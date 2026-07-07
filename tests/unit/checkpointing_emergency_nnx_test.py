# Copyright 2025-2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for restoring pure_nnx emergency checkpoints.

pure_nnx saves in the on-disk checkpoint layout (Linen params + an nnx_aux subtree),
and the emergency manager restores against the abstract it is built with, so that
abstract must be in the same layout. These tests guard it.
"""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from etils import epath
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.common import checkpointing
from maxtext.common import train_state_nnx
import optax


class _Model(nnx.Module):
  """Linear + dropout, so the state has weights, an optimizer, and rngs."""

  def __init__(self, rngs: nnx.Rngs):
    self.linear = nnx.Linear(2, 3, rngs=rngs)
    self.dropout = nnx.Dropout(rate=0.5, rngs=rngs)

  def __call__(self, x, deterministic=False):
    return self.dropout(self.linear(x), deterministic=deterministic)


_TX = optax.adam(1e-3)


def _nnx_abstract_state():
  """An abstract NNX state (model/optimizer), the shape pure_nnx hands the manager."""

  def make():
    model = _Model(nnx.Rngs(0))
    return nnx.state(train_state_nnx.TrainStateNNX(model, nnx.Optimizer(model, _TX, wrt=nnx.Param)))

  return nnx.eval_shape(make)


class TestEmergencyManagerAbstractLayout(unittest.TestCase):
  """Checks the emergency manager is built with a checkpoint-layout abstract.

  Mocks the Orbax manager, so it runs in normal CI without a real EMC setup.
  """

  def _create(self, abstract_state):
    """Calls the constructor with Orbax mocked; returns the abstract it was handed."""
    mesh = jax.sharding.Mesh(jax.devices(), ("x",))
    with (
        mock.patch.object(checkpointing, "EmergencyCheckpointManager") as manager_cls,
        mock.patch.object(checkpointing.gcs_utils, "mkdir_and_check_permissions", side_effect=epath.Path),
    ):
      with tempfile.TemporaryDirectory() as d:
        checkpointing.create_orbax_emergency_checkpoint_manager(
            os.path.join(d, "local"),
            os.path.join(d, "persist"),
            mesh,
            abstract_state,
            local_save_interval_steps=1,
            persistent_save_interval_steps=1,
        )
    return manager_cls.call_args.kwargs["abstract_state"]

  def test_nnx_abstract_is_converted_to_checkpoint_layout(self):
    """An NNX abstract (model/optimizer) is reshaped to the on-disk checkpoint layout."""
    passed = self._create(_nnx_abstract_state())
    self.assertNotIsInstance(passed, nnx.State)
    # The checkpoint layout (Linen params + nnx_aux), not the NNX model/optimizer
    # roots that caused the mismatch.
    self.assertNotIn("model", passed)
    self.assertNotIn("optimizer", passed)
    self.assertIn("params", passed)
    self.assertIn("nnx_aux", passed)  # rngs from the dropout model, via to_checkpoint_dict

  def test_non_nnx_abstract_is_passed_through_unchanged(self):
    """A Linen state (not an nnx.State) is passed through as-is."""
    sentinel = SimpleNamespace(params={"a": 1}, opt_state=(), step=0)  # stand-in Linen state
    passed = self._create(sentinel)
    self.assertIs(passed, sentinel)


class TestEmergencySaveRestoreRoundTrip(unittest.TestCase):
  """Save then restore through the real emergency manager.

  Skips if the emergency manager can't be built in this environment.
  """

  def setUp(self):
    self._dir = tempfile.mkdtemp()

  def _config(self):
    return SimpleNamespace(
        pure_nnx=True,
        enable_diloco=False,
        enable_checkpointing=True,
        enable_continuous_checkpointing=False,
        enable_emergency_checkpoint=True,
        enable_autocheckpoint=False,
        enable_multi_tier_checkpointing=False,
        checkpoint_period=1,
        local_checkpoint_period=1,
        async_checkpointing=False,
        dataset_type="tfds",
        lora=None,
        checkpoint_storage_target_data_file_size_bytes=checkpointing.DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE,
        elastic_enabled=False,
    )

  def test_emergency_round_trip_reshapes_back_to_nnx(self):
    mesh = jax.sharding.Mesh(jax.devices(), ("x",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    abstract = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=sharding) if hasattr(x, "shape") else x,
        _nnx_abstract_state(),
    )

    try:
      manager = checkpointing.create_orbax_emergency_checkpoint_manager(
          os.path.join(self._dir, "local"),
          os.path.join(self._dir, "persist"),
          mesh,
          abstract,
          local_save_interval_steps=1,
          persistent_save_interval_steps=1,
      )
    except Exception as e:  # pylint: disable=broad-except
      raise unittest.SkipTest(f"EMC manager unavailable in this environment: {e}")

    # Train one step so weights and optimizer state are non-trivial, then save.
    model = _Model(nnx.Rngs(0))
    state = train_state_nnx.TrainStateNNX(model, nnx.Optimizer(model, _TX, wrt=nnx.Param))
    grads = nnx.grad(lambda m: jnp.mean(m(jnp.ones((4, 2)), deterministic=False) ** 2))(state.model)
    state.apply_gradients(grads)
    saved_kernel = jnp.asarray(nnx.state(state).to_pure_dict()["model"]["linear"]["kernel"])

    checkpointing.maybe_save_checkpoint(manager, nnx.state(state), self._config(), data_iterator=None, step=1)
    manager.wait_until_finished()

    # Restore should not raise, and should come back as NNX.
    full, _ = checkpointing.load_state_if_possible(
        manager,
        data_iterator=None,
        load_parameters_from_path="",
        load_full_state_from_path="",
        checkpoint_storage_concurrent_gb=8,
        abstract_unboxed_pre_state=abstract,
        dataset_type="tfds",
        maxtext_config=self._config(),
    )
    self.assertIn("model", full)
    self.assertIn("optimizer", full)
    self.assertNotIn("params", full)
    self.assertTrue(jnp.allclose(full["model"]["linear"]["kernel"], saved_kernel))


if __name__ == "__main__":
  unittest.main()
