# Copyright 2023–2026 Google LLC
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


"""Create an Orbax CheckpointManager with specified (Async or not) Checkpointer."""

import time
from typing import Any

from etils import epath
from flax import nnx
from flax.training import train_state
import jax
from jax.experimental import multihost_utils
from maxtext.checkpoint_conversion.utils import load_dynamic
from maxtext.common import checkpoint_context
from maxtext.common import emergency_checkpointing
from maxtext.common import grain_utility
from maxtext.common import train_state_nnx
from maxtext.input_pipeline import multihost_dataloading
from maxtext.input_pipeline import synthetic_data_processing
from maxtext.utils import elastic_utils
from maxtext.utils import exceptions
from maxtext.utils import gcs_utils
from maxtext.utils import globals as maxtext_globals
from maxtext.utils import max_logging
from orbax.checkpoint import v1 as ocp
from orbax.checkpoint._src.arrays import sharding as sharding_utils

load_safetensors_dynamic_state = load_dynamic.load_safetensors_dynamic_state
PlaceHolderDataIterator = synthetic_data_processing.PlaceHolderDataIterator
MultiHostDataLoadIterator = multihost_dataloading.MultiHostDataLoadIterator
DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE = maxtext_globals.DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE

# Backward compatibility aliases for v0 emergency managers.
EmergencyCheckpointManager = emergency_checkpointing.CheckpointManager
EmergencyReplicatorCheckpointManager = emergency_checkpointing.ReplicatorCheckpointManager
create_orbax_emergency_checkpoint_manager = emergency_checkpointing.create_emergency_checkpoint_manager
create_orbax_emergency_replicator_checkpoint_manager = emergency_checkpointing.create_replicator_checkpoint_manager

# Union of v1 Checkpointer / the emergency factories return; used in type hints.
CheckpointManager = ocp.training.Checkpointer | EmergencyCheckpointManager | EmergencyReplicatorCheckpointManager


def _load_linen_checkpoint_into_nnx(
    path,
    abstract_nnx_state,
    checkpoint_storage_concurrent_gb,
    use_ocdbt,
    use_zarr3,
    enable_single_replica_ckpt_restoring: bool = False,
):
  """Restores a Linen-layout checkpoint into an NNX state (pure_nnx resume).

  Restores against a Linen-shape abstract, reshapes back via
  `from_linen_checkpoint_dict`, then fills NNX-only rngs/dropout with defaults.
  """
  max_logging.log(f"Restoring Linen-layout checkpoint into NNX state at {path}")
  nnx_abstract_pure = abstract_nnx_state.to_pure_dict()
  linen_abstract = train_state_nnx.to_linen_checkpoint_dict(nnx_abstract_pure)
  context = checkpoint_context.build_context(
      use_ocdbt=use_ocdbt,
      use_zarr3=use_zarr3,
      checkpoint_storage_concurrent_gb=checkpoint_storage_concurrent_gb,
      partial_load=True,
      enable_single_replica_ckpt_restoring=enable_single_replica_ckpt_restoring,
  )
  with context:
    restored = ocp.load(epath.Path(path), linen_abstract)
  partial_nnx = train_state_nnx.from_linen_checkpoint_dict(restored)
  return train_state_nnx.populate_pure_dict_from_partial(nnx_abstract_pure, partial_nnx)


def _restore_emergency_linen_checkpoint_into_nnx(
    checkpoint_manager,
    step,
    abstract_nnx_state,
):
  """Restores an emergency Linen-layout checkpoint into an NNX state."""
  max_logging.log(f"Restoring emergency Linen-layout checkpoint into NNX state at step {step}")
  nnx_abstract_pure = abstract_nnx_state.to_pure_dict()
  linen_abstract = train_state_nnx.to_linen_checkpoint_dict(nnx_abstract_pure)
  restored = emergency_checkpointing.restore(checkpoint_manager, step, linen_abstract)
  partial_nnx = train_state_nnx.from_linen_checkpoint_dict(restored)
  return train_state_nnx.populate_pure_dict_from_partial(nnx_abstract_pure, partial_nnx)


def _load_linen_params_into_nnx(
    path,
    nnx_params_abstract,
    checkpoint_storage_concurrent_gb,
    use_ocdbt,
    use_zarr3,
    enable_single_replica_ckpt_restoring: bool = False,
):
  """Weight-only load of a Linen-layout checkpoint into an NNX params state.

  Reuses `to_linen_checkpoint_dict` (wrapping the params under `model`) to build
  the
  `params/params/...` restore target, then rebinds the restored weights into the
  NNX params Variables.
  """
  max_logging.log(f"Restoring Linen-layout params into NNX state at {path}")
  linen_abstract = train_state_nnx.to_linen_checkpoint_dict({"model": nnx_params_abstract.to_pure_dict()})
  context = checkpoint_context.build_context(
      use_ocdbt=use_ocdbt,
      use_zarr3=use_zarr3,
      checkpoint_storage_concurrent_gb=checkpoint_storage_concurrent_gb,
      partial_load=True,
      enable_single_replica_ckpt_restoring=enable_single_replica_ckpt_restoring,
  )
  with context:
    restored = ocp.load(epath.Path(path), linen_abstract)
  return train_state_nnx.rebuild_nnx_with_values(nnx_params_abstract, restored["params"]["params"])


def _load_full_state_from_path(
    path,
    abstract_unboxed_pre_state,
    checkpoint_conversion_fn,
    source_checkpoint_layout,
    checkpoint_storage_concurrent_gb,
    use_ocdbt,
    use_zarr3,
    enable_single_replica_ckpt_restoring: bool = False,
):
  """Load full state from checkpoint at specified path.

  Args:
    path: path to checkpoint
    abstract_unboxed_pre_state: an abstract state that Orbax matches type
      against.
    checkpoint_conversion_fn: user-provided function to convert checkpoint to
      maxtext-supported state.
    source_checkpoint_layout: String representation of the checkpoint layout of
      the source checkpoint.
    checkpoint_storage_concurrent_gb: concurrent GB for checkpoint byte I/O.
    use_ocdbt: Whether to use OCDBT format.
    use_zarr3: Whether to use Zarr3 format.
    enable_single_replica_ckpt_restoring: bool flag for restoring checkpoint
      with SingleReplicaArrayHandler

  Returns:
    The loaded state.
  """
  # pure_nnx checkpoints are stored in the Linen on-disk layout; reshape to NNX.
  if isinstance(abstract_unboxed_pre_state, nnx.State):
    return _load_linen_checkpoint_into_nnx(
        path,
        abstract_unboxed_pre_state,
        checkpoint_storage_concurrent_gb,
        use_ocdbt,
        use_zarr3,
        enable_single_replica_ckpt_restoring=enable_single_replica_ckpt_restoring,
    )

  if source_checkpoint_layout == "orbax":
    context = checkpoint_context.build_context(
        use_ocdbt=use_ocdbt,
        use_zarr3=use_zarr3,
        checkpoint_storage_concurrent_gb=checkpoint_storage_concurrent_gb,
        checkpoint_layout=ocp.options.CheckpointLayout.ORBAX,
        enable_single_replica_ckpt_restoring=enable_single_replica_ckpt_restoring,
    )
    with context:
      return ocp.load(path, abstract_unboxed_pre_state)

  if source_checkpoint_layout == "safetensors":
    context = checkpoint_context.build_context(
        use_ocdbt=use_ocdbt,
        use_zarr3=use_zarr3,
        checkpoint_storage_concurrent_gb=checkpoint_storage_concurrent_gb,
        checkpoint_layout=ocp.options.CheckpointLayout.SAFETENSORS,
        enable_single_replica_ckpt_restoring=enable_single_replica_ckpt_restoring,
    )
    with context:
      metadata = ocp.metadata(path)
      simple_abstract_state = metadata.metadata
      shardings = sharding_utils.construct_maximal_shardings(simple_abstract_state)

      def combine_sharding(sds, shardings):
        return jax.ShapeDtypeStruct(shape=sds.shape, dtype=sds.dtype, sharding=shardings)

      sharded_abstract_state = jax.tree.map(combine_sharding, simple_abstract_state, shardings)
      pre_transformed_state = ocp.load(path, sharded_abstract_state)
    return checkpoint_conversion_fn(pre_transformed_state)

  raise ocp.errors.InvalidLayoutError(f"Unknown checkpoint layout: {source_checkpoint_layout}")


def create_orbax_checkpoint_manager(
    checkpoint_dir: str,
    enable_checkpointing: bool,
    use_async: bool,
    save_interval_steps: int,
    dataset_type: None | str = None,
    orbax_logger: Any = None,  # pytype: disable=attribute-error
    use_ocdbt: bool = True,
    use_zarr3: bool = True,
    enable_continuous_checkpointing: bool = False,
    max_num_checkpoints_to_keep: int = 10,
    checkpoint_storage_concurrent_gb: int = 96,
    enable_single_controller: bool = False,
    colocated_python_checkpointing: bool = False,
    enable_single_replica_ckpt_restoring: bool = False,
    enable_autocheckpoint: bool = False,
    todelete_subdir: str | None = None,
    todelete_full_path: str | None = None,
    ocdbt_target_data_file_size_bytes: int | None = None,
):
  """Returns an Orbax v1 training ``Checkpointer``, or None if checkpointing is disabled."""
  if not enable_checkpointing:
    max_logging.log("Checkpointing disabled, not creating checkpoint manager.")
    return None

  # TODO: b/529622681 - Remove deprecated settings.
  if orbax_logger is not None:
    max_logging.warning(
        "Cloud logging (enable_checkpoint_cloud_logger) is disabled because"
        " Orbax v1 now configures its own logger internally. This config"
        " setting is ignored and will be removed."
    )
  if dataset_type is not None:
    max_logging.warning(
        "Specifying dataset_type upon checkpointer creation is deprecated and"
        " will be removed soon, this is now handled dynamically by Orbax"
        " Checkpointer."
    )

  max_logging.log(f"Creating checkpointer with ocdbt={use_ocdbt} and zarr3={use_zarr3}")

  validated_path = gcs_utils.mkdir_and_check_permissions(checkpoint_dir)

  if ocdbt_target_data_file_size_bytes is None:
    ocdbt_target_data_file_size_bytes = DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE

  context = checkpoint_context.build_context(
      use_ocdbt=use_ocdbt,
      use_zarr3=use_zarr3,
      ocdbt_target_data_file_size_bytes=ocdbt_target_data_file_size_bytes,
      checkpoint_storage_concurrent_gb=checkpoint_storage_concurrent_gb,
      enable_continuous_checkpointing=enable_continuous_checkpointing,
      todelete_full_path=todelete_full_path,
      todelete_subdir=todelete_subdir,
      enable_single_replica_ckpt_restoring=enable_single_replica_ckpt_restoring,
      colocated_python_checkpointing=(enable_single_controller and colocated_python_checkpointing),
      partial_load=True,
  )

  manager = ocp.training.Checkpointer(
      validated_path,
      context=context,
      save_decision_policy=checkpoint_context.build_save_decision_policy(
          save_interval_steps=save_interval_steps,
          enable_continuous_checkpointing=enable_continuous_checkpointing,
          enable_autocheckpoint=enable_autocheckpoint,
      ),
      preservation_policy=checkpoint_context.build_preservation_policy(
          max_to_keep=max_num_checkpoints_to_keep,
      ),
  )
  # Necessary bridges to support v0 backward compatibility.
  manager.use_async = use_async

  max_logging.log("Checkpoint manager created!")
  return manager


def print_save_message(step, async_checkpointing):
  if async_checkpointing:
    max_logging.log(f"Started an asynchronous checkpoint save for step {step}")
  else:
    max_logging.log(f"Saved a checkpoint at step {step}.")


def latest_step(checkpoint_manager):
  """Latest saved step or None, across the v0 emergency manager and the v1 Checkpointer."""
  if isinstance(
      checkpoint_manager, (emergency_checkpointing.CheckpointManager, emergency_checkpointing.ReplicatorCheckpointManager)
  ):
    return checkpoint_manager.latest_step()
  latest = checkpoint_manager.latest
  return latest.step if latest is not None else None


def wait_until_finished(checkpoint_manager):
  """Blocks until pending saves finish, across the v0 emergency manager and the v1 Checkpointer."""
  if emergency_checkpointing.is_emergency_manager(checkpoint_manager):
    checkpoint_manager.wait_until_finished()
  else:
    checkpoint_manager.wait()


def reached_preemption(checkpoint_manager, step: int) -> bool:
  """Whether a preemption sync point has been reached at ``step`` (manager-agnostic)."""
  if emergency_checkpointing.is_emergency_manager(checkpoint_manager):
    return checkpoint_manager.reached_preemption(step)
  return multihost_utils.reached_preemption_sync_point(step)


def _normalize_checkpoint_root(path_str):
  """Lifts a v0-convention pytree path (".../<step>/items") to its checkpoint root."""
  path = epath.Path(str(path_str).rstrip("/"))
  if path.name == "items":
    path = path.parent
  return str(path)


def load_state_if_possible(
    checkpoint_manager: CheckpointManager | None,
    data_iterator: MultiHostDataLoadIterator | list[MultiHostDataLoadIterator] | None,
    load_parameters_from_path: str,
    load_full_state_from_path: str,
    checkpoint_storage_concurrent_gb: int,
    abstract_unboxed_pre_state: train_state.TrainState | nnx.State,
    enable_single_replica_ckpt_restoring: bool | None = False,
    dataset_type: str | None = "tfds",
    step: int = -1,  # -1 means latest
    use_ocdbt=True,
    use_zarr3=True,
    enable_orbax_v1=False,
    checkpoint_conversion_fn=None,
    source_checkpoint_layout="orbax",
    expansion_factor_real_data: int = -1,
    maxtext_config: Any | None = None,
):
  """Loads TrainState as possible from the inputs.

  Args:
    checkpoint_manager: if the checkpoint_manager has a valid checkpoint, return
      that TrainState. This enables a full reload of a run in progress.
    load_parameters_from_path: if there is no checkpoint in the checkpoint
      manager, load parameters from a parameter only checkpoint at this path.
    load_full_state_from_path: if there is no checkpoint in the checkpoint
      manager, load full state from a full state checkpoint at this path.
    abstract_unboxed_pre_state: an unboxed, abstract TrainState that Orbax
      matches type against.
    enable_single_replica_ckpt_restoring: bool flag for restoring checkpoitn
      with SingleReplicaArrayHandler
    checkpoint_storage_concurrent_gb: concurrent GB for checkpoint byte I/O.
    enable_orbax_v1: bool flag for enabling Orbax v1.
    checkpoint_conversion_fn: function for converting checkpoint to Orbax v1.
    source_checkpoint_layout: Optional checkpoint context to use for loading,
      provided in string format with the default being "orbax".

  Returns:
    A tuple of (train_state, train_state_params) where full_train_state captures
     a full reload and train_state_params just the params for a partial reload.
     At most one will be non-None. Both can be None if neither checkpoint is
     set.
  """

  # TODO: b/529622681 - Remove deprecated settings.
  if enable_orbax_v1:
    max_logging.warning(
        "enable_orbax_v1 is deprecated and will be removed, as Orbax v1 is now the default checkpointing API."
    )

  if load_parameters_from_path:
    load_parameters_from_path = _normalize_checkpoint_root(load_parameters_from_path)
  if load_full_state_from_path:
    load_full_state_from_path = _normalize_checkpoint_root(load_full_state_from_path)

  if checkpoint_manager is not None:
    max_logging.log("checkpoint manager exists so trying to load this run's existing checkpoint")

    step = latest_step(checkpoint_manager) if step < 0 else step
    if step is not None:
      max_logging.log(f"restoring from this run's directory step {step}")

      # pure_nnx saves in the Linen on-disk layout; reshape it back into the NNX state.
      # (Emergency managers use their own restore path below.)
      if isinstance(abstract_unboxed_pre_state, nnx.State) and not emergency_checkpointing.is_emergency_manager(
          checkpoint_manager
      ):
        assert isinstance(checkpoint_manager, ocp.training.Checkpointer)
        nnx_abstract_pure = abstract_unboxed_pre_state.to_pure_dict()
        linen_abstract = train_state_nnx.to_linen_checkpoint_dict(nnx_abstract_pure)
        restored_linen = checkpoint_manager.load(step, linen_abstract, checkpointable_name="items")
        partial_nnx = train_state_nnx.from_linen_checkpoint_dict(restored_linen)
        restored_nnx = train_state_nnx.populate_pure_dict_from_partial(nnx_abstract_pure, partial_nnx)
        return ({"items": restored_nnx}, None)

      # Emergency + NNX: the on-disk layout is Linen; partially restore the Linen
      # tree and reshape it back into the NNX state.
      if isinstance(abstract_unboxed_pre_state, nnx.State) and emergency_checkpointing.is_emergency_manager(
          checkpoint_manager
      ):
        restored_nnx = _restore_emergency_linen_checkpoint_into_nnx(checkpoint_manager, step, abstract_unboxed_pre_state)
        return (restored_nnx, None)

      # Case 1: Matches if 'checkpoint_manager' is an instance of either EmergencyCheckpointManager
      # or EmergencyReplicatorCheckpointManager.
      if emergency_checkpointing.is_emergency_manager(checkpoint_manager):
        return (
            emergency_checkpointing.restore(checkpoint_manager, step, abstract_unboxed_pre_state),
            None,
        )
      # Case 2: Matches if dataset type is "grain" and the data iterator is not a
      # PlaceHolderDataIterator and a specific checkpoint file exists for the iterator
      assert isinstance(checkpoint_manager, ocp.training.Checkpointer)
      abstract_checkpointables = {"items": abstract_unboxed_pre_state}
      if (
          dataset_type == "grain"
          and data_iterator
          and not isinstance(data_iterator, PlaceHolderDataIterator)
          and (checkpoint_manager.directory / str(step) / "iter").exists()
      ):
        abstract_checkpointables["iter"] = grain_utility.for_restore(
            checkpoint_manager, step, data_iterator, expansion_factor_real_data
        )
      # Load the checkpointables from the checkpoint.
      return (
          checkpoint_manager.load_checkpointables(step, abstract_checkpointables),
          None,
      )

  if source_checkpoint_layout == "safetensors_dynamic":
    path = load_parameters_from_path or load_full_state_from_path
    max_logging.log(f"Dynamic On-the-Fly Formatting: Loading SafeTensors from {path}")
    # Delegate to custom MaxText dynamic conversion module
    return load_safetensors_dynamic_state(path, abstract_unboxed_pre_state, maxtext_config)
  elif load_parameters_from_path != "":
    if isinstance(abstract_unboxed_pre_state, nnx.State):
      _, params, _ = nnx.split(abstract_unboxed_pre_state.model, nnx.Param, ...)
    else:
      params = abstract_unboxed_pre_state.params

    restored_params = load_params_from_path(
        load_parameters_from_path,
        params,
        checkpoint_storage_concurrent_gb,
        use_ocdbt=use_ocdbt,
        use_zarr3=use_zarr3,
        enable_single_replica_ckpt_restoring=enable_single_replica_ckpt_restoring,
    )
    return None, restored_params
  elif load_full_state_from_path != "":
    max_logging.log(f"Loading full state from path: {load_full_state_from_path}")
    restored_state = _load_full_state_from_path(
        path=load_full_state_from_path,
        abstract_unboxed_pre_state=abstract_unboxed_pre_state,
        checkpoint_conversion_fn=checkpoint_conversion_fn,
        source_checkpoint_layout=source_checkpoint_layout,
        checkpoint_storage_concurrent_gb=checkpoint_storage_concurrent_gb,
        use_ocdbt=use_ocdbt,
        use_zarr3=use_zarr3,
        enable_single_replica_ckpt_restoring=enable_single_replica_ckpt_restoring,
    )
    return {"items": restored_state}, None
  else:
    max_logging.log("No existing checkpoints found, not restoring checkpoint.")
    return None, None


def setup_checkpoint_logger(config) -> Any | None:  # pytype: disable=attribute-error
  """DEPRECATED: Setup checkpoint logger."""
  # TODO: b/529622681 - Remove this config option entirely.
  if config.enable_checkpoint_cloud_logger:
    max_logging.warning(
        "Cloud logging (enable_checkpoint_cloud_logger) is disabled because"
        " Orbax v1 now configures its own logger internally. This config"
        " setting is ignored and will be removed."
    )


def load_params_from_path(
    load_parameters_from_path,
    abstract_unboxed_params,
    checkpoint_storage_concurrent_gb,
    use_ocdbt=True,
    use_zarr3=True,
    enable_single_replica_ckpt_restoring: bool = False,
):
  """Load decode params from checkpoint at specified path."""
  assert load_parameters_from_path, "load_parameters_from_path is not defined."
  max_logging.log(f"restoring params from {load_parameters_from_path}")

  # Orbax v1 refuses to read an item subdirectory directly, normalize the documented
  # ".../<step>/items" form to its checkpoint root and load it by name below.
  path = epath.Path(_normalize_checkpoint_root(load_parameters_from_path))

  # NNX target: the on-disk checkpoint is in Linen layout; reshape it into the
  # NNX params state.
  if isinstance(abstract_unboxed_params, nnx.State):
    return _load_linen_params_into_nnx(
        path,
        abstract_unboxed_params,
        checkpoint_storage_concurrent_gb,
        use_ocdbt,
        use_zarr3,
        enable_single_replica_ckpt_restoring=enable_single_replica_ckpt_restoring,
    )

  # Memory optimization: restore only the "params" key (the checkpoint may also hold opt_state/step);
  # partial_load drops the rest. The abstract carries shape/dtype/sharding directly.
  context = checkpoint_context.build_context(
      use_ocdbt=use_ocdbt,
      use_zarr3=use_zarr3,
      checkpoint_storage_concurrent_gb=checkpoint_storage_concurrent_gb,
      partial_load=True,
      enable_single_replica_ckpt_restoring=enable_single_replica_ckpt_restoring,
  )
  # Dispatch on the on-disk layout instead of assuming a step root: callers pass
  # step roots, v0-style pytree dirs (normalized above), and v0 flat params-only
  # checkpoints (save_params_to_path wrote the pytree directly at the directory).
  with context:
    if (path / "items").exists():
      restored = ocp.load(path, {"params": abstract_unboxed_params}, checkpointable_name="items")
    else:
      # The path itself is a pytree dir; read it directly.
      restored = ocp.load(path, {"params": abstract_unboxed_params}, checkpointable_name=None)
  return restored["params"]


def save_params_to_path(checkpoint_dir, params, use_ocdbt=True, use_zarr3=True):
  """Save decode params in checkpoint at specified path."""
  assert checkpoint_dir, "checkpoint_dir is not defined."
  max_logging.log(f"Saving params checkpoint with use_ocdbt={use_ocdbt} and" f" use_zarr3={use_zarr3}")
  context = checkpoint_context.build_context(use_ocdbt=use_ocdbt, use_zarr3=use_zarr3)
  with context:
    ocp.save(checkpoint_dir, {"params": params}, checkpointable_name="items", overwrite=True)
  max_logging.log(f"Params checkpoint saved at: {checkpoint_dir}")


def load_checkpoint_metadata(checkpoint_dir_path: str) -> Any:
  """Loads custom metadata from an Orbax checkpoint.

  Args:
    checkpoint_dir_path: Path to the checkpoint directory.

  Returns:
    A dictionary containing custom metadata, or an empty dictionary if none is
    present or loading fails.
  """
  checkpoint_dir = epath.Path(_normalize_checkpoint_root(checkpoint_dir_path))
  try:
    metadata = ocp.checkpointables_metadata(checkpoint_dir)
    return metadata.custom_metadata or {}
  except Exception as e:  # pylint: disable=broad-except
    max_logging.log(f"Warning: Failed to load checkpoint metadata: {e}")
    return {}


def maybe_save_checkpoint(checkpoint_manager, state, config, data_iterator, step=None):
  """Save checkpoint if checkpointing is enabled."""
  if checkpoint_manager is None:
    return

  # Determine the effective step for saving a checkpoint.
  # If 'step' is not provided, this call is for a potential final checkpoint
  # and use the last completed step from the state.
  if step is not None:
    actual_step = int(step)
  else:
    if config.pure_nnx:
      # Under DiLoCo the step lives on the DiLoCoTrainState; otherwise on the optimizer.
      actual_step = int(state.step if config.enable_diloco else state.optimizer.step) - 1
    else:
      # Linen TrainState has .step attribute
      actual_step = int(state.step) - 1

  if latest_step(checkpoint_manager) == actual_step:
    max_logging.log(f"Checkpoint for step {actual_step} already exists, skipping save.")
    return

  if config.pure_nnx:
    # Save in the Linen on-disk layout so pure_nnx and Linen checkpoints are interchangeable.
    if config.enable_diloco:
      # DiLoCoTrainState: persist the synchronized global model (outer params).
      # The per-replica inner optimizer / outer-momentum state is not checkpointed.
      step_value = state.step.get_value() if hasattr(state.step, "get_value") else state.step
      state = train_state_nnx.to_linen_checkpoint_dict({"model": state.params, "optimizer": {"step": step_value}})
    else:
      state = train_state_nnx.to_linen_checkpoint_dict(state.to_pure_dict())

  # Determine if a checkpoint save should be forced, overriding the usual `config.checkpoint_period` logic.
  # This occurs if this function was called:
  # without an explicit 'step' (implying it's a checkpoint save for final step),
  # AND the 'actual_step' is a valid step,
  # AND it's not a step that would normally trigger a checkpoint save.
  force_ckpt_save = step is None and actual_step != -1 and (actual_step % config.checkpoint_period != 0)

  try:
    checkpoint_saved = save_checkpoint(checkpoint_manager, actual_step, state, config, data_iterator, force_ckpt_save)
    if checkpoint_saved:
      print_save_message(actual_step, config.async_checkpointing)
      if config.elastic_enabled:
        elastic_utils.maybe_elastic_scale_up(config, checkpoint_manager)
  except elastic_utils.manager.ScaleUpSignalError as e:
    if config.elastic_enabled:
      max_logging.log(f"Elastic event detected, letting exception bubble up: {e}")
      raise
    else:
      raise exceptions.StopTraining("Job is preempted.") from e
  except jax.errors.JaxRuntimeError as e:
    if config.elastic_enabled:
      max_logging.log(f"Elastic event detected, letting exception bubble up: {e}")
      raise
    else:
      raise exceptions.StopTraining("Job is preempted.") from e
  except Exception as e:
    raise exceptions.StopTraining(f"Checkpointing failed. {str(e)}") from e

  # Wait for any pending checkpoint save to finish during preemption or final step save
  if force_ckpt_save or reached_preemption(checkpoint_manager, actual_step):
    wait_until_finished(checkpoint_manager)

  # Raise exception upon preemption
  if reached_preemption(checkpoint_manager, actual_step):
    raise exceptions.StopTraining("Job is preempted.")


def save_checkpoint(checkpoint_manager, step, state, config=None, data_iterator=None, force=False):
  """Wrapper for saving checkpoint."""
  if config and config.enable_checkpointing:
    if (
        force
        or (step % config.checkpoint_period == 0 and not config.enable_continuous_checkpointing)
        or (config.enable_emergency_checkpoint and step % config.local_checkpoint_period == 0)
        or (config.enable_autocheckpoint and reached_preemption(checkpoint_manager, step))
    ):
      blocking_until_ready_start = time.time()
      max_logging.log(f"Waiting for step {step} to finish before checkpoint...")
      # We block here on the step finishing so that our checkpointing metrics
      # measure only checkpointing time, not training time.
      jax.block_until_ready(state)
      max_logging.log(
          f"Waited {time.time() - blocking_until_ready_start} seconds for step "
          f"{step} to finish before starting checkpointing."
      )

  # Emergency / replicator managers keep the v0 save path.
  if emergency_checkpointing.is_emergency_manager(checkpoint_manager):
    return emergency_checkpointing.save(checkpoint_manager, step, state, config, force)

  # Record config properties needed to validate compatibility at load time
  # (e.g. proactive scan_layers verification, LoRA restore).
  custom_metadata = {}
  if config:
    if hasattr(config, "scan_layers"):
      custom_metadata["scan_layers"] = config.scan_layers
    if hasattr(config, "lora") and config.lora and getattr(config.lora, "lora_rank", 0) > 0:
      custom_metadata["lora"] = config.lora.model_dump()

  # Standard path: Orbax v1 Checkpointer. Storage/chunk options live on the manager's Context.
  checkpointables = {"items": state}
  if config and config.dataset_type == "grain" and not isinstance(data_iterator, PlaceHolderDataIterator):
    checkpointables["iter"] = grain_utility.for_save(step, data_iterator, config.expansion_factor_real_data)
  # The v1 Checkpointer raises for an already-existing step BEFORE consulting the
  # save decision policy; v0 silently skipped such saves (should_save ran first).
  # Preserve v0 semantics: e.g. resuming from a non-latest step into a directory
  # that still holds later/off-interval checkpoints must not kill training.
  try:
    if getattr(checkpoint_manager, "use_async", False):
      # Async save returns once the blocking device-to-host copy is done and writes in
      # the background (v0 enable_async_checkpointing parity); a None response means the
      # save decision policy declined. Background errors surface on the next save/wait.
      response = checkpoint_manager.save_checkpointables_async(
          step, checkpointables, force=force, custom_metadata=custom_metadata
      )
      return response is not None
    return checkpoint_manager.save_checkpointables(step, checkpointables, force=force, custom_metadata=custom_metadata)
  except FileExistsError as e:  # ocp.training StepAlreadyExistsError subclasses FileExistsError
    max_logging.log(f"Checkpoint for step {step} already exists, skipping save. ({e})")
    return False
