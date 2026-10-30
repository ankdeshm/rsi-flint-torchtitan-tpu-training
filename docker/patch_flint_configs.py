#!/usr/bin/env python3
"""Appends Flint MFU hillclimb & v5p/v6e portable recipes to PR #122 config_registry.py."""

from pathlib import Path

TARGET = Path("/torchtitan-native/torchtitan/experiments/tpu/examples/flint/config_registry.py")

ADDITIONS = '''

# ==============================================================================
# RSI POC Hillclimb & Cross-Generation (v7x / v5p / v6e) Configs
# ==============================================================================

_HIER_SPARSECORE_RS_FLAGS = {
    "xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter": True,
    "xla_tpu_enable_sparse_core_hierarchical_reduce_scatter": True,
    "xla_tpu_use_single_sparse_core_for_reduce_scatter_offload": False,
}


def flint_40b_v012_hier_rs() -> TpuGraphTrainer.Config:
  """Hillclimb 1 (v7x): flint_40b_v011_n40 + PR #123 hierarchical ND SparseCore reduce-scatter."""
  config = flint_40b_v011_n40()
  config.libtpu_init_flags = tpu_flags.merge_flags(
      config.libtpu_init_flags, _HIER_SPARSECORE_RS_FLAGS
  )
  return config


def flint_40b_v013_n49() -> TpuGraphTrainer.Config:
  """Hillclimb 2 (v7x-16 / 32 ranks): save Splash attention output in ALL 49 layers + hier RS."""
  config = flint_40b_v012_hier_rs()
  _save_attention_last_n_layers(config, 49)
  return config


def flint_40b_v014_tile1024() -> TpuGraphTrainer.Config:
  """Hillclimb 3 (v7x-16 / 32 ranks): n49 + hier RS + Pallas grouped_mm tile_m=1024."""
  config = flint_40b_v013_n49()
  for _, experts, _, _ in config.model_spec.model.traverse(
      TpuGroupedExperts.Config
  ):
    experts.pallas_grouped_mm_tile_m = 1024
  return config


def flint_40b_v015_compiled_opt() -> TpuGraphTrainer.Config:
  """Hillclimb 4 (v7x-16): n49 + hier RS + compiled optimizer step."""
  config = flint_40b_v013_n49()
  if "compiled_optimizers" not in config.compile.components:
    config.compile.components = list(config.compile.components) + ["compiled_optimizers"]
  return config


def flint_40b_v5p_base() -> TpuGraphTrainer.Config:
  """Portable v5p recipe (no v7x dvfs_p_state=7): GraphTrainer split + Pallas GMM + native layout."""
  config = flint_40b_v005_gt_split()
  _use_pallas_grouped_mm(config)
  for _, experts, _, _ in config.model_spec.model.traverse(
      TpuGroupedExperts.Config
  ):
    experts.use_kernel_native_weight_layout = True
  config.libtpu_init_flags = tpu_flags.merge_flags(
      config.libtpu_init_flags,
      {"xla_tpu_prefer_async_allgather_to_allreduce": True},
  )
  return config


def flint_40b_v5p_n40() -> TpuGraphTrainer.Config:
  """Portable v5p-16 (16 ranks x 95 GiB): save last 40 Splash attention layers."""
  config = flint_40b_v5p_base()
  _save_attention_last_n_layers(config, 40)
  return config


def flint_40b_v5p_n49() -> TpuGraphTrainer.Config:
  """Portable v5p-32 (32 ranks x 95 GiB): save all 49 Splash attention layers + tile_m=1024."""
  config = flint_40b_v5p_base()
  _save_attention_last_n_layers(config, 49)
  for _, experts, _, _ in config.model_spec.model.traverse(
      TpuGroupedExperts.Config
  ):
    experts.pallas_grouped_mm_tile_m = 1024
  return config


def flint_40b_v6e_fullremat() -> TpuGraphTrainer.Config:
  """Portable v6e-32 (32 ranks x 32 GiB HBM): full remat + mb=8 (16,384 tokens/rank) to fit 32 GiB."""
  config = flint_40b_v5p_base()
  config.training.num_tokens_per_microbatch_per_dp_rank = 8 * _40B_CONTEXT_LENGTH
  return config


__all__.extend([
    "flint_40b_v012_hier_rs",
    "flint_40b_v013_n49",
    "flint_40b_v014_tile1024",
    "flint_40b_v015_compiled_opt",
    "flint_40b_v5p_base",
    "flint_40b_v5p_n40",
    "flint_40b_v5p_n49",
    "flint_40b_v6e_fullremat",
])
'''


def main() -> None:
    src = TARGET.read_text()
    if "flint_40b_v012_hier_rs" not in src:
        TARGET.write_text(src + ADDITIONS)
    print(f"Patched {TARGET} successfully.")


if __name__ == "__main__":
    main()
