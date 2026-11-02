#!/usr/bin/env python3
"""Pass 11A & Pass 11B TPU v7x-16 (2x2x4, 16 Ironwood chips / 32 TensorCores) StaticEP MFU Optimization Sweeps.

Resolves the exact `GuardOnDataDependentSymNode` root cause uncovered in Pass 10C/10D when `EP > 1`:
  - Upstream `AllToAllTokenDispatcher` called `.to("cpu").tolist()` on dynamic per-peer token counts
    and launched ragged `all_to_all_single`, creating data-dependent `SymInt`s rejected by XLA `aot_fx_trace`.
  - `StaticTpuAllToAllTokenDispatcher` replaces the `ep_mesh is not None` path with a 100% static-shape,
    scatter-free (`index_select`-only in BOTH forward and backward!) uniform `[C]*EP` `all_to_all_single`
    transport (`C = ceil(1.15 * (T*K)/EP / 512) * 512`), validated for exact numerical parity (`<2e-6`)
    and `AOTAutograd` `min_cut_rematerialization_partition` compatibility.
  - Also injects calibrated Ironwood XLA flags directly into `config.libtpu_init_flags` (which `TpuGraphTrainer`
    reads to overwrite `LIBTPU_INIT_ARGS`) and enables `metrics.log_freq=5` async multi-step pipelining.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import threading
import time

FLINT_FLOPS_PER_TOKEN = 17.385e9
PEAK_BF16_TFLOPS_PER_RANK = {"v7x": 1153.5, "v5p": 459.0, "v6e": 918.0}

STEP_RE = re.compile(
    r"step:\s*(?P<step>\d+).*?loss:\s*(?P<loss>[0-9.]+).*?tps:\s*(?P<tps>[0-9.,]+)",
    re.IGNORECASE,
)
MFU_RE = re.compile(r"mfu:\s*(?P<mfu>[0-9.]+)%", re.IGNORECASE)
MEM_RE = re.compile(r"memory:\s*(?P<mem>[0-9.]+)GiB", re.IGNORECASE)

STATIC_EP_DISPATCHER_PATCH = '''

# ==============================================================================
# Pass 11 StaticTpuAllToAllTokenDispatcher (100% Static-Shape, Scatter-Free TPU EP)
# ==============================================================================
import math as _sep_math
import os as _sep_os
import torch.nn.functional as _sep_F
import spmd_types as _sep_spmd
from torchtitan.distributed.utils import get_spmd_backend as _sep_get_spmd_backend


@dataclass(frozen=True)
class StaticEPDispatchMetadata:
  token_indices_experts_sorted_N: torch.Tensor
  topk_scores_experts_sorted_N: torch.Tensor
  src_idx_S: torch.Tensor
  slot_N: torch.Tensor
  valid_N: torch.Tensor
  exp_perm_R: torch.Tensor
  inv_exp_perm_R: torch.Tensor
  cap_C: int
  ep_size: int


class _StaticEPPack(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x_ND, src_idx_S, slot_N, valid_N):
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(slot_N, valid_N)
    ctx.mark_non_differentiable(src_idx_S, slot_N, valid_N)
    padded = _sep_F.pad(x_ND, (0, 0, 0, 1)) if x_ND.ndim == 2 else _sep_F.pad(x_ND, (0, 1))
    return torch.index_select(padded, 0, src_idx_S)

  @staticmethod
  def backward(ctx, grad_S):
    slot_N, valid_N = ctx.saved_tensors
    if grad_S is None:
      return None, None, None, None
    gathered = torch.index_select(grad_S, 0, slot_N)
    mask = valid_N.unsqueeze(-1) if grad_S.ndim == 2 else valid_N
    return torch.where(mask, gathered, torch.zeros_like(gathered)), None, None, None


class _StaticEPPermute(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x_RD, perm_R, inv_perm_R):
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(inv_perm_R)
    return torch.index_select(x_RD, 0, perm_R)

  @staticmethod
  def backward(ctx, grad_RD):
    (inv_perm_R,) = ctx.saved_tensors
    if grad_RD is None:
      return None, None, None
    return torch.index_select(grad_RD, 0, inv_perm_R), None, None


class _StaticEPUnpack(torch.autograd.Function):
  @staticmethod
  def forward(ctx, ret_SD, slot_N, valid_N, src_idx_S):
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(src_idx_S)
    gathered = torch.index_select(ret_SD, 0, slot_N)
    mask = valid_N.unsqueeze(-1) if ret_SD.ndim == 2 else valid_N
    return torch.where(mask, gathered, torch.zeros_like(gathered))

  @staticmethod
  def backward(ctx, grad_ND):
    (src_idx_S,) = ctx.saved_tensors
    if grad_ND is None:
      return None, None, None, None
    padded = _sep_F.pad(grad_ND, (0, 0, 0, 1)) if grad_ND.ndim == 2 else _sep_F.pad(grad_ND, (0, 1))
    return torch.index_select(padded, 0, src_idx_S), None, None, None


TpuAllToAllTokenDispatcher._orig_dispatch = TpuAllToAllTokenDispatcher.dispatch
TpuAllToAllTokenDispatcher._orig_combine = TpuAllToAllTokenDispatcher.combine


def _static_tpu_ep_dispatch(
    self,
    x_TD: torch.Tensor,
    topk_scores_TK: torch.Tensor,
    topk_expert_ids_TK: torch.Tensor,
    num_local_tokens_per_expert_E: torch.Tensor,
):
  if self.ep_mesh is None:
    return TpuAllToAllTokenDispatcher._orig_dispatch(
        self, x_TD, topk_scores_TK, topk_expert_ids_TK, num_local_tokens_per_expert_E
    )
  ep_size = self.ep_mesh.size()
  E = num_local_tokens_per_expert_E.numel()
  e = E // ep_size
  N = x_TD.shape[0] * self.top_k
  cap_factor = float(_sep_os.environ.get("TT_STATIC_EP_CAP_FACTOR", "1.15"))
  tile_align = int(_sep_os.environ.get("TT_STATIC_EP_ALIGN", "512"))
  cap_C = max(tile_align, _sep_math.ceil(cap_factor * N / ep_size / tile_align) * tile_align)
  S = ep_size * cap_C

  with annotate_region(MOE_ROUTING):
    flat_eids_N = topk_expert_ids_TK.reshape(-1)
    perm_N = torch.argsort(flat_eids_N, stable=True)
    sorted_eids_N = torch.index_select(flat_eids_N, 0, perm_N)

  routed_ND, tok_idx_N, sorted_scores_N = _LocalReorder.apply(
      x_TD, topk_scores_TK, perm_N, self.top_k
  )

  if _sep_get_spmd_backend() == "spmd_types" and _sep_spmd.is_type_checking():
    _sep_spmd.mutate_type(
        num_local_tokens_per_expert_E,
        src=_sep_spmd.P,
        dst={"dp": _sep_spmd.V, "cp": _sep_spmd.V, "tp": _sep_spmd.V},
    )

  with torch.no_grad():
    peer_counts = num_local_tokens_per_expert_E.reshape(ep_size, e).sum(dim=1).to(torch.int64)
    peer_starts = torch.cumsum(peer_counts, dim=0) - peer_counts
    s_arange = torch.arange(S, device=x_TD.device, dtype=torch.int64)
    s_peer = s_arange // cap_C
    s_off = s_arange % cap_C
    s_valid = s_off < peer_counts[s_peer]
    src_idx_S = torch.where(s_valid, peer_starts[s_peer] + s_off, torch.full_like(s_off, N))

    p_N = (sorted_eids_N // e).to(torch.int64)
    j_N = torch.arange(N, device=x_TD.device, dtype=torch.int64) - peer_starts[p_N]
    valid_N = j_N < cap_C
    slot_N = p_N * cap_C + torch.clamp(j_N, min=0, max=cap_C - 1)

    padded_lexp = _sep_F.pad((sorted_eids_N % e).to(torch.int32), (0, 1), value=e - 1)
    send_lexp = torch.index_select(padded_lexp, 0, src_idx_S)

  send_x = _StaticEPPack.apply(routed_ND, src_idx_S, slot_N, valid_N)
  splits = [cap_C] * ep_size

  with maybe_set_sparse_mesh():
    pg = "ep" if _sep_get_spmd_backend() == "spmd_types" else self.ep_mesh.get_group()
    if _sep_get_spmd_backend() == "spmd_types" and _sep_spmd.is_type_checking():
      send_x = _sep_spmd.reinterpret_mesh(send_x, _sep_spmd.current_mesh())
      send_lexp = _sep_spmd.reinterpret_mesh(send_lexp, _sep_spmd.current_mesh())

    recv_x = self._dispatch_token_exchange(send_x, pg, splits, splits)
    with torch.no_grad():
      recv_lexp = self._dispatch_token_exchange(send_lexp, pg, splits, splits)
      exp_perm_R = torch.argsort(recv_lexp, stable=True)
      inv_exp_perm_R = torch.argsort(exp_perm_R)
      counts_e = (
          recv_lexp.unsqueeze(1)
          == torch.arange(e, device=x_TD.device, dtype=torch.int32)
      ).sum(dim=0, dtype=torch.int32)
    routed_RD = _StaticEPPermute.apply(recv_x, exp_perm_R, inv_exp_perm_R)

  meta = StaticEPDispatchMetadata(
      token_indices_experts_sorted_N=tok_idx_N,
      topk_scores_experts_sorted_N=sorted_scores_N,
      src_idx_S=src_idx_S,
      slot_N=slot_N,
      valid_N=valid_N,
      exp_perm_R=exp_perm_R,
      inv_exp_perm_R=inv_exp_perm_R,
      cap_C=cap_C,
      ep_size=ep_size,
  )
  return routed_RD, counts_e, meta


def _static_tpu_ep_combine(
    self,
    routed_output_RD: torch.Tensor,
    metadata,
    x_TD: torch.Tensor,
) -> torch.Tensor:
  if self.ep_mesh is None:
    return TpuAllToAllTokenDispatcher._orig_combine(self, routed_output_RD, metadata, x_TD)
  assert isinstance(metadata, StaticEPDispatchMetadata)
  splits = [metadata.cap_C] * metadata.ep_size
  with maybe_set_sparse_mesh():
    pg = "ep" if _sep_get_spmd_backend() == "spmd_types" else self.ep_mesh.get_group()
    unsorted_RD = _StaticEPPermute.apply(
        routed_output_RD, metadata.inv_exp_perm_R, metadata.exp_perm_R
    )
    ret_SD = self._combine_token_exchange(unsorted_RD, pg, splits, splits)

  if _sep_get_spmd_backend() == "spmd_types" and _sep_spmd.is_type_checking():
    ret_SD = _sep_spmd.reinterpret_mesh(ret_SD, _sep_spmd.current_mesh())

  y_ND = _StaticEPUnpack.apply(ret_SD, metadata.slot_N, metadata.valid_N, metadata.src_idx_S)
  if self.fuse_route_score_backward:
    weighted_ND = scale_route_weights(y_ND, metadata.topk_scores_experts_sorted_N)
  else:
    weighted_ND = (
        y_ND.float() * metadata.topk_scores_experts_sorted_N.reshape(-1, 1)
    ).to(y_ND.dtype)
  return stable_grouped_reduce(
      weighted_ND, metadata.token_indices_experts_sorted_N, self.top_k
  )


TpuAllToAllTokenDispatcher.dispatch = _static_tpu_ep_dispatch
TpuAllToAllTokenDispatcher.combine = _static_tpu_ep_combine
'''

NATIVE_P11_REGISTRY_PATCH = '''

# ==============================================================================
# Pass 11A / 11B Flint v7x-16 StaticEP Configs (EP=8/16/32 ZeRO-2 + WholeStep + Offload)
# ==============================================================================
from torchtitan.experiments.tpu.token_dispatcher import (
    TpuAllToAllTokenDispatcher,
    TpuRoutedExperts,
    TpuRoutedExpertsConverter,
)

_P11_BF16_ROPE = (
    "torchtitan.experiments.tpu.overrides.bf16_rope"
    ".tpu_activation_dtype_cossin_rope"
)

_P11_V7X_BASE_FLAGS = {
    "xla_tpu_dvfs_p_state": 7,
    "xla_tpu_scoped_vmem_limit_kib": 65536,
    "xla_tpu_enable_offloading_copy_to_sparsecore": False,
    "xla_tpu_copy_insertion_use_region_analysis": True,
    "xla_tpu_copy_insertion_use_region_analysis_limit": 1000,
    "xla_tpu_prefer_async_allgather_to_allreduce": True,
    "xla_enable_async_all_gather": True,
    "xla_enable_async_all_reduce": True,
    "xla_tpu_enable_async_all_to_all": True,
    "xla_tpu_overlap_compute_collective_tc": True,
    "xla_tpu_layer_scheduler_min_compute_latency_for_backward_pass": 25000000,
}


def _configure_static_ep_p11(
    config, *, ep_degree: int, tile_m: int = 512, paired: bool = True
) -> None:
  config.parallelism.expert_parallel_degree = ep_degree
  for _, td, _, _ in config.model_spec.model.traverse(TpuAllToAllTokenDispatcher.Config):
    td.fuse_route_score_backward = True
  for _, ex, _, _ in config.model_spec.model.traverse(TpuGroupedExperts.Config):
    ex.use_pallas_grouped_mm = True
    ex.use_pallas_paired_grouped_mm = paired
    ex.pallas_grouped_mm_tile_m = tile_m
    if paired:
      ex.pallas_grouped_mm_tgmm_tiling = (tile_m, 1024, 0)
      ex.pallas_grouped_mm_w2_tgmm_tiling = (256, 512, 0)
      ex.pallas_grouped_mm_w2_transposed_tgmm_tiling = (256, 0, 1024)
    else:
      ex.pallas_grouped_mm_tgmm_tiling = (tile_m, 512, 512)
      ex.pallas_grouped_mm_w2_tgmm_tiling = (256, 512, 512)
      ex.pallas_grouped_mm_w2_transposed_tgmm_tiling = (256, 512, 1024)
    ex.use_kernel_native_weight_layout = True
  if _P11_BF16_ROPE not in config.override.imports:
    config.override.imports.append(_P11_BF16_ROPE)
  config.libtpu_init_flags = tpu_flags.merge_flags(
      config.libtpu_init_flags,
      tpu_flags.SPARSECORE_LATENCY_HIDING_SCHEDULER,
      tpu_flags.DUAL_SPARSECORE_REDUCE_SCATTER,
      tpu_flags.ICI_AG_PIPELINING,
      _HIER_SPARSECORE_RS_FLAGS,
      _P11_V7X_BASE_FLAGS,
  )


def _build_split_staticep_zero2_p11(
    *, ep_degree: int, tile_m: int, save_attn_layers: int, paired: bool = True
) -> TpuGraphTrainer.Config:
  config = flint_40b_v012_hier_rs()
  _configure_static_ep_p11(config, ep_degree=ep_degree, tile_m=tile_m, paired=paired)
  config.parallelism.fsdp_reshard_after_forward = "never"
  config.compile.graph_split.enabled = True
  config.compile.graph_split.forward_layers_per_executable = 1
  config.compile.graph_split.backward_partition_mode = "by_layer"
  config.compile.graph_split.backward_layers_per_executable = 1
  config.compile.memory_policy = "tpu_selective"
  config.compile.save_regions = ("moe.routing",)
  config.compile.save_last_n_layers = {"attention.inner_attention": save_attn_layers}
  config.compile.optimizer_compile = "separate_graph"
  config.compile.compile_grad_norm = True
  config.optimizer.implementation = "fused"
  return config


def _build_wholestep_staticep_zero2_p11(
    *,
    ep_degree: int,
    tile_m: int,
    save_attn_layers: int,
    opt_compile: str = "separate_graph",
    paired: bool = True,
) -> TpuGraphTrainer.Config:
  config = flint_40b_v005_gt_rb()
  _configure_static_ep_p11(config, ep_degree=ep_degree, tile_m=tile_m, paired=paired)
  config.parallelism.fsdp_reshard_after_forward = "never"
  config.compile.graph_split.enabled = False
  config.compile.enable_remat_barriers = True
  config.compile.memory_policy = "tpu_selective"
  config.compile.save_regions = ("moe.routing",)
  config.compile.save_last_n_layers = {"attention.inner_attention": save_attn_layers}
  config.compile.optimizer_compile = opt_compile
  config.compile.compile_grad_norm = (opt_compile != "in_graph")
  config.optimizer.implementation = "fused"
  return config


# --- Pass 11A / 11C / 11D Configs ---
def flint_40b_p11a_split_ep8_zero2_n36() -> TpuGraphTrainer.Config:
  return _build_split_staticep_zero2_p11(ep_degree=8, tile_m=256, save_attn_layers=36)


def flint_40b_p11d_split_ep8_zero3_n36() -> TpuGraphTrainer.Config:
  cfg = _build_split_staticep_zero2_p11(ep_degree=8, tile_m=256, save_attn_layers=36)
  cfg.parallelism.fsdp_reshard_after_forward = "always"
  return cfg


def flint_40b_p11a_split_ep16_zero2_n40() -> TpuGraphTrainer.Config:
  return _build_split_staticep_zero2_p11(ep_degree=16, tile_m=256, save_attn_layers=40)


def flint_40b_p11a_split_ep32_zero2_n45() -> TpuGraphTrainer.Config:
  return _build_split_staticep_zero2_p11(ep_degree=32, tile_m=512, save_attn_layers=45)


def flint_40b_p11c_split_ep32_zero2_unpaired512_n45() -> TpuGraphTrainer.Config:
  return _build_split_staticep_zero2_p11(
      ep_degree=32, tile_m=512, save_attn_layers=45, paired=False
  )


def flint_40b_p11a_split_ep32_zero2_n36() -> TpuGraphTrainer.Config:
  return _build_split_staticep_zero2_p11(ep_degree=32, tile_m=512, save_attn_layers=36)


def flint_40b_p11a_ep1_sbdp_paired256_n36() -> TpuGraphTrainer.Config:
  config = flint_40b_v012_hier_rs()
  config.model_spec.model = (
      TpuRoutedExpertsConverter.Config().build().convert(config.model_spec.model)
  )
  for _, r, _, _ in config.model_spec.model.traverse(TpuRoutedExperts.Config):
    r.scale_before_down_proj = True
  for _, ex, _, _ in config.model_spec.model.traverse(TpuGroupedExperts.Config):
    ex.use_pallas_grouped_mm = True
    ex.use_pallas_paired_grouped_mm = True
    ex.pallas_grouped_mm_tile_m = 256
    ex.pallas_grouped_mm_tgmm_tiling = (256, 1024, 0)
    ex.pallas_grouped_mm_w2_tgmm_tiling = (256, 512, 0)
    ex.pallas_grouped_mm_w2_transposed_tgmm_tiling = (256, 0, 1024)
    ex.use_kernel_native_weight_layout = True
  if _P11_BF16_ROPE not in config.override.imports:
    config.override.imports.append(_P11_BF16_ROPE)
  config.compile.save_last_n_layers = {"attention.inner_attention": 36}
  config.compile.optimizer_compile = "separate_graph"
  config.compile.compile_grad_norm = True
  config.optimizer.implementation = "fused"
  config.libtpu_init_flags = tpu_flags.merge_flags(
      config.libtpu_init_flags,
      tpu_flags.SPARSECORE_LATENCY_HIDING_SCHEDULER,
      tpu_flags.DUAL_SPARSECORE_REDUCE_SCATTER,
      tpu_flags.ICI_AG_PIPELINING,
      _HIER_SPARSECORE_RS_FLAGS,
      _P11_V7X_BASE_FLAGS,
  )
  return config


# --- Pass 11B Configs ---
def flint_40b_p11b_wholestep_ep32_zero2_n49() -> TpuGraphTrainer.Config:
  return _build_wholestep_staticep_zero2_p11(ep_degree=32, tile_m=512, save_attn_layers=45)


def flint_40b_p11b_wholestep_ep8_zero2_n36() -> TpuGraphTrainer.Config:
  return _build_wholestep_staticep_zero2_p11(ep_degree=8, tile_m=256, save_attn_layers=36)


def flint_40b_p11b_wholestep_ep32_zero2_ingraph_n36() -> TpuGraphTrainer.Config:
  return _build_wholestep_staticep_zero2_p11(
      ep_degree=32, tile_m=512, save_attn_layers=36, opt_compile="in_graph"
  )


def flint_40b_p11b_ep1_sbdp_fulloffload() -> TpuGraphTrainer.Config:
  config = flint_40b_p11a_ep1_sbdp_paired256_n36()
  config.compile.memory_policy = "tpu_selective_and_offload"
  config.compile.save_regions = ("moe.routing",)
  config.compile.save_last_n_layers = {}
  config.compile.graph_split.cpu_offload_prefetch_executables = 2
  return config


def flint_40b_p11b_staticep16_fulloffload() -> TpuGraphTrainer.Config:
  config = flint_40b_v012_hier_rs()
  _configure_static_ep_p11(config, ep_degree=16, tile_m=512)
  config.parallelism.fsdp_reshard_after_forward = "always"
  config.compile.memory_policy = "tpu_selective_and_offload"
  config.compile.save_regions = ("moe.routing",)
  config.compile.save_last_n_layers = {}
  config.compile.graph_split.enabled = True
  config.compile.graph_split.forward_layers_per_executable = 1
  config.compile.graph_split.backward_partition_mode = "by_layer"
  config.compile.graph_split.backward_layers_per_executable = 1
  config.compile.graph_split.cpu_offload_prefetch_executables = 2
  config.compile.optimizer_compile = "separate_graph"
  config.compile.compile_grad_norm = True
  config.optimizer.implementation = "fused"
  return config


def flint_40b_p11e_wholestep_ep32_zero3_n36() -> TpuGraphTrainer.Config:
  cfg = _build_wholestep_staticep_zero2_p11(ep_degree=32, tile_m=512, save_attn_layers=36)
  cfg.parallelism.fsdp_reshard_after_forward = "always"
  return cfg


def flint_40b_p11e_split7_fsdppipe_ep32_zero3_n36() -> TpuGraphTrainer.Config:
  cfg = _build_split_staticep_zero2_p11(ep_degree=32, tile_m=512, save_attn_layers=36)
  cfg.parallelism.fsdp_reshard_after_forward = "always"
  cfg.compile.graph_split.forward_layers_per_executable = 7
  cfg.compile.graph_split.backward_partition_mode = "fsdp_pipeline"
  cfg.compile.graph_split.backward_layers_per_executable = 7
  return cfg


def flint_40b_p11e_split1_fsdppipe_ep32_zero3_n36() -> TpuGraphTrainer.Config:
  cfg = _build_split_staticep_zero2_p11(ep_degree=32, tile_m=512, save_attn_layers=36)
  cfg.parallelism.fsdp_reshard_after_forward = "always"
  cfg.compile.graph_split.backward_partition_mode = "fsdp_pipeline"
  return cfg


def flint_40b_p11e_ep1_sbdp_paired1024_n24() -> TpuGraphTrainer.Config:
  cfg = flint_40b_p11a_ep1_sbdp_paired256_n36()
  for _, ex, _, _ in cfg.model_spec.model.traverse(TpuGroupedExperts.Config):
    ex.pallas_grouped_mm_tile_m = 1024
    ex.pallas_grouped_mm_tgmm_tiling = (512, 2048, 0)
    ex.pallas_grouped_mm_w2_tgmm_tiling = (512, 2048, 0)
    ex.pallas_grouped_mm_w2_transposed_tgmm_tiling = (512, 0, 2048)
  cfg.compile.save_last_n_layers = {"attention.inner_attention": 24}
  return cfg


__all__.extend([
    "flint_40b_p11a_split_ep8_zero2_n36",
    "flint_40b_p11d_split_ep8_zero3_n36",
    "flint_40b_p11a_split_ep16_zero2_n40",
    "flint_40b_p11a_split_ep32_zero2_n45",
    "flint_40b_p11c_split_ep32_zero2_unpaired512_n45",
    "flint_40b_p11a_split_ep32_zero2_n36",
    "flint_40b_p11a_ep1_sbdp_paired256_n36",
    "flint_40b_p11b_wholestep_ep32_zero2_n49",
    "flint_40b_p11b_wholestep_ep8_zero2_n36",
    "flint_40b_p11b_wholestep_ep32_zero2_ingraph_n36",
    "flint_40b_p11b_ep1_sbdp_fulloffload",
    "flint_40b_p11b_staticep16_fulloffload",
    "flint_40b_p11e_wholestep_ep32_zero3_n36",
    "flint_40b_p11e_split7_fsdppipe_ep32_zero3_n36",
    "flint_40b_p11e_split1_fsdppipe_ep32_zero3_n36",
    "flint_40b_p11e_ep1_sbdp_paired1024_n24",
])
'''

GROUPED_EXPERTS_EP_PATCH = '''

# StaticTpuAllToAllTokenDispatcher provides static tile-aligned exact-row EP dispatch
# (`R = EP * cap_C`, `R % tile_m == 0`, zero-padded tail rows assigned to last local
# expert and stripped in `_StaticEPUnpack`), enabling `use_pallas_paired_grouped_mm=True`
# under `ep_degree > 1`.
TpuGroupedExperts.parallelize = lambda self, parallel_dims: GroupedExperts.parallelize(
    self, parallel_dims
)
'''

EP_GRAD_NORM_PATCH = '''

# --- Multi-mesh (EP + dense FSDP) CompiledGradNorm support ---
def _ep_aware_global_grad_norm(grads: list[torch.Tensor]) -> torch.Tensor:
  from torch.distributed.tensor import DTensor
  ep_grads: list[torch.Tensor] = []
  non_ep_grads: list[torch.Tensor] = []
  for g in grads:
    if (
        isinstance(g, DTensor)
        and g.device_mesh.mesh_dim_names is not None
        and "ep" in g.device_mesh.mesh_dim_names
    ):
      ep_grads.append(g)
    else:
      non_ep_grads.append(g)
  sq_terms: list[torch.Tensor] = []
  for group in (ep_grads, non_ep_grads):
    if group:
      n = torch.nn.utils.get_total_norm(
          group,
          norm_type=2.0,
          error_if_nonfinite=False,
          foreach=True,
      )
      if isinstance(n, DTensor):
        n = n.full_tensor()
      sq_terms.append(n * n)
  if not sq_terms:
    return torch.zeros((), device=grads[0].device, dtype=torch.float32)
  total_sq = sq_terms[0] + sq_terms[1] if len(sq_terms) == 2 else sq_terms[0]
  return torch.sqrt(total_sq)


global_grad_norm = _ep_aware_global_grad_norm


@torch.no_grad()
def _ep_aware_clip_grad_norm_(
    self,
    parameters: list[torch.Tensor],
    max_norm: float,
) -> torch.Tensor:
  from typing import cast
  from torch.distributed.tensor import DTensor
  grads = cast(
      list[torch.Tensor],
      [parameter.grad for parameter in parameters if parameter.grad is not None],
  )
  signature = _grad_signature(grads)
  if self._runner is None:
    if not grads:
      raise ValueError("compile_grad_norm requires at least one gradient")
    logger.info(
        "[compiled_grad_norm] compiling EP-aware global norm for %d gradients",
        len(grads),
    )
    traced = minimal_fx_tracer(_ep_aware_global_grad_norm)(grads)
    traced.gm = tpu_backend_compile_pass(
        traced.gm,
        traced.example_inputs,
        is_fwd=True,
    )
    self._runner = run_traced(traced, _validate_runtime=True)
    self._signature = signature
  elif signature != self._signature:
    raise ValueError(
        "gradient metadata changed after compiling the gradient norm: "
        f"trace-time {self._signature}, runtime {signature}"
    )
  total_norm = self._runner(grads)
  ep_params: list[torch.Tensor] = []
  non_ep_params: list[torch.Tensor] = []
  for p in parameters:
    if p.grad is None:
      continue
    if (
        isinstance(p.grad, DTensor)
        and p.grad.device_mesh.mesh_dim_names is not None
        and "ep" in p.grad.device_mesh.mesh_dim_names
    ):
      ep_params.append(p)
    else:
      non_ep_params.append(p)
  if ep_params:
    torch.nn.utils.clip_grads_with_norm_(ep_params, max_norm, total_norm, foreach=True)
  if non_ep_params:
    torch.nn.utils.clip_grads_with_norm_(non_ep_params, max_norm, total_norm, foreach=True)
  return total_norm


CompiledGradNorm.clip_grad_norm_ = _ep_aware_clip_grad_norm_
'''

SEPARATE_GRAPH_MESH_SPLIT_PATCH = '''

# --- Split mixed-mesh & oversized AdamW param_groups into homogeneous chunks ---
_orig_build_compiled_optimizer_groups = _build_compiled_optimizer_groups


def _mesh_chunked_build_compiled_optimizer_groups(opt: torch.optim.Optimizer) -> list:
  import os
  from torch.distributed.tensor import DTensor
  max_expert_chunk = int(os.environ.get("TT_OPT_EXPERT_CHUNK", "48"))
  orig_groups = list(opt.param_groups)
  expanded_groups: list[dict] = []
  group_orig_indices: list[int] = []
  for orig_idx, grp in enumerate(orig_groups):
    buckets: dict[tuple, list[torch.Tensor]] = {}
    for p in grp["params"]:
      if isinstance(p, DTensor):
        key = (id(p.device_mesh), repr(p.placements), p.dtype)
      else:
        key = (0, "local", p.dtype)
      buckets.setdefault(key, []).append(p)
    for key, plist in buckets.items():
      is_ep = any(
          isinstance(p, DTensor)
          and p.device_mesh.mesh_dim_names is not None
          and "ep" in p.device_mesh.mesh_dim_names
          for p in plist[:1]
      )
      chunk_sz = max_expert_chunk if is_ep else 96
      for start in range(0, len(plist), chunk_sz):
        sub = dict(grp)
        sub["params"] = plist[start : start + chunk_sz]
        expanded_groups.append(sub)
        group_orig_indices.append(orig_idx)
  opt.param_groups = expanded_groups
  try:
    built = _orig_build_compiled_optimizer_groups(opt)
  finally:
    opt.param_groups = orig_groups
  for (_run, _params, _states, lr_ref), orig_idx in zip(built, group_orig_indices):
    lr_ref._group_idx = orig_idx
  return built


_build_compiled_optimizer_groups = _mesh_chunked_build_compiled_optimizer_groups
'''

HF_HYBRID_AC_PATCH_V3 = '''

import os as _tt_os
import torch._dynamo

torch._dynamo.config.cache_size_limit = 256
torch._dynamo.config.accumulated_cache_size_limit = 1024


class _LayerSplashSaveOps(set):
  def __init__(self, save_attn: bool):
    super().__init__()
    self._save_attn = save_attn

  def __contains__(self, op) -> bool:
    if not self._save_attn:
      return False
    return _PALLAS_ATTENTION_FWD_MARKER in _op_name(op)


def _hybrid_attn_only_wrap_v3(self, module, *, base_fqn=None):
  self.config.preserve_rng_state = False
  n_full = int(_tt_os.environ.get("TT_FULL_AC_FIRST_N", "0"))
  self._cur_layer_save_attn = True
  if n_full > 0 and base_fqn:
    parts = [p for p in base_fqn.split(".") if p.isdigit()]
    if parts and int(parts[0]) < n_full:
      self._cur_layer_save_attn = False
  return SelectiveAC._wrap_block(self, module, base_fqn=base_fqn)


def _hybrid_get_save_ops_v3(self) -> set:
  return _LayerSplashSaveOps(getattr(self, "_cur_layer_save_attn", True))


AttentionOnlyAC._wrap_block = _hybrid_attn_only_wrap_v3
AttentionOnlyAC.get_save_ops = _hybrid_get_save_ops_v3
'''

HF_SBDP_PATCH = '''

_orig_substitute_tpu_experts = substitute_tpu_experts


def _p9_substitute_tpu_experts(model):
  import os
  _orig_substitute_tpu_experts(model)
  if os.environ.get("TT_HF_SBDP", "0") != "1":
    return
  from torchtitan.experiments.tpu.token_dispatcher import (
      TpuRoutedExperts,
      TpuRoutedExpertsConverter,
  )
  rc = TpuRoutedExpertsConverter(TpuRoutedExpertsConverter.Config())
  n_sbdp = 0
  for layer in model.layers.values():
    cfg = getattr(layer, "_native_moe_config", None)
    if cfg is not None:
      cfg = rc.convert(cfg)
      for _, r, _, _ in cfg.traverse(TpuRoutedExperts.Config):
        r.scale_before_down_proj = True
        n_sbdp += 1
      layer._native_moe_config = cfg
  if n_sbdp:
    logger.info("Enabled scale_before_down_proj on %d HF MoE layer(s).", n_sbdp)


substitute_tpu_experts = _p9_substitute_tpu_experts
'''


def probe_compile_opt_args(workdir: str) -> list[str]:
    code = (
        "from torchtitan.experiments.tpu.tpu_job_config import TPUTrainerConfig; "
        "c = TPUTrainerConfig(); out = []; "
        "hasattr(c, 'compile') and hasattr(c.compile, 'compile_grad_norm') and out.append('--compile.compile_grad_norm'); "
        "hasattr(c, 'compile') and hasattr(c.compile, 'optimizer_compile') and out.append('--compile.optimizer_compile=separate_graph'); "
        "hasattr(c, 'tpu_config') and hasattr(c.tpu_config, 'compile_grad_norm') and out.append('--tpu_config.compile_grad_norm'); "
        "hasattr(c, 'tpu_config') and hasattr(c.tpu_config, 'compile_optimizer') and out.append('--tpu_config.compile_optimizer'); "
        "print(' '.join(out))"
    )
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = workdir
        env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
        res = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=20)
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip().split()
    except Exception:
        pass
    return []


def apply_runtime_patches() -> None:
    td_path = Path("/torchtitan-native/torchtitan/experiments/tpu/token_dispatcher.py")
    if td_path.exists():
        txt = td_path.read_text()
        if "_static_tpu_ep_dispatch" not in txt:
            td_path.write_text(txt + STATIC_EP_DISPATCHER_PATCH)
            print("[flint-v7x-p11] Patched token_dispatcher.py with StaticTpuAllToAllTokenDispatcher", flush=True)

    ge_path = Path("/torchtitan-native/torchtitan/experiments/tpu/grouped_experts.py")
    if ge_path.exists():
        txt = ge_path.read_text()
        if "StaticTpuAllToAllTokenDispatcher provides static" not in txt:
            ge_path.write_text(txt + GROUPED_EXPERTS_EP_PATCH)
            print("[flint-v7x-p11] Patched grouped_experts.py to allow StaticEP + PairedGMM", flush=True)

    jc_path = Path("/torchtitan-native/torchtitan/experiments/tpu/tpu_job_config.py")
    if jc_path.exists():
        txt = jc_path.read_text()
        target = (
            "          self.parallelism.expert_parallel_degree > 1\n"
            "          or self.parallelism.pipeline_parallel_degree > 1"
        )
        if target in txt:
            jc_path.write_text(txt.replace(target, "          self.parallelism.pipeline_parallel_degree > 1"))
            print("[flint-v7x-p11] Patched tpu_job_config.py to allow compile_grad_norm with EP > 1", flush=True)

    gn_path = Path("/torchtitan-native/torchtitan/experiments/tpu/grad_norm.py")
    if gn_path.exists():
        txt = gn_path.read_text()
        if "_ep_aware_global_grad_norm" not in txt:
            gn_path.write_text(txt + EP_GRAD_NORM_PATCH)
            print("[flint-v7x-p11] Patched grad_norm.py with EP-aware CompiledGradNorm", flush=True)

    for sg_root in ("/torchtitan-native", "/torchtitan-hf"):
        sg_path = Path(f"{sg_root}/torchtitan/experiments/tpu/compiled_optimizers/separate_graph.py")
        if sg_path.exists():
            txt = sg_path.read_text()
            if "_mesh_chunked_build_compiled_optimizer_groups" not in txt:
                sg_path.write_text(txt + SEPARATE_GRAPH_MESH_SPLIT_PATCH)
                print(f"[flint-v7x-p11] Patched {sg_root} separate_graph.py with mesh-pure 96-chunk AdamW", flush=True)

    nat_reg = Path("/torchtitan-native/torchtitan/experiments/tpu/examples/flint/config_registry.py")
    if nat_reg.exists():
        txt = nat_reg.read_text()
        if "flint_40b_p11a_split_ep8_zero2_n36" not in txt:
            nat_reg.write_text(txt + NATIVE_P11_REGISTRY_PATCH)
            print("[flint-v7x-p11] Patched Native config_registry.py with Pass 11A..11F configs", flush=True)

    hf_ac = Path("/torchtitan-hf/torchtitan/experiments/tpu/activation_checkpoint.py")
    if hf_ac.exists():
        txt = hf_ac.read_text()
        if "_hybrid_attn_only_wrap_v3" not in txt:
            hf_ac.write_text(txt + HF_HYBRID_AC_PATCH_V3)
            print("[flint-v7x-p11] Patched HF activation_checkpoint.py with HybridAC v3", flush=True)

    hf_mod = Path("/torchtitan-hf/torchtitan/experiments/tpu/transformers_modeling_backend/model.py")
    if hf_mod.exists():
        txt = hf_mod.read_text()
        if "_p9_substitute_tpu_experts" not in txt:
            hf_mod.write_text(txt + HF_SBDP_PATCH)
            hf_par = Path("/torchtitan-hf/torchtitan/experiments/tpu/transformers_modeling_backend/parallelize.py")
            if hf_par.exists():
                ptxt = hf_par.read_text()
                if "from .model import substitute_tpu_experts" in ptxt:
                    hf_par.write_text(
                        ptxt.replace(
                            "from .model import substitute_tpu_experts",
                            "from . import model as _hf_model_mod\n"
                            "def substitute_tpu_experts(m): return _hf_model_mod.substitute_tpu_experts(m)",
                        )
                    )
            print("[flint-v7x-p11] Patched HF model.py & parallelize.py with TT_HF_SBDP=1 hook", flush=True)


def filter_supported_xla_flags(candidates: list[str]) -> str:
    try:
        import libtpu

        so_dir = Path(libtpu.__file__).parent
        so_files = list(so_dir.glob("*.so"))
        blob = b"".join(p.read_bytes() for p in so_files) if so_files else b""
    except Exception:
        blob = b""
    if not blob:
        return " ".join(candidates)
    kept, dropped = [], []
    for flag in candidates:
        name = flag.lstrip("-").split("=", 1)[0].encode("utf-8")
        if name in blob:
            kept.append(flag)
        else:
            dropped.append(flag)
    print(f"[flint-v7x-p11] Validated XLA flags: kept={len(kept)} dropped={dropped}", flush=True)
    return " ".join(kept)


def wait_for_master_dns(master_addr: str, timeout_s: int = 300) -> str:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            return socket.gethostbyname(master_addr)
        except socket.gaierror:
            time.sleep(3)
    return master_addr


def build_v7x_pass11_experiments(
    suite: str,
    world_size: int,
    steps: int,
    hf_model_path: str = "/models/models/RecursiveSI/flint",
) -> list[dict]:
    mb32 = 32 * 2048    # 65,536 tok/rank
    mb48 = 48 * 2048    # 98,304 tok/rank
    mb64 = 64 * 2048    # 131,072 tok/rank (3,072 tok/exp @ EP=1)
    mb80 = 80 * 2048    # 163,840 tok/rank (3,840 tok/exp @ EP=1)
    mb96 = 96 * 2048    # 196,608 tok/rank (4,608 tok/exp @ EP=1)
    mb112 = 112 * 2048  # 229,376 tok/rank (5,376 tok/exp @ EP=1)
    mb128 = 128 * 2048  # 262,144 tok/rank (6,144 tok/exp @ EP=1)

    calibrated_v7x_flags = filter_supported_xla_flags([
        "--xla_tpu_dvfs_p_state=7",
        "--xla_tpu_scoped_vmem_limit_kib=65536",
        "--xla_tpu_enable_offloading_copy_to_sparsecore=false",
        "--xla_tpu_copy_insertion_use_region_analysis=true",
        "--xla_tpu_copy_insertion_use_region_analysis_limit=1000",
        "--xla_tpu_enable_sparse_core_collective_aggregator=true",
        "--xla_tpu_enable_latency_hiding_layer_scheduler=true",
        "--xla_tpu_use_single_sparse_core_for_reduce_scatter_offload=false",
        "--xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=true",
        "--xla_tpu_enable_sparse_core_hierarchical_reduce_scatter=true",
        "--xla_tpu_prefer_async_allgather_to_allreduce=true",
        "--xla_enable_async_all_gather=true",
        "--xla_enable_async_all_reduce=true",
        "--xla_tpu_enable_async_all_to_all=true",
        "--xla_tpu_enable_ag_backward_pipelining=true",
        "--xla_tpu_enable_ici_ag_pipelining=true",
        "--xla_tpu_overlap_compute_collective_tc=true",
        "--xla_tpu_layer_scheduler_min_compute_latency_for_backward_pass=25000000",
        "--xla_max_concurrent_async_all_gathers=4",
        "--xla_latency_hiding_scheduler_rerun=2",
        "--xla_enable_transpose_trace=false",
    ])

    def _nat_exp(
        name: str,
        lane: str,
        cfg: str,
        mb: int,
        ga: int,
        align: int = 512,
        log_freq: int = 1,
        cap_factor: str = "1.12",
    ) -> dict:
        return {
            "name": name,
            "lane": lane,
            "workdir": "/torchtitan-native",
            "env": {
                "LIBTPU_INIT_ARGS": calibrated_v7x_flags,
                "TT_STATIC_EP_CAP_FACTOR": cap_factor,
                "TT_STATIC_EP_ALIGN": str(align),
                "TT_OPT_EXPERT_CHUNK": "48",
            },
            "args": [
                "--module=torchtitan.experiments.tpu.examples.flint",
                f"--config={cfg}",
                f"--training.steps={steps}",
                "--training.mixed_precision_reduce=bfloat16",
                f"--metrics.log_freq={log_freq}",
                "--debug.seed=42",
                "--tpu_config.mfu_benchmark_start_step=4",
                f"--training.num_tokens_per_microbatch_per_dp_rank={mb}",
                f"--training.num_tokens_per_train_step={ga * world_size * mb}",
            ],
        }

    if suite == "pass11f":
        hf_comp_opt = probe_compile_opt_args("/torchtitan-hf")
        hf_splash_1024 = [
            "--splash_attention_kernel.sa_block_q=1024",
            "--splash_attention_kernel.sa_block_kv=1024",
            "--splash_attention_kernel.sa_block_kv_compute=512",
            "--splash_attention_kernel.sa_block_q_dkv=1024",
            "--splash_attention_kernel.sa_block_kv_dkv=1024",
            "--splash_attention_kernel.sa_block_kv_dkv_compute=512",
        ]
        gmm_env_512 = (
            "use_pallas_paired_grouped_mm=true,"
            "pallas_grouped_mm_tile_m=512,"
            "pallas_grouped_mm_tgmm_tiling=512:1024:0,"
            "pallas_grouped_mm_w2_tgmm_tiling=256:512:0,"
            "pallas_grouped_mm_w2_transposed_tgmm_tiling=256:0:1024"
        )
        gmm_env_1024 = (
            "use_pallas_paired_grouped_mm=true,"
            "pallas_grouped_mm_tile_m=1024,"
            "pallas_grouped_mm_tgmm_tiling=1024:1024:0,"
            "pallas_grouped_mm_w2_tgmm_tiling=512:512:0,"
            "pallas_grouped_mm_w2_transposed_tgmm_tiling=512:0:1024"
        )
        hf_base = [
            "--module=torchtitan.experiments.tpu.transformers_modeling_backend",
            "--config=hf_moe_auto_max_attnac",
            f"--hf-model={hf_model_path}",
            "--training.max_context_length=2048",
            f"--training.steps={steps}",
            "--training.mixed_precision_reduce=bfloat16",
            "--optimizer.implementation=fused",
            "--lr_scheduler.warmup_steps=0",
            "--lr_scheduler.decay_ratio=0.0",
            "--metrics.log_freq=1",
            "--debug.seed=42",
            "--loss.b_block_size=1024",
            *hf_splash_1024,
            *hf_comp_opt,
        ]
        return [
            {
                "name": "v7x_p11f_01_hf_sbdp_sa1024_hybridac16_tile512_mb64_1x_log1",
                "lane": "v7x-16 HF PR#123 + SBDP + ChunkedOpt + sa=1024 + HybridAC16 + tile512 (mb=64 = 3,072 tok/exp, 1x)",
                "workdir": "/torchtitan-hf",
                "env": {
                    "TT_HF_GMM": gmm_env_512,
                    "TT_HF_SBDP": "1",
                    "TT_FULL_AC_FIRST_N": "16",
                    "TORCH_TPU_DEFER_AND_FUSE": "1",
                    "LIBTPU_INIT_ARGS": calibrated_v7x_flags,
                },
                "args": [
                    *hf_base,
                    f"--training.num_tokens_per_microbatch_per_dp_rank={mb64}",
                    f"--training.num_tokens_per_train_step={world_size * mb64}",
                ],
            },
            {
                "name": "v7x_p11f_02_hf_sbdp_sa1024_hybridac20_tile512_mb80_1x_log1",
                "lane": "v7x-16 HF PR#123 + SBDP + ChunkedOpt + sa=1024 + HybridAC20 + tile512 (mb=80 = 3,840 tok/exp, 1x)",
                "workdir": "/torchtitan-hf",
                "env": {
                    "TT_HF_GMM": gmm_env_512,
                    "TT_HF_SBDP": "1",
                    "TT_FULL_AC_FIRST_N": "20",
                    "TORCH_TPU_DEFER_AND_FUSE": "1",
                    "LIBTPU_INIT_ARGS": calibrated_v7x_flags,
                },
                "args": [
                    *hf_base,
                    f"--training.num_tokens_per_microbatch_per_dp_rank={mb80}",
                    f"--training.num_tokens_per_train_step={world_size * mb80}",
                ],
            },
            {
                "name": "v7x_p11f_03_hf_sbdp_sa1024_hybridac26_tile512_mb96_1x_log1",
                "lane": "v7x-16 HF PR#123 + SBDP + ChunkedOpt + sa=1024 + HybridAC26 + tile512 (mb=96 = 4,608 tok/exp, 1x)",
                "workdir": "/torchtitan-hf",
                "env": {
                    "TT_HF_GMM": gmm_env_512,
                    "TT_HF_SBDP": "1",
                    "TT_FULL_AC_FIRST_N": "26",
                    "TORCH_TPU_DEFER_AND_FUSE": "1",
                    "LIBTPU_INIT_ARGS": calibrated_v7x_flags,
                },
                "args": [
                    *hf_base,
                    f"--training.num_tokens_per_microbatch_per_dp_rank={mb96}",
                    f"--training.num_tokens_per_train_step={world_size * mb96}",
                ],
            },
            {
                "name": "v7x_p11f_04_hf_sbdp_sa1024_hybridac36_tile512_mb112_1x_log1",
                "lane": "v7x-16 HF PR#123 + SBDP + ChunkedOpt + sa=1024 + HybridAC36 + tile512 (mb=112 = 5,376 tok/exp, 1x)",
                "workdir": "/torchtitan-hf",
                "env": {
                    "TT_HF_GMM": gmm_env_512,
                    "TT_HF_SBDP": "1",
                    "TT_FULL_AC_FIRST_N": "36",
                    "TORCH_TPU_DEFER_AND_FUSE": "1",
                    "LIBTPU_INIT_ARGS": calibrated_v7x_flags,
                },
                "args": [
                    *hf_base,
                    f"--training.num_tokens_per_microbatch_per_dp_rank={mb112}",
                    f"--training.num_tokens_per_train_step={world_size * mb112}",
                ],
            },
            {
                "name": "v7x_p11f_05_hf_sbdp_sa1024_hybridac46_tile1024_mb128_1x_log1",
                "lane": "v7x-16 HF PR#123 + SBDP + ChunkedOpt + sa=1024 + HybridAC46 + tile1024 (mb=128 = 6,144 tok/exp, 1x)",
                "workdir": "/torchtitan-hf",
                "env": {
                    "TT_HF_GMM": gmm_env_1024,
                    "TT_HF_SBDP": "1",
                    "TT_FULL_AC_FIRST_N": "46",
                    "TORCH_TPU_DEFER_AND_FUSE": "1",
                    "LIBTPU_INIT_ARGS": calibrated_v7x_flags,
                },
                "args": [
                    *hf_base,
                    f"--training.num_tokens_per_microbatch_per_dp_rank={mb128}",
                    f"--training.num_tokens_per_train_step={world_size * mb128}",
                ],
            },
            {
                "name": "v7x_p11f_06_hf_sbdp_sa1024_hybridac26_tile512_mb96_ga2_log1",
                "lane": "v7x-16 HF PR#123 + SBDP + ChunkedOpt + sa=1024 + HybridAC26 + tile512 (mb=96 = 4,608 tok/exp, 2xGA = 12.58M tok/step)",
                "workdir": "/torchtitan-hf",
                "env": {
                    "TT_HF_GMM": gmm_env_512,
                    "TT_HF_SBDP": "1",
                    "TT_FULL_AC_FIRST_N": "26",
                    "TORCH_TPU_DEFER_AND_FUSE": "1",
                    "LIBTPU_INIT_ARGS": calibrated_v7x_flags,
                },
                "args": [
                    *hf_base,
                    f"--training.num_tokens_per_microbatch_per_dp_rank={mb96}",
                    f"--training.num_tokens_per_train_step={2 * world_size * mb96}",
                ],
            },
            _nat_exp(
                "v7x_p11f_07_split7_fsdppipe_staticep32_zero3_paired512_mb48_ga2_log1",
                "v7x-16 Split-7 fsdp_pipeline + StaticEP=32 (0 device mismatch, 0 expert AG/RS) + ZeRO-3 + PairedGMM-512 (n36, mb=48 2xGA)",
                "flint_40b_p11e_split7_fsdppipe_ep32_zero3_n36",
                mb48,
                2,
                align=512,
                log_freq=1,
            ),
        ]

    if suite == "pass11e":
        return [
            _nat_exp(
                "v7x_p11e_01_split1_zeroargsort_staticep8_zero3_paired256_mb48_ga2_log1",
                "v7x-16 Split-1 + Zero-Argsort Intra-Host StaticEP=8 + ZeRO-3 + PairedGMM-256 (0 argsorts, 0 scatter-adds, n36, mb=48 2xGA)",
                "flint_40b_p11d_split_ep8_zero3_n36",
                mb48,
                2,
                align=256,
                log_freq=1,
            ),
            _nat_exp(
                "v7x_p11e_02_split1_fsdppipe_zeroargsort_staticep32_zero3_paired512_mb48_ga2_log1",
                "v7x-16 Split-1 fsdp_pipeline + Zero-Argsort StaticEP=32 (0 device mismatch, 0 expert AG/RS) + ZeRO-3 + PairedGMM-512 (n36, mb=48 2xGA)",
                "flint_40b_p11e_split1_fsdppipe_ep32_zero3_n36",
                mb48,
                2,
                align=512,
                log_freq=1,
            ),
            _nat_exp(
                "v7x_p11e_03_split7_fsdppipe_zeroargsort_staticep32_zero3_paired512_mb48_ga2_log1",
                "v7x-16 Split-7 fsdp_pipeline + Zero-Argsort StaticEP=32 (0 device mismatch, 0 expert AG/RS, 73,728 tok/exp) + ZeRO-3 + PairedGMM-512 (n36, mb=48 2xGA)",
                "flint_40b_p11e_split7_fsdppipe_ep32_zero3_n36",
                mb48,
                2,
                align=512,
                log_freq=1,
            ),
            _nat_exp(
                "v7x_p11e_04_ep1_sbdp_paired1024_n24_mb64_ga2_log1",
                "v7x-16 EP=1 SBDP + BF16-RoPE + PairedGMM-1024/512 + n24 + mb=64 (8.39M tok/step 2xGA)",
                "flint_40b_p11e_ep1_sbdp_paired1024_n24",
                mb64,
                2,
                align=256,
                log_freq=1,
            ),
            _nat_exp(
                "v7x_p11e_05_wholestep_zeroargsort_staticep32_zero3_paired512_mb48_ga2_log1",
                "v7x-16 WholeStep Single-Graph + Zero-Argsort StaticEP=32 + ZeRO-3 + PairedGMM-512 (n36, mb=48 2xGA)",
                "flint_40b_p11e_wholestep_ep32_zero3_n36",
                mb48,
                2,
                align=512,
                log_freq=1,
            ),
        ]

    if suite == "pass11d":
        return [
            _nat_exp(
                "v7x_p11d_01_split_staticep32_zero2_compgn_paired512_mb48_ga2_log2",
                "v7x-16 GraphSplit + StaticEP=32 + ZeRO-2 + EPCompiledGradNorm + SplitMeshOpt + PairedGMM-512 (0 expert AG/RS, 73,728 tok/exp, n45, mb=48 2xGA, log2)",
                "flint_40b_p11a_split_ep32_zero2_n45",
                mb48,
                2,
                align=512,
                log_freq=2,
            ),
            _nat_exp(
                "v7x_p11d_02_split_staticep32_zero2_compgn_paired512_mb64_1x_log1",
                "v7x-16 GraphSplit + StaticEP=32 + ZeRO-2 + EPCompiledGradNorm + SplitMeshOpt + PairedGMM-512 (98,304 tok/exp, n36, mb=64 1x, log1)",
                "flint_40b_p11a_split_ep32_zero2_n36",
                mb64,
                1,
                align=512,
                log_freq=1,
            ),
            _nat_exp(
                "v7x_p11d_03_split_staticep16_zero2_compgn_paired256_mb48_ga2_log1",
                "v7x-16 GraphSplit + StaticEP=16 + ZeRO-2 + EPCompiledGradNorm + SplitMeshOpt + PairedGMM-256 (36,864 tok/exp, n40, mb=48 2xGA, log1)",
                "flint_40b_p11a_split_ep16_zero2_n40",
                mb48,
                2,
                align=256,
                log_freq=1,
            ),
            _nat_exp(
                "v7x_p11d_04_split_staticep8_zero3_compgn_paired256_mb48_ga2_log1",
                "v7x-16 GraphSplit + Intra-Host StaticEP=8 + ZeRO-3 + EPCompiledGradNorm + SplitMeshOpt + PairedGMM-256 (100% intra-host A2A, n36, mb=48 2xGA, log1)",
                "flint_40b_p11d_split_ep8_zero3_n36",
                mb48,
                2,
                align=256,
                log_freq=1,
            ),
            _nat_exp(
                "v7x_p11d_05_wholestep_staticep32_zero2_compgn_paired512_mb32_ga2_log2",
                "v7x-16 WholeStep Single-Graph + StaticEP=32 + ZeRO-2 + EPCompiledGradNorm + SplitMeshOpt + PairedGMM-512 (n45, 49,152 tok/exp, mb=32 2xGA, log2)",
                "flint_40b_p11b_wholestep_ep32_zero2_n49",
                mb32,
                2,
                align=512,
                log_freq=2,
            ),
            _nat_exp(
                "v7x_p11d_06_wholestep_staticep32_zero2_ingraph_paired512_mb48_1x_log1",
                "v7x-16 WholeStep + in_graph Fused AdamW + StaticEP=32 + ZeRO-2 + PairedGMM-512 (n36, 73,728 tok/exp, mb=48 1x, log1)",
                "flint_40b_p11b_wholestep_ep32_zero2_ingraph_n36",
                mb48,
                1,
                align=512,
                log_freq=1,
            ),
        ]

    if suite == "pass11a":
        return [
            _nat_exp(
                "v7x_p11a_01_split_staticep8_zero2_paired256_mb48_ga2",
                "v7x-16 GraphSplit + Intra-Host StaticEP=8 + ZeRO-2 + PairedGMM-256 (100% intra-host A2A, 18,432 tok/exp, mb=48 2xGA)",
                "flint_40b_p11a_split_ep8_zero2_n36",
                mb48,
                2,
                align=256,
            ),
            _nat_exp(
                "v7x_p11a_02_split_staticep16_zero2_paired256_mb48_ga2",
                "v7x-16 GraphSplit + StaticEP=16 + ZeRO-2 + PairedGMM-256 (36,864 tok/exp, 100.6MB/layer, n40, mb=48 2xGA)",
                "flint_40b_p11a_split_ep16_zero2_n40",
                mb48,
                2,
                align=256,
            ),
            _nat_exp(
                "v7x_p11a_03_split_staticep32_zero2_paired512_mb48_ga2",
                "v7x-16 GraphSplit + StaticEP=32 + ZeRO-2 + PairedGMM-512 (0 expert AG/RS, 50.3MB/layer, 73,728 tok/exp, n45, mb=48 2xGA)",
                "flint_40b_p11a_split_ep32_zero2_n45",
                mb48,
                2,
                align=512,
                log_freq=2,
            ),
            _nat_exp(
                "v7x_p11a_04_split_staticep32_zero2_paired512_mb64_1x",
                "v7x-16 GraphSplit + StaticEP=32 + ZeRO-2 + PairedGMM-512 (98,304 tok/exp, n36, mb=64 = 4.19M tok/step 1x)",
                "flint_40b_p11a_split_ep32_zero2_n36",
                mb64,
                1,
                align=512,
            ),
            _nat_exp(
                "v7x_p11a_05_ep1_sbdp_paired256_libtpuflags_log5_mb48_ga2",
                "v7x-16 EP=1 SBDP + PairedGMM-256 + Direct config.libtpu_init_flags + log_freq=5 (n36, mb=48 2xGA)",
                "flint_40b_p11a_ep1_sbdp_paired256_n36",
                mb48,
                2,
                align=256,
                log_freq=5,
            ),
        ]

    if suite == "pass11c":
        return [
            _nat_exp(
                "v7x_p11c_01_split_staticep8_zero2_paired256_mb48_ga2",
                "v7x-16 GraphSplit + Intra-Host StaticEP=8 + ZeRO-2 + PairedGMM-256 (100% intra-host A2A, 18,432 tok/exp, n36, mb=48 2xGA)",
                "flint_40b_p11a_split_ep8_zero2_n36",
                mb48,
                2,
                align=256,
            ),
            _nat_exp(
                "v7x_p11c_02_split_staticep16_zero2_paired256_mb48_ga2",
                "v7x-16 GraphSplit + StaticEP=16 + ZeRO-2 + PairedGMM-256 (36,864 tok/exp, 100.6MB/layer, n40, mb=48 2xGA)",
                "flint_40b_p11a_split_ep16_zero2_n40",
                mb48,
                2,
                align=256,
            ),
            _nat_exp(
                "v7x_p11c_03_split_staticep32_zero2_paired512_mb48_ga2",
                "v7x-16 GraphSplit + StaticEP=32 + ZeRO-2 + PairedGMM-512 (0 expert AG/RS, 50.3MB/layer, 73,728 tok/exp, n45, mb=48 2xGA)",
                "flint_40b_p11a_split_ep32_zero2_n45",
                mb48,
                2,
                align=512,
                log_freq=2,
            ),
            _nat_exp(
                "v7x_p11c_04_split_staticep32_zero2_unpaired512_mb48_ga2",
                "v7x-16 GraphSplit + StaticEP=32 + ZeRO-2 + MegaBlox UnpairedGMM-512 (0 expert AG/RS, 73,728 tok/exp, n45, mb=48 2xGA)",
                "flint_40b_p11c_split_ep32_zero2_unpaired512_n45",
                mb48,
                2,
                align=512,
                log_freq=2,
            ),
            _nat_exp(
                "v7x_p11c_05_wholestep_staticep32_zero2_paired512_mb32_ga2",
                "v7x-16 WholeStep Single-Graph + StaticEP=32 + ZeRO-2 + PairedGMM-512 (n45, 49,152 tok/exp, mb=32 2xGA)",
                "flint_40b_p11b_wholestep_ep32_zero2_n49",
                mb32,
                2,
                align=512,
                log_freq=2,
            ),
            _nat_exp(
                "v7x_p11c_06_wholestep_staticep8_zero2_paired256_mb32_ga2",
                "v7x-16 WholeStep Single-Graph + Intra-Host StaticEP=8 + ZeRO-2 + PairedGMM-256 (n36, mb=32 2xGA)",
                "flint_40b_p11b_wholestep_ep8_zero2_n36",
                mb32,
                2,
                align=256,
            ),
        ]

    return [
        _nat_exp(
            "v7x_p11b_01_wholestep_staticep32_zero2_paired512_mb32_1x",
            "v7x-16 WholeStep Single-Graph + StaticEP=32 + ZeRO-2 + PairedGMM-512 (n49, 49,152 tok/exp, mb=32 1x)",
            "flint_40b_p11b_wholestep_ep32_zero2_n49",
            mb32,
            1,
            align=512,
            log_freq=2,
        ),
        _nat_exp(
            "v7x_p11b_02_wholestep_staticep8_zero2_paired256_mb32_ga2",
            "v7x-16 WholeStep Single-Graph + Intra-Host StaticEP=8 + ZeRO-2 + PairedGMM-256 (n36, mb=32 2xGA)",
            "flint_40b_p11b_wholestep_ep8_zero2_n36",
            mb32,
            2,
            align=256,
        ),
        _nat_exp(
            "v7x_p11b_03_wholestep_staticep32_zero2_ingraph_mb48_1x",
            "v7x-16 WholeStep + in_graph Fused AdamW + StaticEP=32 + ZeRO-2 (n36, 73,728 tok/exp, mb=48 1x)",
            "flint_40b_p11b_wholestep_ep32_zero2_ingraph_n36",
            mb48,
            1,
            align=512,
        ),
        _nat_exp(
            "v7x_p11b_04_ep1_sbdp_fulloffload_mb96_ga2",
            "v7x-16 EP=1 SBDP + PairedGMM-256 + Full Host Offload (`save_last_n=0, prefetch=2`, mb=96 = 12.58M tok/step 2xGA)",
            "flint_40b_p11b_ep1_sbdp_fulloffload",
            mb96,
            2,
            align=256,
        ),
        _nat_exp(
            "v7x_p11b_05_staticep16_fulloffload_mb96_1x",
            "v7x-16 GraphSplit + StaticEP=16 + Full Host Offload (`save_last_n=0, prefetch=2`, 73,728 tok/exp, mb=96 1x)",
            "flint_40b_p11b_staticep16_fulloffload",
            mb96,
            1,
            align=512,
        ),
    ]


def _kill_proc_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    subprocess.run(["pkill", "-9", "-f", "torchtitan.experiments.tpu.train"], check=False)


def run_with_oom_watchdog(
    cmd: list[str],
    workdir: str,
    env: dict[str, str],
    node_rank: int,
    per_run_timeout: int,
    idle_timeout_s: float = 1200.0,
) -> tuple[int, list[str]]:
    log_chunks: list[str] = []
    proc = subprocess.Popen(
        cmd,
        cwd=workdir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    assert proc.stdout is not None
    state = {"last_line": time.time(), "fatal_ts": None}
    fatal_markers = (
        "RESOURCE_EXHAUSTED",
        "RuntimeProgramAllocationFailure",
        "Ran out of memory in memory space",
        "Out of memory",
        "ChildFailedError",
        "local_exit_code: 1",
    )

    def _watchdog() -> None:
        t_start = time.time()
        while proc.poll() is None:
            now = time.time()
            if now - t_start > per_run_timeout:
                print(f"[watchdog] per_run_timeout ({per_run_timeout}s) reached; killing process group", flush=True)
                _kill_proc_group(proc)
                return
            if now - state["last_line"] > idle_timeout_s:
                print(f"[watchdog] idle_timeout ({idle_timeout_s}s) reached; killing process group", flush=True)
                _kill_proc_group(proc)
                return
            if state["fatal_ts"] is not None and now - state["fatal_ts"] > 10.0:
                print("[watchdog] fatal OOM/VMEM/error detected >10s ago; killing process group", flush=True)
                _kill_proc_group(proc)
                return
            time.sleep(2.0)

    wt = threading.Thread(target=_watchdog, daemon=True)
    wt.start()
    for line in proc.stdout:
        state["last_line"] = time.time()
        if state["fatal_ts"] is None and any(m in line for m in fatal_markers):
            state["fatal_ts"] = time.time()
        sys.stdout.write(line)
        sys.stdout.flush()
        if node_rank == 0:
            log_chunks.append(line)
    rc = proc.wait(timeout=30)
    subprocess.run(["pkill", "-9", "-f", "torchtitan.experiments.tpu.train"], check=False)
    return rc, log_chunks


def parse_metrics(log_text: str, chip_type: str) -> dict:
    steps = []
    for line in log_text.splitlines():
        m = STEP_RE.search(line)
        if not m:
            continue
        step = int(m.group("step"))
        if steps and steps[-1]["step"] == step:
            continue
        loss = float(m.group("loss"))
        tps = float(m.group("tps").replace(",", ""))
        mfu_m = MFU_RE.search(line)
        mem_m = MEM_RE.search(line)
        peak_tf = PEAK_BF16_TFLOPS_PER_RANK.get(chip_type, 1153.5) * 1e12
        calc_mfu = 100.0 * FLINT_FLOPS_PER_TOKEN * tps / peak_tf
        logged_mfu = float(mfu_m.group("mfu")) if mfu_m else calc_mfu
        mem_gib = float(mem_m.group("mem")) if mem_m else None
        steps.append(
            {
                "step": step,
                "loss": loss,
                "tps_per_rank": tps,
                "logged_mfu_pct": round(logged_mfu, 2),
                "mfu_pct": round(calc_mfu, 2),
                "mem_gib": mem_gib,
            }
        )
    steady = [s for s in steps if s["step"] >= 6] or [s for s in steps if s["step"] >= 4] or steps[-2:]
    if not steady:
        return {"num_logged_steps": 0, "steady_tps": 0.0, "steady_mfu_pct": 0.0}
    avg_tps = sum(s["tps_per_rank"] for s in steady) / len(steady)
    max_tps = max(s["tps_per_rank"] for s in steady)
    peak_tf = PEAK_BF16_TFLOPS_PER_RANK.get(chip_type, 1153.5) * 1e12
    return {
        "num_logged_steps": len(steps),
        "first_loss": steps[0]["loss"],
        "final_loss": steps[-1]["loss"],
        "steady_tps": round(avg_tps, 1),
        "peak_tps": round(max_tps, 1),
        "steady_mfu_pct": round(100.0 * FLINT_FLOPS_PER_TOKEN * avg_tps / peak_tf, 2),
        "peak_mfu_pct": round(100.0 * FLINT_FLOPS_PER_TOKEN * max_tps / peak_tf, 2),
        "peak_mem_gib": max((s["mem_gib"] for s in steps if s["mem_gib"] is not None), default=None),
        "steps": steps,
    }


def write_summary(
    out_dir: Path, bench_dir: Path, suite: str, slice_name: str, chip_type: str, world_size: int, results: list[dict]
) -> None:
    payload = {
        "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "suite": suite,
        "slice_name": slice_name,
        "chip_type": chip_type,
        "world_size_ranks": world_size,
        "flops_per_token": FLINT_FLOPS_PER_TOKEN,
        "peak_bf16_tflops_per_rank": PEAK_BF16_TFLOPS_PER_RANK.get(chip_type),
        "results": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    md_lines = [
        f"# Flint (`RecursiveSI/flint` 40B MoE) TPU v7x `{suite.upper()}` (`StaticEP=8/16/32 ZeRO-2` + `WholeStep` + `Offload`) — `{slice_name}`",
        "",
        f"- **Updated (UTC):** `{payload['updated_at_utc']}`",
        f"- **Chip / World Size:** `{chip_type}` (`{world_size}` TensorCore ranks, `{PEAK_BF16_TFLOPS_PER_RANK.get(chip_type)}` BF16 TFLOP/s/rank)",
        "",
        "| Experiment | Lane | Status | Steady tok/s/rank | Steady MFU (%) | Peak MFU (%) | Peak HBM (GiB) | Loss (start -> end) | Wall Time (s) |",
        "|---|---|---|---:|---:|---:|---:|---|---:|",
    ]
    for r in results:
        m = r.get("metrics", {})
        loss_str = f"{m.get('first_loss', '-')} -> {m.get('final_loss', '-')}" if m.get("num_logged_steps") else "-"
        md_lines.append(
            f"| `{r['name']}` | {r['lane']} | `{r['status']}` | "
            f"**{m.get('steady_tps', 0.0):,.1f}** | **{m.get('steady_mfu_pct', 0.0):.2f}%** | "
            f"{m.get('peak_mfu_pct', 0.0):.2f}% | {m.get('peak_mem_gib') or '-'} | {loss_str} | {r['duration_s']:.1f} |"
        )
    md_text = "\n".join(md_lines) + "\n"
    (out_dir / "SUMMARY.md").write_text(md_text)
    if bench_dir.exists():
        (bench_dir / f"flint_{slice_name}_{suite}_summary.json").write_text(json.dumps(payload, indent=2))
        (bench_dir / f"flint_{slice_name}_{suite}_SUMMARY.md").write_text(md_text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite", choices=["pass11a", "pass11b", "pass11c", "pass11d", "pass11e", "pass11f"], default="pass11a"
    )
    parser.add_argument("--chip-type", choices=["v7x", "v5p", "v6e"], default="v7x")
    parser.add_argument("--nnodes", type=int, required=True)
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument("--master-addr", type=str, required=True)
    parser.add_argument("--slice-name", type=str, required=True)
    parser.add_argument("--hf-model", type=str, default="/models/models/RecursiveSI/flint")
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--base-port", type=int, default=29711)
    parser.add_argument("--per-run-timeout", type=int, default=2400)
    parser.add_argument("--output-dir", type=str, default="/models/flint-results")
    args = parser.parse_args()

    apply_runtime_patches()
    node_rank = int(os.environ.get("JOB_COMPLETION_INDEX", os.environ.get("NODE_RANK", "0")))
    world_size = args.nnodes * args.nproc_per_node
    wait_for_master_dns(args.master_addr)

    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir) / f"{args.slice_name}_{args.suite}_{ts}"
    bench_dir = Path("/models/benchmark-results")
    if node_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        bench_dir.mkdir(parents=True, exist_ok=True)

    exps = build_v7x_pass11_experiments(args.suite, world_size, args.steps, args.hf_model)
    results = []
    for idx, exp in enumerate(exps):
        port = args.base_port + idx * 5
        cmd = [
            "torchrun",
            f"--nnodes={args.nnodes}",
            f"--nproc_per_node={args.nproc_per_node}",
            "--rdzv_backend=static",
            f"--rdzv_endpoint={args.master_addr}:{port}",
            f"--node_rank={node_rank}",
            "-m",
            "torchtitan.experiments.tpu.train",
            *exp["args"],
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = exp["workdir"]
        env.update(exp["env"])
        print(f"\n=== [{idx + 1}/{len(exps)}] {exp['name']} ({exp['lane']}) ===", flush=True)
        t0 = time.time()
        try:
            rc, log_chunks = run_with_oom_watchdog(
                cmd, exp["workdir"], env, node_rank, args.per_run_timeout, idle_timeout_s=1200.0
            )
        except Exception as exc:
            rc = -1
            log_chunks = [f"[flint-v7x-p11] Exception: {exc}\n"]
        dur = time.time() - t0
        if node_rank == 0:
            raw_log = "".join(log_chunks)
            (out_dir / f"{exp['name']}.log").write_text(raw_log)
            metrics = parse_metrics(raw_log, args.chip_type)
            status = "PASS" if (rc == 0 and metrics.get("num_logged_steps", 0) > 0) else f"FAIL(rc={rc})"
            results.append(
                {
                    "name": exp["name"],
                    "lane": exp["lane"],
                    "status": status,
                    "return_code": rc,
                    "duration_s": round(dur, 1),
                    "command": cmd,
                    "env_overrides": exp["env"],
                    "metrics": metrics,
                }
            )
            write_summary(out_dir, bench_dir, args.suite, args.slice_name, args.chip_type, world_size, results)
            print(
                f"[flint-v7x-{args.suite}] SAVED {exp['name']}: status={status} "
                f"steady_tps={metrics.get('steady_tps', 0)} steady_mfu={metrics.get('steady_mfu_pct', 0)}%",
                flush=True,
            )
        time.sleep(8)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
