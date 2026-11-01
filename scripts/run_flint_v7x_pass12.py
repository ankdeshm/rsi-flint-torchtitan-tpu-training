#!/usr/bin/env python3
"""Pass 12 TPU v7x-16 (2x2x4, 16 Ironwood chips / 32 TensorCores) MFU Frontier & EP/DP/TP Matrix for Flint.

Unlocks the high-impact optimizations & parallelism combinations left untested when Pass 11 pivoted:
  1. Fixes the Pass 11A `MfuBenchmark` boundary bug (`mfu_benchmark_start_step=5`, `steps=20` with `log_freq=5`)
     and Pass 9's `TpuGraphTrainer` `config.libtpu_init_flags` overwrite so Native `EP=1` finally gets:
     - `tile_m=1024` (`tgmm=(512,2048,0)`, `w2=(512,512,0)`, `w2_T=(512,0,2048)`)
     - `TpuRoutedExpertsConverter(scale_before_down_proj=True)` (`SBDP` — eliminates 47-layer bwd W2 recompute)
     - `BF16-RoPE` (`tpu_activation_dtype_cossin_rope`)
     - Pure single-graph `FusedCompiledOpt` + `CompiledGradNorm` (no 96-param sub-chunking when `EP=1`!)
     - Direct `config.libtpu_init_flags` (`SPARSECORE_LATENCY_HIDING_SCHEDULER`, `DUAL_SPARSECORE_REDUCE_SCATTER`,
       `ICI_AG_PIPELINING`, `HierSC`, `RegionAnalysis`, NO `scoped_vmem_limit_kib` override -> preserves full
       `34.58 GiB` bottom-of-HBM executable reservation!)
     - Tested across `mb=48` (`n45 1x`, `n43 2xGA`, `n45 4xGA` = 12.58M tok/step) and `mb=64` (`n36 2xGA` = 8.39M tok/step).
  2. Upgrades Pass 11F's HF PR#123 record (`18.39%` at `mb=64 1x tile512`) with `tile_m=1024` (`tgmm=512:2048:0`),
     Pure `FusedCompiledOpt`, `Splash sa=1024`, `SBDP`, `2x` GradAccum, `log_freq=5` at `mb=64` (`HybridAC N=12`)
     and `mb=72` (`HybridAC N=16`, safely below the `34.58 GiB` bottom-of-HBM slab limit).
  3. Completes the systematic `(EP, DP/FSDP, HSDP, TP)` parallelism matrix on `v7x-16`:
     - `HSDP (DP_rep=2 × FSDP=16)` (`mb=64 2xGA`)
     - `Intra-Chip TP=2 × FSDP=16` (`100%` intra-socket dual-TensorCore TP, `mb=64 2xGA`)
     - `Intra-Chip StaticEP=2 × ZeRO-3` (`100%` intra-socket dual-TensorCore AllToAll, 7-layer `fsdp_pipeline`)
     - `Intra-Tray StaticEP=4 × ZeRO-3` (`2`-chip AllToAll, `9,216` tok/expert, 7-layer `fsdp_pipeline`)
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
# Pass 12 StaticTpuAllToAllTokenDispatcher (100% Static-Shape, Scatter-Free TPU EP)
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
  cap_factor = float(_sep_os.environ.get("TT_STATIC_EP_CAP_FACTOR", "1.12"))
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
          == torch.arange(e, device=recv_lexp.device, dtype=recv_lexp.dtype)
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

NATIVE_P12_REGISTRY_PATCH = '''

# ==============================================================================
# Pass 12 Flint v7x-16 MFU Frontier & EP/DP/TP Configs
# ==============================================================================
from torchtitan.experiments.tpu.token_dispatcher import (
    TpuAllToAllTokenDispatcher,
    TpuRoutedExperts,
    TpuRoutedExpertsConverter,
)

_P12_BF16_ROPE = (
    "torchtitan.experiments.tpu.overrides.bf16_rope"
    ".tpu_activation_dtype_cossin_rope"
)

# NOTE: Intentionally omits xla_tpu_scoped_vmem_limit_kib override so libtpu preserves
# the full 34.58 GiB bottom-of-HBM contiguous executable reservation on Ironwood!
_P12_V7X_SCOUT_FLAGS = {
    "xla_tpu_dvfs_p_state": 7,
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


def _build_p12_ep1_sbdp_tile1024(save_attn_layers: int) -> TpuGraphTrainer.Config:
  config = flint_40b_v014_tile1024()
  config.model_spec.model = (
      TpuRoutedExpertsConverter.Config().build().convert(config.model_spec.model)
  )
  for _, r, _, _ in config.model_spec.model.traverse(TpuRoutedExperts.Config):
    r.scale_before_down_proj = True
  for _, ex, _, _ in config.model_spec.model.traverse(TpuGroupedExperts.Config):
    ex.use_pallas_grouped_mm = True
    ex.use_pallas_paired_grouped_mm = True
    ex.pallas_grouped_mm_tile_m = 1024
    ex.pallas_grouped_mm_tgmm_tiling = (512, 2048, 0)
    ex.pallas_grouped_mm_w2_tgmm_tiling = (512, 512, 0)
    ex.pallas_grouped_mm_w2_transposed_tgmm_tiling = (512, 0, 2048)
    ex.use_kernel_native_weight_layout = True
  if _P12_BF16_ROPE not in config.override.imports:
    config.override.imports.append(_P12_BF16_ROPE)
  config.compile.memory_policy = "tpu_selective"
  config.compile.save_regions = ("moe.routing",)
  config.compile.save_last_n_layers = {"attention.inner_attention": save_attn_layers}
  config.compile.optimizer_compile = "separate_graph"
  config.compile.compile_grad_norm = True
  config.optimizer.implementation = "fused"
  config.libtpu_init_flags = tpu_flags.merge_flags(
      config.libtpu_init_flags,
      tpu_flags.SPARSECORE_LATENCY_HIDING_SCHEDULER,
      tpu_flags.DUAL_SPARSECORE_REDUCE_SCATTER,
      tpu_flags.ICI_AG_PIPELINING,
      _HIER_SPARSECORE_RS_FLAGS,
      _P12_V7X_SCOUT_FLAGS,
  )
  return config


def _build_p12_staticep(ep_degree: int, tile_m: int, save_attn_layers: int) -> TpuGraphTrainer.Config:
  config = flint_40b_v012_hier_rs()
  config.parallelism.expert_parallel_degree = ep_degree
  config.parallelism.fsdp_reshard_after_forward = "always"
  for _, td, _, _ in config.model_spec.model.traverse(TpuAllToAllTokenDispatcher.Config):
    td.fuse_route_score_backward = True
  for _, ex, _, _ in config.model_spec.model.traverse(TpuGroupedExperts.Config):
    ex.use_pallas_grouped_mm = True
    ex.use_pallas_paired_grouped_mm = True
    ex.pallas_grouped_mm_tile_m = tile_m
    ex.pallas_grouped_mm_tgmm_tiling = (512, 2048, 0) if tile_m >= 512 else (256, 1024, 0)
    ex.pallas_grouped_mm_w2_tgmm_tiling = (256, 512, 0)
    ex.pallas_grouped_mm_w2_transposed_tgmm_tiling = (256, 0, 1024)
    ex.use_kernel_native_weight_layout = True
  if _P12_BF16_ROPE not in config.override.imports:
    config.override.imports.append(_P12_BF16_ROPE)
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
  config.libtpu_init_flags = tpu_flags.merge_flags(
      config.libtpu_init_flags,
      tpu_flags.SPARSECORE_LATENCY_HIDING_SCHEDULER,
      tpu_flags.DUAL_SPARSECORE_REDUCE_SCATTER,
      tpu_flags.ICI_AG_PIPELINING,
      _HIER_SPARSECORE_RS_FLAGS,
      _P12_V7X_SCOUT_FLAGS,
  )
  return config


def flint_40b_p12_ep1_sbdp_tile1024_n45() -> TpuGraphTrainer.Config:
  return _build_p12_ep1_sbdp_tile1024(45)


def flint_40b_p12_ep1_sbdp_tile1024_n43() -> TpuGraphTrainer.Config:
  return _build_p12_ep1_sbdp_tile1024(43)


def flint_40b_p12_ep1_sbdp_tile1024_n36() -> TpuGraphTrainer.Config:
  return _build_p12_ep1_sbdp_tile1024(36)


def flint_40b_p12_staticep2_zero3_n43() -> TpuGraphTrainer.Config:
  return _build_p12_staticep(ep_degree=2, tile_m=512, save_attn_layers=43)


def flint_40b_p12_staticep4_zero3_n40() -> TpuGraphTrainer.Config:
  return _build_p12_staticep(ep_degree=4, tile_m=512, save_attn_layers=40)


def _build_p12c_ep1_p9base_sbdp(save_attn_layers: int) -> TpuGraphTrainer.Config:
  # Preserves flint_40b_v014_tile1024()'s exact libtpu_init_flags, GMM tiling, and
  # for_each separate_graph optimizer from all-time #1 v7x_p9_02 (18.68% MFU),
  # adding ONLY SBDP (eliminates 47-layer bwd W2 recompute) + BF16-RoPE!
  config = flint_40b_v014_tile1024()
  config.model_spec.model = (
      TpuRoutedExpertsConverter.Config().build().convert(config.model_spec.model)
  )
  for _, r, _, _ in config.model_spec.model.traverse(TpuRoutedExperts.Config):
    r.scale_before_down_proj = True
  if _P12_BF16_ROPE not in config.override.imports:
    config.override.imports.append(_P12_BF16_ROPE)
  config.compile.memory_policy = "tpu_selective"
  config.compile.save_regions = ("moe.routing",)
  config.compile.save_last_n_layers = {"attention.inner_attention": save_attn_layers}
  config.compile.optimizer_compile = "separate_graph"
  config.compile.compile_grad_norm = True
  return config


def flint_40b_p12c_ep1_p9base_sbdp_n41() -> TpuGraphTrainer.Config:
  return _build_p12c_ep1_p9base_sbdp(41)


def flint_40b_p12c_ep1_p9base_sbdp_n30() -> TpuGraphTrainer.Config:
  return _build_p12c_ep1_p9base_sbdp(30)


def flint_40b_p12c_staticep2_zero3_n36() -> TpuGraphTrainer.Config:
  return _build_p12_staticep(ep_degree=2, tile_m=512, save_attn_layers=36)


def flint_40b_p12c_staticep2_zero2_n36() -> TpuGraphTrainer.Config:
  cfg = _build_p12_staticep(ep_degree=2, tile_m=512, save_attn_layers=36)
  cfg.parallelism.fsdp_reshard_after_forward = "default"
  return cfg


def flint_40b_p12d_ep1_p9base_sbdp_n47() -> TpuGraphTrainer.Config:
  return _build_p12c_ep1_p9base_sbdp(47)


def flint_40b_p12d_ep1_hostoffload_sbdp() -> TpuGraphTrainer.Config:
  from .config_registry import flint_40b_v014_tile512_host_offload
  cfg = flint_40b_v014_tile512_host_offload()
  cfg.model_spec.model = (
      TpuRoutedExpertsConverter.Config().build().convert(cfg.model_spec.model)
  )
  for _, r, _, _ in cfg.model_spec.model.traverse(TpuRoutedExperts.Config):
    r.scale_before_down_proj = True
  if _P12_BF16_ROPE not in cfg.override.imports:
    cfg.override.imports.append(_P12_BF16_ROPE)
  cfg.compile.optimizer_compile = "separate_graph"
  cfg.compile.compile_grad_norm = True
  return cfg


def flint_40b_p12d_staticep2_zero2_sbdp_n30() -> TpuGraphTrainer.Config:
  cfg = _build_p12_staticep(ep_degree=2, tile_m=512, save_attn_layers=30)
  cfg.model_spec.model = (
      TpuRoutedExpertsConverter.Config().build().convert(cfg.model_spec.model)
  )
  for _, r, _, _ in cfg.model_spec.model.traverse(TpuRoutedExperts.Config):
    r.scale_before_down_proj = True
  cfg.parallelism.fsdp_reshard_after_forward = "default"
  return cfg


def flint_40b_p12e_ep1_p9base_sbdp_n49() -> TpuGraphTrainer.Config:
  return _build_p12c_ep1_p9base_sbdp(49)


def flint_40b_p12e_ep1_p9base_sbdp_fulloffload() -> TpuGraphTrainer.Config:
  cfg = _build_p12c_ep1_p9base_sbdp(0)
  cfg.compile.memory_policy = "tpu_selective_and_offload"
  cfg.compile.save_regions = ("moe.routing",)
  cfg.compile.save_last_n_layers = {}
  cfg.compile.graph_split.cpu_offload_prefetch_executables = 2
  return cfg


def flint_40b_p12e_staticep2_zero2_n49() -> TpuGraphTrainer.Config:
  cfg = _build_p12_staticep(ep_degree=2, tile_m=512, save_attn_layers=49)
  cfg.parallelism.fsdp_reshard_after_forward = "default"
  return cfg


def flint_40b_p12f_staticep2_zero3_sbdp_n43() -> TpuGraphTrainer.Config:
  cfg = _build_p12_staticep(ep_degree=2, tile_m=512, save_attn_layers=43)
  cfg.model_spec.model = (
      TpuRoutedExpertsConverter.Config().build().convert(cfg.model_spec.model)
  )
  for _, r, _, _ in cfg.model_spec.model.traverse(TpuRoutedExperts.Config):
    r.scale_before_down_proj = True
  if _P12_BF16_ROPE not in cfg.override.imports:
    cfg.override.imports.append(_P12_BF16_ROPE)
  return cfg


def flint_40b_p12g_ep1_p9base_sbdp_fusedopt_n49() -> TpuGraphTrainer.Config:
  cfg = _build_p12c_ep1_p9base_sbdp(49)
  cfg.optimizer.implementation = "fused"
  return cfg


def flint_40b_p12g_staticep2_zero3_sbdp_n43() -> TpuGraphTrainer.Config:
  cfg = _build_p12_staticep(ep_degree=2, tile_m=512, save_attn_layers=43)
  for _, td, _, _ in cfg.model_spec.model.traverse(TpuAllToAllTokenDispatcher.Config):
    td.fuse_route_score_backward = False
  cfg.model_spec.model = (
      TpuRoutedExpertsConverter.Config().build().convert(cfg.model_spec.model)
  )
  for _, r, _, _ in cfg.model_spec.model.traverse(TpuRoutedExperts.Config):
    r.scale_before_down_proj = True
  if _P12_BF16_ROPE not in cfg.override.imports:
    cfg.override.imports.append(_P12_BF16_ROPE)
  return cfg


__all__.extend([
    "flint_40b_p12_ep1_sbdp_tile1024_n45",
    "flint_40b_p12_ep1_sbdp_tile1024_n43",
    "flint_40b_p12_ep1_sbdp_tile1024_n36",
    "flint_40b_p12_staticep2_zero3_n43",
    "flint_40b_p12_staticep4_zero3_n40",
    "flint_40b_p12c_ep1_p9base_sbdp_n41",
    "flint_40b_p12c_ep1_p9base_sbdp_n30",
    "flint_40b_p12c_staticep2_zero3_n36",
    "flint_40b_p12c_staticep2_zero2_n36",
    "flint_40b_p12d_ep1_p9base_sbdp_n47",
    "flint_40b_p12d_ep1_hostoffload_sbdp",
    "flint_40b_p12d_staticep2_zero2_sbdp_n30",
    "flint_40b_p12e_ep1_p9base_sbdp_n49",
    "flint_40b_p12e_ep1_p9base_sbdp_fulloffload",
    "flint_40b_p12e_staticep2_zero2_n49",
    "flint_40b_p12f_staticep2_zero3_sbdp_n43",
    "flint_40b_p12g_ep1_p9base_sbdp_fusedopt_n49",
    "flint_40b_p12g_staticep2_zero3_sbdp_n43",
])
'''

GROUPED_EXPERTS_EP_PATCH = '''

TpuGroupedExperts.parallelize = lambda self, parallel_dims: GroupedExperts.parallelize(
    self, parallel_dims
)
'''

EP_GRAD_NORM_PATCH = '''

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
  if not ep_grads:
    n = torch.nn.utils.get_total_norm(
        non_ep_grads, norm_type=2.0, error_if_nonfinite=False, foreach=True
    )
    return n.full_tensor() if isinstance(n, DTensor) else n
  sq_terms: list[torch.Tensor] = []
  for group in (ep_grads, non_ep_grads):
    if group:
      n = torch.nn.utils.get_total_norm(
          group, norm_type=2.0, error_if_nonfinite=False, foreach=True
      )
      if isinstance(n, DTensor):
        n = n.full_tensor()
      sq_terms.append(n * n)
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
    traced = minimal_fx_tracer(_ep_aware_global_grad_norm)(grads)
    traced.gm = tpu_backend_compile_pass(
        traced.gm, traced.example_inputs, is_fwd=True
    )
    self._runner = run_traced(traced, _validate_runtime=True)
    self._signature = signature
  elif signature != self._signature:
    raise ValueError(f"gradient metadata changed: {self._signature} vs {signature}")
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

# Preserve 100% pure single-graph compiled AdamW when EP=1; only split meshes when EP > 1!
_orig_build_compiled_optimizer_groups = _build_compiled_optimizer_groups


def _mesh_chunked_build_compiled_optimizer_groups(opt: torch.optim.Optimizer) -> list:
  import os
  from torch.distributed.tensor import DTensor
  has_ep = any(
      isinstance(p, DTensor)
      and p.device_mesh.mesh_dim_names is not None
      and "ep" in p.device_mesh.mesh_dim_names
      for grp in opt.param_groups
      for p in grp["params"]
  )
  if not has_ep and os.environ.get("TT_FORCE_CHUNK_OPT", "0") != "1":
    return _orig_build_compiled_optimizer_groups(opt)

  max_expert_chunk = int(os.environ.get("TT_OPT_EXPERT_CHUNK", "48"))
  orig_groups = list(opt.param_groups)
  expanded_groups: list[dict] = []
  group_orig_indices: list[int] = []
  for orig_idx, grp in enumerate(orig_groups):
    buckets: dict[tuple, list[torch.Tensor]] = {}
    for p in grp["params"]:
      key = (id(p.device_mesh), repr(p.placements), p.dtype) if isinstance(p, DTensor) else (0, "local", p.dtype)
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
            print("[flint-v7x-p12] Patched token_dispatcher.py with StaticTpuAllToAllTokenDispatcher", flush=True)

    ge_path = Path("/torchtitan-native/torchtitan/experiments/tpu/grouped_experts.py")
    if ge_path.exists():
        txt = ge_path.read_text()
        if "GroupedExperts.parallelize" not in txt or "lambda self, parallel_dims" not in txt:
            ge_path.write_text(txt + GROUPED_EXPERTS_EP_PATCH)
            print("[flint-v7x-p12] Patched grouped_experts.py for StaticEP + PairedGMM", flush=True)

    jc_path = Path("/torchtitan-native/torchtitan/experiments/tpu/tpu_job_config.py")
    if jc_path.exists():
        txt = jc_path.read_text()
        target = (
            "          self.parallelism.expert_parallel_degree > 1\n"
            "          or self.parallelism.pipeline_parallel_degree > 1"
        )
        if target in txt:
            jc_path.write_text(txt.replace(target, "          self.parallelism.pipeline_parallel_degree > 1"))
            print("[flint-v7x-p12] Patched tpu_job_config.py for EP > 1 compile_grad_norm", flush=True)

    gn_path = Path("/torchtitan-native/torchtitan/experiments/tpu/grad_norm.py")
    if gn_path.exists():
        txt = gn_path.read_text()
        if "_ep_aware_global_grad_norm" not in txt:
            gn_path.write_text(txt + EP_GRAD_NORM_PATCH)
            print("[flint-v7x-p12] Patched grad_norm.py with EP-aware CompiledGradNorm", flush=True)

    for sg_root in ("/torchtitan-native", "/torchtitan-hf"):
        sg_path = Path(f"{sg_root}/torchtitan/experiments/tpu/compiled_optimizers/separate_graph.py")
        if sg_path.exists():
            txt = sg_path.read_text()
            if "_mesh_chunked_build_compiled_optimizer_groups" not in txt:
                sg_path.write_text(txt + SEPARATE_GRAPH_MESH_SPLIT_PATCH)
                print(f"[flint-v7x-p12] Patched {sg_root} separate_graph.py (pure 1-graph for EP=1)", flush=True)

    nat_reg = Path("/torchtitan-native/torchtitan/experiments/tpu/examples/flint/config_registry.py")
    if nat_reg.exists():
        txt = nat_reg.read_text()
        if "flint_40b_p12_ep1_sbdp_tile1024_n45" not in txt:
            nat_reg.write_text(txt + NATIVE_P12_REGISTRY_PATCH)
            print("[flint-v7x-p12] Patched Native config_registry.py with Pass 12 configs", flush=True)

    hf_ac = Path("/torchtitan-hf/torchtitan/experiments/tpu/activation_checkpoint.py")
    if hf_ac.exists():
        txt = hf_ac.read_text()
        if "_hybrid_attn_only_wrap_v3" not in txt:
            hf_ac.write_text(txt + HF_HYBRID_AC_PATCH_V3)
            print("[flint-v7x-p12] Patched HF activation_checkpoint.py with HybridAC v3", flush=True)

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
            print("[flint-v7x-p12] Patched HF model.py & parallelize.py with TT_HF_SBDP=1 hook", flush=True)

    hf_sh = Path("/torchtitan-hf/torchtitan/experiments/transformers_modeling_backend/hf_sharding.py")
    if hf_sh.exists():
        txt = hf_sh.read_text()
        target_attn = (
            "    attn._sharding_config = ShardingConfig(\n"
            "        in_src_shardings={"
        )
        repl_attn = (
            "    attn._sharding_config = ShardingConfig(\n"
            '        state_shardings={"sinks": dense_param_placement(tp=spmd.S(0))} if hasattr(attn, "sinks") else {},\n'
            "        in_src_shardings={"
        )
        if target_attn in txt and 'state_shardings={"sinks"' not in txt:
            hf_sh.write_text(txt.replace(target_attn, repl_attn))
            print("[flint-v7x-p12] Patched HF hf_sharding.py for TP=2 FlintAttention.sinks", flush=True)

    hf_at = Path("/torchtitan-hf/torchtitan/experiments/tpu/transformers_modeling_backend/attention.py")
    if hf_at.exists():
        txt = hf_at.read_text()
        if "_dtensor_sbd_splash_wrap" not in txt:
            dt_patch = '''

# _dtensor_sbd_splash_wrap: support DTensor inputs (e.g. TP=2 head-sharded q/k/v + sinks)
_orig_sbd_forward = _sbd_forward


def _dtensor_sbd_splash_wrap(query, key, value, segment_ids, is_causal, scaling, window,
                             dropout, kernel_kwargs, seg_desc, softcap=None, sinks=None):
    from torch.distributed.tensor import DTensor
    if not isinstance(query, DTensor):
        return _orig_sbd_forward(
            query, key, value, segment_ids, is_causal, scaling, window,
            dropout, kernel_kwargs, seg_desc, softcap=softcap, sinks=sinks,
        )
    from torch.distributed.tensor.experimental import local_map
    mesh = query.device_mesh
    q_pl = query.placements
    k_pl = key.placements if isinstance(key, DTensor) else None
    v_pl = value.placements if isinstance(value, DTensor) else None
    if sinks is not None and isinstance(sinks, DTensor):
        s_pl = sinks.placements
        fn = local_map(
            lambda ql, kl, vl, sl: _orig_sbd_forward(
                ql, kl, vl,
                segment_ids.to_local() if isinstance(segment_ids, DTensor) else segment_ids,
                is_causal, scaling, window, dropout, kernel_kwargs, seg_desc,
                softcap=softcap, sinks=sl,
            ),
            out_placements=q_pl,
            in_placements=(q_pl, k_pl, v_pl, s_pl),
            device_mesh=mesh,
        )
        return fn(query, key, value, sinks)
    fn = local_map(
        lambda ql, kl, vl: _orig_sbd_forward(
            ql, kl, vl,
            segment_ids.to_local() if isinstance(segment_ids, DTensor) else segment_ids,
            is_causal, scaling, window, dropout, kernel_kwargs, seg_desc,
            softcap=softcap, sinks=sinks,
        ),
        out_placements=q_pl,
        in_placements=(q_pl, k_pl, v_pl),
        device_mesh=mesh,
    )
    return fn(query, key, value)


_sbd_forward = _dtensor_sbd_splash_wrap
'''
            hf_at.write_text(txt + dt_patch)
            print("[flint-v7x-p12] Patched HF attention.py with DTensor local_map for TP=2 SplashAttention", flush=True)

    hf_base_mod = Path("/torchtitan-hf/torchtitan/experiments/transformers_modeling_backend/model.py")
    if hf_base_mod.exists():
        txt = hf_base_mod.read_text()
        target_pad = "module.weight.data[module.padding_idx].zero_()"
        repl_pad = (
            "module.weight.data[module.padding_idx].zero_() "
            "if module.padding_idx < module.weight.data.shape[0] else None"
        )
        if target_pad in txt and "if module.padding_idx < module.weight.data.shape[0]" not in txt:
            hf_base_mod.write_text(txt.replace(target_pad, repl_pad))
            print("[flint-v7x-p12] Patched HF base model.py for TP=2 Embedding padding_idx bounds", flush=True)

    hf_emb = Path("/torchtitan-hf/torchtitan/models/common/embedding.py")
    if hf_emb.exists():
        txt = hf_emb.read_text()
        if "_safe_tp_f_embedding" not in txt:
            emb_patch = '''

_orig_f_embedding = F.embedding


def _safe_tp_f_embedding(input, weight, padding_idx=None, *args, **kwargs):
    if padding_idx is not None and (padding_idx < 0 or padding_idx >= weight.shape[0]):
        padding_idx = None
    return _orig_f_embedding(input, weight, padding_idx, *args, **kwargs)


F.embedding = _safe_tp_f_embedding
'''
            hf_emb.write_text(txt + emb_patch)
            print("[flint-v7x-p12] Patched HF common/embedding.py for TP=2 F.embedding padding_idx", flush=True)

    hf_loss = Path("/torchtitan-hf/torchtitan/experiments/tpu/loss.py")
    if hf_loss.exists():
        txt = hf_loss.read_text()
        if "_tp_aware_pallas_ce" not in txt:
            loss_patch = '''

_orig_pallas_ce = pallas_cross_entropy_loss


def _tp_aware_pallas_ce(x, labels, weight, *args, **kwargs):
    import torch
    import torch.distributed.nn.functional as dist_nn
    from torch.distributed.tensor import DTensor, Replicate

    tp_mesh = None
    if (
        isinstance(weight, DTensor)
        and weight.device_mesh.mesh_dim_names
        and "tp" in weight.device_mesh.mesh_dim_names
    ):
        tp_mesh = weight.device_mesh["tp"]
        if tp_mesh.size() > 1:
            w_pl = [
                Replicate() if n == "tp" else p
                for n, p in zip(weight.device_mesh.mesh_dim_names, weight.placements)
            ]
            weight = weight.redistribute(placements=w_pl)
            if all(isinstance(p, Replicate) for p in weight.placements):
                weight = weight.to_local()

    if (
        isinstance(x, DTensor)
        and x.device_mesh.mesh_dim_names
        and "tp" in x.device_mesh.mesh_dim_names
    ):
        if tp_mesh is None:
            tp_mesh = x.device_mesh["tp"]
        if tp_mesh.size() > 1:
            x_pl = [
                Replicate() if n == "tp" else p
                for n, p in zip(x.device_mesh.mesh_dim_names, x.placements)
            ]
            x = x.redistribute(placements=x_pl)
            if all(isinstance(p, Replicate) for p in x.placements):
                x = x.to_local()
    elif not isinstance(x, DTensor) and tp_mesh is not None and tp_mesh.size() > 1:
        tok_x = x.numel() // x.shape[-1]
        if tok_x * tp_mesh.size() == labels.numel():
            gdim = 1 if x.ndim == 3 else 0
            parts = dist_nn.all_gather(x.contiguous(), group=tp_mesh.get_group())
            x = torch.cat(parts, dim=gdim)

    return _orig_pallas_ce(x, labels, weight, *args, **kwargs)


pallas_cross_entropy_loss = _tp_aware_pallas_ce
'''
            hf_loss.write_text(txt + loss_patch)
            print("[flint-v7x-p12] Patched HF experiments/tpu/loss.py with TP=2 Pallas CE support", flush=True)


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
    print(f"[flint-v7x-p12] Validated XLA flags: kept={len(kept)} dropped={dropped}", flush=True)
    return " ".join(kept)


def wait_for_master_dns(master_addr: str, timeout_s: int = 300) -> str:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            return socket.gethostbyname(master_addr)
        except socket.gaierror:
            time.sleep(3)
    return master_addr


def build_v7x_pass12_experiments(
    world_size: int,
    hf_model_path: str = "/models/models/RecursiveSI/flint",
) -> list[dict]:
    mb48 = 48 * 2048  # 98,304 tok/rank
    mb64 = 64 * 2048  # 131,072 tok/rank (3,072 tok/exp @ EP=1)
    mb72 = 72 * 2048  # 147,456 tok/rank (3,456 tok/exp @ EP=1, safely < 34.58 GiB slab!)

    # Intentionally omits --xla_tpu_scoped_vmem_limit_kib override to preserve full 34.58 GiB bottom-of-HBM!
    v7x_p12_flags = filter_supported_xla_flags([
        "--xla_tpu_dvfs_p_state=7",
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

    hf_comp_opt = probe_compile_opt_args("/torchtitan-hf")
    hf_splash_1024 = [
        "--splash_attention_kernel.sa_block_q=1024",
        "--splash_attention_kernel.sa_block_kv=1024",
        "--splash_attention_kernel.sa_block_kv_compute=512",
        "--splash_attention_kernel.sa_block_q_dkv=1024",
        "--splash_attention_kernel.sa_block_kv_dkv=1024",
        "--splash_attention_kernel.sa_block_kv_dkv_compute=512",
    ]
    # Hardware-verified 32 MiB Scoped-VMEM-safe forward tile_m=1024 + backward tgmm_tiling=512:2048:0
    paired_gmm_1024_safe = (
        "use_pallas_paired_grouped_mm=true,"
        "pallas_grouped_mm_tile_m=1024,"
        "pallas_grouped_mm_tgmm_tiling=512:2048:0,"
        "pallas_grouped_mm_w2_tgmm_tiling=512:512:0,"
        "pallas_grouped_mm_w2_transposed_tgmm_tiling=512:0:2048"
    )

    def _nat_exp(name: str, lane: str, cfg: str, mb: int, ga: int, log_freq: int = 5, steps: int = 20) -> dict:
        mfu_start = 5 if log_freq == 5 else 4
        return {
            "name": name,
            "lane": lane,
            "workdir": "/torchtitan-native",
            "env": {
                "LIBTPU_INIT_ARGS": v7x_p12_flags,
                "TT_STATIC_EP_CAP_FACTOR": "1.12",
                "TT_STATIC_EP_ALIGN": "512",
                "TT_FORCE_CHUNK_OPT": "1",
                "TT_OPT_EXPERT_CHUNK": "96",
            },
            "args": [
                "--module=torchtitan.experiments.tpu.examples.flint",
                f"--config={cfg}",
                f"--training.steps={steps}",
                "--training.mixed_precision_reduce=bfloat16",
                f"--metrics.log_freq={log_freq}",
                "--debug.seed=42",
                f"--tpu_config.mfu_benchmark_start_step={mfu_start}",
                f"--training.num_tokens_per_microbatch_per_dp_rank={mb}",
                f"--training.num_tokens_per_train_step={ga * world_size * mb}",
            ],
        }

    def _hf_exp(
        name: str,
        lane: str,
        mb: int,
        ga: int,
        n_full_ac: int,
        log_freq: int = 5,
        steps: int = 20,
        extra_args: list[str] | None = None,
        dp_ranks: int | None = None,
    ) -> dict:
        eff_dp = dp_ranks if dp_ranks is not None else world_size
        return {
            "name": name,
            "lane": lane,
            "workdir": "/torchtitan-hf",
            "env": {
                "TT_HF_GMM": paired_gmm_1024_safe,
                "TT_HF_SBDP": "1",
                "TT_FULL_AC_FIRST_N": str(n_full_ac),
                "TORCH_TPU_DEFER_AND_FUSE": "1",
                "LIBTPU_INIT_ARGS": v7x_p12_flags,
            },
            "args": [
                "--module=torchtitan.experiments.tpu.transformers_modeling_backend",
                "--config=hf_moe_auto_max_attnac",
                f"--hf-model={hf_model_path}",
                "--training.max_context_length=2048",
                f"--training.steps={steps}",
                "--training.mixed_precision_reduce=bfloat16",
                "--optimizer.implementation=fused",
                "--lr_scheduler.warmup_steps=0",
                "--lr_scheduler.decay_ratio=0.0",
                f"--metrics.log_freq={log_freq}",
                "--debug.seed=42",
                "--loss.b_block_size=1024",
                *hf_splash_1024,
                *hf_comp_opt,
                *(extra_args or []),
                f"--training.num_tokens_per_microbatch_per_dp_rank={mb}",
                f"--training.num_tokens_per_train_step={ga * eff_dp * mb}",
            ],
        }

    return [
        # 1. Direct upgrade over all-time #1 v7x_p9_02 (18.68% steady / 18.76% peak):
        #    Native tile_m=1024 + SBDP + BF16-RoPE + PureFusedOpt + Direct config.libtpu_init_flags + log_freq=5 @ mb=48 2xGA
        _nat_exp(
            "v7x_p12_01_native_tile1024_sbdp_bf16rope_fusedopt_libtpu_n43_mb48_ga2_log5",
            "v7x-16 Native PR#122 + tile1024 + SBDP + BF16-RoPE + PureFusedOpt + CalibratedLibtpuFlags + n43 (mb=48, 2xGA = 6.29M tok/step, log5)",
            "flint_40b_p12_ep1_sbdp_tile1024_n43",
            mb48,
            2,
            log_freq=5,
            steps=20,
        ),
        # 2. Push Native saved SplashAttention n43->n45 + 4x GradAccum (12.58M tok/step, 4x optimizer/RS amortization!)
        _nat_exp(
            "v7x_p12_02_native_tile1024_sbdp_bf16rope_fusedopt_libtpu_n45_mb48_ga4_log5",
            "v7x-16 Native PR#122 + tile1024 + SBDP + BF16-RoPE + PureFusedOpt + CalibratedLibtpuFlags + n45 (mb=48, 4xGA = 12.58M tok/step, log5)",
            "flint_40b_p12_ep1_sbdp_tile1024_n45",
            mb48,
            4,
            log_freq=5,
            steps=15,
        ),
        # 3. Native mb=64 (131,072 tok/rank = 3,072 tok/exp) + SBDP + BF16-RoPE + tile1024 + n36 + 2xGA (8.39M tok/step)
        _nat_exp(
            "v7x_p12_03_native_tile1024_sbdp_bf16rope_fusedopt_libtpu_n36_mb64_ga2_log5",
            "v7x-16 Native PR#122 + tile1024 + SBDP + BF16-RoPE + PureFusedOpt + CalibratedLibtpuFlags + n36 (mb=64 = 3,072 tok/exp, 2xGA = 8.39M tok/step, log5)",
            "flint_40b_p12_ep1_sbdp_tile1024_n36",
            mb64,
            2,
            log_freq=5,
            steps=15,
        ),
        # 4. Upgrade over Pass 11F HF #1 winner (18.39%): HF PR#123 + tile1024 (safe tgmm) + PureFusedOpt + SBDP + sa=1024 + HybridAC12 @ mb=64 2xGA
        _hf_exp(
            "v7x_p12_04_hf_pr123_sbdp_purefused_sa1024_tile1024_hybridac12_mb64_ga2_log5",
            "v7x-16 HF PR#123 + SBDP + PureFusedOpt + sa=1024 + tile1024 + HybridAC12 (mb=64 = 3,072 tok/exp, 2xGA = 8.39M tok/step, log5)",
            mb64,
            2,
            n_full_ac=12,
            log_freq=5,
            steps=15,
        ),
        # 5. HF PR#123 @ mb=72 (147,456 tok/rank = 3,456 tok/exp, safely < 34.58 GiB slab!) + HybridAC16 + 2xGA (9.44M tok/step)
        _hf_exp(
            "v7x_p12_05_hf_pr123_sbdp_purefused_sa1024_tile1024_hybridac16_mb72_ga2_log5",
            "v7x-16 HF PR#123 + SBDP + PureFusedOpt + sa=1024 + tile1024 + HybridAC16 (mb=72 = 3,456 tok/exp, 2xGA = 9.44M tok/step, log5)",
            mb72,
            2,
            n_full_ac=16,
            log_freq=5,
            steps=15,
        ),
        # 6. Parallelism Matrix (HSDP): DP_replicate=2 × DP_shard=16 (8-chip shards) + SBDP + tile1024 + mb=64 2xGA
        _hf_exp(
            "v7x_p12_06_hf_pr123_hsdp2x16_sbdp_purefused_sa1024_tile1024_hybridac20_mb64_ga2_log5",
            "v7x-16 HSDP (DP_rep=2 x FSDP=16) + SBDP + PureFusedOpt + sa=1024 + tile1024 + HybridAC20 (mb=64, 2xGA = 8.39M tok/step, log5)",
            mb64,
            2,
            n_full_ac=20,
            log_freq=5,
            steps=15,
            extra_args=[
                "--parallelism.data_parallel_replicate_degree=2",
                "--parallelism.data_parallel_shard_degree=16",
            ],
        ),
        # 7. Parallelism Matrix (TP × DP): Intra-Chip TP=2 (dual-TensorCore socket) × FSDP=16 + SBDP + tile1024 + mb=64 2xGA
        _hf_exp(
            "v7x_p12_07_hf_pr123_tp2_dp16_sbdp_purefused_sa1024_tile1024_hybridac8_mb64_ga2_log5",
            "v7x-16 Intra-Chip TP=2 x FSDP=16 + SBDP + PureFusedOpt + sa=1024 + tile1024 + HybridAC8 (mb=64/dp_rank, 2xGA = 4.19M tok/step, log5)",
            mb64,
            2,
            n_full_ac=8,
            log_freq=5,
            steps=15,
            extra_args=["--parallelism.tensor_parallel_degree=2"],
            dp_ranks=world_size // 2,
        ),
        # 8. Parallelism Matrix (Intra-Chip EP=2 × FSDP=16): 100% intra-socket dual-TensorCore AllToAll, 4,608 tok/exp
        _nat_exp(
            "v7x_p12_08_native_intrachip_staticep2_zero3_paired512_n43_mb48_ga2_log5",
            "v7x-16 Intra-Chip StaticEP=2 x FSDP=16 (Zero Inter-Chip A2A, 4,608 tok/exp) + ZeRO-3 + PairedGMM-512 + n43 (mb=48, 2xGA, log5)",
            "flint_40b_p12_staticep2_zero3_n43",
            mb48,
            2,
            log_freq=5,
            steps=15,
        ),
        # 9. Parallelism Matrix (Intra-Tray EP=4 × FSDP=8): 2-chip intra-tray AllToAll, 9,216 tok/exp
        _nat_exp(
            "v7x_p12_09_native_intratray_staticep4_zero3_paired512_n40_mb48_ga2_log5",
            "v7x-16 Intra-Tray StaticEP=4 x FSDP=8 (2-Chip Intra-Host A2A, 9,216 tok/exp) + ZeRO-3 + PairedGMM-512 + n40 (mb=48, 2xGA, log5)",
            "flint_40b_p12_staticep4_zero3_n40",
            mb48,
            2,
            log_freq=5,
            steps=15,
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


def parse_metrics(log_text: str, chip_type: str, tp_degree: int = 1) -> dict:
    steps = []
    for line in log_text.splitlines():
        m = STEP_RE.search(line)
        if not m:
            continue
        step = int(m.group("step"))
        if steps and steps[-1]["step"] == step:
            continue
        loss = float(m.group("loss"))
        raw_tps = float(m.group("tps").replace(",", ""))
        # Note: When TP > 1, TorchTitan logs tokens/sec per DP group; normalize per TensorCore rank!
        tps = raw_tps / tp_degree
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
                "tps_per_rank": round(tps, 1),
                "logged_mfu_pct": round(logged_mfu, 2),
                "mfu_pct": round(calc_mfu, 2),
                "mem_gib": mem_gib,
            }
        )
    if not steps:
        return {"num_logged_steps": 0, "steady_tps": 0.0, "steady_mfu_pct": 0.0}
    max_step = max(s["step"] for s in steps)
    if max_step >= 15 and any(s["step"] >= 10 for s in steps):
        steady = [s for s in steps if s["step"] >= 10]
    else:
        steady = [s for s in steps if s["step"] >= 6] or steps[-2:]
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
        f"# Flint (`RecursiveSI/flint` 40B MoE) TPU v7x `{suite.upper()}` (MFU Frontier + `EP/DP/TP` Matrix) — `{slice_name}`",
        "",
        f"- **Updated (UTC):** `{payload['updated_at_utc']}`",
        f"- **Chip / World Size:** `{chip_type}` (`{world_size}` TensorCore ranks, `{PEAK_BF16_TFLOPS_PER_RANK.get(chip_type)}` BF16 TFLOP/s/rank)",
        "",
        "| Experiment | Lane | Status | Steady tok/s/rank | Cluster tok/s | Steady MFU (%) | Peak MFU (%) | Peak HBM (GiB) | Loss (start -> end) | Wall Time (s) |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for r in results:
        m = r.get("metrics", {})
        loss_str = f"{m.get('first_loss', '-')} -> {m.get('final_loss', '-')}" if m.get("num_logged_steps") else "-"
        st_tps = m.get("steady_tps", 0.0)
        md_lines.append(
            f"| `{r['name']}` | {r['lane']} | `{r['status']}` | "
            f"**{st_tps:,.1f}** | {st_tps * world_size:,.0f} | **{m.get('steady_mfu_pct', 0.0):.2f}%** | "
            f"{m.get('peak_mfu_pct', 0.0):.2f}% | {m.get('peak_mem_gib') or '-'} | {loss_str} | {r['duration_s']:.1f} |"
        )
    md_text = "\n".join(md_lines) + "\n"
    (out_dir / "SUMMARY.md").write_text(md_text)
    if bench_dir.exists():
        (bench_dir / f"flint_{slice_name}_{suite}_summary.json").write_text(json.dumps(payload, indent=2))
        (bench_dir / f"flint_{slice_name}_{suite}_SUMMARY.md").write_text(md_text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="pass12")
    parser.add_argument("--chip-type", choices=["v7x", "v5p", "v6e"], default="v7x")
    parser.add_argument("--nnodes", type=int, required=True)
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument("--master-addr", type=str, required=True)
    parser.add_argument("--slice-name", type=str, required=True)
    parser.add_argument("--hf-model", type=str, default="/models/models/RecursiveSI/flint")
    parser.add_argument("--base-port", type=int, default=29740)
    parser.add_argument("--per-run-timeout", type=int, default=2100)
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

    exps = build_v7x_pass12_experiments(world_size, args.hf_model)
    if args.suite == "pass12b":
        hf_ga4 = dict(exps[3])
        hf_ga4["name"] = "v7x_p12b_04_hf_pr123_sbdp_purefused_sa1024_tile1024_hybridac12_mb64_ga4_log5"
        hf_ga4["lane"] = "v7x-16 HF PR#123 + SBDP + PureFusedOpt + sa=1024 + tile1024 + HybridAC12 (mb=64, 4xGA = 16.78M tok/step, log5)"
        hf_ga4["args"] = [
            f"--training.num_tokens_per_train_step={4 * world_size * 64 * 2048}"
            if a.startswith("--training.num_tokens_per_train_step=")
            else a
            for a in exps[3]["args"]
        ]
        exps = [exps[0], exps[1], exps[2], hf_ga4, exps[7], exps[8]]
    elif args.suite == "pass12c":
        mb48 = 48 * 2048
        mb64 = 64 * 2048
        mb80 = 80 * 2048
        p9_scout_flags = filter_supported_xla_flags([
            "--xla_tpu_enable_sparse_core_collective_aggregator=true",
            "--xla_tpu_enable_latency_hiding_layer_scheduler=true",
            "--xla_tpu_use_single_sparse_core_for_reduce_scatter_offload=false",
            "--xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=true",
            "--xla_tpu_enable_sparse_core_hierarchical_reduce_scatter=true",
            "--xla_tpu_dvfs_p_state=7",
            "--xla_enable_async_all_gather=true",
            "--xla_enable_async_all_reduce=true",
            "--xla_tpu_enable_async_all_to_all=true",
            "--xla_tpu_enable_ag_backward_pipelining=true",
            "--xla_tpu_enable_ici_ag_pipelining=true",
            "--xla_tpu_overlap_compute_collective_tc=true",
            "--xla_max_concurrent_async_all_gathers=4",
            "--xla_latency_hiding_scheduler_rerun=2",
            "--xla_enable_transpose_trace=false",
        ])
        base_nat_env = dict(exps[7]["env"])
        p9_nat_env = {
            "LIBTPU_INIT_ARGS": p9_scout_flags,
            "TT_FORCE_CHUNK_OPT": "0",
        }
        hsdp_48 = dict(exps[5])
        hsdp_48["name"] = "v7x_p12c_05_hf_pr123_hsdp2x16_sbdp_purefused_sa1024_tile1024_hybridac28_mb48_ga2_log5"
        hsdp_48["lane"] = "v7x-16 HSDP (DP_rep=2 x FSDP=16) + SBDP + PureFusedOpt + sa=1024 + tile1024 + HybridAC28 (mb=48, 2xGA = 6.29M tok/step, log5)"
        hsdp_48["env"] = {**exps[5]["env"], "TT_FULL_AC_FIRST_N": "28"}
        hsdp_48["args"] = [
            f"--training.num_tokens_per_microbatch_per_dp_rank={mb48}"
            if a.startswith("--training.num_tokens_per_microbatch_per_dp_rank=")
            else (
                f"--training.num_tokens_per_train_step={2 * world_size * mb48}"
                if a.startswith("--training.num_tokens_per_train_step=")
                else a
            )
            for a in exps[5]["args"]
        ]
        tp2_64 = dict(exps[6])
        tp2_64["name"] = "v7x_p12c_06_hf_pr123_tp2_dp16_sbdp_purefused_sa1024_tile1024_hybridac16_mb48_ga2_log5"
        tp2_64["lane"] = "v7x-16 Intra-Chip TP=2 x FSDP=16 + SBDP + PureFusedOpt + sa=1024 + tile1024 + HybridAC16 + SinksTP (mb=48/dp_rank, 2xGA = 3.15M tok/step, log5)"
        tp2_64["env"] = {**exps[6]["env"], "TT_FULL_AC_FIRST_N": "16"}
        tp2_64["args"] = [
            f"--training.num_tokens_per_microbatch_per_dp_rank={mb48}"
            if a.startswith("--training.num_tokens_per_microbatch_per_dp_rank=")
            else (
                f"--training.num_tokens_per_train_step={2 * (world_size // 2) * mb48}"
                if a.startswith("--training.num_tokens_per_train_step=")
                else a
            )
            for a in exps[6]["args"]
        ]
        exps = [
            # 1. Exact v7x_p9_02 (18.68% MFU) base + SBDP + BF16-RoPE + n41 @ mb=48, 2xGA (zero bottom-of-HBM overflow!)
            {
                "name": "v7x_p12c_01_native_p9base_sbdp_bf16rope_n41_mb48_ga2_log5",
                "lane": "v7x-16 Native PR#122 (v7x_p9_02 base + SBDP + BF16-RoPE + n41, mb=48, 2xGA = 6.29M tok/step, log5)",
                "workdir": "/torchtitan-native",
                "env": p9_nat_env,
                "args": [
                    "--module=torchtitan.experiments.tpu.examples.flint",
                    "--config=flint_40b_p12c_ep1_p9base_sbdp_n41",
                    "--training.steps=20",
                    "--training.mixed_precision_reduce=bfloat16",
                    "--metrics.log_freq=5",
                    "--debug.seed=42",
                    "--tpu_config.mfu_benchmark_start_step=5",
                    f"--training.num_tokens_per_microbatch_per_dp_rank={mb48}",
                    f"--training.num_tokens_per_train_step={2 * world_size * mb48}",
                ],
            },
            # 2. Exact v7x_p9_02 base + SBDP + BF16-RoPE scaled to mb=64 (3,072 tok/exp) + n30, 2xGA = 8.39M tok/step
            {
                "name": "v7x_p12c_02_native_p9base_sbdp_bf16rope_n30_mb64_ga2_log5",
                "lane": "v7x-16 Native PR#122 (v7x_p9_02 base + SBDP + BF16-RoPE + n30, mb=64 = 3,072 tok/exp, 2xGA = 8.39M tok/step, log5)",
                "workdir": "/torchtitan-native",
                "env": p9_nat_env,
                "args": [
                    "--module=torchtitan.experiments.tpu.examples.flint",
                    "--config=flint_40b_p12c_ep1_p9base_sbdp_n30",
                    "--training.steps=20",
                    "--training.mixed_precision_reduce=bfloat16",
                    "--metrics.log_freq=5",
                    "--debug.seed=42",
                    "--tpu_config.mfu_benchmark_start_step=5",
                    f"--training.num_tokens_per_microbatch_per_dp_rank={mb64}",
                    f"--training.num_tokens_per_train_step={2 * world_size * mb64}",
                ],
            },
            # 3. Intra-Chip StaticEP=2 x FSDP=16 scaled to mb=80 (7,680 tok/exp!) + ZeRO-3 + PairedGMM-512
            {
                "name": "v7x_p12c_03_native_intrachip_staticep2_zero3_paired512_n36_mb80_ga2_log5",
                "lane": "v7x-16 Intra-Chip StaticEP=2 x FSDP=16 (Zero Inter-Chip A2A, 7,680 tok/exp) + ZeRO-3 + PairedGMM-512 + n36 (mb=80, 2xGA = 10.49M tok/step)",
                "workdir": "/torchtitan-native",
                "env": base_nat_env,
                "args": [
                    "--module=torchtitan.experiments.tpu.examples.flint",
                    "--config=flint_40b_p12c_staticep2_zero3_n36",
                    "--training.steps=15",
                    "--training.mixed_precision_reduce=bfloat16",
                    "--metrics.log_freq=5",
                    "--debug.seed=42",
                    "--tpu_config.mfu_benchmark_start_step=5",
                    f"--training.num_tokens_per_microbatch_per_dp_rank={mb80}",
                    f"--training.num_tokens_per_train_step={2 * world_size * mb80}",
                ],
            },
            # 4. Intra-Chip StaticEP=2 x FSDP=16 scaled to mb=80 + ZeRO-2 (skips 100% of backward parameter AllGather!)
            {
                "name": "v7x_p12c_04_native_intrachip_staticep2_zero2_paired512_n36_mb80_ga2_log5",
                "lane": "v7x-16 Intra-Chip StaticEP=2 x FSDP=16 (Zero Inter-Chip A2A, 7,680 tok/exp) + ZeRO-2 (No Bwd AG) + PairedGMM-512 + n36 (mb=80, 2xGA = 10.49M tok/step)",
                "workdir": "/torchtitan-native",
                "env": base_nat_env,
                "args": [
                    "--module=torchtitan.experiments.tpu.examples.flint",
                    "--config=flint_40b_p12c_staticep2_zero2_n36",
                    "--training.steps=15",
                    "--training.mixed_precision_reduce=bfloat16",
                    "--metrics.log_freq=5",
                    "--debug.seed=42",
                    "--tpu_config.mfu_benchmark_start_step=5",
                    f"--training.num_tokens_per_microbatch_per_dp_rank={mb80}",
                    f"--training.num_tokens_per_train_step={2 * world_size * mb80}",
                ],
            },
            # 5. HSDP (DP_rep=2 x FSDP=16) @ mb=48, HybridAC28, 2xGA
            hsdp_48,
            # 6. Intra-Chip TP=2 x FSDP=16 with FlintAttention.sinks + DTensor SplashAttention
            tp2_64,
        ]
    elif args.suite == "pass12d":
        mb48 = 48 * 2048
        mb64 = 64 * 2048
        p9_scout_flags = filter_supported_xla_flags([
            "--xla_tpu_enable_sparse_core_collective_aggregator=true",
            "--xla_tpu_enable_latency_hiding_layer_scheduler=true",
            "--xla_tpu_use_single_sparse_core_for_reduce_scatter_offload=false",
            "--xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=true",
            "--xla_tpu_enable_sparse_core_hierarchical_reduce_scatter=true",
            "--xla_tpu_dvfs_p_state=7",
            "--xla_enable_async_all_gather=true",
            "--xla_enable_async_all_reduce=true",
            "--xla_tpu_enable_async_all_to_all=true",
            "--xla_tpu_enable_ag_backward_pipelining=true",
            "--xla_tpu_enable_ici_ag_pipelining=true",
            "--xla_tpu_overlap_compute_collective_tc=true",
            "--xla_max_concurrent_async_all_gathers=4",
            "--xla_latency_hiding_scheduler_rerun=2",
            "--xla_enable_transpose_trace=false",
        ])
        base_nat_env = dict(exps[7]["env"])
        p9_nat_env = {
            "LIBTPU_INIT_ARGS": p9_scout_flags,
            "TT_FORCE_CHUNK_OPT": "0",
        }
        hsdp_64 = dict(exps[5])
        hsdp_64["name"] = "v7x_p12d_04_hf_pr123_hsdp2x16_sbdp_purefused_sa1024_tile1024_hybridac16_mb64_ga2_log5"
        hsdp_64["lane"] = "v7x-16 HSDP (DP_rep=2 x FSDP=16) + SBDP + PureFusedOpt + sa=1024 + tile1024 + HybridAC16 (mb=64 = 3,072 tok/exp, 2xGA = 8.39M tok/step, log5)"
        hsdp_64["env"] = {**exps[5]["env"], "TT_FULL_AC_FIRST_N": "16"}

        tp2_48 = dict(exps[6])
        tp2_48["name"] = "v7x_p12d_05_hf_pr123_tp2_dp16_sbdp_purefused_sa1024_tile1024_hybridac16_mb48_ga2_log5"
        tp2_48["lane"] = "v7x-16 Intra-Chip TP=2 x FSDP=16 + SBDP + PureFusedOpt + sa=1024 + tile1024 + HybridAC16 + SinksTP + PadFix (mb=48/dp_rank, 2xGA = 3.15M tok/step, log5)"
        tp2_48["env"] = {**exps[6]["env"], "TT_FULL_AC_FIRST_N": "16"}
        tp2_48["args"] = [
            f"--training.num_tokens_per_microbatch_per_dp_rank={mb48}"
            if a.startswith("--training.num_tokens_per_microbatch_per_dp_rank=")
            else (
                f"--training.num_tokens_per_train_step={2 * (world_size // 2) * mb48}"
                if a.startswith("--training.num_tokens_per_train_step=")
                else a
            )
            for a in exps[6]["args"]
        ]
        exps = [
            # 1. Push past 20.61% MFU: v7x_p12c_01 + n47 (47/49 layers saved!) + 4x GA (12.58M tok/step)
            {
                "name": "v7x_p12d_01_native_p9base_sbdp_bf16rope_n47_mb48_ga4_log5",
                "lane": "v7x-16 Native PR#122 (v7x_p12c_01 + n47 + 4xGA = 12.58M tok/step, SBDP + BF16-RoPE, mb=48, log5)",
                "workdir": "/torchtitan-native",
                "env": p9_nat_env,
                "args": [
                    "--module=torchtitan.experiments.tpu.examples.flint",
                    "--config=flint_40b_p12d_ep1_p9base_sbdp_n47",
                    "--training.steps=20",
                    "--training.mixed_precision_reduce=bfloat16",
                    "--metrics.log_freq=5",
                    "--debug.seed=42",
                    "--tpu_config.mfu_benchmark_start_step=5",
                    f"--training.num_tokens_per_microbatch_per_dp_rank={mb48}",
                    f"--training.num_tokens_per_train_step={4 * world_size * mb48}",
                ],
            },
            # 2. Full Host Offload (save_last_n=0, prefetch=2 -> 1 single layer template, zero host RAM compile OOM!) + SBDP + BF16-RoPE @ mb=64 (3,072 tok/exp)
            {
                "name": "v7x_p12d_02_native_hostoffload_sbdp_bf16rope_mb64_ga2_log5",
                "lane": "v7x-16 Native PR#122 (Full Host Offload prefetch=2 + SBDP + BF16-RoPE, mb=64 = 3,072 tok/exp, 2xGA = 8.39M tok/step, log5)",
                "workdir": "/torchtitan-native",
                "env": p9_nat_env,
                "args": [
                    "--module=torchtitan.experiments.tpu.examples.flint",
                    "--config=flint_40b_p12d_ep1_hostoffload_sbdp",
                    "--training.steps=20",
                    "--training.mixed_precision_reduce=bfloat16",
                    "--metrics.log_freq=5",
                    "--debug.seed=42",
                    "--tpu_config.mfu_benchmark_start_step=5",
                    f"--training.num_tokens_per_microbatch_per_dp_rank={mb64}",
                    f"--training.num_tokens_per_train_step={2 * world_size * mb64}",
                ],
            },
            # 3. Intra-Chip StaticEP=2 x FSDP=16 + ZeRO-2 + SBDP + BF16-RoPE + PairedGMM-512 @ mb=64 (6,144 tok/exp), n30
            {
                "name": "v7x_p12d_03_native_intrachip_staticep2_zero2_sbdp_paired512_n30_mb64_ga2_log5",
                "lane": "v7x-16 Intra-Chip StaticEP=2 x FSDP=16 (Zero Inter-Chip A2A, 6,144 tok/exp) + ZeRO-2 + SBDP + PairedGMM-512 + n30 (mb=64, 2xGA = 8.39M tok/step)",
                "workdir": "/torchtitan-native",
                "env": base_nat_env,
                "args": [
                    "--module=torchtitan.experiments.tpu.examples.flint",
                    "--config=flint_40b_p12d_staticep2_zero2_sbdp_n30",
                    "--training.steps=15",
                    "--training.mixed_precision_reduce=bfloat16",
                    "--metrics.log_freq=5",
                    "--debug.seed=42",
                    "--tpu_config.mfu_benchmark_start_step=5",
                    f"--training.num_tokens_per_microbatch_per_dp_rank={mb64}",
                    f"--training.num_tokens_per_train_step={2 * world_size * mb64}",
                ],
            },
            # 4. HSDP (DP_rep=2 x FSDP=16) scaled to mb=64 (3,072 tok/exp), HybridAC16, 2xGA
            hsdp_64,
            # 5. Intra-Chip TP=2 x FSDP=16 with SinksTP + DTensor SplashAttention + Embedding padding_idx fix
            tp2_48,
        ]
    elif args.suite == "pass12e":
        mb48 = 48 * 2048
        mb64 = 64 * 2048
        p9_scout_flags = filter_supported_xla_flags([
            "--xla_tpu_enable_sparse_core_collective_aggregator=true",
            "--xla_tpu_enable_latency_hiding_layer_scheduler=true",
            "--xla_tpu_use_single_sparse_core_for_reduce_scatter_offload=false",
            "--xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=true",
            "--xla_tpu_enable_sparse_core_hierarchical_reduce_scatter=true",
            "--xla_tpu_dvfs_p_state=7",
            "--xla_enable_async_all_gather=true",
            "--xla_enable_async_all_reduce=true",
            "--xla_tpu_enable_async_all_to_all=true",
            "--xla_tpu_enable_ag_backward_pipelining=true",
            "--xla_tpu_enable_ici_ag_pipelining=true",
            "--xla_tpu_overlap_compute_collective_tc=true",
            "--xla_max_concurrent_async_all_gathers=4",
            "--xla_latency_hiding_scheduler_rerun=2",
            "--xla_enable_transpose_trace=false",
        ])
        base_nat_env = dict(exps[7]["env"])
        p9_nat_env = {
            "LIBTPU_INIT_ARGS": p9_scout_flags,
            "TT_FORCE_CHUNK_OPT": "0",
        }
        tp2_48_clean = dict(exps[6])
        tp2_48_clean["name"] = "v7x_p12e_02_hf_pr123_tp2_dp16_sbdp_purefused_sa1024_tile1024_hybridac24_mb48_ga2_log5"
        tp2_48_clean["lane"] = "v7x-16 Intra-Chip TP=2 x FSDP=16 + SBDP + PureFusedOpt + sa=1024 + tile1024 + HybridAC24 + SinksTP + PadFix (mb=48/dp_rank, 2xGA = 3.15M tok/step, log5)"
        tp2_48_clean["env"] = {**exps[6]["env"], "TT_FULL_AC_FIRST_N": "24"}
        tp2_48_clean["args"] = [
            f"--training.num_tokens_per_microbatch_per_dp_rank={mb48}"
            if a.startswith("--training.num_tokens_per_microbatch_per_dp_rank=")
            else (
                f"--training.num_tokens_per_train_step={2 * (world_size // 2) * mb48}"
                if a.startswith("--training.num_tokens_per_train_step=")
                else a
            )
            for a in exps[6]["args"]
        ]

        hsdp_64_ac36 = dict(exps[5])
        hsdp_64_ac36["name"] = "v7x_p12e_06_hf_pr123_hsdp2x16_sbdp_purefused_sa1024_tile1024_hybridac36_mb64_ga2_log5"
        hsdp_64_ac36["lane"] = "v7x-16 HSDP (DP_rep=2 x FSDP=16) + SBDP + PureFusedOpt + sa=1024 + tile1024 + HybridAC36 (mb=64 = 3,072 tok/exp, 2xGA = 8.39M tok/step, log5)"
        hsdp_64_ac36["env"] = {**exps[5]["env"], "TT_FULL_AC_FIRST_N": "36"}

        exps = [
            # 1. Push past 20.68% MFU: n49 (49/49 layers saved — 0% attention recompute!) + 4x GA (12.58M tok/step)
            {
                "name": "v7x_p12e_01_native_p9base_sbdp_bf16rope_n49_mb48_ga4_log5",
                "lane": "v7x-16 Native PR#122 (n49 = 0% Attn Recompute! + 4xGA = 12.58M tok/step, SBDP + BF16-RoPE, mb=48, log5)",
                "workdir": "/torchtitan-native",
                "env": p9_nat_env,
                "args": [
                    "--module=torchtitan.experiments.tpu.examples.flint",
                    "--config=flint_40b_p12e_ep1_p9base_sbdp_n49",
                    "--training.steps=20",
                    "--training.mixed_precision_reduce=bfloat16",
                    "--metrics.log_freq=5",
                    "--debug.seed=42",
                    "--tpu_config.mfu_benchmark_start_step=5",
                    f"--training.num_tokens_per_microbatch_per_dp_rank={mb48}",
                    f"--training.num_tokens_per_train_step={4 * world_size * mb48}",
                ],
            },
            # 2. Clean Intra-Chip TP=2 x FSDP=16 run early on pristine slice
            tp2_48_clean,
            # 3. Single-template n49 (98% XLA cache hit rate!) scaled to mb=64 (3,072 tok/exp), 2x GA (8.39M tok/step)
            {
                "name": "v7x_p12e_03_native_p9base_sbdp_bf16rope_n49_mb64_ga2_log5",
                "lane": "v7x-16 Native PR#122 (n49 Single-Template + SBDP + BF16-RoPE, mb=64 = 3,072 tok/exp, 2xGA = 8.39M tok/step, log5)",
                "workdir": "/torchtitan-native",
                "env": p9_nat_env,
                "args": [
                    "--module=torchtitan.experiments.tpu.examples.flint",
                    "--config=flint_40b_p12e_ep1_p9base_sbdp_n49",
                    "--training.steps=20",
                    "--training.mixed_precision_reduce=bfloat16",
                    "--metrics.log_freq=5",
                    "--debug.seed=42",
                    "--tpu_config.mfu_benchmark_start_step=5",
                    f"--training.num_tokens_per_microbatch_per_dp_rank={mb64}",
                    f"--training.num_tokens_per_train_step={2 * world_size * mb64}",
                ],
            },
            # 4. Full Host Activation Offload (tpu_selective_and_offload + prefetch=2) + SBDP + BF16-RoPE @ mb=64 (3,072 tok/exp)
            {
                "name": "v7x_p12e_04_native_fulloffload_sbdp_bf16rope_mb64_ga2_log5",
                "lane": "v7x-16 Native PR#122 (Full Host Offload prefetch=2 + SBDP + BF16-RoPE, mb=64 = 3,072 tok/exp, 2xGA = 8.39M tok/step, log5)",
                "workdir": "/torchtitan-native",
                "env": p9_nat_env,
                "args": [
                    "--module=torchtitan.experiments.tpu.examples.flint",
                    "--config=flint_40b_p12e_ep1_p9base_sbdp_fulloffload",
                    "--training.steps=20",
                    "--training.mixed_precision_reduce=bfloat16",
                    "--metrics.log_freq=5",
                    "--debug.seed=42",
                    "--tpu_config.mfu_benchmark_start_step=5",
                    f"--training.num_tokens_per_microbatch_per_dp_rank={mb64}",
                    f"--training.num_tokens_per_train_step={2 * world_size * mb64}",
                ],
            },
            # 5. Intra-Chip StaticEP=2 x FSDP=16 + ZeRO-2 + PairedGMM-512 + n49 @ mb=64 (6,144 tok/exp), 4x GA (16.78M tok/step)
            {
                "name": "v7x_p12e_05_native_intrachip_staticep2_zero2_paired512_n49_mb64_ga4_log5",
                "lane": "v7x-16 Intra-Chip StaticEP=2 x FSDP=16 (Zero Inter-Chip A2A, 6,144 tok/exp) + ZeRO-2 + PairedGMM-512 + n49 (mb=64, 4xGA = 16.78M tok/step)",
                "workdir": "/torchtitan-native",
                "env": base_nat_env,
                "args": [
                    "--module=torchtitan.experiments.tpu.examples.flint",
                    "--config=flint_40b_p12e_staticep2_zero2_n49",
                    "--training.steps=15",
                    "--training.mixed_precision_reduce=bfloat16",
                    "--metrics.log_freq=5",
                    "--debug.seed=42",
                    "--tpu_config.mfu_benchmark_start_step=5",
                    f"--training.num_tokens_per_microbatch_per_dp_rank={mb64}",
                    f"--training.num_tokens_per_train_step={4 * world_size * mb64}",
                ],
            },
            # 6. HSDP (DP_rep=2 x FSDP=16) scaled to mb=64 with HybridAC36 (<20 GiB HBM), 2x GA
            hsdp_64_ac36,
        ]
    elif args.suite == "pass12f":
        mb48 = 48 * 2048
        mb52 = 52 * 2048
        mb56 = 56 * 2048
        p9_scout_flags = filter_supported_xla_flags([
            "--xla_tpu_enable_sparse_core_collective_aggregator=true",
            "--xla_tpu_enable_latency_hiding_layer_scheduler=true",
            "--xla_tpu_use_single_sparse_core_for_reduce_scatter_offload=false",
            "--xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=true",
            "--xla_tpu_enable_sparse_core_hierarchical_reduce_scatter=true",
            "--xla_tpu_dvfs_p_state=7",
            "--xla_enable_async_all_gather=true",
            "--xla_enable_async_all_reduce=true",
            "--xla_tpu_enable_async_all_to_all=true",
            "--xla_tpu_enable_ag_backward_pipelining=true",
            "--xla_tpu_enable_ici_ag_pipelining=true",
            "--xla_tpu_overlap_compute_collective_tc=true",
            "--xla_max_concurrent_async_all_gathers=4",
            "--xla_latency_hiding_scheduler_rerun=2",
            "--xla_enable_transpose_trace=false",
        ])
        base_nat_env = dict(exps[7]["env"])
        p9_nat_env = {
            "LIBTPU_INIT_ARGS": p9_scout_flags,
            "TT_FORCE_CHUNK_OPT": "0",
        }
        tp2_48_fixed = dict(exps[6])
        tp2_48_fixed["name"] = "v7x_p12f_03_hf_pr123_tp2_dp16_sbdp_purefused_sa1024_tile1024_hybridac24_mb48_ga2_log5"
        tp2_48_fixed["lane"] = "v7x-16 Intra-Chip TP=2 x FSDP=16 + SBDP + PureFusedOpt + sa=1024 + tile1024 + HybridAC24 + SinksTP + EmbPadFix (mb=48/dp_rank, 2xGA = 3.15M tok/step, log5)"
        tp2_48_fixed["env"] = {**exps[6]["env"], "TT_FULL_AC_FIRST_N": "24"}
        tp2_48_fixed["args"] = [
            f"--training.num_tokens_per_microbatch_per_dp_rank={mb48}"
            if a.startswith("--training.num_tokens_per_microbatch_per_dp_rank=")
            else (
                f"--training.num_tokens_per_train_step={2 * (world_size // 2) * mb48}"
                if a.startswith("--training.num_tokens_per_train_step=")
                else a
            )
            for a in exps[6]["args"]
        ]

        hsdp_64_ac49 = dict(exps[5])
        hsdp_64_ac49["name"] = "v7x_p12f_04_hf_pr123_hsdp2x16_sbdp_purefused_sa1024_tile1024_fullac49_mb64_ga2_log5"
        hsdp_64_ac49["lane"] = "v7x-16 HSDP (DP_rep=2 x FSDP=16) + SBDP + PureFusedOpt + sa=1024 + tile1024 + FullAC49 (mb=64 = 3,072 tok/exp, 2xGA = 8.39M tok/step, log5)"
        hsdp_64_ac49["env"] = {**exps[5]["env"], "TT_FULL_AC_FIRST_N": "49"}

        exps = [
            # 1. Push past 20.77% MFU: n49 (0% Attn Recompute!) + 8x GA (25.17M tok/step, halves optimizer overhead!)
            {
                "name": "v7x_p12f_01_native_p9base_sbdp_bf16rope_n49_mb48_ga8_log5",
                "lane": "v7x-16 Native PR#122 (n49 = 0% Attn Recompute! + 8xGA = 25.17M tok/step, SBDP + BF16-RoPE, mb=48, log5)",
                "workdir": "/torchtitan-native",
                "env": p9_nat_env,
                "args": [
                    "--module=torchtitan.experiments.tpu.examples.flint",
                    "--config=flint_40b_p12e_ep1_p9base_sbdp_n49",
                    "--training.steps=15",
                    "--training.mixed_precision_reduce=bfloat16",
                    "--metrics.log_freq=5",
                    "--debug.seed=42",
                    "--tpu_config.mfu_benchmark_start_step=5",
                    f"--training.num_tokens_per_microbatch_per_dp_rank={mb48}",
                    f"--training.num_tokens_per_train_step={8 * world_size * mb48}",
                ],
            },
            # 2. n49 + SBDP + BF16-RoPE @ mb=52 (2,496 tok/exp, ~25 GiB HBM), 4x GA (13.63M tok/step)
            {
                "name": "v7x_p12f_02_native_p9base_sbdp_bf16rope_n49_mb52_ga4_log5",
                "lane": "v7x-16 Native PR#122 (n49 + SBDP + BF16-RoPE, mb=52 = 2,496 tok/exp, 4xGA = 13.63M tok/step, log5)",
                "workdir": "/torchtitan-native",
                "env": p9_nat_env,
                "args": [
                    "--module=torchtitan.experiments.tpu.examples.flint",
                    "--config=flint_40b_p12e_ep1_p9base_sbdp_n49",
                    "--training.steps=15",
                    "--training.mixed_precision_reduce=bfloat16",
                    "--metrics.log_freq=5",
                    "--debug.seed=42",
                    "--tpu_config.mfu_benchmark_start_step=5",
                    f"--training.num_tokens_per_microbatch_per_dp_rank={mb52}",
                    f"--training.num_tokens_per_train_step={4 * world_size * mb52}",
                ],
            },
            # 3. Intra-Chip TP=2 x FSDP=16 with F.embedding padding_idx fix
            tp2_48_fixed,
            # 4. HSDP (DP_rep=2 x FSDP=16) @ mb=64 with FullAC49, 2x GA (8.39M tok/step)
            hsdp_64_ac49,
            # 5. Intra-Chip StaticEP=2 x FSDP=16 + ZeRO-3 + SBDP + BF16-RoPE + PairedGMM-512 + n43 @ mb=56 (5,376 tok/exp), 2x GA
            {
                "name": "v7x_p12f_05_native_intrachip_staticep2_zero3_sbdp_paired512_n43_mb56_ga2_log5",
                "lane": "v7x-16 Intra-Chip StaticEP=2 x FSDP=16 + ZeRO-3 + SBDP + BF16-RoPE + PairedGMM-512 + n43 (mb=56 = 5,376 tok/exp, 2xGA = 7.34M tok/step)",
                "workdir": "/torchtitan-native",
                "env": base_nat_env,
                "args": [
                    "--module=torchtitan.experiments.tpu.examples.flint",
                    "--config=flint_40b_p12f_staticep2_zero3_sbdp_n43",
                    "--training.steps=15",
                    "--training.mixed_precision_reduce=bfloat16",
                    "--metrics.log_freq=5",
                    "--debug.seed=42",
                    "--tpu_config.mfu_benchmark_start_step=5",
                    f"--training.num_tokens_per_microbatch_per_dp_rank={mb56}",
                    f"--training.num_tokens_per_train_step={2 * world_size * mb56}",
                ],
            },
        ]
    elif args.suite == "pass12g":
        mb40 = 40 * 2048
        mb48 = 48 * 2048
        mb56 = 56 * 2048
        p9_scout_flags = filter_supported_xla_flags([
            "--xla_tpu_enable_sparse_core_collective_aggregator=true",
            "--xla_tpu_enable_latency_hiding_layer_scheduler=true",
            "--xla_tpu_use_single_sparse_core_for_reduce_scatter_offload=false",
            "--xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=true",
            "--xla_tpu_enable_sparse_core_hierarchical_reduce_scatter=true",
            "--xla_tpu_dvfs_p_state=7",
            "--xla_enable_async_all_gather=true",
            "--xla_enable_async_all_reduce=true",
            "--xla_tpu_enable_async_all_to_all=true",
            "--xla_tpu_enable_ag_backward_pipelining=true",
            "--xla_tpu_enable_ici_ag_pipelining=true",
            "--xla_tpu_overlap_compute_collective_tc=true",
            "--xla_max_concurrent_async_all_gathers=4",
            "--xla_latency_hiding_scheduler_rerun=2",
            "--xla_enable_transpose_trace=false",
        ])
        base_nat_env = dict(exps[7]["env"])
        p9_nat_env = {
            "LIBTPU_INIT_ARGS": p9_scout_flags,
            "TT_FORCE_CHUNK_OPT": "0",
        }
        tp2_48_ce_fixed = dict(exps[6])
        tp2_48_ce_fixed["name"] = "v7x_p12g_03_hf_pr123_tp2_dp16_sbdp_purefused_sa1024_tile1024_hybridac24_mb48_ga2_log5"
        tp2_48_ce_fixed["lane"] = "v7x-16 Intra-Chip TP=2 x FSDP=16 + SBDP + PureFusedOpt + sa=1024 + tile1024 + HybridAC24 + TP2PallasCE (mb=48/dp_rank, 2xGA = 3.15M tok/step, log5)"
        tp2_48_ce_fixed["env"] = {**exps[6]["env"], "TT_FULL_AC_FIRST_N": "24"}
        tp2_48_ce_fixed["args"] = [
            f"--training.num_tokens_per_microbatch_per_dp_rank={mb48}"
            if a.startswith("--training.num_tokens_per_microbatch_per_dp_rank=")
            else (
                f"--training.num_tokens_per_train_step={2 * (world_size // 2) * mb48}"
                if a.startswith("--training.num_tokens_per_train_step=")
                else a
            )
            for a in exps[6]["args"]
        ]

        hsdp_40_ac49 = dict(exps[5])
        hsdp_40_ac49["name"] = "v7x_p12g_05_hf_pr123_hsdp2x16_sbdp_purefused_sa1024_tile1024_fullac49_mb40_ga2_log5"
        hsdp_40_ac49["lane"] = "v7x-16 HSDP (DP_rep=2 x FSDP=16) + SBDP + PureFusedOpt + sa=1024 + tile1024 + FullAC49 (mb=40 = 1,920 tok/exp, 2xGA = 5.24M tok/step, log5)"
        hsdp_40_ac49["env"] = {**exps[5]["env"], "TT_FULL_AC_FIRST_N": "49"}
        hsdp_40_ac49["args"] = [
            f"--training.num_tokens_per_microbatch_per_dp_rank={mb40}"
            if a.startswith("--training.num_tokens_per_microbatch_per_dp_rank=")
            else (
                f"--training.num_tokens_per_train_step={2 * world_size * mb40}"
                if a.startswith("--training.num_tokens_per_train_step=")
                else a
            )
            for a in exps[5]["args"]
        ]

        exps = [
            # 1. Push past 20.75% Steady / 20.77% Peak: n49 + PureFusedOpt + 16x GA (50.33M tok/step!)
            {
                "name": "v7x_p12g_01_native_p9base_sbdp_bf16rope_fusedopt_n49_mb48_ga16_log5",
                "lane": "v7x-16 Native PR#122 (n49 + PureFusedOpt + 16xGA = 50.33M tok/step, SBDP + BF16-RoPE, mb=48, log5)",
                "workdir": "/torchtitan-native",
                "env": p9_nat_env,
                "args": [
                    "--module=torchtitan.experiments.tpu.examples.flint",
                    "--config=flint_40b_p12g_ep1_p9base_sbdp_fusedopt_n49",
                    "--training.steps=15",
                    "--training.mixed_precision_reduce=bfloat16",
                    "--metrics.log_freq=5",
                    "--debug.seed=42",
                    "--tpu_config.mfu_benchmark_start_step=5",
                    f"--training.num_tokens_per_microbatch_per_dp_rank={mb48}",
                    f"--training.num_tokens_per_train_step={16 * world_size * mb48}",
                ],
            },
            # 2. PureFusedOpt A/B on exact 8xGA 20.75% champion (25.17M tok/step)
            {
                "name": "v7x_p12g_02_native_p9base_sbdp_bf16rope_fusedopt_n49_mb48_ga8_log5",
                "lane": "v7x-16 Native PR#122 (n49 + PureFusedOpt + 8xGA = 25.17M tok/step, SBDP + BF16-RoPE, mb=48, log5)",
                "workdir": "/torchtitan-native",
                "env": p9_nat_env,
                "args": [
                    "--module=torchtitan.experiments.tpu.examples.flint",
                    "--config=flint_40b_p12g_ep1_p9base_sbdp_fusedopt_n49",
                    "--training.steps=15",
                    "--training.mixed_precision_reduce=bfloat16",
                    "--metrics.log_freq=5",
                    "--debug.seed=42",
                    "--tpu_config.mfu_benchmark_start_step=5",
                    f"--training.num_tokens_per_microbatch_per_dp_rank={mb48}",
                    f"--training.num_tokens_per_train_step={8 * world_size * mb48}",
                ],
            },
            # 3. Intra-Chip TP=2 x FSDP=16 with TP=2 Pallas CE + SinksTP + DTensor SplashAttention + EmbPadFix
            tp2_48_ce_fixed,
            # 4. Intra-Chip StaticEP=2 x FSDP=16 + ZeRO-3 + SBDP (fuse_route_score_backward=False) @ mb=56 (5,376 tok/exp)
            {
                "name": "v7x_p12g_04_native_intrachip_staticep2_zero3_sbdp_paired512_n43_mb56_ga2_log5",
                "lane": "v7x-16 Intra-Chip StaticEP=2 x FSDP=16 + ZeRO-3 + SBDP + BF16-RoPE + PairedGMM-512 + n43 (mb=56 = 5,376 tok/exp, 2xGA = 7.34M tok/step)",
                "workdir": "/torchtitan-native",
                "env": base_nat_env,
                "args": [
                    "--module=torchtitan.experiments.tpu.examples.flint",
                    "--config=flint_40b_p12g_staticep2_zero3_sbdp_n43",
                    "--training.steps=15",
                    "--training.mixed_precision_reduce=bfloat16",
                    "--metrics.log_freq=5",
                    "--debug.seed=42",
                    "--tpu_config.mfu_benchmark_start_step=5",
                    f"--training.num_tokens_per_microbatch_per_dp_rank={mb56}",
                    f"--training.num_tokens_per_train_step={2 * world_size * mb56}",
                ],
            },
            # 5. HSDP (DP_rep=2 x FSDP=16) @ mb=40 (<20.66 GiB slab) + FullAC49, 2x GA
            hsdp_40_ac49,
        ]
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
        tp_deg = 2 if any("tensor_parallel_degree=2" in a for a in exp["args"]) else 1
        print(f"\n=== [{idx + 1}/{len(exps)}] {exp['name']} ({exp['lane']}) ===", flush=True)
        t0 = time.time()
        try:
            rc, log_chunks = run_with_oom_watchdog(
                cmd, exp["workdir"], env, node_rank, args.per_run_timeout, idle_timeout_s=1200.0
            )
        except Exception as exc:
            rc = -1
            log_chunks = [f"[flint-v7x-p12] Exception: {exc}\n"]
        dur = time.time() - t0
        if node_rank == 0:
            raw_log = "".join(log_chunks)
            (out_dir / f"{exp['name']}.log").write_text(raw_log)
            metrics = parse_metrics(raw_log, args.chip_type, tp_degree=tp_deg)
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
        time.sleep(20 if rc != 0 else 10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
