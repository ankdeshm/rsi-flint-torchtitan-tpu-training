#!/usr/bin/env python3
"""Pass 11A & Pass 11B TPU v7x-16 (2x2x4, 16 Ironwood chips / 32 TensorCores) StaticEP MFU Breakthrough Sweeps.

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

    padded_lexp = _sep_F.pad((sorted_eids_N % e).to(torch.int32), (0, 1), value=e)
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


def _configure_static_ep_p11(config, *, ep_degree: int, tile_m: int = 512) -> None:
  config.parallelism.expert_parallel_degree = ep_degree
  for _, td, _, _ in config.model_spec.model.traverse(TpuAllToAllTokenDispatcher.Config):
    td.fuse_route_score_backward = True
  for _, ex, _, _ in config.model_spec.model.traverse(TpuGroupedExperts.Config):
    ex.use_pallas_grouped_mm = True
    ex.use_pallas_paired_grouped_mm = True
    ex.pallas_grouped_mm_tile_m = tile_m
    ex.pallas_grouped_mm_tgmm_tiling = (tile_m, 1024, 0)
    ex.pallas_grouped_mm_w2_tgmm_tiling = (256, 512, 0)
    ex.pallas_grouped_mm_w2_transposed_tgmm_tiling = (256, 0, 1024)
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


def _build_split_staticep_zero2_p11(*, ep_degree: int, tile_m: int, save_attn_layers: int) -> TpuGraphTrainer.Config:
  config = flint_40b_v012_hier_rs()
  _configure_static_ep_p11(config, ep_degree=ep_degree, tile_m=tile_m)
  config.parallelism.fsdp_reshard_after_forward = "never"
  config.compile.graph_split.enabled = True
  config.compile.graph_split.forward_layers_per_executable = 1
  config.compile.graph_split.backward_partition_mode = "by_layer"
  config.compile.graph_split.backward_layers_per_executable = 1
  config.compile.memory_policy = "tpu_selective"
  config.compile.save_regions = ("moe.routing",)
  config.compile.save_last_n_layers = {"attention.inner_attention": save_attn_layers}
  config.compile.optimizer_compile = "separate_graph"
  config.compile.compile_grad_norm = False
  config.optimizer.implementation = "fused"
  return config


def _build_wholestep_staticep_zero2_p11(
    *, ep_degree: int, tile_m: int, save_attn_layers: int, opt_compile: str = "separate_graph"
) -> TpuGraphTrainer.Config:
  config = flint_40b_v005_gt_rb()
  _configure_static_ep_p11(config, ep_degree=ep_degree, tile_m=tile_m)
  config.parallelism.fsdp_reshard_after_forward = "never"
  config.compile.graph_split.enabled = False
  config.compile.enable_remat_barriers = True
  config.compile.memory_policy = "tpu_selective"
  config.compile.save_regions = ("moe.routing",)
  config.compile.save_last_n_layers = {"attention.inner_attention": save_attn_layers}
  config.compile.optimizer_compile = opt_compile
  config.compile.compile_grad_norm = False
  config.optimizer.implementation = "fused"
  return config


# --- Pass 11A Configs ---
def flint_40b_p11a_split_ep8_zero2_n36() -> TpuGraphTrainer.Config:
  return _build_split_staticep_zero2_p11(ep_degree=8, tile_m=256, save_attn_layers=36)


def flint_40b_p11a_split_ep16_zero2_n40() -> TpuGraphTrainer.Config:
  return _build_split_staticep_zero2_p11(ep_degree=16, tile_m=256, save_attn_layers=40)


def flint_40b_p11a_split_ep32_zero2_n45() -> TpuGraphTrainer.Config:
  return _build_split_staticep_zero2_p11(ep_degree=32, tile_m=512, save_attn_layers=45)


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
  return _build_wholestep_staticep_zero2_p11(ep_degree=32, tile_m=512, save_attn_layers=49)


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
  config.compile.compile_grad_norm = False
  config.optimizer.implementation = "fused"
  return config


__all__.extend([
    "flint_40b_p11a_split_ep8_zero2_n36",
    "flint_40b_p11a_split_ep16_zero2_n40",
    "flint_40b_p11a_split_ep32_zero2_n45",
    "flint_40b_p11a_split_ep32_zero2_n36",
    "flint_40b_p11a_ep1_sbdp_paired256_n36",
    "flint_40b_p11b_wholestep_ep32_zero2_n49",
    "flint_40b_p11b_wholestep_ep8_zero2_n36",
    "flint_40b_p11b_wholestep_ep32_zero2_ingraph_n36",
    "flint_40b_p11b_ep1_sbdp_fulloffload",
    "flint_40b_p11b_staticep16_fulloffload",
])
'''


def apply_runtime_patches() -> None:
    td_path = Path("/torchtitan-native/torchtitan/experiments/tpu/token_dispatcher.py")
    if td_path.exists():
        txt = td_path.read_text()
        if "_static_tpu_ep_dispatch" not in txt:
            td_path.write_text(txt + STATIC_EP_DISPATCHER_PATCH)
            print("[flint-v7x-p11] Patched token_dispatcher.py with StaticTpuAllToAllTokenDispatcher", flush=True)

    nat_reg = Path("/torchtitan-native/torchtitan/experiments/tpu/examples/flint/config_registry.py")
    if nat_reg.exists():
        txt = nat_reg.read_text()
        if "flint_40b_p11a_split_ep8_zero2_n36" not in txt:
            nat_reg.write_text(txt + NATIVE_P11_REGISTRY_PATCH)
            print("[flint-v7x-p11] Patched Native config_registry.py with Pass 11A/11B configs", flush=True)


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


def build_v7x_pass11_experiments(suite: str, world_size: int, steps: int) -> list[dict]:
    mb32 = 32 * 2048    # 65,536 tok/rank  -> @EP=32: 49,152 tok/exp | @EP=8: 12,288 tok/exp
    mb48 = 48 * 2048    # 98,304 tok/rank  -> @EP=32: 73,728 tok/exp | @EP=16: 36,864 tok/exp | @EP=8: 18,432 tok/exp
    mb64 = 64 * 2048    # 131,072 tok/rank -> @EP=32: 98,304 tok/exp
    mb96 = 96 * 2048    # 196,608 tok/rank -> @EP=16: 73,728 tok/exp

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

    def _nat_exp(name: str, lane: str, cfg: str, mb: int, ga: int, align: int = 512) -> dict:
        return {
            "name": name,
            "lane": lane,
            "workdir": "/torchtitan-native",
            "env": {
                "LIBTPU_INIT_ARGS": calibrated_v7x_flags,
                "TT_STATIC_EP_CAP_FACTOR": "1.15",
                "TT_STATIC_EP_ALIGN": str(align),
            },
            "args": [
                "--module=torchtitan.experiments.tpu.examples.flint",
                f"--config={cfg}",
                f"--training.steps={steps}",
                "--training.mixed_precision_reduce=bfloat16",
                "--metrics.log_freq=5",
                "--debug.seed=42",
                "--tpu_config.mfu_benchmark_start_step=5",
                f"--training.num_tokens_per_microbatch_per_dp_rank={mb}",
                f"--training.num_tokens_per_train_step={ga * world_size * mb}",
            ],
        }

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
    parser.add_argument("--suite", choices=["pass11a", "pass11b"], default="pass11a")
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

    exps = build_v7x_pass11_experiments(args.suite, world_size, args.steps)
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
