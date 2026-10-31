#!/usr/bin/env python3
"""Pass 9B & Pass 9C TPU v5p-32 (2x4x4) & v5p-16 (2x2x4) MFU Breakthrough Sweep for Flint.

Builds directly on our Pass 9 record (`v5p_p9_02` @ 25.76% Steady / 25.81% Peak MFU on v5p-16 and
24.88% Steady / 24.91% Peak MFU on v5p-32, which used only 22.84 GiB / 95 GiB = 24.04% of v5p HBM!)
AND brings our verified `StaticTpuAllToAllTokenDispatcher` (`StaticEP=4, 8, 16` + `ZeRO-2`) to v5p:

  Suite `pass9b` (runs on BOTH `tpu-v5p-32-pool` [32 chips] AND `tpu-v5p-16-pool` [16 chips]):
    1. `mb=96` (196,608 tok/rank = 4,608 tok/exp, ~31 GiB HBM) + HF PR#123 + SBDP + FusedOpt + sa=1024 + HybridAC N=16 + log_freq=5
    2. `mb=128` (262,144 tok/rank = 6,144 tok/exp, ~40 GiB HBM) + HF PR#123 + SBDP + FusedOpt + sa=1024 + HybridAC N=16 + tile_m=512 + log_freq=5
    3. `mb=160` (327,680 tok/rank = 7,680 tok/exp, ~50 GiB HBM) + HF PR#123 + SBDP + FusedOpt + sa=1024 + HybridAC N=24 + tile_m=512 + log_freq=5
    4. `mb=96` + `HSDP 2x(W/2)` (`replicate=2, shard=W/2`) + `2xGA` + SBDP + FusedOpt + sa=1024 + HybridAC N=20 + log_freq=5
    5. Native `TpuGraphTrainer` + `StaticEP=8` + `ZeRO-2` + PairedGMM-256 + `config.libtpu_init_flags` (`mb=48`, `2xGA`, `log_freq=5`)

  Suite `pass9c` (runs in parallel on `tpu-v5p-16-pool` [16 chips]):
    1. `StaticEP=4` (100% Intra-Host AllToAll across 4 v5p chips/host!) + `ZeRO-2` + PairedGMM-256 (`n40`, `mb=48`, `2xGA`, `log_freq=5`)
    2. `StaticEP=8` + `ZeRO-2` + PairedGMM-256 (`n45`, `mb=64`, `1x`, `24,576` tok/exp, `log_freq=5`)
    3. `StaticEP=16` (`0` expert AG/RS on v5p-16! `100.6 MB/layer`, `36,864` tok/exp) + `ZeRO-2` + PairedGMM-512 (`n45`, `mb=48`, `2xGA`, `log_freq=5`)
    4. `WholeStep` (`graph_split.enabled=False, enable_remat_barriers=True`) + `StaticEP=16` + `ZeRO-2` (`n49`, `mb=32`, `1x`, `log_freq=5`)
    5. HF PR#123 + SBDP + FusedOpt + sa=1024 + `AttnOnlyAC (N=0, 0% attn recompute!)` @ `mb=96` (`1x`, `log_freq=5`)
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
# Pass 9B/9C StaticTpuAllToAllTokenDispatcher (100% Static-Shape, Scatter-Free TPU EP)
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
  tile_align = int(_sep_os.environ.get("TT_STATIC_EP_ALIGN", "256"))
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

NATIVE_V5P_P9BC_PATCH = '''

# ==============================================================================
# Pass 9B / 9C Flint v5p StaticEP Configs (EP=4/8/16 + ZeRO-2 + WholeStep)
# ==============================================================================
from torchtitan.experiments.tpu.token_dispatcher import TpuAllToAllTokenDispatcher

_P9BC_BF16_ROPE = (
    "torchtitan.experiments.tpu.overrides.bf16_rope"
    ".tpu_activation_dtype_cossin_rope"
)

_P9BC_V5P_FLAGS = {
    "xla_tpu_scoped_vmem_limit_kib": 81920,
    "xla_tpu_enable_offloading_copy_to_sparsecore": False,
    "xla_tpu_copy_insertion_use_region_analysis": True,
    "xla_tpu_copy_insertion_use_region_analysis_limit": 1000,
    "xla_tpu_prefer_async_allgather_to_allreduce": True,
    "xla_enable_async_all_gather": True,
    "xla_enable_async_all_reduce": True,
    "xla_tpu_enable_async_all_to_all": True,
    "xla_tpu_overlap_compute_collective_tc": True,
}


def _build_v5p_staticep_zero2(
    *, ep_degree: int, tile_m: int, save_attn_layers: int, wholestep: bool = False
) -> TpuGraphTrainer.Config:
  config = flint_40b_v005_gt_rb() if wholestep else flint_40b_v5p_n40()
  config.parallelism.expert_parallel_degree = ep_degree
  config.parallelism.fsdp_reshard_after_forward = "never"
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
  if _P9BC_BF16_ROPE not in config.override.imports:
    config.override.imports.append(_P9BC_BF16_ROPE)
  if wholestep:
    config.compile.graph_split.enabled = False
    config.compile.enable_remat_barriers = True
  else:
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
  config.libtpu_init_flags = tpu_flags.merge_flags(
      config.libtpu_init_flags,
      tpu_flags.ICI_AG_PIPELINING,
      _P9BC_V5P_FLAGS,
  )
  return config


def flint_40b_v5p_p9bc_staticep4_zero2_n40() -> TpuGraphTrainer.Config:
  return _build_v5p_staticep_zero2(ep_degree=4, tile_m=256, save_attn_layers=40)


def flint_40b_v5p_p9bc_staticep8_zero2_n45() -> TpuGraphTrainer.Config:
  return _build_v5p_staticep_zero2(ep_degree=8, tile_m=256, save_attn_layers=45)


def flint_40b_v5p_p9bc_staticep16_zero2_n45() -> TpuGraphTrainer.Config:
  return _build_v5p_staticep_zero2(ep_degree=16, tile_m=512, save_attn_layers=45)


def flint_40b_v5p_p9bc_wholestep_staticep16_zero2_n49() -> TpuGraphTrainer.Config:
  return _build_v5p_staticep_zero2(ep_degree=16, tile_m=512, save_attn_layers=49, wholestep=True)


__all__.extend([
    "flint_40b_v5p_p9bc_staticep4_zero2_n40",
    "flint_40b_v5p_p9bc_staticep8_zero2_n45",
    "flint_40b_v5p_p9bc_staticep16_zero2_n45",
    "flint_40b_v5p_p9bc_wholestep_staticep16_zero2_n49",
])
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


def apply_runtime_patches() -> None:
    td_path = Path("/torchtitan-native/torchtitan/experiments/tpu/token_dispatcher.py")
    if td_path.exists():
        txt = td_path.read_text()
        if "_static_tpu_ep_dispatch" not in txt:
            td_path.write_text(txt + STATIC_EP_DISPATCHER_PATCH)
            print("[flint-v5p-p9bc] Patched Native token_dispatcher.py with StaticTpuAllToAllTokenDispatcher", flush=True)

    nat_reg = Path("/torchtitan-native/torchtitan/experiments/tpu/examples/flint/config_registry.py")
    if nat_reg.exists():
        txt = nat_reg.read_text()
        if "flint_40b_v5p_p9bc_staticep4_zero2_n40" not in txt:
            nat_reg.write_text(txt + NATIVE_V5P_P9BC_PATCH)
            print("[flint-v5p-p9bc] Patched Native config_registry.py with Pass 9B/9C configs", flush=True)

    hf_ac = Path("/torchtitan-hf/torchtitan/experiments/tpu/activation_checkpoint.py")
    if hf_ac.exists():
        txt = hf_ac.read_text()
        if "_hybrid_attn_only_wrap_v3" not in txt:
            hf_ac.write_text(txt + HF_HYBRID_AC_PATCH_V3)
            print("[flint-v5p-p9bc] Patched HF activation_checkpoint.py with HybridAC v3", flush=True)

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
            print("[flint-v5p-p9bc] Patched HF model.py & parallelize.py with TT_HF_SBDP=1 hook", flush=True)


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
    print(f"[flint-v5p-p9bc] Validated XLA flags: kept={len(kept)} dropped={dropped}", flush=True)
    return " ".join(kept)


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


def wait_for_master_dns(master_addr: str, timeout_s: int = 300) -> str:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            return socket.gethostbyname(master_addr)
        except socket.gaierror:
            time.sleep(3)
    return master_addr


def build_v5p_pass9bc_experiments(suite: str, world_size: int, hf_model_path: str, steps: int) -> list[dict]:
    mb32 = 32 * 2048    # 65,536 tok/rank
    mb48 = 48 * 2048    # 98,304 tok/rank
    mb64 = 64 * 2048    # 131,072 tok/rank
    mb96 = 96 * 2048    # 196,608 tok/rank  (4,608 tok/expert @ EP=1)
    mb128 = 128 * 2048  # 262,144 tok/rank  (6,144 tok/expert @ EP=1)
    mb160 = 160 * 2048  # 327,680 tok/rank  (7,680 tok/expert @ EP=1)

    v5p_flags = filter_supported_xla_flags([
        "--xla_tpu_scoped_vmem_limit_kib=81920",
        "--xla_tpu_enable_offloading_copy_to_sparsecore=false",
        "--xla_tpu_copy_insertion_use_region_analysis=true",
        "--xla_tpu_copy_insertion_use_region_analysis_limit=1000",
        "--xla_tpu_prefer_async_allgather_to_allreduce=true",
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

    hf_comp_opt = probe_compile_opt_args("/torchtitan-hf")
    hf_splash_1024 = [
        "--splash_attention_kernel.sa_block_q=1024",
        "--splash_attention_kernel.sa_block_kv=1024",
        "--splash_attention_kernel.sa_block_kv_compute=512",
        "--splash_attention_kernel.sa_block_q_dkv=1024",
        "--splash_attention_kernel.sa_block_kv_dkv=1024",
        "--splash_attention_kernel.sa_block_kv_dkv_compute=512",
    ]
    gmm_env_256 = (
        "use_pallas_paired_grouped_mm=true,"
        "pallas_grouped_mm_tile_m=256,"
        "pallas_grouped_mm_tgmm_tiling=256:1024:0,"
        "pallas_grouped_mm_w2_tgmm_tiling=256:512:0,"
        "pallas_grouped_mm_w2_transposed_tgmm_tiling=256:0:1024"
    )
    gmm_env_512 = (
        "use_pallas_paired_grouped_mm=true,"
        "pallas_grouped_mm_tile_m=512,"
        "pallas_grouped_mm_tgmm_tiling=512:1024:0,"
        "pallas_grouped_mm_w2_tgmm_tiling=256:512:0,"
        "pallas_grouped_mm_w2_transposed_tgmm_tiling=256:0:1024"
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
        "--metrics.log_freq=5",
        "--debug.seed=42",
        "--loss.b_block_size=1024",
        *hf_splash_1024,
        *hf_comp_opt,
    ]

    if suite == "pass9b":
        hsdp_shard = world_size // 2
        return [
            {
                "name": "v5p_p9b_01_hf_sbdp_sa1024_hybridac16_mb96_1x_log5",
                "lane": f"v5p-{world_size} HF PR#123 + SBDP + FusedOpt + sa=1024 + HybridAC16 + log5 (mb=96 = 4,608 tok/exp, 1x)",
                "workdir": "/torchtitan-hf",
                "env": {
                    "TT_HF_GMM": gmm_env_256,
                    "TT_HF_SBDP": "1",
                    "TT_FULL_AC_FIRST_N": "16",
                    "TORCH_TPU_DEFER_AND_FUSE": "1",
                    "LIBTPU_INIT_ARGS": v5p_flags,
                },
                "args": [
                    *hf_base,
                    f"--training.num_tokens_per_microbatch_per_dp_rank={mb96}",
                    f"--training.num_tokens_per_train_step={world_size * mb96}",
                ],
            },
            {
                "name": "v5p_p9b_02_hf_sbdp_sa1024_hybridac16_tile512_mb128_1x_log5",
                "lane": f"v5p-{world_size} HF PR#123 + SBDP + FusedOpt + sa=1024 + HybridAC16 + tile512 + log5 (mb=128 = 6,144 tok/exp, 1x)",
                "workdir": "/torchtitan-hf",
                "env": {
                    "TT_HF_GMM": gmm_env_512,
                    "TT_HF_SBDP": "1",
                    "TT_FULL_AC_FIRST_N": "16",
                    "TORCH_TPU_DEFER_AND_FUSE": "1",
                    "LIBTPU_INIT_ARGS": v5p_flags,
                },
                "args": [
                    *hf_base,
                    f"--training.num_tokens_per_microbatch_per_dp_rank={mb128}",
                    f"--training.num_tokens_per_train_step={world_size * mb128}",
                ],
            },
            {
                "name": "v5p_p9b_03_hf_sbdp_sa1024_hybridac24_tile512_mb160_1x_log5",
                "lane": f"v5p-{world_size} HF PR#123 + SBDP + FusedOpt + sa=1024 + HybridAC24 + tile512 + log5 (mb=160 = 7,680 tok/exp, 1x)",
                "workdir": "/torchtitan-hf",
                "env": {
                    "TT_HF_GMM": gmm_env_512,
                    "TT_HF_SBDP": "1",
                    "TT_FULL_AC_FIRST_N": "24",
                    "TORCH_TPU_DEFER_AND_FUSE": "1",
                    "LIBTPU_INIT_ARGS": v5p_flags,
                },
                "args": [
                    *hf_base,
                    f"--training.num_tokens_per_microbatch_per_dp_rank={mb160}",
                    f"--training.num_tokens_per_train_step={world_size * mb160}",
                ],
            },
            {
                "name": f"v5p_p9b_04_hf_hsdp2x{hsdp_shard}_sbdp_sa1024_hybridac20_mb96_ga2_log5",
                "lane": f"v5p-{world_size} HF PR#123 + HSDP 2x{hsdp_shard} + SBDP + sa=1024 + HybridAC20 (mb=96, 2xGA)",
                "workdir": "/torchtitan-hf",
                "env": {
                    "TT_HF_GMM": gmm_env_256,
                    "TT_HF_SBDP": "1",
                    "TT_FULL_AC_FIRST_N": "20",
                    "TORCH_TPU_DEFER_AND_FUSE": "1",
                    "LIBTPU_INIT_ARGS": v5p_flags,
                },
                "args": [
                    *hf_base,
                    "--parallelism.data_parallel_replicate_degree=2",
                    f"--parallelism.data_parallel_shard_degree={hsdp_shard}",
                    f"--training.num_tokens_per_microbatch_per_dp_rank={mb96}",
                    f"--training.num_tokens_per_train_step={2 * world_size * mb96}",
                ],
            },
            {
                "name": "v5p_p9b_05_native_staticep8_zero2_paired256_mb48_ga2_log5",
                "lane": f"v5p-{world_size} Native TpuGraphTrainer + StaticEP=8 + ZeRO-2 + PairedGMM-256 (18,432 tok/exp, mb=48 2xGA)",
                "workdir": "/torchtitan-native",
                "env": {
                    "LIBTPU_INIT_ARGS": v5p_flags,
                    "TT_STATIC_EP_CAP_FACTOR": "1.15",
                    "TT_STATIC_EP_ALIGN": "256",
                },
                "args": [
                    "--module=torchtitan.experiments.tpu.examples.flint",
                    "--config=flint_40b_v5p_p9bc_staticep8_zero2_n45",
                    f"--training.steps={steps}",
                    "--training.mixed_precision_reduce=bfloat16",
                    "--metrics.log_freq=5",
                    "--debug.seed=42",
                    "--tpu_config.mfu_benchmark_start_step=5",
                    f"--training.num_tokens_per_microbatch_per_dp_rank={mb48}",
                    f"--training.num_tokens_per_train_step={2 * world_size * mb48}",
                ],
            },
        ]

    # Suite pass9c: StaticEP=4 / 8 / 16 + WholeStep + Zero-Recompute mb=96 on tpu-v5p-16-pool (16 chips)
    return [
        {
            "name": "v5p_p9c_01_staticep4_zero2_paired256_mb48_ga2_log5",
            "lane": f"v5p-{world_size} Intra-Host StaticEP=4 + ZeRO-2 + PairedGMM-256 (100% intra-host A2A, 9,216 tok/exp, mb=48 2xGA)",
            "workdir": "/torchtitan-native",
            "env": {
                "LIBTPU_INIT_ARGS": v5p_flags,
                "TT_STATIC_EP_CAP_FACTOR": "1.15",
                "TT_STATIC_EP_ALIGN": "256",
            },
            "args": [
                "--module=torchtitan.experiments.tpu.examples.flint",
                "--config=flint_40b_v5p_p9bc_staticep4_zero2_n40",
                f"--training.steps={steps}",
                "--training.mixed_precision_reduce=bfloat16",
                "--metrics.log_freq=5",
                "--debug.seed=42",
                "--tpu_config.mfu_benchmark_start_step=5",
                f"--training.num_tokens_per_microbatch_per_dp_rank={mb48}",
                f"--training.num_tokens_per_train_step={2 * world_size * mb48}",
            ],
        },
        {
            "name": "v5p_p9c_02_staticep8_zero2_paired256_mb64_1x_log5",
            "lane": f"v5p-{world_size} StaticEP=8 + ZeRO-2 + PairedGMM-256 (24,576 tok/exp, n45, mb=64 1x)",
            "workdir": "/torchtitan-native",
            "env": {
                "LIBTPU_INIT_ARGS": v5p_flags,
                "TT_STATIC_EP_CAP_FACTOR": "1.15",
                "TT_STATIC_EP_ALIGN": "256",
            },
            "args": [
                "--module=torchtitan.experiments.tpu.examples.flint",
                "--config=flint_40b_v5p_p9bc_staticep8_zero2_n45",
                f"--training.steps={steps}",
                "--training.mixed_precision_reduce=bfloat16",
                "--metrics.log_freq=5",
                "--debug.seed=42",
                "--tpu_config.mfu_benchmark_start_step=5",
                f"--training.num_tokens_per_microbatch_per_dp_rank={mb64}",
                f"--training.num_tokens_per_train_step={world_size * mb64}",
            ],
        },
        {
            "name": "v5p_p9c_03_staticep16_zero2_paired512_mb48_ga2_log5",
            "lane": f"v5p-{world_size} StaticEP=16 + ZeRO-2 + PairedGMM-512 (0 expert AG/RS, 36,864 tok/exp, n45, mb=48 2xGA)",
            "workdir": "/torchtitan-native",
            "env": {
                "LIBTPU_INIT_ARGS": v5p_flags,
                "TT_STATIC_EP_CAP_FACTOR": "1.15",
                "TT_STATIC_EP_ALIGN": "512",
            },
            "args": [
                "--module=torchtitan.experiments.tpu.examples.flint",
                "--config=flint_40b_v5p_p9bc_staticep16_zero2_n45",
                f"--training.steps={steps}",
                "--training.mixed_precision_reduce=bfloat16",
                "--metrics.log_freq=5",
                "--debug.seed=42",
                "--tpu_config.mfu_benchmark_start_step=5",
                f"--training.num_tokens_per_microbatch_per_dp_rank={mb48}",
                f"--training.num_tokens_per_train_step={2 * world_size * mb48}",
            ],
        },
        {
            "name": "v5p_p9c_04_wholestep_staticep16_zero2_mb32_1x_log5",
            "lane": f"v5p-{world_size} WholeStep Single-Graph + StaticEP=16 + ZeRO-2 + PairedGMM-512 (n49, mb=32 1x)",
            "workdir": "/torchtitan-native",
            "env": {
                "LIBTPU_INIT_ARGS": v5p_flags,
                "TT_STATIC_EP_CAP_FACTOR": "1.15",
                "TT_STATIC_EP_ALIGN": "512",
            },
            "args": [
                "--module=torchtitan.experiments.tpu.examples.flint",
                "--config=flint_40b_v5p_p9bc_wholestep_staticep16_zero2_n49",
                f"--training.steps={steps}",
                "--training.mixed_precision_reduce=bfloat16",
                "--metrics.log_freq=5",
                "--debug.seed=42",
                "--tpu_config.mfu_benchmark_start_step=5",
                f"--training.num_tokens_per_microbatch_per_dp_rank={mb32}",
                f"--training.num_tokens_per_train_step={world_size * mb32}",
            ],
        },
        {
            "name": "v5p_p9c_05_hf_sbdp_sa1024_attnac_zero_recomp_mb96_1x_log5",
            "lane": f"v5p-{world_size} HF PR#123 + SBDP + FusedOpt + sa=1024 + AttnOnlyAC (0% attn recompute!) (mb=96, 1x)",
            "workdir": "/torchtitan-hf",
            "env": {
                "TT_HF_GMM": gmm_env_256,
                "TT_HF_SBDP": "1",
                "TT_FULL_AC_FIRST_N": "0",
                "TORCH_TPU_DEFER_AND_FUSE": "1",
                "LIBTPU_INIT_ARGS": v5p_flags,
            },
            "args": [
                *hf_base,
                f"--training.num_tokens_per_microbatch_per_dp_rank={mb96}",
                f"--training.num_tokens_per_train_step={world_size * mb96}",
            ],
        },
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
        peak_tf = PEAK_BF16_TFLOPS_PER_RANK.get(chip_type, 459.0) * 1e12
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
    peak_tf = PEAK_BF16_TFLOPS_PER_RANK.get(chip_type, 459.0) * 1e12
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
        f"# Flint (`RecursiveSI/flint` 40B MoE) TPU v5p `{suite.upper()}` (`mb=96..160` + `StaticEP=4/8/16 ZeRO-2`) — `{slice_name}`",
        "",
        f"- **Updated (UTC):** `{payload['updated_at_utc']}`",
        f"- **Chip / World Size:** `{chip_type}` (`{world_size}` chips/ranks, `{PEAK_BF16_TFLOPS_PER_RANK.get(chip_type)}` BF16 TFLOP/s/rank)",
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
    parser.add_argument("--suite", choices=["pass9b", "pass9c"], default="pass9b")
    parser.add_argument("--chip-type", choices=["v7x", "v5p", "v6e"], default="v5p")
    parser.add_argument("--nnodes", type=int, required=True)
    parser.add_argument("--nproc-per-node", type=int, default=4)
    parser.add_argument("--master-addr", type=str, required=True)
    parser.add_argument("--slice-name", type=str, required=True)
    parser.add_argument("--hf-model", type=str, default="/models/models/RecursiveSI/flint")
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--base-port", type=int, default=29811)
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

    exps = build_v5p_pass9bc_experiments(args.suite, world_size, args.hf_model, args.steps)
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
            log_chunks = [f"[flint-v5p-p9bc] Exception: {exc}\n"]
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
                f"[flint-v5p-{args.suite}] SAVED {exp['name']}: status={status} "
                f"steady_tps={metrics.get('steady_tps', 0)} steady_mfu={metrics.get('steady_mfu_pct', 0)}%",
                flush=True,
            )
        time.sleep(8)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
