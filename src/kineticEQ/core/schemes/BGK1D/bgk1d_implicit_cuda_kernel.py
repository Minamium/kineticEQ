# kineticEQ/src/kineticEQ/core/schemes/BGK1D/bgk1d_implicit_cuda_kernel.py
from __future__ import annotations
from typing import Callable
import torch, math
from kineticEQ.api.config import Config
from kineticEQ.core.states.state_1d import State1D1V
from kineticEQ.core.schemes.BGK1D.bgk1d_utils.bgk1d_implicit_ws import ImplicitWorkspace, allocate_implicit_workspace
from kineticEQ.core.schemes.BGK1D.bgk1d_utils.bgk1d_set_initial_condition import set_initial_condition
from kineticEQ.core.schemes.BGK1D.bgk1d_utils.bgk1d_maxwellian import maxwellian
from kineticEQ.core.schemes.BGK1D.bgk1d_utils.bgk1d_check_CFL import bgk1d_check_CFL
from kineticEQ.core.schemes.BGK1D.bgk1d_utils.bgk1d_compute_moments import calculate_moments
from kineticEQ.cuda_kernel.compile import load_implicit_fused
from kineticEQ.cuda_kernel.compile import load_gtsv
from kineticEQ.CNN.BGK1D1V.models import MomentCNN1D

import logging
logger = logging.getLogger(__name__)
Stepper = Callable[[int], None]

# CNNモデルのロード
def _load_ckpt_state(ckpt_path: str) -> dict:
    obj = torch.load(ckpt_path, map_location="cpu")
    if isinstance(obj, dict):
        for k in ("model", "model_state", "state_dict", "model_state_dict"):
            if k in obj and isinstance(obj[k], dict):
                obj = obj[k]
                break
    if not isinstance(obj, dict):
        raise ValueError(f"Unsupported checkpoint format: {type(obj)}")
    if any(k.startswith("module.") for k in obj.keys()):
        obj = {k[len("module."):]: v for k, v in obj.items()}
    return obj

def _infer_arch_from_state(state: dict) -> tuple[int,int,int,int]:
    # stem.0.weight: (hidden, in_ch, kernel)
    w = state.get("stem.0.weight", None)
    if w is None or w.ndim != 3:
        raise KeyError("Cannot infer arch: missing stem.0.weight")
    hidden = int(w.shape[0])
    in_ch  = int(w.shape[1])
    kernel = int(w.shape[2])

    idx = set()
    for k in state.keys():
        if k.startswith("blocks."):
            p = k.split(".")
            if len(p) >= 2 and p[1].isdigit():
                idx.add(int(p[1]))
    if not idx:
        raise ValueError("Cannot infer n_blocks from blocks.{i}.* keys")
    n_blocks = max(idx) + 1
    return in_ch, hidden, kernel, n_blocks

def load_moments_cnn_model(model_path: str, device: torch.device) -> MomentCNN1D:
    sd = _load_ckpt_state(model_path)
    in_ch, hidden, kernel, n_blocks = _infer_arch_from_state(sd)

    model = MomentCNN1D(in_ch=in_ch, hidden=hidden, out_ch=3, kernel=kernel, n_blocks=n_blocks)
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    return model


# モデルによる推論
@torch.no_grad()
def predict_next_moments_delta(
    model: MomentCNN1D,
    n0: torch.Tensor, u0: torch.Tensor, T0: torch.Tensor,
    logdt: float, logtau: float,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor
]:
    nx = n0.numel()
    x = torch.empty((1, 5, nx), device=n0.device, dtype=torch.float32)
    x[0, 0] = n0.to(torch.float32)
    x[0, 1] = u0.to(torch.float32)
    x[0, 2] = T0.to(torch.float32)
    x[0, 3].fill_(float(logdt))
    x[0, 4].fill_(float(logtau))

    dy = model(x)[0]  # (3, nx) float32
    dn = dy[0].to(n0.dtype)
    dT = dy[2].to(T0.dtype)
    du = dy[1].to(n0.dtype)

    u1 = u0 + du
    n1 = n0 + dn
    T1 = T0 + dT
    return n1[1:-1], u1[1:-1], T1[1:-1], dn, du, dT

@torch.no_grad()
def step(
    state: State1D1V, 
    cfg: Config, 
    ws: ImplicitWorkspace, 
    cuda_module, 
    gtsv_module, 
    num_steps: int,
    model: MomentCNN1D | None = None,
) -> tuple[State1D1V, dict]:
    # 初期候補：前ステップを参照, 外部フックがあればそちらを優先, CNNモデルを最優先
    if cfg.model_cfg.scheme_params.moments_cnn_modelpath is not None:
        # 入力のモーメント値計算
        n0, u0, T0 = calculate_moments(state, state.f)

        state.n = n0.clone()
        state.u = u0.clone()
        state.T = T0.clone()

        state.n[1:-1], state.u[1:-1], state.T[1:-1], _, _, _ = predict_next_moments_delta(
            model,
            n0, u0, T0,
            math.log10(cfg.model_cfg.time.dt),
            math.log10(cfg.model_cfg.params.tau_tilde)
        )
        ws.fz = maxwellian(state)
    else:
        init_fz = getattr(ws, "_init_fz", None)
        if init_fz is None:
            ws.fz.copy_(state.f)
        else:
            ws.fz.copy_(init_fz)   # shape (nx, nv)
            ws._init_fz = None

    residual_val = float('inf')

    # scheme_params から取得
    picard_iter = cfg.model_cfg.scheme_params.picard_iter
    picard_tol = cfg.model_cfg.scheme_params.picard_tol
    abs_tol = cfg.model_cfg.scheme_params.abs_tol

    latest = ws.fz

    for z in range(picard_iter):
        # (a,b,c,B) を一括構築（Maxwellの境界寄与も旧実装と同等）
        cuda_module.build_system_fused(
            state.f, ws.fz, state.v,
            float(state.dv), float(cfg.model_cfg.time.dt), float(state.dx),
            float(cfg.model_cfg.params.tau_tilde), float(state.inv_sqrt_2pi.item()),
            ws.dl, ws.dd, ws.du, ws.B
        )

        # 既存 cuSOLVER バインダで一括解法（戻り値 shape: (nv, nx-2)）
        solution = gtsv_module.gtsv_strided(
                ws.dl.contiguous(),
                ws.dd.contiguous(),
                ws.du.contiguous(),
                ws.B.contiguous()
            )

        # 内部セルのみ書き戻し。境界は前状態を維持
        ws.fn_tmp.copy_(ws.fz)
        ws.fn_tmp[1:-1, :].copy_(solution.T)

        # 正規化誤差
        df  = torch.abs(ws.fn_tmp - ws.fz)
        ref = torch.maximum(torch.abs(ws.fn_tmp), torch.abs(ws.fz))
        den = abs_tol + picard_tol * ref

        residual = torch.max(df / den)
        residual_val = float((torch.max(df) / torch.max(ref)).item())
        std_residual_val = float(residual.item())

        latest = ws.fn_tmp

        if residual <= 1.0:
            break

        # 次反復へ
        ws.fz, ws.fn_tmp = ws.fn_tmp, ws.fz
    
    state.f_tmp.copy_(latest)
    state.f_tmp[0, :].copy_(state.f[0, :])
    state.f_tmp[-1, :].copy_(state.f[-1, :])

    # NaN/Infチェック
    if num_steps % 100 == 0:
        logger.debug(f"NaN/Inf check executed at step: {num_steps}")
        if not torch.isfinite(state.f_tmp).all():
            raise ValueError("NaN/Inf detected in f_tmp")

    # swap
    state.f, state.f_tmp = state.f_tmp, state.f

    # benchlog
    benchlog = {
        "picard_iter": z + 1,
        "picard_residual": residual_val,
        "std_picard_residual": std_residual_val,
    }

    return state, benchlog

def build_stepper(cfg: Config, state: State1D1V) -> Stepper:
    # CFL条件チェック
    #bgk1d_check_CFL(cfg)

    # JITコンパイル
    cuda_module = load_implicit_fused()
    gtsv_module = load_gtsv()

    # implicit 専用ワークスペース確保
    nx, nv = state.f.shape
    ws = allocate_implicit_workspace(nx, nv, state.f.device, state.f.dtype)

    # モデルパスがあればロード
    if cfg.model_cfg.scheme_params.moments_cnn_modelpath is not None:
        model = load_moments_cnn_model(
            cfg.model_cfg.scheme_params.moments_cnn_modelpath,
            device=state.f.device
        )
    else:
        model = None

    # 初期条件設定
    set_initial_condition(state, cfg)
    def _stepper(num_steps: int) -> None:
        _, benchlog = step(state, cfg, ws, cuda_module, gtsv_module, num_steps, model)
        _stepper.benchlog = benchlog  # bench-logを属性として載せる

    _stepper.benchlog = None  # 初期値
    _stepper.ws = ws 
    return _stepper
