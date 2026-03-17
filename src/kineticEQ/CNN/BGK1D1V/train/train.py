# kineticEQ/CNN/BGK1D1V/train/train.py
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader

from ..util.models import MomentCNN1D
from ..gen_traindata_v1.dataloader_npz import BGK1D1VNPZDeltaDataset
from ..gen_traindata_v2.dataloader_pt import BGK1D1VShardDeltaDataset

from ..evaluation.train_eval import TrainEvaluator, build_train_eval_spec_from_args


# ---------------- losses ----------------
from ..util.losses import (
    compute_stdW_residuals,
    build_shock_mask_from_x,
    std_w_loss_from_residuals_shock,
)

# ---------------- utils ----------------
def save_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


_RETRAIN_RUNTIME_OVERRIDE_KEYS = {
    "config",
    "retrain",
    "save_dir",
    "device",
    "workers",
    "prefetch_factor",
    "cache_shards",
    "log_interval",
    "epochs",
    "seed",
    "warm_eval",
    "warm_eval_start",
    "warm_eval_tau",
    "warm_eval_dt",
    "warm_eval_T_total",
    "warm_eval_picard_iter",
    "warm_eval_picard_tol",
    "warm_eval_abs_tol",
    "warm_eval_nx",
    "warm_eval_nv",
    "warm_eval_debug_steps",
    "warm_eval_n_floor",
    "warm_eval_T_floor",
}


def _sha256_file(path: str | Path | None) -> str | None:
    if not path:
        return None
    p = Path(path).expanduser().resolve()
    if not p.exists() or not p.is_file():
        return None

    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _jsonable_value(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, tuple):
        return [_jsonable_value(v) for v in x]
    if isinstance(x, list):
        return [_jsonable_value(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _jsonable_value(v) for k, v in x.items()}
    return x


def _normalized_compare_value(key: str, value: Any) -> Any:
    if key == "manifest" and value:
        try:
            return str(Path(value).expanduser().resolve())
        except Exception:
            return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return tuple(_normalized_compare_value(key, v) for v in value)
    if isinstance(value, list):
        return tuple(_normalized_compare_value(key, v) for v in value)
    return value


def _model_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "in_ch": 5,
        "hidden": int(args.hidden),
        "out_ch": 3,
        "kernel": int(args.kernel),
        "n_blocks": int(args.n_blocks),
        "gn_groups": int(args.gn_groups),
        "bottleneck": float(args.bottleneck),
        "dilation_cycle": tuple(int(v) for v in args.dilation_cycle),
        "use_gate_head": bool(args.use_gate_head),
        "gate_bias_init": float(args.gate_bias_init),
        "gate_scale": float(args.gate_scale),
        "gate_per_channel": bool(args.gate_per_channel),
    }


def _build_checkpoint_meta(
    args: argparse.Namespace,
    *,
    model_kwargs: dict[str, Any],
    retrain_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta = {
        "checkpoint_format_version": 2,
        "saved_at_unix": float(time.time()),
        "manifest_path": str(Path(args.manifest).expanduser().resolve()) if args.manifest else None,
        "manifest_sha256": _sha256_file(args.manifest),
        "delta_type": str(args.delta_type),
        "model_kwargs": _jsonable_value(model_kwargs),
    }
    if retrain_info:
        meta["retrain"] = _jsonable_value(retrain_info)
    return meta


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"rng_state_cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["rng_state_cuda"] = [s.cpu() for s in torch.cuda.get_rng_state_all()]
    return state


def _restore_rng_state(ckpt: dict[str, Any], device: torch.device) -> None:
    cpu_state = ckpt.get("rng_state_cpu")
    if isinstance(cpu_state, torch.Tensor):
        torch.set_rng_state(cpu_state.cpu())

    cuda_states = ckpt.get("rng_state_cuda")
    if device.type != "cuda" or not isinstance(cuda_states, list):
        return

    try:
        if len(cuda_states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all([s.cpu() for s in cuda_states])
    except Exception:
        pass


def _move_optimizer_state_to_device(opt: torch.optim.Optimizer, device: torch.device) -> None:
    for state in opt.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device=device, non_blocking=True)


def _compare_manifest_identity(current_manifest: str | None, saved_manifest: str | None) -> tuple[bool, str | None]:
    current_norm = _normalized_compare_value("manifest", current_manifest)
    saved_norm = _normalized_compare_value("manifest", saved_manifest)
    if current_norm == saved_norm:
        return True, None

    current_hash = _sha256_file(current_manifest)
    saved_hash = _sha256_file(saved_manifest)
    if current_hash is not None and saved_hash is not None and current_hash == saved_hash:
        return True, None

    detail = f"manifest: current={current_norm!r}, saved={saved_norm!r}"
    if current_hash is not None or saved_hash is not None:
        detail += f" (sha256 current={current_hash}, saved={saved_hash})"
    return False, detail


def _validate_retrain_args(
    args: argparse.Namespace,
    saved_args: dict[str, Any],
    *,
    ckpt_manifest: str | None,
    ckpt_model_kwargs: dict[str, Any] | None = None,
) -> None:
    mismatches: list[str] = []

    manifest_ok, manifest_detail = _compare_manifest_identity(
        getattr(args, "manifest", None),
        saved_args.get("manifest", ckpt_manifest),
    )
    if not manifest_ok and manifest_detail is not None:
        mismatches.append(manifest_detail)

    current_dict = vars(args)
    keys = (set(saved_args.keys()) & set(current_dict.keys())) - _RETRAIN_RUNTIME_OVERRIDE_KEYS - {"manifest"}
    for key in sorted(keys):
        current_v = _normalized_compare_value(key, current_dict[key])
        saved_v = _normalized_compare_value(key, saved_args[key])
        if current_v != saved_v:
            mismatches.append(f"{key}: current={current_v!r}, saved={saved_v!r}")

    if isinstance(ckpt_model_kwargs, dict):
        current_model_kwargs = _model_kwargs_from_args(args)
        for key in sorted(set(ckpt_model_kwargs.keys()) & set(current_model_kwargs.keys())):
            current_v = _normalized_compare_value(key, current_model_kwargs[key])
            saved_v = _normalized_compare_value(key, ckpt_model_kwargs[key])
            if current_v != saved_v:
                mismatches.append(f"model_kwargs.{key}: current={current_v!r}, saved={saved_v!r}")

    if mismatches:
        detail = "\n  - ".join(mismatches)
        raise ValueError(
            "retrain checkpoint is incompatible with current training arguments:\n"
            f"  - {detail}\n"
            "Only runtime/output settings may differ for --retrain."
        )


def _load_retrain_checkpoint(
    ckpt_path: str | Path,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Any], int, float, float, dict[str, Any]]:
    ckpt_file = Path(ckpt_path).expanduser().resolve()
    if not ckpt_file.exists():
        raise FileNotFoundError(f"retrain checkpoint not found: {ckpt_file}")

    save_dir = Path(args.save_dir).expanduser().resolve()
    if save_dir == ckpt_file.parent.resolve():
        raise ValueError("--retrain requires a new --save_dir; source and destination directories match")

    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model" not in ckpt or "opt" not in ckpt:
        raise ValueError(f"unsupported retrain checkpoint format: {ckpt_file}")

    saved_args = ckpt.get("args")
    if not isinstance(saved_args, dict):
        raise ValueError(f"checkpoint has no saved args metadata: {ckpt_file}")

    ckpt_model_kwargs = ckpt.get("model_kwargs")
    if not isinstance(ckpt_model_kwargs, dict):
        meta = ckpt.get("meta")
        if isinstance(meta, dict) and isinstance(meta.get("model_kwargs"), dict):
            ckpt_model_kwargs = meta.get("model_kwargs")

    _validate_retrain_args(
        args,
        saved_args,
        ckpt_manifest=ckpt.get("manifest"),
        ckpt_model_kwargs=ckpt_model_kwargs,
    )

    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_val = float(ckpt.get("best_val", float("inf")))
    best_speed = float(ckpt.get("best_speed", ckpt.get("best_speed_picard_sum", 0.0)) or 0.0)
    retrain_info = {
        "source_checkpoint": str(ckpt_file),
        "source_epoch": int(ckpt.get("epoch", 0)),
        "source_save_dir": str(ckpt_file.parent.resolve()),
    }

    print(
        f"[retrain] source={ckpt_file} start_epoch={start_epoch} "
        f"additional_epochs={int(args.epochs)} save_dir={save_dir}",
        flush=True,
    )
    _restore_rng_state(ckpt, device)
    return ckpt, start_epoch, best_val, best_speed, retrain_info


def _build_checkpoint_payload(
    *,
    epoch: int,
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    scaler: torch.amp.GradScaler,
    best_val: float,
    best_speed: float,
    train_loss: float,
    val_loss: float,
    val_base_loss: float,
    val_shock_loss: float,
    args: argparse.Namespace,
    model_kwargs: dict[str, Any],
    retrain_info: dict[str, Any] | None,
) -> dict[str, Any]:
    payload = {
        "epoch": int(epoch),
        "model": model.state_dict(),
        "model_kwargs": _jsonable_value(model_kwargs),
        "opt": opt.state_dict(),
        "best_val": float(best_val),
        "best_speed": float(best_speed),
        "best_speed_picard_sum": float(best_speed),
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "val_base_loss": float(val_base_loss),
        "val_shock_loss": float(val_shock_loss),
        "manifest": args.manifest,
        "args": _jsonable_value(vars(args)),
        "meta": _build_checkpoint_meta(args, model_kwargs=model_kwargs, retrain_info=retrain_info),
        **_capture_rng_state(),
    }
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if scaler.is_enabled():
        payload["scaler"] = scaler.state_dict()
    return payload


def _load_config_defaults(parser: argparse.ArgumentParser, config_path: str) -> dict:
    path = Path(config_path).expanduser().resolve()
    obj = json.loads(path.read_text())
    if not isinstance(obj, dict):
        raise ValueError(f"config must be a JSON object: {path}")

    # Reuse the same base_args envelope as multi_train.json when present.
    cfg = obj.get("base_args", obj)
    if not isinstance(cfg, dict):
        raise ValueError(f"config.base_args must be a JSON object: {path}")

    known = {action.dest for action in parser._actions}
    defaults = {}
    unknown = []
    for key, value in cfg.items():
        if str(key).startswith("_"):
            continue
        if key not in known:
            unknown.append(str(key))
            continue
        defaults[str(key)] = value

    if unknown:
        unknown_s = ", ".join(sorted(unknown))
        raise ValueError(f"unknown config keys in {path}: {unknown_s}")

    defaults["config"] = str(path)
    return defaults


def _read_manifest_format(manifest: str) -> str | None:
    try:
        obj = json.loads(Path(manifest).read_text())
    except Exception:
        return None
    fmt = obj.get("format", None)
    return str(fmt) if fmt is not None else None


def make_loader(
    manifest: str,
    split: str,
    batch: int,
    workers: int,
    pin: bool,
    prefetch_factor: int,
    target: str,
    shuffle: bool,
    *,
    split_key: str,
    cache_shards: int,
):
    manifest_format = _read_manifest_format(manifest)

    if manifest_format == "kineticEQ_BGK1D1V_pt_v2":
        ds = BGK1D1VShardDeltaDataset(
            dataset_manifest_path=manifest,
            split=split,
            split_key=split_key,
            target=target,
            cache_shards=int(cache_shards),
        )
    else:
        ds = BGK1D1VNPZDeltaDataset(
            manifest_path=manifest,
            split=split,
            mmap=True,
            cache_npz=True,
            target=target,
        )

    dl = DataLoader(
        ds,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin,
        persistent_workers=(workers > 0),
        drop_last=(split == "train"),
        prefetch_factor=prefetch_factor if workers > 0 else None,
    )
    return ds, dl

# ---------------- args ----------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=str,
        default=None,
        help="optional single-train JSON config; explicit CLI args override file values",
    )
    ap.add_argument(
        "--retrain",
        type=str,
        default=None,
        help="resume training from a saved last.pt checkpoint into a new save_dir",
    )
    ap.add_argument("--manifest", type=str, default=None)
    ap.add_argument("--split_key", type=str, default="split_iid", help="split key for v2 manifest")
    ap.add_argument("--cache_shards", type=int, default=2, help="LRU shard cache size for v2 loader")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--prefetch_factor", type=int, default=2)
    ap.add_argument("--amp", action="store_true")

    ap.add_argument("--save_dir", type=str, default="cnn_runs/bgk1d1v_stdW")
    ap.add_argument("--seed", type=int, default=0)

    # learning target
    ap.add_argument("--delta_type", type=str, default="dnu", choices=["dnu", "dw"])
    ap.add_argument(
        "--conv_type",
        type=str,
        default="f",
        choices=["f", "w"],
        help="Picard convergence metric for warm_eval: f (distribution) or w (moments)",
    )

    # shuffle
    ap.add_argument("--no_shuffle", action="store_true")

    # loss knobs
    # base loss knobs
    ap.add_argument("--loss_kind", type=str, default="smoothl1", choices=["smoothl1", "mse", "l1","softmax"])
    ap.add_argument("--loss_softmax_beta", type=float, default=20.0)
    ap.add_argument("--loss_eps", type=float, default=1e-12)
    ap.add_argument("--nb", type=int, default=10)
    ap.add_argument("--n_floor", type=float, default=1e-8)
    ap.add_argument("--T_floor", type=float, default=1e-8)

    # shock-mask loss knobs
    ap.add_argument("--shock_loss_kind", type=str, default="softmax", choices=["mse","softmax"])
    ap.add_argument("--shock_loss_softmax_beta", type=float, default=20.0)
    ap.add_argument("--shock_loss_eps", type=float, default=1e-12)
    ap.add_argument("--shock_ratio", type=float, default=0.8)
    ap.add_argument("--shock_q", type=float, default=0.90)  # 0.90 -> top10% shock

    # model architecture
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--n_blocks", type=int, default=6)
    ap.add_argument("--kernel", type=int, default=5)
    ap.add_argument("--gn_groups", type=int, default=32)
    ap.add_argument("--bottleneck", type=float, default=0.5)
    ap.add_argument("--dilation_cycle", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--use_gate_head", type=int, default=1, help="0=off, 1=on")
    ap.add_argument("--gate_bias_init", type=float, default=-4.0)
    ap.add_argument("--gate_scale", type=float, default=1.0)
    ap.add_argument("--gate_per_channel", type=int, default=0, help="0=off, 1=on")

    # optimization knobs
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--log_interval", type=int, default=50)

    # optional scheduler
    ap.add_argument("--sched_plateau", action="store_true")
    ap.add_argument("--sched_patience", type=int, default=3)
    ap.add_argument("--sched_factor", type=float, default=0.5)
    ap.add_argument("--sched_min_lr", type=float, default=1e-6)

    # ---- epoch-end warmstart eval (print only; uses LAST model) ----
    ap.add_argument("--warm_eval", action="store_true", help="run warmstart debug at each epoch end (print only)")
    ap.add_argument(
        "--warm_eval_start",
        type=int,
        default=0,
        help="skip warm_eval through this epoch; e.g. 10 means first eval runs at epoch 11",
    )
    ap.add_argument("--warm_eval_tau", type=float, default=5e-7)
    ap.add_argument("--warm_eval_dt", type=float, default=5e-5)
    ap.add_argument("--warm_eval_T_total", type=float, default=0.05)
    ap.add_argument("--warm_eval_picard_iter", type=int, default=1000)
    ap.add_argument("--warm_eval_picard_tol", type=float, default=1e-3)
    ap.add_argument("--warm_eval_abs_tol", type=float, default=1e-13)
    ap.add_argument("--warm_eval_nx", type=int, default=512)
    ap.add_argument("--warm_eval_nv", type=int, default=256)
    ap.add_argument("--warm_eval_debug_steps", type=int, default=0, help="0 disables per-step debug_log collection")
    ap.add_argument("--warm_eval_n_floor", type=float, default=1e-12)
    ap.add_argument("--warm_eval_T_floor", type=float, default=1e-12)

    # Anderson Acceleration (implicit Picard solver)
    ap.add_argument("--aa_enable", action="store_true", help="enable Anderson Acceleration")
    ap.add_argument("--aa_m", type=int, default=6, help="AA history depth")
    ap.add_argument("--aa_beta", type=float, default=1.0, help="AA damping parameter")
    ap.add_argument("--aa_stride", type=int, default=1, help="apply AA every k Picard iterations")
    ap.add_argument("--aa_start_iter", type=int, default=2, help="first Picard iter to allow AA")
    ap.add_argument("--aa_reg", type=float, default=1e-10, help="AA Gram regularization")
    ap.add_argument("--aa_alpha_max", type=float, default=50.0, help="AA alpha clamp")
    return ap


def parse_args():
    parser = build_parser()
    bootstrap_args, _ = parser.parse_known_args()
    if bootstrap_args.config:
        parser.set_defaults(**_load_config_defaults(parser, bootstrap_args.config))
    args = parser.parse_args()
    if not args.manifest:
        parser.error("--manifest is required unless provided by --config")
    return args


# ---------------- main ----------------
def main():
    args = parse_args()

    if args.config:
        print(f"config file: {args.config}", flush=True)
    print(f"loss function: {args.loss_kind}", flush=True)
    print(f"shock_ratio: {args.shock_ratio}", flush=True)
    print(f"shock_q: {args.shock_q}", flush=True)
    print(f"delta_type: {args.delta_type}", flush=True)

    # Picard-sum speedup against the cached baseline warm-eval.
    best_speed = 0.0

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    pin = (device.type == "cuda")
    use_amp = bool(args.amp) and (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if device.type == "cuda":
        print(f"Device_name: {torch.cuda.get_device_name(0)}", flush=True)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if args.retrain and any((save_dir / name).exists() for name in ("log.jsonl", "last.pt", "best.pt", "best_speed.pt")):
        raise FileExistsError(f"--retrain requires a fresh save_dir; found existing artifacts in {save_dir}")

    model_kwargs = _model_kwargs_from_args(args)
    retrain_info: dict[str, Any] | None = None

    model = MomentCNN1D(
        **model_kwargs,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr))

    scheduler = None
    if bool(args.sched_plateau):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=float(args.sched_factor),
            patience=int(args.sched_patience),
            min_lr=float(args.sched_min_lr),
        )

    train_ds, train_dl = make_loader(
        args.manifest,
        "train",
        args.batch,
        args.workers,
        pin=pin,
        prefetch_factor=args.prefetch_factor,
        target=args.delta_type,
        shuffle=(not args.no_shuffle),
        split_key=str(args.split_key),
        cache_shards=int(args.cache_shards),
    )
    val_ds, val_dl = make_loader(
        args.manifest,
        "val",
        args.batch,
        args.workers,
        pin=pin,
        prefetch_factor=args.prefetch_factor,
        target=args.delta_type,
        shuffle=False,
        split_key=str(args.split_key),
        cache_shards=int(args.cache_shards),
    )

    best_val = float("inf")
    start_epoch = 1
    log_path = save_dir / "log.jsonl"
    train_evaluator = None
    warm_eval_start = max(int(args.warm_eval_start), 0)
    if args.retrain:
        ckpt, start_epoch, best_val, best_speed, retrain_info = _load_retrain_checkpoint(
            args.retrain,
            args,
            device,
        )
        model.load_state_dict(ckpt["model"], strict=True)
        opt.load_state_dict(ckpt["opt"])
        _move_optimizer_state_to_device(opt, device)

        sched_state = ckpt.get("scheduler")
        if scheduler is not None and isinstance(sched_state, dict):
            scheduler.load_state_dict(sched_state)
        elif scheduler is not None and sched_state is None:
            print("[retrain] checkpoint has no scheduler state; scheduler restarted from current args", flush=True)

        scaler_state = ckpt.get("scaler")
        if scaler.is_enabled() and isinstance(scaler_state, dict):
            scaler.load_state_dict(scaler_state)
        elif scaler.is_enabled() and scaler_state is None:
            print("[retrain] checkpoint has no AMP scaler state; scaler restarted", flush=True)

    config_obj = dict(vars(args))
    if retrain_info is not None:
        config_obj["_retrain"] = {
            **retrain_info,
            "resume_start_epoch": int(start_epoch),
            "additional_epochs": int(args.epochs),
        }
    save_json(save_dir / "config.json", _jsonable_value(config_obj))

    if bool(args.warm_eval):
        eval_cache_dir = save_dir / "eval_cache"
        eval_cache_dir.mkdir(parents=True, exist_ok=True)
        eval_spec = build_train_eval_spec_from_args(
            args,
            cache_dir=str(eval_cache_dir),
        )
        train_evaluator = TrainEvaluator(
            eval_spec=eval_spec,
            device=str(device),
            cache_dir=str(eval_cache_dir),
            verbose=True,
        )
        if warm_eval_start > 0:
            print(
                f"[warm-eval] deferred until epoch {warm_eval_start + 1:d} "
                f"(skip epochs 1-{warm_eval_start:d})",
                flush=True,
            )

    end_epoch = start_epoch + int(args.epochs)
    for ep in range(start_epoch, end_epoch):
        # ---------------- train ----------------
        model.train()
        t0 = time.time()

        tr_loss_sum = 0.0
        tr_n = 0
        tr_base_loss_sum = 0.0
        tr_shock_wsum = 0.0          # sum(shock_loss * shock_count)
        tr_shock_wden = 0.0  

        # channel-wise stats on residuals
        tr_rn_abs_sum = 0.0
        tr_ru_abs_sum = 0.0
        tr_rT_abs_sum = 0.0
        tr_rn_abs_max = 0.0
        tr_ru_abs_max = 0.0
        tr_rT_abs_max = 0.0
        tr_count = 0.0  # total valid elements per channel

        pbar = tqdm(train_dl, desc=f"train ep{ep:03d}", dynamic_ncols=True, leave=False)
        for it, (x, y) in enumerate(pbar, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(x)

                rn, ru, rT, valid = compute_stdW_residuals(
                    pred, y, x,
                    nb=int(args.nb),
                    n_floor=float(args.n_floor),
                    T_floor=float(args.T_floor),
                    eps=float(args.loss_eps),
                    delta_type=args.delta_type
                )

                shock_mask = build_shock_mask_from_x(
                    x, nb=int(args.nb), shock_q=float(args.shock_q)
                )

                loss, base_loss, shock_loss = std_w_loss_from_residuals_shock(
                    rn, ru, rT, valid, shock_mask,
                    kind=str(args.loss_kind),
                    shock_ratio=float(args.shock_ratio),
                    softmax_beta=float(args.loss_softmax_beta),
                    shock_loss_kind=str(args.shock_loss_kind),
                    shock_loss_softmax_beta=float(args.shock_loss_softmax_beta),
                    shock_loss_eps=float(args.shock_loss_eps),
                )

            scaler.scale(loss).backward()
            if float(args.grad_clip) > 0.0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
            scaler.step(opt)
            scaler.update()

            bs = x.size(0)
            
            # total/base
            tr_loss_sum += float(loss.item()) * bs
            tr_base_loss_sum += float(base_loss.item()) * bs

            # shock
            shock_count = float(shock_mask.sum().item()) * 3.0
            tr_shock_wsum += float(shock_loss.item()) * shock_count
            tr_shock_wden += shock_count

            tr_n += bs

            B = rn.shape[0]
            batch_count = float(valid.item()) * float(B)
            tr_count += batch_count

            arn = rn.abs()
            aru = ru.abs()
            arT = rT.abs()
            tr_rn_abs_sum += float(arn.sum().item())
            tr_ru_abs_sum += float(aru.sum().item())
            tr_rT_abs_sum += float(arT.sum().item())
            tr_rn_abs_max = max(tr_rn_abs_max, float(arn.max().item()))
            tr_ru_abs_max = max(tr_ru_abs_max, float(aru.max().item()))
            tr_rT_abs_max = max(tr_rT_abs_max, float(arT.max().item()))

            if (it == 1) or (it % int(args.log_interval) == 0):
                lr = opt.param_groups[0]["lr"]
                rn_mae = tr_rn_abs_sum / max(tr_count, 1.0)
                ru_mae = tr_ru_abs_sum / max(tr_count, 1.0)
                rT_mae = tr_rT_abs_sum / max(tr_count, 1.0)
                pbar.set_postfix({
                    "loss": f"{(tr_loss_sum/max(tr_n,1)):.3e}",
                    "shock": f"{( (tr_shock_wsum / max(tr_shock_wden,1.0)) * float(args.shock_ratio) ):.3e}",
                    "base": f"{float(tr_base_loss_sum/max(tr_n,1)):.3e}",
                    "lr": f"{lr:.1e}",
                    "|rn|": f"{rn_mae:.2e}",
                    "|ru|": f"{ru_mae:.2e}",
                    "|rT|": f"{rT_mae:.2e}",
                    "rn_max": f"{tr_rn_abs_max:.2e}",
                    "ru_max": f"{tr_ru_abs_max:.2e}",
                    "rT_max": f"{tr_rT_abs_max:.2e}",
                })

        train_loss = tr_loss_sum / max(tr_n, 1)
        train_shock_loss_raw = tr_shock_wsum / max(tr_shock_wden, 1.0)
        train_shock_loss = train_shock_loss_raw * float(args.shock_ratio)
        train_time = time.time() - t0
        tr_rn_mae = tr_rn_abs_sum / max(tr_count, 1.0)
        tr_ru_mae = tr_ru_abs_sum / max(tr_count, 1.0)
        tr_rT_mae = tr_rT_abs_sum / max(tr_count, 1.0)

        # ---------------- val ----------------
        model.eval()
        va_loss_sum = 0.0
        va_n = 0
        va_base_loss_sum = 0.0
        va_shock_wsum = 0.0
        va_shock_wden = 0.0

        va_rn_abs_sum = 0.0
        va_ru_abs_sum = 0.0
        va_rT_abs_sum = 0.0
        va_rn_abs_max = 0.0
        va_ru_abs_max = 0.0
        va_rT_abs_max = 0.0
        va_count = 0.0

        with torch.no_grad():
            vbar = tqdm(val_dl, desc=f"val   ep{ep:03d}", dynamic_ncols=True, leave=False)
            for x, y in vbar:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred = model(x)
                    rn, ru, rT, valid = compute_stdW_residuals(
                        pred, y, x,
                        nb=int(args.nb),
                        n_floor=float(args.n_floor),
                        T_floor=float(args.T_floor),
                        eps=float(args.loss_eps),
                        delta_type=args.delta_type
                    )
                    shock_mask = build_shock_mask_from_x(
                        x, nb=int(args.nb), shock_q=float(args.shock_q)
                    )
                    loss, val_base_loss, val_shock_loss = std_w_loss_from_residuals_shock(
                        rn, ru, rT, valid, shock_mask,
                        kind=str(args.loss_kind),
                        shock_ratio=float(args.shock_ratio),
                        softmax_beta=float(args.loss_softmax_beta),
                        shock_loss_kind=str(args.shock_loss_kind),
                        shock_loss_softmax_beta=float(args.shock_loss_softmax_beta),
                        shock_loss_eps=float(args.shock_loss_eps),
                    )

                bs = x.size(0)
                va_loss_sum += float(loss.item()) * bs
                va_base_loss_sum += float(val_base_loss.item()) * bs

                shock_count = float(shock_mask.sum().item()) * 3.0
                va_shock_wsum += float(val_shock_loss.item()) * shock_count
                va_shock_wden += shock_count

                va_n += bs

                B = rn.shape[0]
                batch_count = float(valid.item()) * float(B)
                va_count += batch_count

                arn = rn.abs()
                aru = ru.abs()
                arT = rT.abs()
                va_rn_abs_sum += float(arn.sum().item())
                va_ru_abs_sum += float(aru.sum().item())
                va_rT_abs_sum += float(arT.sum().item())
                va_rn_abs_max = max(va_rn_abs_max, float(arn.max().item()))
                va_ru_abs_max = max(va_ru_abs_max, float(aru.max().item()))
                va_rT_abs_max = max(va_rT_abs_max, float(arT.max().item()))

        val_loss = va_loss_sum / max(va_n, 1)
        val_base_loss = va_base_loss_sum / max(va_n, 1)
        val_shock_loss_raw = va_shock_wsum / max(va_shock_wden, 1.0)
        val_shock_loss = val_shock_loss_raw * float(args.shock_ratio)
        va_rn_mae = va_rn_abs_sum / max(va_count, 1.0)
        va_ru_mae = va_ru_abs_sum / max(va_count, 1.0)
        va_rT_mae = va_rT_abs_sum / max(va_count, 1.0)

        if scheduler is not None:
            scheduler.step(val_loss)

        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss

        print(
            f"[epoch {ep:03d}] "
            f"train={train_loss:.6e}, val={val_loss:.6e}, val_base={val_base_loss:.6e}, val_shock={val_shock_loss:.6e} "
            f"time={train_time:.1f}s best={best_val:.6e} "
            f"|rn|={va_rn_mae:.2e} |ru|={va_ru_mae:.2e} |rT|={va_rT_mae:.2e} "
            f"ru_max={va_ru_abs_max:.2e}",
            flush=True,
        )

        ckpt = _build_checkpoint_payload(
            epoch=ep,
            model=model,
            opt=opt,
            scheduler=scheduler,
            scaler=scaler,
            best_val=best_val,
            best_speed=best_speed,
            train_loss=train_loss,
            val_loss=val_loss,
            val_base_loss=val_base_loss,
            val_shock_loss=val_shock_loss,
            args=args,
            model_kwargs=model_kwargs,
            retrain_info=retrain_info,
        )
        torch.save(ckpt, save_dir / "last.pt")
        if is_best:
            torch.save(ckpt, save_dir / "best.pt")

        with log_path.open("a") as f:
            f.write(json.dumps({
                "epoch": ep,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_base_loss": float(val_base_loss),
                "val_shock_loss": float(val_shock_loss),
                "train_shock_loss_raw": float(train_shock_loss_raw),
                "train_shock_loss": float(train_shock_loss),          
                "val_shock_loss_raw": float(val_shock_loss_raw),
                "val_shock_loss": float(val_shock_loss),

                "best_val": best_val,
                "lr": opt.param_groups[0]["lr"],
                "train_time_sec": train_time,

                # channel-wise abs residual stats (train/val)
                "train_rn_abs_mean": tr_rn_mae,
                "train_ru_abs_mean": tr_ru_mae,
                "train_rT_abs_mean": tr_rT_mae,
                "train_rn_abs_max": tr_rn_abs_max,
                "train_ru_abs_max": tr_ru_abs_max,
                "train_rT_abs_max": tr_rT_abs_max,

                "val_rn_abs_mean": va_rn_mae,
                "val_ru_abs_mean": va_ru_mae,
                "val_rT_abs_mean": va_rT_mae,
                "val_rn_abs_max": va_rn_abs_max,
                "val_ru_abs_max": va_ru_abs_max,
                "val_rT_abs_max": va_rT_abs_max,

                # config echoes
                "loss_kind": str(args.loss_kind),
                "loss_eps": float(args.loss_eps),
                "nb": int(args.nb),
                "n_floor": float(args.n_floor),
                "T_floor": float(args.T_floor),
                "shock_ratio": float(args.shock_ratio),
                "shock_q": float(args.shock_q),
                "grad_clip": float(args.grad_clip),
                "amp": bool(args.amp),
                "warm_eval_start": warm_eval_start,
            }) + "\n")

        # ---------------- epoch-end warmstart debug (LATEST model via last.pt; save best_speed.pt) ----------------
        if train_evaluator is not None and ep > warm_eval_start:
            speed_ep_best = 0.0
            speed_ep_by_alpha = {}

            try:
                model.eval()
                last_ckpt = save_dir / "last.pt"
                eval_summary = train_evaluator.evaluate_epoch(last_ckpt)
                base_sum = int(eval_summary["picard_sum_base"])
                warm_sum = int(eval_summary["picard_sum_warm"])
                speed = float(eval_summary["speedup_picard_sum"])

                speed_ep_by_alpha["engine"] = speed
                speed_ep_best = speed

                print(
                    f"[warm-eval ep{ep:03d}] "
                    f"tau={args.warm_eval_tau:.3e} "
                    f"picard_sum base={base_sum} warm={warm_sum} (x{speed:.2f}) "
                    f"t_base={float(eval_summary['walltime_base_sec']):.3f}s "
                    f"t_warm={float(eval_summary['walltime_warm_sec']):.3f}s",
                    flush=True,
                )

                with log_path.open("a") as f:
                    f.write(json.dumps({
                        "epoch": ep,
                        "warm_eval": True,
                        "picard_sum_base": base_sum,
                        "picard_sum_warm": warm_sum,
                        "speedup_picard_sum": float(speed),
                        "walltime_base_sec": float(eval_summary["walltime_base_sec"]),
                        "walltime_warm_sec": float(eval_summary["walltime_warm_sec"]),
                        "speedup_walltime": eval_summary["speedup_walltime"],
                        "best_speed_so_far": float(max(speed_ep_best, best_speed)),
                    }) + "\n")

            except Exception as e:
                print(f"[warm-eval ep{ep:03d}] FAILED: {type(e).__name__}: {e}", flush=True)

            if speed_ep_best > best_speed:
                best_speed = speed_ep_best
                torch.save(
                    _build_checkpoint_payload(
                        epoch=ep,
                        model=model,
                        opt=opt,
                        scheduler=scheduler,
                        scaler=scaler,
                        best_val=best_val,
                        best_speed=best_speed,
                        train_loss=train_loss,
                        val_loss=val_loss,
                        val_base_loss=val_base_loss,
                        val_shock_loss=val_shock_loss,
                        args=args,
                        model_kwargs=model_kwargs,
                        retrain_info=retrain_info,
                    ),
                    save_dir / "last.pt",
                )
                torch.save({
                    "epoch": ep,
                    "model": model.state_dict(),
                    "model_kwargs": _jsonable_value(model_kwargs),
                    "best_speed_picard_sum": float(best_speed),
                    "speed_by_alpha": speed_ep_by_alpha,
                    "args": _jsonable_value(vars(args)),
                    "meta": _build_checkpoint_meta(args, model_kwargs=model_kwargs, retrain_info=retrain_info),
                }, save_dir / "best_speed.pt")

    train_ds.close()
    val_ds.close()


if __name__ == "__main__":
    main()
