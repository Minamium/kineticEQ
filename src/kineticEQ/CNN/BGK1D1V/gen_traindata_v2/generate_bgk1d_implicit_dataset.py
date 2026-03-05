from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.distributed as dist

from kineticEQ import BGK1D, Config, Engine
from kineticEQ.core.schemes.BGK1D.bgk1d_utils.bgk1d_compute_moments import calculate_moments


FAMILY_NAMES = (
    "shock_basic",
    "shock_strong",
    "isobaric_contact",
    "double_hotspot_isobaric",
    "velocity_ramp",
    "thermal_velocity_mixed",
)

DEFAULT_FAMILY_WEIGHTS = (
    "shock_basic:0.25,"
    "shock_strong:0.20,"
    "isobaric_contact:0.15,"
    "double_hotspot_isobaric:0.15,"
    "velocity_ramp:0.15,"
    "thermal_velocity_mixed:0.10"
)


@dataclass(frozen=True)
class CaseSpec:
    case_id: int
    family: str
    seed: int

    nx: int
    nv: int
    Lx: float
    v_max: float

    dt: float
    tau_tilde: float
    T_total: float

    picard_iter: int
    picard_tol: float
    abs_tol: float
    conv_type: str

    initial_regions: tuple[dict, ...]

    dt_over_tau: float
    log10_dt: float
    log10_tau: float
    log10_dt_over_tau: float

    n_ratio: float
    T_ratio: float
    u_rms_over_sqrtT: float


@dataclass
class CaseRun:
    W: np.ndarray
    picard_iter: np.ndarray
    std_residual: np.ndarray
    elapsed_sec: float


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _safe_log10(x: float, floor: float = 1e-300) -> float:
    return float(math.log10(max(float(x), floor)))


def _region_stats(initial_regions: tuple[dict, ...]) -> tuple[float, float, float]:
    widths: list[float] = []
    n_vals: list[float] = []
    T_vals: list[float] = []
    u_vals: list[float] = []

    for reg in initial_regions:
        x0, x1 = reg["x_range"]
        w = max(float(x1) - float(x0), 0.0)
        widths.append(w)
        n_vals.append(float(reg["n"]))
        T_vals.append(float(reg["T"]))
        u_vals.append(float(reg["u"]))

    wsum = float(sum(widths)) if widths else 1.0
    wnorm = [w / max(wsum, 1e-12) for w in widths]

    n_min = max(min(n_vals), 1e-12)
    T_min = max(min(T_vals), 1e-12)
    n_ratio = float(max(n_vals) / n_min)
    T_ratio = float(max(T_vals) / T_min)

    u_rms = math.sqrt(sum(w * (u * u) for w, u in zip(wnorm, u_vals)))
    T_mean = sum(w * t for w, t in zip(wnorm, T_vals))
    u_rms_over_sqrtT = float(u_rms / math.sqrt(max(T_mean, 1e-12)))

    return n_ratio, T_ratio, u_rms_over_sqrtT


def _sample_centered_log_uniform(
    rng: np.random.Generator,
    *,
    center_value: float,
    center_prob: float,
    vmin: float,
    vmax: float,
) -> float:
    p = float(min(max(center_prob, 0.0), 1.0))
    if rng.random() < p:
        return float(center_value)
    lo = _safe_log10(vmin)
    hi = _safe_log10(vmax)
    return float(10.0 ** rng.uniform(lo, hi))


def _build_shock_basic(rng: np.random.Generator) -> tuple[dict, ...]:
    x0 = float(rng.uniform(0.4, 0.6))

    n_ratio = float(10.0 ** rng.uniform(math.log10(1.2), math.log10(5.0)))
    T_ratio = float(10.0 ** rng.uniform(math.log10(1.2), math.log10(5.0)))

    if rng.random() < 0.5:
        nL, nR = 1.0, 1.0 / n_ratio
        TL, TR = 1.0, 1.0 / T_ratio
    else:
        nL, nR = 1.0, n_ratio
        TL, TR = 1.0, 1.0 / T_ratio

    return (
        {"x_range": (0.0, x0), "n": float(nL), "u": 0.0, "T": float(TL)},
        {"x_range": (x0, 1.0), "n": float(nR), "u": 0.0, "T": float(TR)},
    )


def _build_shock_strong(rng: np.random.Generator) -> tuple[dict, ...]:
    x0 = float(rng.uniform(0.45, 0.55))

    n_ratio = float(10.0 ** rng.uniform(math.log10(2.0), math.log10(20.0)))
    T_ratio = float(10.0 ** rng.uniform(math.log10(2.0), math.log10(20.0)))

    if rng.random() < 0.5:
        nL, nR = 1.0, n_ratio
        TL, TR = 1.0, 1.0 / T_ratio
    else:
        nL, nR = n_ratio, 1.0
        TL, TR = 1.0 / T_ratio, 1.0

    return (
        {"x_range": (0.0, x0), "n": float(nL), "u": 0.0, "T": float(TL)},
        {"x_range": (x0, 1.0), "n": float(nR), "u": 0.0, "T": float(TR)},
    )


def _build_isobaric_contact(rng: np.random.Generator) -> tuple[dict, ...]:
    x0 = float(rng.uniform(0.35, 0.65))
    p0 = float(10.0 ** rng.uniform(math.log10(0.2), math.log10(1.5)))

    T_low = float(10.0 ** rng.uniform(math.log10(0.2), math.log10(0.8)))
    Tr = float(10.0 ** rng.uniform(math.log10(1.2), math.log10(8.0)))
    T_high = float(min(T_low * Tr, 2.0))

    if rng.random() < 0.5:
        TL, TR = T_high, T_low
    else:
        TL, TR = T_low, T_high

    nL = p0 / TL
    nR = p0 / TR

    return (
        {"x_range": (0.0, x0), "n": float(nL), "u": 0.0, "T": float(TL)},
        {"x_range": (x0, 1.0), "n": float(nR), "u": 0.0, "T": float(TR)},
    )


def _build_double_hotspot_isobaric(rng: np.random.Generator) -> tuple[dict, ...]:
    p0 = float(10.0 ** rng.uniform(math.log10(0.15), math.log10(1.2)))
    Tc = float(10.0 ** rng.uniform(math.log10(0.15), math.log10(0.5)))
    Th = float(10.0 ** rng.uniform(math.log10(0.8), math.log10(1.6)))

    w_hot = float(rng.uniform(0.06, 0.16))
    left_c = float(rng.uniform(0.15, 0.35))
    right_c = float(rng.uniform(0.65, 0.85))

    a = max(left_c - 0.5 * w_hot, 0.0)
    b = min(left_c + 0.5 * w_hot, 1.0)
    c = max(right_c - 0.5 * w_hot, 0.0)
    d = min(right_c + 0.5 * w_hot, 1.0)

    if not (0.0 < a < b < c < d < 1.0):
        a, b, c, d = 0.15, 0.25, 0.75, 0.85

    return (
        {"x_range": (0.0, a), "n": float(p0 / Tc), "u": 0.0, "T": float(Tc)},
        {"x_range": (a, b), "n": float(p0 / Th), "u": 0.0, "T": float(Th)},
        {"x_range": (b, c), "n": float(p0 / Tc), "u": 0.0, "T": float(Tc)},
        {"x_range": (c, d), "n": float(p0 / Th), "u": 0.0, "T": float(Th)},
        {"x_range": (d, 1.0), "n": float(p0 / Tc), "u": 0.0, "T": float(Tc)},
    )


def _build_velocity_ramp(rng: np.random.Generator, segments: int) -> tuple[dict, ...]:
    N = max(int(segments), 4)
    U = float(rng.uniform(0.2, 0.8))
    n0 = float(rng.uniform(0.8, 1.2))
    T0 = float(rng.uniform(0.8, 1.2))

    regions: list[dict] = []
    for k in range(N):
        x0 = k / N
        x1 = (k + 1) / N
        xc = 0.5 * (x0 + x1)
        u = -U + 2.0 * U * xc
        regions.append({"x_range": (float(x0), float(x1)), "n": n0, "u": float(u), "T": T0})
    return tuple(regions)


def _build_thermal_velocity_mixed(rng: np.random.Generator, edge_width: float) -> tuple[dict, ...]:
    delta = float(min(max(edge_width, 1e-3), 0.2))
    U = float(rng.uniform(0.2, 0.8))

    TL = float(rng.uniform(0.8, 1.4))
    TR = float(rng.uniform(0.2, 0.8))
    TM = 0.5 * (TL + TR)

    p0 = float(10.0 ** rng.uniform(math.log10(0.3), math.log10(1.2)))
    nL = p0 / TL
    nM = p0 / TM
    nR = p0 / TR

    return (
        {"x_range": (0.0, delta), "n": float(nL), "u": float(-U), "T": float(TL)},
        {"x_range": (delta, 1.0 - delta), "n": float(nM), "u": 0.0, "T": float(TM)},
        {"x_range": (1.0 - delta, 1.0), "n": float(nR), "u": float(+U), "T": float(TR)},
    )


def _build_initial_regions(
    family: str,
    rng: np.random.Generator,
    *,
    velocity_segments: int,
    edge_width: float,
) -> tuple[dict, ...]:
    if family == "shock_basic":
        return _build_shock_basic(rng)
    if family == "shock_strong":
        return _build_shock_strong(rng)
    if family == "isobaric_contact":
        return _build_isobaric_contact(rng)
    if family == "double_hotspot_isobaric":
        return _build_double_hotspot_isobaric(rng)
    if family == "velocity_ramp":
        return _build_velocity_ramp(rng, segments=velocity_segments)
    if family == "thermal_velocity_mixed":
        return _build_thermal_velocity_mixed(rng, edge_width=edge_width)
    raise ValueError(f"unknown family: {family}")


def _parse_family_weights(spec: str) -> tuple[list[str], np.ndarray]:
    pairs = [s.strip() for s in str(spec).split(",") if s.strip()]
    if not pairs:
        raise ValueError("family_weights is empty")

    names: list[str] = []
    weights: list[float] = []
    for p in pairs:
        if ":" not in p:
            raise ValueError(f"invalid family weight entry: {p!r}")
        name, w = p.split(":", 1)
        name = name.strip()
        if name not in FAMILY_NAMES:
            raise ValueError(f"unknown family in family_weights: {name!r}")
        names.append(name)
        weights.append(float(w))

    w_arr = np.asarray(weights, dtype=np.float64)
    if np.any(w_arr < 0.0):
        raise ValueError("family weights must be non-negative")
    s = float(w_arr.sum())
    if s <= 0.0:
        raise ValueError("sum of family weights must be > 0")
    w_arr /= s
    return names, w_arr


def _sample_case_spec(case_id: int, family: str, args: argparse.Namespace) -> CaseSpec:
    case_seed = int(args.seed) * 1_000_003 + int(case_id)
    rng = np.random.default_rng(case_seed)

    dt = _sample_centered_log_uniform(
        rng,
        center_value=float(args.dt_center),
        center_prob=float(args.dt_center_prob),
        vmin=float(args.dt_min),
        vmax=float(args.dt_max),
    )
    tau = _sample_centered_log_uniform(
        rng,
        center_value=float(args.tau_center),
        center_prob=float(args.tau_center_prob),
        vmin=float(args.tau_min),
        vmax=float(args.tau_max),
    )

    initial_regions = _build_initial_regions(
        family,
        rng,
        velocity_segments=int(args.velocity_segments),
        edge_width=float(args.edge_width),
    )
    n_ratio, T_ratio, u_rms_over_sqrtT = _region_stats(initial_regions)

    dt_over_tau = float(dt / max(tau, 1e-30))

    return CaseSpec(
        case_id=int(case_id),
        family=str(family),
        seed=int(case_seed),
        nx=int(args.nx),
        nv=int(args.nv),
        Lx=float(args.Lx),
        v_max=float(args.v_max),
        dt=float(dt),
        tau_tilde=float(tau),
        T_total=float(args.T_total),
        picard_iter=int(args.picard_iter),
        picard_tol=float(args.picard_tol),
        abs_tol=float(args.abs_tol),
        conv_type=str(args.conv_type).lower(),
        initial_regions=initial_regions,
        dt_over_tau=dt_over_tau,
        log10_dt=_safe_log10(dt),
        log10_tau=_safe_log10(tau),
        log10_dt_over_tau=_safe_log10(dt_over_tau),
        n_ratio=float(n_ratio),
        T_ratio=float(T_ratio),
        u_rms_over_sqrtT=float(u_rms_over_sqrtT),
    )


def _build_all_case_specs(args: argparse.Namespace) -> list[CaseSpec]:
    fam_names, fam_weights = _parse_family_weights(args.family_weights)
    rng = np.random.default_rng(int(args.seed))

    fam_idx = rng.choice(len(fam_names), size=int(args.cases), p=fam_weights)
    specs: list[CaseSpec] = []
    for cid in range(int(args.cases)):
        fam = fam_names[int(fam_idx[cid])]
        specs.append(_sample_case_spec(case_id=cid, family=fam, args=args))
    return specs


def _setup_dist(device_arg: str) -> tuple[bool, int, int, int, torch.device]:
    is_dist = ("RANK" in os.environ) and ("WORLD_SIZE" in os.environ)
    if is_dist:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
        return True, rank, local_rank, world_size, device

    if device_arg == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)
    return False, 0, 0, 1, device


def _build_cfg(case: CaseSpec, device: torch.device) -> Config:
    scheme_params = BGK1D.implicit.Params(
        picard_iter=int(case.picard_iter),
        picard_tol=float(case.picard_tol),
        abs_tol=float(case.abs_tol),
    )

    fnames = {f.name for f in fields(scheme_params)}
    if "conv_type" in fnames:
        scheme_params = replace(scheme_params, conv_type=str(case.conv_type))
    if "warm_enable" in fnames:
        scheme_params = replace(scheme_params, warm_enable=False)
    if "moments_cnn_modelpath" in fnames:
        scheme_params = replace(scheme_params, moments_cnn_modelpath=None)

    model_cfg = BGK1D.ModelConfig(
        grid=BGK1D.Grid1D1V(
            nx=int(case.nx),
            nv=int(case.nv),
            Lx=float(case.Lx),
            v_max=float(case.v_max),
        ),
        time=BGK1D.TimeConfig(
            dt=float(case.dt),
            T_total=float(case.T_total),
        ),
        params=BGK1D.BGK1D1VParams(
            tau_tilde=float(case.tau_tilde),
        ),
        scheme_params=scheme_params,
        initial=BGK1D.InitialCondition1D(initial_regions=tuple(case.initial_regions)),
    )

    return Config(
        model="BGK1D1V",
        scheme="implicit",
        backend="cuda_kernel",
        model_cfg=model_cfg,
        device=str(device),
        log_level="err",
        use_tqdm=False,
    )


def _run_case(case: CaseSpec, device: torch.device) -> CaseRun:
    cfg = _build_cfg(case, device)
    maker = Engine(cfg)

    n_steps = int(cfg.model_cfg.time.n_steps)
    nx = int(cfg.model_cfg.grid.nx)

    W = np.empty((n_steps + 1, 3, nx), dtype=np.float32)
    picard_iter = np.empty((n_steps + 1,), dtype=np.int32)
    std_resid = np.empty((n_steps + 1,), dtype=np.float32)

    with torch.no_grad():
        n0, u0, T0 = calculate_moments(maker.state, maker.state.f)
    W[0, 0] = n0.detach().cpu().float().numpy()
    W[0, 1] = u0.detach().cpu().float().numpy()
    W[0, 2] = T0.detach().cpu().float().numpy()
    picard_iter[0] = 0
    std_resid[0] = 0.0

    _sync(device)
    t0 = time.perf_counter()

    for step in range(n_steps):
        maker.stepper(step)
        bench = getattr(maker.stepper, "benchlog", None) or {}

        with torch.no_grad():
            n, u, T = calculate_moments(maker.state, maker.state.f)

        W[step + 1, 0] = n.detach().cpu().float().numpy()
        W[step + 1, 1] = u.detach().cpu().float().numpy()
        W[step + 1, 2] = T.detach().cpu().float().numpy()
        picard_iter[step + 1] = int(bench.get("picard_iter", -1))
        std_resid[step + 1] = float(bench.get("std_picard_residual", np.nan))

    _sync(device)
    elapsed = float(time.perf_counter() - t0)

    return CaseRun(
        W=W,
        picard_iter=picard_iter,
        std_residual=std_resid,
        elapsed_sec=elapsed,
    )


class ShardWriter:
    def __init__(self, shards_dir: Path, rank: int, cases_per_shard: int):
        self.shards_dir = Path(shards_dir)
        self.rank = int(rank)
        self.cases_per_shard = max(int(cases_per_shard), 1)

        self._buffer_cases: list[CaseSpec] = []
        self._buffer_runs: list[CaseRun] = []
        self._shard_idx = 0

    def add_case(self, case: CaseSpec, run: CaseRun) -> list[dict]:
        self._buffer_cases.append(case)
        self._buffer_runs.append(run)
        if len(self._buffer_cases) >= self.cases_per_shard:
            return self.flush()
        return []

    def flush(self) -> list[dict]:
        if not self._buffer_cases:
            return []

        n_cases = len(self._buffer_cases)
        nx0 = int(self._buffer_cases[0].nx)
        total_frames = int(sum(int(r.W.shape[0]) for r in self._buffer_runs))

        W_all = torch.empty((total_frames, 3, nx0), dtype=torch.float32)
        it_all = torch.empty((total_frames,), dtype=torch.int32)
        resid_all = torch.empty((total_frames,), dtype=torch.float32)

        case_ptr = [0]
        case_meta: list[dict] = []

        cursor = 0
        for case, run in zip(self._buffer_cases, self._buffer_runs):
            frames = int(run.W.shape[0])
            if int(case.nx) != nx0:
                raise ValueError("All cases in a shard must have same nx")

            W_all[cursor:cursor + frames].copy_(torch.from_numpy(run.W))
            it_all[cursor:cursor + frames].copy_(torch.from_numpy(run.picard_iter.astype(np.int32, copy=False)))
            resid_all[cursor:cursor + frames].copy_(torch.from_numpy(run.std_residual.astype(np.float32, copy=False)))

            case_info = asdict(case)
            case_info["initial_regions"] = [dict(r) for r in case.initial_regions]
            case_info["run_elapsed_sec"] = float(run.elapsed_sec)

            it_valid = run.picard_iter[1:]
            it_valid = it_valid[it_valid > 0]
            case_info["picard_iter_sum"] = int(np.sum(it_valid)) if it_valid.size else 0
            case_info["picard_iter_mean"] = float(np.mean(it_valid)) if it_valid.size else float("nan")

            std_tail = run.std_residual[1:]
            case_info["std_residual_max"] = float(np.nanmax(std_tail)) if std_tail.size else float("nan")

            case_meta.append(case_info)

            cursor += frames
            case_ptr.append(cursor)

        shard_name = f"shard_rank{self.rank:02d}_{self._shard_idx:05d}.pt"
        shard_rel = Path("shards") / shard_name
        shard_path = self.shards_dir / shard_name

        payload = {
            "schema_version": 2,
            "W": W_all,
            "picard_iter": it_all,
            "std_picard_residual": resid_all,
            "case_ptr": torch.tensor(case_ptr, dtype=torch.int64),
            "case_meta": case_meta,
        }
        torch.save(payload, shard_path)

        records: list[dict] = []
        for local_idx, meta in enumerate(case_meta):
            frame_start = int(case_ptr[local_idx])
            frame_end = int(case_ptr[local_idx + 1])

            rec = dict(meta)
            rec["shard_path"] = shard_rel.as_posix()
            rec["local_case_index"] = int(local_idx)
            rec["frame_start"] = frame_start
            rec["frame_end"] = frame_end
            rec["n_steps"] = int(max(frame_end - frame_start - 1, 0))
            records.append(rec)

        self._buffer_cases.clear()
        self._buffer_runs.clear()
        self._shard_idx += 1
        return records


def _write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    if not path.exists():
        return out
    with path.open("r") as f:
        for line in f:
            s = line.strip()
            if s:
                out.append(json.loads(s))
    return out


def _assign_random_split(
    records: list[dict],
    *,
    split_key: str,
    indices: list[int],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> None:
    if not indices:
        return

    rng = np.random.default_rng(seed)
    arr = np.asarray(indices, dtype=np.int64)
    rng.shuffle(arr)

    n = int(arr.size)
    n_train = int(round(float(train_ratio) * n))
    n_val = int(round(float(val_ratio) * n))
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)

    for i, ridx in enumerate(arr.tolist()):
        if i < n_train:
            split = "train"
        elif i < n_train + n_val:
            split = "val"
        else:
            split = "test"
        records[ridx][split_key] = split


def _merge_and_write_manifests(
    *,
    out_root: Path,
    world_size: int,
    args: argparse.Namespace,
) -> None:
    rank_records: list[dict] = []
    for rank in range(int(world_size)):
        p = out_root / "manifests" / f"case_manifest_rank{rank:02d}.jsonl"
        rank_records.extend(_read_jsonl(p))

    if not rank_records:
        raise RuntimeError("No case records found from worker ranks")

    rank_records.sort(key=lambda x: int(x["case_id"]))

    n_all = len(rank_records)
    all_indices = list(range(n_all))

    _assign_random_split(
        rank_records,
        split_key="split_iid",
        indices=all_indices,
        seed=int(args.split_seed),
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
    )

    holdout_fams = {s.strip() for s in str(args.ood_family_holdout).split(",") if s.strip()}
    in_dist_idx: list[int] = []
    for i, rec in enumerate(rank_records):
        if str(rec["family"]) in holdout_fams:
            rec["split_ood_family"] = "test"
        else:
            in_dist_idx.append(i)

    ratio_den = max(float(args.train_ratio) + float(args.val_ratio), 1e-12)
    _assign_random_split(
        rank_records,
        split_key="split_ood_family",
        indices=in_dist_idx,
        seed=int(args.split_seed) + 17,
        train_ratio=float(args.train_ratio) / ratio_den,
        val_ratio=float(args.val_ratio) / ratio_den,
        test_ratio=0.0,
    )

    in_range_idx: list[int] = []
    for i, rec in enumerate(rank_records):
        n_ratio = float(rec.get("n_ratio", 1.0))
        T_ratio = float(rec.get("T_ratio", 1.0))
        log_dt_tau = float(rec.get("log10_dt_over_tau", 0.0))

        is_ood = (
            (n_ratio >= float(args.ood_ratio_threshold))
            or (T_ratio >= float(args.ood_ratio_threshold))
            or (log_dt_tau < float(args.ood_log10_dt_over_tau_min))
            or (log_dt_tau > float(args.ood_log10_dt_over_tau_max))
        )

        if is_ood:
            rec["split_ood_range"] = "test"
        else:
            in_range_idx.append(i)

    _assign_random_split(
        rank_records,
        split_key="split_ood_range",
        indices=in_range_idx,
        seed=int(args.split_seed) + 31,
        train_ratio=float(args.train_ratio) / ratio_den,
        val_ratio=float(args.val_ratio) / ratio_den,
        test_ratio=0.0,
    )

    case_manifest = out_root / "case_manifest.jsonl"
    _write_jsonl(case_manifest, rank_records)

    fam_count: dict[str, int] = {}
    for rec in rank_records:
        fam = str(rec["family"])
        fam_count[fam] = fam_count.get(fam, 0) + 1

    split_count: dict[str, dict[str, int]] = {}
    for key in ("split_iid", "split_ood_family", "split_ood_range"):
        c = {"train": 0, "val": 0, "test": 0}
        for rec in rank_records:
            c[str(rec[key])] += 1
        split_count[key] = c

    n_shards = len(list((out_root / "shards").glob("*.pt")))

    dataset_manifest = {
        "schema_version": 2,
        "format": "kineticEQ_BGK1D1V_pt_v2",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "generator": {
            "script": "kineticEQ.CNN.BGK1D1V.gen_traindata_v2.generate_bgk1d_implicit_dataset",
            "args": vars(args),
        },
        "num_cases": int(len(rank_records)),
        "num_shards": int(n_shards),
        "world_size": int(world_size),
        "families": fam_count,
        "split_counts": split_count,
        "paths": {
            "shards_dir": "shards",
            "case_manifest": "case_manifest.jsonl",
        },
    }

    dataset_manifest_path = out_root / "dataset_manifest.json"
    dataset_manifest_path.write_text(json.dumps(dataset_manifest, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--cases", type=int, default=1200)
    p.add_argument("--cases_per_shard", type=int, default=200)
    p.add_argument("--overwrite", action="store_true")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split_seed", type=int, default=123)
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.1)

    p.add_argument("--device", type=str, default="auto")

    p.add_argument("--nx", type=int, default=512)
    p.add_argument("--nv", type=int, default=256)
    p.add_argument("--Lx", type=float, default=1.0)
    p.add_argument("--v_max", type=float, default=10.0)
    p.add_argument("--T_total", type=float, default=0.05)

    p.add_argument("--dt_center", type=float, default=5e-4)
    p.add_argument("--dt_center_prob", type=float, default=0.5)
    p.add_argument("--dt_min", type=float, default=5e-5)
    p.add_argument("--dt_max", type=float, default=5e-3)

    p.add_argument("--tau_center", type=float, default=5e-7)
    p.add_argument("--tau_center_prob", type=float, default=0.4)
    p.add_argument("--tau_min", type=float, default=5e-9)
    p.add_argument("--tau_max", type=float, default=5e-6)

    p.add_argument("--picard_iter", type=int, default=10000)
    p.add_argument("--picard_tol", type=float, default=1e-8)
    p.add_argument("--abs_tol", type=float, default=1e-13)
    p.add_argument("--conv_type", type=str, choices=["f", "w"], default="w")

    p.add_argument("--family_weights", type=str, default=DEFAULT_FAMILY_WEIGHTS)
    p.add_argument("--velocity_segments", type=int, default=16)
    p.add_argument("--edge_width", type=float, default=0.02)

    p.add_argument("--ood_family_holdout", type=str, default="isobaric_contact,double_hotspot_isobaric")
    p.add_argument("--ood_ratio_threshold", type=float, default=6.0)
    p.add_argument("--ood_log10_dt_over_tau_min", type=float, default=-2.0)
    p.add_argument("--ood_log10_dt_over_tau_max", type=float, default=4.0)

    p.add_argument("--log_interval", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    split_sum = float(args.train_ratio) + float(args.val_ratio) + float(args.test_ratio)
    if abs(split_sum - 1.0) > 1e-8:
        raise ValueError(f"train/val/test ratio must sum to 1.0, got {split_sum}")

    if float(args.dt_min) <= 0.0 or float(args.dt_max) <= 0.0:
        raise ValueError("dt_min and dt_max must be positive")
    if float(args.dt_min) > float(args.dt_max):
        raise ValueError(f"dt_min must be <= dt_max, got {args.dt_min} > {args.dt_max}")
    if float(args.tau_min) <= 0.0 or float(args.tau_max) <= 0.0:
        raise ValueError("tau_min and tau_max must be positive")
    if float(args.tau_min) > float(args.tau_max):
        raise ValueError(f"tau_min must be <= tau_max, got {args.tau_min} > {args.tau_max}")

    is_dist, rank, local_rank, world_size, device = _setup_dist(args.device)
    out_root = Path(args.out_dir).resolve()

    if rank == 0 and args.overwrite and out_root.exists():
        shutil.rmtree(out_root)

    if is_dist:
        dist.barrier()

    (out_root / "shards").mkdir(parents=True, exist_ok=True)
    (out_root / "manifests").mkdir(parents=True, exist_ok=True)

    all_specs = _build_all_case_specs(args)
    my_specs = all_specs[rank::world_size]

    writer = ShardWriter(
        shards_dir=out_root / "shards",
        rank=rank,
        cases_per_shard=int(args.cases_per_shard),
    )

    local_records: list[dict] = []
    t_rank0 = time.perf_counter()

    for i, case in enumerate(my_specs, start=1):
        run = _run_case(case, device=device)
        local_records.extend(writer.add_case(case, run))

        if (i == 1) or (i % max(int(args.log_interval), 1) == 0) or (i == len(my_specs)):
            print(
                f"[rank {rank}] {i}/{len(my_specs)} cases done "
                f"(case_id={case.case_id}, family={case.family}, elapsed={run.elapsed_sec:.2f}s)",
                flush=True,
            )

    local_records.extend(writer.flush())

    local_manifest_path = out_root / "manifests" / f"case_manifest_rank{rank:02d}.jsonl"
    _write_jsonl(local_manifest_path, local_records)

    if is_dist:
        dist.barrier()

    if rank == 0:
        _merge_and_write_manifests(
            out_root=out_root,
            world_size=world_size,
            args=args,
        )
        elapsed_all = time.perf_counter() - t_rank0
        print(f"[rank 0] finished dataset generation in {elapsed_all:.1f}s", flush=True)
        print(f"[rank 0] dataset_manifest: {(out_root / 'dataset_manifest.json').as_posix()}", flush=True)
        print(f"[rank 0] case_manifest: {(out_root / 'case_manifest.jsonl').as_posix()}", flush=True)

    if is_dist:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
