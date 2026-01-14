# kineticEQ/analysis/BGK1D/utils/compute_err.py
from __future__ import annotations
import numpy as np


def append_errors(out: dict, *, kind: str = "nearest") -> dict:
    """
    out["records"] の snapshot(fields) から ref(最細) を選び、coarse に写像して L1/L2/Linf を計算して out["errors"] に追加する。
    bench_type: meta["bench_type"] in {"x_grid","v_grid"} のみ対応
    補間: kind in {"nearest","linear"} のみ
    """
    meta = out.get("meta", {})
    bench_type = meta.get("bench_type")
    if bench_type not in ("x_grid", "v_grid"):
        return out

    snaps = [r for r in out.get("records", []) if isinstance(r.get("fields"), dict)]
    snaps = [r for r in snaps if all(k in r["fields"] for k in ("x", "v", "f", "n", "u", "T", "dx", "dv"))]
    if len(snaps) < 2:
        return out

    if kind not in ("nearest", "linear"):
        raise ValueError(f"kind must be 'nearest' or 'linear', got {kind}")

    def _nx_nv(r):
        sw = r.get("sweep", {})
        return int(sw.get("nx", -1)), int(sw.get("nv", -1))

    ref = max(snaps, key=(lambda r: _nx_nv(r)[0] if bench_type == "x_grid" else _nx_nv(r)[1]))
    rx = np.asarray(ref["fields"]["x"])
    rv = np.asarray(ref["fields"]["v"])
    rf = np.asarray(ref["fields"]["f"])

    def _interp_1d(y_ref: np.ndarray, x_ref: np.ndarray, x_new: np.ndarray) -> np.ndarray:
        if kind == "linear":
            return np.interp(x_new, x_ref, y_ref)
        # nearest
        idx = np.searchsorted(x_ref, x_new)
        idx = np.clip(idx, 0, len(x_ref) - 1)
        idx2 = np.clip(idx - 1, 0, len(x_ref) - 1)
        pick = np.where(np.abs(x_ref[idx] - x_new) <= np.abs(x_ref[idx2] - x_new), idx, idx2)
        return y_ref[pick]

    def _map_ref_f_to_coarse(cx: np.ndarray, cv: np.ndarray) -> np.ndarray:
        # out shape = (len(cx), len(cv))
        if bench_type == "x_grid":
            tmp = rf
            if not np.allclose(rv, cv):
                tmp2 = np.empty((rf.shape[0], len(cv)), dtype=rf.dtype)
                for xi in range(rf.shape[0]):
                    tmp2[xi, :] = _interp_1d(rf[xi, :], rv, cv)
                tmp = tmp2
            out_f = np.empty((len(cx), tmp.shape[1]), dtype=tmp.dtype)
            for vj in range(tmp.shape[1]):
                out_f[:, vj] = _interp_1d(tmp[:, vj], rx, cx)
            return out_f

        # v_grid
        tmp = rf
        if not np.allclose(rx, cx):
            tmp2 = np.empty((len(cx), rf.shape[1]), dtype=rf.dtype)
            for vj in range(rf.shape[1]):
                tmp2[:, vj] = _interp_1d(rf[:, vj], rx, cx)
            tmp = tmp2
        out_f = np.empty((tmp.shape[0], len(cv)), dtype=tmp.dtype)
        for xi in range(tmp.shape[0]):
            out_f[xi, :] = _interp_1d(tmp[xi, :], rv, cv)
        return out_f

    def _norms(cur: dict, ref_on: dict) -> dict:
        dx = float(cur["dx"]); dv = float(cur["dv"])
        w2 = dx * dv; w1 = dx

        f_err = np.asarray(cur["f"]) - np.asarray(ref_on["f"])
        n_err = np.asarray(cur["n"]) - np.asarray(ref_on["n"])
        u_err = np.asarray(cur["u"]) - np.asarray(ref_on["u"])
        T_err = np.asarray(cur["T"]) - np.asarray(ref_on["T"])

        return {
            "L1": {"f": float(np.sum(np.abs(f_err)) * w2), "n": float(np.sum(np.abs(n_err)) * w1),
                   "u": float(np.sum(np.abs(u_err)) * w1), "T": float(np.sum(np.abs(T_err)) * w1)},
            "L2": {"f": float(np.sqrt(np.sum(f_err**2) * w2)), "n": float(np.sqrt(np.sum(n_err**2) * w1)),
                   "u": float(np.sqrt(np.sum(u_err**2) * w1)), "T": float(np.sqrt(np.sum(T_err**2) * w1))},
            "Linf": {"f": float(np.max(np.abs(f_err))), "n": float(np.max(np.abs(n_err))),
                     "u": float(np.max(np.abs(u_err))), "T": float(np.max(np.abs(T_err)))},
        }

    ref_sw = {"nx": _nx_nv(ref)[0], "nv": _nx_nv(ref)[1], "tag": ref.get("tag", "")}

    errs = []
    for r in snaps:
        if r is ref:
            continue
        cur = r["fields"]
        cx = np.asarray(cur["x"])
        cv = np.asarray(cur["v"])

        ref_f = _map_ref_f_to_coarse(cx, cv)

        # n/u/T は簡易に補間（短くするため）。厳密にするなら ref_f から再計算へ。
        if bench_type == "x_grid":
            ref_n = _interp_1d(np.asarray(ref["fields"]["n"]), rx, cx)
            ref_u = _interp_1d(np.asarray(ref["fields"]["u"]), rx, cx)
            ref_T = _interp_1d(np.asarray(ref["fields"]["T"]), rx, cx)
        else:
            if np.allclose(rx, cx):
                ref_n = np.asarray(ref["fields"]["n"])
                ref_u = np.asarray(ref["fields"]["u"])
                ref_T = np.asarray(ref["fields"]["T"])
            else:
                ref_n = _interp_1d(np.asarray(ref["fields"]["n"]), rx, cx)
                ref_u = _interp_1d(np.asarray(ref["fields"]["u"]), rx, cx)
                ref_T = _interp_1d(np.asarray(ref["fields"]["T"]), rx, cx)

        norms = _norms(cur, {"f": ref_f, "n": ref_n, "u": ref_u, "T": ref_T})
        nx, nv = _nx_nv(r)
        errs.append({"sweep": {"nx": nx, "nv": nv}, "ref": ref_sw, "kind": kind, "norms": norms})

    out.setdefault("errors", [])
    out["errors"].extend(errs)
    return out
