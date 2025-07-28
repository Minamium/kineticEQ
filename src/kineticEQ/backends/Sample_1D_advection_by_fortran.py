"""
Runtime-built Fortran backend (OpenMP)
-------------------------------------
初回 import 時に f2py でビルドし、build/ に .so をキャッシュ。
"""
from __future__ import annotations
import hashlib, importlib.util, os, subprocess, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SRC  = HERE / "fortran" / "Sample_1Dadvection" / "Sample_1D_advection.f90"
BUILD_DIR = HERE.parent.parent.parent.parent / "build"  # kineticEQ/build
BUILD_DIR = BUILD_DIR.resolve()
BUILD_DIR.mkdir(exist_ok=True)

# — ハッシュ（ソース + フラグ）で一意に
def _tag() -> str:
    """ソースファイルとコンパイルフラグから一意のハッシュを生成"""
    if not SRC.exists():
        raise FileNotFoundError(f"Fortranソースファイルが見つかりません: {SRC}")
    
    txt = SRC.read_bytes()
    flags = os.environ.get("KINEQ_FFLAGS", "-O3 -fopenmp").encode()
    return hashlib.sha1(txt + flags).hexdigest()[:12]

def _build() -> Path:
    """F2PYを使用してFortranモジュールをビルド"""
    try:
        tag = _tag()
        so_name = f"advection1d_{tag}{importlib.machinery.EXTENSION_SUFFIXES[0]}"
        so_path = BUILD_DIR / so_name
        
        # 既存のビルドをチェック
        if so_path.exists() and not os.getenv("KINEQ_FORCE_REBUILD"):
            print(f"[kineticEQ] 既存のFortranバックエンドを使用: {so_path.name}")
            return so_path

        # F2PYビルドコマンド
        cmd = [
            sys.executable, "-m", "numpy.f2py",
            "-c", str(SRC),
            "-m", "advection1d",
            f"--f90exec={os.getenv('FC', 'gfortran')}",
            f"--f90flags={os.getenv('KINEQ_FFLAGS', '-O3 -fopenmp')}",
            "-lgomp",
            f"--build-dir={BUILD_DIR}",
            "-o", str(so_path)
        ]
        
        print(f"[kineticEQ] Fortranバックエンドをビルド中 → {so_path.name}")
        print(f"[kineticEQ] コマンド: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        if result.returncode != 0:
            print(f"[kineticEQ] F2PYビルドエラー:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            raise RuntimeError(f"F2PYビルドが失敗しました (exit code: {result.returncode})")
        
        if not so_path.exists():
            raise RuntimeError(f"ビルドは成功しましたが、出力ファイルが見つかりません: {so_path}")
        
        print(f"[kineticEQ] ビルド完了: {so_path}")
        return so_path
        
    except Exception as e:
        print(f"[kineticEQ] Fortranバックエンドのビルドに失敗: {e}")
        raise

# — 動的 import with エラーハンドリング
try:
    _so = _build()
    _spec = importlib.util.spec_from_file_location("advection1d", _so)
    if _spec is None or _spec.loader is None:
        raise ImportError(f"モジュールスペックの作成に失敗: {_so}")
    
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    
    # モジュール構造を確認
    if not hasattr(_mod, 'sample_1d_advection_main_module'):
        available_attrs = [attr for attr in dir(_mod) if not attr.startswith('_')]
        raise AttributeError(f"期待されるモジュール 'sample_1d_advection_main_module' が見つかりません。利用可能: {available_attrs}")
    
    print(f"[kineticEQ] Fortranバックエンドのロードが完了")
    
except Exception as e:
    print(f"[kineticEQ] Fortranバックエンドの初期化に失敗: {e}")
    _mod = None

# — 公開 API
def step(q: np.ndarray, dt: float, dx: float, u: float, nt: int = 1) -> np.ndarray:
    """一次風上差分を nt ステップ進める（周期境界）
    
    Args:
        q: 初期濃度分布 [nx]
        dt: 時間刻み
        dx: 空間刻み
        u: 移流速度
        nt: 時間ステップ数
        
    Returns:
        最終濃度分布 [nx]
    """
    if _mod is None:
        raise RuntimeError("Fortranバックエンドが利用できません。ビルドエラーを確認してください。")
    
    # 入力検証
    q = np.ascontiguousarray(q, dtype=np.float64)
    if q.ndim != 1:
        raise ValueError(f"qは1次元配列である必要があります。受信: {q.shape}")
    
    nx = q.size
    if nx < 2:
        raise ValueError(f"格子点数は2以上である必要があります。受信: {nx}")
    
    # Fortranサブルーチンを呼び出し
    try:
        q_final = np.zeros_like(q)
        _mod.sample_1d_advection_main_module.advec_upwind(
            nt, nx, dt, dx, u, q, q_final
        )
        return q_final
        
    except Exception as e:
        raise RuntimeError(f"Fortran計算中にエラーが発生: {e}")

def is_available() -> bool:
    """Fortranバックエンドが利用可能かチェック"""
    return _mod is not None

def get_info() -> dict:
    """バックエンド情報を取得"""
    return {
        "backend": "Fortran (F2PY + OpenMP)",
        "available": is_available(),
        "source_file": str(SRC),
        "build_dir": str(BUILD_DIR),
        "compiler": os.getenv('FC', 'gfortran'),
        "flags": os.getenv('KINEQ_FFLAGS', '-O3 -fopenmp')
    }
