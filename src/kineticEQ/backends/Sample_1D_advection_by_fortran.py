"""
Runtime-built Fortran backend (OpenMP)
-------------------------------------
初回 import 時に f2py でビルドし、build/ に .so をキャッシュ。
"""
from __future__ import annotations
import hashlib, importlib.util, os, subprocess, sys
from pathlib import Path
import numpy as np
import glob

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
        
        # 既存のビルドをチェック（パターンマッチング）
        existing_files = list(BUILD_DIR.glob(f"advection1d_{tag}*.so"))
        if existing_files and not os.getenv("KINEQ_FORCE_REBUILD"):
            so_path = existing_files[0]
            print(f"[kineticEQ] 既存のFortranバックエンドを使用: {so_path.name}")
            return so_path

        # F2PYビルドコマンド（-oオプションを使わず、build-dirのみ指定）
        cmd = [
            sys.executable, "-m", "numpy.f2py",
            "-c", str(SRC),
            "-m", "advection1d",
            f"--f90exec={os.getenv('FC', 'gfortran')}",
            f"--f90flags={os.getenv('KINEQ_FFLAGS', '-O3 -fopenmp')}",
            "-lgomp",
            f"--build-dir={BUILD_DIR}"
        ]
        
        print(f"[kineticEQ] Fortranバックエンドをビルド中...")
        print(f"[kineticEQ] コマンド: {' '.join(cmd)}")
        
        # 現在の作業ディレクトリを変更してF2PYを実行
        original_cwd = os.getcwd()
        try:
            os.chdir(BUILD_DIR)
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        finally:
            os.chdir(original_cwd)
        
        if result.returncode != 0:
            print(f"[kineticEQ] F2PYビルドエラー:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            raise RuntimeError(f"F2PYビルドが失敗しました (exit code: {result.returncode})")
        
        # ビルド後に生成されたファイルを検索
        so_patterns = [
            f"advection1d*.so",
            f"advection1d*.pyd",  # Windows
            f"advection1d*{importlib.machinery.EXTENSION_SUFFIXES[0]}"
        ]
        
        found_files = []
        for pattern in so_patterns:
            found_files.extend(BUILD_DIR.glob(pattern))
        
        if not found_files:
            # より詳細な検索
            all_files = list(BUILD_DIR.glob("*"))
            print(f"[kineticEQ] ビルドディレクトリの内容: {[f.name for f in all_files]}")
            raise RuntimeError(f"ビルド後に.soファイルが見つかりません。パターン: {so_patterns}")
        
        # 最新のファイルを選択
        so_path = max(found_files, key=lambda p: p.stat().st_mtime)
        
        # タグ付きファイル名にリネーム
        target_name = f"advection1d_{tag}{so_path.suffix}"
        target_path = BUILD_DIR / target_name
        
        if so_path != target_path:
            so_path.rename(target_path)
            so_path = target_path
        
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
    
    # モジュール構造を確認・デバッグ
    available_attrs = [attr for attr in dir(_mod) if not attr.startswith('_')]
    print(f"[kineticEQ] 利用可能なモジュール属性: {available_attrs}")
    
    # 期待されるモジュール名をチェック
    expected_modules = [
        'sample_1d_advection_main_module',
        'sample_1d_advection_step_module'
    ]
    
    found_module = None
    for mod_name in expected_modules:
        if hasattr(_mod, mod_name):
            found_module = getattr(_mod, mod_name)
            print(f"[kineticEQ] 使用するモジュール: {mod_name}")
            break
    
    if found_module is None:
        raise AttributeError(f"期待されるモジュールが見つかりません。利用可能: {available_attrs}")
    
    _main_module = found_module
    print(f"[kineticEQ] Fortranバックエンドのロードが完了")
    
except Exception as e:
    print(f"[kineticEQ] Fortranバックエンドの初期化に失敗: {e}")
    _mod = None
    _main_module = None

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
    if _mod is None or _main_module is None:
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
        _main_module.advec_upwind(nt, nx, dt, dx, u, q, q_final)
        return q_final
        
    except Exception as e:
        raise RuntimeError(f"Fortran計算中にエラーが発生: {e}")

def is_available() -> bool:
    """Fortranバックエンドが利用可能かチェック"""
    return _mod is not None and _main_module is not None

def get_info() -> dict:
    """バックエンド情報を取得"""
    return {
        "backend": "Fortran (F2PY + OpenMP)",
        "available": is_available(),
        "source_file": str(SRC),
        "build_dir": str(BUILD_DIR),
        "compiler": os.getenv('FC', 'gfortran'),
        "flags": os.getenv('KINEQ_FFLAGS', '-O3 -fopenmp'),
        "module_loaded": _mod is not None,
        "main_module_loaded": _main_module is not None
    }
