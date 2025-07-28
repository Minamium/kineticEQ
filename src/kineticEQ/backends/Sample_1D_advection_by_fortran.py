"""
Runtime-built Fortran backend (OpenMP)
-------------------------------------
初回 import 時に f2py でビルドし、build/ に .so をキャッシュ。
"""
from __future__ import annotations
import hashlib, importlib.util, os, subprocess, sys
from pathlib import Path
import numpy as np
import tempfile
import shutil

HERE = Path(__file__).resolve().parent
SRC  = HERE / "fortran" / "Sample_1Dadvection" / "Sample_1D_advection.f90"
BUILD_DIR = HERE.parent.parent.parent.parent / "build"  # kineticEQ/build
BUILD_DIR = BUILD_DIR.resolve()
BUILD_DIR.mkdir(exist_ok=True)

def _get_openmp_settings():
    """OpenMP設定を取得"""
    # 環境変数でOpenMPの有効/無効を制御
    enable_openmp = os.getenv("KINEQ_ENABLE_OPENMP", "true").lower() in ("true", "1", "yes")
    
    if enable_openmp:
        flags = os.getenv("KINEQ_FFLAGS", "-O3 -fopenmp")
        link_flags = ["-lgomp"] if "-fopenmp" in flags else []
        return flags, link_flags, True
    else:
        flags = os.getenv("KINEQ_FFLAGS", "-O3")
        return flags, [], False

# — ハッシュ（ソース + フラグ）で一意に
def _tag() -> str:
    """ソースファイルとコンパイルフラグから一意のハッシュを生成"""
    if not SRC.exists():
        raise FileNotFoundError(f"Fortranソースファイルが見つかりません: {SRC}")
    
    txt = SRC.read_bytes()
    flags, _, _ = _get_openmp_settings()
    return hashlib.sha1(txt + flags.encode()).hexdigest()[:12]

def _build() -> Path:
    """F2PYを使用してFortranモジュールをビルド"""
    try:
        tag = _tag()
        flags, link_flags, openmp_enabled = _get_openmp_settings()
        
        # 既存のビルドをチェック（パターンマッチング）
        existing_files = list(BUILD_DIR.glob(f"advection1d_{tag}*.so"))
        if existing_files and not os.getenv("KINEQ_FORCE_REBUILD"):
            so_path = existing_files[0]
            print(f"[kineticEQ] 既存のFortranバックエンドを使用: {so_path.name}")
            return so_path

        # 一時ディレクトリでビルド（mesonの問題を回避）
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            temp_src = temp_path / "Sample_1D_advection.f90"
            
            # ソースファイルをコピー
            shutil.copy2(SRC, temp_src)
            
            # F2PYコマンド（OpenMP設定に応じて構築）
            cmd = [
                sys.executable, "-m", "numpy.f2py",
                "-c", str(temp_src),
                "-m", "advection1d",
                f"--f90exec={os.getenv('FC', 'gfortran')}",
                f"--f90flags={flags}"
            ]
            
            # OpenMP有効時はリンクフラグを追加
            cmd.extend(link_flags)
            
            openmp_status = "有効" if openmp_enabled else "無効"
            print(f"[kineticEQ] Fortranバックエンドをビルド中... (OpenMP: {openmp_status})")
            print(f"[kineticEQ] 一時ディレクトリ: {temp_dir}")
            print(f"[kineticEQ] コマンド: {' '.join(cmd)}")
            
            # 一時ディレクトリで実行
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_path)
                result = subprocess.run(cmd, capture_output=True, text=True, check=False)
                
                print(f"[kineticEQ] F2PY終了コード: {result.returncode}")
                if result.stdout:
                    print(f"[kineticEQ] STDOUT: {result.stdout[:500]}...")
                if result.stderr:
                    print(f"[kineticEQ] STDERR: {result.stderr[:500]}...")
                
                if result.returncode != 0:
                    # OpenMP有効時にエラーが発生した場合、OpenMPなしで再試行
                    if openmp_enabled:
                        print(f"[kineticEQ] OpenMP有効でビルド失敗。OpenMPなしで再試行...")
                        os.environ["KINEQ_ENABLE_OPENMP"] = "false"
                        return _build()  # 再帰呼び出し
                    else:
                        raise RuntimeError(f"F2PYビルドが失敗しました (exit code: {result.returncode})")
                
                # 生成されたファイルを検索
                so_patterns = [
                    "advection1d*.so",
                    "advection1d*.pyd",
                    "*.so"  # より広範囲に検索
                ]
                
                found_files = []
                for pattern in so_patterns:
                    found_files.extend(temp_path.glob(pattern))
                
                if not found_files:
                    # 一時ディレクトリの内容を表示
                    all_files = list(temp_path.glob("*"))
                    print(f"[kineticEQ] 一時ディレクトリの内容: {[f.name for f in all_files]}")
                    raise RuntimeError(f"ビルド後に.soファイルが見つかりません")
                
                # 最新のファイルを選択
                so_path_temp = max(found_files, key=lambda p: p.stat().st_mtime)
                print(f"[kineticEQ] 生成されたファイル: {so_path_temp.name}")
                
                # ビルドディレクトリにコピー
                target_name = f"advection1d_{tag}{so_path_temp.suffix}"
                target_path = BUILD_DIR / target_name
                shutil.copy2(so_path_temp, target_path)
                
                print(f"[kineticEQ] ビルド完了: {target_path}")
                return target_path
                
            finally:
                os.chdir(original_cwd)
        
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
    
    # 関数シグネチャを確認
    if hasattr(_main_module, 'advec_upwind'):
        func = _main_module.advec_upwind
        print(f"[kineticEQ] 関数シグネチャ: {func.__doc__}")
    
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
    # F2PYの正しい引数順序: (nt, dt, dx, u, q_init)
    try:
        func = _main_module.advec_upwind
        q_final = func(nt, dt, dx, u, q)
        return q_final
        
    except Exception as e:
        raise RuntimeError(f"Fortran計算中にエラーが発生: {e}")

def is_available() -> bool:
    """Fortranバックエンドが利用可能かチェック"""
    return _mod is not None and _main_module is not None

def get_info() -> dict:
    """バックエンド情報を取得"""
    _, _, openmp_enabled = _get_openmp_settings()
    flags, _, _ = _get_openmp_settings()
    
    backend_name = "Fortran (F2PY + OpenMP)" if openmp_enabled else "Fortran (F2PY)"
    
    return {
        "backend": backend_name,
        "available": is_available(),
        "source_file": str(SRC),
        "build_dir": str(BUILD_DIR),
        "compiler": os.getenv('FC', 'gfortran'),
        "flags": flags,
        "module_loaded": _mod is not None,
        "main_module_loaded": _main_module is not None,
        "openmp_enabled": openmp_enabled
    }
