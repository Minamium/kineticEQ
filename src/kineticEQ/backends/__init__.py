"""
kineticEQ backends module
-------------------------
Fortranバックエンドによる高速計算機能を提供
"""

try:
    from .Sample_1D_advection_by_fortran import step, is_available, get_info
    __all__ = ["step", "is_available", "get_info"]
    
    # バックエンドの状態を確認
    if is_available():
        print("[kineticEQ.backends] Fortranバックエンドが利用可能です")
    else:
        print("[kineticEQ.backends] 警告: Fortranバックエンドが利用できません")
        
except ImportError as e:
    print(f"[kineticEQ.backends] Fortranバックエンドのインポートに失敗: {e}")
    
    # フォールバック関数を定義
    def step(*args, **kwargs):
        raise RuntimeError("Fortranバックエンドが利用できません。ビルドエラーを確認してください。")
    
    def is_available():
        return False
    
    def get_info():
        return {"backend": "Fortran (unavailable)", "available": False, "error": str(e)}
    
    __all__ = ["step", "is_available", "get_info"]
