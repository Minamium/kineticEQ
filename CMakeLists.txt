cmake_minimum_required(VERSION 3.18)
project(kineticEQ LANGUAGES Fortran C)

# ────────────────────────────────────────────────────────────────
#  scikit-build-core のマクロを読み込む
# ────────────────────────────────────────────────────────────────
include(ScikitBuildCore)        # ← これが無いとマクロを認識できません

# ────────────────────────────────────────────────────────────────
#  Fortran → F2PY 拡張ライブラリ
#   ▷ パスは実際に存在するソースへ合わせてください
# ────────────────────────────────────────────────────────────────
scikit_build_add_f2py_extension(advection1d
    MODULE_NAME advection1d
    SOURCES
        src/kineticEQ/backends/fortran/Sample_1Dadvection/Sample_1D_advection.f90
    INSTALL_DESTINATION kineticEQ/backends      # wheel 内で import される場所
)

# ────────────────────────────────────────────────────────────────
#  OpenMP を使う場合（Fortran に !$omp がある場合だけ有効化）
# ────────────────────────────────────────────────────────────────
find_package(OpenMP)
if(OpenMP_Fortran_FOUND)
    target_link_libraries(advection1d PRIVATE OpenMP::OpenMP_Fortran)
endif()
