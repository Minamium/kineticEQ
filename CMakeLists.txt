cmake_minimum_required(VERSION 3.18)
project(kineticEQ LANGUAGES Fortran C)

# ❶ scikit-build-core のマクロをロード
include(skbuild_core)                      # ← これが正しい

# ❷ （必要なら）OpenMP
find_package(OpenMP)                       # !$omp を書いてないなら削除でOK

# ❸ Fortran → f2py 拡張
skbuild_add_f2py_extension(advection1d
  MODULE_NAME advection1d
  SOURCES
    src/kineticEQ/backends/fortran/Sample_1Dadvection/Sample_1D_advection.f90
  INSTALL_DESTINATION kineticEQ/backends
)

if(OpenMP_Fortran_FOUND)
  target_link_libraries(advection1d PRIVATE OpenMP::OpenMP_Fortran)
endif()
