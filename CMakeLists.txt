cmake_minimum_required(VERSION 3.18)
project(kineticEQ LANGUAGES Fortran C)

# ❶ scikit-build-core を検出し、CMAKE_MODULE_PATH をセット
find_package(scikit_build_core CONFIG REQUIRED)

# ❷ マクロを読み込む  ※ここは 'skbuild'
include(skbuild)

# ❸ （OpenMP を使う場合だけ）
find_package(OpenMP)

# ❹ Fortran → f2py 拡張
skbuild_add_f2py_extension(advection1d
  MODULE_NAME advection1d
  SOURCES
    src/kineticEQ/backends/fortran/Sample_1Dadvection/Sample_1D_advection.f90
  INSTALL_DESTINATION kineticEQ/backends
)

if(OpenMP_Fortran_FOUND)
  target_link_libraries(advection1d PRIVATE OpenMP::OpenMP_Fortran)
endif()
