cmake_minimum_required(VERSION 3.18)
project(kineticEQ LANGUAGES Fortran C)

# ▼ 新しいマクロ名
include(skbuild_core)                    

find_package(PythonExtensions REQUIRED)
find_package(OpenMP)

# Fortran → f2py 拡張
skbuild_add_f2py_extension(advection1d         # ← 小文字 skbuild_
  MODULE_NAME advection1d
  SOURCES
    src/kineticEQ/backends/fortran/Sample_1Dadvection/Sample_1D_advection.f90
  INSTALL_DESTINATION kineticEQ/backends
)

if(OpenMP_Fortran_FOUND)
  target_link_libraries(advection1d PRIVATE OpenMP::OpenMP_Fortran)
endif()
