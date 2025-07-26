cmake_minimum_required(VERSION 3.18)
project(kineticEQ LANGUAGES Fortran C)

include(skbuild)                        # ← これが正しいモジュール名

find_package(PythonExtensions REQUIRED) # ← これで .so の拡張名処理などを取得
find_package(OpenMP)                    # (OpenMP を使わないなら削除可)

skbuild_add_f2py_extension(advection1d
  MODULE_NAME advection1d
  SOURCES
    src/kineticEQ/backends/fortran/Sample_1Dadvection/Sample_1D_advection.f90
  INSTALL_DESTINATION kineticEQ/backends
)

if(OpenMP_Fortran_FOUND)
  target_link_libraries(advection1d PRIVATE OpenMP::OpenMP_Fortran)
endif()
