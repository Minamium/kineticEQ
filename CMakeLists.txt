cmake_minimum_required(VERSION 3.18)
project(kineticEQ LANGUAGES Fortran C)

# --- Python / NumPy 情報取得 -------------------------------------------
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# NumPy のヘッダーパスを取得
execute_process(
    COMMAND "${Python3_EXECUTABLE}" - <<PY
import numpy, pathlib, sys; print(pathlib.Path(numpy.get_include()).as_posix())
PY
    OUTPUT_VARIABLE NUMPY_INCLUDE
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# --- Fortran → F2PY 拡張ライブラリ -------------------------------------
# skbuild_add_f2py_extension は scikit-build-core 0.7 で利用可能
skbuild_add_f2py_extension(advection1d
    MODULE_NAME advection1d                           # import 名
    SOURCES
        src/kineticEQ/backends/fortran/Sample_1Dadvection/Sample_1D_advection.f90
    INCLUDE_DIRECTORIES "${NUMPY_INCLUDE}"
    INSTALL_DESTINATION kineticEQ/backends            # wheel 内の配置先
)

# 例：OpenMP を有効にしたい場合
# find_package(OpenMP)
# if(OpenMP_Fortran_FOUND)
#   target_link_libraries(advection1d PRIVATE OpenMP::OpenMP_Fortran)
# endif()