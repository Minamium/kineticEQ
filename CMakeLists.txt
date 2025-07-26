cmake_minimum_required(VERSION 3.18)
project(kineticEQ LANGUAGES Fortran C)

include(ScikitBuildCore)

# --- Python / NumPy 情報取得 -------------------------------------------
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# NumPy のヘッダーパスを取得
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# --- F2PY を直接使用してFortran拡張をビルド ---
find_program(F2PY_EXECUTABLE f2py REQUIRED)

# Fortran ソースファイル
set(FORTRAN_SOURCE src/kineticEQ/backends/fortran/Sample_1Dadvection/Sample_1D_advection.f90)

# F2PY専用の作業ディレクトリを作成（一意な名前を使用）
set(F2PY_WORK_DIR ${CMAKE_CURRENT_BINARY_DIR}/f2py_build_${CMAKE_BUILD_TYPE})
file(MAKE_DIRECTORY ${F2PY_WORK_DIR})

# F2PY で拡張モジュールを直接ビルド
add_custom_target(advection1d ALL
    COMMAND ${CMAKE_COMMAND} -E env FC=${CMAKE_Fortran_COMPILER}
        ${F2PY_EXECUTABLE} 
        -c ${CMAKE_CURRENT_SOURCE_DIR}/${FORTRAN_SOURCE} 
        -m advection1d 
        --build-dir ${F2PY_WORK_DIR}
        --f90exec=${CMAKE_Fortran_COMPILER}
    DEPENDS ${FORTRAN_SOURCE}
    WORKING_DIRECTORY ${F2PY_WORK_DIR}
    COMMENT "Building F2PY extension advection1d"
    VERBATIM
)

# 生成された共有ライブラリをインストール
install(CODE "
    file(GLOB F2PY_LIBS \"${F2PY_WORK_DIR}/advection1d*.so\")
    if(F2PY_LIBS)
        file(INSTALL \${F2PY_LIBS} DESTINATION \${CMAKE_INSTALL_PREFIX}/kineticEQ/backends)
    endif()
")