cmake_minimum_required(VERSION 3.18)
project(kineticEQ LANGUAGES Fortran C)

add_library(advection1d MODULE
    src/kineticEQ/backends/fortran/Sample_1Dadvection/Sample_1D_advection.f90)
set_target_properties(advection1d PROPERTIES
    OUTPUT_NAME "advection1d" PREFIX "")

install(TARGETS advection1d
        LIBRARY DESTINATION kineticEQ/backends)