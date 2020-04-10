cmake_minimum_required (VERSION 3.14)
set(CMAKE_CXX_STANDARD 17)

project(AventOfCode C CXX)
set(EXECUTABLE_OUTPUT_PATH "out")

add_subdirectory(Utilities)
add_subdirectory(Y_2018)
add_subdirectory(Y_2019)
add_subdirectory(Run)