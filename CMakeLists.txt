cmake_minimum_required (VERSION 3.14)
set(CMAKE_CXX_STANDARD 17)

project(AventOfCode C CXX)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(Utilities)
add_subdirectory(Y_2018)
add_subdirectory(Y_2019)
add_subdirectory(Run)