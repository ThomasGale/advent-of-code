cmake_minimum_required(VERSION 3.17.2)
project(AdventOfCode
    VERSION 0.1.0
    DESCRIPTION "Advent of Code"
    LANGUAGES CXX CUDA)

# Project wide C++ standard
set(CMAKE_CXX_STANDARD 17)

add_subdirectory(Utilities)
add_subdirectory(Y_2018)
add_subdirectory(Y_2019)
add_subdirectory(Run)
