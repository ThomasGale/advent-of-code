cmake_minimum_required (VERSION 3.14)
set(CMAKE_CXX_STANDARD 17)

add_library (Utilities
"reader.hpp" "utils.hpp"
"utilities.cpp"
)

set_target_properties(Utilities PROPERTIES LINKER_LANGUAGE CXX)