cmake_minimum_required (VERSION 3.14)
set(CMAKE_CXX_STANDARD 17)

add_library (Utilities
STATIC
"reader.h" "reader.cpp"
"utils.hpp"
)

set_target_properties(Utilities PROPERTIES LINKER_LANGUAGE CXX)