cmake_minimum_required (VERSION 3.14)
set(CMAKE_CXX_STANDARD 17)

add_library (Y_2019
STATIC
"default.h"
"d_01.h"
"y_2019.h" "y_2019.cpp"
)

target_link_libraries(Y_2019
Utilities
)