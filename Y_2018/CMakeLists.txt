cmake_minimum_required (VERSION 3.14)
set(CMAKE_CXX_STANDARD 17)

add_library (Y_2018
STATIC
"default.h"
"d_01.h"
"d_02.h"
"y_2018.h" "y_2018.cpp"
)

target_link_libraries(Y_2018
Utilities
)