cmake_minimum_required (VERSION 3.14)
set(CMAKE_CXX_STANDARD 17)

add_library (Y_2019
STATIC
"default.h"
"d_01.h"
"d_02.h"
"d_03.h"
"d_04.h"
"d_05.h"
"d_06.h"
"d_07.h"
"d_08.h"
"d_09.h"
"d_10.h"
"d_11.h"
"y_2019.h" "y_2019.cpp"
)

target_link_libraries(Y_2019
Utilities
)