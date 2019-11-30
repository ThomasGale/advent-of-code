cmake_minimum_required (VERSION 3.14)
set(CMAKE_CXX_STANDARD 17)

configure_file("../Y_2018/d_01.txt" "Y_2018/d_01.txt" COPYONLY)

add_executable (Run
"run.h" "run.cpp"
)

target_link_libraries(Run
#Utilities
Y_2018
)