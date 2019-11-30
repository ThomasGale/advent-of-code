cmake_minimum_required (VERSION 3.14)
set(CMAKE_CXX_STANDARD 17)

configure_file("../2018/day_01.txt" "2018/day_01.txt" COPYONLY)

add_executable (Run
"run.h" "run.cpp"
)

target_link_libraries(Run
Utilities
2018
)