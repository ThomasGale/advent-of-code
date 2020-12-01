cmake_minimum_required (VERSION 3.14)
set(CMAKE_CXX_STANDARD 17)

configure_file("../Y_2018/d_01.txt" "Y_2018/d_01.txt" COPYONLY)
configure_file("../Y_2018/d_02.txt" "Y_2018/d_02.txt" COPYONLY)
configure_file("../Y_2018/d_03.txt" "Y_2018/d_03.txt" COPYONLY)

configure_file("../Y_2019/d_01.txt" "Y_2019/d_01.txt" COPYONLY)
configure_file("../Y_2019/d_02.txt" "Y_2019/d_02.txt" COPYONLY)
configure_file("../Y_2019/d_03.txt" "Y_2019/d_03.txt" COPYONLY)
configure_file("../Y_2019/d_04.txt" "Y_2019/d_04.txt" COPYONLY)
configure_file("../Y_2019/d_05.txt" "Y_2019/d_05.txt" COPYONLY)
configure_file("../Y_2019/d_06.txt" "Y_2019/d_06.txt" COPYONLY)
configure_file("../Y_2019/d_07.txt" "Y_2019/d_07.txt" COPYONLY)
configure_file("../Y_2019/d_08.txt" "Y_2019/d_08.txt" COPYONLY)
configure_file("../Y_2019/d_09.txt" "Y_2019/d_09.txt" COPYONLY)
configure_file("../Y_2019/d_10.txt" "Y_2019/d_10.txt" COPYONLY)
configure_file("../Y_2019/d_11.txt" "Y_2019/d_11.txt" COPYONLY)
configure_file("../Y_2019/d_12.txt" "Y_2019/d_12.txt" COPYONLY)
configure_file("../Y_2019/d_13.txt" "Y_2019/d_13.txt" COPYONLY)
configure_file("../Y_2019/d_14.txt" "Y_2019/d_14.txt" COPYONLY)
configure_file("../Y_2019/d_15.txt" "Y_2019/d_15.txt" COPYONLY)
configure_file("../Y_2019/d_16.txt" "Y_2019/d_16.txt" COPYONLY)
configure_file("../Y_2019/d_17.txt" "Y_2019/d_17.txt" COPYONLY)
configure_file("../Y_2019/d_18.txt" "Y_2019/d_18.txt" COPYONLY)
configure_file("../Y_2019/d_19.txt" "Y_2019/d_19.txt" COPYONLY)
configure_file("../Y_2019/d_20.txt" "Y_2019/d_20.txt" COPYONLY)
configure_file("../Y_2019/d_21.txt" "Y_2019/d_21.txt" COPYONLY)
configure_file("../Y_2019/d_22.txt" "Y_2019/d_22.txt" COPYONLY)
configure_file("../Y_2019/d_23.txt" "Y_2019/d_23.txt" COPYONLY)

add_executable (Run
    "run.cpp")

target_link_libraries(Run
    Y_2018
    Y_2019)