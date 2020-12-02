file(GLOB INPUTS inputs/*.in)
file(COPY ${INPUTS} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

add_executable (main
    "Main.cpp")

target_link_libraries(main
    y2018
    #y2019
    y2020)