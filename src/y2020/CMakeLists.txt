add_library(y2020 INTERFACE)

target_include_directories(y2020 INTERFACE
    .)

target_link_libraries(y2020 INTERFACE
    solution)
