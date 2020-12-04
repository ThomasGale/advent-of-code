add_library(y2020 INTERFACE)

find_package(Thrust REQUIRED CONFIG)
thrust_create_target(Thrust)

target_link_libraries(y2020 INTERFACE
    utilities
    Thrust)
