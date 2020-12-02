add_library (utilities
    "Reader.cpp"
    "BasicWindow.cpp"
    "Solution.cpp")

find_package(Curses REQUIRED)

target_include_directories(utilities PUBLIC
    .
    ${CURSES_INCLUDE_DIR})

target_link_libraries(utilities PUBLIC
    ${CURSES_LIBRARIES})
