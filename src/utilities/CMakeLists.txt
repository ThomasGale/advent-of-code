find_package(Curses REQUIRED)

add_library (utilities
"reader.cpp"
"BasicWindow.cpp")

target_include_directories(utilities PUBLIC
    .
    ${CURSES_INCLUDE_DIR})

target_link_libraries(utilities PUBLIC
    ${CURSES_LIBRARIES})
