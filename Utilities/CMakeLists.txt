cmake_minimum_required (VERSION 3.14)
set(CMAKE_CXX_STANDARD 17)

find_package(Curses REQUIRED)

add_library (Utilities
STATIC
"reader.h" "reader.cpp"
"utils.hpp"
"basic_window.h" "basic_window.cpp"
"vec2.h" 
"vec3.h"
"mat2x2.h"
)

target_include_directories(Utilities
PUBLIC ${CURSES_INCLUDE_DIR}
)

target_link_libraries(Utilities
${CURSES_LIBRARIES}
)

set_target_properties(Utilities PROPERTIES LINKER_LANGUAGE CXX)