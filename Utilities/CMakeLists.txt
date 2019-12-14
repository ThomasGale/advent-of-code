cmake_minimum_required (VERSION 3.14)
set(CMAKE_CXX_STANDARD 17)

file(GLOB PDCurses_SRC
"PDCurses/pdcurses/*.c"
"PDCurses/wincon/*.c" # Windows Specific. x11 for Linux.
)

add_library(PDCurses
STATIC
"PDCurses/curspriv.h"
${PDCurses_SRC}
)

target_include_directories(PDCurses 
PUBLIC
	"PDCurses/"
)

add_library (Utilities
STATIC
"reader.h" "reader.cpp"
"utils.hpp"
"basic_window.h" "basic_window.cpp"
)

target_link_libraries(Utilities
PDCurses
)

set_target_properties(Utilities PROPERTIES LINKER_LANGUAGE CXX)