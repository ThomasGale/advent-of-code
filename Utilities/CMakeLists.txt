cmake_minimum_required (VERSION 3.14)
set(CMAKE_CXX_STANDARD 17)

#file(GLOB NCurses_SRC
#"ncurses/pdcurses/*.c"
#"PDCurses/x11/*.c"
#)

#add_library(PDCurses
#STATIC
#"PDCurses/curspriv.h"
#${PDCurses_SRC}
#)

#target_include_directories(PDCurses 
#PUBLIC#
	#"PDCurses/"
#	"/usr/include/X11"
#)

#target_link_libraries(PDCurses
#PRIVATE
#	Xaw Xmu Xt X11 SM ICE Xext Xpm
#)

add_library (Utilities
STATIC
"reader.h" "reader.cpp"
"utils.hpp"
"basic_window.h" "basic_window.cpp"
"vec2.h" 
"vec3.h"
"mat2x2.h"
)

find_package(Curses REQUIRED)
target_include_directories(Utilities
PUBLIC ${CURSES_INCLUDE_DIR}
)

target_link_libraries(Utilities
${CURSES_LIBRARIES}
)

set_target_properties(Utilities PROPERTIES LINKER_LANGUAGE CXX)

add_executable(TestCurses
test.cpp
)

target_link_libraries(TestCurses
Utilities
)