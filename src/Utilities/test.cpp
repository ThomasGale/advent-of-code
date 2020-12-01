#include "basic_window.h"
#include <iostream>

int main(int argc, char** argv)
{
    std::cout << "Testing ncurses" << "\n";
    aoc::utils::BasicWindow window(100, 100);
    window.SetChar(50, 0, '0');
    window.Update();
    std::cout << "End" << "\n";
}