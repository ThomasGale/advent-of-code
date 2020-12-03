file(GLOB INPUTS inputs/*.in)
file(COPY ${INPUTS} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

add_executable (main
    Main.cpp)

target_link_libraries(main
    utilities
    #y2018
    #y2019
    y2020)

# For CUDA 2020.
set_source_files_properties(Main.cpp PROPERTIES LANGUAGE CUDA)
# For CUDA dynamic parallelism
set_target_properties(main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
