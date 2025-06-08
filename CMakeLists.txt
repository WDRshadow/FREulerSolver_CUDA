cmake_minimum_required(VERSION 3.31)
project(FR2DEulerSolver LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include_directories(include)
file(GLOB SRC_FILES src/*.cu src/*.cpp)

add_library(fr STATIC ${SRC_FILES})

add_executable(FR2DEulerSolver main.cpp)
target_link_libraries(FR2DEulerSolver fr)

enable_testing()
add_subdirectory(test)
