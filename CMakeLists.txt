cmake_minimum_required(VERSION 3.31)
project(FR2DEulerSolver LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "CMake is in Debug mode — enabling CUDA debug flags (-G -g -O0)")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g -O0")
endif()

include_directories(include)
file(GLOB SRC_FILES src/*.cu src/*.cpp)

add_library(fr STATIC ${SRC_FILES})

add_executable(FR2DEulerSolver main.cpp)
target_link_libraries(FR2DEulerSolver fr)

enable_testing()
add_subdirectory(test)
