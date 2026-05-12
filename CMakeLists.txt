cmake_minimum_required(VERSION 3.16)
project(SUS2_SH VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(sus2-sh src/sus2_sh.cpp)
target_compile_options(sus2-sh PRIVATE -O3 -Wall -Wextra -Wpedantic)
