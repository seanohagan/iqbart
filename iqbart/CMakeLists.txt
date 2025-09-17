cmake_minimum_required(VERSION 3.12)

project(iqbart_cpp)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# add_subdirectory(pybind11)
find_package(pybind11 REQUIRED)

pybind11_add_module(iqbart_cpp
    iqbart_py.cpp
    iqbart.cpp
    cppmonbart.cpp
    funs.cpp
    bd.cpp
    tree.cpp
    gig.cpp
)
