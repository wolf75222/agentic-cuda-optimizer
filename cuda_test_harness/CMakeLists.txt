cmake_minimum_required(VERSION 3.24)

project(cuda_test_harness LANGUAGES CXX)
set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT cuda_test_harness)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR "Use an out-of-source build: cmake -S . -B build")
endif()

file(GLOB_RECURSE HARNESS_FILES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cuh"
)

# Do not collect compiler probes or other generated sources.
set(HARNESS_SOURCES)
set(HARNESS_HEADERS)
foreach(FILE_PATH IN LISTS HARNESS_FILES)
    cmake_path(IS_PREFIX CMAKE_BINARY_DIR "${FILE_PATH}" NORMALIZE IN_BUILD_DIR)
    if(IN_BUILD_DIR OR FILE_PATH MATCHES "/(build[^/]*|cmake-build[^/]*|out|CMakeFiles)/")
        continue()
    endif()
    if(FILE_PATH MATCHES "\\.(cpp|cu)$")
        list(APPEND HARNESS_SOURCES "${FILE_PATH}")
    else()
        list(APPEND HARNESS_HEADERS "${FILE_PATH}")
    endif()
endforeach()

if(HARNESS_SOURCES)
    add_executable(cuda_test_harness ${HARNESS_SOURCES} ${HARNESS_HEADERS})
    target_include_directories(cuda_test_harness PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
    target_link_libraries(cuda_test_harness PRIVATE CUDA::cuda_driver CUDA::nvrtc)
    list(GET CUDAToolkit_INCLUDE_DIRS 0 NVRTC_INCLUDE_DIR)
    target_compile_definitions(cuda_test_harness PRIVATE
        CUDA_HARNESS_INCLUDE_DIR="${NVRTC_INCLUDE_DIR}")
else()
    # Keep configuration working while the harness contains only headers.
    add_library(cuda_test_harness INTERFACE ${HARNESS_HEADERS})
    target_include_directories(cuda_test_harness INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
    target_link_libraries(cuda_test_harness INTERFACE CUDA::cuda_driver CUDA::nvrtc)
    message(STATUS "Add a .cpp file with main() to create the harness executable.")
endif()

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${HARNESS_SOURCES} ${HARNESS_HEADERS})


# NVRTC and its builtins must be available when launched outside a developer shell.
if(WIN32 AND HARNESS_SOURCES)
    file(GLOB NVRTC_DLLS "${CUDAToolkit_BIN_DIR}/nvrtc*.dll" "${CUDAToolkit_BIN_DIR}/x64/nvrtc*.dll")
    foreach(NVRTC_DLL IN LISTS NVRTC_DLLS)
        add_custom_command(TARGET cuda_test_harness POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${NVRTC_DLL}" "$<TARGET_FILE_DIR:cuda_test_harness>"
            VERBATIM)
    endforeach()
endif()
