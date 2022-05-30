cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(GC)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CUDA_NVCC_FLAGS "-allow-unsupported-compiler")

file(GLOB_RECURSE CPP_FILES source/*.cpp)
file(GLOB_RECURSE H_FILES source/*.h)
file(GLOB_RECURSE CU_FILES source/*.cu)
file(GLOB_RECURSE CUH_FILES source/*.cuh)

function(assign_source_group)
    foreach(_source IN ITEMS ${ARGN})
        if (IS_ABSOLUTE "${_source}")
            file(RELATIVE_PATH _source_rel "${CMAKE_CURRENT_SOURCE_DIR}" "${_source}")
        else()
            set(_source_rel "${_source}")
        endif()
        get_filename_component(_source_path "${_source_rel}" PATH)
        string(REPLACE "/" "\\" _source_path_msvc "${_source_path}")
        source_group("${_source_path_msvc}" FILES "${_source}")
    endforeach()
endfunction(assign_source_group)

function(my_add_executable)
    foreach(_source IN ITEMS ${ARGN})
        assign_source_group(${_source})
    endforeach()
    cuda_add_executable(${ARGV})
endfunction(my_add_executable)

my_add_executable(GC ${CPP_FILES} ${H_FILES} ${CU_FILES} ${CUH_FILES})
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT GC)
target_link_libraries(GC "${TORCH_LIBRARIES}/*")
set_property(TARGET GC PROPERTY CXX_STANDARD 14)
include_directories("C:/ProgramData/Miniconda3/envs/gc/include")
target_link_libraries(GC "C:/ProgramData/Miniconda3/envs/gc/libs/*")

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET GC
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:GC>)
endif (MSVC)