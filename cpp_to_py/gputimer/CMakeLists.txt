# Files
file(GLOB_RECURSE SRC_FILES_GT ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
file(GLOB_RECURSE SRC_FILES_GT_CUDA ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)

# OpenMP
find_package(OpenMP REQUIRED)


# CUDA/CPP GGR Kernel
add_library(gt STATIC ${CMAKE_CURRENT_SOURCE_DIR}/../io_parser/gp/GPDatabase.cpp
                       ${SRC_FILES_GT}
                       ${SRC_FILES_GT_CUDA})

set_target_properties(gt PROPERTIES
                       CUDA_RESOLVE_DEVICE_SYMBOLS ON
                       POSITION_INDEPENDENT_CODE ON)

target_include_directories(gt PRIVATE ${PROJECT_SOURCE_DIR}/cpp_to_py ${TORCH_INCLUDE_DIRS} ${FLUTE_INCLUDE_DIR})
target_link_libraries(gt PRIVATE torch ${TORCH_PYTHON_LIBRARY} xplace_common io_parser OpenMP::OpenMP_CXX flute stdc++fs pthread)
target_compile_options(gt PRIVATE -fPIC)
target_compile_options(gt PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>")

pybind11_add_module(gputimer MODULE PyBindCppMain.cpp)

target_include_directories(gputimer PRIVATE ${TORCH_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/cpp_to_py ${FLUTE_INCLUDE_DIR})
target_link_libraries(gputimer PRIVATE torch ${TORCH_PYTHON_LIBRARY} xplace_common flute io_parser gt)
target_compile_definitions(gputimer PRIVATE 
                            TORCH_EXTENSION_NAME=gpugr
                            TORCH_VERSION_MAJOR=${TORCH_VERSION_MAJOR}
                            TORCH_VERSION_MINOR=${TORCH_VERSION_MINOR}
                            ENABLE_CUDA=${TORCH_ENABLE_CUDA})

install(TARGETS gputimer DESTINATION ${XPLACE_LIB_DIR})