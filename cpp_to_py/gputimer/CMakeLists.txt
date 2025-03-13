# Files
file(GLOB_RECURSE SRC_FILES_GT ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
file(GLOB_RECURSE SRC_FILES_GT_CUDA ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)

# OpenMP
find_package(OpenMP REQUIRED)

# CUDA GT Kernel
cuda_add_library(gt_cuda_tmp STATIC ${SRC_FILES_GT_CUDA})

set_target_properties(gt_cuda_tmp PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(gt_cuda_tmp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(gt_cuda_tmp PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(gt_cuda_tmp PRIVATE ${PROJECT_SOURCE_DIR}/cpp_to_py ${TORCH_INCLUDE_DIRS} ${FLUTE_INCLUDE_DIR})
target_link_libraries(gt_cuda_tmp torch ${TORCH_PYTHON_LIBRARY} xplace_common io_parser flute)
# target_compile_options(gt_cuda_tmp PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-use_fast_math>") # not work...

# CPU GT object
add_library(gt SHARED ${SRC_FILES_GT} 
                      ${SRC_FILES_GT_CUDA})

target_include_directories(gt PRIVATE ${PROJECT_SOURCE_DIR}/cpp_to_py ${TORCH_INCLUDE_DIRS} ${FLUTE_INCLUDE_DIR})
target_link_libraries(gt PRIVATE torch ${TORCH_PYTHON_LIBRARY} xplace_common io_parser OpenMP::OpenMP_CXX flute stdc++fs gt_cuda_tmp pthread)
target_compile_options(gt PRIVATE -fPIC)

install(TARGETS gt DESTINATION ${XPLACE_LIB_DIR})

# For Pybind
add_pytorch_extension(gputimer PyBindCppMain.cpp
    EXTRA_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/cpp_to_py ${FLUTE_INCLUDE_DIR}
    EXTRA_LINK_LIBRARIES xplace_common flute io_parser gt gt_cuda_tmp)

install(TARGETS gputimer DESTINATION ${XPLACE_LIB_DIR})


################ For debug only ##################

# # Files
# file(GLOB_RECURSE SRC_FILES_GT ${CMAKE_CURRENT_SOURCE_DIR}/db/*.cpp
#                                 ${CMAKE_CURRENT_SOURCE_DIR}/lib/*.cpp
#                                 ${CMAKE_CURRENT_SOURCE_DIR}/core/*.cpp)
# file(GLOB_RECURSE SRC_FILES_GT_CUDA ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)

# # For Pybind
# add_pytorch_extension(gputimer PyBindCppMain.cpp
#     ${SRC_FILES_GT} ${SRC_FILES_GT_CUDA}
#     EXTRA_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/cpp_to_py ${FLUTE_INCLUDE_DIR}
#     EXTRA_LINK_LIBRARIES xplace_common flute stdc++fs io_parser)

# install(TARGETS gputimer DESTINATION ${XPLACE_LIB_DIR})
