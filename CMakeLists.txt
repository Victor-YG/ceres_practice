cmake_minimum_required(VERSION 3.5)

project(ceres_practice)

find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED COMPONENTS SuiteSparse EigenSparse)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(./src)

# denoising
add_executable(denoising ./src/denoising.cc ./src/fields_of_experts.cc)
target_link_libraries(denoising Ceres::ceres)

# diffusion
add_executable(diffusion ./src/diffusion.cc)
target_link_libraries(diffusion Ceres::ceres ${OpenCV_LIBS})

# completion
add_executable(completion ./completion.cc)
target_link_libraries(completion Ceres::ceres ${OpenCV_LIBS})

# refinement
add_executable(refinement ./src/refinement.cc)
target_link_libraries(refinement Ceres::ceres ${OpenCV_LIBS})
