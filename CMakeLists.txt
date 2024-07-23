# =============================================================================
# Copyright (c) 2023-2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

# ------------- configure rapids-cmake --------------#

include(cmake/thirdparty/fetch_rapids.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

# ------------- configure project --------------#

rapids_cuda_init_architectures(conv_cagra_index)

project(conv_cagra_index LANGUAGES CXX CUDA)

# ------------- configure raft -----------------#

rapids_cpm_init()
include(cmake/thirdparty/get_raft.cmake)

# -------------- compile tasks ----------------- #

find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_executable(conv src/conv.cu)
target_include_directories(conv PUBLIC src/anns_dataset/include)
target_link_libraries(conv PRIVATE raft::raft raft::compiled)

add_executable(graph_stats src/graph_stats.cu)
target_include_directories(graph_stats PUBLIC src/anns_dataset/include src/histo/include)

target_link_libraries(graph_stats PRIVATE
  OpenMP::OpenMP_CXX
  )
