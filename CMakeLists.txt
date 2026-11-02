cmake_minimum_required(VERSION 3.24)
project(kvpage LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(KVP_BUILD_TESTS "Build GoogleTest unit tests" ON)

# ---------------------------------------------------------------- CUDA (optional)
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)  # RTX 4080 (Ada)
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  find_package(CUDAToolkit REQUIRED)
  set(KVP_HAVE_CUDA ON)
  message(STATUS "kvpage: CUDA ${CMAKE_CUDA_COMPILER_VERSION}, arch ${CMAKE_CUDA_ARCHITECTURES}")
else()
  message(STATUS "kvpage: no CUDA compiler found, building CPU-only targets")
endif()

# ---------------------------------------------------------------- CPU core + simulator
add_library(kvp_core STATIC
  src/core/block_allocator.cpp
  src/core/prefix_cache.cpp
  src/core/kv_manager.cpp
  src/core/paged_manager.cpp
  src/core/contiguous_manager.cpp
  src/core/scheduler.cpp
  src/sim/workload.cpp
  src/sim/simulator.cpp)
target_include_directories(kvp_core PUBLIC src)
target_compile_options(kvp_core PRIVATE -Wall -Wextra)

add_executable(kvsim src/main_sim.cpp)
target_link_libraries(kvsim PRIVATE kvp_core)

# ---------------------------------------------------------------- GPU engine
if(KVP_HAVE_CUDA)
  add_library(kvp_gpu STATIC
    src/cuda/kv_cache.cu
    src/cuda/paged_attention.cu
    src/cuda/kernels.cu
    src/model/weights.cpp
    src/model/llama.cpp
    src/engine/engine.cpp
    src/engine/metrics.cpp)
  target_include_directories(kvp_gpu PUBLIC src)
  target_link_libraries(kvp_gpu PUBLIC kvp_core CUDA::cudart CUDA::cublas)
  target_compile_options(kvp_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --expt-relaxed-constexpr -Xptxas=-v>)
  set_target_properties(kvp_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

  add_executable(kvbench src/main_bench.cpp)
  target_link_libraries(kvbench PRIVATE kvp_gpu)
  add_executable(kvcorrect src/main_correct.cpp)
  target_link_libraries(kvcorrect PRIVATE kvp_gpu)
  add_executable(kvkernel src/main_kernel_bench.cpp)
  target_link_libraries(kvkernel PRIVATE kvp_gpu)
endif()

# ---------------------------------------------------------------- tests
if(KVP_BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  enable_testing()
  include(GoogleTest)

  add_executable(test_core
    tests/test_block_allocator.cpp
    tests/test_managers.cpp
    tests/test_prefix_cache.cpp
    tests/test_scheduler.cpp)
  target_link_libraries(test_core PRIVATE kvp_core GTest::gtest_main)
  gtest_discover_tests(test_core)

  if(KVP_HAVE_CUDA)
    add_executable(test_cuda tests/test_paged_attention.cu tests/test_kernels.cu)
    target_link_libraries(test_cuda PRIVATE kvp_gpu GTest::gtest_main)
    gtest_discover_tests(test_cuda DISCOVERY_TIMEOUT 60)
  endif()
endif()
