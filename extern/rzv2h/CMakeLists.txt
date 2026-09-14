# Build the `drpai_runtime` Python extension for RZ/V2H.
#
# This mirrors the SDK's apps/CMakeLists.txt (same TVM includes, same V2H
# runtime libraries) but produces a Python module instead of an executable.
# It MUST be configured with the SDK cross-toolchain and built inside the
# RZ/V2H DRP-AI TVM SDK Docker. See README.md.
#
# Required env: TVM_ROOT  (root of rzv_drp-ai_tvm), SDK (Yocto cross SDK)
# Required -D : PYBIND11_INCLUDE_DIR, PYTHON_INCLUDE_DIR (target aarch64 python)
cmake_minimum_required(VERSION 3.16)
project(drpai_runtime CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ENV{TVM_ROOT})
  message(FATAL_ERROR "TVM_ROOT not set — source the DRP-AI TVM SDK env first")
endif()
set(TVM_ROOT "$ENV{TVM_ROOT}")

set(DRPAI_APPS "${TVM_ROOT}/apps" CACHE PATH "rzv_drp-ai_tvm/apps directory")
set(PYBIND11_INCLUDE_DIR "" CACHE PATH "pybind11 include directory")
set(PYTHON_INCLUDE_DIR "" CACHE PATH "target python3 include dir")
set(LIBMERA_RT_PATH ${TVM_ROOT}/obj/build_runtime/v2h/lib)

add_library(drpai_runtime MODULE
  drpai_runtime_pybind.cpp
  ${DRPAI_APPS}/MeraDrpRuntimeWrapper.cpp
)
set_target_properties(drpai_runtime PROPERTIES PREFIX "" SUFFIX ".so")

target_include_directories(drpai_runtime PRIVATE
  ${DRPAI_APPS}
  ${TVM_ROOT}/tvm/include
  ${TVM_ROOT}/setup/include
  ${TVM_ROOT}/tvm/3rdparty/dlpack/include
  ${TVM_ROOT}/tvm/3rdparty/dmlc-core/include
  ${TVM_ROOT}/tvm/3rdparty/compiler-rt
  ${PYBIND11_INCLUDE_DIR}
  ${PYTHON_INCLUDE_DIR}
)

add_definitions(-DMERA_DRP_RUNTIME)
target_compile_definitions(drpai_runtime PUBLIC KDLDRPAI)
target_link_directories(drpai_runtime PRIVATE ${LIBMERA_RT_PATH})
target_link_libraries(drpai_runtime PRIVATE
  mera2_runtime
  mera2_plan_io
  drp_tvm_rt
  pthread
)
set_target_properties(drpai_runtime PROPERTIES
  LINK_FLAGS "-Wl,-rpath,${LIBMERA_RT_PATH} -Wl,-rpath-link,${LIBMERA_RT_PATH}")

target_compile_options(drpai_runtime PRIVATE -O3 -mtune=cortex-a55 -Wall -fvisibility=hidden)
