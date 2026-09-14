// drpai_runtime_pybind.cpp
// Copyright (C) 2024-2026 Collabora Ltd. — LGPL (see COPYING).
//
// pybind11 binding around the Renesas DRP-AI TVM runtime
// (MeraDrpRuntimeWrapper, powered by EdgeCortix MERA(TM)) for RZ/V2H.
//
// Exposes a minimal `drpai_runtime.Runtime` class to Python so the pure-Python
// `drpai_engine.py` can drive the DRP-AI NPU:
//
//     import drpai_runtime
//     rt = drpai_runtime.Runtime()
//     rt.load("/path/to/deploy_dir")     # deploy.so/json/params
//     rt.set_input(0, nchw_float32_numpy)
//     rt.run()
//     out0 = rt.get_output(0)            # numpy (float32, fp16 upcast)
//
// Build with CMake against the board's DRP-AI TVM runtime — see CMakeLists.txt
// and README.md. This compiles only inside the RZ/V2H DRP-AI TVM SDK and runs
// only on the board (it talks to /dev/drpai0).

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/drpai.h>

#include "MeraDrpRuntimeWrapper.h"

namespace py = pybind11;

static float fp16_to_fp32(uint16_t h) {
  uint32_t sign = static_cast<uint32_t>(h & 0x8000) << 16;
  uint32_t exp = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x3FF;
  uint32_t f;
  if (exp == 0) {
    if (mant == 0) {
      f = sign;
    } else {
      exp = 127 - 15 + 1;
      while ((mant & 0x400) == 0) {
        mant <<= 1;
        exp--;
      }
      mant &= 0x3FF;
      f = sign | (exp << 23) | (mant << 13);
    }
  } else if (exp == 0x1F) {
    f = sign | 0x7F800000 | (mant << 13);  // Inf / NaN
  } else {
    f = sign | ((exp - 15 + 127) << 23) | (mant << 13);
  }
  float out;
  std::memcpy(&out, &f, sizeof(out));
  return out;
}

static uint64_t get_drpai_start_addr() {
  int fd = open("/dev/drpai0", O_RDWR);
  if (fd < 0) {
    throw std::runtime_error("Failed to open /dev/drpai0 (run on the board, as root?)");
  }
  drpai_data_t drpai_data;
  int ret = ioctl(fd, DRPAI_GET_DRPAI_AREA, &drpai_data);
  close(fd);
  if (ret == -1) {
    throw std::runtime_error("ioctl(DRPAI_GET_DRPAI_AREA) failed");
  }
  return drpai_data.address;
}

class Runtime {
 public:
  Runtime() : rt_() {}

  bool load(const std::string& model_dir) {
    model_dir_ = model_dir;
    return rt_.LoadModel(model_dir, get_drpai_start_addr());
  }

  void set_input(int index,
                 py::array_t<float, py::array::c_style | py::array::forcecast> data) {
    rt_.SetInput(index, static_cast<const float*>(data.data()));
  }

  void run() { rt_.Run(); }

  int num_input() { return rt_.GetNumInput(model_dir_); }
  int num_output() { return rt_.GetNumOutput(); }

  py::array get_output(int index) {
    auto out = rt_.GetOutput(index);
    InOutDataType dtype = std::get<0>(out);
    const void* ptr = std::get<1>(out);
    int64_t size = std::get<2>(out);

    switch (dtype) {
      case InOutDataType::FLOAT16: {
        const uint16_t* src = reinterpret_cast<const uint16_t*>(ptr);
        py::array_t<float> result(size);
        float* dst = static_cast<float*>(result.request().ptr);
        for (int64_t i = 0; i < size; ++i) dst[i] = fp16_to_fp32(src[i]);
        return result;
      }
      case InOutDataType::FLOAT32: {
        py::array_t<float> result(size);
        std::memcpy(result.request().ptr, ptr, size * sizeof(float));
        return result;
      }
      case InOutDataType::INT32: {
        py::array_t<int32_t> result(size);
        std::memcpy(result.request().ptr, ptr, size * sizeof(int32_t));
        return result;
      }
      case InOutDataType::INT64: {
        py::array_t<int64_t> result(size);
        std::memcpy(result.request().ptr, ptr, size * sizeof(int64_t));
        return result;
      }
      default:
        throw std::runtime_error("Unsupported DRP-AI output data type");
    }
  }

 private:
  MeraDrpRuntimeWrapper rt_;
  std::string model_dir_;
};

PYBIND11_MODULE(drpai_runtime, m) {
  m.doc() = "pybind11 binding for the Renesas DRP-AI TVM runtime (RZ/V2H)";
  py::class_<Runtime>(m, "Runtime")
      .def(py::init<>())
      .def("load", &Runtime::load, py::arg("model_dir"),
           "Load a DRP-AI TVM deploy directory (deploy.so/json/params).")
      .def("set_input", &Runtime::set_input, py::arg("index"), py::arg("data"))
      .def("run", &Runtime::run)
      .def("num_input", &Runtime::num_input)
      .def("num_output", &Runtime::num_output)
      .def("get_output", &Runtime::get_output, py::arg("index"));
}
