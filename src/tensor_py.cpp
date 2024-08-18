#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "tensor.h"
#include <sstream>
#include <memory>
#include <pybind11/operators.h>

namespace py = pybind11;

PYBIND11_MODULE(tensor_module, m) {
    py::enum_<DType>(m, "DType")
        .value("FLOAT16", DType::FLOAT16)
        .value("FLOAT32", DType::FLOAT32)
        .value("INT8", DType::INT8)
        .value("INT32", DType::INT32)
        .value("UINT8", DType::UINT8)
        .value("UINT32", DType::UINT32)
        .export_values();

    py::enum_<Device>(m, "Device")
        .value("CUDA", Device::CUDA)
        .value("CPU", Device::CPU)
        .export_values();

    py::class_<Tensor<FLOAT32>, std::shared_ptr<Tensor<FLOAT32>>>(m, "TensorFLOAT32")
        .def(py::init<>())
        .def(py::init<const std::vector<int>&>())
        .def(py::init<std::vector<float>&, std::vector<int>&>()) 
        .def_static("ones", &Tensor<FLOAT32>::ones)
        .def_static("zeros", &Tensor<FLOAT32>::zeros)
        .def_static("rand", &Tensor<FLOAT32>::rand)
        .def("get", &Tensor<FLOAT32>::get)
        .def("set", &Tensor<FLOAT32>::set)
        .def("reshape", &Tensor<FLOAT32>::reshape)
        .def("get_children_size", &Tensor<FLOAT32>::get_children_size)
        .def("get_children", &Tensor<FLOAT32>::get_children)
        .def("set_children", &Tensor<FLOAT32>::set_children)
        .def("get_device", &Tensor<FLOAT32>::get_device)
        .def("change_device", &Tensor<FLOAT32>::change_device)
        .def("size", &Tensor<FLOAT32>::size)
        .def("data", &Tensor<FLOAT32>::data)
        .def("data_set", &Tensor<FLOAT32>::data_set)
        .def("matmul", &matmul<FLOAT32>)
        .def("__add__", [](const Tensor<FLOAT32>& lhs, const Tensor<FLOAT32>& rhs) { return lhs + rhs; })
        .def("__sub__", [](const Tensor<FLOAT32>& lhs, const Tensor<FLOAT32>& rhs) { return lhs - rhs; })
        .def("__mul__", [](const Tensor<FLOAT32>& lhs, const Tensor<FLOAT32>& rhs) { return lhs * rhs; })
        //.def("hstack", static_cast<Tensor<FLOAT32> (*)(const Tensor<FLOAT32>&, const Tensor<FLOAT32>&)>(&hstack<FLOAT32>)) 
        //.def("vstack", static_cast<Tensor<FLOAT32> (*)(const Tensor<FLOAT32>&, const Tensor<FLOAT32>&)>(&vstack<FLOAT32>)) 
        .def("__str__", [](const Tensor<FLOAT32> &tensor) {
            std::ostringstream oss;
            oss << tensor;
            return oss.str();
        });
}
