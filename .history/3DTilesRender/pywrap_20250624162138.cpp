// pywrap.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <opencv2/opencv.hpp>
#include "ModelRenderScene.h"

std::string type2emu(int type)
{
    std::string r;

    unsigned char depth = type & CV_MAT_DEPTH_MASK;
    unsigned char chans = 1 + (type >> CV_CN_SHIFT);

    switch (depth)
    {
    case CV_8U:
        r = "8U";
        break;
    case CV_8S:
        r = "8S";
        break;
    case CV_16U:
        r = "16U";
        break;
    case CV_16S:
        r = "16S";
        break;
    case CV_32S:
        r = "32S";
        break;
    case CV_32F:
        r = "32F";
        break;
    case CV_64F:
        r = "64F";
        break;
    default:
        r = "User";
        break;
    }

    r += "C";
    r += (chans + '0');

    return r;
}

std::string type2str(int type)
{
    std::string r;

    unsigned char depth = type & CV_MAT_DEPTH_MASK;

    switch (depth)
    {
    case CV_8U:
        r = pybind11::format_descriptor<unsigned char>::format();
        break;
    case CV_8S:
        r = pybind11::format_descriptor<char>::format();
        break;
    case CV_16U:
        r = pybind11::format_descriptor<unsigned short>::format();
        break;
    case CV_16S:
        r = pybind11::format_descriptor<short>::format();
        break;
    case CV_32S:
        r = pybind11::format_descriptor<int>::format();
        break;
    case CV_32F:
        r = pybind11::format_descriptor<float>::format();
        break;
    case CV_64F:
        r = pybind11::format_descriptor<double>::format();
        break;
    default:
        r = "User";
        break;
    }
    return r;
}

size_t type2size(int type)
{
    size_t size;
    unsigned char depth = type & CV_MAT_DEPTH_MASK;

    switch (depth)
    {
    case CV_8U:
    case CV_8S:
        size = 1;
        break;
    case CV_16U:
    case CV_16S:
        size = 2;
        break;
    case CV_32S:
    case CV_32F:
        size = 4;
        break;
    case CV_64F:
        size = 8;
        break;
    default:
        size = 0;
        break;
    }

    return size;
}

namespace py = pybind11;
constexpr auto byref = py::return_value_policy::reference_internal;
constexpr auto byauto = py::return_value_policy::automatic;

PYBIND11_MODULE(ModelRenderScene, m)
{
    m.doc() = "3DTiles model render module";

    py::class_<ModelRenderScene>(m, "ModelRenderScene")
        .def(py::init<std::string, uint32_t, uint32_t, double, double, double, double>())
        .def("nextFrame", &ModelRenderScene::nextFrame, py::call_guard<py::gil_scoped_release>())
        .def("getColorImageHigh", &ModelRenderScene::getColorImageHigh, byauto)
        .def("getColorImageWidth", &ModelRenderScene::getColorImageWidth, byauto)
        .def("getColorImage", &ModelRenderScene::getColorImage, byref)
        .def("getDepthImage", &ModelRenderScene::getDepthImage, byref)
        .def("saveDepthImage", &ModelRenderScene::saveDepthImage, byauto)
        .def("saveColorImage", &ModelRenderScene::saveColorImage, byauto)
        // .def("updateFOV", &ModelRenderScene::updateFOV, byauto)
        .def("updateViewPoint", &ModelRenderScene::updateViewPoint, byauto);
    pybind11::class_<cv::Mat>(m, "Image", pybind11::buffer_protocol())
        .def_buffer([](cv::Mat &im) -> pybind11::buffer_info
                    { return pybind11::buffer_info(
                          // Pointer to buffer
                          im.data,
                          // Size of one scalar
                          type2size(im.type()),
                          // Python struct-style format descriptor
                          type2str(im.type()),
                          // Number of dimensions
                          3,
                          // Buffer dimensions
                          {im.rows, im.cols, im.channels()},
                          // Strides (in bytes) for each index
                          {
                              type2size(im.type()) * im.channels() * im.cols,
                              type2size(im.type()) * im.channels(),
                              type2size(im.type())}); });
}