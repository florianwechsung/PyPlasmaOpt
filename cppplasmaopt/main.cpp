#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "xtensor/xmath.hpp"              // xtensor import for the C++ universal functions
#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"     // Numpy bindings

#include "biot_savart.h"

int add(int i, int j) {
    return i + j;
}

namespace py = pybind11;

PYBIND11_MODULE(cppplasmaopt, m) {
    xt::import_numpy();

    m.def("add", &add, R"pbdoc(
        Add two numbers
        Some other explanation about the add function.
    )pbdoc");

    m.def("biot_savart_B", &biot_savart_B);
    m.def("biot_savart_dB_by_dX", &biot_savart_dB_by_dX);
    m.def("biot_savart_dB_by_dcoilcoeff", &biot_savart_dB_by_dcoilcoeff);
    m.def("biot_savart_dB_by_dcoilcoeff_via_chainrule", &biot_savart_dB_by_dcoilcoeff_via_chainrule);
    m.def("biot_savart_d2B_by_dXdcoilcoeff", &biot_savart_d2B_by_dXdcoilcoeff);
    m.def("biot_savart_d2B_by_dXdX", &biot_savart_d2B_by_dXdX);


#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
