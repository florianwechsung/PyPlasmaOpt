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
        Add two numbers.
        Just here to test that the C++/Python interface works.
    )pbdoc");

    m.def("biot_savart_all",               & biot_savart_all);
    m.def("biot_savart_B_only",            & biot_savart_B_only);
    m.def("biot_savart_by_dcoilcoeff_all", & biot_savart_by_dcoilcoeff_all);
    m.def("biot_savart_by_dcoilcoeff_all_vjp", & biot_savart_by_dcoilcoeff_all_vjp);
    

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
