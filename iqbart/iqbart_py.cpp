#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // For automatic conversion of std::vector
#include <pybind11/numpy.h> // For handling NumPy arrays

#include "iqbart.h" // Include your main iqbart header

namespace py = pybind11;

// This is the main wrapper function. It has a Python-friendly interface
// and translates the arguments for the C++ iqbart function.
IQBartParResults iqbart_py_wrapper(
    py::array_t<double, py::array::c_style | py::array::forcecast> x,
    py::array_t<double, py::array::c_style | py::array::forcecast> y,
    py::array_t<double, py::array::c_style | py::array::forcecast> xp,
    py::array_t<double, py::array::c_style | py::array::forcecast> qp,
    double tau,
    double nu,
    double lambda,
    double alpha,
    double mybeta,
    double phi,
    size_t nd,
    size_t burn,
    size_t m,
    size_t nm,
    size_t nkeeptrain,
    size_t nkeeptest,
    size_t nkeeptestme,
    size_t nkeeptreedraws,
    size_t printevery,
    bool data_aug,
    unsigned int seed,
    size_t num_chains
) {
    // 1. Get direct access to the NumPy array buffers.
    // pybind11 will throw an error if the arrays are not the correct type/shape.
    py::buffer_info x_buf = x.request();
    py::buffer_info y_buf = y.request();
    py::buffer_info xp_buf = xp.request();
    py::buffer_info qp_buf = qp.request();

    size_t n = x.shape(0);
    size_t p = x.shape(1);
    size_t np = xp.shape(0);
    size_t l_qp = qp.size();

    double* x_ptr = static_cast<double*>(x_buf.ptr);
    double* y_ptr = static_cast<double*>(y_buf.ptr);
    double* xp_ptr = static_cast<double*>(xp_buf.ptr);
    double* qp_ptr = static_cast<double*>(qp_buf.ptr);

    IQBartParResults results;

    if (num_chains == 1) {
        IQBartResults single_result = iqbart(
            x_ptr, y_ptr, p, n,
            xp_ptr, np,
            qp_ptr, l_qp,
            tau, nu, lambda, alpha, mybeta, phi,
            nd, burn, m, nm,
            nkeeptrain, nkeeptest, nkeeptestme, nkeeptreedraws,
            printevery, data_aug, seed
    );
    results = {{single_result}, 1};
    } else {
        results = iqbart_par(
            x_ptr, y_ptr, p, n,
            xp_ptr, np,
            qp_ptr, l_qp,
            tau, nu, lambda, alpha, mybeta, phi,
            nd, burn, m, nm,
            nkeeptrain, nkeeptest, nkeeptestme, nkeeptreedraws,
            printevery, data_aug, seed, num_chains
        );
    }
    return results;
}


// This is the macro that creates the Python module.
// The first argument is the name of the compiled module file (e.g., iqbart_cpp.so).
// The second argument, 'm', is a variable that represents the module.
PYBIND11_MODULE(iqbart_cpp, m) {
    m.doc() = "Python bindings for the Instrumented Quantile BART C++ implementation";

    // Define the IQBartResults struct so pybind11 knows how to handle it.
    // This creates a Python class that mirrors the C++ struct.
    py::class_<IQBartResults>(m, "IQBartResults")
        .def(py::init<>()) // Default constructor
        // Expose the members as readable properties in Python
        .def_readonly("yhat_test_draws", &IQBartResults::yhat_test_draws)
        .def_readonly("yhat_test_mean", &IQBartResults::yhat_test_mean)
        .def_readonly("yhat_train_draws", &IQBartResults::yhat_train_draws)
        .def_readonly("yhat_train_mean", &IQBartResults::yhat_train_mean);
        // Note: We are not exposing the complex TreeDraws for simplicity.
    py::class_<IQBartParResults>(m, "IQBartParResults")
        .def(py::init<>())
        .def_readonly("chain_results", &IQBartParResults::chain_results)
        .def_readonly("num_chains", &IQBartParResults::num_chains);

    // Define the main function that will be callable from Python.
    m.def("iqbart", &iqbart_py_wrapper, "Run the Instrumented Quantile BART model",
        // Here we define the Python arguments with default values
        py::arg("x"),
        py::arg("y"),
        py::arg("xp"),
        py::arg("qp"),
        py::arg("tau") = 0.95,
        py::arg("nu") = 3.0,
        py::arg("lambda_val") = 0.9,
        py::arg("alpha") = 0.25,
        py::arg("mybeta") = 0.8,
        py::arg("phi") = 1.0,
        py::arg("nd") = 1000,
        py::arg("burn") = 500,
        py::arg("m") = 200,
        py::arg("nm") = 200,
        py::arg("nkeeptrain") = 100,
        py::arg("nkeeptest") = 100,
        py::arg("nkeeptestme") = 100,
        py::arg("nkeeptreedraws") = 0,
        py::arg("printevery") = 100,
        py::arg("data_aug") = true,
        py::arg("seed") = 1u,
        py::arg("num_chains") = 4
    );
}
