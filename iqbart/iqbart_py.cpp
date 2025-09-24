
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // For automatic conversion of std::vector (if any TreeDraws uses it)
#include <pybind11/numpy.h> // For handling NumPy arrays and buffer_info

#include "iqbart.h" // Include your main iqbart header (assumes Tensor<double> everywhere)

namespace py = pybind11;

// Function to expose Tensor<T> as ND NumPy array (zero-copy, with ownership capsule)
// Takes Tensor by move (&&): Moves into unique_ptr for RAII; Python array owns the Tensor, deletes on ref loss.
// Handles 1D/2D/3D+ uniformly via shape/strides (recomputes row-major strides since private).
template <typename T>
inline py::array_t<T> as_pyarray(Tensor<T>&& tensor) {
    const size_t nd = tensor.ndim();
    if (nd == 0 || tensor.empty()) {
        // Empty Tensor: Return empty array (ndim=0 or 1D size=0; NumPy-compatible)
        return py::array_t<T>(0);  // Empty 0D/1D
    }

    // Shape: std::vector<py::ssize_t> (signed for pybind11)
    std::vector<py::ssize_t> py_shape;
    py_shape.reserve(nd);
    for (auto dim : tensor.shape()) {
        py_shape.push_back(static_cast<py::ssize_t>(dim));
    }

    // Strides in BYTES (row-major C-order; recompute to match Tensor logic)
    std::vector<py::ssize_t> py_strides(nd);
    if (nd > 0) {
        py_strides[nd - 1] = sizeof(T);  // Innermost: itemsize
        for (int i = static_cast<int>(nd) - 2; i >= 0; --i) {
            py_strides[i] = py_strides[i + 1] * py_shape[i + 1];
        }
    }

    // Move Tensor into unique_ptr for ownership
    std::unique_ptr<Tensor<T>> tensor_ptr = std::make_unique<Tensor<T>>(std::move(tensor));

    // Buffer info: Borrow contiguous data ptr, with shape/strides
    auto info = py::buffer_info(
        tensor_ptr->data(),                 // Contiguous ptr (borrowed)
        sizeof(T),                          // Itemsize
        py::format_descriptor<T>::format(), // Format (e.g., 'd' for double)
        static_cast<py::ssize_t>(nd),       // ndim
        py_shape,                           // Shape vector (pybind11 accepts directly)
        py_strides                          // Strides vector (accepts directly)
    );

    // Capsule: Deletes Tensor (frees data_) when Python array is GC'd (refcount=0)
    auto capsule = py::capsule(static_cast<void*>(tensor_ptr.release()), [](void* p) {
        delete reinterpret_cast<Tensor<T>*>(p);  // Calls ~Tensor (deallocs data_)
    });

    // Return array with buffer + capsule (ND, c_style row-major)
    return py::array_t<T>(info, capsule);
}

// This is the main wrapper function. It has a Python-friendly interface
// and translates the arguments for the C++ iqbart function.
IQBartParResults iqbart_py_wrapper(
    py::array_t<double, py::array::c_style | py::array::forcecast> x,
    py::array_t<double, py::array::c_style | py::array::forcecast> y,
    py::array_t<double, py::array::c_style | py::array::forcecast> xp_augmented,
    // py::array_t<double, py::array::c_style | py::array::forcecast> qp,
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
    auto x_buf = x.request();
    auto y_buf = y.request();
    auto xp_buf = xp_augmented.request();
    // auto qp_buf = qp.request();

    // Extract dimensions (cast to size_t for C++ params)
    size_t n = static_cast<size_t>(x.shape(0));
    size_t p = static_cast<size_t>(x.shape(1));
    size_t np_val = static_cast<size_t>(xp_augmented.shape(0));
    // size_t l_qp_val = static_cast<size_t>(qp.size());

    // Pointers to contiguous data (zero-copy to C++)
    double* x_ptr = static_cast<double*>(x_buf.ptr);
    double* y_ptr = static_cast<double*>(y_buf.ptr);
    double* xp_ptr = static_cast<double*>(xp_buf.ptr);
    // double* qp_ptr = static_cast<double*>(qp_buf.ptr);

    IQBartParResults results;

    if (num_chains == 1) {
        IQBartResults single_result = iqbart(
            x_ptr, y_ptr, p, n,
            xp_ptr, np_val,
            // qp_ptr, l_qp_val,
            tau, nu, lambda, alpha, mybeta, phi,
            nd, burn, m, nm,
            nkeeptrain, nkeeptest, nkeeptestme, nkeeptreedraws,
            printevery, data_aug, seed
        );
        results.chain_results = {single_result};
        results.num_chains = 1;
    } else {
        results = iqbart_par(
            x_ptr, y_ptr, p, n,
            xp_ptr, np_val,
            // qp_ptr, l_qp_val,
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

    // Helper: Bind Tensor<double> class (expose methods like shape, if needed; mainly for buffer protocol)
    py::class_<Tensor<double>>(m, "Tensor", py::buffer_protocol())
        .def(py::init<std::initializer_list<size_t>>())  // Brace init: Tensor{2,3}
        .def("shape", &Tensor<double>::shape)
        .def("ndim", &Tensor<double>::ndim)
        .def("size", &Tensor<double>::size)
        .def_buffer([](Tensor<double>& self) -> py::buffer_info {
            // Buffer protocol for direct py::array(self.yhat_test_draws) if needed
            const size_t nd = self.ndim();
            if (nd == 0 || self.empty()) return py::buffer_info();

            std::vector<py::ssize_t> py_shape(self.shape().begin(), self.shape().end());
            std::vector<py::ssize_t> py_strides(nd);
            py_strides[nd - 1] = sizeof(double);
            for (int i = static_cast<int>(nd) - 2; i >= 0; --i) {
                py_strides[i] = py_strides[i + 1] * py_shape[i + 1];
            }
            return py::buffer_info(
                self.data(), sizeof(double), py::format_descriptor<double>::format(),
                static_cast<py::ssize_t>(nd), py_shape, py_strides
            );
        });

    // Define IQBartResults: Properties return ND np.arrays (owning the moved Tensor data)
    py::class_<IQBartResults>(m, "IQBartResults")
        .def(py::init<>())  // Default ctor (empty Tensors)
        // yhat_test_draws: 3D Tensor → 3D np.array (np, l_qp, num_draws)
        .def_property_readonly("yhat_test_draws", [](IQBartResults& self) {
            return as_pyarray(std::move(self.yhat_test_draws));
        })
        // yhat_test_mean: 2D Tensor → 2D np.array (np, l_qp)
        .def_property_readonly("yhat_test_mean", [](IQBartResults& self) {
            return as_pyarray(std::move(self.yhat_test_mean));
        })
        // yhat_train_draws: 2D Tensor → 2D np.array (n_draws, n_train)
        .def_property_readonly("yhat_train_draws", [](IQBartResults& self) {
            return as_pyarray(std::move(self.yhat_train_draws));
        })
        // yhat_train_mean: 1D Tensor → 1D np.array (n_train)
        .def_property_readonly("yhat_train_mean", [](IQBartResults& self) {
            return as_pyarray(std::move(self.yhat_train_mean));
        })
        // tree_draws: Stub (expose if it's a simple struct; full binding if complex)
        .def_readonly("tree_draws", &IQBartResults::tree_draws);

    // IQBartParResults: Vector of results (auto-converts to Python list of IQBartResults)
    py::class_<IQBartParResults>(m, "IQBartParResults")
        .def(py::init<>())
        .def_readonly("chain_results", &IQBartParResults::chain_results)  // list[IQBartResults]
        .def_readonly("num_chains", &IQBartParResults::num_chains);

    // Main function: Callable from Python (inputs as NumPy arrays, zero-copy to C++)
    m.def("iqbart", &iqbart_py_wrapper, "Run the Instrumented Quantile BART model",
        py::arg("x"),           // np.array (n, p)
        py::arg("y"),           // np.array (n,)
        py::arg("xp_augmented"),          // np.array (np, p)
        // py::arg("qp"),          // np.array (l_qp,)
        py::arg("tau") = 0.95,
        py::arg("nu") = 3.0,
        py::arg("lambda_val") = 0.9,  // Note: renamed to match arg name
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
