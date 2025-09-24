#ifndef CPPMONBART_H
#define CPPMONBART_H

#include <vector>
#include <string>
#include "tensor.h"

// Struct to hold tree information
struct TreeDraws {
    std::string trees;
    std::vector<std::vector<double>> cutpoints;
};

// Struct to hold the results
struct MonBartResults {
    Tensor<double> yhat_train_draws;
    Tensor<double> yhat_test_draws;
    Tensor<double> yhat_train_mean;
    Tensor<double> yhat_test_mean;
    TreeDraws tree_draws;
};

// Declaration of the main cmonbart function
MonBartResults cmonbart(
    double* x,
    double* y,
    size_t p,
    size_t n,
    double* xp,
    size_t np,
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
    std::vector<bool>& monotone_flags,
    std::vector<double>& tau_quantile_vec,
    bool data_aug,
    unsigned int seed
);

#endif // CPPMONBART_H
