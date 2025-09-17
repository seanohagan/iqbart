#ifndef GUARD_iqbart_h
#define GUARD_iqbart_h

#include "cppmonbart.h"
#include <vector>

// Define a new results struct for iqbart to handle the 3D output shape
struct IQBartResults {
    // A 3D vector for test draws: [num_test_points][num_quantiles][num_draws]
    std::vector<std::vector<std::vector<double>>> yhat_test_draws;

    // A 2D vector for the posterior mean: [num_test_points][num_quantiles]
    std::vector<std::vector<double>> yhat_test_mean;

    std::vector<std::vector<double>> yhat_train_draws;
    std::vector<double> yhat_train_mean;
    TreeDraws tree_draws;
};

struct IQBartParResults {

    std::vector<IQBartResults> chain_results;
    size_t num_chains;

};


// Wrapper function for Instrumented Quantile BART (iqbart)
// This function treats the quantile level 'tau' as a predictor.
IQBartResults iqbart(
    double* x,
    double* y,
    size_t p,
    size_t n,
    double* xp,
    size_t np,
    double* qp, // Vector of quantiles to predict for the test set
    size_t l_qp, // Length of the qp vector
    double tau, // Note: this tau is for the mu prior, not for quantiles
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
    unsigned int seed
);


IQBartParResults iqbart_par(
    double* x,
    double* y,
    size_t p,
    size_t n,
    double* xp,
    size_t np,
    double* qp, // Vector of quantiles to predict for the test set
    size_t l_qp, // Length of the qp vector
    double tau, // Note: this tau is for the mu prior, not for quantiles
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
);

#endif
