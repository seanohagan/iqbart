#include <vector>
#include <iostream>
#include <future>
#include "cppmonbart.h"
#include "iqbart.h"
#include "rrn.h"

IQBartResults iqbart(
    double* x,
    double* y,
    size_t p,
    size_t n,
    double* xp,
    size_t np,
    double* qp,
    size_t l_qp,
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
    unsigned int seed
) {
    //std::cout << "***** Into main of iqbart *****" << std::endl;

    rrn gen(seed);

    std::vector<double> tau_quantile_vec(n);
    for (size_t i = 0; i < n; ++i) {
        tau_quantile_vec[i] = gen.uniform(0.001, 0.999);
    }
    //std::cout << "Generated " << n << " random tau values for training." << std::endl;


    // --- Step 2: Create the new (p+1) dimensional predictor matrix for training ---
    size_t p_new = p + 1;
    std::vector<double> x_augmented(n * p_new);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < p; ++j) {
            x_augmented[i * p_new + j] = x[i * p + j];
        }
        x_augmented[i * p_new + p] = tau_quantile_vec[i];
    }
    //std::cout << "Augmented training X matrix with new shape (" << n << ", " << p_new << ")." << std::endl;

    // --- Step 3: Construct the augmented test dataset (Cartesian product of xp and qp) ---
    size_t np_new = np * l_qp;
    std::vector<double> xp_augmented(np_new * p_new);
    if (np > 0 && l_qp > 0) {
        for (size_t i = 0; i < np; ++i) { // For each original test point
            for (size_t j = 0; j < l_qp; ++j) { // For each desired quantile
                size_t new_row_idx = i * l_qp + j;
                // Copy original p predictors
                for (size_t k = 0; k < p; ++k) {
                    xp_augmented[new_row_idx * p_new + k] = xp[i * p + k];
                }
                // Add the specific quantile as the last predictor
                xp_augmented[new_row_idx * p_new + p] = qp[j];
            }
        }
        //std::cout << "Created augmented test matrix with shape (" << np_new << ", " << p_new << ")." << std::endl;
    }

    // --- Step 4: Create the new monotone_flags vector ---
    std::vector<bool> monotone_flags(p_new, false);
    monotone_flags[p] = true; // The last predictor (tau) is monotonic
    //std::cout << "Set monotonicity flag for the new tau predictor to true." << std::endl;


    // --- Step 5: Call the core cmonbart function ---
    //std::cout << "\n***** Calling core cmonbart function... *****" << std::endl;
    MonBartResults monbart_results = cmonbart(
        x_augmented.data(), y, p_new, n,
        xp_augmented.data(), np_new,
        tau, nu, lambda, alpha, mybeta, phi,
        nd, burn, m, nm,
        nkeeptrain, nkeeptest, nkeeptestme, nkeeptreedraws,
        printevery,
        monotone_flags,
        tau_quantile_vec,
        data_aug,
        seed
    );

    // --- Step 6: Reshape the results and return the new IQBartResults struct ---
    IQBartResults iq_results;

    // Copy over the training results and tree draws
    iq_results.yhat_train_draws = std::move(monbart_results.yhat_train_draws);
    iq_results.yhat_train_mean = std::move(monbart_results.yhat_train_mean);
    iq_results.tree_draws = std::move(monbart_results.tree_draws);

    // Reshape the posterior mean for test predictions
    if (np > 0 && l_qp > 0 && !monbart_results.yhat_test_mean.empty()) {
        iq_results.yhat_test_mean.resize(np, std::vector<double>(l_qp));
        for (size_t i = 0; i < np; ++i) {
            for (size_t j = 0; j < l_qp; ++j) {
                size_t flat_index = i * l_qp + j;
                iq_results.yhat_test_mean[i][j] = monbart_results.yhat_test_mean[flat_index];
            }
        }
    }

    // Reshape the posterior draws for test predictions
    if (np > 0 && l_qp > 0 && !monbart_results.yhat_test_draws.empty()) {
        size_t num_draws = monbart_results.yhat_test_draws.size();
        iq_results.yhat_test_draws.resize(np, std::vector<std::vector<double>>(l_qp, std::vector<double>(num_draws)));

        for (size_t d = 0; d < num_draws; ++d) {
            for (size_t i = 0; i < np; ++i) {
                for (size_t j = 0; j < l_qp; ++j) {
                    size_t flat_index = i * l_qp + j;
                    iq_results.yhat_test_draws[i][j][d] = monbart_results.yhat_test_draws[d][flat_index];
                }
            }
        }
    }

    //std::cout << "***** Reshaped results and returning from iqbart *****" << std::endl;
    return iq_results;
}

IQBartParResults iqbart_par(
    double* x,
    double* y,
    size_t p,
    size_t n,
    double* xp,
    size_t np,
    double* qp,
    size_t l_qp,
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
    //std::cout << "Starting parallel execution of " << num_chains << " IQ-BART chains." << std::endl;

    std::vector<std::future<IQBartResults>> futures;

    for (size_t i = 0; i < num_chains; ++i) {
        unsigned int chain_seed = seed + 5000 + i;
        size_t chain_printevery = (printevery == 0)? 0 : nd + burn + 1;

        futures.push_back(std::async(std::launch::async, iqbart,
            x, y, p, n, xp, np, qp, l_qp, tau, nu, lambda, alpha, mybeta, phi,
            nd, burn, m, nm, nkeeptrain, nkeeptest, nkeeptestme, nkeeptreedraws,
            chain_printevery, data_aug, chain_seed
        ));
    }

    std::vector<IQBartResults> chain_results;
    for (auto& fut : futures) {
        chain_results.push_back(fut.get());
    }
    //std::cout << "All chains finished." << std::endl;

    return {chain_results, num_chains};

}
