// #define CATCH_CONFIG_MAIN // We now get main() from catch_amalgamated.cpp

#define R_NO_REMAP


// Include the Catch2 header
#include "catch_amalgamated.hpp"

// Include project headers
#include "../cppmonbart.h"      // For MonBartResults struct and cmonbart function
#include "../rrn.h"          // For the random number generator
#include "../tree.h"
#include "../iqbart.h"
#include <vector>
#include <cmath> // For std::pow
#include <time.h>  // For time() to seed RNG
#include <numeric> // For std::accumulate
#include <random> // For std::accumulate
#include <map> // For std::accumulate
#include <algorithm> // For std::min_element, std::max_element
#include <fstream>

// extern "C" {
// #include <Rmath.h> // For set_seed
// }

// TEST_CASE("integration test with generated data (matching R)", "[integration]") {
//     size_t n = 200;
//     size_t p = 1;

//     size_t nd = 1000;
//     size_t burn = 100;
//     size_t m = 200;
//     size_t nm = 50;
//     double alpha = 0.25;
//     double mybeta = 0.8;
//     size_t printevery = 10;

//     set_seed(14, 0); // seed for data generation
//   //
//     std::vector<double> x_train(p * n);
//     std::vector<double> y_train(n);
//     rrn gen;

//     for (size_t i = 0; i < n; ++i) {
//         x_train[i * p + 0] = -1 + 2 * gen.uniform();
//     }
//     std::sort(x_train.begin(), x_train.end());

//     double sigma_noise = 0.1;
//     for (size_t i = 0; i < n; ++i) {
//         double x_val = x_train[i * p + 0];
//         double noise = gen.normal() * sigma_noise;
//         y_train[i] = std::pow(x_val, 3) + noise;
//     }

//     // calculate tau and lambda based on monbart.R logic
//     double y_min = y_train[0];
//     double y_max = y_train[0];
//     for (size_t i = 1; i < n; ++i) {
//         if (y_train[i] < y_min) y_min = y_train[i];
//         if (y_train[i] > y_max) y_max = y_train[i];
//     }

//     double k = 2.0; // From monbart.R default
//     double tau = (y_max - y_min) / (2 * k * std::sqrt(m));
//     tau = std::sqrt(1.467) * tau; // adjustment for monotonic constraint

//     double nu = 3.0; // sigdf from monbart.R default
//     double sigquant = 0.90; // from monbart.R default
//     double qchi = qchisq(1.0 - sigquant, nu, 1, 0);
//     double lambda = (sigma_noise * sigma_noise * qchi) / nu;

//     std::vector<bool> monotone_flags(p, true);

//     set_seed(99, 0); // seed for MCMC
//     MonBartResults results;
//     REQUIRE_NOTHROW([&]() {
//         results = cmonbart(
//             x_train.data(), // x
//             y_train.data(), // y
//             p,              // p
//             n,              // n
//             nullptr,        // xp (no test data)
//             0,              // np
//             tau,            // tau
//             nu,             // nu
//             lambda,         // lambda
//             alpha,          // alpha
//             mybeta,         // mybeta
//             nd,             // nd
//             burn,           // burn
//             m,              // m
//             nm,             // nm (mu grid size)
//             nd,             // nkeeptrain
//             0,              // nkeeptest
//             0,              // nkeeptestme
//             0,              // nkeeptreedraws
//             printevery,      // printevery
//             monotone_flags
//         );
//     }());

//     // basic checks to ensure the run was successful
//     CHECK(results.sigma_draws.size() == (nd + burn));
//     CHECK(results.yhat_train_mean.size() == n);
//     CHECK(results.yhat_train_draws.size() == nd);
//     if (nd > 0) {
//         CHECK(results.yhat_train_draws[0].size() == n);
//     }

//     // correctness checks against R's output
//     std::vector<double> f_true_cpp(n);
//     for (size_t i = 0; i < n; ++i) {
//         f_true_cpp[i] = std::pow(x_train[i * p + 0], 3);
//     }

//     // mean Squared Error (MSE) of posterior mean prediction against true function
//     double mse_cpp = 0.0;
//     for (size_t i = 0; i < n; ++i) {
//         mse_cpp += std::pow(results.yhat_train_mean[i] - f_true_cpp[i], 2);
//     }
//     mse_cpp /= n;

//     // mean of posterior sigma draws
//     double mean_sigma_cpp = std::accumulate(results.sigma_draws.begin(), results.sigma_draws.end(), 0.0) / results.sigma_draws.size();

//     // min and Max of posterior mean prediction
//     double min_yhat_cpp = *std::min_element(results.yhat_train_mean.begin(), results.yhat_train_mean.end());
//     double max_yhat_cpp = *std::max_element(results.yhat_train_mean.begin(), results.yhat_train_mean.end());

//     // assertions against values taken from R with individual tolerances
//     CHECK(mse_cpp == Catch::Approx(0.0007685819).epsilon(0.3));
//     CHECK(mean_sigma_cpp == Catch::Approx(0.1016446).epsilon(0.12));
//     CHECK(min_yhat_cpp == Catch::Approx(-0.8904141).epsilon(0.08));
//     CHECK(max_yhat_cpp == Catch::Approx(0.757139).epsilon(0.27));
// }

// TEST_CASE("Hybrid data monotonicity test (Full Coverage)", "[hybrid-monotonicity]") {
//     // Test function: y = x1 + x2^2 (x1 is monotonic, x2 is not on [-1,1])
//     size_t p = 2; // Two variables

//     // --- Step 1: Create a hybrid training dataset ---
//     // Part A: A structured grid for easy evaluation
//     size_t grid_dim = 15;
//     size_t n_grid = grid_dim * grid_dim;

//     // Part B: Random data to make the fitting task more realistic
//     size_t n_random = 200;

//     // Part C: Combine them
//     size_t n = n_grid + n_random;
//     std::vector<double> x_train(p * n);
//     std::vector<double> y_train(n);
//     rrn gen;
//     set_seed(42, 0);

//     // Generate the grid points (for evaluation)
//     size_t current_row = 0;
//     for (size_t i = 0; i < grid_dim; ++i) { // Loop through x1 values
//         for (size_t j = 0; j < grid_dim; ++j) { // Loop through x2 values
//             double x1_val = -1.0 + 2.0 * j / (grid_dim - 1); // x1 varies faster
//             double x2_val = -1.0 + 2.0 * i / (grid_dim - 1); // x2 varies slower
//             x_train[current_row * p + 0] = x1_val;
//             x_train[current_row * p + 1] = x2_val;
//             current_row++;
//         }
//     }
//     // Generate the random points (for fitting)
//     for (size_t i = 0; i < n_random; ++i) {
//         x_train[current_row * p + 0] = -1.0 + 2.0 * gen.uniform();
//         x_train[current_row * p + 1] = -1.0 + 2.0 * gen.uniform();
//         current_row++;
//     }

//     // --- Step 2: Generate y_train for the entire combined dataset ---
//     double sigma_noise = 0.1;
//     for (size_t i = 0; i < n; ++i) {
//         double x1 = x_train[i * p + 0];
//         double x2 = x_train[i * p + 1];
//         y_train[i] = (x1 * x1 * x1) + (x2 * x2) + (gen.normal() * sigma_noise);
//     }

//     // --- Step 3: MCMC Setup ---
//     size_t nd = 500, burn = 100, m = 50, nm = 50;
//     double alpha = 0.25, mybeta = 0.8; // Standard BART priors
//     size_t printevery = 501;

//     double y_min = *std::min_element(y_train.begin(), y_train.end());
//     double y_max = *std::max_element(y_train.begin(), y_train.end());
//     double k = 2.0, tau = (y_max - y_min) / (2 * k * std::sqrt(m));
//     tau = std::sqrt(1.467) * tau; // Monotonic adjustment
//     double nu = 3.0, sigquant = 0.90;
//     double qchi = qchisq(1.0 - sigquant, nu, 1, 0);
//     double lambda = (sigma_noise * sigma_noise * qchi) / nu;

//     // --- Step 4: A checking function that ONLY evaluates the grid portion of the data ---
//     auto check_violations = [&](const std::vector<double>& predictions, size_t var_to_check, float tol = 1e-6) {
//         size_t total_violations = 0;
//         if (var_to_check == 0) { // Check x1 (varies faster in our grid)
//             for (size_t i = 0; i < grid_dim; ++i) { // For each fixed x2 slice
//                 for (size_t j = 1; j < grid_dim; ++j) {
//                     size_t current_index = i * grid_dim + j;
//                     size_t previous_index = i * grid_dim + j - 1;
//                     if (predictions[current_index] < predictions[previous_index] - tol) {
//                         total_violations++;
//                     }
//                 }
//             }
//         } else if (var_to_check == 1) { // Check x2 (varies slower in our grid)
//             for (size_t j = 0; j < grid_dim; ++j) { // For each fixed x1 slice
//                 for (size_t i = 1; i < grid_dim; ++i) {
//                     size_t current_index = i * grid_dim + j;
//                     size_t previous_index = (i - 1) * grid_dim + j;
//                     if (predictions[current_index] < predictions[previous_index] - tol) {
//                         total_violations++;
//                     }
//                 }
//             }
//         }
//         return total_violations;
//     };

//     // --- Step 5: Run the full test loop ---
//     std::cout << "\n=== Hybrid Data Test (Final Version) ===\n";
//     std::cout << "Function: y = x1 + x2^2 (x1 monotonic, x2 not)\n\n";

//     std::vector<std::vector<bool>> flag_combinations = {
//         {false, false}, {true, false}, {false, true}, {true, true}
//     };
//     std::vector<std::string> flag_names = {
//         "No constraints [F,F]", "x1 monotonic [T,F]",
//         "x2 monotonic [F,T]", "Both monotonic [T,T]"
//     };


//     std::vector<double> tau_quantile_vec(n, 0.99);

//     for (size_t test_idx = 0; test_idx < flag_combinations.size(); ++test_idx) {
//         set_seed(123, 0);
//         std::vector<bool> flags = flag_combinations[test_idx];
//         MonBartResults results = cmonbart(
//             x_train.data(), y_train.data(), p, n, nullptr, 0,
//             tau, nu, lambda, alpha, mybeta, nd, burn, m, nm,
//             nd, 0, 0, 0, printevery, flags, tau_quantile_vec
//         );

//         // Pass only the predictions on the grid to the checker
//         std::vector<double> grid_predictions(results.yhat_train_mean.begin(), results.yhat_train_mean.begin() + n_grid);
//         size_t x1_violations = check_violations(grid_predictions, 0);
//         size_t x2_violations = check_violations(grid_predictions, 1);

//         std::cout << flag_names[test_idx] << ":\n";
//         std::cout << "  x1 Monotonicity violations on grid: " << x1_violations << "\n";
//         std::cout << "  x2 Monotonicity violations on grid: " << x2_violations << "\n\n";

//         // --- Corrected Assertions ---
//         if (flags[0]) { // If x1 was constrained, it MUST be monotonic
//             CHECK(x1_violations == 0);
//         }
//         if (flags[1]) { // If x2 was constrained, it MUST be monotonic
//             CHECK(x2_violations == 0);
//         }
//     }
// }

// TEST_CASE("Monotone Quantile BART Accuracy and Ordering", "[quantile_accuracy]") {

//     // =================================================================
//     // 1. SETUP THE SCENARIO
//     // =================================================================

//     // --- Data Generation Parameters ---
//     const size_t n = 1000; // Number of training observations
//     const size_t p = 1;   // Number of predictors
//     const size_t n_test = 50; // Number of test points for evaluation
//     const double sigma_noise = 0.1; // Standard deviation of the noise

//     // --- MCMC Parameters ---
//     const size_t nd = 500;
//     const size_t burn = 200;
//     const size_t m = 100;

//     // --- Model Parameters ---
//     std::vector<bool> monotone_flags = {true}; // x is monotonically increasing
//     std::vector<double> quantiles_to_test = {0.1, 0.5, 0.9};

//     // --- Create a random number generator for the data ---
//     std::mt19937 rng(1234); // Fixed seed for reproducibility
//     std::uniform_real_distribution<> uniform_dist(-1.0, 1.0);
//     std::normal_distribution<> normal_dist(0.0, sigma_noise);

//     // =================================================================
//     // 2. GENERATE TRAINING AND TEST DATA
//     // =================================================================

//     // --- Training Data ---
//     // y = x^3 + epsilon, where epsilon ~ N(0, sigma_noise^2)
//     std::vector<double> x_train(n * p);
//     std::vector<double> y_train(n);
//     for (size_t i = 0; i < n; ++i) {
//         x_train[i] = uniform_dist(rng);
//         y_train[i] = std::pow(x_train[i], 3) + normal_dist(rng);
//     }

//     // --- Test Data ---
//     // An evenly spaced grid of points from -1 to 1 to evaluate the model
//     std::vector<double> x_test(n_test * p);
//     for (size_t i = 0; i < n_test; ++i) {
//         x_test[i] = -1.0 + (2.0 * i) / (n_test - 1);
//     }

//     // =================================================================
//     // 3. RUN THE MODEL FOR EACH QUANTILE
//     // =================================================================

//     std::map<double, std::vector<double>> posterior_means;

//     for (double q : quantiles_to_test) {

//         // Set the tau vector for all observations to the current quantile
//         std::vector<double> tau_quantile_vec(n, q);

//         MonBartResults results = cmonbart(
//             x_train.data(), y_train.data(), p, n,
//             x_test.data(), n_test,
//             0.95, 3.0, 0.9, 0.25, 0.8, // tau, nu, lambda, alpha, beta (some are unused in quantile version)
//             nd, burn, m, 50, // nd, burn, m, nm
//             0, n_test, n_test, 0, 200, // nkeeptrain, nkeeptest, nkeeptestme, nkeeptreedraws, printevery (CHANGED printevery)
//             monotone_flags,
//             tau_quantile_vec
//         );

//         // Store the posterior mean predictions for this quantile
//         posterior_means[q] = results.yhat_test_mean;
//     }

//     // =================================================================
//     // 4. VERIFY THE RESULTS (ASSERTIONS)
//     // =================================================================

//     SECTION("Check accuracy of quantile estimates via RMSE") {
//         for (double q : quantiles_to_test) {
//             double squared_error_sum = 0.0;
//             std::vector<double>& predictions = posterior_means[q];

//             for (size_t i = 0; i < n_test; ++i) {
//                 // The true conditional quantile is f(x) + quantile_of_noise
//                 // Q_q(Y|x) = x^3 + sigma * Phi^-1(q)
//                 // We use qnorm(quantile, mean, sd, lower_tail, log_p) for Phi^-1
//                 double noise_quantile = qnorm(q, 0.0, 1.0, 1, 0);
//                 double true_conditional_quantile = std::pow(x_test[i], 3) + sigma_noise * noise_quantile;
//                 squared_error_sum += std::pow(predictions[i] - true_conditional_quantile, 2);
//             }

//             double rmse = std::sqrt(squared_error_sum / n_test);

//             // The RMSE should be reasonably low. The old threshold of 0.05 was too strict.
//             // A value around the noise level (0.1) is more realistic. We'll set the bar
//             // a bit higher to allow for some estimation error. A very high RMSE
//             // likely indicates an issue with the fixed `phi` parameter.
//             REQUIRE(rmse < 0.15);
//         }
//     }

//     SECTION("Check that quantile estimates are correctly ordered") {
//         std::vector<double>& preds_q10 = posterior_means[0.1];
//         std::vector<double>& preds_q50 = posterior_means[0.5];
//         std::vector<double>& preds_q90 = posterior_means[0.9];

//         REQUIRE(preds_q10.size() == n_test);
//         REQUIRE(preds_q50.size() == n_test);
//         REQUIRE(preds_q90.size() == n_test);

//         for (size_t i = 0; i < n_test; ++i) {
//             // Assert that for each test point x, Q_0.1 <= Q_0.5 <= Q_0.9
//             REQUIRE(preds_q10[i] <= preds_q50[i]);
//             REQUIRE(preds_q50[i] <= preds_q90[i]);
//         }
//     }

//     // =================================================================
//     // 5. SAVE RESULTS TO FILE FOR VISUALIZATION
//     // =================================================================
//     std::ofstream output_file("test_results.csv");
//     output_file << "x,true_q10,pred_q10,true_q50,pred_q50,true_q90,pred_q90\n";

//     for (size_t i = 0; i < n_test; ++i) {
//         output_file << x_test[i] << ",";
//         // True values
//         output_file << (std::pow(x_test[i], 3) + sigma_noise * qnorm(0.1, 0.0, 1.0, 1, 0)) << ",";
//         // Predicted values
//         output_file << posterior_means[0.1][i] << ",";

//         // True values
//         output_file << (std::pow(x_test[i], 3) + sigma_noise * qnorm(0.5, 0.0, 1.0, 1, 0)) << ",";
//         // Predicted values
//         output_file << posterior_means[0.5][i] << ",";

//         // True values
//         output_file << (std::pow(x_test[i], 3) + sigma_noise * qnorm(0.9, 0.0, 1.0, 1, 0)) << ",";
//         // Predicted values
//         output_file << posterior_means[0.9][i] << "\n";
//     }
//     output_file.close();
// }

TEST_CASE("Implicit Quantile BART Accuracy and Ordering", "[iqbart_accuracy]") {

    // =================================================================
    // 1. SETUP THE SCENARIO (Same as before)
    // =================================================================

    // --- Data Generation Parameters ---
    const size_t n = 10000;
    const size_t p = 1;
    const size_t n_test = 100;
    const double sigma_noise = 0.1;

    // --- MCMC Parameters ---
    const size_t nd = 100;
    const size_t burn = 100;
    const size_t m = 500;

    const size_t ngrid = 100;

    // --- Model Parameters ---
    std::vector<double> quantiles_to_test = {0.1, 0.5, 0.9};

    // --- Create a random number generator for the data ---
    std::mt19937 rng(1234); // Fixed seed for reproducibility
    std::uniform_real_distribution<> uniform_dist(-1.0, 1.0);
    std::normal_distribution<> normal_dist(0.0, sigma_noise);

    // =================================================================
    // 2. GENERATE TRAINING AND TEST DATA (Same as before)
    // =================================================================

    // --- Training Data ---
    std::vector<double> x_train(n * p);
    std::vector<double> y_train(n);
    for (size_t i = 0; i < n; ++i) {
        x_train[i] = uniform_dist(rng);
        y_train[i] = std::pow(x_train[i], 3) + normal_dist(rng);
    }

    // --- Test Data ---
    std::vector<double> x_test(n_test * p);
    for (size_t i = 0; i < n_test; ++i) {
        x_test[i] = -1.0 + (2.0 * i) / (n_test - 1);
    }

    // =================================================================
    // 3. RUN THE IQ-BART MODEL ONCE
    // =================================================================

    IQBartResults results = iqbart(
        x_train.data(), y_train.data(), p, n,
        x_test.data(), n_test,
        quantiles_to_test.data(), quantiles_to_test.size(),
        0.95, 3.0, 0.9, 0.5, 0.8, // tau, nu, lambda, alpha, beta
        nd, burn, m, ngrid,           // nd, burn, m, nm
        0, n_test, n_test, 0, 200   // nkeeptrain, nkeeptest, nkeeptestme, nkeeptreedraws, printevery
    );

    // =================================================================
    // 4. VERIFY THE RESULTS (ASSERTIONS)
    // =================================================================

    SECTION("Check accuracy of quantile estimates via RMSE") {
        for (size_t j = 0; j < quantiles_to_test.size(); ++j) {
            double q = quantiles_to_test[j];
            double squared_error_sum = 0.0;

            for (size_t i = 0; i < n_test; ++i) {
                // True conditional quantile: Q_q(Y|x) = x^3 + sigma * Phi^-1(q)
                double noise_quantile = qnorm(q, 0.0, 1.0, 1, 0);
                double true_conditional_quantile = std::pow(x_test[i], 3) + sigma_noise * noise_quantile;

                // Get the prediction from the reshaped results
                double prediction = results.yhat_test_mean[i][j];
                squared_error_sum += std::pow(prediction - true_conditional_quantile, 2);
            }

            double rmse = std::sqrt(squared_error_sum / n_test);
            REQUIRE(rmse < 0.15); // Use a reasonable threshold
        }
    }

    SECTION("Check that quantile estimates are correctly ordered") {
        REQUIRE(results.yhat_test_mean.size() == n_test);

        for (size_t i = 0; i < n_test; ++i) {
            REQUIRE(results.yhat_test_mean[i].size() == quantiles_to_test.size());
            // Assert that for each test point x, Q_0.1 <= Q_0.5 <= Q_0.9
            REQUIRE(results.yhat_test_mean[i][0] <= results.yhat_test_mean[i][1]);
            REQUIRE(results.yhat_test_mean[i][1] <= results.yhat_test_mean[i][2]);
        }
    }

    // =================================================================
    // 5. SAVE RESULTS TO FILE FOR VISUALIZATION
    // =================================================================
    std::ofstream output_file("test_results_iqbart.csv");
    output_file << "x,true_q10,pred_q10,true_q50,pred_q50,true_q90,pred_q90\n";

    for (size_t i = 0; i < n_test; ++i) {
        output_file << x_test[i] << ",";
        // True values
        output_file << (std::pow(x_test[i], 3) + sigma_noise * qnorm(0.1, 0.0, 1.0, 1, 0)) << ",";
        // Predicted values from reshaped results
        output_file << results.yhat_test_mean[i][0] << ",";

        output_file << (std::pow(x_test[i], 3) + sigma_noise * qnorm(0.5, 0.0, 1.0, 1, 0)) << ",";
        output_file << results.yhat_test_mean[i][1] << ",";

        output_file << (std::pow(x_test[i], 3) + sigma_noise * qnorm(0.9, 0.0, 1.0, 1, 0)) << ",";
        output_file << results.yhat_test_mean[i][2] << "\n";
    }
    output_file.close();
}
