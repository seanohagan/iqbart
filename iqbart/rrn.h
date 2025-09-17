#ifndef RRN_H
#define RRN_H

#include "rn.h"
#include <limits>
#include <random>
#include <chrono>

/**
 * @brief A thread-safe, seedable random number generator class.
 * This class is the primary way to get random numbers for the MCMC sampler.
 */
class rrn: public rn
{
public:
    // Constructor: Initializes the engine with a provided seed.
    // If the seed is 0 (default), it uses a high-resolution clock for a random seed.
    rrn(unsigned int seed = 0) {
        if (seed == 0) {
            seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        }
        engine.seed(seed);
    }

    // Virtual destructor
    virtual ~rrn() {}

    // Methods now use the object's own private engine
    virtual double normal() {
        std::normal_distribution<double> dist(0.0, 1.0);
        return dist(engine);
    }

    // Generates from U(0,1)
    virtual double uniform() {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        return dist(engine);
    }

    // Generates from U(min, max)
    virtual double uniform(double min, double max) {
        std::uniform_real_distribution<double> dist(min, max);
        return dist(engine);
    }

    virtual double gamma(double shape, double scale) {
        if (shape <= 0.0 || scale <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        std::gamma_distribution<double> dist(shape, scale);
        return dist(engine);
    }

    virtual double exp() {
        std::exponential_distribution<double> dist(1.0);
        return dist(engine);
    }

private:
    std::mt19937 engine; // Each rrn object has its own private engine
};


// =============================================================================
// Standalone, Thread-Safe RNG Wrappers
// These functions provide the convenience of the old global functions but are
// now thread-safe because they operate on an explicit generator object.
// =============================================================================

inline double runif(double min, double max, rrn& gen) {
    return gen.uniform(min, max);
}

inline double rgamma(double shape, double scale, rrn& gen) {
    return gen.gamma(shape, scale);
}


// =============================================================================
// Standalone, Deterministic Math Utilities (Inherently Thread-Safe)
// =============================================================================

namespace detail {
    // Private implementation of the inverse error function, needed for qnorm.
    inline double erf_inv(double x) {
        if (x < -1.0 || x > 1.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        if (std::abs(x) == 1.0) {
            return std::copysign(std::numeric_limits<double>::infinity(), x);
        }

        double a[] = {0.886226899, -1.645349621, 0.914624893, -0.140543331};
        double b[] = {-2.118377725, 1.442710462, -0.329097515, 0.012229801};
        double c[] = {-1.970840454, -1.624906493, 3.429567803, 1.641345311};
        double d[] = {3.543889200, 1.637067800};

        double x2 = x * x;

        if (std::abs(x) <= 0.7) {
            double num = a[0] + x2 * (a[1] + x2 * (a[2] + x2 * a[3]));
            double den = 1.0 + x2 * (b[0] + x2 * (b[1] + x2 * (b[2] + x2 * b[3])));
            return x * num / den;
        } else {
            double y = std::sqrt(-std::log((1.0 - std::abs(x)) / 2.0));
            double num = c[0] + y * (c[1] + y * (c[2] + y * c[3]));
            double den = 1.0 + y * (d[0] + y * d[1]);
            return std::copysign(num / den, x);
        }
    }
} // namespace detail


/**
 * @brief Computes the quantile function (inverse CDF) of the normal distribution.
 */
inline double qnorm(double p, double mean = 0.0, double sd = 1.0, int _lt=1, int _lb=0) {
    if (p < 0.0 || p > 1.0 || sd < 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (p == 0.0) {
        return -std::numeric_limits<double>::infinity();
    }
    if (p == 1.0) {
        return std::numeric_limits<double>::infinity();
    }
    if (sd == 0.0) {
        return mean;
    }

    return mean + sd * std::sqrt(2.0) * detail::erf_inv(2.0 * p - 1.0);
}

#endif
