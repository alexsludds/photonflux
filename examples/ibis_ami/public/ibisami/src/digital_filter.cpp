/** \file digital_filter.cpp
 *  \brief Implementation of DigitalFilter class.
 *
 * Original author: David Banas <br>
 * Original date:   May 7, 2015
 *
 * Copyright (c) 2015 David Banas; all rights reserved World wide.
 */

#include <vector>
#include "include/digital_filter.h"

/// Constructor
/**
 * \param num A vector of doubles forming the numerator of the filter response.
 * \param den A vector of doubles forming the denominator of the filter response.
 */
DigitalFilter::DigitalFilter(const std::vector<double>& num,
                             const std::vector<double>& den) {
    // Our "apply" function assumes den_[0] (i.e. - a0) = 1,
    // and that num_ and den_ have the same length.
    num_ = num;
    den_ = den;
    auto len = num_.size();
    if (den_.size() > len) {
        len = den_.size();
        num_.resize(len, 0.0);
    } else {
        den_.resize(len, 0.0);
    }
    auto norm = den_[0];
    auto i = 0;
    for (auto dummy : num_) {
        num_[i] /= norm;
        den_[i] /= norm;
        i++;
    }
    // Initialize our internal state.
    state_.clear();
    state_.resize(len, 0.0);
    num_taps_ = len;
}

/// Filter application
/**
 * \param sig A pointer to the vector of doubles to be processed.
 * \param len The number of samples to process.
 */
void DigitalFilter::apply(double *sig, const long len) {
    double accum;

    // This "direct form 2" implementation comes from
    // "Digital Signal Processing" by Mullis & Roberts.
    for (auto i = 0; i < len; i++) {
        // Advance the "w"s.
        for (auto j = (num_taps_ - 1); j > 0; j--)
            state_[j] = state_[j - 1];
        // Calculate the new w[0] value.
        accum = sig[i];
        for (auto j = 1; j < num_taps_; j++)
            accum -= state_[j] * den_[j];
        state_[0] = accum;
        // Calculate the next `y' value.
        accum = 0;
        for (auto j = 0; j < num_taps_; j++)
            accum += state_[j] * num_[j];
        sig[i] = accum;
    }
}

