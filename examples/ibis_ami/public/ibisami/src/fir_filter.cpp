/** \file fir_filter.cpp
 *  \brief Implementation of FIRFilter class.
 *
 * Original author: David Banas <br>
 * Original date:   June 8, 2015
 *
 * Copyright (c) 2015 David Banas; all rights reserved World wide.
 */

#include <vector>
#include <algorithm>
#include "include/fir_filter.h"

/// Constructor
/**
 * \param weights A vector of doubles containing the desired filter tap weights.
 * \param oversample_factor An integer giving the number of signal vector elements per weight.
 */
FIRFilter::FIRFilter(const std::vector<double>& weights, int oversample_factor) {
    weights_ = weights;
    oversample_factor_ = oversample_factor;
    delay_chain_ = std::vector<double>(weights.size() * oversample_factor, 0.0);
}

/// Filter application
/**
 * \param sig A pointer to the vector of doubles to be processed, in place.
 * \param len The number of samples to process.
 */
void FIRFilter::apply(double *sig, const long len) {
    std::vector<double> tmp_weights = weights_;
    std::reverse(tmp_weights.begin(), tmp_weights.end());
    for (auto i = 0; i < len; i++) {
        // Shift next sample into delay chain.
        std::rotate(delay_chain_.begin(), delay_chain_.begin() + 1, delay_chain_.end());
        delay_chain_.pop_back();
        delay_chain_.push_back(sig[i]);
        // Calculate next output.
        double accum = 0.0;
        int j = 0;
        for (auto weight : tmp_weights) {
            accum += weight * delay_chain_[j];
            j += oversample_factor_;
        }
        sig[i] = accum;
    }
}

/// Filter single stepping
/**
 * \param x The input value to use, when stepping the filter.
 * \return The next output from the filter, after single stepping.
 */
double FIRFilter::step(double x) {
    apply(&x, 1L);
    return x;
}

std::vector<double> FIRFilter::get_weights() {
    return weights_;
}

std::vector<double> FIRFilter::get_values() {
    std::vector<double> tmp_values = delay_chain_;
    std::reverse(tmp_values.begin(), tmp_values.end());
    return tmp_values;
}

