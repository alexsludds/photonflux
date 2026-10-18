/** \file fir_filter.h
 *  \brief Interface to FIRFilter class.
 *
 * Original author: David Banas <br>
 * Original date:   June 7, 2015
 *
 * Copyright (c) 2015 David Banas; all rights reserved World wide.
 *
 * This class provides a finite impulse response digital filter with optional over-sampling.
 */

#ifndef INCLUDE_FIR_FILTER_H_
#define INCLUDE_FIR_FILTER_H_

#include <vector>

/// A FIR (finite impulse response) filter implementation, with optional over-sampling.
class FIRFilter {
 public:
    explicit FIRFilter(const std::vector<double>& weights, int oversample_factor=1);
    virtual ~FIRFilter() {}
    void apply(double *sig, const long len);
    double step(double x);
    void set_weights(const std::vector<double>& new_weights) {weights_ = new_weights;}
    std::vector<double> get_weights();
    std::vector<double> get_values();

 protected:
    std::vector<double> weights_, delay_chain_;
    int oversample_factor_;
};

#endif  // INCLUDE_FIR_FILTER_H_

