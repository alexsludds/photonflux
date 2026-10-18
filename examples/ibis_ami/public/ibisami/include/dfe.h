/** \file dfe.h
 *  \brief Interface to DFE class.
 *
 * Original author: David Banas <br>
 * Original date:   June 4, 2015
 *
 * Copyright (c) 2015 David Banas; all rights reserved World wide.
 *
 * This class provides a decision feedback equalizer (DFE).
 */

#ifndef INCLUDE_DFE_H_
#define INCLUDE_DFE_H_

#include <vector>
#include <string>
#include "include/fir_filter.h"

/// A generic DFE implementation.
class DFE {
 public:
    DFE(double slicer_output_mag, double error_gain, int adapt_mode,
        double sample_interval, double clock_per,
        const std::vector<double>& tap_weights);
    virtual ~DFE() {delete backward_filter_;}
    virtual bool apply(double *sig, const long len, double *clock_times);
    int mode;
    std::vector<double> get_weights() {return backward_filter_->get_weights();}
    void set_weights(std::vector<double> weights) {backward_filter_->set_weights(weights);}
    void set_max_weights(std::vector<double> weights) {max_weights_ = weights;}
    void set_min_weights(std::vector<double> weights) {min_weights_ = weights;}
    double slicer_mag() {return slicer_mag_;}
    const std::vector<std::vector<double>>& adaptation() {return weights_;}

 protected:
    FIRFilter *backward_filter_;
    std::vector<double> clock_pers_, tap_weights_, tap_values_, min_weights_, max_weights_;
    std::vector<std::vector<double>> weights_;
    unsigned long clk_cntr_;
    bool cdr_locked_;
    double sim_time_, next_ui_edge_, backward_filter_output_, edge_sample_,
           last_summer_output_, sample_interval_, clock_per_, next_clock_,
           last_clk_sample_, slicer_mag_, err_gain_, next_backward_filter_output_;
};

#endif  // INCLUDE_DFE_H_

