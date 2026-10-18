/** \file dfe.cpp
 *  \brief Implementation of DFE class.
 *
 * Original author: David Banas <br>
 * Original date:   June 2, 2015
 *
 * Copyright (c) 2015 David Banas; all rights reserved World wide.
 */

#include <cmath>
#include <vector>
#include <string>
#include <sstream>
#include "include/dfe.h"
#include "include/util.h"

#define CLKS_PER_ADAPTATION 1

extern std::ostringstream debug_stream;

/// Constructor
DFE::DFE(double slicer_output_mag, double error_gain, int adapt_mode,
    double sample_interval, double clock_per,
    const std::vector<double>& tap_weights) {
    // Capture the incoming configuration variables.
    slicer_mag_ = slicer_output_mag;
    err_gain_ = error_gain;
    mode = adapt_mode;
    tap_weights_ = tap_weights;
    // Initialize our internal state.
    clock_pers_.clear();
    weights_.clear();
    clk_cntr_ = 0L;
    cdr_locked_ = true;
    sim_time_ = 0.0;
    next_ui_edge_ = 0.0;
    backward_filter_output_ = 0.0;
    next_backward_filter_output_ = 0.0;
    edge_sample_ = 0.0;
    last_summer_output_ = 0.0;
    sample_interval_ = sample_interval;
    clock_per_ = clock_per;
    next_clock_ = clock_per / 2.0;
    last_clk_sample_ = 0.0;
    backward_filter_ = new FIRFilter(tap_weights);
}

/// Filter application.
bool DFE::apply(double *sig, const long len, double *clock_times) {
    for (auto i = 0; i < len; i++) {
        double summer_output = sig[i] - backward_filter_output_;
        // Test if we're at a UI boundary, and act accordingly.
        if (sim_time_ >= next_ui_edge_) {
            backward_filter_output_ = next_backward_filter_output_;
            summer_output = sig[i] - backward_filter_output_;
            edge_sample_ = interp(last_summer_output_, summer_output, next_ui_edge_, sim_time_, sample_interval_);
            next_ui_edge_ += clock_per_;
        }
        // Test if we're at a clock instant, and act accordingly.
        if (sim_time_ >= next_clock_) {
            *clock_times++ = next_clock_ - clock_per_ / 2.0;  // IBIS-AMI clocks are edge-aligned.
            double clk_sample = interp(last_summer_output_, summer_output, next_clock_, sim_time_, sample_interval_);
            // clock_per_ = cdr_.adjust(last_clk_sample_, edge_sample_, clk_sample);
            clock_pers_.push_back(clock_per_);
            next_ui_edge_ = next_clock_ + clock_per_ / 2.0;
            next_clock_ += clock_per_;
            double slicer_out = slicer_mag_ * sgn(clk_sample);
            if ((mode > 1) && cdr_locked_ && !(clk_cntr_ % CLKS_PER_ADAPTATION)) {
                double err = clk_sample - slicer_out;
                std::vector<double> weights = backward_filter_->get_weights();
                std::vector<double> values = backward_filter_->get_values();
                int j = 0;
                std::vector<double> new_weights;
                for (auto weight : weights) {
                    double new_weight = weight + err * err_gain_ * values[j];
                    if (new_weight < min_weights_[j])
                        new_weight = min_weights_[j];
                    if (new_weight > max_weights_[j])
                        new_weight = max_weights_[j];
                    new_weights.push_back(new_weight);
                    j++;
                }
                backward_filter_->set_weights(new_weights);
                weights_.push_back(new_weights);
            }
            next_backward_filter_output_ = backward_filter_->step(slicer_out);
            last_clk_sample_ = clk_sample;
            clk_cntr_++;
        }
        sim_time_ += sample_interval_;
        last_summer_output_ = summer_output;
        sig[i] = summer_output;
    }

    // Report on state of adaptation convergence, if appropriate.
    // if (mode > 1) {
    if (0) {
        // Divide the captured weights vector into 10 segments, for slope comparisons.
        int slope_meas_interval = weights_.size() / 10;
        double first_slope, last_slope;
        for (auto i = 0; i < weights_[0].size(); i++) {
            first_slope = std::abs(weights_[slope_meas_interval][i] - weights_[0][i]);
            last_slope = std::abs(weights_.back()[i] - weights_[weights_.size() - slope_meas_interval][i]);
            if (last_slope > 0.1 * first_slope)
                return false;  // We've not, yet, converged.
        }
        return true;
    } else
        return true;
}

