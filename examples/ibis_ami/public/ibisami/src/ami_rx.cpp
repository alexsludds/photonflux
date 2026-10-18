/** \file ami_rx.cpp
 *  \brief Implementation of AmiRx class.
 *
 * Original author: David Banas <br>
 * Original date:   May 20, 2015
 *
 * Copyright (c) 2015 David Banas; all rights reserved World wide.
 */

#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <memory>
#include "include/ami_rx.h"
#include "include/util.h"

#define MAX_INIT_ADAPT_BITS 10000

extern std::ostringstream debug_stream;

/// Initializer
void AmiRx::init(double *impulse_matrix, const long number_of_rows,
                 const long aggressors, const double sample_interval,
                 const double bit_time, const std::string& AMI_parameters_in) {
    // Let our base class do its thing.
    inherited::init(impulse_matrix, number_of_rows, aggressors, sample_interval, bit_time, AMI_parameters_in);

    // Initialize our own state.
    delete ctle_;
    delete dfe_;
    ctle_ = nullptr;
    dfe_ = nullptr;
    dump_dfe_adaptation_ = 0;
    dump_adaptation_input_ = 0;
    sig_tap_ = 0;
    init_adapt_tap_ = 0;

    // Grab debugging related parameters and configure things accordingly.
    std::vector<std::string> node_names; node_names.clear();
    node_names.push_back("debug");
    node_names.push_back("enable");
    int dbg_enable = get_param_int(node_names, 0);
    node_names.pop_back();
    node_names.push_back("sig_tap");
    int sig_tap = get_param_int(node_names, 0);
    node_names.pop_back();
    node_names.push_back("init_adapt_tap");
    int init_adapt_tap = get_param_int(node_names, 0);
    node_names.pop_back();
    node_names.pop_back();
    if (dbg_enable) {
        sig_tap_ = sig_tap;
        init_adapt_tap_ = init_adapt_tap;
    }
    std::ostringstream msg_stream;
    msg_stream << "Debugging parameters:\n";
    msg_stream << "\tenable: " << dbg_enable << "\n";
    msg_stream << "\tsig_tap: " << sig_tap << "\n";
    msg_stream << "\tinit_adapt_tap: " << init_adapt_tap << "\n";
    msg_ += msg_stream.str();
}

double maxn (double *sig, long len) {
    double current_max = 0.0;

    for (auto ix = 0; ix < len; ix++)
        if (abs(sig[ix]) > current_max)
            current_max = abs(sig[ix]);
    return current_max;
}

/// Override of AMIModel::proc_sig() specific to Rx models.
bool AmiRx::proc_sig(double *sig, long len, double *clock_times) {
    std::ostringstream params;
    params.flush();
    params << "(example_rx\n";

    if (ctle_) {
        ctle_->apply(sig, len);
    }

    bool rslt = true;
    clock_times[0] = -1;  // Flags tool that we haven't populated 'clock_times'.
    if (dfe_) {
        rslt = dfe_->apply(sig, len, clock_times);
        int k = 1;
        for (auto &weight: dfe_->get_weights())
            params << "\t(dfe_tap" << k++ << " " << weight << ")" << std::endl;
    }

    params << ")\n";
    param_str_ = params.str();

    return rslt;
}

/// Override of AMIModel::proc_imp() specific to Rx models.
void AmiRx::proc_imp() {
    std::ostringstream params;
    params.flush();
    params << "(example_rx\n";

    if (sig_tap_ == 1) // Return input unmodified.
        return;

    // Apply our filter.
    if (ctle_)
        ctle_->apply(impulse_matrix_, number_of_rows_);
    if (sig_tap_ == 2) // Return post-CTLE signal.
        return;
    if (dfe_) {
        // Assemble the step and pulse responses.
        double  *step_resp  = new double[number_of_rows_];
        double  *pulse_resp = new double[number_of_rows_];
        if (!step_resp || !pulse_resp)
            throw std::runtime_error("ERROR: AmiRx::proc_imp() could not allocate required working vectors!");
        step_resp[0] = 0.0;
        for (auto ix = 0; ix < (number_of_rows_ - 1); ix++)
            step_resp[ix + 1] = step_resp[ix] + impulse_matrix_[ix] * sample_interval_;
        for (auto ix = 0; ix < samples_per_bit_; ix++)
            pulse_resp[ix] = step_resp[ix];
        for (auto ix = samples_per_bit_; ix < number_of_rows_; ix++)
            pulse_resp[ix] = step_resp[ix] - step_resp[ix - samples_per_bit_];

        // Locate the main cursor location.
        unsigned long offset, edge = 0;
        double max_pulse = 0.0;
        for (auto i = 0; i < number_of_rows_; i++)
            if (pulse_resp[i] > max_pulse) {
                offset = i;
                max_pulse = pulse_resp[i];
            }
        edge = offset - samples_per_bit_ / 2;

        // Approximate DFE effect on impulse response, for statistical analysis.
        if (dfe_->mode > 1) {  // Emulate adaptation, using pulse response zero forcing.
            double                  err;
            std::vector<double>     weights;
            for (auto weight : dfe_->get_weights()) {
                offset += samples_per_bit_;  // Move 'offset' to next cursor location.
                weights.push_back(pulse_resp[offset] / dfe_->slicer_mag());
            }
            dfe_->set_weights(weights);
        }
        int tap_num = 1;
        for (auto weight : dfe_->get_weights()) {
            edge += samples_per_bit_;
            impulse_matrix_[edge] -= weight * dfe_->slicer_mag() / sample_interval_;
            params << "\t(tap" << tap_num++ << " " << weight << ")\n";
        }
    }

    params << ")\n";
    param_str_ = params.str();

    msg_ += debug_stream.str();
}

