/** \file ami_rx.h
 *  \brief Interface to AmiRx class.
 *
 * Original author: David Banas <br>
 * Original date:   May 20, 2015
 *
 * Copyright (c) 2015 David Banas; all rights reserved World wide.
 */

#ifndef INCLUDE_AMI_RX_H_
#define INCLUDE_AMI_RX_H_

#include <string>
#include <vector>
#include "include/amimodel.h"
#include "include/digital_filter.h"
#include "include/dfe.h"

/// A generic IBIS-AMI Rx model implementation.
/**
 * This abstract class provides generic Rx model capability.
 * Device specific Rx models should derive from this class.
 */
class AmiRx : public AMIModel {
    typedef AMIModel inherited;
 public:
    AmiRx() {}
    virtual ~AmiRx() {delete ctle_; delete dfe_;}
    void init(double *impulse_matrix, const long number_of_rows,
              const long aggressors, const double sample_interval,
              const double bit_time, const std::string& AMI_parameters_in) override;
    void proc_imp() override;
    bool proc_sig(double *sig, long len, double *clock_times) override;

 protected:
    DigitalFilter *ctle_;
    DFE *dfe_;
    std::ofstream dfe_dump_stream_, dfe_input_stream_;
    bool dump_dfe_adaptation_, dump_adaptation_input_;
    std::string dfe_dump_file_, dfe_input_file_;
    int sig_tap_, init_adapt_tap_;
};

#endif  // INCLUDE_AMI_RX_H_

