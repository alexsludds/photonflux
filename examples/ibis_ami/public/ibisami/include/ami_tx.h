/** \file ami_tx.h
 *  \brief Interface to AmiTx class.
 *
 * Original author: David Banas <br>
 * Original date:   May 6, 2015
 *
 * Copyright (c) 2015 David Banas; all rights reserved World wide.
 */

#ifndef INCLUDE_AMI_TX_H_
#define INCLUDE_AMI_TX_H_

#include <string>
#include <vector>
#include "include/amimodel.h"
#include "include/digital_filter.h"

/// A generic IBIS-AMI Tx model implementation.
/**
 * This abstract class provides generic Tx model capability.
 * Device specific Tx models should derive from this class.
 */
class AmiTx : public AMIModel {
    typedef AMIModel inherited;
 public:
    virtual ~AmiTx() {delete filter_;}
    void proc_imp() override;
    bool proc_sig(double *sig, long len, double *clock_times) override;

 protected:
    DigitalFilter *filter_;  ///< Used for pre-emphasis.
    bool have_preemph_;  ///< True, if I have a pre-emphasis filter.
    int num_taps_;  ///< Number of taps in my pre-emphasis filter.
    std::vector<double> tap_weights_;  ///< Tap weights for pre-emphasis filter.
};

#endif  // INCLUDE_AMI_TX_H_

