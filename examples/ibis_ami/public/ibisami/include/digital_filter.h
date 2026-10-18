/** \file digital_filter.h
 *  \brief Interface to DigitalFilter class.
 *
 * Original author: David Banas <br>
 * Original date:   May 7, 2015
 *
 * Copyright (c) 2015 David Banas; all rights reserved World wide.
 *
 * This class provides a generic digital filter.
 */

#ifndef INCLUDE_DIGITAL_FILTER_H_
#define INCLUDE_DIGITAL_FILTER_H_

#include <vector>

/// A generic digital filter implementation, using "Direct Form 2" processing.
class DigitalFilter {
 public:
    DigitalFilter(const std::vector<double>& num,
                  const std::vector<double>& den);
    virtual ~DigitalFilter() {}
    void apply(double *sig, const long len);
    std::vector<double> num() {return num_;}
    std::vector<double> den() {return den_;}
    std::vector<double> state() {return state_;}
    int num_taps() {return num_taps_;}
    void clear() {
        state_.clear();
        state_.resize(num_taps_, 0.0);
    }

 protected:
    std::vector<double> num_, den_, state_;
    int num_taps_;
};

#endif  // INCLUDE_DIGITAL_FILTER_H_

