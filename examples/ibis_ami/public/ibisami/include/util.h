/** \file util.h
 *  \brief Small, generally useful utilities.
 *
 * Original author: David Banas <br>
 * Original date:   June 23, 2015
 *
 * Copyright (c) 2015 David Banas; all rights reserved World wide.
 */

#ifndef INCLUDE_UTIL_H_
#define INCLUDE_UTIL_H_

/// Linear interpolation.
double interp(double last_x, double x, double ref_time, double time, double ref_period);

/// The signum function.
template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

#endif  // INCLUDE_UTIL_H_

