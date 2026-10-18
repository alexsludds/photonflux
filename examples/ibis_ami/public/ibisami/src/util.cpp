/** \file util.cpp
 *  \brief Small, generally useful utilities.
 *
 * Original author: David Banas <br>
 * Original date:   June 23, 2015
 *
 * Copyright (c) 2015 David Banas; all rights reserved World wide.
 */

/// Linear interpolation.
double interp(double last_x, double x, double ref_time, double time, double ref_period) {
    return last_x + (x - last_x) * (1 - (time - ref_time) / ref_period);
}

