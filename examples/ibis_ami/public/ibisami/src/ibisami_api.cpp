/** \file ibisami_api.cpp
*   \brief Provides IBIS-AMI API and necessary bootstrapping.
*
* Original author: David Banas <br>
* Original date:   April 29, 2015
*
* Copyright (c) 2015 David Banas; all rights reserved World wide.
*
* Provides the API (application programming interface) to the
* *.SO (shared object) or *.DLL (dynamically linked library) file.
*
* Note: The required API for a IBIS-AMI model is defined by the IBIS
*       standard, which is available here: http://www.eda.org/ibis/
*/

#include <stdexcept>
#include <string>
#include "include/amimodel.h"

#define AMI_GETWAVE

extern AMIModel *ami_model;  ///< Defined in our device-specific source code file.

/// Holds the pointers, which we pass back to the AMI_Init() caller.
struct AmiPointers {
    AMIModel *model;
    char     *msg;
    char     *params;
};

static char panic_msg[] = "ERROR: Failed to initialize!";
static char panic_msg2[] = "ERROR: GetWave() received a null pointer, in 'AMI_memory'!";
static char panic_msg3[] = "ERROR: GetWave() received a null pointer, in 'wave'!";

extern "C" {
// The following is to accomodate Windows and make this file OS agnostic.
// It is taken directly from the Microsoft Visual Studio header file examples.
#ifdef _MSC_VER  // Gets defined, only when the MSVS compiler is invoked.
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT
#endif

/// IBIS-AMI model initialization and impulse response processing. (Required)
/**
 * \param impulse_matrix A matrix of doubles containing, at least, the impulse response to be processed (i.e. - the victim), in place.
 * \param number_of_rows The length of all impulse responses provided (victim and aggressors).
 * \param aggressors The number of aggressors included in impulse_matrix.
 * \param sample_interval The time interval between adjacent elements of the vectors contained in impulse_matrix.
 * \param bit_time The unit interval.
 * \param AMI_parameters_in A pointer to a char array containing the input AMI parameter string.
 * \param AMI_parameters_out A handle to be updated with the pointer to the output AMI parameter string.
 * \param AMI_memory_handle A handle to be updated with the pointer to the model memory structure.
 * \param msg A handle to be updated with the pointer to the message being returned by the model.
 * \return '1' for success, '0' for failure.
 */
DLL_EXPORT long AMI_Init(
        double * impulse_matrix,
        long     number_of_rows,
        long     aggressors,
        double   sample_interval,
        double   bit_time,
        char   * AMI_parameters_in,
        char  ** AMI_parameters_out,
        void  ** AMI_memory_handle,
        char  ** msg
    ) {
    // Even though it seems silly to use new for such a small structure,
    // we must do this, in order to remain reentrant.
    AmiPointers *self = new AmiPointers;
    if (self) {
        self->model = nullptr;
        self->msg = nullptr;
        self->params = nullptr;
        *AMI_memory_handle = self;
    } else {
        *msg = panic_msg;
        return 0;
    }

    // Attempt to initialize the device-specific model.
    try {
        ami_model->init(impulse_matrix, number_of_rows, aggressors,
            sample_interval, bit_time, AMI_parameters_in);
        ami_model->proc_imp();
    } catch(std::runtime_error err) {
        std::string tmp_str = err.what();
        self->msg = new char[tmp_str.length() + 1];
        if (!self->msg) {
            *msg = panic_msg;
            return 0;
        }
        strcpy(self->msg, tmp_str.c_str());
        *msg = self->msg;
        return 0;
    }
    self->model = ami_model;

    // Attempt to create storage for message returned by model maker.
    self->msg = new char[self->model->msg().length() + 1];
    if (!self->msg) {
        *msg = panic_msg;
        return 0;
    }
    strcpy(self->msg, self->model->msg().c_str());
    *msg = self->msg;

    // Attempt to create storage for parameters returned by model maker.
    self->params = new char[self->model->param_str().length() + 1];
    if (!self->params) {
        *msg = panic_msg;
        return 0;
    }
    strcpy(self->params, self->model->param_str().c_str());
    *AMI_parameters_out = self->params;

    return 1;
}

#ifdef AMI_GETWAVE
// Time domain signal processing. (Optional)
DLL_EXPORT long AMI_GetWave(
        double * wave,
        long     wave_size,
        double * clock_times,
        char  ** AMI_parameters_out,
        void   * AMI_memory
    ) {

    // Sanity check the input.
    AmiPointers *self = (AmiPointers *)AMI_memory;
    if (self == nullptr) {
        *AMI_parameters_out = panic_msg2;
        return 0;
    }
    if (wave == nullptr) {
        *AMI_parameters_out = panic_msg3;
        return 0;
    }

    // Attempt to process the signal.
    try {
        self->model->proc_sig(wave, wave_size, clock_times);
    } catch(std::runtime_error err) {
        std::string param_str = "(" + self->model->name() + " (ERROR \"" + err.what() + "\"))";
        delete self->msg;
        self->msg = new char[param_str.length() + 1];
        if (!self->msg) {
            *AMI_parameters_out = panic_msg;
            return 0;
        }
        strcpy(self->msg, param_str.c_str());
        *AMI_parameters_out = self->msg;
        return 0;
    }

    // Attempt to create storage for parameters returned by model maker.
    delete self->params;
    self->params = new char[self->model->param_str().length() + 1];
    if (!self->params) {
        *AMI_parameters_out = panic_msg;
        return 0;
    }
    strcpy(self->params, self->model->param_str().c_str());
    *AMI_parameters_out = self->params;

    return 1;
}
#endif

/// Clean-up. (Required)
/**
 * \param AMI_memory A pointer to the model memory structure.
 * \return '1'
 */
DLL_EXPORT long AMI_Close(
        void * AMI_memory
     ) {
    AmiPointers *self = (AmiPointers *)AMI_memory;
    delete self->msg;
    delete self->params;
    delete self;
    return 1L;
}
}

