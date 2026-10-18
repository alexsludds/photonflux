/* pf_ami.h -- shared helpers for the photonflux IBIS-AMI models.
 *
 * The photonflux models (pf_tx, pf_rx) are plain C99 with no dependencies
 * beyond libm, so the same sources build as a macOS dylib, a Linux .so or a
 * Windows .dll. Everything here is model-private plumbing: the AMI parameter
 * string tokenizer, a growable string buffer (AMI_parameters_out / msg are
 * owned by the model and freed in AMI_Close), and a few DSP blocks shared by
 * Init (applied to the impulse response) and GetWave (applied sample by
 * sample), which is what keeps the two modes consistent.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
#ifndef PF_AMI_H
#define PF_AMI_H

#include <stddef.h>

#if defined(_WIN32)
#define PF_AMI_EXPORT __declspec(dllexport)
#else
#define PF_AMI_EXPORT __attribute__((visibility("default")))
#endif

/* ---- AMI parameter string ------------------------------------------------
 * "(root (name value) (branch (leaf v1 v2 ...)))" -> flat token list.
 * Lookups search the whole tree for a leaf by name (photonflux model leaf
 * names are unique), so branch nesting is irrelevant to the model. */
typedef struct {
    char **tok;
    int n;
} pf_params;

int pf_params_parse(pf_params *p, const char *s);
void pf_params_free(pf_params *p);
/* number of atoms of leaf `name` (0 if absent); fills vals[] up to max */
int pf_param_atoms(const pf_params *p, const char *name, const char **vals,
                   int max);
double pf_param_double(const pf_params *p, const char *name, double dflt);
long pf_param_long(const pf_params *p, const char *name, long dflt);
/* 1 and the (unquoted) value copied into buf when found, else 0 */
int pf_param_string(const pf_params *p, const char *name, char *buf,
                    size_t n);

/* ---- growable string ----------------------------------------------------- */
typedef struct {
    char *s;
    size_t len, cap;
} pf_str;

void pf_str_clear(pf_str *b);
void pf_str_printf(pf_str *b, const char *fmt, ...);
void pf_str_free(pf_str *b);

/* ---- DSP blocks ---------------------------------------------------------- */

/* Direct-form-I biquad, y = (b0 + b1 z^-1 + b2 z^-2)/(1 + a1 z^-1 + a2 z^-2) */
typedef struct {
    double b0, b1, b2, a1, a2;
    double x1, x2, y1, y2;
} pf_biquad;

void pf_biquad_reset(pf_biquad *f);
double pf_biquad_step(pf_biquad *f, double x);
/* Bilinear-transform design of
 *   H(s) = k (1 + s/wz) / ((1 + s/wp1)(1 + s/wp2))
 * (wz <= 0 drops the zero; wp2 <= 0 drops the second pole). */
void pf_biquad_design(pf_biquad *f, double ts, double k, double wz,
                      double wp1, double wp2);

/* Delay line for UI-spaced FIRs on an oversampled waveform. */
typedef struct {
    double *buf;
    long len, pos; /* pos = index of the newest sample */
} pf_delay;

int pf_delay_init(pf_delay *d, long len);
void pf_delay_free(pf_delay *d);
void pf_delay_push(pf_delay *d, double x);
/* sample `back` samples ago (0 = newest); fractional via linear interp */
double pf_delay_at(const pf_delay *d, double back);

#define PF_PI 3.14159265358979323846
#define PF_MIN(a, b) ((a) < (b) ? (a) : (b))
#define PF_MAX(a, b) ((a) > (b) ? (a) : (b))
#define PF_CLAMP(x, lo, hi) PF_MIN(PF_MAX((x), (lo)), (hi))

#endif
