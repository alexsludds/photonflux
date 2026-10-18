/* pf_tx.c -- photonflux IBIS-AMI transmitter: FFE + swing + driver bandwidth.
 *
 * A dual-mode (Init + GetWave) Tx model in the shape of a real SerDes driver:
 *
 *   x --> 6-tap UI-spaced FFE (2 pre, main, 3 post) --> swing --> 2-pole
 *         driver bandwidth --> out
 *
 * The FFE taps are linear and signed, relative to a main tap that takes up
 * the rest of the swing budget: c0 = 1 - sum|c_k|, so the peak output swing
 * stays at tx_swing whatever the de-emphasis. tx_preset selects one of the
 * PCIe 3.0-6.0 coefficient presets P0..P9 instead of the manual taps.
 *
 * The FIR is causal: the main tap sits two UI (the pre-cursor span) behind
 * the newest input, in Init and GetWave alike, so the two modes produce the
 * same pulse response sample for sample. The FFE and the bandwidth filter
 * keep their state across GetWave calls (any block size).
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "pf_ami.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NPRE 2
#define NPOST 3
#define NTAPS (NPRE + 1 + NPOST)

typedef struct {
    double ts, ui;
    long nspui;
    double c[NTAPS]; /* c[0] = c(-2) ... c[NPRE] = main ... */
    double swing;
    double bw;
    pf_biquad drv;
    pf_delay hist;
    long preset;
    pf_str params_out, msg;
} pf_tx;

/* PCIe Gen3+ transmitter presets (c-1, c+1), linear coefficients */
static const double PRESETS[10][2] = {
    {0.000, -0.250}, /* P0 */
    {0.000, -0.167}, /* P1 */
    {0.000, -0.200}, /* P2 */
    {0.000, -0.125}, /* P3 */
    {0.000, 0.000},  /* P4 */
    {-0.100, 0.000}, /* P5 */
    {-0.125, 0.000}, /* P6 */
    {-0.100, -0.200}, /* P7 */
    {-0.125, -0.125}, /* P8 */
    {-0.166, 0.000}, /* P9 */
};

static void configure(pf_tx *m, const pf_params *p)
{
    static const char *names[NTAPS] = {"tx_tap_m2", "tx_tap_m1", NULL,
                                       "tx_tap_p1", "tx_tap_p2", "tx_tap_p3"};
    double side = 0.0;
    m->preset = pf_param_long(p, "tx_preset", -1);
    for (int k = 0; k < NTAPS; k++)
        m->c[k] = names[k] ? pf_param_double(p, names[k], 0.0) : 0.0;
    if (m->preset >= 0 && m->preset <= 9) {
        for (int k = 0; k < NTAPS; k++)
            m->c[k] = 0.0;
        m->c[NPRE - 1] = PRESETS[m->preset][0];
        m->c[NPRE + 1] = PRESETS[m->preset][1];
    }
    for (int k = 0; k < NTAPS; k++)
        if (k != NPRE)
            side += fabs(m->c[k]);
    if (side > 0.9) { /* keep a usable main cursor */
        for (int k = 0; k < NTAPS; k++)
            if (k != NPRE)
                m->c[k] *= 0.9 / side;
        pf_str_printf(&m->msg, "pf_tx: side taps scaled to sum|c| = 0.9; ");
        side = 0.9;
    }
    m->c[NPRE] = 1.0 - side;
    m->swing = pf_param_double(p, "tx_swing", 1.0);
    m->bw = pf_param_double(p, "tx_bw", 0.0);
}

static void design_driver(pf_tx *m)
{
    if (m->bw > 0.0) {
        /* two coincident real poles with a combined -3 dB point at tx_bw */
        double wp = 2.0 * PF_PI * m->bw / sqrt(sqrt(2.0) - 1.0);
        pf_biquad_design(&m->drv, m->ts, 1.0, 0.0, wp, wp);
    } else {
        pf_biquad_design(&m->drv, m->ts, 1.0, 0.0, 0.0, 0.0);
    }
}

static void report(pf_tx *m)
{
    pf_str_clear(&m->params_out);
    pf_str_printf(&m->params_out, "(pf_tx (tx_taps");
    for (int k = 0; k < NTAPS; k++)
        pf_str_printf(&m->params_out, " %.6g", m->c[k]);
    pf_str_printf(&m->params_out, ") (tx_preset %ld) (tx_swing %.6g))",
                  m->preset, m->swing);
}

/* y[n] = swing * drv( sum_k c[k] x[n - (NTAPS-1-k) * nspui] ) is the
 * GetWave recursion; Init runs the same FIR on the impulse response. */
static void fir_block(const pf_tx *m, const double *x, double *y, long n)
{
    for (long i = 0; i < n; i++) {
        double acc = 0.0;
        for (int k = 0; k < NTAPS; k++) {
            long j = i - (long)(NTAPS - 1 - k) * m->nspui;
            if (j >= 0)
                acc += m->c[k] * x[j];
        }
        y[i] = acc;
    }
}

PF_AMI_EXPORT long AMI_Init(double *impulse_matrix, long row_size,
                            long aggressors, double sample_interval,
                            double bit_time, char *AMI_parameters_in,
                            char **AMI_parameters_out,
                            void **AMI_memory_handle, char **msg)
{
    pf_tx *m = (pf_tx *)calloc(1, sizeof(pf_tx));
    pf_params p;
    if (!m)
        return 0;
    *AMI_memory_handle = m;
    m->ts = sample_interval;
    m->ui = bit_time;
    m->nspui = (long)floor(bit_time / sample_interval + 0.5);
    if (m->nspui < 1)
        m->nspui = 1;
    if (fabs(m->nspui * sample_interval - bit_time) > 1e-3 * bit_time)
        pf_str_printf(&m->msg, "pf_tx: bit_time is not a whole number of "
                               "samples; taps spaced %ld samples; ",
                      m->nspui);
    if (!pf_params_parse(&p, AMI_parameters_in))
        return 0;
    configure(m, &p);
    pf_params_free(&p);
    design_driver(m);
    if (!pf_delay_init(&m->hist, (NTAPS - 1) * m->nspui + 2))
        return 0;

    double *tmp = (double *)malloc((size_t)row_size * sizeof(double));
    if (!tmp)
        return 0;
    for (long r = 0; r <= aggressors; r++) {
        double *h = impulse_matrix + r * row_size;
        fir_block(m, h, tmp, row_size);
        pf_biquad_reset(&m->drv);
        for (long i = 0; i < row_size; i++)
            h[i] = m->swing * pf_biquad_step(&m->drv, tmp[i]);
    }
    free(tmp);
    pf_biquad_reset(&m->drv);

    report(m);
    pf_str_printf(&m->msg, "pf_tx: %d-tap FFE, main %.4f, swing %.3g, "
                           "bw %.3g Hz", NTAPS, m->c[NPRE], m->swing, m->bw);
    *AMI_parameters_out = m->params_out.s;
    *msg = m->msg.s;
    return 1;
}

PF_AMI_EXPORT long AMI_GetWave(double *wave, long wave_size,
                               double *clock_times, char **AMI_parameters_out,
                               void *AMI_memory)
{
    pf_tx *m = (pf_tx *)AMI_memory;
    if (!m || !wave)
        return 0;
    for (long i = 0; i < wave_size; i++) {
        pf_delay_push(&m->hist, wave[i]);
        double acc = 0.0;
        for (int k = 0; k < NTAPS; k++)
            acc += m->c[k] *
                   pf_delay_at(&m->hist, (double)((NTAPS - 1 - k) * m->nspui));
        wave[i] = m->swing * pf_biquad_step(&m->drv, acc);
    }
    if (clock_times)
        clock_times[0] = -1.0; /* a transmitter recovers no clock */
    report(m);
    if (AMI_parameters_out)
        *AMI_parameters_out = m->params_out.s;
    return 1;
}

PF_AMI_EXPORT long AMI_Close(void *AMI_memory)
{
    pf_tx *m = (pf_tx *)AMI_memory;
    if (!m)
        return 1;
    pf_delay_free(&m->hist);
    pf_str_free(&m->params_out);
    pf_str_free(&m->msg);
    free(m);
    return 1;
}
