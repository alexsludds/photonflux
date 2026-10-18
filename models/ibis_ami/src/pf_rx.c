/* pf_rx.c -- photonflux IBIS-AMI receiver: CTLE + AGC + FFE + DFE + CDR.
 *
 * A dual-mode (Init + GetWave) SerDes / optical-DSP receiver:
 *
 *   in --> CTLE (1 zero, 2 poles) --> AGC --> UI-spaced FFE --(+)--> out
 *                                                     ^       |
 *                                         DFE feedback|       v
 *                                                    slicer <- CDR clock
 *
 * Init (statistical flow) sees the channel impulse response and
 *   - picks the CTLE peaking (ctle_mode 2) and the sampling phase that
 *     maximize the eye figure of merit: half eye over the RMS of residual
 *     ISI plus the CTLE/FFE-shaped input noise (rx_noise_nv),
 *   - designs the FFE by MMSE (ffe_mode >= 1) around a DFE that zero-forces
 *     dfe_ntaps post-cursors (dfe_mode >= 1),
 *   - sets the AGC gain so the outer data level sits at agc_target,
 *   - returns the equalized impulse response (CTLE, FFE, AGC on every row;
 *     the DFE notches on the victim row only) and seeds GetWave with it.
 *
 * GetWave (time-domain flow) runs the same filters sample by sample and
 *   - recovers the clock: bang-bang (Alexander, cdr_mode 1) or
 *     Mueller-Muller (cdr_mode 2) phase detector, proportional-integral loop,
 *   - slices NRZ or PAM4 (Modulation) against adapted data levels,
 *   - adapts the DFE (dfe_mode 2) and FFE (ffe_mode 2) taps by LMS and the
 *     AGC (agc_mode 2) toward agc_target,
 *   - writes the equalized waveform (the slicer input) back into `wave`,
 *     the clock ticks (sampling instant - UI/2, IBIS convention) into
 *     clock_times, and the live adaptation state into AMI_parameters_out
 *     after every call, which is what the photonflux link engine plots.
 *
 * The stimulus convention is IBIS's: data symbols drive +/-0.5 V, so the
 * outer level of a unit-gain path is 0.5 * pulse main cursor.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "pf_ami.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXFFE 31
#define MAXDFE 16
#define KPRE 8      /* pre-cursors examined by the Init equalizer design */
#define KPOST_MAX 400

typedef struct {
    double ts, ui;
    long nspui;
    int levels; /* 2 = NRZ, 4 = PAM4 */

    long ctle_mode;
    double ctle_peak, ctle_fp1, ctle_fp2m, ctle_max, ctle_step;
    pf_biquad ctle;

    long agc_mode;
    double agc_gain, agc_target, agc_mu;

    long ffe_mode, ffe_n, ffe_pre;
    double ffe_w[MAXFFE], ffe_mu;
    pf_delay ffe_in;

    long dfe_mode, dfe_n;
    double dfe_b[MAXDFE], dfe_mu, dfe_max;
    double dhist[MAXDFE];
    double fb_now, fb_next, t_fb;

    double amp, amp_mu;

    long cdr_mode;
    double kp, ki, integ;
    double t, t_clk, t_edge, edge_val, y_prev, yk_prev, dk_prev;
    int edge_done;
    long nsym;

    double rx_noise_nv, init_fom_db, init_phase;
    pf_str params_out, msg;
} pf_rx;

/* ---- configuration ------------------------------------------------------- */

static void configure(pf_rx *m, const pf_params *p)
{
    char mod[16] = "NRZ";
    pf_param_string(p, "Modulation", mod, sizeof mod);
    m->levels = (!strcmp(mod, "PAM4") || !strcmp(mod, "pam4")) ? 4 : 2;

    m->ctle_mode = pf_param_long(p, "ctle_mode", 2);
    m->ctle_peak = pf_param_double(p, "ctle_peaking_db", 6.0);
    m->ctle_fp1 = pf_param_double(p, "ctle_fp1", 10e9);
    m->ctle_fp2m = pf_param_double(p, "ctle_fp2_mult", 2.0);
    m->ctle_max = pf_param_double(p, "ctle_max_db", 16.0);
    m->ctle_step = pf_param_double(p, "ctle_step_db", 1.0);

    m->agc_mode = pf_param_long(p, "agc_mode", 1);
    m->agc_target = pf_param_double(p, "agc_target", 0.2);
    m->agc_mu = pf_param_double(p, "agc_mu", 1e-4);
    m->agc_gain = 1.0;

    m->ffe_mode = pf_param_long(p, "ffe_mode", 0);
    m->ffe_n = PF_CLAMP(pf_param_long(p, "ffe_ntaps", 7), 1, MAXFFE);
    m->ffe_pre = PF_CLAMP(pf_param_long(p, "ffe_pre", 1), 0, m->ffe_n - 1);
    m->ffe_mu = pf_param_double(p, "ffe_mu", 2e-3);
    if (m->ffe_mode <= 0) {
        m->ffe_n = 1;
        m->ffe_pre = 0;
    }
    memset(m->ffe_w, 0, sizeof m->ffe_w);
    m->ffe_w[m->ffe_pre] = 1.0;

    m->dfe_mode = pf_param_long(p, "dfe_mode", 2);
    m->dfe_n = PF_CLAMP(pf_param_long(p, "dfe_ntaps", 5), 0, MAXDFE);
    m->dfe_mu = pf_param_double(p, "dfe_mu", 2e-3);
    m->dfe_max = pf_param_double(p, "dfe_tap_max", 0.6);
    if (m->dfe_mode <= 0)
        m->dfe_n = 0;

    m->amp_mu = pf_param_double(p, "level_mu", 1e-3);
    m->cdr_mode = pf_param_long(p, "cdr_mode", 1);
    m->kp = pf_param_double(p, "cdr_kp", 4e-3);
    m->ki = pf_param_double(p, "cdr_ki", 4e-6);
    m->init_phase = pf_param_double(p, "cdr_init_phase", 0.5);
    m->rx_noise_nv = pf_param_double(p, "rx_noise_nv", 5.0);
}

static void design_ctle(pf_biquad *f, double ts, double peak_db, double fp1,
                        double fp2m)
{
    /* H(s) = (1/k) (1 + s/wz) / ((1 + s/wp1)(1 + s/wp2)), wz = wp1/k: the
     * peaked band sits near 0 dB and DC is attenuated by peak_db */
    double wp1 = 2.0 * PF_PI * PF_MAX(fp1, 1.0);
    double k = pow(10.0, peak_db / 20.0);
    pf_biquad_design(f, ts, 1.0 / k, wp1 / k, wp1, fp2m * wp1);
}

/* ---- Init-time equalizer design ------------------------------------------ */

static int solve(double *A, double *b, int n) /* A x = b, in place -> b */
{
    for (int c = 0; c < n; c++) {
        int piv = c;
        for (int r = c + 1; r < n; r++)
            if (fabs(A[r * n + c]) > fabs(A[piv * n + c]))
                piv = r;
        if (fabs(A[piv * n + c]) < 1e-300)
            return 0;
        if (piv != c) {
            for (int j = 0; j < n; j++) {
                double t = A[c * n + j];
                A[c * n + j] = A[piv * n + j];
                A[piv * n + j] = t;
            }
            double t = b[c];
            b[c] = b[piv];
            b[piv] = t;
        }
        for (int r = c + 1; r < n; r++) {
            double f = A[r * n + c] / A[c * n + c];
            for (int j = c; j < n; j++)
                A[r * n + j] -= f * A[c * n + j];
            b[r] -= f * b[c];
        }
    }
    for (int r = n - 1; r >= 0; r--) {
        double s = b[r];
        for (int j = r + 1; j < n; j++)
            s -= A[r * n + j] * b[j];
        b[r] = s / A[r * n + r];
    }
    return 1;
}

/* MMSE FFE around a zero-forcing DFE, from symbol-spaced cursors c[0..lc)
 * with the main cursor at index KPRE. Writes w[0..nf), main tap = 1. */
static void mmse_ffe(const double *c, int lc, int nf, int pre, int ndfe,
                     double noise_rel, double *w) /* noise_rel: sn^2/sa^2 */
{
    int rows = lc + nf - 1, cur = KPRE + pre;
    double *A = (double *)calloc((size_t)nf * nf, sizeof(double));
    double *b = (double *)calloc((size_t)nf, sizeof(double));
    if (!A || !b) {
        free(A);
        free(b);
        for (int j = 0; j < nf; j++)
            w[j] = j == pre;
        return;
    }
    /* H[i + j][j] = c[i]; rows cur+1..cur+ndfe are left to the DFE */
    for (int r = 0; r < rows; r++) {
        if (r > cur && r <= cur + ndfe)
            continue;
        double hr[MAXFFE];
        for (int j = 0; j < nf; j++) {
            int i = r - j;
            hr[j] = (i >= 0 && i < lc) ? c[i] : 0.0;
        }
        for (int j = 0; j < nf; j++) {
            for (int k = 0; k < nf; k++)
                A[j * nf + k] += hr[j] * hr[k];
            if (r == cur)
                b[j] += hr[j];
        }
    }
    /* (H'H + sigma_n^2/sigma_a^2 I) w = H' e, plus a tiny ridge */
    double tr = 0.0;
    for (int j = 0; j < nf; j++)
        tr += A[j * nf + j];
    for (int j = 0; j < nf; j++)
        A[j * nf + j] += noise_rel + 1e-9 * tr / nf;
    if (!solve(A, b, nf) || fabs(b[pre]) < 1e-30) {
        for (int j = 0; j < nf; j++)
            w[j] = j == pre;
    } else {
        for (int j = 0; j < nf; j++)
            w[j] = b[j] / b[pre];
    }
    free(A);
    free(b);
}

typedef struct {
    double fom, main, noise_gain;
    long n0;              /* sampling sample of the unequalized pulse */
    double w[MAXFFE], b[MAXDFE];
} eval_t;

/* Score one CTLE setting: equalize the pulse at every candidate phase and
 * keep the best. hc = CTLE-filtered impulse, g2 = sum of the squared
 * CTLE impulse taps (noise power gain before the FFE). */
static eval_t evaluate(const pf_rx *m, const double *hc, long n, double g2,
                       double *pulse, double *cur)
{
    eval_t best;
    memset(&best, 0, sizeof best);
    best.fom = -1.0;
    double acc = 0.0;
    long npk = 0;
    for (long i = 0; i < n; i++) {
        acc += hc[i];
        if (i >= m->nspui)
            acc -= hc[i - m->nspui];
        pulse[i] = acc * m->ts;
        if (pulse[i] > pulse[npk])
            npk = i;
    }
    double L = m->levels;
    double var_a = m->levels == 4 ? 5.0 / 9.0 : 1.0;
    /* white input noise of rx_noise_nv nV/rtHz, per-sample variance */
    double sn2 = pow(m->rx_noise_nv * 1e-9, 2) / (2.0 * m->ts);
    long step = PF_MAX(1, m->nspui / 32);
    for (long n0 = npk - m->nspui / 2; n0 <= npk + m->nspui / 2; n0 += step) {
        if (n0 < 0 || n0 >= n)
            continue;
        int lc = KPRE + 1 + (int)PF_MIN((n - 1 - n0) / m->nspui, KPOST_MAX);
        for (int k = 0; k < lc; k++) {
            long idx = n0 + (long)(k - KPRE) * m->nspui;
            cur[k] = (idx >= 0 && idx < n) ? pulse[idx] : 0.0;
        }
        eval_t e;
        memset(&e, 0, sizeof e);
        int nf = (int)m->ffe_n, pre = (int)m->ffe_pre;
        if (m->ffe_mode > 0) {
            mmse_ffe(cur, lc, nf, pre, (int)m->dfe_n,
                     sn2 * g2 / (0.25 * var_a), e.w);
        } else {
            e.w[0] = 1.0;
        }
        /* equalized cursors q = cur (*) w, main at KPRE + pre */
        int lq = lc + nf - 1, mq = KPRE + pre;
        double resid = 0.0, wg = 0.0;
        double q_main = 0.0;
        for (int j = 0; j < nf; j++)
            wg += e.w[j] * e.w[j];
        for (int i = 0; i < lq; i++) {
            double q = 0.0;
            for (int j = 0; j < nf; j++)
                if (i - j >= 0 && i - j < lc)
                    q += e.w[j] * cur[i - j];
            if (i == mq) {
                q_main = q;
                continue;
            }
            if (i > mq && i <= mq + m->dfe_n) {
                e.b[i - mq - 1] = q; /* normalized below */
                continue;
            }
            resid += q * q;
        }
        if (q_main <= 0.0)
            continue;
        for (int k = 0; k < m->dfe_n; k++) {
            double bk = e.b[k] / q_main;
            double cl = PF_CLAMP(bk, -m->dfe_max, m->dfe_max);
            resid += pow((bk - cl) * q_main, 2);
            e.b[k] = cl;
        }
        /* outer level 0.5 q_main; eye spacing over (L-1) */
        double sig = 0.5 * q_main / (L - 1.0);
        double s2 = 0.25 * var_a * resid + sn2 * g2 * wg;
        e.fom = sig / sqrt(s2 + 1e-60);
        e.main = q_main;
        e.noise_gain = sqrt(g2 * wg);
        e.n0 = n0;
        if (e.fom > best.fom)
            best = e;
    }
    return best;
}

/* ---- reporting ----------------------------------------------------------- */

static void report(pf_rx *m)
{
    pf_str *b = &m->params_out;
    pf_str_clear(b);
    double ph = fmod(m->t_clk, m->ui) / m->ui;
    pf_str_printf(b, "(pf_rx (ctle_peaking_db %.4g) (agc_gain %.6g) "
                     "(level %.6g) (cdr_phase_ui %.6g) (cdr_freq_ppm %.6g) "
                     "(symbols %ld)",
                  m->ctle_mode > 0 ? m->ctle_peak : 0.0, m->agc_gain, m->amp,
                  ph, m->integ * 1e6, m->nsym);
    if (m->ffe_mode > 0) {
        pf_str_printf(b, " (ffe_taps");
        for (int j = 0; j < m->ffe_n; j++)
            pf_str_printf(b, " %.6g", m->ffe_w[j]);
        pf_str_printf(b, ")");
    }
    if (m->dfe_n > 0) {
        pf_str_printf(b, " (dfe_taps");
        for (int j = 0; j < m->dfe_n; j++)
            pf_str_printf(b, " %.6g", m->dfe_b[j]);
        pf_str_printf(b, ")");
    }
    pf_str_printf(b, " (init_fom_db %.4g))", m->init_fom_db);
}

/* ---- AMI API ------------------------------------------------------------- */

PF_AMI_EXPORT long AMI_Init(double *impulse_matrix, long row_size,
                            long aggressors, double sample_interval,
                            double bit_time, char *AMI_parameters_in,
                            char **AMI_parameters_out,
                            void **AMI_memory_handle, char **msg)
{
    pf_rx *m = (pf_rx *)calloc(1, sizeof(pf_rx));
    pf_params p;
    if (!m)
        return 0;
    *AMI_memory_handle = m;
    m->ts = sample_interval;
    m->ui = bit_time;
    m->nspui = PF_MAX(1, (long)floor(bit_time / sample_interval + 0.5));
    if (!pf_params_parse(&p, AMI_parameters_in))
        return 0;
    configure(m, &p);
    pf_params_free(&p);

    long n = row_size;
    double *h = impulse_matrix;
    double *hc = (double *)malloc((size_t)n * sizeof(double));
    double *pulse = (double *)malloc((size_t)n * sizeof(double));
    double *cur = (double *)malloc((KPRE + 1 + KPOST_MAX) * sizeof(double));
    double *g = (double *)malloc((size_t)n * sizeof(double));
    if (!hc || !pulse || !cur || !g) {
        free(hc); free(pulse); free(cur); free(g);
        return 0;
    }

    /* CTLE candidates: the configured peaking, or a sweep (mode 2) */
    int ncand = 1;
    double cand[257];
    cand[0] = m->ctle_peak;
    if (m->ctle_mode == 2) {
        double st = m->ctle_step > 0.0 ? m->ctle_step : 1.0;
        ncand = 0;
        for (double pk = 0.0; pk <= m->ctle_max + 1e-9 && ncand < 257;
             pk += st)
            cand[ncand++] = pk;
    }
    eval_t best;
    memset(&best, 0, sizeof best);
    best.fom = -1.0;
    double best_pk = cand[0];
    for (int c = 0; c < ncand; c++) {
        double g2 = 1.0;
        memcpy(hc, h, (size_t)n * sizeof(double));
        if (m->ctle_mode > 0) {
            design_ctle(&m->ctle, m->ts, cand[c], m->ctle_fp1, m->ctle_fp2m);
            for (long i = 0; i < n; i++)
                hc[i] = pf_biquad_step(&m->ctle, hc[i]);
            /* noise power gain from the CTLE's own impulse response */
            pf_biquad_reset(&m->ctle);
            g2 = 0.0;
            for (long i = 0; i < n; i++) {
                double v = pf_biquad_step(&m->ctle, i == 0 ? 1.0 : 0.0);
                g2 += v * v;
            }
        }
        eval_t e = evaluate(m, hc, n, g2, pulse, cur);
        if (e.fom > best.fom) {
            best = e;
            best_pk = cand[c];
        }
    }
    if (best.fom < 0.0) {
        pf_str_printf(&m->msg, "pf_rx: no positive main cursor in the "
                               "impulse response; equalizer left at defaults");
        best.w[m->ffe_pre] = 1.0;
        best.main = 1.0;
        best.n0 = 0;
    }
    m->ctle_peak = best_pk;
    memcpy(m->ffe_w, best.w, sizeof m->ffe_w);
    memcpy(m->dfe_b, best.b, sizeof m->dfe_b);
    m->init_fom_db = best.fom > 0.0 ? 20.0 * log10(best.fom) : -99.0;
    if (m->agc_mode > 0)
        m->agc_gain = m->agc_target / (0.5 * best.main);
    m->amp = 0.5 * best.main * m->agc_gain;

    /* apply the linear chain to every row; DFE notches on the victim only */
    for (long r = 0; r <= aggressors; r++) {
        double *row = impulse_matrix + r * n;
        if (m->ctle_mode > 0) {
            design_ctle(&m->ctle, m->ts, m->ctle_peak, m->ctle_fp1,
                        m->ctle_fp2m);
            for (long i = 0; i < n; i++)
                row[i] = pf_biquad_step(&m->ctle, row[i]);
        }
        for (long i = n - 1; i >= 0; i--) { /* causal FIR, in place */
            double acc = 0.0;
            for (int j = 0; j < m->ffe_n; j++) {
                long k = i - (long)j * m->nspui;
                if (k >= 0)
                    acc += m->ffe_w[j] * row[k];
            }
            g[i] = acc;
        }
        for (long i = 0; i < n; i++)
            row[i] = g[i] * m->agc_gain;
    }
    long n_main = best.n0 + m->ffe_pre * m->nspui;
    for (long k = 1; k <= m->dfe_n; k++) {
        long idx = n_main + k * m->nspui - m->nspui / 2;
        if (idx >= 0 && idx < n)
            h[idx] -= m->dfe_b[k - 1] * 2.0 * m->amp / m->ts;
    }

    /* GetWave state, seeded from the Init design */
    design_ctle(&m->ctle, m->ts, m->ctle_peak, m->ctle_fp1, m->ctle_fp2m);
    pf_delay_init(&m->ffe_in, (m->ffe_n - 1) * m->nspui + 4);
    if (best.fom > 0.0)
        m->init_phase = fmod(n_main * m->ts, m->ui) / m->ui;
    m->t_clk = m->init_phase * m->ui;
    if (m->t_clk < 0.5 * m->ui) /* clock ticks (t_clk - UI/2) stay >= 0 */
        m->t_clk += m->ui;
    m->t_edge = m->t_clk - 0.5 * m->ui;
    m->edge_done = m->t_edge < 0.0;
    m->t_fb = 1e300;

    free(hc); free(pulse); free(cur); free(g);
    report(m);
    pf_str_printf(&m->msg, "pf_rx: %s, CTLE %.3g dB%s, FFE %ld taps, DFE "
                           "%ld taps, FOM %.2f dB, sampling phase %.3f UI",
                  m->levels == 4 ? "PAM4" : "NRZ", m->ctle_peak,
                  m->ctle_mode == 2 ? " (adapted)" : "",
                  m->ffe_mode > 0 ? m->ffe_n : 0L, m->dfe_n, m->init_fom_db,
                  m->init_phase);
    *AMI_parameters_out = m->params_out.s;
    *msg = m->msg.s;
    return 1;
}

static double slice(const pf_rx *m, double y)
{
    if (m->levels == 4) {
        double a = m->amp * (2.0 / 3.0);
        if (y > a)
            return 1.0;
        if (y > 0.0)
            return 1.0 / 3.0;
        if (y > -a)
            return -1.0 / 3.0;
        return -1.0;
    }
    return y >= 0.0 ? 1.0 : -1.0;
}

PF_AMI_EXPORT long AMI_GetWave(double *wave, long wave_size,
                               double *clock_times, char **AMI_parameters_out,
                               void *AMI_memory)
{
    pf_rx *m = (pf_rx *)AMI_memory;
    if (!m || !wave)
        return 0;
    long nclk = 0, max_clk = wave_size / m->nspui + 1;
    double ui = m->ui, ts = m->ts;
    for (long i = 0; i < wave_size; i++) {
        double x = wave[i];
        if (m->ctle_mode > 0)
            x = pf_biquad_step(&m->ctle, x);
        x *= m->agc_gain;
        pf_delay_push(&m->ffe_in, x);
        double z = 0.0;
        for (int j = 0; j < m->ffe_n; j++)
            z += m->ffe_w[j] * pf_delay_at(&m->ffe_in, (double)j * m->nspui);
        if (m->t >= m->t_fb) {
            m->fb_now = m->fb_next;
            m->t_fb = 1e300;
        }
        double y = z - m->fb_now;

        if (!m->edge_done && m->t >= m->t_edge) {
            double fr = (m->t_edge - (m->t - ts)) / ts;
            m->edge_val = m->y_prev + PF_CLAMP(fr, 0.0, 1.0) * (y - m->y_prev);
            m->edge_done = 1;
        }
        if (m->t >= m->t_clk) {
            double fr = PF_CLAMP((m->t_clk - (m->t - ts)) / ts, 0.0, 1.0);
            double yk = m->y_prev + fr * (y - m->y_prev);
            double d = slice(m, yk);
            double amp = m->amp > 1e-30 ? m->amp : 1e-30;
            double e = yk - d * amp;
            if (clock_times && nclk < max_clk)
                clock_times[nclk++] = m->t_clk - 0.5 * ui;

            if (m->dfe_mode == 2)
                for (int j = 0; j < m->dfe_n; j++)
                    m->dfe_b[j] = PF_CLAMP(m->dfe_b[j] + m->dfe_mu * (e / amp) *
                                                             m->dhist[j],
                                           -m->dfe_max, m->dfe_max);
            if (m->ffe_mode == 2 && m->ffe_n > 1) {
                double r[MAXFFE], nrm = 1e-30;
                for (int j = 0; j < m->ffe_n; j++) {
                    r[j] = pf_delay_at(&m->ffe_in,
                                       (1.0 - fr) + (double)j * m->nspui);
                    nrm += r[j] * r[j];
                }
                /* the main tap is fixed (the AGC owns gain); with a
                 * Mueller-Muller CDR the first pre-cursor tap is held too:
                 * the CDR's lock point is defined by h(-1), and letting
                 * both adapt it walks phase and taps off together */
                long hold = m->cdr_mode == 2 ? m->ffe_pre - 1 : -1;
                for (int j = 0; j < m->ffe_n; j++)
                    if (j != m->ffe_pre && j != hold)
                        m->ffe_w[j] -= m->ffe_mu * e * r[j] / nrm;
            }
            m->amp += m->amp_mu * (yk / d - m->amp);
            if (m->agc_mode == 2)
                m->agc_gain *= 1.0 + m->agc_mu * (m->agc_target - yk / d) /
                                         m->agc_target;

            double pd = 0.0;
            if (m->cdr_mode == 1) {
                int sym = m->levels == 2 ||
                          fabs(fabs(m->dk_prev) - fabs(d)) < 1e-9;
                if (m->dk_prev * d < 0.0 && sym && m->edge_done)
                    pd = (m->edge_val * m->dk_prev > 0.0) ? 1.0 : -1.0;
            } else if (m->cdr_mode == 2) {
                pd = (yk * m->dk_prev - m->yk_prev * d) / amp;
            }
            m->integ += m->ki * pd;
            double stp = m->kp * pd + m->integ;

            for (int j = MAXDFE - 1; j > 0; j--)
                m->dhist[j] = m->dhist[j - 1];
            m->dhist[0] = d;
            double fb = 0.0;
            for (int j = 0; j < m->dfe_n; j++)
                fb += m->dfe_b[j] * m->dhist[j];
            m->fb_next = fb * m->amp;
            m->t_fb = m->t_clk + 0.5 * ui;

            m->yk_prev = yk;
            m->dk_prev = d;
            m->nsym++;
            m->t_clk += ui * (1.0 + stp);
            m->t_edge = m->t_clk - 0.5 * ui;
            m->edge_done = 0;
        }
        wave[i] = y;
        m->y_prev = y;
        m->t += ts;
    }
    if (clock_times)
        clock_times[nclk] = -1.0;
    report(m);
    if (AMI_parameters_out)
        *AMI_parameters_out = m->params_out.s;
    return 1;
}

PF_AMI_EXPORT long AMI_Close(void *AMI_memory)
{
    pf_rx *m = (pf_rx *)AMI_memory;
    if (!m)
        return 1;
    pf_delay_free(&m->ffe_in);
    pf_str_free(&m->params_out);
    pf_str_free(&m->msg);
    free(m);
    return 1;
}
