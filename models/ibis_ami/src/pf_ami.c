/* pf_ami.c -- shared helpers for the photonflux IBIS-AMI models (pf_ami.h).
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "pf_ami.h"

#include <ctype.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- AMI parameter string ------------------------------------------------ */

static int push_tok(pf_params *p, int *cap, const char *s, size_t n)
{
    if (p->n == *cap) {
        int nc = *cap ? 2 * *cap : 64;
        char **t = (char **)realloc(p->tok, (size_t)nc * sizeof(char *));
        if (!t)
            return 0;
        p->tok = t;
        *cap = nc;
    }
    char *c = (char *)malloc(n + 1);
    if (!c)
        return 0;
    memcpy(c, s, n);
    c[n] = '\0';
    p->tok[p->n++] = c;
    return 1;
}

int pf_params_parse(pf_params *p, const char *s)
{
    int cap = 0;
    p->tok = NULL;
    p->n = 0;
    if (!s)
        return 1;
    while (*s) {
        if (isspace((unsigned char)*s)) {
            s++;
        } else if (*s == '(' || *s == ')') {
            if (!push_tok(p, &cap, s, 1))
                return 0;
            s++;
        } else if (*s == '"') { /* quoted string: keep the inside */
            const char *e = strchr(s + 1, '"');
            if (!e)
                e = s + strlen(s);
            if (!push_tok(p, &cap, s + 1, (size_t)(e - s - 1)))
                return 0;
            s = *e ? e + 1 : e;
        } else {
            const char *e = s;
            while (*e && !isspace((unsigned char)*e) && *e != '(' && *e != ')')
                e++;
            if (!push_tok(p, &cap, s, (size_t)(e - s)))
                return 0;
            s = e;
        }
    }
    return 1;
}

void pf_params_free(pf_params *p)
{
    for (int i = 0; i < p->n; i++)
        free(p->tok[i]);
    free(p->tok);
    p->tok = NULL;
    p->n = 0;
}

int pf_param_atoms(const pf_params *p, const char *name, const char **vals,
                   int max)
{
    for (int i = 0; i + 1 < p->n; i++) {
        if (strcmp(p->tok[i], "(") != 0 || strcmp(p->tok[i + 1], name) != 0)
            continue;
        int k = 0;
        for (int j = i + 2; j < p->n; j++) {
            if (!strcmp(p->tok[j], "(") || !strcmp(p->tok[j], ")"))
                break;
            if (k < max)
                vals[k] = p->tok[j];
            k++;
        }
        if (k > 0)
            return k;
    }
    return 0;
}

double pf_param_double(const pf_params *p, const char *name, double dflt)
{
    const char *v;
    if (pf_param_atoms(p, name, &v, 1) < 1)
        return dflt;
    if (!strcmp(v, "True") || !strcmp(v, "true") || !strcmp(v, "TRUE"))
        return 1.0;
    if (!strcmp(v, "False") || !strcmp(v, "false") || !strcmp(v, "FALSE"))
        return 0.0;
    char *end;
    double x = strtod(v, &end);
    return end == v ? dflt : x;
}

long pf_param_long(const pf_params *p, const char *name, long dflt)
{
    double x = pf_param_double(p, name, (double)dflt);
    return (long)floor(x + 0.5);
}

int pf_param_string(const pf_params *p, const char *name, char *buf, size_t n)
{
    const char *v;
    if (pf_param_atoms(p, name, &v, 1) < 1 || n == 0)
        return 0;
    strncpy(buf, v, n - 1);
    buf[n - 1] = '\0';
    return 1;
}

/* ---- growable string ----------------------------------------------------- */

void pf_str_clear(pf_str *b)
{
    b->len = 0;
    if (b->s)
        b->s[0] = '\0';
}

void pf_str_printf(pf_str *b, const char *fmt, ...)
{
    va_list ap;
    for (;;) {
        size_t room = b->cap - b->len;
        int need;
        if (b->cap) {
            va_start(ap, fmt);
            need = vsnprintf(b->s + b->len, room, fmt, ap);
            va_end(ap);
            if (need < 0)
                return;
            if ((size_t)need < room) {
                b->len += (size_t)need;
                return;
            }
        } else {
            need = 255;
        }
        size_t nc = b->cap ? 2 * b->cap : 256;
        while (nc < b->len + (size_t)need + 1)
            nc *= 2;
        char *s = (char *)realloc(b->s, nc);
        if (!s)
            return;
        if (!b->cap)
            s[0] = '\0';
        b->s = s;
        b->cap = nc;
    }
}

void pf_str_free(pf_str *b)
{
    free(b->s);
    b->s = NULL;
    b->len = b->cap = 0;
}

/* ---- DSP blocks ---------------------------------------------------------- */

void pf_biquad_reset(pf_biquad *f)
{
    f->x1 = f->x2 = f->y1 = f->y2 = 0.0;
}

double pf_biquad_step(pf_biquad *f, double x)
{
    double y = f->b0 * x + f->b1 * f->x1 + f->b2 * f->x2 - f->a1 * f->y1 -
               f->a2 * f->y2;
    f->x2 = f->x1;
    f->x1 = x;
    f->y2 = f->y1;
    f->y1 = y;
    return y;
}

void pf_biquad_design(pf_biquad *f, double ts, double k, double wz,
                      double wp1, double wp2)
{
    /* N(s) = n0 + n1 s, D(s) = d0 + d1 s + d2 s^2 */
    double n0 = k, n1 = wz > 0.0 ? k / wz : 0.0, n2 = 0.0;
    double d0 = 1.0, d1 = 0.0, d2 = 0.0;
    if (wp1 > 0.0 && wp2 > 0.0) {
        d1 = 1.0 / wp1 + 1.0 / wp2;
        d2 = 1.0 / (wp1 * wp2);
    } else if (wp1 > 0.0) {
        d1 = 1.0 / wp1;
    }
    double c = 2.0 / ts; /* s = c (1 - z^-1) / (1 + z^-1) */
    double B0 = n0 + n1 * c + n2 * c * c;
    double B1 = 2.0 * n0 - 2.0 * n2 * c * c;
    double B2 = n0 - n1 * c + n2 * c * c;
    double A0 = d0 + d1 * c + d2 * c * c;
    double A1 = 2.0 * d0 - 2.0 * d2 * c * c;
    double A2 = d0 - d1 * c + d2 * c * c;
    f->b0 = B0 / A0;
    f->b1 = B1 / A0;
    f->b2 = B2 / A0;
    f->a1 = A1 / A0;
    f->a2 = A2 / A0;
    pf_biquad_reset(f);
}

int pf_delay_init(pf_delay *d, long len)
{
    d->len = len < 2 ? 2 : len;
    d->pos = 0;
    d->buf = (double *)calloc((size_t)d->len, sizeof(double));
    return d->buf != NULL;
}

void pf_delay_free(pf_delay *d)
{
    free(d->buf);
    d->buf = NULL;
}

void pf_delay_push(pf_delay *d, double x)
{
    d->pos = (d->pos + 1) % d->len;
    d->buf[d->pos] = x;
}

double pf_delay_at(const pf_delay *d, double back)
{
    if (back < 0.0)
        back = 0.0;
    if (back > (double)(d->len - 2))
        back = (double)(d->len - 2);
    long i = (long)floor(back);
    double fr = back - (double)i;
    long a = ((d->pos - i) % d->len + d->len) % d->len;
    long b = ((a - 1) % d->len + d->len) % d->len;
    return d->buf[a] * (1.0 - fr) + d->buf[b] * fr;
}
