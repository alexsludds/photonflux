/* ami_host.c -- serve one IBIS-AMI shared library's calls over a pipe.
 *
 *   ami_host <model.so>
 *
 * photonflux runs this when a model must not (or cannot) load into the
 * Python process: a local subprocess for crash isolation, or inside a Linux
 * container whose platform matches the library (linux/386 for the 32-bit
 * vendor models of the 2008-2011 IBIS-AMI toolkits, linux/amd64 for x86-64
 * builds on an Apple Silicon host). It is compiled for the model's own ABI,
 * so `long` here is the model's `long`.
 *
 * Protocol (little-endian; ints on the wire are 64-bit; strings are a u64
 * length + bytes). The model's own stdout is redirected to stderr first, so
 * models that printf cannot corrupt the stream.
 *
 *   start   -> "AMIH" u32 version u32 sizeof(long) u32 has_getwave
 *              or "ERR!" str message
 *   'I' u64 row_size u64 aggressors f64 ts f64 ui str params
 *       f64[(aggressors+1)*row_size] impulse
 *           -> i64 ret f64[...] impulse_out str params_out str msg
 *   'G' u64 n u64 clock_capacity f64[n] wave
 *           -> i64 ret f64[n] wave_out u64 nclk f64[nclk] clocks str params_out
 *   'C'     -> i64 ret, then exit
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef long (*init_fn)(double *, long, long, double, double, char *, char **,
                        void **, char **);
typedef long (*getwave_fn)(double *, long, double *, char **, void *);
typedef long (*close_fn)(void *);

static FILE *in, *out;

static void die(const char *what)
{
    fprintf(stderr, "ami_host: %s\n", what);
    exit(2);
}

static void rd(void *p, size_t n)
{
    if (n && fread(p, 1, n, in) != n)
        exit(0); /* parent went away */
}

static void wr(const void *p, size_t n)
{
    if (n && fwrite(p, 1, n, out) != n)
        die("write failed");
}

static uint64_t rd_u64(void)
{
    uint64_t v;
    rd(&v, 8);
    return v;
}

static void wr_u64(uint64_t v) { wr(&v, 8); }

static void wr_i64(int64_t v) { wr(&v, 8); }

static void wr_str(const char *s)
{
    uint64_t n = s ? strlen(s) : 0;
    wr_u64(n);
    wr(s, n);
}

int main(int argc, char **argv)
{
    /* protocol on a private dup of stdout; the model's stdout -> stderr */
    int fd = dup(1);
    if (fd < 0 || dup2(2, 1) < 0)
        die("cannot redirect stdout");
    out = fdopen(fd, "wb");
    in = stdin;
    if (!out)
        die("fdopen failed");
    if (argc != 2) {
        wr("ERR!", 4);
        wr_str("usage: ami_host <model.so>");
        fflush(out);
        return 2;
    }
    void *lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
        wr("ERR!", 4);
        wr_str(dlerror());
        fflush(out);
        return 2;
    }
    init_fn ami_init = (init_fn)dlsym(lib, "AMI_Init");
    getwave_fn ami_getwave = (getwave_fn)dlsym(lib, "AMI_GetWave");
    close_fn ami_close = (close_fn)dlsym(lib, "AMI_Close");
    if (!ami_init || !ami_close) {
        wr("ERR!", 4);
        wr_str("library exports no AMI_Init / AMI_Close");
        fflush(out);
        return 2;
    }
    uint32_t hdr[3] = {1, (uint32_t)sizeof(long), ami_getwave != NULL};
    wr("AMIH", 4);
    wr(hdr, sizeof hdr);
    fflush(out);

    void *handle = NULL;
    double *buf = NULL, *clk = NULL;
    size_t buf_cap = 0, clk_cap = 0;
    for (;;) {
        int cmd = fgetc(in);
        if (cmd == EOF)
            break;
        if (cmd == 'I') {
            uint64_t n = rd_u64(), agg = rd_u64();
            double ts, ui;
            rd(&ts, 8);
            rd(&ui, 8);
            uint64_t plen = rd_u64();
            char *params = (char *)calloc(plen + 1, 1);
            if (!params)
                die("out of memory");
            rd(params, plen);
            size_t cnt = (size_t)((agg + 1) * n);
            double *imp = (double *)malloc(cnt * sizeof(double) + 8);
            if (!imp)
                die("out of memory");
            rd(imp, cnt * sizeof(double));
            char *po = NULL, *msg = NULL;
            if (handle)
                ami_close(handle);
            handle = NULL;
            long ret = ami_init(imp, (long)n, (long)agg, ts, ui, params, &po,
                                &handle, &msg);
            wr_i64(ret);
            wr(imp, cnt * sizeof(double));
            wr_str(po);
            wr_str(msg);
            fflush(out);
            free(imp);
            /* params may be referenced by the model until AMI_Close: keep */
        } else if (cmd == 'G') {
            uint64_t n = rd_u64(), nclk = rd_u64();
            if (n > buf_cap) {
                buf = (double *)realloc(buf, n * sizeof(double));
                buf_cap = n;
            }
            if (nclk + 1 > clk_cap) {
                clk = (double *)realloc(clk, (nclk + 1) * sizeof(double));
                clk_cap = nclk + 1;
            }
            if (!buf || !clk)
                die("out of memory");
            rd(buf, n * sizeof(double));
            for (uint64_t i = 0; i <= nclk; i++)
                clk[i] = -1.0;
            char *po = NULL;
            long ret = ami_getwave ? ami_getwave(buf, (long)n, clk, &po, handle)
                                   : 0;
            uint64_t nc = 0;
            while (nc < nclk && clk[nc] != -1.0)
                nc++;
            wr_i64(ret);
            wr(buf, n * sizeof(double));
            wr_u64(nc);
            wr(clk, nc * sizeof(double));
            wr_str(po);
            fflush(out);
        } else if (cmd == 'C') {
            long ret = handle ? ami_close(handle) : 1;
            handle = NULL;
            wr_i64(ret);
            fflush(out);
            break;
        } else {
            die("unknown command");
        }
    }
    if (handle)
        ami_close(handle);
    return 0;
}
