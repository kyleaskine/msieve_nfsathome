/* synthmat: measure the structure of an msieve .mat file and generate
   synthetic matrices with the same structure, for timing Lanczos without
   shipping real matrices (tools/tune_la.sh). Build:
   gcc -O2 -fopenmp -o synthmat tools/synthmat.c -lm

   synthmat profile IN.mat OUT.prof
       row weights by rank (rows sorted by weight), the sorted column
       weight distribution, dense row densities
   synthmat gen PROFILE OUT.mat [ncols [sparse_weight_per_col [seed]]]
       defaults: the profiled matrix's size and weight, seed 1
   synthmat blockstats IN.mat BLOCK_NNZ SKIP_ROWS
       nonzeros per row inside consecutive column blocks of BLOCK_NNZ
       nonzeros, ignoring the first SKIP_ROWS rows (msieve packs those)

   File format (lanczos_io.c dump_matrix): uint32 nrows, num_dense_rows,
   ncols; per column uint32 weight, weight sparse row indices (sorted, all
   >= num_dense_rows), then (num_dense_rows + 31) / 32 dense words.

   Generation: every sparse row gets its profiled weight; it picks that
   many distinct columns with probability proportional to each column's
   target weight. Targets follow the profiled distribution, sorted and
   then alternated light/heavy as reduce_matrix leaves them. Column
   weights therefore come out near their targets (Poisson noise); the few
   columns that fall below the profile's lightest are topped up, which
   adds those entries to rows. The lightest row and column weights stay
   at the profile's at any density (real C168 TD 70/90/110 matrices all
   had 3 and 6). The matrix is written to OUT.part and renamed to OUT
   when complete. */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <math.h>
#include <omp.h>

/* see the row weight comment in generate() */
#define HEAD_FRAC 0.01
#define HEAD_ALPHA 0.1

typedef uint32_t u32;
typedef uint64_t u64;

/* an output file this program created, removed if it dies */
static const char *partial_out;

static void die(const char *msg) {
	fprintf(stderr, "error: %s\n", msg);
	if (partial_out != NULL)
		remove(partial_out);
	exit(1);
}

static void xwrite(const void *p, size_t size, size_t n, FILE *fp) {
	if (fwrite(p, size, n, fp) != n)
		die("write failed");
}

static void xclose(FILE *fp) {
	int bad = ferror(fp);
	if (fclose(fp) != 0 || bad)
		die("write failed");
}

/* a whole-argument unsigned number */
static u64 parse_u64(const char *s, const char *what) {
	char *end = (char *)s;
	unsigned long long v = 0;
	errno = 0;
	if (*s >= '0' && *s <= '9')
		v = strtoull(s, &end, 10);
	if (end == s || *end != '\0' || errno == ERANGE) {
		fprintf(stderr, "error: bad %s '%s'\n", what, s);
		exit(1);
	}
	return v;
}

static void *xmalloc(size_t n) {
	void *p = malloc(n ? n : 1);
	if (p == NULL)
		die("out of memory");
	return p;
}

static void *xcalloc(size_t n, size_t s) {
	void *p = calloc(n ? n : 1, s);
	if (p == NULL)
		die("out of memory");
	return p;
}

/* buffered uint32 reader */

typedef struct {
	FILE *fp;
	u32 *buf;
	size_t len, pos;
} reader_t;

#define RBUF (1 << 22)

static void reader_open(reader_t *r, const char *name) {
	r->fp = fopen(name, "rb");
	if (r->fp == NULL)
		die("cannot open input");
	r->buf = xmalloc(RBUF * sizeof(u32));
	r->len = r->pos = 0;
}

static u32 rd(reader_t *r) {
	if (r->pos == r->len) {
		r->len = fread(r->buf, sizeof(u32), RBUF, r->fp);
		r->pos = 0;
		if (r->len == 0)
			die("truncated matrix file");
	}
	return r->buf[r->pos++];
}

static int cmp_u32_desc(const void *a, const void *b) {
	u32 x = *(const u32 *)a, y = *(const u32 *)b;
	return (x < y) - (x > y);
}

static int cmp_u32(const void *a, const void *b) {
	u32 x = *(const u32 *)a, y = *(const u32 *)b;
	return (x > y) - (x < y);
}

/* ---------------- profile ---------------- */

static void profile(const char *in, const char *out) {
	reader_t r;
	u32 nrows, ndense, ncols, words, i, j;
	u32 *rowcount, *colw, *sorted;
	u64 *densebits, nnz = 0, unsorted_cols = 0, bad = 0;
	u64 row_desc = 0, col_asc = 0;
	FILE *fp;
	u32 nseg = 256, s;

	reader_open(&r, in);
	nrows = rd(&r);
	ndense = rd(&r);
	ncols = rd(&r);
	words = (ndense + 31) / 32;
	rowcount = xcalloc(nrows, sizeof(u32));
	colw = xmalloc((size_t)ncols * sizeof(u32));
	densebits = xcalloc(ndense ? ndense : 1, sizeof(u64));

	for (i = 0; i < ncols; i++) {
		u32 w = rd(&r), prev = 0;
		colw[i] = w;
		nnz += w;
		for (j = 0; j < w; j++) {
			u32 row = rd(&r);
			if (row < ndense || row >= nrows)
				bad++;
			else
				rowcount[row]++;
			if (j > 0 && row <= prev)
				unsorted_cols++;
			prev = row;
		}
		for (j = 0; j < words; j++) {
			u32 d = rd(&r), b;
			for (b = 0; b < 32 && 32 * j + b < ndense; b++)
				densebits[32 * j + b] += (d >> b) & 1;
		}
	}
	fclose(r.fp);

	for (i = ndense + 1; i < nrows; i++)
		row_desc += rowcount[i] <= rowcount[i - 1];
	for (i = 1; i < ncols; i++)
		col_asc += colw[i] >= colw[i - 1];

	printf("%s: %u x %u, %u dense rows, %llu sparse nonzeros (%.2f/col)\n",
		in, nrows, ncols, ndense, (unsigned long long)nnz,
		(double)nnz / ncols);
	printf("  out-of-range entries %llu, out-of-order pairs %llu\n",
		(unsigned long long)bad, (unsigned long long)unsorted_cols);
	printf("  sparse rows non-increasing in file order: %.4f%%\n",
		100.0 * row_desc / (nrows - ndense - 1));
	printf("  columns non-decreasing in file order: %.4f%%\n",
		100.0 * col_asc / (ncols - 1));

	/* rows sorted by weight, descending */
	u32 S = nrows - ndense;
	sorted = xmalloc((size_t)S * sizeof(u32));
	memcpy(sorted, rowcount + ndense, (size_t)S * sizeof(u32));
	qsort(sorted, S, sizeof(u32), cmp_u32_desc);
	{
		u64 acc = 0;
		u32 marks[] = {10, 100, 136, 392, 1000, 10000, 100000, 1000000};
		u32 m = 0;
		printf("  heaviest sparse rows:");
		for (i = 0; i < 8; i++)
			printf(" %u", sorted[i]);
		printf("\n  share of sparse nonzeros in the top N rows:");
		for (i = 0; i < S && m < 8; i++) {
			acc += sorted[i];
			if (i + 1 == marks[m]) {
				printf(" %u:%.1f%%", marks[m], 100.0 * acc / nnz);
				m++;
			}
		}
		printf("\n  row weight at rank fraction 0.01/0.1/0.5/0.9/1: "
			"%u %u %u %u %u\n", sorted[S / 100], sorted[S / 10],
			sorted[S / 2], sorted[(u64)S * 9 / 10], sorted[S - 1]);
	}

	fp = fopen(out, "w");
	if (fp == NULL)
		die("cannot write profile");
	fprintf(fp, "synthmat 2\n%u %u %u %llu\n", nrows, ndense, ncols,
		(unsigned long long)nnz);
	fprintf(fp, "dense %u\n", ndense);
	for (i = 0; i < ndense; i++)
		fprintf(fp, "%.6f\n", (double)densebits[i] / ncols);

	/* row points: every rank below 64, then 2% steps, then the last */
	{
		u32 *pts = xmalloc(4096 * sizeof(u32)), np = 0;
		u32 rk = 0;
		while (rk < S - 1 && np < 4095) {
			pts[np++] = rk;
			if (rk < 64)
				rk++;
			else {
				u32 nx = (u32)(rk * 1.02);
				rk = nx > rk + 1 ? nx : rk + 1;
			}
		}
		pts[np++] = S - 1;
		fprintf(fp, "rows %u\n", np);
		for (i = 0; i < np; i++)
			fprintf(fp, "%.7g %.7g\n", (double)pts[i] / S,
				(double)sorted[pts[i]] / ncols);
		free(pts);
	}

	/* column weights: the sorted distribution (msieve sorts columns by
	   weight, then alternates light and heavy ones), relative to the mean */
	qsort(colw, ncols, sizeof(u32), cmp_u32);
	printf("  column weight at quantile 0/0.01/0.1/0.25/0.5/0.75/0.9/0.99/1: "
		"%u %u %u %u %u %u %u %u %u\n", colw[0], colw[ncols / 100],
		colw[ncols / 10], colw[ncols / 4], colw[ncols / 2],
		colw[(u64)ncols * 3 / 4], colw[(u64)ncols * 9 / 10],
		colw[(u64)ncols * 99 / 100], colw[ncols - 1]);
	fprintf(fp, "colq %u\n", nseg + 1);
	for (s = 0; s <= nseg; s++) {
		u64 k = (u64)(ncols - 1) * s / nseg;
		fprintf(fp, "%.7g %.7g\n", (double)s / nseg,
			colw[k] / ((double)nnz / ncols));
	}
	xclose(fp);
	free(rowcount);
	free(colw);
	free(sorted);
	free(densebits);
}

/* ---------------- generation ---------------- */

static inline u64 splitmix(u64 *s) {
	u64 z = (*s += 0x9e3779b97f4a7c15ULL);
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
	z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
	return z ^ (z >> 31);
}

static double interp(const double *x, const double *y, u32 n, double q) {
	u32 lo = 0, hi = n - 1;
	if (q <= x[0])
		return y[0];
	if (q >= x[n - 1])
		return y[n - 1];
	while (hi - lo > 1) {
		u32 mid = (lo + hi) / 2;
		if (x[mid] <= q)
			lo = mid;
		else
			hi = mid;
	}
	return y[lo] + (y[hi] - y[lo]) * (q - x[lo]) / (x[hi] - x[lo]);
}

typedef struct {
	u32 nrows, ndense, ncols;
	u64 nnz;
	double *dense;
	u32 nrp, ncp;
	double *rx, *ry, *cx, *cy;
} prof_t;

static void read_profile(const char *name, prof_t *p) {
	FILE *fp = fopen(name, "r");
	unsigned long long nnz;
	u32 i;
	char tag[16];

	if (fp == NULL || fscanf(fp, "synthmat 2 %u %u %u %llu", &p->nrows,
			&p->ndense, &p->ncols, &nnz) != 4)
		die("bad profile");
	p->nnz = nnz;
	if (fscanf(fp, "%15s %u", tag, &i) != 2 || strcmp(tag, "dense") ||
	    i != p->ndense)
		die("bad profile dense section");
	p->dense = xmalloc((p->ndense + 1) * sizeof(double));
	for (i = 0; i < p->ndense; i++)
		if (fscanf(fp, "%lf", p->dense + i) != 1)
			die("bad dense value");
	if (fscanf(fp, "%15s %u", tag, &p->nrp) != 2 || strcmp(tag, "rows"))
		die("bad profile rows section");
	p->rx = xmalloc(p->nrp * sizeof(double));
	p->ry = xmalloc(p->nrp * sizeof(double));
	for (i = 0; i < p->nrp; i++)
		if (fscanf(fp, "%lf %lf", p->rx + i, p->ry + i) != 2)
			die("bad row point");
	if (fscanf(fp, "%15s %u", tag, &p->ncp) != 2 || strcmp(tag, "colq"))
		die("bad profile cols section");
	p->cx = xmalloc(p->ncp * sizeof(double));
	p->cy = xmalloc(p->ncp * sizeof(double));
	for (i = 0; i < p->ncp; i++)
		if (fscanf(fp, "%lf %lf", p->cx + i, p->cy + i) != 2)
			die("bad col point");
	fclose(fp);
}

/* Vose alias table for sampling index i with probability t[i] / sum(t);
   t is overwritten */

static void build_alias(double *t, u32 n, u32 *thresh, u32 *alias) {
	u32 *small = xmalloc((size_t)n * sizeof(u32));
	u32 *large = xmalloc((size_t)n * sizeof(u32));
	u32 ns = 0, nl = 0, i;
	double tsum = 0;

	for (i = 0; i < n; i++)
		tsum += t[i];
	for (i = 0; i < n; i++) {
		t[i] = t[i] * n / tsum;
		if (t[i] < 1.0)
			small[ns++] = i;
		else
			large[nl++] = i;
	}
	while (ns && nl) {
		u32 a = small[--ns], b = large[nl - 1];
		thresh[a] = (u32)(t[a] * 4294967296.0 > 4294967295.0 ?
				4294967295.0 : t[a] * 4294967296.0);
		alias[a] = b;
		t[b] -= 1.0 - t[a];
		if (t[b] < 1.0) {
			nl--;
			small[ns++] = b;
		}
	}
	while (nl) {
		u32 b = large[--nl];
		thresh[b] = 0xffffffffu;
		alias[b] = b;
	}
	while (ns) {
		u32 a = small[--ns];
		thresh[a] = 0xffffffffu;
		alias[a] = a;
	}
	free(small);
	free(large);
}

static inline u32 alias_draw(u64 u, u32 n, const u32 *thresh,
			const u32 *alias) {
	u32 j = (u32)(((u >> 32) * (u64)n) >> 32);
	return (u32)u < thresh[j] ? j : alias[j];
}

/* draw the distinct columns of one row into colcount (pass 1) or into
   the column lists (pass 2); identical draws in both passes */

static void draw_row(u32 row, u32 count, u64 seed, u32 ncols,
		const u32 *thresh, const u32 *alias, u32 *mark,
		u32 *colcount, u32 *cursor, const u64 *offset, u32 *data) {
	u64 s = seed ^ ((u64)row * 0xd1342543de82ef95ULL);
	u32 got = 0;

	splitmix(&s);
	while (got < count) {
		u32 j = alias_draw(splitmix(&s), ncols, thresh, alias);
		if (mark[j] == row + 1)
			continue;
		mark[j] = row + 1;
		got++;
		if (data == NULL)
			__atomic_fetch_add(colcount + j, 1, __ATOMIC_RELAXED);
		else {
			u32 pos = __atomic_fetch_add(cursor + j, 1,
						__ATOMIC_RELAXED);
			data[offset[j] + pos] = row;
		}
	}
}

static void generate(const char *pname, const char *out, u64 ncols_arg,
		double weight, u64 seed) {
	prof_t p;
	u32 ncols, nrows, ndense, S, i;
	u32 *rowcount, *colcount, *cursor, *data, *thresh, *alias;
	u64 *offset, total = 0;
	double *t, rsum, pweight;
	u32 minw, minr, *extra;
	int64_t nr;
	int nthreads = omp_get_max_threads();
	u32 **marks;
	char *part;
	FILE *fp;

	read_profile(pname, &p);
	pweight = (double)p.nnz / p.ncols;
	if (ncols_arg == 0)
		ncols_arg = p.ncols;
	if (weight <= 0)
		weight = pweight;
	ndense = p.ndense;
	/* as many surplus columns as the profiled matrix */
	nr = (int64_t)ncols_arg - ((int64_t)p.ncols - p.nrows);
	if (ncols_arg > 0xffffffffULL || nr - ndense < 1000 || nr > 0xffffffffLL)
		die("ncols out of range for this profile");
	ncols = (u32)ncols_arg;
	nrows = (u32)nr;
	S = nrows - ndense;

	/* the lightest column and row weights, which do not scale with
	   density (cy and ry are relative to the profiled mean and ncols) */
	minw = (u32)floor(p.cy[0] * pweight + 0.5);
	minr = (u32)floor(p.ry[p.nrp - 1] * p.ncols + 0.5);
	if (minr < 1)
		minr = 1;

	/* row weights, in file order = descending weight. The heaviest rows
	   are small-prime ideals, the same ones at any matrix size, so the
	   first HEAD_FRAC of the profiled rows keep their absolute rank and
	   their share of each column (scaled with the column weight, as the
	   C168 TD 70/90/110 heads were), shrinking slowly with size: real
	   heads held 0.87x the C168 share both at 26.7M (C191, top 136 sparse
	   rows 7.5%) and 41.6M (C199, top 392 12.8%), which (size ratio)^-0.1
	   fits. The rest of the profile is stretched over the remaining rows
	   and rescaled to the target weight. At the profiled size and weight
	   this is the profile itself. */
	rowcount = xmalloc((size_t)S * sizeof(u32));
	{
		double *raw = xmalloc((size_t)S * sizeof(double));
		u64 s = seed * 0x2545f4914f6cdd1dULL + 7;
		u32 S0 = p.nrows - p.ndense, H = (u32)(HEAD_FRAC * S0);
		double ratio = (double)S / S0, hsum = 0, tsum = 0, tscale;
		double hscale = weight / ((double)p.nnz / p.ncols) *
				pow(ratio, -HEAD_ALPHA);
		u64 acc = 0;
		u32 report_ranks[] = {136, 392, 1000, 10000, 100000}, m = 0, k;

		if (H > S / 2)
			H = S / 2;
		for (i = 0; i < H; i++) {
			raw[i] = interp(p.rx, p.ry, p.nrp, (double)i / S0) *
					ncols * hscale;
			hsum += raw[i];
		}
		for (i = H; i < S; i++) {
			double x = (H + (double)(i - H) * (S0 - H) / (S - H)) / S0;
			raw[i] = interp(p.rx, p.ry, p.nrp, x) * ncols;
			tsum += raw[i];
		}
		rsum = weight * ncols;
		if (hsum >= rsum)
			die("target weight too small for the profiled heavy rows");
		/* scale the tail to the target weight, counting the rows
		   held at the profile's lightest weight */
		tscale = (rsum - hsum) / tsum;
		for (k = 0; k < 20; k++) {
			double held = 0, rest = 0, next;
			for (i = H; i < S; i++) {
				if (raw[i] * tscale < minr)
					held += minr;
				else
					rest += raw[i];
			}
			if (rest == 0 || held >= rsum - hsum)
				die("target weight too small for the profile's "
					"lightest rows");
			next = (rsum - hsum - held) / rest;
			if (fabs(next - tscale) <= 1e-9 * tscale)
				break;
			tscale = next;
		}
		for (i = H; i < S; i++)
			raw[i] *= tscale;
		for (i = 0; i < S; i++) {
			double c = raw[i], fl;
			if (c > 0.9 * ncols)
				c = 0.9 * ncols;
			if (c < minr)
				c = minr;
			fl = floor(c);
			rowcount[i] = (u32)fl + ((splitmix(&s) >> 11) *
				(1.0 / 9007199254740992.0) < c - fl);
			total += rowcount[i];
		}
		free(raw);
		/* rounding jitters neighbors; msieve's rows are sorted */
		qsort(rowcount, S, sizeof(u32), cmp_u32_desc);
		fprintf(stderr, "heavy rows: %u at %.3fx the profiled column share; "
			"share of nonzeros in the top N rows:", H,
			hscale * ((double)p.nnz / p.ncols) / weight);
		for (i = 0; i < S && m < 5; i++) {
			acc += rowcount[i];
			if (i + 1 == report_ranks[m])
				fprintf(stderr, " %u:%.1f%%", report_ranks[m++],
					100.0 * acc / total);
		}
		fprintf(stderr, "\n");
	}

	/* column sampling weights: the profiled sorted distribution, no
	   lighter than the profile's lightest column, then msieve's
	   light/heavy alternation (reduce_matrix) */
	t = xmalloc((size_t)ncols * sizeof(double));
	for (i = 0; i < ncols; i++) {
		t[i] = interp(p.cx, p.cy, p.ncp, (i + 0.5) / ncols);
		if (t[i] * weight < minw)
			t[i] = minw / weight;
	}
	{
		u32 a, b;
		for (a = 1, b = ncols - 2; a < b; a += 2, b -= 2) {
			double tmp = t[a];
			t[a] = t[b];
			t[b] = tmp;
		}
	}
	thresh = xmalloc((size_t)ncols * sizeof(u32));
	alias = xmalloc((size_t)ncols * sizeof(u32));
	build_alias(t, ncols, thresh, alias);
	free(t);

	marks = xmalloc(nthreads * sizeof(u32 *));
	for (i = 0; i < (u32)nthreads; i++)
		marks[i] = xcalloc(ncols, sizeof(u32));
	colcount = xcalloc(ncols, sizeof(u32));

	fprintf(stderr, "generating %u x %u, %u dense rows, %llu sparse "
		"nonzeros (%.2f/col), seed %llu, %d threads\n", nrows, ncols,
		ndense, (unsigned long long)total, (double)total / ncols,
		(unsigned long long)seed, nthreads);

#pragma omp parallel for schedule(dynamic, 1024)
	for (i = 0; i < S; i++)
		draw_row(ndense + i, rowcount[i], seed, ncols, thresh, alias,
			marks[omp_get_thread_num()], colcount, NULL, NULL, NULL);

	/* msieve deletes the lightest columns; top up the few that Poisson
	   noise left below the profiled minimum */
	extra = xcalloc(ncols, sizeof(u32));
	offset = xmalloc(((size_t)ncols + 1) * sizeof(u64));
	offset[0] = 0;
	for (i = 0; i < ncols; i++) {
		if (colcount[i] < minw)
			extra[i] = minw - colcount[i];
		offset[i + 1] = offset[i] + colcount[i] + extra[i];
	}
	data = xmalloc(offset[ncols] * sizeof(u32));
	cursor = xcalloc(ncols, sizeof(u32));
	for (i = 0; i < (u32)nthreads; i++)
		memset(marks[i], 0, (size_t)ncols * sizeof(u32));

#pragma omp parallel for schedule(dynamic, 1024)
	for (i = 0; i < S; i++)
		draw_row(ndense + i, rowcount[i], seed, ncols, thresh, alias,
			marks[omp_get_thread_num()], NULL, cursor, offset, data);

	{
		u32 *rthresh = xmalloc((size_t)S * sizeof(u32));
		u32 *ralias = xmalloc((size_t)S * sizeof(u32));
		double *rw = xmalloc((size_t)S * sizeof(double));
		u64 added = 0, short_cols = 0;
		for (i = 0; i < S; i++)
			rw[i] = rowcount[i];
		build_alias(rw, S, rthresh, ralias);
		free(rw);
#pragma omp parallel for schedule(dynamic, 4096) reduction(+:added,short_cols)
		for (i = 0; i < ncols; i++) {
			u64 s = (seed + 7777) ^ ((u64)i * 0xd1342543de82ef95ULL);
			u32 *col = data + offset[i];
			splitmix(&s);
			short_cols += extra[i] > 0;
			while (extra[i]) {
				u32 row = ndense + alias_draw(splitmix(&s), S,
							rthresh, ralias);
				u32 k;
				for (k = 0; k < colcount[i] && col[k] != row; k++)
					;
				if (k < colcount[i])
					continue;
				col[colcount[i]++] = row;
				extra[i]--;
				added++;
			}
			qsort(col, colcount[i], sizeof(u32), cmp_u32);
		}
		fprintf(stderr, "topped up %llu columns below weight %u with "
			"%llu nonzeros\n", (unsigned long long)short_cols, minw,
			(unsigned long long)added);
		free(rthresh);
		free(ralias);
		free(extra);
	}

	for (i = 0; i < (u32)nthreads; i++)
		free(marks[i]);
	free(marks);
	free(cursor);

	/* dense rows: independent bits at the profiled densities */
	u32 words = (ndense + 31) / 32;
	u32 *dense = xcalloc((size_t)ncols * (words ? words : 1), sizeof(u32));
#pragma omp parallel for schedule(static)
	for (i = 0; i < ncols; i++) {
		/* not splitmix's increment: neighbors would share a stream */
		u64 s = (seed + 99) ^ ((u64)i * 0xff51afd7ed558ccdULL);
		u32 b;
		splitmix(&s);
		for (b = 0; b < ndense; b++)
			if ((splitmix(&s) >> 11) * (1.0 / 9007199254740992.0) <
					p.dense[b])
				dense[(size_t)i * words + b / 32] |= 1u << (b % 32);
	}

	/* a complete file appears under its name or not at all */
	part = xmalloc(strlen(out) + 6);
	sprintf(part, "%s.part", out);
	fp = fopen(part, "wb");
	if (fp == NULL)
		die("cannot write matrix");
	partial_out = part;
	setvbuf(fp, NULL, _IOFBF, 1 << 24);
	{
		u32 hdr[3] = {nrows, ndense, ncols};
		xwrite(hdr, sizeof(u32), 3, fp);
	}
	for (i = 0; i < ncols; i++) {
		xwrite(colcount + i, sizeof(u32), 1, fp);
		xwrite(data + offset[i], sizeof(u32), colcount[i], fp);
		if (words)
			xwrite(dense + (size_t)i * words, sizeof(u32), words, fp);
	}
	xclose(fp);
	if (rename(part, out) != 0)
		die("cannot rename the finished matrix");
	partial_out = NULL;
	free(part);
	free(data);
	free(dense);
	free(offset);
	free(colcount);
	free(rowcount);
	free(thresh);
	free(alias);
}

/* ---------------- block statistics ---------------- */

static void blockstats(const char *in, u64 block_nnz, u32 skip) {
	reader_t r;
	u32 nrows, ndense, ncols, words, i, j;
	u32 *seg;
	u64 hist[12] = {0}, inblock = 0, blocks = 0, nz = 0, rows_hit = 0;
	u64 heavy_nz = 0;
	const char *lab[12] = {"0", "1", "2", "3-4", "5-8", "9-16", "17-32",
		"33-64", "65-128", "129-256", "257-1024", ">1024"};

	reader_open(&r, in);
	nrows = rd(&r);
	ndense = rd(&r);
	ncols = rd(&r);
	words = (ndense + 31) / 32;
	seg = xcalloc(nrows, sizeof(u32));

	for (i = 0; i <= ncols; i++) {
		if (i == ncols || inblock >= block_nnz) {
			/* close the block */
			u32 k;
			for (k = skip; k < nrows; k++) {
				u32 c = seg[k], b;
				if (c == 0)
					b = 0;
				else if (c <= 2)
					b = c;
				else if (c <= 1024) {
					b = 3;
					while (c > (4u << (b - 3)) && b < 10)
						b++;
				}
				else
					b = 11;
				hist[b]++;
				rows_hit += c > 0;
				if (c > 64)
					heavy_nz += c;
			}
			memset(seg, 0, (size_t)nrows * sizeof(u32));
			blocks++;
			inblock = 0;
			if (i == ncols)
				break;
		}
		u32 w = rd(&r);
		for (j = 0; j < w; j++) {
			u32 row = rd(&r);
			if (row >= skip) {
				seg[row]++;
				inblock++;
				nz++;
			}
		}
		for (j = 0; j < words; j++)
			rd(&r);
	}
	fclose(r.fp);

	printf("%s: %llu blocks of ~%llu nonzeros (rows >= %u), %.2f nonzeros "
		"per row per block, %.1f%% of rows hit per block, %.1f%% of "
		"nonzeros in row segments over 64\n", in,
		(unsigned long long)blocks, (unsigned long long)block_nnz, skip,
		(double)nz / blocks / (nrows - skip),
		100.0 * rows_hit / blocks / (nrows - skip),
		100.0 * heavy_nz / nz);
	printf("  row segment length histogram (%% of rows over all blocks):\n ");
	for (i = 0; i < 12; i++)
		printf(" %s:%.3f", lab[i],
			100.0 * hist[i] / blocks / (nrows - skip));
	printf("\n");
	free(seg);
}

int main(int argc, char **argv) {
	if (argc >= 4 && !strcmp(argv[1], "profile"))
		profile(argv[2], argv[3]);
	else if (argc >= 4 && !strcmp(argv[1], "gen")) {
		double weight = 0;
		if (argc > 5) {
			char *end;
			weight = strtod(argv[5], &end);
			if (end == argv[5] || *end != '\0' || !(weight >= 0)) {
				fprintf(stderr, "error: bad weight '%s'\n", argv[5]);
				return 1;
			}
		}
		generate(argv[2], argv[3],
			argc > 4 ? parse_u64(argv[4], "ncols") : 0, weight,
			argc > 6 ? parse_u64(argv[6], "seed") : 1);
	}
	else if (argc >= 5 && !strcmp(argv[1], "blockstats")) {
		u64 block_nnz = parse_u64(argv[3], "BLOCK_NNZ");
		u64 skip = parse_u64(argv[4], "SKIP_ROWS");
		if (block_nnz == 0 || skip > 0xffffffffULL) {
			fprintf(stderr, "error: bad BLOCK_NNZ or SKIP_ROWS\n");
			return 1;
		}
		blockstats(argv[2], block_nnz, (u32)skip);
	}
	else {
		fprintf(stderr, "usage: synthmat profile IN.mat OUT.prof\n"
			"       synthmat gen PROFILE OUT.mat [ncols [weight [seed]]]\n"
			"       synthmat blockstats IN.mat BLOCK_NNZ SKIP_ROWS\n");
		return 1;
	}
	return 0;
}
