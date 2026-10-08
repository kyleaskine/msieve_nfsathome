/* simple DSO interface for sparse matrix multiply */

#ifndef _SPMV_ENGINE_H_
#define _SPMV_ENGINE_H_

#include <stdlib.h>
#include <cuda.h>
#include "../common/lanczos/gpu/lanczos_gpu_core.h"

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct {
	int num_rows;
	uint32_t num_col_entries;
	CUdeviceptr vector_in;    /* v_t */
	CUdeviceptr vector_out;    /* v_t */
	CUdeviceptr col_entries;    /* uint32 */
	CUdeviceptr row_entries;    /* uint32 */
} spmv_data_t;

typedef void * (*spmv_engine_init_func)(int * vbits);

typedef void (*spmv_engine_free_func)(void * e);

typedef void (*spmv_engine_run_func)(void * e,
				spmv_data_t * spmv_data);

/* Optional new entry points; the original struct and functions keep their
   ABI for external engines. col_bytes is 3 or 4; row_format is from
   spmv_format.h. Both products use the same packed representation. */
typedef void (*spmv_engine_run_packed_func)(void *e, spmv_data_t *data,
				uint32_t col_bytes, uint32_t row_format);

/* optional: kernel for the gather products (spmv_engine_run) */
enum {
	SPMV_KERNEL_AUTO = 0,	/* per block, from nonzeros per row */
	SPMV_KERNEL_WARPMERGE = 1,
	SPMV_KERNEL_SEGSCAN = 2
};

/* SPMV_KERNEL_AUTO uses segscan for blocks averaging fewer nonzeros per
   row than this, warpmerge for the rest. On an RTX 5070 segscan was 7-39%
   faster from 7 to 24 nonzeros per row (VBITS 64-512); warpmerge was 2%
   faster on A100 blocks with 29-57 per row (MPI pieces whose forward
   blocks average ~29). 28 sits between the two. msieve logs the split it
   expects from this value; an engine built with another value differs. */
#ifndef SEGSCAN_MAX_ROW_MEAN
#define SEGSCAN_MAX_ROW_MEAN 28
#endif
typedef void (*spmv_engine_set_kernel_func)(void * e, int kernel);

#ifdef __cplusplus
}
#endif

#endif /* !_SPMV_ENGINE_H_ */
