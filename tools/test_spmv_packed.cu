/* Independent CPU-oracle checks for every CSR format and SpMV direction.
   nvcc -O2 -arch=sm_120 -DVBITS=128 -DDISABLE_NVTX -I. -Icub \
       tools/test_spmv_packed.cu -o /tmp/test_spmv_packed
   Run for VBITS=64,128,256. No production matrix or job files are used. */
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "cub/spmv_engine.h"
#include "cub/spmv_format.h"

extern "C" void *spmv_engine_init(int *);
extern "C" void spmv_engine_free(void *);
extern "C" void spmv_engine_set_kernel(void *, int);
extern "C" void spmv_engine_run_packed(void *, spmv_data_t *, uint32_t, uint32_t);
extern "C" void spmv_engine_run_trans_packed(void *, spmv_data_t *, uint32_t, uint32_t);

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); exit(1); } } while (0)
#define GPU(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)

static uint64_t state = 0x8b713ccaaa234defULL;
static uint64_t random_word() {
    state ^= state << 13; state ^= state >> 7; state ^= state << 17;
    return state;
}

static void test_codecs() {
    for (uint32_t rows : {0u, 1u, 126u, 127u, 128u, 129u, 255u, 256u, 513u}) {
        std::vector<uint32_t> ptr((size_t)rows + 1);
        ptr[0] = 0xf0000000u;
        for (uint32_t r = 0; r < rows; r++)
            ptr[r + 1] = ptr[r] + ((r % 129 == 64) ? 70000 : (random_word() % 5));
        size_t bytes = spmv_row16_bytes(ptr.data(), rows);
        std::vector<uint32_t> packed((bytes + 3) / 4 + 1, 0xdeadbeef);
        spmv_pack_rows(packed.data(), ptr.data(), rows);
        CHECK(packed.back() == 0xdeadbeef);
        // the autotuner sizes its copy from the last header word alone
        CHECK(spmv_row16_bytes_from_last(rows,
                packed[2 * spmv_row_groups(rows) - 1]) == bytes);
        spmv_row_cursor<SPMV_ROWS_BASE16> cursor(packed.data());
        spmv_row_cursor<SPMV_ROWS_U32> raw(ptr.data());
        for (uint32_t r = 0; r <= rows; r++) {
            CHECK(spmv_row16_at(packed.data(), r) == ptr[r]);
            CHECK(cursor.at(r) == ptr[r]);
            CHECK(raw.at(r) == ptr[r]);
        }
        // Nonmonotone accesses also reload metadata at a group boundary.
        for (uint32_t j = 0; j < 1024; j++) {
            uint32_t r = random_word() % (rows + 1);
            CHECK(cursor.at(r) == ptr[r]);
        }
    }
    for (uint32_t count = 0; count < 513; count++) {
        std::vector<uint32_t> encoded((3 * count + 3) / 4 + 1, 0);
        std::vector<uint32_t> values(count);
        uint8_t *out = (uint8_t *)encoded.data();
        for (uint32_t j = 0; j < count; j++) {
            values[j] = j % 7 == 0 ? 0xffffff : random_word() & 0xffffff;
            spmv_col24_put(out, j, values[j]);
            for (uint32_t k = 0; k < 3; k++)
                CHECK(out[3 * j + k] == ((values[j] >> (8 * k)) & 255));
        }
        for (uint32_t j = 0; j < count; j++) CHECK(spmv_col24_at(encoded.data(), j) == values[j]);
    }
    // 24-bit columns only for blocks whose every local index fits
    CHECK(spmv_col_bytes(1, 0) == 3);
    CHECK(spmv_col_bytes(1, 1u << 24) == 3);
    CHECK(spmv_col_bytes(1, (1u << 24) + 1) == 4);
    CHECK(spmv_col_bytes(1, UINT32_MAX) == 4);
    CHECK(spmv_col_bytes(0, 100) == 4);
}

static void test_matrix(uint32_t rows, uint32_t cols, int shape, bool prime_empty) {
    std::vector<uint32_t> ptr(rows + 1, 0), idx;
    for (uint32_t r = 0; r < rows; r++) {
        uint32_t len = shape == 0 ? (r % 7 == 0 ? 0 : random_word() % 13) :
                       shape == 1 ? 1024 : shape == 2 ? 2200 :
                       shape == 4 ? random_word() % 41 : 0;
        if (shape == 0 && r == 129) len = 70000;  // wide row group, warp splits
        if (r + 1 == rows) len = 0;               // trailing empty row
        for (uint32_t j = 0; j < len; j++) {
            uint32_t c = random_word() % cols;
            // shape 4: a block 2^24 columns wide, the widest that keeps
            // 24-bit columns; indices at the top end and with a high byte
            // of 0xff exercise every byte of the device decoder
            if (shape == 4 && j % 4 == 0) c = cols - 1 - random_word() % 256;
            if (shape == 4 && j % 4 == 1) c = (0xff0000u | (uint32_t)random_word()) % cols;
            idx.push_back(c);
        }
        ptr[r + 1] = idx.size();
    }
    std::vector<v_t> input(std::max(rows, cols));
    for (v_t &v : input) for (int w = 0; w < VWORDS; w++) v.w[w] = random_word();
    v_t *dx, *dy;
    GPU(cudaMalloc(&dx, input.size() * sizeof(v_t)));
    GPU(cudaMalloc(&dy, input.size() * sizeof(v_t)));
    GPU(cudaMemcpy(dx, input.data(), input.size() * sizeof(v_t), cudaMemcpyHostToDevice));
    for (uint32_t cb : {4u, 3u}) for (uint32_t rf : {SPMV_ROWS_U32, SPMV_ROWS_BASE16}) {
        size_t rb = rf ? spmv_row16_bytes(ptr.data(), rows) : ptr.size() * 4;
        std::vector<uint32_t> row((rb + 3) / 4, 0);
        if (rf) spmv_pack_rows(row.data(), ptr.data(), rows);
        else memcpy(row.data(), ptr.data(), rb);
        size_t ib = (idx.size() * cb + 3) & ~(size_t)3;
        std::vector<uint32_t> col(std::max((size_t)1, ib / 4), 0);
        if (cb == 4 && !idx.empty()) memcpy(col.data(), idx.data(), idx.size() * 4);
        else for (size_t j = 0; j < idx.size(); j++)
            spmv_col24_put(col.data(), j, idx[j]);
        uint32_t *dr, *dc;
        GPU(cudaMalloc(&dr, rb)); GPU(cudaMalloc(&dc, col.size() * 4));
        GPU(cudaMemcpy(dr, row.data(), rb, cudaMemcpyHostToDevice));
        GPU(cudaMemcpy(dc, col.data(), col.size() * 4, cudaMemcpyHostToDevice));
        int bits;
        void *engine = spmv_engine_init(&bits);
        CHECK(bits == VBITS);
        spmv_data_t data = {};
        data.vector_in = (CUdeviceptr)dx; data.vector_out = (CUdeviceptr)dy;
        data.col_entries = (CUdeviceptr)dc; data.row_entries = (CUdeviceptr)dr;
        data.num_rows = rows; data.num_col_entries = idx.size();
        if (prime_empty) {
            uint32_t *empty;
            GPU(cudaMalloc(&empty, ptr.size() * 4)); GPU(cudaMemset(empty, 0, ptr.size() * 4));
            spmv_data_t zero = data; zero.row_entries = (CUdeviceptr)empty; zero.num_col_entries = 0;
            spmv_engine_run_packed(engine, &zero, 4, SPMV_ROWS_U32);
            GPU(cudaDeviceSynchronize()); GPU(cudaFree(empty));
        }
        for (int mode = 0; mode <= 2; mode++) for (int transpose = 0; transpose <= 1; transpose++) {
            spmv_engine_set_kernel(engine, mode);
            size_t n = transpose ? cols : rows;
            std::vector<v_t> expected(n), actual(n);
            for (v_t &v : expected) for (int w = 0; w < VWORDS; w++) v.w[w] = random_word();
            GPU(cudaMemcpy(dy, expected.data(), n * sizeof(v_t), cudaMemcpyHostToDevice));
            for (uint32_t r = 0; r < rows; r++) for (uint32_t j = ptr[r]; j < ptr[r + 1]; j++)
                for (int w = 0; w < VWORDS; w++)
                    expected[transpose ? idx[j] : r].w[w] ^= input[transpose ? r : idx[j]].w[w];
            if (transpose) spmv_engine_run_trans_packed(engine, &data, cb, rf);
            else spmv_engine_run_packed(engine, &data, cb, rf);
            GPU(cudaMemcpy(actual.data(), dy, n * sizeof(v_t), cudaMemcpyDeviceToHost));
            CHECK(memcmp(actual.data(), expected.data(), n * sizeof(v_t)) == 0);
        }
        spmv_engine_free(engine); GPU(cudaFree(dr)); GPU(cudaFree(dc));
    }
    GPU(cudaFree(dx)); GPU(cudaFree(dy));
}

int main(int argc, char **argv) {
    test_codecs();
    if (argc > 1 && strcmp(argv[1], "--host-only") == 0) {
        printf("PASS packed CSR host codecs\n"); return 0;
    }
    GPU(cudaSetDevice(0));
    test_matrix(4097, 2053, 0, false);
    test_matrix(4097, 2053, 0, true);  // 256-item tiles after empty autotune
    test_matrix(257, 2053, 1, false);  // 1024-item tiles
    test_matrix(129, 2053, 2, false);  // 2048-item tiles
    test_matrix(137, 19, 3, false);   // no nonzeros
    test_matrix(301, 1u << 24, 4, false);  // full 24-bit column range
    printf("PASS packed CSR: VBITS=%d, 4 formats, 3 gather modes, both directions, 6 shapes\n", VBITS);
}

/* One CUDA translation unit: lanczos_gpu_core.h contains device definitions
   that cannot be linked from two separately compiled CUDA sources. */
#include "../cub/spmv_engine.cu"
