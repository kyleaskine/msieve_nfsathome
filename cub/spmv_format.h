/* Lossless CSR storage formats shared by the host packer and CUDA kernels. */
#ifndef SPMV_FORMAT_H
#define SPMV_FORMAT_H

#include <stdint.h>
#include <stddef.h>

#define SPMV_ROW_GROUP 128u
#define SPMV_ROWS_U32 0u
#define SPMV_ROWS_BASE16 1u

/* BASE16: one (absolute base, byte offset) uint32 pair per group of 128
   pointers, followed by relative offsets. Offset bit 0 selects uint32
   instead of uint16 for a group spanning more than 65535 nonzeros.
   Payloads are aligned to four bytes. The terminal CSR pointer is encoded
   like every other pointer. This preserves random access and empty rows. */
static inline size_t spmv_row_groups(uint32_t rows) {
    return ((size_t)rows + 1 + SPMV_ROW_GROUP - 1) / SPMV_ROW_GROUP;
}

/* payload of one group of count pointers, padded to four bytes */
static inline size_t spmv_row16_payload_bytes(size_t count, int wide) {
    return (count * (wide ? 4 : 2) + 3) & ~(size_t)3;
}

static inline size_t spmv_row16_bytes(const uint32_t *ptr, uint32_t rows) {
    size_t n = (size_t)rows + 1;
    size_t bytes = 8 * spmv_row_groups(rows);
    for (size_t first = 0; first < n; first += SPMV_ROW_GROUP) {
        size_t count = n - first;
        if (count > SPMV_ROW_GROUP) count = SPMV_ROW_GROUP;
        bytes += spmv_row16_payload_bytes(count,
                    ptr[first + count - 1] - ptr[first] > UINT16_MAX);
    }
    return bytes;
}

/* The same total from an encoded array's last header word alone (the
   offset of the last group's payload, which ends the array), for code
   that holds the array but not the pointers it was built from */
static inline size_t spmv_row16_bytes_from_last(uint32_t rows, uint32_t last) {
    size_t count = (size_t)rows + 1 - (spmv_row_groups(rows) - 1) * SPMV_ROW_GROUP;
    return (last & ~(size_t)1) + spmv_row16_payload_bytes(count, last & 1);
}

/* Bytes per local column index in a block num_cols wide: 3 when 24-bit
   columns were requested and every index fits, else 4 */
static inline uint32_t spmv_col_bytes(int col24, uint32_t num_cols) {
    return col24 && num_cols <= (1u << 24) ? 3 : 4;
}

static inline void spmv_pack_rows(void *out, const uint32_t *ptr, uint32_t rows) {
    size_t n = (size_t)rows + 1;
    size_t offset = 8 * spmv_row_groups(rows);
    uint32_t *header = (uint32_t *)out;
    for (size_t first = 0, page = 0; first < n; first += SPMV_ROW_GROUP, page++) {
        size_t count = n - first;
        if (count > SPMV_ROW_GROUP) count = SPMV_ROW_GROUP;
        uint32_t base = ptr[first];
        int wide = ptr[first + count - 1] - base > UINT16_MAX;
        header[2 * page] = base;
        header[2 * page + 1] = (uint32_t)offset | (uint32_t)wide;
        uint32_t *payload = header + offset / 4;
        if (wide) {
            for (size_t j = 0; j < count; j++)
                payload[j] = ptr[first + j] - base;
        } else {
            /* Use word stores, including the zero-filled half-word tail.
               This also permits packing into a C++ vector<uint32_t>. */
            for (size_t j = 0; j < count; j += 2) {
                uint32_t pair = ptr[first + j] - base;
                if (j + 1 < count)
                    pair |= (ptr[first + j + 1] - base) << 16;
                payload[j / 2] = pair;
            }
        }
        offset += spmv_row16_payload_bytes(count, wide);
    }
}

#ifdef __CUDACC__
#define SPMV_FORMAT_INLINE __host__ __device__ __forceinline__
#else
#define SPMV_FORMAT_INLINE static inline
#endif

SPMV_FORMAT_INLINE uint32_t spmv_row16_value(const uint32_t *data,
        uint32_t base, uint32_t offset, uint32_t row) {
    const uint32_t *payload = data + (offset >> 2);
    uint32_t local = row % SPMV_ROW_GROUP;
    if (offset & 1) return base + payload[local];
    /* Word loads avoid reading a uint32_t object through a uint16_t*,
       which violates C++ aliasing rules in the host autotuner. */
    return base + ((payload[local / 2] >> ((local & 1) * 16)) & 0xffffu);
}

SPMV_FORMAT_INLINE uint32_t spmv_row16_at(const uint32_t *data, uint32_t row) {
    uint32_t group = row / SPMV_ROW_GROUP;
    return spmv_row16_value(data, data[2 * (size_t)group],
                           data[2 * (size_t)group + 1], row);
}

SPMV_FORMAT_INLINE void spmv_col24_put(void *data, size_t col, uint32_t value) {
    uint8_t *out = (uint8_t *)data + col * 3;
    out[0] = (uint8_t)value;
    out[1] = (uint8_t)(value >> 8);
    out[2] = (uint8_t)(value >> 16);
}

SPMV_FORMAT_INLINE uint32_t spmv_col24_at(const uint32_t *data, uint32_t col) {
    size_t byte = (size_t)col * 3;
    uint32_t shift = (uint32_t)(byte & 3) * 8;
    uint32_t value = data[byte / 4] >> shift;
    if (shift > 8) value |= data[byte / 4 + 1] << (32 - shift);
    return value & 0xffffffu;
}

#ifdef __cplusplus
/* Build with -DSPMV_CACHE_ROW_GROUPS=0 for a decoder-only A/B control. */
#ifndef SPMV_CACHE_ROW_GROUPS
#define SPMV_CACHE_ROW_GROUPS 1
#endif

#ifdef __CUDACC__
#define SPMV_CURSOR_INLINE __host__ __device__ __forceinline__
#else
#define SPMV_CURSOR_INLINE inline
#endif

template<unsigned FORMAT> struct spmv_row_cursor {
    const uint32_t *data;
    uint32_t group, base, offset;

    SPMV_CURSOR_INLINE explicit spmv_row_cursor(const uint32_t *p)
        : data(p), group(UINT32_MAX), base(0), offset(0) {}

    SPMV_CURSOR_INLINE uint32_t at(uint32_t row) {
        if (FORMAT == SPMV_ROWS_U32) return data[row];
        if (!SPMV_CACHE_ROW_GROUPS) return spmv_row16_at(data, row);
        uint32_t next = row / SPMV_ROW_GROUP;
        if (next != group) {
            group = next;
            base = data[2 * (size_t)group];
            offset = data[2 * (size_t)group + 1];
        }
        return spmv_row16_value(data, base, offset, row);
    }
};
#undef SPMV_CURSOR_INLINE
#endif

#undef SPMV_FORMAT_INLINE
#endif
