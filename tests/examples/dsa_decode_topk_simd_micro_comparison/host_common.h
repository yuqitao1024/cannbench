#ifndef DSA_DECODE_TOPK_COMPARISON_HOST_COMMON_H
#define DSA_DECODE_TOPK_COMPARISON_HOST_COMMON_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "acl/acl.h"
#include "simt_api/asc_bf16.h"

constexpr int32_t kRowCount = 4;
constexpr int32_t kContextCount = 32768;
constexpr int32_t kTopK = 2048;
constexpr int32_t kContextShardCount = 16;
constexpr int32_t kRadixBinCount = 256;
constexpr int32_t kStateWordsPerRow = 4;
constexpr int32_t kOffsetWordsPerShard = 2;

typedef void (*TopKLaunch)(const bfloat16_t*, uint32_t*, uint32_t*,
                           uint32_t*, uint32_t*, int32_t*, aclrtStream);

typedef struct {
    uint16_t* scores;
    uint32_t* high_histogram;
    uint32_t* low_histogram;
    uint32_t* state;
    uint32_t* shard_offsets;
    int32_t* indices;
} TopKDeviceBuffers;

static int check_acl(aclError result, const char* operation)
{
    if (result == ACL_SUCCESS) return 1;
    fprintf(stderr, "ACL call failed: %s, error=%d\n", operation, (int)result);
    return 0;
}

static uint16_t float_to_bfloat16_bits(float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    return (uint16_t)(bits >> 16U);
}

static float bfloat16_bits_to_float(uint16_t bits)
{
    uint32_t wide = (uint32_t)bits << 16U;
    float value;
    memcpy(&value, &wide, sizeof(value));
    return value;
}

static void fill_deterministic_bf16_scores(uint16_t* scores)
{
    for (int32_t row = 0; row < kRowCount; ++row) {
        for (int32_t index = 0; index < kContextCount; ++index) {
            int32_t quantized = ((index * 73 + row * 109) % 4096) - 2048;
            scores[(int64_t)row * kContextCount + index] =
                float_to_bfloat16_bits((float)quantized * 0.0625F);
        }
    }
}

static int compare_float_desc(const void* lhs, const void* rhs)
{
    float a = *(const float*)lhs;
    float b = *(const float*)rhs;
    return (a < b) - (a > b);
}

static int verify_score_set(const uint16_t* scores, const int32_t* indices)
{
    float* sorted = (float*)malloc((size_t)kContextCount * sizeof(float));
    uint8_t* seen = (uint8_t*)malloc((size_t)kContextCount);
    if (sorted == NULL || seen == NULL) {
        fprintf(stderr, "host oracle allocation failed\n");
        free(seen); free(sorted); return 0;
    }
    for (int32_t row = 0; row < kRowCount; ++row) {
        memset(seen, 0, (size_t)kContextCount);
        for (int32_t index = 0; index < kContextCount; ++index) {
            sorted[index] = bfloat16_bits_to_float(
                scores[(int64_t)row * kContextCount + index]);
        }
        qsort(sorted, kContextCount, sizeof(float), compare_float_desc);
        float threshold = sorted[kTopK - 1];
        int32_t greater_count = 0;
        int32_t threshold_equal_selected = 0;
        for (int32_t index = 0; index < kContextCount; ++index) {
            greater_count += sorted[index] > threshold ? 1 : 0;
        }
        int32_t expected_threshold_equal = kTopK - greater_count;
        for (int32_t slot = 0; slot < kTopK; ++slot) {
            int32_t index = indices[(int64_t)row * kTopK + slot];
            if (!(index >= 0 && index < kContextCount)) {
                fprintf(stderr, "row=%d slot=%d index out of bounds: %d\n", row, slot, index);
                free(seen); free(sorted); return 0;
            }
            if (seen[index]) {
                fprintf(stderr, "row=%d duplicate index=%d\n", row, index);
                free(seen); free(sorted); return 0;
            }
            seen[index] = 1;
            float score = bfloat16_bits_to_float(scores[(int64_t)row * kContextCount + index]);
            if (score < threshold) {
                fprintf(stderr, "row=%d selected score below threshold: %.6f < %.6f\n", row, score, threshold);
                free(seen); free(sorted); return 0;
            }
            if (score > threshold) continue;
            ++threshold_equal_selected;
        }
        if (threshold_equal_selected != expected_threshold_equal) {
            fprintf(stderr, "row=%d threshold equal mismatch: expected=%d actual=%d\n",
                    row, expected_threshold_equal, threshold_equal_selected);
            free(seen); free(sorted); return 0;
        }
        printf("row=%d threshold=%.6f greater=%d threshold_equal_selected=%d expected_threshold_equal=%d\n",
               row, threshold, greater_count, threshold_equal_selected, expected_threshold_equal);
    }
    free(seen); free(sorted); return 1;
}

static int run_topk_example(TopKLaunch launch)
{
    const size_t score_bytes = (size_t)kRowCount * kContextCount * sizeof(uint16_t);
    const size_t index_bytes = (size_t)kRowCount * kTopK * sizeof(int32_t);
    const size_t histogram_bytes = (size_t)kRowCount * kContextShardCount *
        kRadixBinCount * sizeof(uint32_t);
    const size_t state_bytes = (size_t)kRowCount * kStateWordsPerRow * sizeof(uint32_t);
    const size_t offset_bytes = (size_t)kRowCount * kContextShardCount *
        kOffsetWordsPerShard * sizeof(uint32_t);
    uint16_t* scores = (uint16_t*)malloc(score_bytes);
    int32_t* indices = (int32_t*)malloc(index_bytes);
    TopKDeviceBuffers device = {NULL, NULL, NULL, NULL, NULL, NULL};
    aclrtStream stream = NULL;
    int initialized = 0, selected = 0, ok = 0;
    if (scores == NULL || indices == NULL) goto cleanup;
    fill_deterministic_bf16_scores(scores);
    for (int32_t i = 0; i < kRowCount * kTopK; ++i) indices[i] = -1;
    if (!check_acl(aclInit(NULL), "aclInit")) goto cleanup;
    initialized = 1;
    if (!check_acl(aclrtSetDevice(0), "aclrtSetDevice")) goto cleanup;
    selected = 1;
    if (!check_acl(aclrtCreateStream(&stream), "aclrtCreateStream")) goto cleanup;
    if (!check_acl(aclrtMalloc((void**)&device.scores, score_bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(scores)")) goto cleanup;
    if (!check_acl(aclrtMalloc((void**)&device.high_histogram, histogram_bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(high_histogram)")) goto cleanup;
    if (!check_acl(aclrtMalloc((void**)&device.low_histogram, histogram_bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(low_histogram)")) goto cleanup;
    if (!check_acl(aclrtMalloc((void**)&device.state, state_bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(state)")) goto cleanup;
    if (!check_acl(aclrtMalloc((void**)&device.shard_offsets, offset_bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(shard_offsets)")) goto cleanup;
    if (!check_acl(aclrtMalloc((void**)&device.indices, index_bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(indices)")) goto cleanup;
    if (!check_acl(aclrtMemcpy(device.scores, score_bytes, scores, score_bytes, ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy(scores H2D)")) goto cleanup;
    if (!check_acl(aclrtMemcpy(device.indices, index_bytes, indices, index_bytes, ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy(indices H2D)")) goto cleanup;
    launch((const bfloat16_t*)device.scores, device.high_histogram,
           device.low_histogram, device.state, device.shard_offsets,
           device.indices, stream);
    if (!check_acl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream")) goto cleanup;
    if (!check_acl(aclrtMemcpy(indices, index_bytes, device.indices, index_bytes, ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy(indices D2H)")) goto cleanup;
    ok = verify_score_set(scores, indices);
cleanup:
    if (device.indices != NULL) (void)aclrtFree(device.indices);
    if (device.shard_offsets != NULL) (void)aclrtFree(device.shard_offsets);
    if (device.state != NULL) (void)aclrtFree(device.state);
    if (device.low_histogram != NULL) (void)aclrtFree(device.low_histogram);
    if (device.high_histogram != NULL) (void)aclrtFree(device.high_histogram);
    if (device.scores != NULL) (void)aclrtFree(device.scores);
    if (stream != NULL) (void)aclrtDestroyStream(stream);
    if (selected) (void)aclrtResetDevice(0);
    if (initialized) (void)aclFinalize();
    free(indices); free(scores);
    printf("%s\n", ok ? "Verification PASSED" : "Verification FAILED");
    return ok ? 0 : 1;
}

#endif
