#pragma once

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "acl/acl.h"
#include "simt_api/asc_bf16.h"

#define SOFTMAX_ROWS 512
#define SELECTED_TOKENS 2048
#define INDICES_ROWS 4
#define TILE_TOKENS 128
#define TILE_COUNT (SELECTED_TOKENS / TILE_TOKENS)
#define CONTEXT_TOKENS 32768
#define QUERY_TOKENS 2
#define S1_TEMPLATE_ROWS 64
#define ROWS_PER_BLOCK 32
#define BLOCK_DIM (SOFTMAX_ROWS / ROWS_PER_BLOCK)
#define VEC1_SRC_STRIDE ((S1_TEMPLATE_ROWS >> 1) + 1)
#define STAGE_TILE_ELEMENTS (VEC1_SRC_STRIDE * TILE_TOKENS)
#define OUTPUT_TILE_ELEMENTS (S1_TEMPLATE_ROWS * TILE_TOKENS)
#define THREADS_PER_BLOCK 1024
#define SOFTMAX_SCALE (1.0F / 24.0F)
#define SCORE_MIN (-24.0F)
#define SCORE_MAX 24.0F
#define RANDOM_SEED 20260813U

#define MAX_ATOL 2.0e-5
#define MAX_RTOL 2.0e-5
#define SUM_ATOL 2.0e-3
#define SUM_RTOL 2.0e-3
#define NUMERATOR_ATOL 8.0e-3
#define NUMERATOR_RTOL 8.0e-3
#define PROBABILITY_ATOL 2.0e-4
#define PROBABILITY_RTOL 2.0e-3
#define ROW_SUM_ATOL 2.0e-3

typedef struct {
    float* scores;
    int32_t* indices;
    bfloat16_t* numerators;
    float* running_max;
    float* running_sum;
    float* old_scale;
} DeviceBuffers;

typedef struct {
    uint16_t* numerators;
    float* running_max;
    float* running_sum;
    float* old_scale;
} HostOutputs;

static int check_acl(aclError result, const char* expression)
{
    if (result == ACL_SUCCESS) {
        return 1;
    }
    fprintf(stderr, "ACL call failed: %s, ret=%d\n", expression, (int)result);
    return 0;
}

static uint16_t float_to_bfloat16_bits(float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    uint32_t lsb = (bits >> 16U) & 1U;
    bits += 0x7fffU + lsb;
    return (uint16_t)(bits >> 16U);
}

static float bfloat16_bits_to_float(uint16_t value)
{
    uint32_t bits = (uint32_t)value << 16U;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static size_t numerator_physical_offset(uint32_t row, uint32_t selected)
{
    uint32_t block = row / ROWS_PER_BLOCK;
    uint32_t template_block = block / 2U;
    uint32_t sub_block = block % 2U;
    uint32_t local_row = sub_block * ROWS_PER_BLOCK + row % ROWS_PER_BLOCK;
    uint32_t tile = selected / TILE_TOKENS;
    uint32_t local_selected = selected % TILE_TOKENS;
    uint32_t nz_offset = (local_selected / 16U) * 16U * S1_TEMPLATE_ROWS +
        (local_row / 16U) * 16U * 16U + (local_row % 16U) * 16U + local_selected % 16U;
    return (size_t)template_block * TILE_COUNT * OUTPUT_TILE_ELEMENTS +
        (size_t)tile * OUTPUT_TILE_ELEMENTS + nz_offset;
}

static size_t old_scale_physical_offset(uint32_t row, uint32_t tile)
{
    return (size_t)tile * SOFTMAX_ROWS + row;
}

static int score_is_valid(const int32_t* indices, uint32_t row, uint32_t selected)
{
    uint32_t batch = row / 256U;
    uint32_t query = (row / 128U) % QUERY_TOKENS;
    uint32_t index_row = batch * QUERY_TOKENS + query;
    int32_t context_index = indices[(size_t)index_row * SELECTED_TOKENS + selected];
    int32_t causal_limit = CONTEXT_TOKENS - QUERY_TOKENS + (int32_t)query;
    return context_index >= 0 && context_index < CONTEXT_TOKENS && context_index <= causal_limit;
}

static void fill_deterministic_scores(float* scores, int32_t* indices)
{
    uint32_t state = RANDOM_SEED;
    for (size_t index = 0; index < (size_t)SOFTMAX_ROWS * SELECTED_TOKENS; ++index) {
        state = state * 1664525U + 1013904223U;
        float unit = (float)(state & 0x00ffffffU) / 16777215.0F;
        scores[index] = SCORE_MIN + (SCORE_MAX - SCORE_MIN) * unit;
    }
    for (uint32_t row = 0; row < INDICES_ROWS; ++row) {
        for (uint32_t selected = 0; selected < SELECTED_TOKENS; ++selected) {
            indices[(size_t)row * SELECTED_TOKENS + selected] =
                (int32_t)((selected * 13U + row * 17U) % (CONTEXT_TOKENS - 4U));
        }
        indices[(size_t)row * SELECTED_TOKENS] = -1;
        indices[(size_t)row * SELECTED_TOKENS + 1U] = CONTEXT_TOKENS;
        indices[(size_t)row * SELECTED_TOKENS + 2U] = CONTEXT_TOKENS - 1;
    }
}

static void materialize_softmax_stage_scores(float* scores, const int32_t* indices)
{
    for (uint32_t row = 0; row < SOFTMAX_ROWS; ++row) {
        for (uint32_t selected = 0; selected < SELECTED_TOKENS; ++selected) {
            if (!score_is_valid(indices, row, selected)) {
                scores[(size_t)row * SELECTED_TOKENS + selected] = -INFINITY;
            }
        }
    }
}

static void build_online_softmax_oracle(
    const float* scores, const int32_t* indices, HostOutputs* expected)
{
    for (uint32_t row = 0; row < SOFTMAX_ROWS; ++row) {
        float running_max = -INFINITY;
        float running_sum = 0.0F;
        for (uint32_t tile = 0; tile < TILE_COUNT; ++tile) {
            uint32_t begin = tile * TILE_TOKENS;
            float tile_max = -INFINITY;
            for (uint32_t local = 0; local < TILE_TOKENS; ++local) {
                uint32_t selected = begin + local;
                if (score_is_valid(indices, row, selected)) {
                    float scaled = scores[(size_t)row * SELECTED_TOKENS + selected] * SOFTMAX_SCALE;
                    tile_max = scaled > tile_max ? scaled : tile_max;
                }
            }
            float new_max = running_sum > 0.0F && running_max > tile_max ? running_max : tile_max;
            float scale = running_sum > 0.0F ? expf(running_max - new_max) : 0.0F;
            float tile_sum = 0.0F;
            for (uint32_t local = 0; local < TILE_TOKENS; ++local) {
                uint32_t selected = begin + local;
                float numerator = 0.0F;
                if (score_is_valid(indices, row, selected)) {
                    float scaled = scores[(size_t)row * SELECTED_TOKENS + selected] * SOFTMAX_SCALE;
                    numerator = expf(scaled - new_max);
                    tile_sum += numerator;
                }
                expected->numerators[numerator_physical_offset(row, selected)] =
                    float_to_bfloat16_bits(numerator);
            }
            expected->old_scale[old_scale_physical_offset(row, tile)] = scale;
            running_max = new_max;
            running_sum = scale * running_sum + tile_sum;
        }
        expected->running_max[row] = running_max;
        expected->running_sum[row] = running_sum;
    }
}

static int close_enough(float actual, float expected, float atol, float rtol)
{
    return isfinite(actual) && isfinite(expected) &&
        fabsf(actual - expected) <= atol + rtol * fabsf(expected);
}

static int verify_online_softmax(
    const float* scores, const int32_t* indices, const HostOutputs* actual,
    const HostOutputs* expected, const char* implementation)
{
    int vllm_ascend = strcmp(implementation, "vllm_ascend") == 0;
    float worst_probability = 0.0F;
    float worst_row_sum = 0.0F;
    for (uint32_t row = 0; row < SOFTMAX_ROWS; ++row) {
        if (!close_enough(actual->running_max[row], expected->running_max[row], MAX_ATOL, MAX_RTOL) ||
            !close_enough(actual->running_sum[row], expected->running_sum[row], SUM_ATOL, SUM_RTOL)) {
            fprintf(stderr, "state mismatch at row %u: max=%g/%g sum=%g/%g\n", row,
                    actual->running_max[row], expected->running_max[row],
                    actual->running_sum[row], expected->running_sum[row]);
            return 0;
        }
        float row_sum = 0.0F;
        for (uint32_t tile = 0; tile < TILE_COUNT; ++tile) {
            float actual_scale = actual->old_scale[old_scale_physical_offset(row, tile)];
            float expected_scale = expected->old_scale[old_scale_physical_offset(row, tile)];
            if (tile == 0U && vllm_ascend) {
                if (!isnan(actual_scale)) {
                    fprintf(stderr, "vLLM tile-zero old scale was unexpectedly written at row %u: %g\n",
                            row, actual_scale);
                    return 0;
                }
            } else if (!close_enough(actual_scale, expected_scale, SUM_ATOL, SUM_RTOL)) {
                fprintf(stderr, "old scale mismatch at row %u tile %u: %g/%g\n",
                        row, tile, actual_scale, expected_scale);
                return 0;
            }
            float future_scale = 1.0F;
            for (uint32_t future = tile + 1U; future < TILE_COUNT; ++future) {
                future_scale *= actual->old_scale[old_scale_physical_offset(row, future)];
            }
            for (uint32_t local = 0; local < TILE_TOKENS; ++local) {
                uint32_t selected = tile * TILE_TOKENS + local;
                size_t offset = numerator_physical_offset(row, selected);
                float numerator = bfloat16_bits_to_float(actual->numerators[offset]);
                float expected_numerator = bfloat16_bits_to_float(expected->numerators[offset]);
                if (!close_enough(numerator, expected_numerator, NUMERATOR_ATOL, NUMERATOR_RTOL)) {
                    fprintf(stderr, "BF16 numerator mismatch at row %u selected %u: %g/%g\n",
                            row, selected, numerator, expected_numerator);
                    return 0;
                }
                float probability = numerator * future_scale / actual->running_sum[row];
                float expected_probability = score_is_valid(indices, row, selected)
                    ? expf(scores[(size_t)row * SELECTED_TOKENS + selected] * SOFTMAX_SCALE -
                           expected->running_max[row]) / expected->running_sum[row]
                    : 0.0F;
                float difference = fabsf(probability - expected_probability);
                worst_probability = difference > worst_probability ? difference : worst_probability;
                if (!close_enough(probability, expected_probability, PROBABILITY_ATOL, PROBABILITY_RTOL)) {
                    fprintf(stderr, "reconstructed probability mismatch at row %u selected %u: %g/%g\n",
                            row, selected, probability, expected_probability);
                    return 0;
                }
                row_sum += probability;
            }
        }
        float row_error = fabsf(row_sum - 1.0F);
        worst_row_sum = row_error > worst_row_sum ? row_error : worst_row_sum;
        if (!isfinite(row_sum) || row_error > ROW_SUM_ATOL) {
            fprintf(stderr, "row sum mismatch at row %u: %g\n", row, row_sum);
            return 0;
        }
    }
    printf("max_probability_abs_error=%g max_row_sum_abs_error=%g\n",
           worst_probability, worst_row_sum);
    return 1;
}

static void reset_device_buffers(DeviceBuffers* buffers)
{
    memset(buffers, 0, sizeof(*buffers));
}

static void free_device_buffers(DeviceBuffers* buffers)
{
    (void)aclrtFree(buffers->old_scale);
    (void)aclrtFree(buffers->running_sum);
    (void)aclrtFree(buffers->running_max);
    (void)aclrtFree(buffers->numerators);
    (void)aclrtFree(buffers->indices);
    (void)aclrtFree(buffers->scores);
    reset_device_buffers(buffers);
}

static int allocate_device_buffers(DeviceBuffers* buffers)
{
    size_t scores_bytes = (size_t)SOFTMAX_ROWS * SELECTED_TOKENS * sizeof(float);
    size_t indices_bytes = (size_t)INDICES_ROWS * SELECTED_TOKENS * sizeof(int32_t);
    size_t numerators_bytes = (size_t)SOFTMAX_ROWS * SELECTED_TOKENS * sizeof(uint16_t);
    size_t state_bytes = (size_t)SOFTMAX_ROWS * sizeof(float);
    size_t scale_bytes = (size_t)SOFTMAX_ROWS * TILE_COUNT * sizeof(float);
    return check_acl(aclrtMalloc((void**)&buffers->scores, scores_bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc scores") &&
        check_acl(aclrtMalloc((void**)&buffers->indices, indices_bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc indices") &&
        check_acl(aclrtMalloc((void**)&buffers->numerators, numerators_bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc numerators") &&
        check_acl(aclrtMalloc((void**)&buffers->running_max, state_bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc running_max") &&
        check_acl(aclrtMalloc((void**)&buffers->running_sum, state_bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc running_sum") &&
        check_acl(aclrtMalloc((void**)&buffers->old_scale, scale_bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc old_scale");
}

typedef void (*LaunchStage)(DeviceBuffers*, aclrtStream);

static int run_program(LaunchStage launch_stage, const char* implementation)
{
    size_t scores_count = (size_t)SOFTMAX_ROWS * SELECTED_TOKENS;
    size_t indices_count = (size_t)INDICES_ROWS * SELECTED_TOKENS;
    size_t scale_count = (size_t)SOFTMAX_ROWS * TILE_COUNT;
    float* scores = (float*)malloc(scores_count * sizeof(float));
    int32_t* indices = (int32_t*)malloc(indices_count * sizeof(int32_t));
    float* old_scale_sentinel = (float*)malloc(scale_count * sizeof(float));
    HostOutputs actual = {}, expected = {};
    DeviceBuffers buffers;
    aclrtStream stream = NULL;
    int initialized = 0, device_set = 0, ok = 0;
    reset_device_buffers(&buffers);
    actual.numerators = (uint16_t*)malloc(scores_count * sizeof(uint16_t));
    expected.numerators = (uint16_t*)malloc(scores_count * sizeof(uint16_t));
    actual.running_max = (float*)malloc(SOFTMAX_ROWS * sizeof(float));
    expected.running_max = (float*)malloc(SOFTMAX_ROWS * sizeof(float));
    actual.running_sum = (float*)malloc(SOFTMAX_ROWS * sizeof(float));
    expected.running_sum = (float*)malloc(SOFTMAX_ROWS * sizeof(float));
    actual.old_scale = (float*)malloc(scale_count * sizeof(float));
    expected.old_scale = (float*)malloc(scale_count * sizeof(float));
    if (!scores || !indices || !old_scale_sentinel || !actual.numerators || !expected.numerators ||
        !actual.running_max || !expected.running_max || !actual.running_sum ||
        !expected.running_sum || !actual.old_scale || !expected.old_scale) {
        fprintf(stderr, "host allocation failed\n");
        goto cleanup;
    }
    fill_deterministic_scores(scores, indices);
    build_online_softmax_oracle(scores, indices, &expected);
    materialize_softmax_stage_scores(scores, indices);
    for (size_t index = 0; index < scale_count; ++index) {
        old_scale_sentinel[index] = NAN;
    }
    if (!check_acl(aclInit(NULL), "aclInit")) goto cleanup;
    initialized = 1;
    if (!check_acl(aclrtSetDevice(0), "aclrtSetDevice(0)")) goto cleanup;
    device_set = 1;
    if (!check_acl(aclrtCreateStream(&stream), "aclrtCreateStream")) goto cleanup;
    if (!allocate_device_buffers(&buffers)) goto cleanup;
    if (!check_acl(aclrtMemcpy(buffers.scores, scores_count * sizeof(float), scores,
            scores_count * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy scores H2D") ||
        !check_acl(aclrtMemcpy(buffers.indices, indices_count * sizeof(int32_t), indices,
            indices_count * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy indices H2D") ||
        !check_acl(aclrtMemcpy(buffers.old_scale, scale_count * sizeof(float), old_scale_sentinel,
            scale_count * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy old scale H2D")) goto cleanup;
    launch_stage(&buffers, stream);
    if (!check_acl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream")) goto cleanup;
    if (!check_acl(aclrtMemcpy(actual.numerators, scores_count * sizeof(uint16_t), buffers.numerators,
            scores_count * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy numerators D2H") ||
        !check_acl(aclrtMemcpy(actual.running_max, SOFTMAX_ROWS * sizeof(float), buffers.running_max,
            SOFTMAX_ROWS * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy max D2H") ||
        !check_acl(aclrtMemcpy(actual.running_sum, SOFTMAX_ROWS * sizeof(float), buffers.running_sum,
            SOFTMAX_ROWS * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy sum D2H") ||
        !check_acl(aclrtMemcpy(actual.old_scale, scale_count * sizeof(float), buffers.old_scale,
            scale_count * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy old scale D2H")) goto cleanup;
    ok = verify_online_softmax(scores, indices, &actual, &expected, implementation);
    printf("implementation=%s rows=%d selected_tokens=%d tile_tokens=%d scale=1/24 result=%s\n",
           implementation, SOFTMAX_ROWS, SELECTED_TOKENS, TILE_TOKENS, ok ? "PASS" : "FAIL");
cleanup:
    free_device_buffers(&buffers);
    if (stream) (void)aclrtDestroyStream(stream);
    if (device_set) (void)aclrtResetDevice(0);
    if (initialized) (void)aclFinalize();
    free(expected.old_scale); free(actual.old_scale);
    free(expected.running_sum); free(actual.running_sum);
    free(expected.running_max); free(actual.running_max);
    free(expected.numerators); free(actual.numerators);
    free(old_scale_sentinel); free(indices); free(scores);
    printf("%s\n", ok ? "Verification PASSED" : "Verification FAILED");
    return ok ? 0 : 1;
}
