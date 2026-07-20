#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "acl/acl_rt.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_adaptive_max_pool3d_backward.h"

#define CHECK_ACL(expr)                                                                                 \
    do {                                                                                                \
        auto ret = (expr);                                                                              \
        int32_t code = static_cast<int32_t>(ret);                                                       \
        if (code != 0) {                                                                                \
            std::fprintf(stderr, "[ERROR] %s failed at %s:%d, ret=%d\n", #expr, __FILE__, __LINE__, code); \
            return code;                                                                                \
        }                                                                                               \
    } while (0)

namespace {

template <typename T>
int CopyToDevice(void* device, const std::vector<T>& host)
{
    return static_cast<int>(aclrtMemcpy(
        device,
        host.size() * sizeof(T),
        host.data(),
        host.size() * sizeof(T),
        ACL_MEMCPY_HOST_TO_DEVICE));
}

template <typename T>
int CopyToHost(std::vector<T>& host, void* device)
{
    return static_cast<int>(aclrtMemcpy(
        host.data(),
        host.size() * sizeof(T),
        device,
        host.size() * sizeof(T),
        ACL_MEMCPY_DEVICE_TO_HOST));
}

aclTensor* CreateTensor(
    const std::vector<int64_t>& shape,
    aclDataType dtype,
    aclFormat format,
    void* deviceMem)
{
    return aclCreateTensor(
        shape.data(),
        shape.size(),
        dtype,
        nullptr,
        0,
        format,
        shape.data(),
        shape.size(),
        deviceMem);
}

int RunProbe(bool indicesInt64, aclFormat format)
{
    const int32_t deviceId = 0;
    aclrtStream stream = nullptr;

    CHECK_ACL(aclnnInit(nullptr));
    CHECK_ACL(aclrtSetDevice(deviceId));
    CHECK_ACL(aclrtCreateStream(&stream));

    const std::vector<int64_t> selfShape = {1, 2, 2, 2, 2};
    const std::vector<int64_t> gradShape = {1, 2, 1, 1, 1};
    const int64_t selfCount = 16;
    const int64_t gradCount = 2;

    void* selfDev = nullptr;
    void* gradDev = nullptr;
    void* indicesDev = nullptr;
    void* outDev = nullptr;
    CHECK_ACL(aclrtMalloc(&selfDev, selfCount * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&gradDev, gradCount * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&outDev, selfCount * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    const size_t indicesBytes = gradCount * (indicesInt64 ? sizeof(int64_t) : sizeof(int32_t));
    CHECK_ACL(aclrtMalloc(&indicesDev, indicesBytes, ACL_MEM_MALLOC_HUGE_FIRST));

    std::vector<float> selfHost(selfCount);
    for (int64_t i = 0; i < selfCount; ++i) {
        selfHost[i] = static_cast<float>(i);
    }
    std::vector<float> gradHost = {1.0F, 2.0F};
    std::vector<float> outHost(selfCount, 0.0F);

    CHECK_ACL(CopyToDevice(selfDev, selfHost));
    CHECK_ACL(CopyToDevice(gradDev, gradHost));
    CHECK_ACL(aclrtMemset(outDev, selfCount * sizeof(float), 0, selfCount * sizeof(float)));
    if (indicesInt64) {
        const std::vector<int64_t> indicesHost = {7, 7};
        CHECK_ACL(CopyToDevice(indicesDev, indicesHost));
    } else {
        const std::vector<int32_t> indicesHost = {7, 7};
        CHECK_ACL(CopyToDevice(indicesDev, indicesHost));
    }

    aclTensor* self = CreateTensor(selfShape, ACL_FLOAT, format, selfDev);
    aclTensor* grad = CreateTensor(gradShape, ACL_FLOAT, format, gradDev);
    aclTensor* indices = CreateTensor(
        gradShape,
        indicesInt64 ? ACL_INT64 : ACL_INT32,
        format,
        indicesDev);
    aclTensor* out = CreateTensor(selfShape, ACL_FLOAT, format, outDev);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    const aclnnStatus workspaceStatus = aclnnAdaptiveMaxPool3dBackwardGetWorkspaceSize(
        grad,
        self,
        indices,
        out,
        &workspaceSize,
        &executor);
    std::printf(
        "GetWorkspaceSize status=%d workspace=%lu indices=%s format=%d\n",
        static_cast<int>(workspaceStatus),
        workspaceSize,
        indicesInt64 ? "int64" : "int32",
        static_cast<int>(format));
    if (workspaceStatus == 0) {
        void* workspace = nullptr;
        if (workspaceSize > 0) {
            CHECK_ACL(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
        }
        const aclnnStatus runStatus = aclnnAdaptiveMaxPool3dBackward(
            workspace,
            workspaceSize,
            executor,
            stream);
        std::printf("Run status=%d\n", static_cast<int>(runStatus));
        if (runStatus == 0) {
            CHECK_ACL(aclrtSynchronizeStream(stream));
            CHECK_ACL(CopyToHost(outHost, outDev));
            std::printf("output:");
            for (float value : outHost) {
                std::printf(" %.1f", value);
            }
            std::printf("\n");
        }
        if (workspace != nullptr) {
            CHECK_ACL(aclrtFree(workspace));
        }
    }

    aclDestroyTensor(self);
    aclDestroyTensor(grad);
    aclDestroyTensor(indices);
    aclDestroyTensor(out);
    CHECK_ACL(aclrtFree(selfDev));
    CHECK_ACL(aclrtFree(gradDev));
    CHECK_ACL(aclrtFree(indicesDev));
    CHECK_ACL(aclrtFree(outDev));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclnnFinalize());
    return static_cast<int>(workspaceStatus);
}

} // namespace

int main(int argc, char** argv)
{
    bool indicesInt64 = true;
    aclFormat format = ACL_FORMAT_ND;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--indices-int32") {
            indicesInt64 = false;
        } else if (arg == "--format-ncdhw") {
            format = ACL_FORMAT_NCDHW;
        }
    }
    return RunProbe(indicesInt64, format);
}
