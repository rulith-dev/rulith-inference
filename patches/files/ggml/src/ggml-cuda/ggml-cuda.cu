#include "hc-mix.cuh"
#include "idx-relu-sum.cuh"
#include "mmb.cuh"
#include "gdn-conv.cuh"
#include "ple-conv.cuh"
#include "norm-gated.cuh"
#include "expand.cuh"
#include <array>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif
#include "hc-cn.cuh"
#include "mmh-moe.cuh"
#include "ggml-cuda.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "ggml-cuda/allreduce.cuh"
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/acc.cuh"
#include "ggml-cuda/add-id.cuh"
#include "ggml-cuda/arange.cuh"
#include "ggml-cuda/argmax.cuh"
#include "ggml-cuda/argsort.cuh"
#include "ggml-cuda/binbcast.cuh"
#include "ggml-cuda/clamp.cuh"
#include "ggml-cuda/col2im-1d.cuh"
#include "ggml-cuda/concat.cuh"
#include "ggml-cuda/conv-transpose-1d.cuh"
#include "ggml-cuda/conv2d.cuh"
#include "ggml-cuda/conv2d-dw.cuh"
#include "ggml-cuda/conv2d-transpose.cuh"
#include "ggml-cuda/conv3d.cuh"
#include "ggml-cuda/convert.cuh"
#include "ggml-cuda/count-equal.cuh"
#include "ggml-cuda/cpy.cuh"
#include "ggml-cuda/cross-entropy-loss.cuh"
#include "ggml-cuda/cumsum.cuh"
#include "ggml-cuda/diagmask.cuh"
#include "ggml-cuda/diag.cuh"
#include "ggml-cuda/fattn.cuh"
#include "ggml-cuda/qsa.cuh"
#include "ggml-cuda/mul-mat-had.cuh"
#include "ggml-cuda/fwht.cuh"
#include "ggml-cuda/getrows.cuh"
#include "ggml-cuda/im2col.cuh"
#include "ggml-cuda/mmf.cuh"
#include "ggml-cuda/mmq.cuh"
#include "ggml-cuda/mmvf.cuh"
#include "ggml-cuda/mmvq.cuh"
#include "ggml-cuda/moe-weighted-reduction.cuh"
// the 5-argument form lives in this backend's own moe-weighted-reduction.cu (shared-expert merge folded in; the gated
// shared expert's tail as well)
void ggml_cuda_op_moe_weighted_reduction(ggml_backend_cuda_context & ctx, const ggml_tensor * experts,
                                         const ggml_tensor * expert_scale, const ggml_tensor * weights,
                                         ggml_tensor * dst, const ggml_tensor * merge, const ggml_tensor * shexp,
                                         const ggml_tensor * sgate);
#include "ggml-cuda/norm.cuh"
#include "ggml-cuda/opt-step-adamw.cuh"
#include "ggml-cuda/opt-step-sgd.cuh"
#include "ggml-cuda/out-prod.cuh"
#include "ggml-cuda/pad.cuh"
#include "ggml-cuda/pool2d.cuh"
#include "ggml-cuda/pool1d.cuh"
#include "ggml-cuda/quantize.cuh"
#include "ggml-cuda/rope.cuh"
#include "ggml-cuda/roll.cuh"
#include "ggml-cuda/scale.cuh"
#include "ggml-cuda/snake.cuh"
#include "ggml-cuda/softcap.cuh"
#include "ggml-cuda/softmax.cuh"
#include "ggml-cuda/ssm-conv.cuh"
#include "ggml-cuda/ssm-scan.cuh"
#include "ggml-cuda/sum.cuh"
#include "ggml-cuda/sumrows.cuh"
#include "ggml-cuda/top-k.cuh"
#include "ggml-cuda/mean.cuh"
#include "ggml-cuda/tsembd.cuh"
#include "ggml-cuda/topk-moe.cuh"
#include "ggml-cuda/unary.cuh"
#include "ggml-cuda/upscale.cuh"
#include "ggml-cuda/wkv.cuh"
#include "ggml-cuda/gla.cuh"
#include "ggml-cuda/gated_delta_net.cuh"
#include "ggml-cuda/strixllama-chain.cuh"
#include "ggml-cuda/strixllama-getrows-cast.cuh"
#include "ggml-cuda/dsv4-hc.cuh"
#include "ggml-cuda/set.cuh"
#include "ggml-cuda/set-rows.cuh"
#include "ggml-cuda/pad_reflect_1d.cuh"
#include "ggml-cuda/solve_tri.cuh"
#include "ggml-cuda/tri.cuh"
#include "ggml-cuda/cumsum.cuh"
#include "ggml-cuda/fill.cuh"
#include "ggml-cuda/lightning-indexer.cuh"
#include "ggml-cuda/top-k-mask.cuh"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <cinttypes>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cfloat>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static_assert(sizeof(half) == sizeof(ggml_fp16_t), "wrong fp16 size");

#define GGML_LOG_WARN_ONCE(str) \
    { static std::once_flag warn_flag; std::call_once(warn_flag, []() { GGML_LOG_WARN(str); }); }

[[noreturn]]
void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    int id = -1; // in case cudaGetDevice fails
    (void)cudaGetDevice(&id);

    GGML_LOG_ERROR(GGML_CUDA_NAME " error: %s\n", msg);
    GGML_LOG_ERROR("  current device: %d, in function %s at %s:%d\n", id, func, file, line);
    GGML_LOG_ERROR("  %s\n", stmt);
    // abort with GGML_ABORT to get a stack trace
    GGML_ABORT(GGML_CUDA_NAME " error");
}

// map a (possibly virtual) device id to the physical CUDA device that backs it
static int ggml_cuda_get_physical_device(int device) {
    const ggml_cuda_device_info & info = ggml_cuda_info();
    GGML_ASSERT(device >= 0 && device < info.device_count);
    return info.devices[device].physical_device;
}

// this is faster on Windows
// probably because the Windows CUDA libraries forget to make this check before invoking the drivers
void ggml_cuda_set_device(int device) {
    // translate the (possibly virtual) device id to the physical CUDA device that backs it
    const int physical_device = ggml_cuda_get_physical_device(device);

    int current_device;
    CUDA_CHECK(cudaGetDevice(&current_device));

    if (physical_device == current_device) {
        return;
    }

    CUDA_CHECK(cudaSetDevice(physical_device));
}

int ggml_cuda_get_device() {
    int id;
    CUDA_CHECK(cudaGetDevice(&id));
    return id;
}

static bool ggml_cuda_env_flag(const char * name) {
    const char * env = getenv(name);
    return env != nullptr && std::atoi(env) != 0;
}

static int ggml_cuda_env_int(const char * name, int default_value) {
    const char * env = getenv(name);
    return env != nullptr ? std::atoi(env) : default_value;
}

static bool ggml_cuda_env_set(const char * name) {
    return getenv(name) != nullptr;
}

static bool ggml_cuda_unified_memory_enabled() {
    static const bool enabled = ggml_cuda_env_flag("GGML_CUDA_ENABLE_UNIFIED_MEMORY");
    return enabled;
}

static bool ggml_cuda_pinned_memory_disabled() {
    static const bool disabled = ggml_cuda_env_flag("GGML_CUDA_NO_PINNED");
    return disabled;
}

static bool ggml_cuda_register_host_enabled() {
    static const bool enabled = ggml_cuda_env_flag("GGML_CUDA_REGISTER_HOST");
    return enabled;
}

static bool ggml_cuda_fusion_disabled() {
    static const bool disabled = ggml_cuda_env_flag("GGML_CUDA_DISABLE_FUSION");
    return disabled;
}

static bool ggml_cuda_p2p_enabled() {
    static const bool enabled = ggml_cuda_env_set("GGML_CUDA_P2P");
    return enabled;
}

static int ggml_cuda_op_offload_min_batch() {
    static const int min_batch = ggml_cuda_env_int("GGML_OP_OFFLOAD_MIN_BATCH", 32);
    return min_batch;
}

static cudaError_t ggml_cuda_device_malloc(void ** ptr, size_t size, int device) {
    ggml_cuda_set_device(device);
    cudaError_t err;
    if (ggml_cuda_unified_memory_enabled()) {
        err = cudaMallocManaged(ptr, size);
#if defined(GGML_USE_HIP)
        if (err == hipSuccess) {
            // hipMemAdviseSetCoarseGrain is an optional performance hint;
            // ignore errors (e.g. hipErrorInvalidValue on some APU/iGPU configs).
            (void)cudaMemAdvise(*ptr, size, hipMemAdviseSetCoarseGrain, device);
            (void)hipGetLastError(); // clear any error
        }

        // fall back to cudaMalloc if not supported (e.g. on Windows)
        if (err == hipErrorNotSupported) {
            static bool warned_unsupported = false;
            if (!warned_unsupported) {
                GGML_LOG_WARN("hipMallocManaged unsupported, falling back to hipMalloc.\n");
                warned_unsupported = true;
            }

            err = cudaMalloc(ptr, size);
        }
#endif // defined(GGML_USE_HIP)
    } else {
        err = cudaMalloc(ptr, size);
    }
    return err;
}

#if defined(GGML_USE_HIP)
static int ggml_cuda_parse_id(char devName[]) {
    // A list of possible Target IDs can be found under the rocclr/clr repo in device.cpp
    // these values are not stable so this is susceptible to breakage
    // https://github.com/ROCm/clr/blob/amd-staging/rocclr/device/device.cpp
    int archMajor = 0x0;
    int archMinor = 0x0;
    int archNum = GGML_CUDA_CC_OFFSET_AMD;
    int archLen = strlen(devName);
    char archName[archLen + 1];

    // strip leading 'gfx' while copying into our buffer
    if (archLen > 3) {
        strcpy(archName, &devName[3]);
        archLen -= 3;
    }

    // trim trailing :xnack- or :sramecc- statuses
    archLen = strcspn(archName, ":");
    archName[archLen] = '\0';

    // tease out the version information
    if (archLen > 8) {
        // versions labeled generic use '-' as delimiter
        // strip the trailing "-generic" then iterate through what remains
        if ((strstr(archName, "-generic"))) {
            archName[archLen - 8] = '\0';
            char * pch;
            if ((pch = strtok(archName, "-"))) {
                archMajor = (int)strtoul(pch, 0, 16);
                if ((pch = strtok(NULL, "-"))) {
                    archMinor = 0x10 * (int)strtoul(pch, 0, 16);
                }
            }
        }
    } else if (archLen >= 3) {
        // last two digits should be the minor * 0x10 + stepping
        archMinor = (int)strtoul(&archName[archLen - 2], 0, 16);
        archName[archLen - 2] = '\0';

        // only the major version remains
        archMajor = (int)strtoul(archName, 0, 16);
    }
    archNum += archMajor * 0x100;
    archNum += archMinor;

    return archNum;
}
#endif // defined(GGML_USE_HIP)

static ggml_cuda_device_info ggml_cuda_init() {
    ggml_cuda_device_info info = {};

    cudaError_t err = cudaGetDeviceCount(&info.physical_device_count);
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("%s: failed to initialize " GGML_CUDA_NAME ": %s\n", __func__, cudaGetErrorString(err));
        return info;
    }

    GGML_ASSERT(info.physical_device_count <= GGML_CUDA_MAX_DEVICES);

    // by default expose exactly the physical devices; GGML_CUDA_DEVICES can request a different
    // number of (virtual) devices to emulate multi-GPU systems on a machine with fewer GPUs
    info.device_count = info.physical_device_count;

    const char * devices_env = getenv("GGML_CUDA_DEVICES");
    if (devices_env != nullptr && info.physical_device_count > 0) {
        const int requested = atoi(devices_env);
        if (requested > 0) {
            info.device_count = requested;
        } else {
            GGML_LOG_WARN("%s: ignoring invalid GGML_CUDA_DEVICES=\"%s\"\n", __func__, devices_env);
        }
    }

    if (info.device_count > GGML_CUDA_MAX_DEVICES) {
        GGML_LOG_WARN("%s: requested %d devices, clamping to GGML_CUDA_MAX_DEVICES=%d\n",
                      __func__, info.device_count, GGML_CUDA_MAX_DEVICES);
        info.device_count = GGML_CUDA_MAX_DEVICES;
    }

    // map each (virtual) device to a backing physical device (round-robin), assign each its index
    // among the (virtual) devices sharing that physical GPU, and store the per-physical share count
    int physical_share_count[GGML_CUDA_MAX_DEVICES] = {};
    GGML_ASSERT(info.device_count == 0 || info.physical_device_count > 0);
    for (int id = 0; id < info.device_count; ++id) {
        info.devices[id].physical_device = id % info.physical_device_count;
        info.devices[id].virtual_index  = physical_share_count[info.devices[id].physical_device]++;
    }

    int64_t total_vram = 0;
    for (int id = 0; id < info.physical_device_count; ++id) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, id));
        total_vram += prop.totalGlobalMem;
    }
    GGML_LOG_INFO("%s: found %d " GGML_CUDA_NAME " devices (Total VRAM: %zu MiB):\n",
                  __func__, info.physical_device_count, (size_t)(total_vram / (1024 * 1024)));
    if (info.device_count != info.physical_device_count) {
        GGML_LOG_INFO("%s: emulating %d virtual device(s) on %d physical device(s) (GGML_CUDA_DEVICES)\n",
                      __func__, info.device_count, info.physical_device_count);
    }
    total_vram = 0;

    std::vector<std::pair<int, std::string>> turing_devices_without_mma;
    for (int id = 0; id < info.device_count; ++id) {
        const int physical_id = info.devices[id].physical_device;

        int device_vmm = 0;

#if defined(GGML_USE_VMM)
        CUdevice device;
        CU_CHECK(cuDeviceGet(&device, physical_id));
        CU_CHECK(cuDeviceGetAttribute(&device_vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device));

        if (device_vmm) {
            CUmemAllocationProp alloc_prop = {};
            alloc_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            alloc_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            alloc_prop.location.id = physical_id;
            CU_CHECK(cuMemGetAllocationGranularity(&info.devices[id].vmm_granularity, &alloc_prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
        }
#endif // defined(GGML_USE_VMM)
        info.devices[id].vmm = !!device_vmm;

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, physical_id));

        // a virtual device owns only a share of its physical GPU's memory; report that share so the
        // logged per-device VRAM sums to the physical total above.
        GGML_ASSERT(physical_share_count[physical_id] > 0);
        info.devices[id].physical_share_count = physical_share_count[physical_id];
        const size_t device_vram = prop.totalGlobalMem / info.devices[id].physical_share_count;
        const size_t device_vram_mib = device_vram / (1024 * 1024);

        info.default_tensor_split[id] = total_vram;
        total_vram += device_vram;
#if defined(GGML_USE_HIP)
        // Report the real flag on HIP. The corrupted output this was disabled for (#15034)
        // came from graph inputs being overwritten while an asynchronous backend still read
        // them; the scheduler's graph input ring buffer fixes that at the source, so an APU
        // can use its host-visible buffers again instead of staging every input through VRAM.
        info.devices[id].integrated = prop.integrated;
#else
        info.devices[id].integrated = false; // Temporarily disabled due to issues with corrupted output (e.g. #15034)
#endif
        info.devices[id].nsm        = prop.multiProcessorCount;
        info.devices[id].smpb       = prop.sharedMemPerBlock;
        info.devices[id].warp_size  = prop.warpSize;

        int supports_coop_launch = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&supports_coop_launch, cudaDevAttrCooperativeLaunch, physical_id));
        info.devices[id].supports_cooperative_launch = !!supports_coop_launch;

#if defined(GGML_USE_HIP)
        info.devices[id].smpbo = prop.sharedMemPerBlock;

        info.devices[id].cc = ggml_cuda_parse_id(prop.gcnArchName);
        if ((info.devices[id].cc & 0xff00) == 0x0) {
            GGML_LOG_WARN("invalid architecture ID received for device %d %s: %s  cc %d.%d\n",
                            id, prop.name, prop.gcnArchName, prop.major, prop.minor);

            // Fallback to prop.major and prop.minor
            if (prop.major > 0) {
                info.devices[id].cc = GGML_CUDA_CC_OFFSET_AMD + prop.major * 0x100;
                info.devices[id].cc += prop.minor * 0x10;
            }
        }
        GGML_LOG_INFO("  Device %d: %s, %s (0x%x), VMM: %s, Wave Size: %d, VRAM: %zu MiB\n",
                      id, prop.name, prop.gcnArchName, info.devices[id].cc & 0xffff,
                      device_vmm ? "yes" : "no", prop.warpSize,
                      device_vram_mib);
#elif defined(GGML_USE_MUSA)
        info.devices[id].smpbo = prop.sharedMemPerBlockOptin;
        info.devices[id].cc = GGML_CUDA_CC_OFFSET_MTHREADS + prop.major * 0x100;
        info.devices[id].cc += prop.minor * 0x10;
        GGML_LOG_INFO("  Device %d: %s, compute capability %d.%d, VMM: %s, VRAM: %zu MiB\n",
                      id, prop.name, prop.major, prop.minor, device_vmm ? "yes" : "no",
                      device_vram_mib);
#else
        info.devices[id].smpbo = prop.sharedMemPerBlockOptin;
        info.devices[id].cc = 100*prop.major + 10*prop.minor;
        GGML_LOG_INFO("  Device %d: %s, compute capability %d.%d, VMM: %s, VRAM: %zu MiB\n",
                      id, prop.name, prop.major, prop.minor, device_vmm ? "yes" : "no",
                      device_vram_mib);
        std::string device_name(prop.name);
        if (device_name == "NVIDIA GeForce MX450") {
            turing_devices_without_mma.push_back({ id, device_name });
        } else if (device_name == "NVIDIA GeForce MX550") {
            turing_devices_without_mma.push_back({ id, device_name });
        } else if (device_name.substr(0, 21) == "NVIDIA GeForce GTX 16") {
            turing_devices_without_mma.push_back({ id, device_name });
        }

        // Temporary performance fix:
        // Setting device scheduling strategy for iGPUs with cc121 to "spinning" to avoid delays in cuda synchronize calls.
        // TODO: Check for future drivers the default scheduling strategy and
        // remove this call again when cudaDeviceScheduleSpin is default.
        if (prop.major == 12 && prop.minor == 1) {
            CUDA_CHECK(cudaSetDevice(physical_id));
            CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
        }

#endif  // defined(GGML_USE_HIP)
    }

    if (ggml_cuda_highest_compiled_arch(GGML_CUDA_CC_TURING) >= GGML_CUDA_CC_TURING && !turing_devices_without_mma.empty()) {
        GGML_LOG_INFO("The following devices will have suboptimal performance due to a lack of tensor cores:\n");
        for (size_t device_pos = 0; device_pos < turing_devices_without_mma.size(); device_pos++) {
            GGML_LOG_INFO(
                "  Device %d: %s\n", turing_devices_without_mma[device_pos].first, turing_devices_without_mma[device_pos].second.c_str());
        }
        GGML_LOG_INFO(
            "Consider compiling with CMAKE_CUDA_ARCHITECTURES=61-virtual;80-virtual and DGGML_CUDA_FORCE_MMQ to force the use of the Pascal code for Turing.\n");
    }

    for (int id = 0; id < info.device_count; ++id) {
        info.default_tensor_split[id] /= total_vram;
    }

    // configure logging to stdout
    // CUBLAS_CHECK(cublasLoggerConfigure(1, 1, 0, nullptr));

    if (ggml_cuda_p2p_enabled()) {
        for (int id = 0; id < info.physical_device_count; ++id) {
            CUDA_CHECK(cudaSetDevice(id));
            for (int id_other = 0; id_other < info.physical_device_count; ++id_other) {
                if (id == id_other) {
                    continue;
                }
                int can_access_peer;
                CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, id, id_other));
                if (can_access_peer) {
                    CUDA_CHECK(cudaDeviceEnablePeerAccess(id_other, 0));
                }
            }
        }
    }

    return info;
}

const ggml_cuda_device_info & ggml_cuda_info() {
    static ggml_cuda_device_info info = ggml_cuda_init();
    return info;
}

// #define DEBUG_CUDA_MALLOC

// buffer pool for cuda (legacy)
struct ggml_cuda_pool_leg : public ggml_cuda_pool {
    static const int MAX_BUFFERS = 256;

    ggml_backend_cuda_context & ctx;
    int device;
    struct ggml_cuda_buffer {
        void * ptr = nullptr;
        size_t size = 0;
    };

    ggml_cuda_buffer buffer_pool[MAX_BUFFERS] = {};
    size_t pool_size = 0;

    ggml_cuda_pool_leg(ggml_backend_cuda_context & ctx, int device) :
        ctx(ctx),
        device(device) {
    }

    ~ggml_cuda_pool_leg() {
        clear_pool();
        GGML_ASSERT(pool_size == 0);
    }

    void clear_pool() {
        ggml_cuda_set_device(device);
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer & b = buffer_pool[i];
            if (b.ptr != nullptr) {
                CUDA_CHECK(cudaFree(b.ptr));
                pool_size -= b.size;
                b.ptr  = nullptr;
                b.size = 0;
            }
        }
    }

    // captured CUDA graphs can still use freed buffers, so they must be captured again
    void invalidate_cuda_graphs() {
#ifdef USE_CUDA_GRAPH
        int n_captured = 0;
        for (const auto & [key, graph] : ctx.cuda_graphs) {
            n_captured += graph->instance != nullptr;
        }
        if (n_captured > 0) {
            GGML_LOG_DEBUG(GGML_CUDA_NAME " pool[%d]: memory freed, %d captured CUDA graphs must be captured again\n", device, n_captured);
        }
        ctx.cuda_graphs.clear();
#endif // USE_CUDA_GRAPH
    }

    void * alloc(size_t size, size_t * actual_size) override {
#ifdef DEBUG_CUDA_MALLOC
        int nnz = 0;
        size_t max_size = 0;
#endif
        size_t best_diff = 1ull << 36;
        int ibest = -1;
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer& b = buffer_pool[i];
            if (b.ptr != nullptr) {
#ifdef DEBUG_CUDA_MALLOC
                ++nnz;
                if (b.size > max_size) max_size = b.size;
#endif
                if (b.size >= size) {
                    size_t diff = b.size - size;
                    if (diff < best_diff) {
                        best_diff = diff;
                        ibest = i;
                        if (!best_diff) {
                            void * ptr = b.ptr;
                            *actual_size = b.size;
                            b.ptr = nullptr;
                            b.size = 0;
                            return ptr;
                        }
                    }
                }
            }
        }
        if (ibest >= 0) {
            ggml_cuda_buffer& b = buffer_pool[ibest];
            void * ptr = b.ptr;
            *actual_size = b.size;
            b.ptr = nullptr;
            b.size = 0;
            return ptr;
        }
        void * ptr;
        size_t look_ahead_size = (size_t) (1.05 * size);
        look_ahead_size = 256 * ((look_ahead_size + 255)/256);
        ggml_cuda_set_device(device);
        cudaError_t err = ggml_cuda_device_malloc(&ptr, look_ahead_size, device);
        if (err == cudaErrorMemoryAllocation) {
            (void)cudaGetLastError();
            const size_t cached_bytes = pool_size;
            GGML_LOG_DEBUG(GGML_CUDA_NAME " pool[%d]: alloc of %.2f MiB failed, flushing %.2f MiB of cached buffers and retrying\n",
                           device, look_ahead_size/1024.0/1024.0, cached_bytes/1024.0/1024.0);
            CUDA_CHECK(cudaDeviceSynchronize());
            clear_pool();
            invalidate_cuda_graphs();
            err = ggml_cuda_device_malloc(&ptr, look_ahead_size, device);
            if (err == cudaSuccess) {
                GGML_LOG_DEBUG(GGML_CUDA_NAME " pool[%d]: retry succeeded\n", device);
            }
        }
        CUDA_CHECK(err);
        *actual_size = look_ahead_size;
        pool_size += look_ahead_size;
#ifdef DEBUG_CUDA_MALLOC
        GGML_LOG_INFO("%s[%d]: %d buffers, max_size = %u MB, pool_size = %u MB, requested %u MB\n", __func__, device, nnz,
                           (uint32_t)(max_size / 1024 / 1024), (uint32_t)(pool_size / 1024 / 1024), (uint32_t)(size / 1024 / 1024));
#endif
        return ptr;
    }

    void free(void * ptr, size_t size) override {
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer& b = buffer_pool[i];
            if (b.ptr == nullptr) {
                b.ptr = ptr;
                b.size = size;
                return;
            }
        }
        GGML_LOG_DEBUG(GGML_CUDA_NAME " buffer pool full, increase MAX_CUDA_BUFFERS\n");
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaFree(ptr));
        pool_size -= size;
        invalidate_cuda_graphs();
    }
};

// pool with virtual memory
#if defined(GGML_USE_VMM)
struct ggml_cuda_pool_vmm : public ggml_cuda_pool {
    static const size_t CUDA_POOL_VMM_MAX_SIZE = 1ull << 35; // 32 GB

    int device;
    int physical_device;
    CUdeviceptr pool_addr = 0;
    size_t pool_used = 0;
    size_t pool_size = 0;
    size_t granularity;
#if defined(GGML_USE_HIP)
    std::vector<std::pair<CUdeviceptr, size_t>> mappings;
#endif

    explicit ggml_cuda_pool_vmm(int device) :
        device(device),
        physical_device(ggml_cuda_get_physical_device(device)),
        granularity(ggml_cuda_info().devices[device].vmm_granularity) {
    }

    ~ggml_cuda_pool_vmm() {
        if (pool_addr != 0) {
#if defined(GGML_USE_HIP)
            // Workaround for https://github.com/ROCm/ROCR-Runtime/issues/285
            for (std::pair<CUdeviceptr, size_t> & mapping : mappings) {
                CU_CHECK(cuMemUnmap(mapping.first, mapping.second));
            }
#else
            CU_CHECK(cuMemUnmap(pool_addr, pool_size));
#endif
            CU_CHECK(cuMemAddressFree(pool_addr, CUDA_POOL_VMM_MAX_SIZE));
        }
    }

    void * alloc(size_t size, size_t * actual_size) override {
        // round up the allocation size to the alignment to ensure that all allocations are aligned for all data types
        const size_t alignment = 128;
        size = alignment * ((size + alignment - 1) / alignment);

        size_t avail = pool_size - pool_used;

        if (size > avail) {
            // round up to the next multiple of the granularity
            size_t reserve_size = size - avail;
            reserve_size = granularity * ((reserve_size + granularity - 1) / granularity);

            GGML_ASSERT(pool_size + reserve_size <= CUDA_POOL_VMM_MAX_SIZE);

            // allocate more physical memory
            CUmemAllocationProp prop = {};
            prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id = physical_device;
            CUmemGenericAllocationHandle handle;
            CU_CHECK(cuMemCreate(&handle, reserve_size, &prop, 0));

            // reserve virtual address space (if not already reserved)
            if (pool_addr == 0) {
                CU_CHECK(cuMemAddressReserve(&pool_addr, CUDA_POOL_VMM_MAX_SIZE, 0, 0, 0));
            }

            // map at the end of the pool
            CUdeviceptr start_ptr = (CUdeviceptr)((char *)(pool_addr) + pool_size);
            CU_CHECK(cuMemMap(start_ptr, reserve_size, 0, handle, 0));
#if defined(GGML_USE_HIP)
            mappings.push_back({start_ptr, reserve_size});
#endif

            // the memory allocation handle is no longer needed after mapping
            CU_CHECK(cuMemRelease(handle));

            // VMM Bug fix for P2P access if GGML_CUDA_P2P is set, or if NCCL build
            bool use_peer_access = ggml_cuda_p2p_enabled();
#if defined(GGML_USE_NCCL)
            use_peer_access = true;
#endif // defined(GGML_USE_NCCL)

            if (use_peer_access) {
                // NCCL implicitly enables peer access (cudaDeviceEnablePeerAccess), and
                // GGML_CUDA_P2P enables it explicitly. Unlike cudaMalloc buffers, VMM
                // allocations do not become peer-accessible from that alone, so access
                // must be granted explicitly here. With virtual devices, grant access
                // on the backing *physical* devices (deduplicated, since several
                // virtual devices can map to the same physical GPU).
                std::vector<CUmemAccessDesc> access_descs;
                bool physical_seen[GGML_CUDA_MAX_DEVICES] = {};
                const int device_count = ggml_cuda_info().device_count;
                for (int id = 0; id < device_count; ++id) {
                    const int id_physical = ggml_cuda_get_physical_device(id);
                    if (id_physical != physical_device) {
                        int can_access_peer = 0;
                        CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, id_physical, physical_device));
                        if (!can_access_peer) {
                            continue;
                        }
                    }
                    if (physical_seen[id_physical]) {
                        continue;
                    }
                    physical_seen[id_physical] = true;
                    CUmemAccessDesc access = {};
                    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                    access.location.id = id_physical;
                    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                    access_descs.push_back(access);
                }
                CU_CHECK(cuMemSetAccess(start_ptr, reserve_size, access_descs.data(), access_descs.size()));
            } else {
                // set access for non P2P
                CUmemAccessDesc access = {};
                access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                access.location.id = physical_device;
                access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                CU_CHECK(cuMemSetAccess(start_ptr, reserve_size, &access, 1));
            }

            // add to the pool
            pool_size += reserve_size;

            //printf("cuda pool[%d]: size increased to %llu MB (reserved %llu MB)\n",
            //       device, (unsigned long long) (pool_size/1024/1024),
            //       (unsigned long long) (reserve_size/1024/1024));
        }

        GGML_ASSERT(pool_addr != 0);

        void * ptr = (void *) ((CUdeviceptr)((char *)(pool_addr) + pool_used));
        *actual_size = size;
        pool_used += size;

#ifdef DEBUG_CUDA_MALLOC
        printf("cuda pool[%d]: allocated %llu bytes at %llx\n", device, (unsigned long long) size, ptr);
#endif

        return ptr;
    }

    void free(void * ptr, size_t size) override {
#ifdef DEBUG_CUDA_MALLOC
        printf("cuda pool[%d]: freed %llu bytes at %llx\n", device, (unsigned long long) size, ptr);
#endif

        pool_used -= size;

        // all deallocations must be in reverse order of the allocations
        GGML_ASSERT(ptr == (void *) ((char *)(pool_addr) + pool_used));
    }
};
#endif // defined(GGML_USE_VMM)

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(ggml_backend_cuda_context & ctx,
                                                                               int                         device,
                                                                               [[maybe_unused]] int        stream_no) {
#if defined(GGML_USE_VMM)
    if (ggml_cuda_info().devices[device].vmm) {
        return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_vmm(device));
    }
#endif // defined(GGML_USE_VMM)
    return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_leg(ctx, device));
}

// destroying a cuBLAS handle while a graph is being captured in a different thread can result in a CUDA error
// this lock is used to ensure that no cuBLAS handle is destroyed while a graph is being captured

static std::mutex ggml_cuda_lock;
static std::condition_variable ggml_cuda_lock_cv;
static std::atomic<int> ggml_cuda_lock_counter;

ggml_backend_cuda_context::~ggml_backend_cuda_context() {
    std::unique_lock<std::mutex> lock(ggml_cuda_lock);
    ggml_cuda_lock_cv.wait(lock, []{ return ggml_cuda_lock_counter.load(std::memory_order_relaxed) == 0; });

    if (copy_event != nullptr) {
        CUDA_CHECK(cudaEventDestroy(copy_event));
    }
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
        for (int j = 0; j < GGML_CUDA_MAX_STREAMS; ++j) {
            if (streams[i][j] != nullptr) {
                CUDA_CHECK(cudaStreamDestroy(streams[i][j]));
            }
            if (cublas_handles[i][j] != nullptr) {
                CUBLAS_CHECK(cublasDestroy(cublas_handles[i][j]));
            }
            if (cublas_workspaces[i][j] != nullptr) {
                CUDA_CHECK(cudaFree(cublas_workspaces[i][j]));
            }
        }
    }
}

// cuda buffer

struct ggml_backend_cuda_buffer_context {
    int device;
    void * dev_ptr = nullptr;
    std::string name;

    ggml_backend_cuda_buffer_context(int device, void * dev_ptr) :
        device(device), dev_ptr(dev_ptr),
        name(GGML_CUDA_NAME + std::to_string(device)) {
    }

    ~ggml_backend_cuda_buffer_context() {
        CUDA_CHECK(cudaFree(dev_ptr));
    }
};

static void ggml_backend_cuda_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    delete ctx;
}

static bool ggml_backend_buffer_is_cuda(ggml_backend_buffer_t buffer) {
    return buffer->iface.free_buffer == ggml_backend_cuda_buffer_free_buffer;
}

static void * ggml_backend_cuda_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    return ctx->dev_ptr;
}

static enum ggml_status ggml_backend_cuda_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    if (tensor->view_src != NULL) {
        assert(tensor->view_src->buffer->buft == buffer->buft);
        return GGML_STATUS_SUCCESS;
    }

    if (ggml_is_quantized(tensor->type) && tensor->view_src == nullptr && ggml_backend_buffer_get_usage(buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        // initialize padding to 0 to avoid possible NaN values
        const size_t original_size = ggml_nbytes(tensor);
        const size_t padded_size = ggml_backend_buft_get_alloc_size(buffer->buft, tensor);

        if (padded_size > original_size) {
            ggml_cuda_set_device(ctx->device);
            CUDA_CHECK(cudaMemsetAsync((char *)tensor->data + original_size, 0, padded_size - original_size, cudaStreamPerThread));
            CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        }
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_cuda_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemsetAsync((char *) tensor->data + offset, value, size, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpyAsync((char *) tensor->data + offset, data, size, cudaMemcpyHostToDevice, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpyAsync(data, (const char *) tensor->data + offset, size, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_set_tensor_2d(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpy2DAsync(
        (char *) tensor->data + offset, stride_tensor, data, stride_data, size, n_copies, cudaMemcpyHostToDevice, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_get_tensor_2d(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpy2DAsync(
        data, stride_data, (const char *) tensor->data + offset, stride_tensor, size, n_copies, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static bool ggml_backend_cuda_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (ggml_backend_buffer_is_cuda(src->buffer)) {
        ggml_backend_cuda_buffer_context * src_ctx = (ggml_backend_cuda_buffer_context *)src->buffer->context;
        ggml_backend_cuda_buffer_context * dst_ctx = (ggml_backend_cuda_buffer_context *)dst->buffer->context;
        // compare the backing physical devices: distinct virtual devices may share one physical GPU,
        // in which case a same-device copy (not a peer copy) is required
        const int src_physical = ggml_cuda_get_physical_device(src_ctx->device);
        const int dst_physical = ggml_cuda_get_physical_device(dst_ctx->device);
        if (src_physical == dst_physical) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(src), cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        } else {
#ifdef GGML_CUDA_NO_PEER_COPY
            return false;
#else
            CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, dst_physical, src->data, src_physical, ggml_nbytes(src), cudaStreamPerThread));
#endif
        }
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

static void ggml_backend_cuda_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemsetAsync(ctx->dev_ptr, value, buffer->size, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static const ggml_backend_buffer_i ggml_backend_cuda_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_cuda_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cuda_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_cuda_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_cuda_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cuda_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cuda_buffer_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_cuda_buffer_set_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_cuda_buffer_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_cuda_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cuda_buffer_clear,
    /* .reset           = */ NULL,
};

// cuda buffer type
struct ggml_backend_cuda_buffer_type_context {
    int device;
    std::string name;
};

static const char * ggml_backend_cuda_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_buffer_type_context * ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;

    return ctx->name.c_str();
}

static bool ggml_backend_buft_is_cuda(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cuda_buffer_type_get_name;
}

static ggml_backend_buffer_t ggml_backend_cuda_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;

    ggml_cuda_set_device(buft_ctx->device);

    void * dev_ptr;
    cudaError_t err = ggml_cuda_device_malloc(&dev_ptr, size, buft_ctx->device);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
        GGML_LOG_ERROR("%s: allocating %.2f MiB on device %d: cudaMalloc failed: %s\n", __func__, size / 1024.0 / 1024.0, buft_ctx->device, cudaGetErrorString(err));
        return nullptr;
    }

    ggml_backend_cuda_buffer_context * ctx = new ggml_backend_cuda_buffer_context(buft_ctx->device, dev_ptr);

    return ggml_backend_buffer_init(buft, ggml_backend_cuda_buffer_interface, ctx, size);
}

static size_t ggml_backend_cuda_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 128;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_cuda_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *) buft->context;

    size_t size = tensor->op == GGML_OP_FLASH_ATTN_EXT
        ? ggml_cuda_flash_attn_ext_get_alloc_size(buft_ctx->device, tensor)
        : ggml_nbytes(tensor);
    int64_t ne0 = tensor->ne[0];

    // [TAG_ALLOC_SIZE_EXPAND]
    if (ggml_is_quantized(tensor->type)) {
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            GGML_ASSERT(tensor->nb[0] == ggml_element_size(tensor));
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    return size;
}

static const ggml_backend_buffer_type_i ggml_backend_cuda_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_cuda_buffer_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_cuda_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ NULL,
    /* .get_alignment       = */ ggml_backend_cuda_buffer_type_get_alignment,
    /* .get_max_size        = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size      = */ ggml_backend_cuda_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ NULL,
    /* .is_host             = */ NULL,
};

ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    if (device >= ggml_backend_cuda_get_device_count()) {
        return nullptr;
    }

    static ggml_backend_buffer_type ggml_backend_cuda_buffer_types[GGML_CUDA_MAX_DEVICES];

    static bool ggml_backend_cuda_buffer_type_initialized = false;

    if (!ggml_backend_cuda_buffer_type_initialized) {
        for (int i = 0; i < ggml_backend_cuda_get_device_count(); i++) {
            ggml_backend_cuda_buffer_types[i] = {
                /* .iface    = */ ggml_backend_cuda_buffer_type_interface,
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), i),
                /* .context  = */ new ggml_backend_cuda_buffer_type_context{i, GGML_CUDA_NAME + std::to_string(i)},
            };
        }
        ggml_backend_cuda_buffer_type_initialized = true;
    }

    return &ggml_backend_cuda_buffer_types[device];
}

// Communication context for multi-GPU AllReduce during tensor parallelism.
//
// Created once per meta backend instance.  Resources for the selected mode
// (NCCL communicators or the internal AllReduce pipeline) are initialised
// eagerly during comm_init so any init failure surfaces at startup rather
// than mid-run.
struct ggml_backend_cuda_comm_context {
    using try_allreduce_fn = bool(*)(ggml_backend_cuda_comm_context *, struct ggml_tensor **);

    std::vector<ggml_backend_t> backends;
    std::vector<int>            dev_ids;

    // Set by the init chain (comm_init_{nccl, internal, none}) to one of
    // try_allreduce_{nccl, internal, butterfly}.  nccl needs `comms`,
    // internal needs `ar_pipeline`, butterfly needs nothing.  Per-call
    // failures return false; the meta backend's generic implementation then
    // handles that call.
    try_allreduce_fn            try_allreduce = nullptr;

    ggml_cuda_ar_pipeline *     ar_pipeline = nullptr;

#ifdef GGML_USE_NCCL
    std::vector<ncclComm_t>     comms;
#endif // GGML_USE_NCCL

    ~ggml_backend_cuda_comm_context() {
#ifdef GGML_USE_NCCL
        for (ncclComm_t comm : comms) {
            NCCL_CHECK(ncclCommDestroy(comm));
        }
#endif // GGML_USE_NCCL
        ggml_cuda_ar_pipeline_free(ar_pipeline);
    }
};

#ifdef GGML_USE_NCCL
// AllReduce via NCCL. Reduces as FP32 for small tensors and BF16 for large
// tensors (bandwidth-bound), then converts back to FP32.
static bool ggml_backend_cuda_comm_allreduce_nccl(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    const int64_t ne = ggml_nelements(tensors[0]);
    // FIXME the input of llm_graph_context::build_in_out_ids can produce a tensor with 0 elements if n_outputs == 0
    // This then causes a crash in this function
    if (ne == 0) {
        return true;
    }

    const size_t n_backends = comm_ctx->backends.size();

    for (size_t i = 0; i < n_backends; ++i) {
        GGML_ASSERT(tensors[i] != nullptr);
        GGML_ASSERT(ggml_nelements(tensors[i]) == ne);
        GGML_ASSERT(ggml_is_contiguously_allocated(tensors[i]));
    }

    // For small tensors, simply reduce them as FP32.
    // The following heuristic for how "small" a tensor should be is based on RTX 4090s connected via 16x PCIe 4.0.
    if ((n_backends <= 2 && ne < 32768) || (n_backends == 3 && ne < 131072) || (n_backends >= 4 && ne < 262144)) {
        for (size_t i = 0; i < n_backends; ++i) {
            if ((tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
                ggml_cuda_set_device(cuda_ctx->device);
                CUDA_CHECK(cudaMemsetAsync(tensors[i]->data, 0, ggml_nbytes(tensors[i]), cuda_ctx->stream()));
            }
        }
        NCCL_CHECK(ncclGroupStart());
        for (size_t i = 0; i < n_backends; ++i) {
            ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
            NCCL_CHECK(ncclAllReduce(tensors[i]->data, tensors[i]->data, ne, ncclFloat, ncclSum, comm_ctx->comms[i], cuda_ctx->stream()));
        }
        NCCL_CHECK(ncclGroupEnd());
        return true;
    }

    // For large tensors it's faster to compress them to BF16 for the reduction:
    to_bf16_cuda_t to_bf16 = ggml_get_to_bf16_cuda(GGML_TYPE_F32);
    to_fp32_cuda_t to_fp32 = ggml_get_to_fp32_cuda(GGML_TYPE_BF16);

    ggml_cuda_pool_alloc<nv_bfloat16> tmp[GGML_CUDA_MAX_DEVICES];
    for (size_t i = 0; i < n_backends; ++i) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
        tmp[i].pool = &cuda_ctx->pool();
        tmp[i].alloc(ne);

        ggml_cuda_set_device(cuda_ctx->device);
        if (tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) {
            to_bf16(tensors[i]->data, tmp[i].get(), ne, cuda_ctx->stream());
        } else {
            CUDA_CHECK(cudaMemsetAsync(tmp[i].get(), 0, ne * sizeof(nv_bfloat16), cuda_ctx->stream()));
        }
        CUDA_CHECK(cudaGetLastError());
    }

    NCCL_CHECK(ncclGroupStart());
    for (size_t i = 0; i < n_backends; ++i) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
        NCCL_CHECK(ncclAllReduce(tmp[i].get(), tmp[i].get(), ne, ncclBfloat16, ncclSum, comm_ctx->comms[i], cuda_ctx->stream()));
    }
    NCCL_CHECK(ncclGroupEnd());

    for (size_t i = 0; i < n_backends; ++i) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;

        ggml_cuda_set_device(cuda_ctx->device);
        to_fp32(tmp[i].get(), (float *) tensors[i]->data, ne, cuda_ctx->stream());
        CUDA_CHECK(cudaGetLastError());
    }

    return true;
}
#endif // GGML_USE_NCCL

// Run the internal AR pipeline.  Returns false on unsupported / failed input
// -- the caller decides whether to abort (env-forced) or fall back silently.
static bool ggml_backend_cuda_comm_allreduce_internal(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    GGML_ASSERT(comm_ctx->ar_pipeline != nullptr);

    const size_t n_backends = comm_ctx->backends.size();
    GGML_ASSERT(n_backends == 2);
    GGML_ASSERT(tensors[0] != nullptr);

    const int64_t   ne   = ggml_nelements(tensors[0]);
    const ggml_type type = tensors[0]->type;

    if (type != GGML_TYPE_F32 && type != GGML_TYPE_F16 && type != GGML_TYPE_BF16) {
        GGML_LOG_DEBUG("%s: internal unsupported: type=%d\n", __func__, (int) type);
        return false;
    }

    if (ne == 0) {
        return true;
    }

    for (size_t i = 0; i < n_backends; ++i) {
        if (tensors[i] == nullptr) {
            GGML_LOG_ERROR("%s: internal failed: tensor[%zu] is null\n", __func__, i);
            return false;
        }
        if (ggml_nelements(tensors[i]) != ne || tensors[i]->type != type) {
            GGML_LOG_ERROR("%s: internal failed: tensor[%zu] ne=%" PRId64 " type=%d expected ne=%" PRId64 " type=%d\n",
                           __func__, i, ggml_nelements(tensors[i]), (int) tensors[i]->type, ne, (int) type);
            return false;
        }
        if (!ggml_is_contiguously_allocated(tensors[i])) {
            GGML_LOG_DEBUG("%s: internal unsupported: tensor[%zu] is not contiguously allocated: ne=%" PRId64 " nbytes=%zu packed=%zu type=%d\n",
                           __func__, i, ne, ggml_nbytes(tensors[i]),
                           (size_t) ne * ggml_type_size(type) / ggml_blck_size(type), (int) type);
            return false;
        }
        if (((uintptr_t) tensors[i]->data & 0xF) != 0) {
            GGML_LOG_DEBUG("%s: internal unsupported: tensor[%zu] data pointer is not 16-byte aligned: %p type=%d ne=%" PRId64 "\n",
                           __func__, i, tensors[i]->data, (int) type, ne);
            return false;
        }
        GGML_ASSERT((ggml_nbytes(tensors[i]) & 0xF) == 0);
    }

    return ggml_cuda_ar_allreduce(comm_ctx->ar_pipeline, comm_ctx->backends.data(), tensors);
}

// ---------------------------------------------------------------------------
// Per-call dispatch -- three variants, one per backend.  Each is set as
// comm_ctx->try_allreduce by the matching init step.  Per-call failure
// returns false; the meta backend's generic implementation handles that call.
// ---------------------------------------------------------------------------

#ifdef GGML_USE_NCCL
static bool ggml_backend_cuda_comm_try_allreduce_nccl(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    return ggml_backend_cuda_comm_allreduce_nccl(comm_ctx, tensors);
}
#endif // GGML_USE_NCCL

static bool ggml_backend_cuda_comm_try_allreduce_internal(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    return ggml_backend_cuda_comm_allreduce_internal(comm_ctx, tensors);
}

static bool ggml_backend_cuda_comm_try_allreduce_butterfly(
        ggml_backend_cuda_comm_context *, struct ggml_tensor **) {
    return false;
}

static void ggml_backend_cuda_comm_free(void * comm_ctx_v) {
    if (comm_ctx_v == nullptr) {
        return;
    }
    delete static_cast<ggml_backend_cuda_comm_context *>(comm_ctx_v);
}

// ---------------------------------------------------------------------------
// Init -- chained nccl -> internal -> none.  Each step tries to bring up its
// resource; on failure it warns and recurses into the next step.
// ---------------------------------------------------------------------------
static void ggml_backend_cuda_comm_init_none(ggml_backend_cuda_comm_context * ret) {
    ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_butterfly;
}

static void ggml_backend_cuda_comm_init_internal(ggml_backend_cuda_comm_context * ret) {
    ret->ar_pipeline = ggml_cuda_ar_pipeline_init(ret->dev_ids.data(), ret->dev_ids.size());
    if (ret->ar_pipeline) {
        ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_internal;
        return;
    }

    // Clear sticky CUDA error from the failed init.
    (void) cudaGetLastError();
    GGML_LOG_WARN("internal AllReduce init failed (n_devices != 2?); "
                  "falling back to meta-backend butterfly\n");
    ggml_backend_cuda_comm_init_none(ret);
}

static void ggml_backend_cuda_comm_init_nccl(ggml_backend_cuda_comm_context * ret) {
#ifdef GGML_USE_NCCL
    // Disabling NCCL path when CUDA virtual devices are in use since NCCL requires one distinct physical GPU per rank.
    const ggml_cuda_device_info & info = ggml_cuda_info();
    if (info.device_count > info.physical_device_count) {
        GGML_LOG_WARN("NCCL disabled: virtual devices in use; "
                      "falling back to internal AllReduce\n");
        ggml_backend_cuda_comm_init_internal(ret);
        return;
    }

    const size_t n = ret->dev_ids.size();
    ret->comms.resize(n);
    ncclResult_t rc = ncclCommInitAll(ret->comms.data(), (int) n, ret->dev_ids.data());
    if (rc == ncclSuccess) {
        ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_nccl;
        return;
    }

    ret->comms.clear();
    GGML_LOG_WARN("NCCL init failed (%s); falling back to internal AllReduce\n",
                  ncclGetErrorString(rc));
#else // GGML_USE_NCCL
#ifndef GGML_USE_HIP
    GGML_LOG_WARN("NCCL not compiled in; falling back to internal AllReduce.  "
                  "Recompile with -DGGML_CUDA_NCCL=ON for best multi-GPU performance.\n");
#endif // !GGML_USE_HIP
#endif // GGML_USE_NCCL

    ggml_backend_cuda_comm_init_internal(ret);
}

// Top-level init.  Picks one of the three init paths based on
// GGML_CUDA_ALLREDUCE (or the platform default) and lets the chain handle
// any fallback.  Unrecognised env values warn and fall through to the
// platform default.
static void * ggml_backend_cuda_comm_init(ggml_backend_t * backends, size_t n_backends) {
    for (size_t i = 0; i < n_backends; i++) {
        if (!ggml_backend_is_cuda(backends[i])) {
            return nullptr;
        }
    }

    auto * ret = new ggml_backend_cuda_comm_context;
    ret->backends.assign(backends, backends + n_backends);
    ret->dev_ids.reserve(n_backends);
    for (size_t i = 0; i < n_backends; i++) {
        ret->dev_ids.push_back(static_cast<ggml_backend_cuda_context *>(backends[i]->context)->device);
    }

    const char * env = getenv("GGML_CUDA_ALLREDUCE");
    if (!env) {
        // Platform default: Linux uses NCCL, otherwise (generally Windows) internal
#if defined(__linux__)
        ggml_backend_cuda_comm_init_nccl(ret);
#else
        ggml_backend_cuda_comm_init_internal(ret);
#endif // defined(__linux__)
    } else {
        std::string env_str(env);
        if (env_str == "nccl") {
            ggml_backend_cuda_comm_init_nccl(ret);
        } else if (env_str == "internal") {
            ggml_backend_cuda_comm_init_internal(ret);
        } else if (env_str == "none") {
            ggml_backend_cuda_comm_init_none(ret);
        } else {
            GGML_LOG_WARN("unknown GGML_CUDA_ALLREDUCE value: %s\n", env);
            ggml_backend_cuda_comm_init_none(ret);
        }
    }

    return ret;
}

// Top-level dispatch -- calls the function pointer chosen by comm_init.
// Returns false to let the meta-backend's butterfly run.
static bool ggml_backend_cuda_comm_allreduce_tensor(void * comm_ctx_v, struct ggml_tensor ** tensors) {
    if (comm_ctx_v == nullptr) {
        return false;
    }
    auto * comm_ctx = static_cast<ggml_backend_cuda_comm_context *>(comm_ctx_v);
    return comm_ctx->try_allreduce(comm_ctx, tensors);
}

// host buffer type

static const char * ggml_backend_cuda_host_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return GGML_CUDA_NAME "_Host";

    GGML_UNUSED(buft);
}

static bool ggml_backend_buft_is_cuda_host(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cuda_host_buffer_type_name;
}

static void ggml_backend_cuda_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    CUDA_CHECK(cudaFreeHost(buffer->context));
}

static void * ggml_cuda_host_malloc(size_t size) {
    if (ggml_cuda_pinned_memory_disabled()) {
        return nullptr;
    }

    void * ptr = nullptr;
    cudaError_t err = cudaMallocHost((void **) &ptr, size);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
        GGML_LOG_DEBUG("%s: failed to allocate %.2f MiB of pinned memory: %s\n", __func__,
                           size / 1024.0 / 1024.0, cudaGetErrorString(err));
        return nullptr;
    }

    return ptr;
}

// strixllama: an integrated GPU computes on tensors in its host buffers in place (supports_buft), so quantized weights
// there need what a device buffer gives them: rows padded to MATRIX_ROW_PADDING, the padding zeroed. MMQ reads each
// matrix's last row that far; without it, with experts in system memory (-ot ...=ROCm_Host), the 640-wide down experts
// read the next tensor's bytes as block scales and every batch MMQ took (17 tokens up) came out NaN
static size_t ggml_backend_cuda_host_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    size_t size = ggml_nbytes(tensor);
    if (ggml_is_quantized(tensor->type) && tensor->ne[0] % MATRIX_ROW_PADDING != 0) {
        size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - tensor->ne[0] % MATRIX_ROW_PADDING);
    }
    return size;

    GGML_UNUSED(buft);
}

static enum ggml_status ggml_backend_cuda_host_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    if (tensor->view_src == nullptr && ggml_is_quantized(tensor->type) &&
            ggml_backend_buffer_get_usage(buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t original_size = ggml_nbytes(tensor);
        const size_t padded_size   = ggml_backend_buft_get_alloc_size(buffer->buft, tensor);
        if (padded_size > original_size) {
            memset((char *) tensor->data + original_size, 0, padded_size - original_size);
        }
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_backend_buffer_t ggml_backend_cuda_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * ptr = ggml_cuda_host_malloc(size);

    if (ptr == nullptr) {
        // fallback to cpu buffer
        return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    }

    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft = buft;
    buffer->iface.free_buffer = ggml_backend_cuda_host_buffer_free_buffer;
    buffer->iface.init_tensor = ggml_backend_cuda_host_buffer_init_tensor;

    return buffer;
}

static size_t ggml_backend_cuda_host_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    // very large pinned allocations can fail (e.g. on Windows) and fall back to pageable memory, smaller ones succeed
    return 4ull*1024*1024*1024;
}

ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type() {
    static struct ggml_backend_buffer_type ggml_backend_cuda_buffer_type_host = {
        /* .iface    = */ {
            /* .get_name            = */ ggml_backend_cuda_host_buffer_type_name,
            /* .alloc_buffer        = */ ggml_backend_cuda_host_buffer_type_alloc_buffer,
            /* .alloc_buffer_n      = */ NULL,
            /* .get_alignment       = */ ggml_backend_cpu_buffer_type()->iface.get_alignment,
            /* .get_max_size        = */ ggml_backend_cuda_host_buffer_type_get_max_size,
            /* .get_alloc_size      = */ ggml_backend_cuda_host_buffer_type_get_alloc_size,
            /* .get_alloc_size_n    = */ NULL,
            /* .is_host             = */ ggml_backend_cpu_buffer_type()->iface.is_host,
        },
        /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), 0),
        /* .context  = */ nullptr,
    };

    return &ggml_backend_cuda_buffer_type_host;
}

//static bool ggml_backend_buffer_is_cuda_host(ggml_backend_buffer_t buffer) {
//    return buffer->buft->iface.get_name == ggml_backend_cuda_host_buffer_type_name;
//}

/// kernels

typedef void (*ggml_cuda_op_mul_mat_t)(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

static __global__ void k_compute_batched_ptrs(
        const void * src0_as_f16, const void * src1_as_f16, char * dst,
        const void ** ptrs_src, void ** ptrs_dst,
        int64_t ne12, int64_t ne13,
        int64_t ne23,
        size_t  nb02, size_t  nb03,
        size_t  nb12, size_t  nb13,
        size_t  nbd2, size_t  nbd3,
        int64_t r2,   int64_t r3) {
    const int64_t i13 = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t i12 = blockIdx.y * blockDim.y + threadIdx.y;

    if (i13 >= ne13 || i12 >= ne12) {
        return;
    }

    const int64_t i03 = i13 / r3;
    const int64_t i02 = i12 / r2;

    ptrs_src[0*ne23 + i12 + i13*ne12] = (const char *) src0_as_f16 + i02*nb02 + i03*nb03;
    ptrs_src[1*ne23 + i12 + i13*ne12] = (const char *) src1_as_f16 + i12*nb12 + i13*nb13;
    ptrs_dst[0*ne23 + i12 + i13*ne12] = (      char *)         dst + i12*nbd2 + i13*nbd3;
}

// Type traits for mapping ggml types to CUDA/cuBLAS types
template<ggml_type T>
struct batched_mul_mat_traits;

template<>
struct batched_mul_mat_traits<GGML_TYPE_F32> {
    using cuda_type = float;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    static inline const cudaDataType_t data_type = CUDA_R_32F;
    static inline const ggml_type ggml_type_val = GGML_TYPE_F32;
    static inline const float alpha = 1.0f;
    static inline const float beta = 0.0f;
    static inline const void* get_alpha() { static const float val = alpha; return &val; }
    static inline const void* get_beta() { static const float val = beta; return &val; }
    static inline auto convert(ggml_type src_type) { return ggml_get_to_fp32_cuda(src_type); }
    static inline auto convert_nc(ggml_type src_type) { return ggml_get_to_fp32_nc_cuda(src_type); }
};

template<>
struct batched_mul_mat_traits<GGML_TYPE_BF16> {
    using cuda_type = nv_bfloat16;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    static inline const cudaDataType_t data_type = CUDA_R_16BF;
    static inline const ggml_type ggml_type_val = GGML_TYPE_BF16;
    static inline const float alpha = 1.0f;
    static inline const float beta = 0.0f;
    static inline const void* get_alpha() { static const float val = alpha; return &val; }
    static inline const void* get_beta() { static const float val = beta; return &val; }
    static inline auto convert(ggml_type src_type) { return ggml_get_to_bf16_cuda(src_type); }
    static inline auto convert_nc(ggml_type src_type) { return ggml_get_to_bf16_nc_cuda(src_type); }
};

template<>
struct batched_mul_mat_traits<GGML_TYPE_F16> {
    using cuda_type = half;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
    static inline const cudaDataType_t data_type = CUDA_R_16F;
    static inline const ggml_type ggml_type_val = GGML_TYPE_F16;
    static inline const half alpha = 1.0;
    static inline const half beta = 0.0;
    static inline const void* get_alpha() { static const half val = alpha; return &val; }
    static inline const void* get_beta() { static const half val = beta; return &val; }
    static inline auto convert(ggml_type src_type) { return ggml_get_to_fp16_cuda(src_type); }
    static inline auto convert_nc(ggml_type src_type) { return ggml_get_to_fp16_nc_cuda(src_type); }
};

#if defined(GGML_USE_HIP)
// strixllama: an F32 product of one column without rocBLAS, which faults in host code on those (see
// ggml_cuda_mul_mat_cublas): a warp a row, rows of any length, src0 broadcast over dims 2 and 3 as the GEMMs do
static __global__ void ggml_cuda_mul_mat_f32_one_col(const float * x, const float * y, float * dst, const int64_t ne00,
        const int64_t s01, const int64_t s02, const int64_t s03, const int64_t s12, const int64_t s13,
        const int64_t r2, const int64_t r3, const int64_t sd2, const int64_t sd3) {
    const int64_t row = blockIdx.x;
    const int64_t i12 = blockIdx.y;
    const int64_t i13 = blockIdx.z;
    const float * xr = x + (i13/r3)*s03 + (i12/r2)*s02 + row*s01;
    const float * yc = y + i13*s13 + i12*s12;
    float sum = 0.0f;
    for (int64_t k = threadIdx.x; k < ne00; k += WARP_SIZE) {
        sum += xr[k]*yc[k];
    }
    sum = warp_reduce_sum(sum);
    if (threadIdx.x == 0) {
        dst[i13*sd3 + i12*sd2 + row] = sum;
    }
}
#endif // defined(GGML_USE_HIP)

template<ggml_type compute_type>
static void ggml_cuda_mul_mat_cublas_impl(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    using traits = batched_mul_mat_traits<compute_type>;
    using cuda_t = typename traits::cuda_type;

    // F32 chunks can have padding between rows to preserve the original destination stride.
    GGML_ASSERT(ggml_is_contiguous(dst) ||
        (compute_type == GGML_TYPE_F32 && ggml_is_contiguous_1(dst)));

    GGML_TENSOR_BINARY_OP_LOCALS

    const int64_t ldc = nb1 / sizeof(float);
    const int64_t ne_dst = ggml_nelements(dst);
    cudaStream_t main_stream = ctx.stream();
    cublasHandle_t cublas_h = ctx.cublas_handle();

    const size_t src0_ts = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == src0_ts);
    int64_t s01 = nb01 / src0_ts;
    int64_t s02 = nb02 / src0_ts;
    int64_t s03 = nb03 / src0_ts;

    const size_t src1_ts = ggml_type_size(src1->type);
    GGML_ASSERT(nb10 == src1_ts);
    int64_t s11 = nb11 / src1_ts;
    int64_t s12 = nb12 / src1_ts;
    int64_t s13 = nb13 / src1_ts;

    float * dst_ddf = (float *) dst->data;

    const cuda_t * src0_ptr = nullptr;
    const cuda_t * src1_ptr = nullptr;

    ggml_cuda_pool_alloc<cuda_t> src0_alloc(ctx.pool());
    ggml_cuda_pool_alloc<cuda_t> src1_alloc(ctx.pool());

    bool is_src0_cont_2 = ggml_is_contiguous_2(src0);
    bool is_src1_cont_2 = ggml_is_contiguous_2(src1);

    if (src0->type == compute_type) {
        src0_ptr = (const cuda_t *) src0->data;
    } else {
        src0_alloc.alloc(ggml_nelements(src0));

        if (ggml_is_contiguously_allocated(src0)) {
            const auto convert_func = traits::convert(src0->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src0->data, src0_alloc.get(), ggml_nelements(src0), main_stream);
            const size_t src0_bs = ggml_blck_size(src0->type);
            s01 *= src0_bs;
            s02 *= src0_bs;
            s03 *= src0_bs;
        } else {
            const auto convert_func = traits::convert_nc(src0->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src0->data, src0_alloc.get(), ne00, ne01, ne02, ne03, s01, s02, s03, main_stream);
            s01 = ne00;
            s02 = ne01*s01;
            s03 = ne02*s02;
            is_src0_cont_2 = true;
        }
        src0_ptr = src0_alloc.get();
    }

    if (src1->type == compute_type) {
        src1_ptr = (const cuda_t *) src1->data;
    } else {
        src1_alloc.alloc(ggml_nelements(src1));

        if (ggml_is_contiguously_allocated(src1)) {
            const auto convert_func = traits::convert(src1->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src1->data, src1_alloc.get(), ggml_nelements(src1), main_stream);
            const size_t src1_bs = ggml_blck_size(src1->type);
            s11 *= src1_bs;
            s12 *= src1_bs;
            s13 *= src1_bs;
        } else {
            const auto convert_func = traits::convert_nc(src1->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src1->data, src1_alloc.get(), ne10, ne11, ne12, ne13, s11, s12, s13, main_stream);
            s11 = ne10;
            s12 = ne11*s11;
            s13 = ne12*s12;
            is_src1_cont_2 = true;
        }
        src1_ptr = src1_alloc.get();
    }

    ggml_cuda_pool_alloc<cuda_t> dst_temp(ctx.pool());
    char * dst_ptr;
    size_t nbd2 = dst->nb[2];
    size_t nbd3 = dst->nb[3];

    cublasComputeType_t cu_compute_type = traits::compute_type;
    cudaDataType_t cu_data_type = traits::data_type;
    cudaDataType_t cu_data_type_a = traits::data_type;
    cudaDataType_t cu_data_type_b = traits::data_type;
    const void * alpha = traits::get_alpha();
    const void * beta = traits::get_beta();

    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    bool prefer_f32_output = false;
    if (compute_type == GGML_TYPE_F16) {
        prefer_f32_output = cc == GGML_CUDA_CC_VOLTA || GGML_CUDA_CC_IS_RDNA4(cc) || GGML_CUDA_CC_IS_CDNA(cc);
    } else if (compute_type == GGML_TYPE_BF16) {
        prefer_f32_output = !GGML_CUDA_CC_IS_RDNA3(cc) && !GGML_CUDA_CC_IS_CDNA(cc);
    }

    if (prefer_f32_output) {
        dst_ptr = (char *) dst_ddf;
        cu_compute_type = batched_mul_mat_traits<GGML_TYPE_F32>::compute_type;
        cu_data_type = batched_mul_mat_traits<GGML_TYPE_F32>::data_type;
        alpha = batched_mul_mat_traits<GGML_TYPE_F32>::get_alpha();
        beta = batched_mul_mat_traits<GGML_TYPE_F32>::get_beta();
    } else {
        if constexpr (compute_type == GGML_TYPE_F32) {
            dst_ptr = (char *) dst_ddf;  // Direct F32 output
        } else {
            dst_ptr = (char *) dst_temp.alloc(ne_dst);
            nbd2 /= sizeof(float) / sizeof(cuda_t);
            nbd3 /= sizeof(float) / sizeof(cuda_t);
        }
    }

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);

    // broadcast factors
    const int64_t r2 = ne12/ne02;
    const int64_t r3 = ne13/ne03;

#if defined(GGML_USE_HIP)
    if constexpr (compute_type == GGML_TYPE_F32) {
        if (ne11 == 1) {
            const dim3 grid(ne01, ne12, ne13);
            ggml_cuda_mul_mat_f32_one_col<<<grid, WARP_SIZE, 0, main_stream>>>((const float *) src0_ptr, (const float *) src1_ptr,
                (float *) dst_ptr, ne10, s01, s02, s03, s12, s13, r2, r3, nbd2/sizeof(float), nbd3/sizeof(float));
            CUDA_CHECK(cudaGetLastError());
            return;
        }
    }
#endif // defined(GGML_USE_HIP)

    // Theoretically cublasGemmStridedBatchedEx would always work, even for a single matrix.
    // However, for some old NVIDIA and AMD GPUs the strided/Ex GEMM is much slower,
    //     probably because the internal kernel selection logic is suboptimal.
    if (compute_type == GGML_TYPE_F32 && ne12 == 1 && ne13 == 1) {
        CUBLAS_CHECK(
            cublasSgemm(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                    ne01, ne11, ne10,
                    (const float *) alpha, (const float *) src0_ptr, s01,
                                           (const float *) src1_ptr, s11,
                    (const float *) beta,  (float       *)  dst_ptr, ldc));
    } else if (ne12 == 1 && ne13 == 1) {
        CUBLAS_CHECK(
            cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                    ne01, ne11, ne10,
                    alpha, src0_ptr, cu_data_type_a, s01,
                           src1_ptr, cu_data_type_b, s11,
                    beta,   dst_ptr, cu_data_type,   ldc,
                    cu_compute_type,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else if (r2 == 1 && r3 == 1 && is_src0_cont_2 && is_src1_cont_2) {
        // with a [0, 2, 1, 3] perm. and ne02==1 the matrix strides need to be determined from dim 3:
        const int64_t sma = ne02 == 1 ? s03 : s02;
        const int64_t smb = ne12 == 1 ? s13 : s12;

        // there is no broadcast and src0, src1 are contiguous across dims 2, 3
        // use cublasGemmStridedBatchedEx
        CUBLAS_CHECK(
        cublasGemmStridedBatchedEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, src0_ptr, cu_data_type_a, s01, sma,     // strideA
                       src1_ptr, cu_data_type_b, s11, smb,     // strideB
                beta,   dst_ptr, cu_data_type,   ldc, ne1*ldc, // strideC
                ne12*ne13,
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else {
        // use cublasGemmBatchedEx
        const int64_t ne23 = ne12*ne13;

        ggml_cuda_pool_alloc<const void *> ptrs_src(ctx.pool(), 2*ne23);
        ggml_cuda_pool_alloc<      void *> ptrs_dst(ctx.pool(), 1*ne23);

        const size_t src_type_size = sizeof(cuda_t);

        const int threads_x = 16;
        const int threads_y = 16;
        const dim3 block_dims(threads_x, threads_y);

        const dim3 grid_dims(
            (ne13 + threads_x - 1) / threads_x,
            (ne12 + threads_y - 1) / threads_y
        );
        k_compute_batched_ptrs<<<grid_dims, block_dims, 0, main_stream>>>(
                src0_ptr, src1_ptr, dst_ptr,
                ptrs_src.get(), ptrs_dst.get(),
                ne12, ne13,
                ne23,
                s02*src_type_size, s03*src_type_size,
                s12*src_type_size, s13*src_type_size,
                nbd2, nbd3,
                r2, r3);

        CUDA_CHECK(cudaGetLastError());

        CUBLAS_CHECK(
        cublasGemmBatchedEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, (const void **) (ptrs_src.get() + 0*ne23), cu_data_type_a, s01,
                       (const void **) (ptrs_src.get() + 1*ne23), cu_data_type_b, s11,
                beta,  (      void **) (ptrs_dst.get() + 0*ne23), cu_data_type,   ldc,
                ne23,
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    // Convert output back to F32 if needed
    if (cu_data_type != CUDA_R_32F) {
        const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(traits::ggml_type_val);
        to_fp32_cuda(dst_temp.get(), dst_ddf, ne_dst, main_stream);
    }
}

static void ggml_cuda_mul_mat_cublas(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const ggml_prec prec = (ggml_prec) ggml_get_op_params_i32(dst, 0);
    ggml_type compute_type = src0->type;
    if (ggml_is_quantized(compute_type)) {
        compute_type = fast_fp16_hardware_available(cc) ? GGML_TYPE_F16 : GGML_TYPE_F32;
    } else if (compute_type == GGML_TYPE_F16 && !fast_fp16_hardware_available(cc)) {
        compute_type = GGML_TYPE_F32;
    } else if (compute_type == GGML_TYPE_BF16 && !fast_bf16_hardware_available(cc)) {
        if (GGML_CUDA_CC_IS_AMD(cc) && src1->ne[1] > 32) {
            compute_type = GGML_TYPE_F32;
        }
        if (GGML_CUDA_CC_IS_NVIDIA(cc) && src1->ne[1] > (cc >= GGML_CUDA_CC_VOLTA ? 8 : 128)) {
            compute_type = GGML_TYPE_F32;
        }
    }
    // F16 is the only compute type that can not satisfy a request for BF16
    if (prec == GGML_PREC_BF16 && compute_type == GGML_TYPE_F16) {
        compute_type = fast_bf16_hardware_available(cc) ? GGML_TYPE_BF16 : GGML_TYPE_F32;
    } else if (prec == GGML_PREC_F32) {
        compute_type = GGML_TYPE_F32;
    }

    const char * env_c = getenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE");
    if (env_c != nullptr) {
        std::string env_cpp = env_c;
        for (char & c : env_cpp) {
            c = std::tolower(c);
        }
        if (env_cpp == "f32" || env_cpp == "fp32") {
            compute_type = GGML_TYPE_F32;
        } else if (env_cpp == "f16" || env_cpp == "fp16") {
            compute_type = GGML_TYPE_F16;
        } else if (env_cpp == "bf16") {
            compute_type = GGML_TYPE_BF16;
        } else if (env_cpp != "auto") {
            GGML_LOG_WARN("%s: unknown value for GGML_CUDA_CUBLAS_COMPUTE_TYPE: %s", __func__, env_cpp.c_str());
        }
    }

#if defined(GGML_USE_HIP)
    // strixllama: rocBLAS (TheRock 10.2 nightly, Windows) faults in host code on an F32 GEMM of one column - sgemm and
    // GemmEx and the batched forms alike (test-backend-ops MUL_MAT f32 x bf16, n = 1, as ggml_conv_1d_dw emits it). One
    // column of an F32 / F16 / BF16 weight takes the vector kernel instead, a non-F32 column widened to F32 first; what
    // the vector kernel does not take (an odd row length) goes through ggml_cuda_mul_mat_f32_one_col in the F32 impl
    if (src1->ne[1] == 1 && compute_type == GGML_TYPE_BF16) {
        compute_type = GGML_TYPE_F32;
    }
    if (compute_type == GGML_TYPE_F32 && src1->ne[1] == 1 && dst->type == GGML_TYPE_F32 && ggml_is_contiguous(src1) &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16) &&
            ggml_cuda_should_use_mmvf(src0->type, cc, ggml_cuda_info().devices[ctx.device].warp_size, src0->ne, src0->nb, 1)) {
        if (src1->type == GGML_TYPE_F32) {
            ggml_cuda_mul_mat_vec_f(ctx, src0, src1, nullptr, dst);
            return;
        }
        const to_fp32_cuda_t to_fp32 = ggml_get_to_fp32_cuda(src1->type);
        if (to_fp32 != nullptr) {
            ggml_cuda_pool_alloc<float> src1_f32(ctx.pool(), ggml_nelements(src1));
            to_fp32(src1->data, src1_f32.get(), ggml_nelements(src1), ctx.stream());
            ggml_tensor src1_col = *src1;
            src1_col.type  = GGML_TYPE_F32;
            src1_col.data  = src1_f32.get();
            src1_col.nb[0] = sizeof(float);
            for (int i = 1; i < GGML_MAX_DIMS; ++i) {
                src1_col.nb[i] = src1_col.nb[i - 1]*src1_col.ne[i - 1];
            }
            ggml_cuda_mul_mat_vec_f(ctx, src0, &src1_col, nullptr, dst);
            return;
        }
    }
#endif // defined(GGML_USE_HIP)

    constexpr size_t max_src0_convert_size = 512ull * 1024 * 1024;
    const size_t src0_f32_size = ggml_nelements(src0) * sizeof(float);

    if (compute_type == GGML_TYPE_F32 &&
            (src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16) &&
            src0_f32_size > max_src0_convert_size) {
        const size_t f32_row_size = src0_f32_size / src0->ne[1];
        const int64_t rows_per_chunk = std::max<int64_t>(1, (int64_t) (max_src0_convert_size / f32_row_size));

        if (rows_per_chunk < src0->ne[1]) {
            ggml_tensor src0_chunk = *src0;
            ggml_tensor dst_chunk = *dst;

            for (int64_t i01 = 0; i01 < src0->ne[1]; i01 += rows_per_chunk) {
                src0_chunk.ne[1] = std::min(rows_per_chunk, src0->ne[1] - i01);
                src0_chunk.data = (char *) src0->data + i01*src0->nb[1];

                dst_chunk.ne[0] = src0_chunk.ne[1];
                dst_chunk.data = (char *) dst->data + i01*dst->nb[0];

                ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F32>(ctx, &src0_chunk, src1, &dst_chunk);
            }
            return;
        }
    }

    switch (compute_type) {
        case GGML_TYPE_F32:
            ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F32>(ctx, src0, src1, dst);
            break;
        case GGML_TYPE_BF16:
            ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_BF16>(ctx, src0, src1, dst);
            break;
        case GGML_TYPE_F16:
            ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F16>(ctx, src0, src1, dst);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

static bool ggml_cuda_should_fuse_mul_mat(const ggml_tensor * ffn_up,
                                          const ggml_tensor * ffn_gate,
                                          const ggml_tensor * glu,
                                          const ggml_tensor * ffn_up_bias = nullptr,
                                          const ggml_tensor * ffn_gate_bias = nullptr,
                                          const ggml_tensor * ffn_up_scale = nullptr,
                                          const ggml_tensor * ffn_gate_scale = nullptr) {
    const bool has_bias = ffn_up_bias != nullptr || ffn_gate_bias != nullptr;
    const bool has_scale = ffn_up_scale != nullptr || ffn_gate_scale != nullptr;

    if (has_bias && (!ffn_up_bias || !ffn_gate_bias)) {
        return false;
    }
    if (has_scale && (!ffn_up_scale || !ffn_gate_scale)) {
        return false;
    }

    const bool is_mul_mat     = ffn_up->op == GGML_OP_MUL_MAT     && ffn_gate->op == GGML_OP_MUL_MAT     && glu->op == GGML_OP_GLU;
    const bool is_mul_mat_id  = ffn_up->op == GGML_OP_MUL_MAT_ID  && ffn_gate->op == GGML_OP_MUL_MAT_ID  && glu->op == GGML_OP_GLU;

    GGML_ASSERT(ffn_up && ffn_gate && glu);

    if (!is_mul_mat && !is_mul_mat_id) {
        return false;
    }

    const ggml_op expected_bias_op = is_mul_mat ? GGML_OP_ADD : GGML_OP_ADD_ID;
    const ggml_tensor * ffn_up_bias_src   = has_scale ? ffn_up_scale   : ffn_up;
    const ggml_tensor * ffn_gate_bias_src = has_scale ? ffn_gate_scale : ffn_gate;
    const ggml_tensor * ffn_up_out        = has_bias ? ffn_up_bias     : ffn_up_bias_src;
    const ggml_tensor * ffn_gate_out      = has_bias ? ffn_gate_bias   : ffn_gate_bias_src;

    if (glu->src[0] != ffn_gate_out || glu->src[1] != ffn_up_out) {
        return false;
    }

    if (has_scale) {
        if (ffn_up_scale->op != GGML_OP_MUL || ffn_gate_scale->op != GGML_OP_MUL) {
            return false;
        }
        const bool up_has_mm   = ffn_up_scale->src[0] == ffn_up || ffn_up_scale->src[1] == ffn_up;
        const bool gate_has_mm = ffn_gate_scale->src[0] == ffn_gate || ffn_gate_scale->src[1] == ffn_gate;
        if (!up_has_mm || !gate_has_mm) {
            return false;
        }
    }

    if (has_bias) {
        if (ffn_up_bias->op != expected_bias_op || ffn_gate_bias->op != expected_bias_op) {
            return false;
        }

        if (expected_bias_op == GGML_OP_ADD) {
            const bool up_has_mul   = ffn_up_bias->src[0] == ffn_up_bias_src || ffn_up_bias->src[1] == ffn_up_bias_src;
            const bool gate_has_mul = ffn_gate_bias->src[0] == ffn_gate_bias_src || ffn_gate_bias->src[1] == ffn_gate_bias_src;
            if (!up_has_mul || !gate_has_mul) {
                return false;
            }
        } else { // GGML_OP_ADD_ID
            if (ffn_up_bias->src[0] != ffn_up_bias_src || ffn_gate_bias->src[0] != ffn_gate_bias_src) {
                return false;
            }
            if (ffn_up_bias->src[2] != ffn_up->src[2] || ffn_gate_bias->src[2] != ffn_gate->src[2]) {
                return false;
            }
        }
    }

    if (ffn_up->src[0]->type != ffn_gate->src[0]->type || !ggml_are_same_shape(ffn_up->src[0], ffn_gate->src[0]) ||
        !ggml_are_same_stride(ffn_up->src[0], ffn_gate->src[0])) {
        return false;
    }

    if (ffn_up->src[1] != ffn_gate->src[1]) {
        return false;
    }

    if (is_mul_mat_id && ffn_up->src[2] != ffn_gate->src[2]) {
        return false;
    }

    static constexpr std::array<ggml_glu_op, 4> valid_glu_ops = { GGML_GLU_OP_SWIGLU, GGML_GLU_OP_GEGLU, GGML_GLU_OP_SWIGLU_OAI, GGML_GLU_OP_SWIGLU_CLAMP };

    if (std::find(valid_glu_ops.begin(), valid_glu_ops.end(), ggml_get_glu_op(glu)) == valid_glu_ops.end()) {
        return false;
    }

    if (const bool swapped = ggml_get_op_params_i32(glu, 1); swapped) {
        return false;
    }

    return true;
}

static bool ggml_cuda_should_fuse_mul_mat_vec_f(const ggml_tensor * tensor) {
    ggml_tensor *       src0 = tensor->src[0];
    ggml_tensor *       src1 = tensor->src[1];
    const ggml_tensor * dst  = tensor;

    const bool is_mul_mat_id = tensor->op == GGML_OP_MUL_MAT_ID;

    bool use_mul_mat_vec_f =
        (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16) &&
        src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;

    const int cc        = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    use_mul_mat_vec_f = use_mul_mat_vec_f && ggml_cuda_should_use_mmvf(src0->type, cc, warp_size, src0->ne, src0->nb, is_mul_mat_id ? src1->ne[2] : src1->ne[1]);

    //we only support fusion for ncols_dst = 1
    if (tensor->op == GGML_OP_MUL_MAT && dst->ne[1] != 1) {
        return false;
    }

    if (tensor->op == GGML_OP_MUL_MAT_ID && dst->ne[2] != 1) {
        return false;
    }

    return use_mul_mat_vec_f;
}

static bool ggml_cuda_should_fuse_mul_mat_vec_q(const ggml_tensor * tensor) {
    ggml_tensor *       src0 = tensor->src[0];
    ggml_tensor *       src1 = tensor->src[1];
    const ggml_tensor * dst  = tensor;

    const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
                                   ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) &&
                                   src0->view_src;

    bool use_mul_mat_vec_q = ggml_is_quantized(src0->type) && !bad_padding_clear && src1->type == GGML_TYPE_F32 &&
                             dst->type == GGML_TYPE_F32 && src1->ne[1] <= MMVQ_MAX_BATCH_SIZE;

    // fusion is not universally faster on Pascal
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (cc <= GGML_CUDA_CC_PASCAL) {
        return false;
    }
    //we only support fusion for ncols_dst = 1
    if (tensor->op == GGML_OP_MUL_MAT && dst->ne[1] != 1) {
        return false;
    }

    if (tensor->op == GGML_OP_MUL_MAT_ID && dst->ne[2] > get_mmvq_mmid_max_batch(src0->type, cc)) {
        return false;
    }

    return use_mul_mat_vec_q;
}

// strixllama: a narrow F32 weight [K x M], M <= 64, against more than 64 columns - the hyper-connection inject (M 4,
// K 10240), the GDN beta and alpha (48, 2560), the shared expert's gate (1, 2560): Unsloth's files keep these small
// weights F32. The tile GEMM pads M to 128 and the library GEMM (under 512 columns) ran them at ~50 GB/s of the
// activations they read. M <= 8: a block per column streams the column once with the M sums in registers, the partial
// sums reduced in a fixed order (per wave, then the eight waves in turn) - the inject at 2030 columns 0.37 ms, as a
// plain read of the same bytes, against 1.40 in the tile GEMM; the gate 0.09 against 0.24 in the transposed vector
// kernel. Wider: the WMMA kernel of mmb.cu (beta/alpha 0.36 -> 0.14 ms). A column's sums do not depend on the rest
// of the batch. STRIX_SKINNY_F32=0 leaves them to the tile GEMM and the library.
template <int M>
static __global__ void k_skinny_f32_direct(const float * __restrict__ w, const float * __restrict__ x, float * __restrict__ d,
        const int K, const int64_t sw, const int64_t sx, const int64_t sd) {
    __shared__ float red[8][M];
    const int t = blockIdx.x;
    const float4 * xc = (const float4 *) (x + (int64_t) t * sx);
    float acc[M];
#pragma unroll
    for (int m = 0; m < M; ++m) {
        acc[m] = 0.0f;
    }
    for (int k4 = threadIdx.x; k4 < K / 4; k4 += 256) {
        const float4 xv = xc[k4];
#pragma unroll
        for (int m = 0; m < M; ++m) {
            const float4 wv = ((const float4 *) (w + (int64_t) m * sw))[k4];
            acc[m] = fmaf(wv.x, xv.x, acc[m]);
            acc[m] = fmaf(wv.y, xv.y, acc[m]);
            acc[m] = fmaf(wv.z, xv.z, acc[m]);
            acc[m] = fmaf(wv.w, xv.w, acc[m]);
        }
    }
    const int lane = threadIdx.x % 32, wave = threadIdx.x / 32;
#pragma unroll
    for (int m = 0; m < M; ++m) {
        const float v = warp_reduce_sum<32>(acc[m]);
        if (lane == 0) {
            red[wave][m] = v;
        }
    }
    __syncthreads();
    if (threadIdx.x < M) {
        float v = red[0][threadIdx.x];
#pragma unroll
        for (int i = 1; i < 8; ++i) {
            v += red[i][threadIdx.x];
        }
        d[(int64_t) t * sd + threadIdx.x] = v;
    }
}

// strixllama: an F16 x F32 product with a tiny inner dimension - the compact sparse-attention scorer's block membership,
// K = the sequences of a ubatch - went to hipBLAS, which loads a kernel from disk the first time it meets a shape: a
// 272 ms stall in the first step three conversations decode together, and again as their blocks grow. One thread an
// output element, the K products summed in order (STRIX_SMALL_K=0: as before). The membership is 0 or 1, so exact.
static __global__ void k_small_k_f16(const half * __restrict__ a, const float * __restrict__ b, float * __restrict__ d,
        const int K, const int M, const int64_t sa, const int64_t sb, const int64_t sd) {
    const int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) {
        return;
    }
    const half  * ap = a + (int64_t) m * sa;
    const float * bp = b + (int64_t) blockIdx.y * sb;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc = fmaf(__half2float(ap[k]), bp[k], acc);
    }
    d[(int64_t) blockIdx.y * sd + m] = acc;
}

static bool ggml_cuda_small_k(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    static const bool on = !getenv("STRIX_SMALL_K") || atoi(getenv("STRIX_SMALL_K")) != 0;
    const int64_t K = src0->ne[0], M = src0->ne[1], N = src1->ne[1];
    if (!on || src0->type != GGML_TYPE_F16 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || K > 32 ||
            src1->ne[0] != K || dst->ne[0] != M || dst->ne[1] != N || N > 65535 || M > INT_MAX ||
            src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 || dst->ne[2] != 1 || dst->ne[3] != 1 ||
            src0->nb[0] != sizeof(half) || src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float) ||
            src0->nb[1] % sizeof(half) != 0 || src1->nb[1] % sizeof(float) != 0 || dst->nb[1] % sizeof(float) != 0) {
        return false;
    }
    const dim3 grid((unsigned) ((M + 255) / 256), (unsigned) N), block(256);
    k_small_k_f16<<<grid, block, 0, ctx.stream()>>>((const half *) src0->data, (const float *) src1->data, (float *) dst->data,
        (int) K, (int) M, src0->nb[1] / sizeof(half), src1->nb[1] / sizeof(float), dst->nb[1] / sizeof(float));
    CUDA_CHECK(cudaGetLastError());
    return true;
}

// strixllama: whether ggml_cuda_skinny_f32 takes this product (and, with narrow, whether it hands it to the narrow WMMA kernel)
static bool ggml_cuda_skinny_f32_ok(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, bool * narrow) {
    static const bool on = !getenv("STRIX_SKINNY_F32") || atoi(getenv("STRIX_SKINNY_F32")) != 0;
    const int64_t K = src0->ne[0], M = src0->ne[1], T = src1->ne[1];
    if (!on || src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
            M > 64 || T <= 64 || T > INT_MAX || K % 4 != 0 || src1->ne[0] != K || dst->ne[0] != M || dst->ne[1] != T ||
            ggml_nrows(src0) != M || src1->ne[2] != 1 || src1->ne[3] != 1 || dst->ne[2] != 1 || dst->ne[3] != 1 ||
            src0->nb[0] != sizeof(float) || src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float) ||
            src0->nb[1] % 16 != 0 || src1->nb[1] % 16 != 0 || ((uintptr_t) src0->data) % 16 != 0 || ((uintptr_t) src1->data) % 16 != 0 ||
            ggml_cuda_info().devices[ctx.device].warp_size != 32) {
        return false;
    }
    if (narrow) { *narrow = M > 8 || (M != 1 && M != 2 && M != 4 && M != 8); }
    return true;
}

static bool ggml_cuda_skinny_f32(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    bool narrow = false;
    if (!ggml_cuda_skinny_f32_ok(ctx, src0, src1, dst, &narrow)) {
        return false;
    }
    const int64_t K = src0->ne[0], M = src0->ne[1], T = src1->ne[1];
    if (narrow) {
        return ggml_cuda_mmb_f32_narrow(ctx, src0, src1, dst);
    }
    const int64_t sw = src0->nb[1] / sizeof(float), sx = src1->nb[1] / sizeof(float), sd = dst->nb[1] / sizeof(float);
    const float * w = (const float *) src0->data;
    const float * x = (const float *) src1->data;
    float * d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();
    const dim3 grid((unsigned) T), block(256);
    switch (M) {
        case 1: k_skinny_f32_direct<1><<<grid, block, 0, stream>>>(w, x, d, (int) K, sw, sx, sd); break;
        case 2: k_skinny_f32_direct<2><<<grid, block, 0, stream>>>(w, x, d, (int) K, sw, sx, sd); break;
        case 4: k_skinny_f32_direct<4><<<grid, block, 0, stream>>>(w, x, d, (int) K, sw, sx, sd); break;
        default: k_skinny_f32_direct<8><<<grid, block, 0, stream>>>(w, x, d, (int) K, sw, sx, sd); break;
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

// strixllama: the hc_inject that reads a fused combine's xn [hc * n_embd, T] with F32 weights [hc * n_embd, hc] makes
// one more pass over the F32 xn (84 MB at 2K tokens) for four dot products a token. The combine computes them per
// stream while xn is in registers; the MUL_MAT then only sums the streams (ggml_cuda_hc_inject_take). Prefill-sized
// batches only (the skinny kernel's range): decode keeps its numerics.
struct ggml_cuda_hc_inject_pending {
    const ggml_tensor * node    = nullptr;
    const void *        xn_data = nullptr;
    const float *       part    = nullptr;
};
static ggml_cuda_hc_inject_pending g_hc_inject;

// the pending inject's MUL_MAT: sum the streams' partials the combine wrote
static bool ggml_cuda_hc_inject_take(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, ggml_tensor * dst) {
    if (!g_hc_inject.node || dst != g_hc_inject.node) {
        return false;
    }
    const bool ok = src1->data == g_hc_inject.xn_data;
    const float * part = g_hc_inject.part;
    g_hc_inject = {};
    if (!ok) {
        return false;
    }
    ggml_cuda_hc_inject_reduce(ctx, part, dst);
    return true;
}

// strixllama: upstream's fusion of the routed experts' gate/up with the shared expert's (MMVQ, and the node rotation in
// graph_optimize that feeds it) changes the decode graph's kernels; off on HIP until it is measured on gfx1151
// (STRIX_SHEXP_FUSE=1 turns it on)
static bool ggml_cuda_shexp_fuse() {
#ifdef GGML_USE_HIP
    static const bool on = getenv("STRIX_SHEXP_FUSE") != nullptr && atoi(getenv("STRIX_SHEXP_FUSE")) != 0;
#else
    static const bool on = true;
#endif
    return on;
}

static bool ggml_cuda_match_shared_expert(const ggml_cgraph * graph, int routed_idx, int shared_idx) {
    if (routed_idx + 2 >= graph->n_nodes || shared_idx + 2 >= graph->n_nodes || shared_idx < routed_idx + 3) {
        return false;
    }
    const int nodes[] = { routed_idx, routed_idx + 1, routed_idx + 2, shared_idx, shared_idx + 1, shared_idx + 2 };
    const ggml_op ops[] = { GGML_OP_MUL_MAT_ID, GGML_OP_MUL_MAT_ID, GGML_OP_GLU,
                           GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU };
    const int outputs[] = { routed_idx + 2, shared_idx + 2 };
    if (!ggml_can_fuse_subgraph_ext(graph, nodes, 6, ops, outputs, 2)) {
        return false;
    }

    const ggml_tensor * routed = graph->nodes[routed_idx + 2];
    const ggml_tensor * shared = graph->nodes[shared_idx + 2];
    const ggml_tensor * gate = routed->src[0];
    const ggml_tensor * up = routed->src[1];
    const ggml_tensor * shared_gate = shared->src[0];
    const ggml_tensor * shared_up = shared->src[1];
    const auto is_pair = [&](const ggml_tensor * a, const ggml_tensor * b, int idx) {
        return (a == graph->nodes[idx] && b == graph->nodes[idx + 1]) ||
               (b == graph->nodes[idx] && a == graph->nodes[idx + 1]);
    };
    // only batch-size independent checks here: graph_optimize must produce the same graph topology for every ubatch
    // size, otherwise ggml-alloc has to re-reserve (and the scheduler to synchronize) at runtime.
    // the MMVQ batch size check is done in ggml_cuda_try_fuse
    if (!is_pair(gate, up, routed_idx) || !is_pair(shared_gate, shared_up, shared_idx) ||
            !ggml_cuda_should_fuse_mul_mat(up, gate, routed) ||
            !ggml_cuda_should_fuse_mul_mat(shared_up, shared_gate, shared) ||
            !up->src[0]->buffer ||
            !ggml_is_quantized(up->src[0]->type)) {
        return false;
    }
    const ggml_tensor * input = up->src[1];
    const ggml_tensor * weight = up->src[0];
    const ggml_tensor * shared_weight = shared_up->src[0];
    if (input->op != GGML_OP_RESHAPE || input->src[0] != shared_up->src[1] ||
            input->ne[1] != 1 || input->ne[3] != 1 || !ggml_is_contiguous(input) ||
            !ggml_is_contiguous(shared_up->src[1]) || !ggml_is_matrix(shared_up->src[1]) ||
            weight->type != shared_weight->type || weight->ne[0] != shared_weight->ne[0] ||
            weight->ne[1] != shared_weight->ne[1] || weight->nb[1] != shared_weight->nb[1] || weight->ne[3] != 1 ||
            !ggml_is_matrix(shared_weight) || !ggml_is_contiguous(shared_weight) ||
            !ggml_is_contiguous(shared_gate->src[0]) || !ggml_is_contiguous(routed) || !ggml_is_contiguous(shared)) {
        return false;
    }
    if (shared_weight->op != GGML_OP_NONE || shared_gate->src[0]->op != GGML_OP_NONE ||
            ggml_get_glu_op(routed) != ggml_get_glu_op(shared) ||
            ggml_get_op_params_f32(routed, 3) != ggml_get_op_params_f32(shared, 3)) {
        return false;
    }
    return true;
}

static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS

    if (ggml_cuda_op_mul_mat_use_fwht(dst) && ggml_cuda_op_fwht(ctx, src1, dst)) {
        return;
    }

    // strixllama: our narrow-product paths below skip a Hadamard product (the hint), as before the hint moved into fwht.cu
    const int32_t hint = ggml_get_op_params_i32(dst, 1);

    if (hint != GGML_HINT_SRC0_IS_HADAMARD && ggml_cuda_hc_inject_take(ctx, src1, dst)) {
        return;
    }

    if (hint != GGML_HINT_SRC0_IS_HADAMARD && ggml_cuda_small_k(ctx, src0, src1, dst)) {
        return;
    }

    if (hint != GGML_HINT_SRC0_IS_HADAMARD && ggml_cuda_skinny_f32(ctx, src0, src1, dst)) {
        return;
    }

    // strixllama: the BF16 twin of an F32 weight takes the vector kernel up to 8 columns, as its original does - bf16
    // widens to f32 exactly and mul_mat_vec_f's bf16 loop is its f32 loop, so the products are the same bits (bf16
    // alone would take the WMMA kernel from 4 columns on RDNA, with other sums)
    if (src0->type == GGML_TYPE_BF16 && src0->op == GGML_OP_NONE && ggml_get_op_params_i32(src0, 0) == GGML_BF16_TWIN_MAGIC &&
            ne11 <= MMVF_MAX_BATCH_SIZE && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
            ggml_cuda_should_use_mmvf(src0->type, ggml_cuda_info().devices[ctx.device].cc, ggml_cuda_info().devices[ctx.device].warp_size, src0->ne, src0->nb, 1)) {
        ggml_cuda_mul_mat_vec_f(ctx, src0, src1, nullptr, dst);
        return;
    }

    // If src0 is a temporary compute buffer it may have some padding that needs to be cleared for mul_mat_vec_q or mul_mat_q.
    // But if src0 is also a view of another tensor then this cannot be done safely because it may overwrite valid tensor data.
    // Therefore, in such cases use cuBLAS.
    const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE
        && ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) && src0->view_src;
    if (bad_padding_clear || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ggml_cuda_mul_mat_cublas(ctx, src0, src1, dst);
        return;
    }

    const int cc        = ggml_cuda_info().devices[ctx.device].cc;
    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;

    if (ggml_cuda_should_use_mmvf(src0->type, cc, warp_size, src0->ne, src0->nb, ne11)) {
        // The custom F16 vector kernel can be used over batched cuBLAS GEMM.
        // But this is only faster for GPUs without tensor cores or with a thin src0 matrix (particularly KQV in attention)
        ggml_cuda_mul_mat_vec_f(ctx, src0, src1, nullptr, dst);
        return;
    }
    // strixllama: a few-row F32 src0 with more columns than the vector kernel takes - the hyper-connection
    // inject [10240 x 4] and the GDN beta / alpha [2560 x 48] once a step verifies 16 tokens (four slots) -
    // is too narrow for mul_mat_f and fell through to cuBLAS: ~120 us per call against ~8 us per 8 columns
    // in the vector kernel, 95 + 72 such calls per pass. Run the vector kernel over column chunks instead.
    // Measured on gfx1151: [4 x 10240] 117 -> 16 us at 16 columns, [48 x 2560] 37 -> 11.
    // strixllama: and an F32 weight of any height at 9-32 columns (STRIX_F32_VEC_CHUNK_MAX): the MoE router
    // [2560 x 512] of a nine-token verify step - three conversations drafting two tokens each - took the MMB
    // kernel's 128-row tiles there, four workgroups for the whole GPU: ~250 us a product, the step +11 ms
    // strixllama: and a BF16 weight of any height at 9-32 columns (STRIX_BF16_VEC_CHUNK_MAX): the sparse-attention
    // indexer's k projection [2560 x 128] took the MMB kernel's one 128-row tile there, one workgroup for the whole GPU
    static const int64_t f32_vec_chunk_max  = getenv("STRIX_F32_VEC_CHUNK_MAX")  ? atoll(getenv("STRIX_F32_VEC_CHUNK_MAX"))  : 32;
    static const int64_t bf16_vec_chunk_max = getenv("STRIX_BF16_VEC_CHUNK_MAX") ? atoll(getenv("STRIX_BF16_VEC_CHUNK_MAX")) : 32;
    const bool vec_chunks = (ne01 <= 64 && ne11 <= 64) || (src0->type == GGML_TYPE_F32 && ne11 <= f32_vec_chunk_max)
        || (src0->type == GGML_TYPE_BF16 && ne11 <= bf16_vec_chunk_max);
    if (!ggml_is_quantized(src0->type) && vec_chunks && ne11 > MMVF_MAX_BATCH_SIZE
            && src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1
            && ggml_cuda_should_use_mmvf(src0->type, cc, warp_size, src0->ne, src0->nb, /*ne11 =*/ 1)) {
        int64_t chunk = MMVF_MAX_BATCH_SIZE;
        while (chunk > 1 && !ggml_cuda_should_use_mmvf(src0->type, cc, warp_size, src0->ne, src0->nb, chunk)) {
            --chunk;
        }
        for (int64_t c0 = 0; c0 < ne11; c0 += chunk) {
            const int64_t nc = std::min<int64_t>(chunk, ne11 - c0);
            ggml_tensor src1_c = *src1;
            src1_c.ne[1] = nc;
            src1_c.data  = (char *) src1->data + c0*src1->nb[1];
            ggml_tensor dst_c = *dst;
            dst_c.ne[1] = nc;
            dst_c.data  = (char *) dst->data + c0*dst->nb[1];
            ggml_cuda_mul_mat_vec_f(ctx, src0, &src1_c, nullptr, &dst_c);
        }
        return;
    }
    // A transposed vector can still use MMVQ (i.e. ne01 == 1)
    if (ne01 == 1 && ne11 > MMVF_MAX_BATCH_SIZE && ne2 == 1 && ne3 == 1
            && src0->type == GGML_TYPE_F32
            && ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst)
            && ggml_cuda_should_use_mmvf(src1->type, cc, warp_size, src1->ne, src1->nb, /*ne11 =*/ 1)) {
        ggml_tensor dst_vec = *dst;
        dst_vec.ne[0] = ne11;
        dst_vec.ne[1] = 1;
        dst_vec.nb[1] = dst_vec.nb[0]*ne11;
        dst_vec.nb[2] = dst_vec.nb[1];
        dst_vec.nb[3] = dst_vec.nb[1];
        ggml_cuda_mul_mat_vec_f(ctx, src1, src0, nullptr, &dst_vec);
        return;
    }
    if (ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, ne11, /*mul_mat_id =*/ false)) {
        ggml_cuda_mul_mat_f(ctx, src0, src1, nullptr, dst);
        return;
    }
    // strixllama: a quantized weight of at most STRIX_Q_VEC_CHUNK_ROWS output rows (1024) at 9-32 columns
    // (STRIX_Q_VEC_CHUNK_MAX, 0: off) - the hyper-connection down projection [10240 x 320], the attention k and v, the
    // shared expert - is a handful of MMQ tiles, and MMQ has no stream-k on RDNA: three to five workgroups for the whole
    // GPU. The vector kernel over 8-column chunks reads the weight twice and still takes about half the time. Taller
    // weights stay on MMQ: for the [6144 x 2560] projections the second read cost more than the tiles lose
    static const int64_t q_vec_chunk_max  = getenv("STRIX_Q_VEC_CHUNK_MAX")  ? atoll(getenv("STRIX_Q_VEC_CHUNK_MAX"))  : 32;
    static const int64_t q_vec_chunk_rows = getenv("STRIX_Q_VEC_CHUNK_ROWS") ? atoll(getenv("STRIX_Q_VEC_CHUNK_ROWS")) : 1024;
    if (ggml_is_quantized(src0->type) && ne11 > MMVQ_MAX_BATCH_SIZE && ne11 <= q_vec_chunk_max && ne01 <= q_vec_chunk_rows
            && src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1
            && ggml_cuda_should_use_mmvq(src0->type, cc, MMVQ_MAX_BATCH_SIZE)) {
        const int64_t chunk = MMVQ_MAX_BATCH_SIZE;
        for (int64_t c0 = 0; c0 < ne11; c0 += chunk) {
            const int64_t nc = std::min<int64_t>(chunk, ne11 - c0);
            ggml_tensor src1_c = *src1;
            src1_c.ne[1] = nc;
            src1_c.data  = (char *) src1->data + c0*src1->nb[1];
            ggml_tensor dst_c = *dst;
            dst_c.ne[1] = nc;
            dst_c.data  = (char *) dst->data + c0*dst->nb[1];
            ggml_cuda_mul_mat_vec_q(ctx, src0, &src1_c, nullptr, &dst_c);
        }
        return;
    }
    if (ggml_cuda_should_use_mmvq(src0->type, cc, ne11)) {
        ggml_cuda_mul_mat_vec_q(ctx, src0, src1, nullptr, dst);
        return;
    }
    if (ggml_cuda_mmb_supported_mm(src0, src1, dst)) {
        ggml_cuda_mul_mat_mmb(ctx, src0, src1, dst);
        return;
    }
    if (ggml_cuda_should_use_mmq(src0->type, cc, ne11, /*n_experts =*/ 0)) {
        ggml_cuda_mul_mat_q(ctx, src0, src1, nullptr, dst);
        return;
    }
    ggml_cuda_mul_mat_cublas(ctx, src0, src1, dst);
}

// returns true when ggml_cuda_mul_mat_id takes the fallback path that requires stream synchronization
// [TAG_MUL_MAT_ID_CUDA_GRAPHS]
static bool ggml_cuda_mul_mat_id_needs_sync(const ggml_tensor * dst, const int cc) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;

    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return true;
    }

    if (dst->ne[2] <= MMVQ_MAX_BATCH_SIZE) {
        if (ggml_is_quantized(src0->type)) {
            if (dst->ne[2] <= get_mmvq_mmid_max_batch(src0->type, cc)) {
                return false;
            }
        } else if (GGML_CUDA_CC_IS_AMD(cc)) {
            return false;
        }
    }

    if (ggml_cuda_should_use_mmq(src0->type, cc, src1->ne[2], /*n_experts=*/src0->ne[2])) {
        return false;
    }

    if (ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, src1->ne[2], /*mul_mat_id=*/true)) {
        return false;
    }

    return true;
}

static void ggml_cuda_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;

    // [TAG_MUL_MAT_ID_CUDA_GRAPHS]
    if (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        static_assert(MMVQ_MAX_BATCH_SIZE == MMVF_MAX_BATCH_SIZE);
        // strixllama: STRIX_MOE_VEC_MAX=n sends the routed experts of more than n tokens to the tiled kernel, which
        // dequantizes an expert's weights once for all its tokens where the vector kernel does it per token. At 6,
        // eight conversations decode +6% summed, four no worse. The manager sets it when several slots are
        // configured; the default is upstream's limit (8), so one conversation's results do not change
        static const int moe_vec_max = getenv("STRIX_MOE_VEC_MAX") ? atoi(getenv("STRIX_MOE_VEC_MAX")) : 8;
        if (ne2 <= MMVQ_MAX_BATCH_SIZE && ne2 <= moe_vec_max) {
            if (ggml_is_quantized(src0->type)) {
                const int mmvq_mmid_max = get_mmvq_mmid_max_batch(src0->type, cc);
                if (ne2 <= mmvq_mmid_max) {
                    ggml_cuda_mul_mat_vec_q(ctx, src0, src1, ids, dst);
                    return;
                }
            } else {
                if (GGML_CUDA_CC_IS_AMD(cc)) {
                    ggml_cuda_mul_mat_vec_f(ctx, src0, src1, ids, dst);
                    return;
                }
            }
        }

        // strixllama: a verify step of several conversations (5-16 tokens) is more than the vector kernel takes
        // (IQ3_S 4, IQ4_NL 6 on RDNA 3.5), and the tiled kernel pays per expert for staging a tile it barely fills: a
        // nine-token step's routed experts 397 -> 289 us a product for the gate/up, 499 -> 387 for the down, the step
        // -5 ms. Chunks of STRIX_MOE_VEC_CHUNK tokens (0: off) through the vector kernel instead, up to
        // STRIX_MOE_VEC_CHUNK_MAX_T tokens; past that the tiled kernel's one read of an expert for all its tokens wins
        static const int moe_vec_chunk = getenv("STRIX_MOE_VEC_CHUNK") ? atoi(getenv("STRIX_MOE_VEC_CHUNK")) : 4;
        static const int moe_vec_chunk_max_t = getenv("STRIX_MOE_VEC_CHUNK_MAX_T") ? atoi(getenv("STRIX_MOE_VEC_CHUNK_MAX_T")) : 16;
        if (moe_vec_chunk > 0 && ggml_is_quantized(src0->type) && ne2 > 1 && ne2 <= moe_vec_chunk_max_t && ne12 == ne2) {
            const int chunk = std::min(moe_vec_chunk, get_mmvq_mmid_max_batch(src0->type, cc));
            if (chunk >= 1) {
                for (int64_t t0 = 0; t0 < ne2; t0 += chunk) {
                    const int64_t nt = std::min<int64_t>(chunk, ne2 - t0);
                    ggml_tensor src1_c = *src1;
                    src1_c.ne[2] = nt;
                    src1_c.data  = (char *) src1->data + t0*src1->nb[2];
                    ggml_tensor ids_c = *ids;
                    ids_c.ne[1] = nt;
                    ids_c.data  = (char *) ids->data + t0*ids->nb[1];
                    ggml_tensor dst_c = *dst;
                    dst_c.ne[2] = nt;
                    dst_c.data  = (char *) dst->data + t0*dst->nb[2];
                    ggml_cuda_mul_mat_vec_q(ctx, src0, &src1_c, &ids_c, &dst_c);
                }
                return;
            }
        }

        if (ggml_cuda_mmb_supported_mmid(src0, src1, ids, dst)) {
            ggml_cuda_mul_mat_id_mmb(ctx, src0, src1, ids, dst);
            return;
        }
        if (ggml_cuda_should_use_mmq(src0->type, cc, ne12, /*n_experts=*/ne02)) {
            ggml_cuda_mul_mat_q(ctx, src0, src1, ids, dst);
            return;
        }

        if (ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, src1->ne[2], /*mul_mat_id=*/true)) {
            ggml_cuda_mul_mat_f(ctx, src0, src1, ids, dst);
            return;
        }
    }

    // note: this path should not be reached when recording CUDA graphs, because it requires stream synchronization
    GGML_ASSERT(ggml_cuda_mul_mat_id_needs_sync(dst, cc));
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2  % nb1  == 0);

    const ggml_type type_src1_sorted = (src0->type == GGML_TYPE_F16 && !fast_fp16_hardware_available(cc))
        || ggml_is_quantized(src0->type) ? GGML_TYPE_F32 : src0->type;
    const ggml_type type_dst_sorted  = GGML_TYPE_F32;
    const size_t ts_src1_sorted = ggml_type_size(type_src1_sorted);
    const size_t ts_dst_sorted  = ggml_type_size(type_dst_sorted);

    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows = ne12 * n_expert_used;

    std::vector<int32_t> ids_to_sorted_host;
    ids_to_sorted_host.reserve(2*ne_get_rows);
    std::vector<int32_t> ids_from_sorted_host(ne_get_rows);

    ggml_cuda_pool_alloc<int32_t> ids_buf_dev(ctx.pool(), 2*ne_get_rows);

    std::vector<int32_t> tokens_per_expert(ne02);

    ggml_cuda_pool_alloc<char> src1_sorted(ctx.pool(), ne12*n_expert_used*ne10*ts_src1_sorted);
    ggml_cuda_pool_alloc<char>  dst_sorted(ctx.pool(), ne2 *n_expert_used* ne0*ts_dst_sorted);

    std::vector<char> ids_host(ggml_nbytes(ids));
    CUDA_CHECK(cudaMemcpyAsync(ids_host.data(), ids->data, ggml_nbytes(ids), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int64_t i02 = 0; i02 < ne02; ++i02) { // expert matrices
        for (int64_t i12 = 0; i12 < ne12; ++i12) { // tokens
            for (int64_t iex = 0; iex < n_expert_used; ++iex) {
                const int32_t expert_to_use = *(const int32_t *)(ids_host.data() + i12*ids->nb[1] + iex*ids->nb[0]);
                assert(expert_to_use >= 0 && expert_to_use < ne02);
                if (expert_to_use == i02) {
                    ids_from_sorted_host[i12*n_expert_used + iex] = ids_to_sorted_host.size();
                    ids_to_sorted_host.push_back(i12*ne11 + iex % ne11);
                    tokens_per_expert[i02]++;
                    break;
                }
            }
        }
    }
    GGML_ASSERT(ids_to_sorted_host.size() == size_t(ne_get_rows));

    ids_to_sorted_host.insert(ids_to_sorted_host.end(), ids_from_sorted_host.begin(), ids_from_sorted_host.end());

    CUDA_CHECK(cudaMemcpyAsync(ids_buf_dev.ptr, ids_to_sorted_host.data(), 2*ne_get_rows*sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const int32_t * ids_to_sorted   = ids_buf_dev.ptr + 0*ne_get_rows;
    const int32_t * ids_from_sorted = ids_buf_dev.ptr + 1*ne_get_rows;

    get_rows_cuda(src1->data, src1->type, ids_to_sorted, src1_sorted.ptr, type_src1_sorted,
        ne10, nb11, nb12, nb13,
        ne_get_rows, 1, 1, sizeof(int32_t), ne_get_rows*sizeof(int32_t), ne_get_rows*sizeof(int32_t),
        ne10*ts_src1_sorted, ne_get_rows*ne10*ts_src1_sorted, ne_get_rows*ne10*ts_src1_sorted, stream);
    CUDA_CHECK(cudaGetLastError());

    char * src1_data_cur = (char *) src1_sorted.ptr;
    char *  dst_data_cur = (char *)  dst_sorted.ptr;
    for (int64_t i02 = 0; i02 < ne02; ++i02) {
        if (tokens_per_expert[i02] == 0) {
            continue;
        }

        ggml_tensor src0_slice = *src0;
        src0_slice.ne[2]    = 1;
        src0_slice.nb[3]    = src0_slice.nb[2];
        src0_slice.op       = GGML_OP_VIEW;
        src0_slice.view_src = dst->src[0]; // non-const pointer to src0
        src0_slice.data     = (char *) src0->data + i02*nb02;

        ggml_tensor src1_slice;
        memset(&src1_slice, 0, sizeof(src1_slice));
        src1_slice.buffer = src1->buffer;
        src1_slice.type   = type_src1_sorted;
        src1_slice.ne[0]  = ne10;
        src1_slice.ne[1]  = tokens_per_expert[i02];
        src1_slice.ne[2]  = 1;
        src1_slice.ne[3]  = 1;
        src1_slice.nb[0]  = ts_src1_sorted;
        src1_slice.nb[1]  = src1_slice.ne[0] * src1_slice.nb[0];
        src1_slice.nb[2]  = src1_slice.ne[1] * src1_slice.nb[1];
        src1_slice.nb[3]  = src1_slice.ne[2] * src1_slice.nb[2];
        src1_slice.data   = src1_data_cur;

        ggml_tensor dst_slice;
        memset(&dst_slice, 0, sizeof(dst_slice));
        memcpy(dst_slice.op_params, dst->op_params, sizeof(dst_slice.op_params));
        dst_slice.buffer = dst->buffer;
        dst_slice.type   = type_dst_sorted;
        dst_slice.ne[0]  = ne0;
        dst_slice.ne[1]  = tokens_per_expert[i02];
        dst_slice.ne[2]  = 1;
        dst_slice.ne[3]  = 1;
        dst_slice.nb[0]  = ts_dst_sorted;
        dst_slice.nb[1]  = dst_slice.ne[0] * dst_slice.nb[0];
        dst_slice.nb[2]  = dst_slice.ne[1] * dst_slice.nb[1];
        dst_slice.nb[3]  = dst_slice.ne[2] * dst_slice.nb[2];
        dst_slice.data   = dst_data_cur;

        ggml_cuda_mul_mat(ctx, &src0_slice, &src1_slice, &dst_slice);
        CUDA_CHECK(cudaGetLastError());

        src1_data_cur += src1_slice.nb[2];
        dst_data_cur  +=  dst_slice.nb[2];
    }

    get_rows_cuda(dst_sorted.ptr, type_dst_sorted, ids_from_sorted, dst->data, dst->type,
        ne0, ne0*ts_dst_sorted, ne_get_rows*ne0*ts_dst_sorted, ne_get_rows*ne0*ts_dst_sorted,
        ne_get_rows, 1, 1, sizeof(int32_t), ne_get_rows*sizeof(int32_t), ne_get_rows*sizeof(int32_t),
        nb1, nb2, nb3, stream);
}

// strixllama DEV: STRIX_REPEAT=topk:N,fa:N,kvrows:N,argsort:N,idxdec:N dispatches those ops N more times (each writes what
// it wrote before), so a step's time grows by N times the op's in-graph cost; 0 when unset
static int strix_repeat(const char * key) {
    static const std::string spec = getenv("STRIX_REPEAT") ? getenv("STRIX_REPEAT") : "";
    if (spec.empty()) { return 0; }
    const std::string k = std::string(key) + ":";
    const size_t at = spec.find(k);
    return at == std::string::npos ? 0 : atoi(spec.c_str() + at + k.size());
}

// strixllama DEV: STRIX_REPEAT_FILE=<path>, the in-graph cost of any op class by repetition, changeable while the server
// runs. The file holds items OP[:SUB][@lo[-hi]][/K]=N, separated by commas, semicolons or newlines: OP a ggml op name or
// *, SUB a unary or glu op name, lo-hi the node's ne[0], K its src[0]'s ne[0] (e.g. MUL_MAT@10240/320=2 is the
// hyper-connections' up product). A matching node - a fused chain by its first node - is dispatched N more times, so a
// step grows by N times the class's cost. Only nodes whose output overlaps none of their inputs repeat (each repeat
// writes what the node wrote), never the ops that write beyond their output (cache and state writes); a chain repeats
// only if all of its nodes qualify. The item `dump` lists every dispatched class of the next graphs with its count. The
// file is read again when its content changes (checked at most five times a second), and the HIP graphs are captured
// again then.
struct strix_repeat_item {
    std::string op, sub;
    long long lo = std::numeric_limits<long long>::min(), hi = std::numeric_limits<long long>::max(), k = -1;
    int n = 0;
};
static std::vector<strix_repeat_item> g_strix_rep;
static int     g_strix_rep_gen  = 0;
static int     g_strix_rep_dump = 0;    // graphs left to list
static int64_t g_strix_rep_checked_us = 0;
static std::map<std::string, std::pair<int, int>> g_strix_rep_counts;   // the dump: dispatches and repeatable ones a class

// the dump of the graph that just ran, at the start of the next one
static void strix_repeat_dump_flush(int n_nodes_next) {
    if (g_strix_rep_dump <= 0 || g_strix_rep_counts.empty()) {
        return;
    }
    int total = 0;
    for (const auto & kv : g_strix_rep_counts) {
        total += kv.second.first;
    }
    fprintf(stderr, "STRIX_REPEAT_FILE dump: a graph of %d dispatches in %zu classes (next graph: %d nodes)\n", total,
            g_strix_rep_counts.size(), n_nodes_next);
    for (const auto & kv : g_strix_rep_counts) {
        fprintf(stderr, "STRIX_REPEAT_FILE dump:   %-64s %6d  repeatable %6d\n", kv.first.c_str(), kv.second.first,
                kv.second.second);
    }
    g_strix_rep_counts.clear();
    --g_strix_rep_dump;
}

static void strix_repeat_file_reload() {
    static const char * path = getenv("STRIX_REPEAT_FILE");
    if (!path) {
        return;
    }
    const int64_t now = ggml_time_us();
    if (now - g_strix_rep_checked_us < 200000) {
        return;
    }
    g_strix_rep_checked_us = now;
    std::string text;
    if (FILE * f = fopen(path, "rb")) {
        char buf[4096];
        size_t got;
        while ((got = fread(buf, 1, sizeof(buf), f)) > 0) {
            text.append(buf, got);
        }
        fclose(f);
    } else {
        return;
    }
    static std::string last = "\x01";
    if (text == last) {
        return;
    }
    last = text;
    std::vector<strix_repeat_item> items;
    bool dump = false;
    std::string cur;
    auto flush = [&]() {
        if (cur.empty() || cur[0] == '#') {
            return;
        }
        if (cur == "dump") {
            dump = true;
            return;
        }
        const size_t eq = cur.find('=');
        if (eq == std::string::npos) {
            return;
        }
        strix_repeat_item it;
        it.n = atoi(cur.c_str() + eq + 1);
        std::string lhs = cur.substr(0, eq);
        const size_t sl = lhs.find('/');
        if (sl != std::string::npos) {
            it.k = atoll(lhs.c_str() + sl + 1);
            lhs = lhs.substr(0, sl);
        }
        const size_t at = lhs.find('@');
        if (at != std::string::npos) {
            it.lo = atoll(lhs.c_str() + at + 1);
            const size_t dash = lhs.find('-', at + 1);
            it.hi = dash != std::string::npos ? atoll(lhs.c_str() + dash + 1) : it.lo;
            lhs = lhs.substr(0, at);
        }
        const size_t colon = lhs.find(':');
        if (colon != std::string::npos) {
            it.sub = lhs.substr(colon + 1);
            lhs = lhs.substr(0, colon);
        }
        it.op = lhs;
        if (it.n > 0) {
            items.push_back(it);
        }
    };
    for (char c : text + ",") {
        if (c == ',' || c == ';' || c == '\n' || c == '\r') {
            flush();
            cur.clear();
        } else if (c != ' ' && c != '\t') {
            cur.push_back(c);
        }
    }
    g_strix_rep = items;
    g_strix_rep_dump = dump ? 4 : 0;
    ++g_strix_rep_gen;
    fprintf(stderr, "STRIX_REPEAT_FILE: generation %d, %zu items%s\n", g_strix_rep_gen, items.size(), dump ? ", dump" : "");
}

static const char * strix_repeat_sub(const ggml_tensor * node) {
    return node->op == GGML_OP_UNARY ? ggml_unary_op_name(ggml_get_unary_op(node)) :
           node->op == GGML_OP_GLU   ? ggml_glu_op_name(ggml_get_glu_op(node)) : "";
}

static int strix_repeat_file_count(const ggml_tensor * node) {
    for (const strix_repeat_item & it : g_strix_rep) {
        if (it.op != "*" && it.op != ggml_op_name(node->op)) {
            continue;
        }
        if (!it.sub.empty() && it.sub != strix_repeat_sub(node)) {
            continue;
        }
        if (node->ne[0] < it.lo || node->ne[0] > it.hi) {
            continue;
        }
        if (it.k >= 0 && (!node->src[0] || node->src[0]->ne[0] != it.k)) {
            continue;
        }
        return it.n;
    }
    return 0;
}

static bool strix_repeat_overlaps(const ggml_tensor * a, const ggml_tensor * b) {
    const char * a0 = (const char *) a->data;
    const char * b0 = (const char *) b->data;
    return a0 && b0 && a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
}

static bool strix_repeat_is_view(const ggml_tensor * t) {
    return ggml_is_empty(t) || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE || t->op == GGML_OP_VIEW ||
           t->op == GGML_OP_PERMUTE || t->op == GGML_OP_NONE;
}

// t is one of nodes [i, i + skip], or a view of one
static bool strix_repeat_in_chain(const ggml_cgraph * cgraph, int i, int skip, const ggml_tensor * t) {
    for (; t; t = t->view_src) {
        for (int j = i; j <= i + skip; ++j) {
            if (cgraph->nodes[j] == t) {
                return true;
            }
        }
    }
    return false;
}

// nodes [i, i + skip] write only their outputs, and none of those overlaps an input from outside them
static bool strix_repeat_chain_safe(const ggml_cgraph * cgraph, int i, int skip) {
    for (int j = i; j <= i + skip; ++j) {
        const ggml_tensor * w = cgraph->nodes[j];
        if (strix_repeat_is_view(w)) {
            continue;
        }
        switch (w->op) {
            case GGML_OP_SET_ROWS: case GGML_OP_CPY: case GGML_OP_DUP: case GGML_OP_SET: case GGML_OP_ACC:
            case GGML_OP_GATED_DELTA_NET: case GGML_OP_SSM_CONV: case GGML_OP_SSM_SCAN:
            case GGML_OP_CUSTOM: case GGML_OP_MAP_CUSTOM1: case GGML_OP_MAP_CUSTOM2: case GGML_OP_MAP_CUSTOM3:
                return false;
            default:
                break;
        }
        for (int jj = i; jj <= i + skip; ++jj) {
            const ggml_tensor * r = cgraph->nodes[jj];
            for (int k = 0; k < GGML_MAX_SRC && r->src[k]; ++k) {
                if (!strix_repeat_in_chain(cgraph, i, skip, r->src[k]) && strix_repeat_overlaps(w, r->src[k])) {
                    return false;
                }
            }
        }
    }
    return true;
}

// a node that writes its output and nothing else, and reads none of it
static bool strix_repeat_safe(const ggml_tensor * node) {
    switch (node->op) {
        case GGML_OP_SET_ROWS: case GGML_OP_CPY: case GGML_OP_DUP: case GGML_OP_SET: case GGML_OP_ACC:
        case GGML_OP_GATED_DELTA_NET: case GGML_OP_SSM_CONV: case GGML_OP_SSM_SCAN:
        case GGML_OP_CUSTOM: case GGML_OP_MAP_CUSTOM1: case GGML_OP_MAP_CUSTOM2: case GGML_OP_MAP_CUSTOM3:
            return false;
        default:
            break;
    }
    for (int s = 0; s < GGML_MAX_SRC && node->src[s]; ++s) {
        if (strix_repeat_overlaps(node, node->src[s])) {
            return false;
        }
    }
    return true;
}

// how many more times nodes [i, i + skip] run (a fused chain when skip > 0)
static int strix_repeat_times(const ggml_cgraph * cgraph, int i, int skip) {
    const ggml_tensor * first = cgraph->nodes[i];
    if (g_strix_rep_dump > 0) {
        char key[160];
        snprintf(key, sizeof(key), "%s%s%s@%lld/%lld%s", ggml_op_name(first->op), *strix_repeat_sub(first) ? ":" : "",
                 strix_repeat_sub(first), (long long) first->ne[0], first->src[0] ? (long long) first->src[0]->ne[0] : -1LL,
                 skip > 0 ? " (fused)" : "");
        const bool safe = strix_repeat_chain_safe(cgraph, i, skip);
        auto & c = g_strix_rep_counts[key];
        ++c.first;
        c.second += safe ? 1 : 0;
    }
    if (g_strix_rep.empty()) {
        return 0;
    }
    const int n = strix_repeat_file_count(first);
    if (n <= 0) {
        return 0;
    }
    return strix_repeat_chain_safe(cgraph, i, skip) ? n : 0;
}

// strixllama DEV: STRIX_GPU_TIMELINE=N, where a decode step's GPU time goes, by op class. The graph evaluation writes the
// GPU's wall clock (100 MHz) twice before the first dispatch and once after every dispatch - a one-thread kernel,
// captured into the HIP graph like the rest - and every Nth compute of a graph whose first node has fewer than 64 rows
// is read back. A dispatch's time is its timestamp less the one before, less the timestamp kernel's own cost (the two
// at the start: ~0.7 us); a fused chain counts as its first node's class. Every 8 readings print a table: per class the
// dispatches a graph and the us a graph, and the graph's span (first to last timestamp) and the sum.
static int strix_tl_every() {
    static const int n = getenv("STRIX_GPU_TIMELINE") ? atoi(getenv("STRIX_GPU_TIMELINE")) : 0;
    return n;
}

static __global__ void strix_tl_ts(uint64_t * dst) {
    *dst = wall_clock64();
}

struct strix_tl_capture {
    std::vector<std::string> keys;   // the class of each timestamped dispatch
    int64_t rows = 0;                // the first node's rows
};
static std::unordered_map<const void *, strix_tl_capture> g_strix_tl;
static uint64_t *       g_strix_tl_buf  = nullptr;
static const int        STRIX_TL_MAX    = 1 << 15;
static strix_tl_capture * g_strix_tl_cur = nullptr;   // the capture being written by the evaluation

// strixllama DEV: STRIX_AB=<n>, an in-run A/B: every n decode graphs (first node < 64 rows) the side flips between A
// and B and every HIP graph is captured anew (as a STRIX_REPEAT_FILE change does). Switches that read
// ggml_cuda_strix_ab() follow it; the GPU timeline keeps the sides apart. Drift then falls on both sides alike.
static int g_strix_ab = 0;
static int g_strix_ab_since = 0;   // decode graphs since the last flip: the first two (captured anew) are not timed
int ggml_cuda_strix_ab() {
    return g_strix_ab;
}

static void strix_ab_step(const ggml_cgraph * cgraph);

struct strix_tl_acc {
    std::map<std::string, std::pair<double, double>> by_class;   // dispatches, us
    double span_us = 0.0, sum_us = 0.0, overhead_us = 0.0;
    int graphs = 0, readings = 0;
};
static std::map<std::pair<size_t, int64_t>, strix_tl_acc> g_strix_tl_accs;   // by graph: dispatches, first node's rows (+ 1e9 on side B)

static std::string strix_tl_class(const ggml_cgraph * cgraph, int i, int skip) {
    const ggml_tensor * first = cgraph->nodes[i];
    const char * sub = first->op == GGML_OP_UNARY ? ggml_unary_op_name(ggml_get_unary_op(first)) :
                       first->op == GGML_OP_GLU   ? ggml_glu_op_name(ggml_get_glu_op(first)) : "";
    char key[400];
    snprintf(key, sizeof(key), "%s%s%s@%lld/%lld%s", ggml_op_name(first->op), *sub ? ":" : "", sub,
             (long long) first->ne[0], first->src[0] ? (long long) first->src[0]->ne[0] : -1LL, skip > 0 ? " (fused)" : "");
    if (skip > 0 && getenv("STRIX_GPU_TIMELINE_CHAINS")) {
        std::string ops = " =";
        for (int j = i; j <= i + skip; ++j) {
            const ggml_tensor * t = cgraph->nodes[j];
            if (strix_repeat_is_view(t)) {
                continue;
            }
            ops += " ";
            ops += t->op == GGML_OP_UNARY ? ggml_unary_op_name(ggml_get_unary_op(t)) :
                   t->op == GGML_OP_GLU   ? ggml_glu_op_name(ggml_get_glu_op(t)) : ggml_op_name(t->op);
        }
        const size_t n = strlen(key);
        snprintf(key + n, sizeof(key) - n, "%s", ops.c_str());
    }
    static const bool names = getenv("STRIX_GPU_TIMELINE_NAMES") != nullptr;   // DEV: the first node's name too
    if (names) {
        const size_t n = strlen(key);
        snprintf(key + n, sizeof(key) - n, " {%s}", first->name);
    }
    if ((first->op == GGML_OP_MUL_MAT || first->op == GGML_OP_MUL_MAT_ID) && first->src[1]) {
        const ggml_tensor * b = first->src[1];
        const size_t n = strlen(key);
        snprintf(key + n, sizeof(key) - n, " x[%lld,%lld,%lld]%s", (long long) b->ne[1], (long long) b->ne[2], (long long) b->ne[3],
                 ggml_is_contiguous(b) ? "" : " nc");
    }
    return key;
}

static void strix_tl_start(ggml_backend_cuda_context * ctx, const ggml_cgraph * cgraph, const void * graph_key) {
    g_strix_tl_cur = nullptr;
    if (strix_tl_every() <= 0 || !g_strix_tl_buf) {
        return;
    }
    strix_tl_capture & c = g_strix_tl[graph_key];
    c.keys.clear();
    c.rows = cgraph->n_nodes ? ggml_nrows(cgraph->nodes[0]) : 0;
    g_strix_tl_cur = &c;
    strix_tl_ts<<<1, 1, 0, ctx->stream()>>>(g_strix_tl_buf + 0);
    strix_tl_ts<<<1, 1, 0, ctx->stream()>>>(g_strix_tl_buf + 1);
}

static void strix_tl_mark(ggml_backend_cuda_context * ctx, const ggml_cgraph * cgraph, int i, int skip) {
    if (!g_strix_tl_cur || (int) g_strix_tl_cur->keys.size() + 2 >= STRIX_TL_MAX) {
        return;
    }
    g_strix_tl_cur->keys.push_back(strix_tl_class(cgraph, i, skip));
    strix_tl_ts<<<1, 1, 0, ctx->stream()>>>(g_strix_tl_buf + 1 + g_strix_tl_cur->keys.size());
}

static void strix_tl_after(ggml_backend_cuda_context * ctx, const void * graph_key) {
    g_strix_tl_cur = nullptr;
    const int every = strix_tl_every();
    if (every <= 0 || !g_strix_tl_buf) {
        return;
    }
    auto it = g_strix_tl.find(graph_key);
    static const int64_t tl_rows = getenv("STRIX_GPU_TIMELINE_ROWS") ? atoll(getenv("STRIX_GPU_TIMELINE_ROWS")) : 64;   // DEV: prompt graphs too
    if (it == g_strix_tl.end() || it->second.rows >= tl_rows || it->second.keys.empty()) {
        return;
    }
    static int computes = 0;
    if (++computes % every != 0) {
        return;
    }
    if (getenv("STRIX_AB") && g_strix_ab_since < 2) {
        return;
    }
    const strix_tl_capture & c = it->second;
    std::vector<uint64_t> ts(c.keys.size() + 2);
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream()));
    CUDA_CHECK(cudaMemcpy(ts.data(), g_strix_tl_buf, ts.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    const double us = 0.01;   // 100 MHz
    const double overhead = (ts[1] - ts[0]) * us;
    strix_tl_acc & a = g_strix_tl_accs[std::make_pair(c.keys.size(), c.rows + (g_strix_ab ? 1000000000LL : 0))];
    a.overhead_us += overhead;
    a.span_us += (ts.back() - ts[0]) * us;
    for (size_t k = 0; k < c.keys.size(); ++k) {
        const double d = (double) (int64_t) (ts[k + 2] - ts[k + 1]) * us - overhead;
        auto & e = a.by_class[c.keys[k]];
        e.first += 1.0;
        e.second += d;
        a.sum_us += d;
    }
    ++a.graphs;
    // STRIX_GPU_TIMELINE_ORDER=n: the first n dispatches of the second reading, in order
    static const int order_n = getenv("STRIX_GPU_TIMELINE_ORDER") ? atoi(getenv("STRIX_GPU_TIMELINE_ORDER")) : 0;
    if (order_n > 0 && a.readings == 1) {
        for (size_t k = 0; k < c.keys.size() && (int) k < order_n; ++k) {
            fprintf(stderr, "GPU_ORDER %5zu %9.2f us  %s\n", k, (double) (int64_t) (ts[k + 2] - ts[k + 1]) * us - overhead, c.keys[k].c_str());
        }
    }
    // STRIX_GPU_TIMELINE_PRINT=n: a table every n readings (default 8; 1 for a single prompt)
    static const int print_every = getenv("STRIX_GPU_TIMELINE_PRINT") ? std::max(1, atoi(getenv("STRIX_GPU_TIMELINE_PRINT"))) : 8;
    if (++a.readings % print_every != 0) {
        return;
    }
    std::vector<std::pair<std::string, std::pair<double, double>>> rows(a.by_class.begin(), a.by_class.end());
    std::sort(rows.begin(), rows.end(), [](const auto & x, const auto & y) { return x.second.second > y.second.second; });
    const double g = a.graphs;
    fprintf(stderr, "GPU_TIMELINE: [%lld rows%s] %d graphs, a graph: span %.0f us, dispatches %.0f, their sum %.0f us, timestamp cost %.2f us each\n",
            (long long) c.rows, getenv("STRIX_AB") ? (g_strix_ab ? ", B" : ", A") : "", a.graphs, a.span_us / g,
            (double) c.keys.size(), a.sum_us / g, a.overhead_us / g);
    for (const auto & r : rows) {
        fprintf(stderr, "GPU_TIMELINE:   %-100s %6.0f x %8.1f us  %6.2f us each  %5.1f%%\n", r.first.c_str(), r.second.first / g,
                r.second.second / g, r.second.second / r.second.first, 100.0 * r.second.second / a.sum_us);
    }
    a = strix_tl_acc();
}

bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
    // strixllama (dev): STRIX_SKIP_NODES="a,b,..." leaves out the decode-size nodes (<= 16 rows) whose op name is one of the
    // strings or whose first source's name contains one (of more than 3 characters) - wrong outputs; the generation speed with and without them is what they
    // cost in the real step (graphs on). Fused paths that do not come through here are not affected.
    {
        static const std::vector<std::string> skip = [] {
            std::vector<std::string> v;
            const char * e = getenv("STRIX_SKIP_NODES");
            for (std::string s = e ? e : ""; !s.empty(); ) {
                const size_t c = s.find(',');
                v.push_back(s.substr(0, c));
                s = c == std::string::npos ? "" : s.substr(c + 1);
            }
            return v;
        }();
        if (!skip.empty() && ggml_nrows(dst) <= 16*16 && dst->ne[1] <= 16) {
            const char * op = ggml_op_name(dst->op);
            const char * s0 = dst->src[0] ? dst->src[0]->name : "";
            for (const auto & k : skip) {
                if (strcmp(op, k.c_str()) == 0 || (k.size() > 3 && strstr(s0, k.c_str()))) {
                    return true;
                }
            }
        }
    }
    if (ggml_cuda_mmb_marks_count() > 0 && dst->op != GGML_OP_MUL_MAT && dst->op != GGML_OP_MUL_MAT_ID && dst->op != GGML_OP_VIEW &&
            dst->op != GGML_OP_RESHAPE && dst->op != GGML_OP_PERMUTE && dst->op != GGML_OP_TRANSPOSE && dst->op != GGML_OP_NONE) {
        for (int s = 0; s < GGML_MAX_SRC && dst->src[s]; ++s) {
            const ggml_tensor * src = dst->src[s];
            if (ggml_cuda_mmb_is_bf16_only(src) || (src->view_src && ggml_cuda_mmb_is_bf16_only(src->view_src))) {
                GGML_ABORT("MMB: %s(%s) reads BF16-only tensor %s through the unfused path", ggml_op_name(dst->op), dst->name, src->name);
            }
        }
    }
    switch (dst->op) {
        case GGML_OP_ARGMAX:
            ggml_cuda_argmax(ctx, dst);
            break;
        case GGML_OP_COUNT_EQUAL:
            ggml_cuda_count_equal(ctx, dst);
            break;
        case GGML_OP_REPEAT:
            ggml_cuda_op_repeat(ctx, dst);
            break;
        case GGML_OP_REPEAT_BACK:
            ggml_cuda_op_repeat_back(ctx, dst);
            break;
        case GGML_OP_GET_ROWS:
            ggml_cuda_op_get_rows(ctx, dst);
            if (dst->type == GGML_TYPE_F16 && dst->ne[0] == 512) {
                for (int r = strix_repeat("kvrows"); r > 0; --r) { ggml_cuda_op_get_rows(ctx, dst); }
            }
            break;
        case GGML_OP_GET_ROWS_BACK:
            ggml_cuda_op_get_rows_back(ctx, dst);
            break;
        case GGML_OP_SET_ROWS:
            ggml_cuda_op_set_rows(ctx, dst);
            break;
        case GGML_OP_SET:
            ggml_cuda_op_set(ctx, dst);
            break;
        case GGML_OP_DUP:
            ggml_cuda_dup(ctx, dst);
            break;
        case GGML_OP_CPY:
            ggml_cuda_cpy(ctx, dst->src[0], dst->src[1]);
            break;
        case GGML_OP_CONT:
            ggml_cuda_dup(ctx, dst);
            break;
        case GGML_OP_ADD:
        case GGML_OP_ADD1: // TODO: more efficient implementation
            ggml_cuda_op_add(ctx, dst);
            break;
        case GGML_OP_ADD_ID:
            ggml_cuda_op_add_id(ctx, dst);
            break;
        case GGML_OP_SUB:
            ggml_cuda_op_sub(ctx, dst);
            break;
        case GGML_OP_ACC:
            ggml_cuda_op_acc(ctx, dst);
            break;
        case GGML_OP_MUL:
            ggml_cuda_op_mul(ctx, dst);
            break;
        case GGML_OP_DIV:
            ggml_cuda_op_div(ctx, dst);
            break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(dst)) {
                case GGML_UNARY_OP_ABS:
                    ggml_cuda_op_abs(ctx, dst);
                    break;
                case GGML_UNARY_OP_SGN:
                    ggml_cuda_op_sgn(ctx, dst);
                    break;
                case GGML_UNARY_OP_NEG:
                    ggml_cuda_op_neg(ctx, dst);
                    break;
                case GGML_UNARY_OP_STEP:
                    ggml_cuda_op_step(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU:
                    ggml_cuda_op_gelu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SILU:
                    ggml_cuda_op_silu(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_ERF:
                    ggml_cuda_op_gelu_erf(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_QUICK:
                    ggml_cuda_op_gelu_quick(ctx, dst);
                    break;
                case GGML_UNARY_OP_TANH:
                    ggml_cuda_op_tanh(ctx, dst);
                    break;
                case GGML_UNARY_OP_RELU:
                    ggml_cuda_op_relu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SIGMOID:
                    ggml_cuda_op_sigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSIGMOID:
                    ggml_cuda_op_hardsigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSWISH:
                    ggml_cuda_op_hardswish(ctx, dst);
                    break;
                case GGML_UNARY_OP_EXP:
                    ggml_cuda_op_exp(ctx, dst);
                    break;
                case GGML_UNARY_OP_ELU:
                    ggml_cuda_op_elu(ctx, dst);
                    break;
                case GGML_UNARY_OP_XIELU:
                    ggml_cuda_op_xielu(ctx, dst);
                    break;
                case GGML_UNARY_OP_FLOOR:
                    ggml_cuda_op_floor(ctx, dst);
                    break;
                case GGML_UNARY_OP_CEIL:
                    ggml_cuda_op_ceil(ctx, dst);
                    break;
                case GGML_UNARY_OP_ROUND:
                    ggml_cuda_op_round(ctx, dst);
                    break;
                case GGML_UNARY_OP_TRUNC:
                    ggml_cuda_op_trunc(ctx, dst);
                    break;
                case GGML_UNARY_OP_EXPM1:
                    ggml_cuda_op_expm1(ctx, dst);
                    break;
                case GGML_UNARY_OP_SOFTPLUS:
                    ggml_cuda_op_softplus(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(dst)) {
                case GGML_GLU_OP_REGLU:
                    ggml_cuda_op_reglu(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU:
                    ggml_cuda_op_geglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU:
                    ggml_cuda_op_swiglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU_OAI:
                    ggml_cuda_op_swiglu_oai(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_ERF:
                    ggml_cuda_op_geglu_erf(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_QUICK:
                    ggml_cuda_op_geglu_quick(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    ggml_cuda_op_swiglu_clamp(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_NORM:
            ggml_cuda_op_norm(ctx, dst);
            break;
        case GGML_OP_GROUP_NORM:
            ggml_cuda_op_group_norm(ctx, dst);
            break;
        case GGML_OP_L2_NORM:
            ggml_cuda_op_l2_norm(ctx, dst);
            break;
        case GGML_OP_CONCAT:
            ggml_cuda_op_concat(ctx, dst);
            break;
        case GGML_OP_UPSCALE:
            ggml_cuda_op_upscale(ctx, dst);
            break;
        case GGML_OP_PAD:
            ggml_cuda_op_pad(ctx, dst);
            break;
        case GGML_OP_PAD_REFLECT_1D:
            ggml_cuda_op_pad_reflect_1d(ctx, dst);
            break;
        case GGML_OP_ARANGE:
            ggml_cuda_op_arange(ctx, dst);
            break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            ggml_cuda_op_timestep_embedding(ctx, dst);
            break;
        case GGML_OP_LEAKY_RELU:
            ggml_cuda_op_leaky_relu(ctx, dst);
            break;
        case GGML_OP_SILU_BACK:
            ggml_cuda_op_silu_back(ctx, dst);
            break;
        case GGML_OP_RMS_NORM:
            ggml_cuda_op_rms_norm(ctx, dst);
            break;
        case GGML_OP_RMS_NORM_BACK:
            ggml_cuda_op_rms_norm_back(ctx, dst);
            break;
        case GGML_OP_MUL_MAT:
            ggml_cuda_mul_mat(ctx, dst->src[0], dst->src[1], dst);
            break;
        case GGML_OP_MUL_MAT_ID:
            ggml_cuda_mul_mat_id(ctx, dst);
            break;
        case GGML_OP_OUT_PROD:
            ggml_cuda_out_prod(ctx, dst);
            break;
        case GGML_OP_SCALE:
            ggml_cuda_op_scale(ctx, dst);
            break;
        case GGML_OP_SQR:
            ggml_cuda_op_sqr(ctx, dst);
            break;
        case GGML_OP_SQRT:
            ggml_cuda_op_sqrt(ctx, dst);
            break;
        case GGML_OP_SIN:
            ggml_cuda_op_sin(ctx, dst);
            break;
        case GGML_OP_COS:
            ggml_cuda_op_cos(ctx, dst);
            break;
        case GGML_OP_CLAMP:
            ggml_cuda_op_clamp(ctx, dst);
            break;
        case GGML_OP_LOG:
            ggml_cuda_op_log(ctx, dst);
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
                break;
        case GGML_OP_DIAG:
            ggml_cuda_op_diag(ctx, dst);
            break;
        case GGML_OP_DIAG_MASK_INF:
            ggml_cuda_op_diag_mask_inf(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX:
            ggml_cuda_op_soft_max(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX_BACK:
            ggml_cuda_op_soft_max_back(ctx, dst);
            break;
        case GGML_OP_ROPE:
            ggml_cuda_op_rope(ctx, dst);
            break;
        case GGML_OP_ROPE_BACK:
            ggml_cuda_op_rope_back(ctx, dst);
            break;
        case GGML_OP_ROLL:
            ggml_cuda_op_roll(ctx, dst);
            break;
        case GGML_OP_IM2COL:
            ggml_cuda_op_im2col(ctx, dst);
            break;
        case GGML_OP_IM2COL_3D:
            ggml_cuda_op_im2col_3d(ctx, dst);
            break;
        case GGML_OP_CONV_2D:
            ggml_cuda_op_conv2d(ctx, dst);
            break;
        case GGML_OP_CONV_3D:
            ggml_cuda_op_conv3d(ctx, dst);
            break;
        case GGML_OP_CONV_2D_DW:
            ggml_cuda_op_conv2d_dw(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_2D:
            ggml_cuda_conv_2d_transpose_p0(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            ggml_cuda_op_conv_transpose_1d(ctx,dst);
            break;
        case GGML_OP_COL2IM_1D:
            ggml_cuda_op_col2im_1d(ctx, dst);
            break;
        case GGML_OP_POOL_2D:
            ggml_cuda_op_pool2d(ctx, dst);
            break;
        case GGML_OP_POOL_1D:
            ggml_cuda_op_pool1d(ctx, dst);
            break;
        case GGML_OP_SUM:
            ggml_cuda_op_sum(ctx, dst);
            break;
        case GGML_OP_CUMSUM:
            ggml_cuda_op_cumsum(ctx, dst);
            break;
        case GGML_OP_SUM_ROWS:
            ggml_cuda_op_sum_rows(ctx, dst);
            break;
        case GGML_OP_MEAN:
            ggml_cuda_op_mean(ctx, dst);
            break;
        case GGML_OP_SSM_CONV:
            ggml_cuda_op_ssm_conv(ctx, dst);
            break;
        case GGML_OP_SSM_SCAN:
            ggml_cuda_op_ssm_scan(ctx, dst);
            break;
        case GGML_OP_TOP_K:
            ggml_cuda_op_top_k(ctx, dst);
            if (dst->ne[0] == 512) {
                for (int r = strix_repeat("topk"); r > 0; --r) { ggml_cuda_op_top_k(ctx, dst); }
            }
            break;
        case GGML_OP_ARGSORT:
            ggml_cuda_op_argsort(ctx, dst);
            for (int r = strix_repeat("argsort"); r > 0; --r) { ggml_cuda_op_argsort(ctx, dst); }
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            ggml_cuda_flash_attn_ext(ctx, dst);
            if (dst->ne[1] == 24 && dst->ne[2] == 1) {
                for (int r = strix_repeat("fa"); r > 0; --r) { ggml_cuda_flash_attn_ext(ctx, dst); }
            }
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS:
            ggml_cuda_cross_entropy_loss(ctx, dst);
            break;
        case GGML_OP_TRI:
            ggml_cuda_op_tri(ctx, dst);
            break;
        case GGML_OP_RWKV_WKV6:
            ggml_cuda_op_rwkv_wkv6(ctx, dst);
            break;
        case GGML_OP_GATED_LINEAR_ATTN:
            ggml_cuda_op_gated_linear_attn(ctx, dst);
            break;
        case GGML_OP_GATED_DELTA_NET:
            ggml_cuda_op_gated_delta_net(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_COMB:
            ggml_cuda_op_dsv4_hc_comb(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_PRE:
            ggml_cuda_op_dsv4_hc_pre(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_POST:
            ggml_cuda_op_dsv4_hc_post(ctx, dst);
            break;
        case GGML_OP_RWKV_WKV7:
            ggml_cuda_op_rwkv_wkv7(ctx, dst);
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            ggml_cuda_cross_entropy_loss_back(ctx, dst);
            break;
        case GGML_OP_OPT_STEP_ADAMW:
            ggml_cuda_opt_step_adamw(ctx, dst);
            break;
        case GGML_OP_OPT_STEP_SGD:
            ggml_cuda_opt_step_sgd(ctx, dst);
            break;
        case GGML_OP_SOLVE_TRI:
            ggml_cuda_op_solve_tri(ctx, dst);
            break;
        case GGML_OP_FILL:
            ggml_cuda_op_fill(ctx, dst);
            break;
        case GGML_OP_LIGHTNING_INDEXER:
            ggml_cuda_lightning_indexer(ctx, dst);
            break;
        case GGML_OP_QSA_PACK:
            ggml_cuda_qsa_pack(ctx, dst);
            break;
        case GGML_OP_MUL_MAT_HAD:
            ggml_cuda_mul_mat_had(ctx, dst);
            break;
        case GGML_OP_TOP_K_MASK:
            ggml_cuda_op_top_k_mask(ctx, dst);
            break;
        default:
            return false;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("%s: %s failed\n", __func__, ggml_op_desc(dst));
        // strixllama: name the node, its sources and where each lies in its buffer - what a report of a kernel fault needs
        GGML_LOG_ERROR("  dst '%s' %s [%lld,%lld,%lld,%lld] nb [%zu,%zu,%zu] off %lld of buffer %zu, extent %zu\n", dst->name, ggml_type_name(dst->type),
            (long long) dst->ne[0], (long long) dst->ne[1], (long long) dst->ne[2], (long long) dst->ne[3], dst->nb[1], dst->nb[2], dst->nb[3],
            dst->buffer ? (long long) ((const char *) dst->data - (const char *) ggml_backend_buffer_get_base(dst->buffer)) : -1LL,
            dst->buffer ? ggml_backend_buffer_get_size(dst->buffer) : (size_t) 0, ggml_nbytes(dst));
        for (int k = 0; k < GGML_MAX_SRC && dst->src[k]; ++k) {
            const ggml_tensor * s = dst->src[k];
            const ggml_tensor * base = s->view_src ? s->view_src : s;
            const char * b0 = s->buffer ? (const char *) ggml_backend_buffer_get_base(s->buffer) : nullptr;
            const size_t bsz = s->buffer ? ggml_backend_buffer_get_size(s->buffer) : 0;
            GGML_LOG_ERROR("  src%d '%s' (%s, view of '%s' %s) %s [%lld,%lld,%lld,%lld] nb [%zu,%zu,%zu] off %lld of buffer %zu, extent %zu\n",
                k, s->name, ggml_op_desc(s), base->name, ggml_op_desc(base), ggml_type_name(s->type),
                (long long) s->ne[0], (long long) s->ne[1], (long long) s->ne[2], (long long) s->ne[3], s->nb[1], s->nb[2], s->nb[3],
                b0 ? (long long) ((const char *) s->data - b0) : -1LL, bsz, ggml_nbytes(s));
        }
        CUDA_CHECK(err);
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////

// backend

static const char * ggml_backend_cuda_get_name(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    return cuda_ctx->name.c_str();
}

static void ggml_backend_cuda_free(ggml_backend_t backend) {
    ggml_cuda_mmb_release_all();   // release cached BF16 conversions before the pool is destroyed (pool leak assert)
    ggml_cuda_hc_release();
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    delete cuda_ctx;
    delete backend;
}

// strixllama: a stream's wait by polling host memory instead of hipStreamSynchronize (STRIX_POLL_SYNC; 0: the runtime's
// sync; ab: by the side of the in-run A/B). On Windows the runtime reports a 2000-node graph done ~0.65 ms after its last
// kernel has finished, and the last batch of a launch is not submitted until the host synchronizes or queries
// (ROCm/TheRock#8786). So the stream's end writes a sequence number into fine-grained host memory, an event is recorded
// and queried once (which submits what the launch left batched), and the host polls the number. Past 100 ms (a prefill
// ubatch) the runtime's sync takes over: a spinning core buys nothing there, and that sync also reports a fault the poll
// cannot see. A stream nothing has gone into since it was last seen done is not waited for at all: the scheduler
// synchronizes before every input it copies, and a poll's round trip each time (or the runtime's sync, still behind on
// a stream the poll saw done) cost more than the poll saves. So every way work enters a stream marks it (strix_poll_dirty):
// the graph, the async copies, an event wait.
static int strix_poll_sync_mode() {
    static const int m = [] {
        const char * e = getenv("STRIX_POLL_SYNC");
        return !e ? 1 : strcmp(e, "ab") == 0 ? 2 : atoi(e) != 0 ? 1 : 0;
    }();
    return m;
}

struct strix_poll_state {
    unsigned long long * host  = nullptr;  // fine-grained host memory: the GPU's write needs no flush to be seen
    unsigned long long * dev   = nullptr;
    unsigned long long   seq   = 0;
    cudaEvent_t          ev    = nullptr;
    bool                 ok    = true;
    bool                 dirty = true;     // work went in since the stream was last seen done
    bool                 dma   = false;    // a DMA copy to the host went in: the poll does not see it end
};

static strix_poll_state * strix_poll_get(cudaStream_t stream) {
    static std::mutex mtx;
    static std::unordered_map<cudaStream_t, strix_poll_state> states;
    std::lock_guard<std::mutex> lock(mtx);
    return &states[stream];
}

static void strix_poll_dirty(cudaStream_t stream) {
    if (strix_poll_sync_mode() != 0) {
        strix_poll_get(stream)->dirty = true;
    }
}

static void ggml_backend_cuda_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    strix_poll_dirty(cuda_ctx->stream());
    CUDA_CHECK(cudaMemcpyAsync((char *) tensor->data + offset, data, size, cudaMemcpyHostToDevice, cuda_ctx->stream()));
}

#if defined(GGML_USE_HIP)
// strixllama: with the polling sync, a copy to the host must end before the poll's mark: the mark is a kernel of the
// stream, and seeing it run did not mean a DMA copy queued before it had landed (0.6.7: the 2-3 MB of logits of an MTP
// verification were read partly old, and under sampling EXL3 3.05 answered in garbled text). So a copy into pinned host
// memory is a kernel of the same queue, every store at system scope; any other copy to the host makes the next sync the
// runtime's (strix_poll_state::dma)
static __global__ void strix_d2h_words(const uint32_t * __restrict__ src, uint32_t * dst, const size_t n) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        __hip_atomic_store(dst + i, src[i], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    }
    __threadfence_system();
}

// the device address of pinned host memory at p (nullptr: not pinned host memory)
static void * strix_pinned_device_ptr(void * p) {
    hipPointerAttribute_t attr = {};
    if (hipPointerGetAttributes(&attr, p) != hipSuccess) {
        (void) hipGetLastError();
        return nullptr;
    }
    return attr.type == hipMemoryTypeHost ? attr.devicePointer : nullptr;
}

// true when the copy went in as a kernel
static bool strix_poll_d2h(cudaStream_t stream, void * dst, const void * src, const size_t size) {
    if (strix_poll_sync_mode() == 0) {
        return false;
    }
    static const bool kernel_env = !getenv("STRIX_POLL_D2H_KERNEL") || atoi(getenv("STRIX_POLL_D2H_KERNEL")) != 0;
    void * d = kernel_env && size % 4 == 0 && ((uintptr_t) dst) % 4 == 0 && ((uintptr_t) src) % 4 == 0 ? strix_pinned_device_ptr(dst) : nullptr;
    if (d == nullptr) {
        strix_poll_get(stream)->dma = true;
        return false;
    }
    const size_t n = size / 4;
    if (n == 0) {
        return true;
    }
    const unsigned blocks = (unsigned) std::min<size_t>((n + 255) / 256, 1024);
    strix_d2h_words<<<blocks, 256, 0, stream>>>((const uint32_t *) src, (uint32_t *) d, n);
    CUDA_CHECK(cudaGetLastError());
    return true;
}
#endif // defined(GGML_USE_HIP)

static void ggml_backend_cuda_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    strix_poll_dirty(cuda_ctx->stream());
#if defined(GGML_USE_HIP)
    if (strix_poll_d2h(cuda_ctx->stream(), data, (const char *) tensor->data + offset, size)) {
        return;
    }
#endif // defined(GGML_USE_HIP)
    CUDA_CHECK(cudaMemcpyAsync(data, (const char *) tensor->data + offset, size, cudaMemcpyDeviceToHost, cuda_ctx->stream()));
}

static void ggml_backend_cuda_set_tensor_2d_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    strix_poll_dirty(cuda_ctx->stream());
    CUDA_CHECK(cudaMemcpy2DAsync(
        (char *) tensor->data + offset, stride_tensor, data, stride_data, size, n_copies, cudaMemcpyHostToDevice, cuda_ctx->stream()));
}

static void ggml_backend_cuda_get_tensor_2d_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    strix_poll_dirty(cuda_ctx->stream());
    if (strix_poll_sync_mode() != 0) {
        strix_poll_get(cuda_ctx->stream())->dma = true;   // a DMA copy to the host: the next sync is the runtime's
    }
    CUDA_CHECK(cudaMemcpy2DAsync(
        data, stride_data, (const char *) tensor->data + offset, stride_tensor, size, n_copies, cudaMemcpyDeviceToHost, cuda_ctx->stream()));
}

static bool ggml_backend_cuda_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    ggml_backend_buffer_t buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;

    if (!ggml_backend_is_cuda(backend_src) || !ggml_backend_is_cuda(backend_dst)) {
        return false;
    }

    if (!ggml_backend_buffer_is_cuda(buf_src) || !ggml_backend_buffer_is_cuda(buf_dst)) {
        return false;
    }

    // device -> device copy
    ggml_backend_cuda_context * cuda_ctx_src = (ggml_backend_cuda_context *) backend_src->context;
    ggml_backend_cuda_context * cuda_ctx_dst = (ggml_backend_cuda_context *) backend_dst->context;
    strix_poll_dirty(cuda_ctx_src->stream());
    strix_poll_dirty(cuda_ctx_dst->stream());

    ggml_backend_cuda_buffer_context * buf_ctx_src = (ggml_backend_cuda_buffer_context *) buf_src->context;
    ggml_backend_cuda_buffer_context * buf_ctx_dst = (ggml_backend_cuda_buffer_context *) buf_dst->context;

    if (cuda_ctx_src->device != buf_ctx_src->device || cuda_ctx_dst->device != buf_ctx_dst->device) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: backend and buffer devices do not match\n", __func__);
#endif // NDEBUG
        return false;
    }

    if (backend_src != backend_dst) {
        // copy on src stream
        // compare the backing physical devices: distinct virtual devices may share one physical GPU,
        // in which case a same-device copy (not a peer copy) is required
        const int src_physical = ggml_cuda_get_physical_device(cuda_ctx_src->device);
        const int dst_physical = ggml_cuda_get_physical_device(cuda_ctx_dst->device);
        if (src_physical == dst_physical) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, cuda_ctx_src->stream()));
        } else {
#ifdef GGML_CUDA_NO_PEER_COPY
            return false;
#else
            CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, dst_physical, src->data, src_physical, ggml_nbytes(dst), cuda_ctx_src->stream()));
#endif // GGML_CUDA_NO_PEER_COPY
        }

        // record event on src stream after the copy
        if (!cuda_ctx_src->copy_event) {
            ggml_cuda_set_device(cuda_ctx_src->device);
            CUDA_CHECK(cudaEventCreateWithFlags(&cuda_ctx_src->copy_event, cudaEventDisableTiming));
        }

        CUDA_CHECK(cudaEventRecord(cuda_ctx_src->copy_event, cuda_ctx_src->stream()));

        // wait on dst stream for the copy to complete
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx_dst->stream(), cuda_ctx_src->copy_event, 0));
    } else {
        // src and dst are on the same backend
        CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, cuda_ctx_src->stream()));
    }
    return true;
}

#if defined(GGML_USE_HIP)
static __global__ void strix_poll_mark(unsigned long long * flag, unsigned long long seq) {
    __threadfence_system();
    *(volatile unsigned long long *) flag = seq;
}

// true when the stream's work is done; false: wait with the runtime's sync instead
static bool strix_poll_wait(cudaStream_t stream, strix_poll_state * st) {
    if (!st->ok) {
        return false;
    }
    if (st->host == nullptr) {
        st->ok = hipHostMalloc((void **) &st->host, 64, hipHostMallocCoherent | hipHostMallocMapped) == hipSuccess &&
                 hipHostGetDevicePointer((void **) &st->dev, st->host, 0) == hipSuccess &&
                 hipEventCreateWithFlags(&st->ev, hipEventDisableTiming) == hipSuccess;
        (void) hipGetLastError();
        if (!st->ok) {
            GGML_LOG_WARN("%s: no fine-grained host memory to poll, the runtime's sync waits instead\n", __func__);
            return false;
        }
        *(volatile unsigned long long *) st->host = 0;
    }
    const unsigned long long seq = ++st->seq;
    strix_poll_mark<<<1, 1, 0, stream>>>(st->dev, seq);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(hipEventRecord(st->ev, stream));
    (void) hipEventQuery(st->ev);
    (void) hipGetLastError();                   // hipErrorNotReady, which a later check must not take for a failure
    const volatile unsigned long long * h = st->host;
    const int64_t t0 = ggml_time_us();
    for (uint32_t i = 1; *h < seq; ++i) {
#if defined(__x86_64__) || defined(_M_X64)
        _mm_pause();
#endif
        if ((i & 4095) == 0 && ggml_time_us() - t0 > 100000) {
            return false;
        }
    }
    return true;
}
#endif // defined(GGML_USE_HIP)

static void ggml_backend_cuda_synchronize(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    const int mode = strix_poll_sync_mode();
    if (mode != 0) {
        strix_poll_state * st = strix_poll_get(cuda_ctx->stream());
        if (!st->dirty) {
            return;                             // nothing went in since the stream was last seen done
        }
#if defined(GGML_USE_HIP)
        if (!st->dma && (mode == 1 || ggml_cuda_strix_ab()) && strix_poll_wait(cuda_ctx->stream(), st)) {
            st->dirty = false;
            return;
        }
#endif
        CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));
        st->dirty = false;
        st->dma   = false;
        return;
    }
    CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));

    GGML_UNUSED(backend);
}

// strixllama ablation helper: which op classes STRIX_SKIP_OPS asks to replace by a zero fill
static bool strixllama_skip_op(const ggml_tensor * node) {
    static const std::vector<std::string> wanted = []() {
        std::vector<std::string> v;
        if (const char * e = getenv("STRIX_SKIP_OPS")) {
            std::string cur;
            for (const char * c = e; ; ++c) {
                if (*c == ',' || *c == 0) { if (!cur.empty()) { v.push_back(cur); } cur.clear(); if (*c == 0) { break; } }
                else { cur.push_back(*c); }
            }
            if (!v.empty()) { fprintf(stderr, "STRIX_SKIP_OPS: %zu op classes zero-filled instead of computed\n", v.size()); }
        }
        return v;
    }();
    if (wanted.empty()) { return false; }
    const char * op = ggml_op_name(node->op);
    for (const auto & w : wanted) {
        // DEV: OP@ne0 skips only the nodes of that op whose ne[0] is ne0 (e.g. MUL_MAT@320); OP@lo-hi those whose ne[0]
        // is in [lo, hi]; * for OP takes every op
        const size_t at = w.find('@');
        if (at != std::string::npos) {
            const bool any_op = at == 1 && w[0] == '*';
            if (any_op || w.compare(0, at, op) == 0) {
                const long long lo = atoll(w.c_str() + at + 1);
                const size_t dash = w.find('-', at + 1);
                const long long hi = dash != std::string::npos ? atoll(w.c_str() + dash + 1) : lo;
                if (node->ne[0] >= lo && node->ne[0] <= hi) { return true; }
            }
            continue;
        }
        const size_t colon = w.find(':');
        if (colon == std::string::npos) { if (w == op) { return true; } continue; }
        if (w.compare(0, colon, op) != 0) { continue; }
        if (node->op == GGML_OP_UNARY && w.compare(colon + 1, std::string::npos, ggml_unary_op_name(ggml_get_unary_op(node))) == 0) { return true; }
        if (node->op == GGML_OP_GLU && w.compare(colon + 1, std::string::npos, ggml_glu_op_name(ggml_get_glu_op(node))) == 0) { return true; }
    }
    return false;
}

static bool ggml_cuda_is_view_or_noop(const ggml_tensor * t) {
    return ggml_is_empty(t) || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE ||
           t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_NONE;
}

#ifdef USE_CUDA_GRAPH
static bool ggml_cuda_graph_debug_enabled() {
    static const bool enabled = ggml_cuda_env_flag("GGML_CUDA_GRAPH_DEBUG");
    return enabled;
}

// strixllama: a graph the scheduler did not split again - llama reused it, so the split keeps its cgraph and its uid - has the
// HIP graph key and the compatibility it had. Hashing a decode graph's ~8000 nodes and walking them again for compatibility
// took ~0.28 ms of every step on the host before the launch, the GPU idle
struct ggml_cuda_graph_memo {
    const ggml_cgraph * g   = nullptr;
    uint64_t            uid = 0;
    uint64_t            key = 0;
    int                 compat = -1;
};
static ggml_cuda_graph_memo * ggml_cuda_graph_memo_find(const ggml_cgraph * cgraph, bool add) {
    static ggml_cuda_graph_memo memo[8];
    static int next = 0;
    if (cgraph->uid == 0) {
        return nullptr;
    }
    for (auto & m : memo) {
        if (m.g == cgraph && m.uid == cgraph->uid) {
            return &m;
        }
    }
    if (!add) {
        return nullptr;
    }
    ggml_cuda_graph_memo & m = memo[next];
    next = (next + 1) % 8;
    m = {};
    m.g   = cgraph;
    m.uid = cgraph->uid;
    return &m;
}

static bool ggml_cuda_graph_check_compability_scan(ggml_cgraph * cgraph);

static bool ggml_cuda_graph_check_compability(ggml_cgraph * cgraph) {
    ggml_cuda_graph_memo * m = ggml_cuda_graph_memo_find(cgraph, true);
    if (m && m->compat >= 0) {
        return m->compat != 0;
    }
    const bool res = ggml_cuda_graph_check_compability_scan(cgraph);
    if (m) {
        m->compat = res ? 1 : 0;
    }
    return res;
}

static bool ggml_cuda_graph_check_compability_scan(ggml_cgraph * cgraph) {

    bool use_cuda_graph = true;
    // Loop over nodes in GGML graph to obtain info needed for CUDA graph

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        if (ggml_cuda_is_view_or_noop(node)) {
            continue;
        }

        // [TAG_MUL_MAT_ID_CUDA_GRAPHS]
        if (node->op == GGML_OP_MUL_MAT_ID) {
            const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
            if (ggml_cuda_mul_mat_id_needs_sync(node, cc)) {
                // the mul_mat_id fallback path synchronizes the stream, so we cannot use CUDA graphs
                // ref: https://github.com/ggml-org/llama.cpp/pull/18958
                use_cuda_graph = false;
                if (getenv("LLAMA_GRAPH_TRACE")) {   // the notice below is compiled out of a release build
                    static unsigned strixllama_hits = 0;
                    if (strixllama_hits++ < 4) {
                        fprintf(stderr, "GRAPH_TRACE: graphs OFF, mul_mat_id needs sync: %s type=%s ne=[%lld,%lld,%lld]\n",
                                node->name, ggml_type_name(node->src[0]->type),
                                (long long) node->ne[0], (long long) node->ne[1], (long long) node->ne[2]);
                    }
                }
                if (ggml_cuda_graph_debug_enabled()) {
                    GGML_LOG_DEBUG("%s: disabling CUDA graphs due to unsupported node type\n", __func__);
                }
            }
        }

        if (!use_cuda_graph) {
            break;
        }
    }

    return use_cuda_graph;
}

static const void * ggml_cuda_graph_get_key(ggml_cgraph * cgraph) {
    // strixllama: fold the graph's shape into the key. One context alternates between batch shapes on the
    // same first node (the MTP draft's catch-up at 1+n_draft tokens then its draft steps at 1, the
    // target at 1+n_draft), and keying on the first node alone made every switch look like a property
    // change: warmup reset, the next graph run direct, the one after captured again. With the shape in
    // the key each shape keeps its own instance and replays. The value is only ever a map key.
    // STRIX_GRAPH_KEY_SHAPE=0 restores the first-node key.
    static const bool by_shape = [] {
        const char * e = getenv("STRIX_GRAPH_KEY_SHAPE");
        return !e || atoi(e) != 0;
    }();
    if (!by_shape || cgraph->n_nodes == 0) {
        return cgraph->nodes[0];
    }
    ggml_cuda_graph_memo * m = ggml_cuda_graph_memo_find(cgraph, true);
    if (m && m->key != 0) {
        return (const void *) (uintptr_t) m->key;
    }
    uint64_t h = (uint64_t) (uintptr_t) cgraph->nodes[0];
    h ^= (uint64_t) cgraph->n_nodes * 0x9E3779B97F4A7C15ull;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * t = cgraph->nodes[i];
        h = (h ^ (uint64_t) t->op) * 0x100000001B3ull;
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            h = (h ^ (uint64_t) t->ne[d]) * 0x100000001B3ull;
        }
    }
    if (m) {
        m->key = h;
    }
    return (const void *) (uintptr_t) h;
}

static ggml_cuda_graph::node_properties ggml_cuda_graph_node_props(const ggml_tensor * node) {
    ggml_cuda_graph::node_properties prop = {};
    memcpy(&prop.node, node, sizeof(ggml_tensor));

    for (int j = 0; j < GGML_MAX_SRC; ++j) {
        if (node->src[j]) {
            prop.node_src_data_ptrs[j] = node->src[j]->data;
            memcpy(prop.node_src_ne[j], node->src[j]->ne, sizeof(prop.node_src_ne[j]));
            memcpy(prop.node_src_nb[j], node->src[j]->nb, sizeof(prop.node_src_nb[j]));
        }
    }

    return prop;
}

static bool ggml_cuda_graph_verify_uid() {
#ifndef NDEBUG
    return true;
#else
    static const bool verify = getenv("GGML_CUDA_GRAPH_VERIFY_UID") != nullptr;
    return verify;
#endif
}

// see docs/development/backend-scheduler.md
static bool ggml_cuda_graph_update_required(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, const void * graph_key) {
    bool res = false;

    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);

    if (cgraph->uid != 0 &&
        cgraph->uid == graph->uid) {
        if (ggml_cuda_graph_debug_enabled()) {
            GGML_LOG_DEBUG("CUDA Graph id %zu reused\n", cgraph->uid);
        }
        GGML_ASSERT((int)graph->node_props.size() == cgraph->n_nodes);
        if (ggml_cuda_graph_verify_uid()) {
            for (int i = 0; i < cgraph->n_nodes; i++) {
                const ggml_cuda_graph::node_properties prop = ggml_cuda_graph_node_props(cgraph->nodes[i]);
                if (memcmp(&graph->node_props[i], &prop, sizeof(prop)) != 0) {
                    GGML_LOG_ERROR("%s: node %d (%s) changed while cgraph->uid stayed %llu\n",
                            __func__, i, cgraph->nodes[i]->name, (unsigned long long) cgraph->uid);
                    GGML_ABORT("CUDA graph uid reused after node properties changed - see [TAG_CUDA_GRAPH_UID]");
                }
            }
        }
        return false;
    }

    graph->uid = cgraph->uid;

    // Check if the graph size has changed
    if ((int)graph->node_props.size() != cgraph->n_nodes) {
        res = true;
        graph->node_props.resize(cgraph->n_nodes);
    }

    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_cuda_graph::node_properties prop = ggml_cuda_graph_node_props(cgraph->nodes[i]);

        if (res || memcmp(&graph->node_props[i], &prop, sizeof(prop)) != 0) {
            graph->node_props[i] = prop;
            res = true;
        }
    }

    return res;
}

static void ggml_cuda_graph_update_executable(ggml_backend_cuda_context * cuda_ctx, const void * graph_key) {
    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);

#if CUDART_VERSION >= 12000
    cudaGraphExecUpdateResultInfo result_info;
    cudaError_t stat = cudaGraphExecUpdate(graph->instance, graph->graph, &result_info);
#else
    cudaGraphNode_t errorNode;
    cudaGraphExecUpdateResult result_info;
    cudaError_t stat = cudaGraphExecUpdate(graph->instance, graph->graph, &errorNode, &result_info);
#endif // CUDART_VERSION >= 12000

    if (stat == cudaErrorGraphExecUpdateFailure) {
        if (ggml_cuda_graph_debug_enabled()) {
            GGML_LOG_DEBUG("%s: CUDA graph update failed\n", __func__);
        }

        // The pre-existing graph exec cannot be updated due to violated constraints
        // so instead clear error and re-instantiate
        (void)cudaGetLastError();
        CUDA_CHECK(cudaGraphExecDestroy(graph->instance));
        graph->instance = nullptr;
        CUDA_CHECK(cudaGraphInstantiate(&graph->instance, graph->graph, NULL, NULL, 0));
    } else {
        GGML_ASSERT(stat == cudaSuccess);
    }
}
#endif // USE_CUDA_GRAPH

static bool ggml_cuda_should_fuse_rope_set_rows(const ggml_tensor * rope,
                                                const ggml_tensor * view,
                                                const ggml_tensor * set_rows) {

    if (rope->op != GGML_OP_ROPE || view->op != GGML_OP_VIEW || set_rows->op != GGML_OP_SET_ROWS) {
        return false;
    }
    // ne3 not tested
    if (rope->src[0]->ne[3] != 1) {
        return false;
    }

    if (set_rows->type != GGML_TYPE_F32 && set_rows->type != GGML_TYPE_F16) {
        return false;
    }

    if (set_rows->src[1]->type != GGML_TYPE_I64) {
        return false;
    }

    // The view should flatten two dims of rope into one dim
    if (!ggml_is_contiguous(view) || view->ne[0] != rope->ne[0] * rope->ne[1]) {
        return false;
    }

    // Only norm/neox shaders have the fusion code
    const int mode = ((const int32_t *) rope->op_params)[2];
    if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX) {
        return false;
    }

    return true;
}

static bool ggml_cuda_should_fuse_rms_norm_mul_rope(const ggml_tensor * rms_norm,
                                                    const ggml_tensor * mul,
                                                    const ggml_tensor * rope) {
    if (rms_norm->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL || rope->op != GGML_OP_ROPE) {
        return false;
    }

    if (rms_norm->src[0]->type != GGML_TYPE_F32 || rms_norm->type != GGML_TYPE_F32 ||
        mul->src[0]->type != GGML_TYPE_F32 || mul->src[1]->type != GGML_TYPE_F32 ||
        mul->type != GGML_TYPE_F32 || rope->type != GGML_TYPE_F32) {
        return false;
    }

    if (rope->src[0] != mul) {
        return false;
    }

    //if rms norm is the B operand, then we don't handle broadcast
    if (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
        return false;
    }

    if (!ggml_are_same_shape(rms_norm, mul)) {
        return false;
    }

    //rms_norm kernel assumes contiguous rows
    if (!ggml_is_contiguous_rows(rms_norm->src[0]) ||
        !ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
        return false;
    }

    // the fused kernel handles the norm/neox rope modes only
    const int mode = ((const int32_t *) rope->op_params)[2];
    if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX) {
        return false;
    }

    const int n_dims = ((const int32_t *) rope->op_params)[1];
    if (n_dims % 2 != 0 || rope->src[0]->ne[0] % 2 != 0) {
        return false;
    }

    // ggml_rope_set_offset is not yet supported in the fused kernel
    const int n_offs = ((const int32_t *) rope->op_params)[15];
    if (n_offs != 0) {
        return false;
    }

    return true;
}

// match gated_delta_net + the strided cpy that scatters its state snapshots into the cache
// (slot i -> rollback group i, slot 0 newest), so the kernel can write them and skip the cpy.
static int ggml_cuda_try_gdn_cache_fusion(
        const ggml_cgraph * cgraph, int node_idx, ggml_cuda_gated_delta_net_fused_cache & fused_state_cpy) {
    const ggml_tensor * gdn = cgraph->nodes[node_idx];
    // the kernel skips the snapshot tail, so the gdn output must not be a graph output; the deferred-rollback net and
    // the replay (op_params[1]) write no snapshots
    if (gdn->op != GGML_OP_GATED_DELTA_NET || gdn->type != GGML_TYPE_F32 ||
        (gdn->flags & GGML_TENSOR_FLAG_OUTPUT) || ggml_get_op_params_i32(gdn, 1) != 0) {
        return 0;
    }

    const ggml_tensor * src_v     = gdn->src[2];
    const int64_t       S_v       = src_v->ne[0];
    const int64_t       H         = src_v->ne[1];
    const int64_t       n_tokens  = src_v->ne[2];
    const int64_t       n_seqs    = src_v->ne[3];
    const int64_t       D         = S_v * S_v * H;
    const int64_t       K         = ggml_get_op_params_i32(gdn, 0); // snapshot slot count
    const int64_t       n_written = std::min<int64_t>(n_tokens, K); // newest n_written slots are written

    // snapshot tail starts right after the attention scores
    const size_t tail_off = ggml_row_size(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    // snapshot cpy is the first real node after the gdn (skip views/no-ops)
    const ggml_tensor * cpy  = nullptr;
    int                 skip = 0;
    for (int j = node_idx + 1; j < cgraph->n_nodes && cpy == nullptr; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(n)) {
            continue;
        }
        if (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return 0;
        }
        cpy  = n;
        skip = j - node_idx;
    }
    if (cpy == nullptr) {
        return 0;
    }

    const ggml_tensor * src = cpy->src[0]; // view of the gdn snapshot tail
    const ggml_tensor * dst = cpy->src[1]; // cache view the kernel writes to

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    if (src->op != GGML_OP_VIEW || src->view_src != gdn || src->view_offs != tail_off ||
        !ggml_is_contiguous(src)) {
        return 0;
    }

    // dst is the [D, n_seqs, n_written] cache view; require nb[1] == D (the per-seq stride the kernel
    // assumes). ggml_cpy pins src to the same element count.
    const std::array<int64_t, GGML_MAX_DIMS> expected_ne = { D, n_seqs, n_written, 1 };
    if (dst->op != GGML_OP_VIEW || dst->type != GGML_TYPE_F32 || dst->data == nullptr ||
        !std::equal(expected_ne.begin(), expected_ne.end(), dst->ne) ||
        dst->nb[0] != ggml_type_size(GGML_TYPE_F32) || dst->nb[1] != (size_t) ggml_row_size(GGML_TYPE_F32, D)) {
        return 0;
    }

    fused_state_cpy.data        = (float *) dst->data; // rollback group 0 (newest)
    fused_state_cpy.slot_stride = K > 1 ? (int64_t) (dst->nb[2] / sizeof(float)) : 0;
    return skip;
}

// strixllama: a batch's recurrent states are gathered (GET_ROWS by s_copy) into scratch before the gated delta net,
// which writes its new states and rollback snapshots to the cells they came from. When llama marks the gather ('SGIP'
// in op_params[0]: every sequence reads a row of its own cell - its newest state, or a rollback snapshot after
// rejected drafts), the net reads the row s_copy names straight from the cache and the gather is skipped: 3 MB a
// sequence a layer, ~7 ms a step at eight sequences. A sequence's rows are read and written by its own blocks only,
// and each warp reads its columns before it writes them. Only when the net's writes go to that tensor (the cache
// fusion) and nothing else reads the gathered rows. STRIX_GDN_GATHER_SKIP=0 (read on the llama side) keeps the gather.
struct ggml_cuda_gdn_direct { const ggml_tensor * gdn; const float * base; const int32_t * rows; };
static std::vector<ggml_cuda_gdn_direct> g_gdn_direct;

static bool ggml_cuda_gdn_gather_skip(const ggml_cgraph * cgraph, int i) {
    static const bool no_fusion = getenv("GGML_CUDA_DISABLE_FUSION") != nullptr && std::atoi(getenv("GGML_CUDA_DISABLE_FUSION"));
    const ggml_tensor * g = cgraph->nodes[i];
    if (no_fusion || g->op != GGML_OP_GET_ROWS || g->op_params[0] != 0x53474950 || g->type != GGML_TYPE_F32 ||
            g->src[0] == nullptr || g->src[0]->type != GGML_TYPE_F32 || (g->flags & GGML_TENSOR_FLAG_OUTPUT) ||
            ggml_node_get_use_count(cgraph, i) != 1) {
        return false;
    }
    // its one user is a view, used once in turn, by a gated delta net as its state (the users sit a few nodes on)
    auto first_user = [cgraph](int from, const ggml_tensor * t) {
        for (int j = from; j < cgraph->n_nodes; ++j) {
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                if (cgraph->nodes[j]->src[s] == t) {
                    return j;
                }
            }
        }
        return -1;
    };
    const int j_view = first_user(i + 1, g);
    if (j_view < 0) {
        return false;
    }
    const ggml_tensor * view = cgraph->nodes[j_view];
    if (!ggml_cuda_is_view_or_noop(view) || view->view_src != g || ggml_node_get_use_count(cgraph, j_view) != 1) {
        return false;
    }
    const int j_gdn = first_user(j_view + 1, view);
    if (j_gdn < 0 || cgraph->nodes[j_gdn]->op != GGML_OP_GATED_DELTA_NET || cgraph->nodes[j_gdn]->src[5] != view) {
        return false;
    }
    const ggml_tensor * gdn = cgraph->nodes[j_gdn];
    ggml_cuda_gated_delta_net_fused_cache fc;
    if (ggml_cuda_try_gdn_cache_fusion(cgraph, j_gdn, fc) <= 0) {
        return false;
    }
    // the cells written must be rows of the tensor the gather reads, a row one sequence's state [S_v, S_v, H], and
    // the indices one per sequence
    const ggml_tensor * all  = g->src[0];
    const ggml_tensor * idx  = g->src[1];
    const ggml_tensor * v    = gdn->src[2];
    const size_t        row  = ggml_row_size(GGML_TYPE_F32, v->ne[0] * v->ne[0] * v->ne[1]);
    const ptrdiff_t     off  = (const char *) fc.data - (const char *) all->data;
    if (off < 0 || (size_t) off >= ggml_nbytes(all) || off % (ptrdiff_t) all->nb[1] != 0 || all->nb[1] != g->nb[1] ||
            all->nb[1] != row || idx == nullptr || idx->type != GGML_TYPE_I32 || !ggml_is_contiguous(idx) ||
            idx->ne[0] != v->ne[3] || ggml_nelements(idx) != v->ne[3]) {
        return false;
    }
    g_gdn_direct.push_back({gdn, (const float *) all->data, (const int32_t *) idx->data});
    return true;
}

// match ssm_scan + the strided cpy that scatters its state snapshots into the cache, so the kernel writes them and skips the cpy
static int ggml_cuda_try_ssm_scan_cache_fusion(
        const ggml_cgraph * cgraph, int node_idx, ggml_cuda_ssm_scan_fused_cache & fused_state_cpy) {
    const ggml_tensor * ssm = cgraph->nodes[node_idx];
    // the kernel skips the snapshot tail, so the scan output must not be a graph output
    if (ssm->op != GGML_OP_SSM_SCAN || ssm->type != GGML_TYPE_F32 || (ssm->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return 0;
    }

    const int64_t K = ggml_get_op_params_i32(ssm, 0); // snapshot slot count

    const ggml_tensor * s = ssm->src[0];
    const ggml_tensor * x = ssm->src[1];
    const ggml_tensor * A = ssm->src[3];

    const int64_t d_state = s->ne[0];
    const int64_t D       = d_state * s->ne[1] * x->ne[1]; // d_state * head_dim * n_head
    const int64_t n_tok   = x->ne[2];
    const int64_t n_seqs  = x->ne[3];

    // only the mamba-2 kernels (group scan and SSD) write to the cache; mamba-1 still uses the cpy
    if (A->nb[1] != sizeof(float) || (d_state != 96 && d_state != 128 && d_state != 256)) {
        return 0;
    }

    // the scan reads its input rows from the cache (picked by ids), so with more than one seq a seq can read a row that another seq writes in the same launch
    if (n_seqs != 1) {
        return 0;
    }

    const int64_t n_written = std::min<int64_t>(n_tok, K);
    const size_t  tail_off  = ggml_row_size(GGML_TYPE_F32, ggml_nelements(x));

    // snapshot cpy is the first real node after the scan (skip views/no-ops)
    const ggml_tensor * cpy  = nullptr;
    int                 skip = 0;
    for (int j = node_idx + 1; j < cgraph->n_nodes && cpy == nullptr; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(n)) {
            continue;
        }
        if (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return 0;
        }
        cpy  = n;
        skip = j - node_idx;
    }
    if (cpy == nullptr) {
        return 0;
    }

    const ggml_tensor * src = cpy->src[0]; // view of the scan snapshot tail
    const ggml_tensor * dst = cpy->src[1]; // cache view the kernel writes to

    // src must be this scan's snapshot tail (contiguous, at the tail offset)
    if (src->op != GGML_OP_VIEW || src->view_src != ssm || src->view_offs != tail_off ||
        !ggml_is_contiguous(src)) {
        return 0;
    }

    // dst is the [D, n_seqs, n_written] cache view; require nb[1] == D, the per-seq stride the kernel takes from src0->nb[3]
    const std::array<int64_t, GGML_MAX_DIMS> expected_ne = { D, n_seqs, n_written, 1 };
    if (dst->op != GGML_OP_VIEW || dst->type != GGML_TYPE_F32 || dst->data == nullptr ||
        !std::equal(expected_ne.begin(), expected_ne.end(), dst->ne) ||
        dst->nb[0] != ggml_type_size(GGML_TYPE_F32) || dst->nb[1] != (size_t) ggml_row_size(GGML_TYPE_F32, D)) {
        return 0;
    }

    fused_state_cpy.data        = (float *) dst->data; // rollback slot 0 (newest)
    fused_state_cpy.slot_stride = K > 1 ? (int64_t) (dst->nb[2] / sizeof(float)) : 0;
    return skip;
}

// expert group selection from build_moe_ffn (n_expert_groups > 1)
// src0/src1 index into this table, -1 is the (biased) probs, -2 is not checked
static const struct {
    ggml_op op;
    int     src0;
    int     src1;
} topk_moe_group_pattern[] = {
    { GGML_OP_RESHAPE,  -1, -2 }, //  0: [n_exp_per_group, n_expert_groups, n_tokens]
    { GGML_OP_RESHAPE,   0, -2 }, //  1: [1, n_exp_per_group, n_expert_groups, n_tokens]
    { GGML_OP_ARGSORT,   0, -2 }, //  2
    { GGML_OP_VIEW,      2, -2 }, //  3: top-2 per group
    { GGML_OP_GET_ROWS,  1,  3 }, //  4
    { GGML_OP_RESHAPE,   4, -2 }, //  5
    { GGML_OP_SUM_ROWS,  5, -2 }, //  6: group scores
    { GGML_OP_RESHAPE,   6, -2 }, //  7
    { GGML_OP_ARGSORT,   7, -2 }, //  8
    { GGML_OP_VIEW,      8, -2 }, //  9: top n_group_used groups
    { GGML_OP_GET_ROWS,  0,  9 }, // 10
    { GGML_OP_FILL,      0, -2 }, // 11
    { GGML_OP_SET_ROWS, 10,  9 }, // 12: unused groups -> -inf
    { GGML_OP_RESHAPE,  12, -2 }, // 13: [n_expert, n_tokens]
};

static bool ggml_cuda_topk_moe_group_fusion(const struct ggml_cgraph * cgraph, int node_idx, const ggml_tensor * selection, ggml_cuda_topk_moe_args & args) {
#if defined(GGML_USE_HIP) && HIP_VERSION < 60300000
    GGML_UNUSED_VARS(cgraph, node_idx, selection, args);
    return false;
#else
    const int n_ops = (int) std::size(topk_moe_group_pattern);
    if (node_idx + n_ops > cgraph->n_nodes) {
        return false;
    }

    ggml_tensor ** nodes = cgraph->nodes + node_idx;
    auto src_ok = [&](const ggml_tensor * src, int idx) {
        return idx == -2 || src == (idx == -1 ? selection : nodes[idx]);
    };

    for (int i = 0; i < n_ops; i++) {
        const auto & p = topk_moe_group_pattern[i];
        if (nodes[i]->op != p.op || !src_ok(nodes[i]->src[0], p.src0) || !src_ok(nodes[i]->src[1], p.src1)) {
            return false;
        }
    }

    const int n_groups     = nodes[0]->ne[1];
    const int n_group_used = nodes[9]->ne[0];

    if (nodes[3]->ne[0] != 2 || nodes[12]->src[2] != nodes[11] || ggml_get_op_params_f32(nodes[11], 0) != -INFINITY ||
            n_groups > WARP_SIZE) {
        return false;
    }

    args.n_groups     = n_groups;
    args.n_group_used = n_group_used;
    return true;
#endif // defined(GGML_USE_HIP) && HIP_VERSION < 60300000
}

static bool ggml_cuda_topk_moe_fusion(const struct ggml_cgraph * cgraph, int node_idx, ggml_cuda_topk_moe_args & args) {
    args.sigmoid         = false;
    args.sqrt_softplus   = false;
    args.softmax         = false;
    args.delayed_softmax = false;
    args.prob_bias       = false;
    args.norm            = false;
    args.scale           = false;
    args.n_groups        = 0;
    args.n_group_used    = 0;

    const int      n_nodes = cgraph->n_nodes;
    ggml_tensor ** nodes   = cgraph->nodes;

    if (nodes[node_idx]->op == GGML_OP_SOFT_MAX) {
        args.softmax = true;
    }

    if (nodes[node_idx]->op == GGML_OP_UNARY) {
        const ggml_unary_op unary_op = ggml_get_unary_op(nodes[node_idx]);
        if (unary_op == GGML_UNARY_OP_SIGMOID) {
            args.sigmoid = true;
        } else if (unary_op == GGML_UNARY_OP_SOFTPLUS && node_idx + 1 < n_nodes &&
                   nodes[node_idx + 1]->op == GGML_OP_SQRT && nodes[node_idx + 1]->src[0] == nodes[node_idx]) {
            // sqrt(softplus(x)) scoring (DeepSeek-V4)
            args.sqrt_softplus = true;
            node_idx++;
        } else {
            return false;
        }
    }

    if (nodes[node_idx]->op == GGML_OP_ARGSORT) {
        args.delayed_softmax = true;
    }

    node_idx++;

    if (args.sigmoid || args.sqrt_softplus || args.softmax) {
        // SOFTMAX -> RESHAPE
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_RESHAPE ||
                nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        ggml_tensor * probs_reshaped = nodes[node_idx];
        node_idx++;

        if (node_idx >= n_nodes) {
            return false;
        }

        // src of bias add is the unreshaped probs (-2 instead of -1)
        const ggml_tensor * selection = nodes[node_idx - 2];
        if (nodes[node_idx]->op == GGML_OP_ADD && nodes[node_idx]->src[0] == selection) {
            args.prob_bias = true;
            selection = nodes[node_idx];
            node_idx++;
        }

        if (ggml_cuda_topk_moe_group_fusion(cgraph, node_idx, selection, args)) {
            node_idx += (int) std::size(topk_moe_group_pattern);
            selection = nodes[node_idx - 1];
        }

        // RESHAPE/ADD/group mask -> ARGSORT
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_ARGSORT || nodes[node_idx]->src[0] != selection) {
            return false;
        }

        node_idx++;

        // ARGSORT-> VIEW
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_VIEW ||
                nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }

        // kept groups must hold all used experts
        if (args.n_groups > 0 && nodes[node_idx]->ne[0] > args.n_group_used * (probs_reshaped->ne[1] / args.n_groups)) {
            return false;
        }
        node_idx++;

        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_GET_ROWS) {
            return false;
        }

        // GET_ROWS
        if (nodes[node_idx]->src[0] != probs_reshaped || nodes[node_idx]->src[1] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;
    } else if (args.delayed_softmax) {
        if (node_idx - 2 < 0) {
            return false;
        }
        ggml_tensor * probs_reshaped = nodes[node_idx - 2];

        // VIEW->ARGSORT
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_VIEW ||
            nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;

        // GET_ROWS
        if (node_idx >= n_nodes || nodes[node_idx]->src[1] != nodes[node_idx - 1] ||
                nodes[node_idx]->src[0] != probs_reshaped) {
            return false;
        }
        node_idx++;

        static const std::vector<ggml_op> remaining_ops = { GGML_OP_RESHAPE, GGML_OP_SOFT_MAX, GGML_OP_RESHAPE };

        for (const ggml_op op : remaining_ops) {
            if (node_idx >= n_nodes || nodes[node_idx]->op != op || nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
                return false;
            }
            node_idx++;
        }
    }

    // At this point we can check for norm + scale. Everything is now at least valid till the norm
    if (node_idx >= n_nodes) {
        return true;
    }

    if (nodes[node_idx]->op == GGML_OP_RESHAPE) {
        //check RESHAPE->SUM_ROWS->CLAMP->DIV->RESHAPE
        static const std::vector<ggml_op> norm_ops = { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP };

        args.norm = true;
        for (const ggml_op op : norm_ops) {
            if (node_idx < n_nodes && nodes[node_idx]->op == op && nodes[node_idx]->src[0] == nodes[node_idx - 1]) {
                node_idx++;
            } else {
                args.norm = false;
                return true;
            }
        }

        // DIV <- CLAMP, RESHAPE
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_DIV || nodes[node_idx]->src[1] != nodes[node_idx - 1] ||
            nodes[node_idx]->src[0] != nodes[node_idx - 3]) {
            args.norm = false;
            return true;
        }
        node_idx++;

        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_RESHAPE || nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            args.norm = false;
            return true;
        }

        node_idx++;
    }

    if (node_idx < n_nodes && nodes[node_idx]->op == GGML_OP_SCALE && nodes[node_idx]->src[0] == nodes[node_idx - 1]) {
        args.scale = true;
    }

    return true;
}

struct ggml_cuda_topk_moe_match {
    ggml_cuda_topk_moe_args args;
    const ggml_tensor * logits = nullptr;
    const ggml_tensor * bias = nullptr;
    const ggml_tensor * clamp = nullptr;
    const ggml_tensor * scale = nullptr;
    ggml_tensor * weights = nullptr;
    ggml_tensor * ids = nullptr;
    int node_count = 0;
    int out_nodes[2] = {};
};

static bool ggml_cuda_match_topk_moe(const ggml_cgraph * graph, int i, ggml_cuda_topk_moe_match & match) {
    const ggml_tensor * node = graph->nodes[i];
    if (node->op != GGML_OP_UNARY && node->op != GGML_OP_SOFT_MAX && node->op != GGML_OP_ARGSORT) {
        return false;
    }
    if (!ggml_cuda_topk_moe_fusion(graph, i, match.args)) {
        return false;
    }

    std::vector<ggml_op> ops;
    ops.reserve(13);  // max ops; avoids gcc -Wstringop-overflow false positive
    const ggml_tensor * gating = node;
    match.logits = node->src[0];
    if (!match.args.delayed_softmax) {
        if (match.args.sigmoid) {
            ops.insert(ops.end(), { GGML_OP_UNARY });
        } else if (match.args.sqrt_softplus) {
            ops.insert(ops.end(), { GGML_OP_UNARY, GGML_OP_SQRT });
        } else {
            ops.insert(ops.end(), { GGML_OP_SOFT_MAX });
        }
        ops.push_back(GGML_OP_RESHAPE);
        if (match.args.prob_bias) {
            ops.push_back(GGML_OP_ADD);
            match.bias = graph->nodes[i + ops.size() - 1]->src[1];
        }
        if (match.args.n_groups > 0) {
            for (const auto & p : topk_moe_group_pattern) {
                ops.push_back(p.op);
            }
        }
        ops.insert(ops.end(), { GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS });
        match.out_nodes[0] = i + ops.size() - 2;
        match.ids = graph->nodes[match.out_nodes[0]];
        if (match.args.norm) {
            ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE });
            match.clamp = graph->nodes[i + ops.size() - 3];
        }
        if (match.args.scale) {
            ops.insert(ops.end(), { GGML_OP_SCALE });
            match.scale = graph->nodes[i + ops.size() - 1];
        }
        match.out_nodes[1] = i + ops.size() - 1;
        match.weights = graph->nodes[match.out_nodes[1]];
    } else if (!match.args.norm && !match.args.prob_bias) {
        ops.insert(ops.end(), { GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_RESHAPE, GGML_OP_SOFT_MAX, GGML_OP_RESHAPE });
        match.weights = graph->nodes[i + 5];
        match.ids = graph->nodes[i + 1];
        gating = graph->nodes[i + 4];
        match.out_nodes[0] = i + 1;
        match.out_nodes[1] = i + 5;
    } else {
        return false;
    }
    match.node_count = ops.size();
    return ggml_can_fuse_subgraph(graph, i, ops.size(), ops.data(), match.out_nodes, 2) &&
           ggml_cuda_should_use_topk_moe(gating, match.weights, match.logits, match.ids);
}

// returns whether the write (out) nodes overwrite the read nodes in operation
static bool ggml_cuda_check_fusion_memory_ranges(const ggml_cgraph * cgraph,
                                                 const int           node_idx,
                                                 const int           node_count,
                                                 const int *         out_nodes,
                                                 const int           out_count,
                                                 const bool          is_topk_moe = false) {
    auto nodes_overlap = [&](const ggml_tensor * a, const ggml_tensor * b) {
        const int64_t a_start = (int64_t) a->data;
        const int64_t a_end   = a_start + ggml_backend_buft_get_alloc_size(a->buffer->buft, a);

        const int64_t b_start = (int64_t) b->data;
        const int64_t b_end   = b_start + ggml_backend_buft_get_alloc_size(b->buffer->buft, b);

        if ((b_start <= a_start && a_start < b_end) || (a_start <= b_start && b_start < a_end)) {
            return true;
        }

        return false;
    };

    bool is_ok = true;
    // one block reads all logits before it writes, so logits may alias the out nodes
    const ggml_tensor * logits_may_alias = nullptr;
    if (is_topk_moe && ggml_nrows(cgraph->nodes[node_idx]) <= TOPK_MOE_ROWS_PER_BLOCK) {
        logits_may_alias = cgraph->nodes[node_idx]->src[0];
    }

    for (int i = 0; i < out_count; ++i) {
        const ggml_tensor * dst = cgraph->nodes[out_nodes[i]];

        for (int j = node_idx; j < node_idx + node_count; ++j) {
            // Loop over all srcs of all nodes in the fusion. If the src overlaps
            // the destination and the src is not an intermediate node that's being
            // elided, then disable fusion.

            for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                const ggml_tensor * src = cgraph->nodes[j]->src[src_idx];

                if (!src || src->op == GGML_OP_NONE || src == logits_may_alias) {
                    continue;
                }

                if (nodes_overlap(dst, src)) {
                    bool found = false;

                    for (int k = node_idx; k < j; ++k) {
                        if (cgraph->nodes[k] == src) {
                            found = true;
                            break;
                        }
                    }

                    if (!found) {
                        is_ok = false;
                        break;
                    }
                }
            }
        }
    }

    return is_ok;
}

// The long form spans 2*k + 1 nodes. ggml_can_fuse_subgraph() accepts at most
// 31 nodes, so k <= 15; larger values use the per-operation path.
static constexpr int MOE_WEIGHTED_REDUCTION_MAX_EXPERTS = 15;

// strixllama: set by the LLAMA_GRAPH_TRACE counter so the fusion skip site can report how many nodes
// it swallowed; null whenever the trace is off
static unsigned * g_strixllama_fused_away = nullptr;

// strixllama: STRIX_NODE_TIMING=1 (needs GGML_CUDA_DISABLE_GRAPHS=1) brackets every dispatch with
// stream events and prints the costliest nodes of each graph; see the report after the dispatch loop
#if defined(GGML_USE_HIP) && !defined(cudaEventElapsedTime)
#define cudaEventElapsedTime hipEventElapsedTime
#endif
static const bool g_strixllama_node_timing = getenv("STRIX_NODE_TIMING") != nullptr;
static std::vector<cudaEvent_t> g_strixllama_ev0, g_strixllama_ev1;
static std::vector<int> g_strixllama_ev_node;

struct ggml_cuda_moe_weighted_reduction_match {
    const ggml_tensor * experts      = nullptr;
    const ggml_tensor * expert_scale = nullptr;
    const ggml_tensor * weights      = nullptr;
    ggml_tensor *       dst          = nullptr;
    int                 node_count   = 0;
};

static bool ggml_cuda_match_moe_weighted_reduction(
        const ggml_cgraph * cgraph,
        int node_idx,
        ggml_cuda_moe_weighted_reduction_match & match) {
    const ggml_tensor * first = cgraph->nodes[node_idx];
    if (first->op != GGML_OP_MUL || first->type != GGML_TYPE_F32 || !ggml_is_contiguous(first)) {
        return false;
    }

    auto split_mul = [](const ggml_tensor * mul, const ggml_tensor *& full, const ggml_tensor *& broadcast) {
        auto is_weights = [mul](const ggml_tensor * tensor) {
            return tensor && tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor) && tensor->ne[0] == 1 &&
                tensor->ne[1] == mul->ne[1] && tensor->ne[2] == mul->ne[2] && tensor->ne[3] == mul->ne[3];
        };
        auto is_experts = [mul](const ggml_tensor * tensor) {
            return tensor && tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor) &&
                ggml_are_same_shape(tensor, mul);
        };

        if (is_experts(mul->src[0]) && is_weights(mul->src[1])) {
            full      = mul->src[0];
            broadcast = mul->src[1];
            return true;
        }
        if (is_experts(mul->src[1]) && is_weights(mul->src[0])) {
            full      = mul->src[1];
            broadcast = mul->src[0];
            return true;
        }
        return false;
    };

    const ggml_tensor * weighted     = first;
    const ggml_tensor * experts      = nullptr;
    const ggml_tensor * expert_scale = nullptr;
    const ggml_tensor * weights      = nullptr;
    int                 mul_count    = 1;

    // Match both structural forms:
    //   (experts * expert_scale) * router_weight
    //   experts * router_weight
    // The matcher does not depend on the model or quantization type.
    if (node_idx + 1 < cgraph->n_nodes) {
        const ggml_tensor * second = cgraph->nodes[node_idx + 1];
        const ggml_tensor * scaled = nullptr;
        const ggml_tensor * route  = nullptr;
        const ggml_tensor * raw    = nullptr;
        const ggml_tensor * scale  = nullptr;
        if (second->op == GGML_OP_MUL && second->type == GGML_TYPE_F32 && ggml_is_contiguous(second) &&
                split_mul(second, scaled, route) && scaled == first && split_mul(first, raw, scale)) {
            weighted     = second;
            experts      = raw;
            expert_scale = scale;
            weights      = route;
            mul_count    = 2;
        }
    }

    if (experts == nullptr && !split_mul(first, experts, weights)) {
        return false;
    }

    const int     n_expert_used = (int) weighted->ne[1];
    const int64_t n_tokens      = weighted->ne[2] * weighted->ne[3];
    if (n_expert_used < 2 || n_expert_used > MOE_WEIGHTED_REDUCTION_MAX_EXPERTS) {
        return false;
    }

    const int node_count = 2 * n_expert_used + mul_count - 1;
    if (node_idx + node_count > cgraph->n_nodes) {
        return false;
    }

    std::vector<ggml_op> ops(node_count, GGML_OP_VIEW);
    ops[0] = GGML_OP_MUL;
    if (mul_count == 2) {
        ops[1] = GGML_OP_MUL;
    }
    std::vector<const ggml_tensor *> views;
    views.reserve(n_expert_used);
    const ggml_tensor * previous = nullptr;
    int n_adds = 0;
    for (int offset = mul_count; offset < node_count; ++offset) {
        const ggml_tensor * candidate = cgraph->nodes[node_idx + offset];
        ops[offset] = candidate->op;

        if (candidate->op == GGML_OP_VIEW) {
            const int expert = (int) views.size();
            if (expert >= n_expert_used || candidate->src[0] != weighted || candidate->view_src != weighted ||
                    candidate->type != GGML_TYPE_F32 || candidate->ne[0] != weighted->ne[0] ||
                    candidate->ne[1] != n_tokens || candidate->ne[2] != 1 || candidate->ne[3] != 1 ||
                    candidate->nb[0] != weighted->nb[0] || candidate->nb[1] != weighted->nb[2] ||
                    candidate->view_offs != (size_t) expert * weighted->nb[1]) {
                return false;
            }
            views.push_back(candidate);
            continue;
        }

        if (candidate->op != GGML_OP_ADD || views.size() < 2 || n_adds + 1 >= (int) views.size()) {
            return false;
        }
        const ggml_tensor * lhs = n_adds == 0 ? views[0] : previous;
        const ggml_tensor * rhs = views[n_adds + 1];
        if (candidate->src[0] != lhs || candidate->src[1] != rhs || candidate->type != GGML_TYPE_F32) {
            return false;
        }
        previous = candidate;
        ++n_adds;
    }

    if ((int) views.size() != n_expert_used || n_adds != n_expert_used - 1 || previous == nullptr) {
        return false;
    }
    if (!ggml_is_contiguous(previous) || previous->ne[0] != weighted->ne[0] ||
            previous->ne[1] != n_tokens || previous->ne[2] != 1 || previous->ne[3] != 1) {
        return false;
    }

    const int output_idx = node_idx + node_count - 1;
    if (!ggml_can_fuse_subgraph(cgraph, node_idx, node_count, ops.data(), &output_idx, 1)) {
        return false;
    }

    match.experts      = experts;
    match.expert_scale = expert_scale;
    match.weights      = weights;
    match.dst          = cgraph->nodes[output_idx];
    match.node_count   = node_count;
    return true;
}

struct ggml_cuda_get_rows_mean4_match {
    const ggml_tensor * src = nullptr;
    const ggml_tensor * ids = nullptr;
    ggml_tensor * dst = nullptr;
    int node_count = 14;
};

static bool ggml_cuda_match_get_rows_mean4(
        const ggml_cgraph * graph, int i, ggml_cuda_get_rows_mean4_match & match) {
    if (i < 0 || i + match.node_count > graph->n_nodes || graph->nodes[i]->op != GGML_OP_GET_ROWS) {
        return false;
    }
    if (!ggml_can_fuse_subgraph(graph, i,
            { GGML_OP_GET_ROWS, GGML_OP_RESHAPE,
              GGML_OP_VIEW, GGML_OP_CONT, GGML_OP_VIEW, GGML_OP_CONT, GGML_OP_ADD,
              GGML_OP_VIEW, GGML_OP_CONT, GGML_OP_ADD, GGML_OP_VIEW, GGML_OP_CONT, GGML_OP_ADD,
              GGML_OP_SCALE }, { i + 13 })) {
        return false;
    }
    const ggml_tensor * rows = graph->nodes[i];
    const ggml_tensor * src = rows->src[0];
    const ggml_tensor * ids = rows->src[1];
    const ggml_tensor * members = graph->nodes[i+1];
    ggml_tensor * dst = graph->nodes[i+13];
    if ((src->type != GGML_TYPE_F16 && src->type != GGML_TYPE_F32) || ids->type != GGML_TYPE_I32 ||
            src->ne[0] != 128 || src->ne[1] <= 0 || src->ne[2] <= 0 || src->nb[0] != ggml_type_size(src->type) || src->ne[3] != 1 ||
            ids->nb[0] != sizeof(int32_t) || ids->ne[0] <= 0 || ids->ne[0] % 4 != 0 || ids->ne[2] != 1 || ids->ne[3] != 1 ||
            ids->ne[1] != src->ne[2] || ids->ne[1] > 65535 || ids->ne[0]/4 > INT32_MAX) {
        return false;
    }
    if (members->src[0] != rows || members->ne[0] != 128 || members->ne[1] != 4 ||
            members->ne[2] != ids->ne[0]/4 || members->ne[3] != ids->ne[1] ||
            !ggml_is_contiguous(members)) {
        return false;
    }
    for (int n = 0; n < match.node_count; ++n) {
        if (graph->nodes[i+n]->type != GGML_TYPE_F32) {
            return false;
        }
    }
    const int offsets[4] = { 2, 4, 7, 10 };
    const ggml_tensor * sum = nullptr;
    for (int n = 0; n < 4; ++n) {
        const ggml_tensor * view = graph->nodes[i+offsets[n]];
        const ggml_tensor * copy = graph->nodes[i+offsets[n]+1];
        if (view->src[0] != members || view->ne[0] != 128 || view->ne[1] != members->ne[2] ||
                view->ne[2] != members->ne[3] || view->ne[3] != 1 || view->nb[0] != sizeof(float) ||
                view->nb[1] != members->nb[2] || view->nb[2] != members->nb[3] ||
                view->view_offs != members->view_offs + n*members->nb[1] ||
                copy->src[0] != view || !ggml_are_same_shape(copy, view) || !ggml_is_contiguous(copy)) {
            return false;
        }
        if (n == 0) {
            sum = copy;
        } else {
            const ggml_tensor * add = graph->nodes[i+offsets[n]+2];
            if (add->src[0] != sum || add->src[1] != copy || !ggml_are_same_shape(add, copy)) {
                return false;
            }
            sum = add;
        }
    }
    if (dst->src[0] != sum || !ggml_are_same_shape(dst, sum) || !ggml_is_contiguous(dst) ||
            ggml_get_op_params_f32(dst, 0) != 0.25f || ggml_get_op_params_f32(dst, 1) != 0.0f) {
        return false;
    }
    match.src = src;
    match.ids = ids;
    match.dst = dst;
    return true;
}

static bool ggml_cuda_mean4_ranges_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    const uintptr_t av = (uintptr_t) a->data;
    const uintptr_t bv = (uintptr_t) b->data;
    return av <= bv ? bv-av < ggml_nbytes(a) : av-bv < ggml_nbytes(b);
}

#include "match.inc"

static int ggml_cuda_match_hc_combine_norm(
        const ggml_cgraph * graph, int i, int warp_size, ggml_cuda_hc_combine_norm_args & args) {
    const ggml_tensor * scale1 = graph->nodes[i];
    if (scale1->op != GGML_OP_SCALE || i + 8 >= graph->n_nodes) {
        return 0;
    }
    const ggml_tensor * sigmoid = graph->nodes[i + 1];
    const ggml_tensor * scale2 = graph->nodes[i + 2];
    if (sigmoid->op != GGML_OP_UNARY || ggml_get_unary_op(sigmoid) != GGML_UNARY_OP_SIGMOID ||
            sigmoid->src[0] != scale1 || scale2->op != GGML_OP_SCALE || scale2->src[0] != sigmoid) {
        return 0;
    }
    int j = i + 3;
    while (j < graph->n_nodes && j < i + 7 && graph->nodes[j]->op == GGML_OP_RESHAPE) {
        ++j;
    }
    if (j + 4 >= graph->n_nodes) {
        return 0;
    }
    const ggml_tensor * repeat = graph->nodes[j];
    const ggml_tensor * mul = graph->nodes[j + 1];
    ggml_tensor * add = graph->nodes[j + 2];
    const ggml_tensor * rms = graph->nodes[j + 3];
    if (repeat->op != GGML_OP_REPEAT || mul->op != GGML_OP_MUL || add->op != GGML_OP_ADD ||
            rms->op != GGML_OP_RMS_NORM || rms->src[0] != add) {
        return 0;
    }
    int end = j + 4;
    const ggml_tensor * norm = rms;
    if (graph->nodes[end]->op == GGML_OP_RESHAPE) {
        norm = graph->nodes[end++];
        if (norm->src[0] != rms || norm->view_src != rms || end >= graph->n_nodes) {
            return 0;
        }
    }
    ggml_tensor * mulg = graph->nodes[end];
    if (mulg->op != GGML_OP_MUL || (mulg->src[0] != norm && mulg->src[1] != norm)) {
        return 0;
    }
    const ggml_tensor * inject = scale1->src[0];
    const ggml_tensor * block = repeat->src[0];
    const ggml_tensor * weights = mul->src[0] == repeat ? mul->src[1] : mul->src[0];
    const ggml_tensor * residual = add->src[0] == mul ? add->src[1] : add->src[0];
    const ggml_tensor * gamma = mulg->src[0] == norm ? mulg->src[1] : mulg->src[0];
    const int64_t width = add->ne[0], hc = add->ne[1], tokens = add->ne[2];
    for (const ggml_tensor * tensor : {scale1, sigmoid, scale2, inject, block, weights,
            repeat, mul, (const ggml_tensor *) add, residual, rms, norm, (const ggml_tensor *) mulg, gamma}) {
        if (tensor->type != GGML_TYPE_F32 || !ggml_is_contiguous(tensor)) {
            return 0;
        }
    }
    if (add->ne[3] != 1 || tokens < 1 || tokens > 65535 ||
            !ggml_are_same_shape(scale1, inject) || !ggml_are_same_shape(scale2, inject) ||
            ggml_nelements(inject) != hc * tokens || ggml_nrows(inject) != tokens ||
            !(weights == scale2 || (weights->op == GGML_OP_RESHAPE && weights->src[0] == scale2 &&
                weights->view_src == scale2)) ||
            weights->ne[0] != 1 || weights->ne[1] != hc || weights->ne[2] != tokens || weights->ne[3] != 1 ||
            block->ne[0] != width || block->ne[1] != 1 || block->ne[2] != tokens || block->ne[3] != 1 ||
            (mul->src[0] != repeat && mul->src[1] != repeat) || weights == repeat ||
            (add->src[0] != mul && add->src[1] != mul) || residual == mul || gamma == norm ||
            !ggml_are_same_shape(repeat, add) || !ggml_are_same_shape(mul, add) ||
            !ggml_are_same_shape(residual, add) || !ggml_are_same_shape(rms, add) ||
            !ggml_are_same_shape(mulg, norm) || ggml_nelements(norm) != ggml_nelements(add) ||
            ggml_nelements(gamma) != width * hc) {
        return 0;
    }
    if (norm == rms) {
        if (gamma->ne[0] != width || gamma->ne[1] != hc) {
            return 0;
        }
    } else if (norm->ne[0] != width * hc || norm->ne[1] != tokens || norm->ne[2] != 1 || norm->ne[3] != 1 ||
            gamma->ne[0] != width * hc) {
        return 0;
    }
    args.inject = inject;
    args.residual = residual;
    args.block_out = block;
    args.gamma = gamma;
    args.out_res = add;
    args.out_xn = mulg;
    args.s1 = ggml_get_op_params_f32(scale1, 0);
    args.b1 = ggml_get_op_params_f32(scale1, 1);
    args.s2 = ggml_get_op_params_f32(scale2, 0);
    args.b2 = ggml_get_op_params_f32(scale2, 1);
    args.eps = ggml_get_op_params_f32(rms, 0);
    if (!ggml_cuda_hc_combine_norm_supported(args, warp_size)) {
        return 0;
    }
    int indices[16], count = 0;
    ggml_op ops[16];
    for (int k = i; k <= end; ++k) {
        const ggml_tensor * tensor = graph->nodes[k];
        if (k > i + 2 && k < j && tensor != weights) {
            if (tensor != block) {
                return 0;
            }
            continue;
        }
        indices[count] = k;
        ops[count++] = tensor->op;
    }
    const int outputs[] = {j + 2, end};
    return ggml_can_fuse_subgraph_ext(graph, indices, count, ops, outputs, 2) ? end - i : 0;
}

static void ggml_cuda_hc_inject_plan(const ggml_cgraph * graph, const int last, ggml_cuda_hc_combine_norm_args & args,
        ggml_backend_cuda_context & ctx) {
    g_hc_inject = {};
    const ggml_tensor * xn = args.out_xn;
    const int64_t K = args.out_res->ne[0] * args.out_res->ne[1], T = args.out_res->ne[2];
    if (T <= 64 || !ggml_cuda_hc_inject_fusable(args) || !args.store_xn_f32 || xn->type != GGML_TYPE_F32 ||
            ggml_nelements(xn) != K * T) {
        return;
    }
    // the inject is expanded into the graph by the combine that reads it, so it follows the whole block
    for (int n = last + 1; n < graph->n_nodes && n <= last + 4096; ++n) {
        const ggml_tensor * t = graph->nodes[n];
        if (t->op != GGML_OP_MUL_MAT || !t->src[0] || !t->src[1]) {
            continue;
        }
        const ggml_tensor * w = t->src[0], * x = t->src[1];
        if (x != xn && !(x->view_src == xn && x->view_offs == 0)) {
            continue;
        }
        if (w->type != GGML_TYPE_F32 || !ggml_is_contiguous(w) || w->ne[0] != K || w->ne[1] != 4 || w->ne[2] != 1 ||
                w->ne[3] != 1 || ((uintptr_t) w->data) % 8 != 0 || x->ne[0] != K || x->ne[1] != T || x->ne[2] != 1 ||
                x->ne[3] != 1 || !ggml_is_contiguous(x) || t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t) ||
                t->ne[0] != 4 || t->ne[1] != T || t->ne[2] != 1 || t->ne[3] != 1 ||
                ggml_get_op_params_i32(t, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
            continue;
        }
        args.inject_w    = (const float *) w->data;
        args.inject_part = ggml_cuda_hc_inject_part(ctx);
        g_hc_inject = { t, xn->data, args.inject_part };
        return;
    }
}

static bool ggml_cuda_hc_combine_norm_alias_ok(const ggml_cuda_hc_combine_norm_args & args) {
    auto overlap = [](const ggml_tensor * a, const ggml_tensor * b) {
        const uintptr_t av = (uintptr_t) a->data, bv = (uintptr_t) b->data;
        return av <= bv ? bv - av < ggml_nbytes(a) : av - bv < ggml_nbytes(b);
    };
    for (const ggml_tensor * output : {args.out_res, args.out_xn}) {
        for (const ggml_tensor * input : {args.block_out, args.inject, args.gamma}) {
            if (overlap(output, input)) {
                return false;
            }
        }
        if (overlap(output, args.residual) && output->data != args.residual->data) {
            return false;
        }
    }
    return !overlap(args.out_res, args.out_xn);
}

#include "hc-match.inc"

// strixllama (STRIX_HC_XRES): whether the combine's F32 normalised streams have a reader left. The HC down GEMM reads the
// BF16 copy, the inject is folded into the combine (inject_plan), and the gate GEMM's fused mix, which with Q8_0 weights
// reads the F32 streams, can form them again from the combine's F32 residual output, the per-row scale it then stores and
// gamma - the same two products in the same order, so the same bits. With no other reader the combine stops writing
// them: 40 KB a token per combine, two combines a layer
static void ggml_cuda_hc_xres_plan(const ggml_cgraph * graph, const int last, ggml_cuda_hc_combine_norm_args & args,
        ggml_backend_cuda_context & ctx) {
    static const bool on = !getenv("STRIX_HC_XRES") || atoi(getenv("STRIX_HC_XRES")) != 0;
    ggml_cuda_hc_xres_set(nullptr, nullptr, nullptr, nullptr);
    const ggml_tensor * xn = args.out_xn;
    const int64_t hc = args.out_res->ne[1], T = args.out_res->ne[2];
    if (!on || !args.store_xn_f32 || !args.out_xn_bf16 || args.res_out_bf16 || hc != 4 || T < 512 ||
            args.out_res->type != GGML_TYPE_F32 || args.gamma->type != GGML_TYPE_F32 || !ggml_is_contiguous(args.out_res) ||
            !ggml_is_contiguous(args.gamma) || xn->type != GGML_TYPE_F32 || !ggml_cuda_hc_cn_b256(args)) {
        return;
    }
    bool formed = false;
    int nread = 0, last_read = last;
    for (int n = last + 1; n < graph->n_nodes; ++n) {
        const ggml_tensor * t = graph->nodes[n];
        bool reads = false;
        for (int s = 0; s < GGML_MAX_SRC && t->src[s]; ++s) {
            if (t->src[s] == xn || t->src[s]->view_src == xn) { reads = true; }
        }
        if (!reads) {
            continue;
        }
        ++nread;
        last_read = n;
        if (g_hc_inject.node == t) {
            continue;                       // the inject, folded into the combine
        }
        if (t->op == GGML_OP_MUL_MAT && t->src[1] == xn && ggml_cuda_mmb_supported_mm(t->src[0], t->src[1], t)) {
            continue;                       // the HC down GEMM: the BF16 copy
        }
        if (t->op == GGML_OP_MUL && n >= 2) {
            ggml_cuda_hc_mix_args ma;
            const ggml_tensor * g = graph->nodes[n - 2];
            bool f32 = false;
            if (ggml_cuda_hc_mix_closed(graph, n - 1, ma) > 0 && ma.xn == xn && ma.gate == g && g->op == GGML_OP_MUL_MAT &&
                    g->src[0]->ne[0] == 320 && g->src[0]->ne[1] == 10240 && ggml_node_has_n_uses(graph, n - 2, 1) &&
                    ggml_cuda_mmb_supported_mm(g->src[0], g->src[1], g) &&
                    ggml_cuda_hc_gate_mix_ok(g->src[0], g->src[1], xn, ma.dst, ma.hc, &f32)) {
                formed = formed || f32;
                continue;                   // the gate GEMM's fused mix
            }
        }
        return;                             // another reader: the F32 streams stay
    }
    if (nread == 0) {
        return;
    }
    if (formed) {
        // The mix then reads the combine's residual output, which the graph does not list among its inputs: it must
        // stay allocated until the mix has run - read again later (the next combine) or kept as an output. The last
        // layer's residual has neither, and the head's own down projection could be allocated over it: issue #7,
        // llama-perplexity at -ub 512 / 2048 read the first 64 rows of it overwritten by that GEMM's output
        const ggml_tensor * r = args.out_res->view_src ? args.out_res->view_src : args.out_res;
        bool alive = (r->flags & GGML_TENSOR_FLAG_OUTPUT) != 0;
        for (int n = last_read + 1; n < graph->n_nodes && !alive; ++n) {
            const ggml_tensor * t = graph->nodes[n];
            for (int s = 0; s < GGML_MAX_SRC && t->src[s]; ++s) {
                if (t->src[s] == r || t->src[s]->view_src == r) { alive = true; break; }
            }
        }
        if (!alive) {
            return;                         // the F32 streams stay
        }
    }
    args.store_xn_f32 = false;
    if (formed) {
        args.row_scale = ggml_cuda_hc_row_scale_buf(ctx);
        ggml_cuda_hc_xres_set(xn->data, (const float *) args.out_res->data, args.row_scale, (const float *) args.gamma->data);
    }
    static unsigned hits = 0; if (hits++ < 2) fprintf(stderr, "HC_CN no F32 streams (%s): T=%lld\n", formed ? "the mix forms them" : "no F32 reader", (long long) T);
}

static bool ggml_cuda_can_fuse(const struct ggml_cgraph *                cgraph,
                               int                                       node_idx,
                               std::initializer_list<enum ggml_op>       ops,
                               std::initializer_list<enum ggml_unary_op> unary_ops) {
#ifndef NDEBUG
    const size_t num_unary = std::count(ops.begin(), ops.end(), GGML_OP_UNARY);
    GGML_ASSERT(unary_ops.size() == num_unary);
#endif

    const auto is_equal = [](const std::initializer_list<enum ggml_op> & list1,
                             const std::initializer_list<enum ggml_op> & list2) {
        return std::equal(list1.begin(), list1.end(), list2.begin(), list2.end());
    };

    std::initializer_list<enum ggml_op> mul_mat_bias_glu_ops    = { GGML_OP_MUL_MAT,    GGML_OP_ADD,    GGML_OP_MUL_MAT,    GGML_OP_ADD,    GGML_OP_GLU };
    std::initializer_list<enum ggml_op> mul_mat_id_bias_glu_ops = { GGML_OP_MUL_MAT_ID, GGML_OP_ADD_ID, GGML_OP_MUL_MAT_ID, GGML_OP_ADD_ID, GGML_OP_GLU };

    std::initializer_list<enum ggml_op> mul_mat_id_glu_ops = { GGML_OP_MUL_MAT_ID, GGML_OP_MUL_MAT_ID, GGML_OP_GLU };
    std::initializer_list<enum ggml_op> mul_mat_glu_ops    = { GGML_OP_MUL_MAT,    GGML_OP_MUL_MAT,    GGML_OP_GLU };

    if ((is_equal(mul_mat_bias_glu_ops, ops) || is_equal(mul_mat_id_bias_glu_ops, ops)) &&
        ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 4 })) {
        const ggml_tensor * ffn_gate      = cgraph->nodes[node_idx];
        const ggml_tensor * ffn_gate_bias = cgraph->nodes[node_idx + 1];
        const ggml_tensor * ffn_up        = cgraph->nodes[node_idx + 2];
        const ggml_tensor * ffn_up_bias   = cgraph->nodes[node_idx + 3];
        const ggml_tensor * glu           = cgraph->nodes[node_idx + 4];

        if (ggml_cuda_should_fuse_mul_mat(ffn_up, ffn_gate, glu, ffn_up_bias, ffn_gate_bias)) {
            int out_nodes[] = { node_idx + 4 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    if ((is_equal(mul_mat_id_glu_ops, ops) || is_equal(mul_mat_glu_ops, ops)) &&
        ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {
        const ggml_tensor * ffn_gate = cgraph->nodes[node_idx];
        const ggml_tensor * ffn_up   = cgraph->nodes[node_idx + 1];
        const ggml_tensor * glu      = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_mul_mat(ffn_up, ffn_gate, glu)) {
            int out_nodes[] = { node_idx + 2 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    std::initializer_list<enum ggml_op> rms_norm_mul_rope_ops          = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE };
    std::initializer_list<enum ggml_op> rms_norm_mul_rope_set_rows_ops = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS };

    if (is_equal(rms_norm_mul_rope_set_rows_ops, ops) && ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 4 })) {
        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * mul      = cgraph->nodes[node_idx + 1];
        const ggml_tensor * rope     = cgraph->nodes[node_idx + 2];
        const ggml_tensor * view     = cgraph->nodes[node_idx + 3];
        const ggml_tensor * set_rows = cgraph->nodes[node_idx + 4];

        if (ggml_check_edges(cgraph, node_idx, {{1, 0, 0}, {2, 0, 1}, {3, 0, 2}, {4, 0, 3}}) &&
            ggml_cuda_should_fuse_rms_norm_mul_rope(rms_norm, mul, rope) &&
            ggml_cuda_should_fuse_rope_set_rows(rope, view, set_rows)) {
            int out_nodes[] = { node_idx + 4 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    if (is_equal(rms_norm_mul_rope_ops, ops) && ggml_can_fuse(cgraph, node_idx, ops)) {
        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * mul      = cgraph->nodes[node_idx + 1];
        const ggml_tensor * rope     = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_rms_norm_mul_rope(rms_norm, mul, rope)) {
            int out_nodes[] = { node_idx + 2 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
        return false;
    }

    std::initializer_list<enum ggml_op> rope_set_rows_ops = { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS };

    if (is_equal(rope_set_rows_ops, ops) && ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {
        const ggml_tensor * rope     = cgraph->nodes[node_idx];
        const ggml_tensor * view     = cgraph->nodes[node_idx + 1];
        const ggml_tensor * set_rows = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_rope_set_rows(rope, view, set_rows)) {
            int out_nodes[] = { node_idx + 2 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    // scale_out*sigmoid(scale_in*post) as the hc_post weights (qwen4exp hc_combine)
    std::initializer_list<enum ggml_op> hc_post_gate_ops = { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_DSV4_HC_POST };

    if (is_equal(hc_post_gate_ops, ops) && ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 3 })) {
        const ggml_tensor * scale_in  = cgraph->nodes[node_idx];
        const ggml_tensor * sigmoid   = cgraph->nodes[node_idx + 1];
        const ggml_tensor * scale_out = cgraph->nodes[node_idx + 2];

        // the chain must feed hc_post->src[2]; a bias on either scale is not handled
        if (ggml_check_edges(cgraph, node_idx, {{1, 0, 0}, {2, 0, 1}, {3, 2, 2}}) &&
            ggml_get_unary_op(sigmoid) == GGML_UNARY_OP_SIGMOID &&
            ggml_get_op_params_f32(scale_in, 1) == 0.0f && ggml_get_op_params_f32(scale_out, 1) == 0.0f &&
            scale_in->src[0]->type == GGML_TYPE_F32 && scale_out->type == GGML_TYPE_F32 &&
            ggml_are_same_shape(scale_in->src[0], scale_out)) {
            int out_nodes[] = { node_idx + 3 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
        return false;
    }

    if (!ggml_can_fuse(cgraph, node_idx, ops)) {
        return false;
    }

    if ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        const ggml_tensor *rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor *mul      = cgraph->nodes[node_idx+1];
        const ggml_tensor *add      = nullptr;

        if (ops.size() == 3 && ops.begin()[2] == GGML_OP_ADD) {
            add = cgraph->nodes[node_idx+2];
        }

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);

        //rms norm only supports F32
        if (mul->src[0]->type != GGML_TYPE_F32 ||
            mul->src[1]->type != GGML_TYPE_F32 ||
            mul->type != GGML_TYPE_F32) {
            return false;
        }

        if (add && (add->src[0]->type != GGML_TYPE_F32 ||
            add->src[1]->type != GGML_TYPE_F32 ||
            add->type != GGML_TYPE_F32) ) {
            return false;
        }

        //if rms norm is the B operand, then we don't handle broadcast
        if (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
            return false;
        }

        //rms_norm kernel assumes contiguous rows
        if (!ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
            return false;
        }

        if (add && (!ggml_is_contiguous(add->src[0]) || !ggml_is_contiguous_rows(add->src[1]))) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_SCALE) {
        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * scale    = cgraph->nodes[node_idx+1];

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);

        float bias;
        memcpy(&bias, (const float *) scale->op_params + 1, sizeof(float));

        return bias == 0.0f && scale->type == GGML_TYPE_F32;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_UNARY
     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        const ggml_tensor * ssm_conv = cgraph->nodes[node_idx];
        const ggml_tensor * silu     = cgraph->nodes[node_idx+1];
        if (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            return false;
        }

        if (ssm_conv->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            return false;
        }

        return true;
    }

    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_ADD
     && ops.begin()[2] == GGML_OP_UNARY && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        const ggml_tensor * ssm_conv = cgraph->nodes[node_idx];
        const ggml_tensor * add      = cgraph->nodes[node_idx+1];
        const ggml_tensor * silu     = cgraph->nodes[node_idx+2];
        if (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            return false;
        }

        if (ssm_conv->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            return false;
        }

        // ADD must consume ssm_conv's output and broadcast a 1-D channel-wise bias.
        const ggml_tensor * bias = (add->src[0] == ssm_conv) ? add->src[1] : add->src[0];
        if (bias->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias)) {
            return false;
        }
        if (ggml_nelements(bias) != ssm_conv->ne[0] || bias->ne[0] != ssm_conv->ne[0]) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_MUL
     && unary_ops.size() == 1 && (unary_ops.begin()[0] == GGML_UNARY_OP_SILU || unary_ops.begin()[0] == GGML_UNARY_OP_SIGMOID || unary_ops.begin()[0] == GGML_UNARY_OP_SOFTPLUS)) {
        const ggml_tensor * unary = cgraph->nodes[node_idx];
        const ggml_tensor * mul   = cgraph->nodes[node_idx+1];

        if (ggml_get_unary_op(unary) != unary_ops.begin()[0]) {
            return false;
        }

        if (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16 && unary->type != GGML_TYPE_BF16) {
            return false;
        }

        if (unary->type != mul->type) {
            return false;
        }

        const ggml_tensor * other = (mul->src[0] == unary) ? mul->src[1] : mul->src[0];
        if (other->type != unary->type) {
            return false;
        }
        if (!ggml_is_contiguous_1(other) || !ggml_is_contiguous_1(unary->src[0]) || !ggml_are_same_shape(other, unary)) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_SQR
     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_RELU) {
        const ggml_tensor * unary = cgraph->nodes[node_idx];
        const ggml_tensor * sqr   = cgraph->nodes[node_idx+1];

        if (ggml_get_unary_op(unary) != GGML_UNARY_OP_RELU) {
            return false;
        }

        if (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16) {
            return false;
        }

        if (unary->type != sqr->type) {
            return false;
        }

        if (!ggml_is_contiguous(unary->src[0])) {
            return false;
        }

        return true;
    }

    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_SCALE && ops.begin()[1] == GGML_OP_UNARY && ops.begin()[2] == GGML_OP_SCALE
     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_TANH) {
        const ggml_tensor *scale  = cgraph->nodes[node_idx];
        const ggml_tensor *tanh   = cgraph->nodes[node_idx+1];
        const ggml_tensor *scale2 = cgraph->nodes[node_idx+2];

        // the fused softcap kernel is F32 only
        if (scale->src[0]->type != GGML_TYPE_F32 || scale->type != GGML_TYPE_F32) {
            return false;
        }

        if (ggml_get_unary_op(tanh) != GGML_UNARY_OP_TANH) {
            return false;
        }

        // Check for bias
        if (ggml_get_op_params_f32(scale, 1) != 0.0f || ggml_get_op_params_f32(scale2, 1) != 0.0f) {
            return false;
        }

        return true;
    }

    // strixllama: the same runs but with SIGMOID, which is what this architecture gates with - a
    // decode pass dispatches 195 SIGMOID and zero TANH, so the softcap rule above never fires.
    // The fused kernel folds in both biases, so unlike softcap there is nothing to reject here.
    if (unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SIGMOID &&
            ops.size() >= 2 && ops.begin()[0] == GGML_OP_SCALE && ops.begin()[1] == GGML_OP_UNARY &&
            (ops.size() == 2 || (ops.size() == 3 && ops.begin()[2] == GGML_OP_SCALE))) {
        const ggml_tensor * scale = cgraph->nodes[node_idx];
        const ggml_tensor * sig   = cgraph->nodes[node_idx + 1];
        if (ggml_get_unary_op(sig) != GGML_UNARY_OP_SIGMOID) {
            return false;
        }
        if (scale->src[0]->type != GGML_TYPE_F32 || scale->type != GGML_TYPE_F32) {
            return false;
        }
        return true;
    }

    return false;
}

// try and fuse nodes and return the number of nodes to skip
// indexer head reduction: RELU, VIEW(h=0), CONT, then (VIEW(h), ADD) for the remaining heads
static int ggml_cuda_match_idx_relu_sum(const ggml_cgraph * g, int i, ggml_cuda_idx_relu_sum_args & a) {
    if (!ggml_cuda_idx_relu_sum_enabled() || i + 4 >= g->n_nodes) return 0;
    const ggml_tensor * relu = g->nodes[i];
    if (relu->op != GGML_OP_UNARY || ggml_get_unary_op(relu) != GGML_UNARY_OP_RELU) return 0;
    if (relu->type != GGML_TYPE_F32 || !ggml_is_contiguous(relu)) return 0;
    const ggml_tensor * src = relu->src[0];
    if (!src || src->type != GGML_TYPE_F32 || !ggml_is_contiguous(src) || !ggml_are_same_shape(src, relu)) return 0;
    const int64_t nb = relu->ne[0], H = relu->ne[1], nt = relu->ne[2], ns = relu->ne[3];
    if (H < 2 || H > 32 || nb < 64 || nt * ns < 64) return 0;
    auto is_slice = [&](const ggml_tensor * v, int64_t h) {
        return v->op == GGML_OP_VIEW && v->view_src == relu && v->type == GGML_TYPE_F32 &&
               v->ne[0] == nb && v->ne[1] == nt && v->ne[2] == ns && v->ne[3] == 1 &&
               v->nb[0] == sizeof(float) && v->nb[1] == relu->nb[2] && v->nb[2] == relu->nb[3] &&
               v->view_offs == (size_t) h * relu->nb[1];
    };
    const ggml_tensor * cont = g->nodes[i + 2];
    if (!is_slice(g->nodes[i + 1], 0) || cont->op != GGML_OP_CONT || cont->src[0] != g->nodes[i + 1]) return 0;
    const ggml_tensor * prev = cont;
    int k = i + 3;
    for (int64_t h = 1; h < H; ++h) {
        if (k + 1 >= g->n_nodes) return 0;
        const ggml_tensor * v = g->nodes[k]; const ggml_tensor * add = g->nodes[k + 1];
        if (!is_slice(v, h) || add->op != GGML_OP_ADD || add->type != GGML_TYPE_F32) return 0;
        if (add->src[0] != prev || add->src[1] != v || !ggml_are_same_shape(add, cont) || !ggml_is_contiguous(add)) return 0;
        prev = add; k += 2;
    }
    const int count = k - i;
    if (count > 32) return 0;
    int indices[32]; ggml_op ops[32];
    for (int j = 0; j < count; ++j) { indices[j] = i + j; ops[j] = g->nodes[i + j]->op; }
    const int output = i + count - 1;
    if (!ggml_can_fuse_subgraph_ext(g, indices, count, ops, &output, 1)) return 0;
    if (src->data && prev->data) {
        const uintptr_t av = (uintptr_t) src->data, bv = (uintptr_t) prev->data;
        const bool overlap = av <= bv ? bv - av < ggml_nbytes(src) : av - bv < ggml_nbytes(prev);
        if (overlap) { static unsigned w = 0; if (w++ < 2) fprintf(stderr, "IDX_RELU_SUM: skipped, score and output overlap\n"); return 0; }
    }
    a.score = src; a.dst = (ggml_tensor *) prev; a.heads = (int) H;
    return count;
}

// strixllama: the lightning indexer's score for one query strip (qwen4exp build_qsa_top_k, compact path, one
// sequence): MUL_MAT (block keys x the strip's queries, four heads) -> RESHAPE -> RELU -> the head sum (the relu-sum
// chain above) -> + log(step(tail - start)), the visibility of each block to each query. The chain writes a
// [blocks x heads x queries] F32 score and passes over it four more times (1.9 + 0.9 + 1.1 ms a strip at 64K tokens);
// k_idx_score computes the [blocks x queries] result in one kernel with the chain's arithmetic. STRIX_IDX_SCORE=0 keeps
// the chain.
static int ggml_cuda_match_idx_score(const ggml_cgraph * g, int i, ggml_cuda_idx_score_args & a) {
    static const bool on = !getenv("STRIX_IDX_SCORE") || atoi(getenv("STRIX_IDX_SCORE")) != 0;
    if (!on || i + 20 >= g->n_nodes) return 0;
    const ggml_tensor * mm = g->nodes[i];
    if (mm->op != GGML_OP_MUL_MAT || mm->type != GGML_TYPE_F32 || !ggml_node_has_n_uses(g, i, 1)) return 0;
    const ggml_tensor * k = mm->src[0], * q = mm->src[1];
    if (k->type != GGML_TYPE_F32 || q->type != GGML_TYPE_F32 || k->ne[0] != 128 || q->ne[0] != 128 ||
        k->nb[0] != sizeof(float) || q->nb[0] != sizeof(float) || k->nb[1] % 16 != 0 || q->nb[1] % 16 != 0 ||
        k->ne[2] != 1 || k->ne[3] != 1 || q->ne[2] != 1 || q->ne[3] != 1 || q->ne[1] % 4 != 0) return 0;
    const ggml_tensor * rs = g->nodes[i + 1];
    const int64_t nb = mm->ne[0], nq = q->ne[1] / 4;
    if (rs->op != GGML_OP_RESHAPE || rs->src[0] != mm || rs->ne[0] != nb || rs->ne[1] != 4 || rs->ne[2] != nq || rs->ne[3] != 1 ||
        ggml_node_get_use_count(g, i + 1) != 1) return 0;
    ggml_cuda_idx_relu_sum_args ra;
    const int c = ggml_cuda_match_idx_relu_sum(g, i + 2, ra);
    if (c <= 0 || ra.score != rs || ra.heads != 4) return 0;
    const ggml_tensor * summed = ra.dst;
    // the visibility, in the order build_forward_expand leaves it: VIEW(limits) CPY RESHAPE REPEAT VIEW(limits) CPY SUB
    // STEP LOG ADD
    const int j = i + 2 + c;
    if (j + 9 >= g->n_nodes) return 0;
    const ggml_tensor * const * n = g->nodes + j;
    const ggml_tensor * tv = n[0], * tc = n[1], * tr = n[2], * rp = n[3], * sv = n[4], * sc = n[5], * sub = n[6], * st = n[7], * lg = n[8], * add = n[9];
    auto limits_view = [](const ggml_tensor * v, int64_t len) {
        return v->op == GGML_OP_VIEW && v->type == GGML_TYPE_I32 && v->view_src && v->view_src->type == GGML_TYPE_I32 &&
               v->ne[0] == len && v->ne[1] == 1 && v->ne[2] == 1 && v->ne[3] == 1 && v->nb[0] == sizeof(int32_t);
    };
    auto cast_of = [](const ggml_tensor * c, const ggml_tensor * v) {
        return c->op == GGML_OP_CPY && c->src[0] == v && c->type == GGML_TYPE_F32 && ggml_nelements(c) == ggml_nelements(v) && ggml_is_contiguous(c);
    };
    if (!limits_view(tv, nq) || !cast_of(tc, tv) || tr->op != GGML_OP_RESHAPE || tr->src[0] != tc || tr->ne[0] != 1 || tr->ne[1] != nq ||
        rp->op != GGML_OP_REPEAT || rp->src[0] != tr || !ggml_are_same_shape(rp, summed) ||
        !limits_view(sv, nb) || sv->view_src != tv->view_src || !cast_of(sc, sv) ||
        sub->op != GGML_OP_SUB || sub->src[0] != rp || sub->src[1] != sc ||
        st->op != GGML_OP_UNARY || ggml_get_unary_op(st) != GGML_UNARY_OP_STEP || st->src[0] != sub ||
        lg->op != GGML_OP_LOG || lg->src[0] != st ||
        add->op != GGML_OP_ADD || add->src[0] != summed || add->src[1] != lg || add->type != GGML_TYPE_F32 ||
        !ggml_are_same_shape(add, summed) || !ggml_is_contiguous(add)) return 0;
    // nothing between the MUL_MAT and the ADD is read outside the chain (a cast is its own src[1])
    const int uses[10] = {1, 2, 1, 1, 1, 2, 1, 1, 1, -1};
    for (int u = 0; u < 9; ++u) {
        if (ggml_node_get_use_count(g, j + u) != uses[u]) return 0;
    }
    if (!ggml_node_has_n_uses(g, i + 2 + c - 1, 1)) return 0;   // the head sum feeds only the ADD
    a.keys = k; a.q = q; a.out = (ggml_tensor *) add;
    a.starts = (const int32_t *) sv->data; a.tails = (const int32_t *) tv->data;
    return j + 10 - i;
}

// strixllama: the lightning indexer's score for a decode-sized strip (at most 8 queries, ggml_cuda_match_idx_score's gates
// are for prefill strips): GET_ROWS (the block keys out of the block-key cache, F16 -> F32) -> RESHAPE, the strip's query
// nodes, then MUL_MAT (keys x queries, four heads) -> RESHAPE -> RELU -> the head sum (CONT of head 0, then VIEW + ADD a head)
// -> + log(step(tail - start)). k_idx_score_dec computes it from the cache in one kernel at the MUL_MAT; the GET_ROWS is
// skipped, since the MUL_MAT was its only reader. At 110K tokens the chain took ~0.5 ms a layer, 12 layers a token.
// STRIX_IDX_SCORE_DEC=0 keeps the nodes.
struct ggml_cuda_idx_dec_match {
    int gr  = -1;    // the GET_ROWS
    int mm  = -1;    // the MUL_MAT
    int end = -1;    // the visibility ADD, the chain's output
    ggml_cuda_idx_score_dec_args a;
    const ggml_tensor * sbv = nullptr;   // several sequences: the membership inputs' views
    const ggml_tensor * stv = nullptr;
    bool check = false;                  // STRIX_IDX_SCORE_DEC=2: the chain runs, the kernel is compared with it at `end`
};

static bool ggml_cuda_match_idx_score_dec(const ggml_cgraph * g, int i, ggml_cuda_idx_dec_match & m) {
    static const bool on = !getenv("STRIX_IDX_SCORE_DEC") || atoi(getenv("STRIX_IDX_SCORE_DEC")) != 0;
    if (!on || i + 24 >= g->n_nodes) return false;
    const ggml_tensor * gr = g->nodes[i];
    if (gr->op != GGML_OP_GET_ROWS || gr->type != GGML_TYPE_F32) return false;
    const ggml_tensor * kb = gr->src[0], * rows = gr->src[1];
    if (!kb || !rows || kb->type != GGML_TYPE_F16 || kb->ne[0] != 128 || kb->ne[2] != 1 || kb->ne[3] != 1 ||
        kb->nb[0] != sizeof(half) || kb->nb[1] % 16 != 0 || ((uintptr_t) kb->data) % 16 != 0 ||
        rows->type != GGML_TYPE_I32 || rows->ne[1] != 1 || rows->ne[2] != 1 || rows->ne[3] != 1 || !ggml_is_contiguous(rows) ||
        gr->ne[0] != 128 || gr->ne[1] != rows->ne[0] || gr->ne[2] != 1 || gr->ne[3] != 1 || !ggml_is_contiguous(gr) ||
        rows->ne[0] < 1 || rows->ne[0] > INT_MAX || !ggml_node_has_n_uses(g, i, 1)) return false;
    const ggml_tensor * rs = g->nodes[i + 1];
    if (rs->op != GGML_OP_RESHAPE || rs->src[0] != gr || ggml_node_get_use_count(g, i + 1) != 1) return false;
    // the MUL_MAT that reads the keys, past the strip's query nodes, none of which reads them
    int j = i + 2;
    for (; j < i + 32 && j < g->n_nodes; ++j) {
        const ggml_tensor * n = g->nodes[j];
        if (n->op == GGML_OP_MUL_MAT && n->src[0] == rs) break;
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (n->src[s] == gr || n->src[s] == rs) return false;
        }
    }
    if (j >= i + 32 || j + 20 >= g->n_nodes) return false;
    const ggml_tensor * mm = g->nodes[j];
    const ggml_tensor * q = mm->src[1];
    const int64_t nb = gr->ne[1];
    if (mm->type != GGML_TYPE_F32 || !ggml_node_has_n_uses(g, j, 1) || !ggml_is_contiguous(mm) ||
        q->type != GGML_TYPE_F32 || q->ne[0] != 128 || q->ne[1] % 4 != 0 || q->ne[1] < 4 || q->ne[1] > 32 ||
        q->ne[2] != 1 || q->ne[3] != 1 || q->nb[0] != sizeof(float) || q->nb[1] % 8 != 0 || ((uintptr_t) q->data) % 8 != 0 ||
        mm->ne[0] != nb || mm->ne[1] != q->ne[1]) return false;
    const int64_t nq = q->ne[1] / 4;
    // RESHAPE [nb, 4, nq] -> RELU -> CONT(head 0) and VIEW + ADD for heads 1..3, as build_qsa_top_k leaves them
    const ggml_tensor * rs2 = g->nodes[j + 1], * relu = g->nodes[j + 2];
    if (rs2->op != GGML_OP_RESHAPE || rs2->src[0] != mm || rs2->ne[0] != nb || rs2->ne[1] != 4 || rs2->ne[2] != nq || rs2->ne[3] != 1 ||
        ggml_node_get_use_count(g, j + 1) != 1) return false;
    if (relu->op != GGML_OP_UNARY || ggml_get_unary_op(relu) != GGML_UNARY_OP_RELU || relu->src[0] != rs2 ||
        relu->type != GGML_TYPE_F32 || !ggml_is_contiguous(relu)) return false;
    auto is_slice = [&](const ggml_tensor * v, int64_t h) {
        return v->op == GGML_OP_VIEW && v->view_src == relu && v->type == GGML_TYPE_F32 &&
               v->ne[0] == nb && v->ne[1] == nq && v->ne[2] == 1 && v->ne[3] == 1 &&
               v->nb[0] == sizeof(float) && v->nb[1] == relu->nb[2] && v->view_offs == (size_t) h * relu->nb[1];
    };
    const ggml_tensor * cont = g->nodes[j + 4];
    if (!is_slice(g->nodes[j + 3], 0) || cont->op != GGML_OP_CONT || cont->src[0] != g->nodes[j + 3] ||
        ggml_node_get_use_count(g, j + 4) != 1) return false;
    const ggml_tensor * summed = cont;
    int k = j + 5;
    for (int64_t h = 1; h < 4; ++h) {
        const ggml_tensor * v = g->nodes[k], * add = g->nodes[k + 1];
        if (!is_slice(v, h) || add->op != GGML_OP_ADD || add->type != GGML_TYPE_F32 || add->src[0] != summed || add->src[1] != v ||
            !ggml_are_same_shape(add, cont) || !ggml_is_contiguous(add) || ggml_node_get_use_count(g, k + 1) != 1) return false;
        summed = add;
        k += 2;
    }
    if (ggml_node_get_use_count(g, j + 2) != 4) return false;   // the relu feeds its four head slices only
    // the visibility: VIEW(limits) CPY RESHAPE REPEAT VIEW(limits) CPY SUB STEP LOG ADD, as in ggml_cuda_match_idx_score; with
    // several sequences in the strip VIEW(seq_blk) CPY(F16) VIEW(seq_tok) MUL_MAT SCALE STEP MUL between the STEP and the LOG
    if (k + 9 >= g->n_nodes) return false;
    const ggml_tensor * const * n = g->nodes + k;
    const ggml_tensor * tv = n[0], * tc = n[1], * tr = n[2], * rp = n[3], * sv = n[4], * sc = n[5], * sub = n[6], * st = n[7], * lg = n[8], * add = n[9];
    const ggml_tensor * mbv = nullptr, * mtv = nullptr;
    if (lg->op == GGML_OP_VIEW) {
        if (k + 16 >= g->n_nodes) return false;
        const ggml_tensor * bv = n[8], * bc = n[9], * tvw = n[10], * mm2 = n[11], * sb = n[12], * st2 = n[13], * mul = n[14];
        lg = n[15];
        add = n[16];
        const int64_t ns = bv->ne[0];
        float sb_scale = 0.0f, sb_bias = 0.0f;
        if (sb->op == GGML_OP_SCALE) {
            memcpy(&sb_scale, (const float *) sb->op_params + 0, sizeof(float));
            memcpy(&sb_bias,  (const float *) sb->op_params + 1, sizeof(float));
        }
        if (ns < 1 || ns > 64 ||
            bv->type != GGML_TYPE_F32 || bv->ne[1] < nb || bv->ne[2] != 1 || bv->ne[3] != 1 || bv->nb[0] != sizeof(float) ||
            bv->nb[1] != ns * sizeof(float) || !bv->view_src ||
            bc->op != GGML_OP_CPY || bc->src[0] != bv || bc->type != GGML_TYPE_F16 || bc->ne[0] != ns || bc->ne[1] != nb ||
            tvw->op != GGML_OP_VIEW || tvw->type != GGML_TYPE_F32 || tvw->ne[0] != ns || tvw->ne[1] != nq || tvw->nb[0] != sizeof(float) ||
            tvw->nb[1] != ns * sizeof(float) || !tvw->view_src ||
            mm2->op != GGML_OP_MUL_MAT || mm2->src[0] != bc || mm2->src[1] != tvw || mm2->ne[0] != nb || mm2->ne[1] != nq ||
            sb->op != GGML_OP_SCALE || sb->src[0] != mm2 || sb_scale != 1.0f || sb_bias != -0.5f ||
            st2->op != GGML_OP_UNARY || ggml_get_unary_op(st2) != GGML_UNARY_OP_STEP || st2->src[0] != sb ||
            mul->op != GGML_OP_MUL || mul->src[0] != st || mul->src[1] != st2 || !ggml_are_same_shape(mul, summed) ||
            lg->op != GGML_OP_LOG || lg->src[0] != mul) return false;
        const int uses_m[7] = {1, 2, 1, 1, 1, 1, 1};
        for (int u = 0; u < 7; ++u) {
            if (ggml_node_get_use_count(g, k + 8 + u) != uses_m[u]) return false;
        }
        mbv = bv;
        mtv = tvw;
    }
    auto limits_view = [](const ggml_tensor * v, int64_t len) {
        return v->op == GGML_OP_VIEW && v->type == GGML_TYPE_I32 && v->view_src && v->view_src->type == GGML_TYPE_I32 &&
               v->ne[0] == len && v->ne[1] == 1 && v->ne[2] == 1 && v->ne[3] == 1 && v->nb[0] == sizeof(int32_t);
    };
    auto cast_of = [](const ggml_tensor * c, const ggml_tensor * v) {
        return c->op == GGML_OP_CPY && c->src[0] == v && c->type == GGML_TYPE_F32 && ggml_nelements(c) == ggml_nelements(v) && ggml_is_contiguous(c);
    };
    if (!limits_view(tv, nq) || !cast_of(tc, tv) || tr->op != GGML_OP_RESHAPE || tr->src[0] != tc || tr->ne[0] != 1 || tr->ne[1] != nq ||
        rp->op != GGML_OP_REPEAT || rp->src[0] != tr || !ggml_are_same_shape(rp, summed) ||
        !limits_view(sv, nb) || sv->view_src != tv->view_src || !cast_of(sc, sv) ||
        sub->op != GGML_OP_SUB || sub->src[0] != rp || sub->src[1] != sc ||
        st->op != GGML_OP_UNARY || ggml_get_unary_op(st) != GGML_UNARY_OP_STEP || st->src[0] != sub ||
        (!mbv && (lg->op != GGML_OP_LOG || lg->src[0] != st)) ||
        add->op != GGML_OP_ADD || add->src[0] != summed || add->src[1] != lg || add->type != GGML_TYPE_F32 ||
        !ggml_are_same_shape(add, summed) || !ggml_is_contiguous(add)) return false;
    const int uses[9] = {1, 2, 1, 1, 1, 2, 1, 1, 1};
    for (int u = 0; u < 8; ++u) {
        if (ggml_node_get_use_count(g, k + u) != uses[u]) return false;
    }
    if (!mbv && ggml_node_get_use_count(g, k + 8) != 1) return false;
    if (mbv && ggml_node_get_use_count(g, k + 15) != 1) return false;
    if (!ggml_node_has_n_uses(g, k - 1, 1)) return false;   // the head sum feeds only the ADD
    m.gr = i; m.mm = j; m.end = mbv ? k + 16 : k + 9;
    m.a.kb = kb; m.a.rows = rows; m.a.q = q; m.a.out = (ggml_tensor *) add;
    m.a.starts = (const int32_t *) sv->data; m.a.tails = (const int32_t *) tv->data;
    if (mbv) {
        m.a.seq_blk = (const float *) mbv->data;
        m.a.seq_tok = (const float *) mtv->data;
        m.a.n_slots = (int) mbv->ne[0];
        m.sbv = mbv;
        m.stv = mtv;
    }
    return true;
}

// a matched decode strip: the GET_ROWS skipped, the kernel run at the MUL_MAT (reset at every graph evaluation)
static ggml_cuda_idx_dec_match g_idx_dec_pending;

static int ggml_cuda_idx_dec_mode() {
    static const int mode = getenv("STRIX_IDX_SCORE_DEC") ? atoi(getenv("STRIX_IDX_SCORE_DEC")) : 1;
    return mode;
}

// STRIX_IDX_SCORE_DEC=2, after the chain's last node ran: the kernel into scratch, both copied back and compared
static void ggml_cuda_idx_dec_check(ggml_backend_cuda_context & ctx, const ggml_cuda_idx_dec_match & m) {
    const size_t n = ggml_nelements(m.a.out);
    ggml_cuda_pool_alloc<float> scratch(ctx.pool(), n);
    ggml_tensor out = *m.a.out;
    out.data = scratch.get();
    ggml_cuda_idx_score_dec_args a = m.a;
    a.out = &out;
    ggml_cuda_op_idx_score_dec(ctx, a);
    std::vector<float> want(n), got(n);
    CUDA_CHECK(cudaMemcpyAsync(want.data(), m.a.out->data, n*sizeof(float), cudaMemcpyDeviceToHost, ctx.stream()));
    CUDA_CHECK(cudaMemcpyAsync(got.data(), scratch.get(), n*sizeof(float), cudaMemcpyDeviceToHost, ctx.stream()));
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
    static int64_t n_checked = 0;
    if (memcmp(want.data(), got.data(), n*sizeof(float)) != 0) {
        size_t k = 0;
        while (k < n && memcmp(&want[k], &got[k], sizeof(float)) == 0) { ++k; }
        fprintf(stderr, "IDX_SCORE_DEC check: element %zu of %zu differs: chain %.9g, kernel %.9g (%lld queries, %d slots)\n",
                k, n, want[k], got[k], (long long) (m.a.q->ne[1]/4), m.a.n_slots);
        GGML_ABORT("IDX_SCORE_DEC check failed");
    }
    if (++n_checked % 64 == 0) {
        fprintf(stderr, "IDX_SCORE_DEC check: %lld strips, the kernel's scores equal the chain's\n", (long long) n_checked);
    }
}

// strixllama: SSM_SCAN and the CPY of its new states into the recurrent cache (Mamba-2, one snapshot slot): the scan
// writes the cache rows itself, several sequences at once (ggml_cuda_ssm_scan_cache), and the CPY dispatches nothing.
// At eight conversations the copy was 310 us of each of Nemotron 3.5 Super's 40 Mamba layers (4 MiB of state a sequence).
static int ggml_cuda_ssm_scan_cache_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
#ifdef GGML_USE_HIP
    static const bool on = !getenv("STRIX_SSM_CACHE") || atoi(getenv("STRIX_SSM_CACHE")) != 0;
    ggml_tensor * ssm = cgraph->nodes[i];
    if (!on || ssm->op != GGML_OP_SSM_SCAN || ssm->type != GGML_TYPE_F32 || (ssm->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return 0;
    }
    const ggml_tensor * s = ssm->src[0];
    const ggml_tensor * x = ssm->src[1];
    if (ggml_get_op_params_i32(ssm, 0) > 1 || ssm->src[3]->nb[1] != sizeof(float) || s->type != GGML_TYPE_F32) {
        return 0;
    }
    // the CPY is the first node after the scan that dispatches anything (the extra states' copy is empty)
    const ggml_tensor * cpy = nullptr;
    int skip = 0;
    for (int j = i + 1; j < cgraph->n_nodes && cpy == nullptr; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(n)) {
            continue;
        }
        if (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return 0;
        }
        cpy  = n;
        skip = j - i;
    }
    if (cpy == nullptr) {
        return 0;
    }
    const int64_t n_seqs = x->ne[3];
    const int64_t D      = s->ne[0] * s->ne[1] * s->ne[2];
    const ggml_tensor * tail = cpy->src[0];
    const ggml_tensor * rows = cpy->src[1];
    if (tail->op != GGML_OP_VIEW || tail->view_src != ssm || tail->view_offs != (size_t) ggml_nelements(x) * sizeof(float) || !ggml_is_contiguous(tail) ||
            tail->ne[0] != D || ggml_nelements(tail) != D * n_seqs) {
        return 0;
    }
    // the rows are rows of the states the scan reads, which its ids count from s's first
    const ggml_tensor * root = s->view_src ? s->view_src : s;
    if (rows->op != GGML_OP_VIEW || rows->type != GGML_TYPE_F32 || rows->data == nullptr || rows->view_src != root ||
            s->data != root->data || rows->ne[0] != D || rows->ne[1] != n_seqs || ggml_nelements(rows) != D * n_seqs ||
            rows->nb[0] != sizeof(float) || rows->nb[1] != s->nb[3] || s->nb[3] != (size_t) D * sizeof(float)) {
        return 0;
    }
    const ptrdiff_t off = (const char *) rows->data - (const char *) s->data;
    if (off < 0 || off % (ptrdiff_t) s->nb[3] != 0) {
        return 0;
    }
    ggml_cuda_ssm_scan_cache c;
    c.rows = (char *) rows->data;
    c.row0 = getenv("STRIX_SSM_CACHE_FIXUP_ALL") ? -1 : off / (ptrdiff_t) s->nb[3];   // DEV: every sequence through the fixup
    ggml_cuda_op_ssm_scan(ctx, ssm, &c);
    return skip;
#else
    GGML_UNUSED(ctx); GGML_UNUSED(cgraph); GGML_UNUSED(i);
    return 0;
#endif
}

// strixllama: the routed EXL3 experts of a prompt-sized batch as one pipeline (mmh-moe.cu). Main part at the gate node:
// {MUL_MAT_HAD gate, MUL_MAT_HAD up, GLU swiglu, MUL_MAT_HAD down} - sort by expert, gather, gate / up, SwiGLU, down -
// with the down product's sorted F32 rows left in the down node's own buffer (the gate or up node's buffer carries the
// intermediate rows). Reduce part at the weighted sum, which qwen4exp builds after the shared expert's nodes: the sum
// over the picks and, right after it, the shared expert's gated tail {MUL, ADD}. Both parts are decided at the gate node,
// so a refused block runs as before throughout. The arrays the reduce needs live in a small buffer of this file (grown
// with the batch, never per call). STRIX_MMH_MOE = the smallest batch it takes (default 32 tokens), 0 = off;
// STRIX_MMH_MOE_DEBUG=1 says why a block was refused.
struct ggml_cuda_mmh_moe_pending {
    const ggml_tensor *    down = nullptr;   // the routed down product, whose buffer holds the sorted rows
    int                    device = -1;
    ggml_cuda_mmh_moe_args a;
    // a decode block's shared expert folded into the pipeline (ggml_cuda_mmh_moe_match_shexp): its products' nodes are
    // skipped, the sum's nodes too (the reduce runs at fold_mul, after the gate's nodes, which run without fusions)
    const ggml_tensor *    fold_skip[6] = {};
    const ggml_tensor *    fold_gin = nullptr, * fold_sig = nullptr, * fold_mul = nullptr, * fold_add = nullptr;
    bool                   fold_sum_seen = false;
};
static ggml_cuda_mmh_moe_pending g_mmh_moe;

// strixllama: a node whose product the decode pipeline already made (the folded shared expert's)
static bool ggml_cuda_mmh_moe_fold_skip(const ggml_tensor * node) {
    if (!g_mmh_moe.fold_mul) {
        return false;
    }
    for (const ggml_tensor * t : g_mmh_moe.fold_skip) {
        if (t == node) {
            return true;
        }
    }
    return false;
}

// strixllama: the shared expert after a decode block's weighted sum (graph order of qwen4exp for batches under 512
// tokens): {MUL_MAT_HAD, MUL_MAT_HAD} of the block's input x, GLU swiglu of the two, MUL_MAT_HAD down, the gate {MUL_MAT
// of x, SIGMOID}, then {MUL(down, gate), ADD(sum, that)}; nothing else in between, every product read only by the next
struct ggml_cuda_mmh_moe_fold_match {
    const ggml_tensor * gate = nullptr, * up = nullptr, * glu = nullptr, * down = nullptr, * gin = nullptr, * sig = nullptr;
    ggml_tensor *       mul = nullptr, * add = nullptr;
    int                 bits = 0, I = 0;
};

static const char * ggml_cuda_mmh_moe_match_shexp(const ggml_cgraph * cgraph, const int last, const ggml_tensor * x,
        const ggml_tensor * sum, const int H, const int T, ggml_cuda_mmh_moe_fold_match & f) {
    auto is_x = [&](const ggml_tensor * t) {
        return t && t->data == x->data && t->type == GGML_TYPE_F32 && t->ne[0] == H && t->ne[1] == T && t->ne[2] == 1 &&
            t->ne[3] == 1 && t->nb[0] == sizeof(float) && t->nb[1] == x->nb[2];
    };
    const ggml_tensor * prod[2] = {};
    int n_prod = 0, k_mul = -1, k_prod[2] = { -1, -1 }, k_glu = -1, k_down = -1, k_gin = -1, k_sig = -1;
    for (int k = last + 1; k < std::min(cgraph->n_nodes, last + 24) && k_mul < 0; ++k) {
        ggml_tensor * n = cgraph->nodes[k];
        if (ggml_cuda_is_view_or_noop(n)) {
            const ggml_tensor * v = n->view_src ? n->view_src : n->src[0];
            if (v && (v == prod[0] || v == prod[1] || v == f.glu || v == f.down || v == f.gin || v == f.sig)) {
                return "a view of a shared expert product";
            }
            continue;
        }
        if (n->op == GGML_OP_MUL_MAT_HAD && !n->src[2] && !f.glu && n_prod < 2 && is_x(n->src[1])) {
            k_prod[n_prod] = k;
            prod[n_prod++] = n;
        } else if (n->op == GGML_OP_GLU && n_prod == 2 && !f.glu) {
            f.glu = n;  k_glu = k;
        } else if (n->op == GGML_OP_MUL_MAT_HAD && !n->src[2] && f.glu && !f.down && n->src[1] == f.glu) {
            f.down = n;  k_down = k;
        } else if (n->op == GGML_OP_MUL_MAT && !f.gin && is_x(n->src[1])) {
            f.gin = n;  k_gin = k;
        } else if (n->op == GGML_OP_UNARY && ggml_get_unary_op(n) == GGML_UNARY_OP_SIGMOID && f.gin && !f.sig && n->src[0] == f.gin) {
            f.sig = n;  k_sig = k;
        } else if (n->op == GGML_OP_MUL && f.down && f.sig) {
            k_mul = k;
        } else {
            return "another node between the sum and the shared expert's tail";
        }
    }
    if (k_mul < 0 || k_mul + 1 >= cgraph->n_nodes) {
        return "no shared expert tail";
    }
    const ggml_tensor * glu = f.glu;
    if (ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU || !glu->src[1] || ggml_get_op_params_i32(glu, 1) != 0 ||
            !((glu->src[0] == prod[0] && glu->src[1] == prod[1]) || (glu->src[0] == prod[1] && glu->src[1] == prod[0]))) {
        return "shared GLU";
    }
    f.gate = glu->src[0];
    f.up   = glu->src[1];
    const ggml_tensor * wg = f.gate->src[0], * wu = f.up->src[0], * wd = f.down->src[0];
    auto bits_of = [](const ggml_type t) { return t >= GGML_TYPE_EXL3_M1 && t <= GGML_TYPE_EXL3_M8 ? 1 + (int) (t - GGML_TYPE_EXL3_M1) : 0; };
    f.bits = bits_of(wg->type);
    f.I    = (int) wg->ne[1];
    if (!f.bits || wu->type != wg->type || wd->type != wg->type || wg->ne[0] != H || wu->ne[0] != H || wu->ne[1] != f.I ||
            wd->ne[0] != f.I || wd->ne[1] != H) {
        return "shared weight types or shapes";
    }
    for (const ggml_tensor * w : { wg, wu, wd }) {
        if (w->ne[2] != 1 || w->ne[3] != 1 || w->nb[1] != ggml_row_size(w->type, w->ne[0]) || ((uintptr_t) w->data) % 16 != 0) {
            return "shared weight layout";
        }
    }
    for (const ggml_tensor * n : { f.gate, f.up, f.down }) {
        if (ggml_get_op_params_i32(n, 0) != 128 || n->src[5]) {
            return "shared product kind";
        }
        const ggml_tensor * ri = n->src[3], * ro = n->src[4];
        for (const ggml_tensor * r : { ri, ro }) {
            if (!r || r->type != GGML_TYPE_F16 || !ggml_is_contiguous(r) || r->ne[1] != 1 || ((uintptr_t) r->data) % 8 != 0) {
                return "shared rotations";
            }
        }
        if (ri->ne[0] != n->src[0]->ne[0] || ro->ne[0] != n->src[0]->ne[1]) {
            return "shared rotation lengths";
        }
    }
    f.mul = cgraph->nodes[k_mul];
    f.add = cgraph->nodes[k_mul + 1];
    const ggml_tensor * mul = f.mul;
    const ggml_tensor * add = f.add;
    if (!((mul->src[0] == f.down && mul->src[1] == f.sig) || (mul->src[1] == f.down && mul->src[0] == f.sig)) ||
            add->op != GGML_OP_ADD || !((add->src[0] == sum && add->src[1] == mul) || (add->src[1] == sum && add->src[0] == mul))) {
        return "shared tail";
    }
    if (f.sig->type != GGML_TYPE_F32 || !ggml_is_contiguous(f.sig) || f.sig->ne[0] != 1 || f.sig->ne[1] != T ||
            ggml_nelements(f.sig) != T || mul->type != GGML_TYPE_F32 || !ggml_are_same_shape(mul, f.down) ||
            add->type != GGML_TYPE_F32 || !ggml_is_contiguous(add) || add->ne[0] != H || add->ne[1] != T ||
            ggml_nelements(add) != (int64_t) H * T) {
        return "shared tail shapes";
    }
    // every product read only by the next step (the sum, node `last`, only by the ADD)
    if (cgraph->nodes[last] != sum) {
        return "the sum is not the block's last node";
    }
    for (const int k : { k_prod[0], k_prod[1], k_glu, k_down, k_gin, k_sig, k_mul, last }) {
        if (!ggml_node_has_n_uses(cgraph, k, 1)) {
            return "a shared product has other users";
        }
    }
    return nullptr;
}

static void * ggml_cuda_mmh_moe_persist(const int device, const size_t bytes, cudaStream_t stream) {
    static void * buf[GGML_CUDA_MAX_DEVICES] = {};
    static size_t size[GGML_CUDA_MAX_DEVICES] = {};
    if (bytes > size[device]) {
#if defined(GGML_USE_HIP)
        hipStreamCaptureStatus cap = hipStreamCaptureStatusNone;
        CUDA_CHECK(hipStreamIsCapturing(stream, &cap));
        const bool capturing = cap != hipStreamCaptureStatusNone;
#else
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        CUDA_CHECK(cudaStreamIsCapturing(stream, &cap));
        const bool capturing = cap != cudaStreamCaptureStatusNone;
#endif // defined(GGML_USE_HIP)
        if (capturing) {
            return nullptr;   // no allocation inside a captured graph: the block runs the plain way
        }
        if (buf[device]) {
            CUDA_CHECK(cudaStreamSynchronize(stream));   // the last reduce may still read it
            CUDA_CHECK(cudaFree(buf[device]));
        }
        const size_t n = std::max<size_t>(bytes, 1 << 20) * 5 / 4;
        CUDA_CHECK(cudaMalloc(&buf[device], n));
        size[device] = n;
    }
    return buf[device];
}

// strixllama: the decode pipeline's scratch (ggml_cuda_mmh_moe_run_dec): one buffer a device, allocated once and never
// moved, because decode runs inside captured graphs that keep its address. The first allocation may happen while a
// graph is being captured (relaxed capture mode; no free and no sync is needed for it)
static void * ggml_cuda_mmh_moe_dec_buf(const int device, const size_t bytes) {
    static void * buf[GGML_CUDA_MAX_DEVICES] = {};
    static size_t size[GGML_CUDA_MAX_DEVICES] = {};
    if (!buf[device]) {
        if (cudaMalloc(&buf[device], bytes) != cudaSuccess) {
            (void) cudaGetLastError();
            buf[device] = nullptr;
            return nullptr;
        }
        size[device] = bytes;
    }
    return size[device] >= bytes ? buf[device] : nullptr;
}

static int ggml_cuda_try_mmh_moe(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, const int i) {
#if defined(GGML_USE_HIP)
    static const int moe_min = getenv("STRIX_MMH_MOE") ? atoi(getenv("STRIX_MMH_MOE")) : 32;
    // blocks of STRIX_MMH_MOE_DEC_MIN .. GGML_CUDA_MMH_MOE_DEC_MAX_T tokens (decode, MTP verification) take the decode
    // pipeline; STRIX_MMH_MOE_DEC=0 keeps the GEMVs
    static const bool moe_dec = !getenv("STRIX_MMH_MOE_DEC") || atoi(getenv("STRIX_MMH_MOE_DEC")) != 0;
    static const bool moe_dbg = getenv("STRIX_MMH_MOE_DEBUG") && atoi(getenv("STRIX_MMH_MOE_DEBUG"));
#define MOE_REFUSE(why) do { if (moe_dbg) { fprintf(stderr, "MMH_MOE refused at node %d (%s): %s\n", i, cgraph->nodes[i]->name, why); } return 0; } while (0)
    if (moe_min <= 0 || i + 5 >= cgraph->n_nodes || !GGML_CUDA_CC_IS_RDNA3_5(ggml_cuda_info().devices[ctx.device].cc)) {
        return 0;
    }
    ggml_tensor * n0 = cgraph->nodes[i], * n1 = cgraph->nodes[i + 1], * glu = cgraph->nodes[i + 2], * down = cgraph->nodes[i + 3];
    if (n0->op != GGML_OP_MUL_MAT_HAD || n1->op != GGML_OP_MUL_MAT_HAD || glu->op != GGML_OP_GLU || down->op != GGML_OP_MUL_MAT_HAD ||
            ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU || !glu->src[1] || ggml_get_op_params_i32(glu, 1) != 0) {
        return 0;
    }
    ggml_tensor * gate = glu->src[0], * up = glu->src[1];
    if (!((gate == n0 && up == n1) || (gate == n1 && up == n0))) {
        return 0;
    }
    const ggml_tensor * ids = gate->src[2];
    const ggml_tensor * x   = gate->src[1];
    if (!ids || up->src[2] != ids || down->src[2] != ids || up->src[1] != x || down->src[1] != glu) {
        MOE_REFUSE("not one expert block");
    }
    // the ids are the top-k view of the router's argsort: rows of n_used ids, row stride ids->nb[1]
    if (gate->src[5] || up->src[5] || down->src[5] || ids->type != GGML_TYPE_I32 || ids->nb[0] != sizeof(int32_t) ||
            ids->nb[1] % sizeof(int32_t) != 0) {
        MOE_REFUSE("classes or ids layout");
    }
    const int n_used = (int) ids->ne[0], T = (int) ids->ne[1];
    // STRIX_MMH_MOE_DEC_MIN: the fewest tokens it takes. One token too since the stream ring runs the jobs (+2% decode;
    // the fused GEMV launch had measured 2% slower than the GEMVs there)
    static const int dec_min = getenv("STRIX_MMH_MOE_DEC_MIN") ? atoi(getenv("STRIX_MMH_MOE_DEC_MIN")) : 1;
    const bool dec = moe_dec && T >= dec_min && T <= GGML_CUDA_MMH_MOE_DEC_MAX_T && n_used <= 16;
    if (!dec && T < moe_min) {
        return 0;
    }
    if (g_mmh_moe.down) {
        MOE_REFUSE("an earlier block's reduce is still pending");
    }
    if (x->type != GGML_TYPE_F32 || x->ne[1] != 1 || x->ne[2] != T || x->ne[3] != 1 || x->nb[0] != sizeof(float) ||
            x->nb[2] % 16 != 0 || ((uintptr_t) x->data) % 16 != 0) {
        MOE_REFUSE("input layout");
    }
    const ggml_tensor * wg = gate->src[0], * wu = up->src[0], * wd = down->src[0];
    auto bits_of = [](const ggml_type t) { return t == GGML_TYPE_EXL3_M4 ? 4 : t == GGML_TYPE_EXL3_M3 ? 3 : 0; };
    const int H = (int) wg->ne[0], I = (int) wg->ne[1], E = (int) wg->ne[2];
    const int bits_gu = bits_of(wg->type), bits_d = bits_of(wd->type);
    if (wu->type != wg->type || wu->ne[0] != H || wu->ne[1] != I || wu->ne[2] != E || wd->ne[0] != I || wd->ne[1] != H || wd->ne[2] != E ||
            x->ne[0] != H || !ggml_cuda_mmh_moe_supported(H, I, n_used, bits_gu, bits_d)) {
        MOE_REFUSE("weight types or shapes");
    }
    if (dec && bits_gu != bits_d) {
        MOE_REFUSE("decode: the gate / up and down widths differ");
    }
    for (const ggml_tensor * w : { wg, wu, wd }) {
        if (w->nb[1] != ggml_row_size(w->type, w->ne[0]) || w->nb[2] != w->ne[1] * w->nb[1]) {
            MOE_REFUSE("weight strides");
        }
    }
    // rotations: F16, one row per expert or one row shared by all of them
    const ggml_tensor * rot[6] = { gate->src[3], gate->src[4], up->src[3], up->src[4], down->src[3], down->src[4] };
    const int64_t rot_len[6] = { H, I, H, I, I, H };
    int64_t rs[6];
    for (int r = 0; r < 6; ++r) {
        const ggml_tensor * t = rot[r];
        if (!t || t->type != GGML_TYPE_F16 || !ggml_is_contiguous(t) || t->ne[0] != rot_len[r] || (t->ne[1] != E && t->ne[1] != 1) ||
                ((uintptr_t) t->data) % 8 != 0) {
            MOE_REFUSE("rotations");
        }
        rs[r] = t->ne[1] == 1 ? 0 : t->ne[0];
    }
    // the products feed only the next step
    if (!ggml_node_has_n_uses(cgraph, i, 1) || !ggml_node_has_n_uses(cgraph, i + 1, 1) || !ggml_node_has_n_uses(cgraph, i + 2, 1) ||
            !ggml_node_has_n_uses(cgraph, i + 3, 1)) {
        MOE_REFUSE("a product has other users");
    }
    // the weighted sum: the first node after the block that reads the down product (the nodes in between - the shared
    // expert's - must not read the block's products), and the shared expert's tail right after the sum
    int j = -1;
    for (int k = i + 4; k < std::min(cgraph->n_nodes, i + 64) && j < 0; ++k) {
        const ggml_tensor * nk = cgraph->nodes[k];
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            const ggml_tensor * src = nk->src[s];
            if (src == gate || src == up || src == glu) {
                MOE_REFUSE("a later node reads the gate, up or SwiGLU product");
            }
            if (src == down) {
                j = k;
            }
        }
    }
    ggml_cuda_moe_weighted_reduction_match m;
    if (j < 0 || !ggml_cuda_match_moe_weighted_reduction(cgraph, j, m) || m.experts != down || m.expert_scale != nullptr ||
            m.weights->ne[1] != n_used || m.weights->ne[2] != T || ((uintptr_t) m.weights->data) % 4 != 0) {
        MOE_REFUSE("no weighted sum of the down product");
    }
    int last = j + m.node_count - 1;
    ggml_tensor * out = m.dst;
    const ggml_tensor * sx = nullptr, * sg = nullptr;
    {
        const int im = last + 1, ia = last + 2;
        if (ia < cgraph->n_nodes) {
            const ggml_tensor * mul = cgraph->nodes[im];
            ggml_tensor *       add = cgraph->nodes[ia];
            const ggml_tensor * a0  = mul->op == GGML_OP_MUL ? mul->src[0] : nullptr;
            const ggml_tensor * a1  = mul->op == GGML_OP_MUL ? mul->src[1] : nullptr;
            if (a0 && a1 && add->op == GGML_OP_ADD && ((add->src[0] == m.dst && add->src[1] == mul) || (add->src[1] == m.dst && add->src[0] == mul)) &&
                    mul->type == GGML_TYPE_F32 && add->type == GGML_TYPE_F32 && a0->type == GGML_TYPE_F32 && a1->type == GGML_TYPE_F32 &&
                    ggml_is_contiguous(a0) && ggml_is_contiguous(a1) && ggml_is_contiguous(add) &&
                    ggml_are_same_shape(a0, m.dst) && ggml_are_same_shape(add, m.dst) && ggml_are_same_shape(mul, m.dst) &&
                    a1->ne[0] == 1 && a1->ne[1] == T && a1->ne[2] == 1 && a1->ne[3] == 1 &&
                    ggml_node_has_n_uses(cgraph, last, 1) && ggml_node_has_n_uses(cgraph, im, 1) && !ggml_cuda_mmb_is_bf16_only(add)) {
                sx = a0; sg = a1; out = add; last = ia;
            }
        }
    }
    // strixllama: a decode block's shared expert as one more job of the pipeline (an expert's shape: one I x H gate, up
    // and down). Its products' nodes and the sum are skipped, its gate runs as nodes, and the reduce at the gate's MUL
    // adds its rows times the gate and writes the ADD. STRIX_MMH_MOE_SHEXP=0: its own GEMVs as before
    static const int fold_env = getenv("STRIX_MMH_MOE_SHEXP") ? atoi(getenv("STRIX_MMH_MOE_SHEXP")) : 1;   // DEV -1: the STRIX_AB side
    const bool fold_on = fold_env < 0 ? ggml_cuda_strix_ab() != 0 : fold_env != 0;
    ggml_cuda_mmh_moe_fold_match fold;
    bool folded = false;
    if (dec && fold_on && !sx) {
        const char * why = ggml_cuda_mmh_moe_match_shexp(cgraph, last, x, m.dst, H, T, fold);
        if (!why && !ggml_cuda_mmh_moe_dec_shexp_ok(T, n_used, I, bits_gu, fold.I, fold.bits)) {
            why = "the shared expert's width or rows";
        }
        if (!why && ggml_cuda_mmb_is_bf16_only(fold.add)) {
            why = "the shared tail's output";
        }
        if (why) {
            if (moe_dbg) {
                fprintf(stderr, "MMH_MOE shared expert not folded at node %d: %s\n", i, why);
            }
        } else {
            folded = true;
            out = fold.add;
        }
    }
    if (out->type != GGML_TYPE_F32 || !ggml_is_contiguous(out) || out->ne[0] != H || out->ne[1] != T || ggml_cuda_mmb_is_bf16_only(out)) {
        MOE_REFUSE("output");
    }
    // buffers: the gate (or up) product's carries the intermediate rows during the main part, the down product's the
    // sorted rows until the reduce; neither may overlap what the parts read or write while it is in use
    auto overlaps = [](const void * a, const size_t na, const void * b, const size_t nb) {
        return a && b && (const char *) a < (const char *) b + nb && (const char *) b < (const char *) a + na;
    };
    const size_t bytes_gu = (size_t) T * n_used * I * sizeof(float), bytes_had = (size_t) T * n_used * H * sizeof(float);
    if (ggml_nbytes(down) < bytes_had || (!dec && (ggml_nbytes(gate) < bytes_gu || ggml_nbytes(up) < bytes_gu))) {
        MOE_REFUSE("product buffers too small");
    }
    void * buf_had = down->data;
    const ggml_tensor * keep_had[] = { x, ids, m.weights, out };   // sx and sg are made while the down product is alive
    for (const ggml_tensor * k : keep_had) {
        if (overlaps(buf_had, bytes_had, k->data, ggml_nbytes(k))) {
            MOE_REFUSE("the down product's buffer overlaps a live tensor");
        }
    }
    if (folded) {
        // the shared gate's nodes write after the sum's last read: not into what the reduce still reads
        for (const ggml_tensor * k : { fold.gin, fold.sig }) {
            if (overlaps(buf_had, bytes_had, k->data, ggml_nbytes(k)) || overlaps(m.weights->data, ggml_nbytes(m.weights), k->data, ggml_nbytes(k))) {
                MOE_REFUSE("the shared gate overlaps what the reduce reads");
            }
        }
        if (overlaps(out->data, ggml_nbytes(out), fold.sig->data, ggml_nbytes(fold.sig))) {
            MOE_REFUSE("the output overlaps the shared gate");
        }
    }
    void * buf_gu = nullptr;
    for (ggml_tensor * cand : { gate, up }) {
        if (dec) {
            break;   // the decode pipeline keeps the gate / up products in its own scratch
        }
        bool ok = !overlaps(cand->data, bytes_gu, buf_had, bytes_had);
        for (const ggml_tensor * k : { x, ids }) {
            ok = ok && !overlaps(cand->data, bytes_gu, k->data, ggml_nbytes(k));
        }
        if (ok) {
            buf_gu = cand->data;
            break;
        }
    }
    if (!buf_gu && !dec) {
        MOE_REFUSE("the gate and up products' buffers overlap live tensors");
    }
    void * dec_scratch = nullptr;
    if (dec) {
        dec_scratch = ggml_cuda_mmh_moe_dec_buf(ctx.device, ggml_cuda_mmh_moe_dec_scratch_bytes(E, H, I));
        if (!dec_scratch) {
            MOE_REFUSE("decode scratch");
        }
    }
    const ggml_cuda_mmh_moe_scratch s = ggml_cuda_mmh_moe_scratch_size(T, n_used, E);
    void * persist = dec ? nullptr : ggml_cuda_mmh_moe_persist(ctx.device, s.persist_bytes, ctx.stream());
    if (!persist && !dec) {
        MOE_REFUSE("graph capture");
    }

    ggml_cuda_mmh_moe_args a = {};
    a.x = (const float *) x->data;  a.x_ld = (int64_t) (x->nb[2] / sizeof(float));
    a.ids = (const int32_t *) ids->data;  a.ids_ld = (int64_t) (ids->nb[1] / sizeof(int32_t));  a.weights = (const float *) m.weights->data;
    a.T = T;  a.n_used = n_used;  a.E = E;  a.H = H;  a.I = I;  a.bits_gu = bits_gu;  a.bits_d = bits_d;
    a.w_gate = wg->data;  a.w_up = wu->data;  a.w_down = wd->data;
    a.es_gate = wg->nb[2];  a.es_up = wu->nb[2];  a.es_down = wd->nb[2];
    a.rin_g = rot[0]->data;  a.rout_g = rot[1]->data;  a.rin_u = rot[2]->data;  a.rout_u = rot[3]->data;
    a.rin_d = rot[4]->data;  a.rout_d = rot[5]->data;
    a.rs_in_g = rs[0];  a.rs_out_g = rs[1];  a.rs_in_u = rs[2];  a.rs_out_u = rs[3];  a.rs_in_d = rs[4];  a.rs_out_d = rs[5];
    a.sx = sx ? (const float *) sx->data : nullptr;  a.sg = sg ? (const float *) sg->data : nullptr;
    a.out = (float *) out->data;  a.buf_gu = buf_gu;  a.buf_had = buf_had;
    if (folded) {
        a.s_w_gate = fold.gate->src[0]->data;  a.s_w_up = fold.up->src[0]->data;  a.s_w_down = fold.down->src[0]->data;
        a.s_rin_g = fold.gate->src[3]->data;  a.s_rout_g = fold.gate->src[4]->data;
        a.s_rin_u = fold.up->src[3]->data;    a.s_rout_u = fold.up->src[4]->data;
        a.s_rin_d = fold.down->src[3]->data;  a.s_rout_d = fold.down->src[4]->data;
        a.bits_s = fold.bits;
        a.sg = (const float *) fold.sig->data;
        // an F32 gate weight: the prep launch makes the gate (sigmoid of its dot with x), and its nodes are skipped too
        const ggml_tensor * gw = fold.gin->src[0];
        static const bool gate_in = !getenv("STRIX_MMH_MOE_SHEXP_GATE") || atoi(getenv("STRIX_MMH_MOE_SHEXP_GATE")) != 0;   // DEV
        if (gate_in && (gw->type == GGML_TYPE_F32 || gw->type == GGML_TYPE_BF16) && ggml_is_contiguous(gw) && gw->ne[0] == H &&
                ggml_nelements(gw) == H && fold.gin->src[1]->nb[1] % 16 == 0 && ((uintptr_t) gw->data) % 16 == 0 && H % 4 == 0) {
            a.s_gin_w    = gw->data;
            a.s_gin_bf16 = gw->type == GGML_TYPE_BF16;
        } else if (moe_dbg) {
            fprintf(stderr, "MMH_MOE shared gate stays nodes: type %d ne %lld %lld nb1 %zu align %d\n", (int) gw->type,
                    (long long) gw->ne[0], (long long) gw->ne[1], fold.gin->src[1]->nb[1], (int) (((uintptr_t) gw->data) % 16));
        }
    }
    if (dec) {
        ggml_cuda_mmh_moe_dec_carve(a, dec_scratch);
        if (a.s_gin_w) {
            a.sg = a.s_gate;
        }
        ggml_cuda_mmh_moe_run_dec(a, ctx.stream());
    } else {
        ggml_cuda_pool_alloc<char> temp(ctx.pool(), s.temp_bytes);
        ggml_cuda_mmh_moe_scratch_carve(a, persist, temp.get());
        ggml_cuda_mmh_moe_run_main(a, ctx.stream());
    }
    CUDA_CHECK(cudaGetLastError());
    g_mmh_moe.down   = down;
    g_mmh_moe.device = ctx.device;
    g_mmh_moe.a      = a;
    g_mmh_moe.fold_sum_seen = false;
    if (folded) {
        g_mmh_moe.fold_skip[0] = fold.gate;  g_mmh_moe.fold_skip[1] = fold.up;
        g_mmh_moe.fold_skip[2] = fold.glu;   g_mmh_moe.fold_skip[3] = fold.down;
        g_mmh_moe.fold_skip[4] = a.s_gin_w ? fold.gin : nullptr;
        g_mmh_moe.fold_skip[5] = a.s_gin_w ? fold.sig : nullptr;
        g_mmh_moe.fold_gin = fold.gin;  g_mmh_moe.fold_sig = fold.sig;
        g_mmh_moe.fold_mul = fold.mul;  g_mmh_moe.fold_add = fold.add;
    } else {
        g_mmh_moe.fold_mul = nullptr;
    }
    static unsigned hits = 0, dec_hits = 0, fold_hits = 0;
    if (folded && fold_hits++ == 0) {
        fprintf(stderr, "MMH_MOE decode pipeline: the shared expert (bits %d, I %d) as one more job\n", fold.bits, fold.I);
    }
    if (!dec && hits++ == 0) {
        fprintf(stderr, "MMH_MOE grouped expert pipeline: T=%d E=%d H=%d I=%d picks=%d bits=%d/%d shexp=%d sum %d nodes on\n", T, E, H, I,
                n_used, bits_gu, bits_d, int(sx != nullptr), j - i);
    }
    if (dec && dec_hits++ == 0) {
        fprintf(stderr, "MMH_MOE decode pipeline: T=%d E=%d H=%d I=%d picks=%d bits=%d shexp=%d sum %d nodes on\n", T, E, H, I, n_used,
                bits_gu, int(sx != nullptr), j - i);
    }
    return 3;
#undef MOE_REFUSE
#else
    GGML_UNUSED(ctx); GGML_UNUSED(cgraph); GGML_UNUSED(i);
    return 0;
#endif
}

// the reduce part at the weighted sum of a block whose main part ran (ggml_cuda_try_mmh_moe): the nodes it covers after
// node i, or -1 when node i is not that sum
static int ggml_cuda_mmh_moe_reduce(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, const int i) {
#if defined(GGML_USE_HIP)
    ggml_tensor * node = cgraph->nodes[i];
    if (g_mmh_moe.fold_mul && node == g_mmh_moe.fold_mul) {
        // the folded shared expert's MUL: the sum plus its rows times the gate, written as the ADD
        GGML_ASSERT(g_mmh_moe.device == ctx.device && g_mmh_moe.fold_sum_seen && i + 1 < cgraph->n_nodes &&
                    cgraph->nodes[i + 1] == g_mmh_moe.fold_add && g_mmh_moe.fold_add->data == g_mmh_moe.a.out &&
                    "MMH_MOE: the folded shared expert's tail is not where the gate node saw it");
        ggml_cuda_mmh_moe_run_reduce(g_mmh_moe.a, ctx.stream());
        CUDA_CHECK(cudaGetLastError());
        g_mmh_moe.down = nullptr;
        g_mmh_moe.fold_mul = nullptr;
        return 1;
    }
    if (!g_mmh_moe.down || (node->src[0] != g_mmh_moe.down && node->src[1] != g_mmh_moe.down)) {
        return -1;
    }
    GGML_ASSERT(g_mmh_moe.device == ctx.device);
    ggml_cuda_moe_weighted_reduction_match m;
    GGML_ASSERT(ggml_cuda_match_moe_weighted_reduction(cgraph, i, m) && m.experts == g_mmh_moe.down &&
                "MMH_MOE: the weighted sum seen at the gate node is not there");
    int last = i + m.node_count - 1;
    if (g_mmh_moe.fold_mul) {
        GGML_ASSERT(!g_mmh_moe.fold_sum_seen);
        g_mmh_moe.fold_sum_seen = true;   // the reduce runs at the shared expert's MUL
        return last - i;
    }
    if (g_mmh_moe.a.sx) {
        last += 2;
        GGML_ASSERT(last < cgraph->n_nodes && cgraph->nodes[last]->data == g_mmh_moe.a.out);
    } else {
        GGML_ASSERT(m.dst->data == g_mmh_moe.a.out);
    }
    ggml_cuda_mmh_moe_run_reduce(g_mmh_moe.a, ctx.stream());
    CUDA_CHECK(cudaGetLastError());
    g_mmh_moe.down = nullptr;
    return last - i;
#else
    GGML_UNUSED(ctx); GGML_UNUSED(cgraph); GGML_UNUSED(i);
    return -1;
#endif
}

static int ggml_cuda_try_fuse(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {

    if (ggml_cuda_fusion_disabled()) {
        return 0;
    }

    ggml_tensor * node = cgraph->nodes[i];

    if (g_mmh_moe.fold_mul && (node == g_mmh_moe.fold_gin || node == g_mmh_moe.fold_sig)) {
        return 0;   // the folded shared expert's gate: plain nodes, its MUL is the reduce's
    }

    if (node->op == GGML_OP_SSM_SCAN) {
        const int sk = ggml_cuda_ssm_scan_cache_fuse(*cuda_ctx, cgraph, i);
        if (sk > 0) {
            return sk;
        }
    }

    if (node->op == GGML_OP_MUL_MAT_HAD && node->src[2]) {
        const int skip = ggml_cuda_try_mmh_moe(*cuda_ctx, cgraph, i);
        if (skip > 0) {
            return skip;
        }
    }
    if (node->op == GGML_OP_MUL) {
        const int skip = ggml_cuda_mmh_moe_reduce(*cuda_ctx, cgraph, i);
        if (skip >= 0) {
            return skip;
        }
    }

    // strixllama: {MUL_MAT_HAD, MUL_MAT_HAD, GLU swiglu}: the second product writes silu(gate)*up in its epilogue (EXL3
    // experts and shared experts: no GLU pass, no write and read of its second input)
    if (node->op == GGML_OP_MUL_MAT_HAD && i + 2 < cgraph->n_nodes) {
        static const bool glu_env = !getenv("STRIX_MMH_GLU") || atoi(getenv("STRIX_MMH_GLU")) != 0;
        ggml_tensor * a   = cgraph->nodes[i];
        ggml_tensor * b   = cgraph->nodes[i + 1];
        ggml_tensor * glu = cgraph->nodes[i + 2];
        if (glu_env && b->op == GGML_OP_MUL_MAT_HAD && glu->op == GGML_OP_GLU && ggml_get_glu_op(glu) == GGML_GLU_OP_SWIGLU &&
                glu->src[1] && ggml_get_op_params_i32(glu, 1) == 0 &&
                ((glu->src[0] == a && glu->src[1] == b) || (glu->src[0] == b && glu->src[1] == a)) &&
                glu->type == GGML_TYPE_F32 && ggml_are_same_shape(a, b) && ggml_are_same_shape(a, glu) &&
                ggml_is_contiguous(a) && ggml_is_contiguous(b) && ggml_is_contiguous(glu) &&
                ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_MUL_MAT_HAD, GGML_OP_MUL_MAT_HAD, GGML_OP_GLU }, { i + 2 })) {
            int out_nodes[] = { i + 2 };
            if (ggml_cuda_check_fusion_memory_ranges(cgraph, i, 3, out_nodes, 1)) {
                ggml_cuda_mul_mat_had(*cuda_ctx, a);
                ggml_cuda_mul_mat_had_glu(*cuda_ctx, b, a, glu, glu->src[0] == b);
                return 2;
            }
        }
    }

    // strixllama: {MUL_MAT_HAD with ids, MUL by the routing weights [1, n_used, T]} (the experts' down and its weighting):
    // the product writes y*w in its epilogue - one dispatch and one pass over the f32 slots less per layer
    if (node->op == GGML_OP_MUL_MAT_HAD && node->src[2] && i + 1 < cgraph->n_nodes) {
        static const bool rw_env = !getenv("STRIX_MMH_ROWMUL") || atoi(getenv("STRIX_MMH_ROWMUL")) != 0;
        ggml_tensor * mul = cgraph->nodes[i + 1];
        if (rw_env && mul->op == GGML_OP_MUL && mul->src[0] == node && mul->src[1] && mul->type == GGML_TYPE_F32) {
            const ggml_tensor * wts = mul->src[1];
            if (wts->type == GGML_TYPE_F32 && wts->ne[0] == 1 && wts->ne[1] == node->ne[1] && wts->ne[2] == node->ne[2] &&
                    wts->ne[3] == 1 && node->ne[3] == 1 && ggml_are_same_shape(mul, node) &&
                    ggml_is_contiguous(node) && ggml_is_contiguous(mul) &&
                    ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_MUL_MAT_HAD, GGML_OP_MUL }, { i + 1 })) {
                int out_nodes[] = { i + 1 };
                if (ggml_cuda_check_fusion_memory_ranges(cgraph, i, 2, out_nodes, 1)) {
                    ggml_cuda_mul_mat_had_rowmul(*cuda_ctx, node, mul);
                    return 1;
                }
            }
        }
    }

    // strixllama: a decode strip's indexer score (ggml_cuda_match_idx_score_dec): skip the keys' GET_ROWS and its RESHAPE,
    // run the whole score at the MUL_MAT
    if (node->op == GGML_OP_MUL_MAT && g_idx_dec_pending.mm == i && g_idx_dec_pending.a.out) {
        ggml_cuda_op_idx_score_dec(*cuda_ctx, g_idx_dec_pending.a);
        for (int r = strix_repeat("idxdec"); r > 0; --r) { ggml_cuda_op_idx_score_dec(*cuda_ctx, g_idx_dec_pending.a); }
        const int skip = g_idx_dec_pending.end - i;
        g_idx_dec_pending = {};
        return skip;
    }
    if (node->op == GGML_OP_GET_ROWS && GGML_CUDA_CC_IS_RDNA3_5(ggml_cuda_info().devices[cuda_ctx->device].cc)) {
        ggml_cuda_idx_dec_match m;
        if (ggml_cuda_match_idx_score_dec(cgraph, i, m)) {
            if (ggml_cuda_idx_dec_mode() == 2) {
                m.check = true;
                m.mm = -1;
                g_idx_dec_pending = m;
                return 0;
            }
            g_idx_dec_pending = m;
            return 1;
        }
    }

    if (node->op == GGML_OP_CONT) {
        int sk = ggml_cuda_qsa_pack_fuse(*cuda_ctx, cgraph, i);
        if (sk > 0) {
            return sk;
        }
        sk = ggml_cuda_cont_sigmoid_fuse(*cuda_ctx, cgraph, i);
        if (sk > 0) {
            return sk;
        }
    }
    if (node->op == GGML_OP_CONCAT || node->op == GGML_OP_CONT) {
        ggml_cuda_ple_conv_match pm;
        if (node->op == GGML_OP_CONCAT && ggml_cuda_ple_conv_match_at_concat(cgraph, i, pm)) { ggml_cuda_ple_conv_write_tail(*cuda_ctx, pm); return 1; }
        if (node->op == GGML_OP_CONT && ggml_cuda_ple_conv_match_at_tap(cgraph, i, pm)) { ggml_cuda_ple_conv_direct(*cuda_ctx, pm); return pm.silu_idx - i; }
    }
    if (node->op == GGML_OP_RMS_NORM) {
        ggml_cuda_norm_gated_match nm;
        int sk = ggml_cuda_norm_gated_match_at(cgraph, i, nm);
        if (sk > 0) {
            // strixllama: the gate GEMM's only reader is this norm - run them as one kernel, z never written (mmb.cu)
            if (nm.pre >= 0 && ggml_node_has_n_uses(cgraph, nm.pre, 1) && (nm.z == cgraph->nodes[nm.pre] || nm.z->view_src == cgraph->nodes[nm.pre]) &&
                    ggml_cuda_mmb_gemm_gnorm(*cuda_ctx, cgraph->nodes[nm.pre], nm.x, nm.w, nm.eps, nm.dst)) {
                return sk;
            }
            // strixllama: an EXL3 gate in the tiled GEMM's epilogue (mul-mat-had.cu)
            if (nm.pre >= 0 && cgraph->nodes[nm.pre]->op == GGML_OP_MUL_MAT_HAD && ggml_node_has_n_uses(cgraph, nm.pre, 1) &&
                    (nm.z == cgraph->nodes[nm.pre] || nm.z->view_src == cgraph->nodes[nm.pre]) &&
                    ggml_cuda_mul_mat_had_gnorm(*cuda_ctx, cgraph->nodes[nm.pre], nm.x, nm.w, nm.eps, nm.dst)) {
                return sk;
            }
            if (nm.pre >= 0) { if (!ggml_cuda_compute_forward(*cuda_ctx, cgraph->nodes[nm.pre])) { GGML_ABORT("norm-gated: gate MUL_MAT dispatch failed"); } }
            ggml_cuda_op_norm_gated(*cuda_ctx, nm); return sk;
        }
        sk = ggml_cuda_norm_rows_match_at(cgraph, i, nm);
        if (sk > 0) { ggml_cuda_op_norm_gated(*cuda_ctx, nm); return sk; }
        ggml_cuda_norm_scale_match sm;
        if (ggml_cuda_norm_scale_match_at(cgraph, i, sm) > 0) { ggml_cuda_op_norm_scale(*cuda_ctx, sm); return 1; }
    }
    if (node->op == GGML_OP_CONCAT || node->op == GGML_OP_SSM_CONV) {
        ggml_cuda_gdn_conv_match gm;
        if (node->op == GGML_OP_CONCAT && ggml_cuda_gdn_conv_match_at_concat(cgraph, i, gm)) {
            ggml_cuda_gdn_conv_write_tail(*cuda_ctx, gm);
            return 1;
        }
        if (node->op == GGML_OP_SSM_CONV && ggml_cuda_gdn_conv_match_at_conv(cgraph, i, gm)) {
            // strixllama: with the q and k norms after it in the same pass (STRIX_GDN_CONV_NORM=0: off)
            ggml_cuda_gdn_conv_norm gn;
            if (ggml_cuda_gdn_conv_norm_match(cgraph, gm, gn)) {
                ggml_cuda_gdn_conv_norm_run(*cuda_ctx, gm, gn);
                return gn.last - i;
            }
            ggml_cuda_gdn_conv_direct(*cuda_ctx, gm);
            return 1;
        }
    }

    // strixllama: two narrow F32 products of the same activations next to each other (the GDN beta and alpha, which
    // qwen4exp puts side by side) in one pass over them - only where each would have gone to the narrow kernel on its
    // own, whose results these are bitwise. The second's output is not read or written between the two nodes
    if (node->op == GGML_OP_MUL_MAT && i + 1 < cgraph->n_nodes && cgraph->nodes[i + 1]->op == GGML_OP_MUL_MAT &&
            cgraph->nodes[i + 1]->src[1] == node->src[1] && node->src[0]->type == GGML_TYPE_F32 && cgraph->nodes[i + 1]->src[0]->type == GGML_TYPE_F32) {
        ggml_tensor * nx = cgraph->nodes[i + 1];
        bool n0 = false, n1 = false;
        if (ggml_get_op_params_i32(node, 1) != GGML_HINT_SRC0_IS_HADAMARD && ggml_get_op_params_i32(nx, 1) != GGML_HINT_SRC0_IS_HADAMARD &&
                g_hc_inject.node != node && g_hc_inject.node != nx &&
                ggml_cuda_skinny_f32_ok(*cuda_ctx, node->src[0], node->src[1], node, &n0) && n0 &&
                ggml_cuda_skinny_f32_ok(*cuda_ctx, nx->src[0], nx->src[1], nx, &n1) && n1 &&
                ggml_cuda_mmb_f32_narrow_pair(*cuda_ctx, node->src[0], nx->src[0], node->src[1], node, nx)) {
            return 1;
        }
    }

    if (node->op == GGML_OP_MUL_MAT && ggml_cuda_mmb_gatemix() && i + 1 < cgraph->n_nodes && GGML_CUDA_CC_IS_RDNA3_5(ggml_cuda_info().devices[cuda_ctx->device].cc)) {
        // HC gate GEMM [320 -> 10240] whose only consumer is the fused stream mix: run GEMM + sigmoid + mix in one kernel
        const ggml_tensor * w = node->src[0], * lo = node->src[1];
        // strixllama: Q8_0 as well - Unsloth's file keeps the HC weights at Q8_0
        if ((w->type == GGML_TYPE_IQ4_NL || w->type == GGML_TYPE_Q8_0) && w->ne[0] == 320 && w->ne[1] == 10240 && ggml_node_has_n_uses(cgraph, i, 1) && ggml_cuda_mmb_supported_mm(w, lo, node)) {
            ggml_cuda_hc_mix_args ma;
            const int count = ggml_cuda_hc_mix_closed(cgraph, i + 1, ma);
            if (count > 0 && ma.gate == node && ggml_cuda_hc_gate_mix(*cuda_ctx, w, lo, ma.xn, ma.dst, ma.hc, ma.scale, ma.bias)) return count;
        }
    }

    if (node->op == GGML_OP_MUL_MAT && GGML_CUDA_CC_IS_RDNA3_5(ggml_cuda_info().devices[cuda_ctx->device].cc)) {
        ggml_cuda_idx_score_args sa;
        const int count = ggml_cuda_match_idx_score(cgraph, i, sa);
        if (count > 0) { ggml_cuda_op_idx_score(*cuda_ctx, sa); return count - 1; }
    }

    if (node->op == GGML_OP_UNARY && GGML_CUDA_CC_IS_RDNA3_5(ggml_cuda_info().devices[cuda_ctx->device].cc)) {
        ggml_cuda_idx_relu_sum_args args;
        const int count = ggml_cuda_match_idx_relu_sum(cgraph, i, args);
        if (count > 0) { ggml_cuda_op_idx_relu_sum(*cuda_ctx, args); return count - 1; }
    }

    // strixllama: gather straight into F16 rather than through an F32 copy (strixllama-getrows-cast.cuh)
    if (node->op == GGML_OP_GET_ROWS) {
        if (const int skip = ggml_cuda_get_rows_cast_try(*cuda_ctx, cgraph, i)) {
            return skip;
        }
    }

    if (GGML_CUDA_CC_IS_RDNA3_5(ggml_cuda_info().devices[cuda_ctx->device].cc)) {
        ggml_cuda_hc_mix_args args;
        const int count=ggml_cuda_hc_mix_closed(cgraph,i,args);
        if (count>0) {
            ggml_cuda_op_hc_mix_reduce(*cuda_ctx,args);
            return count-1;
        }
    }

    if (GGML_CUDA_CC_IS_RDNA3_5(ggml_cuda_info().devices[cuda_ctx->device].cc)) {
        ggml_cuda_qsa_expand_args args;
        const int skip = ggml_cuda_match_qsa_expand(cgraph,i,args);
        if (skip) {
            const bool alias_ok = ggml_cuda_qsa_expand_alias_ok(args);
            if (getenv("QSA_EXPAND_MATCH_DEBUG")) fprintf(stderr,"QSA_EXPAND_MEMORY explicit_rw_disjoint=%d cast_self=%d,%d,%d\n",alias_ok,
                args.cast_blocks->src[1]==args.cast_blocks,args.cast_cells->src[1]==args.cast_cells,args.expanded->src[1]==args.expanded);
            if (alias_ok) {
                ggml_cuda_op_qsa_expand(*cuda_ctx,args);
                return skip;
            }
        }
    }

    if (GGML_CUDA_CC_IS_RDNA3_5(ggml_cuda_info().devices[cuda_ctx->device].cc)) {
        ggml_cuda_hc_combine_norm_args args;
        const int skip = ggml_cuda_match_hc_combine_norm(
            cgraph, i, ggml_cuda_info().devices[cuda_ctx->device].warp_size, args);
        if (skip > 0 && !ggml_cuda_hc_combine_norm_alias_ok(args) && ggml_cuda_mmb_is_bf16_only(args.out_xn)) {
            GGML_ABORT("hc_combine_norm: output %s is marked BF16-only but the fused kernel was refused (buffer aliasing)", args.out_xn->name);
        }
        if (skip > 0 && ggml_cuda_hc_combine_norm_alias_ok(args)) {
            // strixllama: the BF16 copy's rows (a token's hc streams) are padded when the F32 streams are stored as well
            // (mmb_pad_ld); a BF16-only xn keeps dense rows, as its other readers expect
            if (ggml_cuda_mmb_is_bf16_only(args.out_xn)) {
                args.out_xn_bf16 = ggml_cuda_mmb_cache_reserve(*cuda_ctx, args.out_xn, (size_t) ggml_nelements(args.out_xn));
            } else {
                // a token's hc streams make one row: K = n_embd * hc, one row a token. The shapes come from the residual
                // [n_embd, hc, T] - xn itself is the 2-D [n_embd * hc, T] of the norm's MUL
                const int64_t K = args.out_res->ne[0] * args.out_res->ne[1], rows = args.out_res->ne[2];
                GGML_ASSERT(K * rows == ggml_nelements(args.out_xn) && args.out_res->ne[3] == 1);
                args.out_xn_bf16 = ggml_cuda_mmb_cache_reserve_ld(*cuda_ctx, args.out_xn, rows, K, &args.xn_bf16_ld);
                GGML_ASSERT(args.xn_bf16_ld == 0 || args.xn_bf16_ld == K + 64);
            }
            args.store_xn_f32 = !(args.out_xn_bf16 && ggml_cuda_mmb_is_bf16_only(args.out_xn));
            if (ggml_cuda_mmb_res16()) {   // 16-bit residual stream, in place over each tensor's own buffer
                const ggml_tensor * rin  = args.residual->view_src ? args.residual->view_src : args.residual;
                const ggml_tensor * rout = args.out_res->view_src  ? args.out_res->view_src  : args.out_res;
                const bool in16  = ggml_cuda_mmb_is_bf16_only(rin)  && args.residual->view_offs == 0;
                const bool out16 = ggml_cuda_mmb_is_bf16_only(rout) && args.out_res->view_offs == 0;
                if (args.out_res->data == args.residual->data && in16 != out16) {
                    GGML_ABORT("hc_combine_norm: in-place residual with mismatched BF16 marks (%s); the graph_optimize "
                               "allocation dependency should have separated them", args.out_res->name);
                }
                const ggml_tensor * rblk = args.block_out->view_src ? args.block_out->view_src : args.block_out;
                args.blk_in_bf16 = (ggml_cuda_mmb_blk16() && ggml_cuda_mmb_is_bf16_only(rblk) && args.block_out->view_offs == 0)
                                 ? (const uint16_t *) args.block_out->data : nullptr;
                args.res_in_bf16  = in16  ? (const uint16_t *) args.residual->data : nullptr;
                args.res_out_bf16 = out16 ? (uint16_t *) args.out_res->data : nullptr;
                static unsigned dbg = 0;
                if (getenv("LLAMA_HC16_DEBUG") && ggml_nrows(args.out_res) >= 512 && dbg++ < 40)
                    fprintf(stderr, "HC_RES16 out=%s in=%s(root %s) in16=%d out16=%d inplace=%d\n", args.out_res->name,
                        args.residual->name, rin->name, (int) in16, (int) out16,
                        (int) (args.out_res->data == args.residual->data));
            }
            { static unsigned dbg = 0; if (getenv("LLAMA_HC16_DEBUG") && ggml_nrows(args.out_xn) >= 4096 && dbg++ < 8) fprintf(stderr, "HC_CN dispatch xn=%s bf16=%d store_f32=%d\n", args.out_xn->name, args.out_xn_bf16 != nullptr, (int) args.store_xn_f32); }
            ggml_cuda_hc_inject_plan(cgraph, i + skip, args, *cuda_ctx);
            ggml_cuda_hc_xres_plan(cgraph, i + skip, args, *cuda_ctx);
            ggml_cuda_op_hc_combine_norm(*cuda_ctx, args);
            return skip;
        }
    }

    if (GGML_CUDA_CC_IS_RDNA3_5(ggml_cuda_info().devices[cuda_ctx->device].cc)) {
        ggml_cuda_get_rows_mean4_match match;
        if (ggml_cuda_match_get_rows_mean4(cgraph, i, match)) {
            const int output_idx = i + match.node_count - 1;
            if (ggml_cuda_check_fusion_memory_ranges(cgraph, i, match.node_count, &output_idx, 1) &&
                    !ggml_cuda_mean4_ranges_overlap(match.src, match.dst) &&
                    !ggml_cuda_mean4_ranges_overlap(match.ids, match.dst)) {
                ggml_cuda_op_get_rows_mean4(*cuda_ctx, match.src, match.ids, match.dst);
                return match.node_count - 1;
            }
        }
    }

    if (ggml_cuda_shexp_fuse() && node->op == GGML_OP_MUL_MAT_ID && cuda_ctx->stream_context().concurrent_events.empty() &&
            ggml_cuda_match_shared_expert(cgraph, i, i + 3) &&
            ggml_cuda_should_fuse_mul_mat_vec_q(cgraph->nodes[i + 2]->src[1])) {
        const int outputs[] = { i + 2, i + 5 };
        if (ggml_cuda_check_fusion_memory_ranges(cgraph, i, 6, outputs, 2)) {
            ggml_tensor * routed = cgraph->nodes[i + 2];
            ggml_tensor * shared = cgraph->nodes[i + 5];
            const ggml_tensor * up = routed->src[1];
            ggml_cuda_mm_fusion_args_host fusion{};
            fusion.gate = routed->src[0]->src[0];
            fusion.glu_op = ggml_get_glu_op(routed);
            fusion.glu_limit = ggml_get_op_params_f32(routed, 3);
            fusion.shared_up = shared->src[1]->src[0];
            fusion.shared_gate = shared->src[0]->src[0];
            fusion.shared_dst = shared;
            ggml_cuda_mul_mat_vec_q(*cuda_ctx, up->src[0], up->src[1], up->src[2], routed, &fusion);
            return 5;
        }
    }

    if (node->op == GGML_OP_MUL) {
        ggml_cuda_moe_weighted_reduction_match match;
        if (ggml_cuda_match_moe_weighted_reduction(cgraph, i, match)) {
            // strixllama: the shared expert's tail right after the weighted sum - MUL(shexp, sigmoid gate [1, T]), ADD(sum,
            // that) - in the same kernel (qwen4exp builds the shared expert first for prompt-sized batches so that the
            // two follow the sum): the sum and the gated shared expert are no longer written out and read back, bitwise
            // the same (moe_shexp_tail). STRIX_SHEXP_TAIL=0: off
            static const bool shexp_tail = !getenv("STRIX_SHEXP_TAIL") || atoi(getenv("STRIX_SHEXP_TAIL")) != 0;
            const int im = i + match.node_count, ia = im + 1;
            if (shexp_tail && ia < cgraph->n_nodes && !ggml_cuda_mmb_is_bf16_only(match.dst)) {
                const ggml_tensor * mul = cgraph->nodes[im];
                ggml_tensor *       add = cgraph->nodes[ia];
                const ggml_tensor * sx  = mul->op == GGML_OP_MUL ? mul->src[0] : nullptr;
                const ggml_tensor * sg  = mul->op == GGML_OP_MUL ? mul->src[1] : nullptr;
                const bool shape_ok = sx && sg && add->op == GGML_OP_ADD &&
                    ((add->src[0] == match.dst && add->src[1] == mul) || (add->src[1] == match.dst && add->src[0] == mul)) &&
                    mul->type == GGML_TYPE_F32 && add->type == GGML_TYPE_F32 && sx->type == GGML_TYPE_F32 && sg->type == GGML_TYPE_F32 &&
                    ggml_is_contiguous(sx) && ggml_is_contiguous(sg) && ggml_is_contiguous(add) && ggml_is_contiguous(mul) &&
                    ggml_are_same_shape(sx, match.dst) && ggml_are_same_shape(add, match.dst) && ggml_are_same_shape(mul, match.dst) &&
                    sg->ne[0] == 1 && sg->ne[1] == match.dst->ne[1] && sg->ne[2] == 1 && sg->ne[3] == 1 &&
                    match.dst->ne[0] % 4 == 0;
                if (shape_ok && ggml_node_has_n_uses(cgraph, im - 1, 1) && ggml_node_has_n_uses(cgraph, im, 1)) {
                    const int count = match.node_count + 2;
                    if (ggml_cuda_check_fusion_memory_ranges(cgraph, i, count, &ia, 1)) {
                        ggml_cuda_op_moe_weighted_reduction(*cuda_ctx, match.experts, match.expert_scale, match.weights, add,
                                                            nullptr, sx, sg);
                        return count - 1;
                    }
                }
            }
            int count = match.node_count;
            const ggml_tensor * merge = nullptr;
            {   // shared-expert merge: ADD(reduction, ffn_shexp_gated) right after, with the reduction used only there
                const int nadd = i + match.node_count;
                if (ggml_cuda_mmb_blk16() && nadd < cgraph->n_nodes && ggml_node_has_n_uses(cgraph, i + match.node_count - 1, 1)) {
                    const ggml_tensor * add = cgraph->nodes[nadd];
                    if (add->op == GGML_OP_ADD && add->type == GGML_TYPE_F32 && ggml_is_contiguous(add) &&
                            ggml_are_same_shape(add, match.dst) && ggml_cuda_mmb_is_bf16_only(add)) {
                        const ggml_tensor * o = add->src[0] == match.dst ? add->src[1] : (add->src[1] == match.dst ? add->src[0] : nullptr);
                        if (o && o->type == GGML_TYPE_F32 && ggml_is_contiguous(o) && ggml_are_same_shape(o, match.dst)) {
                            merge = o; match.dst = const_cast<ggml_tensor *>(add); count = match.node_count + 1;
                        }
                    }
                }
            }
            const int output_idx = i + count - 1;
            if (ggml_cuda_check_fusion_memory_ranges(cgraph, i, count, &output_idx, 1)) {
                ggml_cuda_op_moe_weighted_reduction(
                    *cuda_ctx, match.experts, match.expert_scale, match.weights, match.dst, merge);
                return count - 1;
            }
        }
    }

    // gated_delta_net -> cpy: scatter recurrent-state snapshots into the cache
    if (node->op == GGML_OP_GATED_DELTA_NET) {
        ggml_cuda_gated_delta_net_fused_cache fused_state_cpy;
        const int nodes_to_skip = ggml_cuda_try_gdn_cache_fusion(cgraph, i, fused_state_cpy);
        if (nodes_to_skip > 0) {
            const auto it = std::find_if(g_gdn_direct.begin(), g_gdn_direct.end(),
                    [node](const ggml_cuda_gdn_direct & d) { return d.gdn == node; });
            if (it != g_gdn_direct.end()) {
                fused_state_cpy.state_base = it->base;
                fused_state_cpy.state_rows = it->rows;
                g_gdn_direct.erase(it);
            }
#ifdef GGML_CUDA_DEBUG
            GGML_LOG_INFO("%s: fused gated_delta_net snapshot copies for %s (skipped %d nodes)\n",
                          __func__, node->name, nodes_to_skip);
#endif
            ggml_cuda_op_gated_delta_net_fused_cache(*cuda_ctx, node, fused_state_cpy);
            return nodes_to_skip;
        }
    }

    // ssm_scan -> cpy: scatter recurrent-state snapshots into the cache
    if (node->op == GGML_OP_SSM_SCAN) {
        ggml_cuda_ssm_scan_fused_cache fused_state_cpy;
        const int nodes_to_skip = ggml_cuda_try_ssm_scan_cache_fusion(cgraph, i, fused_state_cpy);
        if (nodes_to_skip > 0) {
#ifdef GGML_CUDA_DEBUG
            GGML_LOG_INFO("%s: fused ssm_scan snapshot copies for %s (skipped %d nodes)\n",
                          __func__, node->name, nodes_to_skip);
#endif
            ggml_cuda_op_ssm_scan_fused_cache(*cuda_ctx, node, fused_state_cpy);
            return nodes_to_skip;
        }
    }

    ggml_cuda_topk_moe_match topk;
    if (ggml_cuda_match_topk_moe(cgraph, i, topk) &&
            ggml_cuda_check_fusion_memory_ranges(cgraph, i, topk.node_count, topk.out_nodes, 2, true)) {
        ggml_cuda_op_topk_moe(*cuda_ctx, topk.logits, topk.weights, topk.ids, topk.clamp, topk.scale, topk.bias, topk.args);
        return topk.node_count - 1;
    }

    //RoPE + view + set-rows
    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, {})) {
        ggml_tensor * rope     = cgraph->nodes[i];
        ggml_tensor * set_rows = cgraph->nodes[i + 2];

        ggml_cuda_op_rope_fused(*cuda_ctx, rope, set_rows);
        return 2;
    }

    // Snake activation: y = x + sin(a*x)^2 * inv_b
    // Naive 5-op decomposition emitted by frontends: mul -> sin -> sqr -> mul -> add
    if (ggml_can_fuse_subgraph(cgraph, i,
            { GGML_OP_MUL, GGML_OP_SIN, GGML_OP_SQR, GGML_OP_MUL, GGML_OP_ADD },
            { i + 4 })) {
        const ggml_tensor * mul0 = cgraph->nodes[i];
        const ggml_tensor * sqr  = cgraph->nodes[i + 2];
        const ggml_tensor * mul1 = cgraph->nodes[i + 3];
        ggml_tensor *       add  = cgraph->nodes[i + 4];

        // x carries the full activation shape, a is the broadcast operand
        const ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
        const ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

        // mul1 reads sqr and inv_b in either operand order
        const ggml_tensor * inv_b = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];

        // closure check: the trailing add must read the same x as the leading mul
        const ggml_tensor * x_in_add = (add->src[0] == mul1) ? add->src[1] : add->src[0];

        // Kernel iterates over total = T * C, so x and add must be 2D and
        // a / inv_b must collapse to [1, C, 1, 1]. Higher dims are not handled.
        const bool dim_ok   = (x->ne[2]   == 1 && x->ne[3]   == 1) &&
                              (add->ne[2] == 1 && add->ne[3] == 1) &&
                              (a->ne[2]   == 1 && a->ne[3]   == 1);
        const bool shape_ok = ggml_are_same_shape(a, inv_b) && a->ne[0] == 1 && a->ne[1] == x->ne[1];

        // x is in the supported whitelist and every chain intermediate shares
        // x's type. launch_snake reads a and inv_b as const float *, so they
        // stay F32.
        const ggml_tensor * sin1 = cgraph->nodes[i + 1];
        const bool types_ok = (x->type == GGML_TYPE_F32 || x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_BF16) &&
                              (a->type    == GGML_TYPE_F32) && (inv_b->type == GGML_TYPE_F32) &&
                              (mul0->type == x->type) && (sin1->type  == x->type) &&
                              (sqr->type  == x->type) && (mul1->type  == x->type) &&
                              (add->type  == x->type);

        // kernel reads x[idx] and a[c] / inv_b[c] linearly, so every operand is contiguous
        const bool contig_ok = ggml_is_contiguous(x) && ggml_is_contiguous(add) &&
                               ggml_is_contiguous(a) && ggml_is_contiguous(inv_b);

        if (types_ok && shape_ok && dim_ok && contig_ok && x_in_add == x) {
            ggml_cuda_op_snake_fused(*cuda_ctx, x, a, inv_b, add);
            return 4;
        }
    }

    // multi-(add or mul)
    if (node->op == GGML_OP_ADD || node->op == GGML_OP_MUL) {
        int     n_fuse = 0;
        ggml_op ops[8];
        std::fill(ops, ops + 8, node->op);

        for (; n_fuse <= 6; ++n_fuse) {
            if (!ggml_can_fuse(cgraph, i + n_fuse, ops + n_fuse, 2)) {
                break;
            }
            if (cgraph->nodes[i + n_fuse] != cgraph->nodes[i + n_fuse + 1]->src[0]) {
                break;
            }
            if (!ggml_are_same_layout(cgraph->nodes[i + n_fuse]->src[1], cgraph->nodes[i + n_fuse + 1]->src[1])) {
                break;
            }
        }

        n_fuse++;

        if (n_fuse > 1) {
            ggml_tensor fused_node;
            memcpy(&fused_node, node, sizeof(ggml_tensor));
            for (int j = 0; j < n_fuse - 1; ++j) {
                fused_node.src[j + 2] = cgraph->nodes[i + j + 1]->src[1];
            }
            fused_node.data = cgraph->nodes[i + n_fuse - 1]->data;
            if (node->op == GGML_OP_ADD) {
                ggml_cuda_op_fused_add(*cuda_ctx, &fused_node, n_fuse);
            } else {
                ggml_cuda_op_fused_mul(*cuda_ctx, &fused_node, n_fuse);
            }
            return n_fuse - 1;
        }
    }

    bool fused_mul_mat_vec = false;
    int  fused_node_count  = 0;

    auto get_mul_mat_scale = [](const ggml_tensor * scale_node, const ggml_tensor * mm_node) -> const ggml_tensor * {
        const bool scale_lhs_mm = scale_node->src[0] == mm_node;
        const bool scale_rhs_mm = scale_node->src[1] == mm_node;
        if (!scale_lhs_mm && !scale_rhs_mm) {
            return nullptr;
        }

        const ggml_tensor * scale = scale_lhs_mm ? scale_node->src[1] : scale_node->src[0];
        if (mm_node->src[0]->type != GGML_TYPE_NVFP4 || scale_node->type != GGML_TYPE_F32 ||
                scale->type != GGML_TYPE_F32 || !ggml_is_contiguous(scale) || ggml_nelements(scale) != 1 ||
                !ggml_are_same_shape(scale_node, mm_node)) {
            return nullptr;
        }

        return scale;
    };

    auto get_mul_mat_id_scale = [](const ggml_tensor * reshape, const ggml_tensor * repeat, const ggml_tensor * getrows,
            const ggml_tensor * scale_node, const ggml_tensor * mm_node) -> const ggml_tensor * {
        if (repeat->src[0] != reshape || getrows->src[0] != repeat || getrows->src[1] != mm_node->src[2]) {
            return nullptr;
        }
        if (!((scale_node->src[0] == mm_node && scale_node->src[1] == getrows) ||
                (scale_node->src[0] == getrows && scale_node->src[1] == mm_node))) {
            return nullptr;
        }

        const ggml_tensor * scale = reshape->src[0];
        if (mm_node->src[0]->type != GGML_TYPE_NVFP4 || scale_node->type != GGML_TYPE_F32 ||
                scale->type != GGML_TYPE_F32 || !ggml_is_contiguous(scale) || ggml_nelements(scale) != mm_node->src[0]->ne[2] ||
                !ggml_are_same_shape(scale_node, mm_node)) {
            return nullptr;
        }

        return scale;
    };

    auto get_bias_tensor = [](const ggml_tensor * bias_node, const ggml_tensor * mul_node, ggml_op op_bias) -> const ggml_tensor * {
        if (op_bias == GGML_OP_ADD) {
            if (bias_node->src[0] == mul_node) {
                return bias_node->src[1];
            }
            if (bias_node->src[1] == mul_node) {
                return bias_node->src[0];
            }
            return nullptr;
        }
        GGML_ASSERT(op_bias == GGML_OP_ADD_ID);
        GGML_ASSERT(bias_node->src[0] == mul_node);
        return bias_node->src[1];
    };

    // gate + glu + up, with optional scale/bias on both lanes.
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        if (op == GGML_OP_MUL_MAT) {
            for (const bool with_bias : { false, true }) {
                const int gate_idx       = i;
                const int gate_scale_idx = i + 1;
                const int gate_bias_idx  = with_bias ? i + 2 : -1;
                const int up_idx         = with_bias ? i + 3 : i + 2;
                const int up_scale_idx   = up_idx + 1;
                const int up_bias_idx    = with_bias ? up_idx + 2 : -1;
                const int glu_idx        = with_bias ? up_idx + 3 : up_idx + 2;

                const int out_nodes[] = { glu_idx };
                ggml_op ops[7];
                if (with_bias) {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                    ops[2] = bias_op;
                    ops[3] = op;
                    ops[4] = GGML_OP_MUL;
                    ops[5] = bias_op;
                    ops[6] = GGML_OP_GLU;
                } else {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                    ops[2] = op;
                    ops[3] = GGML_OP_MUL;
                    ops[4] = GGML_OP_GLU;
                }
                const int n_ops = with_bias ? 7 : 5;

                if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1) ||
                        !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                    continue;
                }

                ggml_tensor * gate_n       = cgraph->nodes[gate_idx];
                ggml_tensor * gate_scale_n = cgraph->nodes[gate_scale_idx];
                ggml_tensor * gate_out_n   = with_bias ? cgraph->nodes[gate_bias_idx] : gate_scale_n;
                ggml_tensor * up_n         = cgraph->nodes[up_idx];
                ggml_tensor * up_scale_n   = cgraph->nodes[up_scale_idx];
                ggml_tensor * up_out_n     = with_bias ? cgraph->nodes[up_bias_idx] : up_scale_n;
                const ggml_tensor * glu = cgraph->nodes[glu_idx];

                if (!ggml_cuda_should_fuse_mul_mat(up_n, gate_n, glu,
                        with_bias ? up_out_n : nullptr, with_bias ? gate_out_n : nullptr, up_scale_n, gate_scale_n)) {
                    continue;
                }

                const ggml_tensor * gate_scale = get_mul_mat_scale(gate_scale_n, gate_n);
                const ggml_tensor * up_scale   = get_mul_mat_scale(up_scale_n, up_n);
                if (!gate_scale || !up_scale) {
                    continue;
                }

                const ggml_tensor * up_bias   = with_bias ? get_bias_tensor(up_out_n, up_scale_n, bias_op) : nullptr;
                const ggml_tensor * gate_bias = with_bias ? get_bias_tensor(gate_out_n, gate_scale_n, bias_op) : nullptr;
                if (with_bias && (!ggml_are_same_shape(gate_out_n->src[0], gate_out_n->src[1]) ||
                        !ggml_are_same_shape(up_out_n->src[0], up_out_n->src[1]))) {
                    continue;
                }

                const ggml_tensor * src0 = up_n->src[0];
                const ggml_tensor * src1 = up_n->src[1];
                const ggml_tensor * ids  = up_n->src[2];

                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate       = gate_n->src[0];
                fusion_data.x_bias     = up_bias;
                fusion_data.gate_bias  = gate_bias;
                fusion_data.x_scale    = up_scale;
                fusion_data.gate_scale = gate_scale;
                fusion_data.glu_op     = ggml_get_glu_op(glu);
                fusion_data.glu_limit  = ggml_get_op_params_f32(glu, 3);

                if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n)) {
                    ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, cgraph->nodes[glu_idx], &fusion_data);
                    fused_mul_mat_vec = true;
                    fused_node_count  = n_ops;
                    break;
                }
            }

            if (fused_mul_mat_vec) {
                break;
            }
        } else {
            for (const bool with_bias : { false, true }) {
                const int gate_idx       = i;
                const int gate_scale_idx = i + 4;
                const int gate_bias_idx  = with_bias ? i + 5 : -1;
                const int up_idx         = with_bias ? i + 6 : i + 5;
                const int up_scale_idx   = up_idx + 4;
                const int up_bias_idx    = with_bias ? up_idx + 5 : -1;
                const int glu_idx        = with_bias ? up_idx + 6 : up_idx + 5;

                const int out_nodes[] = { glu_idx };
                ggml_op ops[13];
                if (with_bias) {
                    ops[0]  = op;
                    ops[1]  = GGML_OP_RESHAPE;
                    ops[2]  = GGML_OP_REPEAT;
                    ops[3]  = GGML_OP_GET_ROWS;
                    ops[4]  = GGML_OP_MUL;
                    ops[5]  = bias_op;
                    ops[6]  = op;
                    ops[7]  = GGML_OP_RESHAPE;
                    ops[8]  = GGML_OP_REPEAT;
                    ops[9]  = GGML_OP_GET_ROWS;
                    ops[10] = GGML_OP_MUL;
                    ops[11] = bias_op;
                    ops[12] = GGML_OP_GLU;
                } else {
                    ops[0]  = op;
                    ops[1]  = GGML_OP_RESHAPE;
                    ops[2]  = GGML_OP_REPEAT;
                    ops[3]  = GGML_OP_GET_ROWS;
                    ops[4]  = GGML_OP_MUL;
                    ops[5]  = op;
                    ops[6]  = GGML_OP_RESHAPE;
                    ops[7]  = GGML_OP_REPEAT;
                    ops[8]  = GGML_OP_GET_ROWS;
                    ops[9]  = GGML_OP_MUL;
                    ops[10] = GGML_OP_GLU;
                }
                const int n_ops = with_bias ? 13 : 11;

                if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1) ||
                        !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                    continue;
                }

                ggml_tensor * gate_n       = cgraph->nodes[gate_idx];
                ggml_tensor * gate_scale_n = cgraph->nodes[gate_scale_idx];
                ggml_tensor * gate_out_n   = with_bias ? cgraph->nodes[gate_bias_idx] : gate_scale_n;
                ggml_tensor * up_n         = cgraph->nodes[up_idx];
                ggml_tensor * up_scale_n   = cgraph->nodes[up_scale_idx];
                ggml_tensor * up_out_n     = with_bias ? cgraph->nodes[up_bias_idx] : up_scale_n;
                const ggml_tensor * glu = cgraph->nodes[glu_idx];

                if (!ggml_cuda_should_fuse_mul_mat(up_n, gate_n, glu,
                        with_bias ? up_out_n : nullptr, with_bias ? gate_out_n : nullptr, up_scale_n, gate_scale_n)) {
                    continue;
                }

                const ggml_tensor * gate_scale = get_mul_mat_id_scale(cgraph->nodes[gate_idx + 1], cgraph->nodes[gate_idx + 2],
                        cgraph->nodes[gate_idx + 3], gate_scale_n, gate_n);
                const ggml_tensor * up_scale = get_mul_mat_id_scale(cgraph->nodes[up_idx + 1], cgraph->nodes[up_idx + 2],
                        cgraph->nodes[up_idx + 3], up_scale_n, up_n);
                if (!gate_scale || !up_scale) {
                    continue;
                }

                const ggml_tensor * up_bias   = with_bias ? get_bias_tensor(up_out_n, up_scale_n, bias_op) : nullptr;
                const ggml_tensor * gate_bias = with_bias ? get_bias_tensor(gate_out_n, gate_scale_n, bias_op) : nullptr;

                const ggml_tensor * src0 = up_n->src[0];
                const ggml_tensor * src1 = up_n->src[1];
                const ggml_tensor * ids  = up_n->src[2];

                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate       = gate_n->src[0];
                fusion_data.x_bias     = up_bias;
                fusion_data.gate_bias  = gate_bias;
                fusion_data.x_scale    = up_scale;
                fusion_data.gate_scale = gate_scale;
                fusion_data.glu_op     = ggml_get_glu_op(glu);
                fusion_data.glu_limit  = ggml_get_op_params_f32(glu, 3);

                if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n)) {
                    ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, cgraph->nodes[glu_idx], &fusion_data);
                    fused_mul_mat_vec = true;
                    fused_node_count  = n_ops;
                    break;
                }
            }

            if (fused_mul_mat_vec) {
                break;
            }
        }

        if (ggml_cuda_can_fuse(cgraph, i, { op, bias_op, op, bias_op, GGML_OP_GLU }, {})) {
            ggml_tensor * glu         = cgraph->nodes[i + 4];
            ggml_tensor * gate_bias_n = glu->src[0];
            ggml_tensor * up_bias_n   = glu->src[1];

            //we don't assume the order for {gate, up}. Instead infer it from the bias tensor
            ggml_tensor * gate_n = nullptr;
            ggml_tensor * up_n   = nullptr;

            if (gate_bias_n->src[0] == cgraph->nodes[i] || gate_bias_n->src[1] == cgraph->nodes[i]) {
                gate_n = cgraph->nodes[i];
                up_n   = cgraph->nodes[i + 2];
            } else if (gate_bias_n->src[0] == cgraph->nodes[i + 2] || gate_bias_n->src[1] == cgraph->nodes[i + 2]) {
                gate_n = cgraph->nodes[i + 2];
                up_n   = cgraph->nodes[i];
            } else {
                continue;
            }

            const ggml_tensor * up_bias_tensor   = get_bias_tensor(up_bias_n, up_n, bias_op);
            const ggml_tensor * gate_bias_tensor = get_bias_tensor(gate_bias_n, gate_n, bias_op);

            if (!up_bias_tensor || !gate_bias_tensor) {
                continue;
            }

            // we don't support repeating adds
            if (bias_op == GGML_OP_ADD && (!ggml_are_same_shape(gate_bias_n->src[0], gate_bias_n->src[1]) ||
                                           !ggml_are_same_shape(up_bias_n->src[0], up_bias_n->src[1]))) {
                continue;
            }

            const ggml_tensor * src0 = up_n->src[0];
            const ggml_tensor * src1 = up_n->src[1];
            const ggml_tensor * ids  = up_n->src[2];

            if (ggml_cuda_should_fuse_mul_mat_vec_f(up_n)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate_n->src[0];
                fusion_data.x_bias    = up_bias_tensor;
                fusion_data.gate_bias = gate_bias_tensor;
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 5;
                break;
            }

            if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate_n->src[0];
                fusion_data.x_bias    = up_bias_tensor;
                fusion_data.gate_bias = gate_bias_tensor;
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 5;
                break;
            }
        } else if (ggml_cuda_can_fuse(cgraph, i, { op, op, GGML_OP_GLU }, {})) {
            ggml_tensor * glu  = cgraph->nodes[i + 2];
            ggml_tensor * gate = glu->src[0];
            ggml_tensor * up   = glu->src[1];

            bool ok = (gate == cgraph->nodes[i] && up == cgraph->nodes[i + 1]) ||
                      (gate == cgraph->nodes[i + 1] && up == cgraph->nodes[i]);

            if (!ok) {
                continue;
            }

            const ggml_tensor * src0 = up->src[0];
            const ggml_tensor * src1 = up->src[1];
            const ggml_tensor * ids  = up->src[2];

            if (op == GGML_OP_MUL_MAT_ID && ggml_cuda_mmb_supported_glu(gate->src[0], up->src[0], src1, ids, glu)) {
                ggml_cuda_mul_mat_id_mmb_glu(*cuda_ctx, gate->src[0], up->src[0], src1, ids, glu);
                fused_mul_mat_vec = true;
                fused_node_count  = 3;
                break;
            }

            if (ggml_cuda_should_fuse_mul_mat_vec_f(up)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate->src[0];
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 3;
                break;
            }

            if (ggml_cuda_should_fuse_mul_mat_vec_q(up)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate->src[0];
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 3;
                break;
            }
        }
    }

    if (fused_mul_mat_vec) {
        return fused_node_count - 1;
    }

    fused_mul_mat_vec = false;
    fused_node_count  = 0;

    // mul_mat + scale + optional bias
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        for (const bool with_bias : { false, true }) {
            const int n_ops = op == GGML_OP_MUL_MAT ? (with_bias ? 3 : 2) : (with_bias ? 6 : 5);
            const int out_nodes[] = { i + n_ops - 1 };
            ggml_op ops[6];
            if (op == GGML_OP_MUL_MAT) {
                if (with_bias) {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                    ops[2] = bias_op;
                } else {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                }
            } else {
                if (with_bias) {
                    ops[0] = op;
                    ops[1] = GGML_OP_RESHAPE;
                    ops[2] = GGML_OP_REPEAT;
                    ops[3] = GGML_OP_GET_ROWS;
                    ops[4] = GGML_OP_MUL;
                    ops[5] = bias_op;
                } else {
                    ops[0] = op;
                    ops[1] = GGML_OP_RESHAPE;
                    ops[2] = GGML_OP_REPEAT;
                    ops[3] = GGML_OP_GET_ROWS;
                    ops[4] = GGML_OP_MUL;
                }
            }

            if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1) ||
                    !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                continue;
            }

            ggml_tensor * mm_node    = cgraph->nodes[i];
            ggml_tensor * scale_node = op == GGML_OP_MUL_MAT ? cgraph->nodes[i + 1] : cgraph->nodes[i + 4];
            ggml_tensor * out_node   = with_bias ? cgraph->nodes[i + n_ops - 1] : scale_node;

            const ggml_tensor * scale = nullptr;
            if (op == GGML_OP_MUL_MAT) {
                scale = get_mul_mat_scale(scale_node, mm_node);
            } else {
                scale = get_mul_mat_id_scale(cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[i + 3], scale_node, mm_node);
            }
            if (!scale) {
                continue;
            }

            const ggml_tensor * bias = with_bias ? get_bias_tensor(out_node, scale_node, bias_op) : nullptr;
            if (with_bias && !bias) {
                continue;
            }
            if (with_bias && bias_op == GGML_OP_ADD && !ggml_are_same_shape(out_node->src[0], out_node->src[1])) {
                continue;
            }
            if (with_bias && bias_op == GGML_OP_ADD_ID && out_node->src[2] != mm_node->src[2]) {
                continue;
            }

            const ggml_tensor * src0 = mm_node->src[0];
            const ggml_tensor * src1 = mm_node->src[1];
            const ggml_tensor * ids  = mm_node->src[2];

            ggml_cuda_mm_fusion_args_host fusion_data{};
            fusion_data.x_bias  = bias;
            fusion_data.x_scale = scale;

            if (ggml_cuda_should_fuse_mul_mat_vec_q(mm_node)) {
                ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, out_node, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = n_ops;
                break;
            }
        }
        if (fused_mul_mat_vec) {
            break;
        }
    }

    if (fused_mul_mat_vec) {
        return fused_node_count - 1;
    }

    // mul_mat + add
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        if (!ggml_can_fuse(cgraph, i, { op, bias_op })) {
            continue;
        }

        ggml_tensor * mm_node   = cgraph->nodes[i];
        ggml_tensor * bias_node = cgraph->nodes[i + 1];

        ggml_tensor * bias_tensor = nullptr;
        if (bias_op == GGML_OP_ADD) {
            if (bias_node->src[0] == mm_node) {
                bias_tensor = bias_node->src[1];
            } else if (bias_node->src[1] == mm_node) {
                bias_tensor = bias_node->src[0];
            } else {
                continue;
            }
        } else {
            if (bias_node->src[0] != mm_node) {
                continue;
            }
            bias_tensor = bias_node->src[1];
        }

        const ggml_tensor * src0 = mm_node->src[0];
        const ggml_tensor * src1 = mm_node->src[1];
        const ggml_tensor * ids  = mm_node->src[2];

        if (bias_op == GGML_OP_ADD_ID && bias_node->src[2] != ids) {
            continue;
        }

        if (bias_op == GGML_OP_ADD && !ggml_are_same_shape(bias_node->src[0], bias_node->src[1])) {
            continue;
        }

        ggml_cuda_mm_fusion_args_host fusion_data{};
        fusion_data.x_bias = bias_tensor;

        if (ggml_cuda_should_fuse_mul_mat_vec_f(mm_node)) {
            ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, bias_node, &fusion_data);
            fused_mul_mat_vec = true;
            fused_node_count  = 2;
            break;
        }

        if (ggml_cuda_should_fuse_mul_mat_vec_q(mm_node)) {
            ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, bias_node, &fusion_data);
            fused_mul_mat_vec = true;
            fused_node_count  = 2;
            break;
        }
    }

    if (fused_mul_mat_vec) {
        return fused_node_count - 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, {})) {
        ggml_cuda_op_rms_norm_mul_rope_fused(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[i + 4]);
        return 4;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE }, {})) {
        ggml_cuda_op_rms_norm_mul_rope_fused(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], nullptr);
        return 2;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD }, {})) {
        ggml_cuda_op_rms_norm_fused_add(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
        return 2;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL }, {})) {
        ggml_cuda_op_rms_norm_fused(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_SCALE }, {})) {
        ggml_cuda_op_rms_norm_scale_fused(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_ADD, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
        ggml_cuda_op_ssm_conv(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
        return 2;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
        ggml_cuda_op_ssm_conv(*cuda_ctx, node, /*bias_add_node=*/ nullptr, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SILU }) ||
        ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SIGMOID }) ||
        ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SOFTPLUS })) {
        ggml_cuda_op_unary_mul(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_SQR }, { GGML_UNARY_OP_RELU })) {
        ggml_cuda_op_relu_sqr(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_DSV4_HC_POST }, { GGML_UNARY_OP_SIGMOID })) {
        ggml_cuda_op_dsv4_hc_post(*cuda_ctx, cgraph->nodes[i + 3], node, cgraph->nodes[i + 2]);
        return 3;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE }, { GGML_UNARY_OP_TANH })) {
        ggml_cuda_op_softcap(*cuda_ctx, cgraph->nodes[i + 2], node);
        return 2;
    }

    // strixllama: try the triple before the pair, so a scale/sigmoid/scale run collapses to one
    // launch rather than two. STRIX_NO_SIGMOID_FUSE=1 takes this path out without touching the
    // IQ3_S kernel, which is how the two are told apart when hunting a numerics regression.
    static const bool strixllama_sigmoid_fuse = !getenv("STRIX_NO_SIGMOID_FUSE");
    if (strixllama_sigmoid_fuse &&
        ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE }, { GGML_UNARY_OP_SIGMOID })) {
        ggml_cuda_op_scale_sigmoid(*cuda_ctx, cgraph->nodes[i + 2], node, /*post_scale =*/ true);
        return 2;
    }

    if (strixllama_sigmoid_fuse &&
        ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SCALE, GGML_OP_UNARY }, { GGML_UNARY_OP_SIGMOID })) {
        ggml_cuda_op_scale_sigmoid(*cuda_ctx, cgraph->nodes[i + 1], node, /*post_scale =*/ false);
        return 1;
    }

    // strixllama: consecutive elementwise chains (see strixllama-chain.cuh), tried after every specific pattern
    if (const int chain = ggml_cuda_strixllama_chain_try(*cuda_ctx, cgraph, i)) {
        return chain;
    }
    return 0;
}

static void ggml_cuda_graph_evaluate_and_capture(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, const bool use_cuda_graph, const bool cuda_graph_update_required, const void * graph_key) {
    bool graph_evaluated_or_captured = false;

    g_idx_dec_pending = {};

    // strixllama: whether HIP graphs actually engage decides how much of a decode step is kernel
    // launch overhead, and the upstream notice is a GGML_LOG_DEBUG compiled out of a release
    // build. Report the first few graphs either way under LLAMA_GRAPH_TRACE=1.
    if (getenv("LLAMA_GRAPH_TRACE")) {
        static unsigned strixllama_calls = 0, strixllama_used = 0, strixllama_recaptured = 0;
        ++strixllama_calls;
        if (use_cuda_graph) { ++strixllama_used; }
        if (use_cuda_graph && cuda_graph_update_required) { ++strixllama_recaptured; }
        if (strixllama_calls % 64 == 0) {
            fprintf(stderr, "GRAPH_TRACE: %u calls, graph used %u (%.0f%%), re-captured %u (%.0f%% of used), nodes=%d\n",
                    strixllama_calls, strixllama_used, 100.0 * strixllama_used / strixllama_calls,
                    strixllama_recaptured, strixllama_used ? 100.0 * strixllama_recaptured / strixllama_used : 0.0,
                    cgraph->n_nodes);
        }
        // once: where the nodes go, since with graphs replaying properly the node COUNT is what a
        // decode step costs (7325 nodes over 48 layers is ~153 per layer)
        static bool strixllama_histogrammed = false;
        if (!strixllama_histogrammed && cgraph->n_nodes > 1000 && getenv("LLAMA_GRAPH_DUMP")) {
            // LLAMA_GRAPH_DUMP=<n>: print the op and name of nodes [n, n+220), enough to read one
            // layer's chain and see which elementwise runs are fusable
            const int from = atoi(getenv("LLAMA_GRAPH_DUMP"));
            for (int i = from; i < cgraph->n_nodes && i < from + 220; ++i) {
                const ggml_tensor * t = cgraph->nodes[i];
                char strixllama_op[48];
                if (t->op == GGML_OP_UNARY) {
                    snprintf(strixllama_op, sizeof(strixllama_op), "UNARY:%s", ggml_unary_op_name(ggml_get_unary_op(t)));
                } else {
                    snprintf(strixllama_op, sizeof(strixllama_op), "%s", ggml_op_name(t->op));
                }
                fprintf(stderr, "GRAPH_DUMP %4d %-16s %-28s ne=[%lld,%lld,%lld]%s\n", i,
                        strixllama_op, t->name,
                        (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2],
                        ggml_cuda_is_view_or_noop(t) ? "  (view/noop)" : "");
            }
        }
        if (!strixllama_histogrammed && cgraph->n_nodes > 1000) {
            strixllama_histogrammed = true;
            int counts[GGML_OP_COUNT] = {0}, skipped = 0;
            for (int i = 0; i < cgraph->n_nodes; ++i) {
                if (ggml_cuda_is_view_or_noop(cgraph->nodes[i])) { ++skipped; continue; }
                ++counts[cgraph->nodes[i]->op];
            }
            fprintf(stderr, "GRAPH_TRACE: node census of %d (%d are views/noops):\n", cgraph->n_nodes, skipped);
            for (int pass = 0; pass < 14; ++pass) {   // print the 14 biggest, descending
                int best = -1;
                for (int op = 0; op < GGML_OP_COUNT; ++op) {
                    if (counts[op] > 0 && (best < 0 || counts[op] > counts[best])) { best = op; }
                }
                if (best < 0) { break; }
                fprintf(stderr, "GRAPH_TRACE:   %-24s %5d  (%.1f per layer)\n",
                        ggml_op_name((ggml_op) best), counts[best], counts[best] / 48.0);
                counts[best] = 0;
            }
        }
    }

    // flag used to determine whether it is an integrated_gpu
    const bool integrated            = ggml_cuda_info().devices[cuda_ctx->device].integrated;

    ggml_cuda_stream_context & stream_ctx = cuda_ctx->stream_context();
    bool                         is_concurrent_event_active = false;
    ggml_cuda_concurrent_event * concurrent_event           = nullptr;
    bool                         should_launch_concurrent_events = false;

    const auto try_launch_concurrent_event = [&](const ggml_tensor * node) {
        if (stream_ctx.concurrent_events.find(node) != stream_ctx.concurrent_events.end()) {
            concurrent_event = &stream_ctx.concurrent_events[node];

            is_concurrent_event_active = true;

            GGML_LOG_DEBUG("Launching %d streams at %s\n", concurrent_event->n_streams, node->name);

            cudaStream_t main_stream = cuda_ctx->stream();  // this should be stream 0
            GGML_ASSERT(cuda_ctx->curr_stream_no == 0);
            CUDA_CHECK(cudaEventRecord(concurrent_event->fork_event, main_stream));

            for (int i = 1; i <= concurrent_event->n_streams; ++i) {
                cudaStream_t stream = cuda_ctx->stream(cuda_ctx->device, i);
                CUDA_CHECK(cudaStreamWaitEvent(stream, concurrent_event->fork_event));
            }
        }
    };

    while (!graph_evaluated_or_captured) {
        // Only perform the graph execution if CUDA graphs are not enabled, or we are capturing the graph.
        // With the use of CUDA graphs, the execution will be performed by the graph launch.
        if (!use_cuda_graph || cuda_graph_update_required) {
            [[maybe_unused]] int prev_i = 0;
            strix_tl_start(cuda_ctx, cgraph, graph_key);
            // strixllama: no conv state copy left marked from a graph that stopped early (node addresses are reused)
            ggml_cuda_gdn_conv_tail_reset();

            if (stream_ctx.concurrent_events.size() > 0) {
                should_launch_concurrent_events = true;
                for (const auto & [tensor, event] : stream_ctx.concurrent_events) {
                    should_launch_concurrent_events = should_launch_concurrent_events && event.is_valid();
                }
            }

            if (should_launch_concurrent_events) {
                // Restore original node order within each concurrent region to enable fusion within streams

                std::unordered_map<const ggml_tensor *, int> node_to_idx;
                node_to_idx.reserve(cgraph->n_nodes);
                for (int i = 0; i < cgraph->n_nodes; ++i) {
                    node_to_idx[cgraph->nodes[i]] = i;
                }

                for (auto & [fork_node, event] : stream_ctx.concurrent_events) {
                    // Find positions of all nodes from this event in the current graph
                    std::vector<int> positions;
                    positions.reserve(event.original_order.size());

                    bool all_found = true;
                    for (const ggml_tensor * orig_node : event.original_order) {
                        auto it = node_to_idx.find(orig_node);
                        if (it != node_to_idx.end()) {
                            positions.push_back(it->second);
                        } else {
                            all_found = false;
                            break;
                        }
                    }

                    if (!all_found || positions.size() != event.original_order.size()) {
                        continue;
                    }

                    // Sort positions to get contiguous range
                    std::vector<int> sorted_positions = positions;
                    std::sort(sorted_positions.begin(), sorted_positions.end());

                    bool is_contiguous = true;
                    for (size_t i = 1; i < sorted_positions.size(); ++i) {
                        if (sorted_positions[i] != sorted_positions[i-1] + 1) {
                            is_contiguous = false;
                            break;
                        }
                    }

                    if (!is_contiguous) {
                        continue;
                    }

                    // Restore original order at the sorted positions
                    int start_pos = sorted_positions[0];
                    for (size_t i = 0; i < event.original_order.size(); ++i) {
                        cgraph->nodes[start_pos + i] = const_cast<ggml_tensor *>(event.original_order[i]);
                    }
                }
            } else {
                stream_ctx.concurrent_events.clear();
            }

            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];
                if (is_concurrent_event_active) {
                    GGML_ASSERT(concurrent_event);

                    if (node == concurrent_event->join_node) {
                        cuda_ctx->curr_stream_no = 0;
                        for (int i = 1; i <= concurrent_event->n_streams; ++i) {
                            // Wait on join events of forked streams in the main stream
                            CUDA_CHECK(cudaEventRecord(concurrent_event->join_events[i - 1],
                                                       cuda_ctx->stream(cuda_ctx->device, i)));
                            CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), concurrent_event->join_events[i - 1]));
                        }

                        is_concurrent_event_active = false;
                        concurrent_event           = nullptr;
                    } else {
                        GGML_ASSERT (concurrent_event->stream_mapping.find(node) != concurrent_event->stream_mapping.end());
                        cuda_ctx->curr_stream_no = concurrent_event->stream_mapping[node];
                        GGML_LOG_DEBUG("Setting stream no to %d for node %s\n", cuda_ctx->curr_stream_no, node->name);
                    }
                } else if (i - prev_i > 1) {
                    //the previous node was fused
                    const ggml_tensor * prev_node = cgraph->nodes[i - 1];
                    try_launch_concurrent_event(prev_node);

                    if (is_concurrent_event_active) {
                        cuda_ctx->curr_stream_no = concurrent_event->stream_mapping[node];
                        GGML_LOG_DEBUG("Setting stream no to %d for node %s\n", cuda_ctx->curr_stream_no, node->name);
                    }
                }

                prev_i = i;

                // strixllama: the graph's node count is NOT the kernel count - views and noops are
                // skipped and fusions swallow whole runs. Count what actually dispatches, since
                // that is what a decode step pays for.
                if (getenv("LLAMA_GRAPH_TRACE")) {
                    static unsigned strixllama_passes = 0, strixllama_dispatch = 0, strixllama_fused_away = 0, strixllama_views = 0;
                    static unsigned strixllama_by_op[GGML_OP_COUNT] = {0};
                    static unsigned strixllama_by_unary[GGML_UNARY_OP_COUNT] = {0};
                    if (i == 0) {
                        if (strixllama_passes == 1) {      // report the second pass: the first still captures
                            fprintf(stderr, "GRAPH_TRACE: per pass - %u dispatched, %u swallowed by fusions, %u views/noops, %d nodes\n",
                                    strixllama_dispatch, strixllama_fused_away, strixllama_views, cgraph->n_nodes);
                            unsigned tmp[GGML_OP_COUNT];
                            memcpy(tmp, strixllama_by_op, sizeof(tmp));
                            for (int pass = 0; pass < 12; ++pass) {   // the 12 biggest DISPATCHED ops
                                int best = -1;
                                for (int op = 0; op < GGML_OP_COUNT; ++op) {
                                    if (tmp[op] > 0 && (best < 0 || tmp[op] > tmp[best])) { best = op; }
                                }
                                if (best < 0) { break; }
                                fprintf(stderr, "GRAPH_TRACE:   dispatched %-18s %5u  (%.1f per layer)\n",
                                        ggml_op_name((ggml_op) best), tmp[best], tmp[best] / 48.0);
                                tmp[best] = 0;
                            }
                            for (int u = 0; u < GGML_UNARY_OP_COUNT; ++u) {   // UNARY hides its real op
                                if (strixllama_by_unary[u]) {
                                    fprintf(stderr, "GRAPH_TRACE:     unary %-16s %5u  (%.1f per layer)\n",
                                            ggml_unary_op_name((ggml_unary_op) u), strixllama_by_unary[u],
                                            strixllama_by_unary[u] / 48.0);
                                }
                            }
                        }
                        ++strixllama_passes;
                        strixllama_dispatch = strixllama_fused_away = strixllama_views = 0;
                        memset(strixllama_by_op, 0, sizeof(strixllama_by_op));
                        memset(strixllama_by_unary, 0, sizeof(strixllama_by_unary));
                    }
                    if (ggml_cuda_is_view_or_noop(node)) { ++strixllama_views; }
                    else {
                        ++strixllama_dispatch; ++strixllama_by_op[node->op];
                        if (node->op == GGML_OP_UNARY) { ++strixllama_by_unary[ggml_get_unary_op(node)]; }
                    }
                    g_strixllama_fused_away = &strixllama_fused_away;
                }

                if (ggml_cuda_is_view_or_noop(node)) {
                    continue;
                }

                if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                    continue;
                }

                // strixllama ablation: STRIX_SKIP_OPS=SCALE,UNARY,UNARY:SIGMOID,... replaces those nodes by a
                // zero fill so the per-token cost of an op class can be measured (output is garbage)
                if (strixllama_skip_op(node)) {
                    // STRIX_SKIP_NOFILL=1: no zero fill either, so the dispatch itself goes (its cost with it)
                    static const bool nofill = getenv("STRIX_SKIP_NOFILL") && atoi(getenv("STRIX_SKIP_NOFILL"));
                    if (!nofill) {
                        CUDA_CHECK(cudaMemsetAsync(node->data, 0, ggml_nbytes(node), cuda_ctx->stream()));
                    }
                    continue;
                }

                // strixllama: STRIX_NODE_TIMING=1 (with GGML_CUDA_DISABLE_GRAPHS=1) brackets every dispatch,
                // fused or not, with stream events; the top entries are printed after the graph
                if (g_strixllama_node_timing) {
                    cudaEvent_t e; CUDA_CHECK(cudaEventCreateWithFlags(&e, 0)); CUDA_CHECK(cudaEventRecord(e, cuda_ctx->stream()));
                    g_strixllama_ev0.push_back(e); g_strixllama_ev_node.push_back(i);
                }
                struct strixllama_timing_close {
                    bool on; ggml_backend_cuda_context * ctx;
                    ~strixllama_timing_close() {
                        if (!on) { return; }
                        cudaEvent_t e; CUDA_CHECK(cudaEventCreateWithFlags(&e, 0)); CUDA_CHECK(cudaEventRecord(e, ctx->stream()));
                        g_strixllama_ev1.push_back(e);
                    }
                } strixllama_timing_guard{g_strixllama_node_timing, cuda_ctx};

                // strixllama: a state gather the gated delta net does without (see ggml_cuda_gdn_gather_skip)
                if (ggml_cuda_gdn_gather_skip(cgraph, i)) {
                    continue;
                }
                // strixllama: a conv state copy the conv's kernel does (ggml_cuda_gdn_conv_match::n_tail)
                if (ggml_cuda_gdn_conv_tail_skip(node)) {
                    continue;
                }
                // strixllama: a shared expert product the decode pipeline made (ggml_cuda_mmh_moe_fold_skip)
                if (ggml_cuda_mmh_moe_fold_skip(node)) {
                    continue;
                }

                const ggml_cuda_idx_dec_match strix_pending_before = g_idx_dec_pending;
                int nodes_to_skip = ggml_cuda_try_fuse(cuda_ctx, cgraph, i);

                if (nodes_to_skip != 0) {
                    for (int r = strix_repeat_times(cgraph, i, nodes_to_skip); r > 0; --r) {
                        const ggml_cuda_idx_dec_match strix_pending_after = g_idx_dec_pending;
                        g_idx_dec_pending = strix_pending_before;
                        GGML_ASSERT(ggml_cuda_try_fuse(cuda_ctx, cgraph, i) == nodes_to_skip);
                        g_idx_dec_pending = strix_pending_after;
                    }
                    strix_tl_mark(cuda_ctx, cgraph, i, nodes_to_skip);
#ifdef GGML_CUDA_DEBUG
                    const int last_fused = i + nodes_to_skip;
                    GGML_LOG_INFO("nodes_fused: %d, first: %s (%s), last: %s (%s)\n",
                            nodes_to_skip + 1, ggml_op_name(node->op), node->name,
                            ggml_op_name(cgraph->nodes[last_fused]->op), cgraph->nodes[last_fused]->name);
#endif
                    if (g_strixllama_fused_away) { *g_strixllama_fused_away += nodes_to_skip; }
                    i += nodes_to_skip;
                    continue;
                }
#ifndef NDEBUG
                // On integrated GPUs (APUs, e.g. RDNA3.5) the scheduler may place a
                // node's output on the host-visible buffer, which the compute path
                // handles. Allow that here, mirroring the src-tensor check below.
                assert(node->buffer->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) ||
                       (integrated && ggml_backend_buft_is_cuda_host(node->buffer->buft)));
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    if (node->src[j] != nullptr) {
                        assert(node->src[j]->buffer);
                        assert(node->src[j]->buffer->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) ||
                               (integrated && ggml_backend_buft_is_cuda_host(node->src[j]->buffer->buft)));
                    }
                }
#else
                GGML_UNUSED(integrated);
#endif  // NDEBUG

                bool ok = ggml_cuda_compute_forward(*cuda_ctx, node);
                for (int r = ok ? strix_repeat_times(cgraph, i, 0) : 0; r > 0; --r) {
                    ggml_cuda_compute_forward(*cuda_ctx, node);
                }
                if (ok) {
                    strix_tl_mark(cuda_ctx, cgraph, i, 0);
                }
                if (ok && g_idx_dec_pending.check && i == g_idx_dec_pending.end) {
                    ggml_cuda_idx_dec_check(*cuda_ctx, g_idx_dec_pending);
                    g_idx_dec_pending = {};
                }
                if (!ok) {
                    GGML_LOG_ERROR("%s: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
                }
                GGML_ASSERT(ok);

                if (!is_concurrent_event_active) {
                    try_launch_concurrent_event(node);
               }
            }

            // strixllama: per-node timing report (STRIX_NODE_TIMING=1)
            if (g_strixllama_node_timing && !g_strixllama_ev0.empty() && g_strixllama_ev0.size() == g_strixllama_ev1.size()) {
                CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));
                struct row { float ms; int node; };
                std::vector<row> rows;
                float total = 0.0f;
                for (size_t k = 0; k < g_strixllama_ev0.size(); ++k) {
                    float ms = 0.0f;
                    CUDA_CHECK(cudaEventElapsedTime(&ms, g_strixllama_ev0[k], g_strixllama_ev1[k]));
                    rows.push_back({ms, g_strixllama_ev_node[k]});
                    total += ms;
                    CUDA_CHECK(cudaEventDestroy(g_strixllama_ev0[k]));
                    CUDA_CHECK(cudaEventDestroy(g_strixllama_ev1[k]));
                }
                // aggregate by op + shape so the 48 copies of a layer op add up
                std::map<std::string, std::pair<float, int>> by_kind;
                for (const row & r : rows) {
                    const ggml_tensor * t = cgraph->nodes[r.node];
                    char key[256];
                    snprintf(key, sizeof(key), "%-18s %-12s [%lld,%lld,%lld,%lld]", ggml_op_name(t->op),
                             t->op == GGML_OP_UNARY ? ggml_unary_op_name(ggml_get_unary_op(t)) : (t->op == GGML_OP_GLU ? ggml_glu_op_name(ggml_get_glu_op(t)) : ""),
                             (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);
                    auto & e = by_kind[key];
                    e.first += r.ms; e.second += 1;
                }
                std::vector<std::pair<std::string, std::pair<float, int>>> agg(by_kind.begin(), by_kind.end());
                std::sort(agg.begin(), agg.end(), [](const auto & a, const auto & b) { return a.second.first > b.second.first; });
                fprintf(stderr, "NODE_TIMING: graph of %d nodes, %zu dispatches, %.2f ms of GPU time; top kinds:\n", cgraph->n_nodes, rows.size(), total);
                for (size_t k = 0; k < agg.size() && k < 24; ++k) {
                    fprintf(stderr, "  %8.3f ms %4d x %s\n", agg[k].second.first, agg[k].second.second, agg[k].first.c_str());
                }
                // STRIX_NODE_TIMING_LIST=n: the first n dispatches of a graph of more than 1000 tokens' rows, in order
                // STRIX_NODE_TIMING_LIST_T=t picks the graphs whose routed experts see t tokens instead (the first 3)
                int64_t list_t = -1;
                for (int k = 0; k < cgraph->n_nodes && list_t < 0; ++k) {
                    if (cgraph->nodes[k]->op == GGML_OP_MUL_MAT_ID ||
                        (cgraph->nodes[k]->op == GGML_OP_MUL_MAT_HAD && cgraph->nodes[k]->src[2])) {   // EXL3 experts
                        list_t = cgraph->nodes[k]->ne[2];
                    }
                }
                static int n_listed_t = 0;
                const bool list_by_t = getenv("STRIX_NODE_TIMING_LIST_T") && list_t == atoll(getenv("STRIX_NODE_TIMING_LIST_T")) && n_listed_t < 3;
                if (list_by_t) {
                    ++n_listed_t;
                }
                if (getenv("STRIX_NODE_TIMING_LIST") && !rows.empty() && (list_by_t || (!getenv("STRIX_NODE_TIMING_LIST_T") && cgraph->nodes[rows[0].node]->ne[1] >= 1000))) {
                    const size_t n_list = (size_t) atoi(getenv("STRIX_NODE_TIMING_LIST"));
                    for (size_t k = 0; k < rows.size() && k < n_list; ++k) {
                        const ggml_tensor * t = cgraph->nodes[rows[k].node];
                        const ggml_tensor * s0 = t->src[0];
                        fprintf(stderr, "  LIST %4zu n%-5d %8.3f ms %-14s %-28s [%lld,%lld,%lld,%lld] %s src0 %s [%lld,%lld,%lld,%lld] %s\n", k, rows[k].node, rows[k].ms, ggml_op_name(t->op), t->name,
                                (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3], ggml_type_name(t->type),
                                s0 ? s0->name : "-", s0 ? (long long) s0->ne[0] : 0LL, s0 ? (long long) s0->ne[1] : 0LL, s0 ? (long long) s0->ne[2] : 0LL,
                                s0 ? (long long) s0->ne[3] : 0LL, s0 ? ggml_type_name(s0->type) : "-");
                    }
                }
                std::sort(rows.begin(), rows.end(), [](const row & a, const row & b) { return a.ms > b.ms; });
                for (size_t k = 0; k < rows.size() && k < 6; ++k) {
                    const ggml_tensor * t = cgraph->nodes[rows[k].node];
                    fprintf(stderr, "  single: %8.3f ms %s '%s' [%lld,%lld,%lld,%lld]\n", rows[k].ms, ggml_op_name(t->op), t->name,
                            (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);
                }
                g_strixllama_ev0.clear(); g_strixllama_ev1.clear(); g_strixllama_ev_node.clear();
            }
        }

#ifdef USE_CUDA_GRAPH
        ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);
        if (use_cuda_graph && cuda_graph_update_required) { // End CUDA graph capture
            if (graph->graph != nullptr) {
                CUDA_CHECK(cudaGraphDestroy(graph->graph));
                graph->graph = nullptr;
            }

            CUDA_CHECK(cudaStreamEndCapture(cuda_ctx->stream(), &graph->graph));
            graph_evaluated_or_captured = true; // CUDA graph has been captured

            std::lock_guard<std::mutex> lock(ggml_cuda_lock);
            if (ggml_cuda_lock_counter.fetch_sub(1, std::memory_order_relaxed) == 1) {
                ggml_cuda_lock_cv.notify_all();
            }
        } else {
            graph_evaluated_or_captured = true; // ggml graph has been directly evaluated
        }
    }

    if (use_cuda_graph) {
        ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);
        if (graph->instance == nullptr) { // Create executable graph from captured graph.
            CUDA_CHECK(cudaGraphInstantiate(&graph->instance, graph->graph, NULL, NULL, 0));
        }
        if (cuda_graph_update_required) { // Update graph executable
            ggml_cuda_graph_update_executable(cuda_ctx, graph_key);
        }
        // Launch graph
        CUDA_CHECK(cudaGraphLaunch(graph->instance, cuda_ctx->stream()));
#else
        GGML_UNUSED(graph_key);
        graph_evaluated_or_captured = true;
#endif  // USE_CUDA_GRAPH
    }
}

#ifdef USE_CUDA_GRAPH
static bool ggml_cuda_graph_set_enabled(ggml_backend_cuda_context * cuda_ctx, const void * graph_key) {
    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);

    if (graph->graph == nullptr) {
        if (ggml_cuda_info().devices[cuda_ctx->device].cc < GGML_CUDA_CC_VOLTA) {
            if (!graph->disable_due_to_gpu_arch && ggml_cuda_graph_debug_enabled()) {
                GGML_LOG_DEBUG("%s: disabling CUDA graphs due to GPU architecture\n", __func__);
            }
            graph->disable_due_to_gpu_arch = true;
        }
    }

    return graph->is_enabled();
}
#endif // USE_CUDA_GRAPH

static void strix_ab_step(const ggml_cgraph * cgraph) {
    static const int every = getenv("STRIX_AB") ? atoi(getenv("STRIX_AB")) : 0;
    // STRIX_AB_MIN_NODES=n: only graphs of at least n nodes count (with MTP: the verify passes, a step apart)
    static const int min_nodes = getenv("STRIX_AB_MIN_NODES") ? atoi(getenv("STRIX_AB_MIN_NODES")) : 64;
    if (every <= 0 || cgraph->n_nodes < min_nodes || ggml_nrows(cgraph->nodes[0]) >= 64) {
        return;
    }
    // the wall time from one decode graph's compute to the next, by side (a whole step for one conversation without
    // MTP: host work, launch, GPU, wait); steps right after a flip (graphs captured anew) and gaps over 1 s (idle
    // between requests) are left out. Medians every 256 steps a side
    {
        static int64_t prev = 0;
        static int prev_side = -1;
        static std::vector<double> per_side[2];
        const int64_t now = ggml_time_us();
        if (prev && prev_side == g_strix_ab && g_strix_ab_since >= 2 && now - prev < 1000000) {
            per_side[g_strix_ab].push_back((now - prev) / 1000.0);
            if (per_side[0].size() >= 256 && per_side[1].size() >= 256) {
                double med[2];
                for (int s = 0; s < 2; ++s) {
                    std::sort(per_side[s].begin(), per_side[s].end());
                    med[s] = per_side[s][per_side[s].size() / 2];
                }
                fprintf(stderr, "STRIX_AB step: A %.3f ms, B %.3f ms (B - A %+.3f ms, medians of %zu / %zu decode graphs)\n",
                        med[0], med[1], med[1] - med[0], per_side[0].size(), per_side[1].size());
                per_side[0].clear();
                per_side[1].clear();
            }
        }
        prev = now;
        prev_side = g_strix_ab;
    }
    static int computes = 0;
    if (++computes % every == 0) {
        g_strix_ab ^= 1;
        ++g_strix_rep_gen;      // every graph captured anew
        g_strix_ab_since = 0;
    } else {
        ++g_strix_ab_since;
    }
}

// graph timing / mark-lifetime instrumentation (LLAMA_GRAPH_TIMING=1)
static int64_t      g_gt_prev_return   = 0;
static bool         g_gt_after_compute = true;
static const void * g_gt_first_split   = nullptr;
static bool graph_timing_enabled() { static const bool e = getenv("LLAMA_GRAPH_TIMING") && atoi(getenv("LLAMA_GRAPH_TIMING")); return e; }

static enum ggml_status ggml_backend_cuda_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    const int64_t gt_t0 = ggml_time_us();
    ggml_cuda_mmb_begin_graph();
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;

    ggml_cuda_set_device(cuda_ctx->device);
    strix_poll_dirty(cuda_ctx->stream());

    bool use_cuda_graph             = false;
    bool cuda_graph_update_required = false;
    const void * graph_key = nullptr;

#ifdef USE_CUDA_GRAPH
    graph_key = ggml_cuda_graph_get_key(cgraph);

    ggml_cuda_graph_set_enabled(cuda_ctx, graph_key);

    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);
    strix_repeat_dump_flush(cgraph->n_nodes);
    strix_repeat_file_reload();
    strix_ab_step(cgraph);
    if (strix_tl_every() > 0 && !g_strix_tl_buf) {
        CUDA_CHECK(cudaMalloc(&g_strix_tl_buf, STRIX_TL_MAX * sizeof(uint64_t)));
    }
    {
        static std::unordered_map<const void *, int> strix_rep_seen;
        int & seen = strix_rep_seen[(const void *) graph];
        if (seen != g_strix_rep_gen) {
            seen = g_strix_rep_gen;
            graph->node_props.clear();
            graph->uid = 0;
        }
    }
    // [GRAPH_DIAG] one-shot report for the big prefill graphs: why HIP graphs do or do not engage
    const bool graph_diag = getenv("LLAMA_GRAPH_DIAG") && atoi(getenv("LLAMA_GRAPH_DIAG")) != 0 && cgraph->n_nodes > 1000;
    if (graph->is_enabled()) {
        const bool graph_compatible = ggml_cuda_graph_check_compability(cgraph);
        if (graph_diag) {
            static unsigned n = 0;
            if (n++ < 12) {
                fprintf(stderr, "GRAPH_DIAG nodes=%d key=%p enabled=1 compatible=%d warmup_complete=%d uid=%llu\n",
                        cgraph->n_nodes, (const void *) graph_key, (int) graph_compatible, (int) graph->warmup_complete,
                        (unsigned long long) cgraph->uid);
            }
        }
        if (graph_compatible) {
            const bool properties_changed = ggml_cuda_graph_update_required(cuda_ctx, cgraph, graph_key);

            if (!graph->warmup_complete) {
                // Warmup: need at least 2 calls with no property change on the 2nd call
                if (graph_diag) { static unsigned m = 0; if (m++ < 12) {
                    fprintf(stderr, "GRAPH_DIAG   warmup: properties_changed=%d\n", (int) properties_changed); } }
                if (!properties_changed) {
                    graph->warmup_complete = true;
                    if (ggml_cuda_graph_debug_enabled()) {
                        GGML_LOG_DEBUG("%s: CUDA graph warmup complete\n", __func__);
                    }
                    use_cuda_graph = true;
                    cuda_graph_update_required = true;
                }
                // else: properties changed or first call - execute directly (use_cuda_graph stays false)
            } else {
                // Post-warmup: normal CUDA graph operation
                if (properties_changed) {
                    // Properties changed - reset warmup, execute directly until stable again
                    graph->warmup_complete = false;
                    if (ggml_cuda_graph_debug_enabled()) {
                        GGML_LOG_DEBUG("%s: CUDA graph warmup reset\n", __func__);
                    }
                } else {
                    use_cuda_graph = true;
                    cuda_graph_update_required = graph->instance == nullptr;
                }
            }
        }
    }
#endif // USE_CUDA_GRAPH

    if (use_cuda_graph && cuda_graph_update_required) {
        // Start CUDA graph capture
        {
            std::lock_guard<std::mutex> lock(ggml_cuda_lock);
            ggml_cuda_lock_counter.fetch_add(1, std::memory_order_relaxed);
        }

        CUDA_CHECK(cudaStreamBeginCapture(cuda_ctx->stream(), cudaStreamCaptureModeRelaxed));
    }

    ggml_cuda_graph_evaluate_and_capture(cuda_ctx, cgraph, use_cuda_graph, cuda_graph_update_required, graph_key);
    strix_tl_after(cuda_ctx, graph_key);

    if (graph_timing_enabled()) {
        const int64_t gt_t1 = ggml_time_us();
        CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));
        const int64_t gt_t2 = ggml_time_us();
        const ggml_tensor * f = cgraph->n_nodes ? cgraph->nodes[0] : nullptr; const ggml_tensor * l = cgraph->n_nodes ? cgraph->nodes[cgraph->n_nodes - 1] : nullptr;
        fprintf(stderr, "GT compute n=%d first=%s(%lld) last=%s graph=%d upd=%d hostgap=%.1fms submit=%.1fms gpu_wait=%.1fms\n", cgraph->n_nodes,
            f ? f->name : "-", f ? (long long) ggml_nrows(f) : 0, l ? l->name : "-", (int) use_cuda_graph, (int) cuda_graph_update_required,
            (g_gt_prev_return ? (gt_t0 - g_gt_prev_return) : 0) / 1000.0, (gt_t1 - gt_t0) / 1000.0, (gt_t2 - gt_t1) / 1000.0);
    }
    g_gt_prev_return = ggml_time_us(); g_gt_after_compute = true;
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_cuda_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    CUDA_CHECK(cudaEventRecord((cudaEvent_t)event->context, cuda_ctx->stream()));
}

static void ggml_backend_cuda_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    if (ggml_backend_is_cuda(backend)) {
        strix_poll_dirty(cuda_ctx->stream());
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), (cudaEvent_t)event->context, 0));
    } else {
#if 0
        // untested
        auto wait_fn = [](void * user_data) {
            ggml_backend_event_t event = (ggml_backend_event_t)user_data;
            ggml_backend_event_synchronize(event);
        };

        CUDA_CHECK(cudaLaunchHostFunc(cuda_ctx->stream(), wait_fn, event));
#endif
        GGML_ABORT("fatal error");
    }
}

static void ggml_backend_cuda_graph_optimize(ggml_backend_t backend, ggml_cgraph * cgraph, ggml_backend_graph_optimize_params * params) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    const int64_t gt_t0 = ggml_time_us();
    {   // marks live for the whole scheduled graph (all splits): clear at the first optimize after a compute, or when the first split repeats
        const void * key = cgraph->n_nodes ? cgraph->nodes[0] : nullptr;
        if (g_gt_after_compute || g_gt_first_split == nullptr || key == g_gt_first_split) { ggml_cuda_mmb_marks_clear(); g_gt_first_split = key; g_gt_after_compute = false; }
    }
    {   // HC16 step 2: mark HC normalized-stream (xn) and gate tensors whose consumers all read the BF16 copies
        static const int hc16 = getenv("LLAMA_MMB_HC16") ? atoi(getenv("LLAMA_MMB_HC16")) : 0;
        if (hc16 >= 2 && GGML_CUDA_CC_IS_RDNA3_5(ggml_cuda_info().devices[cuda_ctx->device].cc)) {
            auto reads = [](const ggml_tensor * t, const ggml_tensor * x) { for (int s = 0; s < GGML_MAX_SRC && t->src[s]; ++s) if (t->src[s] == x || t->src[s]->view_src == x) return true; return false; };
            for (int i = 0; i < cgraph->n_nodes; ++i) {
                ggml_cuda_hc_combine_norm_args ca;
                const int skip = ggml_cuda_match_hc_combine_norm(cgraph, i, ggml_cuda_info().devices[cuda_ctx->device].warp_size, ca);
                if (skip <= 0) continue;   // structure only: data pointers are not assigned yet inside graph_optimize (alias check would always fail)
                const ggml_tensor * xn = ca.out_xn;
                if (ggml_nrows(xn) < 512) continue;
                bool ok = true; int nread = 0;
                for (int n = i + 1; n < cgraph->n_nodes && ok; ++n) {
                    const ggml_tensor * t = cgraph->nodes[n];
                    if (!reads(t, xn)) continue;
                    ++nread;
                    if (t->op == GGML_OP_MUL_MAT && ggml_cuda_mmb_supported_mm(t->src[0], t->src[1], t)) continue;
                    if (t->op == GGML_OP_MUL && n >= 1) { ggml_cuda_hc_mix_args ma; if (ggml_cuda_hc_mix_closed(cgraph, n - 1, ma) > 0 && (ma.xn == t->src[0] || ma.xn == t->src[1]) && (ma.xn == xn || ma.xn->view_src == xn)) continue; }
                    if (t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE) continue;
                    { static unsigned dbg = 0; if (getenv("LLAMA_HC16_DEBUG") && ggml_nrows(xn) >= 4096 && dbg++ < 12) fprintf(stderr, "HC16 xn=%s blocked by consumer %s(%s) src0=%s\n", xn->name, ggml_op_name(t->op), t->name, t->src[0] ? t->src[0]->name : "-"); }
                    ok = false;
                }
                { static unsigned dbg2 = 0; if (getenv("LLAMA_HC16_DEBUG") && ggml_nrows(xn) >= 4096 && dbg2++ < 12) fprintf(stderr, "HC16 xn=%s rows=%lld consumers=%d ok=%d\n", xn->name, (long long) ggml_nrows(xn), nread, (int) ok); }
                if (ok && nread > 0) ggml_cuda_mmb_mark_bf16_only(xn);
            }
            if (ggml_cuda_mmb_blk16()) {   // block_out (attention out-proj / MoE merge) is read only by the fused combine
                const int ws = ggml_cuda_info().devices[cuda_ctx->device].warp_size;
                std::vector<const ggml_tensor *> blks;
                for (int i = 0; i < cgraph->n_nodes; ++i) {
                    ggml_cuda_hc_combine_norm_args ca;
                    if (ggml_cuda_match_hc_combine_norm(cgraph, i, ws, ca) > 0)
                        blks.push_back(ca.block_out->view_src ? ca.block_out->view_src : ca.block_out);
                }
                for (const ggml_tensor * b : blks) {
                    if (ggml_nrows(b) < 512 || b->ne[0] % 8 != 0) continue;
                    // only mark when the producer is one that honours a BF16 mark
                    bool producer_ok = false;
                    if (b->op == GGML_OP_MUL_MAT && ggml_cuda_mmb_supported_mm(b->src[0], b->src[1], b)) producer_ok = true;
                    if (!producer_ok) {
                        for (int k = 0; k < cgraph->n_nodes && !producer_ok; ++k) {
                            ggml_cuda_moe_weighted_reduction_match mm;
                            if (cgraph->nodes[k]->op != GGML_OP_MUL) continue;
                            if (!ggml_cuda_match_moe_weighted_reduction(cgraph, k, mm)) continue;
                            if (mm.dst == b) producer_ok = true;                                  // reduction writes it
                            else if (k + mm.node_count < cgraph->n_nodes &&
                                     cgraph->nodes[k + mm.node_count] == b && b->op == GGML_OP_ADD &&
                                     ggml_node_has_n_uses(cgraph, k + mm.node_count - 1, 1) &&
                                     (b->src[0] == mm.dst || b->src[1] == mm.dst)) producer_ok = true;
                        }
                    }
                    if (!producer_ok) { static unsigned w = 0; if (getenv("LLAMA_HC16_DEBUG") && w++ < 6) fprintf(stderr, "HC_BLK16 %s not marked: producer %s does not honour BF16\n", b->name, ggml_op_name(b->op)); continue; }
                    int bi = -1;
                    for (int k = 0; k < cgraph->n_nodes; ++k) if (cgraph->nodes[k] == b) { bi = k; break; }
                    if (bi < 0) continue;
                    bool ok = true; int nread = 0;
                    for (int n = bi + 1; n < cgraph->n_nodes && ok; ++n) {
                        const ggml_tensor * t = cgraph->nodes[n];
                        if (!reads(t, b)) continue;
                        ++nread;
                        if (t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_REPEAT) continue;
                        if (t->op == GGML_OP_MUL || t->op == GGML_OP_ADD) continue;
                        { static unsigned w = 0; if (getenv("LLAMA_HC16_DEBUG") && w++ < 6) fprintf(stderr, "HC_BLK16 %s blocked by %s(%s)\n", b->name, ggml_op_name(t->op), t->name); }
                        ok = false;
                    }
                    if (ok && nread > 0) ggml_cuda_mmb_mark_bf16_only(b);
                }
            }

            if (ggml_cuda_mmb_res16()) {   // residual stream: the only readers are the fused norm and the next combine
                const int ws = ggml_cuda_info().devices[cuda_ctx->device].warp_size;
                std::vector<std::pair<const ggml_tensor *, const ggml_tensor *>> comb;
                for (int i = 0; i < cgraph->n_nodes; ++i) {
                    ggml_cuda_hc_combine_norm_args ca;
                    if (ggml_cuda_match_hc_combine_norm(cgraph, i, ws, ca) > 0) comb.emplace_back(ca.residual, ca.out_res);
                }
                { static unsigned t = 0; if (getenv("LLAMA_HC16_DEBUG") && t == 0) { t = 1;
                    for (size_t z = 0; z < comb.size() && z < 6; ++z) {
                        const ggml_tensor * dr = comb[z].first->view_src ? comb[z].first->view_src : comb[z].first;
                        fprintf(stderr, "HC_RES16 chain[%zu]: residual=%s(root %s) -> out_res=%s\n", z, comb[z].first->name, dr->name, comb[z].second->name);
                    } } }
                for (const auto & c : comb) {
                    const ggml_tensor * r = c.second;
                    if (ggml_nrows(r) < 512) continue;
                    int ri = -1;
                    for (int k = 0; k < cgraph->n_nodes; ++k) if (cgraph->nodes[k] == r) { ri = k; break; }
                    if (ri < 0) continue;
                    bool ok = true; int nread = 0;
                    for (int n = ri + 1; n < cgraph->n_nodes && ok; ++n) {
                        const ggml_tensor * t = cgraph->nodes[n];
                        if (!reads(t, r)) continue;
                        ++nread;
                        if (t->op == GGML_OP_RMS_NORM) continue;
                        if (t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE) continue;
                        bool next_combine = false;
                        for (const auto & d : comb) {
                            const ggml_tensor * dr = d.first->view_src ? d.first->view_src : d.first;
                            if (dr == r && d.second == t) { next_combine = true; break; }
                        }
                        if (next_combine) continue;
                        { static unsigned w = 0; if (getenv("LLAMA_HC16_DEBUG") && w++ < 6) fprintf(stderr, "HC_RES16 %s blocked by %s(%s)\n", r->name, ggml_op_name(t->op), t->name); }
                        ok = false;
                    }
                    if (ok && nread > 0) ggml_cuda_mmb_mark_bf16_only(r);
                }
            }

            for (int i = 0; i < cgraph->n_nodes; ++i) {
                ggml_cuda_hc_mix_args ma;
                if (ggml_cuda_hc_mix_closed(cgraph, i, ma) <= 0 || !ma.dst) continue;
                const ggml_tensor * d = ma.dst;
                if (ggml_nrows(d) < 512) continue;
                bool ok = true; int nread = 0;
                for (int n = i + 1; n < cgraph->n_nodes && ok; ++n) {
                    const ggml_tensor * t = cgraph->nodes[n];
                    if (!reads(t, d)) continue;
                    ++nread;
                    if (t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE) continue;
                    if (t->op == GGML_OP_MUL_MAT && ggml_cuda_mmb_supported_mm(t->src[0], t->src[1], t)) continue;
                    ok = false;
                }
                if (ok && nread > 0) ggml_cuda_mmb_mark_bf16_only(d);
            }

            for (int i = 0; i < cgraph->n_nodes; ++i) {
                const ggml_tensor * t = cgraph->nodes[i];
                if (t->op != GGML_OP_MUL_MAT || t->src[0]->ne[0] != 320 || t->src[0]->ne[1] != 10240 || !ggml_cuda_mmb_supported_mm(t->src[0], t->src[1], t)) continue;
                bool ok = true; int nread = 0;
                for (int n = i + 1; n < cgraph->n_nodes && ok; ++n) {
                    const ggml_tensor * u = cgraph->nodes[n];
                    if (!reads(u, t)) continue;
                    ++nread;
                    if (u->op == GGML_OP_UNARY && ggml_get_unary_op(u) == GGML_UNARY_OP_SIGMOID) { ggml_cuda_hc_mix_args ma; if (ggml_cuda_hc_mix_closed(cgraph, n, ma) > 0 && ma.gate == t) continue; }
                    ok = false;
                }
                if (ok && nread > 0) ggml_cuda_mmb_mark_bf16_only(t);
            }
        }
    }

    {
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            const ggml_tensor * t = cgraph->nodes[i];
            if (t->op == GGML_OP_MUL_MAT && (t->src[0]->type == GGML_TYPE_IQ4_NL || t->src[0]->type == GGML_TYPE_Q6_K) && ggml_cuda_mmb_supported_mm(t->src[0], t->src[1], t)) ggml_cuda_mmb_shadow_prepare(*cuda_ctx, t->src[0]);
            if (t->op == GGML_OP_MUL_MAT && t->src[0]->type == GGML_TYPE_F32 && t->src[0]->ne[1] <= 64 && t->src[1]->ne[1] > 64) ggml_cuda_mmb_f16w_prepare(*cuda_ctx, t->src[0]);
        }
    }
    if (ggml_cuda_mmb_down16()) {
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            ggml_cuda_moe_weighted_reduction_match match;
            if (!ggml_cuda_match_moe_weighted_reduction(cgraph, i, match)) continue;
            const ggml_tensor * ex = match.experts;
            int xi = -1;
            for (int k = 0; k < i; ++k) { if (cgraph->nodes[k] == ex) { xi = k; break; } }
            if (xi < 0 || ex->op != GGML_OP_MUL_MAT_ID || ex->type != GGML_TYPE_F32) continue;
            if (!ggml_cuda_mmb_supported_mmid(ex->src[0], ex->src[1], ex->src[2], const_cast<ggml_tensor *>(ex))) continue;
            if (!ggml_node_has_n_uses(cgraph, xi, 1)) continue;
            if (ex->ne[0] % 8 != 0 || ggml_nrows(ex) < 512) continue;
            ggml_cuda_mmb_mark_bf16_only(ex);
        }
    }

    {
        auto reads = [](const ggml_tensor * t, const ggml_tensor * x) { for (int s = 0; s < GGML_MAX_SRC && t->src[s]; ++s) if (t->src[s] == x || t->src[s]->view_src == x) return true; return false; };
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            const ggml_tensor * glu = cgraph->nodes[i];
            if (glu->op != GGML_OP_GLU || !glu->src[0] || !glu->src[1] || glu->src[0]->op != GGML_OP_MUL_MAT_ID) continue;
            if (!ggml_cuda_mmb_supported_glu(glu->src[0]->src[0], glu->src[1]->src[0], glu->src[0]->src[1], glu->src[0]->src[2], glu)) continue;
            bool ok = true; int nread = 0;
            for (int n = i + 1; n < cgraph->n_nodes && ok; ++n) {
                const ggml_tensor * t = cgraph->nodes[n];
                if (!reads(t, glu)) continue;
                ++nread;
                if (t->op == GGML_OP_MUL_MAT_ID && t->src[1] == glu && ggml_cuda_mmb_supported_mmid(t->src[0], t->src[1], t->src[2], t)) continue;
                ok = false;
            }
            if (ok && nread > 0) ggml_cuda_mmb_mark_bf16_only(glu);
        }
    }

    if (ggml_cuda_mmb_res16()) {   // 16-bit residual stream: force producer and consumer onto different buffers
        const int ws = ggml_cuda_info().devices[cuda_ctx->device].warp_size;
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            ggml_cuda_hc_combine_norm_args ca;
            if (ggml_cuda_match_hc_combine_norm(cgraph, i, ws, ca) <= 0) continue;
            ggml_tensor * rr = const_cast<ggml_tensor *>(ca.residual->view_src ? ca.residual->view_src : ca.residual);
            params->add_alloc_dep(params->user_data, rr, ca.out_res);
            params->add_alloc_dep(params->user_data, rr, ca.out_xn);
        }
    }

    {   // strixllama: the QSA pack and the attention gate's copy + sigmoid read their source while they write
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            if (cgraph->nodes[i]->op != GGML_OP_CONT) continue;
            ggml_tensor * src_root = nullptr, * out = nullptr;
            if (ggml_cuda_qsa_pack_deps(cgraph, i, &src_root, &out) || ggml_cuda_cont_sigmoid_deps(cgraph, i, &src_root, &out)) {
                params->add_alloc_dep(params->user_data, src_root, out);
            }
        }
    }

    {   // strixllama: the decode strip's fused score reads the rows, the queries and the limits at the MUL_MAT, after the rows'
        // GET_ROWS would have: all three stay allocated until the result is written
        auto root = [](const ggml_tensor * t) { return const_cast<ggml_tensor *>(t->view_src ? t->view_src : t); };
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            if (cgraph->nodes[i]->op != GGML_OP_GET_ROWS) continue;
            ggml_cuda_idx_dec_match m;
            if (ggml_cuda_match_idx_score_dec(cgraph, i, m)) {
                // the limits' starts view: five nodes before the end in the one-sequence chain, twelve with the membership
                const ggml_tensor * sv = cgraph->nodes[m.end - (m.sbv ? 12 : 5)];
                params->add_alloc_dep(params->user_data, root(m.a.rows), m.a.out);
                params->add_alloc_dep(params->user_data, root(m.a.q), m.a.out);
                params->add_alloc_dep(params->user_data, root(sv), m.a.out);
                if (m.sbv) {
                    params->add_alloc_dep(params->user_data, root(m.sbv), m.a.out);
                    params->add_alloc_dep(params->user_data, root(m.stv), m.a.out);
                }
            }
        }
    }

    {   // strixllama: the fused indexer score reads the keys and the queries while it writes the strip's result
        auto root = [](const ggml_tensor * t) { return const_cast<ggml_tensor *>(t->view_src ? t->view_src : t); };
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            if (cgraph->nodes[i]->op != GGML_OP_MUL_MAT) continue;
            ggml_cuda_idx_score_args sa;
            if (ggml_cuda_match_idx_score(cgraph, i, sa) > 0) {
                params->add_alloc_dep(params->user_data, root(sa.keys), sa.out);
                params->add_alloc_dep(params->user_data, root(sa.q), sa.out);
            }
        }
    }

    {
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            if (cgraph->nodes[i]->op != GGML_OP_UNARY) continue;
            ggml_cuda_idx_relu_sum_args ia;
            if (ggml_cuda_match_idx_relu_sum(cgraph, i, ia) > 0 && ia.score && ia.dst) {
                ggml_tensor * root = ia.score->view_src ? ia.score->view_src : const_cast<ggml_tensor *>(ia.score);
                params->add_alloc_dep(params->user_data, root, ia.dst);
            }
        }
    }

    {
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            if (cgraph->nodes[i]->op != GGML_OP_RMS_NORM) continue;
            ggml_cuda_norm_gated_match nm;
            const int sk = ggml_cuda_norm_gated_match_at(cgraph, i, nm);
            if (sk > 0 && nm.pre >= 0) {
                params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(nm.x), nm.dst);
                params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(nm.w), nm.dst);
                params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(nm.z->view_src ? nm.z->view_src : nm.z), nm.dst);
            }
        }
    }

    if (graph_timing_enabled()) {
        const ggml_tensor * f = cgraph->n_nodes ? cgraph->nodes[0] : nullptr;
        fprintf(stderr, "GT optimize n=%d first=%s(%lld) took=%.1fms marks=%zu\n", cgraph->n_nodes, f ? f->name : "-", f ? (long long) ggml_nrows(f) : 0, (ggml_time_us() - gt_t0) / 1000.0, ggml_cuda_mmb_marks_count());
    }
    const bool disable_fusion = ggml_cuda_fusion_disabled();

    if (!disable_fusion) {
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            if (GGML_CUDA_CC_IS_RDNA3_5(ggml_cuda_info().devices[cuda_ctx->device].cc)) {
                ggml_cuda_hc_mix_args args;
                const int count=ggml_cuda_hc_mix_closed(cgraph,i,args);
                if (count>0) {
                    params->add_alloc_dep(params->user_data,const_cast<ggml_tensor *>(args.xn),args.dst);
                    params->add_alloc_dep(params->user_data,const_cast<ggml_tensor *>(args.gate),args.dst);
                    i+=count-1;
                    continue;
                }
            }
            if (GGML_CUDA_CC_IS_RDNA3_5(ggml_cuda_info().devices[cuda_ctx->device].cc)) {
                ggml_cuda_qsa_expand_args args;
                const int skip = ggml_cuda_match_qsa_expand(cgraph,i,args);
                if (skip) {
                    for (const auto * input : {args.blocks,args.scores,args.cells,args.tail,
                            (const ggml_tensor *)args.cast_blocks,(const ggml_tensor *)args.cast_cells,(const ggml_tensor *)args.expanded}) {
                        params->add_alloc_dep(params->user_data,const_cast<ggml_tensor *>(input),args.output);
                    }
                    i += skip;
                    continue;
                }
            }
            if (GGML_CUDA_CC_IS_RDNA3_5(ggml_cuda_info().devices[cuda_ctx->device].cc)) {
                ggml_cuda_hc_combine_norm_args args;
                const int skip = ggml_cuda_match_hc_combine_norm(
                    cgraph, i, ggml_cuda_info().devices[cuda_ctx->device].warp_size, args);
                if (skip > 0) {
                    params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(args.block_out), args.out_xn);
                    params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(args.inject), args.out_xn);
                    i += skip;
                    continue;
                }
            }

            if (GGML_CUDA_CC_IS_RDNA3_5(ggml_cuda_info().devices[cuda_ctx->device].cc)) {
                ggml_cuda_get_rows_mean4_match pool;
                if (ggml_cuda_match_get_rows_mean4(cgraph, i, pool)) {
                    params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(pool.src), pool.dst);
                    params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(pool.ids), pool.dst);
                    i += pool.node_count - 1;
                    continue;
                }
            }
            ggml_cuda_topk_moe_match topk;
            if (ggml_cuda_match_topk_moe(cgraph, i, topk)) {
                if (ggml_nrows(cgraph->nodes[i]) > TOPK_MOE_ROWS_PER_BLOCK) {
                    params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(topk.logits), topk.ids);
                    params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(topk.logits), topk.weights);
                }
                if (topk.bias != nullptr) {
                    params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(topk.bias), topk.ids);
                    params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(topk.bias), topk.weights);
                }
                i += topk.node_count - 1;
                continue;
            }
            if (cgraph->nodes[i]->op == GGML_OP_CONCAT) {
                ggml_cuda_ple_conv_match pm;
                if (ggml_cuda_ple_conv_match_at_concat(cgraph, i, pm)) {
                    params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(pm.x), pm.out);
                    params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(pm.state), pm.out);
                    continue;
                }
                ggml_cuda_gdn_conv_match gm;
                if (ggml_cuda_gdn_conv_match_at_concat(cgraph, i, gm)) {
                    params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(gm.x), cgraph->nodes[gm.conv_idx]);
                    params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(gm.state), cgraph->nodes[gm.conv_idx]);
                }
                continue;
            }
        }
        if (ggml_cuda_shexp_fuse()) {
            // add alloc deps for performance positive fusions. This may increase the overall compute buffer size.
            ggml_cuda_set_device(cuda_ctx->device);
            for (int i = 0; i + 5 < cgraph->n_nodes; ++i) {
                if (cgraph->nodes[i]->op != GGML_OP_MUL_MAT_ID) {
                    continue;
                }
                for (int j = i + 3; j + 2 < cgraph->n_nodes; ++j) {
                    if (cgraph->nodes[j]->op == GGML_OP_MUL_MAT_ID && cgraph->nodes[j + 1]->op == GGML_OP_MUL_MAT_ID) {
                        break;
                    }
                    if (cgraph->nodes[j]->op != GGML_OP_MUL_MAT || !ggml_cuda_match_shared_expert(cgraph, i, j)) {
                        continue;
                    }
                    // Group both outputs before allocation so the shared result cannot alias intervening nodes.
                    std::rotate(cgraph->nodes + i + 3, cgraph->nodes + j, cgraph->nodes + j + 3);
                    ggml_tensor * up = cgraph->nodes[i + 2]->src[1];
                    params->add_alloc_dep(params->user_data, up->src[1], cgraph->nodes[i + 5]);
                    params->add_alloc_dep(params->user_data, up->src[2], cgraph->nodes[i + 5]);
                    i += 5;
                    break;
                }
            }
        }
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            if (cgraph->nodes[i]->op != GGML_OP_MUL) {
                continue;
            }

            ggml_cuda_moe_weighted_reduction_match match;
            if (!ggml_cuda_match_moe_weighted_reduction(cgraph, i, match)) {
                continue;
            }

            params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(match.experts), match.dst);
            params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(match.weights), match.dst);
            if (match.expert_scale != nullptr) {
                params->add_alloc_dep(
                    params->user_data, const_cast<ggml_tensor *>(match.expert_scale), match.dst);
            }
            i += match.node_count - 1;
        }
    }

#ifdef USE_CUDA_GRAPH
    const void * graph_key = ggml_cuda_graph_get_key(cgraph);
    const bool use_cuda_graph = ggml_cuda_graph_set_enabled(cuda_ctx, graph_key);
#else
    const bool use_cuda_graph = false;
    GGML_UNUSED(cuda_ctx);
    GGML_UNUSED(cgraph);
#endif

    static bool enable_graph_optimization = [] {
        const char * env     = getenv("GGML_CUDA_GRAPH_OPT");
        return env != nullptr && atoi(env) == 1;
    }();

    if (!enable_graph_optimization) {
        return;
    }

    ggml_cuda_stream_context & stream_context = cuda_ctx->stream_context();
    stream_context.reset();

    if (!use_cuda_graph) {
        return;
    }

    ggml_cuda_set_device(cuda_ctx->device);

    // number of out-degrees for a particular node
    std::unordered_map<const ggml_tensor *, int> fan_out;
    // reverse mapping of node to index in the cgraph
    std::unordered_map<const ggml_tensor *, int> node_indices;

    const auto & is_noop = [](const ggml_tensor * node) -> bool {
        return ggml_is_empty(node) || node->op == GGML_OP_NONE || node->op == GGML_OP_RESHAPE ||
               node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE;
    };

    const auto & depends_on = [](const ggml_tensor * dst, const ggml_tensor * src) -> bool {
        for (uint32_t s = 0; s < GGML_MAX_SRC; ++s) {
            if (dst->src[s] == src) {
                return true;
            }
        }
        // implicit dependency if they view the same tensor
        const ggml_tensor * dst2 = dst->view_src ? dst->view_src : dst;
        const ggml_tensor * src2 = src->view_src ? src->view_src : src;
        if (dst2 == src2) {
            return true;
        }
        return false;
    };

    for (int node_idx = 0; node_idx < cgraph->n_nodes; node_idx++) {
        const ggml_tensor * node = cgraph->nodes[node_idx];
        node_indices[node]       = node_idx;

        if (is_noop(node)) {
            continue;
        }
        for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
            const ggml_tensor * src = cgraph->nodes[node_idx]->src[src_idx];
            //TODO: check why nrows > 1 fails
            if (node && !is_noop(node) && ggml_nrows(node) <= 1) {
                fan_out[src] += 1;
            }
        }
    }

    // Target Q, K, V for concurrency
    // this is a more general way to find nodes which can be candidates for concurrency (although it has not been tested for anything else):
    // 1. find fan-out (fork) nodes where the same input is used at least N times (in QKV, it would be "attn-norm")
    // 2. find the join node, where 2 or more of the outputs are required (in QKV, this would "KQ" or "flash-attn")
    // 3. account for all branches from the fork to the join
    // 4. To extend lifetimes of the tensors, we interleave the branches (see below for more details)
    // 5. save the original cgraph and restore it in graph_compute, to enable fusion within streams
    // See discussion: https://github.com/ggml-org/llama.cpp/pull/16991#issuecomment-3522620030

    const int min_fan_out = 3;
    const int max_fan_out = 3;

    // store {fork_idx, join_idx}
    std::vector<std::pair<int, int>> concurrent_node_ranges;

    for (const auto & [root_node, count] : fan_out) {
        if (count >= min_fan_out && count <= max_fan_out) {
            const int root_node_idx = node_indices[root_node];

            // only optimize for attn_norm
            // TODO: make this more generic
            if (!strstr(root_node->name, "attn_norm")) {
                continue;
            }

            bool is_part_of_event = false;
            for (const auto & [start, end] : concurrent_node_ranges) {
                if (root_node_idx >= start && root_node_idx <= end) {
                    is_part_of_event = true;
                }
            }

            if (is_part_of_event) {
                continue;
            }

            std::vector<std::vector<const ggml_tensor *>> nodes_per_branch;
            for (int i = root_node_idx + 1; i < cgraph->n_nodes; ++i) {
                const ggml_tensor * node = cgraph->nodes[i];
                if (!is_noop(node) && depends_on(node, root_node)) {
                    nodes_per_branch.push_back({ node });
                }
            }

            GGML_ASSERT(nodes_per_branch.size() == (size_t) count);

            //find the join point
            const ggml_tensor * join_node = nullptr;

            const auto & belongs_to_branch = [&](const ggml_tensor *                      node,
                                                 const std::vector<const ggml_tensor *> & branch) -> bool {
                for (const ggml_tensor * n : branch) {
                    if (depends_on(node, n)) {
                        return true;
                    }
                }
                return false;
            };

            for (int i = root_node_idx + 1; i < cgraph->n_nodes; ++i) {
                const ggml_tensor * curr_node = cgraph->nodes[i];

                int num_joins = 0;
                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    if (belongs_to_branch(curr_node, nodes_per_branch[branch_idx])) {
                        num_joins++;
                    }
                }

                if (num_joins >= 2) {
                    join_node = curr_node;
                    break;
                }

                bool found_branch = false;
                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    std::vector<const ggml_tensor *> & branch_vec = nodes_per_branch[branch_idx];
                    if (belongs_to_branch(curr_node, branch_vec)) {
                        //continue accumulating
                        if (std::find(branch_vec.begin(), branch_vec.end(), curr_node) == branch_vec.end()) {
                            branch_vec.push_back(curr_node);
                        }
                        found_branch = true;
                    }
                }

                if (!found_branch && is_noop(curr_node)) {
                    // we can put it in any branch because it will be ignored
                    nodes_per_branch[0].push_back({ curr_node });
                }
            }

            if (join_node) {
                //Create ggml_cuda_concurrent_event
                ggml_cuda_concurrent_event concurrent_event(nodes_per_branch.size());
                concurrent_event.join_node = join_node;

                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    for (const ggml_tensor * n : nodes_per_branch[branch_idx]) {
                        concurrent_event.stream_mapping[n] = branch_idx + 1;
                    }
                }

                int fork_node_idx = node_indices[root_node];
                int join_node_idx = node_indices[join_node];

                int       current_branch_idx = 0;
                int       current_node_idx   = fork_node_idx + 1;
                const int n_branches         = nodes_per_branch.size();

                int total_branch_nodes = 0;
                for (std::vector<const ggml_tensor *> branch_nodes : nodes_per_branch) {
                    total_branch_nodes += branch_nodes.size();
                }

                // there are other nodes in the middle which are unaccounted for
                // usually (cpy) nodes, then ignore this fork
                if (join_node_idx - fork_node_idx - 1 != total_branch_nodes) {
                    GGML_LOG_DEBUG(
                        "Skipping %s because the number of nodes in the middle is not equal to the total number of "
                        "branch nodes %d != %d\n",
                        root_node->name, join_node_idx - fork_node_idx - 1, total_branch_nodes);
                    continue;
                }

                // Save the original order of nodes in this region before interleaving
                // This is used later to restore grouping for fusion within streams
                concurrent_event.original_order.reserve(total_branch_nodes);
                for (int i = fork_node_idx + 1; i < join_node_idx; ++i) {
                    concurrent_event.original_order.push_back(cgraph->nodes[i]);
                }

                std::unordered_map<const ggml_tensor *, ggml_cuda_concurrent_event> & concurrent_events = cuda_ctx->stream_context().concurrent_events;
                GGML_ASSERT(concurrent_events.find(root_node) == concurrent_events.end());
                concurrent_events.emplace(root_node, std::move(concurrent_event));
                GGML_LOG_DEBUG("Adding stream at node %s %p\n", root_node->name, root_node);
                concurrent_node_ranges.emplace_back(fork_node_idx, join_node_idx);

                // interleave tensors to extend lifetimes so that ggml graph doesn't recycle them
                // example transformation:
                // [attn-norm, QMul, QNorm, QRope, KMul, KNorm, KRope, VMul, attn] ->
                // [attn-norm, QMul, KMul, VMul, QNorm, VNorm, QRope, KRope, attn]
                while (current_node_idx < join_node_idx) {
                    std::vector<const ggml_tensor *> & branch_nodes = nodes_per_branch[current_branch_idx];

                    bool has_node = false;
                    for (std::vector<const ggml_tensor *> branch_node : nodes_per_branch) {
                        has_node |= branch_node.size() > 0;
                    }

                    GGML_ASSERT(has_node);

                    if (branch_nodes.empty()) {
                        current_branch_idx = (current_branch_idx + 1) % n_branches;
                        continue;
                    }

                    cgraph->nodes[current_node_idx] = const_cast<ggml_tensor *>(branch_nodes.front());
                    current_node_idx++;
                    branch_nodes.erase(branch_nodes.begin());

                    // append all empty nodes
                    while (!branch_nodes.empty() && is_noop(branch_nodes.front())) {
                        cgraph->nodes[current_node_idx] = const_cast<ggml_tensor *>(branch_nodes.front());
                        current_node_idx++;
                        branch_nodes.erase(branch_nodes.begin());
                    }

                    current_branch_idx = (current_branch_idx + 1) % n_branches;
                }
            }
        }
    }
}

static const ggml_backend_i ggml_backend_cuda_interface = {
    /* .get_name                = */ ggml_backend_cuda_get_name,
    /* .free                    = */ ggml_backend_cuda_free,
    /* .set_tensor_async        = */ ggml_backend_cuda_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_cuda_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_cuda_set_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_cuda_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ ggml_backend_cuda_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_cuda_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_cuda_graph_compute,
    /* .event_record            = */ ggml_backend_cuda_event_record,
    /* .event_wait              = */ ggml_backend_cuda_event_wait,
    /* .graph_optimize          = */ ggml_backend_cuda_graph_optimize,
};

static ggml_guid_t ggml_backend_cuda_guid() {
    static ggml_guid guid = { 0x2c, 0xdd, 0xe8, 0x1c, 0x65, 0xb3, 0x65, 0x73, 0x6a, 0x12, 0x88, 0x61, 0x1c, 0xc9, 0xdc, 0x25 };
    return &guid;
}

bool ggml_backend_is_cuda(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_cuda_guid());
}

int ggml_backend_cuda_get_device_count() {
    return ggml_cuda_info().device_count;
}

static std::string ggml_cuda_device_description(int device) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, ggml_cuda_get_physical_device(device)));

    const ggml_cuda_device_info & info = ggml_cuda_info();
    std::string description = prop.name;
    if (info.device_count > info.physical_device_count) {
        description += " (dev p" + std::to_string(info.devices[device].physical_device) +
                       "/v" + std::to_string(info.devices[device].virtual_index) + ")";
    }
    return description;
}

void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size) {
    snprintf(description, description_size, "%s", ggml_cuda_device_description(device).c_str());
}

static int ggml_cuda_physical_device_share_count(int device) {
    const ggml_cuda_device_info & info = ggml_cuda_info();
    GGML_ASSERT(device >= 0 && device < info.device_count);
    return info.devices[device].physical_share_count;
}

void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total) {
    ggml_cuda_set_device(device);

    CUDA_CHECK(cudaMemGetInfo(free, total));

    // virtual devices sharing one physical GPU share its memory pool; split it between them
    const int share_count = ggml_cuda_physical_device_share_count(device);
    *free  /= share_count;
    *total /= share_count;
}

bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size) {
    if (!ggml_cuda_register_host_enabled()) {
        return false;
    }

#if CUDART_VERSION >= 11010 || defined(GGML_USE_MUSA) || defined(GGML_USE_HIP)
    cudaError_t err = cudaHostRegister(buffer, size, cudaHostRegisterPortable | cudaHostRegisterReadOnly);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();

        GGML_LOG_DEBUG("%s: failed to register %.2f MiB of pinned memory: %s\n", __func__,
                           size / 1024.0 / 1024.0, cudaGetErrorString(err));
        return false;
    }
    return true;
#else
    GGML_UNUSED(buffer);
    GGML_UNUSED(size);
    return false;
#endif // CUDART_VERSION >= 11010 || defined(GGML_USE_MUSA)
}

void ggml_backend_cuda_unregister_host_buffer(void * buffer) {
    if (!ggml_cuda_register_host_enabled()) {
        return;
    }

    cudaError_t err = cudaHostUnregister(buffer);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
    }
}

// backend device

struct ggml_backend_cuda_device_context {
    int device;
    std::string name;
    std::string description;
    std::string pci_bus_id;
    int op_offload_min_batch_size;
};

static const char * ggml_backend_cuda_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ctx->name.c_str();
}

static const char * ggml_backend_cuda_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ctx->description.c_str();
}

#if defined(__linux__)
// Helper function to get available memory from /proc/meminfo for UMA systems
static bool ggml_backend_cuda_get_available_uma_memory(long * available_memory_kb, long * free_swap_kb) {
    FILE * meminfo_file = nullptr;
    // 2KB buffer for reading /proc/meminfo since it does not report size info, should be enough
    const size_t BUFFER_SIZE = 2048;
    auto file_buffer = std::make_unique<char[]>(BUFFER_SIZE);
    size_t bytes_read = 0;
    long huge_tlb_total_pages = -1;
    long huge_tlb_free_pages = -1;
    long huge_tlb_page_size = -1;

    if (available_memory_kb == nullptr || free_swap_kb == nullptr) {
        return false;
    }

    meminfo_file = fopen("/proc/meminfo", "r");
    if (meminfo_file == nullptr) {
        GGML_LOG_ERROR("%s: failed to open /proc/meminfo\n", __func__);
        return false;
    }

    // Read file into buffer
    bytes_read = fread(file_buffer.get(), 1, BUFFER_SIZE - 1, meminfo_file);
    fclose(meminfo_file);

    if (bytes_read == 0) {
        GGML_LOG_ERROR("%s: failed to read from /proc/meminfo\n", __func__);
        return false;
    }
    file_buffer[bytes_read] = '\0';

    *available_memory_kb = -1;
    *free_swap_kb = -1;

    // Parse the file buffer line by line
    char * line = file_buffer.get();
    char * line_next;
    while (line < file_buffer.get() + bytes_read) {
        // Find the end of the current line
        line_next = strchr(line, '\n');
        if (line_next != nullptr) {
            *line_next = '\0';
            line_next++;
        } else {
            line_next = file_buffer.get() + bytes_read;
        }

        long value;
        if (sscanf(line, "MemAvailable: %ld kB", &value) == 1) {
            *available_memory_kb = value;
        } else if (sscanf(line, "SwapFree: %ld kB", &value) == 1) {
            *free_swap_kb = value;
        } else if (sscanf(line, "HugePages_Total: %ld", &value) == 1) {
            huge_tlb_total_pages = value;
        } else if (sscanf(line, "HugePages_Free: %ld", &value) == 1) {
            huge_tlb_free_pages = value;
        } else if (sscanf(line, "Hugepagesize: %ld kB", &value) == 1) {
            huge_tlb_page_size = value;
        }

        line = line_next;
    }

    if (huge_tlb_total_pages != 0 && huge_tlb_total_pages != -1) {
        *available_memory_kb = huge_tlb_free_pages * huge_tlb_page_size;

        // Hugetlbfs pages are not swappable.
        *free_swap_kb = 0;
    }

    GGML_LOG_DEBUG("%s: final available_memory_kb: %ld\n", __func__, *available_memory_kb);
    return true;
}
#endif // defined(__linux__)

static void ggml_backend_cuda_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    ggml_cuda_set_device(ctx->device);
    cudaError_t err = cudaMemGetInfo(free, total);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        GGML_LOG_WARN("%s: cudaMemGetInfo failed (%s), returning 0/0\n", __func__, cudaGetErrorString(err));
        *free = 0;
        *total = 0;
        return;
    }

// ref: https://github.com/ggml-org/llama.cpp/pull/17368
#if defined(__linux__) && !defined(GGML_USE_HIP)
    // Check if this is a UMA (Unified Memory Architecture) system
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, ggml_cuda_get_physical_device(ctx->device)));

    // Check if UMA is explicitly enabled via environment variable
    bool is_uma = prop.integrated > 0 || ggml_cuda_unified_memory_enabled();

    if (is_uma) {
        // For UMA systems (like DGX Spark), use system memory info
        long available_memory_kb = 0;
        long free_swap_kb = 0;

        if (ggml_backend_cuda_get_available_uma_memory(&available_memory_kb, &free_swap_kb) && available_memory_kb > 0) {
            *free = (size_t)available_memory_kb * 1024;
        } else {
            GGML_LOG_ERROR("%s: /proc/meminfo reading failed, using cudaMemGetInfo\n", __func__);
        }
    }
#endif // defined(__linux__) && !defined(GGML_USE_HIP)

    // virtual devices sharing one physical GPU share its memory pool; split it between them
    const int share_count = ggml_cuda_physical_device_share_count(ctx->device);
    *free  /= share_count;
    *total /= share_count;
}

static enum ggml_backend_dev_type ggml_backend_cuda_device_get_type(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *) dev->context;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, ggml_cuda_get_physical_device(ctx->device)));

    return prop.integrated
        ? GGML_BACKEND_DEVICE_TYPE_IGPU
        : GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_cuda_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;

    props->name        = ggml_backend_cuda_device_get_name(dev);
    props->description = ggml_backend_cuda_device_get_description(dev);
    props->type        = ggml_backend_cuda_device_get_type(dev);
    props->device_id   = ctx->pci_bus_id.empty() ? nullptr : ctx->pci_bus_id.c_str();
    ggml_backend_cuda_device_get_memory(dev, &props->memory_free, &props->memory_total);

    const bool host_buffer = !ggml_cuda_pinned_memory_disabled();
#ifdef GGML_CUDA_NO_PEER_COPY
    bool events = false;
#else
    bool events = true;
#endif

    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ host_buffer,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ events,
        /* .mmap_support          = */ props->type != GGML_BACKEND_DEVICE_TYPE_IGPU,
    };
}

static ggml_backend_t ggml_backend_cuda_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ggml_backend_cuda_init(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_cuda_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ggml_backend_cuda_buffer_type(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_cuda_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_cuda_host_buffer_type();
}

// TODO: move these functions here
static bool ggml_backend_cuda_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;

    // check if all the sources are allocated on this device
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (op->src[i] && op->src[i]->buffer && ggml_backend_buft_is_cuda(op->src[i]->buffer->buft)) {
            ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)op->src[i]->buffer->buft->context;
            if (buft_ctx->device != dev_ctx->device) {
                return false;
            }
        }
    }

    switch (op->op) {
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_EXPM1:
                case GGML_UNARY_OP_SOFTPLUS:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_XIELU:
                case GGML_UNARY_OP_FLOOR:
                case GGML_UNARY_OP_CEIL:
                case GGML_UNARY_OP_ROUND:
                case GGML_UNARY_OP_TRUNC:
                    if (op->src[0]->type == GGML_TYPE_BF16 && ggml_get_unary_op(op) == GGML_UNARY_OP_XIELU) {
                        return false;
                    }
                    return op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_BF16;
                default:
                    return false;
            }
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    if (op->src[0]->type == GGML_TYPE_BF16 &&
                            (ggml_get_glu_op(op) == GGML_GLU_OP_SWIGLU_OAI || ggml_get_glu_op(op) == GGML_GLU_OP_SWIGLU_CLAMP)) {
                        return false;
                    }
                    if (op->src[0]->type == GGML_TYPE_F16 && ggml_get_glu_op(op) == GGML_GLU_OP_SWIGLU_OAI) {
                        return false;
                    }
                    return ggml_is_contiguous_1(op->src[0]);
                default:
                    return false;
            }
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
            {
                struct ggml_tensor * a = op->src[0];
                struct ggml_tensor * b = op->src[1];
                if (a->nb[0] != ggml_element_size(a) || b->nb[0] != ggml_element_size(b)) {
                    return false; // TODO this could in principle be implemented though currently there is no use case.
                }
                if (b->type == GGML_TYPE_F16 && a->type != GGML_TYPE_F16 && !ggml_cuda_op_mul_mat_use_fwht(op)) {
                    return false;
                }
                if (op->op == GGML_OP_MUL_MAT_ID && ggml_get_op_params_i32(op, 3) == GGML_PREC_F32) {
                    return false;
                }
#ifdef GGML_USE_MUSA
                const int cc = ggml_cuda_info().devices[dev_ctx->device].cc;
                if (b->ne[2]*b->ne[3] > 1 && !ggml_is_transposed(a) && !ggml_is_transposed(b)) {
                    if (GGML_CUDA_CC_IS_QY1(cc) && op->op == GGML_OP_MUL_MAT &&
                            a->type == GGML_TYPE_F16 && b->type == GGML_TYPE_F16) {
                        return false;
                    }
                    if (GGML_CUDA_CC_IS_QY2(cc) && op->op == GGML_OP_MUL_MAT_ID &&
                            a->type == GGML_TYPE_Q2_K && b->type == GGML_TYPE_F32) {
                        return false;
                    }
                }
#endif // GGML_USE_MUSA
                switch (a->type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
                    case GGML_TYPE_Q1_0:
                    case GGML_TYPE_Q2_0:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_MXFP4:
                    case GGML_TYPE_NVFP4:
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q5_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_Q8_K:
                    case GGML_TYPE_IQ1_M:
                    case GGML_TYPE_IQ1_S:
                    case GGML_TYPE_IQ2_S:
                    case GGML_TYPE_IQ2_XS:
                    case GGML_TYPE_IQ2_XXS:
                    case GGML_TYPE_IQ3_S:
                    case GGML_TYPE_IQ3_XXS:
                    case GGML_TYPE_IQ4_NL:
                    case GGML_TYPE_IQ4_XS:
                    case GGML_TYPE_BF16:
                        return true;
                    default:
                        return false;
                }
            }
        case GGML_OP_OUT_PROD:
            return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32;
        case GGML_OP_GET_ROWS:
            {
                switch (op->src[0]->type) {
                    case GGML_TYPE_F16:
                    case GGML_TYPE_F32:
                    case GGML_TYPE_BF16:
                    case GGML_TYPE_I32:
                    case GGML_TYPE_Q1_0:
                    case GGML_TYPE_Q2_0:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q5_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_IQ2_XXS:
                    case GGML_TYPE_IQ2_XS:
                    case GGML_TYPE_IQ2_S:
                    case GGML_TYPE_IQ3_XXS:
                    case GGML_TYPE_IQ3_S:
                    case GGML_TYPE_IQ1_S:
                    case GGML_TYPE_IQ1_M:
                    case GGML_TYPE_IQ4_XS:
                        return true;
                    case GGML_TYPE_IQ4_NL:
                    case GGML_TYPE_MXFP4:
                        // 32-value sub-blocks, the row size does not guarantee
                        // the QK_K super-blocks the get_rows kernel iterates on
                        return op->src[0]->ne[0] % QK_K == 0;
                    default:
                        return false;
                }
            }
        case GGML_OP_GET_ROWS_BACK:
            return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 && op->ne[2] == 1 && op->ne[3] == 1;
        case GGML_OP_SET_ROWS:
            {
                return (
                           (
                               (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_BF16 ||
                               op->type == GGML_TYPE_Q4_0 || op->type == GGML_TYPE_Q4_1 || op->type == GGML_TYPE_Q5_0 ||
                               op->type == GGML_TYPE_Q5_1 || op->type == GGML_TYPE_Q8_0 || op->type == GGML_TYPE_IQ4_NL) &&
                               op->src[0]->type == GGML_TYPE_F32
                           ) || (
                               op->type == GGML_TYPE_F16 && op->src[0]->type == GGML_TYPE_F16
                           )
                       ) &&
                       (op->src[1]->type == GGML_TYPE_I64 || op->src[1]->type == GGML_TYPE_I32);
            }
        case GGML_OP_SET:
            {
                const ggml_type t = op->type;
                return (t == GGML_TYPE_F32 || t == GGML_TYPE_I32) &&
                    t == op->src[0]->type &&
                    t == op->src[1]->type;
            }
        case GGML_OP_CPY:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if ((src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_BF16 || src0_type == GGML_TYPE_F16) &&
                    (src1_type == GGML_TYPE_F32 || src1_type == GGML_TYPE_BF16 || src1_type == GGML_TYPE_F16)
                ) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q8_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q8_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q8_0 && src1_type == GGML_TYPE_F16) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q4_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q4_1 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q5_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q5_1 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_IQ4_NL) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_I32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_I32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_I32 && src1_type == GGML_TYPE_I32) {
                    return true;
                }
                if (src0_type == src1_type && ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) {
                    return true;
                }
                return false;
            }
        case GGML_OP_DUP:
                return true;
        case GGML_OP_ARGMAX:
        case GGML_OP_COUNT_EQUAL:
            return true;
        case GGML_OP_REPEAT:
            {
                // the CUDA REPEAT path only implements F32/F16; other types assert at runtime
                ggml_type src0_type = op->src[0]->type;
                return src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_F16;
            }
        case GGML_OP_REPEAT_BACK:
                return op->type == GGML_TYPE_F32 && (op->src[0]->ne[2]*op->src[0]->ne[3]) <= (1 << 15);
        case GGML_OP_CONCAT:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                const int32_t dim = op->op_params[0];
                return src0_type == src1_type &&
                       src0_type == op->type &&
                       (
                           (
                               ggml_is_quantized(src0_type) &&
                               (
                                   (
                                       dim == 3 &&
                                       ggml_is_contiguous(op->src[0]) &&
                                       ggml_is_contiguous(op->src[1])
                                   ) || (
                                       dim != 3 &&
                                       ggml_is_contiguous_to_3(op->src[0]) &&
                                       ggml_is_contiguous_to_3(op->src[1])
                                   )
                               ) &&
                               op->src[0]->ne[0] % ggml_blck_size(src0_type) == 0 &&
                               op->src[1]->ne[0] % ggml_blck_size(src0_type) == 0
                           ) || (
                               !ggml_is_quantized(src0_type) &&
                               ggml_blck_size(src0_type) == 1 &&
                               (
                                   ggml_type_size(src0_type) == 1 ||
                                   ggml_type_size(src0_type) == 2 ||
                                   ggml_type_size(src0_type) == 4 ||
                                   ggml_type_size(src0_type) == 8
                               )
                           )
                       );
            }
        case GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                return false;
            }
        case GGML_OP_COL2IM_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                return (src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_F16 || src0_type == GGML_TYPE_BF16) &&
                    op->type == src0_type &&
                    ggml_is_contiguous(op->src[0]) &&
                    ggml_is_contiguous(op);
            }
        case GGML_OP_SILU_BACK:
            return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
        case GGML_OP_L2_NORM:
            return ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_RMS_NORM_BACK:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_ADD_ID:
        case GGML_OP_ADD1:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_LOG:
            return true;
        case GGML_OP_SCALE:
            return (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_BF16) && op->type == op->src[0]->type;
        case GGML_OP_CLAMP:
            return ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            if (op->src[0]->type == GGML_TYPE_BF16 || op->src[1]->type == GGML_TYPE_BF16 || op->type == GGML_TYPE_BF16) {
                return op->src[0]->type == GGML_TYPE_BF16 && op->type == GGML_TYPE_BF16 &&
                    (op->src[1]->type == GGML_TYPE_BF16 || op->src[1]->type == GGML_TYPE_F32);
            }
            return (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) &&
                   (op->type         == GGML_TYPE_F32 || op->type         == GGML_TYPE_F16);
        case GGML_OP_SSM_SCAN: {
            const int32_t K = ggml_get_op_params_i32(op, 0);

            if (op->src[3]->ne[0] == 1) {
                // Mamba2
                // (kernel only supports (d_state == 96 || d_state == 128 || d_state == 256) && d_head % 16 == 0)
                const int64_t d_state = op->src[0]->ne[0];
                return (d_state == 96 || d_state == 128 || d_state == 256) && op->src[0]->ne[1] % 16 == 0;
            } else {
                if (K > 1) {
                    return false;
                }

                // Mamba
                // (kernel only supports d_state == 16, d_head == 1, n_head % 128 == 0, n_group == 1)
                return op->src[0]->ne[0] == 16 && op->src[0]->ne[1] == 1 && op->src[0]->ne[2] % 128 == 0 && op->src[4]->ne[1] == 1;
            }
        }
        case GGML_OP_SSM_CONV: {
            // assumes d_inner % threads == 0
            return op->src[0]->ne[1] % 128 == 0;
        }
        case GGML_OP_CONT:
            return true;
        case GGML_OP_DIAG_MASK_INF:
            return true;
        case GGML_OP_SOFT_MAX:
            return true;
        case GGML_OP_SOFT_MAX_BACK: {
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) op->op_params + 1, sizeof(float));
            return max_bias == 0.0f;
        }
        case GGML_OP_ROLL:
            if(op->src[0]->type == GGML_TYPE_F32) {
                return true;
            }
            return false;
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK: {
            return op->src[0]->nb[0] == ggml_type_size(op->src[0]->type) && ggml_is_contiguous_2(op->src[0]);
        }
        case GGML_OP_IM2COL:
        case GGML_OP_IM2COL_3D:
        case GGML_OP_CONV_2D:
            return (ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]));
        case GGML_OP_CONV_3D:
            return (op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32) &&
                   op->src[1]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]) && ggml_is_contiguous(op);
        case GGML_OP_CONV_2D_DW:
            return (op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32) &&
                   op->src[1]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        case GGML_OP_CONV_TRANSPOSE_2D:
        case GGML_OP_POOL_1D:
        case GGML_OP_POOL_2D:
            return true;
        case GGML_OP_ACC:
            // TODO: extend support like so:
            //return ggml_is_contiguous_rows(op->src[0]) && ggml_is_contiguous_rows(op->src[1]);
            return ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]);
        case GGML_OP_SUM:
            return ggml_is_contiguously_allocated(op->src[0]);
        case GGML_OP_TOP_K:
            return op->src[0]->ne[0] <= INT_MAX;
        case GGML_OP_ARGSORT:
#ifndef GGML_CUDA_USE_CUB
            {
                // bitonic path: the padded row must fit in shared memory
                int64_t ncols_pad = 1;
                while (ncols_pad < op->src[0]->ne[0]) {
                    ncols_pad *= 2;
                }
                return ncols_pad * sizeof(int) <= ggml_cuda_info().devices[dev_ctx->device].smpb;
            }
#else
            return op->src[0]->ne[0] <= INT_MAX;
#endif
        case GGML_OP_SUM_ROWS:
            return op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_MEAN:
            return op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_GROUP_NORM:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_PAD:
            return true;
        case GGML_OP_UPSCALE:
        case GGML_OP_PAD_REFLECT_1D:
        case GGML_OP_ARANGE:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_LEAKY_RELU:
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_GATED_LINEAR_ATTN:
        case GGML_OP_RWKV_WKV7:
            return true;
        case GGML_OP_GATED_DELTA_NET:
            return true;
        case GGML_OP_DSV4_HC_COMB:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        case GGML_OP_DSV4_HC_PRE:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32;
        case GGML_OP_DSV4_HC_POST:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 && (op->src[3] == nullptr || op->src[3]->type == GGML_TYPE_F32) &&
                op->type == GGML_TYPE_F32;
        case GGML_OP_FLASH_ATTN_EXT:
            return ggml_cuda_flash_attn_ext_supported(dev_ctx->device, op);
        case GGML_OP_CROSS_ENTROPY_LOSS:
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
        case GGML_OP_OPT_STEP_ADAMW:
        case GGML_OP_OPT_STEP_SGD:
        case GGML_OP_FILL:
        case GGML_OP_CUMSUM:
        case GGML_OP_TRI:
        case GGML_OP_DIAG:
        case GGML_OP_SOLVE_TRI:
            return true;
        case GGML_OP_LIGHTNING_INDEXER:
            return ggml_cuda_lightning_indexer_supported(dev_ctx->device, op);
        case GGML_OP_QSA_PACK:
            return ggml_cuda_qsa_pack_supported(op);
        case GGML_OP_MUL_MAT_HAD:
            return ggml_cuda_mul_mat_had_supported(dev_ctx->device, op);
        case GGML_OP_TOP_K_MASK:
            return op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32;

        default:
            return false;
    }
}

static bool ggml_backend_cuda_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;
    const bool integrated = ggml_cuda_info().devices[dev_ctx->device].integrated;
    return (ggml_backend_buft_is_cuda(buft) && buft->device == dev) || (integrated && ggml_backend_buft_is_cuda_host(buft));
}

static int64_t get_op_batch_size(const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_GET_ROWS:
            return 0;
        case GGML_OP_MUL_MAT:
            return op->ne[1];
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
            return op->ne[2];
        default:
            return ggml_nrows(op);
    }
}

static bool ggml_backend_cuda_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;

    return get_op_batch_size(op) >= dev_ctx->op_offload_min_batch_size;
}

static ggml_backend_event_t ggml_backend_cuda_device_event_new(ggml_backend_dev_t dev) {
#ifdef GGML_CUDA_NO_PEER_COPY
    GGML_UNUSED(dev);
    return nullptr;
#else
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *)dev->context;

    ggml_cuda_set_device(dev_ctx->device);

    cudaEvent_t event;
    CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));

    return new ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ event,
    };
#endif
}

static void ggml_backend_cuda_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);

    CUDA_CHECK(cudaEventDestroy((cudaEvent_t)event->context));
    delete event;
}

static void ggml_backend_cuda_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    CUDA_CHECK(cudaEventSynchronize((cudaEvent_t)event->context));
}

static const ggml_backend_device_i ggml_backend_cuda_device_interface = {
    /* .get_name                = */ ggml_backend_cuda_device_get_name,
    /* .get_description         = */ ggml_backend_cuda_device_get_description,
    /* .get_memory              = */ ggml_backend_cuda_device_get_memory,
    /* .get_type                = */ ggml_backend_cuda_device_get_type,
    /* .get_props               = */ ggml_backend_cuda_device_get_props,
    /* .init_backend            = */ ggml_backend_cuda_device_init_backend,
    /* .get_buffer_type         = */ ggml_backend_cuda_device_get_buffer_type,
    /* .get_host_buffer_type    = */ ggml_backend_cuda_device_get_host_buffer_type,
    /* .buffer_from_host_ptr    = */ NULL,
    /* .supports_op             = */ ggml_backend_cuda_device_supports_op,
    /* .supports_buft           = */ ggml_backend_cuda_device_supports_buft,
    /* .offload_op              = */ ggml_backend_cuda_device_offload_op,
    /* .event_new               = */ ggml_backend_cuda_device_event_new,
    /* .event_free              = */ ggml_backend_cuda_device_event_free,
    /* .event_synchronize       = */ ggml_backend_cuda_device_event_synchronize,
};

// backend reg

struct ggml_backend_cuda_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

static const char * ggml_backend_cuda_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_CUDA_NAME;
}

static size_t ggml_backend_cuda_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_cuda_reg_context * ctx = (ggml_backend_cuda_reg_context *)reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_cuda_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_cuda_reg_context * ctx = (ggml_backend_cuda_reg_context *)reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

static ggml_backend_feature * ggml_backend_cuda_get_features(ggml_backend_reg_t reg) {
    static std::vector<ggml_backend_feature> features = []() {
        std::vector<ggml_backend_feature> features;
    #define _STRINGIFY(...) #__VA_ARGS__
    #define STRINGIFY(...) _STRINGIFY(__VA_ARGS__)

    #ifdef __CUDA_ARCH_LIST__
        features.push_back({ "ARCHS", STRINGIFY(__CUDA_ARCH_LIST__) });
    #endif

    #ifdef GGML_CUDA_FORCE_MMQ
        features.push_back({ "FORCE_MMQ", "1" });
    #endif

    #ifdef GGML_CUDA_FORCE_CUBLAS
        features.push_back({ "FORCE_CUBLAS", "1" });
    #endif

    #ifndef GGML_USE_VMM
        features.push_back({ "NO_VMM", "1" });
    #endif

    #ifdef GGML_CUDA_NO_PEER_COPY
        features.push_back({ "NO_PEER_COPY", "1" });
    #endif

    #ifdef GGML_CUDA_USE_GRAPHS
        features.push_back({ "USE_GRAPHS", "1" });
    #endif

    #ifdef GGML_CUDA_FA_QUANTS
        features.push_back({ "FA_QUANTS", GGML_CUDA_FA_QUANTS });
    #endif

    {
        const auto & info = ggml_cuda_info();
        for (int id = 0; id < info.device_count; ++id) {
            if (blackwell_mma_available(info.devices[id].cc)) {
                features.push_back({ "BLACKWELL_NATIVE_FP4", "1"});
                break;
            }
        }
    }

    #undef _STRINGIFY
    #undef STRINGIFY

        features.push_back({ nullptr, nullptr });

        return features;
    }();

    return features.data();

    GGML_UNUSED(reg);
}

static void * ggml_backend_cuda_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    if (strcmp(name, "ggml_backend_comm_init") == 0) {
        return (void *)ggml_backend_cuda_comm_init;
    }
    if (strcmp(name, "ggml_backend_comm_free") == 0) {
        return (void *)ggml_backend_cuda_comm_free;
    }
    if (strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        return (void *)ggml_backend_cuda_comm_allreduce_tensor;
    }
    if (strcmp(name, "ggml_backend_register_host_buffer") == 0) {
        return (void *)ggml_backend_cuda_register_host_buffer;
    }
    if (strcmp(name, "ggml_backend_unregister_host_buffer") == 0) {
        return (void *)ggml_backend_cuda_unregister_host_buffer;
    }
    if (strcmp(name, "ggml_backend_get_features") == 0) {
        return (void *)ggml_backend_cuda_get_features;
    }
    // strixllama DEV: the side of the in-run A/B (STRIX_AB), for switches outside the backend
    if (strcmp(name, "ggml_backend_cuda_strix_ab") == 0) {
        return (void *)ggml_cuda_strix_ab;
    }
    // strixllama: the delta net's record replay for the memory's checkpoints and copies (gated_delta_net.cuh)
    if (strcmp(name, "ggml_backend_cuda_gdn_replay") == 0) {
        return (void *)ggml_backend_cuda_gdn_replay;
    }
    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_cuda_reg_interface = {
    /* .get_name          = */ ggml_backend_cuda_reg_get_name,
    /* .get_device_count  = */ ggml_backend_cuda_reg_get_device_count,
    /* .get_device        = */ ggml_backend_cuda_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_cuda_reg_get_proc_address,
};

// backend registry
ggml_backend_reg_t ggml_backend_cuda_reg() {
    static ggml_backend_reg reg;
    static bool initialized = false;

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            ggml_backend_cuda_reg_context * ctx = new ggml_backend_cuda_reg_context;
            const int min_batch_size = ggml_cuda_op_offload_min_batch();

            const ggml_cuda_device_info & info = ggml_cuda_info();
            const bool virtual_devices = info.device_count > info.physical_device_count;

            for (int i = 0; i < info.device_count; i++) {
                const int physical_id = info.devices[i].physical_device;

                ggml_backend_cuda_device_context * dev_ctx = new ggml_backend_cuda_device_context;
                dev_ctx->device = i;
                dev_ctx->name = GGML_CUDA_NAME + std::to_string(i);
                dev_ctx->description = ggml_cuda_device_description(i);

                char pci_bus_id[32] = {};
                CUDA_CHECK(cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), physical_id));
                dev_ctx->pci_bus_id = pci_bus_id;
                if (virtual_devices) {
                    // make the pci bus id unique for virtual devices
                    dev_ctx->pci_bus_id += "-v" + std::to_string(i);
                }
                for (char & c : dev_ctx->pci_bus_id) {
                    c = std::tolower(c);
                }
                dev_ctx->op_offload_min_batch_size = min_batch_size;

                ggml_backend_dev_t dev = new ggml_backend_device {
                    /* .iface   = */ ggml_backend_cuda_device_interface,
                    /* .reg     = */ &reg,
                    /* .context = */ dev_ctx
                };
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg {
                /* .api_version = */ GGML_BACKEND_API_VERSION,
                /* .iface       = */ ggml_backend_cuda_reg_interface,
                /* .context     = */ ctx
            };
        }

        initialized = true;
    }

    return &reg;
}

ggml_backend_t ggml_backend_cuda_init(int device) {
    if (device < 0 || device >= ggml_backend_cuda_get_device_count()) {
        GGML_LOG_ERROR("%s: invalid device %d\n", __func__, device);
        return nullptr;
    }

    ggml_backend_cuda_context * ctx = new ggml_backend_cuda_context(device);
    if (ctx == nullptr) {
        GGML_LOG_ERROR("%s: failed to allocate context\n", __func__);
        return nullptr;
    }

    ggml_backend_t cuda_backend = new ggml_backend {
        /* .guid    = */ ggml_backend_cuda_guid(),
        /* .iface   = */ ggml_backend_cuda_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), device),
        /* .context = */ ctx,
    };

    return cuda_backend;
}

GGML_BACKEND_DL_IMPL(ggml_backend_cuda_reg)
