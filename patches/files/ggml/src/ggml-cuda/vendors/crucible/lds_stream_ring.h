// Wave-specialized stream handoff. The caller owns the payload and LDS size.
#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>
#include <type_traits>

namespace crucible {

// Monotonic counters avoid reset races when loaders run ahead of consumers.
// Each wave calls these methods together. Only lane 0 updates a counter.
template <int Slots, int Loaders, int Consumers>
struct LdsStreamRing {
  static_assert(Slots > 1 && Loaders > 0 && Consumers > 0);
  uint32_t filled[Slots], consumed[Slots];

  __device__ __forceinline__ void Initialize(int thread) {
    if (thread < Slots) filled[thread] = consumed[thread] = 0;
  }
  __device__ __forceinline__ void WaitWritable(uint32_t seq) {
    const uint32_t need = Consumers * (seq / Slots);
    while (__hip_atomic_load(&consumed[seq % Slots], __ATOMIC_RELAXED,
                              __HIP_MEMORY_SCOPE_WORKGROUP) < need)
      __builtin_amdgcn_s_sleep(1);
    asm volatile("" ::: "memory");
  }
  __device__ __forceinline__ void Publish(uint32_t seq, int lane) {
    // A release atomic also drains VMEM and destroys loader overlap.
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    if (lane == 0)
      __hip_atomic_fetch_add(&filled[seq % Slots], 1u, __ATOMIC_RELAXED,
                              __HIP_MEMORY_SCOPE_WORKGROUP);
  }
  __device__ __forceinline__ void WaitReady(uint32_t seq) {
    const uint32_t need = Loaders * (seq / Slots + 1);
    while (__hip_atomic_load(&filled[seq % Slots], __ATOMIC_RELAXED,
                              __HIP_MEMORY_SCOPE_WORKGROUP) < need)
      __builtin_amdgcn_s_sleep(1);
    asm volatile("" ::: "memory");
  }
  __device__ __forceinline__ void Release(uint32_t seq, int lane) {
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    if (lane == 0)
      __hip_atomic_fetch_add(&consumed[seq % Slots], 1u, __ATOMIC_RELAXED,
                              __HIP_MEMORY_SCOPE_WORKGROUP);
  }
};

// Scalar table loads do not drain outstanding vector weight loads.
template <class T>
__device__ __forceinline__ T StreamConstant(const T* p) {
  static_assert(std::is_trivially_copyable_v<T> && sizeof(T) % 4 == 0);
  using CT = const __attribute__((address_space(4))) uint32_t;
  const CT* source = (const CT*)p;
  uint32_t raw[sizeof(T) / 4];
#pragma unroll
  for (int i = 0; i < sizeof(T) / 4; ++i) raw[i] = source[i];
  T value;
  __builtin_memcpy(&value, raw, sizeof(T));
  return value;
}

}  // namespace crucible
