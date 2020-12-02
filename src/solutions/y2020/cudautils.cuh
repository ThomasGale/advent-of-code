#pragma once

namespace aoc {
namespace y2020 {
namespace cudautils {
namespace semaphore {

__device__ volatile int sem = 0;

__device__ void acquire_semaphore(volatile int* lock) {
    while (atomicCAS((int*)lock, 0, 1) != 0)
        ;
}

__device__ void release_semaphore(volatile int* lock) {
    *lock = 0;
    __threadfence();
}

} // namespace semaphore
} // namespace cudautils
} // namespace y2020
} // namespace aoc