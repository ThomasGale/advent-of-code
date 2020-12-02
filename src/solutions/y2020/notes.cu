// Atomic lock = https://stackoverflow.com/questions/18963293/cuda-atomics-change-flag/18968893#18968893

__device__ volatile int sem = 0;

__device__ void acquire_semaphore(volatile int* lock) {
    while (atomicCAS((int*)lock, 0, 1) != 0)
        ;
}

__device__ void release_semaphore(volatile int* lock) {
    *lock = 0;
    __threadfence();
}

__global__ void please(uint* ans) {
    *ans = 42;
}

__syncthreads();
if (threadIdx.x == 0)
    acquire_semaphore(&sem);
__syncthreads();

// Single Thread here

__threadfence();
__syncthreads();
if (threadIdx.x == 0)
    release_semaphore(&sem);
__syncthreads();

// ---------------------------
