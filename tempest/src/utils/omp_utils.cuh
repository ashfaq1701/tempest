#ifndef UTILS_OMP_H
#define UTILS_OMP_H

#include <omp.h>
#include <vector>

template <typename T>
void parallel_prefix_sum(const T* input, T* output, size_t n) {
    static_assert(std::is_arithmetic_v<T>, "parallel_prefix_sum_ptr requires a numeric type");

    if (n == 0) return;

    int num_threads = 0;
    std::vector<T> thread_sums;

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const int nt = omp_get_num_threads();

        #pragma omp single
        {
            num_threads = nt;
            thread_sums.resize(nt + 1, T(0));
        }

        T local_sum = T(0);

        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            local_sum += input[i];
            output[i] = local_sum;
        }

        thread_sums[tid + 1] = local_sum;

        #pragma omp barrier

        #pragma omp single
        {
            for (int i = 1; i <= num_threads; ++i) {
                thread_sums[i] += thread_sums[i - 1];
            }
        }

        T offset = thread_sums[tid];

        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            output[i] -= input[i];
            output[i] += offset;
        }
    }
}

template <typename T>
void parallel_exclusive_scan(const T* input, T* output_with_offset, const size_t n) {
    static_assert(std::is_arithmetic_v<T>, "T must be numeric");
    if (n == 0) {
        output_with_offset[0] = 0;
        return;
    }

    int num_threads = 0;
    std::vector<T> thread_sums;

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const int nt = omp_get_num_threads();

        #pragma omp single
        {
            num_threads = nt;
            thread_sums.resize(nt + 1, T(0));
        }

        T local_sum = 0;

        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            local_sum += input[i];
            output_with_offset[i + 1] = local_sum;
        }

        thread_sums[tid + 1] = local_sum;

        #pragma omp barrier
        #pragma omp single
        {
            for (int i = 1; i <= num_threads; ++i) {
                thread_sums[i] += thread_sums[i - 1];
            }
        }

        T offset = thread_sums[tid];

        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            output_with_offset[i + 1] += offset;
        }
    }

    output_with_offset[0] = 0;
}

template <typename T>
void parallel_inclusive_scan(T* data, size_t n) {
    static_assert(std::is_arithmetic_v<T>, "parallel_inclusive_scan_ptr requires a numeric type");
    if (n == 0) return;

    int num_threads = 0;
    std::vector<T> thread_sums;

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const int nt  = omp_get_num_threads();

        #pragma omp single
        {
            num_threads = nt;
            thread_sums.resize(nt + 1, T(0));
        }

        T local_sum = T(0);

        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            local_sum += data[i];
            data[i] = local_sum;
        }

        thread_sums[tid + 1] = local_sum;

        #pragma omp barrier

        #pragma omp single
        {
            for (int i = 1; i <= num_threads; ++i) {
                thread_sums[i] += thread_sums[i - 1];
            }
        }

        T offset = thread_sums[tid];

        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            data[i] += offset;
        }
    }
}

#endif // UTILS_OMP_H
