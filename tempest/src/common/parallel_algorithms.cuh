#ifndef PARALLEL_ALGORITHMS_H
#define PARALLEL_ALGORITHMS_H

#include <algorithm>
#include <iterator>
#include <type_traits>

#ifndef IS_MACOS
#include <execution>
#endif

namespace parallel {

    template<typename RandomIt>
    void sort(RandomIt first, RandomIt last) {
        #ifdef IS_MACOS
        std::sort(first, last);
        #else
        std::sort(std::execution::par_unseq, first, last);
        #endif
    }

    template<typename RandomIt, typename Compare>
    void sort(RandomIt first, RandomIt last, Compare comp) {
        #ifdef IS_MACOS
        std::sort(first, last, comp);
        #else
        std::sort(std::execution::par_unseq, first, last, comp);
        #endif
    }

    template<typename RandomIt>
    void stable_sort(RandomIt first, RandomIt last) {
        #ifdef IS_MACOS
        std::stable_sort(first, last);
        #else
        std::stable_sort(std::execution::par_unseq, first, last);
        #endif
    }

    template<typename RandomIt, typename Compare>
    void stable_sort(RandomIt first, RandomIt last, Compare comp) {
        #ifdef IS_MACOS
        std::stable_sort(first, last, comp);
        #else
        std::stable_sort(std::execution::par_unseq, first, last, comp);
        #endif
    }

    template<typename InputIt1, typename InputIt2, typename OutputIt>
    OutputIt merge(InputIt1 first1, InputIt1 last1,
                   InputIt2 first2, InputIt2 last2,
                   OutputIt d_first) {
        #ifdef IS_MACOS
        return std::merge(first1, last1, first2, last2, d_first);
        #else
        return std::merge(std::execution::par_unseq, first1, last1, first2, last2, d_first);
        #endif
    }

    template<typename InputIt1, typename InputIt2,
        typename OutputIt, typename Compare>
    OutputIt merge(InputIt1 first1, InputIt1 last1,
                   InputIt2 first2, InputIt2 last2,
                   OutputIt d_first, Compare comp) {
        #ifdef IS_MACOS
        return std::merge(first1, last1, first2, last2, d_first, comp);
        #else
        return std::merge(std::execution::par_unseq, first1, last1, first2, last2, d_first, comp);
        #endif
    }

} // namespace parallel

#endif // PARALLEL_ALGORITHMS_H
