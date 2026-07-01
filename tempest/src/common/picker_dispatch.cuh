#ifndef PICKER_DISPATCH_CUH
#define PICKER_DISPATCH_CUH

#include "../data/enums.cuh"

// runtime->constexpr tag dispatch (C++17-compatible).

template <RandomPickerType PT>
struct PickerTag {
    static constexpr RandomPickerType value = PT;
};

template <bool B>
struct BoolTag {
    static constexpr bool value = B;
};

template <typename Visitor>
inline void dispatch_picker_type(RandomPickerType pt, Visitor&& visit) {
    switch (pt) {
        case RandomPickerType::Uniform:
            visit(PickerTag<RandomPickerType::Uniform>{}); break;
        case RandomPickerType::Linear:
            visit(PickerTag<RandomPickerType::Linear>{}); break;
        case RandomPickerType::ExponentialIndex:
            visit(PickerTag<RandomPickerType::ExponentialIndex>{}); break;
        case RandomPickerType::ExponentialWeight:
            visit(PickerTag<RandomPickerType::ExponentialWeight>{}); break;
        case RandomPickerType::TemporalNode2Vec:
            visit(PickerTag<RandomPickerType::TemporalNode2Vec>{}); break;
        case RandomPickerType::TEST_FIRST:
            visit(PickerTag<RandomPickerType::TEST_FIRST>{}); break;
        case RandomPickerType::TEST_LAST:
            visit(PickerTag<RandomPickerType::TEST_LAST>{}); break;
    }
}

template <typename Visitor>
inline void dispatch_bool(bool b, Visitor&& visit) {
    if (b) visit(BoolTag<true>{});
    else   visit(BoolTag<false>{});
}

#endif // PICKER_DISPATCH_CUH
