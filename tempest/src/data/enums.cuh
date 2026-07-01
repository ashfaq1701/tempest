#ifndef ENUMS_H
#define ENUMS_H

enum RandomPickerType {
    Uniform,
    Linear,
    ExponentialIndex,
    ExponentialWeight,
    TemporalNode2Vec,

    // ONLY FOR TESTS
    TEST_FIRST,
    TEST_LAST
};

enum WalkDirection {
    Forward_In_Time,
    Backward_In_Time
};

enum KernelLaunchType {
    FULL_WALK,
    NODE_GROUPED,
    // ablation: NODE_GROUPED scheduler but routes every coop task through *_global
    // (no smem panel preload). isolates coop from smem-preload contribution.
    NODE_GROUPED_GLOBAL_ONLY
};

constexpr KernelLaunchType DEFAULT_KERNEL_LAUNCH_TYPE =
    KernelLaunchType::NODE_GROUPED;

#endif // ENUMS_H
