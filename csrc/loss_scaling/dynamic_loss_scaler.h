// csrc/loss_scaling/dynamic_loss_scaler.h
#pragma once

// GPU-side dynamic loss scaling can be added here.
// Currently loss scaling is implemented in Python (mixed_precision/training/loss_scaling.py).

void dynamic_loss_scale_step(
    const float* grads_fp32, int n,
    float* scale_inout, int* good_steps_inout,
    float scale_factor, int scale_window,
    cudaStream_t stream = 0
);
