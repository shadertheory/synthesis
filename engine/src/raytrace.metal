#pragma once

#include <metal_raytracing>
using namespace metal::raytracing;

// A payload structure to pass custom data from intersection functions.
struct RayPayload {
    float3 normal;
};