#pragma once

#include <metal_stdlib>
using namespace metal;

struct CameraData {
    float4x4 projection;
    float4x4 projection_inverse;
    float4x4 view;
    float4x4 transform;
    float4 resolution;
    uint4 info;
};
