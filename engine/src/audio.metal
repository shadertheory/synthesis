#pragma once

#include <metal_stdlib>
#include "raytrace.metal"
using namespace metal;

struct Source {
    uint id;
    float3 position;
    float decibels;
    float radius;
    float occlusion;
};

struct Listener {
    float4x4 transform;
};

struct Probe {
    float outdoor;
    float delay;
    float decay;
    float3 ambient;
};


kernel void scan(
    uint2 tid [[thread_position_in_grid]],
    instance_acceleration_structure accel [[buffer(1)]],
    intersection_function_table<instancing> intersectionFunctionTable [[buffer(2)]]
)
{
}

kernel void occlude(
    uint2 tid [[thread_position_in_grid]],
    instance_acceleration_structure accel [[buffer(1)]],
    intersection_function_table<instancing> intersectionFunctionTable [[buffer(2)]]
)
{
}

kernel void visualize(
    uint2 tid [[thread_position_in_grid]],
    instance_acceleration_structure accel [[buffer(1)]],
    intersection_function_table<instancing> intersectionFunctionTable [[buffer(2)]]
)
{
}
