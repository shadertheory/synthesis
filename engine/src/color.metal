#pragma once

#include <metal_stdlib>
using namespace metal;

float3 rgb2hsv(float3 c) {
    float4 K = float4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    float4 p = mix(float4(c.bg, K.wz), float4(c.gb, K.xy), step(c.b, c.g));
    float4 q = mix(float4(p.xyw, c.r), float4(c.r, p.yzx), step(p.x, c.r));

    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return float3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}
inline float3 hsv2rgb(float3 c) {
    // K stores the offsets for R, G, B phases and the math constants
    float4 K = float4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    
    // 1. Vectorized Hue calculation (R, G, B calculated at once)
    // "fract" handles the wrapping (mod 1.0) automatically
    float3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    
    // 2. Apply Saturation and Value using mix (lerp) and clamp
    // saturate() is a free intrinsic on Metal (clamp 0.0 to 1.0)
    return c.z * mix(K.xxx, saturate(p - K.xxx), c.y);
}
