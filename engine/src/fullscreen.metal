#pragma once

#include <metal_stdlib>
#include "palette.metal"
using namespace metal;

struct RasterizerData
{
    float4 position [[position]];
    float4 color;
    float2 uv;  // Add UV coordinate
};

vertex RasterizerData
upscale_vert(uint id [[vertex_id]])
{
    RasterizerData out;
    float2 uv = float2((id << 1) & 2, id & 2);
    out.position = float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);   
    out.color = float4(1, 0, 1, 1);
    out.uv = uv;  // Pass UV directly (already 0-1)
    return out;
}


fragment float4 shade_frag(RasterizerData in [[stage_in]],
        constant float3* palettes [[buffer(5)]],
        texture2d<float, access::read> world_position [[texture(2)]],
        texture2d<float, access::read> world_normal [[texture(3)]],
        texture2d<float, access::read> scratch [[texture(0)]]) 
{    
    // 1. Standard G-Buffer Read
    float2 inputSize = float2(world_normal.get_width(), world_normal.get_height());
    uint2 coord = uint2(in.uv * inputSize);
    
    float3 normal = world_normal.read(coord).xyz;    

    // Background check
    float3 lit_color = float3(0.1, 0.1, 0.2); 
    if (any(normal != float3(0.0))) {        
        
        // Note: Replace this hardcoded pink with your voxel albedo texture read
        float3 voxel_albedo = float3(0.2, 1.0, 0.2); 
        lit_color = voxel_albedo;
        // 2. Calculate Lighting
        float3 light_dir = normalize(float3(2.0, 1.5, 10.0));    
        float diffuse = max(0.2, dot(light_dir, normal));        
        lit_color *= diffuse;
    }

    return float4(palettize(palettes, lit_color), 1.0);
}
