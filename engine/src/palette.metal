#pragma once

#include "color.metal";
constant uint PALETTE_SIZE = 64;

constant float3 WEIGHTS = float3(40.0, 20.0, 10.0);

float3 palettize(

        constant float3* palettes,
        float3 color) {
    float3 target_hsv = rgb2hsv(color);
    float min_dist_sq = 1e9; 
    float3 best_match = target_hsv; // Fallback to original if nothing matches

    for (uint i = 0; i < PALETTE_SIZE; i++) {
        float3 p_hsv = palettes[i];

        // 1. Hue Diff (Circular)
        float h_diff = abs(target_hsv.x - p_hsv.x);
        h_diff = min(h_diff, 1.0 - h_diff);

        // 2. Sat/Val Diff
        float s_diff = target_hsv.y - p_hsv.y;
        float v_diff = target_hsv.z - p_hsv.z;

        // 3. Calculate Weighted Distance
        float d_sq = (h_diff * h_diff * WEIGHTS.x) + 
                     (s_diff * s_diff * WEIGHTS.y) + 
                     (v_diff * v_diff * WEIGHTS.z);

        // 4. BLACKGUARD: Extra penalty for picking Black (V near 0) 
        // if our target is actually bright (V > 0.2).
        // This stops "Lazy Black" matching.
        if (p_hsv.z < 0.2 && target_hsv.z > 0.2) {
             d_sq += 100.0; // Massive penalty
        }

        if (d_sq < min_dist_sq) {
            min_dist_sq = d_sq;
            best_match = p_hsv;
        }
    }
    
   
    return hsv2rgb(best_match);// Simple struct for bounding box intersection return
}
