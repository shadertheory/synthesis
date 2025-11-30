#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace raytracing;
struct CameraData {
    float4x4 projection;
    float4x4 projection_inverse;
    float4x4 view;
    float4x4 transform;
    float4 resolution;
    uint4 info;
};

struct PerInstanceData {
    float4x4 transform;
    float4x4 inverse_transform;
};

// Manual derivative functions for compute shaders
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

// --- MAIN FRAGMENT SHADER ---

constant uint PALETTE_SIZE = 64;

// Tunable Weights for your specific "Grass stays Green" requirement
// Hue is weighted massively so the shader prefers a Dark Green over a Dark Blue
// even if the Blue is mathematically closer in Euclidean RGB space.
constant float3 WEIGHTS = float3(40.0, 20.0, 10.0);
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
    }   

    float3 target_hsv = rgb2hsv(lit_color);
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
    
    if (any(normal != float3(0.0))) {
        // 2. Calculate Lighting
        float3 light_dir = normalize(float3(2.0, 1.5, 10.0));    
        float diffuse = max(0.2, dot(light_dir, normal));        
        best_match.z *= diffuse;
    }
    return float4(hsv2rgb(best_match), 1.0);// Simple struct for bounding box intersection return
}
struct IntersectionResult {
    bool accept [[accept_intersection]];
    bool continueSearch [[continue_search]];
    float distance [[distance]];
};


typedef IntersectionResult IntersectionFunction(float3,
                                                     float3,
                                                     float,
                                                     float);
// A payload structure to pass custom data from intersection functions.
struct RayPayload {
    float3 normal;
};

struct BoundingBox {
    float3 min;
    float3 max;
};

// Option 1: As a constant in your shader
constant BoundingBox sphereBounds = {
    .min = float3(-0.5, -0.5, -0.5),
    .max = float3(0.5, 0.5, 0.5)
};

float cube_sdf(float3 p, float3 s) {
    float3 d = abs(p) - s;
    return length(max(d, 0.0)) + min(max(d.x, max(d.y, d.z)), 0.0);
}

float3 cube_normal(float3 p, float3 s, float e = 0.001f) {
    float2 h = float2(e, 0);

    float nx = cube_sdf(p + h.xyy, s) - cube_sdf(p - h.xyy, s);
    float ny = cube_sdf(p + h.yxy, s) - cube_sdf(p - h.yxy, s);
    float nz = cube_sdf(p + h.yyx, s) - cube_sdf(p - h.yyx, s);

    return normalize(float3(nx, ny, nz));
}

float sphere_sdf(float3 p, float r) {
    return length(p) - r;
}

float3 sphere_normal(float3 p) {
    return normalize(p);
}

[[intersection(bounding_box, instancing)]]
IntersectionResult grid_intersect(
    float3 origin [[origin]],
    float3 direction [[direction]],
    uint primitiveID [[primitive_id]],
    float minDistance [[min_distance]],
    float maxDistance [[max_distance]],
    ray_data RayPayload& payload [[payload]]
)
{
    IntersectionResult result;
    result.accept = false;
    result.continueSearch = false;
    result.distance = maxDistance;
    float denom = dot(float3(0, 0, 1), direction);
    
    // Check if ray is parallel to plane (denominator ≈ 0)
    if (abs(denom) < 1e-6) {
        return result; // No intersection
    }
    
    float t = dot(float3(0, 0, 1), float3(0) - origin) / denom;
   
 
    
    if (t < 0) {
        return result;
    }
    
    // Calculate intersection point
    float3 intersection = origin + t * direction;

    float2 grid = fract(intersection.xy);
    
    float2 distEdge = min(grid, 1.0 - grid);
    float invEdge = 2 * max(distEdge.x, distEdge.y);

    if(invEdge > 0.95) {
        result.accept = true;
        result.distance = maxDistance - 0.1;
        result.continueSearch = true;
    }
    return result;
}
[[intersection(bounding_box, instancing)]]
IntersectionResult voxel_intersect(
    float3 origin [[origin]],
    float3 direction [[direction]],
    uint primitiveID [[primitive_id]],
    float minDistance [[min_distance]],
    float maxDistance [[max_distance]],
    ray_data RayPayload& payload [[payload]]
)
{
    IntersectionResult result;
    result.accept = false;
    result.continueSearch = true;
    result.distance = maxDistance;
    
    // Standard AABB test
    float3 invDir = 1.0 / direction;
    float3 t0s = (float3(0.0) - origin) * invDir;
    float3 t1s = (float3(1.0) - origin) * invDir;
    float3 tmin = min(t0s, t1s);
    float3 tmax = max(t0s, t1s);
    
    float t_enter = max(max(tmin.x, tmin.y), tmin.z);
    float t_exit = min(min(tmax.x, tmax.y), tmax.z);
    
    // Reject if no valid intersection
    if (t_enter > t_exit) {
        return result;
    }
    
    // ONLY accept the ENTRY point, never the exit
    // Skip if entry point is before minDistance or after maxDistance
    if (t_enter < minDistance || t_enter >= maxDistance) {
        result.continueSearch = true;
        return result;
    }
    
    result.accept = true;
    result.distance = t_enter;
    
    // Calculate normal for entry face
    float3 normal;
    if (tmin.x >= tmin.y && tmin.x >= tmin.z) {
        normal = (direction.x > 0.0) ? float3(-1.0, 0.0, 0.0) : float3(1.0, 0.0, 0.0);
    }
    else if (tmin.y >= tmin.z) {
        normal = (direction.y > 0.0) ? float3(0.0, -1.0, 0.0) : float3(0.0, 1.0, 0.0);
    }
    else {
        normal = (direction.z > 0.0) ? float3(0.0, 0.0, -1.0) : float3(0.0, 0.0, 1.0);
    }
    
    payload.normal = normal;
    return result;
}

kernel void raytrace(
    uint2 tid [[thread_position_in_grid]],
    constant CameraData& camera [[buffer(0)]],
    instance_acceleration_structure accel [[buffer(1)]],
    texture2d<float, access::write> world_position [[texture(0)]],
    texture2d<float, access::write> world_normal [[texture(1)]],
    constant PerInstanceData* instance_data [[buffer(3)]],
    intersection_function_table<instancing> intersectionFunctionTable [[buffer(2)]]
)
{
    if (tid.x >= uint(camera.resolution.x) || tid.y >= uint(camera.resolution.y)) return;
    uint frame = camera.info[0];
    // Generate ray
    float2 uv = float2(tid) / camera.resolution.xy;
 
    float2 ndc;
    ndc.x = uv.x * 2.0 - 1.0;
    ndc.y = uv.y * 2.0 - 1.0;
    
    float4 nearPoint = float4(ndc.x, ndc.y, 1.0, 1.0);
    float4 farPoint = float4(ndc.x, ndc.y, 0.999, 1.0);
    
    float4 nearView = camera.projection_inverse * nearPoint;
    float4 farView = camera.projection_inverse * farPoint;
    
    nearView /= nearView.w;
    farView /= farView.w;
    
    float4 nearWorld = camera.transform * nearView;
    float4 farWorld = camera.transform * farView;

    float3 light = normalize(float3(12, 6, 3));
    ray r;
    r.origin = nearWorld.xyz; 
    r.direction = normalize(farWorld.xyz - nearWorld.xyz);
    r.min_distance = 0.001;
    r.max_distance = 999.5;
    
    // Declare the payload that will be filled by the intersection function.
    RayPayload payload;
    
    intersector<instancing> i;
    i.force_opacity(forced_opacity::non_opaque);
    i.accept_any_intersection(false);  // We want the NEAREST intersection

    auto result = i.intersect(r, accel, 0xFF, intersectionFunctionTable, payload);
    
    float3 hit = float3(0.0);
    float4 normal = float4(0.0);

    if (result.type != intersection_type::none) { 
        normal.xyz = payload.normal;
        normal.w = 1.0;
    }

    world_position.write(float4(hit, 1.0), tid);
    world_normal.write(normal, tid);
}
