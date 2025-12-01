#include "raytrace.metal"
#include "camera.metal"
using namespace metal;

struct PerInstanceData {
    float4x4 transform;
    float4x4 inverse_transform;
};

// Manual derivative functions for compute shaders


// --- MAIN FRAGMENT SHADER ---

struct IntersectionResult {
    bool accept [[accept_intersection]];
    bool continueSearch [[continue_search]];
    float distance [[distance]];
};


typedef IntersectionResult IntersectionFunction(float3,
                                                     float3,
                                                     float,
                                                     float);

struct BoundingBox {
    packed_float3 min;
    packed_float3 max;
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
        result.distance = t;
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
