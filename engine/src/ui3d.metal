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
};

// Manual derivative functions for compute shaders
float2 dfdx(float2 value, float2 valueDx) {
    return valueDx - value;
}

float2 dfdy(float2 value, float2 valueDy) {
    return valueDy - value;
}
struct SoftwareRay {
    float3 origin;
    float3 direction;
};

struct SoftwarePlane {
    float3 point;   // A point on the plane
    float3 normal;  // Normal vector (should be normalized)
};
float4 intersect_plane(SoftwareRay ray, SoftwarePlane plane) {
    // SoftwarePlane equation: dot(normal, P - point) = 0
    // SoftwareRay equation: P(t) = origin + t * direction
    // Substitute: dot(normal, origin + t * direction - point) = 0
    // Solve for t: t = dot(normal, point - origin) / dot(normal, direction)
    
    float denom = dot(plane.normal, ray.direction);
    
    // Check if ray is parallel to plane (denominator ≈ 0)
    if (abs(denom) < 1e-6) {
        return float4(0.0); // No intersection
    }
    
    float t = dot(plane.normal, plane.point - ray.origin) / denom;
   
 
    
    // Calculate intersection point
    float3 intersection = ray.origin + t * ray.direction;
    return float4(intersection, 1.0); // .w = 1 indicates valid intersection
}

kernel void reference_grid(
    texture2d<float, access::write> scratch [[texture(0)]],
    const device CameraData* cameras [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {

    if(gid.x >= scratch.get_width() || gid.y >= scratch.get_height()) {
        return;
    }
    CameraData camera = cameras[0];
    float3 origin = float3(camera.transform[3].xyz);
    
    float2 uv = float2(gid) / (camera.resolution.xy);


    float2 ndc;
    ndc.x = uv.x * 2.0 - 1.0;
    ndc.y = (1.0 - uv.y) * 2.0 - 1.0;  // Flip Y for typical UV coords
    
    // Two points: near and far plane
    float4 nearPoint = float4(ndc.x, ndc.y, 1.0, 1.0);  // Near plane in NDC
    float4 farPoint = float4(ndc.x, ndc.y, 0.0, 1.0);    // Far plane in NDC
    
    float4x4 coord = float4x4(
        1.0,  0.0,  0.0,  0.0,  // X stays the same
        0.0,  0.0, -1.0,  0.0,  // Y ← -Z (forward becomes -forward)
        0.0,  1.0,  0.0,  0.0,  // Z ← Y (up stays up)
        0.0,  0.0,  0.0,  1.0
    );

    // Transform to view space
    float4 nearView = camera.projection_inverse * nearPoint;
    float4 farView = camera.projection_inverse * farPoint;
    
    // Perspective divide
    nearView /= nearView.w;
    farView /= farView.w;
    
    float4 nearWorld = camera.transform * nearView;
    float4 farWorld = camera.transform * farView;

    SoftwareRay ray;
    ray.origin = nearWorld.xyz;
    ray.direction = normalize(nearWorld.xyz - camera.transform[3].xyz); 

    SoftwarePlane plane;
    plane.point = (float4(0, 0, 0.0, 1)).xyz;
    plane.normal = (float4(0, 0, 1, 0)).xyz;
       
       
    float4 intersection =   intersect_plane(ray, plane);

   // Offset the grid by camera position so it appears to move with you


    float2 grid = fract(intersection.xy);
    
    float2 distEdge = min(grid, 1.0 - grid);
    float invEdge = 2 * max(distEdge.x, distEdge.y);
    float alpha = invEdge > 0.96 ? 1.0 : 0.0;

      
    scratch.write(float4(float3(alpha), 1), gid);
}

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

fragment float4 upscale_frag(RasterizerData in [[stage_in]],
    texture2d<float, access::read> scratch [[texture(0)]])
{
    float2 inputSize = float2(scratch.get_width(), scratch.get_height());
    float2 texCoord = in.uv * inputSize;  // Scale UV to input texture size
    uint2 coord = uint2(texCoord);
    return scratch.read(coord);
}

// Simple struct for bounding box intersection return
struct BoundingBoxIntersection {
    bool accept;
    float distance;
};



typedef BoundingBoxIntersection IntersectionFunction(float3,
                                                     float3,
                                                     float,
                                                     float);
struct IntersectionResult {
    bool accept [[accept_intersection]];
    bool continueSearch [[continue_search]];
    float distance [[distance]];
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
[[intersection(bounding_box, instancing, world_space_data)]]
IntersectionResult sphere_intersect(
    float3 origin [[world_space_origin]],
    float3 direction [[world_space_direction]],
    uint primitiveID [[primitive_id]],
    float minDistance [[min_distance]],
    float maxDistance [[max_distance]]
)
{
    IntersectionResult result;
    result.accept = false;
    result.distance = maxDistance;
    
    float3 box_min = float3(-0.5);
    float3 box_max = float3(0.5);
    float3 direction_inv = 1.0 / direction;
    float3 t0 = (box_min - origin) * direction_inv;
    float3 t1 = (box_max - origin) * direction_inv;
    float3 a = min(t0, t1);
    float3 b = max(t0, t1);
    float box_tmin = max(max(a.x, a.y), max(a.z, minDistance));
    float box_tmax = min(min(b.x, b.y), min(b.z, maxDistance));
    float box_hit = step(box_tmin, box_tmax);
    
    const int VOXEL_RESOLUTION = 8;
    const float VOXEL_SIZE = 1.0 / float(VOXEL_RESOLUTION);
    const float RADIUS = 0.4;
    const float EPSILON = 0.000001 * float(VOXEL_RESOLUTION);
    
    float3 start = origin + direction * box_tmin;
    float3 voxel_pos = (start - box_min) / VOXEL_SIZE;
    int3 pos = int3(floor(voxel_pos));
    int3 step_dir = int3(sign(direction));
    float3 delta = abs(VOXEL_SIZE / direction);
    float3 fraction = fract(voxel_pos);
    float3 positive = float3(step_dir > 0);
    float3 tMax = delta * (positive * (1.0 - fraction) + (1.0 - positive) * fraction);
    
    float closest_hit = maxDistance;
    float hit_found = 0.0;
    
    const int MAX_STEPS = 64;
    
    for (int iter = 0; iter < MAX_STEPS; ++iter) {
        // Check bounds
        float3 in_low = step(0.0, float3(pos));
        float3 in_high = step(float3(pos), float3(VOXEL_RESOLUTION - 1));
        float in_bounds = in_low.x * in_low.y * in_low.z * in_high.x * in_high.y * in_high.z;
        
        // Voxel center
        float3 center = box_min + (float3(pos) + 0.5) * VOXEL_SIZE;
        
        // Check if sphere surface in voxel
        float dist = sphere_sdf(center, RADIUS);
        float contains = step(dist, VOXEL_SIZE * 0.866) * step(-VOXEL_SIZE * 0.866, dist);
        
        // Hit distance at voxel entry
        float3 voxel_min = box_min + float3(pos) * VOXEL_SIZE;
        float3 voxel_max = voxel_min + VOXEL_SIZE;
        float3 t0_voxel = (voxel_min - origin) * direction_inv;
        float3 t1_voxel = (voxel_max - origin) * direction_inv;
        float3 a_voxel = min(t0_voxel, t1_voxel);
        float3 b_voxel = max(t0_voxel, t1_voxel);
        float t_enter = max(max(a_voxel.x, a_voxel.y), max(a_voxel.z, box_tmin)) + EPSILON;
        
        // Record hit
        float valid = in_bounds * contains * (1.0 - hit_found);
        closest_hit = mix(closest_hit, t_enter, valid);
        hit_found = max(hit_found, valid);
        
        // Step to next voxel
        float step_x = step(tMax.x, tMax.y) * step(tMax.x, tMax.z);
        float step_y = (1.0 - step_x) * step(tMax.y, tMax.z);
        float step_z = (1.0 - step_x) * (1.0 - step_y);
        
        pos.x += int(step_x) * step_dir.x;
        pos.y += int(step_y) * step_dir.y;
        pos.z += int(step_z) * step_dir.z;
        
        tMax.x += step_x * delta.x;
        tMax.y += step_y * delta.y;
        tMax.z += step_z * delta.z;
    }
    
    result.accept = box_hit * hit_found > 0.5;
    result.distance = closest_hit;
    
    return result;
}
    
kernel void raytrace(
    uint2 tid [[thread_position_in_grid]],
    constant CameraData& camera [[buffer(0)]],
    instance_acceleration_structure accel [[buffer(1)]],
    texture2d<float, access::write> output [[texture(0)]],
    intersection_function_table<instancing, world_space_data> intersectionFunctionTable [[buffer(2)]]  // ← Changed type!
)
{
    if (tid.x >= uint(camera.resolution.x) || tid.y >= uint(camera.resolution.y)) return;
    // Generate ray
    float2 uv = float2(tid) / camera.resolution.xy;
 
    float2 ndc;
    ndc.x = uv.x * 2.0 - 1.0;
    ndc.y = uv.y * 2.0 - 1.0;  // Flip Y for typical UV coords
    
    // Two points: near and far plane
    float4 nearPoint = float4(ndc.x, ndc.y, 1.0, 1.0);  // Near plane in NDC
    float4 farPoint = float4(ndc.x, ndc.y, 0.999, 1.0);    // Far plane in NDC
       // Transform to view space
    float4 nearView = camera.projection_inverse * nearPoint;
    float4 farView = camera.projection_inverse * farPoint;
    
    // Perspective divide
    nearView /= nearView.w;
    farView /= farView.w;
    
    float4 nearWorld = camera.transform * nearView;
    float4 farWorld = camera.transform * farView;

    ray r;
    r.origin = nearWorld.xyz; 
    r.direction = normalize(farWorld.xyz - nearWorld.xyz);
    r.min_distance = 0.001;
    r.max_distance = 99.5;    
    // Debug: Test if ray should hit a sphere at origin
    // Manual sphere intersection for debugging
    
    
    float3 color = float3(0.0); // Start black
    
      
    // Create intersection query
    intersector<instancing, world_space_data> i;
    
    // Call intersect() with the visible function table parameter!
    auto result = i.intersect(r, accel, 0xFF, intersectionFunctionTable);
    
    // Check if we hit something
    if (result.type != intersection_type::none) { 
        float distance = result.distance;
        float3 hit = r.origin + r.direction * distance;
        
        // Recalculate which voxel was hit (same logic as intersection function)
        const int VOXEL_RESOLUTION = 8;
        const float VOXEL_SIZE = 1.0 / float(VOXEL_RESOLUTION);
        float3 box_min = float3(-0.5);
        
        // Find which voxel contains the hit point
        float3 voxel_pos = (hit - box_min) / VOXEL_SIZE;
        int3 pos = int3(floor(voxel_pos));
        
        // Get the voxel bounds
        float3 voxel_min = box_min + float3(pos) * VOXEL_SIZE;
        float3 voxel_max = voxel_min + VOXEL_SIZE;
        
        // Determine which face of the voxel cube was hit by checking distances
        float3 t_min = (voxel_min - hit) / r.direction;
        float3 t_max = (voxel_max - hit) / r.direction;
        
        // The face with the smallest positive t value is the one we hit
        float3 t_near = min(t_min, t_max);
        
        // Find which component is largest (that's the face we hit)
        float3 normal = float3(0.0);
        if (t_near.x >= t_near.y && t_near.x >= t_near.z) {
            normal = float3(-sign(r.direction.x), 0.0, 0.0);
        } else if (t_near.y >= t_near.z) {
            normal = float3(0.0, -sign(r.direction.y), 0.0);
        } else {
            normal = float3(0.0, 0.0, -sign(r.direction.z));
        }        float3 light_dir = normalize(float3(3, 2, 1));
        float diffuse = max(0.2, dot(light_dir, normal));
        
        color = diffuse * float3(1.0, 0.0, 1.0);
    }
    
    output.write(float4(color, 1.0), tid);
} 
