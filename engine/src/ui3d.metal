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
[[visible]]
BoundingBoxIntersection sphere_intersect(
                                                   // Ray parameters passed to the intersection query below
                                                   float3 origin,
                                                   float3 direction,
                                                   float minDistance,
                                                   float maxDistance)
{
    // Look up the resources for this piece of sphere geometry.

    // Check for intersection between the ray and sphere mathematically.
    BoundingBoxIntersection ret;

    float3 oc = origin;

    float a = dot(direction, direction);
    float b = 2 * dot(oc, direction);
    float c = dot(oc, oc) - 0.25;

    float disc = b * b - 4 * a * c;


    if (disc <= 0.0f)
    {
        // If the ray missed the sphere, return `false`.
        ret.accept = false;
    }
    else
    {
        // Otherwise, compute the intersection distance.
        ret.distance = (-b - sqrt(disc)) / (2 * a);

        // The intersection function must also check whether the intersection distance is
        // within the acceptable range. Intersections are not reported in any particular order,
        // so the maximum distance may be different from the one passed into the intersection query.
        ret.accept = ret.distance >= minDistance && ret.distance <= maxDistance;
    }

    return ret;
}



kernel void raytrace(
    uint2 tid [[thread_position_in_grid]],
    constant CameraData& camera [[buffer(0)]],
    instance_acceleration_structure accel [[buffer(1)]],
    texture2d<float, access::write> output [[texture(0)]],
    visible_function_table<IntersectionFunction> intersectionFunctionTable [[buffer(2)]]
)
{
    if (tid.x >= uint(camera.resolution.x) || tid.y >= uint(camera.resolution.y)) return;
    // Generate ray
    float2 uv = float2(tid) / camera.resolution.xy;
 
    float2 ndc;
    ndc.x = uv.x * 2.0 - 1.0;
    ndc.y = (1.0 - uv.y) * 2.0 - 1.0;  // Flip Y for typical UV coords
    
    // Two points: near and far plane
    float4 nearPoint = float4(ndc.x, ndc.y, 1.0, 1.0);  // Near plane in NDC
    float4 farPoint = float4(ndc.x, ndc.y, 0.0, 1.0);    // Far plane in NDC
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
    r.direction = -normalize(nearWorld.xyz - camera.transform[3].xyz);
    r.min_distance = 0.001;
    r.max_distance = 1000.0;    
    // Debug: Test if ray should hit a sphere at origin
    // Manual sphere intersection for debugging
    float3 center = float3(0.0, 0.0, 0.0);
    float radius = 0.5;
    float3 oc = r.origin - center;
    float a = dot(r.direction, r.direction);
    float b = 2.0 * dot(oc, r.direction);
    float c = dot(oc, oc) - radius * radius;
    float disc = b * b - 4.0 * a * c;
    
    float3 color = float3(0.0, 0.0, 0.0); // Start black
    
    // Show ray origin as color (scaled and offset to be visible)
    // If camera is far from origin, you'll see colors
    color.r = abs(r.origin.x) * 0.1;
    color.g = abs(r.origin.y) * 0.1;
    color.b = abs(r.origin.z) * 0.1;
    
    // Show ray direction (normalized, so map from -1..1 to 0..1)
    color = r.direction * 0.5 + 0.5;
    
    // Create intersection query
    intersection_query<instancing> i;
    intersection_params params;
    
    // Initialize the query
    i.reset(r, accel, 1, params);
    
    // If we enter the loop, turn magenta
    while (i.next()) {
        color = float3(1.0, 0.0, 1.0); // Magenta = Metal found bounding box
    }
        
    float3 ray_origin = i.get_candidate_ray_origin();
    float3 ray_direction = i.get_candidate_ray_direction();
    float min_dist = i.get_ray_min_distance();
    float max_dist = i.get_committed_distance();
    
    BoundingBoxIntersection bb = intersectionFunctionTable[0](
        ray_origin,
        ray_direction,
        min_dist,
        max_dist
    );
    
    if (bb.accept) {
        i.commit_bounding_box_intersection(bb.distance);
        float distance = i.get_committed_distance();
        float3 hit = r.origin + r.direction * distance;
        float3 normal = normalize(hit);
        
        float3 light_dir = normalize(float3(1, 1, 1));
        float diffuse = max(0.2, dot(normal, light_dir));
        
        color = float3(0.7, 0.3, 0.5) * diffuse; // Final shaded color
    }
    
    output.write(float4(color, 1.0), tid);
}
