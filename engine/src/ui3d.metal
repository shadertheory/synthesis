#include <metal_stdlib>
using namespace metal;

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
struct Ray {
    float3 origin;
    float3 direction;
};

struct Plane {
    float3 point;   // A point on the plane
    float3 normal;  // Normal vector (should be normalized)
};
float4 intersect_plane(Ray ray, Plane plane) {
    // Plane equation: dot(normal, P - point) = 0
    // Ray equation: P(t) = origin + t * direction
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
    texture2d<float, access::write> output [[texture(0)]],
    device CameraData* cameras [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    CameraData camera = cameras[0];
    float3 origin = float3(camera.transform[3].xyz);
    float native_scale = camera.resolution.z;
    float2 uv = float2(gid) / (camera.resolution.xy * native_scale);


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

Ray ray;
ray.origin = nearWorld.xyz;
ray.direction = normalize(nearWorld.xyz - camera.transform[3].xyz); 

    Plane plane;
    plane.point = (float4(0, 0, 0.0, 1)).xyz;
    plane.normal = (float4(0, 0, 1, 0)).xyz;
       
       
    float4 intersection =   intersect_plane(ray, plane);

   // Offset the grid by camera position so it appears to move with you


    float2 grid = fract(intersection.xy);
    
    float2 distEdge = min(grid, 1.0 - grid);
    float invEdge = 2 * max(distEdge.x, distEdge.y);
    float alpha = invEdge > 0.96 ? 1.0 : 0.0;

   
    output.write(float4(float3(alpha), 1.0), gid);

}
