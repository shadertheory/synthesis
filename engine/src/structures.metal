#pragma once

#include <metal_stdlib>
using namespace metal;

struct ListenerGPU {
    packed_float3 position;
    packed_float3 up;
    packed_float3 right;
    float radius; // Listener head radius for intersection
};

struct SourceGPU {
    packed_float3 position;
    packed_float3 color;
    float volume_db;
    float computed_radius;
    uint id;
    uint active;
};

// --- OCCLUSION STRUCTS ---
struct OcclusionResultGPU {
    float transmission_factor; // 0.0 (Blocked) to 1.0 (Clear)
};

// --- PHONON STRUCTS ---
struct Phonon {
    packed_float3 position;
    packed_float3 direction;
    float energy;
    float total_distance;
    uint source_id;
    uint active; 
};

struct PhononHit {
    float total_distance; // Determines delay
    float energy;         // Determines volume
    packed_float3 direction; // Determines binaural panning
    uint source_id;
};

struct AtomicCounter {
    atomic_uint value;
};

struct IndirectDispatchArgs {
    packed_uint3 threadgroups_per_grid;
};

