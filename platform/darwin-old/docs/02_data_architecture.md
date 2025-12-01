# Data Architecture

To support the Universal Simulation Engine, the data architecture prioritizes GPU occupancy, memory coalescing, and unified access patterns.

## 1. The Unified Agent Struct (`Agent`)
To ensure maximum parallel throughput, all agents share a "base" structure. Specialized data is handled via extension buffers, but the core physics/movement logic operates on this common denominator.

**Memory Layout (SoA/AoS Hybrid):**
Designed to align to 64 bytes (cache line friendly).

```rust
// The Base "Soul"
struct Agent {
    vec3 position;   // 12 bytes
    float energy;    // 4 bytes (Mass / Loudness / Health)
    vec3 direction;  // 12 bytes (Velocity Vector)
    float time;      // 4 bytes (Accumulated Lifetime / Phase)
    uint type;       // 4 bytes (Discriminator: 0=Phonon, 1=Fluid, 2=Horde)
    uint flags;      // 4 bytes (State Mask / Active Status)
    uint3 next;      // 12 bytes (Spatial Hashing / Linked List indices)
    uint padding;    // 4 bytes
}
```

*   **Divergence Handling:** A GPU Radix Sort is applied to the Agent buffer every frame, sorting by `type`. This ensures that a GPU warp (32 threads) processes only one type of agent (e.g., only Phonons), eliminating instruction divergence penalties during the "React" phase.

## 2. Specialized State Buffers
While the base `Agent` struct handles movement and basic interaction, specific behaviors require extra data. These are stored in parallel StructuredBuffers.

*   **FluidState:** `pressure` (float), `density` (float), `vorticity` (vec3).
*   **HordeState:** `target_id` (uint), `aggression` (float), `flock_vector` (vec3).

## 3. The Universal Material (`Behavior`)
Instead of separate "Physics Materials" (friction/restitution) and "Render Materials" (albedo/roughness), we define a holistic material structure. This allows a single ray hit to resolve all interaction types.

```rust
struct Behavior {
    float restitution;   // Physics: Bounciness (0.0 - 1.0)
    float friction;      // Physics: Slide resistance
    float absorption;    // Audio: Damping factor (0.0 = reflect, 1.0 = silent)
    float scattering;    // Audio/Light: Diffusion factor
    float transmission;  // Audio/Light: Transparency/Wall-banging capability
    float refraction;    // Fluid/Light: Index of Refraction
    uint mask;           // Collision Layers (Bitmask)
}
```
