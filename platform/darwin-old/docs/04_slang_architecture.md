# Slang as the Universal Language

To achieve true cross-platform compatibility (Metal for Apple, DirectX 12 for Windows/Xbox) without maintaining duplicate shader codebases, we utilize **Slang**. Slang allows us to write the simulation logic once and compile it to the native shader language of the target platform.

## 1. The Abstraction Layer (`IRayTracer`)
We define a Slang interface that abstracts the hardware-specific intrinsics of raytracing.

```csharp
// defined in scene.slang
interface IRayTracer {
    Hit trace(RayDesc ray, uint mask);
}
```

*   **Metal Implementation:** Wraps `metal::raytracing`. It constructs an `intersector<instancing>` object, sets the acceleration structure, and calls `intersect()`. It converts the Metal `intersection_result` into our generic `Hit` struct.
*   **DirectX 12 Implementation:** Wraps `RayQuery<RAY_FLAG_NONE>`. It calls `TraceRayInline()` (Inline Raytracing). This is preferred over the Shader Binding Table (SBT) approach for this specific architecture because our intersection logic is simple (all materials are in a global buffer), avoiding the overhead of complex shader tables.

## 2. The Module System
The shader codebase is organized into modular units:

*   **`types.slang`:**
    *   Contains `struct Agent`, `struct Behavior`, etc.
    *   Designed to be byte-compatible with C/C++/Rust via `#[repr(C)]`.
    *   Serves as the "Contract" between CPU and GPU.
*   **`scene.slang`:**
    *   Defines the `IRayTracer` interface.
    *   Defines `GlobalParams` (Time, Frame Count).
    *   Provides helper functions for reading the `Behavior` buffer.
*   **`kernels.slang`:**
    *   Contains the actual `[shader("compute")]` entry points: `simulateAgents`, `renderView`, etc.
    *   Imports `scene` and `types`.
    *   **Platform Agnostic:** This file contains *zero* platform-specific ifdefs. It purely describes the simulation logic using the provided interfaces.

## 3. Compilation Pipeline
The Rust build system (`build.rs`) automates the compilation:

1.  **Detection:** Detects the target OS (`target_os = "macos"` vs `windows`).
2.  **Invocation:** Calls the Slang compiler (`slangc`).
    *   **MacOS:** `slangc kernels.slang -target metal -o kernels.metallib`
    *   **Windows:** `slangc kernels.slang -target dxil -o kernels.dxil`
3.  **Binding:** The engine loads the resulting binary blob at runtime. The Rust `engine-hal` layer maps the Slang entry point bindings (registers/buffers) to the native API slots (Metal Argument Buffers or DX12 Root Signatures).
