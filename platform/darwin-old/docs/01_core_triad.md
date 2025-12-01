# The Core Triad: Scene, Observer, Agent

The architecture of the Universal Simulation Engine is built upon a fundamental triangular relationship of data flow. This model unifies distinct domains (Rendering, Audio, Physics, AI) into a single cohesive system.

## A. The Agent (The Actor)
The Agent is the fundamental atom of the universe. Unlike traditional engines that segregate `Particle` classes for VFX, `AudioSource` classes for sound, and `RigidBody` classes for physics, this engine defines a single, polymorphic GPU structure.

*   **Phonons (Audio):** Agents that carry acoustic energy, bounce off walls, and travel at the speed of sound. Their simulation drives the acoustic properties of the environment (reverb, occlusion).
*   **Photons (Light):** Agents (or transient rays) that carry radiance, bounce instantaneously (per frame), and travel at $c$. Their accumulation produces the visual frame.
*   **Fluid Atoms (Physics):** Agents that carry pressure, vorticity, and density. They interact with each other (via Smoothed Particle Hydrodynamics or Grid-based methods) and with the Scene boundaries.
*   **Hordes (AI):** Agents that carry intent (aggression, target ID), internal state timers, and flocking vectors. They interact with the Scene to navigate (pathfind) and with each other to flock.

## B. The Scene (The Truth)
The Scene is the unified spatial representation of the world, serving as the single source of truth for all interactions.

*   **Representation:** It is primarily represented by the **TLAS (Top-Level Acceleration Structure)** on the GPU.
*   **Content:**
    *   **Static Geometry:** Walls, terrain, buildings (triangle meshes).
    *   **Dynamic Bodies:** Moving platforms, doors, rigid bodies.
*   **Role:** It is the *only* entity in the simulation that can be "hit" by a ray.
*   **Material System (`Behavior`):** It provides the lookup table for interactions. When an Agent hits the Scene, the Scene provides a `Behavior` struct defining properties like restitution (bounciness), absorption (audio damping), and friction.

## C. The Observer (The Sensor)
The Observer is the window into the simulation. It does not simulate or alter the world; it *measures* the state of Agents to produce output for the user.

*   **Camera:** Measures the accumulation of Photons (or casts reverse rays into the scene) to produce a 2D image buffer for the screen.
*   **Microphone:** Measures the density, energy, and directionality of Phonons within a specific spatial radius to produce a PCM audio buffer.
*   **Trigger/Sensor:** Measures the presence or density of specific Agent types (e.g., Hordes) in a region to trigger game logic events (e.g., "Player detected").
