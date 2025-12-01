# The Universal Simulation Loop

The engine's frame execution is a structured 3-stage compute pipeline executed entirely on the GPU. It replaces the traditional `Update()` -> `Render()` loop with a massive parallel simulation.

## Stage 1: Integrate (The "Move" Kernel)
This stage predicts the future state of agents based on forces and their current momentum.

*   **Input:** Agent Buffer (Read/Write), Global Uniforms (Gravity, Time).
*   **Logic:**
    1.  **Global Forces:** Apply gravity (`F = m*g`).
    2.  **Local Forces:**
        *   *Fluid:* Apply pressure and viscosity forces (calculated from neighbors).
        *   *Horde:* Apply flocking cohesion and separation vectors.
    3.  **Integration:** Semi-Implicit Euler integration:
        *   `velocity += acceleration * dt`
        *   `position += velocity * dt`
*   **Output:** The *Predicted* Position for the next frame.

## Stage 2: Sense (The "Raytrace" Kernel)
This stage validates the predicted movement against the Scene (The Truth).

*   **Input:** Agent Buffer (Predicted Position), TLAS (Scene).
*   **Logic:**
    1.  Every active Agent generates a **Request** (Ray).
    2.  **Ray Generation Strategy:**
        *   *Phonons/Physics:* Cast ray from `CurrentPos` to `PredictedPos`.
        *   *Horde:* Cast ray towards `TargetID` (Line-of-sight) or downwards (Ground grounding).
    3.  **Traversal:** The GPU hardware raytracer (Metal/DXR) traverses the TLAS.
*   **Output:** **Hit** Buffer containing:
    *   `Distance` (t-value)
    *   `Normal` (Surface normal)
    *   `MaterialID` (Index into Behavior buffer)

## Stage 3: Resolve (The "React" Kernel)
This stage reconciles the prediction with the sensor data, applying collision response and state mutation.

*   **Input:** Hit Buffer, Agent Buffer, Behavior Buffer.
*   **Logic:**
    1.  **Collision Response:**
        *   If `Hit.distance < MoveDistance`:
            *   **Reflect:** `velocity = reflect(velocity, Hit.normal)`.
            *   **Damp:** `velocity *= Behavior.restitution`.
            *   **Clamp:** `position = Hit.position + Hit.normal * epsilon`.
    2.  **State Mutation (Type-Specific):**
        *   *Phonon:* `energy *= (1.0 - Behavior.absorption)`. If `energy < epsilon`, mark dead.
        *   *Horde:* If `Hit.normal.y > 0.7` (Floor), reset "Grounded" timer. If `Hit.target` is visible, increase "Aggression".
*   **Output:** The Final Agent Buffer for the frame.
