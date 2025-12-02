When planning gameplay tasks, adhere to this heuristic:

1.  **Define the Simulation Truth (The 1Hz Tick):**
    *   How does this mechanic function in the deterministic server tick?
    *   Does it involve the **Macro** or **Micro** layer?
    *   How does it affect the **ECS Hierarchy** (Parent/Child entities)?

2.  **Define the Interpolated Beauty (The 60fps Frame):**
    *   How does the client render the gap between ticks?
    *   Define the **Prediction Logic**: What does the "Ghost" state look like?
    *   Define the **Confirmation**: How does the world react when the server validates the move?

3.  **Define the Invisible Interaction:**
    *   How does the player trigger this *without a menu*?
    *   Input must be converted to **World Space Intent** immediately.

4.  **Define the Systemic Ripple:**
    *   **Resonance:** Does this action create sound? (Raytraced Audio).
    *   **Photon:** Does this action emit light? (Raytraced Vision).
    *   **Scent:** Does this release particles?
    *   *Crucial:* How will the Zombies react to these stimuli?

5.  **Define the Progression:**
    *   Is this a "Scavenged" capability?
    *   Does it scale from "Hand-Built" to "Automated"?
