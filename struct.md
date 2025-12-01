
   /
   ├── Cargo.toml (Workspace root)
   ├── core/               # (New) Platform-agnostic Logic
   │   ├─│   │   ├── math/       # shared math types
   │     │   └── Cargo.toml
   ├── backends/
   │   ├── metal/          # (New) Refactored darwin.rs + .metal files
   │   ├── dx12/           # (New) Rust port of xbone rendering logic
   │   └── vulkan/         # (Future)
   ├── engine/             # High-level glue
   │   ├]
   │   └── ...
   ├── apps/
   │   ├── ios-tower/      # Existing 'tower' project
   │   └── win-xbone/      # 'xbone' project, but stripped of logic, just bootstraps Rust
