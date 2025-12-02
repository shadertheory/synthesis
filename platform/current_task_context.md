# Current Task Context

## 1. `clangd` Formatting Issues and Resolution

**Problem:** `clangd` was not applying the desired formatting, specifically regarding tab width, indentation, and function name line breaks. An error "Invalid fallback style: file" was observed in `clangd` logs.

**Actions Taken:**
*   Created/Modified `.clang-format` file in the project root (`/Volumes/Remote/rog-ally/.clang-format`) with the following settings:
    ```
    BasedOnStyle: Microsoft
    IndentWidth: 4
    TabWidth: 4
    UseTab: Always
    BreakBeforeBraces: Allman
    AllowShortIfStatementsOnASingleLine: false
    AllowShortLoopsOnASingleLine: false
    ColumnLimit: 120
    AccessModifierOffset: -4
    PointerAlignment: Left
    ReferenceAlignment: Left
    NamespaceIndentation: All
    IndentCaseLabels: true
    IndentExternBlock: Indent
    ReturnTypeBreakingStyle: None
    ```
*   Moved `.clang-format` from `/Volumes/Remote/rog-ally/platform/` to `/Volumes/Remote/rog-ally/` to ensure `clangd` could discover it for all project files.
*   Modified `~/.config/nvim/coc-settings.json` to fix `clangd.arguments`:
    *   Changed `--fallback-style=file` to `--fallback-style=Microsoft` to resolve the "Invalid fallback style" error.
    *   Removed a trailing comma in the JSON.

**Current Status:** The `.clang-format` file is correctly placed and configured. The `coc-settings.json` has been updated to use a valid fallback style. User needs to restart `coc.nvim` for changes to fully apply.

## 2. `key_code` Enum Expansion and i18n Support

**Problem:** Expand the `enum key_code` in `app/xbone/Main.cpp` to include a comprehensive set of keys for game input and discuss i18n support.

**Discussion & Recommendation:**
*   `key_code` enum should represent **physical key locations (scancodes)**, not characters, to ensure consistent game input logic across different keyboard layouts.
*   For **text input (i18n)**, the application should listen for OS-specific character events (e.g., `WM_CHAR` on Windows) which provide the actual Unicode characters based on the user's active keyboard layout.
*   For game actions, a keybinding system would map `key_code` to in-game actions. When displaying key names in UI, use OS APIs to convert `key_code` to localized key names.
*   **Alternative for Scancodes:** Instead of a large enum, a `typedef` to an integer type (e.g., `uint16_t`) combined with `constexpr` constants in a namespace (e.g., `namespace Key { constexpr KeyCode A = 0x1E; }`) is a more flexible and future-proof approach for managing raw scancodes, especially for multi-platform support. This allows for direct use of OS scancode values and easier data-driven configuration, while maintaining readability.

**Next Steps:** Implement the chosen approach for the `key_code` definition in `app/xbone/Main.cpp` based on the discussion (either expanding the enum or moving to `typedef` with constants).
