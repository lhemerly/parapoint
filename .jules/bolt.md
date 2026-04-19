## 2026-04-19 - SpatialGridIndex Query Performance
**Learning:** In Taichi and numpy heavy codebase, `.to_numpy()` and `.from_numpy()` and repeated `ti.ndarray` allocations inside frequently called functions (like spatial index queries) are major performance bottlenecks.
**Action:** When implementing tight loops, pre-allocate `ti.ndarray` buffers as class attributes and cache `.to_numpy()` outputs if the underlying data is immutable.
