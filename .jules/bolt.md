
## 2024-04-21 - Caching Taichi fields as NumPy arrays for rapid Python lookups
**Learning:** Calling `.to_numpy()` on a Taichi field inside a tight loop or frequently called function (like `query_points_in_cell`) is extremely slow due to repeated data conversions and allocations.
**Action:** When a Taichi field's data becomes immutable after an initial build phase (e.g. `indexed_point_indices` in spatial indexes), cache its `.to_numpy()` representation as an instance attribute (`self.indexed_point_indices_np`) and perform queries against the cached NumPy array instead of the Taichi field.
