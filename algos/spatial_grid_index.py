import taichi as ti
import numpy as np

# Taichi initialization (global, executed once on module import)
try:
    ti.init(arch=ti.gpu, log_level=ti.WARN)
    print("Taichi initialized with GPU backend for SpatialGridIndex.")
except Exception:  # Broad exception to catch various Taichi init errors
    ti.init(arch=ti.cpu, log_level=ti.WARN)
    print("Taichi initialized with CPU backend for SpatialGridIndex.")

# --- KERNELS ---


@ti.kernel
def assign_points_to_grid_and_count_kernel(
    points_x: ti.types.ndarray(ti.f32, ndim=1),
    points_y: ti.types.ndarray(ti.f32, ndim=1),
    min_coord_x: ti.f32,
    min_coord_y: ti.f32,
    resolution: ti.f32,
    grid_dim_x: ti.i32,
    grid_dim_y: ti.i32,
    cell_point_counts: ti.template(),
):
    for i in range(points_x.shape[0]):
        px, py = points_x[i], points_y[i]
        cell_x = ti.floor((px - min_coord_x) / resolution)
        cell_y = ti.floor((py - min_coord_y) / resolution)
        if 0 <= cell_x < grid_dim_x and 0 <= cell_y < grid_dim_y:
            int_cell_x = ti.cast(cell_x, ti.i32)
            int_cell_y = ti.cast(cell_y, ti.i32)
            ti.atomic_add(cell_point_counts[int_cell_x, int_cell_y], 1)


@ti.kernel
def calculate_offsets_kernel(
    cell_point_counts: ti.template(),
    grid_dim_x: ti.i32,
    grid_dim_y: ti.i32,
    cell_offsets: ti.template(),
):
    current_offset = 0
    for j in range(grid_dim_y):  # Row-major
        for i in range(grid_dim_x):
            cell_offsets[i, j] = current_offset
            current_offset += cell_point_counts[i, j]


@ti.kernel
def populate_indexed_points_kernel(
    points_x: ti.types.ndarray(ti.f32, ndim=1),
    points_y: ti.types.ndarray(ti.f32, ndim=1),
    original_indices: ti.types.ndarray(ti.i32, ndim=1),
    min_coord_x: ti.f32,
    min_coord_y: ti.f32,
    resolution: ti.f32,
    grid_dim_x: ti.i32,
    grid_dim_y: ti.i32,
    cell_offsets: ti.template(),
    cell_current_insertion_counts: ti.template(),
    indexed_point_indices: ti.template(),
):
    for i in range(points_x.shape[0]):
        px, py = points_x[i], points_y[i]
        cell_x_float = ti.floor((px - min_coord_x) / resolution)
        cell_y_float = ti.floor((py - min_coord_y) / resolution)
        if 0 <= cell_x_float < grid_dim_x and 0 <= cell_y_float < grid_dim_y:
            cell_x = ti.cast(cell_x_float, ti.i32)
            cell_y = ti.cast(cell_y_float, ti.i32)
            base_offset = cell_offsets[cell_x, cell_y]
            local_idx = ti.atomic_add(cell_current_insertion_counts[cell_x, cell_y], 1)
            final_idx = base_offset + local_idx
            if final_idx < indexed_point_indices.shape[0]:  # Boundary check
                indexed_point_indices[final_idx] = original_indices[i]


# --- SPATIAL GRID INDEX CLASS ---


class SpatialGridIndex:
    def __init__(
        self,
        points_xy: np.ndarray,
        resolution: float,
        extent: tuple = None,
        min_points_per_cell_for_debug: int = 0,
    ):
        if (
            not isinstance(points_xy, np.ndarray)
            or points_xy.ndim != 2
            or points_xy.shape[1] != 2
        ):
            raise ValueError("points_xy must be a NumPy array of shape (N, 2).")
        if not isinstance(resolution, (float, int)) or resolution <= 0:
            raise ValueError("resolution must be a positive number.")

        self.resolution = float(resolution)
        self.num_points = points_xy.shape[0]

        if self.num_points == 0:
            # Handle empty input gracefully
            self.min_x, self.min_y, self.max_x, self.max_y = (0.0, 0.0, 0.0, 0.0)
            if extent:
                self.min_x, self.min_y, self.max_x, self.max_y = map(float, extent)
            self.grid_dim_x, self.grid_dim_y = (
                0,
                0,
            )  # Keep logical dimensions as 0 for empty case
            self.points_x_np = np.array([], dtype=np.float32)
            self.points_y_np = np.array([], dtype=np.float32)
            # Taichi fields need positive dimensions, so use 1x1 but never access them
            self.cell_point_counts = ti.field(dtype=ti.i32, shape=(1, 1))
            self.cell_offsets = ti.field(dtype=ti.i32, shape=(1, 1))
            self.indexed_point_indices = ti.field(dtype=ti.i32, shape=1)
            # Initialize to zero
            self.cell_point_counts.fill(0)
            self.cell_offsets.fill(0)
            self.indexed_point_indices.fill(0)
            return

        # 1. Determine extent and grid dimensions
        if extent:
            self.min_x, self.min_y, self.max_x, self.max_y = map(float, extent)
        else:
            self.min_x, self.min_y = np.min(points_xy, axis=0)
            self.max_x, self.max_y = np.max(points_xy, axis=0)
            # If all points are the same, max_x/max_y might be equal to min_x/min_y.
            # Add a small epsilon to max_x/max_y if they are same as min_x/min_y to ensure non-zero grid dimensions.
            if self.max_x == self.min_x:
                self.max_x += self.resolution
            if self.max_y == self.min_y:
                self.max_y += self.resolution

        self.grid_dim_x = max(
            1, int(np.ceil((self.max_x - self.min_x) / self.resolution))
        )
        self.grid_dim_y = max(
            1, int(np.ceil((self.max_y - self.min_y) / self.resolution))
        )

        # 2. Store points as float32 NumPy arrays and prepare original indices
        self.points_x_np = points_xy[:, 0].astype(np.float32)
        self.points_y_np = points_xy[:, 1].astype(np.float32)
        original_indices_np = np.arange(self.num_points, dtype=np.int32)

        # Convert to Taichi ndarrays for kernel calls
        points_x_ti = ti.ndarray(ti.f32, shape=self.num_points)
        points_y_ti = ti.ndarray(ti.f32, shape=self.num_points)
        original_indices_ti = ti.ndarray(ti.i32, shape=self.num_points)
        points_x_ti.from_numpy(self.points_x_np)
        points_y_ti.from_numpy(self.points_y_np)
        original_indices_ti.from_numpy(original_indices_np)

        # 3. Initialize Taichi fields
        self.cell_point_counts = ti.field(
            dtype=ti.i32, shape=(self.grid_dim_x, self.grid_dim_y)
        )
        self.cell_offsets = ti.field(
            dtype=ti.i32, shape=(self.grid_dim_x, self.grid_dim_y)
        )
        cell_current_insertion_counts = ti.field(
            dtype=ti.i32, shape=(self.grid_dim_x, self.grid_dim_y)
        )

        # 4. Call Kernel 1
        self.cell_point_counts.fill(0)
        assign_points_to_grid_and_count_kernel(
            points_x_ti,
            points_y_ti,
            self.min_x,
            self.min_y,
            self.resolution,
            self.grid_dim_x,
            self.grid_dim_y,
            self.cell_point_counts,
        )

        if min_points_per_cell_for_debug > 0:
            counts_np = self.cell_point_counts.to_numpy()
            if counts_np.max() > min_points_per_cell_for_debug:
                print(
                    f"Warning: Max points in a single cell is {counts_np.max()}, "
                    f"exceeding debug threshold {min_points_per_cell_for_debug}."
                )

        # 5. Call Kernel 2
        self.cell_offsets.fill(0)
        calculate_offsets_kernel(
            self.cell_point_counts, self.grid_dim_x, self.grid_dim_y, self.cell_offsets
        )

        # 6. Determine total indexed points and initialize indexed_point_indices
        # The total number of points is the offset of the last cell + count of the last cell
        # Or sum of all cell_point_counts
        if self.grid_dim_x > 0 and self.grid_dim_y > 0:
            # last_cell_offset_val = self.cell_offsets[self.grid_dim_x - 1, self.grid_dim_y - 1]
            # last_cell_count_val = self.cell_point_counts[self.grid_dim_x - 1, self.grid_dim_y - 1]
            # total_indexed_points = last_cell_offset_val + last_cell_count_val
            # Using sum of counts is more robust if grid can be very sparse or empty
            total_indexed_points = int(self.cell_point_counts.to_numpy().sum())

        else:  # Should not happen if num_points > 0 due to max(1, ..) for grid_dim
            total_indexed_points = 0

        self.indexed_point_indices = ti.field(
            dtype=ti.i32, shape=max(1, total_indexed_points)
        )  # Ensure positive shape
        if total_indexed_points > 0:
            self.indexed_point_indices.fill(-1)  # Sentinel for debugging
        else:
            self.indexed_point_indices.fill(0)  # Fill with dummy value for empty case

        # 7. Call Kernel 3
        if total_indexed_points > 0:
            cell_current_insertion_counts.fill(0)
            populate_indexed_points_kernel(
                points_x_ti,
                points_y_ti,
                original_indices_ti,
                self.min_x,
                self.min_y,
                self.resolution,
                self.grid_dim_x,
                self.grid_dim_y,
                self.cell_offsets,
                cell_current_insertion_counts,
                self.indexed_point_indices,
            )

    def get_cell_indices(self, x: float, y: float) -> tuple[int, int]:
        cell_idx = int(ti.floor((x - self.min_x) / self.resolution))
        cell_idy = int(ti.floor((y - self.min_y) / self.resolution))
        return cell_idx, cell_idy

    def query_points_in_cell(self, cell_x_idx: int, cell_y_idx: int) -> np.ndarray:
        # Handle empty index case first
        if self.num_points == 0:
            return np.array([], dtype=np.int32)

        if not (
            0 <= cell_x_idx < self.grid_dim_x and 0 <= cell_y_idx < self.grid_dim_y
        ):
            return np.array([], dtype=np.int32)

        if self.indexed_point_indices.shape[0] == 0:
            return np.array([], dtype=np.int32)

        start_offset = self.cell_offsets[cell_x_idx, cell_y_idx]
        count = self.cell_point_counts[cell_x_idx, cell_y_idx]

        if count == 0:
            return np.array([], dtype=np.int32)

        # Slice the Taichi field to get results.
        # Taichi field slicing `field[start:end]` creates a Taichi expression, not a NumPy array directly.
        # To get a NumPy array, we need to use .to_numpy() on the whole field and then slice,
        # or use a helper kernel to copy to a new (smaller) Taichi field/ndarray then .to_numpy().
        # For typical use cases where this slice isn't enormous, full .to_numpy() is simpler.
        all_indices_np = self.indexed_point_indices.to_numpy()
        return all_indices_np[start_offset : start_offset + count].copy()

    @ti.kernel
    def _radius_query_filter_kernel(
        self,
        num_candidates: ti.i32,
        candidate_indices: ti.types.ndarray(ti.i32, ndim=1),
        # Pass original points' coordinates (already stored as self.points_x_np, self.points_y_np)
        # Must be passed as Taichi ndarrays to the kernel.
        points_x_coords: ti.types.ndarray(ti.f32, ndim=1),
        points_y_coords: ti.types.ndarray(ti.f32, ndim=1),
        query_x: ti.f32,
        query_y: ti.f32,
        radius_sq: ti.f32,
        result_indices: ti.types.ndarray(ti.i32, ndim=1),  # Output buffer
        result_count: ti.template(),  # Atomic counter (ti.field(ti.i32, shape=()))
    ):
        for k in range(num_candidates):
            candidate_original_idx = candidate_indices[k]
            px = points_x_coords[candidate_original_idx]
            py = points_y_coords[candidate_original_idx]

            dist_sq = (px - query_x) ** 2 + (py - query_y) ** 2
            if dist_sq < radius_sq:
                current_idx = ti.atomic_add(result_count[None], 1)
                if current_idx < result_indices.shape[0]:
                    result_indices[current_idx] = candidate_original_idx
                # Else: more hits than buffer, data loss. Buffer should be num_candidates.

    def query_points_in_radius(
        self, query_x: float, query_y: float, radius: float
    ) -> np.ndarray:
        if self.num_points == 0 or radius <= 0:
            return np.array([], dtype=np.int32)

        min_search_x = query_x - radius
        max_search_x = query_x + radius
        min_search_y = query_y - radius
        max_search_y = query_y + radius

        min_cell_x, min_cell_y = self.get_cell_indices(min_search_x, min_search_y)
        max_cell_x, max_cell_y = self.get_cell_indices(max_search_x, max_search_y)

        min_cell_x = max(0, min_cell_x)
        min_cell_y = max(0, min_cell_y)
        max_cell_x = min(self.grid_dim_x - 1, max_cell_x)
        max_cell_y = min(self.grid_dim_y - 1, max_cell_y)

        candidate_indices_list = []
        for cur_cell_x in range(min_cell_x, max_cell_x + 1):
            for cur_cell_y in range(min_cell_y, max_cell_y + 1):
                # query_points_in_cell handles boundary checks internally,
                # but here cur_cell_x/y are already clamped.
                points_in_this_cell = self.query_points_in_cell(cur_cell_x, cur_cell_y)
                if points_in_this_cell.size > 0:
                    candidate_indices_list.append(points_in_this_cell)

        if not candidate_indices_list:
            return np.array([], dtype=np.int32)

        candidate_indices_np = np.concatenate(candidate_indices_list)
        if candidate_indices_np.size == 0:  # Should be caught by previous check
            return np.array([], dtype=np.int32)

        # Convert NumPy arrays to Taichi ndarrays for the kernel
        candidate_indices_ti = ti.ndarray(ti.i32, shape=candidate_indices_np.shape[0])
        candidate_indices_ti.from_numpy(candidate_indices_np)

        # These are the full original coordinate arrays
        points_x_coords_ti = ti.ndarray(ti.f32, shape=self.points_x_np.shape[0])
        points_y_coords_ti = ti.ndarray(ti.f32, shape=self.points_y_np.shape[0])
        points_x_coords_ti.from_numpy(self.points_x_np)
        points_y_coords_ti.from_numpy(self.points_y_np)

        # Output buffer for results from kernel, same size as candidates
        result_buffer_ti = ti.ndarray(ti.i32, shape=candidate_indices_np.shape[0])

        result_count_ti = ti.field(dtype=ti.i32, shape=())
        result_count_ti.fill(0)

        self._radius_query_filter_kernel(
            candidate_indices_np.shape[0],
            candidate_indices_ti,
            points_x_coords_ti,
            points_y_coords_ti,
            float(query_x),
            float(query_y),
            float(radius**2),
            result_buffer_ti,
            result_count_ti,
        )

        num_actual_hits = result_count_ti[None]
        # Convert Taichi ndarray (result_buffer_ti) back to NumPy array to slice it
        final_results_np = result_buffer_ti.to_numpy()

        return final_results_np[:num_actual_hits].copy()


# --- MAIN EXAMPLE USAGE (Updated for Class) ---
if __name__ == "__main__":
    print("\n--- Testing SpatialGridIndex ---")

    test_min_val_x, test_max_val_x = 0.0, 10.0
    test_min_val_y, test_max_val_y = 0.0, 10.0
    test_res = 1.0

    np.random.seed(0)  # For reproducibility
    num_base_points = 200
    test_points_xy = np.random.rand(num_base_points, 2).astype(np.float32)
    test_points_xy[:, 0] = (
        test_points_xy[:, 0] * (test_max_val_x - test_min_val_x) + test_min_val_x
    )
    test_points_xy[:, 1] = (
        test_points_xy[:, 1] * (test_max_val_y - test_min_val_y) + test_min_val_y
    )

    # Add specific points for targeted tests
    extra_points = np.array(
        [
            [test_min_val_x, test_min_val_y],  # Min corner
            [
                test_max_val_x - 1e-5,
                test_max_val_y - 1e-5,
            ],  # Near max corner (should be in last cell)
            [5.0, 5.0],  # Point for cell query
            [5.5, 5.5],  # Another point
            [test_max_val_x, test_max_val_y],  # Point exactly at max extent
        ],
        dtype=np.float32,
    )
    test_points_xy = np.vstack([test_points_xy, extra_points])
    num_test_points = test_points_xy.shape[0]
    print(f"Total test points: {num_test_points}")

    # Instantiate the index
    # Explicitly define extent to ensure points on max_x, max_y are handled by grid_dim calc
    # The grid covers [min_x, min_x + grid_dim_x * res), etc.
    # Points exactly on max_x or max_y might be tricky.
    # If max_x is 10.0 and res is 1.0, grid_dim_x should be 10.
    # A point at 10.0 would be cell_x = floor((10.0 - 0.0)/1.0) = 10.
    # This would be out of bounds if grid_dim_x is 10 (indices 0-9).
    # The `ceil` in `__init__` for grid_dim and the `< grid_dim` check in kernels should handle this.
    # Let's test with extent slightly larger if points can be ON max_x, max_y.
    # Or rely on the current cell calculation: cell_x = floor(...)
    # If extent is (0,0,10,10) and res=1, grid_dim=10. Point at 10 gives cell_idx 10.
    # Kernel checks `0 <= cell_x < grid_dim_x`. So point at 10 is excluded.
    # If we want to include points *at* max_x, max_y, the extent should effectively be slightly larger
    # OR the comparison in kernel should be `0 <= cell_x <= grid_dim_x -1` and `px <= max_x` etc.
    # The current implementation using `np.ceil((self.max_x - self.min_x) / self.resolution)` for grid_dim
    # and `0 <= cell_x < grid_dim_x` in kernel is standard. Points exactly on max_x/max_y are outside.
    # If a point is exactly `max_x`, `(px - min_coord_x) / resolution` can be `grid_dim_x`, making it out of bounds.
    # This is generally acceptable.

    # Test extent definition
    # If point is exactly at max_x, e.g. 10.0, cell_x = floor((10.0 - 0.0)/1.0) = 10.
    # If grid_dim_x = ceil((10.0-0.0)/1.0) = 10. Then cell_x=10 is out of bounds (0-9).
    # So points at the max boundary are excluded by current logic.
    # Let's define extent carefully:
    effective_max_x = (
        test_max_val_x + test_res * 0.1
    )  # Ensure grid covers up to max_val_x
    effective_max_y = test_max_val_y + test_res * 0.1
    # However, the points are generated up to test_max_val_x.
    # The class calculates its own max_x, max_y if extent is None.
    # If a point is (10,10) and max_x from points is 10, grid_dim_x = ceil((10-0)/1) = 10.
    # cell_x for point 10 is floor((10-0)/1) = 10. This is outside grid cell_point_counts[0..9].
    # This means points exactly on the maximum boundary of the point cloud will be excluded.
    # This is a common behavior.

    spatial_index = SpatialGridIndex(
        test_points_xy,
        test_res,
        extent=(test_min_val_x, test_min_val_y, test_max_val_x, test_max_val_y),
        min_points_per_cell_for_debug=50,
    )

    print(
        f"Grid definition: min=({spatial_index.min_x:.2f}, {spatial_index.min_y:.2f}), max=({spatial_index.max_x:.2f}, {spatial_index.max_y:.2f})"
    )
    print(
        f"Grid dimensions: ({spatial_index.grid_dim_x}, {spatial_index.grid_dim_y}) cells"
    )

    counts_sum = np.sum(spatial_index.cell_point_counts.to_numpy())
    print(f"Total indexed points (sum of cell_point_counts): {counts_sum}")
    if (
        spatial_index.indexed_point_indices.shape[0] > 0 or counts_sum > 0
    ):  # Check if shape[0] is valid
        print(
            f"Length of final indexed_point_indices array: {spatial_index.indexed_point_indices.shape[0]}"
        )
        assert counts_sum == spatial_index.indexed_point_indices.shape[0], (
            "Sum of counts must match length of indexed array"
        )

    # Test query_points_in_cell
    test_cell_x, test_cell_y = 5, 5
    print(f"\nQuerying points in cell ({test_cell_x}, {test_cell_y}):")
    points_in_cell = spatial_index.query_points_in_cell(test_cell_x, test_cell_y)
    print(f"Found {points_in_cell.size} points: {points_in_cell}")
    # Verification
    if points_in_cell.size > 0:
        for pt_idx in points_in_cell:
            px, py = test_points_xy[pt_idx, 0], test_points_xy[pt_idx, 1]
            actual_cx, actual_cy = spatial_index.get_cell_indices(px, py)
            if not (actual_cx == test_cell_x and actual_cy == test_cell_y):
                print(
                    f"  VERIFICATION ERROR: Point {pt_idx} ({px:.2f}, {py:.2f}) is in cell ({actual_cx},{actual_cy}), not ({test_cell_x},{test_cell_y})"
                )

    # Test query_points_in_cell for an out-of-bounds cell
    oob_cell_x = spatial_index.grid_dim_x
    oob_cell_y = spatial_index.grid_dim_y
    print(f"\nQuerying points in out-of-bounds cell ({oob_cell_x}, {oob_cell_y}):")
    points_in_oob_cell = spatial_index.query_points_in_cell(oob_cell_x, oob_cell_y)
    print(f"Found {points_in_oob_cell.size} points: {points_in_oob_cell} (expected 0)")
    assert points_in_oob_cell.size == 0

    # Test query_points_in_radius
    query_cx, query_cy = 5.2, 5.3
    query_r = 1.5
    print(
        f"\nQuerying points in radius: center=({query_cx:.2f}, {query_cy:.2f}), radius={query_r:.2f}"
    )

    points_in_radius = spatial_index.query_points_in_radius(query_cx, query_cy, query_r)
    print(f"Found {points_in_radius.size} points within radius: {points_in_radius}")

    # Verification for radius query
    manual_check_indices = []
    for pt_idx in range(num_test_points):
        px, py = test_points_xy[pt_idx, 0], test_points_xy[pt_idx, 1]
        dist_sq = (px - query_cx) ** 2 + (py - query_cy) ** 2
        if dist_sq < query_r**2:
            manual_check_indices.append(pt_idx)

    print(f"Manual check found {len(manual_check_indices)} points.")
    # Sort both arrays before comparison for set equality
    if not np.array_equal(
        np.sort(points_in_radius), np.sort(np.array(manual_check_indices))
    ):
        print("  VERIFICATION ERROR: Radius query result mismatch with manual check.")
        print(f"    Kernel results: {np.sort(points_in_radius)}")
        print(f"    Manual results: {np.sort(np.array(manual_check_indices))}")
    else:
        print("  Radius query result matches manual check.")

    # Test radius query that should yield no results (center far away)
    query_far_x, query_far_y = 100.0, 100.0
    query_small_r = 0.1
    print(
        f"\nQuerying points in radius (expected empty): center=({query_far_x}, {query_far_y}), radius={query_small_r}"
    )
    points_in_far_radius = spatial_index.query_points_in_radius(
        query_far_x, query_far_y, query_small_r
    )
    print(
        f"Found {points_in_far_radius.size} points: {points_in_far_radius} (expected 0)"
    )
    assert points_in_far_radius.size == 0

    # Test with zero points
    print("\n--- Testing SpatialGridIndex with zero points ---")
    empty_points = np.array([[]], dtype=np.float32).reshape(0, 2)
    empty_index = SpatialGridIndex(empty_points, resolution=1.0)
    print(
        f"Grid dimensions for empty index: ({empty_index.grid_dim_x}, {empty_index.grid_dim_y})"
    )
    empty_rad_res = empty_index.query_points_in_radius(0, 0, 1)
    print(f"Radius query on empty index: {empty_rad_res.size} points (expected 0)")
    assert empty_rad_res.size == 0
    empty_cell_res = empty_index.query_points_in_cell(0, 0)
    print(f"Cell query on empty index: {empty_cell_res.size} points (expected 0)")
    assert empty_cell_res.size == 0

    print("\n--- Testing SpatialGridIndex with one point ---")
    one_point = np.array([[1.5, 2.5]], dtype=np.float32)
    one_point_index = SpatialGridIndex(one_point, resolution=1.0, extent=(0, 0, 5, 5))
    print(
        f"Grid for one point: dim=({one_point_index.grid_dim_x}, {one_point_index.grid_dim_y}), total_idx_pts={one_point_index.indexed_point_indices.shape[0]}"
    )
    assert one_point_index.indexed_point_indices.shape[0] == 1
    # Cell for (1.5, 2.5) with min (0,0), res 1.0 is (1,2)
    pts_in_cell_1_2 = one_point_index.query_points_in_cell(1, 2)
    print(f"Query cell (1,2) for one point: {pts_in_cell_1_2} (expected [0])")
    assert np.array_equal(pts_in_cell_1_2, np.array([0]))
    pts_in_rad = one_point_index.query_points_in_radius(1.5, 2.5, 0.1)
    print(f"Query radius at (1.5, 2.5) for one point: {pts_in_rad} (expected [0])")
    assert np.array_equal(pts_in_rad, np.array([0]))
    pts_in_rad_miss = one_point_index.query_points_in_radius(3.5, 3.5, 0.1)
    print(f"Query radius at (3.5, 3.5) for one point: {pts_in_rad_miss} (expected [])")
    assert pts_in_rad_miss.size == 0

    print("\n--- SpatialGridIndex tests completed ---")
    # Note: Taichi messages about ndarray arguments being automatically converted to Taichi fields are normal.
    # e.g. "Argument 'candidate_indices' is a numpy array. It is automatically converted to a Taichi field..."
    # These are info messages, not errors.
    # Using ti.init(log_level=ti.WARN) reduces these.
