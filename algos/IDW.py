import numpy as np
import taichi as ti # Import the Taichi library
import time # For basic timing

# --- Taichi Initialization ---
# You can choose the backend. ti.gpu is usually faster if you have a compatible GPU.
# If not, ti.cpu will use multi-core CPU.
try:
    ti.init(arch=ti.gpu, log_level=ti.WARN) # Use WARN to reduce Taichi's own log verbosity
    print("Taichi initialized with GPU backend.")
except Exception as e_gpu:
    print(f"GPU backend for Taichi failed: {e_gpu}")
    print("Falling back to CPU backend for Taichi.")
    ti.init(arch=ti.cpu, log_level=ti.WARN)
    print("Taichi initialized with CPU backend.")

# --- Taichi Kernel for IDW ---
@ti.kernel
def idw_interpolation_kernel(
    points_x: ti.types.ndarray(ti.f32, ndim=1), # Input: X coordinates of points
    points_y: ti.types.ndarray(ti.f32, ndim=1), # Input: Y coordinates of points
    points_z: ti.types.ndarray(ti.f32, ndim=1), # Input: Z coordinates of points
    dtm_field: ti.template(),          # Output: Taichi field to store DTM Z values
    min_x_dtm: ti.f32,                 # DTM grid minimum X
    min_y_dtm: ti.f32,                 # DTM grid minimum Y
    resolution_dtm: ti.f32,            # DTM grid resolution
    grid_width: ti.i32,                # DTM grid width in cells
    grid_height: ti.i32,               # DTM grid height in cells
    search_radius: ti.f32,             # Max distance to look for points
    power_p: ti.f32,                   # Power parameter for IDW
    nodata_value_kernel: ti.f32        # Nodata value to assign
):
    """
    Performs IDW interpolation for each cell in the DTM grid.
    Iterates through DTM cells in parallel.
    """
    num_points = points_x.shape[0]

    # Parallel loop over DTM grid cells (Taichi handles the parallelization)
    for r, c in ti.ndrange(grid_height, grid_width): # r = row (y_idx), c = col (x_idx)
        # Calculate the center coordinates of the current DTM cell
        cell_center_x = min_x_dtm + (ti.cast(c, ti.f32) + 0.5) * resolution_dtm
        cell_center_y = min_y_dtm + (ti.cast(r, ti.f32) + 0.5) * resolution_dtm

        sum_weighted_z = 0.0
        sum_weights = 0.0
        exact_match_z = -1.0 # Sentinel for exact match
        found_exact_match = False

        # Iterate through all input points to find neighbors for this cell
        # This is the computationally intensive part.
        # For very large point clouds, a spatial index (e.g., grid pre-binning)
        # would be needed here for better performance.
        for i in range(num_points):
            px, py, pz = points_x[i], points_y[i], points_z[i]
            
            # Calculate squared distance first to avoid sqrt until necessary
            dist_sq = (px - cell_center_x)**2 + (py - cell_center_y)**2

            if dist_sq == 0.0: # Exact match at cell center
                exact_match_z = pz
                found_exact_match = True
                break # No need to check other points if we have an exact match

            if dist_sq < search_radius**2:
                dist = ti.sqrt(dist_sq)
                if dist > 1e-6: # Avoid division by zero if point is extremely close but not exact
                    weight = 1.0 / (dist**power_p)
                    sum_weighted_z += weight * pz
                    sum_weights += weight
        
        if found_exact_match:
            dtm_field[r, c] = exact_match_z
        elif sum_weights > 1e-6: # If any suitable neighbors were found
            dtm_field[r, c] = sum_weighted_z / sum_weights
        else: # No points found within search_radius
            dtm_field[r, c] = nodata_value_kernel


# --- Main Python Function for IDW ---
def create_dtm_with_taichi_idw(
    ground_points_xyz: np.ndarray, # NumPy array of shape (N, 3) with X, Y, Z columns
    dtm_resolution: float,
    search_radius: float,
    power: float = 2.0,
    dtm_extent_user: tuple = None, # Optional: (min_x, min_y, max_x, max_y)
    nodata_value: float = -9999.0
) -> np.ndarray:
    """
    Creates a DTM NumPy array from ground points using Taichi for IDW interpolation.

    Args:
        ground_points_xyz: NumPy array of shape (N,3) containing X, Y, Z of ground points.
        dtm_resolution: The desired resolution of the output DTM grid.
        search_radius: Maximum distance from a DTM cell center to consider input points.
        power: The power parameter for IDW (typically 1, 2, or 3).
        dtm_extent_user: Optional tuple (min_x, min_y, max_x, max_y) to define the DTM area.
                         If None, it's calculated from the input points.
        nodata_value: Value to use for DTM cells with no underlying points within search radius.

    Returns:
        A 2D NumPy array representing the DTM (rows, cols) -> (height, width).
    """
    print(f"Starting DTM creation with Taichi (IDW): {ground_points_xyz.shape[0]} points, res {dtm_resolution}, radius {search_radius}, power {power}")
    start_time = time.time()

    if ground_points_xyz.shape[0] == 0:
        print("Warning: No ground points provided. Returning empty DTM.")
        return np.array([[]], dtype=np.float32)

    # Ensure input is float32 for Taichi kernel
    points_x_np = ground_points_xyz[:, 0].astype(np.float32)
    points_y_np = ground_points_xyz[:, 1].astype(np.float32)
    points_z_np = ground_points_xyz[:, 2].astype(np.float32)

    # Determine DTM extent
    if dtm_extent_user:
        min_x, min_y, max_x, max_y = dtm_extent_user
    else:
        min_x = np.min(points_x_np)
        min_y = np.min(points_y_np)
        max_x = np.max(points_x_np)
        max_y = np.max(points_y_np)
    
    print(f"DTM Extent: X({min_x:.2f} - {max_x:.2f}), Y({min_y:.2f} - {max_y:.2f})")

    # Calculate DTM grid dimensions
    if ground_points_xyz.shape[0] == 0: # Should have been caught by earlier check, but defensive
        grid_width = 0
        grid_height = 0
    else:
        grid_width = max(1, int(np.ceil((max_x - min_x) / dtm_resolution)))
        grid_height = max(1, int(np.ceil((max_y - min_y) / dtm_resolution)))
    
    # Adjust max_x, max_y to be the actual extent of the grid for consistency
    # This is important for calculating cell centers accurately.
    actual_max_x = min_x + grid_width * dtm_resolution
    actual_max_y = min_y + grid_height * dtm_resolution


    if grid_width <= 0 or grid_height <= 0:
        print("Error: Calculated DTM grid dimensions are invalid (<=0). Check resolution and point extent.")
        return np.array([[]], dtype=np.float32)
        
    print(f"DTM Grid Dimensions: {grid_width} (width) x {grid_height} (height) cells")

    # Initialize Taichi field for the DTM output
    # DTM shape is (height, width) which corresponds to (num_rows, num_cols)
    dtm_output_field = ti.field(dtype=ti.f32, shape=(grid_height, grid_width))
    dtm_output_field.fill(nodata_value) # Initialize with nodata

    # Call the Taichi kernel
    print("Launching Taichi IDW kernel...")
    kernel_start_time = time.time()
    idw_interpolation_kernel(
        points_x_np, points_y_np, points_z_np,
        dtm_output_field,
        min_x, min_y, dtm_resolution, # Use original min_x, min_y for origin
        grid_width, grid_height,
        search_radius,
        power,
        nodata_value # Pass nodata_value to kernel
    )
    ti.sync() # Ensure kernel execution finishes before proceeding
    kernel_end_time = time.time()
    print(f"Taichi kernel execution finished in {kernel_end_time - kernel_start_time:.2f} seconds.")

    # Convert Taichi field back to NumPy array
    dtm_np = dtm_output_field.to_numpy()
    
    total_time = time.time() - start_time
    print(f"IDW DTM creation complete in {total_time:.2f} seconds.")
    return dtm_np

# --- Example Usage ---
def run_idw_example(
    num_sample_points=500, # Reduced default for faster example/testing
    extent_size=20.0,    # Reduced default
    resolution=1.0,
    idw_search_radius=5.0,
    idw_power=2.0,
    dtm_extent_user=None,
    nodata_value=-9999.0,
    verbose=True # Control print statements for testing
):
    """Runs a full example of IDW DTM generation."""
    if verbose:
        print("--- Running Taichi DTM IDW Interpolation Example ---")
    
    # 1. Create some sample ground points
    sample_points = np.random.rand(num_sample_points, 3).astype(np.float32) * extent_size
    if num_sample_points > 0: # Avoid division by zero if num_sample_points is 0
        sample_points[:, 2] = (np.sin(sample_points[:, 0] / (extent_size/10 if extent_size > 0 else 1.0)) * 10 +
                               np.cos(sample_points[:, 1] / (extent_size/10 if extent_size > 0 else 1.0)) * 10 +
                               sample_points[:,0] * 0.05 + # Gentle slope
                               np.random.rand(num_sample_points) * 2) # Some noise
    
    if verbose:
        print(f"Generated {sample_points.shape[0]} sample points.")

    # 2. Create the DTM
    dtm_array = create_dtm_with_taichi_idw(
        sample_points,
        resolution,
        idw_search_radius,
        power=idw_power,
        dtm_extent_user=user_defined_extent,
        nodata_value=nodata_value
    )

    # 3. Output information about the DTM (if verbose)
    if verbose:
        if dtm_array.size > 0:
            print(f"\nGenerated IDW DTM shape: {dtm_array.shape}") # (height, width)
            valid_dtm_values = dtm_array[dtm_array != nodata_value]
            if valid_dtm_values.size > 0:
                print(f"DTM min value (excluding NoData): {np.min(valid_dtm_values):.2f}")
                print(f"DTM max value (excluding NoData): {np.max(valid_dtm_values):.2f}")
            else:
                print("DTM contains only NoData values.")
            num_nodata_cells = np.sum(dtm_array == nodata_value)
            print(f"Number of NoData cells: {num_nodata_cells} out of {dtm_array.size} total cells ({num_nodata_cells/dtm_array.size*100:.1f}%)")
        else:
            print("IDW DTM generation resulted in an empty array.")
    
    return dtm_array

if __name__ == "__main__":
    # Run with default parameters for the main execution.
    # These can be larger/more demanding than the test version.
    run_idw_example(
        num_sample_points=500000,
        extent_size=200.0,
        resolution=1.0,
        idw_search_radius=5.0,
        idw_power=2.0,
        user_defined_extent=None,
        nodata_value=-9999.0,
        verbose=True
    )

