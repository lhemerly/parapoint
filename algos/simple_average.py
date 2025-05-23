import numpy as np
import taichi as ti # Import the Taichi library

# --- Taichi Initialization ---
# You can choose the backend. ti.gpu is usually faster if you have a compatible GPU.
# If not, ti.cpu will use multi-core CPU.
# ti.init(arch=ti.gpu, log_level=ti.INFO) 
# For broader compatibility initially, let's default to CPU, user can change this.
try:
    ti.init(arch=ti.gpu)
    print("Taichi initialized with GPU backend.")
except Exception as e_gpu:
    print(f"GPU backend for Taichi failed: {e_gpu}")
    print("Falling back to CPU backend for Taichi.")
    ti.init(arch=ti.cpu)
    print("Taichi initialized with CPU backend.")

# --- Taichi Kernel ---
@ti.kernel
def assign_points_to_grid_kernel(
    points_x: ti.types.ndarray(ti.f32, ndim=1), # Input: X coordinates of points
    points_y: ti.types.ndarray(ti.f32, ndim=1), # Input: Y coordinates of points
    points_z: ti.types.ndarray(ti.f32, ndim=1), # Input: Z coordinates of points
    sum_z_field: ti.template(),        # Output: Taichi field to store sum of Z values for each cell
    count_field: ti.template(),        # Output: Taichi field to store point count for each cell
    min_x_dtm: ti.f32,                 # DTM grid minimum X
    min_y_dtm: ti.f32,                 # DTM grid minimum Y
    resolution_dtm: ti.f32,            # DTM grid resolution
    grid_width: ti.i32,                # DTM grid width in cells
    grid_height: ti.i32                # DTM grid height in cells
):
    """
    Assigns points to a DTM grid and accumulates Z values and counts.
    """
    num_points = points_x.shape[0]
    for i in range(num_points):
        # Calculate grid cell indices (floating point first)
        gx_float = (points_x[i] - min_x_dtm) / resolution_dtm
        gy_float = (points_y[i] - min_y_dtm) / resolution_dtm

        # Convert to integer indices
        grid_idx_x = ti.floor(gx_float)
        grid_idx_y = ti.floor(gy_float)
        
        # Cast to int32 for indexing Taichi fields
        gix = ti.cast(grid_idx_x, ti.i32)
        giy = ti.cast(grid_idx_y, ti.i32)

        # Boundary check: ensure the point falls within the defined DTM grid
        if 0 <= gix < grid_width and 0 <= giy < grid_height:
            # Atomically add to the sum and count for the cell
            # This is crucial for parallel execution to avoid race conditions
            ti.atomic_add(sum_z_field[gix, giy], points_z[i])
            ti.atomic_add(count_field[gix, giy], 1)

# --- Main Python Function ---
def create_dtm_with_taichi_averaging(
    ground_points_xyz: np.ndarray, # NumPy array of shape (N, 3) with X, Y, Z columns
    dtm_resolution: float,
    dtm_extent_user: tuple = None, # Optional: (min_x, min_y, max_x, max_y)
    nodata_value: float = -9999.0
) -> np.ndarray:
    """
    Creates a DTM NumPy array from ground points using Taichi for gridding and averaging.

    Args:
        ground_points_xyz: NumPy array of shape (N,3) containing X, Y, Z of ground points.
        dtm_resolution: The desired resolution of the output DTM grid (e.g., 1.0 for 1 meter).
        dtm_extent_user: Optional tuple (min_x, min_y, max_x, max_y) to define the DTM area.
                         If None, it's calculated from the input points.
        nodata_value: Value to use for DTM cells with no underlying points.

    Returns:
        A 2D NumPy array representing the DTM.
    """
    print(f"Starting DTM creation with Taichi: {ground_points_xyz.shape[0]} points, resolution {dtm_resolution}")

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
    grid_width = int(np.ceil((max_x - min_x) / dtm_resolution))
    grid_height = int(np.ceil((max_y - min_y) / dtm_resolution))

    # Ensure at least a 1x1 grid if there are points and extent is zero,
    # otherwise, the previous calculation stands.
    if ground_points_xyz.shape[0] > 0:
        if max_x == min_x: # Typically for a single point column
            grid_width = 1
        if max_y == min_y: # Typically for a single point row
            grid_height = 1
    
    # If after calculation, it's still zero (e.g. no points, or specific user extent)
    if grid_width <= 0 or grid_height <= 0:
        print("Error: Calculated DTM grid dimensions are invalid (<=0). Check resolution and point extent.")
        return np.array([[]], dtype=np.float32)
        
    print(f"DTM Grid Dimensions: {grid_width} (width) x {grid_height} (height) cells")

    # Initialize Taichi fields for sum of Z and counts
    # These fields will store the accumulated values per grid cell
    sum_z_field = ti.field(dtype=ti.f32, shape=(grid_width, grid_height))
    count_field = ti.field(dtype=ti.i32, shape=(grid_width, grid_height))

    # Reset fields to zero before processing
    sum_z_field.fill(0.0)
    count_field.fill(0)

    # Call the Taichi kernel
    print("Launching Taichi kernel to assign points to grid...")
    assign_points_to_grid_kernel(
        points_x_np, points_y_np, points_z_np,
        sum_z_field, count_field,
        min_x, min_y, dtm_resolution,
        grid_width, grid_height
    )
    print("Taichi kernel execution finished.")

    # Convert Taichi fields back to NumPy arrays for final computation
    sum_z_np = sum_z_field.to_numpy()
    count_np = count_field.to_numpy()

    # Create the DTM: average Z where count > 0, otherwise nodata_value
    print("Calculating final DTM averages...")
    dtm_np = np.full((grid_height, grid_width), nodata_value, dtype=np.float32) # Note: DTM often (rows, cols) -> (height, width)
    
    # Avoid division by zero: only calculate average where count_np > 0
    valid_cells_mask = count_np > 0
    dtm_np[valid_cells_mask.T] = sum_z_np[valid_cells_mask] / count_np[valid_cells_mask] # Transpose mask to match dtm_np shape
    
    print("DTM creation complete.")
    return dtm_np # Return as (height, width) which is common for rasters

# --- Example Usage ---
def run_simple_average_example(
    num_sample_points=100, # Reduced default for faster example/testing
    extent_size=20.0,    # Reduced default
    resolution=1.0,
    dtm_extent_user=None,
    nodata_value=-9999.0,
    verbose=True # Control print statements for testing
):
    """Runs a full example of Simple Average DTM generation."""
    if verbose:
        print("--- Running Taichi DTM Averaging Example ---")
    
    # 1. Create some sample ground points
    sample_points = np.random.rand(num_sample_points, 3).astype(np.float32) * extent_size
    if num_sample_points > 0: # Avoid division by zero if num_sample_points is 0
        sample_points[:, 2] = (np.sin(sample_points[:, 0] / (extent_size/10 if extent_size > 0 else 1.0)) * 5 +
                               np.cos(sample_points[:, 1] / (extent_size/10 if extent_size > 0 else 1.0)) * 5 +
                               sample_points[:, 2] * 0.1) # Give some Z variation
    
    if verbose:
        print(f"Generated {sample_points.shape[0]} sample points.")

    # 2. Create the DTM
    dtm_array = create_dtm_with_taichi_averaging(
        sample_points,
        resolution,
        dtm_extent_user=dtm_extent_user,
        nodata_value=nodata_value
    )

    # 3. Output information about the DTM (if verbose)
    if verbose:
        if dtm_array.size > 0:
            print(f"\nGenerated DTM shape: {dtm_array.shape}") # (height, width)
            valid_dtm_values = dtm_array[dtm_array != nodata_value]
            if valid_dtm_values.size > 0:
                print(f"DTM min value (excluding NoData): {np.min(valid_dtm_values):.2f}")
                print(f"DTM max value (excluding NoData): {np.max(valid_dtm_values):.2f}") # Max of valid
            else:
                print("DTM contains only NoData values.")
            num_nodata_cells = np.sum(dtm_array == nodata_value)
            print(f"Number of NoData cells: {num_nodata_cells} out of {dtm_array.size} total cells ({num_nodata_cells/dtm_array.size*100:.1f}%)")
        else:
            print("DTM generation resulted in an empty array.")
    
    return dtm_array

if __name__ == "__main__":
    # Run with default parameters for the main execution.
    # These can be larger/more demanding than the test version.
    run_simple_average_example(
        num_sample_points=100000,
        extent_size=100.0, # Original example extent
        resolution=1.0,
        dtm_extent_user=None, # Original example extent was None
        nodata_value=-9999.0,
        verbose=True
    )

