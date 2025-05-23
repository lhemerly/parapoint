import numpy as np
import taichi as ti # Import the Taichi library
import time # For basic timing
from algos.spatial_grid_index import SpatialGridIndex # Import SpatialGridIndex

# --- Taichi Initialization ---
try:
    ti.init(arch=ti.gpu, log_level=ti.WARN) 
    print("Taichi initialized with GPU backend.")
except Exception as e_gpu:
    print(f"GPU backend for Taichi failed: {e_gpu}")
    print("Falling back to CPU backend for Taichi.")
    ti.init(arch=ti.cpu, log_level=ti.WARN)
    print("Taichi initialized with CPU backend.")

# --- Taichi Kernel for IDW (Refactored) ---
@ti.kernel
def idw_interpolation_kernel(
    points_x: ti.types.ndarray(ti.f32, ndim=1), 
    points_y: ti.types.ndarray(ti.f32, ndim=1), 
    points_z: ti.types.ndarray(ti.f32, ndim=1), 
    dtm_field: ti.template(),          
    min_x_dtm: ti.f32,                 
    min_y_dtm: ti.f32,                 
    resolution_dtm: ti.f32,            
    grid_width: ti.i32,                
    grid_height: ti.i32,               
    search_radius: ti.f32,             
    power_p: ti.f32,                   
    nodata_value_kernel: ti.f32,
    # New parameters for spatial index:
    indexed_point_indices: ti.template(), 
    cell_offsets: ti.template(),          
    cell_point_counts: ti.template(),     
    index_min_x: ti.f32,
    index_min_y: ti.f32,
    index_resolution: ti.f32,
    index_grid_dim_x: ti.i32,
    index_grid_dim_y: ti.i32
):
    for r, c in ti.ndrange(grid_height, grid_width): 
        cell_center_x = min_x_dtm + (ti.cast(c, ti.f32) + 0.5) * resolution_dtm
        cell_center_y = min_y_dtm + (ti.cast(r, ti.f32) + 0.5) * resolution_dtm

        sum_weighted_z = 0.0
        sum_weights = 0.0
        exact_match_z = -1.0 
        found_exact_match = False

        # Efficient search using SpatialGridIndex
        # 1. Determine search range in terms of index cells
        search_min_gx_float = (cell_center_x - search_radius - index_min_x) / index_resolution
        search_max_gx_float = (cell_center_x + search_radius - index_min_x) / index_resolution
        search_min_gy_float = (cell_center_y - search_radius - index_min_y) / index_resolution
        search_max_gy_float = (cell_center_y + search_radius - index_min_y) / index_resolution
        
        min_gix = ti.cast(ti.max(0, ti.floor(search_min_gx_float)), ti.i32)
        max_gix = ti.cast(ti.min(index_grid_dim_x - 1, ti.floor(search_max_gx_float)), ti.i32)
        min_giy = ti.cast(ti.max(0, ti.floor(search_min_gy_float)), ti.i32)
        max_giy = ti.cast(ti.min(index_grid_dim_y - 1, ti.floor(search_max_gy_float)), ti.i32)

        # 2. Iterate through candidate index cells
        for cur_gix_loop in range(min_gix, max_gix + 1):
            if found_exact_match: break 
            for cur_giy_loop in range(min_giy, max_giy + 1):
                if found_exact_match: break 

                # Ensure cur_gix, cur_giy are within valid index grid bounds
                # This check might be slightly redundant due to clamping in min_gix/max_gix calculation,
                # but it's a safe guard, especially if clamping logic changes or has edge cases.
                cur_gix = ti.cast(cur_gix_loop, ti.i32) # Ensure type for field access
                cur_giy = ti.cast(cur_giy_loop, ti.i32) # Ensure type for field access

                if not (0 <= cur_gix < index_grid_dim_x and 0 <= cur_giy < index_grid_dim_y):
                    continue # Should not happen if clamping is correct

                num_points_in_cell = cell_point_counts[cur_gix, cur_giy]
                if num_points_in_cell > 0:
                    start_idx_in_flat_array = cell_offsets[cur_gix, cur_giy]
                    for k_offset in range(num_points_in_cell):
                        original_point_idx = indexed_point_indices[start_idx_in_flat_array + k_offset]
                        
                        px = points_x[original_point_idx] 
                        py = points_y[original_point_idx]
                        pz = points_z[original_point_idx]

                        dist_sq = (px - cell_center_x)**2 + (py - cell_center_y)**2
                        
                        if dist_sq == 0.0: 
                            exact_match_z = pz
                            found_exact_match = True
                            break 
                        
                        if dist_sq < search_radius**2: 
                            dist = ti.sqrt(dist_sq)
                            if dist > 1e-6: 
                                weight = 1.0 / (dist**power_p)
                                sum_weighted_z += weight * pz
                                sum_weights += weight
                if found_exact_match: break
            if found_exact_match: break
        
        if found_exact_match:
            dtm_field[r, c] = exact_match_z
        elif sum_weights > 1e-6: 
            dtm_field[r, c] = sum_weighted_z / sum_weights
        else: 
            dtm_field[r, c] = nodata_value_kernel


# --- Main Python Function for IDW (Refactored) ---
def create_dtm_with_taichi_idw(
    ground_points_xyz: np.ndarray, 
    dtm_resolution: float,
    search_radius: float,
    power: float = 2.0,
    dtm_extent_user: tuple = None, 
    nodata_value: float = -9999.0
) -> np.ndarray:
    print(f"Starting DTM creation with Taichi (IDW Refactored): {ground_points_xyz.shape[0]} points, res {dtm_resolution}, radius {search_radius}, power {power}")
    start_time = time.time()

    if ground_points_xyz.shape[0] == 0:
        print("Warning: No ground points provided. Returning empty DTM.")
        return np.array([[]], dtype=np.float32)

    points_x_np = ground_points_xyz[:, 0].astype(np.float32)
    points_y_np = ground_points_xyz[:, 1].astype(np.float32)
    points_z_np = ground_points_xyz[:, 2].astype(np.float32)

    if dtm_extent_user:
        min_x, min_y, max_x, max_y = dtm_extent_user
    else:
        min_x = np.min(points_x_np)
        min_y = np.min(points_y_np)
        max_x = np.max(points_x_np)
        max_y = np.max(points_y_np)
    
    print(f"DTM Extent: X({min_x:.2f} - {max_x:.2f}), Y({min_y:.2f} - {max_y:.2f})")

    # DTM grid dimensions calculation
    # Using min_x, min_y from user/points and calculated width/height to derive actual DTM grid coverage
    min_x_dtm_grid = min_x 
    min_y_dtm_grid = min_y
    grid_width = max(1, int(np.ceil((max_x - min_x_dtm_grid) / dtm_resolution)))
    grid_height = max(1, int(np.ceil((max_y - min_y_dtm_grid) / dtm_resolution)))
    
    # actual_max_x/y for DTM grid (used for SpatialGridIndex extent if it aligns with DTM)
    actual_dtm_grid_max_x = min_x_dtm_grid + grid_width * dtm_resolution
    actual_dtm_grid_max_y = min_y_dtm_grid + grid_height * dtm_resolution

    if grid_width <= 0 or grid_height <= 0:
        print("Error: Calculated DTM grid dimensions are invalid (<=0).")
        return np.array([[]], dtype=np.float32)
        
    print(f"DTM Grid Dimensions: {grid_width} (width) x {grid_height} (height) cells")

    # Create SpatialGridIndex
    points_xy_for_index = np.vstack((points_x_np, points_y_np)).T
    # Use dtm_resolution for the index resolution.
    # The extent for the index should cover all *source points*.
    # So, use min_x, min_y, max_x, max_y derived from points_x_np, points_y_np.
    # If dtm_extent_user was provided, it might be larger or smaller than points extent.
    # The spatial index should be built on the actual point data's extent.
    idx_min_x, idx_min_y = np.min(points_x_np), np.min(points_y_np)
    idx_max_x, idx_max_y = np.max(points_x_np), np.max(points_y_np)

    # Small epsilon to ensure points on the max boundary are included if idx_max_x aligns with a cell boundary.
    # This helps make the index inclusive of points at the very edge of the data extent.
    epsilon_adjust = dtm_resolution * 1e-5 
    idx_max_x_adjusted = idx_max_x + epsilon_adjust
    idx_max_y_adjusted = idx_max_y + epsilon_adjust

    print(f"Initializing SpatialGridIndex for IDW with resolution: {dtm_resolution} over points extent X({idx_min_x:.2f}-{idx_max_x_adjusted:.2f}), Y({idx_min_y:.2f}-{idx_max_y_adjusted:.2f})")
    spatial_index = SpatialGridIndex(points_xy_for_index, 
                                     resolution=dtm_resolution, 
                                     extent=(idx_min_x, idx_min_y, idx_max_x_adjusted, idx_max_y_adjusted))
    
    if spatial_index.num_points == 0 and points_x_np.size > 0 : # check if points_x_np.size > 0 to ensure it's not just an empty input
        # This might happen if all points fall outside the provided extent or due to an issue in SpatialGridIndex init
        print("Error: Spatial index created with zero points, but input points were provided. Check extent or SpatialGridIndex logs.")
        # Fallback to nodata DTM if index is unexpectedly empty
        dtm_output = np.full((grid_height, grid_width), nodata_value, dtype=np.float32)
        return dtm_output
    elif spatial_index.num_points == 0 : # Handles case where input ground_points_xyz was empty and spatial_index is also empty
        print("Warning: Spatial index is empty (likely due to no input points). DTM will be all nodata.")
        dtm_output = np.full((grid_height, grid_width), nodata_value, dtype=np.float32)
        return dtm_output


    dtm_output_field = ti.field(dtype=ti.f32, shape=(grid_height, grid_width))
    dtm_output_field.fill(nodata_value) 

    print("Launching Taichi IDW kernel (with Spatial Index)...")
    kernel_start_time = time.time()
    idw_interpolation_kernel(
        points_x_np, points_y_np, points_z_np, # Original point data (NumPy)
        dtm_output_field,
        min_x_dtm_grid, min_y_dtm_grid, dtm_resolution, # DTM grid origin and resolution
        grid_width, grid_height,
        search_radius, power, nodata_value, # Kernel needs float for nodata
        # New parameters for spatial index:
        spatial_index.indexed_point_indices, 
        spatial_index.cell_offsets,          
        spatial_index.cell_point_counts,     
        spatial_index.min_x,                 # Index origin x (from SpatialGridIndex)
        spatial_index.min_y,                 # Index origin y (from SpatialGridIndex)
        spatial_index.resolution,            # Index cell size (from SpatialGridIndex)
        spatial_index.grid_dim_x,            # Index grid dimension x
        spatial_index.grid_dim_y             # Index grid dimension y
    )
    ti.sync() 
    kernel_end_time = time.time()
    print(f"Taichi kernel execution finished in {kernel_end_time - kernel_start_time:.2f} seconds.")

    dtm_np = dtm_output_field.to_numpy()
    
    total_time = time.time() - start_time
    print(f"IDW DTM creation complete in {total_time:.2f} seconds.")
    return dtm_np

# --- Example Usage (Unchanged from previous version, but should work with refactored code) ---
def run_idw_example(
    num_sample_points=500, 
    extent_size=20.0,    
    resolution=1.0,
    idw_search_radius=5.0,
    idw_power=2.0,
    dtm_extent_user=None, # Renamed from user_defined_extent for clarity
    nodata_value=-9999.0,
    verbose=True 
):
    if verbose:
        print("--- Running Taichi DTM IDW Interpolation Example (Refactored with Spatial Index) ---")
    
    sample_points = np.random.rand(num_sample_points, 3).astype(np.float32) * extent_size
    if num_sample_points > 0: 
        sample_points[:, 2] = (np.sin(sample_points[:, 0] / (extent_size/10 if extent_size > 0 else 1.0)) * 10 +
                               np.cos(sample_points[:, 1] / (extent_size/10 if extent_size > 0 else 1.0)) * 10 +
                               sample_points[:,0] * 0.05 + 
                               np.random.rand(num_sample_points) * 2) 
    
    if verbose:
        print(f"Generated {sample_points.shape[0]} sample points.")

    dtm_array = create_dtm_with_taichi_idw(
        sample_points,
        resolution,
        idw_search_radius,
        power=idw_power,
        dtm_extent_user=dtm_extent_user, # Pass the renamed variable
        nodata_value=nodata_value
    )

    if verbose:
        if dtm_array.size > 0:
            print(f"\nGenerated IDW DTM shape: {dtm_array.shape}") 
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
    run_idw_example(
        num_sample_points=50000, # Reduced for faster example run if testing locally
        extent_size=200.0,
        resolution=1.0,
        idw_search_radius=5.0,
        idw_power=2.0,
        dtm_extent_user=None, # Use None to auto-calculate from points
        nodata_value=-9999.0,
        verbose=True
    )
