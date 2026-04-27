import numpy as np
import taichi as ti

# --- Taichi Initialization ---
try:
    ti.init(arch=ti.cpu, log_level=ti.WARN)
    print("Taichi initialized with GPU backend.")
except Exception as e_gpu:
    print(f"GPU backend for Taichi failed: {e_gpu}")
    print("Falling back to CPU backend for Taichi.")
    ti.init(arch=ti.cpu, log_level=ti.WARN)
    print("Taichi initialized with CPU backend.")


# --- Taichi Kernel for Min/Max ---
@ti.kernel
def min_max_grid_kernel(
    points_x: ti.types.ndarray(ti.f32, ndim=1),
    points_y: ti.types.ndarray(ti.f32, ndim=1),
    points_z: ti.types.ndarray(ti.f32, ndim=1),
    min_z_field: ti.types.ndarray(ti.f32, ndim=2),
    max_z_field: ti.types.ndarray(ti.f32, ndim=2),
    count_field: ti.types.ndarray(ti.i32, ndim=2),
    min_x_dtm: ti.f32,
    min_y_dtm: ti.f32,
    resolution_dtm: ti.f32,
    grid_width: ti.i32,
    grid_height: ti.i32,
):
    num_points = points_x.shape[0]
    for i in range(num_points):
        gx_float = (points_x[i] - min_x_dtm) / resolution_dtm
        gy_float = (points_y[i] - min_y_dtm) / resolution_dtm

        gix = ti.cast(ti.floor(gx_float), ti.i32)
        giy = ti.cast(ti.floor(gy_float), ti.i32)

        if 0 <= gix < grid_width and 0 <= giy < grid_height:
            ti.atomic_add(count_field[gix, giy], 1)
            ti.atomic_min(min_z_field[gix, giy], points_z[i])
            ti.atomic_max(max_z_field[gix, giy], points_z[i])


def _base_min_max(
    ground_points_xyz: np.ndarray,
    dtm_resolution: float,
    dtm_extent_user: tuple = None,
    nodata_value: float = -9999.0,
    return_min: bool = True,
) -> np.ndarray:
    print(
        f"Starting Min/Max DTM creation with Taichi: {ground_points_xyz.shape[0]} points, resolution {dtm_resolution}"
    )

    if ground_points_xyz.shape[0] == 0:
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

    grid_width = int(np.ceil((max_x - min_x) / dtm_resolution))
    grid_height = int(np.ceil((max_y - min_y) / dtm_resolution))

    if ground_points_xyz.shape[0] > 0:
        if max_x == min_x:
            grid_width = 1
        if max_y == min_y:
            grid_height = 1

    if grid_width <= 0 or grid_height <= 0:
        return np.array([[]], dtype=np.float32)

    min_z_np = np.full((grid_width, grid_height), 1e30, dtype=np.float32)
    max_z_np = np.full((grid_width, grid_height), -1e30, dtype=np.float32)
    count_np = np.zeros((grid_width, grid_height), dtype=np.int32)

    min_max_grid_kernel(
        points_x_np,
        points_y_np,
        points_z_np,
        min_z_np,
        max_z_np,
        count_np,
        min_x,
        min_y,
        dtm_resolution,
        grid_width,
        grid_height,
    )

    if return_min:
        res_np = min_z_np
    else:
        res_np = max_z_np

    dtm_np = np.full((grid_height, grid_width), nodata_value, dtype=np.float32)
    valid_cells_mask = count_np > 0
    dtm_np[valid_cells_mask.T] = res_np.T[valid_cells_mask.T]

    return dtm_np


def min_z(
    ground_points_xyz: np.ndarray,
    dtm_resolution: float,
    dtm_extent_user: tuple = None,
    nodata_value: float = -9999.0,
) -> np.ndarray:
    """Creates a DTM where each cell contains the minimum Z value of points within it."""
    return _base_min_max(
        ground_points_xyz,
        dtm_resolution,
        dtm_extent_user,
        nodata_value,
        return_min=True,
    )


def max_z(
    ground_points_xyz: np.ndarray,
    dtm_resolution: float,
    dtm_extent_user: tuple = None,
    nodata_value: float = -9999.0,
) -> np.ndarray:
    """Creates a DTM where each cell contains the maximum Z value of points within it."""
    return _base_min_max(
        ground_points_xyz,
        dtm_resolution,
        dtm_extent_user,
        nodata_value,
        return_min=False,
    )
