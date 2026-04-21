import numpy as np
import taichi as ti

# --- Taichi Initialization ---
try:
    ti.init(arch=ti.cpu, log_level=ti.WARN)
except Exception:
    ti.init(arch=ti.cpu, log_level=ti.WARN)


# --- Taichi Kernels ---
@ti.kernel
def assign_points_to_grid_minmax_kernel(
    points_x: ti.types.ndarray(ti.f32, ndim=1),
    points_y: ti.types.ndarray(ti.f32, ndim=1),
    points_z: ti.types.ndarray(ti.f32, ndim=1),
    min_z_field: ti.template(),
    max_z_field: ti.template(),
    count_field: ti.template(),
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

        grid_idx_x = ti.floor(gx_float)
        grid_idx_y = ti.floor(gy_float)

        gix = ti.cast(grid_idx_x, ti.i32)
        giy = ti.cast(grid_idx_y, ti.i32)

        if 0 <= gix < grid_width and 0 <= giy < grid_height:
            z = points_z[i]
            ti.atomic_min(min_z_field[gix, giy], z)
            ti.atomic_max(max_z_field[gix, giy], z)
            ti.atomic_add(count_field[gix, giy], 1)


# --- Common Function ---
def _minmax_common(
    ground_points_xyz: np.ndarray,
    dtm_resolution: float,
    dtm_extent_user: tuple = None,
    nodata_value: float = -9999.0,
    return_type: str = "min",
) -> np.ndarray:
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

    min_z_field = ti.field(dtype=ti.f32, shape=(grid_width, grid_height))
    max_z_field = ti.field(dtype=ti.f32, shape=(grid_width, grid_height))
    count_field = ti.field(dtype=ti.i32, shape=(grid_width, grid_height))

    # Initialize min with very large number, max with very small number
    min_z_field.fill(np.inf)
    max_z_field.fill(-np.inf)
    count_field.fill(0)

    assign_points_to_grid_minmax_kernel(
        points_x_np,
        points_y_np,
        points_z_np,
        min_z_field,
        max_z_field,
        count_field,
        min_x,
        min_y,
        dtm_resolution,
        grid_width,
        grid_height,
    )
    ti.sync()

    if return_type == "min":
        out_np = min_z_field.to_numpy()
    else:
        out_np = max_z_field.to_numpy()

    count_np = count_field.to_numpy()

    dtm_np = np.where(count_np.T > 0, out_np.T, nodata_value).astype(np.float32)

    return dtm_np


# --- Main Python Functions ---
def min_elevation(
    ground_points_xyz: np.ndarray,
    dtm_resolution: float,
    dtm_extent_user: tuple = None,
    nodata_value: float = -9999.0,
) -> np.ndarray:
    """Creates a DTM using the minimum elevation within each cell."""
    return _minmax_common(
        ground_points_xyz, dtm_resolution, dtm_extent_user, nodata_value, "min"
    )

def max_elevation(
    ground_points_xyz: np.ndarray,
    dtm_resolution: float,
    dtm_extent_user: tuple = None,
    nodata_value: float = -9999.0,
) -> np.ndarray:
    """Creates a DSM using the maximum elevation within each cell."""
    return _minmax_common(
        ground_points_xyz, dtm_resolution, dtm_extent_user, nodata_value, "max"
    )
