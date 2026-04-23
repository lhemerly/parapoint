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


@ti.kernel
def gaussian_filter_kernel(
    dtm: ti.types.ndarray(ti.f32, ndim=2),
    out_dtm: ti.types.ndarray(ti.f32, ndim=2),
    nodata_value: ti.f32,
    rows: ti.i32,
    cols: ti.i32,
    sigma: ti.f32,
    radius: ti.i32,
):
    # Standard deviation calculation is done outside in Python to avoid redundant kernel ops
    # The kernel itself will construct the weights dynamically for simplicity.
    # In a more optimized version, weights could be pre-computed and passed as an array.

    # 2 * sigma^2
    two_sigma_sq = 2.0 * sigma * sigma
    pi = 3.14159265359

    for r, c in ti.ndrange(rows, cols):
        if dtm[r, c] == nodata_value:
            out_dtm[r, c] = nodata_value
        else:
            weight_sum = 0.0
            val_sum = 0.0
            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    # Check boundaries
                    nr = r + i
                    nc = c + j
                    if 0 <= nr < rows and 0 <= nc < cols:
                        neighbor_val = dtm[nr, nc]
                        if neighbor_val != nodata_value:
                            # Calculate Gaussian weight
                            dist_sq = ti.cast(i * i + j * j, ti.f32)
                            weight = ti.exp(-dist_sq / two_sigma_sq) / (
                                pi * two_sigma_sq
                            )
                            val_sum += neighbor_val * weight
                            weight_sum += weight

            if weight_sum > 0.0:
                out_dtm[r, c] = val_sum / weight_sum
            else:
                out_dtm[r, c] = dtm[r, c]  # Should not happen ideally


@ti.kernel
def fill_nodata_kernel(
    dtm: ti.types.ndarray(ti.f32, ndim=2),
    out_dtm: ti.types.ndarray(ti.f32, ndim=2),
    nodata_value: ti.f32,
    rows: ti.i32,
    cols: ti.i32,
    radius: ti.i32,
) -> ti.i32:
    cells_filled = 0
    for r, c in ti.ndrange(rows, cols):
        if dtm[r, c] == nodata_value:
            val_sum = 0.0
            count = 0
            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    nr = r + i
                    nc = c + j
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if i != 0 or j != 0:
                            neighbor_val = dtm[nr, nc]
                            if neighbor_val != nodata_value:
                                val_sum += neighbor_val
                                count += 1
            if count > 0:
                out_dtm[r, c] = val_sum / ti.cast(count, ti.f32)
                ti.atomic_add(cells_filled, 1)
            else:
                out_dtm[r, c] = nodata_value
        else:
            out_dtm[r, c] = dtm[r, c]
    return cells_filled


def gaussian_filter(
    dtm: np.ndarray,
    sigma: float = 1.0,
    radius: int = 2,
    nodata_value: float = -9999.0,
) -> np.ndarray:
    """
    Applies a Gaussian filter to smooth the DTM, reducing noise.

    Args:
        dtm (np.ndarray): The input Digital Terrain Model.
        sigma (float): The standard deviation of the Gaussian distribution.
        radius (int): The radius of the kernel (e.g., 2 means a 5x5 kernel).
        nodata_value (float): The nodata value to ignore.

    Returns:
        np.ndarray: The smoothed DTM.
    """
    if dtm.size == 0 or dtm.ndim != 2:
        return np.array([[]], dtype=np.float32)

    rows, cols = dtm.shape
    dtm_f32 = dtm.astype(np.float32)
    out_dtm = np.empty_like(dtm_f32)

    gaussian_filter_kernel(dtm_f32, out_dtm, nodata_value, rows, cols, sigma, radius)

    return out_dtm


def fill_nodata(
    dtm: np.ndarray,
    max_iterations: int = 5,
    radius: int = 1,
    nodata_value: float = -9999.0,
) -> np.ndarray:
    """
    Iteratively fills nodata gaps in the DTM using the mean of valid neighbors.

    Args:
        dtm (np.ndarray): The input Digital Terrain Model with nodata gaps.
        max_iterations (int): Maximum number of passes to fill larger gaps.
        radius (int): The search radius for neighbors (e.g., 1 is a 3x3 window).
        nodata_value (float): The nodata value to replace.

    Returns:
        np.ndarray: The DTM with filled nodata values.
    """
    if dtm.size == 0 or dtm.ndim != 2:
        return np.array([[]], dtype=np.float32)

    rows, cols = dtm.shape
    current_dtm = dtm.astype(np.float32).copy()
    out_dtm = np.empty_like(current_dtm)

    for i in range(max_iterations):
        cells_filled = fill_nodata_kernel(
            current_dtm, out_dtm, nodata_value, rows, cols, radius
        )
        # Swap arrays for the next iteration
        current_dtm[:] = out_dtm

        if cells_filled == 0:
            break  # No more cells could be filled

    return current_dtm
