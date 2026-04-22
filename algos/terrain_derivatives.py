import numpy as np
import taichi as ti

# --- Taichi Initialization ---
try:
    ti.init(arch=ti.gpu, log_level=ti.WARN)
    print("Taichi initialized with GPU backend.")
except Exception as e_gpu:
    print(f"GPU backend for Taichi failed: {e_gpu}")
    print("Falling back to CPU backend for Taichi.")
    ti.init(arch=ti.cpu, log_level=ti.WARN)
    print("Taichi initialized with CPU backend.")


@ti.kernel
def slope_aspect_hillshade_kernel(
    dtm: ti.types.ndarray(ti.f32, ndim=2),
    out_slope: ti.types.ndarray(ti.f32, ndim=2),
    out_aspect: ti.types.ndarray(ti.f32, ndim=2),
    out_hillshade: ti.types.ndarray(ti.f32, ndim=2),
    nodata_value: ti.f32,
    cell_size: ti.f32,
    z_factor: ti.f32,
    azimuth: ti.f32,  # in degrees
    altitude: ti.f32,  # in degrees
    rows: ti.i32,
    cols: ti.i32,
):
    # Convert azimuth and altitude to radians for math
    azimuth_rad = (360.0 - azimuth + 90.0) * (3.14159265359 / 180.0)
    if azimuth_rad >= 2 * 3.14159265359:
        azimuth_rad -= 2 * 3.14159265359
    altitude_rad = altitude * (3.14159265359 / 180.0)
    zenith_rad = (90.0 * 3.14159265359 / 180.0) - altitude_rad

    for r, c in ti.ndrange((1, rows - 1), (1, cols - 1)):
        # Read the 3x3 window around (r, c)
        z1 = dtm[r - 1, c - 1]
        z2 = dtm[r - 1, c]
        z3 = dtm[r - 1, c + 1]
        z4 = dtm[r, c - 1]
        z5 = dtm[r, c]
        z6 = dtm[r, c + 1]
        z7 = dtm[r + 1, c - 1]
        z8 = dtm[r + 1, c]
        z9 = dtm[r + 1, c + 1]

        # Check for nodata in the window
        has_nodata = False
        if (
            z1 == nodata_value
            or z2 == nodata_value
            or z3 == nodata_value
            or z4 == nodata_value
            or z5 == nodata_value
            or z6 == nodata_value
            or z7 == nodata_value
            or z8 == nodata_value
            or z9 == nodata_value
        ):
            has_nodata = True

        if has_nodata:
            out_slope[r, c] = nodata_value
            out_aspect[r, c] = nodata_value
            out_hillshade[r, c] = nodata_value
        else:
            dz_dx = ((z3 + 2 * z6 + z9) - (z1 + 2 * z4 + z7)) / (8.0 * cell_size)
            dz_dy = ((z7 + 2 * z8 + z9) - (z1 + 2 * z2 + z3)) / (8.0 * cell_size)

            dz_dx *= z_factor
            dz_dy *= z_factor

            # Calculate Slope in radians
            # Taichi provides atan2, so atan(x) can be computed as atan2(x, 1.0)
            slope_rad = ti.atan2(ti.sqrt(dz_dx**2 + dz_dy**2), 1.0)

            # Slope in degrees
            out_slope[r, c] = slope_rad * (180.0 / 3.14159265359)

            # Calculate Aspect in radians
            aspect_rad = 0.0
            if dz_dx != 0.0:
                aspect_rad = ti.atan2(dz_dy, -dz_dx)
                if aspect_rad < 0.0:
                    aspect_rad += 2.0 * 3.14159265359
            elif dz_dy > 0.0:
                aspect_rad = 3.14159265359 / 2.0
            elif dz_dy < 0.0:
                aspect_rad = 2.0 * 3.14159265359 - (3.14159265359 / 2.0)

            # Aspect in degrees
            aspect_deg = aspect_rad * (180.0 / 3.14159265359)
            if aspect_deg == 360.0:
                aspect_deg = 0.0
            # Convert math aspect to compass aspect
            compass_aspect = 90.0 - aspect_deg
            if compass_aspect < 0.0:
                compass_aspect += 360.0

            out_aspect[r, c] = compass_aspect

            # Calculate Hillshade (0-255)
            hs = 255.0 * (
                (ti.cos(zenith_rad) * ti.cos(slope_rad))
                + (
                    ti.sin(zenith_rad)
                    * ti.sin(slope_rad)
                    * ti.cos(azimuth_rad - aspect_rad)
                )
            )
            if hs < 0.0:
                hs = 0.0
            out_hillshade[r, c] = hs


def calculate_terrain_derivatives(
    dtm: np.ndarray,
    cell_size: float,
    nodata_value: float = -9999.0,
    z_factor: float = 1.0,
    azimuth: float = 315.0,
    altitude: float = 45.0,
) -> dict:
    """
    Calculates slope, aspect, and hillshade for a given DTM using Taichi.

    Returns a dictionary with 'slope', 'aspect', and 'hillshade' as numpy arrays.
    Edge cells (where a full 3x3 window is not available) are assigned the nodata_value.
    """
    if dtm.size == 0 or dtm.ndim != 2:
        return {
            "slope": np.array([[]], dtype=np.float32),
            "aspect": np.array([[]], dtype=np.float32),
            "hillshade": np.array([[]], dtype=np.float32),
        }

    rows, cols = dtm.shape
    dtm_f32 = dtm.astype(np.float32)

    out_slope = np.full((rows, cols), nodata_value, dtype=np.float32)
    out_aspect = np.full((rows, cols), nodata_value, dtype=np.float32)
    out_hillshade = np.full((rows, cols), nodata_value, dtype=np.float32)

    if rows > 2 and cols > 2:
        slope_aspect_hillshade_kernel(
            dtm_f32,
            out_slope,
            out_aspect,
            out_hillshade,
            nodata_value,
            cell_size,
            z_factor,
            azimuth,
            altitude,
            rows,
            cols,
        )

    return {"slope": out_slope, "aspect": out_aspect, "hillshade": out_hillshade}
