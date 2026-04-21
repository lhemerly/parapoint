import numpy as np
import taichi as ti

# --- Taichi Initialization ---
try:
    ti.init(arch=ti.cpu, log_level=ti.WARN)
except Exception:
    ti.init(arch=ti.cpu, log_level=ti.WARN)


@ti.kernel
def compute_slope_kernel(
    dtm: ti.types.ndarray(ti.f32, ndim=2),
    slope_out: ti.template(),
    resolution: ti.f32,
    nodata_value: ti.f32,
    rows: ti.i32,
    cols: ti.i32,
):
    for r, c in ti.ndrange((1, rows - 1), (1, cols - 1)):
        # Read the 3x3 window
        z1 = dtm[r - 1, c - 1]
        z2 = dtm[r - 1, c]
        z3 = dtm[r - 1, c + 1]
        z4 = dtm[r, c - 1]
        z5 = dtm[r, c]
        z6 = dtm[r, c + 1]
        z7 = dtm[r + 1, c - 1]
        z8 = dtm[r + 1, c]
        z9 = dtm[r + 1, c + 1]

        # Check if any value is nodata
        if (z1 == nodata_value or z2 == nodata_value or z3 == nodata_value or
            z4 == nodata_value or z5 == nodata_value or z6 == nodata_value or
            z7 == nodata_value or z8 == nodata_value or z9 == nodata_value):
            slope_out[r, c] = nodata_value
        else:
            # Horn's method for slope
            dz_dx = ((z3 + 2*z6 + z9) - (z1 + 2*z4 + z7)) / (8 * resolution)
            dz_dy = ((z7 + 2*z8 + z9) - (z1 + 2*z2 + z3)) / (8 * resolution)

            slope_rad = ti.atan2(ti.sqrt(dz_dx**2 + dz_dy**2), 1.0)
            # Convert to degrees
            slope_out[r, c] = slope_rad * 180.0 / 3.141592653589793


@ti.kernel
def compute_hillshade_kernel(
    dtm: ti.types.ndarray(ti.f32, ndim=2),
    hillshade_out: ti.template(),
    resolution: ti.f32,
    azimuth_rad: ti.f32,
    altitude_rad: ti.f32,
    z_factor: ti.f32,
    nodata_value: ti.f32,
    rows: ti.i32,
    cols: ti.i32,
):
    for r, c in ti.ndrange((1, rows - 1), (1, cols - 1)):
        # Read the 3x3 window
        z1 = dtm[r - 1, c - 1]
        z2 = dtm[r - 1, c]
        z3 = dtm[r - 1, c + 1]
        z4 = dtm[r, c - 1]
        z5 = dtm[r, c]
        z6 = dtm[r, c + 1]
        z7 = dtm[r + 1, c - 1]
        z8 = dtm[r + 1, c]
        z9 = dtm[r + 1, c + 1]

        if (z1 == nodata_value or z2 == nodata_value or z3 == nodata_value or
            z4 == nodata_value or z5 == nodata_value or z6 == nodata_value or
            z7 == nodata_value or z8 == nodata_value or z9 == nodata_value):
            hillshade_out[r, c] = nodata_value
        else:
            dz_dx = ((z3 + 2*z6 + z9) - (z1 + 2*z4 + z7)) / (8 * resolution) * z_factor
            dz_dy = ((z7 + 2*z8 + z9) - (z1 + 2*z2 + z3)) / (8 * resolution) * z_factor

            slope_rad = ti.atan2(ti.sqrt(dz_dx**2 + dz_dy**2), 1.0)
            aspect_rad = 0.0

            if dz_dx != 0.0:
                aspect_rad = ti.atan2(dz_dy, -dz_dx)
                if aspect_rad < 0.0:
                    aspect_rad = 2.0 * 3.141592653589793 + aspect_rad
            elif dz_dy > 0.0:
                aspect_rad = 3.141592653589793 / 2.0
            elif dz_dy < 0.0:
                aspect_rad = 2.0 * 3.141592653589793 - 3.141592653589793 / 2.0

            shade = 255.0 * ((ti.sin(altitude_rad) * ti.cos(slope_rad)) +
                             (ti.cos(altitude_rad) * ti.sin(slope_rad) * ti.cos(azimuth_rad - aspect_rad)))

            # Clamp to [0, 255]
            if shade < 0.0:
                shade = 0.0
            if shade > 255.0:
                shade = 255.0

            hillshade_out[r, c] = shade


def slope(
    dtm: np.ndarray,
    resolution: float,
    nodata_value: float = -9999.0
) -> np.ndarray:
    """Computes the slope (in degrees) of a DTM using Horn's method."""
    if dtm.size == 0 or dtm.shape[0] < 3 or dtm.shape[1] < 3:
        return np.array([[]], dtype=np.float32)

    rows, cols = dtm.shape
    slope_out_field = ti.field(dtype=ti.f32, shape=(rows, cols))
    slope_out_field.fill(nodata_value)

    dtm_np_float32 = dtm.astype(np.float32)

    compute_slope_kernel(
        dtm_np_float32,
        slope_out_field,
        resolution,
        nodata_value,
        rows,
        cols
    )
    ti.sync()

    return slope_out_field.to_numpy()


def hillshade(
    dtm: np.ndarray,
    resolution: float,
    azimuth: float = 315.0,
    altitude: float = 45.0,
    z_factor: float = 1.0,
    nodata_value: float = -9999.0
) -> np.ndarray:
    """Computes the hillshade of a DTM."""
    if dtm.size == 0 or dtm.shape[0] < 3 or dtm.shape[1] < 3:
        return np.array([[]], dtype=np.float32)

    rows, cols = dtm.shape
    hillshade_out_field = ti.field(dtype=ti.f32, shape=(rows, cols))
    hillshade_out_field.fill(nodata_value)

    dtm_np_float32 = dtm.astype(np.float32)

    # Convert azimuth and altitude to radians
    # Math for hillshade expects mathematical angle, not compass angle
    # Compass angle: 0 is North, clockwise
    # Math angle: 0 is East, counter-clockwise
    math_azimuth = 360.0 - azimuth + 90.0
    if math_azimuth >= 360.0:
        math_azimuth -= 360.0

    azimuth_rad = math_azimuth * np.pi / 180.0
    altitude_rad = altitude * np.pi / 180.0

    compute_hillshade_kernel(
        dtm_np_float32,
        hillshade_out_field,
        resolution,
        azimuth_rad,
        altitude_rad,
        z_factor,
        nodata_value,
        rows,
        cols
    )
    ti.sync()

    return hillshade_out_field.to_numpy()
