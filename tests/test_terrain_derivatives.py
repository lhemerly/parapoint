import numpy as np
from algos.terrain_derivatives import (
    calculate_terrain_derivatives,
    calculate_tpi,
    calculate_tri,
)


def test_calculate_terrain_derivatives():
    # Create a simple 3x3 DTM representing a planar slope
    # Z decreases by 1 in X, and decreases by 1 in Y
    # Cell (1,1) is exactly in the center.
    dtm = np.array(
        [[10.0, 9.0, 8.0], [9.0, 8.0, 7.0], [8.0, 7.0, 6.0]], dtype=np.float32
    )

    cell_size = 1.0
    res = calculate_terrain_derivatives(dtm, cell_size, nodata_value=-99.0)

    slope = res["slope"]
    aspect = res["aspect"]
    hillshade = res["hillshade"]

    # The outer edges should be nodata since we can't calculate a 3x3 window
    assert slope[0, 0] == -99.0

    # For the center cell (1, 1), the slope should be constant
    # dz_dx = ( (8 + 2*7 + 6) - (10 + 2*9 + 8) ) / 8 = (28 - 36) / 8 = -1
    # dz_dy = ( (8 + 2*7 + 6) - (10 + 2*9 + 8) ) / 8 = (28 - 36) / 8 = -1
    # slope = atan(sqrt(1 + 1)) = atan(sqrt(2)) = 54.7356 degrees
    assert np.isclose(slope[1, 1], 54.7356, atol=0.01)

    # aspect: atan2(-1, 1) = -pi/4 -> 315 deg math aspect -> 135 deg compass aspect
    assert np.isclose(aspect[1, 1], 135.0, atol=0.01)

    # Hillshade: shouldn't be nodata
    assert hillshade[1, 1] >= 0.0 and hillshade[1, 1] <= 255.0


def test_terrain_derivatives_empty():
    dtm = np.array([[]], dtype=np.float32)
    res = calculate_terrain_derivatives(dtm, cell_size=1.0)
    assert res["slope"].size == 0
    assert res["aspect"].size == 0
    assert res["hillshade"].size == 0


def test_tpi_basic():
    # 3x3 plane, center should have TPI 0
    dtm = np.array(
        [[10.0, 10.0, 10.0], [10.0, 10.0, 10.0], [10.0, 10.0, 10.0]], dtype=np.float32
    )

    tpi = calculate_tpi(dtm, radius=1, nodata_value=-9999.0)
    # Center cell mean of 8 neighbors is 10. 10 - 10 = 0.
    assert np.isclose(tpi[1, 1], 0.0)

    # Peak
    dtm_peak = np.array(
        [[10.0, 10.0, 10.0], [10.0, 20.0, 10.0], [10.0, 10.0, 10.0]], dtype=np.float32
    )

    tpi_peak = calculate_tpi(dtm_peak, radius=1, nodata_value=-9999.0)
    # Mean of neighbors is 10. Center is 20. TPI = 10.
    assert np.isclose(tpi_peak[1, 1], 10.0)


def test_tpi_empty():
    res = calculate_tpi(np.array([[]], dtype=np.float32))
    assert res.size == 0


def test_tri_basic():
    # 3x3 plane, TRI should be 0
    dtm = np.array(
        [[10.0, 10.0, 10.0], [10.0, 10.0, 10.0], [10.0, 10.0, 10.0]], dtype=np.float32
    )

    tri = calculate_tri(dtm, nodata_value=-9999.0)
    assert np.isclose(tri[1, 1], 0.0)

    # Center diff = 1 with all neighbors -> sum of diffs = 8
    dtm_bump = np.array(
        [[10.0, 10.0, 10.0], [10.0, 11.0, 10.0], [10.0, 10.0, 10.0]], dtype=np.float32
    )

    tri_bump = calculate_tri(dtm_bump, nodata_value=-9999.0)
    assert np.isclose(tri_bump[1, 1], 8.0)


def test_tri_empty():
    res = calculate_tri(np.array([[]], dtype=np.float32))
    assert res.size == 0
