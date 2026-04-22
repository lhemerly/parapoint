import numpy as np
from algos.terrain_derivatives import calculate_terrain_derivatives


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
