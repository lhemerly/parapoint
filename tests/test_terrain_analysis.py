import numpy as np
from parapoint import slope, hillshade

def test_slope_basic():
    # A simple 3x3 DTM representing a flat surface inclined slightly
    # Let's create an inclined plane: z = x + y
    dtm = np.array([
        [0.0, 1.0, 2.0],
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0]
    ], dtype=np.float32)

    res = 1.0
    s = slope(dtm, res)

    # Outer boundaries are -9999.0
    assert s.shape == (3, 3)
    assert s[0, 0] == -9999.0
    assert s[2, 2] == -9999.0

    # Center pixel slope
    # dz_dx for plane x+y is 1.0
    # dz_dy for plane x+y is 1.0
    # slope = atan(sqrt(1^2 + 1^2)) = atan(sqrt(2)) = 54.735 degrees
    center_slope = s[1, 1]
    expected_slope = np.arctan(np.sqrt(2)) * 180 / np.pi
    np.testing.assert_almost_equal(center_slope, expected_slope, decimal=3)


def test_hillshade_basic():
    # A simple 3x3 DTM
    dtm = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 0.0]
    ], dtype=np.float32)

    res = 1.0
    h = hillshade(dtm, res)

    assert h.shape == (3, 3)
    assert h[0, 0] == -9999.0

    # Center pixel should have a valid shade value between 0 and 255
    center_shade = h[1, 1]
    assert 0.0 <= center_shade <= 255.0

def test_terrain_analysis_empty():
    dtm = np.empty((0, 0), dtype=np.float32)
    s = slope(dtm, 1.0)
    h = hillshade(dtm, 1.0)

    assert s.size == 0
    assert h.size == 0

def test_terrain_analysis_too_small():
    dtm = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    s = slope(dtm, 1.0)
    h = hillshade(dtm, 1.0)

    assert s.size == 0
    assert h.size == 0
