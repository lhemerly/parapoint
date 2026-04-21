import numpy as np
from parapoint import min_elevation, max_elevation

def test_minmax_elevation_basic():
    # 4 points forming a simple 2x2 grid when res=10.0
    # points inside cells:
    # Cell (0, 0): Z=5, 10 => Min=5, Max=10
    # Cell (1, 1): Z=20 => Min=20, Max=20
    # Other cells: Empty
    points = np.array([
        [1.0, 1.0, 10.0],
        [1.0, 1.0, 5.0],
        [15.0, 15.0, 20.0]
    ], dtype=np.float32)

    res = 10.0
    nodata = -9999.0

    # Test Min Elevation
    dtm_min = min_elevation(points, res, dtm_extent_user=(0.0, 0.0, 20.0, 20.0), nodata_value=nodata)
    assert dtm_min.shape == (2, 2)
    assert dtm_min[0, 0] == 5.0
    assert dtm_min[1, 1] == 20.0
    assert dtm_min[0, 1] == nodata
    assert dtm_min[1, 0] == nodata

    # Test Max Elevation
    dtm_max = max_elevation(points, res, dtm_extent_user=(0.0, 0.0, 20.0, 20.0), nodata_value=nodata)
    assert dtm_max.shape == (2, 2)
    assert dtm_max[0, 0] == 10.0
    assert dtm_max[1, 1] == 20.0
    assert dtm_max[0, 1] == nodata
    assert dtm_max[1, 0] == nodata

def test_minmax_elevation_empty():
    points = np.empty((0, 3), dtype=np.float32)
    dtm_min = min_elevation(points, 1.0)
    dtm_max = max_elevation(points, 1.0)

    assert dtm_min.size == 0
    assert dtm_max.size == 0
