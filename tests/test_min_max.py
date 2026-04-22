import numpy as np
from algos.min_max import min_z, max_z


def test_min_z():
    points = np.array(
        [
            [0.5, 0.5, 10.0],
            [0.6, 0.6, 5.0],
            [0.4, 0.4, 15.0],
            [1.5, 0.5, 20.0],
        ]
    )

    dtm = min_z(points, dtm_resolution=1.0, nodata_value=-99.0)

    # Cell 0,0 contains 10.0, 5.0, 15.0 -> min is 5.0
    # Cell 1,0 contains 20.0 -> min is 20.0
    # Shape is (height, width) -> (1, 2)
    assert dtm.shape == (1, 2)
    assert dtm[0, 0] == 5.0
    assert dtm[0, 1] == 20.0


def test_max_z():
    points = np.array(
        [
            [0.5, 0.5, 10.0],
            [0.6, 0.6, 5.0],
            [0.4, 0.4, 15.0],
            [1.5, 0.5, 20.0],
        ]
    )

    dtm = max_z(points, dtm_resolution=1.0, nodata_value=-99.0)

    # Cell 0,0 contains 10.0, 5.0, 15.0 -> max is 15.0
    # Cell 1,0 contains 20.0 -> max is 20.0
    assert dtm.shape == (1, 2)
    assert dtm[0, 0] == 15.0
    assert dtm[0, 1] == 20.0


def test_min_max_empty():
    points = np.array([], dtype=np.float32).reshape(0, 3)
    dtm_min = min_z(points, dtm_resolution=1.0)
    dtm_max = max_z(points, dtm_resolution=1.0)
    assert dtm_min.size == 0
    assert dtm_max.size == 0
