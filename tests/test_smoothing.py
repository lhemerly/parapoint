import numpy as np
from algos.smoothing import gaussian_filter, fill_nodata


def test_gaussian_filter_basic():
    dtm = np.array(
        [[10.0, 10.0, 10.0], [10.0, 20.0, 10.0], [10.0, 10.0, 10.0]], dtype=np.float32
    )

    # Smooth with a large sigma to blur the spike
    smoothed = gaussian_filter(dtm, sigma=1.0, radius=1, nodata_value=-9999.0)

    # The center value should be significantly reduced from 20
    assert smoothed[1, 1] < 20.0
    # The corners should be slightly pulled up
    assert smoothed[0, 0] > 10.0


def test_gaussian_filter_nodata():
    dtm = np.array(
        [[10.0, 10.0, 10.0], [10.0, -9999.0, 10.0], [10.0, 10.0, 10.0]],
        dtype=np.float32,
    )

    smoothed = gaussian_filter(dtm, sigma=1.0, radius=1, nodata_value=-9999.0)

    # The nodata value should remain nodata
    assert smoothed[1, 1] == -9999.0
    # A neighbor should only average the valid values
    # In this uniform case, averaging 10s should result in ~10
    assert np.isclose(smoothed[0, 1], 10.0)


def test_fill_nodata_basic():
    dtm = np.array(
        [[10.0, 10.0, 10.0], [10.0, -9999.0, 10.0], [10.0, 10.0, 10.0]],
        dtype=np.float32,
    )

    filled = fill_nodata(dtm, max_iterations=1, radius=1, nodata_value=-9999.0)

    # Center should be filled with the average of its 8 neighbors (all 10.0)
    assert filled[1, 1] == 10.0


def test_fill_nodata_iterative():
    dtm = np.array(
        [[10.0, -9999.0, -9999.0], [10.0, -9999.0, -9999.0], [10.0, 10.0, 10.0]],
        dtype=np.float32,
    )

    # 1 iteration might not fill the top right corner
    filled_1 = fill_nodata(dtm, max_iterations=1, radius=1, nodata_value=-9999.0)
    assert filled_1[0, 2] == -9999.0

    # More iterations should fill it
    filled_3 = fill_nodata(dtm, max_iterations=3, radius=1, nodata_value=-9999.0)
    assert filled_3[0, 2] != -9999.0
    assert filled_3[0, 2] > 0


def test_empty_arrays():
    empty_dtm = np.array([[]], dtype=np.float32)

    res1 = gaussian_filter(empty_dtm)
    assert res1.size == 0

    res2 = fill_nodata(empty_dtm)
    assert res2.size == 0
