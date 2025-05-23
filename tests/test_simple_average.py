import pytest
import numpy as np
from parapoint.algos.simple_average import create_dtm_with_taichi_averaging

# Test with basic input
def test_simple_average_basic():
    points = np.array([[0.5, 0.5, 10], [1.5, 1.5, 20]], dtype=np.float32)
    dtm = create_dtm_with_taichi_averaging(points, 1.0)
    assert dtm.shape == (2, 2)
    assert dtm[0, 0] == 10
    assert dtm[1, 1] == 20

# Test with no points
def test_simple_average_no_points():
    points = np.empty((0, 3), dtype=np.float32)
    dtm = create_dtm_with_taichi_averaging(points, 1.0)
    assert dtm.shape == (1, 0) # Behavior for empty array

# Test with custom extent
def test_simple_average_custom_extent():
    points = np.array([[0.5, 0.5, 10]], dtype=np.float32)
    dtm = create_dtm_with_taichi_averaging(points, 1.0, dtm_extent_user=(0, 0, 1, 1))
    assert dtm.shape == (1, 1)
    assert dtm[0, 0] == 10

# Test with nodata value
def test_simple_average_nodata():
    points = np.array([[0.5, 0.5, 10]], dtype=np.float32)
    dtm = create_dtm_with_taichi_averaging(points, 1.0, nodata_value=-1)
    assert dtm[0, 1] == -1 # Assuming second cell is nodata

# Test with multiple points in one cell
def test_simple_average_multiple_points_in_cell():
    points = np.array([[0.5, 0.5, 10], [0.6, 0.6, 20]], dtype=np.float32)
    dtm = create_dtm_with_taichi_averaging(points, 1.0)
    assert dtm[0, 0] == 15 # Average of 10 and 20
