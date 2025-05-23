import pytest
import numpy as np
from parapoint.algos.IDW import create_dtm_with_taichi_idw

# Test with basic input
def test_idw_basic():
    points = np.array([[0.5, 0.5, 10], [1.5, 1.5, 20]], dtype=np.float32)
    dtm = create_dtm_with_taichi_idw(points, 1.0, 1.0)
    assert dtm.shape == (2, 2) 
    # Exact values depend on IDW, so check if they are not nodata
    assert dtm[0,0] != -9999.0 
    assert dtm[1,1] != -9999.0

# Test with no points
def test_idw_no_points():
    points = np.empty((0, 3), dtype=np.float32)
    dtm = create_dtm_with_taichi_idw(points, 1.0, 1.0)
    assert dtm.shape == (1,0) # Behavior for empty array

# Test with custom extent
def test_idw_custom_extent():
    points = np.array([[0.5, 0.5, 10]], dtype=np.float32)
    dtm = create_dtm_with_taichi_idw(points, 1.0, 1.0, dtm_extent_user=(0, 0, 1, 1))
    assert dtm.shape == (1, 1)
    assert dtm[0,0] != -9999.0

# Test with nodata value
def test_idw_nodata():
    points = np.array([[0.5, 0.5, 10]], dtype=np.float32)
    dtm = create_dtm_with_taichi_idw(points, 1.0, 0.1, nodata_value=-1) # Small radius to ensure nodata
    assert dtm[0, 1] == -1 # Assuming second cell is nodata

# Test with different power
def test_idw_different_power():
    points = np.array([[0.5, 0.5, 10], [1.5, 1.5, 20]], dtype=np.float32)
    dtm_power1 = create_dtm_with_taichi_idw(points, 1.0, 1.0, power=1.0)
    dtm_power3 = create_dtm_with_taichi_idw(points, 1.0, 1.0, power=3.0)
    assert dtm_power1.shape == (2,2)
    assert dtm_power3.shape == (2,2)
    # Values should differ with power, but exact check is complex
    assert not np.array_equal(dtm_power1, dtm_power3)
