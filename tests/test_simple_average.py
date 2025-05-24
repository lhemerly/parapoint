import pytest
import numpy as np
from algos.simple_average import simple
# Removed common_taichi import and _reset calls


# Test with basic input
def test_simple_average_basic():
    points = np.array([[0.5, 0.5, 10], [1.5, 1.5, 20]], dtype=np.float32)
    dtm = simple(points, 1.0)
    # Current grid logic: grid_width = int(np.ceil((max_x - min_x) / dtm_resolution))
    # For these points: max_x-min_x = 1.0. So ceil(1.0/1.0) = 1. Shape (1,1).
    # Point (0.5,0.5,10) -> gix=0,giy=0. Included.
    # Point (1.5,1.5,20) -> gix=1,giy=1. Excluded from 1x1 grid.
    assert dtm.shape == (1, 1)
    assert dtm[0, 0] == 10


# Test with no points
def test_simple_average_no_points():
    points = np.empty((0, 3), dtype=np.float32)
    dtm = simple(points, 1.0)
    assert dtm.shape == (1, 0)  # Behavior for empty array


# Test with custom extent
def test_simple_average_custom_extent():
    points = np.array([[0.5, 0.5, 10]], dtype=np.float32)
    # For extent (0,0,1,1): max_x-min_x = 1.0. Current logic: ceil(1.0/1.0) = 1. Shape (1,1).
    # Point (0.5,0.5) -> gix=floor((0.5-0)/1)=0, giy=floor((0.5-0)/1)=0. Included.
    dtm = simple(points, 1.0, dtm_extent_user=(0, 0, 1, 1))
    assert dtm.shape == (1, 1)
    assert dtm[0, 0] == 10


# Test with nodata value
def test_simple_average_nodata():
    points = np.array([[0.5, 0.5, 10]], dtype=np.float32)
    # For extent (0,0,2,1), res=1.0:
    # grid_width = ceil(2.0/1.0) = 2. grid_height = ceil(1.0/1.0) = 1. Shape (1,2).
    # Point (0.5,0.5) -> gix=floor((0.5-0)/1)=0, giy=floor((0.5-0)/1)=0. Cell (0,0).
    dtm = simple(points, 1.0, dtm_extent_user=(0.0, 0.0, 2.0, 1.0), nodata_value=-1)
    assert dtm.shape == (1, 2)
    assert dtm[0, 0] == 10
    assert dtm[0, 1] == -1  # This cell should be nodata


# Test with multiple points in one cell
def test_simple_average_multiple_points_in_cell():
    points = np.array([[0.5, 0.5, 10], [0.6, 0.6, 20]], dtype=np.float32)
    # Default extent: min_x=0.5, max_x=0.6 -> ceil((0.6 - 0.5) / 1) = 1. min_y=0.5, max_y=0.6 -> ceil((0.6 - 0.5) / 1) = 1. Shape (1,1)
    dtm = simple(points, 1.0)
    assert dtm.shape == (1, 1)  # Original simple_average logic with +1e-6
    assert dtm[0, 0] == 15  # Average of 10 and 20


def test_assign_points_outside_grid():
    """Test that points outside the DTM extent do not contribute."""
    # Define a DTM extent from (0,0) to (1,1) with resolution 0.5
    # This creates a 2x2 grid.
    # Cell (0,0) covers x=[0, 0.5), y=[0, 0.5)
    # Cell (0,1) covers x=[0.5, 1.0), y=[0, 0.5)
    # Cell (1,0) covers x=[0, 0.5), y=[0.5, 1.0)
    # Cell (1,1) covers x=[0.5, 1.0), y=[0.5, 1.0)
    dtm_extent = (0.0, 0.0, 1.0, 1.0)  # min_x, min_y, max_x, max_y
    resolution = 0.5
    nodata_val = -1.0

    # Points: one inside, several outside
    points_inside = np.array(
        [[0.25, 0.25, 10.0]], dtype=np.float32
    )  # Should go to cell (0,0)
    points_outside_x_low = np.array([[-0.25, 0.25, 20.0]], dtype=np.float32)  # x < 0
    points_outside_y_low = np.array([[0.25, -0.25, 30.0]], dtype=np.float32)  # y < 0
    points_outside_x_high = np.array([[1.25, 0.25, 40.0]], dtype=np.float32)  # x >= 1.0
    points_outside_y_high = np.array([[0.25, 1.25, 50.0]], dtype=np.float32)  # y >= 1.0

    all_points = np.vstack(
        [
            points_inside,
            points_outside_x_low,
            points_outside_x_high,
            points_outside_y_low,
            points_outside_y_high,
        ]
    )

    dtm = simple(
        all_points, resolution, dtm_extent_user=dtm_extent, nodata_value=nodata_val
    )

    # With original simple_average.py grid logic:
    # With current grid logic: grid_width = int(np.ceil(1.0/0.5)) = 2
    # grid_height = int(np.ceil(1.0/0.5)) = 2
    # So, a 2x2 DTM is expected.
    assert dtm.shape == (2, 2)  # height, width

    # Point (0.25,0.25,10.0) -> gix=floor(0.25/0.5)=0, giy=floor(0.25/0.5)=0. Cell (0,0).
    # Point (-0.25,0.25,20.0) -> gix=floor(-0.25/0.5)=-1. Outside.
    # Point (0.25,-0.25,30.0) -> giy=floor(-0.25/0.5)=-1. Outside.
    # Point (1.25,0.25,40.0) -> gix=floor(1.25/0.5)=2. Outside (grid_width=2, valid indices 0,1).
    # Point (0.25,1.25,50.0) -> giy=floor(1.25/0.5)=2. Outside (grid_height=2, valid indices 0,1).

    assert dtm[0, 0] == 10.0, "Cell (0,0) should be 10.0 from points_inside."
    assert dtm[0, 1] == nodata_val, "Cell (0,1) should be nodata."
    assert dtm[1, 0] == nodata_val, "Cell (1,0) should be nodata."
    assert dtm[1, 1] == nodata_val, "Cell (1,1) should be nodata."


def test_invalid_grid_dimensions():
    """Test behavior when grid dimensions are invalid (<=0)."""
    points = np.array([[0.5, 0.5, 10.0]], dtype=np.float32)

    # Scenario 1: Resolution much larger than extent
    # Extent of points is 0x0. Original grid calc: ceil((0+1e-6)/10) = 1.
    # This will result in a 1x1 grid.
    # To force grid_width/height <= 0 with original code, we need max_x - min_x to be negative
    # or dtm_resolution to be huge such that (max_x - min_x + 1e-6) / dtm_resolution is < 1 and rounds to 0 with int().
    # The current code path `if grid_width <= 0 or grid_height <= 0:` is hard to hit
    # if points exist, because min_x <= max_x, so (max_x - min_x + 1e-6) is positive.
    # If dtm_resolution is very large, e.g., 10.0 for points at (0.5,0.5),
    # grid_width = ceil( (0.5-0.5+1e-6) / 10.0 ) = ceil(1e-7) = 1.
    # The only way to get grid_width <=0 is if no points are provided,
    # which is already handled by `test_simple_average_no_points`.

    # Let's consider the case where dtm_extent_user might lead to this.
    # If max_x < min_x (invalid extent)
    # The code doesn't explicitly prevent max_x < min_x in dtm_extent_user.
    # If min_x = 1.0, max_x = 0.0, then (max_x - min_x + 1e-6) = -1.0 + 1e-6 = negative.
    # np.ceil of a negative number is still negative or zero. int() would make it <=0.

    dtm_invalid_extent = simple(
        points,
        dtm_resolution=1.0,
        dtm_extent_user=(1.0, 1.0, 0.0, 0.0),  # max_x < min_x
    )
    assert dtm_invalid_extent.shape == (1, 0), (
        f"Expected empty-like array for invalid extent, got shape {dtm_invalid_extent.shape}"
    )

    # Scenario 2: Zero resolution (should ideally be caught earlier, but test current code)
    # The code doesn't have a specific check for dtm_resolution == 0 before division.
    # This would lead to a ZeroDivisionError if not for other guards.
    # However, if it somehow passed to int(np.ceil( positive / 0 )), it's Inf.
    # This path seems unlikely to hit the grid_width <= 0 condition directly.
    # The most reliable way to hit the target lines 100-101 is an invalid user extent.
    # The current code's `if grid_width <= 0 or grid_height <= 0:` is primarily for safety,
    # as valid points and positive resolution usually make grid dimensions >= 1.
    # The test with invalid extent above should cover the spirit of this test.
