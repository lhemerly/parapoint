import unittest
import numpy as np
import taichi as ti # For potential ti.reset() if needed, and type hints
from algos.spatial_grid_index import SpatialGridIndex

# Helper function to sort and compare numpy arrays for assertEqual
def assert_np_arrays_equal_after_sorting(case, arr1, arr2, msg=None):
    if not isinstance(arr1, np.ndarray): arr1 = np.array(arr1)
    if not isinstance(arr2, np.ndarray): arr2 = np.array(arr2)
    case.assertTrue(np.array_equal(np.sort(arr1), np.sort(arr2)), msg)

class TestSpatialGridIndex(unittest.TestCase):

    def setUp(self):
        # Potentially reset Taichi before each test if issues arise with global state.
        # For now, assuming the module's own ti.init() on import is sufficient or
        # that successive ti.init() calls are handled gracefully by Taichi.
        # If "ti.kernel.AlreadyCompiledError" or similar occurs, uncommenting ti.reset() might be needed.
        # try:
        #     # print("Attempting ti.reset() in setUp")
        #     ti.reset()
        # except Exception as e:
        #     print(f"Error during ti.reset() in setUp: {e}")
        # Re-initialize Taichi to a known state for each test if required by Taichi's context management
        # However, SpatialGridIndex itself calls ti.init() when imported.
        # This setup might be complex due to that.
        pass

    # 1. Initialization & Basic Properties
    def test_empty_input(self):
        points = np.array([[]], dtype=np.float32).reshape(0, 2)
        index = SpatialGridIndex(points, resolution=1.0)
        self.assertEqual(index.num_points, 0)
        self.assertTrue(index.grid_dim_x == 0 or index.grid_dim_x == 1) # Can be 1 if min_x=max_x leads to resolution-sized grid
        self.assertTrue(index.grid_dim_y == 0 or index.grid_dim_y == 1)
        # Taichi fields might require a shape of at least 1. Actual shape is 1 for empty.
        self.assertEqual(index.indexed_point_indices.shape[0], 1)
        
        # Queries on empty index
        empty_cell_query = index.query_points_in_cell(0, 0)
        self.assertEqual(empty_cell_query.size, 0, "Cell query on empty index should return empty array.")
        
        empty_radius_query = index.query_points_in_radius(0.0, 0.0, 1.0)
        self.assertEqual(empty_radius_query.size, 0, "Radius query on empty index should return empty array.")

    def test_simple_points_auto_extent(self):
        points = np.array([[0.5, 0.5], [1.5, 1.5], [2.5, 2.5]], dtype=np.float32)
        resolution = 1.0
        index = SpatialGridIndex(points, resolution)
        
        self.assertEqual(index.resolution, resolution)
        self.assertAlmostEqual(index.min_x, 0.5)
        self.assertAlmostEqual(index.min_y, 0.5)
        self.assertAlmostEqual(index.max_x, 2.5) # Max_x from points
        self.assertAlmostEqual(index.max_y, 2.5) # Max_y from points
        
        # Grid dimensions: (2.5 - 0.5) / 1.0 = 2.0. ceil(2.0) = 2. So grid_dim should be 2.
        # If max_x is exclusive for cell calculation, points at 2.5 might need grid_dim 3.
        # Current logic: grid_dim_x = max(1, int(np.ceil((self.max_x - self.min_x) / self.resolution)))
        # grid_dim_x = max(1, int(ceil((2.5 - 0.5)/1.0))) = max(1, int(ceil(2.0))) = 2
        # grid_dim_y = max(1, int(ceil((2.5 - 0.5)/1.0))) = 2
        self.assertEqual(index.grid_dim_x, 2)
        self.assertEqual(index.grid_dim_y, 2)
        self.assertEqual(index.num_points, 3)

    def test_simple_points_custom_extent(self):
        points = np.array([[1.0, 1.0], [1.5, 1.5]], dtype=np.float32)
        resolution = 1.0
        extent = (0.0, 0.0, 3.0, 3.0) # min_x, min_y, max_x, max_y
        index = SpatialGridIndex(points, resolution, extent=extent)
        
        self.assertEqual(index.resolution, resolution)
        self.assertAlmostEqual(index.min_x, 0.0)
        self.assertAlmostEqual(index.min_y, 0.0)
        self.assertAlmostEqual(index.max_x, 3.0)
        self.assertAlmostEqual(index.max_y, 3.0)
        
        # grid_dim = ceil((3.0 - 0.0) / 1.0) = 3
        self.assertEqual(index.grid_dim_x, 3)
        self.assertEqual(index.grid_dim_y, 3)
        self.assertEqual(index.num_points, 2)

    def test_indexed_points_length(self):
        points = np.array([[0.1, 0.1], [0.2, 0.2], [2.0, 2.0], [2.5,2.5]], dtype=np.float32)
        # Ensure points are within an extent that would not filter them out.
        # If extent is not provided, it's auto-calculated.
        index_auto_extent = SpatialGridIndex(points, resolution=1.0)
        self.assertEqual(index_auto_extent.indexed_point_indices.shape[0], points.shape[0],
                         "Length of indexed_point_indices should match number of input points if all are within auto-calculated extent.")

        # Test with an extent that might exclude some points (though current kernels process all points given to them)
        # The kernels themselves filter points outside min_coord_x/y and grid_dim_x/y.
        # The number of points in indexed_point_indices is sum of cell_point_counts.
        # So this test is effectively checking if all points were assigned to cells.
        
        # Points outside a custom extent might still be processed if their coordinates
        # fall into cells calculated from that custom extent. The kernels use min_coord and grid_dim.
        # Let's make points and extent such that all points are clearly inside.
        points_inside = np.array([[0.5,0.5], [1.5,1.5]], dtype=np.float32)
        custom_extent = (0.0, 0.0, 2.0, 2.0) # max_x, max_y are exclusive for cell calculation effectively
        index_custom_extent = SpatialGridIndex(points_inside, resolution=1.0, extent=custom_extent)
        # Points (0.5,0.5) -> cell (0,0). Point (1.5,1.5) -> cell (1,1)
        # All points should be counted.
        self.assertEqual(index_custom_extent.indexed_point_indices.shape[0], points_inside.shape[0])

        # Test where some points are outside the *custom* extent, but might still be indexed
        # if their cell (calculated using custom extent's min_x, min_y) falls within grid_dim
        points_mixed = np.array([[0.5,0.5], [3.5,3.5]], dtype=np.float32) # Second point outside extent (0,0,2,2)
        index_mixed_extent = SpatialGridIndex(points_mixed, resolution=1.0, extent=custom_extent)
        # Point (0.5,0.5) is in cell (0,0) - counted.
        # Point (3.5,3.5): cell_x = floor((3.5-0.0)/1.0) = 3. grid_dim_x = ceil((2.0-0.0)/1.0) = 2.
        # This point (3.5,3.5) is outside the grid defined by the extent (0,0,2,2) because cell_x=3 >= grid_dim_x=2.
        # So it should be ignored by assign_points_to_grid_and_count_kernel.
        self.assertEqual(index_mixed_extent.indexed_point_indices.shape[0], 1,
                         "Only points falling within cells defined by custom extent should be indexed.")


    # 2. query_points_in_cell Method
    def test_qpc_empty_cell(self):
        points = np.array([[0.5, 0.5]], dtype=np.float32)
        index = SpatialGridIndex(points, resolution=1.0, extent=(0,0,2,2)) # point [0] is in cell (0,0)
        indices = index.query_points_in_cell(1, 1) # Cell (1,1) should be empty
        self.assertEqual(indices.size, 0)

    def test_qpc_single_point_cell(self):
        points = np.array([[0.5, 0.5]], dtype=np.float32)
        index = SpatialGridIndex(points, resolution=1.0, extent=(0,0,2,2)) # point [0] is in cell (0,0)
        indices = index.query_points_in_cell(0, 0)
        assert_np_arrays_equal_after_sorting(self, indices, [0])

    def test_qpc_multiple_points_cell(self):
        points = np.array([[0.5, 0.5], [0.6, 0.6], [1.5, 0.5]], dtype=np.float32) # First two in cell (0,0)
        index = SpatialGridIndex(points, resolution=1.0, extent=(0,0,2,2))
        indices_cell_00 = index.query_points_in_cell(0, 0)
        assert_np_arrays_equal_after_sorting(self, indices_cell_00, [0, 1])
        indices_cell_10 = index.query_points_in_cell(1,0) # Third point in cell (1,0)
        assert_np_arrays_equal_after_sorting(self, indices_cell_10, [2])


    def test_qpc_out_of_bounds(self):
        points = np.array([[0.5, 0.5]], dtype=np.float32)
        index = SpatialGridIndex(points, resolution=1.0, extent=(0,0,1,1)) # grid_dim is (1,1), valid cell (0,0)
        indices = index.query_points_in_cell(1, 0) # Out of bounds
        self.assertEqual(indices.size, 0)
        indices = index.query_points_in_cell(0, 1) # Out of bounds
        self.assertEqual(indices.size, 0)
        indices = index.query_points_in_cell(-1, 0) # Out of bounds
        self.assertEqual(indices.size, 0)

    def test_qpc_all_points_in_one_cell(self):
        points = np.array([[0.1,0.1], [0.2,0.2], [0.3,0.3], [0.4,0.4]], dtype=np.float32)
        index = SpatialGridIndex(points, resolution=1.0, extent=(0,0,1,1)) # All in cell (0,0)
        indices = index.query_points_in_cell(0,0)
        assert_np_arrays_equal_after_sorting(self, indices, [0,1,2,3])

    # 3. query_points_in_radius Method
    def test_qpr_no_points_found(self):
        points = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32)
        index = SpatialGridIndex(points, resolution=1.0, extent=(-1,-1,11,11))
        indices = index.query_points_in_radius(5.0, 5.0, 0.5) # Query far from points
        self.assertEqual(indices.size, 0)

    def test_qpr_find_one_point(self):
        points = np.array([[0.0, 0.0], [5.0, 5.0]], dtype=np.float32)
        index = SpatialGridIndex(points, resolution=1.0, extent=(-1,-1,6,6))
        indices = index.query_points_in_radius(0.1, 0.1, 0.5) # Should find point [0]
        assert_np_arrays_equal_after_sorting(self, indices, [0])

    def test_qpr_find_multiple_points(self):
        points = np.array([[0.0,0.0], [0.1,0.1], [5.0,5.0]], dtype=np.float32)
        index = SpatialGridIndex(points, resolution=1.0, extent=(-1,-1,6,6))
        indices = index.query_points_in_radius(0.0, 0.0, 0.5) # Should find points [0] and [1]
        assert_np_arrays_equal_after_sorting(self, indices, [0,1])

    def test_qpr_radius_covers_multiple_cells(self):
        points = np.array([[0.5,0.5], [1.5,1.5]], dtype=np.float32) # Pt [0] in cell (0,0), Pt [1] in cell (1,1)
        index = SpatialGridIndex(points, resolution=1.0, extent=(0,0,2,2))
        indices = index.query_points_in_radius(1.0, 1.0, 0.8) # Query at (1,1) with radius 0.8
                                                              # Dist to (0.5,0.5) is sqrt(0.5^2+0.5^2) = sqrt(0.5) approx 0.707
                                                              # Dist to (1.5,1.5) is sqrt(0.5^2+0.5^2) = sqrt(0.5) approx 0.707
        assert_np_arrays_equal_after_sorting(self, indices, [0,1])

    def test_qpr_all_points_in_radius(self):
        points = np.array([[0.0,0.0], [1.0,1.0], [10.0,10.0]], dtype=np.float32)
        index = SpatialGridIndex(points, resolution=1.0, extent=(-1,-1,11,11))
        indices = index.query_points_in_radius(5.0, 5.0, 10.0) # Large radius
        assert_np_arrays_equal_after_sorting(self, indices, [0,1,2])

    def test_qpr_points_on_boundary(self):
        # Kernel uses dist_sq < radius_sq, so points exactly on boundary are excluded.
        points = np.array([[0.0,0.0], [1.0,0.0]], dtype=np.float32)
        index = SpatialGridIndex(points, resolution=0.5, extent=(-1,-1,2,2))
        
        # Point [1] is at distance 1.0 from query point (0.0,0.0)
        indices = index.query_points_in_radius(0.0, 0.0, 1.0) 
        # Should only find point [0] (dist 0), point [1] is excluded as dist == radius
        assert_np_arrays_equal_after_sorting(self, indices, [0])

        # If radius is slightly larger, it should be included
        indices_larger_radius = index.query_points_in_radius(0.0, 0.0, 1.0001)
        assert_np_arrays_equal_after_sorting(self, indices_larger_radius, [0,1])


    def test_qpr_zero_radius(self):
        points = np.array([[0.0,0.0], [0.001, 0.001]], dtype=np.float32)
        index = SpatialGridIndex(points, resolution=0.01, extent=(-1,-1,1,1))
        indices = index.query_points_in_radius(0.0, 0.0, 0.0) # Zero radius
        # Kernel uses dist_sq < radius_sq. If radius is 0, radius_sq is 0.
        # Only points with dist_sq < 0 would be included, which is impossible.
        # So, it should return empty.
        self.assertEqual(indices.size, 0) 

        # Test with a point exactly at query center
        points_at_center = np.array([[0.5, 0.5], [1.0,1.0]], dtype=np.float32)
        index_at_center = SpatialGridIndex(points_at_center, resolution=0.1, extent=(0,0,2,2))
        indices_at_center = index_at_center.query_points_in_radius(0.5,0.5,0.0)
        self.assertEqual(indices_at_center.size, 0) # Still empty due to strict inequality


    def test_qpr_query_outside_extent(self):
        points = np.array([[0.0,0.0], [1.0,1.0]], dtype=np.float32)
        index = SpatialGridIndex(points, resolution=1.0, extent=(0,0,2,2))
        indices = index.query_points_in_radius(100.0, 100.0, 1.0) # Query far outside
        self.assertEqual(indices.size, 0)

    # 4. Specific Scenarios & Edge Cases
    def test_collinear_points(self):
        points = np.array([[0.0,0.0], [1.0,1.0], [2.0,2.0], [3.0,3.0]], dtype=np.float32)
        index = SpatialGridIndex(points, resolution=1.0, extent=(-1,-1,4,4))
        # Cell check for extent=(-1,-1,4,4), res=1.0:
        # Point (0,0) (index 0) -> cell ( floor((0-(-1))/1), floor((0-(-1))/1) ) = (1,1)
        # Point (1,1) (index 1) -> cell ( floor((1-(-1))/1), floor((1-(-1))/1) ) = (2,2)
        indices_c00 = index.query_points_in_cell(0,0) # Cell (0,0) should be empty
        assert_np_arrays_equal_after_sorting(self, indices_c00, [])
        indices_c11 = index.query_points_in_cell(1,1) # Cell (1,1) should contain point 0
        assert_np_arrays_equal_after_sorting(self, indices_c11, [0])
        
        # Radius check
        indices_rad = index.query_points_in_radius(1.5, 1.5, 0.8) # dist to (1,1) is ~0.707, dist to (2,2) is ~0.707
        assert_np_arrays_equal_after_sorting(self, indices_rad, [1,2])

    def test_coincident_points(self):
        points = np.array([[1.0,1.0], [1.0,1.0], [1.0,1.0]], dtype=np.float32)
        index = SpatialGridIndex(points, resolution=0.5, extent=(0,0,2,2))
        # All points are in cell (2,2) for res=0.5, min_coord=0. floor((1.0-0)/0.5) = 2
        indices_cell = index.query_points_in_cell(2,2) 
        assert_np_arrays_equal_after_sorting(self, indices_cell, [0,1,2])

        indices_rad = index.query_points_in_radius(1.0, 1.0, 0.1)
        assert_np_arrays_equal_after_sorting(self, indices_rad, [0,1,2])
        
        indices_rad_miss = index.query_points_in_radius(1.0, 1.0, 0.0) # Strict inequality
        self.assertEqual(indices_rad_miss.size, 0)


    def test_points_on_cell_boundaries(self):
        # Point (1.0, 1.0) on boundary between (0,0)-(1,0) and (0,1)-(1,1) if res=1, min_coord=0
        # cell_x = floor((px - min_x)/res). If px=1.0, min_x=0, res=1 => cell_x = floor(1.0) = 1
        # So (1.0,1.0) goes into cell (1,1)
        points = np.array([[0.9,0.9], [1.0,1.0], [1.1,1.1]], dtype=np.float32)
        index = SpatialGridIndex(points, resolution=1.0, extent=(0,0,3,3))
        
        # (0.9,0.9) -> cell (0,0)
        indices_c00 = index.query_points_in_cell(0,0)
        assert_np_arrays_equal_after_sorting(self, indices_c00, [0])
        
        # (1.0,1.0) -> cell (1,1)
        # (1.1,1.1) -> cell (1,1)
        indices_c11 = index.query_points_in_cell(1,1)
        assert_np_arrays_equal_after_sorting(self, indices_c11, [1,2])

        # Check radius query around boundary point (1.0,1.0)
        indices_rad = index.query_points_in_radius(1.0, 1.0, 0.05) # Small radius around (1.0,1.0)
        assert_np_arrays_equal_after_sorting(self, indices_rad, [1]) # Should only find point [1]

        indices_rad_larger = index.query_points_in_radius(1.0, 1.0, 0.15) # Radius to include all 3
        # dist( (1.0,1.0) to (0.9,0.9) ) = sqrt(0.1^2 + 0.1^2) = sqrt(0.02) approx 0.1414
        # dist( (1.0,1.0) to (1.1,1.1) ) = sqrt(0.1^2 + 0.1^2) = sqrt(0.02) approx 0.1414
        assert_np_arrays_equal_after_sorting(self, indices_rad_larger, [0,1,2])

    def test_single_point_initialization(self):
        points = np.array([[3.5, 4.5]], dtype=np.float32)
        index = SpatialGridIndex(points, resolution=1.0, extent=(0.0, 0.0, 5.0, 5.0))
        self.assertEqual(index.num_points, 1)
        self.assertEqual(index.indexed_point_indices.shape[0], 1)
        
        # Point (3.5, 4.5) -> cell (3,4)
        cell_indices = index.query_points_in_cell(3,4)
        assert_np_arrays_equal_after_sorting(self, cell_indices, [0])
        
        radius_indices = index.query_points_in_radius(3.5, 4.5, 0.1)
        assert_np_arrays_equal_after_sorting(self, radius_indices, [0])
        
        empty_radius_indices = index.query_points_in_radius(0.0, 0.0, 1.0)
        self.assertEqual(empty_radius_indices.size, 0)

if __name__ == '__main__':
    unittest.main()
