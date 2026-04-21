import time
import numpy as np

class MockSpatialGridIndex:
    def __init__(self, num_points=1000000, grid_dim_x=100, grid_dim_y=100):
        self.num_points = num_points
        self.grid_dim_x = grid_dim_x
        self.grid_dim_y = grid_dim_y

        # Mock Taichi fields as something that is slow to access element-wise in python
        class MockField:
            def __init__(self, arr):
                self.arr = arr
            def __getitem__(self, idx):
                # Simulating slow field access
                return self.arr[idx]
            @property
            def shape(self):
                return self.arr.shape
            def to_numpy(self):
                return self.arr.copy()

        self.cell_offsets = MockField(np.zeros((grid_dim_x, grid_dim_y), dtype=np.int32))
        self.cell_point_counts = MockField(np.ones((grid_dim_x, grid_dim_y), dtype=np.int32) * 10)
        self.indexed_point_indices = MockField(np.arange(grid_dim_x * grid_dim_y * 10, dtype=np.int32))

        # New arrays for optimization
        self.cell_offsets_np = self.cell_offsets.to_numpy()
        self.cell_point_counts_np = self.cell_point_counts.to_numpy()
        self.indexed_point_indices_np = self.indexed_point_indices.to_numpy()

    def query_points_in_cell_old(self, cell_x_idx: int, cell_y_idx: int) -> np.ndarray:
        if self.num_points == 0:
            return np.array([], dtype=np.int32)

        if not (
            0 <= cell_x_idx < self.grid_dim_x and 0 <= cell_y_idx < self.grid_dim_y
        ):
            return np.array([], dtype=np.int32)

        if self.indexed_point_indices.shape[0] == 0:
            return np.array([], dtype=np.int32)

        start_offset = self.cell_offsets[cell_x_idx, cell_y_idx]
        count = self.cell_point_counts[cell_x_idx, cell_y_idx]

        if count == 0:
            return np.array([], dtype=np.int32)

        all_indices_np = self.indexed_point_indices.to_numpy()
        return all_indices_np[start_offset : start_offset + count].copy()

    def query_points_in_cell_new(self, cell_x_idx: int, cell_y_idx: int) -> np.ndarray:
        if self.num_points == 0:
            return np.array([], dtype=np.int32)

        if not (
            0 <= cell_x_idx < self.grid_dim_x and 0 <= cell_y_idx < self.grid_dim_y
        ):
            return np.array([], dtype=np.int32)

        if self.indexed_point_indices_np.shape[0] == 0:
            return np.array([], dtype=np.int32)

        start_offset = self.cell_offsets_np[cell_x_idx, cell_y_idx]
        count = self.cell_point_counts_np[cell_x_idx, cell_y_idx]

        if count == 0:
            return np.array([], dtype=np.int32)

        return self.indexed_point_indices_np[start_offset : start_offset + count].copy()


def test():
    index = MockSpatialGridIndex()
    np.random.seed(42)
    num_queries = 10000
    queries = np.random.randint(0, 100, size=(num_queries, 2))

    start_time = time.time()
    for cx, cy in queries:
        index.query_points_in_cell_old(cx, cy)
    time_old = time.time() - start_time

    start_time = time.time()
    for cx, cy in queries:
        index.query_points_in_cell_new(cx, cy)
    time_new = time.time() - start_time

    print(f"Old approach: {time_old:.4f}s")
    print(f"New approach: {time_new:.4f}s")
    print(f"Speedup: {time_old / time_new:.2f}x")

if __name__ == "__main__":
    test()
