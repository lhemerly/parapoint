# Parapoint

Parapoint is a Python library for performing fast point cloud interpolation, primarily designed for LiDAR data. It leverages the Taichi programming language to accelerate computations on multi-core CPUs or GPUs.

The library provides methods for creating Digital Terrain Models (DTMs) from unorganized point cloud data (NumPy arrays).

## Features

- **Fast Interpolation:** Utilizes Taichi for parallel processing, significantly speeding up interpolation tasks.
- **Multiple Algorithms:** Includes:
    - **Inverse Distance Weighting (IDW):** A common method for interpolating scattered data points.
    - **Simple Averaging:** A basic gridding approach that averages Z values of points falling within each grid cell.
- **NumPy Integration:** Accepts and returns NumPy arrays, making it easy to integrate into existing Python workflows.
- **Customizable:** Allows control over DTM resolution, extent, search radius (for IDW), and nodata values.

## Installation

To use Parapoint, you need to have Python installed (>=3.13 recommended) along with NumPy and Taichi.

1.  **Clone the repository (if applicable) or install directly.**

    If you have the source code, you can install it locally using `uv`:

    ```bash
    uv pip install .
    ```

    (This assumes you have a `pyproject.toml` file set up for installation.)

2.  **Install dependencies:**

    The necessary dependencies are listed in `pyproject.toml`. You can create a virtual environment and install them using `uv`:

    ```bash
    uv venv  # Create a virtual environment (e.g., .venv)
    source .venv/bin/activate  # Activate the virtual environment (Linux/macOS)
    # For Windows: .venv\\Scripts\\activate
    uv pip install -r requirements.txt  # If you have a requirements.txt
    # Or, if your dependencies are managed by pyproject.toml directly:
    uv pip install numpy taichi
    ```

## Usage

Here's a basic example of how to use Parapoint:

```python
import numpy as np
import parapoint

# 1. Sample Data (replace with your actual LiDAR data)
# Points should be a NumPy array of shape (N, 3) with columns [X, Y, Z]
num_points = 100000
extent = 100.0
ground_points = np.random.rand(num_points, 3) * extent
ground_points[:, 2] = (np.sin(ground_points[:, 0] / 10) * 5 +
                       np.cos(ground_points[:, 1] / 10) * 5 +
                       np.random.rand(num_points) * 1.5) # Example Z values

# 2. Define DTM Parameters
dtm_resolution = 1.0  # e.g., 1 meter resolution

# --- Using Simple Averaging ---
print("\n--- Running Simple Average DTM --- ")
dtm_avg = parapoint.create_dtm_with_taichi_averaging(
    ground_points_xyz=ground_points,
    dtm_resolution=dtm_resolution,
    nodata_value=-9999.0
)

if dtm_avg.size > 0:
    print(f"Simple Average DTM shape: {dtm_avg.shape}")
    valid_avg_values = dtm_avg[dtm_avg != -9999.0]
    if valid_avg_values.size > 0:
        print(f"Min/Max (avg): {np.min(valid_avg_values):.2f} / {np.max(valid_avg_values):.2f}")
else:
    print("Simple Average DTM is empty.")


# --- Using Inverse Distance Weighting (IDW) ---
print("\n--- Running IDW DTM --- ")
idw_search_radius = 5.0 # Search radius for IDW
idw_power = 2.0         # Power parameter for IDW

dtm_idw = parapoint.create_dtm_with_taichi_idw(
    ground_points_xyz=ground_points,
    dtm_resolution=dtm_resolution,
    search_radius=idw_search_radius,
    power=idw_power,
    nodata_value=-9999.0
)

if dtm_idw.size > 0:
    print(f"IDW DTM shape: {dtm_idw.shape}")
    valid_idw_values = dtm_idw[dtm_idw != -9999.0]
    if valid_idw_values.size > 0:
        print(f"Min/Max (IDW): {np.min(valid_idw_values):.2f} / {np.max(valid_idw_values):.2f}")
else:
    print("IDW DTM is empty.")

# Further processing or visualization of dtm_avg or dtm_idw can be done here.
# For example, using matplotlib or a GIS library like rasterio to save as GeoTIFF.

# Example: Basic plot with Matplotlib (optional)
# import matplotlib.pyplot as plt
# if dtm_idw.size > 0 and np.any(dtm_idw != -9999.0):
#     plt.figure(figsize=(10, 8))
#     plt.imshow(dtm_idw, cmap='terrain', vmin=np.min(valid_idw_values), vmax=np.max(valid_idw_values))
#     plt.colorbar(label='Elevation (m)')
#     plt.title('IDW Interpolated DTM')
#     plt.xlabel('Grid X')
#     plt.ylabel('Grid Y')
#     plt.show()
# else:
#     print("Skipping plot as DTM is empty or all NoData.")

```

### `create_dtm_with_taichi_averaging`

```python
def create_dtm_with_taichi_averaging(
    ground_points_xyz: np.ndarray, 
    dtm_resolution: float,
    dtm_extent_user: tuple = None, # Optional: (min_x, min_y, max_x, max_y)
    nodata_value: float = -9999.0
) -> np.ndarray:
```

-   `ground_points_xyz`: NumPy array of shape (N,3) containing X, Y, Z of ground points.
-   `dtm_resolution`: The desired resolution of the output DTM grid.
-   `dtm_extent_user` (optional): Tuple `(min_x, min_y, max_x, max_y)` to define the DTM area. If `None`, it's calculated from input points.
-   `nodata_value`: Value for DTM cells with no points.

### `create_dtm_with_taichi_idw`

```python
def create_dtm_with_taichi_idw(
    ground_points_xyz: np.ndarray, 
    dtm_resolution: float,
    search_radius: float,
    power: float = 2.0,
    dtm_extent_user: tuple = None, # Optional: (min_x, min_y, max_x, max_y)
    nodata_value: float = -9999.0
) -> np.ndarray:
```

-   `ground_points_xyz`: NumPy array of shape (N,3) containing X, Y, Z of ground points.
-   `dtm_resolution`: The desired resolution of the output DTM grid.
-   `search_radius`: Maximum distance from a DTM cell center to consider input points for IDW.
-   `power`: The power parameter for IDW (typically 1, 2, or 3).
-   `dtm_extent_user` (optional): Tuple `(min_x, min_y, max_x, max_y)` to define the DTM area. If `None`, it's calculated from input points.
-   `nodata_value`: Value for DTM cells with no points within the search radius.

## Taichi Backend

By default, Taichi will try to initialize with a GPU backend (`ti.gpu`). If a compatible GPU is not found or an error occurs, it will fall back to the CPU backend (`ti.cpu`). You can observe messages in the console indicating which backend is being used.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## License

This project is licensed under the MIT License - see the LICENSE file for details (assuming MIT, add a LICENSE file if you choose one).
