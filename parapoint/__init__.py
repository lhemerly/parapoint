from algos.simple_average import simple
from algos.IDW import idw
from algos.min_max import min_z, max_z
from algos.terrain_derivatives import (
    calculate_terrain_derivatives,
    calculate_tpi,
    calculate_tri,
)
from algos.smoothing import gaussian_filter, fill_nodata

# Legacy aliases for backward compatibility
create_dtm_with_taichi_averaging = simple
create_dtm_with_taichi_idw = idw

__all__ = [
    "simple",
    "idw",
    "min_z",
    "max_z",
    "calculate_terrain_derivatives",
    "calculate_tpi",
    "calculate_tri",
    "gaussian_filter",
    "fill_nodata",
    "create_dtm_with_taichi_averaging",
    "create_dtm_with_taichi_idw",
]
