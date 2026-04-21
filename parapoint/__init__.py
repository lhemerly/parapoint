from algos.simple_average import simple
from algos.IDW import idw
from algos.minmax_elevation import min_elevation, max_elevation
from algos.terrain_analysis import hillshade, slope

# Legacy aliases for backward compatibility
create_dtm_with_taichi_averaging = simple
create_dtm_with_taichi_idw = idw
create_dtm_with_taichi_min = min_elevation
create_dtm_with_taichi_max = max_elevation

__all__ = [
    "simple",
    "idw",
    "min_elevation",
    "max_elevation",
    "hillshade",
    "slope",
    "create_dtm_with_taichi_averaging",
    "create_dtm_with_taichi_idw",
    "create_dtm_with_taichi_min",
    "create_dtm_with_taichi_max",
]
