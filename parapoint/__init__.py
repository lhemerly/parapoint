from algos.simple_average import simple
from algos.IDW import idw

# Legacy aliases for backward compatibility
create_dtm_with_taichi_averaging = simple
create_dtm_with_taichi_idw = idw

__all__ = [
    "simple",
    "idw",
    "create_dtm_with_taichi_averaging",
    "create_dtm_with_taichi_idw"
]
