import pytest
import numpy as np
from algos.simple_average import simple

def test_dos_prevention():
    points = np.array([[0.0, 0.0, 10.0], [100.0, 100.0, 20.0]], dtype=np.float32)
    # Very small resolution will cause huge memory allocation
    with pytest.raises(ValueError):
        simple(points, dtm_resolution=1e-6)
