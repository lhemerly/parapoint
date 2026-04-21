import numpy as np
from algos.IDW import idw
from algos.simple_average import simple

points = np.array([[0.0, 0.0, 10.0], [1000.0, 1000.0, 20.0]], dtype=np.float32)

print("Testing IDW small resolution...")
dtm = idw(points, 1e-4, 1.0)
assert dtm.shape == (1, 0)

print("Testing simple small resolution...")
dtm_simple = simple(points, 1e-4)
assert dtm_simple.shape == (1, 0)

print("Testing zero resolution IDW...")
dtm_zero = idw(points, 0.0, 1.0)
assert dtm_zero.shape == (1, 0)

print("Testing zero resolution simple...")
dtm_zero_simple = simple(points, 0.0)
assert dtm_zero_simple.shape == (1, 0)

print("All tests passed!")
