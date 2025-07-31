from threadpoolctl import threadpool_limits
import numpy as np
import time

def heavy_computation():
    # Example: a big matrix multiplication
    a = np.random.rand(3000, 3000)
    b = np.random.rand(3000, 3000)
    return a @ b

print("Running with default thread count...")
start = time.time()
heavy_computation()
print(f"Time elapsed: {time.time() - start:.2f} seconds\n")

print("Running with threadpool_limits(limits=2)...")
with threadpool_limits(limits=2):
    start = time.time()
    heavy_computation()
    print(f"Time elapsed: {time.time() - start:.2f} seconds")
