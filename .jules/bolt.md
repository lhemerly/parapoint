## 2024-04-22 - Avoid dynamic `ti.field` allocation in tight Taichi calls
**Learning:** In Parapoint's Taichi kernels, dynamically instantiating `ti.field` instances inside Python wrapper functions before calling kernels causes excessive allocation overhead and memory operations from `.to_numpy()`. This pattern severely penalizes repeated calls.
**Action:** Replace `ti.field` definitions inside Taichi caller functions with pre-allocated `numpy` arrays. Update the kernel signatures to use `ti.types.ndarray(ti.f32, ndim=2)` instead of `ti.template()` and pass the `numpy` arrays directly.
