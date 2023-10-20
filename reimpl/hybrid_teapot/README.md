# Hybrid teapot / PBIR route

This directory is the active post-R6 research route:

1. remove density inside the official object AABB from the NeRF emitter;
2. expose separate full-NeRF and environment-only query semantics;
3. intersect the exported explicit teapot mesh and sample its material maps;
4. combine finite camera-ray NeRF segments with explicit surface shading.

The analytic-sphere R2–R6 and Demo Phase code remains unchanged as a frozen
correctness/regression baseline. It is not the final object-editing route.

