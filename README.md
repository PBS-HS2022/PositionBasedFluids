<center>
<h1>Position Based Fluids</h1>
<span>Group 10 - Yuto, Andrew, Siddharth</span>
</center>

---

This is our submission for the final project of the PBSCG HS2022 course. Our topic is Position Based Fluids, mainly implementing Macklin and Müller's [Position Based Fluids](https://mmacklin.com/pbf_sig_preprint.pdf) paper in 2D and 3D.

The codebase is built on top of Assignment 4, with large modifications in the simulation. We initially had some of the codebase translated to CUDA for speed, but this has been abandoned for our final submission since the position-based fluids were fast enough on the CPU.

## Instructions to run

- Run CMake on the `CMakeLists.txt` file, and make either the `pbd_2d` or `pbd_3d` executables.
- Click on "Run Simulation".

## Details

- Much of the simulation logic is within `FluidSim.cpp` for both 2D and 3D. The
`integrateSPH()` function in particular (named back when SPH was used, now still
applicable to Position-Based Fluids) is important.
