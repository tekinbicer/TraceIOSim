Code base for simulating Trace I/O pattern.

Tomographic reconstruction is an compute-intensive task. However, it is also highly parallelizable. While the number of processors increase in the system, the I/O load on target file system also increases. Therefore, contention at the file system level becomes the main bottleneck. In order to model the performance of I/O, we created a I/O simulator which generates 3D datasets and pushes them to file system.

Requirements:
- Parallel HDF5
- MPI (e.g. MPICH)
- C++11 compatible compiler (e.g. clang)
