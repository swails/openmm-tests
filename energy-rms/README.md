# Energy conservation tests for OpenMM

These scripts evaluate energy conservation by examining the RMS fluctuations in total energy and drift over short intervals (a few ps) for several test systems:
* a Lennard-Jones fluid
* a TIP3P water box

The dependence of these properties for several parameter choices is evaluated:
* platform (CUDA, OpenCL, CPU, Reference)
* precision model (single, mixed, double)
* switching function on/off
* constraint tolerance (1e-10, 1e-5)
