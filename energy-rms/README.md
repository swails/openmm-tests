# Tests to evaluate whether total energy RMS (standard deviation) follows expected quadratic depencence on timestep

These scripts evaluate energy conservation by examining the RMS fluctuations in total energy and drift over short intervals (a few ps) for several test systems:
* single diatom
* diatomic fluid with and without constraints
* dipolar diatomic fluid with and without constraints
* Lennard-Jones cluster 
* a Lennard-Jones fluid
* a TIP3P water box
* discharged TIP3P water box
* sodium chloride crystal with Ewald electrostatics
* discharged water box with LJ radii on hydrogens

The dependence of these properties for several parameter choices is evaluated:
* integrators (Verlet, VelocityVerlet)
* platform (CUDA, OpenCL, CPU, Reference)
* precision model (single, mixed, double)
* switching function on/off
* constraint tolerance (1e-10, 1e-5)
