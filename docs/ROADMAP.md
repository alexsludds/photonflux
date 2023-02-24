# Roadmap
This document contains the roadmap of photonflux. It also serves as a TODO list for the project.
Please feel free to work on any of these topics or recommend changes to the roadmap.

## Components
A high level goal for components is to create reasonable models of active and passive components for a standard 220nm SOI platform.
Below is a list of components that need models and goals for what should be modeled for specific components.
- [ ] strip + partially etched silicon + nitride waveguides
    - [ ] Wavelength dependence (Dispersion)
    - [ ] Waveguide width index dependence
    - [ ] Waveguide height index dependence
    - Support for components:
        - [ ] [straight](https://gdsfactory.github.io/gdsfactory/components.html#straight)
        - [ ] [straight_rib](https://gdsfactory.github.io/gdsfactory/components.html#straight-rib)
        - [ ] [spiral_double](https://gdsfactory.github.io/gdsfactory/components.html#spiral-double)
        - [ ] [spiral_external_io](https://gdsfactory.github.io/gdsfactory/components.html#spiral-external-io)
        - [ ] [spiral_internal_io](https://gdsfactory.github.io/gdsfactory/components.html#spiral-inner-io)
        - [ ] [spiral_racetrack](https://gdsfactory.github.io/gdsfactory/components.html#spiral-racetrack)

- [ ] Directional couplers, splitters, combiners, MMIs
    - [ ] Wavelength dependence (Dispersion)
    - [ ] Insertion loss and process variation
    - Support for components:
        - [ ] [1x2 MMI](https://github.com/gdsfactory/gdsfactory/blob/main/gdsfactory/components/mmi1x2_with_sbend.py)
        - [ ] [2x2 MMI](https://github.com/gdsfactory/gdsfactory/blob/main/gdsfactory/components/mmi2x2_with_sbend.py)
        - [ ] [directional coupler](https://gdsfactory.github.io/gdsfactory/components.html#coupler)
        - [ ] [splitter_chain](https://gdsfactory.github.io/gdsfactory/components.html#splitter-chain)
        - [ ] [splitter_tree](https://gdsfactory.github.io/gdsfactory/components.html#splitter-tree)

- [ ] Tapers, escallators and terminations
    - Support for components:
        - [ ] [straight_rib_tapered](https://gdsfactory.github.io/gdsfactory/components.html#straight-rib-tapered)
        - [ ] [taper_cross_section_linear](https://gdsfactory.github.io/gdsfactory/components.html#taper-cross-section-linear)
        - [ ] [taper_cross_section_parabolic](https://gdsfactory.github.io/gdsfactory/components.html#taper-cross-section-parabolic)
        - [ ] [taper_cross_section_sine](https://gdsfactory.github.io/gdsfactory/components.html#taper-cross-section-sine)
        - [ ] [terminator](https://gdsfactory.github.io/gdsfactory/components.html#terminator)

- [ ] Grating couplers
    - [ ] Wavelength dependent coupling
    - [ ] Fabry-perot effects
    - [ ] Misalignment in packaging
    - This would be a baseline model, there are so many types of grating couplers in gdsfactory I expect most folks will use their own model

- [ ] Edge couplers
    - [ ] Wavelength dependent coupling 
    - [ ] Fabry-perot effects
    - [ ] Misalignment in packaging
    - Edge couplers are made from a taper_cross_section, so thought must go into how to model edge couplers here

- [ ] Circular and euler bends
    - [ ] Model radius dependent loss from leakage
    - [ ] For circular bends, model mode overlap loss
    - Support bend_circular, bend_circular180, bend_circular_heater, bend_euler, bend_euler180, bend_euler_s, bend_straight_bend and bezier
    - [Related paper](https://arxiv.org/abs/2301.01689)
    - [Related paper](https://ieeexplore.ieee.org/abstract/document/8328825)

- [ ] Silicon crossing
    - [ ] Insertion loss
    - [ ] Crosstalk between ports
    - [ ] Fabry-perot
    - Support for components:
        - [ ] [crossing](https://gdsfactory.github.io/gdsfactory/components.html#crossing)
        - [ ] [crossing45](https://gdsfactory.github.io/gdsfactory/components.html#crossing45)
        - [ ] [crossing_etched](https://gdsfactory.github.io/gdsfactory/components.html#crossing-etched)

- [ ] Germanium photodetectors
    - [ ] Responsivity/quantum efficiency
    - [ ] Dark current
    - [ ] RC time constant
    - [ ] Carrier transit time
    - [ ] Fabry-perot reflection
    - Support for components:
        - [ge_detector_straight_si_contacts](https://gdsfactory.github.io/gdsfactory/components.html#ge-detector-straight-si-contacts) 

- [ ] Thermal phase shifter
    - [x] With and without undercut
    - [ ] Thermal "RC" time constant
    - [x] TiN models and doped silicon models
    - Support for components:
        - [ ] [spiral_racetrack_heater_doped](https://gdsfactory.github.io/gdsfactory/components.html#spiral-racetrack-heater-doped)
        - [ ] [straight_heater_doped_rib](https://gdsfactory.github.io/gdsfactory/components.html#straight-heater-doped-rib)
        - [ ] [straight_heater_doped_strip](https://gdsfactory.github.io/gdsfactory/components.html#straight-heater-doped-strip)
        - [ ] [straight_heater_meander](https://gdsfactory.github.io/gdsfactory/components.html#straight-heater-meander)
        - [ ] [straight_heater_meander_doped](https://gdsfactory.github.io/gdsfactory/components.html#straight-heater-meander-doped)
        - [X] [straight_heater_metal](https://gdsfactory.github.io/gdsfactory/components.html#straight-heater-metal)
        - [X] [straight_heater_metal_undercut](https://gdsfactory.github.io/gdsfactory/components.html#straight-heater-metal-undercut)

- [ ] Carrier phase shifter
    - [x] PIN and PN phase shifters
    - [X] Lateral junction
    - [X] Vertical junction
    - [ ] Interleaved junction
    - [X] U-shaped junction
    - [ ] MOSCAP junction
    - [ ] Carrier injection "RC" time 
    - [ ] Carrier depletion "RC" time
    - [ ] A lookup table from voltage to current and capacitance to calculate static and dynamic power consumption
    - Support for components:
        - [X] [straight_pin](https://gdsfactory.github.io/gdsfactory/components.html#straight-pin)
        - [X] [straight_pn](https://gdsfactory.github.io/gdsfactory/components.html#straight-pn)
        - Note, straight_pn in gdsfactory is a subset of straight_pin, and is referred to by the name straight_pin. Can we fix this?

- [ ] Ring resonators
    - [ ] Process variation
    - [ ] Heaters
    - [ ] PN junctions
    - [ ] Single pass rings, add-drop rings, crow filters/coupler ring resonators
    - Support for components:
        - [ ] [ring_crow](https://gdsfactory.github.io/gdsfactory/components.html#ring-crow)
        - [ ] [ring_single](https://gdsfactory.github.io/gdsfactory/components.html#ring-single)
        - [ ] [ring_double](https://gdsfactory.github.io/gdsfactory/components.html#ring-double)
        - [ ] [ring_single_heater](https://gdsfactory.github.io/gdsfactory/components.html#ring-single-heater)
        - [ ] [ring_double_heater](https://gdsfactory.github.io/gdsfactory/components.html#ring-double-heater)
        - [ ] [ring_single_pn](https://gdsfactory.github.io/gdsfactory/components.html#ring-single-pn)
        - [ ] [ring_double_pn](https://gdsfactory.github.io/gdsfactory/components.html#ring-double-pn)

## Systems
To aid in folk's ability to simulate systems, photonflux will include many working examples of systems 'out of the box' for users.
Below is a list of systems that we intend to have implemented and specific nuances of what is to be modeled in these systems.
- [ ] Ring based WDM transceiver with the following effects modeled:
    - [ ] Specrtal crosstalk between cavities
    - [ ] photon lifetime
    - [ ] RC time constant
    - [ ] shot noise in detection
    - [ ] thermal noise from readout electronics
    - [ ] ring self-heating/two photon absorption effects

- [ ] MZI based incoherent transceiver with the following effects modeled: 
    - [ ] shot noise in detection
    - [ ] thermal noise from readout electronics
    - [ ] RC time constant
    - [ ] travelling wave effects
    - [ ] PAM2 (NRZ) data transmission example
    - [ ] PAM4 data transmission example

- [ ] MZI based coherent transceiver with the following effects modeled: 
    - [ ] shot noise in detection
    - [ ] thermal noise from readout electronics
    - [ ] chirp from free-carrier dispersion
    - [ ] finite laser linewidth
    - [ ] RIN noise
    - [ ] QAM constellation diagram

- [ ] MZI based wavelength filter (lattice filter)
    - [ ] Fabrication error/process variation
    - [ ] Derivation/tutorial on how to design a lattice filter
    - [A relevent paper on design](https://www.spiedigitallibrary.org/journals/optical-engineering/volume-57/issue-12/127103/Ultrabroadband-lattice-filters-for-integrated-photonic-spectroscopy-and-sensing/10.1117/1.OE.57.12.127103.full?SSO=1)
    - [A relevent youtube video](https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3D_N5aHgYAw5Y&psig=AOvVaw224DFLS85Decg32Tx58dwY&ust=1676585226197000&source=images&cd=vfe&ved=2ahUKEwiA8IHoxJj9AhWZJGIAHdBsAvsQr4kDegUIARC7AQ)

- [ ] MZI switch matrix
    - [ ] Fabrication error/process variation in components and how that effects permutation matrix fidelity
    - [ ] Thermal crosstalk between components

- [ ] Programmable photonics using hexagonal meshes
    - [Paper from Jose Capmany's group](https://www.nature.com/articles/s41467-017-00714-1)
    - [ ] Thermal crosstalk between components
    - [ ] Fabry-Perot effects in these meshes

- [ ] Mesh of interferometers similar to the attached paper with the following effects modeled:
    - [Paper from Dirk Englund's group](https://www.nature.com/articles/nphoton.2017.95)
    - [ ] Thermal crosstalk between phase shifters
    - [ ] Calculation of matrix/transform fidelity
    - [ ] Robustness to component variation

- [ ] Tunable lasers using the Vernier effect similar to the below links with the below effects modeled:
    - [JLT paper from Bowers group](https://opg.optica.org/jlt/abstract.cfm?uri=jlt-40-6-1802&ibsearch=false)
    - [Paper from Michael Hochberg](https://opg.optica.org/oe/fulltext.cfm?uri=oe-26-7-7920&id=383841&ibsearch=false)
    - [ ] Cavity fabrication error and tuning procedure
    - [ ] Gain medium noise spectrum and resulting side-mode supression ratio

## Solver
Our above modeling is powered by having an accurate, easy to use and reasonably fast solver.
Below is a list of improvements we intend to make to the solver
- [ ] Simulation of Fabry-Perot reflections in larger systems
- [ ] Simulation of polarization diversity (TE + TM)
- [ ] Speed up simulation time to solve systems with 1,000 components in ~1ms (1,000 graph solves per second)
- [ ] Interoperability to gdsfactory flow
    - [ ] Pull in automatically generated S-parameter data from lumerical FDTD from gdsfactory
    - [ ] Pull in automatically generated S-parameter data from meep from gdsfactory
    - [ ] Pull in automatically generated mode index data from gdsfactory related to [this pull request](https://github.com/gdsfactory/gdsfactory/pull/1055)

## Miscellaneous 
Below that items that don't fall into the other categories or are things that require discussion before becoming an action item
- [ ] Create a logo for the project
- [ ] Create a documentation page, similar to what gdsfactory uses
- [ ] Have a CI/CD pipeline
- [ ] Create unit tests for the graph solver
- [ ] Document the specifications of the default fabrication process
    - [ ] Silicon nitride thickness and gap
    - [ ] Doping concentrations
    - [ ] Germanium dark current
- [ ] Comparisons and benchmarks to lumerical interconnect
- [ ] Should we add multi-mode simulation support? It is only useful in a select few applications and adds complications while slowing down the solver
- [ ] Are there ways to efficiently enable multi-wavelength simulation?
- [ ] Are there useful tools to add to analyze the "yield" of a system under some amount of process variation?
