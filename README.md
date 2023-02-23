# Photonflux 0.2.1

[![PyPI](https://img.shields.io/pypi/v/photonflux)](https://pypi.org/project/photonflux/)
[![PyPI Python](https://img.shields.io/pypi/pyversions/photonflux.svg)](https://pypi.python.org/pypi/photonflux)
[![issues](https://img.shields.io/github/issues/alexsludds/photonflux)](https://github.com/alexsludds/photonflux/issues)
[![forks](https://img.shields.io/github/forks/alexsludds/photonflux.svg)](https://github.com/alexsludds/photonflux/network/members)
[![GitHub stars](https://img.shields.io/github/stars/alexsludds/photonflux.svg)](https://github.com/alexsludds/photonflux/stargazers)
[![Downloads](https://pepy.tech/badge/photonflux)](https://pepy.tech/project/photonflux)
[![Downloads](https://pepy.tech/badge/photonflux/month)](https://pepy.tech/project/photonflux)
[![Downloads](https://pepy.tech/badge/photonflux/week)](https://pepy.tech/project/photonflux)
[![MIT](https://img.shields.io/github/license/alexsludds/photonflux)](https://choosealicense.com/licenses/mit/)

<!-- ![logo](https://i.imgur.com/v4wpHpg.png) -->

Photonflux is a photonic circuit simulation tool for simple, extensible simulation of large active photonic systems.

It solves the problem that existing simulation tools are unable to capture many important phenomena that are key to creating working products including:
- Complex active component models
- Component variation and Monte-Carlo simulation
- Thermal crosstalk between components
- Fabry-Perot reflections from on-chip components

Here, we build a circuit simulator on NetworkX, a complex network solver, to enable arbitrarily reconfigurable circuit models and systems.
As opposed to other approaches, circuits / schematics are defined using a netlist generated from a **layout**. The primary reason for this is that photon designers typically do not use a idea -> schematic -> layout -> fabrication flow, but often skip the schematic step entirely (idea -> layout -> fabrication). The reason for this is that layouts in photonics are expressive and contain far more detail about the operation of systems than a schematic would capture. 

Layouts for simulation in photonflux are defined in a yaml markup format that can be automatically generated using ![gdsfactory](https://gdsfactory.github.io/gdsfactory/).

To see examples of what is possible in photonflux please see [our examples](https://github.com/alexsludds/photonflux/tree/main/examples).

For more information on the future of the package see [our roadmap](https://github.com/alexsludds/photonflux/blob/main/docs/ROADMAP).


## Installation
### Installation for new photonflux users
It's as simple as:
```
python -m pip install photonflux --upgrade
```

### Installation for developers
First install the package from github using:
```
git clone https://github.com/alexsludds/photonflux.git
```
Then, navigate to the download directory and install as an editable python package
```
python -m pip install -e .
```
## Getting started

We include several examples in the examples folder, but if your usecase is not there please create an github issue and we can work together on it.

## Acks

Contributors (in chronological order):

- Alex Sludds (MIT): Initial code and documentation.
