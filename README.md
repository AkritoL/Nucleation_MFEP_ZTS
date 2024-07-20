# Nucleation MFEP - ZTS

## Introduction
Nucleation is a common process via which phase separation occurs in multicomponent mixtures. It is thermally activated and needs to overcome a finite energy barrier, therefore it is difficult for numerical tools such as phase-field method alone to capture such events. To overcome this difficulty, in this work, we use the string method, a numerical scheme for finding transition pathways and critical states, to study the nucleation of ternary mixtures. Two mechanisms are identified, similar to the homogeneous and heterogeneous nucleation of binary mixture.

We calculate the transition pathways of the two mechanisms and find that their nucleation barriers are related by a shape factor, which is a function of surface tensions.Our results suggest that both the nucleation barriers and the nucleation pathways, can be tuned by altering surface tensions. 

This repository contains the code of our work. __Please contact email address akritolee@gmail.com or maosheng@pku.edu.cn for relevant data.__

## Content
This repository contains the following files:

- __BinaryMix__
    Benchmark: A binary component system is used to test whether the ZTS method can accurately simulate the minimum free energy path of nucleation.
    - __1d__ 
        One-dimensional cases are tested using finite difference method
    - __2d__
        Finite difference method and spectrum method (fft) are used to test the case of two-dimensional radial symmetry, cartesian coordinates and polar coordinates respectively.
- __Paraview__
    Contains a python scripts for data type conversion, in order to be read by Paraview
- __TernaryMix__
    - __1d__ 
        It is only used to test the model correctness of our ternary system and to uncover possible problems
    - __2d__
        - __contact_angle__
            To explore the accuracy of our model for the simulation of the contact Angle formed by the ternary system    
        - plot
        - main.py
    - 3d
        - __single_case__ 
            The string method is not used, but a single image is used to iterate to see if it can converge to nucleation
    - simulate.sh
        A console that simulates the triplet splitting process

## Requirement

| Mandatory    | Recommend |
| ------------ | --------- |
| python       | 3.11      |
| line_profiler| 4.1.3     |
| matplotlib   | 3.9.1     |
| jax          | 0.3.18    |
| jaxlib       | 0.3.18+cuda11.cudnn82 |

| Optional     | Minimum | Recommend |
| ------------ | ------- | --------- |
| CUDA         | 11.6    | 12.2      |

## Getting Started

### Installation

Our work is mainly based on JAX, using JAX requires installing two packages: jax, which is pure Python and cross-platform, and jaxlib which contains compiled binaries, and requires different builds for different operating systems and accelerators.

JAX can be installed for CPU on Linux, Windows, and macOS directly from [the Python Package Index:](https://pypi.org/project/jax/)
```bash
pip install jax
```
or, for NVIDIA GPU:
```bash
pip install -U "jax[cuda12]"
```
For more detailed platform-specific installation information, check out [Installing JAX](https://jax.readthedocs.io/en/latest/installation.html#installation).

For specific older GPU wheels, be sure to use the jax_cuda_releases.html URL; for example
```bash
pip install jax jaxlib==0.3.25+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Then, just clone the repository and install locally (in editable mode so changes in the code are immediately reflected without having to reinstall):
```bash
#git clone link
cd Nucleation_MFEP_ZTS
pip install --upgrade line_profiler matplotlib
```

### Usage: 
Becase binary mix situation is just a benchmark, we directly enter TernaryMix
```bash
cd TernaryMix
```

To run the simulations, simply execute the simulate.sh script.
```bash
sh simulate.sh #or bash simulate.sh
```

> __Hardware Requirement :__ 
We cannot provide a specific time or minimum graphics card memory usage. Here we only provide a reference to the hardware usage when we simulate a single example:

__With a A100 80G__, the number in the table below are all approximate
| Dimension | Iteration steps | Approximate time | Graphics memory usage | 
| --------- | --------------- | ---------------- | --------------------- |
| 2d        | 500000          | 10000s           | 70000M                |
| 3d        | 200000          | 100000s          | 70000M                |


The script requires users to specify parameters, as follows:

```bash
DIMENSION=2d 
# Select the dimensions of the simulation, only permit '2d' or '3d'

CASE=AsymmetricCase 
# Select the symmetry of the simulation, only permit 'AsymmetricCase' or 'SymmetricCase'

NUCLEATION_MODE=heterogeneous 
# Select the nucleation mode, only permit 'homogeneous' or 'heterogeneous' 

INTERFACE_WIDTH=0.012 
# Based on phase-field model, the interface has a preset width, the smaller the better
# We found that for the two-dimensional case, at least 0.012 and 0.024 at least for 3D cases
# In order for the gradient term to work 
# make sure that at least three points are on the interface 

GAMMA_AB=1e-2, GAMMA_BC=1e-2, GAMMA_CA=1.3e-2
# Interface tension
# for symmetric case, we fasten GAMMA_BC and GAMMA_CA as 1e-2, range GAMMA_AB from 2e-3 to 1.8e-2
# for asymmetric case, we fasten GAMMA_AB and GAMMA_BC as 1e-2, range GAMMA_CA from 2e-3 to 1.8e-2

TIME_STEP=5e-6
# Better chose 1e-6 ~ 1e-5, and TIME_STEP_INCREMENT is used to make iterations converge faster

--FINAL_RADIUS 0.1 
# Pre-set final nucleation radius, for 2d case, 0.1 is ok; for 3d case, 0.2 is better.

--STEPMAX 500000 
# The total number of steps of the simulated iteration

--GRID_NUMBER 256
# The number of lattice points on a single dimension of the grid, we simply use a uniform grid
# Here, for 2d case 256; for 3d case we use 128 

--NUM_STRING_IMAGES 50
# Based on Zero-temperature String Method, the more images on the string, the better the simulation of the energy pathway
# Here we all use 50 images
```


