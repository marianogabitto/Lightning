# Lightning
Accompanying Code for: A  Bayesian  nonparametric approach to super-resolution single-molecule localization. Gabitto et al. 

# Installation
The current version of Lightning is implemented in Python 3. Clone the github repository to have access to the classes needed to run it.

To install the C++ code that uses Quadtree acceleration, change into the directory:

cd LightningF/Utils/QT/python/

and then run:

python setup.py build_ext --inplace

the following file contains more information regarding the installation of the C++ core LightningF/Utils/QT/README.txt .

# Requirements

Here, we list the main requirements needed to be installed before running the software. 

scipy, numpy, pandas, seaborn, pystan, matplotlib, scikit-learn, cython, disutils, multiprocessing

In addition, we include in the root directory two files: environment.yml and requirements.txt. These files recreate the anaconda environment.  

# Usage
After cloning the repository, we are going to use Lightning by importing the classes into your project.

> from LightningF.Models.pdp_simple import TimeIndepentModelPython as ModelSpacePy

# Examples
The file comparison.py, located in the directory MCMC_compare, generates supplementary figure 2 from the paper.
It shows an example of how to import some of the classes needed to run Lightning.
To run it, change into the directory and execute: 
> python comparison.py

The file should produce a pdf file in the same directory entitled inference_50nm.pdf . In addition, it prints in the console different results. 
