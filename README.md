## Lightning
Accompanying code for the paper:
- Gabitto, Marie-Nelly, Pakman, Pataki, Darzak, Jordan [A  Bayesian  nonparametric approach to super-resolution single-molecule localization](https://www.biorxiv.org/content/10.1101/2020.02.15.950873v3)



## Installation
Lightning is implemented in Python 3. To install it along with the C++ code used for Quadtree acceleration, run the following:

```bash
git clone https://github.com/marianogabitto/Lightning.git
cd LightningF/Utils/QT/python/
python setup.py build_ext --inplace
```

For more information on the installation of the C++ core see `LightningF/Utils/QT/README.txt`.

## Requirements

The main packages required before running the software are:

`scipy, numpy, pandas, seaborn, pystan, matplotlib, scikit-learn, cython, disutils, multiprocessing`

For anaconda users, the files `environment.yml` and `requirements.txt` in the root directory can be used to recreate the environment.  

## Usage
After cloning the repository, we are going to use Lightning by importing the classes into your project.
First, create a folder in the repository. Then, add the following lines to import the corresponding classes:

```
import sys
sys.path.append('../')
from LightningF.Models.pdp_simple import TimeIndepentModelPython as ModelSpacePy
```

Then, within your code, you will run the code by including:

```
modelpy = ModelSpacePy(data=data, init_type='rl_cluster', infer_pi1=True, infer_alpha0=True)
modelpy.fit(iterations=100, pl=0, prt=1)
```

To see the results do:

```
modelpy.pl_bl() 
```

Your data should be formatted as a 4-column / 6-column numpy array. the first 2 / 3 columns
correspond to x,y / x,y,z observation position. The next 2 / 3 coordinates correspond to the
position uncertainty s_x, s_y / s_x, s_y, s_z. 

## Examples
The file `comparison.py`, located in the directory `MCMC_compare`, generates Supplementary Figure 2 from the paper.
It shows an example of how to import some of the classes needed to run Lightning.
To run it, change into the directory and execute: 

```
cd LightningF
python comparison.py
```

The file should produce a pdf file in the same directory entitled inference_50nm.pdf . In addition, it prints in the console different results. 


## Citation

If you use this project, please cite the relevant publication as:

```
@article{gabitto2020bayesian,
  title={A Bayesian nonparametric approach to super-resolution single-molecule localization},
  author={Gabitto, Mariano Ignacio and Marie-Nelly, Herve and Pakman, Ari and Pataki, Andras and Darzacq, Xavier and Jordan, Michael},
  journal={bioRxiv},
  year={2020},
  publisher={Cold Spring Harbor Laboratory}
}
```


