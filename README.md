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

For more information on the installation of the C++ code see `LightningF/Utils/QT/README.txt`.

## Requirements

The main packages required before running the software are:

`scipy, numpy, pandas, seaborn, pystan, matplotlib, scikit-learn, cython, disutils, multiprocessing`

For anaconda users, the files `environment.yml` and `requirements.txt` in the root directory can be used to recreate the environment.  

## Usage

To use Lightning, import in your code the corresponding class:

```
import sys
sys.path.append('/path/to/root/of/the/Lightning/repository/')
from LightningF.Models.pdp_simple import TimeIndepentModelPython as ModelSpacePy
```

The input data should be created as a 4-column / 6-column numpy array. The first 2 / 3 columns
correspond to x,y / x,y,z observation positions. The next 2 / 3 columns correspond to the position uncertainties s_x, s_y / s_x, s_y, s_z. 

The algorithm is then run by including:

```
modelpy = ModelSpacePy(data=data, init_type='rl_cluster', infer_pi1=True, infer_alpha0=True)
modelpy.fit(iterations=100, pl=0, prt=1)
```

And the results can be visualized with a plot generated with:

```
modelpy.pl_bl() 
```



## Example
An example showing how to run Lightning appears in the file `comparison.py`, located in the directory `MCMC_compare`. To run it, execute:

```
cd MCMC_compare
python comparison.py
```

The code loads one of the datasets used in the paper, compares Lightning with MCMC, prints several metrics and creates a pdf file in the same directory entitled `inference_50nm.pdf`, corresponding to Supplementary Figure 2 from the paper.


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


