This repository contains the code for our work [Dissecting Supervised Contrastive Learning](https://proceedings.mlr.press/v139/graf21a.html) which was accepted at ICML'21 (see also the [updated arxiv version](https://arxiv.org/abs/2102.08817)).

# Installation
This code requires a [**pytorch**](https://pytorch.org/) installation (tested on version 1.13.1).

Additionally, the following packages are required: **scikit-learn**, **fastprogress**, **pandas**,  
```
    pip install scikit-learn fastprogress pandas
```

# Application
To reproduce the experiments from Section 5.2 (Theory vs. Practivce), run 
```
    python run_exp_performance.py 
```

To reproduce the random label experiments from Section 5.3, run
```
    python run_exp_noisy_labels.py 
```

Experiments can be evaluated with the notebooks **results_performance.ipynb**, or **results_noisy_labels.ipynb**, respectively.

Notebooks to reproduce Figures 1, 5 and 6 can be found in the **notebooks/** directory.  
These notebooks further include animations to visualize the convergence towards the simplex configuration when optimizing the representations directly.

# Reference
```
@inproceedings{Graf21a,
  author          = {Graf, Florian and Hofer, Christoph and Niethammer, Marc and Kwitt, Roland},
  title           = {Dissecting Supervised Contrastive Learning},
  booktitle       = {ICML},
  year            = {2021}
}
```  
