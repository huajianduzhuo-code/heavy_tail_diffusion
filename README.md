## statement
This repo is the code for "Learning to Simulate from Heavy-tailed Distribution via Diffusion Model". The abstract is included as follows:

Diffusion models, as a class of neural-network based generative models, despite being one of the most prominent tools to learn to simulate from multi-dimensional distributions, typically assume that the data distributions have finite support. However, applications in the fields of operations research and management science often witness distributions with infinite support or even heavy tails. In this work, we theoretically show that existing diffusion models encounter challenges in addressing the tail distribution in both model training and data generation. To address the challenges, we develop a new method extending existing diffusion models to effectively capture the heavy-tailed distribution patterns. Our method accommodates the learning and simulation of both multi-dimensional distributions with potential heavy tails, and conditional distributions with multi-dimensional conditions.

## Requirement

Please install the packages in requirements.txt


## Experiments 

### training and generation
This is an example of training and generation using the Pareto dataset. Replace the name of the .py file to run other experiments.

run the following code to train your own model:

```shell
python exe_pareto.py
```

or run the following line to generate using existing models:

```shell
python exe_pareto.py --modelfolder "pareto_t"
```

### Visualize results
'visualization.ipynb' is a notebook for visualizing results.