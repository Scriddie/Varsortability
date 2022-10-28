# Beware of the Simulated DAG! Causal Discovery Benchmarks May Be Easy To Game

This repository contains stand-alone implementations of **varsortability**, **sortnregress**, and **chain-orientation** as presented in 

[1] Reisach, A. G., Seiler, C., & Weichwald, S. (2021). [Beware of the Simulated DAG! Causal Discovery Benchmarks May Be Easy To Game](https://proceedings.neurips.cc/paper/2021/file/e987eff4a7c7b7e580d659feb6f60c1a-Paper.pdf).

For a **basic experimental set-up for the comparison of causal structure learning algorithms** as shown in the same work, see the [VarsortabilityExperimentSuite](https://github.com/Scriddie/VarsortabilityExperimentSuite) repository.

If you find this code useful, please consider citing:
```
@article{reisach2021beware,
  title={Beware of the Simulated DAG! Causal Discovery Benchmarks May Be Easy to Game},
  author={Reisach, Alexander G. and Seiler, Christof and Weichwald, Sebastian},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```

To run, perform the following actions **within the /src directory**:

1. Install dependencies by running `./install.sh` in this directory.
2. For **varsortability** run `source env/bin/activate; python varsortability.py` in the current directory.
2. For **sortnregress**, run `source env/bin/activate; python sortnregress.py` in the current directory.
2. For **chain-orientation**
    - run `source env/bin/activate; python chain_orientation.py` in the current directory (may take some time).
    - run `source env/bin/activate; python chain_orientation_three_vars_symbolic.py` in the current directory (may take some time).
