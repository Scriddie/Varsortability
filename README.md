# Beware of the Simulated DAG! Causal Discovery Benchmarks May Be Easy To Game

This repository contains stand-alone implementations of **varsortability**, **sortnregress**, and **chain-orientation** as presented in 

[1] Reisach, A. G., Seiler, C., & Weichwald, S. (2021). [Beware of the Simulated DAG! Causal Discovery Benchmarks May Be Easy To Game](https://arxiv.org/abs/2102.13647)

To run, perform the following actions **within the /src directory**:

1. Install dependencies by running `./install.sh` in this directory.
2. For **varsortability** run `source env/bin/activate; python varsortability.py` in the current directory.
2. For **sortnregress**, run `source env/bin/activate; python sortnregress.py` in the current directory.
2. For **chain-orientation**
    - run `source env/bin/activate; python chain_orientation.py` in the current directory (may take some time).
    - run `source env/bin/activate; python chain_orientation_three_vars_symbolic.py` in the current directory (may take some time).