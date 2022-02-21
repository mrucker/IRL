The Python code directory contains a number of IRL and RL algorithms and experiments.

Core, reusable functionality such as algorithms and visualizations are implemented in the combat directory.

One off experiments and visualizations which utilize the core code can be found in the scripts directory.

To make sure all required dependencies have been installed use conda and run `conda env create --f environment.yml` to create and install all package dependencies to an environment called `combat`. Once this is finished use `conda activate combat` to switch to this environment before running any desired files in ~/scripts/<domains>/*.py.