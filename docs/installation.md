# Erebus installation

### Using pip

Todo

### Manually

Todo: add conda installation instructions.

Create a new conda environment on Python 3.12 by running the following in the terminal (feel free to rename the environment).

```
conda create --name erebus_env
conda activate erebus_env
conda install python=3.12
```

Download the Erebus installation file on the GitHub repo from the latest release or from an action artifact (Actions tab, build dev package artifact, tar.gz file at the bottom).

Navigate to the folder containing the archive file and run (substituting in the actual file name)

```
pip install erebus_nicholasconnors-x.y.z.tar.gz
```

### Running Jupyter Notebook demos

Demo notebooks can be found at https://github.com/nicholasconnors/erebus/docs/demos

To set up Jupyter to run in your conda environment, run:

```
conda install jupyter
jupyter notebook
```

To test that Erebus has installed correctly simply run a notebook with

```
import erebus
```
