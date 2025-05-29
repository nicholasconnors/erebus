# Erebus installation

It is recommended to use erebus in a dedicated Anaconda environment.

Todo: add conda installation instructions.

Next, create a new conda environment on Python 3.12 by running the following in the terminal (feel free to rename the environment).

```
conda create --name erebus_env
conda activate erebus_env
conda install python=3.12
```

### Installation using pip

Todo

### Manual installation

Download the Erebus installation file on the GitHub repo from the latest release or from an action artifact (Actions tab, build dev package artifact, erebus file at the bottom).

If using the action artifact, unzip `erebus.zip` to get access to the `.tar.gz` file

Navigate to the folder containing the archive file and run the following from your conda environment (substituting in the actual file name):

```
pip install erebus-x.y.z.tar.gz
```

### Running Jupyter Notebook demos

Demo notebooks can be found at https://github.com/nicholasconnors/erebus/docs/demos

To set up Jupyter to run in your conda environment, run:

```
conda install jupyter
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=erebus_env
jupyter notebook
```

To test that Erebus has installed correctly simply run a notebook with

```
import erebus
```
