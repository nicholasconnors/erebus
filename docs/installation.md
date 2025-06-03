# Erebus installation

It is recommended to use Erebus in a dedicated Anaconda environment. For instructions on how to install Anaconda, read their [docs](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html). For the quickest setup use [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions).

Next, create a new Anaconda environment on Python 3.12 by running the following in the terminal (feel free to rename the environment).

```
conda create --name erebus_env
conda activate erebus_env
conda install python=3.12
```

### Installation using pip

Todo, once Erebus has been added to PyPI.

### Manual installation

Download the Erebus installation file on the GitHub repo from the [latest release](https://github.com/nicholasconnors/erebus/releases) or from a development [workflow artifact](https://github.com/nicholasconnors/erebus/actions/workflows/publish_dev_artifact.yaml).

Unzip `erebus.zip` (it will contain the archive file `erebus-x.y.z.tar.gz`).

From the terminal, vavigate to the folder containing the archive file and run the following from your Anaconda environment (substituting in the actual file name):

```
pip install erebus-x.y.z.tar.gz
```

### Installing from cloned repo

If you are installing Erebus for local development, see the [contributing](contributing.md) page for how to build and install Erebus from its source code.

### Running Jupyter Notebook demos

Demo notebooks can be found at https://github.com/nicholasconnors/erebus/docs/demos

To set up Jupyter to run in your Anaconda environment, run:

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
