# Contributing

Erebus is made with Python 3.12. To develop, create a fresh conda environment in this version and then install the dependencies in requirements.txt.

I am not currently seeking contritions but have written this for posterity and the sake of internal documentation. However, please report any bugs encounted via the issues tab on the GitHub repository, and feel free to suggest features which could be used to improve the user experience.

### Writing new documentation

Documentation for Erebus is made using MkDocs and hosted using GitHub pages. A separate version of the docs is made per version. The main branch is published as `latest` and the dev branch is published as `dev`, and previous versions of the software are to be kept in their own branches labeled with their version number.

The docs have their own separate requirements, but the same Python version as the tool itself.

```
  pip install mkdocs-material
  pip install pillow cairosvg mike
  pip install mkdocstrings-python
  pip install markdown-include
  pip install mkdocs-jupyter
```

When these are installed, open a terminal from the root of the repo and run `mkdocs serve` to host the site locally. The local adress will be displayed to you.

Any new functionality added to Erebus must be documented.

### Testing

Before making a pull request, run all manual tests locally on your branch and ensure they all work as expected and without logging errors.
