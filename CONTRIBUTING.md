# Contributing

Erebus is made with Python 3.12. To develop, create a fresh conda environment in this version and then install the dependencies in requirements.txt.

I am not currently seeking contritions but have written this for posterity and the sake of internal documentation. However, please report any bugs encounted via the issues tab on the GitHub repository, and feel free to suggest features which could be used to improve the user experience.

### Building Erebus locally

Releases are handled via GitHub workflow, however you may still want to build a local version for testing. To do this, run `./install.sh` in the repository root to build and install Erebus in your current Conda environment.

### Writing new documentation

Documentation for Erebus is made using MkDocs and hosted using GitHub pages. A separate version of the docs is made per version. The main branch is published as `latest` and the dev branch is published as `dev`, and previous versions of the software are to be kept in their own branches labeled with their version number.

The docs have their own separate requirements, but the same Python version as the tool itself.

```
pip install mkdocs-material
pip install pillow cairosvg mike
pip install mkdocstrings-python
pip install markdown-include
pip install mkdocs-jupyter
pip install griffe-pydantic
```

When these are installed, open a terminal from the root of the repo and run `mkdocs serve` to host the site locally. The local adress will be displayed to you.

Any new functionality added to Erebus must be documented.

### Testing

Before making a pull request, run all manual tests locally on your branch and ensure they all work as expected and without logging errors. Similarly any maintainer should run all manual tests locally before merging a pull request after providing a thorough code review.

### Semantic versioning

Semantic versioning or [semver](https://semver.org/) is a simple set of rules for interpreting version numbers. Whenever making an update to Erebus we should strive to ensure there is backwards compatibility, and only when that is no longer tenable should we make a major release. Any deprecated functionality must remain in Erebus and not be removed unless part of a major version update (e.g., 1.x.x -> 2.0.0).

### Releasing

New releases are automatically drafted when merging into `main` with a pull-request labeled `release`. The version tag is taken from the title of the pull-request, and the release body is taken from the body of the pull-request.