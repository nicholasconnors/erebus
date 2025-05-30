#!/bin/bash

echo "Building erebus locally"

if [ ! -d "build" ]; then
    mkdir "build"
fi

cd "build"

venv_dir="./venv"

if [ ! -d "$venv_dir" ]; then
    echo "Creating virtual environment"
    python3 -m venv "$venv_dir"
    echo "Done"
fi

echo "Activating virtual environment"
source "$venv_dir/bin/activate"
echo "Done"

echo "Installing depdencies"
pip install --upgrade pip
pip install setuptools
pip install -r ../requirements.txt
echo "Building"
python -m pip install build
python -m build --outdir ./dist ../

echo "Done"
deactivate