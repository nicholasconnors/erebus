#!/bin/bash

echo "Building and installing Erebus locally"

if [ ! -d "build" ]; then
    mkdir "build"
fi

cd "build"

echo "Installing depdencies"
pip install --upgrade pip
pip install setuptools
pip install -r ../requirements.txt
echo "Building"
python -m pip install build
python -m build --outdir ./dist ../

echo "Done"

pip install dist/*.tar.gz