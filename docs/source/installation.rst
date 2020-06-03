============
Installation
============

At the command line::

    $ pip install emgfit



0. Setup Python (version>=3.6), e.g. get Anaconda.

Install via pip:
1. Run
$ pip install emgfit

Anaconda users:
1. Install and update conda and conda-build.


Build & install package & dependencies:

conda build:
create recipe
conda build . # creates tar.gz in Anaconda/... directory

conda install:
conda install --use-local emgfit

Then set up ipykernel for conda environment.

pip build:
pip install --user --upgrade setuptools wheel
python setup.py sdist bdist_wheel 	# creates .whl and tar.gz in /dist directory

pip install:
pip install .whl
pip install -r requirements.txt

Optionally creating a virtual environment for emgfit:
pip install virtualenv
python -m venv emgfit
.\emgfit-env\Scripts\activate	(linux: source emgfit/bin/activate)

then run steps above inside venv
