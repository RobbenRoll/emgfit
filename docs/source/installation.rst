============
Installation
============

Find below instructions for either 'quick and dirty' way or a neater way of
installing emgfit. The second approach is intended for more advanced users
that want to keep their Python setup cleaner by using virtual environments.

The quick and dirty way
-----------------------

0. Download the `emgfit` repository from GitHub. Ensure that Python 3 is
   installed on your system and has been added to your `PATH`.
1. Use your command line to navigate into the emgfit parent directory that
   you've just downloaded and install emgfit by running::

     $ python setup.py install

  The package and all its dependencies should now be installed.

2. To test whether the installation succeeded open Python from your command
   line and import the package::

    $ python
    $ import emgfit

  If the second line does not raise any errors you should be good to go with
  emgfit!

The neater way: Installing emgfit into a virtual environment
------------------------------------------------------------

Find instructions for both conda and pip users below.

For pip users:
^^^^^^^^^^^^^^

0. Download the `emgfit` repository from GitHub. Ensure that Python 3 is
   installed on your system and has been added to your `PATH`.
1. Set up a new virtual environment for `emgfit` to live in. To do so, use your
   command line to navigate to the folder in which you would like to set up the
   virtual environment and run the following::

    $ pip install virtualenv
    $ python -m venv emgfit-env

2. Activate the new `emgfit-env` environment. The command for this depends on
   your platform.

   Windows users run::

    $ emgfit-env\Scripts\activate.bat

   Linux users run::

    $ source emgfit-env/bin/activate

3. Now you are inside your virtual environment and can install `emgfit`.
   Therefore, navigate into the emgfit parent directory that you have downloaded
   in step 0 and install emgfit by running::

     (emgfit-env) $ python setup.py install

  The package and all its dependencies should now be installed inside your
  virtual environment `emgfit-env`.

4. To test whether the installation succeeded open Python from your command
   line (after activating `emgfit-env`!) and import the package::

     (emgfit-env) $ python
     (emgfit-env) $ import emgfit

   If the second line does not raise any errors you should be good to go with
   emgfit!

For conda users:
^^^^^^^^^^^^^^^^

0. Download the `emgfit` repository from GitHub. Ensure that the Anaconda Python
   distribution is installed on your system and has been added to your `PATH`.
1. Set up a new virtual environment for `emgfit` to live in by running the
   following in your command line::

    $ conda create -n emgfit-env python=3.7

2. Activate the conda environment::

    (emgfit-env) $ conda activate emgfit-env

3. Now you are inside your virtual environment and can install `emgfit`.
   Therefore, navigate into the emgfit parent directory that you have downloaded
   in step 0 and install emgfit by running::

     (emgfit-env) $ python setup.py install

  The package and all its dependencies should now be installed inside your
  virtual environment `emgfit-env`.

4. To test whether the installation succeeded open Python from your command
   line and import the package::

     (emgfit-env) $ python
     (emgfit-env) $ import emgfit

   If the second line does not raise any errors you should be good to go with
   emgfit!

Launching Jupyter notebook
--------------------------

To start working with emgfit, start up the Jupyter notebook server using the
following command::

    $ jupyter notebook

**Users that installed emgfit into a virtual environment must first activate
their emgfit environment (see step 2 above) before running this command.**

This will make a window pop up in your default browser. In there, you can
navigate to different directories and create new notebooks (using the `new`
panel on the top right) or open existing ones. Make sure that you use the
default kernel (usually called `Python3`). Once you have imported the `emgfit`
package (`import emgfit`) you should be ready to analyze some data.
