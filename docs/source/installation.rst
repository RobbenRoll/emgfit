============
Installation
============

Find below instructions for either 'quick and dirty' way or a neater way of
installing emgfit. The second approach is intended for more advanced users
that want to keep their Python setup cleaner by using virtual environments.

The quick and dirty way
-----------------------

This will install emgfit system-wide for your global Python version. All emgfit
versions >= 0.2.0 are available as pip install from PyPI.

0. Ensure that Python3 and its package installer pip are available on your
   system and have been added to your `PATH` (pip comes pre-installed with most
   Python distributions).
1. Use your command line to download and install emgfit from PyPI by
   running::

     $ pip install --upgrade emgfit

  The latest version of the package and all its dependencies should now be
  installed under your Python directory in `/Lib/site-packages`. If you want to
  install a specific emgfit version `x.y.z` run ``$ pip install emgfit==x.y.z``
  instead.

2. To test whether the installation succeeded and if you got the correct version
   open Python from your command line and run::

    $ python
    $ import emgfit
    $ print(emgfit.__version__)

   If this doesn't raise any errors you're good to go with emgfit!


The neater way: Installing emgfit into a virtual environment
------------------------------------------------------------

Virtual environments can help you to keep emgfit isolated from other projects
and are particularly useful if you want to use multiple emgfit versions on the
same system. Find instructions for both venv and conda users below.

For venv users:
^^^^^^^^^^^^^^^

0. Ensure that Python3 and its package installer pip are available on your
   system and have been added to your `PATH` (pip comes pre-installed with most
   Python distributions).
1. Create a new virtual environment for `emgfit` to live in. To do so, use your
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
   Download and install emgfit by running::

     (emgfit-env) $ pip install --upgrade emgfit

  The latest version of the package and all its dependencies should now be
  installed inside your virtual environment `emgfit-env` under
  `/Lib/site-packages`. If you want to install a specific emgfit version `x.y.z`
  run ``$ pip install emgfit==x.y.z`` instead.

4. To test whether the installation succeeded and if you got the correct version
   open Python from your command line (after activating `emgfit-env`!) and run::

     (emgfit-env) $ python
     (emgfit-env) $ import emgfit
     (emgfit-env) $ print(emgfit.__version__)

5. Finally, register an IPython kernel for your emgfit environment::

    (emgfit-env) $ python -m ipykernel install --user --name emgfit-env

Now you're good to go with emgfit!

For conda users:
^^^^^^^^^^^^^^^^

0. Ensure that Python3 and its package installer pip are available on your
   system and have been added to your `PATH` (pip comes pre-installed with most
   Python distributions including Anaconda).
1. Set up a new virtual environment for `emgfit` to live in by running the
   following in your command line::

    $ conda create -n emgfit-env python=3.7

2. Activate the conda environment::

    (emgfit-env) $ conda activate emgfit-env

3. Now you are inside your virtual environment and can install `emgfit`.
   Download and install emgfit by running::

     (emgfit-env) $ pip install --upgrade emgfit

   The latest version of the package and all its dependencies should now be
   installed inside your virtual environment `emgfit-env` under
   `/Lib/site-packages`. If you want to install a specific emgfit version `x.y.z`
   run ``$ pip install emgfit==x.y.z`` instead.

4. To test whether the installation succeeded and if you got the correct version
   open Python from your command line and run::

     (emgfit-env) $ python
     (emgfit-env) $ import emgfit
     (emgfit-env) $ print(emgfit.__version__)

 5. Finally, register an IPython kernel for your emgfit environment::

     (emgfit-env) $ python -m ipykernel install --user --name emgfit-env

Now you're good to go with emgfit!

Launching Jupyter notebooks
---------------------------

To start working with emgfit, start up the Jupyter notebook server using the
following command::

    $ jupyter notebook

**Users that installed emgfit into a virtual environment must first activate
their emgfit environment (see step 2 above) before running this command.**

This will make a window pop up in your default browser. In there, you can
navigate to different directories and create new notebooks (using the `new`
panel on the top right) or open existing ones. Ensure the correct kernel is
selected (indicated on the top right of the notebook). If you installed emgfit
globally you can simply use the default kernel (usually named `Python3`). If you
installed the package into a virtual environment you must use the kernel
registered for your emgfit environment (above named `emgfit-env`). You can
easily switch kernels using the ``Change kernel`` option under the ``Kernel``
tab. Once you have imported the `emgfit` package you should be ready to analyze
some data.
