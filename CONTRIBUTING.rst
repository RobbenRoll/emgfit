============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every
little bit helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/RobbenRoll/emgfit/issues.

If you are reporting a bug, please include:

* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug"
is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "feature"
is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

emgfit could always use more documentation, whether
as part of the official emgfit docs, in docstrings,
or even on the web in blog posts, articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/RobbenRoll/emgfit/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
------------

Ready to contribute? Here's how to set up `emgfit` for local development.

1. Fork the `emgfit` repo on GitHub.
2. Clone your fork locally::

    $ git clone git@github.com:your_name_here/emgfit.git

3. Install your local copy into a virtualenv. Assuming you have virtualenvwrapper 
   installed, this is how you set up your fork for local development::

    $ mkvirtualenv emgfit
    $ cd emgfit/
    $ python setup.py develop

4. Install development requirements by executing the following in the emgfit/ directory:
    
    $ pip install -r requirements-dev.txt 

5. Install pandoc following the instructions given for your given operating 
   system at https://pandoc.org/installing.html.
6. Navigate into your local emgfit directory and install nbstripout by running:
    
    $ nbstripout --install 

   This adds a git filter which ensures that the output is stripped from 
   Jupyter notebooks before a commit to avoid blowing up git with bulky output 
   code. This step only needs to be executed once before your first commit.
7. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.
8. When you're done making changes, check that your changes pass flake8 and the 
   tests, including testing other Python versions with tox::

    $ flake8 emgfit
    $ pytest emgfit/tests
    
9. Build the documentation and check that any changes are properly displayed in 
   the html pages under docs/build/html. The documentation build also 
   automatically runs the tutorial notebook which will fail if errors are 
   encountered.

    $ cd docs 
    $ make html 

10. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

11. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in release-history.rst.
3. The pull request should work for Python >= 3.8. Check
   https://github.com/RobbenRoll/emgfit/actions/workflows/CI-tests.yml
   and make sure that the tests pass for all supported Python versions.
