# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    = "-W"  # -W turns warnings into errors
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = emgfit
SOURCEDIR     = source
BUILDDIR      = build
OUTPUTDIR     = $(SOURCEDIR)/outputs

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).

%: Makefile
	rm -f $(OUTPUTDIR)/*.png $(OUTPUTDIR)/*.xlsx $(OUTPUTDIR)/*calib.txt 
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
