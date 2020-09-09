#!/bin/bash
# Script based off of https://github.com/voorhoede/front-end-tooling-recipes/blob/master/travis-deploy-to-gh-pages/scripts/deploy.sh

set -e # exit with nonzero exit code if anything fails

echo "Starting to update gh-pages index"

#go to home and setup git
cd $HOME
git config --global user.email "travis@travis-ci.org"
git config --global user.name "Travis"

#using token clone gh-pages branch
git clone --quiet --branch=gh-pages https://${GH_USER}:${GITHUB_TOKEN}@github.com/${GH_USER}/${GH_REPO}.git gh-pages > /dev/null

# Update gh-pages index
cd gh-pages
bash ./make-index.sh

#add, commit and push files
git add -f .
git commit --allow-empty -m "Update gh-pages index with travis build $TRAVIS_BUILD_NUMBER"
git push -fq origin gh-pages

echo "Done updating gh-pages index"
