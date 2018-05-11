#!/bin/bash

# This assumes you have already installed contact_map using conda
pushd ~
conda uninstall -y contact_map
git clone https://github.com/dwhswenson/contact_map.git
pushd contact_map
pip install -e .

if [ -n "$CONTACT_MAP_BRANCH" ]
then
    echo $CONTACT_MAP_BRANCH
    git checkout -b $CONTACT_MAP_BRANCH origin/$CONTACT_MAP_BRANCH
fi
git branch
popd
popd
