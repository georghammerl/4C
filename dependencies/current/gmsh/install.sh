#!/bin/bash
# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

# Install gmsh
# Call with
# ./install.sh /path/to/install/dir

# Exit the script at the first failure
set -e

INSTALL_DIR="$1"
# Number of procs for building (default 4)
NPROCS=${NPROCS=4}
VERSION="4.15.0"
CHECKSUM="abb2632715bd7d0130ded7144fd6263635cd7dea883b8df61ba4da58ce6a1dfe"

# get precompiled gmsh sdk
wget --no-verbose https://gmsh.info/src/gmsh-${VERSION}-source.tgz

# Verify checksum
if [ $CHECKSUM = `sha256sum gmsh-${VERSION}-source.tgz | awk '{print $1}'` ]
then
  echo "Checksum matches"
else
  echo "Checksum does not match"
  exit 1
fi

# untar
tar -xzf gmsh-${VERSION}-source.tgz

# remove tar file
rm gmsh-${VERSION}-source.tgz

cd gmsh-${VERSION}-source

# build gmsh
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} -DENABLE_BUILD_DYNAMIC=1 ..
make -j${NPROCS}

make install

cd ../../
# remove source folder
rm -rf gmsh-${VERSION}-source