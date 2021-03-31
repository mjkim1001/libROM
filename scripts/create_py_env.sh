#!/usr/bin/env bash

###############################################################################
#
#  Copyright (c) 2013-2019, Lawrence Livermore National Security, LLC
#  and other libROM project developers. See the top-level COPYRIGHT
#  file for details.
#
#  SPDX-License-Identifier: (Apache-2.0 OR MIT)
#
###############################################################################

REPO_PREFIX=$(git rev-parse --show-toplevel)

if [ ! -d "${REPO_PREFIX}/dependencies/py_env" ]; then
  python3 -m venv ${REPO_PREFIX}/dependencies/py_env
  source ${REPO_PREFIX}/dependencies/py_env/bin/activate
  pip install torch scipy
fi
