/******************************************************************************
 *
 * Copyright (c) 2013-2021, Lawrence Livermore National Security, LLC
 * and other libROM project developers. See the top-level COPYRIGHT
 * file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 *
 *****************************************************************************/

// Description: The abstract wrapper class for an abstract SVD algorithm and
//              sampler.  This class provides interfaces to each so that an
//              application only needs to instantiate one concrete
//              implementation of this class to control all aspects of basis
//              vector generation.

#include <iostream>
#include "NMROM.h"

#include <pybind11/embed.h> // everything needed for embedding
namespace py = pybind11;

namespace CAROM {

  void test()
  {
    py::scoped_interpreter guard{}; // start the interpreter and keep it alive

    py::module_ sys = py::module_::import("sys");
    std::cout << PY_SYS_PATH << std::endl;
    sys.attr("path").attr("insert")(1, "..");
    sys.attr("path").attr("insert")(1, PY_SYS_PATH);
    py::print(sys.attr("path"));
    py::module_ train = py::module_::import("train");
    train.attr("train")();

    // np.array([1, 2, 3])
    py::print("Hello, World!"); // use the Python API
  }

}
