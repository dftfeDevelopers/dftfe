// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022 The Regents of the University of Michigan and DFT-FE
// authors.
//
// This file is part of the DFT-FE code.
//
// The DFT-FE code is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the DFT-FE distribution.
//
// ---------------------------------------------------------------------
//
// @author Phani Motamarri, Sambit Das
//
#include <deal.II/base/data_out_base.h>
#include <deal.II/base/parameter_handler.h>
#include <runParameters.h>
#include <fstream>
#include <iostream>

// using namespace dealii;

namespace dftfe
{
  namespace internalRunParameters
  {
    void
    declare_parameters(dealii::ParameterHandler &prm)
    {
      prm.declare_entry(
        "SOLVER MODE",
        "GS",
        dealii::Patterns::Selection("GS|MD|NEB|GEOOPT|NONE"),
        "[Standard] DFT-FE SOLVER MODE: If GS: performs GroundState calculations. If MD: performs Molecular Dynamics Simulation. If NEB: performs a NEB calculation. If GEOOPT: performs an ion and/or cell optimization calculation. If NONE: the density is initialised with superposition of atomic densities and is written to file along with mesh data.");


      prm.declare_entry(
        "VERBOSITY",
        "1",
        dealii::Patterns::Integer(-1, 5),
        "[Standard] Parameter to control verbosity of terminal output. Ranges from 1 for low, 2 for medium (prints some more additional information), 3 for high (prints eigenvalues and fractional occupancies at the end of each self-consistent field iteration), and 4 for very high, which is only meant for code development purposes. VERBOSITY=0 is only used for unit testing and shouldn't be used by standard users. VERBOSITY=-1 ensures no outout is printed, which is useful when DFT-FE is used as a calculator inside a larger workflow where multiple parallel DFT-FE jobs might be running, for example when using ASE or generating training data for ML workflows.");

      prm.declare_entry(
        "RESTART",
        "false",
        dealii::Patterns::Bool(),
        "[Standard] If set to true solvermode triggers restart checks and modifies the input files for coordinates, domain vectors. Default: false.");

      prm.declare_entry("RESTART FOLDER",
                        ".",
                        dealii::Patterns::Anything(),
                        "[Standard] Folder to store restart files.");
    }
  } // namespace internalRunParameters



  void
  runParameters::parse_parameters(const std::string &parameter_file)
  {
    dealii::ParameterHandler prm;
    internalRunParameters::declare_parameters(prm);
    prm.parse_input(parameter_file, "", true);

    verbosity        = prm.get_integer("VERBOSITY");
    solvermode       = prm.get("SOLVER MODE");
    restart          = prm.get_bool("RESTART");
    restartFilesPath = prm.get("RESTART FOLDER");

    const bool printParametersToFile = false;
    if (printParametersToFile &&
        dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        prm.print_parameters(std::cout, dealii::ParameterHandler::OutputStyle::LaTeX);
        exit(0);
      }
  }

} // namespace dftfe
