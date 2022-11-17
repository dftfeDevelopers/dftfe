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

using namespace dealii;

namespace dftfe
{
  namespace internalRunParameters
  {
    void
    declare_parameters(ParameterHandler &prm)
    {
      prm.declare_entry(
        "SOLVER MODE",
        "GS",
        Patterns::Selection("GS|MD|NEB|GEOOPT|NONE"),
        "[Standard] DFT-FE SOLVER MODE: If GS: performs GroundState calculations. If MD: performs Molecular Dynamics Simulation. If NEB: performs a NEB calculation. If GEOOPT: performs an ion and/or cell optimization calculation. If NONE: the density is initialised with superposition of atomic densities and is written to file along with mesh data.");


      prm.declare_entry(
        "VERBOSITY",
        "1",
        Patterns::Integer(-1, 5),
        "[Standard] Parameter to control verbosity of terminal output. Ranges from 1 for low, 2 for medium (prints some more additional information), 3 for high (prints eigenvalues and fractional occupancies at the end of each self-consistent field iteration), and 4 for very high, which is only meant for code development purposes. VERBOSITY=0 is only used for unit testing and shouldn't be used by standard users. VERBOSITY=-1 ensures no outout is printed, which is useful when DFT-FE is used as a calculator inside a larger workflow where multiple parallel DFT-FE jobs might be running, for example when using ASE or generating training data for ML workflows.");

      prm.declare_entry(
        "RESTART",
        "false",
        Patterns::Bool(),
        "[Standard] If set to true solvermode triggers restart checks and modifies the input files for coordinates, domain vectors. Default: false.");

      prm.declare_entry("RESTART FOLDER",
                        ".",
                        Patterns::Anything(),
                        "[Standard] Folder to store restart files.");
      prm.enter_subsection("NEB");
      {
        prm.declare_entry(
          "ALLOW IMAGE FREEZING",
          "false",
          Patterns::Bool(),
          "If true images less than threshold will freeze for optimization step");

        prm.declare_entry(
          "NUMBER OF IMAGES",
          "1",
          Patterns::Integer(1, 50),
          "[Standard] NUMBER OF IMAGES:Default option is 1. When NEB is triggered this controls the total number of images along the MEP including the end points");

        prm.declare_entry(
          "MAXIMUM SPRING CONSTANT",
          "1e-1",
          Patterns::Double(),
          "[Standard] Sets the maximum allowable spring constant in (Ha/bohr^2)");

        prm.declare_entry(
          "MINIMUM SPRING CONSTANT",
          "5e-2",
          Patterns::Double(),
          "[Standard] Sets the minimum allowable spring constant in (Ha/bohr^2)");

        prm.declare_entry(
          "PATH THRESHOLD",
          "1e-1",
          Patterns::Double(),
          "[Standard] Simulation stops when the error(norm of force orthogonal to path in Ha/bohr) is less than PATH THRESHOLD ");


        prm.declare_entry(
          "MAXIMUM NUMBER OF NEB ITERATIONS",
          "100",
          Patterns::Integer(1, 250),
          "[Standard] Maximum number of NEB iterations that will be performed in the simulation");

        prm.declare_entry(
          "NEB COORDINATES FILE",
          "",
          Patterns::Anything(),
          "[Standard] Atomic-coordinates input file name. For fully non-periodic domain give Cartesian coordinates of the atoms (in a.u) with respect to origin at the center of the domain. For periodic and semi-periodic domain give fractional coordinates of atoms. File format (example for two atoms): Atom1-atomic-charge Atom1-valence-charge x1 y1 z1 (row1), Atom2-atomic-charge Atom2-valence-charge x2 y2 z2 (row2). The number of rows must be equal to NATOMS, and number of unique atoms must be equal to NATOM TYPES.");

        prm.declare_entry(
          "NEB DOMAIN VECTORS FILE",
          "",
          Patterns::Anything(),
          "[Standard] Atomic-coordinates input file name. For fully non-periodic domain give Cartesian coordinates of the atoms (in a.u) with respect to origin at the center of the domain. For periodic and semi-periodic domain give fractional coordinates of atoms. File format (example for two atoms): Atom1-atomic-charge Atom1-valence-charge x1 y1 z1 (row1), Atom2-atomic-charge Atom2-valence-charge x2 y2 z2 (row2). The number of rows must be equal to NATOMS, and number of unique atoms must be equal to NATOM TYPES.");
      }
      prm.leave_subsection();
    }
  } // namespace internalRunParameters



  void
  runParameters::parse_parameters(const std::string &parameter_file)
  {
    ParameterHandler prm;
    internalRunParameters::declare_parameters(prm);
    prm.parse_input(parameter_file, "", true);

    verbosity        = prm.get_integer("VERBOSITY");
    solvermode       = prm.get("SOLVER MODE");
    restart          = prm.get_bool("RESTART");
    restartFilesPath = prm.get("RESTART FOLDER");
    prm.enter_subsection("NEB");
    {
      numberOfImages      = prm.get_integer("NUMBER OF IMAGES");
      imageFreeze         = prm.get_bool("ALLOW IMAGE FREEZING");
      Kmax                = prm.get_double("MAXIMUM SPRING CONSTANT");
      Kmin                = prm.get_double("MINIMUM SPRING CONSTANT");
      pathThreshold       = prm.get_double("PATH THRESHOLD");
      maximumNEBiteration = prm.get_integer("MAXIMUM NUMBER OF NEB ITERATIONS");
      coordinatesFileNEB  = prm.get("NEB COORDINATES FILE");
      domainVectorsFileNEB = prm.get("NEB DOMAIN VECTORS FILE");
    }



    const bool printParametersToFile = false;
    if (printParametersToFile &&
        Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        prm.print_parameters(std::cout, ParameterHandler::OutputStyle::LaTeX);
        exit(0);
      }
  }

} // namespace dftfe
