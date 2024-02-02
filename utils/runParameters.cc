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

#include <runParameters.h>



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
        dealii::Patterns::Selection("GS|MD|NEB|GEOOPT|NONE|NSCF"),
        "[Standard] DFT-FE SOLVER MODE: If GS: performs GroundState calculations. If MD: performs Molecular Dynamics Simulation. If NEB: performs a NEB calculation. If GEOOPT: performs an ion and/or cell optimization calculation. If NONE: the density is initialised with superposition of atomic densities and is written to file along with mesh data. If NSCF: The density from the restart files of the GS run are used to perform NSCF calculation at the k-points specified");


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
      prm.enter_subsection("NEB");
      {
        prm.declare_entry(
          "ALLOW IMAGE FREEZING",
          "false",
          dealii::Patterns::Bool(),
          "If true images less than threshold will freeze for optimization step");

        prm.declare_entry(
          "NUMBER OF IMAGES",
          "7",
          dealii::Patterns::Integer(1, 50),
          "[Standard] NUMBER OF IMAGES:Default option is 7. When NEB is triggered this controls the total number of images along the MEP including the end points");

        prm.declare_entry(
          "MAXIMUM SPRING CONSTANT",
          "5e-3",
          dealii::Patterns::Double(),
          R"([Standard] Sets the maximum allowable spring constant in (Ha/bohr\^2))");

        prm.declare_entry(
          "MINIMUM SPRING CONSTANT",
          "2e-3",
          dealii::Patterns::Double(),
          R"([Standard] Sets the minimum allowable spring constant in (Ha/bohr\^2))");

        prm.declare_entry(
          "PATH THRESHOLD",
          "5e-4",
          dealii::Patterns::Double(),
          "[Standard] Simulation stops when the error(norm of force orthogonal to path in Ha/bohr) is less than PATH THRESHOLD ");


        prm.declare_entry(
          "MAXIMUM NUMBER OF NEB ITERATIONS",
          "100",
          dealii::Patterns::Integer(1, 250),
          "[Standard] Maximum number of NEB iterations that will be performed in the simulation");

        prm.declare_entry(
          "NEB OPT SOLVER",
          "LBFGS",
          dealii::Patterns::Selection("BFGS|LBFGS|CGPRP"),
          "[Standard] Method for Ion relaxation solver. LBFGS is the default");
        prm.declare_entry(
          "MAXIMUM ION UPDATE STEP",
          "0.5",
          dealii::Patterns::Double(0, 5.0),
          "[Standard] Sets the maximum allowed step size (displacement in a.u.) during ion relaxation.");
        prm.declare_entry(
          "MAX LINE SEARCH ITER",
          "5",
          dealii::Patterns::Integer(1, 100),
          "[Standard] Sets the maximum number of line search iterations in the case of CGPRP. Default is 5.");
        prm.declare_entry(
          "ION RELAX FLAGS FILE",
          "",
          dealii::Patterns::Anything(),
          "[Standard] File specifying the permission flags (1-free to move, 0-fixed) and external forces for the 3-coordinate directions and for all atoms. File format (example for two atoms with atom 1 fixed and atom 2 free and 0.01 Ha/Bohr force acting on atom 2): 0 0 0 0.0 0.0 0.0(row1), 1 1 1 0.0 0.0 0.01(row2). External forces are optional.");
        prm.declare_entry(
          "BFGS STEP METHOD",
          "QN",
          dealii::Patterns::Selection("QN|RFO"),
          "[Standard] Method for computing update step in BFGS. Quasi-Newton step (default) or Rational Function Step as described in JPC 1985, 89:52-57.");
        prm.declare_entry(
          "LBFGS HISTORY",
          "5",
          dealii::Patterns::Integer(1, 20),
          "[Standard] Number of previous steps to considered for the LBFGS update.");

        prm.declare_entry(
          "NEB COORDINATES FILE",
          "",
          dealii::Patterns::Anything(),
          "[Standard] Atomic-coordinates input file name. For fully non-periodic domain give Cartesian coordinates of the atoms (in a.u) with respect to origin at the center of the domain. For periodic and semi-periodic domain give fractional coordinates of atoms. File format (example for two atoms): Atom1-atomic-charge Atom1-valence-charge x1 y1 z1 (row1), Atom2-atomic-charge Atom2-valence-charge x2 y2 z2 (row2). The number of rows must be equal to NATOMS, and number of unique atoms must be equal to NATOM TYPES.");

        prm.declare_entry(
          "NEB DOMAIN VECTORS FILE",
          "",
          dealii::Patterns::Anything(),
          "[Standard] Atomic-coordinates input file name. For fully non-periodic domain give Cartesian coordinates of the atoms (in a.u) with respect to origin at the center of the domain. For periodic and semi-periodic domain give fractional coordinates of atoms. File format (example for two atoms): Atom1-atomic-charge Atom1-valence-charge x1 y1 z1 (row1), Atom2-atomic-charge Atom2-valence-charge x2 y2 z2 (row2). The number of rows must be equal to NATOMS, and number of unique atoms must be equal to NATOM TYPES.");
      }
      prm.leave_subsection();
    }
  } // namespace internalRunParameters

  void
  runParameters::print_parameters()
  {
    const bool printParametersToFile = false;
    if (printParametersToFile &&
        dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        prm.print_parameters(std::cout,
                             dealii::ParameterHandler::OutputStyle::LaTeX);
        exit(0);
      }

    if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 &&
        verbosity >= 1 && true)
      {
        prm.print_parameters(std::cout, dealii::ParameterHandler::ShortText);
      }
  }

  void
  runParameters::parse_parameters(const std::string &parameter_file)
  {
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
      domainVectorsFileNEB      = prm.get("NEB DOMAIN VECTORS FILE");
      maxLineSearchIterCGPRP    = prm.get_integer("MAX LINE SEARCH ITER");
      bfgsStepMethod            = prm.get("BFGS STEP METHOD");
      optimizermaxIonUpdateStep = prm.get_double("MAXIMUM ION UPDATE STEP");
      lbfgsNumPastSteps         = prm.get_integer("LBFGS HISTORY");
      optimizationSolver        = prm.get("NEB OPT SOLVER");
      ionRelaxFlagsFile         = prm.get("ION RELAX FLAGS FILE");
    }
  }

} // namespace dftfe
