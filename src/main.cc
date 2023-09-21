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
// @author Phani Motamarri, Denis Davydov, Sambit Das
//

//
// dft header
//
#include "dftfeWrapper.h"
#include "runParameters.h"
#include "molecularDynamicsClass.h"
#include "nudgedElasticBandClass.h"
#include "geometryOptimizationClass.h"

//
// C++ headers
//
#include <fstream>
#include <iostream>
#include <list>
#include <sstream>
#include <sys/stat.h>

#if defined(DFTFE_WITH_MDI)
#  include <mdi.h>
#  include "MDIEngine.h"
#endif

int
main(int argc, char *argv[])
{
  //
  MPI_Init(&argc, &argv);

#if defined(DFTFE_WITH_MDI)
  MPI_Comm mpi_world_comm;
  int      ret;
  // Initialize MDI
  ret = MDI_Init(&argc, &argv);
  if (ret != 0)
    {
      throw std::runtime_error(
        "The MDI library was not initialized correctly.");
    }

  // Confirm that MDI was initialized successfully
  int initialized_mdi;
  ret = MDI_Initialized(&initialized_mdi);
  if (ret != 0)
    {
      throw std::runtime_error("MDI_Initialized failed.");
    }
  if (!initialized_mdi)
    {
      throw std::runtime_error(
        "MDI not initialized: did you provide the -mdi option?.");
    }

  // get the MPI communicator that spans all ranks running DFTFE
  // when using MDI, this may be a subset of MPI_COMM_WORLD

  ret = MDI_MPI_get_world_comm(&mpi_world_comm);
  if (ret != 0)
    {
      throw std::runtime_error("MDI_MPI_get_world_comm failed.");
    }

  dftfe::dftfeWrapper::globalHandlesInitialize(mpi_world_comm);
  dftfe::MDIEngine mdiEngine(mpi_world_comm, argc, argv);
  dftfe::dftfeWrapper::globalHandlesFinalize();
  MPI_Barrier(mpi_world_comm);
#else
  dftfe::dftfeWrapper::globalHandlesInitialize(MPI_COMM_WORLD);
  const double start = MPI_Wtime();
  int          world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // deal.II tests expect parameter file as a first (!) argument
  AssertThrow(argc > 1,
              dealii::ExcMessage(
                "Usage:\n"
                "mpirun -np nProcs executable parameterfile.prm\n"
                "\n"));
  const std::string parameter_file = argv[1];

  dftfe::runParameters runParams;
  runParams.parse_parameters(parameter_file);

  if (runParams.verbosity >= 1 && world_rank == 0)
    {
      std::cout
        << "=========================================================================================================="
        << std::endl;
      std::cout
        << "=========================================================================================================="
        << std::endl;
      std::cout
        << "			Welcome to the Open Source program DFT-FE version	1.1.0-pre		        "
        << std::endl;
      std::cout
        << "This is a C++ code for materials modeling from first principles using Kohn-Sham density functional theory."
        << std::endl;
      std::cout
        << "This is a real-space code for periodic, semi-periodic and non-periodic pseudopotential"
        << std::endl;
      std::cout
        << "and all-electron calculations, and is based on adaptive finite-element discretization."
        << std::endl;
      std::cout
        << "For further details, and citing, please refer to our website: https://sites.google.com/umich.edu/dftfe"
        << std::endl;
      std::cout
        << "=========================================================================================================="
        << std::endl;
      std::cout
        << " DFT-FE Mentors and Development leads (alphabetically) :									"
        << std::endl;
      std::cout << "														" << std::endl;
      std::cout << " Sambit Das               - University of Michigan, USA"
                << std::endl;
      std::cout << " Vikram Gavini            - University of Michigan, USA"
                << std::endl;
      std::cout
        << " Phani Motamarri          - Indian Institute of Science, India"
        << std::endl;
      std::cout
        << " (A complete list of the many authors that have contributed to DFT-FE can be found in the authors file)"
        << std::endl;
      std::cout
        << "=========================================================================================================="
        << std::endl;
      std::cout
        << " 	     Copyright (c) 2017-2022 The Regents of the University of Michigan and DFT-FE authors         "
        << std::endl;
      std::cout
        << " 			DFT-FE is published under [LGPL v2.1 or newer] 				"
        << std::endl;
      std::cout
        << "=========================================================================================================="
        << std::endl;
      std::cout
        << "=========================================================================================================="
        << std::endl;

      runParams.print_parameters();
    }


  if (runParams.solvermode == "MD")
    {
      dftfe::molecularDynamicsClass mdClass(parameter_file,
                                            runParams.restartFilesPath,
                                            MPI_COMM_WORLD,
                                            runParams.restart,
                                            runParams.verbosity);

      int status = mdClass.runMD();
    }

  else if (runParams.solvermode == "NEB")
    {
      dftfe::nudgedElasticBandClass nebClass(
        parameter_file,
        runParams.restartFilesPath,
        MPI_COMM_WORLD,
        runParams.restart,
        runParams.verbosity,
        runParams.numberOfImages,
        runParams.imageFreeze,
        runParams.Kmax,
        runParams.Kmin,
        runParams.pathThreshold,
        runParams.maximumNEBiteration,
        runParams.maxLineSearchIterCGPRP,
        runParams.lbfgsNumPastSteps,
        runParams.bfgsStepMethod,
        runParams.optimizermaxIonUpdateStep,
        runParams.optimizationSolver,
        runParams.coordinatesFileNEB,
        runParams.domainVectorsFileNEB,
        runParams.ionRelaxFlagsFile);

      int status = nebClass.findMEP();
    }
  else if (runParams.solvermode == "GEOOPT")
    {
      dftfe::geometryOptimizationClass geoOpt(parameter_file,
                                              runParams.restartFilesPath,
                                              MPI_COMM_WORLD,
                                              runParams.restart,
                                              runParams.verbosity);
      geoOpt.runOpt();
    }
  else if (runParams.solvermode == "NONE")
    {
      dftfe::dftfeWrapper dftfeWrapped(parameter_file,
                                       MPI_COMM_WORLD,
                                       true,
                                       true,
                                       "NONE",
                                       runParams.restartFilesPath,
                                       runParams.verbosity);
      dftfeWrapped.writeMesh();
    }
  else if (runParams.solvermode == "NSCF")
    {
      dftfe::dftfeWrapper dftfeWrapped(parameter_file,
                                       MPI_COMM_WORLD,
                                       true,
                                       true,
                                       "NSCF",
                                       runParams.restartFilesPath,
                                       runParams.verbosity);
      dftfeWrapped.run();
    }

  else
    {
      dftfe::dftfeWrapper dftfeWrapped(parameter_file,
                                       MPI_COMM_WORLD,
                                       true,
                                       true,
                                       "GS",
                                       runParams.restartFilesPath,
                                       runParams.verbosity);
      dftfeWrapped.run();
    }


  const double end = MPI_Wtime();
  if (runParams.verbosity >= 1 && world_rank == 0)
    {
      std::cout
        << "============================================================================================="
        << std::endl;
      std::cout
        << "DFT-FE Program ends. Elapsed wall time since start of the program: "
        << end - start << " seconds." << std::endl;
      std::cout
        << "============================================================================================="
        << std::endl;
    }

  dftfe::dftfeWrapper::globalHandlesFinalize();
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  MPI_Finalize();
  return 0;
}
