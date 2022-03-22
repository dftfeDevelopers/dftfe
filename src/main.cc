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
// deal.II header
//
#include <deal.II/base/data_out_base.h>

//
// dft header
//
#include <cudaHelpers.h>
#include <dftParameters.h>
#include <dftUtils.h>

#include "constants.h"
#include "dft.h"


//
// C++ headers
//
#include <fstream>
#include <iostream>
#include <list>

using namespace dealii;

// The central DFT-FE run invocation:
template <int n1, int n2>
void
run_problem(const MPI_Comm &    mpi_comm_replica,
            const MPI_Comm &    interpoolcomm,
            const MPI_Comm &    interBandGroupComm,
            const unsigned int &numberEigenValues)
{
  dftfe::dftClass<n1, n2> problemFE(mpi_comm_replica,
                                    interpoolcomm,
                                    interBandGroupComm);
  problemFE.d_numEigenValues = numberEigenValues;
  problemFE.set();
  problemFE.init();
  problemFE.run();
}

// Dynamically access dftClass<n> objects by order.
//  Note that we can't store a list of classes because the types differ,
//  but we can store a list of functions that use them in an n-independent way.
//
//  Also note element 0 is order 1.
//
typedef void (*run_fn)(const MPI_Comm &    mpi_comm_replica,
                       const MPI_Comm &    interpoolcomm,
                       const MPI_Comm &    interBandGroupComm,
                       const unsigned int &numberEigenValues);

static run_fn order_list[] = {
#ifdef DFTFE_MINIMAL_COMPILE
  run_problem<2, 2>,
  run_problem<3, 3>,
  run_problem<4, 4>,
  run_problem<5, 5>,
  run_problem<6, 6>,
  run_problem<6, 7>,
  run_problem<6, 8>,
  run_problem<6, 9>,
  run_problem<7, 7>
#else
  run_problem<1, 1>,  run_problem<1, 2>,  run_problem<2, 2>,
  run_problem<2, 3>,  run_problem<2, 4>,  run_problem<3, 3>,
  run_problem<3, 4>,  run_problem<3, 5>,  run_problem<3, 6>,
  run_problem<4, 4>,  run_problem<4, 5>,  run_problem<4, 6>,
  run_problem<4, 7>,  run_problem<4, 8>,  run_problem<5, 5>,
  run_problem<5, 6>,  run_problem<5, 7>,  run_problem<5, 8>,
  run_problem<5, 9>,  run_problem<5, 10>, run_problem<6, 6>,
  run_problem<6, 7>,  run_problem<6, 8>,  run_problem<6, 9>,
  run_problem<6, 10>, run_problem<6, 11>, run_problem<6, 12>,
  run_problem<7, 7>,  run_problem<7, 8>,  run_problem<7, 9>,
  run_problem<7, 10>, run_problem<7, 11>, run_problem<7, 12>,
  run_problem<7, 13>, run_problem<7, 14>, run_problem<8, 8>,
  run_problem<8, 9>,  run_problem<8, 10>, run_problem<8, 11>,
  run_problem<8, 12>, run_problem<8, 13>, run_problem<8, 14>,
  run_problem<8, 15>, run_problem<8, 16>
#endif
};

int
main(int argc, char *argv[])
{
  // deal.II tests expect parameter file as a first (!) argument
  AssertThrow(argc > 1,
              ExcMessage("Usage:\n"
                         "mpirun -np nProcs executable parameterfile.prm\n"
                         "\n"));
  //
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);
  const double                     start = MPI_Wtime();

  //
  ParameterHandler prm;
  dftfe::dftParameters::declare_parameters(prm);
  const std::string parameter_file = argv[1];
  prm.parse_input(parameter_file);
  dftfe::dftParameters::parse_parameters(prm);

  deallog.depth_console(0);

  dftfe::dftUtils::Pool kPointPool(MPI_COMM_WORLD, dftfe::dftParameters::npool);
  dftfe::dftUtils::Pool bandGroupsPool(kPointPool.get_intrapool_comm(),
                                       dftfe::dftParameters::nbandGrps);

  std::srand(dealii::Utilities::MPI::this_mpi_process(
    bandGroupsPool.get_intrapool_comm()));
  if (dftfe::dftParameters::verbosity >= 1)
    {
      dealii::ConditionalOStream pcout(
        std::cout,
        (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0));
      pcout
        << "=================================MPI Parallelization========================================="
        << std::endl;
      pcout << "Total number of MPI tasks: "
            << Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) << std::endl;
      pcout << "k-point parallelization processor groups: "
            << Utilities::MPI::n_mpi_processes(kPointPool.get_interpool_comm())
            << std::endl;
      pcout << "Band parallelization processor groups: "
            << Utilities::MPI::n_mpi_processes(
                 bandGroupsPool.get_interpool_comm())
            << std::endl;
      pcout << "Number of MPI tasks for finite-element domain decomposition: "
            << Utilities::MPI::n_mpi_processes(
                 bandGroupsPool.get_intrapool_comm())
            << std::endl;
      pcout
        << "============================================================================================"
        << std::endl;
    }

#ifdef DFTFE_WITH_GPU
  if (dftfe::dftParameters::useGPU)
    {
      dftfe::cudaUtils::setupGPU();
    }
#endif

  // set stdout precision
  std::cout << std::scientific << std::setprecision(18);

  int order = dftfe::dftParameters::finiteElementPolynomialOrder;
  int orderElectro =
    dftfe::dftParameters::finiteElementPolynomialOrderElectrostatics;

#ifdef DFTFE_MINIMAL_COMPILE
  if (order < 2 || order > 7)
    {
      std::cout << "Invalid DFT-FE order " << order << std::endl;
      return -1;
    }

  if (order > 5 && order < 7)
    {
      if (orderElectro < order || orderElectro > (order + 3))
        {
          std::cout << "Invalid DFT-FE order electrostatics " << orderElectro
                    << std::endl;
          return -1;
        }
    }
  else
    {
      if (orderElectro != order)
        {
          std::cout << "Invalid DFT-FE order electrostatics " << orderElectro
                    << std::endl;
          return -1;
        }
    }

  int listIndex = 0;
  for (int i = 2; i <= order; i++)
    {
      int maxElectroOrder = (i < order) ? (i + 3) : orderElectro;
      if (i != 6)
        maxElectroOrder = i;
      for (int j = i; j <= maxElectroOrder; j++)
        listIndex++;
    }
#else
  if (order < 1 || order > 8)
    {
      std::cout << "Invalid DFT-FE order " << order << std::endl;
      return -1;
    }

  if (orderElectro < order || orderElectro > order * 2)
    {
      std::cout << "Invalid DFT-FE order electrostatics " << orderElectro
                << std::endl;
      return -1;
    }


  int listIndex = 0;
  for (int i = 1; i <= order; i++)
    {
      int maxElectroOrder = (i < order) ? 2 * i : orderElectro;
      for (int j = i; j <= maxElectroOrder; j++)
        listIndex++;
    }
#endif

  run_fn run = order_list[listIndex - 1];
  run(bandGroupsPool.get_intrapool_comm(),
      kPointPool.get_interpool_comm(),
      bandGroupsPool.get_interpool_comm(),
      dftfe::dftParameters::numberEigenValues);

  const double end = MPI_Wtime();
  if (dftfe::dftParameters::verbosity >= 1 &&
      dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
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
  return 0;
}
