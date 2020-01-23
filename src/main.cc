// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2018 The Regents of the University of Michigan and DFT-FE authors.
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
//deal.II header
//
#include <deal.II/base/data_out_base.h>

//
//dft header
//
#include "constants.h"
#include "dft.h"
#include <dftUtils.h>
#include <dftParameters.h>
#include <setupGPU.h>


//
//C++ headers
//
#include <list>
#include <iostream>
#include <fstream>

using namespace dealii;

// The central DFT-FE run invocation:
template <int n>
void run_problem(const MPI_Comm &mpi_comm_replica,
		   const MPI_Comm &interpoolcomm,
		   const MPI_Comm &interBandGroupComm,
		   const unsigned int &numberEigenValues) {
      dftfe::dftClass<n> problemFE(mpi_comm_replica, interpoolcomm, interBandGroupComm);
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
typedef void (*run_fn)(const MPI_Comm &mpi_comm_replica,
			 const MPI_Comm &interpoolcomm,
			 const MPI_Comm &interBandGroupComm,
			 const unsigned int &numberEigenValues);

static run_fn order_list[] = {
	run_problem<1>,
	run_problem<2>,
	run_problem<3>,
	run_problem<4>,
	run_problem<5>,
	run_problem<6>,
	run_problem<7>,
	run_problem<8>,
	run_problem<9>,
	run_problem<10>,
	run_problem<11>,
	run_problem<12> };

int main (int argc, char *argv[])
{
  // deal.II tests expect parameter file as a first (!) argument
  AssertThrow(argc > 1,
              ExcMessage("Usage:\n"
                         "mpirun -np nProcs executable parameterfile.prm\n"
                         "\n"));
  //
  Utilities::MPI::MPI_InitFinalize mpi_initialization (argc, argv);
  const double start = MPI_Wtime();

  //
  ParameterHandler prm;
  dftfe::dftParameters::declare_parameters (prm);
  const std::string parameter_file = argv[1];
  prm.parse_input(parameter_file);
  dftfe::dftParameters::parse_parameters(prm);

  deallog.depth_console(0);

  dftfe::dftUtils::Pool kPointPool(MPI_COMM_WORLD, dftfe::dftParameters::npool);
  dftfe::dftUtils::Pool bandGroupsPool(kPointPool.get_intrapool_comm(), dftfe::dftParameters::nbandGrps);

  std::srand(dealii::Utilities::MPI::this_mpi_process(bandGroupsPool.get_intrapool_comm()));
  if (dftfe::dftParameters::verbosity>=1)
  {
      dealii::ConditionalOStream   pcout(std::cout, (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0));
      pcout <<"=================================MPI Parallelization=========================================" << std::endl ;
      pcout << "Total number of MPI tasks: "
	    << Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)
	    << std::endl;
      pcout << "k-point parallelization processor groups: "
	    << Utilities::MPI::n_mpi_processes(kPointPool.get_interpool_comm())
	    << std::endl;
      pcout << "Band parallelization processor groups: "
	    << Utilities::MPI::n_mpi_processes(bandGroupsPool.get_interpool_comm())
	    << std::endl;
      pcout << "Number of MPI tasks for finite-element domain decomposition: "
	    << Utilities::MPI::n_mpi_processes(bandGroupsPool.get_intrapool_comm())
	    << std::endl;
      pcout <<"============================================================================================" << std::endl ;
  }

#ifdef DFTFE_WITH_GPU
  if (dftfe::dftParameters::useGPU)
  {
     dftfe::setupGPU();
  }
#endif

  // set stdout precision
  std::cout << std::scientific << std::setprecision(18);

  int order = dftfe::dftParameters::finiteElementPolynomialOrder;
  if(order < 1 || order-1 >= sizeof(order_list)/sizeof(order_list[0])) {
    std::cout << "Invalid DFT-FE order " << order << std::endl;
    return -1;
  }

  run_fn run = order_list[order - 1];
  run(bandGroupsPool.get_intrapool_comm(),
      kPointPool.get_interpool_comm(),
      bandGroupsPool.get_interpool_comm(),
      dftfe::dftParameters::numberEigenValues);

  const double end = MPI_Wtime();
  if (dftfe::dftParameters::verbosity>=1 && dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
  {
      std::cout<<"============================================================================================="<<std::endl;
      std::cout << "DFT-FE Program ends. Elapsed wall time since start of the program: " << end-start << " seconds."<<std::endl;
      std::cout<<"============================================================================================="<<std::endl;
  }
  return 0;
}
