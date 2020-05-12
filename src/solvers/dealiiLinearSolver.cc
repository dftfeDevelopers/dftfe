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
// @author Sambit Das
//

#include <dealiiLinearSolver.h>
#include <dftParameters.h>

namespace dftfe {

    //constructor
    dealiiLinearSolver::dealiiLinearSolver(const MPI_Comm &mpi_comm,
	                                       const solverType type):
      mpi_communicator (mpi_comm),
      d_type(type),
      n_mpi_processes (dealii::Utilities::MPI::n_mpi_processes(mpi_comm)),
      this_mpi_process (dealii::Utilities::MPI::this_mpi_process(mpi_comm)),
      pcout (std::cout, (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0))
    {

    }


    //solve
    void dealiiLinearSolver::solve(dealiiLinearSolverProblem & problem,
				   const double absTolerance,
				   const unsigned int maxNumberIterations,
				   const unsigned int  debugLevel,
				   bool distributeFlag)
    {
      int this_process;
      MPI_Comm_rank(mpi_communicator, &this_process);
      MPI_Barrier(mpi_communicator);
      double start_time=MPI_Wtime();
      double time;

      //compute RHS
      distributedCPUVec<double> rhs;
      problem.computeRhs(rhs);

      MPI_Barrier(mpi_communicator);
      time = MPI_Wtime();

      if (dftParameters::verbosity>=4)
         pcout<<"Time for compute rhs: "<<time-start_time<<std::endl;


      //create dealii solver control object
      dealii::SolverControl solverControl(maxNumberIterations,absTolerance);


      //initialize preconditioner
      dealii::PreconditionJacobi<dealiiLinearSolverProblem> preconditioner;
      preconditioner.initialize (problem, 0.3);

      distributedCPUVec<double> & x= problem.getX();
      try{
	x.update_ghost_values();

	if (d_type==CG)
	{
	  dealii::SolverCG<distributedCPUVec<double>> solver(solverControl);
	  solver.solve(problem,x, rhs, preconditioner);
	}
	else if (d_type==GMRES)
	{
	  dealii::SolverGMRES<distributedCPUVec<double>> solver(solverControl);
	  solver.solve(problem,x, rhs, preconditioner);
	}

	if(distributeFlag)
	  problem.distributeX();

	x.update_ghost_values();
      }
      catch (...)
      {
	AssertThrow(false,dealii::ExcMessage("DFT-FE Error: Poisson solver did not converge as per set tolerances. consider increasing MAXIMUM ITERATIONS in Poisson problem parameters."));
	pcout << "\nWarning: solver did not converge as per set tolerances. consider increasing maxLinearSolverIterations or decreasing relLinearSolverTolerance.\n";
	pcout << "Current abs. residual: "<<solverControl.last_value()<<std::endl;
      }

      if (debugLevel>=2)
      {
	pcout<<std::endl;
	char buffer[200];
	sprintf(buffer, "initial abs. residual: %12.6e, current abs. residual: %12.6e, nsteps: %u, abs. tolerance criterion: %12.6e\n\n", \
	      solverControl.initial_value(),				\
	      solverControl.last_value(),					\
	      solverControl.last_step(), solverControl.tolerance());
	pcout<<buffer;
      }

      MPI_Barrier(mpi_communicator);
      time = MPI_Wtime() - time;

      if (dftParameters::verbosity>=4)
         pcout<<"Time for Poisson/Helmholtz problem CG/GMRES iterations: "<<time<<std::endl;

    }
}
