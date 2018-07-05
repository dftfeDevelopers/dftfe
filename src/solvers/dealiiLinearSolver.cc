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
		                     const double relTolerance,
		                     const unsigned int maxNumberIterations,
		                     const unsigned int  debugLevel)
    {
      //compute RHS
      vectorType rhs;
      problem.computeRhs(rhs);

      //create dealii solver control object
      dealii::SolverControl solverControl(maxNumberIterations,relTolerance*rhs.l2_norm());


      //initialize preconditioner
      dealii::PreconditionJacobi<dealiiLinearSolverProblem> preconditioner;
      preconditioner.initialize (problem, 0.3);

      vectorType & x= problem.getX();
      try{
	x.update_ghost_values();

	if (d_type==CG)
	{
	  dealii::SolverCG<vectorType> solver(solverControl);
	  solver.solve(problem,x, rhs, preconditioner);
	}
	else if (d_type==GMRES)
	{
	  dealii::SolverGMRES<vectorType> solver(solverControl);
	  solver.solve(problem,x, rhs, preconditioner);
	}

	problem.distributeX();
	x.update_ghost_values();
      }
      catch (...) {
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
    }
}
