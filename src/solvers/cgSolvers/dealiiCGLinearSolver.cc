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

#include <dealiiCGLinearSolver.h>

namespace dftfe {

    //constructor
    dealiiCGLinearSolver::dealiiCGLinearSolver(const MPI_Comm &mpi_comm):
      mpi_communicator (mpi_comm),
      n_mpi_processes (dealii::Utilities::MPI::n_mpi_processes(mpi_comm)),
      this_mpi_process (dealii::Utilities::MPI::this_mpi_process(mpi_comm)),
      pcout (std::cout, (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0))
    {

    }


    //solve
    void dealiiCGLinearSolver::solve(dealiiLinearSolverFunction & function,
		                     const double relTolerance,
		                     const unsigned int maxNumberIterations,
		                     const unsigned int  debugLevel)
    {
      //compute RHS
      vectorType rhs;
      function.computeRhs(rhs);

      //create dealii solver control object
      dealii::SolverControl solverControl(maxNumberIterations,relTolerance*rhs.l2_norm());


      //initialize preconditioner
      dealii::PreconditionJacobi<dealiiLinearSolverFunction> preconditioner;
      preconditioner.initialize (function, 0.3);

      vectorType & x= function.getX();
      try{
	x.update_ghost_values();

	dealii::SolverCG<vectorType> solver(solverControl);
	solver.solve(function,x, rhs, preconditioner);

	function.distributeX();
	x.update_ghost_values();
      }
      catch (...) {
	pcout << "\nWarning: solver did not converge as per set tolerances. consider increasing maxLinearSolverIterations or decreasing relLinearSolverTolerance.\n";
      }

      if (debugLevel==2)
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
