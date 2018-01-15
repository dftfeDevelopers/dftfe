//
// File:      NonLinearSolver.cc
// Package:   dft
//
// Density Functional Theory
//
#if defined(HAVE_CONFIG_H)
#include "dft_config.h"
#endif // HAVE_CONFIG_H

#include "NonLinearSolver.h"

#if defined(HAVE_MPI)
#include <utils/mpi/MPIController.h>
#endif //HAVE_MPI

//
//
//

namespace dft {

  //
  // Constructor.
  //
  NonLinearSolver::NonLinearSolver(double tolerance,
                                   int    maxNumberIterations,
                                   int    debugLevel) :
    d_debugLevel(debugLevel),
    d_maxNumberIterations(maxNumberIterations),
    d_tolerance(tolerance)
  {
    
#if defined(HAVE_MPI)

    //
    // get MPI controller
    //
    const Utils::MPIController & mpiController = 
      Utils::MPIControllerSingleton::getInstance();

    //
    // set debugLevel to zero for all tasks other than root task
    //
    const Utils::MPIController::mpi_task_id_type rootTaskId = 
      mpiController.getRootId();
    const Utils::MPIController::mpi_task_id_type taskId = 
      mpiController.getId();
    
    if (taskId != rootTaskId)
      d_debugLevel = 0;

#endif // HAVE_MPI 


    //
    //
    //
    return;

  }

  //
  // Destructor.
  //
  NonLinearSolver::~NonLinearSolver()
  {

    //
    //
    //
    return;

  }


  //
  // Solve non-linear algebraic equation.
  //
  NonLinearSolver::ReturnValueType
  NonLinearSolver::solve(SolverFunction & function)
  {
    
    //
    //
    //
    return this->solve(function,
                       d_tolerance,
                       d_maxNumberIterations,
                       d_debugLevel);
    
  }

  //
  // Get tolerance.
  //
  double 
  NonLinearSolver::getTolerance() const
  {

    //
    //
    //
    return d_tolerance;

  }
  
  //
  // Get maximum number of iterations.
  //
  int 
  NonLinearSolver::getMaximumNumberIterations() const
  {

    //
    //
    //
    return d_maxNumberIterations;

  }


  //
  // Get debug level.
  //
  int 
  NonLinearSolver::getDebugLevel() const
  {
    
    //
    //
    //
    return d_debugLevel;

  }


}
