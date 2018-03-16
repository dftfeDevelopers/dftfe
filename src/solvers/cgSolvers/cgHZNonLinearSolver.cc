//
// File:      CGNonLinearSolver.cc
// Package:   dft
//
// Density Functional Theory
//
#if defined(HAVE_CONFIG_H)
#include "dft_config.h"
#endif // HAVE_CONFIG_H

#include "CGNonLinearSolver.h"

#include "SolverFunction.h"

#if defined(HAVE_MPI)
#include <utils/exceptions/LengthError.h>
#include <utils/mpi/Accumulator.h>
#include <utils/mpi/Assembler.h>
#endif // HAVE_MPI

#if defined(HAVE_CMATH)
#include <cmath>
#else
#error cmath header file not available.
#endif // HAVE_CMATH

#if defined(HAVE_IOSTREAM)
#include <iostream>
#else
#error iostream header file not available.
#endif // HAVE_IOSTREAM

//
//
//
namespace dft {

  //
  // Constructor.
  //
  CGNonLinearSolver::CGNonLinearSolver(double tolerance,
                                       int    maxNumberIterations,
                                       int    debugLevel,
                                       double lineSearchTolerance,
				       int    lineSearchMaxIterations) :
    NonLinearSolver(tolerance,
                    maxNumberIterations,
                    debugLevel),
    d_lineSearchTolerance(lineSearchTolerance),
    d_lineSearchMaxIterations(lineSearchMaxIterations)
  {

    //
    //
    //
    return;

  }

  //
  // Destructor.
  //
  CGNonLinearSolver::~CGNonLinearSolver()
  {

    //
    //
    //
    return;

  }

  //
  // initialize direction
  //
  void
  CGNonLinearSolver::initializeDirection()
  {

    //
    // firewalls
    //
#if defined(HAVE_MPI)
    if(d_unknownCountFlag.empty() == true) {

      const std::string message("For MPI the dot product fuctor must be "
				"initialized.");
      throw LengthError(message);

    }
#endif // HAVE_MPI

    //
    // initialize delta new
    //
    d_deltaNew = 0.0;
	
    //
    // iterate over unknowns
    //
    for (int i = 0; i < d_numberUnknowns; ++i) {

#if defined(HAVE_MPI)
      const double factor = d_unknownCountFlag[i];
#else
      const double factor = 1.0;
#endif // HAVE_MPI

      const double r = -d_gradient[i];
      const double s = r;//d_s[i];

      d_sOld[i]        = s;
      d_direction[i]   = s;
      d_deltaNew      += factor*r*d_direction[i];

    }

#if defined(HAVE_MPI)
    //
    // accumulate d_deltaNew
    //
    // d_deltaNew = Utils::Accumulator().accumulate(d_deltaNew);

#endif // HAVE_MPI

    //
    //
    //
    return;

  }

  //
  // Compute delta_d and eta_p.
  //
  std::pair<double, double>
  CGNonLinearSolver::computeDeltaD()
  {

    //
    // firewalls
    //
#if defined(HAVE_MPI)
    if(d_unknownCountFlag.empty() == true) {

      const std::string message("For MPI the dot product fuctor must be "
				"initialized.");
      throw LengthError(message);

    }
#endif // HAVE_MPI

    //
    // initialize delta_d and eta_p
    //
    double deltaD = 0.0;
    double etaP   = 0.0;

    //
    // iterate over unknowns
    //
    for (int i = 0; i < d_numberUnknowns; ++i) {

#if defined(HAVE_MPI)
      const double factor = d_unknownCountFlag[i];
#else
      const double factor = 1.0;
#endif // HAVE_MPI

      const double direction = d_direction[i];

      deltaD += factor*direction*direction;
      etaP   += factor*d_gradient[i]*direction;

    }

#if defined(HAVE_MPI)
    //
    // instantiate Accumulator
    //
    Utils::Accumulator accumulator;

    //
    // accumulate deltaD
    //
    //deltaD = accumulator.accumulate(deltaD);

    //
    // accumulate etaP
    //
    //etaP = accumulator.accumulate(etaP);

#endif // HAVE_MPI

    //
    //
    //
    return std::make_pair(deltaD, etaP);

  }

  //
  // Compute eta.
  //
  double
  CGNonLinearSolver::computeEta()
  {

    //
    // firewalls
    //
#if defined(HAVE_MPI)
    if(d_unknownCountFlag.empty() == true) {

      const std::string message("For MPI the dot product fuctor must be "
				"initialized.");
      throw LengthError(message);

    }
#endif // HAVE_MPI
      
    //
    // initialize eta
    //
    double eta = 0.0;

    //
    // iterate over unknowns
    //
    for (int i = 0; i < d_numberUnknowns; ++i) {

#if defined(HAVE_MPI)
      const double factor = d_unknownCountFlag[i];
#else
      const double factor = 1.0;
#endif // HAVE_MPI
	
      const double direction = d_direction[i];
      eta += factor*d_gradient[i]*direction;

    }

#if defined(HAVE_MPI)
    //
    // accumulate eta
    //
    //eta = Utils::Accumulator().accumulate(eta);

#endif // HAVE_MPI

    //
    //
    //
    return eta;

  }

  //
  // Compute delta new and delta mid.
  //
  void
  CGNonLinearSolver::computeDeltas()
  {

    //
    // firewalls
    //
#if defined(HAVE_MPI)
    if(d_unknownCountFlag.empty() == true) {

      const std::string message("For MPI the dot product fuctor must be "
				"initialized.");
      throw LengthError(message);

    }
#endif // HAVE_MPI

    //
    // initialize delta new and delta mid.
    //
    d_deltaMid = 0.0;
    d_deltaNew = 0.0;

    //
    // iterate over unknowns
    //
    for (int i = 0; i < d_numberUnknowns; ++i) {

#if defined(HAVE_MPI)
      const double factor = d_unknownCountFlag[i];
#else
      const double factor = 1.0;
#endif // HAVE_MPI

      const double r    = -d_gradient[i];
      const double s    = r;//d_s[i];
      const double sOld = d_sOld[i];

      //
      // compute delta mid
      //
      d_deltaMid += factor*r*sOld;

      //
      // save gradient old
      //
      d_sOld[i] = s;

      //
      // compute delta new
      //
      d_deltaNew += factor*r*s;

    }

#if defined(HAVE_MPI)
    //
    // instantiate Accumulator
    //
    Utils::Accumulator accumulator;

    //
    // accumulate d_deltaMid
    //
    //d_deltaMid = accumulator.accumulate(d_deltaMid);

    //
    // accumulate d_deltaNew
    //
    //d_deltaNew = accumulator.accumulate(d_deltaNew);

#endif // HAVE_MPI

    //
    //
    //
    return;

  }

  //
  // Update direction.
  //
  void
  CGNonLinearSolver::updateDirection()
  {

    //
    // iterate over unknowns
    //
    for (int i = 0; i < d_numberUnknowns; ++i) {

      d_direction[i] *= d_beta;
      d_direction[i] += d_s[i];

    }

    //
    //
    //
    return;

  }

  //
  // Compute residual L2-norm.
  //
  double
  CGNonLinearSolver::computeResidualL2Norm() const
  {

    //
    // firewalls
    //
#if defined(HAVE_MPI)
    if(d_unknownCountFlag.empty() == true) {

      const std::string message("For MPI the dot product fuctor must be "
				"initialized.");
      throw LengthError(message);

    }
#endif // HAVE_MPI

    //
    // initialize norm
    //
    double norm = 0.0;

    //
    // iterate over unknowns
    //
    for (int i = 0; i < d_numberUnknowns; ++i) {
	
#if defined(HAVE_MPI)
      const double factor = d_unknownCountFlag[i];
#else
      const double factor = 1.0;
#endif // HAVE_MPI

      const double gradient = d_gradient[i];
      norm += factor*gradient*gradient;

    }

#if defined(HAVE_MPI)
    //
    // accumulate eta
    //
    //norm = Utils::Accumulator().accumulate(norm);

#endif // HAVE_MPI

    //
    // take square root
    // 
    norm = std::sqrt(norm);

    //
    //
    //
    return norm;

  }

  //
  // Compute the total number of unknowns in all processors.
  //
  int
  CGNonLinearSolver::computeTotalNumberUnknowns() const
  {

#if defined(HAVE_MPI)
    //
    // initialize total number of unknowns
    //
    int totalNumberUnknowns = 0;

    //
    // iterate over unknowns
    //
    for (int i = 0; i < d_numberUnknowns; ++i) {

      const int factor = d_unknownCountFlag[i];

      totalNumberUnknowns += factor;

    }

    //
    // accumulate totalnumberUnknowns
    //
    totalNumberUnknowns = static_cast<int>(Utils::Accumulator().accumulate(totalNumberUnknowns));

    //
    //
    //
    return totalNumberUnknowns;

#else
    return d_numberUnknowns;
#endif // HAVE_MPI

  }

  //
  // Update solution x -> x + \alpha direction.
  //
  void
  CGNonLinearSolver::updateSolution(double                      alpha,
				    const std::vector<double> & direction,
				    SolverFunction            & function)
  {

    //
    // get the solution from solver function
    //
    //function.solution(d_solution);

    //
    // get unknownCountFlag
    //
    //const std::vector<int> unknownCountFlag = function.getUnknownCountFlag();

    std::vector<double> displacements;
    
    //
    // get the size of solution
    //
    const std::vector<double>::size_type solutionSize = d_numberUnknowns;
    displacements.resize(d_numberUnknowns);

    //
    // get the size of solution
    //
    //const std::vector<double>::size_type solutionSize = d_solution.size();
    
    //
    // update solution
    //
    for (std::vector<double>::size_type i = 0; i < solutionSize; ++i)
      displacements[i] = alpha*direction[i];
      //d_solution[i] = (d_solution[i] + alpha*direction[i])*unknownCountFlag[i];
    
    //
    // store solution
    //
    //function.update(d_solution);
    function.update(displacements);

    //
    //
    //
    return;

  }

  //
  // Perform line search.
  //
  CGNonLinearSolver::ReturnValueType
  CGNonLinearSolver::lineSearch(SolverFunction & function,
				double           tolerance,
				int              maxNumberIterations,
				int              debugLevel)
  {
    //
    // local data
    //
    const double toleranceSqr = tolerance*tolerance;

    //
    // value of sigma0
    //
    const double sigma0 = 0.1;

    //
    // set the initial value of alpha
    //
    double alpha = -sigma0;

    //
    // update unknowns
    //
    CGNonLinearSolver::updateSolution(sigma0,
				      d_direction,
				      function);

    //
    // evaluate function gradient
    //
    function.gradient(d_gradient);

    //
    // compute delta_d and eta_p
    //
    std::pair<double, double> deltaDReturnValue = 
      CGNonLinearSolver::computeDeltaD();
    double deltaD = deltaDReturnValue.first;
    double etaP   = deltaDReturnValue.second;
     
    //
    // update unknowns removing earlier update
    //
    CGNonLinearSolver::updateSolution(-sigma0,
				      d_direction,
				      function);

    //
    // begin iteration
    //
    for (int iter = 0; iter < maxNumberIterations; ++iter) {

      //
      // evaluate function gradient
      //
      function.gradient(d_gradient);

      //
      // compute eta
      //
      const double eta = CGNonLinearSolver::computeEta();

      //
      // update alpha
      //
      alpha *= eta/(etaP - eta);

      //
      // update unknowns
      //
      CGNonLinearSolver::updateSolution(alpha,
					d_direction,
					function);

      //
      // update etaP
      //
      etaP = eta;

      //
      // output 
      //
      if (debugLevel >= 1)
	std::cout << "iteration: " << iter << " alpha: " << alpha 
		  << std::endl;

      //
      // check for convergence
      //
      if (alpha*alpha*deltaD < toleranceSqr)
	return SUCCESS;

    }

    //
    //
    //
    return MAX_ITER_REACHED;

  }



  //
  // Perform function minimization.
  //
  NonLinearSolver::ReturnValueType
  CGNonLinearSolver::solve(SolverFunction & function,
			   double           tolerance,
			   int              maxNumberIterations,
			   int              debugLevel)
  {

#if defined(HAVE_MPI)
    //
    // get MPIController
    //
    const Utils::MPIController & mpiController = 
      Utils::MPIControllerSingleton::getInstance();

    //
    // get root task id
    //
    const Utils::MPIController::mpi_task_id_type rootTaskId = 
      mpiController.getRootId();

    //
    // get task id
    //
    const Utils::MPIController::mpi_task_id_type taskId = 
      mpiController.getId();

    //
    // get dot product factor
    //
    //d_unknownCountFlag = function.getUnknownCountFlag();

#endif // HAVE_MPI

    //
    // method const data
    //
    const double toleranceSqr = tolerance*tolerance;

    //
    // get total number of unknowns in the problem.
    //
    d_numberUnknowns = function.getNumberUnknowns();

    //
    //resize d_unknownCountFlag with numberUnknown and initialize to 1
    //
    d_unknownCountFlag.resize(d_numberUnknowns,1);

    //
    // get total number of unknowns
    //
    const int totalnumberUnknowns = d_numberUnknowns;
      //CGNonLinearSolver::computeTotalNumberUnknowns();

    //
    // allocate space for direction, gradient and old gradient
    // values.
    //
    d_direction.resize(d_numberUnknowns);
    d_gradient.resize(d_numberUnknowns);
    d_sOld.resize(d_numberUnknowns);
    d_s.resize(d_numberUnknowns);

    //
    // compute initial values of function and function gradient
    //
    //double functionValue = function.value();
    function.gradient(d_gradient);

    //
    // apply preconditioner
    //
    //function.precondition(d_s,
    //			  d_gradient);
    for(int i = 0; i < d_s.size(); ++i)
      {
	d_s[i] = -d_gradient[i];
      }

    //
    // initialize delta new and direction
    //
    CGNonLinearSolver::initializeDirection();
      
    //
    // check for convergence
    //
    if ( d_deltaNew < toleranceSqr*totalnumberUnknowns*totalnumberUnknowns)
      return SUCCESS;

   


    for (d_iter = 0; d_iter < maxNumberIterations; ++d_iter) {

      if(taskId == rootTaskId)
	{
	  for(int i = 0; i < d_gradient.size(); ++i)
	    {
	      std::cout<<"d_gradient: "<<d_gradient[i]<<std::endl;
	    }
	}


      //
      // compute L2-norm of the residual (gradient)
      //
      const double residualNorm = computeResidualL2Norm();


      if (debugLevel >= 1)
      std::cout << "Iteration no. | delta new | residual norm " 
	"| residual norm avg" << std::endl;

      //
      // output at the begining of the iteration
      //
#if defined(HAVE_MPI)
      if (taskId == rootTaskId)
#endif // HAVE_MPI
	if (debugLevel >= 1) 
	  std::cout << d_iter << " " 
		    << d_deltaNew << " " 
		    << residualNorm << " " 
		    << residualNorm/totalnumberUnknowns << " "
		    << std::endl;
	
      //
      // perform line search along direction
      //
      ReturnValueType lineSearchReturnValue = 
	CGNonLinearSolver::lineSearch(function,
				      d_lineSearchTolerance,
				      d_lineSearchMaxIterations,
				      debugLevel);
					
      //
      // evaluate gradient
      //
      function.gradient(d_gradient);
	
      //
      // apply preconditioner
      //
      //function.precondition(d_s,
      //			    d_gradient);

      for(int i = 0; i < d_s.size(); ++i)
	{
	  d_s[i] = -d_gradient[i];
	}

      //
      // update values of delta_new and delta_mid
      //
      d_deltaOld = d_deltaNew;
      CGNonLinearSolver::computeDeltas();

      //
      // compute beta
      //
      //d_beta = (lineSearchReturnValue == SUCCESS) ? 
      //(d_deltaNew - d_deltaMid)/d_deltaOld : 0.0;

      d_beta = (d_deltaNew - d_deltaMid)/d_deltaOld;

      if(debugLevel >= 1)
	{
	  if(d_beta < 0)
	    {
	      std::cout<<"d_beta is negative: "<<std::endl;
	    }
	}

      if(d_beta <= 0)
	d_beta = 0;

      //
      // update direction
      //
      CGNonLinearSolver::updateDirection();

      //
      // check for convergence
      //
      if (d_deltaNew < toleranceSqr*totalnumberUnknowns*totalnumberUnknowns)
	break;

    }

    //
    // set error condition
    //
    ReturnValueType returnValue = SUCCESS;
      
    if(d_iter == maxNumberIterations) 
      returnValue = MAX_ITER_REACHED;

    //
    // compute function value
    //
    //functionValue = function.value();
      
    //
    // final output
    //
#if defined(HAVE_MPI)
    if (taskId == rootTaskId)
#endif // HAVE_MPI 
      if (debugLevel >= 1) {
	
	if (returnValue == SUCCESS) {
	  std::cout << "Conjugate Gradient converged after " 
		    << d_iter << " iterations." << std::endl;
	} else {
	  std::cout << "Conjugate Gradient failed to converge after "
		    << d_iter << " iterations." << std::endl;
	}

	//std::cout << "Final function value: " << functionValue
	//	  << std::endl;
	
      }

    //
    //
    //
    return SUCCESS;

  }

}
