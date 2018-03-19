// ---------------------------------------------------------------------
//
// Copyright (c) 2017 The Regents of the University of Michigan and DFT-FE authors.
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
// @author Sambit Das (2018)

#include <cgPRPNonLinearSolver.h>
#include <solverFunction.h>


  //
  // Constructor.
  //
  cgPRPNonLinearSolver::cgPRPNonLinearSolver(const double tolerance,
                                             const unsigned int    maxNumberIterations,
                                             const unsigned int    debugLevel,
					     const MPI_Comm &mpi_comm_replica,
                                             const double lineSearchTolerance,
				             const unsigned int    lineSearchMaxIterations,
					     const double lineSearchDampingParameter) :
    d_lineSearchTolerance(lineSearchTolerance),
    d_lineSearchMaxIterations(lineSearchMaxIterations),
    d_lineSearchDampingParameter(lineSearchDampingParameter),
    mpi_communicator (mpi_comm_replica),
    n_mpi_processes (dealii::Utilities::MPI::n_mpi_processes(mpi_communicator)),
    this_mpi_process (dealii::Utilities::MPI::this_mpi_process(mpi_communicator)),
    pcout(std::cout, (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0))    
  {
    d_debugLevel=debugLevel;
    d_maxNumberIterations=maxNumberIterations;
    d_tolerance=tolerance;

  }

  //
  // Destructor.
  //
  cgPRPNonLinearSolver::~cgPRPNonLinearSolver()
  {

    //
    //
    //
    return;

  }

  //
  // reinit cg solve parameters
  //
  void
  cgPRPNonLinearSolver::reinit(const double tolerance,
                               const unsigned int    maxNumberIterations,
                               const unsigned int    debugLevel,
                               const double lineSearchTolerance,
			       const unsigned int    lineSearchMaxIterations,
			       const double lineSearchDampingParameter)
  {
    d_debugLevel=debugLevel;
    d_maxNumberIterations=maxNumberIterations;
    d_tolerance=tolerance; 
    d_lineSearchTolerance=lineSearchTolerance;
    d_lineSearchMaxIterations=lineSearchMaxIterations;
    d_lineSearchDampingParameter=lineSearchDampingParameter;
  }
  //
  // initialize direction
  //
  void
  cgPRPNonLinearSolver::initializeDirection()
  {

    //
    // initialize delta new
    //
    d_deltaNew = 0.0;
	
    //
    // iterate over unknowns
    //
    for (unsigned int i = 0; i < d_numberUnknowns; ++i) {

      const double factor = d_unknownCountFlag[i];
      const double r = -d_gradient[i];
      const double s = r;//d_s[i];

      d_sOld[i]        = s;
      d_direction[i]   = s;
      d_deltaNew      += factor*r*d_direction[i];

    }

    //
    //
    return;

  }

  //
  // Compute delta_d and eta_p.
  //
  std::pair<double, double>
  cgPRPNonLinearSolver::computeDeltaD()
  {
    //
    // initialize delta_d and eta_p
    //
    double deltaD = 0.0;
    double etaP   = 0.0;

    //
    // iterate over unknowns
    //
    for (unsigned int i = 0; i < d_numberUnknowns; ++i) {

      const double factor = d_unknownCountFlag[i];
      const double direction = d_direction[i];

      deltaD += factor*direction*direction;
      etaP   += factor*d_gradient[i]*direction;

    }

    //
    //
    //
    return std::make_pair(deltaD, etaP);

  }

  //
  // Compute eta.
  //
  double
  cgPRPNonLinearSolver::computeEta()
  {

      
    //
    // initialize eta
    //
    double eta = 0.0;

    //
    // iterate over unknowns
    //
    for (unsigned int i = 0; i < d_numberUnknowns; ++i) {

      const double factor = d_unknownCountFlag[i];
      const double direction = d_direction[i];
      eta += factor*d_gradient[i]*direction;

    }

    //
    //
    return eta;

  }

  //
  // Compute delta new and delta mid.
  //
  void
  cgPRPNonLinearSolver::computeDeltas()
  {

    //
    // initialize delta new and delta mid.
    //
    d_deltaMid = 0.0;
    d_deltaNew = 0.0;

    //
    // iterate over unknowns
    //
    for (unsigned int i = 0; i < d_numberUnknowns; ++i) {

      const double factor = d_unknownCountFlag[i];
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
    //
    //
    //
    return;

  }

  //
  // Update direction.
  //
  void
  cgPRPNonLinearSolver::updateDirection()
  {

    //
    // iterate over unknowns
    //
    for (unsigned int i = 0; i < d_numberUnknowns; ++i) {

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
  cgPRPNonLinearSolver::computeResidualL2Norm() const
  {
    // initialize norm
    //
    double norm = 0.0;

    //
    // iterate over unknowns
    //
    for (unsigned int i = 0; i < d_numberUnknowns; ++i) {
      const double factor = d_unknownCountFlag[i];
      const double gradient = d_gradient[i];
      norm += factor*gradient*gradient;
    }


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
  unsigned int
  cgPRPNonLinearSolver::computeTotalNumberUnknowns() const
  {

    //
    // initialize total number of unknowns
    //
    unsigned int totalNumberUnknowns = 0;

    //
    // iterate over unknowns
    //
    for (unsigned int i = 0; i < d_numberUnknowns; ++i) {

      const unsigned int factor = d_unknownCountFlag[i];

      totalNumberUnknowns += factor;

    }

    //
    // accumulate totalnumberUnknowns
    //
    totalNumberUnknowns = dealii::Utilities::MPI::sum(totalNumberUnknowns, mpi_communicator);;

    //
    //
    //
    return totalNumberUnknowns;

  }

  //
  // Update solution x -> x + \alpha direction.
  //
  void
  cgPRPNonLinearSolver::updateSolution(const double                      alpha,
				       const std::vector<double> & direction,
				       solverFunction            & function)
  {


    std::vector<double> incrementVector;
    
    //
    // get the size of solution
    //
    const std::vector<double>::size_type solutionSize = d_numberUnknowns;
    incrementVector.resize(d_numberUnknowns);


    for (std::vector<double>::size_type i = 0; i < solutionSize; ++i)
      incrementVector[i] = alpha*direction[i];
    
    //
    // call solver function update
    //
    function.update(incrementVector);

    //
    //
    //
    return;

  }

  //
  // Perform line search.
  //
  nonLinearSolver::ReturnValueType
  cgPRPNonLinearSolver::lineSearch(solverFunction &       function,
				   const double           tolerance,
				   const unsigned int     maxNumberIterations,
				   const unsigned int     debugLevel)
  {
    //
    // local data
    //
    const double toleranceSqr = tolerance*tolerance;

 
    //
    // set the initial value of alpha
    //
    double alpha = d_lineSearchDampingParameter;

    //
    // evaluate function gradient
    //
    function.gradient(d_gradient);

    //
    // compute delta_d and eta_p
    //
    std::pair<double, double> deltaDReturnValue = 
                 computeDeltaD();
    double deltaD = deltaDReturnValue.first;
    double etaP   = deltaDReturnValue.second;
    double alphaP=0;
    if (debugLevel >= 1)
       std::cout << "Initial guess for secant line search iteration, alpha: " << alpha << std::endl;    
    //
    // update unknowns removing earlier update
    //
    updateSolution(alpha-alphaP,
		   d_direction,
		   function);
    //
    // begin iteration (using secant method)
    //
    for (unsigned int iter = 0; iter < maxNumberIterations; ++iter) {

      //
      // evaluate function gradient
      //
      function.gradient(d_gradient);


      //
      // compute eta
      //
      const double eta = computeEta();

      unsigned int isSuccess=0;
      if (std::fabs(eta) < toleranceSqr*d_numberUnknowns)
         isSuccess=1;
      MPI_Bcast(&(isSuccess),
		1,
		MPI_INT,
		0,
		MPI_COMM_WORLD); 
      if (isSuccess==1)	  
	return SUCCESS;      
      //
      // update alpha
      //
      double alphaNew=(alphaP*eta-alpha*etaP)/(eta-etaP);

      

      //
      // output 
      //
      if (debugLevel >= 1)
	std::cout << "Line search iteration: " << iter << " alphaNew: " << alphaNew << " alpha: "<<alpha<< " alphaP: "<<alphaP <<"  eta: "<< eta << " etaP: "<<etaP << std::endl;      
      //
      // update unknowns
      //
      updateSolution(alphaNew-alpha,
		     d_direction,
		     function);

      //
      // update etaP, alphaP and alpha
      //
      etaP = eta;
      alphaP=alpha;
      alpha=alphaNew;
    }

    //
    //
    //
    return MAX_ITER_REACHED;

  }



  //
  // Perform function minimization.
  //
  nonLinearSolver::ReturnValueType
  cgPRPNonLinearSolver::solve(solverFunction & function)
  {

    //
    // method const data
    //
    const double toleranceSqr = d_tolerance*d_tolerance;

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
    initializeDirection();
      
    //
    // check for convergence
    //
    unsigned int isSuccess=0;    
    if ( d_deltaNew < toleranceSqr*totalnumberUnknowns)
        isSuccess=1;

    MPI_Bcast(&(isSuccess),
	       1,
	       MPI_INT,
	       0,
	       MPI_COMM_WORLD); 
    if (isSuccess==1)	  
       return SUCCESS;  
   


    for (d_iter = 0; d_iter < d_maxNumberIterations; ++d_iter) {

      for(unsigned int i = 0; i < d_gradient.size(); ++i)
      {
	  pcout<<"d_gradient: "<<d_gradient[i]<<std::endl;
      }


      //
      // compute L2-norm of the residual (gradient)
      //
      const double residualNorm = computeResidualL2Norm();


      if (d_debugLevel >= 1)
      std::cout << "Iteration no. | delta new | residual norm " 
	"| residual norm avg" << std::endl;

      //
      // output at the begining of the iteration
      //
      if (d_debugLevel >= 1) 
	  pcout << d_iter << " " 
		    << d_deltaNew << " " 
		    << residualNorm << " " 
		    << residualNorm/totalnumberUnknowns << " "
		    << std::endl;
	
      //
      // perform line search along direction
      //
      ReturnValueType lineSearchReturnValue = 
	                   lineSearch(function,
				      d_lineSearchTolerance,
				      d_lineSearchMaxIterations,
				      d_debugLevel);
				
      //write mesh
      std::string meshFileName="mesh_geo";
      meshFileName+=std::to_string(d_iter);
      //function.writeMesh(meshFileName);      
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
      computeDeltas();

      //
      // compute PRP beta
      //

      d_beta = (d_deltaNew - d_deltaMid)/d_deltaOld;

      pcout<<" CG- d_beta: "<<d_beta<<std::endl;

      unsigned int isBetaZero=0;
      if(d_beta <= 0)
      {
	  pcout<<" Negative d_beta- setting it to zero "<<std::endl;
	  isBetaZero=1;
	  //d_beta=0;
	  //return RESTART;	  
      }
      MPI_Bcast(&(isBetaZero),
		   1,
		   MPI_INT,
		   0,
		   MPI_COMM_WORLD); 
      if (isBetaZero==1)	  
	  d_beta=0; 
      //
      // update direction
      //
      updateDirection();

      //
      // check for convergence
      //
      unsigned int isBreak=0;
      if (d_deltaNew < toleranceSqr*totalnumberUnknowns)
	isBreak=1;
      MPI_Bcast(&(isSuccess),
		   1,
		   MPI_INT,
		   0,
		   MPI_COMM_WORLD); 
      if (isBreak==1)	  
	  break; 
    }

    //
    // set error condition
    //
    ReturnValueType returnValue = SUCCESS;
      
    if(d_iter == d_maxNumberIterations) 
      returnValue = MAX_ITER_REACHED;

    //
    // compute function value
    //
    //functionValue = function.value();
      
    //
    // final output
    //
    if (d_debugLevel >= 1)
    {
    
      if (returnValue == SUCCESS)
      {
        pcout << "Conjugate Gradient converged after " 
		<< d_iter << " iterations." << std::endl;
      } else
      {
        pcout << "Conjugate Gradient failed to converge after "
		<< d_iter << " iterations." << std::endl;
      }

      //std::cout << "Final function value: " << functionValue
      //	  << std::endl;
    
    }

    //
    //
    //
    return returnValue;

  }
