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

#include "../../include/cgPRPNonLinearSolver.h"
#include "../../include/solverFunction.h"


  //
  // Constructor.
  //
  cgPRPNonLinearSolver::cgPRPNonLinearSolver(double tolerance,
                                            int    maxNumberIterations,
                                            int    debugLevel,
					    MPI_Comm &mpi_comm_replica,
                                            double lineSearchTolerance,
				            int    lineSearchMaxIterations) :
    d_lineSearchTolerance(lineSearchTolerance),
    d_lineSearchMaxIterations(lineSearchMaxIterations),
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
  cgPRPNonLinearSolver::reinit(double tolerance,
                               int    maxNumberIterations,
                               int    debugLevel,
                               double lineSearchTolerance,
			       int    lineSearchMaxIterations)
  {
    d_debugLevel=debugLevel;
    d_maxNumberIterations=maxNumberIterations;
    d_tolerance=tolerance; 
    d_lineSearchTolerance=lineSearchTolerance;
    d_lineSearchMaxIterations=lineSearchMaxIterations;
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
    for (int i = 0; i < d_numberUnknowns; ++i) {

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
    for (int i = 0; i < d_numberUnknowns; ++i) {

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
    for (int i = 0; i < d_numberUnknowns; ++i) {

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
    for (int i = 0; i < d_numberUnknowns; ++i) {

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
  cgPRPNonLinearSolver::computeResidualL2Norm() const
  {
    // initialize norm
    //
    double norm = 0.0;

    //
    // iterate over unknowns
    //
    for (int i = 0; i < d_numberUnknowns; ++i) {
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
  int
  cgPRPNonLinearSolver::computeTotalNumberUnknowns() const
  {

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
  cgPRPNonLinearSolver::updateSolution(double                      alpha,
				    const std::vector<double> & direction,
				    solverFunction            & function)
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
  nonLinearSolver::ReturnValueType
  cgPRPNonLinearSolver::lineSearch(solverFunction & function,
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
    const double sigma0 = 0.7;//0.1;

    //
    // set the initial value of alpha
    //
    double alpha = sigma0;//-sigma0;

    //
    // update unknowns
    //
    //updateSolution(sigma0,
    //	           d_direction,
    //		   function);

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
    updateSolution(alpha-alphaP,//-sigma0,
		   d_direction,
		   function);
    //
    // begin iteration (using secant method)
    //
    for (int iter = 0; iter < maxNumberIterations; ++iter) {

      //
      // evaluate function gradient
      //
      function.gradient(d_gradient);


      //
      // compute eta
      //
      const double eta = computeEta();

      if (std::fabs(eta) < tolerance*d_numberUnknowns)
	return SUCCESS;      
      //
      // update alpha
      //
      //alpha *= eta/(etaP - eta);
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

      //
      // check for convergence
      //
      //if (alpha*alpha*deltaD < toleranceSqr)
      //   return SUCCESS;

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
      //cgPRPNonLinearSolver::computeTotalNumberUnknowns();

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
    initializeDirection();
      
    //
    // check for convergence
    //
    if ( d_deltaNew < toleranceSqr*totalnumberUnknowns*totalnumberUnknowns)
      return SUCCESS;

   


    for (d_iter = 0; d_iter < d_maxNumberIterations; ++d_iter) {

      if(this_mpi_process ==0)
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


      if (d_debugLevel >= 1)
      std::cout << "Iteration no. | delta new | residual norm " 
	"| residual norm avg" << std::endl;

      //
      // output at the begining of the iteration
      //
      if (this_mpi_process == 0)
	if (d_debugLevel >= 1) 
	  std::cout << d_iter << " " 
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
      function.writeMesh(meshFileName);      
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
      //d_beta = (lineSearchReturnValue == SUCCESS) ? 
      //(d_deltaNew - d_deltaMid)/d_deltaOld : 0.0;

      d_beta = (d_deltaNew - d_deltaMid)/d_deltaOld;

     if (this_mpi_process == 0)
	   std::cout<<" CG- d_beta: "<<d_beta<<std::endl;
      if(d_beta <= 0)
      {
	  if (this_mpi_process == 0)
	     std::cout<<" Negative d_beta- setting it to zero "<<std::endl;
	  d_beta=0;
	  //return RESTART;	  
      }

      //
      // update direction
      //
      updateDirection();

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
      
    if(d_iter == d_maxNumberIterations) 
      returnValue = MAX_ITER_REACHED;

    //
    // compute function value
    //
    //functionValue = function.value();
      
    //
    // final output
    //
    if (this_mpi_process == 0)
      if (d_debugLevel >= 1) {
	
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
