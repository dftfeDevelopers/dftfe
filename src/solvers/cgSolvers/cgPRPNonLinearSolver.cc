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

#include <cgPRPNonLinearSolver.h>
#include <nonlinearSolverProblem.h>
#include <fileReaders.h>

namespace dftfe {

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
    nonLinearSolver(debugLevel,maxNumberIterations,tolerance),
    mpi_communicator (mpi_comm_replica),
    n_mpi_processes (dealii::Utilities::MPI::n_mpi_processes(mpi_comm_replica)),
    this_mpi_process (dealii::Utilities::MPI::this_mpi_process(mpi_comm_replica)),
    pcout(std::cout, (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0))
  {
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

      d_steepestDirectionOld[i]  =r;
      d_conjugateDirection[i]   = r;
      d_deltaNew      += factor*r*d_conjugateDirection[i];

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
      const double direction = d_conjugateDirection[i];

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
      const double direction = d_conjugateDirection[i];
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
      const double sOld = d_steepestDirectionOld[i];

      //
      // compute delta mid
      //
      d_deltaMid += factor*r*sOld;

      //
      // save gradient old
      //
      d_steepestDirectionOld[i] = r;

      //
      // compute delta new
      //
      d_deltaNew += factor*r*r;

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

      d_conjugateDirection[i] *= d_beta;
      d_conjugateDirection[i] += -d_gradient[i];

    }

  }

  //
  // save checkpoint files.
  //
  void
  cgPRPNonLinearSolver::save(const std::string & checkpointFileName)
  {

      std::vector<std::vector<double>> data;
      for (unsigned int i=0; i< d_conjugateDirection.size();++i)
        data.push_back(std::vector<double>(1,d_conjugateDirection[i]));

      dftUtils::writeDataIntoFile(data,
                                  checkpointFileName);
  }

  //
  // load from checkpoint files.
  //
  void
  cgPRPNonLinearSolver::load(const std::string & checkpointFileName)
  {

      std::vector<std::vector<double>> data;
      dftUtils::readFile(1,data,checkpointFileName);

      d_conjugateDirection.clear();
      for (unsigned int i=0; i< d_numberUnknowns;++i)
        d_conjugateDirection.push_back(data[i][0]);

      AssertThrow (d_conjugateDirection.size()== d_numberUnknowns,
	    dealii::ExcMessage (std::string("DFT-FE Error: data size of cg solver checkpoint file doesn't match with number of unknowns in the problem.")));
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
				       nonlinearSolverProblem            & problem)
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
    // call solver problem update
    //
    problem.update(incrementVector);

    //
    //
    //
    return;

  }

  //
  // Perform line search.
  //
  nonLinearSolver::ReturnValueType
  cgPRPNonLinearSolver::lineSearch(nonlinearSolverProblem &       problem,
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
    // evaluate problem gradient
    //
    problem.gradient(d_gradient);

    //
    // compute delta_d and eta_p
    //
    std::pair<double, double> deltaDReturnValue =
                 computeDeltaD();
    double deltaD = deltaDReturnValue.first;
    double etaP   = deltaDReturnValue.second;
    double alphaP=0;
    if (debugLevel >= 2)
       std::cout << "Initial guess for secant line search iteration, alpha: " << alpha << std::endl;
    //
    // update unknowns removing earlier update
    //
    updateSolution(alpha-alphaP,
		   d_conjugateDirection,
		   problem);
    //
    // begin iteration (using secant method)
    //
    for (unsigned int iter = 0; iter < maxNumberIterations; ++iter) {

      //
      // evaluate problem gradient
      //
      problem.gradient(d_gradient);


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
      if (debugLevel >= 2)
	std::cout << "Line search iteration: " << iter << " alphaNew: " << alphaNew << " alpha: "<<alpha<< " alphaP: "<<alphaP <<"  eta: "<< eta << " etaP: "<<etaP << std::endl;
      //
      // update unknowns
      //
      updateSolution(alphaNew-alpha,
		     d_conjugateDirection,
		     problem);

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
  // Perform problem minimization.
  //
  nonLinearSolver::ReturnValueType
  cgPRPNonLinearSolver::solve(nonlinearSolverProblem & problem,
	                      const std::string checkpointFileName,
			      const bool restart)
  {
    //
    // method const data
    //
    const double toleranceSqr = d_tolerance*d_tolerance;

    //
    // get total number of unknowns in the problem.
    //
    d_numberUnknowns = problem.getNumberUnknowns();

    //
    //resize d_unknownCountFlag with numberUnknown and initialize to 1
    //
    d_unknownCountFlag.resize(d_numberUnknowns,1);

    //
    // allocate space for conjugate direction, gradient and old steepest direction
    // values.
    //
    d_conjugateDirection.resize(d_numberUnknowns);
    d_gradient.resize(d_numberUnknowns);
    d_steepestDirectionOld.resize(d_numberUnknowns);

    //
    // compute initial values of problem and problem gradient
    //
    problem.gradient(d_gradient);

    //
    // initialize delta new and direction
    //
    if (!restart)
      initializeDirection();
    else
    {
      load(checkpointFileName);

      // compute deltaNew, and initialize steepestDirectionOld to current steepest direction
      d_deltaNew = 0.0;
      for (unsigned int i = 0; i < d_numberUnknowns; ++i)
      {
         const double r = -d_gradient[i];
         d_steepestDirectionOld[i]  =r;
         d_deltaNew += d_unknownCountFlag[i]*r*r;
      }
    }
    //
    // check for convergence
    //
    unsigned int isSuccess=0;
    if ( d_deltaNew < toleranceSqr*d_numberUnknowns)
        isSuccess=1;

    MPI_Bcast(&(isSuccess),
	       1,
	       MPI_INT,
	       0,
	       MPI_COMM_WORLD);
    if (isSuccess==1)
       return SUCCESS;



    for (d_iter = 0; d_iter < d_maxNumberIterations; ++d_iter) {

      if (d_debugLevel >= 2)
        for(unsigned int i = 0; i < d_gradient.size(); ++i)
	  pcout<<"d_gradient: "<<d_gradient[i]<<std::endl;


      //
      // compute L2-norm of the residual (gradient)
      //
      const double residualNorm = computeResidualL2Norm();


      if (d_debugLevel >= 2)
      std::cout << "Iteration no. | delta new | residual norm "
	"| residual norm avg" << std::endl;

      //
      // output at the begining of the iteration
      //
      if (d_debugLevel >= 2)
	  pcout << d_iter+1 << " "
		    << d_deltaNew << " "
		    << residualNorm << " "
		    << residualNorm/d_numberUnknowns << " "
		    << std::endl;

      //
      // perform line search along direction
      //
      ReturnValueType lineSearchReturnValue =
	                   lineSearch(problem,
				      d_lineSearchTolerance,
				      d_lineSearchMaxIterations,
				      d_debugLevel);

      //
      // evaluate gradient
      //
      problem.gradient(d_gradient);

      //
      // update values of delta_new and delta_mid
      //
      d_deltaOld = d_deltaNew;
      computeDeltas();

      //
      // compute PRP beta
      //

      d_beta = (d_deltaNew - d_deltaMid)/d_deltaOld;

      if (d_debugLevel >= 2)
         pcout<<" CG- d_beta: "<<d_beta<<std::endl;

      unsigned int isBetaZero=0;
      if(d_beta <= 0)
      {
	  if (d_debugLevel >= 2)
	     pcout<<" Negative d_beta- setting it to zero "<<std::endl;
	  isBetaZero=1;
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

      if (!checkpointFileName.empty())
      {
           save(checkpointFileName);
           problem.save();
      }

      //
      // check for convergence
      //
      unsigned int isBreak=0;
      if (d_deltaNew < toleranceSqr*d_numberUnknowns)
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
    // final output
    //
    if (d_debugLevel >= 1)
    {

      if (returnValue == SUCCESS)
      {
        pcout << "Non-linerar Conjugate Gradient solver converged after "
		<< d_iter+1 << " iterations." << std::endl;
      } else
      {
        pcout << "Non-linear Conjugate Gradient solver failed to converge after "
		<< d_iter << " iterations." << std::endl;
      }

    }

    //
    //
    //
    return returnValue;

  }
}
