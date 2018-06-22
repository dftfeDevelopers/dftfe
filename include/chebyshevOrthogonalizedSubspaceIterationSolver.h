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
// @author Phani Motamarri (2018)

#ifndef chebyshevOrthogonalizedSubspaceIterationSolver_h
#define chebyshevOrthogonalizedSubspaceIterationSolver_h

#include "eigenSolver.h"
#include "dftParameters.h"
#include "eigen.h"


namespace dftfe{

  /**
   * @brief Concrete class implementing Chebyshev filtered orthogonalized subspace
   * iteration solver.
   */

  class chebyshevOrthogonalizedSubspaceIterationSolver : public eigenSolverClass {

  public:
    /**
     * @brief Constructor.
     *
     * @param lowerBoundWantedSpectrum Lower Bound of the Wanted Spectrum.
     * @param lowerBoundUnWantedSpectrum Lower Bound of the UnWanted Spectrum.
     */
    chebyshevOrthogonalizedSubspaceIterationSolver(double lowerBoundWantedSpectrum,
						   double lowerBoundUnWantedSpectrum);


    /**
     * @brief Destructor.
     */
    ~chebyshevOrthogonalizedSubspaceIterationSolver();


    /**
     * @brief Solve a generalized eigen problem.
     */
    eigenSolverClass::ReturnValueType solve(operatorDFTClass & operatorMatrix,
	                                    dealii::parallel::distributed::Vector<dataTypes::number> & eigenVectorsFlattened,
					    vectorType & tempEigenVec,
					    const unsigned int totalNumberWaveFunctions,
					    std::vector<double> & eigenValues,
					    std::vector<double> & residuals,
					    const MPI_Comm &interBandGroupComm);

    /**
     * @brief Solve a generalized eigen problem.
     */
    eigenSolverClass::ReturnValueType solve(operatorDFTClass & operatorMatrix,
	                                    std::vector<vectorType> & eigenVectors,
					    std::vector<double> & eigenValues,
					    std::vector<double> & residuals);

    /**
     * @brief reinit spectrum bounds
     */
    void reinitSpectrumBounds(double lowerBoundWantedSpectrum,
			      double lowerBoundUnWantedSpectrum);


  private:
    //
    //stores lower bound of wanted spectrum
    //
    double d_lowerBoundWantedSpectrum;

    //
    //stores lower bound of unwanted spectrum
    //
    double d_lowerBoundUnWantedSpectrum;

    //
    //variables for printing out and timing
    //
    dealii::ConditionalOStream   pcout;
    dealii::TimerOutput computing_timer;
  };
}
#endif
