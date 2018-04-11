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

/**
 * @brief Concrete class implementing Chebyshev filtered orthogonalized subspace
 * iteration solver.
 */
namespace dftfe{

  class chebyshevOrthogonalizedSubspaceIterationSolver : public eigenSolverClass {
  
    //
    // types
    //
  public:

    //
    // methods
    //
  public:

    /**
     * @brief Constructor.
     *
     * @param lowerBoundWantedSpectrum Lower Bound of the Wanted Spectrum.
     * @param lowerBoundUnWantedSpectrum Lower Bound of the UnWanted Spectrum.
     * @param numberEigenvalues Number of smallest eigenvalues to be
     * solved for.
     * @param verbosityLevel Debug output level:
     *                   0 - very limited debug output
     *                   1 - limited debug output
     *                   2 - all debug output.
     */
    chebyshevOrthogonalizedSubspaceIterationSolver(double lowerBoundWantedSpectrum,
						   double lowerBoundUnWantedSpectrum,
						   const unsigned int numberEigenvalues);

    /**
     * @brief Destructor.
     */
    ~chebyshevOrthogonalizedSubspaceIterationSolver();


    /**
     * @brief Solve a generalized eigen problem. 
     */
    eigenSolverClass::ReturnValueType solve(operatorClass * operatorMatrix,
					    std::vector<vectorType> & eigenVectors,
					    std::vector<double> & eigenValues);

    /**
     * @brief reinit spectrum bounds
     */
    void reinitSpectrumBounds(double lowerBoundWantedSpectrum,
			      double lowerBoundUnWantedSpectrum);
  
    //
    //data
    //
  private:
    double d_lowerBoundWantedSpectrum;
    double d_lowerBoundUnWantedSpectrum;
    const unsigned int d_numberEigenValues;
    dealii::ConditionalOStream   pcout;
    TimerOutput computing_timer;
  };
}
#endif
