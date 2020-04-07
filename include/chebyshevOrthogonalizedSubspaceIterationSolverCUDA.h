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

#if defined(DFTFE_WITH_GPU)
#ifndef chebyshevOrthogonalizedSubspaceIterationSolverCUDA_h
#define chebyshevOrthogonalizedSubspaceIterationSolverCUDA_h


#include "headers.h"
#include "operatorCUDA.h"
#include "dftParameters.h"


namespace dftfe
{

  /**
   * @brief Concrete class implementing Chebyshev filtered orthogonalized subspace
   * iteration solver.
   * @author Phani Motamarri
   */
  class chebyshevOrthogonalizedSubspaceIterationSolverCUDA
  {

    public:
    /**
     * @brief Constructor.
     *
     * @param mpi_comm domain decomposition mpi communicator
     * @param lowerBoundWantedSpectrum Lower Bound of the Wanted Spectrum.
     * @param lowerBoundUnWantedSpectrum Lower Bound of the UnWanted Spectrum.
     */
    chebyshevOrthogonalizedSubspaceIterationSolverCUDA
                                          (const MPI_Comm &mpi_comm,
	                                   double lowerBoundWantedSpectrum,
				           double lowerBoundUnWantedSpectrum);


    /**
     * @brief Destructor.
     */
    ~chebyshevOrthogonalizedSubspaceIterationSolverCUDA();


    /**
     * @brief Solve a generalized eigen problem.
     */
    void solve(operatorDFTCUDAClass & operatorMatrix,
	       double* eigenVectorsFlattenedCUDA,
               double* eigenVectorsRotFracDensityFlattenedCUDA,
               const unsigned int flattenedSize,
	       vectorType & tempEigenVec,
	       const unsigned int totalNumberWaveFunctions,
	       std::vector<double> & eigenValues,
	       std::vector<double> & residuals,
               const MPI_Comm &interBandGroupComm,
               dealii::ScaLAPACKMatrix<double> & projHamPar,
               dealii::ScaLAPACKMatrix<double> & overlapMatPar,
               const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid> & processGrid,
               const bool isXlBOMDLinearizedSolve,
               const bool useCommunAvoidanceCheby,
               const bool useMixedPrecOverall=false,
               const bool isFirstScf=false,
               const bool useFullMassMatrixGEP=false,
               const bool isElpaStep1=false,
               const bool isElpaStep2=false);

    /**
     * @brief Solve a generalized eigen problem.
     */
    void solveNoRR(operatorDFTCUDAClass & operatorMatrix,
	       double* eigenVectorsFlattenedCUDA,
               const unsigned int flattenedSize,
	       vectorType & tempEigenVec,
	       const unsigned int totalNumberWaveFunctions,
	       std::vector<double> & eigenValues,
               const MPI_Comm &interBandGroupComm,
               dealii::ScaLAPACKMatrix<double> & projHamPar,
               dealii::ScaLAPACKMatrix<double> & overlapMatPar,
               const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid> & processGrid,
               const bool isXlBOMDLinearizedSolve,
               const bool useCommunAvoidanceCheby,
               const bool isFirstPass,
               const bool useMixedPrecOverall);

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
#endif
