// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022 The Regents of the University of Michigan and DFT-FE
// authors.
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


#ifndef chebyshevOrthogonalizedSubspaceIterationSolver_h
#define chebyshevOrthogonalizedSubspaceIterationSolver_h

#include "dftParameters.h"
#include "eigenSolver.h"
#include "operator.h"
#include "elpaScalaManager.h"


namespace dftfe
{
  /**
   * @brief Concrete class implementing Chebyshev filtered orthogonalized subspace
   * iteration solver.
   * @author Phani Motamarri, Sambit Das
   */

  class chebyshevOrthogonalizedSubspaceIterationSolver : public eigenSolverClass
  {
  public:
    /**
     * @brief Constructor.
     *
     * @param mpi_comm domain decomposition mpi communicator
     * @param lowerBoundWantedSpectrum Lower Bound of the Wanted Spectrum.
     * @param lowerBoundUnWantedSpectrum Lower Bound of the UnWanted Spectrum.
     */
    chebyshevOrthogonalizedSubspaceIterationSolver(
      const MPI_Comm &mpi_comm,
      double          lowerBoundWantedSpectrum,
      double          lowerBoundUnWantedSpectrum,
      double          upperBoundUnWantedSpectrum);


    /**
     * @brief Destructor.
     */
    ~chebyshevOrthogonalizedSubspaceIterationSolver();


    /**
     * @brief Solve a generalized eigen problem.
     */
    void
    solve(operatorDFTClass &              operatorMatrix,
          elpaScalaManager &              elpaScala,
          std::vector<dataTypes::number> &eigenVectorsFlattened,
          std::vector<dataTypes::number> &eigenVectorsRotFracDensityFlattened,
          const unsigned int              totalNumberWaveFunctions,
          std::vector<double> &           eigenValues,
          std::vector<double> &           residuals,
          const MPI_Comm &                interBandGroupComm,
          const bool                      computeResidual,
          const bool                      useMixedPrec = false,
          const bool                      isFirstScf   = false);

    /**
     * @brief Solve a generalized eigen problem.
     */
    void
    solve(operatorDFTClass &                      operatorMatrix,
          std::vector<distributedCPUVec<double>> &eigenVectors,
          std::vector<double> &                   eigenValues,
          std::vector<double> &                   residuals);

    /**
     * @brief reinit spectrum bounds
     */
    void
    reinitSpectrumBounds(double lowerBoundWantedSpectrum,
                         double lowerBoundUnWantedSpectrum,
                         double upperBoundUnWantedSpectrum);

  private:
    //
    // stores lower bound of wanted spectrum
    //
    double d_lowerBoundWantedSpectrum;

    //
    // stores lower bound of unwanted spectrum
    //
    double d_lowerBoundUnWantedSpectrum;


    //
    // stores upper bound of unwanted spectrum
    //
    double d_upperBoundUnWantedSpectrum;

    //
    // variables for printing out and timing
    //
    dealii::ConditionalOStream pcout;
    dealii::TimerOutput        computing_timer;
  };
} // namespace dftfe
#endif
