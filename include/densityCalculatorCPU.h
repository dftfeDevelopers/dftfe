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

#ifndef densityCalculatorCPU_H_
#define densityCalculatorCPU_H_

#include "headers.h"
#include "operator.h"
#include "FEBasisOperations.h"
#include "dftParameters.h"

namespace dftfe
{
  /**
   * @brief Density calculator class using gemm recasting
   *
   * @author Sambit Das
   */

  template <typename T>
  void
  computeRhoFromPSICPU(
    const T *                               X,
    const T *                               XFrac,
    const unsigned int                      totalNumWaveFunctions,
    const unsigned int                      Nfr,
    const unsigned int                      numLocalDofs,
    const std::vector<std::vector<double>> &eigenValues,
    const double                            fermiEnergy,
    const double                            fermiEnergyUp,
    const double                            fermiEnergyDown,
    operatorDFTClass &                      operatorMatrix,
    std::unique_ptr<
      dftfe::basis::
        FEBasisOperations<T, double, dftfe::utils::MemorySpace::HOST>>
      &                                            basisOperationsPtrHost,
    const dealii::DoFHandler<3> &                  dofHandler,
    const unsigned int                             totalLocallyOwnedCells,
    const unsigned int                             numberNodesPerElement,
    const unsigned int                             numQuadPoints,
    const std::vector<double> &                    kPointWeights,
    std::map<dealii::CellId, std::vector<double>> *rhoValues,
    std::map<dealii::CellId, std::vector<double>> *gradRhoValues,
    std::map<dealii::CellId, std::vector<double>> *rhoValuesSpinPolarized,
    std::map<dealii::CellId, std::vector<double>> *gradRhoValuesSpinPolarized,
    const bool                                     isEvaluateGradRho,
    const MPI_Comm &                               mpiCommParent,
    const MPI_Comm &                               interpoolcomm,
    const MPI_Comm &                               interBandGroupComm,
    const dftParameters &                          dftParams,
    const bool                                     spectrumSplit,
    const bool                                     useFEOrderRhoPlusOneGLQuad);
} // namespace dftfe
#endif
