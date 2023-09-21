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

#if defined(DFTFE_WITH_DEVICE)
#  ifndef densityCalculatorDevice_H_
#    define densityCalculatorDevice_H_

#    include <headers.h>
#    include <operatorDevice.h>
#    include "dftParameters.h"
#    include "FEBasisOperations.h"

namespace dftfe
{
  namespace Device
  {
    template <typename NumberType>
    void
    computeRhoFromPSI(
      const NumberType *                      X,
      const NumberType *                      XFrac,
      const unsigned int                      totalNumWaveFunctions,
      const unsigned int                      Nfr,
      const unsigned int                      numLocalDofs,
      const std::vector<std::vector<double>> &eigenValues,
      const double                            fermiEnergy,
      const double                            fermiEnergyUp,
      const double                            fermiEnergyDown,
      operatorDFTDeviceClass &                operatorMatrix,
      std::unique_ptr<
        dftfe::basis::FEBasisOperations<NumberType,
                                        double,
                                        dftfe::utils::MemorySpace::DEVICE>>
        &                                            basisOperationsPtrDevice,
      const unsigned int                             matrixFreeDofhandlerIndex,
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
      const bool use2pPlusOneGLQuad = false);
  }
} // namespace dftfe
#  endif
#endif
