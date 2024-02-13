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
#  ifndef forceWfcContractionsDevice_H_
#    define forceWfcContractionsDevice_H_

#    include "headers.h"
#    include "operatorDevice.h"
#    include "dftParameters.h"
#    include "FEBasisOperations.h"
#    include <BLASWrapper.h>

namespace dftfe
{
  namespace forceDevice
  {
    void
    wfcContractionsForceKernelsAllH(
      std::shared_ptr<
        dftfe::basis::FEBasisOperations<dataTypes::number,
                                        double,
                                        dftfe::utils::MemorySpace::DEVICE>>
        &basisOperationsPtr,
      const std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
        &                                     BLASWrapperPtr,
      operatorDFTDeviceClass &                operatorMatrix,
      const dataTypes::number *               X,
      const unsigned int                      spinPolarizedFlag,
      const unsigned int                      spinIndex,
      const std::vector<std::vector<double>> &eigenValuesH,
      const std::vector<std::vector<double>> &partialOccupanciesH,
      const std::vector<double> &             kPointCoordinates,
      const unsigned int *                    nonTrivialIdToElemIdMapH,
      const unsigned int *projecterKetTimesFlattenedVectorLocalIdsH,
      const unsigned int  MLoc,
      const unsigned int  N,
      const unsigned int  numCells,
      const unsigned int  numQuads,
      const unsigned int  numQuadsNLP,
      const unsigned int  numNodesPerElement,
      const unsigned int  totalNonTrivialPseudoWfcs,
      double *            eshelbyTensorQuadValuesH,
      dataTypes::number *
        projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedH,
#    ifdef USE_COMPLEX
      dataTypes::number
        *projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedH,
#    endif
      const MPI_Comm &     mpiCommParent,
      const MPI_Comm &     interBandGroupComm,
      const bool           isPsp,
      const bool           isFloatingChargeForces,
      const bool           addEk,
      const dftParameters &dftParams);
  } // namespace forceDevice
} // namespace dftfe
#  endif
#endif
