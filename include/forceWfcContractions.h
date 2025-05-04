// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
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

#ifndef forceWfcContractions_H_
#define forceWfcContractions_H_

#include "headers.h"
#include "dftParameters.h"
#include "FEBasisOperations.h"
#include "oncvClass.h"
#include <memory>
#include <BLASWrapper.h>
#include "hubbardClass.h"

namespace dftfe
{
  namespace force
  {
    template <dftfe::utils::MemorySpace memorySpace>
    void
    wfcContractionsForceKernelsAllH(
      std::shared_ptr<
        dftfe::basis::FEBasisOperations<dataTypes::number, double, memorySpace>>
                       &basisOperationsPtr,
      const dftfe::uInt densityQuadratureId,
      const dftfe::uInt nlpspQuadratureId,
      const std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
        &BLASWrapperPtr,
      std::shared_ptr<dftfe::oncvClass<dataTypes::number, memorySpace>>
                                                               oncvClassPtr,
      std::shared_ptr<hubbard<dataTypes::number, memorySpace>> hubbardClassPtr,
      const bool                                               useHubbard,
      const dataTypes::number                                 *X,
      const dftfe::uInt                       spinPolarizedFlag,
      const dftfe::uInt                       spinIndex,
      const std::vector<std::vector<double>> &eigenValuesH,
      const std::vector<std::vector<double>> &partialOccupanciesH,
      const std::vector<double>              &kPointCoordinates,
      const dftfe::uInt                       MLoc,
      const dftfe::uInt                       N,
      const dftfe::uInt                       numCells,
      const dftfe::uInt                       numQuads,
      const dftfe::uInt                       numQuadsNLP,
      double                                 *eshelbyTensorQuadValuesH,
      dataTypes::number *
        projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedH,
      dataTypes::number *
        projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedHHubbard,
#ifdef USE_COMPLEX
      dataTypes::number
        *projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedH,
      dataTypes::number *
        projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedHHubbard,
#endif
      const MPI_Comm      &mpiCommParent,
      const MPI_Comm      &interBandGroupComm,
      const bool           isPsp,
      const bool           isFloatingChargeForces,
      const bool           addEk,
      const dftParameters &dftParams);
  } // namespace force
} // namespace dftfe
#endif
