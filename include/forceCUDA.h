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

#if defined(DFTFE_WITH_GPU)
#  ifndef forceCUDA_H_
#    define forceCUDA_H_

#    include "headers.h"
#    include "operatorCUDA.h"

namespace dftfe
{
  namespace forceCUDA
  {
    void
    gpuPortedForceKernelsAllH(
      operatorDFTCUDAClass &      operatorMatrix,
      const dataTypes::numberGPU *X,
      const double *              eigenValuesH,
      const double *              partialOccupanciesH,
#    ifdef USE_COMPLEX
      const double kcoordx,
      const double kcoordy,
      const double kcoordz,
#    endif
      const unsigned int *nonTrivialIdToElemIdMapH,
      const unsigned int *projecterKetTimesFlattenedVectorLocalIdsH,
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
      const MPI_Comm &interBandGroupComm,
      const bool      isPsp,
      const bool      isFloatingChargeForces,
      const bool      addEk);
  } // namespace forceCUDA
} // namespace dftfe
#  endif
#endif
