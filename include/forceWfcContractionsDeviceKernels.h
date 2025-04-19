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

#if defined(DFTFE_WITH_DEVICE)
#  ifndef forceWfcContractionsDeviceKernels_H_
#    define forceWfcContractionsDeviceKernels_H_

namespace dftfe
{
  namespace forceDeviceKernels
  {
    template <typename ValueType>
    void
    nlpContractionContributionPsiIndex(
      const dftfe::uInt  wfcBlockSize,
      const dftfe::uInt  blockSizeNlp,
      const dftfe::uInt  numQuadsNLP,
      const dftfe::uInt  startingIdNlp,
      const ValueType   *projectorKetTimesVectorPar,
      const ValueType   *gradPsiOrPsiQuadValuesNLP,
      const double      *partialOccupancies,
      const dftfe::uInt *nonTrivialIdToElemIdMap,
      const dftfe::uInt *projecterKetTimesFlattenedVectorLocalIds,
      ValueType         *nlpContractionContribution);

    template <typename ValueType>
    void
    computeELocWfcEshelbyTensorContributions(const dftfe::uInt wfcBlockSize,
                                             const dftfe::uInt cellsBlockSize,
                                             const dftfe::uInt numQuads,
                                             const ValueType  *psiQuadValues,
                                             const ValueType *gradPsiQuadValues,
                                             const double    *eigenValues,
                                             const double *partialOccupancies,
#    ifdef USE_COMPLEX
                                             const double kcoordx,
                                             const double kcoordy,
                                             const double kcoordz,
#    endif
                                             double *eshelbyTensorContributions
#    ifdef USE_COMPLEX
                                             ,
                                             const bool addEk
#    endif
    );


  } // namespace forceDeviceKernels
} // namespace dftfe
#  endif
#endif
