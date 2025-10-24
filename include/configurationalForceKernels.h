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

#ifndef configurationalForceKernels_H_
#define configurationalForceKernels_H_
#if defined(DFTFE_WITH_DEVICE)

#  include <BLASWrapper.h>
#  include <DataTypeOverloads.h>
#  include <DeviceAPICalls.h>
#  include <DeviceDataTypeOverloads.h>
#  include <DeviceTypeConfig.h>
#  include <DeviceKernelLauncherHelpers.h>
#  include <memory>
#  include <MemoryStorage.h>
namespace dftfe
{
  void
  computeWavefuncEshelbyContributionLocal(
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
                                             &BLASWrapperPtr,
    const std::pair<dftfe::uInt, dftfe::uInt> cellRange,
    const std::pair<dftfe::uInt, dftfe::uInt> vecRange,
    const dftfe::uInt                         nQuadsPerCell,
    const double                              kcoordx,
    const double                              kcoordy,
    const double                              kcoordz,
    double                                   *partialOccupVec,
    double                                   *eigenValuesVec,
    dataTypes::number                        *wfcQuadPointData,
    dataTypes::number                        *gradWfcQuadPointData,
    double                                   *eshelbyContributions,
    double                                   *eshelbyTensor,
    const bool                                floatingNuclearCharges,
    const bool                                isTauMGGA,
    double                                   *pdexTauLocallyOwnedCellsBlock,
    double                                   *pdecTauLocallyOwnedCellsBlock,
    const bool                                computeForce,
    const bool                                computeStress);

  void
  nlpWfcContractionContribution(
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
                            &BLASWrapperPtr,
    const dftfe::uInt        wfcBlockSize,
    const dftfe::uInt        blockSizeNlp,
    const dftfe::uInt        numQuadsNLP,
    const dftfe::uInt        startingIdNlp,
    const dataTypes::number *projectorKetTimesVectorPar,
    const dataTypes::number *gradPsiOrPsiQuadValuesNLP,
    const dftfe::uInt       *nonTrivialIdToElemIdMap,
    const dftfe::uInt       *projecterKetTimesFlattenedVectorLocalIds,
    dataTypes::number       *nlpContractionContribution);
} // namespace dftfe
#endif
#endif
