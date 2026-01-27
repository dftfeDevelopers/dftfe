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
// @author Sambit Das
//

// source file for electron density related computations
#include "densityCalculatorDeviceKernels.h"

namespace dftfe
{
  namespace
  {

    DFTFE_CREATE_KERNEL(
      void,
      computeRhoGradRhoFromInterpolatedValues,
      {
        const dftfe::uInt numEntriesPerCell = numVectors * nQuadsPerCell;
        const dftfe::uInt numberEntries     = numEntriesPerCell * numCells;

        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += nThreadsPerBlock * nThreadBlock)
          {
            const double psi                = wfcContributions[index];
            rhoCellsWfcContributions[index] = psi * psi;

            if (isEvaluateGradRho)
              {
                dftfe::uInt  iCell          = index / numEntriesPerCell;
                dftfe::uInt  intraCellIndex = index - iCell * numEntriesPerCell;
                dftfe::uInt  iQuad          = intraCellIndex / numVectors;
                dftfe::uInt  iVec     = intraCellIndex - iQuad * numVectors;
                const double gradPsiX = //[iVec * numCells * numVectors + + 0]
                  gradwfcContributions[intraCellIndex +
                                       numEntriesPerCell * 3 * iCell];
                gradRhoCellsWfcContributions[iVec + 3 * iQuad * numVectors +
                                             numEntriesPerCell * 3 * iCell] =
                  2.0 * psi * gradPsiX;

                const double gradPsiY =
                  gradwfcContributions[intraCellIndex + numEntriesPerCell +
                                       numEntriesPerCell * 3 * iCell];
                gradRhoCellsWfcContributions[iVec + numVectors +
                                             3 * iQuad * numVectors +
                                             numEntriesPerCell * 3 * iCell] =
                  2.0 * psi * gradPsiY;

                const double gradPsiZ =
                  gradwfcContributions[intraCellIndex + 2 * numEntriesPerCell +
                                       numEntriesPerCell * 3 * iCell];
                gradRhoCellsWfcContributions[iVec + 2 * numVectors +
                                             3 * iQuad * numVectors +
                                             numEntriesPerCell * 3 * iCell] =
                  2.0 * psi * gradPsiZ;
              }
          }
      },
      const dftfe::uInt numVectors,
      const dftfe::uInt numCells,
      const dftfe::uInt nQuadsPerCell,
      double           *wfcContributions,
      double           *gradwfcContributions,
      double           *rhoCellsWfcContributions,
      double           *gradRhoCellsWfcContributions,
      const bool        isEvaluateGradRho);



    DFTFE_CREATE_KERNEL(
      void,
      computeRhoGradRhoFromInterpolatedValues,
      {
        const dftfe::uInt numEntriesPerCell = numVectors * nQuadsPerCell;
        const dftfe::uInt numberEntries     = numEntriesPerCell * numCells;

        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += nThreadsPerBlock * nThreadBlock)
          {
            const dftfe::utils::deviceDoubleComplex psi =
              wfcContributions[index];
            rhoCellsWfcContributions[index] =
              dftfe::utils::realPartDevice(psi) *
                dftfe::utils::realPartDevice(psi) +
              dftfe::utils::imagPartDevice(psi) *
                dftfe::utils::imagPartDevice(psi);

            if (isEvaluateGradRho)
              {
                dftfe::uInt iCell          = index / numEntriesPerCell;
                dftfe::uInt intraCellIndex = index - iCell * numEntriesPerCell;
                dftfe::uInt iQuad          = intraCellIndex / numVectors;
                dftfe::uInt iVec = intraCellIndex - iQuad * numVectors;
                const dftfe::utils::deviceDoubleComplex gradPsiX =
                  gradwfcContributions[intraCellIndex +
                                       numEntriesPerCell * 3 * iCell];
                gradRhoCellsWfcContributions[iVec + 3 * iQuad * numVectors +
                                             numEntriesPerCell * 3 * iCell] =
                  2.0 * (dftfe::utils::realPartDevice(psi) *
                           dftfe::utils::realPartDevice(gradPsiX) +
                         dftfe::utils::imagPartDevice(psi) *
                           dftfe::utils::imagPartDevice(gradPsiX));

                const dftfe::utils::deviceDoubleComplex gradPsiY =
                  gradwfcContributions[intraCellIndex + numEntriesPerCell +
                                       numEntriesPerCell * 3 * iCell];
                gradRhoCellsWfcContributions[iVec + numVectors +
                                             3 * iQuad * numVectors +
                                             numEntriesPerCell * 3 * iCell] =
                  2.0 * (dftfe::utils::realPartDevice(psi) *
                           dftfe::utils::realPartDevice(gradPsiY) +
                         dftfe::utils::imagPartDevice(psi) *
                           dftfe::utils::imagPartDevice(gradPsiY));

                const dftfe::utils::deviceDoubleComplex gradPsiZ =
                  gradwfcContributions[intraCellIndex + 2 * numEntriesPerCell +
                                       numEntriesPerCell * 3 * iCell];
                gradRhoCellsWfcContributions[iVec + 2 * numVectors +
                                             3 * iQuad * numVectors +
                                             numEntriesPerCell * 3 * iCell] =
                  2.0 * (dftfe::utils::realPartDevice(psi) *
                           dftfe::utils::realPartDevice(gradPsiZ) +
                         dftfe::utils::imagPartDevice(psi) *
                           dftfe::utils::imagPartDevice(gradPsiZ));
              }
          }
      },
      const dftfe::uInt                  numVectors,
      const dftfe::uInt                  numCells,
      const dftfe::uInt                  nQuadsPerCell,
      dftfe::utils::deviceDoubleComplex *wfcContributions,
      dftfe::utils::deviceDoubleComplex *gradwfcContributions,
      double                            *rhoCellsWfcContributions,
      double                            *gradRhoCellsWfcContributions,
      const bool                         isEvaluateGradRho);

    DFTFE_CREATE_KERNEL(void,
                        computeNonCollinRhoGradRhoFromInterpolatedValues,
                        {{}},
                        const dftfe::uInt numVectors,
                        const dftfe::uInt numCells,
                        const dftfe::uInt nQuadsPerCell,
                        double           *wfcContributions,
                        double           *gradwfcContributions,
                        double           *rhoCellsWfcContributions,
                        double           *gradRhoCellsWfcContributions,
                        const bool        isEvaluateGradRho,
                        const bool        isNonCollin);

    DFTFE_CREATE_KERNEL(
      void,
      computeNonCollinRhoGradRhoFromInterpolatedValues,
      {
        const dftfe::uInt numEntriesPerCell = numVectors * nQuadsPerCell;
        const dftfe::uInt numberEntries = numVectors * nQuadsPerCell * numCells;

        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += nThreadsPerBlock * nThreadBlock)
          {
            dftfe::uInt iCell          = index / numEntriesPerCell;
            dftfe::uInt intraCellIndex = index - iCell * numEntriesPerCell;
            dftfe::uInt iQuad          = intraCellIndex / numVectors;
            dftfe::uInt iVec           = intraCellIndex - iQuad * numVectors;
            const dftfe::utils::deviceDoubleComplex psiUp =
              wfcContributions[iCell * numEntriesPerCell * 2 +
                               iQuad * numVectors * 2 + iVec];
            const dftfe::utils::deviceDoubleComplex psiDown =
              wfcContributions[iCell * numEntriesPerCell * 2 +
                               iQuad * numVectors * 2 + numVectors + iVec];
            rhoCellsWfcContributions[index] =
              dftfe::utils::abs(dftfe::utils::mult(psiUp, psiUp)) +
              dftfe::utils::abs(dftfe::utils::mult(psiDown, psiDown));
            if (isNonCollin)
              {
                rhoCellsWfcContributions[numberEntries + index] =
                  dftfe::utils::abs(dftfe::utils::mult(psiUp, psiUp)) -
                  dftfe::utils::abs(dftfe::utils::mult(psiDown, psiDown));

                rhoCellsWfcContributions[2 * numberEntries + index] =
                  2.0 *
                  dftfe::utils::imagPartDevice(
                    dftfe::utils::mult(dftfe::utils::conj(psiUp), psiDown));

                rhoCellsWfcContributions[3 * numberEntries + index] =
                  2.0 *
                  dftfe::utils::realPartDevice(
                    dftfe::utils::mult(dftfe::utils::conj(psiUp), psiDown));
              }
            if (isEvaluateGradRho)
              {
                for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
                  {
                    const dftfe::utils::deviceDoubleComplex gradPsiUp =
                      gradwfcContributions[iCell * numEntriesPerCell * 2 * 3 +
                                           iDim * numEntriesPerCell * 2 +
                                           iQuad * numVectors * 2 + iVec];

                    const dftfe::utils::deviceDoubleComplex gradPsiDown =
                      gradwfcContributions[iCell * numEntriesPerCell * 2 * 3 +
                                           iDim * numEntriesPerCell * 2 +
                                           iQuad * numVectors * 2 + numVectors +
                                           iVec];
                    gradRhoCellsWfcContributions[0 * numberEntries * 3 +
                                                 iCell * numEntriesPerCell * 3 +
                                                 iQuad * numVectors * 3 +
                                                 iDim * numVectors + iVec] =
                      2.0 * dftfe::utils::realPartDevice(dftfe::utils::add(
                              dftfe::utils::mult(dftfe::utils::conj(psiUp),
                                                 gradPsiUp),
                              dftfe::utils::mult(dftfe::utils::conj(psiDown),
                                                 gradPsiDown)));
                    if (isNonCollin)
                      {
                        gradRhoCellsWfcContributions[1 * numberEntries * 3 +
                                                     iCell * numEntriesPerCell *
                                                       3 +
                                                     iQuad * numVectors * 3 +
                                                     iDim * numVectors + iVec] =
                          2.0 *
                          dftfe::utils::realPartDevice(dftfe::utils::sub(
                            dftfe::utils::mult(dftfe::utils::conj(psiUp),
                                               gradPsiUp),
                            dftfe::utils::mult(dftfe::utils::conj(psiDown),
                                               gradPsiDown)));
                        gradRhoCellsWfcContributions[2 * numberEntries * 3 +
                                                     iCell * numEntriesPerCell *
                                                       3 +
                                                     iQuad * numVectors * 3 +
                                                     iDim * numVectors + iVec] =
                          2.0 *
                          dftfe::utils::imagPartDevice(dftfe::utils::add(
                            dftfe::utils::mult(dftfe::utils::conj(gradPsiUp),
                                               psiDown),
                            dftfe::utils::mult(dftfe::utils::conj(psiUp),
                                               gradPsiDown)));
                        gradRhoCellsWfcContributions[3 * numberEntries * 3 +
                                                     iCell * numEntriesPerCell *
                                                       3 +
                                                     iQuad * numVectors * 3 +
                                                     iDim * numVectors + iVec] =
                          2.0 *
                          dftfe::utils::realPartDevice(dftfe::utils::add(
                            dftfe::utils::mult(dftfe::utils::conj(gradPsiUp),
                                               psiDown),
                            dftfe::utils::mult(dftfe::utils::conj(psiUp),
                                               gradPsiDown)));
                      }
                  }
              }
          }
      },
      const dftfe::uInt                  numVectors,
      const dftfe::uInt                  numCells,
      const dftfe::uInt                  nQuadsPerCell,
      dftfe::utils::deviceDoubleComplex *wfcContributions,
      dftfe::utils::deviceDoubleComplex *gradwfcContributions,
      double                            *rhoCellsWfcContributions,
      double                            *gradRhoCellsWfcContributions,
      const bool                         isEvaluateGradRho,
      const bool                         isNonCollin);


    DFTFE_CREATE_KERNEL(
      void,
      computeTauFromInterpolatedValues,
      {
        const dftfe::uInt numEntriesPerCell = numVectors * nQuadsPerCell;
        const dftfe::uInt numberEntries     = numEntriesPerCell * numCells;
        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += nThreadsPerBlock * nThreadBlock)
          {
            const double psi = wfcContributions[index];

            dftfe::uInt iCell          = index / numEntriesPerCell;
            dftfe::uInt intraCellIndex = index - iCell * numEntriesPerCell;
            dftfe::uInt iQuad          = intraCellIndex / numVectors;
            dftfe::uInt iVec           = intraCellIndex - iQuad * numVectors;

            double gradPsiDirVal;
            tauCellsWfcContributions[index] = 0.0;
            for (dftfe::uInt dirIdx = 0; dirIdx < 3; dirIdx++)
              {
                gradPsiDirVal =
                  gradwfcContributions[intraCellIndex +
                                       dirIdx * numEntriesPerCell +
                                       numEntriesPerCell * 3 * iCell];

                tauCellsWfcContributions[index] +=
                  gradPsiDirVal * gradPsiDirVal;
              }
            tauCellsWfcContributions[index] =
              0.5 * tauCellsWfcContributions[index];
          }
      },
      const dftfe::uInt numVectors,
      const dftfe::uInt numCells,
      const dftfe::uInt nQuadsPerCell,
      double           *wfcContributions,
      double           *gradwfcContributions,
      double           *kCoord,
      double           *tauCellsWfcContributions);



    DFTFE_CREATE_KERNEL(
      void,
      computeTauFromInterpolatedValues,
      {
        const dftfe::uInt numEntriesPerCell = numVectors * nQuadsPerCell;
        const dftfe::uInt numberEntries     = numEntriesPerCell * numCells;
        const double      kPointCoordSq =
          kCoord[0] * kCoord[0] + kCoord[1] * kCoord[1] + kCoord[2] * kCoord[2];

        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += nThreadsPerBlock * nThreadBlock)
          {
            const dftfe::utils::deviceDoubleComplex psi =
              wfcContributions[index];

            dftfe::uInt iCell          = index / numEntriesPerCell;
            dftfe::uInt intraCellIndex = index - iCell * numEntriesPerCell;
            dftfe::uInt iQuad          = intraCellIndex / numVectors;
            dftfe::uInt iVec           = intraCellIndex - iQuad * numVectors;

            dftfe::utils::deviceDoubleComplex tempImag;

            tempImag = dftfe::utils::makeComplex(0.0, 0.0);

            dftfe::utils::deviceDoubleComplex gradPsiDirVal;
            tauCellsWfcContributions[index] = 0.0;
            for (dftfe::uInt dirIdx = 0; dirIdx < 3; dirIdx++)
              {
                gradPsiDirVal =
                  gradwfcContributions[intraCellIndex +
                                       dirIdx * numEntriesPerCell +
                                       numEntriesPerCell * 3 * iCell];

                tauCellsWfcContributions[index] +=
                  dftfe::utils::realPartDevice(gradPsiDirVal) *
                    dftfe::utils::realPartDevice(gradPsiDirVal) +
                  dftfe::utils::imagPartDevice(gradPsiDirVal) *
                    dftfe::utils::imagPartDevice(gradPsiDirVal);

                tempImag = dftfe::utils::makeComplex(
                  dftfe::utils::realPartDevice(tempImag) +
                    kCoord[dirIdx] *
                      dftfe::utils::realPartDevice(gradPsiDirVal),
                  dftfe::utils::imagPartDevice(tempImag) +
                    kCoord[dirIdx] *
                      dftfe::utils::imagPartDevice(gradPsiDirVal));
              }

            tauCellsWfcContributions[index] =
              0.5 * tauCellsWfcContributions[index];
            tauCellsWfcContributions[index] +=
              0.5 * kPointCoordSq *
              (dftfe::utils::realPartDevice(psi) *
                 dftfe::utils::realPartDevice(psi) +
               dftfe::utils::imagPartDevice(psi) *
                 dftfe::utils::imagPartDevice(psi));
            tauCellsWfcContributions[index] +=
              dftfe::utils::realPartDevice(psi) *
                dftfe::utils::imagPartDevice(tempImag) -
              dftfe::utils::imagPartDevice(psi) *
                dftfe::utils::realPartDevice(tempImag);
          }
      },
      const dftfe::uInt                  numVectors,
      const dftfe::uInt                  numCells,
      const dftfe::uInt                  nQuadsPerCell,
      dftfe::utils::deviceDoubleComplex *wfcContributions,
      dftfe::utils::deviceDoubleComplex *gradwfcContributions,
      double                            *kCoord,
      double                            *tauCellsWfcContributions);


  } // namespace
  template <typename NumberType>
  void
  computeRhoGradRhoFromInterpolatedValues(
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
                                             &BLASWrapperPtr,
    const std::pair<dftfe::uInt, dftfe::uInt> cellRange,
    const std::pair<dftfe::uInt, dftfe::uInt> vecRange,
    const dftfe::uInt                         nQuadsPerCell,
    const dftfe::uInt                         nCells,
    double                                   *partialOccupVec,
    NumberType                               *wfcQuadPointData,
    NumberType                               *gradWfcQuadPointData,
    double                                   *rhoCellsWfcContributions,
    double                                   *gradRhoCellsWfcContributions,
    double                                   *rho,
    double                                   *gradRho,
    const bool                                isEvaluateGradRho,
    const bool                                isNonCollin,
    const bool                                hasSOC)
  {
    const dftfe::uInt cellsBlockSize      = cellRange.second - cellRange.first;
    const dftfe::uInt vectorsBlockSize    = vecRange.second - vecRange.first;
    const dftfe::uInt numComp             = isNonCollin ? 4 : 1;
    const double      scalarCoeffAlphaRho = 1.0;
    const double      scalarCoeffBetaRho  = 1.0;
    const double      scalarCoeffAlphaGradRho = 1.0;
    const double      scalarCoeffBetaGradRho  = 1.0;
    if (isNonCollin || hasSOC)
      DFTFE_LAUNCH_KERNEL(
        computeNonCollinRhoGradRhoFromInterpolatedValues,
        (vectorsBlockSize + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
          dftfe::utils::DEVICE_BLOCK_SIZE * nQuadsPerCell * cellsBlockSize,
        dftfe::utils::DEVICE_BLOCK_SIZE,
        dftfe::utils::defaultStream,
        vectorsBlockSize,
        cellsBlockSize,
        nQuadsPerCell,
        dftfe::utils::makeDataTypeDeviceCompatible(wfcQuadPointData),
        dftfe::utils::makeDataTypeDeviceCompatible(gradWfcQuadPointData),
        dftfe::utils::makeDataTypeDeviceCompatible(rhoCellsWfcContributions),
        dftfe::utils::makeDataTypeDeviceCompatible(
          gradRhoCellsWfcContributions),
        isEvaluateGradRho,
        isNonCollin);
    else
      DFTFE_LAUNCH_KERNEL(
        computeRhoGradRhoFromInterpolatedValues,
        (vectorsBlockSize + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
          dftfe::utils::DEVICE_BLOCK_SIZE * nQuadsPerCell * cellsBlockSize,
        dftfe::utils::DEVICE_BLOCK_SIZE,
        dftfe::utils::defaultStream,
        vectorsBlockSize,
        cellsBlockSize,
        nQuadsPerCell,
        dftfe::utils::makeDataTypeDeviceCompatible(wfcQuadPointData),
        dftfe::utils::makeDataTypeDeviceCompatible(gradWfcQuadPointData),
        dftfe::utils::makeDataTypeDeviceCompatible(rhoCellsWfcContributions),
        dftfe::utils::makeDataTypeDeviceCompatible(
          gradRhoCellsWfcContributions),
        isEvaluateGradRho);
    for (dftfe::uInt iComp = 0; iComp < numComp; ++iComp)
      BLASWrapperPtr->xgemv('T',
                            vectorsBlockSize,
                            cellsBlockSize * nQuadsPerCell,
                            &scalarCoeffAlphaRho,
                            rhoCellsWfcContributions +
                              iComp * vectorsBlockSize * cellsBlockSize *
                                nQuadsPerCell,
                            vectorsBlockSize,
                            partialOccupVec,
                            1,
                            &scalarCoeffBetaRho,
                            rho + cellRange.first * nQuadsPerCell +
                              iComp * nCells * nQuadsPerCell,
                            1);


    if (isEvaluateGradRho)
      {
        for (dftfe::uInt iComp = 0; iComp < numComp; ++iComp)
          BLASWrapperPtr->xgemv('T',
                                vectorsBlockSize,
                                cellsBlockSize * nQuadsPerCell * 3,
                                &scalarCoeffAlphaGradRho,
                                gradRhoCellsWfcContributions +
                                  iComp * vectorsBlockSize * cellsBlockSize *
                                    nQuadsPerCell * 3,
                                vectorsBlockSize,
                                partialOccupVec,
                                1,
                                &scalarCoeffBetaGradRho,
                                gradRho + cellRange.first * nQuadsPerCell * 3 +
                                  iComp * nCells * nQuadsPerCell * 3,
                                1);
      }
  }

  template <typename NumberType>
  void
  computeTauFromInterpolatedValues(
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
                                             &BLASWrapperPtr,
    const std::pair<dftfe::uInt, dftfe::uInt> cellRange,
    const std::pair<dftfe::uInt, dftfe::uInt> vecRange,
    const dftfe::uInt                         nQuadsPerCell,
    double                                   *partialOccupVec,
    double                                   *kCoord,
    NumberType                               *wfcQuadPointData,
    NumberType                               *gradWfcQuadPointData,
    double    *kineticEnergyDensityCellsWfcContributions,
    double    *tau,
    const bool isNonCollin,
    const bool hasSOC)
  {
    const dftfe::uInt cellsBlockSize   = cellRange.second - cellRange.first;
    const dftfe::uInt vectorsBlockSize = vecRange.second - vecRange.first;
    const double      scalarCoeffAlpha = 1.0;
    const double      scalarCoeffBeta  = 1.0;

    DFTFE_LAUNCH_KERNEL(
      computeTauFromInterpolatedValues,
      (vectorsBlockSize + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
        dftfe::utils::DEVICE_BLOCK_SIZE * nQuadsPerCell * cellsBlockSize,
      dftfe::utils::DEVICE_BLOCK_SIZE,
      dftfe::utils::defaultStream,
      vectorsBlockSize,
      cellsBlockSize,
      nQuadsPerCell,
      dftfe::utils::makeDataTypeDeviceCompatible(wfcQuadPointData),
      dftfe::utils::makeDataTypeDeviceCompatible(gradWfcQuadPointData),
      dftfe::utils::makeDataTypeDeviceCompatible(kCoord),
      dftfe::utils::makeDataTypeDeviceCompatible(
        kineticEnergyDensityCellsWfcContributions));

    BLASWrapperPtr->xgemv('T',
                          vectorsBlockSize,
                          cellsBlockSize * nQuadsPerCell,
                          &scalarCoeffAlpha,
                          kineticEnergyDensityCellsWfcContributions,
                          vectorsBlockSize,
                          partialOccupVec,
                          1,
                          &scalarCoeffBeta,
                          tau + cellRange.first * nQuadsPerCell,
                          1);
  }

  template void
  computeRhoGradRhoFromInterpolatedValues(
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
                                             &BLASWrapperPtr,
    const std::pair<dftfe::uInt, dftfe::uInt> cellRange,
    const std::pair<dftfe::uInt, dftfe::uInt> vecRange,
    const dftfe::uInt                         nQuadsPerCell,
    const dftfe::uInt                         nCells,
    double                                   *partialOccupVec,
    dataTypes::number                        *wfcQuadPointData,
    dataTypes::number                        *gradWfcQuadPointData,
    double                                   *rhoCellsWfcContributions,
    double                                   *gradRhoCellsWfcContributions,
    double                                   *rho,
    double                                   *gradRho,
    const bool                                isEvaluateGradRho,
    const bool                                isNonCollin,
    const bool                                hasSOC);

  template void
  computeTauFromInterpolatedValues(
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
                                             &BLASWrapperPtr,
    const std::pair<dftfe::uInt, dftfe::uInt> cellRange,
    const std::pair<dftfe::uInt, dftfe::uInt> vecRange,
    const dftfe::uInt                         nQuadsPerCell,
    double                                   *partialOccupVec,
    double                                   *kCoord,
    dataTypes::number                        *wfcQuadPointData,
    dataTypes::number                        *gradWfcQuadPointData,
    double    *kineticEnergyDensityCellsWfcContributions,
    double    *tau,
    const bool isNonCollin,
    const bool hasSOC);
} // namespace dftfe
