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
// @author Sambit Das, Nikhil Kodali
//
#include <KohnShamHamiltonianOperator.h>
#include <AuxDensityMatrixFE.h>
namespace dftfe
{
  template <dftfe::utils::MemorySpace memorySpace>
  void
  KohnShamHamiltonianOperator<memorySpace>::computeVEffPrime(
    std::shared_ptr<AuxDensityMatrix<memorySpace>>
      auxDensityXCRepresentationPtr,
    const std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &rhoPrimeValues,
    const std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &gradRhoPrimeValues,
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &                phiPrimeValues,
    const unsigned int spinIndex)
  {
    bool isIntegrationByPartsGradDensityDependenceVxc =
      (d_excManagerPtr->getExcSSDFunctionalObj()->getDensityBasedFamilyType() ==
       densityFamilyType::GGA);
    const bool isGGA = isIntegrationByPartsGradDensityDependenceVxc;
    d_basisOperationsPtrHost->reinit(0, 0, d_densityQuadratureID);
    const unsigned int totalLocallyOwnedCells =
      d_basisOperationsPtrHost->nCells();
    const unsigned int numberQuadraturePointsPerCell =
      d_basisOperationsPtrHost->nQuadsPerCell();
#if defined(DFTFE_WITH_DEVICE)
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      d_VeffJxWHost;
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      d_invJacderExcWithSigmaTimesGradRhoJxWHost;
#else
    auto &d_VeffJxWHost = d_VeffJxW;
    auto &d_invJacderExcWithSigmaTimesGradRhoJxWHost =
      d_invJacderExcWithSigmaTimesGradRhoJxW;
#endif
    d_VeffJxWHost.resize(totalLocallyOwnedCells * numberQuadraturePointsPerCell,
                         0.0);
    d_invJacderExcWithSigmaTimesGradRhoJxWHost.clear();
    d_invJacderExcWithSigmaTimesGradRhoJxWHost.resize(
      isGGA ? totalLocallyOwnedCells * numberQuadraturePointsPerCell * 3 : 0,
      0.0);

    std::unordered_map<xcRemainderOutputDataAttributes, std::vector<double>>
      xDataOut;
    std::unordered_map<xcRemainderOutputDataAttributes, std::vector<double>>
      cDataOut;


    std::vector<double> &pdexDensitySpinUp =
      xDataOut[xcRemainderOutputDataAttributes::pdeDensitySpinUp];
    std::vector<double> &pdexDensitySpinDown =
      xDataOut[xcRemainderOutputDataAttributes::pdeDensitySpinDown];
    std::vector<double> &pdecDensitySpinUp =
      cDataOut[xcRemainderOutputDataAttributes::pdeDensitySpinUp];
    std::vector<double> &pdecDensitySpinDown =
      cDataOut[xcRemainderOutputDataAttributes::pdeDensitySpinDown];

    if (isGGA)
      {
        xDataOut[xcRemainderOutputDataAttributes::pdeSigma] =
          std::vector<double>();
        cDataOut[xcRemainderOutputDataAttributes::pdeSigma] =
          std::vector<double>();
      }

    auto quadPointsAll = d_basisOperationsPtrHost->quadPoints();

    auto quadWeightsAll = d_basisOperationsPtrHost->JxW();

    std::vector<double> quadPointsStdVecAll;
    quadPointsStdVecAll.resize(quadPointsAll.size());
    std::vector<double> quadWeightsStdVecAll;
    quadWeightsStdVecAll.resize(quadWeightsAll.size());
    for (unsigned int iQuad = 0; iQuad < quadWeightsStdVecAll.size(); ++iQuad)
      {
        for (unsigned int idim = 0; idim < 3; ++idim)
          quadPointsStdVecAll[3 * iQuad + idim] =
            quadPointsAll[3 * iQuad + idim];
        quadWeightsStdVecAll[iQuad] = std::real(quadWeightsAll[iQuad]);
      }

    const double lambda = 1e-2;
    for (unsigned int iCellQuad = 0;
         iCellQuad < totalLocallyOwnedCells * numberQuadraturePointsPerCell;
         ++iCellQuad)
      d_VeffJxWHost[iCellQuad] =
        phiPrimeValues[iCellQuad] *
        d_basisOperationsPtrHost->JxWBasisData()[iCellQuad];
    std::transform(phiPrimeValues.begin(),
                   phiPrimeValues.end(),
                   d_basisOperationsPtrHost->JxWBasisData().begin(),
                   d_VeffJxWHost.begin(),
                   std::multiplies<>{});

    auto computeXCPerturbedDensity = [&](double densityPerturbCoeff,
                                         double veffCoeff) {
      std::unordered_map<DensityDescriptorDataAttributes, std::vector<double>>
                           densityDataAll;
      std::vector<double> &densitySpinUpAll =
        densityDataAll[DensityDescriptorDataAttributes::valuesSpinUp];
      std::vector<double> &densitySpinDownAll =
        densityDataAll[DensityDescriptorDataAttributes::valuesSpinDown];
      if (isGGA)
        {
          densityDataAll[DensityDescriptorDataAttributes::gradValuesSpinUp] =
            std::vector<double>();
          densityDataAll[DensityDescriptorDataAttributes::gradValuesSpinDown] =
            std::vector<double>();
        }

      auxDensityXCRepresentationPtr->applyLocalOperations(quadPointsStdVecAll,
                                                          densityDataAll);

      std::vector<double> gradDensitySpinUpAll;
      std::vector<double> gradDensitySpinDownAll;

      if (isGGA)
        {
          gradDensitySpinUpAll =
            densityDataAll[DensityDescriptorDataAttributes::gradValuesSpinUp];
          gradDensitySpinDownAll =
            densityDataAll[DensityDescriptorDataAttributes::gradValuesSpinDown];
        }

      std::unordered_map<std::string, std::vector<double>>
        perturbedDensityProjectionInputs;

      std::vector<double> &perturbedDensityValsForXC =
        perturbedDensityProjectionInputs["densityFunc"];
      perturbedDensityProjectionInputs["quadpts"] = quadPointsStdVecAll;
      perturbedDensityProjectionInputs["quadWt"]  = quadWeightsStdVecAll;

      perturbedDensityValsForXC.resize(2 * totalLocallyOwnedCells *
                                         numberQuadraturePointsPerCell,
                                       0);

      const double *rhoTotalPrimeValues = rhoPrimeValues[0].data();
      const double *rhoMagzPrimeValues  = rhoPrimeValues[1].data();
      for (unsigned int iQuad = 0;
           iQuad < totalLocallyOwnedCells * numberQuadraturePointsPerCell;
           ++iQuad)
        perturbedDensityValsForXC[iQuad] =
          densitySpinUpAll[iQuad] +
          densityPerturbCoeff *
            (rhoTotalPrimeValues[iQuad] + rhoMagzPrimeValues[iQuad]) / 2.0;

      for (unsigned int iQuad = 0;
           iQuad < totalLocallyOwnedCells * numberQuadraturePointsPerCell;
           ++iQuad)
        perturbedDensityValsForXC[totalLocallyOwnedCells *
                                    numberQuadraturePointsPerCell +
                                  iQuad] =
          densitySpinDownAll[iQuad] +
          densityPerturbCoeff *
            (rhoTotalPrimeValues[iQuad] - rhoMagzPrimeValues[iQuad]) / 2.0;

      if (isGGA)
        {
          std::vector<double> &perturbedGradDensityValsForXC =
            perturbedDensityProjectionInputs["gradDensityFunc"];

          perturbedGradDensityValsForXC.resize(
            2 * totalLocallyOwnedCells * numberQuadraturePointsPerCell * 3, 0);

          const double *gradRhoTotalPrimeValues = gradRhoPrimeValues[0].data();
          const double *gradRhoMagzPrimeValues  = gradRhoPrimeValues[1].data();
          for (unsigned int i = 0;
               i < totalLocallyOwnedCells * numberQuadraturePointsPerCell * 3;
               ++i)
            for (unsigned int idim = 0; idim < 3; ++idim)
              perturbedGradDensityValsForXC[i] =
                gradDensitySpinUpAll[i] +
                densityPerturbCoeff *
                  (gradRhoTotalPrimeValues[i] + gradRhoMagzPrimeValues[i]) /
                  2.0;

          for (unsigned int i = 0;
               i < totalLocallyOwnedCells * numberQuadraturePointsPerCell * 3;
               ++i)
            for (unsigned int idim = 0; idim < 3; ++idim)
              perturbedGradDensityValsForXC[totalLocallyOwnedCells *
                                              numberQuadraturePointsPerCell *
                                              3 +
                                            i] =
                gradDensitySpinDownAll[i] +
                densityPerturbCoeff *
                  (gradRhoTotalPrimeValues[i] - gradRhoMagzPrimeValues[i]) /
                  2.0;
        }


      std::shared_ptr<AuxDensityMatrix<memorySpace>>
        auxDensityXCPerturbedRepresentationPtr =
          std::make_shared<AuxDensityMatrixFE<memorySpace>>();
      auxDensityXCPerturbedRepresentationPtr->projectDensityStart(
        perturbedDensityProjectionInputs);

      auxDensityXCPerturbedRepresentationPtr->projectDensityEnd(
        d_mpiCommDomain);

      //
      // loop over cell block
      //
      for (unsigned int iCell = 0; iCell < totalLocallyOwnedCells; ++iCell)
        {
          std::vector<double> quadPointsInCell(numberQuadraturePointsPerCell *
                                               3);
          std::vector<double> quadWeightsInCell(numberQuadraturePointsPerCell);
          for (unsigned int iQuad = 0; iQuad < numberQuadraturePointsPerCell;
               ++iQuad)
            {
              for (unsigned int idim = 0; idim < 3; ++idim)
                quadPointsInCell[3 * iQuad + idim] =
                  quadPointsAll[iCell * numberQuadraturePointsPerCell * 3 +
                                3 * iQuad + idim];
              quadWeightsInCell[iQuad] = std::real(
                quadWeightsAll[iCell * numberQuadraturePointsPerCell + iQuad]);
            }

          d_excManagerPtr->getExcSSDFunctionalObj()->computeOutputXCData(
            *auxDensityXCPerturbedRepresentationPtr,
            quadPointsInCell,
            xDataOut,
            cDataOut);


          const std::vector<double> &pdexDensitySpinIndex =
            spinIndex == 0 ? pdexDensitySpinUp : pdexDensitySpinDown;
          const std::vector<double> &pdecDensitySpinIndex =
            spinIndex == 0 ? pdecDensitySpinUp : pdecDensitySpinDown;

          std::vector<double> pdexSigma;
          std::vector<double> pdecSigma;
          if (isGGA)
            {
              pdexSigma = xDataOut[xcRemainderOutputDataAttributes::pdeSigma];
              pdecSigma = cDataOut[xcRemainderOutputDataAttributes::pdeSigma];
            }

          std::unordered_map<DensityDescriptorDataAttributes,
                             std::vector<double>>
                               densityData;
          std::vector<double> &gradDensitySpinUp =
            densityData[DensityDescriptorDataAttributes::gradValuesSpinUp];
          std::vector<double> &gradDensitySpinDown =
            densityData[DensityDescriptorDataAttributes::gradValuesSpinDown];

          if (isGGA)
            auxDensityXCPerturbedRepresentationPtr->applyLocalOperations(
              quadPointsInCell, densityData);

          const std::vector<double> &gradDensityXCSpinIndex =
            spinIndex == 0 ? gradDensitySpinUp : gradDensitySpinDown;
          const std::vector<double> &gradDensityXCOtherSpinIndex =
            spinIndex == 0 ? gradDensitySpinDown : gradDensitySpinUp;



          auto cellJxWPtr = d_basisOperationsPtrHost->JxWBasisData().data() +
                            iCell * numberQuadraturePointsPerCell;


          for (unsigned int iQuad = 0; iQuad < numberQuadraturePointsPerCell;
               ++iQuad)
            {
              d_VeffJxWHost[iCell * numberQuadraturePointsPerCell + iQuad] +=
                veffCoeff *
                (pdexDensitySpinIndex[iQuad] + pdecDensitySpinIndex[iQuad]) *
                cellJxWPtr[iQuad];
            }

          if (isGGA)
            {
              if (d_basisOperationsPtrHost->cellsTypeFlag() != 2)
                {
                  for (unsigned int iQuad = 0;
                       iQuad < numberQuadraturePointsPerCell;
                       ++iQuad)
                    {
                      const double *inverseJacobiansQuadPtr =
                        d_basisOperationsPtrHost->inverseJacobiansBasisData()
                          .data() +
                        (d_basisOperationsPtrHost->cellsTypeFlag() == 0 ?
                           iCell * numberQuadraturePointsPerCell * 9 +
                             iQuad * 9 :
                           iCell * 9);
                      const double *gradDensityQuadPtr =
                        gradDensityXCSpinIndex.data() + iQuad * 3;
                      const double *gradDensityOtherQuadPtr =
                        gradDensityXCOtherSpinIndex.data() + iQuad * 3;
                      const double term =
                        (pdexSigma[iQuad * 3 + 2 * spinIndex] +
                         pdecSigma[iQuad * 3 + 2 * spinIndex]) *
                        cellJxWPtr[iQuad];
                      const double termoff =
                        (pdexSigma[iQuad * 3 + 1] + pdecSigma[iQuad * 3 + 1]) *
                        cellJxWPtr[iQuad];
                      for (unsigned jDim = 0; jDim < 3; ++jDim)
                        for (unsigned iDim = 0; iDim < 3; ++iDim)
                          d_invJacderExcWithSigmaTimesGradRhoJxWHost
                            [iCell * numberQuadraturePointsPerCell * 3 +
                             iQuad * 3 + iDim] +=
                            veffCoeff *
                            inverseJacobiansQuadPtr[3 * jDim + iDim] *
                            (2.0 * gradDensityQuadPtr[jDim] * term +
                             gradDensityOtherQuadPtr[jDim] * termoff);
                    }
                }
              else if (d_basisOperationsPtrHost->cellsTypeFlag() == 2)
                {
                  for (unsigned int iQuad = 0;
                       iQuad < numberQuadraturePointsPerCell;
                       ++iQuad)
                    {
                      const double *inverseJacobiansQuadPtr =
                        d_basisOperationsPtrHost->inverseJacobiansBasisData()
                          .data() +
                        iCell * 3;
                      const double *gradDensityQuadPtr =
                        gradDensityXCSpinIndex.data() + iQuad * 3;
                      const double *gradDensityOtherQuadPtr =
                        gradDensityXCOtherSpinIndex.data() + iQuad * 3;
                      const double term =
                        (pdexSigma[iQuad * 3 + 2 * spinIndex] +
                         pdecSigma[iQuad * 3 + 2 * spinIndex]) *
                        cellJxWPtr[iQuad];
                      const double termoff =
                        (pdexSigma[iQuad * 3 + 1] + pdecSigma[iQuad * 3 + 1]) *
                        cellJxWPtr[iQuad];
                      for (unsigned iDim = 0; iDim < 3; ++iDim)
                        d_invJacderExcWithSigmaTimesGradRhoJxWHost
                          [iCell * numberQuadraturePointsPerCell * 3 +
                           iQuad * 3 + iDim] +=
                          veffCoeff * inverseJacobiansQuadPtr[iDim] *
                          (2.0 * gradDensityQuadPtr[iDim] * term +
                           gradDensityOtherQuadPtr[iDim] * termoff);
                    }
                }
            } // GGA
        }     // cell loop
    };
    computeXCPerturbedDensity(2.0 * lambda, -1.0 / 12.0 / lambda);
    computeXCPerturbedDensity(lambda, 2.0 / 3.0 / lambda);
    computeXCPerturbedDensity(-2.0 * lambda, 1.0 / 12.0 / lambda);
    computeXCPerturbedDensity(-lambda, -2.0 / 3.0 / lambda);



#if defined(DFTFE_WITH_DEVICE)
    d_VeffJxW.resize(d_VeffJxWHost.size());
    d_VeffJxW.copyFrom(d_VeffJxWHost);
    d_invJacderExcWithSigmaTimesGradRhoJxW.resize(
      d_invJacderExcWithSigmaTimesGradRhoJxWHost.size());
    d_invJacderExcWithSigmaTimesGradRhoJxW.copyFrom(
      d_invJacderExcWithSigmaTimesGradRhoJxWHost);
#endif
  }
  template class KohnShamHamiltonianOperator<dftfe::utils::MemorySpace::HOST>;
#if defined(DFTFE_WITH_DEVICE)
  template class KohnShamHamiltonianOperator<dftfe::utils::MemorySpace::DEVICE>;
#endif
} // namespace dftfe
