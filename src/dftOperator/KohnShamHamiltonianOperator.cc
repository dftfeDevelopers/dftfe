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
// @author Nikhil Kodali
//

#include <KohnShamHamiltonianOperator.h>
namespace dftfe
{
  //
  // constructor
  //
  template <dftfe::utils::MemorySpace memorySpace>
  KohnShamHamiltonianOperator<memorySpace>::KohnShamHamiltonianOperator(
    std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
      BLASWrapperPtr,
    std::shared_ptr<
      dftfe::basis::FEBasisOperations<dataTypes::number, double, memorySpace>>
      basisOperationsPtr,
    std::shared_ptr<
      dftfe::basis::FEBasisOperations<dataTypes::number,
                                      double,
                                      dftfe::utils::MemorySpace::HOST>>
      basisOperationsPtrHost,
    std::shared_ptr<dftfe::oncvClass<dataTypes::number, memorySpace>>
                                oncvClassPtr,
    std::shared_ptr<excManager> excManagerPtr,
    dftParameters *             dftParamsPtr,
    const unsigned int          densityQuadratureID,
    const unsigned int          lpspQuadratureID,
    const unsigned int          feOrderPlusOneQuadratureID,
    const MPI_Comm &            mpi_comm_parent,
    const MPI_Comm &            mpi_comm_domain)
    : d_kPointIndex(0)
    , d_spinIndex(0)
    , d_HamiltonianIndex(0)
    , d_BLASWrapperPtr(BLASWrapperPtr)
    , d_basisOperationsPtr(basisOperationsPtr)
    , d_basisOperationsPtrHost(basisOperationsPtrHost)
    , d_oncvClassPtr(oncvClassPtr)
    , d_excManagerPtr(excManagerPtr)
    , d_dftParamsPtr(dftParamsPtr)
    , d_densityQuadratureID(densityQuadratureID)
    , d_lpspQuadratureID(lpspQuadratureID)
    , d_feOrderPlusOneQuadratureID(feOrderPlusOneQuadratureID)
    , d_isExternalPotCorrHamiltonianComputed(false)
    , d_mpiCommParent(mpi_comm_parent)
    , d_mpiCommDomain(mpi_comm_domain)
    , n_mpi_processes(dealii::Utilities::MPI::n_mpi_processes(mpi_comm_domain))
    , this_mpi_process(
        dealii::Utilities::MPI::this_mpi_process(mpi_comm_domain))
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0))
    , computing_timer(mpi_comm_domain,
                      pcout,
                      dealii::TimerOutput::never,
                      dealii::TimerOutput::wall_times)
  {
    if (d_dftParamsPtr->isPseudopotential)
      d_ONCVnonLocalOperator = oncvClassPtr->getNonLocalOperator();
    if (d_dftParamsPtr->isPseudopotential && d_dftParamsPtr->useSinglePrecCheby)
      d_ONCVnonLocalOperatorSinglePrec =
        oncvClassPtr->getNonLocalOperatorSinglePrec();
    d_cellsBlockSizeHamiltonianConstruction =
      memorySpace == dftfe::utils::MemorySpace::HOST ? 1 : 50;
    d_cellsBlockSizeHX = memorySpace == dftfe::utils::MemorySpace::HOST ?
                           1 :
                           d_basisOperationsPtr->nCells();
    d_numVectorsInternal = 0;
  }

  //
  // initialize KohnShamHamiltonianOperator object
  //
  template <dftfe::utils::MemorySpace memorySpace>
  void
  KohnShamHamiltonianOperator<memorySpace>::init(
    const std::vector<double> &kPointCoordinates,
    const std::vector<double> &kPointWeights)
  {
    computing_timer.enter_subsection("KohnShamHamiltonianOperator setup");
    inverseSqrtMassVectorScaledConstraintsNoneDataInfoPtr =
      std::make_shared<dftUtils::constraintMatrixInfo<memorySpace>>(
        d_basisOperationsPtr
          ->d_constraintInfo[d_basisOperationsPtr->d_dofHandlerID]);
    inverseSqrtMassVectorScaledConstraintsNoneDataInfoPtr
      ->initializeScaledConstraints(
        d_basisOperationsPtr->inverseSqrtMassVectorBasisData());
    inverseMassVectorScaledConstraintsNoneDataInfoPtr =
      std::make_shared<dftUtils::constraintMatrixInfo<memorySpace>>(
        d_basisOperationsPtr
          ->d_constraintInfo[d_basisOperationsPtr->d_dofHandlerID]);
    inverseMassVectorScaledConstraintsNoneDataInfoPtr
      ->initializeScaledConstraints(
        d_basisOperationsPtr->inverseMassVectorBasisData());
    d_kPointCoordinates = kPointCoordinates;
    d_kPointWeights     = kPointWeights;
    d_invJacKPointTimesJxW.resize(d_kPointWeights.size());
    d_cellHamiltonianMatrix.resize(
      d_dftParamsPtr->memOptMode ?
        1 :
        (d_kPointWeights.size() * (d_dftParamsPtr->spinPolarized + 1)));
    d_cellHamiltonianMatrixSinglePrec.resize(
      d_dftParamsPtr->useSinglePrecCheby ? d_cellHamiltonianMatrix.size() : 0);

    const unsigned int nCells       = d_basisOperationsPtr->nCells();
    const unsigned int nDofsPerCell = d_basisOperationsPtr->nDofsPerCell();
    tempHamMatrixRealBlock.resize(nDofsPerCell * nDofsPerCell *
                                  d_cellsBlockSizeHamiltonianConstruction);
    if (std::is_same<dataTypes::number, std::complex<double>>::value)
      tempHamMatrixImagBlock.resize(nDofsPerCell * nDofsPerCell *
                                    d_cellsBlockSizeHamiltonianConstruction);
    for (unsigned int iHamiltonian = 0;
         iHamiltonian < d_cellHamiltonianMatrix.size();
         ++iHamiltonian)
      d_cellHamiltonianMatrix[iHamiltonian].resize(nDofsPerCell * nDofsPerCell *
                                                   nCells);
    for (unsigned int iHamiltonian = 0;
         iHamiltonian < d_cellHamiltonianMatrixSinglePrec.size();
         ++iHamiltonian)
      d_cellHamiltonianMatrixSinglePrec[iHamiltonian].resize(
        nDofsPerCell * nDofsPerCell * nCells);

    d_basisOperationsPtrHost->reinit(0, 0, d_densityQuadratureID, false);
    const unsigned int numberQuadraturePoints =
      d_basisOperationsPtrHost->nQuadsPerCell();
    if (std::is_same<dataTypes::number, std::complex<double>>::value)
      for (unsigned int kPointIndex = 0; kPointIndex < d_kPointWeights.size();
           ++kPointIndex)
        {
#if defined(DFTFE_WITH_DEVICE)
          dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
            d_invJacKPointTimesJxWHost;
#else
          auto &d_invJacKPointTimesJxWHost =
            d_invJacKPointTimesJxW[kPointIndex];
#endif
          d_invJacKPointTimesJxWHost.resize(nCells * numberQuadraturePoints * 3,
                                            0.0);
          for (unsigned int iCell = 0; iCell < nCells; ++iCell)
            {
              auto cellJxWPtr =
                d_basisOperationsPtrHost->JxWBasisData().data() +
                iCell * numberQuadraturePoints;
              const double *kPointCoordinatesPtr =
                kPointCoordinates.data() + 3 * kPointIndex;

              if (d_basisOperationsPtrHost->cellsTypeFlag() != 2)
                {
                  for (unsigned int iQuad = 0; iQuad < numberQuadraturePoints;
                       ++iQuad)
                    {
                      const double *inverseJacobiansQuadPtr =
                        d_basisOperationsPtrHost->inverseJacobiansBasisData()
                          .data() +
                        (d_basisOperationsPtrHost->cellsTypeFlag() == 0 ?
                           iCell * numberQuadraturePoints * 9 + iQuad * 9 :
                           iCell * 9);
                      for (unsigned jDim = 0; jDim < 3; ++jDim)
                        for (unsigned iDim = 0; iDim < 3; ++iDim)
                          d_invJacKPointTimesJxWHost[iCell *
                                                       numberQuadraturePoints *
                                                       3 +
                                                     iQuad * 3 + iDim] +=
                            -inverseJacobiansQuadPtr[3 * jDim + iDim] *
                            kPointCoordinatesPtr[jDim] * cellJxWPtr[iQuad];
                    }
                }
              else if (d_basisOperationsPtrHost->cellsTypeFlag() == 2)
                {
                  for (unsigned int iQuad = 0; iQuad < numberQuadraturePoints;
                       ++iQuad)
                    {
                      const double *inverseJacobiansQuadPtr =
                        d_basisOperationsPtrHost->inverseJacobiansBasisData()
                          .data() +
                        iCell * 3;
                      for (unsigned iDim = 0; iDim < 3; ++iDim)
                        d_invJacKPointTimesJxWHost[iCell *
                                                     numberQuadraturePoints *
                                                     3 +
                                                   iQuad * 3 + iDim] =
                          -inverseJacobiansQuadPtr[iDim] *
                          kPointCoordinatesPtr[iDim] * cellJxWPtr[iQuad];
                    }
                }
            }
#if defined(DFTFE_WITH_DEVICE)
          d_invJacKPointTimesJxW[kPointIndex].resize(
            d_invJacKPointTimesJxWHost.size());
          d_invJacKPointTimesJxW[kPointIndex].copyFrom(
            d_invJacKPointTimesJxWHost);
#endif
        }
    computing_timer.leave_subsection("KohnShamHamiltonianOperator setup");
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  KohnShamHamiltonianOperator<memorySpace>::resetExtPotHamFlag()
  {
    d_isExternalPotCorrHamiltonianComputed = false;
  }


  template <dftfe::utils::MemorySpace memorySpace>
  void
  KohnShamHamiltonianOperator<memorySpace>::computeVEff(
    const std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &rhoValues,
    const std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &gradRhoValues,
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &                                                  phiValues,
    const std::map<dealii::CellId, std::vector<double>> &rhoCoreValues,
    const std::map<dealii::CellId, std::vector<double>> &gradRhoCoreValues,
    const unsigned int                                   spinIndex)
  {
    const bool isGGA =
      d_excManagerPtr->getDensityBasedFamilyType() == densityFamilyType::GGA;
    const unsigned int spinPolarizedFactor = 1 + d_dftParamsPtr->spinPolarized;
    const unsigned int spinPolarizedSigmaFactor =
      d_dftParamsPtr->spinPolarized == 0 ? 1 : 3;
    d_basisOperationsPtrHost->reinit(0, 0, d_densityQuadratureID);
    const unsigned int totalLocallyOwnedCells =
      d_basisOperationsPtrHost->nCells();
    const unsigned int numberQuadraturePoints =
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
    d_VeffJxWHost.resize(totalLocallyOwnedCells * numberQuadraturePoints, 0.0);
    d_invJacderExcWithSigmaTimesGradRhoJxWHost.resize(
      isGGA ? totalLocallyOwnedCells * numberQuadraturePoints * 3 : 0, 0.0);

    // allocate storage for exchange potential
    std::vector<double> exchangePotentialVal(numberQuadraturePoints *
                                             spinPolarizedFactor);
    std::vector<double> corrPotentialVal(numberQuadraturePoints *
                                         spinPolarizedFactor);
    std::vector<double> densityValue(numberQuadraturePoints *
                                     spinPolarizedFactor);
    std::vector<double> sigmaValue(
      isGGA ? numberQuadraturePoints * spinPolarizedSigmaFactor : 0);
    std::vector<double> derExchEnergyWithSigmaVal(
      isGGA ? numberQuadraturePoints * spinPolarizedSigmaFactor : 0);
    std::vector<double> derCorrEnergyWithSigmaVal(
      isGGA ? numberQuadraturePoints * spinPolarizedSigmaFactor : 0);
    std::vector<double> gradDensityValue(
      isGGA ? 3 * numberQuadraturePoints * spinPolarizedFactor : 0);
    auto dot3 = [](const double *a, const double *b) {
      double sum = 0.0;
      for (unsigned int i = 0; i < 3; i++)
        {
          sum += a[i] * b[i];
        }
      return sum;
    };

    for (unsigned int iCell = 0; iCell < totalLocallyOwnedCells; ++iCell)
      {
        if (spinPolarizedFactor == 1)
          std::memcpy(densityValue.data(),
                      rhoValues[0].data() + iCell * numberQuadraturePoints,
                      numberQuadraturePoints * sizeof(double));
        else if (spinPolarizedFactor == 2)
          {
            const double *cellRhoValues =
              rhoValues[0].data() + iCell * numberQuadraturePoints;
            const double *cellMagValues =
              rhoValues[1].data() + iCell * numberQuadraturePoints;
            for (unsigned int iQuad = 0; iQuad < numberQuadraturePoints;
                 ++iQuad)
              {
                const double rhoByTwo       = cellRhoValues[iQuad] / 2.0;
                const double magByTwo       = cellMagValues[iQuad] / 2.0;
                densityValue[2 * iQuad]     = rhoByTwo + magByTwo;
                densityValue[2 * iQuad + 1] = rhoByTwo - magByTwo;
              }
          }
        if (isGGA)
          if (spinPolarizedFactor == 1)
            std::memcpy(gradDensityValue.data(),
                        gradRhoValues[0].data() +
                          iCell * numberQuadraturePoints * 3,
                        3 * numberQuadraturePoints * sizeof(double));
          else if (spinPolarizedFactor == 2)
            {
              const double *cellGradRhoValues =
                gradRhoValues[0].data() + 3 * iCell * numberQuadraturePoints;
              const double *cellGradMagValues =
                gradRhoValues[1].data() + 3 * iCell * numberQuadraturePoints;
              for (unsigned int iQuad = 0; iQuad < numberQuadraturePoints;
                   ++iQuad)
                for (unsigned int iDim = 0; iDim < 3; ++iDim)
                  {
                    const double gradRhoByTwo =
                      cellGradRhoValues[3 * iQuad + iDim] / 2.0;
                    const double gradMagByTwo =
                      cellGradMagValues[3 * iQuad + iDim] / 2.0;
                    gradDensityValue[6 * iQuad + iDim] =
                      gradRhoByTwo + gradMagByTwo;
                    gradDensityValue[6 * iQuad + 3 + iDim] =
                      gradRhoByTwo - gradMagByTwo;
                  }
            }
        const double *tempPhi =
          phiValues.data() + iCell * numberQuadraturePoints;


        if (d_dftParamsPtr->nonLinearCoreCorrection)
          if (spinPolarizedFactor == 1)
            {
              std::transform(densityValue.data(),
                             densityValue.data() + numberQuadraturePoints,
                             rhoCoreValues
                               .find(d_basisOperationsPtrHost->cellID(iCell))
                               ->second.data(),
                             densityValue.data(),
                             std::plus<>{});
              if (isGGA)
                std::transform(gradDensityValue.data(),
                               gradDensityValue.data() +
                                 3 * numberQuadraturePoints,
                               gradRhoCoreValues
                                 .find(d_basisOperationsPtrHost->cellID(iCell))
                                 ->second.data(),
                               gradDensityValue.data(),
                               std::plus<>{});
            }
          else if (spinPolarizedFactor == 2)
            {
              const std::vector<double> &temp2 =
                rhoCoreValues.find(d_basisOperationsPtrHost->cellID(iCell))
                  ->second;
              for (unsigned int iQuad = 0; iQuad < numberQuadraturePoints;
                   ++iQuad)
                {
                  densityValue[2 * iQuad] += temp2[iQuad] / 2.0;
                  densityValue[2 * iQuad + 1] += temp2[iQuad] / 2.0;
                }
              if (isGGA)
                {
                  const std::vector<double> &temp3 =
                    gradRhoCoreValues
                      .find(d_basisOperationsPtrHost->cellID(iCell))
                      ->second;
                  for (unsigned int iQuad = 0; iQuad < numberQuadraturePoints;
                       ++iQuad)
                    for (unsigned int iDim = 0; iDim < 3; ++iDim)
                      {
                        gradDensityValue[6 * iQuad + iDim] +=
                          temp3[3 * iQuad + iDim] / 2.0;
                        gradDensityValue[6 * iQuad + iDim + 3] +=
                          temp3[3 * iQuad + iDim] / 2.0;
                      }
                }
            }
        if (isGGA)
          {
            if (spinPolarizedFactor == 1)
              for (unsigned int iQuad = 0; iQuad < numberQuadraturePoints;
                   ++iQuad)
                sigmaValue[iQuad] = dot3(gradDensityValue.data() + 3 * iQuad,
                                         gradDensityValue.data() + 3 * iQuad);
            else if (spinPolarizedFactor == 2)
              for (unsigned int iQuad = 0; iQuad < numberQuadraturePoints;
                   ++iQuad)
                {
                  sigmaValue[3 * iQuad] =
                    dot3(gradDensityValue.data() + 6 * iQuad,
                         gradDensityValue.data() + 6 * iQuad);
                  sigmaValue[3 * iQuad + 1] =
                    dot3(gradDensityValue.data() + 6 * iQuad,
                         gradDensityValue.data() + 6 * iQuad + 3);
                  sigmaValue[3 * iQuad + 2] =
                    dot3(gradDensityValue.data() + 6 * iQuad + 3,
                         gradDensityValue.data() + 6 * iQuad + 3);
                }
          }
        std::map<rhoDataAttributes, const std::vector<double> *> rhoData;

        std::map<VeffOutputDataAttributes, std::vector<double> *>
          outputDerExchangeEnergy;
        std::map<VeffOutputDataAttributes, std::vector<double> *>
          outputDerCorrEnergy;

        rhoData[rhoDataAttributes::values] = &densityValue;

        outputDerExchangeEnergy
          [VeffOutputDataAttributes::derEnergyWithDensity] =
            &exchangePotentialVal;

        outputDerCorrEnergy[VeffOutputDataAttributes::derEnergyWithDensity] =
          &corrPotentialVal;
        if (isGGA)
          {
            rhoData[rhoDataAttributes::sigmaGradValue] = &sigmaValue;
            outputDerExchangeEnergy
              [VeffOutputDataAttributes::derEnergyWithSigmaGradDensity] =
                &derExchEnergyWithSigmaVal;
            outputDerCorrEnergy
              [VeffOutputDataAttributes::derEnergyWithSigmaGradDensity] =
                &derCorrEnergyWithSigmaVal;
          }
        d_excManagerPtr->getExcDensityObj()->computeDensityBasedVxc(
          numberQuadraturePoints,
          rhoData,
          outputDerExchangeEnergy,
          outputDerCorrEnergy);
        auto cellJxWPtr = d_basisOperationsPtrHost->JxWBasisData().data() +
                          iCell * numberQuadraturePoints;
        for (unsigned int iQuad = 0; iQuad < numberQuadraturePoints; ++iQuad)
          {
            if (spinPolarizedFactor == 1)
              d_VeffJxWHost[iCell * numberQuadraturePoints + iQuad] =
                (tempPhi[iQuad] + exchangePotentialVal[iQuad] +
                 corrPotentialVal[iQuad]) *
                cellJxWPtr[iQuad];
            else
              d_VeffJxWHost[iCell * numberQuadraturePoints + iQuad] =
                (tempPhi[iQuad] + exchangePotentialVal[2 * iQuad + spinIndex] +
                 corrPotentialVal[2 * iQuad + spinIndex]) *
                cellJxWPtr[iQuad];
          }
        if (isGGA)
          {
            if (spinPolarizedFactor == 1)
              {
                if (d_basisOperationsPtrHost->cellsTypeFlag() != 2)
                  {
                    for (unsigned int iQuad = 0; iQuad < numberQuadraturePoints;
                         ++iQuad)
                      {
                        const double *inverseJacobiansQuadPtr =
                          d_basisOperationsPtrHost->inverseJacobiansBasisData()
                            .data() +
                          (d_basisOperationsPtrHost->cellsTypeFlag() == 0 ?
                             iCell * numberQuadraturePoints * 9 + iQuad * 9 :
                             iCell * 9);
                        const double *gradDensityQuadPtr =
                          gradDensityValue.data() + 3 * iQuad;
                        const double term = (derExchEnergyWithSigmaVal[iQuad] +
                                             derCorrEnergyWithSigmaVal[iQuad]) *
                                            cellJxWPtr[iQuad];
                        for (unsigned jDim = 0; jDim < 3; ++jDim)
                          for (unsigned iDim = 0; iDim < 3; ++iDim)
                            d_invJacderExcWithSigmaTimesGradRhoJxWHost
                              [iCell * numberQuadraturePoints * 3 + iQuad * 3 +
                               iDim] +=
                              2.0 * inverseJacobiansQuadPtr[3 * jDim + iDim] *
                              gradDensityQuadPtr[jDim] * term;
                      }
                  }
                else if (d_basisOperationsPtrHost->cellsTypeFlag() == 2)
                  {
                    for (unsigned int iQuad = 0; iQuad < numberQuadraturePoints;
                         ++iQuad)
                      {
                        const double *inverseJacobiansQuadPtr =
                          d_basisOperationsPtrHost->inverseJacobiansBasisData()
                            .data() +
                          iCell * 3;
                        const double *gradDensityQuadPtr =
                          gradDensityValue.data() + 3 * iQuad;
                        const double term = (derExchEnergyWithSigmaVal[iQuad] +
                                             derCorrEnergyWithSigmaVal[iQuad]) *
                                            cellJxWPtr[iQuad];
                        for (unsigned iDim = 0; iDim < 3; ++iDim)
                          d_invJacderExcWithSigmaTimesGradRhoJxWHost
                            [iCell * numberQuadraturePoints * 3 + iQuad * 3 +
                             iDim] = 2.0 * inverseJacobiansQuadPtr[iDim] *
                                     gradDensityQuadPtr[iDim] * term;
                      }
                  }
              }
            else if (spinPolarizedFactor == 2)
              {
                if (d_basisOperationsPtrHost->cellsTypeFlag() != 2)
                  {
                    for (unsigned int iQuad = 0; iQuad < numberQuadraturePoints;
                         ++iQuad)
                      {
                        const double *inverseJacobiansQuadPtr =
                          d_basisOperationsPtrHost->inverseJacobiansBasisData()
                            .data() +
                          (d_basisOperationsPtrHost->cellsTypeFlag() == 0 ?
                             iCell * numberQuadraturePoints * 9 + iQuad * 9 :
                             iCell * 9);
                        const double *gradDensityQuadPtr =
                          gradDensityValue.data() + 6 * iQuad + 3 * spinIndex;
                        const double *gradDensityOtherQuadPtr =
                          gradDensityValue.data() + 6 * iQuad +
                          3 * (1 - spinIndex);
                        const double term =
                          (derExchEnergyWithSigmaVal[3 * iQuad +
                                                     2 * spinIndex] +
                           derCorrEnergyWithSigmaVal[3 * iQuad +
                                                     2 * spinIndex]) *
                          cellJxWPtr[iQuad];
                        const double termoff =
                          (derExchEnergyWithSigmaVal[3 * iQuad + 1] +
                           derCorrEnergyWithSigmaVal[3 * iQuad + 1]) *
                          cellJxWPtr[iQuad];
                        for (unsigned jDim = 0; jDim < 3; ++jDim)
                          for (unsigned iDim = 0; iDim < 3; ++iDim)
                            d_invJacderExcWithSigmaTimesGradRhoJxWHost
                              [iCell * numberQuadraturePoints * 3 + iQuad * 3 +
                               iDim] +=
                              inverseJacobiansQuadPtr[3 * jDim + iDim] *
                              (2.0 * gradDensityQuadPtr[jDim] * term +
                               gradDensityOtherQuadPtr[jDim] * termoff);
                      }
                  }
                else if (d_basisOperationsPtrHost->cellsTypeFlag() == 2)
                  {
                    for (unsigned int iQuad = 0; iQuad < numberQuadraturePoints;
                         ++iQuad)
                      {
                        const double *inverseJacobiansQuadPtr =
                          d_basisOperationsPtrHost->inverseJacobiansBasisData()
                            .data() +
                          iCell * 3;
                        const double *gradDensityQuadPtr =
                          gradDensityValue.data() + 6 * iQuad + 3 * spinIndex;
                        const double *gradDensityOtherQuadPtr =
                          gradDensityValue.data() + 6 * iQuad +
                          3 * (1 - spinIndex);
                        const double term =
                          (derExchEnergyWithSigmaVal[3 * iQuad +
                                                     2 * spinIndex] +
                           derCorrEnergyWithSigmaVal[3 * iQuad +
                                                     2 * spinIndex]) *
                          cellJxWPtr[iQuad];
                        const double termoff =
                          (derExchEnergyWithSigmaVal[3 * iQuad + 1] +
                           derCorrEnergyWithSigmaVal[3 * iQuad + 1]) *
                          cellJxWPtr[iQuad];
                        for (unsigned iDim = 0; iDim < 3; ++iDim)
                          d_invJacderExcWithSigmaTimesGradRhoJxWHost
                            [iCell * numberQuadraturePoints * 3 + iQuad * 3 +
                             iDim] = inverseJacobiansQuadPtr[iDim] *
                                     (2.0 * gradDensityQuadPtr[iDim] * term +
                                      gradDensityOtherQuadPtr[iDim] * termoff);
                      }
                  }
              }
          }
      }
#if defined(DFTFE_WITH_DEVICE)
    d_VeffJxW.resize(d_VeffJxWHost.size());
    d_VeffJxW.copyFrom(d_VeffJxWHost);
    d_invJacderExcWithSigmaTimesGradRhoJxW.resize(
      d_invJacderExcWithSigmaTimesGradRhoJxWHost.size());
    d_invJacderExcWithSigmaTimesGradRhoJxW.copyFrom(
      d_invJacderExcWithSigmaTimesGradRhoJxWHost);
#endif
  }
  template <dftfe::utils::MemorySpace memorySpace>
  void
  KohnShamHamiltonianOperator<memorySpace>::computeVEffExternalPotCorr(
    const std::map<dealii::CellId, std::vector<double>> &externalPotCorrValues)
  {
    d_basisOperationsPtrHost->reinit(0, 0, d_lpspQuadratureID, false);
    const unsigned int nCells = d_basisOperationsPtrHost->nCells();
    const int nQuadsPerCell   = d_basisOperationsPtrHost->nQuadsPerCell();
#if defined(DFTFE_WITH_DEVICE)
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      d_VeffExtPotJxWHost;
#else
    auto &d_VeffExtPotJxWHost = d_VeffExtPotJxW;
#endif
    d_VeffExtPotJxWHost.resize(nCells * nQuadsPerCell);

    for (unsigned int iCell = 0; iCell < nCells; ++iCell)
      {
        const auto &temp =
          externalPotCorrValues.find(d_basisOperationsPtrHost->cellID(iCell))
            ->second;
        const double *cellJxWPtr =
          d_basisOperationsPtrHost->JxWBasisData().data() +
          iCell * nQuadsPerCell;
        for (unsigned int iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
          d_VeffExtPotJxWHost[iCell * nQuadsPerCell + iQuad] =
            temp[iQuad] * cellJxWPtr[iQuad];
      }

#if defined(DFTFE_WITH_DEVICE)
    d_VeffExtPotJxW.resize(d_VeffExtPotJxWHost.size());
    d_VeffExtPotJxW.copyFrom(d_VeffExtPotJxWHost);
#endif
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  KohnShamHamiltonianOperator<memorySpace>::reinitkPointSpinIndex(
    const unsigned int kPointIndex,
    const unsigned int spinIndex)
  {
    d_kPointIndex = kPointIndex;
    d_spinIndex   = spinIndex;
    d_HamiltonianIndex =
      d_dftParamsPtr->memOptMode ?
        0 :
        kPointIndex * (d_dftParamsPtr->spinPolarized + 1) + spinIndex;
    if constexpr (dftfe::utils::MemorySpace::DEVICE == memorySpace)
      if (d_dftParamsPtr->isPseudopotential)
        d_ONCVnonLocalOperator->initialiseOperatorActionOnX(d_kPointIndex);
    if constexpr (dftfe::utils::MemorySpace::DEVICE == memorySpace)
      if (d_dftParamsPtr->isPseudopotential &&
          d_dftParamsPtr->useSinglePrecCheby)
        d_ONCVnonLocalOperatorSinglePrec->initialiseOperatorActionOnX(
          d_kPointIndex);
  }


  template <dftfe::utils::MemorySpace memorySpace>
  void
  KohnShamHamiltonianOperator<memorySpace>::reinitNumberWavefunctions(
    const unsigned int numWaveFunctions)
  {
    const unsigned int nCells       = d_basisOperationsPtr->nCells();
    const unsigned int nDofsPerCell = d_basisOperationsPtr->nDofsPerCell();
    if (d_cellWaveFunctionMatrixSrc.size() <
        nCells * nDofsPerCell * numWaveFunctions)
      d_cellWaveFunctionMatrixSrc.resize(nCells * nDofsPerCell *
                                         numWaveFunctions);
    if (d_dftParamsPtr->useSinglePrecCheby &&
        d_cellWaveFunctionMatrixSrcSinglePrec.size() <
          nCells * nDofsPerCell * numWaveFunctions)
      d_cellWaveFunctionMatrixSrcSinglePrec.resize(nCells * nDofsPerCell *
                                                   numWaveFunctions);
    if (d_cellWaveFunctionMatrixDst.size() <
        d_cellsBlockSizeHX * nDofsPerCell * numWaveFunctions)
      d_cellWaveFunctionMatrixDst.resize(d_cellsBlockSizeHX * nDofsPerCell *
                                         numWaveFunctions);
    if (d_dftParamsPtr->useSinglePrecCheby &&
        d_cellWaveFunctionMatrixDstSinglePrec.size() <
          d_cellsBlockSizeHX * nDofsPerCell * numWaveFunctions)
      d_cellWaveFunctionMatrixDstSinglePrec.resize(
        d_cellsBlockSizeHX * nDofsPerCell * numWaveFunctions);

    if (d_dftParamsPtr->isPseudopotential)
      {
        if constexpr (dftfe::utils::MemorySpace::DEVICE == memorySpace)
          {
            d_ONCVnonLocalOperator->initialiseFlattenedDataStructure(
              numWaveFunctions, d_ONCVNonLocalProjectorTimesVectorBlock);
            d_ONCVnonLocalOperator->initialiseCellWaveFunctionPointers(
              d_cellWaveFunctionMatrixSrc);
          }
        else
          d_ONCVnonLocalOperator->initialiseFlattenedDataStructure(
            numWaveFunctions, d_ONCVNonLocalProjectorTimesVectorBlock);
      }
    if (d_dftParamsPtr->isPseudopotential && d_dftParamsPtr->useSinglePrecCheby)
      {
        if constexpr (dftfe::utils::MemorySpace::DEVICE == memorySpace)
          {
            d_ONCVnonLocalOperatorSinglePrec->initialiseFlattenedDataStructure(
              numWaveFunctions,
              d_ONCVNonLocalProjectorTimesVectorBlockSinglePrec);
            d_ONCVnonLocalOperatorSinglePrec
              ->initialiseCellWaveFunctionPointers(
                d_cellWaveFunctionMatrixSrcSinglePrec);
          }
        else
          d_ONCVnonLocalOperatorSinglePrec->initialiseFlattenedDataStructure(
            numWaveFunctions,
            d_ONCVNonLocalProjectorTimesVectorBlockSinglePrec);
      }

    d_basisOperationsPtr->reinit(numWaveFunctions,
                                 d_cellsBlockSizeHX,
                                 d_densityQuadratureID,
                                 false,
                                 false);
    d_numVectorsInternal = numWaveFunctions;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  const MPI_Comm &
  KohnShamHamiltonianOperator<memorySpace>::getMPICommunicatorDomain()
  {
    return d_mpiCommDomain;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  dftUtils::constraintMatrixInfo<dftfe::utils::MemorySpace::HOST> *
  KohnShamHamiltonianOperator<memorySpace>::getOverloadedConstraintMatrixHost()
    const
  {
    return &(d_basisOperationsPtrHost
               ->d_constraintInfo[d_basisOperationsPtr->d_dofHandlerID]);
  }

  template <dftfe::utils::MemorySpace memorySpace>
  const dftfe::utils::MemoryStorage<double, memorySpace> &
  KohnShamHamiltonianOperator<memorySpace>::getInverseSqrtMassVector()
  {
    return d_basisOperationsPtr->inverseSqrtMassVectorBasisData();
  }

  template <dftfe::utils::MemorySpace memorySpace>
  const dftfe::utils::MemoryStorage<double, memorySpace> &
  KohnShamHamiltonianOperator<memorySpace>::getSqrtMassVector()
  {
    return d_basisOperationsPtr->sqrtMassVectorBasisData();
  }

  template <dftfe::utils::MemorySpace memorySpace>
  dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &
  KohnShamHamiltonianOperator<memorySpace>::getScratchFEMultivector(
    const unsigned int numVectors,
    const unsigned int index)
  {
    return d_basisOperationsPtr->getMultiVector(numVectors, index);
  }

  template <dftfe::utils::MemorySpace memorySpace>
  dftfe::linearAlgebra::MultiVector<dataTypes::numberFP32, memorySpace> &
  KohnShamHamiltonianOperator<memorySpace>::getScratchFEMultivectorSinglePrec(
    const unsigned int numVectors,
    const unsigned int index)
  {
    return d_basisOperationsPtr->getMultiVectorSinglePrec(numVectors, index);
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  KohnShamHamiltonianOperator<
    memorySpace>::computeCellHamiltonianMatrixExtPotContribution()
  {
    d_basisOperationsPtr->reinit(0,
                                 d_cellsBlockSizeHamiltonianConstruction,
                                 d_lpspQuadratureID,
                                 false,
                                 true);
    const unsigned int nCells       = d_basisOperationsPtr->nCells();
    const unsigned int nDofsPerCell = d_basisOperationsPtr->nDofsPerCell();
    d_cellHamiltonianMatrixExtPot.resize(nCells * nDofsPerCell * nDofsPerCell);
    d_basisOperationsPtr->computeWeightedCellMassMatrix(
      std::pair<unsigned int, unsigned int>(0, nCells),
      d_VeffExtPotJxW,
      d_cellHamiltonianMatrixExtPot);
    d_isExternalPotCorrHamiltonianComputed = true;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  KohnShamHamiltonianOperator<memorySpace>::computeCellHamiltonianMatrix(
    const bool onlyHPrimePartForFirstOrderDensityMatResponse)
  {
    if ((d_dftParamsPtr->isPseudopotential ||
         d_dftParamsPtr->smearedNuclearCharges) &&
        !onlyHPrimePartForFirstOrderDensityMatResponse)
      if (!d_isExternalPotCorrHamiltonianComputed)
        computeCellHamiltonianMatrixExtPotContribution();
    const unsigned int nCells           = d_basisOperationsPtr->nCells();
    const unsigned int nQuadsPerCell    = d_basisOperationsPtr->nQuadsPerCell();
    const unsigned int nDofsPerCell     = d_basisOperationsPtr->nDofsPerCell();
    const double       scalarCoeffAlpha = 1.0;
    const double       scalarCoeffHalf  = 0.5;
    d_basisOperationsPtr->reinit(0,
                                 d_cellsBlockSizeHamiltonianConstruction,
                                 d_densityQuadratureID,
                                 false,
                                 true);
    for (unsigned int iCell = 0; iCell < nCells;
         iCell += d_cellsBlockSizeHamiltonianConstruction)
      {
        std::pair<unsigned int, unsigned int> cellRange(
          iCell,
          std::min(iCell + d_cellsBlockSizeHamiltonianConstruction, nCells));
        tempHamMatrixRealBlock.setValue(0.0);
        if ((d_dftParamsPtr->isPseudopotential ||
             d_dftParamsPtr->smearedNuclearCharges) &&
            !onlyHPrimePartForFirstOrderDensityMatResponse)
          {
            d_BLASWrapperPtr->xcopy(nDofsPerCell * nDofsPerCell *
                                      (cellRange.second - cellRange.first),
                                    d_cellHamiltonianMatrixExtPot.data() +
                                      cellRange.first * nDofsPerCell *
                                        nDofsPerCell,
                                    1,
                                    tempHamMatrixRealBlock.data(),
                                    1);
          }
        d_basisOperationsPtr->computeWeightedCellMassMatrix(
          cellRange, d_VeffJxW, tempHamMatrixRealBlock);
        if (d_excManagerPtr->getDensityBasedFamilyType() ==
            densityFamilyType::GGA)
          d_basisOperationsPtr->computeWeightedCellNjGradNiPlusNiGradNjMatrix(
            cellRange,
            d_invJacderExcWithSigmaTimesGradRhoJxW,
            tempHamMatrixRealBlock);
        if (!onlyHPrimePartForFirstOrderDensityMatResponse)
          d_BLASWrapperPtr->xaxpy(
            nDofsPerCell * nDofsPerCell * (cellRange.second - cellRange.first),
            &scalarCoeffHalf,
            d_basisOperationsPtr->cellStiffnessMatrixBasisData().data() +
              cellRange.first * nDofsPerCell * nDofsPerCell,
            1,
            tempHamMatrixRealBlock.data(),
            1);

        if constexpr (std::is_same<dataTypes::number,
                                   std::complex<double>>::value)
          {
            tempHamMatrixImagBlock.setValue(0.0);
            if (!onlyHPrimePartForFirstOrderDensityMatResponse)
              {
                const double *kPointCoors =
                  d_kPointCoordinates.data() + 3 * d_kPointIndex;
                const double kSquareTimesHalf =
                  0.5 * (kPointCoors[0] * kPointCoors[0] +
                         kPointCoors[1] * kPointCoors[1] +
                         kPointCoors[2] * kPointCoors[2]);
                d_BLASWrapperPtr->xaxpy(
                  nDofsPerCell * nDofsPerCell *
                    (cellRange.second - cellRange.first),
                  &kSquareTimesHalf,
                  d_basisOperationsPtr->cellMassMatrixBasisData().data() +
                    cellRange.first * nDofsPerCell * nDofsPerCell,
                  1,
                  tempHamMatrixRealBlock.data(),
                  1);
                d_basisOperationsPtr->computeWeightedCellNjGradNiMatrix(
                  cellRange,
                  d_invJacKPointTimesJxW[d_kPointIndex],
                  tempHamMatrixImagBlock);
              }
            d_BLASWrapperPtr->copyRealArrsToComplexArr(
              nDofsPerCell * nDofsPerCell *
                (cellRange.second - cellRange.first),
              tempHamMatrixRealBlock.data(),
              tempHamMatrixImagBlock.data(),
              d_cellHamiltonianMatrix[d_HamiltonianIndex].data() +
                cellRange.first * nDofsPerCell * nDofsPerCell);
          }
        else
          {
            d_BLASWrapperPtr->xcopy(
              nDofsPerCell * nDofsPerCell *
                (cellRange.second - cellRange.first),
              tempHamMatrixRealBlock.data(),
              1,
              d_cellHamiltonianMatrix[d_HamiltonianIndex].data() +
                cellRange.first * nDofsPerCell * nDofsPerCell,
              1);
          }
      }
    if (d_dftParamsPtr->useSinglePrecCheby)
      d_BLASWrapperPtr->copyValueType1ArrToValueType2Arr(
        d_cellHamiltonianMatrix[d_HamiltonianIndex].size(),
        d_cellHamiltonianMatrix[d_HamiltonianIndex].data(),
        d_cellHamiltonianMatrixSinglePrec[d_HamiltonianIndex].data());
    if (d_dftParamsPtr->memOptMode)
      if ((d_dftParamsPtr->isPseudopotential ||
           d_dftParamsPtr->smearedNuclearCharges) &&
          !onlyHPrimePartForFirstOrderDensityMatResponse)
        {
          d_cellHamiltonianMatrixExtPot.clear();
          d_isExternalPotCorrHamiltonianComputed = false;
        }
  }
  template <dftfe::utils::MemorySpace memorySpace>
  void
  KohnShamHamiltonianOperator<memorySpace>::HX(
    dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &src,
    const double                                                       scalarHX,
    const double                                                       scalarY,
    const double                                                       scalarX,
    dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &dst,
    const bool onlyHPrimePartForFirstOrderDensityMatResponse)
  {
    const unsigned int numCells       = d_basisOperationsPtr->nCells();
    const unsigned int numDoFsPerCell = d_basisOperationsPtr->nDofsPerCell();
    const unsigned int numberWavefunctions = src.numVectors();
    if (d_numVectorsInternal != numberWavefunctions)
      reinitNumberWavefunctions(numberWavefunctions);

    if (d_basisOperationsPtr->d_nVectors != numberWavefunctions)
      d_basisOperationsPtr->reinit(numberWavefunctions,
                                   d_cellsBlockSizeHX,
                                   d_densityQuadratureID,
                                   false,
                                   false);

    d_BLASWrapperPtr->axpby(src.locallyOwnedSize() * src.numVectors(),
                            scalarX,
                            src.data(),
                            scalarY,
                            dst.data());

    src.updateGhostValues();
    inverseSqrtMassVectorScaledConstraintsNoneDataInfoPtr->distribute(src);
    const dataTypes::number scalarCoeffAlpha = dataTypes::number(1.0),
                            scalarCoeffBeta  = dataTypes::number(0.0);

    if constexpr (memorySpace == dftfe::utils::MemorySpace::HOST)
      if (d_dftParamsPtr->isPseudopotential)
        d_ONCVnonLocalOperator->initialiseOperatorActionOnX(d_kPointIndex);
    const bool hasNonlocalComponents =
      d_dftParamsPtr->isPseudopotential &&
      (d_ONCVnonLocalOperator->getTotalNonLocalElementsInCurrentProcessor() >
       0) &&
      !onlyHPrimePartForFirstOrderDensityMatResponse;

    for (unsigned int iCell = 0; iCell < numCells; iCell += d_cellsBlockSizeHX)
      {
        std::pair<unsigned int, unsigned int> cellRange(
          iCell, std::min(iCell + d_cellsBlockSizeHX, numCells));
        d_BLASWrapperPtr->stridedBlockScaleCopy(
          numberWavefunctions,
          numDoFsPerCell * (cellRange.second - cellRange.first),
          1.0,
          d_basisOperationsPtr->cellInverseSqrtMassVectorBasisData().data() +
            cellRange.first * numDoFsPerCell,
          src.data(),
          d_cellWaveFunctionMatrixSrc.data() +
            cellRange.first * numDoFsPerCell * numberWavefunctions,
          d_basisOperationsPtr->d_flattenedCellDofIndexToProcessDofIndexMap
              .data() +
            cellRange.first * numDoFsPerCell);
        if (hasNonlocalComponents)
          d_ONCVnonLocalOperator->applyCconjtransOnX(
            d_cellWaveFunctionMatrixSrc.data() +
              cellRange.first * numDoFsPerCell * numberWavefunctions,
            cellRange);
      }
    if (d_dftParamsPtr->isPseudopotential &&
        !onlyHPrimePartForFirstOrderDensityMatResponse)
      {
        d_ONCVNonLocalProjectorTimesVectorBlock.setValue(0);
        d_ONCVnonLocalOperator->applyAllReduceOnCconjtransX(
          d_ONCVNonLocalProjectorTimesVectorBlock);
        d_ONCVnonLocalOperator->applyVOnCconjtransX(
          CouplingStructure::diagonal,
          d_oncvClassPtr->getCouplingMatrix(),
          d_ONCVNonLocalProjectorTimesVectorBlock,
          true);
      }

    for (unsigned int iCell = 0; iCell < numCells; iCell += d_cellsBlockSizeHX)
      {
        std::pair<unsigned int, unsigned int> cellRange(
          iCell, std::min(iCell + d_cellsBlockSizeHX, numCells));

        d_BLASWrapperPtr->xgemmStridedBatched(
          'N',
          'N',
          numberWavefunctions,
          numDoFsPerCell,
          numDoFsPerCell,
          &scalarCoeffAlpha,
          d_cellWaveFunctionMatrixSrc.data() +
            cellRange.first * numDoFsPerCell * numberWavefunctions,
          numberWavefunctions,
          numDoFsPerCell * numberWavefunctions,
          d_cellHamiltonianMatrix[d_HamiltonianIndex].data() +
            cellRange.first * numDoFsPerCell * numDoFsPerCell,
          numDoFsPerCell,
          numDoFsPerCell * numDoFsPerCell,
          &scalarCoeffBeta,
          d_cellWaveFunctionMatrixDst.data(),
          numberWavefunctions,
          numDoFsPerCell * numberWavefunctions,
          cellRange.second - cellRange.first);
        if (hasNonlocalComponents)
          d_ONCVnonLocalOperator->applyCOnVCconjtransX(
            d_cellWaveFunctionMatrixDst.data(), cellRange);
        d_BLASWrapperPtr->axpyStridedBlockAtomicAdd(
          numberWavefunctions,
          numDoFsPerCell * (cellRange.second - cellRange.first),
          scalarHX,
          d_basisOperationsPtr->cellInverseSqrtMassVectorBasisData().data() +
            cellRange.first * numDoFsPerCell,
          d_cellWaveFunctionMatrixDst.data(),
          dst.data(),
          d_basisOperationsPtr->d_flattenedCellDofIndexToProcessDofIndexMap
              .data() +
            cellRange.first * numDoFsPerCell);
      }

    inverseSqrtMassVectorScaledConstraintsNoneDataInfoPtr
      ->distribute_slave_to_master(dst);

    src.zeroOutGhosts();
    inverseSqrtMassVectorScaledConstraintsNoneDataInfoPtr->set_zero(src);
    dst.accumulateAddLocallyOwned();
    dst.zeroOutGhosts();
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  KohnShamHamiltonianOperator<memorySpace>::HXCheby(
    dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &src,
    const double                                                       scalarHX,
    const double                                                       scalarY,
    const double                                                       scalarX,
    dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &dst,
    const bool onlyHPrimePartForFirstOrderDensityMatResponse,
    const bool skip1,
    const bool skip2,
    const bool skip3)
  {
    const unsigned int numCells       = d_basisOperationsPtr->nCells();
    const unsigned int numDoFsPerCell = d_basisOperationsPtr->nDofsPerCell();
    const unsigned int numberWavefunctions = src.numVectors();
    if (d_numVectorsInternal != numberWavefunctions)
      reinitNumberWavefunctions(numberWavefunctions);

    if (d_basisOperationsPtr->d_nVectors != numberWavefunctions)
      d_basisOperationsPtr->reinit(numberWavefunctions,
                                   d_cellsBlockSizeHX,
                                   d_densityQuadratureID,
                                   false,
                                   false);
    const bool hasNonlocalComponents =
      d_dftParamsPtr->isPseudopotential &&
      (d_ONCVnonLocalOperator->getTotalNonLocalElementsInCurrentProcessor() >
       0) &&
      !onlyHPrimePartForFirstOrderDensityMatResponse;
    const dataTypes::number scalarCoeffAlpha = dataTypes::number(1.0),
                            scalarCoeffBeta  = dataTypes::number(0.0);

    if (!skip1 && !skip2 && !skip3)
      src.updateGhostValues();
    if (!skip1)
      {
        d_basisOperationsPtr->distribute(src);
        if constexpr (memorySpace == dftfe::utils::MemorySpace::HOST)
          if (d_dftParamsPtr->isPseudopotential)
            d_ONCVnonLocalOperator->initialiseOperatorActionOnX(d_kPointIndex);
        for (unsigned int iCell = 0; iCell < numCells;
             iCell += d_cellsBlockSizeHX)
          {
            std::pair<unsigned int, unsigned int> cellRange(
              iCell, std::min(iCell + d_cellsBlockSizeHX, numCells));
            d_BLASWrapperPtr->stridedCopyToBlock(
              numberWavefunctions,
              numDoFsPerCell * (cellRange.second - cellRange.first),
              src.data(),
              d_cellWaveFunctionMatrixSrc.data() +
                cellRange.first * numDoFsPerCell * numberWavefunctions,
              d_basisOperationsPtr->d_flattenedCellDofIndexToProcessDofIndexMap
                  .data() +
                cellRange.first * numDoFsPerCell);
            if (hasNonlocalComponents)
              d_ONCVnonLocalOperator->applyCconjtransOnX(
                d_cellWaveFunctionMatrixSrc.data() +
                  cellRange.first * numDoFsPerCell * numberWavefunctions,
                cellRange);
          }
      }
    if (!skip2)
      {
        if (d_dftParamsPtr->isPseudopotential &&
            !onlyHPrimePartForFirstOrderDensityMatResponse)
          {
            d_ONCVNonLocalProjectorTimesVectorBlock.setValue(0);
            d_ONCVnonLocalOperator->applyAllReduceOnCconjtransX(
              d_ONCVNonLocalProjectorTimesVectorBlock, true);
            d_ONCVNonLocalProjectorTimesVectorBlock
              .accumulateAddLocallyOwnedBegin();
          }
        src.zeroOutGhosts();
        inverseMassVectorScaledConstraintsNoneDataInfoPtr->set_zero(src);
        if (d_dftParamsPtr->isPseudopotential &&
            !onlyHPrimePartForFirstOrderDensityMatResponse)
          {
            d_ONCVNonLocalProjectorTimesVectorBlock
              .accumulateAddLocallyOwnedEnd();
            d_ONCVNonLocalProjectorTimesVectorBlock.updateGhostValuesBegin();
          }
        d_BLASWrapperPtr->axpby(src.locallyOwnedSize() * src.numVectors(),
                                scalarX,
                                src.data(),
                                scalarY,
                                dst.data());
        if (d_dftParamsPtr->isPseudopotential &&
            !onlyHPrimePartForFirstOrderDensityMatResponse)
          {
            d_ONCVNonLocalProjectorTimesVectorBlock.updateGhostValuesEnd();
            d_ONCVnonLocalOperator->applyVOnCconjtransX(
              CouplingStructure::diagonal,
              d_oncvClassPtr->getCouplingMatrix(),
              d_ONCVNonLocalProjectorTimesVectorBlock,
              true);
          }
      }
    if (!skip3)
      {
        for (unsigned int iCell = 0; iCell < numCells;
             iCell += d_cellsBlockSizeHX)
          {
            std::pair<unsigned int, unsigned int> cellRange(
              iCell, std::min(iCell + d_cellsBlockSizeHX, numCells));

            d_BLASWrapperPtr->xgemmStridedBatched(
              'N',
              'N',
              numberWavefunctions,
              numDoFsPerCell,
              numDoFsPerCell,
              &scalarCoeffAlpha,
              d_cellWaveFunctionMatrixSrc.data() +
                cellRange.first * numDoFsPerCell * numberWavefunctions,
              numberWavefunctions,
              numDoFsPerCell * numberWavefunctions,
              d_cellHamiltonianMatrix[d_HamiltonianIndex].data() +
                cellRange.first * numDoFsPerCell * numDoFsPerCell,
              numDoFsPerCell,
              numDoFsPerCell * numDoFsPerCell,
              &scalarCoeffBeta,
              d_cellWaveFunctionMatrixDst.data(),
              numberWavefunctions,
              numDoFsPerCell * numberWavefunctions,
              cellRange.second - cellRange.first);
            if (hasNonlocalComponents)
              d_ONCVnonLocalOperator->applyCOnVCconjtransX(
                d_cellWaveFunctionMatrixDst.data(), cellRange);
            d_BLASWrapperPtr->axpyStridedBlockAtomicAdd(
              numberWavefunctions,
              numDoFsPerCell * (cellRange.second - cellRange.first),
              scalarHX,
              d_basisOperationsPtr->cellInverseMassVectorBasisData().data() +
                cellRange.first * numDoFsPerCell,
              d_cellWaveFunctionMatrixDst.data(),
              dst.data(),
              d_basisOperationsPtr->d_flattenedCellDofIndexToProcessDofIndexMap
                  .data() +
                cellRange.first * numDoFsPerCell);
          }

        inverseMassVectorScaledConstraintsNoneDataInfoPtr
          ->distribute_slave_to_master(dst);
      }
    if (!skip1 && !skip2 && !skip3)
      {
        dst.accumulateAddLocallyOwned();
        dst.zeroOutGhosts();
      }
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  KohnShamHamiltonianOperator<memorySpace>::HXCheby(
    dftfe::linearAlgebra::MultiVector<dataTypes::numberFP32, memorySpace> &src,
    const double scalarHX,
    const double scalarY,
    const double scalarX,
    dftfe::linearAlgebra::MultiVector<dataTypes::numberFP32, memorySpace> &dst,
    const bool onlyHPrimePartForFirstOrderDensityMatResponse,
    const bool skip1,
    const bool skip2,
    const bool skip3)
  {
    const unsigned int numCells       = d_basisOperationsPtr->nCells();
    const unsigned int numDoFsPerCell = d_basisOperationsPtr->nDofsPerCell();
    const unsigned int numberWavefunctions = src.numVectors();
    if (d_numVectorsInternal != numberWavefunctions)
      reinitNumberWavefunctions(numberWavefunctions);

    if (d_basisOperationsPtr->d_nVectors != numberWavefunctions)
      d_basisOperationsPtr->reinit(numberWavefunctions,
                                   d_cellsBlockSizeHX,
                                   d_densityQuadratureID,
                                   false,
                                   false);
    const bool hasNonlocalComponents =
      d_dftParamsPtr->isPseudopotential &&
      (d_ONCVnonLocalOperatorSinglePrec
         ->getTotalNonLocalElementsInCurrentProcessor() > 0) &&
      !onlyHPrimePartForFirstOrderDensityMatResponse;
    const dataTypes::numberFP32 scalarCoeffAlpha = dataTypes::numberFP32(1.0),
                                scalarCoeffBeta  = dataTypes::numberFP32(0.0);

    if (!skip1 && !skip2 && !skip3)
      src.updateGhostValues();
    if (!skip1)
      {
        d_basisOperationsPtr
          ->d_constraintInfo[d_basisOperationsPtr->d_dofHandlerID]
          .distribute(src);
        if constexpr (memorySpace == dftfe::utils::MemorySpace::HOST)
          if (d_dftParamsPtr->isPseudopotential)
            d_ONCVnonLocalOperatorSinglePrec->initialiseOperatorActionOnX(
              d_kPointIndex);
        for (unsigned int iCell = 0; iCell < numCells;
             iCell += d_cellsBlockSizeHX)
          {
            std::pair<unsigned int, unsigned int> cellRange(
              iCell, std::min(iCell + d_cellsBlockSizeHX, numCells));
            d_BLASWrapperPtr->stridedCopyToBlock(
              numberWavefunctions,
              numDoFsPerCell * (cellRange.second - cellRange.first),
              src.data(),
              d_cellWaveFunctionMatrixSrcSinglePrec.data() +
                cellRange.first * numDoFsPerCell * numberWavefunctions,
              d_basisOperationsPtr->d_flattenedCellDofIndexToProcessDofIndexMap
                  .data() +
                cellRange.first * numDoFsPerCell);
            if (hasNonlocalComponents)
              d_ONCVnonLocalOperatorSinglePrec->applyCconjtransOnX(
                d_cellWaveFunctionMatrixSrcSinglePrec.data() +
                  cellRange.first * numDoFsPerCell * numberWavefunctions,
                cellRange);
          }
      }
    if (!skip2)
      {
        if (d_dftParamsPtr->isPseudopotential &&
            !onlyHPrimePartForFirstOrderDensityMatResponse)
          {
            d_ONCVNonLocalProjectorTimesVectorBlockSinglePrec.setValue(0);
            d_ONCVnonLocalOperatorSinglePrec->applyAllReduceOnCconjtransX(
              d_ONCVNonLocalProjectorTimesVectorBlockSinglePrec, true);
            d_ONCVNonLocalProjectorTimesVectorBlockSinglePrec
              .accumulateAddLocallyOwnedBegin();
          }
        src.zeroOutGhosts();
        inverseMassVectorScaledConstraintsNoneDataInfoPtr->set_zero(src);
        if (d_dftParamsPtr->isPseudopotential &&
            !onlyHPrimePartForFirstOrderDensityMatResponse)
          {
            d_ONCVNonLocalProjectorTimesVectorBlockSinglePrec
              .accumulateAddLocallyOwnedEnd();
            d_ONCVNonLocalProjectorTimesVectorBlockSinglePrec
              .updateGhostValuesBegin();
          }
        d_BLASWrapperPtr->axpby(src.locallyOwnedSize() * src.numVectors(),
                                scalarX,
                                src.data(),
                                scalarY,
                                dst.data());
        if (d_dftParamsPtr->isPseudopotential &&
            !onlyHPrimePartForFirstOrderDensityMatResponse)
          {
            d_ONCVNonLocalProjectorTimesVectorBlockSinglePrec
              .updateGhostValuesEnd();
            d_ONCVnonLocalOperatorSinglePrec->applyVOnCconjtransX(
              CouplingStructure::diagonal,
              d_oncvClassPtr->getCouplingMatrixSinglePrec(),
              d_ONCVNonLocalProjectorTimesVectorBlockSinglePrec,
              true);
          }
      }
    if (!skip3)
      {
        for (unsigned int iCell = 0; iCell < numCells;
             iCell += d_cellsBlockSizeHX)
          {
            std::pair<unsigned int, unsigned int> cellRange(
              iCell, std::min(iCell + d_cellsBlockSizeHX, numCells));

            d_BLASWrapperPtr->xgemmStridedBatched(
              'N',
              'N',
              numberWavefunctions,
              numDoFsPerCell,
              numDoFsPerCell,
              &scalarCoeffAlpha,
              d_cellWaveFunctionMatrixSrcSinglePrec.data() +
                cellRange.first * numDoFsPerCell * numberWavefunctions,
              numberWavefunctions,
              numDoFsPerCell * numberWavefunctions,
              d_cellHamiltonianMatrixSinglePrec[d_HamiltonianIndex].data() +
                cellRange.first * numDoFsPerCell * numDoFsPerCell,
              numDoFsPerCell,
              numDoFsPerCell * numDoFsPerCell,
              &scalarCoeffBeta,
              d_cellWaveFunctionMatrixDstSinglePrec.data(),
              numberWavefunctions,
              numDoFsPerCell * numberWavefunctions,
              cellRange.second - cellRange.first);
            if (hasNonlocalComponents)
              d_ONCVnonLocalOperatorSinglePrec->applyCOnVCconjtransX(
                d_cellWaveFunctionMatrixDstSinglePrec.data(), cellRange);
            d_BLASWrapperPtr->axpyStridedBlockAtomicAdd(
              numberWavefunctions,
              numDoFsPerCell * (cellRange.second - cellRange.first),
              scalarHX,
              d_basisOperationsPtr->cellInverseMassVectorBasisData().data() +
                cellRange.first * numDoFsPerCell,
              d_cellWaveFunctionMatrixDstSinglePrec.data(),
              dst.data(),
              d_basisOperationsPtr->d_flattenedCellDofIndexToProcessDofIndexMap
                  .data() +
                cellRange.first * numDoFsPerCell);
          }

        inverseMassVectorScaledConstraintsNoneDataInfoPtr
          ->distribute_slave_to_master(dst);
      }
    if (!skip1 && !skip2 && !skip3)
      {
        dst.accumulateAddLocallyOwned();
        dst.zeroOutGhosts();
      }
  }


  template class KohnShamHamiltonianOperator<dftfe::utils::MemorySpace::HOST>;
#if defined(DFTFE_WITH_DEVICE)
  template class KohnShamHamiltonianOperator<dftfe::utils::MemorySpace::DEVICE>;
#endif

} // namespace dftfe
