// ---------------------------------------------------------------------
//
// Copyright (c) 2017-18 The Regents of the University of Michigan and DFT-FE
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
#include <force.h>
#include <dft.h>
#include <eshelbyTensor.h>
#include <eshelbyTensorSpinPolarized.h>
#include <dftUtils.h>
#include <forceWfcContractions.h>

namespace dftfe
{
  //
  // compute configurational force contribution from all terms except the
  // nuclear self energy
  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  forceClass<FEOrder, FEOrderElectro, memorySpace>::
    computeConfigurationalForceEEshelbyTensorFPSPFnlLinFE(
      const dealii::MatrixFree<3, double> &matrixFreeData,
      const unsigned int                   eigenDofHandlerIndex,
      const unsigned int                   smearedChargeQuadratureId,
      const unsigned int                   lpspQuadratureIdElectro,
      const dealii::MatrixFree<3, double> &matrixFreeDataElectro,
      const unsigned int                   phiTotDofHandlerIndexElectro,
      const distributedCPUVec<double> &    phiTotRhoOutElectro,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &rhoOutValues,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &gradRhoOutValues,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &rhoTotalOutValuesLpsp,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &gradRhoTotalOutValuesLpsp,
      const std::map<dealii::CellId, std::vector<double>> &rhoCoreValues,
      const std::map<dealii::CellId, std::vector<double>> &gradRhoCoreValues,
      const std::map<dealii::CellId, std::vector<double>> &hessianRhoCoreValues,
      const std::map<unsigned int,
                     std::map<dealii::CellId, std::vector<double>>>
        &gradRhoCoreAtoms,
      const std::map<unsigned int,
                     std::map<dealii::CellId, std::vector<double>>>
        &                                                  hessianRhoCoreAtoms,
      const std::map<dealii::CellId, std::vector<double>> &pseudoVLocElectro,
      const std::map<unsigned int,
                     std::map<dealii::CellId, std::vector<double>>>
        &                                              pseudoVLocAtomsElectro,
      const vselfBinsManager<FEOrder, FEOrderElectro> &vselfBinsManagerElectro)
  {
    int this_process;
    MPI_Comm_rank(d_mpiCommParent, &this_process);
    MPI_Barrier(d_mpiCommParent);
    double forcetotal_time = MPI_Wtime();

    MPI_Barrier(d_mpiCommParent);
    double init_time = MPI_Wtime();

    const unsigned int numberGlobalAtoms = dftPtr->atomLocations.size();
    std::map<unsigned int, std::vector<double>> forceContributionFnlGammaAtoms;

    const bool isPseudopotential = d_dftParams.isPseudopotential;

    dealii::FEEvaluation<
      3,
      1,
      C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>(),
      3>
      forceEval(matrixFreeData,
                d_forceDofHandlerIndex,
                dftPtr->d_densityQuadratureId);
    dealii::FEEvaluation<3,
                         1,
                         C_num1DQuadNLPSP<FEOrder>() * C_numCopies1DQuadNLPSP(),
                         3>
      forceEvalNLP(matrixFreeData,
                   d_forceDofHandlerIndex,
                   dftPtr->d_nlpspQuadratureId);


    std::map<unsigned int, std::vector<double>>
      forceContributionShadowLocalGammaAtoms;

    const unsigned int numQuadPoints    = forceEval.n_q_points;
    const unsigned int numQuadPointsNLP = forceEvalNLP.n_q_points;

    const unsigned int numEigenVectors = dftPtr->d_numEigenValues;
    const unsigned int numKPoints      = dftPtr->d_kPointWeights.size();
    dealii::DoFHandler<3>::active_cell_iterator           subCellPtr;
    dealii::Tensor<1, 2, dealii::VectorizedArray<double>> zeroTensor1;
    zeroTensor1[0] = dealii::make_vectorized_array(0.0);
    zeroTensor1[1] = dealii::make_vectorized_array(0.0);
    dealii::Tensor<1, 2, dealii::Tensor<1, 3, dealii::VectorizedArray<double>>>
                                                          zeroTensor2;
    dealii::Tensor<1, 3, dealii::VectorizedArray<double>> zeroTensor3;
    dealii::Tensor<2, 3, dealii::VectorizedArray<double>> zeroTensor4;
    for (unsigned int idim = 0; idim < 3; idim++)
      {
        zeroTensor2[0][idim] = dealii::make_vectorized_array(0.0);
        zeroTensor2[1][idim] = dealii::make_vectorized_array(0.0);
        zeroTensor3[idim]    = dealii::make_vectorized_array(0.0);
      }
    for (unsigned int idim = 0; idim < 3; idim++)
      {
        for (unsigned int jdim = 0; jdim < 3; jdim++)
          {
            zeroTensor4[idim][jdim] = dealii::make_vectorized_array(0.0);
          }
      }

    const double spinPolarizedFactor =
      (d_dftParams.spinPolarized == 1) ? 0.5 : 1.0;
    const dealii::VectorizedArray<double> spinPolarizedFactorVect =
      (d_dftParams.spinPolarized == 1) ? dealii::make_vectorized_array(0.5) :
                                         dealii::make_vectorized_array(1.0);

    const unsigned int numPhysicalCells = matrixFreeData.n_physical_cells();

    std::map<dealii::CellId, unsigned int> cellIdToCellNumberMap;

    dealii::DoFHandler<3>::active_cell_iterator cell = dftPtr->dofHandler
                                                         .begin_active(),
                                                endc = dftPtr->dofHandler.end();
    unsigned int iElem                               = 0;
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        {
          cellIdToCellNumberMap[cell->id()] = iElem;
          iElem++;
        }



    // band group parallelization data structures
    const unsigned int numberBandGroups =
      dealii::Utilities::MPI::n_mpi_processes(dftPtr->interBandGroupComm);
    const unsigned int bandGroupTaskId =
      dealii::Utilities::MPI::this_mpi_process(dftPtr->interBandGroupComm);
    std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
    dftUtils::createBandParallelizationIndices(dftPtr->interBandGroupComm,
                                               numEigenVectors,
                                               bandGroupLowHighPlusOneIndices);


    const unsigned int localVectorSize =
      matrixFreeData.get_vector_partitioner()->locally_owned_size();

    const unsigned int numMacroCells = matrixFreeData.n_cell_batches();


    std::vector<std::vector<double>> partialOccupancies(
      numKPoints,
      std::vector<double>((1 + d_dftParams.spinPolarized) * numEigenVectors,
                          0.0));
    for (unsigned int spinIndex = 0;
         spinIndex < (1 + d_dftParams.spinPolarized);
         ++spinIndex)
      for (unsigned int kPoint = 0; kPoint < numKPoints; ++kPoint)
        for (unsigned int iWave = 0; iWave < numEigenVectors; ++iWave)
          {
            const double eigenValue =
              dftPtr->eigenValues[kPoint][numEigenVectors * spinIndex + iWave];
            partialOccupancies[kPoint][numEigenVectors * spinIndex + iWave] =
              dftUtils::getPartialOccupancy(eigenValue,
                                            dftPtr->fermiEnergy,
                                            C_kb,
                                            d_dftParams.TVal);

            if (d_dftParams.constraintMagnetization)
              {
                partialOccupancies[kPoint]
                                  [numEigenVectors * spinIndex + iWave] = 1.0;
                if (spinIndex == 0)
                  {
                    if (eigenValue > dftPtr->fermiEnergyUp)
                      partialOccupancies[kPoint][numEigenVectors * spinIndex +
                                                 iWave] = 0.0;
                  }
                else if (spinIndex == 1)
                  {
                    if (eigenValue > dftPtr->fermiEnergyDown)
                      partialOccupancies[kPoint][numEigenVectors * spinIndex +
                                                 iWave] = 0.0;
                  }
              }
          }

    MPI_Barrier(d_mpiCommParent);
    init_time = MPI_Wtime() - init_time;

    for (unsigned int spinIndex = 0;
         spinIndex < (1 + d_dftParams.spinPolarized);
         ++spinIndex)
      {
        std::vector<double> elocWfcEshelbyTensorQuadValuesH(
          numKPoints * numPhysicalCells * numQuadPoints * 9, 0.0);
        std::vector<dataTypes::number>
          projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattened;

#ifdef USE_COMPLEX
        std::vector<dataTypes::number>
          projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattened;
#endif

        if (isPseudopotential)
          {
            projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattened
              .resize(numKPoints *
                        dftPtr->d_oncvClassPtr->getNonLocalOperator()
                          ->getTotalNonTrivialSphericalFnsOverAllCells() *
                        numQuadPointsNLP * 3,
                      dataTypes::number(0.0));

#ifdef USE_COMPLEX
            projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattened
              .resize(numKPoints *
                        dftPtr->d_oncvClassPtr->getNonLocalOperator()
                          ->getTotalNonTrivialSphericalFnsOverAllCells() *
                        numQuadPointsNLP,
                      dataTypes::number(0.0));
#endif
          }



#if defined(DFTFE_WITH_DEVICE)
        if constexpr (dftfe::utils::MemorySpace::DEVICE == memorySpace)
          {
            MPI_Barrier(d_mpiCommParent);
            double device_time = MPI_Wtime();
            force::wfcContractionsForceKernelsAllH(
              dftPtr->d_basisOperationsPtrDevice,
              dftPtr->d_densityQuadratureId,
              dftPtr->d_nlpspQuadratureId,
              dftPtr->d_BLASWrapperPtr,
              dftPtr->d_oncvClassPtr,
              dftPtr->d_eigenVectorsFlattenedDevice.begin(),
              d_dftParams.spinPolarized,
              spinIndex,
              dftPtr->eigenValues,
              partialOccupancies,
              dftPtr->d_kPointCoordinates,
              localVectorSize,
              numEigenVectors,
              numPhysicalCells,
              numQuadPoints,
              numQuadPointsNLP,
              elocWfcEshelbyTensorQuadValuesH.data(),
              projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattened
                .data(),
#  ifdef USE_COMPLEX
              projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattened
                .data(),
#  endif
              d_mpiCommParent,
              dftPtr->interBandGroupComm,
              isPseudopotential,
              d_dftParams.floatingNuclearCharges,
              false,
              d_dftParams);

            MPI_Barrier(d_mpiCommParent);
            device_time = MPI_Wtime() - device_time;

            if (this_process == 0 && d_dftParams.verbosity >= 4)
              std::cout << "Time for wfc contractions in forces: "
                        << device_time << std::endl;
          }
        else
#endif
          {
            MPI_Barrier(d_mpiCommParent);
            double host_time = MPI_Wtime();
            force::wfcContractionsForceKernelsAllH(
              dftPtr->d_basisOperationsPtrHost,
              dftPtr->d_densityQuadratureId,
              dftPtr->d_nlpspQuadratureId,
              dftPtr->d_BLASWrapperPtrHost,
              dftPtr->d_oncvClassPtr,
              dftPtr->d_eigenVectorsFlattenedHost.begin(),
              d_dftParams.spinPolarized,
              spinIndex,
              dftPtr->eigenValues,
              partialOccupancies,
              dftPtr->d_kPointCoordinates,
              localVectorSize,
              numEigenVectors,
              numPhysicalCells,
              numQuadPoints,
              numQuadPointsNLP,
              elocWfcEshelbyTensorQuadValuesH.data(),
              projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattened
                .data(),
#ifdef USE_COMPLEX
              projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattened
                .data(),
#endif
              d_mpiCommParent,
              dftPtr->interBandGroupComm,
              isPseudopotential,
              d_dftParams.floatingNuclearCharges,
              false,
              d_dftParams);

            MPI_Barrier(d_mpiCommParent);
            host_time = MPI_Wtime() - host_time;

            if (this_process == 0 && d_dftParams.verbosity >= 4)
              std::cout << "Time for wfc contractions in forces: " << host_time
                        << std::endl;
          }


        if (!d_dftParams.floatingNuclearCharges)
          {
            dealii::AlignedVector<
              dealii::Tensor<2, 3, dealii::VectorizedArray<double>>>
              EQuad(numQuadPoints, zeroTensor4);
            for (unsigned int cell = 0; cell < matrixFreeData.n_cell_batches();
                 ++cell)
              {
                forceEval.reinit(cell);

                std::fill(EQuad.begin(), EQuad.end(), zeroTensor4);

                const unsigned int numSubCells =
                  matrixFreeData.n_active_entries_per_cell_batch(cell);

                for (unsigned int iSubCell = 0; iSubCell < numSubCells;
                     ++iSubCell)
                  {
                    subCellPtr =
                      matrixFreeData.get_cell_iterator(cell, iSubCell);
                    const unsigned int physicalCellId =
                      cellIdToCellNumberMap[subCellPtr->id()];
                    for (unsigned int kPoint = 0; kPoint < numKPoints; ++kPoint)
                      {
                        for (unsigned int q = 0; q < numQuadPoints; ++q)
                          {
                            const unsigned int id =
                              kPoint * numPhysicalCells * numQuadPoints * 9 +
                              physicalCellId * numQuadPoints * 9 + q * 9;
                            EQuad[q][0][0][iSubCell] +=
                              dftPtr->d_kPointWeights[kPoint] *
                              elocWfcEshelbyTensorQuadValuesH[id + 0];
                            EQuad[q][0][1][iSubCell] +=
                              dftPtr->d_kPointWeights[kPoint] *
                              elocWfcEshelbyTensorQuadValuesH[id + 1];
                            EQuad[q][0][2][iSubCell] +=
                              dftPtr->d_kPointWeights[kPoint] *
                              elocWfcEshelbyTensorQuadValuesH[id + 2];
                            EQuad[q][1][0][iSubCell] +=
                              dftPtr->d_kPointWeights[kPoint] *
                              elocWfcEshelbyTensorQuadValuesH[id + 3];
                            EQuad[q][1][1][iSubCell] +=
                              dftPtr->d_kPointWeights[kPoint] *
                              elocWfcEshelbyTensorQuadValuesH[id + 4];
                            EQuad[q][1][2][iSubCell] +=
                              dftPtr->d_kPointWeights[kPoint] *
                              elocWfcEshelbyTensorQuadValuesH[id + 5];
                            EQuad[q][2][0][iSubCell] +=
                              dftPtr->d_kPointWeights[kPoint] *
                              elocWfcEshelbyTensorQuadValuesH[id + 6];
                            EQuad[q][2][1][iSubCell] +=
                              dftPtr->d_kPointWeights[kPoint] *
                              elocWfcEshelbyTensorQuadValuesH[id + 7];
                            EQuad[q][2][2][iSubCell] +=
                              dftPtr->d_kPointWeights[kPoint] *
                              elocWfcEshelbyTensorQuadValuesH[id + 8];
                          } // quad loop
                      }     // kpoint loop
                  }         // subcell loop


                for (unsigned int q = 0; q < numQuadPoints; ++q)
                  forceEval.submit_gradient(spinPolarizedFactorVect * EQuad[q],
                                            q);

                forceEval.integrate(dealii::EvaluationFlags::gradients);
#ifdef USE_COMPLEX
                forceEval.distribute_local_to_global(
                  d_configForceVectorLinFEKPoints);
#else
                forceEval.distribute_local_to_global(d_configForceVectorLinFE);
#endif
              } // cell loop
          }

        if (isPseudopotential)
          {
            dealii::AlignedVector<
              dealii::Tensor<1, 3, dealii::VectorizedArray<double>>>
              FVectQuads(numQuadPointsNLP, zeroTensor3);
            for (unsigned int cell = 0; cell < matrixFreeData.n_cell_batches();
                 ++cell)
              {
                forceEvalNLP.reinit(cell);

                // compute FnlGammaAtoms  (contibution due to Gamma(Rj))
                FnlGammaAtomsElementalContribution(
                  forceContributionFnlGammaAtoms,
                  matrixFreeData,
                  forceEvalNLP,
                  cell,
                  cellIdToCellNumberMap,
#ifdef USE_COMPLEX
                  projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattened,
#endif
                  dftPtr->d_oncvClassPtr->getNonLocalOperator()
                    ->getAtomCenteredKpointIndexedSphericalFnQuadValues(),
                  projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattened);


                if (!d_dftParams.floatingNuclearCharges)
                  {
                    FnlGammaxElementalContribution(
                      FVectQuads,
                      matrixFreeData,
                      numQuadPointsNLP,
                      cell,
                      cellIdToCellNumberMap,
                      dftPtr->d_oncvClassPtr->getNonLocalOperator()
                        ->getAtomCenteredKpointIndexedSphericalFnQuadValues(),
                      projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattened);

                    for (unsigned int q = 0; q < numQuadPointsNLP; ++q)
                      forceEvalNLP.submit_value(spinPolarizedFactorVect *
                                                  FVectQuads[q],
                                                q);

                    forceEvalNLP.integrate(dealii::EvaluationFlags::values);

#ifdef USE_COMPLEX
                    forceEvalNLP.distribute_local_to_global(
                      d_configForceVectorLinFEKPoints);
#else
                    forceEvalNLP.distribute_local_to_global(
                      d_configForceVectorLinFE);
#endif
                  } // no floating charges check
              }     // macro cell loop
          }         // pseudopotential check
      }             // spin index

    // add global Fnl contribution due to Gamma(Rj) to the configurational force
    // vector
    if (isPseudopotential)
      {
        if (d_dftParams.spinPolarized == 1)
          for (auto &iter : forceContributionFnlGammaAtoms)
            {
              std::vector<double> &fnlvec = iter.second;
              for (unsigned int i = 0; i < fnlvec.size(); i++)
                fnlvec[i] *= spinPolarizedFactor;
            }

        if (d_dftParams.floatingNuclearCharges)
          {
#ifdef USE_COMPLEX
            accumulateForceContributionGammaAtomsFloating(
              forceContributionFnlGammaAtoms, d_forceAtomsFloatingKPoints);
#else
            accumulateForceContributionGammaAtomsFloating(
              forceContributionFnlGammaAtoms, d_forceAtomsFloating);
#endif
          }
        else
          distributeForceContributionFnlGammaAtoms(
            forceContributionFnlGammaAtoms);
      }


    MPI_Barrier(d_mpiCommParent);
    double enowfc_time = MPI_Wtime();

    /////////// Compute contribution independent of wavefunctions
    ////////////////////
    if (bandGroupTaskId == 0)
      {
        // kpoint group parallelization data structures
        const unsigned int numberKptGroups =
          dealii::Utilities::MPI::n_mpi_processes(dftPtr->interpoolcomm);

        const unsigned int kptGroupTaskId =
          dealii::Utilities::MPI::this_mpi_process(dftPtr->interpoolcomm);
        std::vector<int> kptGroupLowHighPlusOneIndices;

        if (numMacroCells > 0)
          dftUtils::createKpointParallelizationIndices(
            dftPtr->interpoolcomm,
            numMacroCells,
            kptGroupLowHighPlusOneIndices);

        dftPtr->d_basisOperationsPtrHost->reinit(0,
                                                 0,
                                                 dftPtr->d_densityQuadratureId,
                                                 false);
        std::vector<
          dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
          gradDensityOutValuesSpinPolarized;

        bool isGradDensityDataRequired =
          (dftPtr->d_excManagerPtr->getExcSSDFunctionalObj()
             ->getDensityBasedFamilyType() == densityFamilyType::GGA);
        if (isGradDensityDataRequired)
          {
            gradDensityOutValuesSpinPolarized = gradRhoOutValues;

            if (d_dftParams.spinPolarized == 0)
              gradDensityOutValuesSpinPolarized.push_back(
                dftfe::utils::MemoryStorage<double,
                                            dftfe::utils::MemorySpace::HOST>(
                  gradRhoOutValues[0].size(), 0.0));
          }

        dealii::AlignedVector<
          dealii::Tensor<1, 3, dealii::VectorizedArray<double>>>
          gradRhoSpin0QuadsVect(numQuadPoints, zeroTensor3);
        dealii::AlignedVector<
          dealii::Tensor<1, 3, dealii::VectorizedArray<double>>>
          gradRhoSpin1QuadsVect(numQuadPoints, zeroTensor3);
        dealii::AlignedVector<
          dealii::Tensor<2, 3, dealii::VectorizedArray<double>>>
          hessianRhoSpin0Quads(numQuadPoints, zeroTensor4);
        dealii::AlignedVector<
          dealii::Tensor<2, 3, dealii::VectorizedArray<double>>>
                                                               hessianRhoSpin1Quads(numQuadPoints, zeroTensor4);
        dealii::AlignedVector<dealii::VectorizedArray<double>> excQuads(
          numQuadPoints, dealii::make_vectorized_array(0.0));
        dealii::AlignedVector<dealii::VectorizedArray<double>>
          vxcRhoOutSpin0Quads(numQuadPoints,
                              dealii::make_vectorized_array(0.0));
        dealii::AlignedVector<dealii::VectorizedArray<double>>
          vxcRhoOutSpin1Quads(numQuadPoints,
                              dealii::make_vectorized_array(0.0));
        dealii::AlignedVector<
          dealii::Tensor<1, 3, dealii::VectorizedArray<double>>>
          derExchCorrEnergyWithGradRhoOutSpin0Quads(numQuadPoints, zeroTensor3);
        dealii::AlignedVector<
          dealii::Tensor<1, 3, dealii::VectorizedArray<double>>>
          derExchCorrEnergyWithGradRhoOutSpin1Quads(numQuadPoints, zeroTensor3);
        dealii::AlignedVector<
          dealii::Tensor<1, 3, dealii::VectorizedArray<double>>>
          gradRhoCoreQuads(numQuadPoints, zeroTensor3);
        dealii::AlignedVector<
          dealii::Tensor<2, 3, dealii::VectorizedArray<double>>>
          hessianRhoCoreQuads(numQuadPoints, zeroTensor4);
        std::map<unsigned int, std::vector<double>>
          forceContributionNonlinearCoreCorrectionGammaAtoms;

        std::unordered_map<xcRemainderOutputDataAttributes, std::vector<double>>
          xDensityOutDataOut;
        std::unordered_map<xcRemainderOutputDataAttributes, std::vector<double>>
          cDensityOutDataOut;

        std::vector<double> &xEnergyDensityOut =
          xDensityOutDataOut[xcRemainderOutputDataAttributes::e];
        std::vector<double> &cEnergyDensityOut =
          cDensityOutDataOut[xcRemainderOutputDataAttributes::e];

        std::vector<double> &pdexDensityOutSpinUp =
          xDensityOutDataOut[xcRemainderOutputDataAttributes::pdeDensitySpinUp];
        std::vector<double> &pdexDensityOutSpinDown = xDensityOutDataOut
          [xcRemainderOutputDataAttributes::pdeDensitySpinDown];
        std::vector<double> &pdecDensityOutSpinUp =
          cDensityOutDataOut[xcRemainderOutputDataAttributes::pdeDensitySpinUp];
        std::vector<double> &pdecDensityOutSpinDown = cDensityOutDataOut
          [xcRemainderOutputDataAttributes::pdeDensitySpinDown];

        if (isGradDensityDataRequired)
          {
            xDensityOutDataOut[xcRemainderOutputDataAttributes::pdeSigma] =
              std::vector<double>();
            cDensityOutDataOut[xcRemainderOutputDataAttributes::pdeSigma] =
              std::vector<double>();
          }


        auto quadPointsAll = dftPtr->d_basisOperationsPtrHost->quadPoints();

        auto quadWeightsAll = dftPtr->d_basisOperationsPtrHost->JxW();

        for (unsigned int cell = 0; cell < matrixFreeData.n_cell_batches();
             ++cell)
          {
            if (cell < kptGroupLowHighPlusOneIndices[2 * kptGroupTaskId + 1] &&
                cell >= kptGroupLowHighPlusOneIndices[2 * kptGroupTaskId])
              {
                forceEval.reinit(cell);

                std::fill(gradRhoSpin0QuadsVect.begin(),
                          gradRhoSpin0QuadsVect.end(),
                          zeroTensor3);
                std::fill(gradRhoSpin1QuadsVect.begin(),
                          gradRhoSpin1QuadsVect.end(),
                          zeroTensor3);
                std::fill(hessianRhoSpin0Quads.begin(),
                          hessianRhoSpin0Quads.end(),
                          zeroTensor4);
                std::fill(hessianRhoSpin1Quads.begin(),
                          hessianRhoSpin1Quads.end(),
                          zeroTensor4);
                std::fill(excQuads.begin(),
                          excQuads.end(),
                          dealii::make_vectorized_array(0.0));
                std::fill(vxcRhoOutSpin0Quads.begin(),
                          vxcRhoOutSpin0Quads.end(),
                          dealii::make_vectorized_array(0.0));
                std::fill(vxcRhoOutSpin1Quads.begin(),
                          vxcRhoOutSpin1Quads.end(),
                          dealii::make_vectorized_array(0.0));
                std::fill(derExchCorrEnergyWithGradRhoOutSpin0Quads.begin(),
                          derExchCorrEnergyWithGradRhoOutSpin0Quads.end(),
                          zeroTensor3);
                std::fill(derExchCorrEnergyWithGradRhoOutSpin1Quads.begin(),
                          derExchCorrEnergyWithGradRhoOutSpin1Quads.end(),
                          zeroTensor3);
                std::fill(gradRhoCoreQuads.begin(),
                          gradRhoCoreQuads.end(),
                          zeroTensor3);
                std::fill(hessianRhoCoreQuads.begin(),
                          hessianRhoCoreQuads.end(),
                          zeroTensor4);

                const unsigned int numSubCells =
                  matrixFreeData.n_active_entries_per_cell_batch(cell);

                //
                for (unsigned int iSubCell = 0; iSubCell < numSubCells;
                     ++iSubCell)
                  {
                    subCellPtr =
                      matrixFreeData.get_cell_iterator(cell, iSubCell);
                    dealii::CellId subCellId = subCellPtr->id();


                    const unsigned int subCellIndex =
                      dftPtr->d_basisOperationsPtrHost->cellIndex(subCellId);

                    std::vector<double> quadPointsInCell(numQuadPoints * 3);
                    std::vector<double> quadWeightsInCell(numQuadPoints);
                    for (unsigned int iQuad = 0; iQuad < numQuadPoints; ++iQuad)
                      {
                        for (unsigned int idim = 0; idim < 3; ++idim)
                          quadPointsInCell[iQuad * 3 + idim] =
                            quadPointsAll[subCellIndex * numQuadPoints * 3 +
                                          iQuad * 3 + idim];
                        quadWeightsInCell[iQuad] = std::real(
                          quadWeightsAll[subCellIndex * numQuadPoints + iQuad]);
                      }

                    dftPtr->d_excManagerPtr->getExcSSDFunctionalObj()
                      ->computeRhoTauDependentXCData(
                        *(dftPtr->d_auxDensityMatrixXCOutPtr),
                        quadPointsInCell,
                        xDensityOutDataOut,
                        cDensityOutDataOut);

                    std::vector<double> pdexDensityOutSigma;
                    std::vector<double> pdecDensityOutSigma;

                    if (isGradDensityDataRequired)
                      {
                        pdexDensityOutSigma = xDensityOutDataOut
                          [xcRemainderOutputDataAttributes::pdeSigma];
                        pdecDensityOutSigma = cDensityOutDataOut
                          [xcRemainderOutputDataAttributes::pdeSigma];
                      }


                    std::unordered_map<DensityDescriptorDataAttributes,
                                       std::vector<double>>
                                        densityXCOutData;
                    std::vector<double> gradDensityXCOutSpinUp;
                    std::vector<double> gradDensityXCOutSpinDown;

                    if (isGradDensityDataRequired)
                      {
                        densityXCOutData
                          [DensityDescriptorDataAttributes::gradValuesSpinUp] =
                            std::vector<double>();
                        densityXCOutData[DensityDescriptorDataAttributes::
                                           gradValuesSpinDown] =
                          std::vector<double>();
                      }

                    dftPtr->d_auxDensityMatrixXCOutPtr->applyLocalOperations(
                      quadPointsInCell, densityXCOutData);

                    if (isGradDensityDataRequired)
                      {
                        gradDensityXCOutSpinUp = densityXCOutData
                          [DensityDescriptorDataAttributes::gradValuesSpinUp];
                        gradDensityXCOutSpinDown = densityXCOutData
                          [DensityDescriptorDataAttributes::gradValuesSpinDown];
                      }



                    if (isGradDensityDataRequired)
                      {
                        // const std::vector<double> &temp3 =
                        //  (*dftPtr->gradRhoOutValuesSpinPolarized)
                        //    .find(subCellId)
                        //    ->second;

                        const auto &gradRhoTotalOutValues =
                          gradDensityOutValuesSpinPolarized[0];
                        const auto &gradRhoMagOutValues =
                          gradDensityOutValuesSpinPolarized[1];

                        for (unsigned int q = 0; q < numQuadPoints; ++q)
                          for (unsigned int idim = 0; idim < 3; idim++)
                            {
                              gradRhoSpin0QuadsVect[q][idim][iSubCell] =
                                (gradRhoTotalOutValues[subCellIndex *
                                                         numQuadPoints * 3 +
                                                       q * 3 + idim] +
                                 gradRhoMagOutValues[subCellIndex *
                                                       numQuadPoints * 3 +
                                                     q * 3 + idim]) /
                                2.0;
                              gradRhoSpin1QuadsVect[q][idim][iSubCell] =
                                (gradRhoTotalOutValues[subCellIndex *
                                                         numQuadPoints * 3 +
                                                       q * 3 + idim] -
                                 gradRhoMagOutValues[subCellIndex *
                                                       numQuadPoints * 3 +
                                                     q * 3 + idim]) /
                                2.0;
                            }
                      }


                    for (unsigned int q = 0; q < numQuadPoints; ++q)
                      {
                        excQuads[q][iSubCell] =
                          xEnergyDensityOut[q] + cEnergyDensityOut[q];
                        vxcRhoOutSpin0Quads[q][iSubCell] =
                          pdexDensityOutSpinUp[q] + pdecDensityOutSpinUp[q];
                        vxcRhoOutSpin1Quads[q][iSubCell] =
                          pdexDensityOutSpinDown[q] + pdecDensityOutSpinDown[q];
                      }

                    if (isGradDensityDataRequired)
                      {
                        for (unsigned int q = 0; q < numQuadPoints; ++q)
                          {
                            for (unsigned int idim = 0; idim < 3; idim++)
                              {
                                derExchCorrEnergyWithGradRhoOutSpin0Quads
                                  [q][idim][iSubCell] =
                                    2.0 *
                                    (pdexDensityOutSigma[3 * q + 0] +
                                     pdecDensityOutSigma[3 * q + 0]) *
                                    gradDensityXCOutSpinUp[3 * q + idim];
                                derExchCorrEnergyWithGradRhoOutSpin0Quads
                                  [q][idim][iSubCell] +=
                                  (pdexDensityOutSigma[3 * q + 1] +
                                   pdecDensityOutSigma[3 * q + 1]) *
                                  gradDensityXCOutSpinDown[3 * q + idim];

                                derExchCorrEnergyWithGradRhoOutSpin1Quads
                                  [q][idim][iSubCell] +=
                                  2.0 *
                                  (pdexDensityOutSigma[3 * q + 2] +
                                   pdecDensityOutSigma[3 * q + 2]) *
                                  gradDensityXCOutSpinDown[3 * q + idim];
                                derExchCorrEnergyWithGradRhoOutSpin1Quads
                                  [q][idim][iSubCell] +=
                                  (pdexDensityOutSigma[3 * q + 1] +
                                   pdecDensityOutSigma[3 * q + 1]) *
                                  gradDensityXCOutSpinUp[3 * q + idim];
                              }
                          }
                      }


                    if (d_dftParams.nonLinearCoreCorrection == true)
                      for (unsigned int q = 0; q < numQuadPoints; ++q)
                        {
                          const std::vector<double> &temp1 =
                            gradRhoCoreValues.find(subCellId)->second;
                          for (unsigned int q = 0; q < numQuadPoints; ++q)
                            for (unsigned int idim = 0; idim < 3; idim++)
                              gradRhoCoreQuads[q][idim][iSubCell] =
                                temp1[3 * q + idim] / 2.0;

                          if (isGradDensityDataRequired)
                            {
                              const std::vector<double> &temp2 =
                                hessianRhoCoreValues.find(subCellId)->second;
                              for (unsigned int q = 0; q < numQuadPoints; ++q)
                                for (unsigned int idim = 0; idim < 3; ++idim)
                                  for (unsigned int jdim = 0; jdim < 3; ++jdim)
                                    hessianRhoCoreQuads
                                      [q][idim][jdim][iSubCell] =
                                        temp2[9 * q + 3 * idim + jdim] / 2.0;
                            }
                        }

                  } // subcell loop

                if (d_dftParams.nonLinearCoreCorrection)
                  {
                    FNonlinearCoreCorrectionGammaAtomsElementalContributionSpinPolarized(
                      forceContributionNonlinearCoreCorrectionGammaAtoms,
                      forceEval,
                      matrixFreeData,
                      cell,
                      vxcRhoOutSpin0Quads,
                      vxcRhoOutSpin1Quads,
                      derExchCorrEnergyWithGradRhoOutSpin0Quads,
                      derExchCorrEnergyWithGradRhoOutSpin1Quads,
                      gradRhoCoreAtoms,
                      hessianRhoCoreAtoms,
                      isGradDensityDataRequired);
                  }

                for (unsigned int q = 0; q < numQuadPoints; ++q)
                  {
                    dealii::Tensor<2, 3, dealii::VectorizedArray<double>> E =
                      eshelbyTensorSP::getELocXcEshelbyTensor(
                        gradRhoSpin0QuadsVect[q],
                        gradRhoSpin1QuadsVect[q],
                        excQuads[q],
                        derExchCorrEnergyWithGradRhoOutSpin0Quads[q],
                        derExchCorrEnergyWithGradRhoOutSpin1Quads[q]);

                    dealii::Tensor<1, 3, dealii::VectorizedArray<double>> F =
                      zeroTensor3;

                    if (d_dftParams.nonLinearCoreCorrection)
                      F += eshelbyTensorSP::getFNonlinearCoreCorrection(
                        vxcRhoOutSpin0Quads[q],
                        vxcRhoOutSpin1Quads[q],
                        derExchCorrEnergyWithGradRhoOutSpin0Quads[q],
                        derExchCorrEnergyWithGradRhoOutSpin1Quads[q],
                        gradRhoCoreQuads[q],
                        hessianRhoCoreQuads[q],
                        isGradDensityDataRequired);

                    forceEval.submit_value(F, q);
                    forceEval.submit_gradient(E, q);
                  } // quad point loop

                forceEval.integrate(dealii::EvaluationFlags::values |
                                    dealii::EvaluationFlags::gradients);
                forceEval.distribute_local_to_global(
                  d_configForceVectorLinFE); // also takes care of
                                             // constraints
              }                              // kpt paral
          }                                  // cell loop

        if (d_dftParams.nonLinearCoreCorrection)
          {
            if (d_dftParams.floatingNuclearCharges)
              accumulateForceContributionGammaAtomsFloating(
                forceContributionNonlinearCoreCorrectionGammaAtoms,
                d_forceAtomsFloating);
            else
              distributeForceContributionFPSPLocalGammaAtoms(
                forceContributionNonlinearCoreCorrectionGammaAtoms,
                d_atomsForceDofs,
                d_constraintsNoneForce,
                d_configForceVectorLinFE);
          }


        ////Add electrostatic configurational force contribution////////////////
        computeConfigurationalForceEEshelbyEElectroPhiTot(
          matrixFreeDataElectro,
          phiTotDofHandlerIndexElectro,
          smearedChargeQuadratureId,
          lpspQuadratureIdElectro,
          phiTotRhoOutElectro,
          rhoOutValues[0],
          rhoTotalOutValuesLpsp,
          gradRhoOutValues[0],
          gradRhoTotalOutValuesLpsp,
          pseudoVLocElectro,
          pseudoVLocAtomsElectro,
          vselfBinsManagerElectro);
      }

    MPI_Barrier(d_mpiCommParent);
    enowfc_time = MPI_Wtime() - enowfc_time;

    forcetotal_time = MPI_Wtime() - forcetotal_time;

    if (this_process == 0 && d_dftParams.verbosity >= 4)
      std::cout
        << "Total time for configurational force computation except Eself contribution: "
        << forcetotal_time << std::endl;

    if (d_dftParams.verbosity >= 4)
      {
        pcout << " Time taken for initialization in force: " << init_time
              << std::endl;
        pcout << " Time taken for non wfc in force: " << enowfc_time
              << std::endl;
      }
  }

  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  forceClass<FEOrder, FEOrderElectro, memorySpace>::
    computeConfigurationalForceEEshelbyEElectroPhiTot(
      const dealii::MatrixFree<3, double> &matrixFreeDataElectro,
      const unsigned int                   phiTotDofHandlerIndexElectro,
      const unsigned int                   smearedChargeQuadratureId,
      const unsigned int                   lpspQuadratureIdElectro,
      const distributedCPUVec<double> &    phiTotRhoOutElectro,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &rhoTotalOutValues,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &rhoTotalOutValuesLpsp,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &gradRhoTotalOutValues,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &gradRhoTotalOutValuesLpsp,
      const std::map<dealii::CellId, std::vector<double>> &pseudoVLocElectro,
      const std::map<unsigned int,
                     std::map<dealii::CellId, std::vector<double>>>
        &                                              pseudoVLocAtomsElectro,
      const vselfBinsManager<FEOrder, FEOrderElectro> &vselfBinsManagerElectro)
  {
    dealii::FEEvaluation<
      3,
      1,
      C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>(),
      3>
      forceEvalElectro(matrixFreeDataElectro,
                       d_forceDofHandlerIndexElectro,
                       dftPtr->d_densityQuadratureIdElectro);

    dealii::FEEvaluation<
      3,
      FEOrderElectro,
      C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>(),
      1>
      phiTotEvalElectro(matrixFreeDataElectro,
                        phiTotDofHandlerIndexElectro,
                        dftPtr->d_densityQuadratureIdElectro);

    dealii::FEEvaluation<3, -1> phiTotEvalSmearedCharge(
      matrixFreeDataElectro,
      phiTotDofHandlerIndexElectro,
      smearedChargeQuadratureId);

    dealii::FEEvaluation<3, -1, 1, 3> forceEvalSmearedCharge(
      matrixFreeDataElectro,
      d_forceDofHandlerIndexElectro,
      smearedChargeQuadratureId);

    dealii::FEEvaluation<3,
                         1,
                         C_num1DQuadLPSP<FEOrderElectro>() *
                           C_numCopies1DQuadLPSP(),
                         3>
      forceEvalElectroLpsp(matrixFreeDataElectro,
                           d_forceDofHandlerIndexElectro,
                           lpspQuadratureIdElectro);

    std::map<unsigned int, std::vector<double>>
      forceContributionFPSPLocalGammaAtoms;
    std::map<unsigned int, std::vector<double>>
      forceContributionSmearedChargesGammaAtoms;
    std::map<unsigned int, std::vector<double>>
      forceContributionShadowPotentialElectroGammaAtoms;

    const unsigned int numQuadPoints = forceEvalElectro.n_q_points;
    const unsigned int numQuadPointsSmearedb =
      forceEvalSmearedCharge.n_q_points;
    const unsigned int numQuadPointsLpsp = forceEvalElectroLpsp.n_q_points;

    AssertThrow(
      matrixFreeDataElectro.get_quadrature(smearedChargeQuadratureId).size() ==
        numQuadPointsSmearedb,
      dealii::ExcMessage(
        "DFT-FE Error: mismatch in quadrature rule usage in force computation."));

    AssertThrow(
      matrixFreeDataElectro.get_quadrature(lpspQuadratureIdElectro).size() ==
        numQuadPointsLpsp,
      dealii::ExcMessage(
        "DFT-FE Error: mismatch in quadrature rule usage in force computation."));

    // if (gradRhoTotalOutValuesLpsp.size() != 0)
    //  AssertThrow(
    //    gradRhoTotalOutValuesLpsp.begin()->second.size() == 3 *
    //    numQuadPointsLpsp, dealii::ExcMessage(
    //      "DFT-FE Error: mismatch in quadrature rule usage in force
    //      computation."));

    dealii::DoFHandler<3>::active_cell_iterator subCellPtr;

    dealii::FEValues<3> feVselfValuesElectro(
      matrixFreeDataElectro.get_dof_handler(phiTotDofHandlerIndexElectro)
        .get_fe(),
      matrixFreeDataElectro.get_quadrature(lpspQuadratureIdElectro),
      d_dftParams.floatingNuclearCharges && d_dftParams.smearedNuclearCharges ?
        (dealii::update_values | dealii::update_quadrature_points) :
        (dealii::update_values | dealii::update_gradients |
         dealii::update_quadrature_points));

    dealii::QIterated<3 - 1> faceQuadrature(
      dealii::QGauss<1>(C_num1DQuadLPSP<FEOrderElectro>()),
      C_numCopies1DQuadLPSP());
    dealii::FEFaceValues<3> feFaceValuesElectro(
      dftPtr->d_dofHandlerRhoNodal.get_fe(),
      faceQuadrature,
      dealii::update_values | dealii::update_JxW_values |
        dealii::update_normal_vectors | dealii::update_quadrature_points);

    dealii::Tensor<1, 3, dealii::VectorizedArray<double>> zeroTensor;
    for (unsigned int idim = 0; idim < 3; idim++)
      {
        zeroTensor[idim] = dealii::make_vectorized_array(0.0);
      }

    dealii::Tensor<2, 3, dealii::VectorizedArray<double>> zeroTensor2;
    for (unsigned int idim = 0; idim < 3; idim++)
      for (unsigned int jdim = 0; jdim < 3; jdim++)
        {
          zeroTensor2[idim][jdim] = dealii::make_vectorized_array(0.0);
        }


    std::vector<double> tempRhoVal(numQuadPoints, 0);
    std::vector<double> tempLpspRhoVal(numQuadPointsLpsp, 0);
    std::vector<double> tempLpspGradRhoVal(numQuadPointsLpsp * 3, 0);


    dealii::AlignedVector<dealii::VectorizedArray<double>> rhoQuadsElectro(
      numQuadPoints, dealii::make_vectorized_array(0.0));
    dealii::AlignedVector<dealii::VectorizedArray<double>> rhoQuadsElectroLpsp(
      numQuadPointsLpsp, dealii::make_vectorized_array(0.0));
    dealii::AlignedVector<dealii::VectorizedArray<double>> smearedbQuads(
      numQuadPointsSmearedb, dealii::make_vectorized_array(0.0));
    dealii::AlignedVector<dealii::Tensor<1, 3, dealii::VectorizedArray<double>>>
      gradPhiTotSmearedChargeQuads(numQuadPointsSmearedb, zeroTensor);
    dealii::AlignedVector<dealii::Tensor<1, 3, dealii::VectorizedArray<double>>>
      gradRhoQuadsElectro(numQuadPoints, zeroTensor);
    dealii::AlignedVector<dealii::Tensor<1, 3, dealii::VectorizedArray<double>>>
      gradRhoQuadsElectroLpsp(numQuadPointsLpsp, zeroTensor);
    // dealii::AlignedVector<dealii::Tensor<1,3,dealii::VectorizedArray<double>
    // > > gradRhoAtomsQuadsElectro(numQuadPoints,zeroTensor);
    dealii::AlignedVector<dealii::VectorizedArray<double>>
      pseudoVLocQuadsElectro(numQuadPointsLpsp,
                             dealii::make_vectorized_array(0.0));

    // kpoint group parallelization data structures
    const unsigned int numberKptGroups =
      dealii::Utilities::MPI::n_mpi_processes(dftPtr->interpoolcomm);

    const unsigned int kptGroupTaskId =
      dealii::Utilities::MPI::this_mpi_process(dftPtr->interpoolcomm);
    std::vector<int> kptGroupLowHighPlusOneIndices;

    if (matrixFreeDataElectro.n_cell_batches() > 0)
      dftUtils::createKpointParallelizationIndices(
        dftPtr->interpoolcomm,
        matrixFreeDataElectro.n_cell_batches(),
        kptGroupLowHighPlusOneIndices);

    for (unsigned int cell = 0; cell < matrixFreeDataElectro.n_cell_batches();
         ++cell)
      {
        if (cell < kptGroupLowHighPlusOneIndices[2 * kptGroupTaskId + 1] &&
            cell >= kptGroupLowHighPlusOneIndices[2 * kptGroupTaskId])
          {
            std::set<unsigned int> nonTrivialSmearedChargeAtomIdsMacroCell;

            const unsigned int numSubCells =
              matrixFreeDataElectro.n_active_entries_per_cell_batch(cell);
            if (d_dftParams.smearedNuclearCharges)
              for (unsigned int iSubCell = 0; iSubCell < numSubCells;
                   ++iSubCell)
                {
                  subCellPtr =
                    matrixFreeDataElectro.get_cell_iterator(cell, iSubCell);
                  dealii::CellId                   subCellId = subCellPtr->id();
                  const std::vector<unsigned int> &temp =
                    dftPtr->d_bCellNonTrivialAtomIds.find(subCellId)->second;
                  for (int i = 0; i < temp.size(); i++)
                    nonTrivialSmearedChargeAtomIdsMacroCell.insert(temp[i]);
                }

            forceEvalElectro.reinit(cell);
            forceEvalElectroLpsp.reinit(cell);

            phiTotEvalElectro.reinit(cell);
            phiTotEvalElectro.read_dof_values_plain(phiTotRhoOutElectro);
            phiTotEvalElectro.evaluate(dealii::EvaluationFlags::values |
                                       dealii::EvaluationFlags::gradients);

            if (d_dftParams.smearedNuclearCharges &&
                nonTrivialSmearedChargeAtomIdsMacroCell.size() > 0)
              {
                forceEvalSmearedCharge.reinit(cell);
                phiTotEvalSmearedCharge.reinit(cell);
              }

            std::fill(rhoQuadsElectro.begin(),
                      rhoQuadsElectro.end(),
                      dealii::make_vectorized_array(0.0));
            std::fill(rhoQuadsElectroLpsp.begin(),
                      rhoQuadsElectroLpsp.end(),
                      dealii::make_vectorized_array(0.0));
            std::fill(smearedbQuads.begin(),
                      smearedbQuads.end(),
                      dealii::make_vectorized_array(0.0));
            std::fill(gradPhiTotSmearedChargeQuads.begin(),
                      gradPhiTotSmearedChargeQuads.end(),
                      zeroTensor);
            std::fill(gradRhoQuadsElectro.begin(),
                      gradRhoQuadsElectro.end(),
                      zeroTensor);
            std::fill(gradRhoQuadsElectroLpsp.begin(),
                      gradRhoQuadsElectroLpsp.end(),
                      zeroTensor);
            // std::fill(gradRhoAtomsQuadsElectro.begin(),gradRhoAtomsQuadsElectro.end(),zeroTensor);
            std::fill(pseudoVLocQuadsElectro.begin(),
                      pseudoVLocQuadsElectro.end(),
                      dealii::make_vectorized_array(0.0));

            for (unsigned int iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
              {
                subCellPtr =
                  matrixFreeDataElectro.get_cell_iterator(cell, iSubCell);
                dealii::CellId subCellId = subCellPtr->id();

                const unsigned int subCellIndex =
                  dftPtr->d_basisOperationsPtrElectroHost->cellIndex(subCellId);

                for (unsigned int q = 0; q < numQuadPoints; ++q)
                  tempRhoVal[q] =
                    rhoTotalOutValues[subCellIndex * numQuadPoints + q];

                for (unsigned int q = 0; q < numQuadPoints; ++q)
                  rhoQuadsElectro[q][iSubCell] = tempRhoVal[q];
                // rhoOutValuesElectro.find(subCellId)->second[q];

                if (d_dftParams.isPseudopotential ||
                    d_dftParams.smearedNuclearCharges)
                  {
                    const std::vector<double> &tempPseudoVal =
                      pseudoVLocElectro.find(subCellId)->second;
                    // const std::vector<double> &tempLpspRhoVal =
                    //  rhoOutValuesLpsp.find(subCellId)->second;
                    // const std::vector<double> &tempLpspGradRhoVal =
                    //  gradRhoOutValuesLpsp.find(subCellId)->second;



                    for (unsigned int q = 0; q < numQuadPointsLpsp; ++q)
                      {
                        tempLpspRhoVal[q] =
                          rhoTotalOutValuesLpsp[subCellIndex *
                                                  numQuadPointsLpsp +
                                                q];
                        tempLpspGradRhoVal[3 * q + 0] =
                          gradRhoTotalOutValuesLpsp[subCellIndex *
                                                      numQuadPointsLpsp * 3 +
                                                    3 * q + 0];
                        tempLpspGradRhoVal[3 * q + 1] =
                          gradRhoTotalOutValuesLpsp[subCellIndex *
                                                      numQuadPointsLpsp * 3 +
                                                    3 * q + 1];
                        tempLpspGradRhoVal[3 * q + 2] =
                          gradRhoTotalOutValuesLpsp[subCellIndex *
                                                      numQuadPointsLpsp * 3 +
                                                    3 * q + 2];
                      }

                    for (unsigned int q = 0; q < numQuadPointsLpsp; ++q)
                      {
                        pseudoVLocQuadsElectro[q][iSubCell] = tempPseudoVal[q];
                        rhoQuadsElectroLpsp[q][iSubCell]    = tempLpspRhoVal[q];
                        gradRhoQuadsElectroLpsp[q][0][iSubCell] =
                          tempLpspGradRhoVal[3 * q + 0];
                        gradRhoQuadsElectroLpsp[q][1][iSubCell] =
                          tempLpspGradRhoVal[3 * q + 1];
                        gradRhoQuadsElectroLpsp[q][2][iSubCell] =
                          tempLpspGradRhoVal[3 * q + 2];
                      }
                  }

                if (d_dftParams.smearedNuclearCharges &&
                    nonTrivialSmearedChargeAtomIdsMacroCell.size() > 0)
                  {
                    const std::vector<double> &bQuadValuesCell =
                      dftPtr->d_bQuadValuesAllAtoms.find(subCellId)->second;
                    for (unsigned int q = 0; q < numQuadPointsSmearedb; ++q)
                      {
                        smearedbQuads[q][iSubCell] = bQuadValuesCell[q];
                      }
                  }
              }

            if (d_dftParams.isPseudopotential ||
                d_dftParams.smearedNuclearCharges)
              {
                FPSPLocalGammaAtomsElementalContribution(
                  forceContributionFPSPLocalGammaAtoms,
                  feVselfValuesElectro,
                  feFaceValuesElectro,
                  forceEvalElectroLpsp,
                  matrixFreeDataElectro,
                  phiTotDofHandlerIndexElectro,
                  cell,
                  rhoQuadsElectroLpsp,
                  gradRhoQuadsElectroLpsp,
                  pseudoVLocAtomsElectro,
                  vselfBinsManagerElectro,
                  d_cellsVselfBallsClosestAtomIdDofHandlerElectro);
              }

            for (unsigned int q = 0; q < numQuadPoints; ++q)
              {
                dealii::VectorizedArray<double> phiTotElectro_q =
                  phiTotEvalElectro.get_value(q);
                dealii::VectorizedArray<double> phiExtElectro_q =
                  dealii::make_vectorized_array(0.0);
                dealii::Tensor<1, 3, dealii::VectorizedArray<double>>
                  gradPhiTotElectro_q = phiTotEvalElectro.get_gradient(q);

                dealii::Tensor<2, 3, dealii::VectorizedArray<double>> E =
                  eshelbyTensor::getEElectroEshelbyTensor(phiTotElectro_q,
                                                          gradPhiTotElectro_q,
                                                          rhoQuadsElectro[q]);

                dealii::Tensor<1, 3, dealii::VectorizedArray<double>> F =
                  zeroTensor;


                forceEvalElectro.submit_value(F, q);
                forceEvalElectro.submit_gradient(E, q);
              }

            if (d_dftParams.isPseudopotential ||
                d_dftParams.smearedNuclearCharges)
              for (unsigned int q = 0; q < numQuadPointsLpsp; ++q)
                {
                  dealii::VectorizedArray<double> phiExtElectro_q =
                    dealii::make_vectorized_array(0.0);

                  dealii::Tensor<2, 3, dealii::VectorizedArray<double>> E =
                    zeroTensor2;

                  dealii::Tensor<1, 3, dealii::VectorizedArray<double>> F =
                    zeroTensor;
                  dealii::Tensor<1, 3, dealii::VectorizedArray<double>>
                    gradPhiExt_q = zeroTensor;
                  F -= gradRhoQuadsElectroLpsp[q] * pseudoVLocQuadsElectro[q];

                  forceEvalElectroLpsp.submit_value(F, q);
                  forceEvalElectroLpsp.submit_gradient(E, q);
                }

            forceEvalElectro.integrate(dealii::EvaluationFlags::values |
                                       dealii::EvaluationFlags::gradients);
            forceEvalElectro.distribute_local_to_global(
              d_configForceVectorLinFEElectro);

            if (d_dftParams.isPseudopotential ||
                d_dftParams.smearedNuclearCharges)
              {
                forceEvalElectroLpsp.integrate(
                  dealii::EvaluationFlags::values |
                  dealii::EvaluationFlags::gradients);
                forceEvalElectroLpsp.distribute_local_to_global(
                  d_configForceVectorLinFEElectro);
              }

            if (d_dftParams.smearedNuclearCharges &&
                nonTrivialSmearedChargeAtomIdsMacroCell.size() > 0)
              {
                phiTotEvalSmearedCharge.read_dof_values_plain(
                  phiTotRhoOutElectro);
                phiTotEvalSmearedCharge.evaluate(
                  dealii::EvaluationFlags::gradients);

                for (unsigned int q = 0; q < numQuadPointsSmearedb; ++q)
                  {
                    gradPhiTotSmearedChargeQuads[q] =
                      phiTotEvalSmearedCharge.get_gradient(q);

                    dealii::Tensor<1, 3, dealii::VectorizedArray<double>> F =
                      zeroTensor;
                    F = -gradPhiTotSmearedChargeQuads[q] * smearedbQuads[q];

                    forceEvalSmearedCharge.submit_value(F, q);
                  }


                forceEvalSmearedCharge.integrate(
                  dealii::EvaluationFlags::values);
                forceEvalSmearedCharge.distribute_local_to_global(
                  d_configForceVectorLinFEElectro);

                FPhiTotSmearedChargesGammaAtomsElementalContribution(
                  forceContributionSmearedChargesGammaAtoms,
                  forceEvalSmearedCharge,
                  matrixFreeDataElectro,
                  cell,
                  gradPhiTotSmearedChargeQuads,
                  std::vector<unsigned int>(
                    nonTrivialSmearedChargeAtomIdsMacroCell.begin(),
                    nonTrivialSmearedChargeAtomIdsMacroCell.end()),
                  dftPtr->d_bQuadAtomIdsAllAtoms,
                  smearedbQuads);
              }
          } // kpt paral
      }     // macro cell loop

    // add global FPSPLocal contribution due to Gamma(Rj) to the configurational
    // force vector
    if (d_dftParams.isPseudopotential || d_dftParams.smearedNuclearCharges)
      {
        if (d_dftParams.floatingNuclearCharges)
          {
            accumulateForceContributionGammaAtomsFloating(
              forceContributionFPSPLocalGammaAtoms, d_forceAtomsFloating);
          }
        else
          distributeForceContributionFPSPLocalGammaAtoms(
            forceContributionFPSPLocalGammaAtoms,
            d_atomsForceDofsElectro,
            d_constraintsNoneForceElectro,
            d_configForceVectorLinFEElectro);
      }

    if (d_dftParams.smearedNuclearCharges)
      {
        if (d_dftParams.floatingNuclearCharges)
          {
            accumulateForceContributionGammaAtomsFloating(
              forceContributionSmearedChargesGammaAtoms, d_forceAtomsFloating);
          }
        else
          distributeForceContributionFPSPLocalGammaAtoms(
            forceContributionSmearedChargesGammaAtoms,
            d_atomsForceDofsElectro,
            d_constraintsNoneForceElectro,
            d_configForceVectorLinFEElectro);
      }
  }
#include "../force.inst.cc"
} // namespace dftfe
