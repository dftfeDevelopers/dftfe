// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2025  The Regents of the University of Michigan and DFT-FE
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


#include <configurationalForce.h>
#include <configurationalForceKernels.h>
#include <constants.h>
namespace dftfe
{
  template <dftfe::utils::MemorySpace memorySpace>
  configurationalForceClass<memorySpace>::configurationalForceClass(
    std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
      BLASWrapperPtr,
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
      BLASWrapperPtrHost,
    std::shared_ptr<
      dftfe::basis::FEBasisOperations<dataTypes::number, double, memorySpace>>
      basisOperationsPtr,
    std::shared_ptr<
      dftfe::basis::FEBasisOperations<dataTypes::number,
                                      double,
                                      dftfe::utils::MemorySpace::HOST>>
      basisOperationsPtrHost,
    std::shared_ptr<
      dftfe::basis::FEBasisOperations<double, double, memorySpace>>
      basisOperationsPtrElectro,
    std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>
      basisOperationsPtrElectroHost,
    std::shared_ptr<
      dftfe::pseudopotentialBaseClass<dataTypes::number, memorySpace>>
                                             pseudopotentialClassPtr,
    std::shared_ptr<excManager<memorySpace>> d_excManagerPtr,
    const dftfe::uInt                        densityQuadratureId,
    const dftfe::uInt                        densityQuadratureIdElectro,
    const dftfe::uInt                        lpspQuadratureId,
    const dftfe::uInt                        lpspQuadratureIdElectro,
    const dftfe::uInt                        smearedChargeQuadratureIdElectro,
    const MPI_Comm                          &mpi_comm_parent,
    const MPI_Comm                          &mpi_comm_domain,
    const MPI_Comm                          &interpoolcomm,
    const MPI_Comm                          &interBandGroupComm,
    const dftParameters                     &dftParams)
    : d_mpiCommParent(mpi_comm_parent)
    , d_mpiCommDomain(mpi_comm_domain)
    , d_mpiCommInterPool(interpoolcomm)
    , d_mpiCommInterBandGroup(interBandGroupComm)
    , d_dftParams(dftParams)
    , d_BLASWrapperPtr(BLASWrapperPtr)
    , d_BLASWrapperPtrHost(BLASWrapperPtrHost)
    , d_basisOperationsPtr(basisOperationsPtr)
    , d_basisOperationsPtrHost(basisOperationsPtrHost)
    , d_basisOperationsPtrElectro(basisOperationsPtrElectro)
    , d_basisOperationsPtrElectroHost(basisOperationsPtrElectroHost)
    , d_pseudopotentialClassPtr(pseudopotentialClassPtr)
    , d_excManagerPtr(d_excManagerPtr)
    , n_mpi_processes(dealii::Utilities::MPI::n_mpi_processes(mpi_comm_domain))
    , this_mpi_process(
        dealii::Utilities::MPI::this_mpi_process(mpi_comm_domain))
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0))
    , d_densityQuadratureId(densityQuadratureId)
    , d_densityQuadratureIdElectro(densityQuadratureIdElectro)
    , d_lpspQuadratureId(lpspQuadratureId)
    , d_lpspQuadratureIdElectro(lpspQuadratureIdElectro)
    , d_smearedChargeQuadratureIdElectro(smearedChargeQuadratureIdElectro)
    , FEForce(dealii::FE_Q<3>(dealii::QGaussLobatto<1>(2)), 3)
  {
    if (d_dftParams.isPseudopotential)
      d_pseudopotentialNonLocalOperator =
        d_pseudopotentialClassPtr->getNonLocalOperator();
  }
  template <dftfe::utils::MemorySpace memorySpace>
  void
  configurationalForceClass<memorySpace>::computeForceAndStress(
    const dftfe::uInt                         &numEigenValues,
    const std::vector<double>                 &kPointCoords,
    const std::vector<double>                 &kPointWeights,
    const double                               domainVolume,
    const std::shared_ptr<groupSymmetryClass> &groupSymmetryPtr,
    const dispersionCorrection                &dispersionCorr,
    const dftfe::utils::MemoryStorage<dataTypes::number, memorySpace>
                                           &eigenVectors,
    const std::vector<std::vector<double>> &eigenValues,
    const std::vector<std::vector<double>> &partialOccupancies,
    const std::vector<std::vector<double>> &atomLocations,
    const std::vector<dftfe::Int>          &imageIds,
    const std::vector<double>              &imageCharges,
    const std::vector<std::vector<double>> &imagePositions,
    const distributedCPUVec<double>        &phiTotRhoOutValues,
    const distributedCPUVec<double>        &rhoOutNodalValues,
    const std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &densityOutValues,
    const std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &gradDensityOutValues,
    const std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &tauOutValues,
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &rhoTotalOutValuesLpsp,
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &gradRhoTotalOutValuesLpsp,
    const std::shared_ptr<AuxDensityMatrix<memorySpace>>
      auxDensityXCOutRepresentationPtr,
    const std::map<dealii::CellId, std::vector<double>> &rhoCoreValues,
    const std::map<dealii::CellId, std::vector<double>> &gradRhoCoreValues,
    const std::map<dealii::CellId, std::vector<double>> &hessianRhoCoreValues,
    const std::map<dftfe::uInt, std::map<dealii::CellId, std::vector<double>>>
      &gradRhoCoreAtoms,
    const std::map<dftfe::uInt, std::map<dealii::CellId, std::vector<double>>>
                                                        &hessianRhoCoreAtoms,
    const std::map<dealii::CellId, std::vector<double>> &pseudoVLocValues,
    const std::map<dftfe::uInt, std::map<dealii::CellId, std::vector<double>>>
                                &pseudoVLocAtoms,
    const dealii::DoFHandler<3> &dofHandlerRhoNodal,
    const vselfBinsManager      &vselfBinsManager,
    const std::vector<distributedCPUVec<double>>
                      &vselfFieldGateauxDerStrainFDBins,
    const dftfe::uInt &binsStartDofHandlerIndexElectro,
    const std::map<dealii::CellId, std::vector<dftfe::Int>>
      &bQuadAtomIdsAllAtoms,
    const std::map<dealii::CellId, std::vector<dftfe::Int>>
      &bQuadAtomIdsAllAtomsImages,
    const std::map<dealii::CellId, std::vector<double>> &bQuadValuesAllAtoms,
    const bool                                           floatingNuclearCharges,
    const bool                                           computeForce,
    const bool                                           computeStress)
  {
    if (computeForce)
      d_forceTotal.resize(d_dftParams.natoms * 3, 0.0);
    if (computeStress)
      d_stressTotal.resize(9, 0.0);
    std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      densityOutValuesSpinPolarized = densityOutValues;
    std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      gradDensityOutValuesSpinPolarized;
    std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      tauOutValuesSpinPolarized = tauOutValues;

    if (d_dftParams.spinPolarized == 0)
      densityOutValuesSpinPolarized.push_back(
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>(
          densityOutValues[0].size(), 0.0));

    bool isIntegrationByPartsGradDensityDependenceVxc =
      (d_excManagerPtr->getExcSSDFunctionalObj()->getDensityBasedFamilyType() ==
       densityFamilyType::GGA);

    const bool isTauMGGA =
      (d_excManagerPtr->getExcSSDFunctionalObj()->getExcFamilyType() ==
       ExcFamilyType::TauMGGA);

    if (isIntegrationByPartsGradDensityDependenceVxc)
      {
        gradDensityOutValuesSpinPolarized = gradDensityOutValues;

        if (d_dftParams.spinPolarized == 0)
          gradDensityOutValuesSpinPolarized.push_back(
            dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::HOST>(
              gradDensityOutValues[0].size(), 0.0));
      }

    if (isTauMGGA)
      {
        if (d_dftParams.spinPolarized == 0)
          {
            tauOutValuesSpinPolarized.push_back(
              dftfe::utils::MemoryStorage<double,
                                          dftfe::utils::MemorySpace::HOST>(
                tauOutValues[0].size(), 0.0));
          }
      }


    computeWfcContribAll(numEigenValues,
                         kPointCoords,
                         kPointWeights,
                         eigenVectors,
                         eigenValues,
                         partialOccupancies,
                         floatingNuclearCharges,
                         computeForce,
                         computeStress);
    computeWfcContribLocal(numEigenValues,
                           kPointCoords,
                           kPointWeights,
                           eigenVectors,
                           eigenValues,
                           partialOccupancies,
                           floatingNuclearCharges,
                           computeForce,
                           computeStress);
    computeXCContribAll(atomLocations,
                        imageIds,
                        imagePositions,
                        densityOutValuesSpinPolarized,
                        gradDensityOutValuesSpinPolarized,
                        tauOutValuesSpinPolarized,
                        auxDensityXCOutRepresentationPtr,
                        rhoCoreValues,
                        gradRhoCoreValues,
                        gradRhoCoreAtoms,
                        hessianRhoCoreAtoms,
                        floatingNuclearCharges,
                        computeForce,
                        computeStress);
    createBinObjectsForce(dofHandlerRhoNodal,
                          vselfBinsManager,
                          d_cellsVselfBallsDofHandlerElectro,
                          d_cellsVselfBallsDofHandlerForceElectro,
                          d_cellsVselfBallsClosestAtomIdDofHandlerElectro,
                          d_AtomIdBinIdLocalDofHandlerElectro,
                          d_cellFacesVselfBallSurfacesDofHandlerElectro,
                          d_cellFacesVselfBallSurfacesDofHandlerForceElectro);
    computeLPSPContribAll(atomLocations,
                          imageIds,
                          imageCharges,
                          imagePositions,
                          rhoOutNodalValues,
                          rhoTotalOutValuesLpsp,
                          gradRhoTotalOutValuesLpsp,
                          pseudoVLocValues,
                          pseudoVLocAtoms,
                          dofHandlerRhoNodal,
                          vselfBinsManager,
                          vselfFieldGateauxDerStrainFDBins,
                          floatingNuclearCharges,
                          computeForce,
                          computeStress);
    computeSmearedContribAll(atomLocations,
                             imagePositions,
                             vselfBinsManager,
                             binsStartDofHandlerIndexElectro,
                             phiTotRhoOutValues,
                             bQuadAtomIdsAllAtoms,
                             bQuadAtomIdsAllAtomsImages,
                             bQuadValuesAllAtoms,
                             floatingNuclearCharges,
                             computeForce,
                             computeStress);
    computeElectroContribEshelby(phiTotRhoOutValues,
                                 densityOutValuesSpinPolarized[0],
                                 floatingNuclearCharges,
                                 computeForce,
                                 computeStress);
    computeESelfContribEshelby(atomLocations,
                               imageIds,
                               imageCharges,
                               imagePositions,
                               vselfBinsManager,
                               floatingNuclearCharges,
                               computeForce,
                               computeStress);
    if (computeForce)
      {
        if (d_dftParams.dc_dispersioncorrectiontype != 0)
          for (dftfe::uInt iAtom = 0; iAtom < d_dftParams.natoms; iAtom++)
            for (dftfe::uInt idim = 0; idim < 3; idim++)
              d_forceTotal[iAtom * 3 + idim] +=
                dispersionCorr.getForceCorrection(iAtom, idim);

        if (d_dftParams.useSymm)
          groupSymmetryPtr->symmetrizeVectorFieldFromGlobalValues(
            d_forceTotal, dftfe::pointSet::atomicCoord);
        d_forceTotal.copyTo(d_forceVector);
      }


    if (computeStress)
      {
        if (d_dftParams.dc_dispersioncorrectiontype != 0)
          for (dftfe::uInt iDim = 0; iDim < 3; iDim++)
            for (dftfe::uInt jDim = 0; jDim < 3; jDim++)
              d_stressTotal[3 * iDim + jDim] +=
                dispersionCorr.getStressCorrection(iDim, jDim);
        if (d_dftParams.useSymm)
          groupSymmetryPtr->symmetrizeRank2Tensor(d_stressTotal);

        for (dftfe::uInt iDim = 0; iDim < 3; iDim++)
          for (dftfe::uInt jDim = 0; jDim < 3; jDim++)
            d_stressTotal[3 * iDim + jDim] /= domainVolume;
        for (dftfe::uInt iDim = 0; iDim < 3; iDim++)
          for (dftfe::uInt jDim = 0; jDim < 3; jDim++)
            d_stressTensor[iDim][jDim] = d_stressTotal[3 * iDim + jDim];
      }
  }

  template <dftfe::utils::MemorySpace memorySpace>
  std::vector<double> &
  configurationalForceClass<memorySpace>::getAtomsForces()
  {
    return d_forceVector;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  dealii::Tensor<2, 3, double> &
  configurationalForceClass<memorySpace>::getStress()
  {
    return d_stressTensor;
  }


  template <dftfe::utils::MemorySpace memorySpace>
  void
  configurationalForceClass<memorySpace>::printAtomsForces()
  {
    const dftfe::Int numberGlobalAtoms = d_dftParams.natoms;
    if (!d_dftParams.reproducible_output)
      pcout << std::endl << "Ion forces (Hartree/Bohr)" << std::endl;
    else
      pcout << std::endl
            << "Absolute values of ion forces (Hartree/Bohr)" << std::endl;
    if (d_dftParams.verbosity >= 2)
      pcout << "Negative of configurational force (Hartree/Bohr) on atoms"
            << std::endl;

    pcout
      << "--------------------------------------------------------------------------------------------"
      << std::endl;
    // also find the atom with the maximum absolute force and print that
    double                           maxForce           = -1.0;
    double                           sumAbsValForceComp = 0;
    std::vector<double>              sumForce(3);
    dftfe::uInt                      maxForceAtomId = 0;
    std::vector<std::vector<double>> forceData(numberGlobalAtoms,
                                               std::vector<double>(3, 0.0));
    for (dftfe::uInt i = 0; i < numberGlobalAtoms; i++)
      {
        if (!d_dftParams.reproducible_output)
          pcout << std::setw(4) << i << "     " << std::scientific
                << -d_forceTotal[3 * i] << "   " << -d_forceTotal[3 * i + 1]
                << "   " << -d_forceTotal[3 * i + 2] << std::endl;
        else
          {
            std::vector<double> truncatedForce(3);
            for (dftfe::uInt idim = 0; idim < 3; idim++)
              truncatedForce[idim] =
                std::fabs(std::floor(10000000 * (-d_forceTotal[3 * i + idim])) /
                          10000000.0);

            pcout << "AtomId " << std::setw(4) << i << ":  " << std::fixed
                  << std::setprecision(6) << truncatedForce[0] << ","
                  << truncatedForce[1] << "," << truncatedForce[2] << std::endl;
          }

        forceData[i][0] = -d_forceTotal[3 * i];
        forceData[i][1] = -d_forceTotal[3 * i + 1];
        forceData[i][2] = -d_forceTotal[3 * i + 2];

        double absForce = 0.0;
        for (dftfe::uInt idim = 0; idim < 3; idim++)
          {
            absForce += d_forceTotal[3 * i + idim] * d_forceTotal[3 * i + idim];
            sumAbsValForceComp += std::abs(d_forceTotal[3 * i + idim]);
            sumForce[idim] += d_forceTotal[3 * i + idim];
          }
        Assert(absForce >= 0., ExcInternalError());
        absForce = std::sqrt(absForce);
        if (absForce > maxForce)
          {
            maxForce       = absForce;
            maxForceAtomId = i;
          }
      }

    pcout
      << "--------------------------------------------------------------------------------------------"
      << std::endl;

    if (d_dftParams.verbosity >= 1)
      {
        pcout << " Maximum absolute force atom id: " << maxForceAtomId
              << ", Force vec: " << -d_forceTotal[3 * maxForceAtomId] << ","
              << -d_forceTotal[3 * maxForceAtomId + 1] << ","
              << -d_forceTotal[3 * maxForceAtomId + 2] << std::endl;
        pcout
          << " Sum of absolute value of all force components over all atoms: "
          << sumAbsValForceComp << std::endl;
        pcout << " Sum of all forces in each component: " << sumForce[0] << " "
              << sumForce[1] << " " << sumForce[2] << std::endl;
      }

    if (d_dftParams.verbosity >= 1 && !d_dftParams.reproducible_output)
      dftUtils::writeDataIntoFile(forceData, "forces.txt", d_mpiCommParent);
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  configurationalForceClass<memorySpace>::printStress()
  {
    if (!d_dftParams.reproducible_output)
      {
        pcout << std::endl;
        pcout << "Cell stress (Hartree/Bohr^3)" << std::endl;
        pcout
          << "------------------------------------------------------------------------"
          << std::endl;
        for (dftfe::uInt idim = 0; idim < 3; idim++)
          pcout << d_stressTensor[idim][0] << "  " << d_stressTensor[idim][1]
                << "  " << d_stressTensor[idim][2] << std::endl;
        pcout
          << "------------------------------------------------------------------------"
          << std::endl;
      }
    else
      {
        pcout << std::endl;
        pcout << "Absolute value of cell stress (Hartree/Bohr^3)" << std::endl;
        pcout
          << "------------------------------------------------------------------------"
          << std::endl;
        for (dftfe::uInt idim = 0; idim < 3; idim++)
          {
            std::vector<double> truncatedStress(3);
            for (dftfe::uInt jdim = 0; jdim < 3; jdim++)
              truncatedStress[jdim] = std::fabs(
                std::floor(10000000 * d_stressTensor[idim][jdim]) / 10000000.0);
            pcout << std::fixed << std::setprecision(6) << truncatedStress[0]
                  << "  " << truncatedStress[1] << "  " << truncatedStress[2]
                  << std::endl;
          }
        pcout
          << "------------------------------------------------------------------------"
          << std::endl;
      }
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  configurationalForceClass<memorySpace>::computeWfcContribLocal(
    const dftfe::uInt         &numEigenValues,
    const std::vector<double> &kPointCoords,
    const std::vector<double> &kPointWeights,
    const dftfe::utils::MemoryStorage<dataTypes::number, memorySpace>
                                           &eigenVectors,
    const std::vector<std::vector<double>> &eigenValues,
    const std::vector<std::vector<double>> &partialOccupancies,
    const bool                              floatingNuclearCharges,
    const bool                              computeForce,
    const bool                              computeStress)
  {
    std::vector<double> StressLocContrib(9, 0.0);
    d_basisOperationsPtr->reinit(0, 0, d_densityQuadratureId);
    d_basisOperationsPtrHost->reinit(0, 0, d_densityQuadratureId);

    const dftfe::uInt nCells        = d_basisOperationsPtr->nCells();
    const dftfe::uInt nDofsPerCell  = d_basisOperationsPtr->nDofsPerCell();
    const dftfe::uInt nQuadsPerCell = d_basisOperationsPtr->nQuadsPerCell();
    const dftfe::uInt numLocalDofs  = d_basisOperationsPtr->nOwnedDofs();
    const dftfe::uInt totalLocallyOwnedCells = d_basisOperationsPtr->nCells();

    const dftfe::uInt cellsBlockSize =
      memorySpace == dftfe::utils::MemorySpace::DEVICE ?
        (d_dftParams.memOptMode ? 50 : nCells) :
        1;
    const dftfe::uInt numCellBlocks = totalLocallyOwnedCells / cellsBlockSize;
    const dftfe::uInt remCellBlockSize =
      totalLocallyOwnedCells - numCellBlocks * cellsBlockSize;


    const dftfe::uInt numberBandGroups =
      dealii::Utilities::MPI::n_mpi_processes(d_mpiCommInterBandGroup);
    const dftfe::uInt bandGroupTaskId =
      dealii::Utilities::MPI::this_mpi_process(d_mpiCommInterBandGroup);
    std::vector<dftfe::uInt> bandGroupLowHighPlusOneIndices;
    dftUtils::createBandParallelizationIndices(d_mpiCommInterBandGroup,
                                               numEigenValues,
                                               bandGroupLowHighPlusOneIndices);

    const dftfe::uInt wfcBlockSize =
      std::min(d_dftParams.chebyWfcBlockSize,
               bandGroupLowHighPlusOneIndices[1]);

    const double spinPolarizedFactor =
      (d_dftParams.spinPolarized == 1) ? 1.0 : 2.0;
    const dftfe::uInt numSpinComponents =
      (d_dftParams.spinPolarized == 1) ? 2 : 1;

    dftfe::utils::MemoryStorage<dataTypes::number, memorySpace>
      cellWaveFunctionMatrix(cellsBlockSize * nDofsPerCell * wfcBlockSize);
    dftfe::utils::MemoryStorage<dataTypes::number, memorySpace>
      cellWaveFunctionQuadData(cellsBlockSize * nQuadsPerCell * wfcBlockSize);
    dftfe::utils::MemoryStorage<dataTypes::number, memorySpace>
      cellGradWaveFunctionQuadData(cellsBlockSize * nQuadsPerCell *
                                   wfcBlockSize * 3);
    dftfe::utils::MemoryStorage<double, memorySpace> eshelbyContributions(
      cellsBlockSize * nQuadsPerCell * wfcBlockSize * 9, 0.0);
    dftfe::utils::MemoryStorage<double, memorySpace> eshelbyTensor(
      cellsBlockSize * nQuadsPerCell * 9, 0.0);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      eshelbyTensorHost(cellsBlockSize * nQuadsPerCell * 9, 0.0);

    dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace>
      *flattenedArrayBlock;
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      partialOccupVecHost(wfcBlockSize, 0.0);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      eigenValuesVecHost(wfcBlockSize, 0.0);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      kCoordHost(3, 0.0);
#if defined(DFTFE_WITH_DEVICE)
    dftfe::utils::MemoryStorage<double, memorySpace> partialOccupVec(
      partialOccupVecHost.size());
    dftfe::utils::MemoryStorage<double, memorySpace> eigenValuesVec(
      eigenValuesVecHost.size());
    dftfe::utils::MemoryStorage<double, memorySpace> kCoord(kCoordHost.size());
#else
    auto &partialOccupVec     = partialOccupVecHost;
    auto &eigenValuesVec      = eigenValuesVecHost;
    auto &kCoord              = kCoordHost;
#endif
    for (dftfe::uInt kPoint = 0; kPoint < kPointWeights.size(); ++kPoint)
      {
        kCoordHost[0] = kPointCoords[3 * kPoint + 0];
        kCoordHost[1] = kPointCoords[3 * kPoint + 1];
        kCoordHost[2] = kPointCoords[3 * kPoint + 2];

        for (dftfe::uInt spinIndex = 0; spinIndex < numSpinComponents;
             ++spinIndex)
          {
            for (dftfe::uInt jvec = 0; jvec < numEigenValues;
                 jvec += wfcBlockSize)
              {
                const dftfe::uInt currentBlockSize =
                  std::min(wfcBlockSize, numEigenValues - jvec);
                flattenedArrayBlock =
                  &(d_basisOperationsPtr->getMultiVector(currentBlockSize, 0));
                if ((jvec + currentBlockSize) <=
                      bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
                    (jvec + currentBlockSize) >
                      bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
                  {
                    for (dftfe::uInt iEigenVec = 0;
                         iEigenVec < currentBlockSize;
                         ++iEigenVec)
                      partialOccupVecHost[iEigenVec] =
                        partialOccupancies[kPoint][numEigenValues * spinIndex +
                                                   jvec + iEigenVec] *
                        kPointWeights[kPoint] * spinPolarizedFactor;
                    for (dftfe::uInt iEigenVec = 0;
                         iEigenVec < currentBlockSize;
                         ++iEigenVec)
                      eigenValuesVecHost[iEigenVec] =
                        eigenValues[kPoint][numEigenValues * spinIndex + jvec +
                                            iEigenVec];

#if defined(DFTFE_WITH_DEVICE)
                    partialOccupVec.copyFrom(partialOccupVecHost);
                    eigenValuesVec.copyFrom(eigenValuesVecHost);
                    kCoord.copyFrom(kCoordHost);
#endif
                    d_BLASWrapperPtr->stridedCopyToBlockConstantStride(
                      currentBlockSize,
                      numEigenValues,
                      numLocalDofs,
                      jvec,
                      eigenVectors.data() +
                        numLocalDofs * numEigenValues *
                          (numSpinComponents * kPoint + spinIndex),
                      flattenedArrayBlock->data());

                    d_basisOperationsPtr->reinit(currentBlockSize,
                                                 cellsBlockSize,
                                                 0,
                                                 false);

                    flattenedArrayBlock->updateGhostValues();
                    d_basisOperationsPtr->distribute(*(flattenedArrayBlock));

                    for (dftfe::Int iCellBlock = 0;
                         iCellBlock < (numCellBlocks + 1);
                         iCellBlock++)
                      {
                        const dftfe::uInt currentCellsBlockSize =
                          (iCellBlock == numCellBlocks) ? remCellBlockSize :
                                                          cellsBlockSize;
                        if (currentCellsBlockSize > 0)
                          {
                            const dftfe::uInt startingCellId =
                              iCellBlock * cellsBlockSize;
                            std::pair<dftfe::uInt, dftfe::uInt> cellRange(
                              startingCellId,
                              startingCellId + currentCellsBlockSize);
                            std::pair<dftfe::uInt, dftfe::uInt> vecRange(
                              jvec, jvec + currentBlockSize);
                            d_basisOperationsPtr->extractToCellNodalDataKernel(
                              *(flattenedArrayBlock),
                              cellWaveFunctionMatrix.data(),
                              cellRange);
                            d_basisOperationsPtr->interpolateKernel(
                              cellWaveFunctionMatrix.data(),
                              cellWaveFunctionQuadData.data(),
                              cellGradWaveFunctionQuadData.data(),
                              cellRange);
                            dftfe::computeWavefuncEshelbyContributionLocal(
                              d_BLASWrapperPtr,
                              cellRange,
                              vecRange,
                              nQuadsPerCell,
                              kCoordHost[0],
                              kCoordHost[1],
                              kCoordHost[2],
                              partialOccupVec.data(),
                              eigenValuesVec.data(),
                              cellWaveFunctionQuadData.data(),
                              cellGradWaveFunctionQuadData.data(),
                              eshelbyContributions.data(),
                              eshelbyTensor.data(),
                              floatingNuclearCharges,
                              computeForce,
                              computeStress);
                            eshelbyTensorHost.copyFrom(eshelbyTensor);
                            for (dftfe::Int iCell = 0;
                                 iCell < currentCellsBlockSize;
                                 iCell++)
                              {
                                const double *JxWValues =
                                  d_basisOperationsPtrHost->JxWBasisData()
                                    .data() +
                                  nQuadsPerCell * (iCell + startingCellId);

                                for (dftfe::uInt iQuad = 0;
                                     iQuad < nQuadsPerCell;
                                     iQuad++)
                                  for (dftfe::uInt iDim = 0; iDim < 3; iDim++)
                                    for (dftfe::uInt jDim = 0; jDim < 3; jDim++)
                                      {
                                        StressLocContrib[3 * iDim + jDim] +=
                                          eshelbyTensorHost
                                            [iCell * nQuadsPerCell * 9 +
                                             iQuad * 9 + 3 * iDim + jDim] *
                                          JxWValues[iQuad];
                                      }
                              }

                          } // non-trivial cell block check
                      }     // cells block loop
                  }
              }
          }
      }
    if (computeStress)
      {
        MPI_Allreduce(MPI_IN_PLACE,
                      StressLocContrib.data(),
                      9,
                      MPI_DOUBLE,
                      MPI_SUM,
                      d_mpiCommParent);
        // pcout << "Stress Tensor Loc: " << StressLocContrib.size() << std::endl;
        // for (dftfe::uInt iDim = 0; iDim < 3; iDim++)
        //   {
        //     for (dftfe::uInt jDim = 0; jDim < 3; jDim++)
        //       pcout << StressLocContrib[3 * iDim + jDim] << " ";
        //     pcout << std::endl;
        //   }
        for (dftfe::uInt iDim = 0; iDim < 3; iDim++)
          for (dftfe::uInt jDim = 0; jDim < 3; jDim++)
            d_stressTotal[3 * iDim + jDim] += StressLocContrib[3 * iDim + jDim];
      }
  }


  template <dftfe::utils::MemorySpace memorySpace>
  void
  configurationalForceClass<memorySpace>::computeESelfContribEshelby(
    const std::vector<std::vector<double>> &atomLocations,
    const std::vector<dftfe::Int>          &imageIds,
    const std::vector<double>              &imageCharges,
    const std::vector<std::vector<double>> &imagePositions,
    const vselfBinsManager                 &vselfBinsManager,
    const bool                              floatingNuclearCharges,
    const bool                              computeForce,
    const bool                              computeStress)
  {
    std::vector<double> stressContribESelfEshelby(9, 0.0);
    d_basisOperationsPtrElectroHost->reinit(0, 0, d_densityQuadratureIdElectro);
    const dftfe::uInt nCells = d_basisOperationsPtrElectroHost->nCells();
    const dftfe::uInt nQuadsPerCell =
      d_basisOperationsPtrElectroHost->nQuadsPerCell();

    dealii::FEValues<3> feVselfValues(
      d_basisOperationsPtrElectroHost->getDofHandler().get_fe(),
      d_basisOperationsPtrElectroHost->matrixFreeData().get_quadrature(
        d_densityQuadratureId),
      dealii::update_gradients | dealii::update_JxW_values);

    std::vector<dealii::Tensor<1, 3, double>> gradVselfQuad(nQuadsPerCell);
    const dftfe::uInt nVSelfBins = vselfBinsManager.getAtomIdsBins().size();
    for (dftfe::uInt iBin = 0; iBin < nVSelfBins; ++iBin)
      {
        const std::vector<dealii::DoFHandler<3>::active_cell_iterator>
          &cellsVselfBallDofHandler = d_cellsVselfBallsDofHandlerElectro[iBin];
        const distributedCPUVec<double> &iBinVselfField =
          vselfBinsManager.getVselfFieldBins()[iBin];
        std::vector<dealii::DoFHandler<3>::active_cell_iterator>::const_iterator
          iter1;
        for (iter1 = cellsVselfBallDofHandler.begin();
             iter1 != cellsVselfBallDofHandler.end();
             ++iter1)
          {
            dealii::DoFHandler<3>::active_cell_iterator cell = *iter1;
            feVselfValues.reinit(cell);
            feVselfValues.get_function_gradients(iBinVselfField, gradVselfQuad);
            if (computeStress)
              for (dftfe::uInt iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
                {
                  const double diagContrib =
                    1.0 / (8.0 * M_PI) * gradVselfQuad[iQuad] *
                    gradVselfQuad[iQuad] * feVselfValues.JxW(iQuad);
                  for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
                    for (dftfe::uInt jDim = 0; jDim < 3; ++jDim)
                      stressContribESelfEshelby[3 * iDim + jDim] +=
                        -1.0 / (4.0 * M_PI) * gradVselfQuad[iQuad][jDim] *
                        gradVselfQuad[iQuad][iDim] * feVselfValues.JxW(iQuad);

                  for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
                    stressContribESelfEshelby[iDim * 3 + iDim] += diagContrib;
                }
          } // cell loop
      }     // bin loop
    dealii::QGauss<3 - 1>   faceQuadrature(d_dftParams.densityQuadratureRule);
    dealii::FEFaceValues<3> feVselfFaceValues(
      d_basisOperationsPtrElectroHost->getDofHandler().get_fe(),
      faceQuadrature,
      dealii::update_values | dealii::update_JxW_values |
        dealii::update_normal_vectors | dealii::update_quadrature_points);

    const dftfe::uInt nQuadsPerFace = feVselfFaceValues.get_quadrature().size();
    for (dftfe::uInt iBin = 0; iBin < nVSelfBins; ++iBin)
      {
        const std::map<dealii::DoFHandler<3>::active_cell_iterator,
                       std::vector<dftfe::uInt>>
          &cellsVselfBallSurfacesDofHandler =
            d_cellFacesVselfBallSurfacesDofHandlerElectro[iBin];
        const distributedCPUVec<double> &iBinVselfField =
          vselfBinsManager.getVselfFieldBins()[iBin];
        std::map<dealii::DoFHandler<3>::active_cell_iterator,
                 std::vector<dftfe::uInt>>::const_iterator iter1;
        for (iter1 = cellsVselfBallSurfacesDofHandler.begin();
             iter1 != cellsVselfBallSurfacesDofHandler.end();
             ++iter1)
          {
            dealii::DoFHandler<3>::active_cell_iterator cell = iter1->first;
            const dftfe::Int                            closestAtomId =
              d_cellsVselfBallsClosestAtomIdDofHandlerElectro[iBin][cell->id()];
            double           closestAtomCharge;
            dealii::Point<3> closestAtomLocation;
            if (closestAtomId < d_dftParams.natoms)
              {
                closestAtomLocation[0] = atomLocations[closestAtomId][2];
                closestAtomLocation[1] = atomLocations[closestAtomId][3];
                closestAtomLocation[2] = atomLocations[closestAtomId][4];
                if (d_dftParams.isPseudopotential)
                  closestAtomCharge = atomLocations[closestAtomId][1];
                else
                  closestAtomCharge = atomLocations[closestAtomId][0];
              }
            else
              {
                const dftfe::Int imageId = closestAtomId - d_dftParams.natoms;
                closestAtomCharge        = imageCharges[imageId];
                closestAtomLocation[0]   = imagePositions[imageId][0];
                closestAtomLocation[1]   = imagePositions[imageId][1];
                closestAtomLocation[2]   = imagePositions[imageId][2];
              }

            const std::vector<dftfe::uInt> &dirichletFaceIds = iter1->second;
            for (dftfe::uInt index = 0; index < dirichletFaceIds.size();
                 index++)
              {
                const dftfe::uInt faceId = dirichletFaceIds[index];
                feVselfFaceValues.reinit(cell, faceId);

                for (dftfe::uInt iQuad = 0; iQuad < nQuadsPerFace; ++iQuad)
                  {
                    const dealii::Point<3> quadPoint =
                      feVselfFaceValues.quadrature_point(iQuad);
                    const dealii::Tensor<1, 3, double> dispClosestAtom =
                      quadPoint - closestAtomLocation;
                    const double dist = dispClosestAtom.norm();
                    const dealii::Tensor<1, 3, double> gradVselfFaceQuadExact =
                      closestAtomCharge * dispClosestAtom / dist / dist / dist;
                    double diagContrib = 1.0 / (8.0 * M_PI) *
                                         scalar_product(gradVselfFaceQuadExact,
                                                        gradVselfFaceQuadExact);
                    dealii::Tensor<2, 3, double> eshelbyTensor =
                      -1.0 / (4.0 * M_PI) *
                      outer_product(gradVselfFaceQuadExact,
                                    gradVselfFaceQuadExact);

                    eshelbyTensor[0][0] += diagContrib;
                    eshelbyTensor[1][1] += diagContrib;
                    eshelbyTensor[2][2] += diagContrib;

                    dealii::Tensor<2, 3, double> surfaceIntegralContrib =
                      outer_product(dispClosestAtom,
                                    eshelbyTensor *
                                      feVselfFaceValues.normal_vector(iQuad)) *
                      feVselfFaceValues.JxW(iQuad);

                    for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
                      for (dftfe::uInt jDim = 0; jDim < 3; ++jDim)
                        stressContribESelfEshelby[3 * iDim + jDim] -=
                          surfaceIntegralContrib[iDim][jDim];
                  } // q point loop
              }     // face loop
          }         // cell loop
      }             // bin loop
    if (computeStress)
      {
        MPI_Allreduce(MPI_IN_PLACE,
                      stressContribESelfEshelby.data(),
                      9,
                      MPI_DOUBLE,
                      MPI_SUM,
                      d_mpiCommDomain);
        // pcout << "Stress Tensor ESelfEshelby: "
        //       << stressContribESelfEshelby.size() << std::endl;
        // for (dftfe::uInt iDim = 0; iDim < 3; iDim++)
        //   {
        //     for (dftfe::uInt jDim = 0; jDim < 3; jDim++)
        //       pcout << stressContribESelfEshelby[3 * iDim + jDim] << " ";
        //     pcout << std::endl;
        //   }
        for (dftfe::uInt iDim = 0; iDim < 3; iDim++)
          for (dftfe::uInt jDim = 0; jDim < 3; jDim++)
            d_stressTotal[3 * iDim + jDim] +=
              stressContribESelfEshelby[3 * iDim + jDim];
      }
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  configurationalForceClass<memorySpace>::computeElectroContribEshelby(
    const distributedCPUVec<double> &phiTotRhoOutValues,
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
              &rhoOutValues,
    const bool floatingNuclearCharges,
    const bool computeForce,
    const bool computeStress)
  {
    std::vector<double> stressContribElectroEshelby(9, 0.0);
    d_basisOperationsPtrElectroHost->reinit(0, 0, d_densityQuadratureIdElectro);
    const dftfe::uInt nCells = d_basisOperationsPtrElectroHost->nCells();
    const dftfe::uInt nQuadsPerCell =
      d_basisOperationsPtrElectroHost->nQuadsPerCell();
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      phiTotRhoOutQuadValues, gradPhiTotRhoOutQuadValues;
    d_basisOperationsPtrElectroHost->interpolateNoConstraints(
      phiTotRhoOutValues,
      d_basisOperationsPtrElectroHost->d_dofHandlerID,
      d_densityQuadratureIdElectro,
      phiTotRhoOutQuadValues,
      gradPhiTotRhoOutQuadValues,
      gradPhiTotRhoOutQuadValues,
      true,
      false,
      true);
    auto dot3 = [](const double *a, const double *b) noexcept -> double {
      return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    };

    for (dftfe::uInt iCell = 0; iCell < nCells; ++iCell)
      {
        const double *rhoOutValuesCurrentCell =
          rhoOutValues.data() + iCell * nQuadsPerCell;
        const double *phiTotRhoOutQuadValuesCurrentCell =
          phiTotRhoOutQuadValues.data() + iCell * nQuadsPerCell;
        const double *gradPhiTotRhoOutQuadValuesCurrentCell =
          gradPhiTotRhoOutQuadValues.data() + iCell * nQuadsPerCell * 3;
        const double *JxWValues =
          d_basisOperationsPtrElectroHost->JxWBasisData().data() +
          nQuadsPerCell * iCell;
        for (dftfe::uInt iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
          {
            if (computeStress)
              {
                const double diagContrib =
                  (-1.0 / (8.0 * M_PI) *
                     dot3(gradPhiTotRhoOutQuadValuesCurrentCell + iQuad * 3,
                          gradPhiTotRhoOutQuadValuesCurrentCell + iQuad * 3) +
                   rhoOutValuesCurrentCell[iQuad] *
                     phiTotRhoOutQuadValuesCurrentCell[iQuad]) *
                  JxWValues[iQuad];
                for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
                  for (dftfe::uInt jDim = 0; jDim < 3; ++jDim)
                    stressContribElectroEshelby[3 * iDim + jDim] +=
                      1.0 / (4.0 * M_PI) *
                      gradPhiTotRhoOutQuadValuesCurrentCell[iQuad * 3 + jDim] *
                      gradPhiTotRhoOutQuadValuesCurrentCell[iQuad * 3 + iDim] *
                      JxWValues[iQuad];

                for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
                  stressContribElectroEshelby[iDim * 3 + iDim] += diagContrib;
              }
          }
      }
    if (computeStress)
      {
        MPI_Allreduce(MPI_IN_PLACE,
                      stressContribElectroEshelby.data(),
                      9,
                      MPI_DOUBLE,
                      MPI_SUM,
                      d_mpiCommDomain);
        // pcout << "Stress Tensor ElectroEshelby: "
        //       << stressContribElectroEshelby.size() << std::endl;
        // for (dftfe::uInt iDim = 0; iDim < 3; iDim++)
        //   {
        //     for (dftfe::uInt jDim = 0; jDim < 3; jDim++)
        //       pcout << stressContribElectroEshelby[3 * iDim + jDim] << " ";
        //     pcout << std::endl;
        //   }
        for (dftfe::uInt iDim = 0; iDim < 3; iDim++)
          for (dftfe::uInt jDim = 0; jDim < 3; jDim++)
            d_stressTotal[3 * iDim + jDim] +=
              stressContribElectroEshelby[3 * iDim + jDim];
      }
  }


  template <dftfe::utils::MemorySpace memorySpace>
  void
  configurationalForceClass<memorySpace>::computeSmearedContribAll(
    const std::vector<std::vector<double>> &atomLocations,
    const std::vector<std::vector<double>> &imagePositions,
    const vselfBinsManager                 &vselfBinsManager,
    const dftfe::uInt                      &binsStartDofHandlerIndexElectro,
    const distributedCPUVec<double>        &phiTotRhoOutValues,
    const std::map<dealii::CellId, std::vector<dftfe::Int>>
      &bQuadAtomIdsAllAtoms,
    const std::map<dealii::CellId, std::vector<dftfe::Int>>
      &bQuadAtomIdsAllAtomsImages,
    const std::map<dealii::CellId, std::vector<double>> &bQuadValuesAllAtoms,
    const bool                                           floatingNuclearCharges,
    const bool                                           computeForce,
    const bool                                           computeStress)
  {
    std::vector<double> forceContribSmeared(3 * d_dftParams.natoms, 0.0);
    std::vector<double> stressContribSmeared(9, 0.0);
    dftfe::uInt         totalNumAtomsInclImages =
      d_dftParams.natoms + imagePositions.size();
    const dftfe::uInt nCells = d_basisOperationsPtrElectroHost->nCells();
    d_basisOperationsPtrElectroHost->reinit(0,
                                            0,
                                            d_smearedChargeQuadratureIdElectro);
    const dftfe::uInt nQuadsPerCell =
      d_basisOperationsPtrElectroHost->nQuadsPerCell();
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      gradPhiTotRhoOutQuadValues;
    d_basisOperationsPtrElectroHost->interpolateNoConstraints(
      phiTotRhoOutValues,
      d_basisOperationsPtrElectroHost->d_dofHandlerID,
      d_smearedChargeQuadratureIdElectro,
      gradPhiTotRhoOutQuadValues,
      gradPhiTotRhoOutQuadValues,
      gradPhiTotRhoOutQuadValues,
      true,
      false,
      false);
    for (dftfe::uInt iAtom = 0; iAtom < totalNumAtomsInclImages; iAtom++)
      {
        std::array<double, 3> atomLocation;
        if (iAtom < d_dftParams.natoms)
          {
            atomLocation[0] = atomLocations[iAtom][2];
            atomLocation[1] = atomLocations[iAtom][3];
            atomLocation[2] = atomLocations[iAtom][4];
          }
        else
          {
            const dftfe::Int imageId = iAtom - d_dftParams.natoms;
            atomLocation[0]          = imagePositions[imageId][0];
            atomLocation[1]          = imagePositions[imageId][1];
            atomLocation[2]          = imagePositions[imageId][2];
          }
        for (dftfe::uInt iCell = 0; iCell < nCells; ++iCell)
          {
            dealii::CellId currentCellId =
              d_basisOperationsPtrElectroHost->cellID(iCell);
            dealii::DoFHandler<3>::active_cell_iterator currentCellPtr =
              d_basisOperationsPtrElectroHost->getCellIterator(iCell);
            const std::vector<dftfe::Int> &bQuadAtomIdsCell =
              bQuadAtomIdsAllAtoms.find(currentCellId)->second;
            const std::vector<dftfe::Int> &bQuadAtomIdsImagesCell =
              bQuadAtomIdsAllAtomsImages.find(currentCellId)->second;
            const std::vector<double> &bQuadAtomValuesCell =
              bQuadValuesAllAtoms.find(currentCellId)->second;
            const double *JxWValues =
              d_basisOperationsPtrElectroHost->JxWBasisData().data() +
              nQuadsPerCell * iCell;
            const double *gradPhiTotRhoOutQuadValuesCell =
              gradPhiTotRhoOutQuadValues.data() + iCell * nQuadsPerCell * 3;
            const double *quadPointsCurrentCell =
              d_basisOperationsPtrElectroHost->quadPoints().data() +
              iCell * nQuadsPerCell * 3;
            for (dftfe::uInt iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
              {
                if (computeForce)
                  if (bQuadAtomIdsCell[iQuad] == iAtom)
                    {
                      for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
                        forceContribSmeared[3 * iAtom + iDim] +=
                          bQuadAtomValuesCell[iQuad] *
                          gradPhiTotRhoOutQuadValuesCell[iQuad * 3 + iDim] *
                          JxWValues[iQuad];
                    }
                if (computeStress)
                  if (bQuadAtomIdsImagesCell[iQuad] == iAtom)
                    {
                      for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
                        for (dftfe::uInt jDim = 0; jDim < 3; ++jDim)
                          stressContribSmeared[3 * iDim + jDim] -=
                            bQuadAtomValuesCell[iQuad] *
                            gradPhiTotRhoOutQuadValuesCell[iQuad * 3 + iDim] *
                            (quadPointsCurrentCell[iQuad * 3 + jDim] -
                             atomLocation[jDim]) *
                            JxWValues[iQuad];
                    }
              }
          }
      }
    const dftfe::uInt nVSelfBins = vselfBinsManager.getAtomIdsBins().size();
    const std::map<dftfe::Int, std::set<dftfe::Int>> &imageIdsBins =
      vselfBinsManager.getAtomImageIdsBins();
    for (dftfe::uInt iBin = 0; iBin < nVSelfBins; ++iBin)
      {
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
          gradVSelfQuadValuesCurrentBin(3 * nQuadsPerCell * nCells, 0.0);
        d_basisOperationsPtrElectroHost->interpolateNoConstraints(
          vselfBinsManager.getVselfFieldBins()[iBin],
          binsStartDofHandlerIndexElectro + 4 * iBin,
          d_smearedChargeQuadratureIdElectro,
          gradVSelfQuadValuesCurrentBin,
          gradVSelfQuadValuesCurrentBin,
          gradVSelfQuadValuesCurrentBin,
          true,
          false,
          false);
        const std::set<dftfe::Int> &atomIdsInBin =
          imageIdsBins.find(iBin)->second;
        for (const dftfe::Int &iAtom : atomIdsInBin)
          {
            std::array<double, 3> atomLocation;
            if (iAtom < d_dftParams.natoms)
              {
                atomLocation[0] = atomLocations[iAtom][2];
                atomLocation[1] = atomLocations[iAtom][3];
                atomLocation[2] = atomLocations[iAtom][4];
              }
            else
              {
                const dftfe::Int imageId = iAtom - d_dftParams.natoms;
                atomLocation[0]          = imagePositions[imageId][0];
                atomLocation[1]          = imagePositions[imageId][1];
                atomLocation[2]          = imagePositions[imageId][2];
              }
            for (dftfe::uInt iCell = 0; iCell < nCells; ++iCell)
              {
                dealii::CellId currentCellId =
                  d_basisOperationsPtrElectroHost->cellID(iCell);
                dealii::DoFHandler<3>::active_cell_iterator currentCellPtr =
                  d_basisOperationsPtrElectroHost->getCellIterator(iCell);
                const std::vector<dftfe::Int> &bQuadAtomIdsCell =
                  bQuadAtomIdsAllAtoms.find(currentCellId)->second;
                const std::vector<dftfe::Int> &bQuadAtomIdsImagesCell =
                  bQuadAtomIdsAllAtomsImages.find(currentCellId)->second;
                const std::vector<double> &bQuadAtomValuesCell =
                  bQuadValuesAllAtoms.find(currentCellId)->second;
                const double *JxWValues =
                  d_basisOperationsPtrElectroHost->JxWBasisData().data() +
                  nQuadsPerCell * iCell;
                const double *gradVSelfQuadValuesCurrentBinCell =
                  gradVSelfQuadValuesCurrentBin.data() +
                  iCell * nQuadsPerCell * 3;
                const double *quadPointsCurrentCell =
                  d_basisOperationsPtrElectroHost->quadPoints().data() +
                  iCell * nQuadsPerCell * 3;
                for (dftfe::uInt iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
                  {
                    if (computeForce)
                      if (bQuadAtomIdsCell[iQuad] == iAtom)
                        {
                          for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
                            forceContribSmeared[3 * iAtom + iDim] -=
                              bQuadAtomValuesCell[iQuad] *
                              gradVSelfQuadValuesCurrentBinCell[iQuad * 3 +
                                                                iDim] *
                              JxWValues[iQuad];
                        }
                    if (computeStress)
                      if (bQuadAtomIdsImagesCell[iQuad] == iAtom)
                        {
                          for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
                            for (dftfe::uInt jDim = 0; jDim < 3; ++jDim)
                              stressContribSmeared[3 * iDim + jDim] +=
                                bQuadAtomValuesCell[iQuad] *
                                gradVSelfQuadValuesCurrentBinCell[iQuad * 3 +
                                                                  iDim] *
                                (quadPointsCurrentCell[iQuad * 3 + jDim] -
                                 atomLocation[jDim]) *
                                JxWValues[iQuad];
                        }
                  }
              }
          }
      }
    if (computeForce)
      {
        MPI_Allreduce(MPI_IN_PLACE,
                      forceContribSmeared.data(),
                      3 * d_dftParams.natoms,
                      MPI_DOUBLE,
                      MPI_SUM,
                      d_mpiCommDomain);
        // pcout << "Force Vector Smeared: " << forceContribSmeared.size()
        //       << std::endl;
        // for (dftfe::uInt iAtom = 0; iAtom < d_dftParams.natoms; iAtom++)
        //   {
        //     for (dftfe::uInt iDim = 0; iDim < 3; iDim++)
        //       pcout << forceContribSmeared[3 * iAtom + iDim] << " ";
        //     pcout << std::endl;
        //   }
        for (dftfe::uInt iAtom = 0; iAtom < d_dftParams.natoms; iAtom++)
          for (dftfe::uInt iDim = 0; iDim < 3; iDim++)
            d_forceTotal[3 * iAtom + iDim] +=
              forceContribSmeared[3 * iAtom + iDim];
      }
    if (computeStress)
      {
        MPI_Allreduce(MPI_IN_PLACE,
                      stressContribSmeared.data(),
                      9,
                      MPI_DOUBLE,
                      MPI_SUM,
                      d_mpiCommDomain);
        // pcout << "Stress Tensor Smeared: " << stressContribSmeared.size()
        //       << std::endl;
        // for (dftfe::uInt iDim = 0; iDim < 3; iDim++)
        //   {
        //     for (dftfe::uInt jDim = 0; jDim < 3; jDim++)
        //       pcout << stressContribSmeared[3 * iDim + jDim] << " ";
        //     pcout << std::endl;
        //   }
        for (dftfe::uInt iDim = 0; iDim < 3; iDim++)
          for (dftfe::uInt jDim = 0; jDim < 3; jDim++)
            d_stressTotal[3 * iDim + jDim] +=
              stressContribSmeared[3 * iDim + jDim];
      }
  }


  template <dftfe::utils::MemorySpace memorySpace>
  void
  configurationalForceClass<memorySpace>::computeLPSPContribAll(
    const std::vector<std::vector<double>> &atomLocations,
    const std::vector<dftfe::Int>          &imageIds,
    const std::vector<double>              &imageCharges,
    const std::vector<std::vector<double>> &imagePositions,
    const distributedCPUVec<double>        &rhoOutNodalValues,
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &rhoTotalOutValuesLpsp,
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &gradRhoTotalOutValuesLpsp,
    const std::map<dealii::CellId, std::vector<double>> &pseudoVLocValues,
    const std::map<dftfe::uInt, std::map<dealii::CellId, std::vector<double>>>
                                &pseudoVLocAtoms,
    const dealii::DoFHandler<3> &dofHandlerRhoNodal,
    const vselfBinsManager      &vselfBinsManager,
    const std::vector<distributedCPUVec<double>>
              &vselfFieldGateauxDerStrainFDBins,
    const bool floatingNuclearCharges,
    const bool computeForce,
    const bool computeStress)
  {
    std::vector<double> forceContribLPSP(3 * d_dftParams.natoms, 0.0);
    std::vector<double> stressContribLPSP(9, 0.0);
    dftfe::uInt totalNumAtomsInclImages = d_dftParams.natoms + imageIds.size();
    const dftfe::uInt nCells = d_basisOperationsPtrElectroHost->nCells();
    d_basisOperationsPtrElectroHost->reinit(0, 0, d_lpspQuadratureIdElectro);
    const dftfe::uInt nQuadsPerCell =
      d_basisOperationsPtrElectroHost->nQuadsPerCell();


    dealii::QIterated<3 - 1> faceQuadrature(
      dealii::QGauss<1>(C_num1DQuadLPSP(
        d_dftParams.finiteElementPolynomialOrderElectrostatics)),
      C_numCopies1DQuadLPSP());
    dealii::FEFaceValues<3> feFaceValuesElectro(
      dofHandlerRhoNodal.get_fe(),
      faceQuadrature,
      dealii::update_values | dealii::update_JxW_values |
        dealii::update_normal_vectors | dealii::update_quadrature_points);

    const dftfe::uInt nQuadsPerFace =
      feFaceValuesElectro.get_quadrature().size();

    dealii::FEValues<3> feVselfValuesElectro(
      d_basisOperationsPtrElectroHost->getDofHandler().get_fe(),
      d_basisOperationsPtrElectroHost->matrixFreeData().get_quadrature(
        d_lpspQuadratureIdElectro),
      d_dftParams.floatingNuclearCharges && d_dftParams.smearedNuclearCharges ?
        (dealii::update_values | dealii::update_quadrature_points) :
        (dealii::update_values | dealii::update_gradients |
         dealii::update_quadrature_points));

    std::vector<double> surfaceIntegralForceContrib(3, 0.0);
    std::vector<double> surfaceIntegralStressContrib(9, 0.0);
    std::vector<double> rhoFaceQuads(nQuadsPerFace);
    std::vector<double> vselfQuads(nQuadsPerCell, 0.0);
    std::vector<double> pseudoVLocAtomsQuads(nQuadsPerCell, 0.0);
    std::vector<double> vselfDerRQuads(nQuadsPerCell * 3, 0.0);
    std::vector<double> vselfFDStrainQuads(nQuadsPerCell * 9, 0.0);
    std::vector<double> forceContribCurrentCellAtom(3, 0.0);
    std::vector<double> stressContribCurrentCellAtom(9, 0.0);

    dealii::DoFHandler<3>::active_cell_iterator cellPtr;
    auto distance3 = [](const double *a, const double *b) noexcept -> double {
      double dx = a[0] - b[0];
      double dy = a[1] - b[1];
      double dz = a[2] - b[2];
      return std::sqrt(dx * dx + dy * dy + dz * dz);
    };


    for (dftfe::uInt iAtom = 0; iAtom < totalNumAtomsInclImages; iAtom++)
      {
        bool isLocalDomainOutsideVselfBall = false;
        bool isLocalDomainOutsidePspTail   = false;
        if (pseudoVLocAtoms.find(iAtom) == pseudoVLocAtoms.end())
          isLocalDomainOutsidePspTail = true;

        double                atomCharge;
        dftfe::uInt           atomId = iAtom;
        std::array<double, 3> atomLocation;
        if (iAtom < d_dftParams.natoms)
          {
            atomLocation[0] = atomLocations[iAtom][2];
            atomLocation[1] = atomLocations[iAtom][3];
            atomLocation[2] = atomLocations[iAtom][4];
            if (d_dftParams.isPseudopotential)
              atomCharge = atomLocations[iAtom][1];
            else
              atomCharge = atomLocations[iAtom][0];
          }
        else
          {
            const dftfe::Int imageId = iAtom - d_dftParams.natoms;
            atomId                   = imageIds[imageId];
            atomCharge               = imageCharges[imageId];
            atomLocation[0]          = imagePositions[imageId][0];
            atomLocation[1]          = imagePositions[imageId][1];
            atomLocation[2]          = imagePositions[imageId][2];
          }

        dftfe::uInt                                        binIdiAtom;
        std::map<dftfe::uInt, dftfe::uInt>::const_iterator it1 =
          vselfBinsManager.getAtomIdBinIdMapLocalAllImages().find(atomId);
        if (it1 == vselfBinsManager.getAtomIdBinIdMapLocalAllImages().end())
          isLocalDomainOutsideVselfBall = true;
        else
          binIdiAtom = it1->second;

        // Assuming psp tail is larger than vself ball
        if (isLocalDomainOutsidePspTail && isLocalDomainOutsideVselfBall)
          continue;

        std::fill(vselfQuads.begin(), vselfQuads.end(), 0.0);
        std::fill(pseudoVLocAtomsQuads.begin(),
                  pseudoVLocAtomsQuads.end(),
                  0.0);
        std::fill(vselfDerRQuads.begin(), vselfDerRQuads.end(), 0.0);

        bool isTrivial = true;
        for (dftfe::uInt iCell = 0; iCell < nCells; ++iCell)
          {
            std::fill(surfaceIntegralForceContrib.begin(),
                      surfaceIntegralForceContrib.end(),
                      0.0);
            std::fill(surfaceIntegralStressContrib.begin(),
                      surfaceIntegralStressContrib.end(),
                      0.0);
            std::fill(forceContribCurrentCellAtom.begin(),
                      forceContribCurrentCellAtom.end(),
                      0.0);
            std::fill(stressContribCurrentCellAtom.begin(),
                      stressContribCurrentCellAtom.end(),
                      0.0);
            cellPtr = d_basisOperationsPtrElectroHost->getCellIterator(iCell);
            dealii::CellId cellId = cellPtr->id();
            const double  *quadPointsCurrentCell =
              d_basisOperationsPtrElectroHost->quadPoints().data() +
              iCell * nQuadsPerCell * 3;
            const double *JxWValues =
              d_basisOperationsPtrElectroHost->JxWBasisData().data() +
              nQuadsPerCell * iCell;
            // get derivative R vself for iAtom
            bool isCellOutsideVselfBall = true;
            if (!isLocalDomainOutsideVselfBall)
              {
                std::map<dealii::CellId, dftfe::uInt>::const_iterator it2 =
                  d_cellsVselfBallsClosestAtomIdDofHandlerElectro[binIdiAtom]
                    .find(cellId);
                if (it2 !=
                    d_cellsVselfBallsClosestAtomIdDofHandlerElectro[binIdiAtom]
                      .end())
                  {
                    std::array<double, 3> closestAtomLocation;
                    const dftfe::uInt     closestAtomId = it2->second;
                    if (it2->second >= d_dftParams.natoms)
                      {
                        const dftfe::uInt imageIdTrunc =
                          closestAtomId - d_dftParams.natoms;
                        closestAtomLocation[0] =
                          imagePositions[imageIdTrunc][0];
                        closestAtomLocation[1] =
                          imagePositions[imageIdTrunc][1];
                        closestAtomLocation[2] =
                          imagePositions[imageIdTrunc][2];
                      }
                    else
                      {
                        closestAtomLocation[0] =
                          atomLocations[closestAtomId][2];
                        closestAtomLocation[1] =
                          atomLocations[closestAtomId][3];
                        closestAtomLocation[2] =
                          atomLocations[closestAtomId][4];
                      }

                    if (distance3(atomLocation.data(),
                                  closestAtomLocation.data()) < 1e-5)
                      {
                        feVselfValuesElectro.reinit(cellPtr);
                        isCellOutsideVselfBall = false;

                        if (d_dftParams.floatingNuclearCharges &&
                            d_dftParams.smearedNuclearCharges)
                          {
                            std::vector<double> vselfDerRQuadsCell(
                              nQuadsPerCell);
                            for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
                              {
                                feVselfValuesElectro.get_function_values(
                                  vselfBinsManager
                                    .getVselfFieldDerRBins()[3 * binIdiAtom +
                                                             iDim],
                                  vselfDerRQuadsCell);
                                for (dftfe::uInt iQuad = 0;
                                     iQuad < nQuadsPerCell;
                                     ++iQuad)
                                  vselfDerRQuads[iQuad * 3 + iDim] =
                                    vselfDerRQuadsCell[iQuad];
                              }
                          }
                        if (computeStress)
                          {
                            std::vector<double> vselfFDStrainQuadsCell(
                              nQuadsPerCell);
                            dftfe::uInt flattenedIdCount = 0;
                            for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
                              for (dftfe::uInt jDim = 0; jDim <= iDim; jDim++)
                                {
                                  feVselfValuesElectro.get_function_values(
                                    vselfFieldGateauxDerStrainFDBins
                                      [6 * binIdiAtom + flattenedIdCount],
                                    vselfFDStrainQuadsCell);
                                  for (dftfe::uInt iQuad = 0;
                                       iQuad < nQuadsPerCell;
                                       ++iQuad)
                                    {
                                      vselfFDStrainQuads[iQuad * 9 + iDim * 3 +
                                                         jDim] =
                                        vselfFDStrainQuadsCell[iQuad];
                                      vselfFDStrainQuads[iQuad * 9 + jDim * 3 +
                                                         iDim] =
                                        vselfFDStrainQuadsCell[iQuad];
                                    }

                                  flattenedIdCount += 1;
                                }

                            feVselfValuesElectro.get_function_values(
                              vselfBinsManager.getVselfFieldBins()[binIdiAtom],
                              vselfQuads);
                          }
                      }
                  }
              }

            // get grad pseudo VLoc for iAtom
            bool isCellOutsidePspTail = true;
            if (!isLocalDomainOutsidePspTail)
              {
                std::map<dealii::CellId, std::vector<double>>::const_iterator
                  it = pseudoVLocAtoms.find(iAtom)->second.find(cellId);
                if (it != pseudoVLocAtoms.find(iAtom)->second.end())
                  {
                    isCellOutsidePspTail = false;
                    for (dftfe::uInt iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
                      pseudoVLocAtomsQuads[iQuad] = (it->second)[iQuad];
                  }
              }
            else if (!isCellOutsideVselfBall)
              {
                for (dftfe::uInt iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
                  {
                    const double dist =
                      distance3(quadPointsCurrentCell + iQuad * 3,
                                atomLocation.data());
                    pseudoVLocAtomsQuads[iQuad] = -atomCharge / dist;
                  }
              }

            if (isCellOutsideVselfBall && !isCellOutsidePspTail)
              {
                for (dftfe::uInt iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
                  {
                    const double dist =
                      distance3(quadPointsCurrentCell + iQuad * 3,
                                atomLocation.data());
                    vselfQuads[iQuad] = -atomCharge / dist;
                  }
              }

            if (!isCellOutsideVselfBall)
              {
                const std::map<dealii::DoFHandler<3>::active_cell_iterator,
                               std::vector<dftfe::uInt>>
                  &cellsVselfBallSurfacesDofHandler =
                    d_cellFacesVselfBallSurfacesDofHandlerElectro[binIdiAtom];

                if (cellsVselfBallSurfacesDofHandler.find(cellPtr) !=
                    cellsVselfBallSurfacesDofHandler.end())
                  {
                    const std::vector<dftfe::uInt> &dirichletFaceIds =
                      cellsVselfBallSurfacesDofHandler.find(cellPtr)->second;
                    for (dftfe::uInt index = 0; index < dirichletFaceIds.size();
                         index++)
                      {
                        const dftfe::uInt faceId = dirichletFaceIds[index];

                        feFaceValuesElectro.reinit(
                          d_cellIdToActiveCellIteratorMapDofHandlerRhoNodalElectro
                            .find(cellId)
                            ->second,
                          faceId);
                        feFaceValuesElectro.get_function_values(
                          rhoOutNodalValues, rhoFaceQuads);
                        for (dftfe::uInt iQuad = 0; iQuad < nQuadsPerFace;
                             ++iQuad)
                          {
                            std::array<double, 3> quadPoint;
                            quadPoint[0] =
                              feFaceValuesElectro.quadrature_point(iQuad)[0];
                            quadPoint[1] =
                              feFaceValuesElectro.quadrature_point(iQuad)[1];
                            quadPoint[2] =
                              feFaceValuesElectro.quadrature_point(iQuad)[2];
                            const double dist =
                              distance3(quadPoint.data(), atomLocation.data());
                            const double vselfFaceQuadExact =
                              -atomCharge / dist;
                            for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
                              surfaceIntegralForceContrib[iDim] -=
                                rhoFaceQuads[iQuad] * vselfFaceQuadExact *
                                feFaceValuesElectro.normal_vector(iQuad)[iDim] *
                                feFaceValuesElectro.JxW(iQuad);
                            if (computeStress)
                              for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
                                for (dftfe::uInt jDim = 0; jDim < 3; ++jDim)
                                  surfaceIntegralStressContrib[3 * iDim +
                                                               jDim] +=
                                    rhoFaceQuads[iQuad] * vselfFaceQuadExact *
                                    feFaceValuesElectro.normal_vector(
                                      iQuad)[iDim] *
                                    (quadPoint[jDim] - atomLocation[jDim]) *
                                    feFaceValuesElectro.JxW(iQuad);
                          } // q point loop
                      }     // face loop
                  }         // surface cells
              }             // inside or intersecting vself ball

            if (isCellOutsideVselfBall && !isCellOutsidePspTail)
              {
                isTrivial = false;
                for (dftfe::uInt iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
                  for (dftfe::uInt iDim = 0; iDim < 3; iDim++)
                    forceContribCurrentCellAtom[iDim] +=
                      (-gradRhoTotalOutValuesLpsp[iCell * nQuadsPerCell * 3 +
                                                  iQuad * 3 + iDim] *
                         vselfQuads[iQuad] +
                       gradRhoTotalOutValuesLpsp[iCell * nQuadsPerCell * 3 +
                                                 iQuad * 3 + iDim] *
                         pseudoVLocAtomsQuads[iQuad]) *
                      JxWValues[iQuad];
                if (computeStress)
                  for (dftfe::uInt iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
                    for (dftfe::uInt iDim = 0; iDim < 3; iDim++)
                      for (dftfe::uInt jDim = 0; jDim < 3; jDim++)
                        stressContribCurrentCellAtom[iDim * 3 + jDim] +=
                          gradRhoTotalOutValuesLpsp[iCell * nQuadsPerCell * 3 +
                                                    iQuad * 3 + iDim] *
                          (vselfQuads[iQuad] - pseudoVLocAtomsQuads[iQuad]) *
                          (quadPointsCurrentCell[iQuad * 3 + jDim] -
                           atomLocation[jDim]) *
                          JxWValues[iQuad];
              }
            else if (!isCellOutsideVselfBall)
              {
                isTrivial = false;
                for (dftfe::uInt iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
                  for (dftfe::uInt iDim = 0; iDim < 3; iDim++)
                    forceContribCurrentCellAtom[iDim] +=
                      (-rhoTotalOutValuesLpsp[iCell * nQuadsPerCell + iQuad] *
                         vselfDerRQuads[iQuad * 3 + iDim] +
                       gradRhoTotalOutValuesLpsp[iCell * nQuadsPerCell * 3 +
                                                 iQuad * 3 + iDim] *
                         pseudoVLocAtomsQuads[iQuad]) *
                      JxWValues[iQuad];

                if (computeStress)
                  for (dftfe::uInt iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
                    for (dftfe::uInt iDim = 0; iDim < 3; iDim++)
                      for (dftfe::uInt jDim = 0; jDim < 3; jDim++)
                        {
                          stressContribCurrentCellAtom[iDim * 3 + jDim] -=
                            (rhoTotalOutValuesLpsp[iCell * nQuadsPerCell +
                                                   iQuad] *
                               vselfFDStrainQuads[iQuad * 9 + iDim * 3 + jDim] +
                             gradRhoTotalOutValuesLpsp[iCell * nQuadsPerCell *
                                                         3 +
                                                       iQuad * 3 + iDim] *
                               pseudoVLocAtomsQuads[iQuad] *
                               (quadPointsCurrentCell[iQuad * 3 + jDim] -
                                atomLocation[jDim])) *
                            JxWValues[iQuad];

                          if (iDim == jDim)
                            {
                              stressContribCurrentCellAtom[iDim * 3 + jDim] -=
                                rhoTotalOutValuesLpsp[iCell * nQuadsPerCell +
                                                      iQuad] *
                                vselfQuads[iQuad] * JxWValues[iQuad];
                            }
                        }
              }

            if (isTrivial)
              continue;


            for (dftfe::uInt iDim = 0; iDim < 3; iDim++)
              forceContribLPSP[3 * atomId + iDim] +=
                surfaceIntegralForceContrib[iDim] +
                forceContribCurrentCellAtom[iDim];

            if (computeStress)
              for (dftfe::uInt iDim = 0; iDim < 3; iDim++)
                for (dftfe::uInt jDim = 0; jDim < 3; jDim++)
                  stressContribLPSP[3 * iDim + jDim] +=
                    surfaceIntegralStressContrib[3 * iDim + jDim] +
                    stressContribCurrentCellAtom[3 * iDim + jDim];

          } // cell loop
      }     // iAtom loop
    if (computeForce)
      {
        MPI_Allreduce(MPI_IN_PLACE,
                      forceContribLPSP.data(),
                      3 * d_dftParams.natoms,
                      MPI_DOUBLE,
                      MPI_SUM,
                      d_mpiCommDomain);
        // pcout << "Force Vector LPSP: " << forceContribLPSP.size() << std::endl;
        // for (dftfe::uInt iAtom = 0; iAtom < d_dftParams.natoms; iAtom++)
        //   {
        //     for (dftfe::uInt iDim = 0; iDim < 3; iDim++)
        //       pcout << forceContribLPSP[3 * iAtom + iDim] << " ";
        //     pcout << std::endl;
        //   }
        for (dftfe::uInt iAtom = 0; iAtom < d_dftParams.natoms; iAtom++)
          for (dftfe::uInt iDim = 0; iDim < 3; iDim++)
            d_forceTotal[3 * iAtom + iDim] +=
              forceContribLPSP[3 * iAtom + iDim];
      }
    if (computeStress)
      {
        MPI_Allreduce(MPI_IN_PLACE,
                      stressContribLPSP.data(),
                      9,
                      MPI_DOUBLE,
                      MPI_SUM,
                      d_mpiCommDomain);
        // pcout << "Stress Tensor LPSP: " << stressContribLPSP.size()
        //       << std::endl;
        // for (dftfe::uInt iDim = 0; iDim < 3; iDim++)
        //   {
        //     for (dftfe::uInt jDim = 0; jDim < 3; jDim++)
        //       pcout << stressContribLPSP[3 * iDim + jDim] << " ";
        //     pcout << std::endl;
        //   }
        for (dftfe::uInt iDim = 0; iDim < 3; iDim++)
          for (dftfe::uInt jDim = 0; jDim < 3; jDim++)
            d_stressTotal[3 * iDim + jDim] +=
              stressContribLPSP[3 * iDim + jDim];
      }
  }


  template <dftfe::utils::MemorySpace memorySpace>
  void
  configurationalForceClass<memorySpace>::computeXCContribAll(
    const std::vector<std::vector<double>> &atomLocations,
    const std::vector<dftfe::Int>          &imageIds,
    const std::vector<std::vector<double>> &imagePositions,
    const std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &densityOutValues,
    const std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &gradDensityOutValues,
    const std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &tauOutValues,
    const std::shared_ptr<AuxDensityMatrix<memorySpace>>
      auxDensityXCOutRepresentationPtr,
    const std::map<dealii::CellId, std::vector<double>> &rhoCoreValues,
    const std::map<dealii::CellId, std::vector<double>> &gradRhoCoreValues,
    const std::map<dftfe::uInt, std::map<dealii::CellId, std::vector<double>>>
      &gradRhoCoreAtoms,
    const std::map<dftfe::uInt, std::map<dealii::CellId, std::vector<double>>>
              &hessianRhoCoreAtoms,
    const bool floatingNuclearCharges,
    const bool computeForce,
    const bool computeStress)
  {
    dftfe::uInt totalNumAtomsInclImages = d_dftParams.natoms + imageIds.size();
    std::vector<double> forceContribXC(3 * d_dftParams.natoms, 0.0);
    std::vector<double> stressContribXC(9, 0.0);
    d_basisOperationsPtrElectroHost->reinit(0, 0, d_densityQuadratureIdElectro);
    const dftfe::uInt nCells = d_basisOperationsPtrElectroHost->nCells();
    const dftfe::uInt nQuadsPerCell =
      d_basisOperationsPtrElectroHost->nQuadsPerCell();


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
    std::vector<double> &pdexDensityOutSpinDown =
      xDensityOutDataOut[xcRemainderOutputDataAttributes::pdeDensitySpinDown];
    std::vector<double> &pdecDensityOutSpinUp =
      cDensityOutDataOut[xcRemainderOutputDataAttributes::pdeDensitySpinUp];
    std::vector<double> &pdecDensityOutSpinDown =
      cDensityOutDataOut[xcRemainderOutputDataAttributes::pdeDensitySpinDown];

    bool isIntegrationByPartsGradDensityDependenceVxc =
      (d_excManagerPtr->getExcSSDFunctionalObj()->getDensityBasedFamilyType() ==
       densityFamilyType::GGA);

    if (isIntegrationByPartsGradDensityDependenceVxc)
      {
        xDensityOutDataOut[xcRemainderOutputDataAttributes::pdeSigma] =
          std::vector<double>();
        cDensityOutDataOut[xcRemainderOutputDataAttributes::pdeSigma] =
          std::vector<double>();
      }

    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      excTimesJxW(nQuadsPerCell);

    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      VxcSpin0TimesJxW(nQuadsPerCell);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      VxcSpin1TimesJxW(nQuadsPerCell);

    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      derExcWithGradRhoSpin0TimesJxW(
        isIntegrationByPartsGradDensityDependenceVxc ? nQuadsPerCell * 3 : 0);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      derExcWithGradRhoSpin1TimesJxW(
        isIntegrationByPartsGradDensityDependenceVxc ? nQuadsPerCell * 3 : 0);

    for (dftfe::uInt iCell = 0; iCell < nCells; ++iCell)
      {
        d_excManagerPtr->getExcSSDFunctionalObj()->computeRhoTauDependentXCData(
          *auxDensityXCOutRepresentationPtr,
          std::make_pair(iCell * nQuadsPerCell, (iCell + 1) * nQuadsPerCell),
          xDensityOutDataOut,
          cDensityOutDataOut);

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

        std::vector<double> pdexDensityOutSigma;
        std::vector<double> pdecDensityOutSigma;
        if (isIntegrationByPartsGradDensityDependenceVxc)
          {
            pdexDensityOutSigma =
              xDensityOutDataOut[xcRemainderOutputDataAttributes::pdeSigma];
            pdecDensityOutSigma =
              cDensityOutDataOut[xcRemainderOutputDataAttributes::pdeSigma];
          }

        std::unordered_map<DensityDescriptorDataAttributes, std::vector<double>>
                             densityXCOutData;
        std::vector<double> &gradDensityXCOutSpinUp =
          densityXCOutData[DensityDescriptorDataAttributes::gradValuesSpinUp];
        std::vector<double> &gradDensityXCOutSpinDown =
          densityXCOutData[DensityDescriptorDataAttributes::gradValuesSpinDown];

        if (isIntegrationByPartsGradDensityDependenceVxc)
          auxDensityXCOutRepresentationPtr->applyLocalOperations(
            std::make_pair(iCell * nQuadsPerCell, (iCell + 1) * nQuadsPerCell),
            densityXCOutData);

        const double *JxWValues =
          d_basisOperationsPtrElectroHost->JxWBasisData().data() +
          nQuadsPerCell * iCell;
        for (dftfe::uInt iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
          {
            excTimesJxW[iQuad] =
              (xEnergyDensityOut[iQuad] + cEnergyDensityOut[iQuad]) *
              JxWValues[iQuad];
            VxcSpin0TimesJxW[iQuad] =
              (pdexDensityOutSpinUp[iQuad] + pdecDensityOutSpinUp[iQuad]) *
              JxWValues[iQuad];
            VxcSpin1TimesJxW[iQuad] =
              (pdexDensityOutSpinDown[iQuad] + pdecDensityOutSpinDown[iQuad]) *
              JxWValues[iQuad];
          }
        if (isIntegrationByPartsGradDensityDependenceVxc)
          for (dftfe::uInt iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
            for (dftfe::uInt iDim = 0; iDim < 3; iDim++)
              {
                derExcWithGradRhoSpin0TimesJxW[iQuad * 3 + iDim] =
                  (2.0 *
                     (pdexDensityOutSigma[3 * iQuad + 0] +
                      pdecDensityOutSigma[3 * iQuad + 0]) *
                     gradDensityXCOutSpinUp[3 * iQuad + iDim] +
                   (pdexDensityOutSigma[3 * iQuad + 1] +
                    pdecDensityOutSigma[3 * iQuad + 1]) *
                     gradDensityXCOutSpinDown[3 * iQuad + iDim]) *
                  JxWValues[iQuad];
                derExcWithGradRhoSpin1TimesJxW[iQuad * 3 + iDim] =
                  (2.0 *
                     (pdexDensityOutSigma[3 * iQuad + 2] +
                      pdecDensityOutSigma[3 * iQuad + 2]) *
                     gradDensityXCOutSpinDown[3 * iQuad + iDim] +
                   (pdexDensityOutSigma[3 * iQuad + 1] +
                    pdecDensityOutSigma[3 * iQuad + 1]) *
                     gradDensityXCOutSpinUp[3 * iQuad + iDim]) *
                  JxWValues[iQuad];
              }
        if (computeForce)
          {
            for (dftfe::uInt iAtom = 0; iAtom < totalNumAtomsInclImages;
                 iAtom++)
              {
                dftfe::uInt atomId = iAtom < d_dftParams.natoms ?
                                       iAtom :
                                       imageIds[iAtom - d_dftParams.natoms];
                if (gradRhoCoreAtoms.find(iAtom) == gradRhoCoreAtoms.end())
                  continue;
                const auto &gradRhoCoreAtomValuesAllCells =
                  gradRhoCoreAtoms.find(iAtom)->second;
                dealii::CellId currentCellId =
                  d_basisOperationsPtrElectroHost->cellID(iCell);
                if (gradRhoCoreAtomValuesAllCells.find(currentCellId) ==
                    gradRhoCoreAtomValuesAllCells.end())
                  continue;
                const std::vector<double> &gradRhoCoreAtomValuesCurrentCell =
                  gradRhoCoreAtomValuesAllCells.find(currentCellId)->second;
                for (dftfe::uInt iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
                  for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
                    forceContribXC[atomId * 3 + iDim] -=
                      gradRhoCoreAtomValuesCurrentCell[3 * iQuad + iDim] *
                      (VxcSpin0TimesJxW[iQuad] + VxcSpin1TimesJxW[iQuad]) * 0.5;
                if (isIntegrationByPartsGradDensityDependenceVxc)
                  {
                    const std::vector<double>
                      &hessianRhoCoreAtomValuesCurrentCell =
                        hessianRhoCoreAtoms.find(iAtom)
                          ->second.find(currentCellId)
                          ->second;

                    for (dftfe::uInt iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
                      for (dftfe::uInt iDim = 0; iDim < 3; iDim++)
                        for (dftfe::uInt jDim = 0; jDim < 3; jDim++)
                          forceContribXC[atomId * 3 + iDim] -=
                            hessianRhoCoreAtomValuesCurrentCell[iQuad * 3 * 3 +
                                                                3 * jDim +
                                                                iDim] *
                            (derExcWithGradRhoSpin0TimesJxW[iQuad * 3 + jDim] +
                             derExcWithGradRhoSpin1TimesJxW[iQuad * 3 + jDim]) *
                            0.5;
                  }
              }
          }
        if (computeStress)
          {
            double integralexc = 0.0;
            for (dftfe::uInt iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
              integralexc += excTimesJxW[iQuad];
            for (dftfe::uInt iDim = 0; iDim < 3; iDim++)
              stressContribXC[iDim * 3 + iDim] += integralexc;
            if (isIntegrationByPartsGradDensityDependenceVxc)
              {
                const double *cellGradRhoValues =
                  gradDensityOutValues[0].data() + iCell * nQuadsPerCell * 3;
                const double *cellGradMagZValues =
                  gradDensityOutValues[1].data() + iCell * nQuadsPerCell * 3;
                for (dftfe::uInt iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
                  for (dftfe::uInt iDim = 0; iDim < 3; iDim++)
                    for (dftfe::uInt jDim = 0; jDim < 3; jDim++)
                      stressContribXC[iDim * 3 + jDim] -=
                        derExcWithGradRhoSpin0TimesJxW[iQuad * 3 + iDim] *
                          (cellGradRhoValues[3 * iQuad + jDim] +
                           cellGradMagZValues[3 * iQuad + jDim]) *
                          0.5 +
                        derExcWithGradRhoSpin1TimesJxW[iQuad * 3 + iDim] *
                          (cellGradRhoValues[3 * iQuad + jDim] -
                           cellGradMagZValues[3 * iQuad + jDim]) *
                          0.5;
              }
            for (dftfe::uInt iAtom = 0; iAtom < totalNumAtomsInclImages;
                 iAtom++)
              {
                dftfe::uInt atomId = iAtom < d_dftParams.natoms ?
                                       iAtom :
                                       imageIds[iAtom - d_dftParams.natoms];
                if (gradRhoCoreAtoms.find(iAtom) == gradRhoCoreAtoms.end())
                  continue;
                const auto &gradRhoCoreAtomValuesAllCells =
                  gradRhoCoreAtoms.find(iAtom)->second;
                dealii::CellId currentCellId =
                  d_basisOperationsPtrElectroHost->cellID(iCell);
                if (gradRhoCoreAtomValuesAllCells.find(currentCellId) ==
                    gradRhoCoreAtomValuesAllCells.end())
                  continue;
                const std::vector<double> &gradRhoCoreAtomValuesCurrentCell =
                  gradRhoCoreAtomValuesAllCells.find(currentCellId)->second;

                const double *quadPointsCurrentCell =
                  d_basisOperationsPtrElectroHost->quadPoints().data() +
                  iCell * nQuadsPerCell * 3;
                for (dftfe::uInt iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
                  {
                    std::vector<double> dispAtomToQuad(3, 0.0);
                    if (iAtom < d_dftParams.natoms)
                      for (dftfe::uInt iDim = 0; iDim < 3; iDim++)
                        dispAtomToQuad[iDim] =
                          quadPointsCurrentCell[3 * iQuad + iDim] -
                          atomLocations[iAtom][2 + iDim];
                    else
                      for (dftfe::uInt iDim = 0; iDim < 3; iDim++)
                        dispAtomToQuad[iDim] =
                          quadPointsCurrentCell[3 * iQuad + iDim] -
                          imagePositions[iAtom - d_dftParams.natoms][iDim];

                    for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
                      for (dftfe::uInt jDim = 0; jDim < 3; jDim++)
                        stressContribXC[iDim * 3 + jDim] +=
                          dispAtomToQuad[jDim] *
                          gradRhoCoreAtomValuesCurrentCell[3 * iQuad + iDim] *
                          (VxcSpin0TimesJxW[iQuad] + VxcSpin1TimesJxW[iQuad]) *
                          0.5;
                  }
                if (isIntegrationByPartsGradDensityDependenceVxc)
                  {
                    const std::vector<double>
                      &hessianRhoCoreAtomValuesCurrentCell =
                        hessianRhoCoreAtoms.find(iAtom)
                          ->second.find(currentCellId)
                          ->second;

                    for (dftfe::uInt iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
                      {
                        std::vector<double> dispAtomToQuad(3, 0.0);
                        if (iAtom < d_dftParams.natoms)
                          for (dftfe::uInt iDim = 0; iDim < 3; iDim++)
                            dispAtomToQuad[iDim] =
                              quadPointsCurrentCell[3 * iQuad + iDim] -
                              atomLocations[iAtom][2 + iDim];
                        else
                          for (dftfe::uInt iDim = 0; iDim < 3; iDim++)
                            dispAtomToQuad[iDim] =
                              quadPointsCurrentCell[3 * iQuad + iDim] -
                              imagePositions[iAtom - d_dftParams.natoms][iDim];

                        for (dftfe::uInt iDim = 0; iDim < 3; iDim++)
                          for (dftfe::uInt jDim = 0; jDim < 3; jDim++)
                            for (dftfe::uInt kDim = 0; kDim < 3; kDim++)
                              stressContribXC[iDim * 3 + jDim] +=
                                dispAtomToQuad[jDim] *
                                hessianRhoCoreAtomValuesCurrentCell
                                  [iQuad * 3 * 3 + 3 * kDim + iDim] *
                                (derExcWithGradRhoSpin0TimesJxW[iQuad * 3 +
                                                                kDim] +
                                 derExcWithGradRhoSpin1TimesJxW[iQuad * 3 +
                                                                kDim]) *
                                0.5;
                      }
                  }
              }
          }
      } // cell loop
    if (computeForce)
      {
        MPI_Allreduce(MPI_IN_PLACE,
                      forceContribXC.data(),
                      3 * d_dftParams.natoms,
                      MPI_DOUBLE,
                      MPI_SUM,
                      d_mpiCommDomain);
        // pcout << "Force Vector XC: " << forceContribXC.size() << std::endl;
        // for (dftfe::uInt iAtom = 0; iAtom < d_dftParams.natoms; iAtom++)
        //   {
        //     for (dftfe::uInt iDim = 0; iDim < 3; iDim++)
        //       pcout << forceContribXC[3 * iAtom + iDim] << " ";
        //     pcout << std::endl;
        //   }
        for (dftfe::uInt iAtom = 0; iAtom < d_dftParams.natoms; iAtom++)
          for (dftfe::uInt iDim = 0; iDim < 3; iDim++)
            d_forceTotal[3 * iAtom + iDim] += forceContribXC[3 * iAtom + iDim];
      }
    if (computeStress)
      {
        MPI_Allreduce(MPI_IN_PLACE,
                      stressContribXC.data(),
                      9,
                      MPI_DOUBLE,
                      MPI_SUM,
                      d_mpiCommDomain);
        // pcout << "Stress Tensor XC: " << stressContribXC.size() << std::endl;
        // for (dftfe::uInt iDim = 0; iDim < 3; iDim++)
        //   {
        //     for (dftfe::uInt jDim = 0; jDim < 3; jDim++)
        //       pcout << stressContribXC[3 * iDim + jDim] << " ";
        //     pcout << std::endl;
        //   }
        for (dftfe::uInt iDim = 0; iDim < 3; iDim++)
          for (dftfe::uInt jDim = 0; jDim < 3; jDim++)
            d_stressTotal[3 * iDim + jDim] += stressContribXC[3 * iDim + jDim];
      }
  }


  template <dftfe::utils::MemorySpace memorySpace>
  void
  configurationalForceClass<memorySpace>::computeWfcContribAll(
    const dftfe::uInt         &numEigenValues,
    const std::vector<double> &kPointCoords,
    const std::vector<double> &kPointWeights,
    const dftfe::utils::MemoryStorage<dataTypes::number, memorySpace>
                                           &eigenVectors,
    const std::vector<std::vector<double>> &eigenValues,
    const std::vector<std::vector<double>> &partialOccupancies,
    const bool                              floatingNuclearCharges,
    const bool                              computeForce,
    const bool                              computeStress)
  {
    std::vector<dataTypes::number> ForceNlocContrib(d_dftParams.natoms * 3,
                                                    0.0);
    std::vector<dataTypes::number> StressNlocContrib(9, 0.0);
    std::vector<dataTypes::number> generatorAtAtomsNlocContribForce(
      d_dftParams.natoms, 0.0);
    std::vector<dataTypes::number> generatorAtAtomsNlocContribStress(
      d_dftParams.natoms * 3, 0.0);
    const dftfe::uInt nCells       = d_basisOperationsPtr->nCells();
    const dftfe::uInt nDofsPerCell = d_basisOperationsPtr->nDofsPerCell();
    const dftfe::uInt numLocalDofs = d_basisOperationsPtr->nOwnedDofs();
    const dftfe::uInt totalLocallyOwnedCells = d_basisOperationsPtr->nCells();

    const dftfe::uInt cellsBlockSize =
      memorySpace == dftfe::utils::MemorySpace::DEVICE ?
        (d_dftParams.memOptMode ? 50 : nCells) :
        1;
    const dftfe::uInt numCellBlocks = totalLocallyOwnedCells / cellsBlockSize;
    const dftfe::uInt remCellBlockSize =
      totalLocallyOwnedCells - numCellBlocks * cellsBlockSize;


    const dftfe::uInt numberBandGroups =
      dealii::Utilities::MPI::n_mpi_processes(d_mpiCommInterBandGroup);
    const dftfe::uInt bandGroupTaskId =
      dealii::Utilities::MPI::this_mpi_process(d_mpiCommInterBandGroup);
    std::vector<dftfe::uInt> bandGroupLowHighPlusOneIndices;
    dftUtils::createBandParallelizationIndices(d_mpiCommInterBandGroup,
                                               numEigenValues,
                                               bandGroupLowHighPlusOneIndices);

    const dftfe::uInt wfcBlockSize =
      std::min(d_dftParams.chebyWfcBlockSize,
               bandGroupLowHighPlusOneIndices[1]);

    const double spinPolarizedFactor =
      (d_dftParams.spinPolarized == 1) ? 1.0 : 2.0;
    const dftfe::uInt numSpinComponents =
      (d_dftParams.spinPolarized == 1) ? 2 : 1;

    dftfe::utils::MemoryStorage<dataTypes::number, memorySpace>
      cellWaveFunctionMatrix(cellsBlockSize * nDofsPerCell * wfcBlockSize);

    dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace>
      *flattenedArrayBlock;
    dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace>
      couplingMatrixTimesPseudopotentialNonLocalProjectorTimesVectorBlock;
    dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace>
      pseudopotentialNonLocalProjectorTimesVectorBlock;
    dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace>
      pseudopotentialNonLocalProjectorTimesXTimesVectorBlock;
    dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace>
      pseudopotentialNonLocalProjectorTimesGradientVectorBlock;
    dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace>
      pseudopotentialNonLocalProjectorTimesRDyadicGradientVectorBlock;
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      sqrtPartialOccupVecHost(wfcBlockSize, 0.0);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      kCoordHost(3, 0.0);
#if defined(DFTFE_WITH_DEVICE)
    dftfe::utils::MemoryStorage<double, memorySpace> sqrtPartialOccupVec(
      sqrtPartialOccupVecHost.size());
    dftfe::utils::MemoryStorage<double, memorySpace> kCoord(kCoordHost.size());
#else
    auto &sqrtPartialOccupVec = sqrtPartialOccupVecHost;
    auto &kCoord              = kCoordHost;
#endif
    for (dftfe::uInt kPoint = 0; kPoint < kPointWeights.size(); ++kPoint)
      {
        kCoordHost[0]           = kPointCoords[3 * kPoint + 0];
        kCoordHost[1]           = kPointCoords[3 * kPoint + 1];
        kCoordHost[2]           = kPointCoords[3 * kPoint + 2];
        const bool isGammaPoint = (std::abs(kCoordHost[0] - 0.0) < 1e-8 &&
                                   std::abs(kCoordHost[1] - 0.0) < 1e-8 &&
                                   std::abs(kCoordHost[2] - 0.0) < 1e-8);
        std::vector<allReduceVectorType> nonLocalOperationsList{
          allReduceVectorType::CconjTransX};
        if (computeForce)
          nonLocalOperationsList.push_back(allReduceVectorType::DconjTransX);
        if (computeStress)
          nonLocalOperationsList.push_back(
            allReduceVectorType::DDyadicRconjTransX);
        if (computeStress && !isGammaPoint)
          nonLocalOperationsList.push_back(allReduceVectorType::CRconjTransX);
        d_pseudopotentialNonLocalOperator->initialiseOperatorActionOnX(
          kPoint, allReduceVectorType::CconjTransX);
        d_pseudopotentialNonLocalOperator->initialiseFlattenedDataStructure(
          wfcBlockSize,
          couplingMatrixTimesPseudopotentialNonLocalProjectorTimesVectorBlock,
          allReduceVectorType::CconjTransX);
        if (computeForce)
          {
            d_pseudopotentialNonLocalOperator->initialiseOperatorActionOnX(
              kPoint, allReduceVectorType::DconjTransX);
            d_pseudopotentialNonLocalOperator->initialiseFlattenedDataStructure(
              wfcBlockSize,
              pseudopotentialNonLocalProjectorTimesGradientVectorBlock,
              allReduceVectorType::DconjTransX);
          }
        if (computeStress)
          {
            d_pseudopotentialNonLocalOperator->initialiseOperatorActionOnX(
              kPoint, allReduceVectorType::DDyadicRconjTransX);
            d_pseudopotentialNonLocalOperator->initialiseFlattenedDataStructure(
              wfcBlockSize,
              pseudopotentialNonLocalProjectorTimesRDyadicGradientVectorBlock,
              allReduceVectorType::DDyadicRconjTransX);
            if (!isGammaPoint)
              {
                d_pseudopotentialNonLocalOperator->initialiseOperatorActionOnX(
                  kPoint, allReduceVectorType::CRconjTransX);
                d_pseudopotentialNonLocalOperator
                  ->initialiseFlattenedDataStructure(
                    wfcBlockSize,
                    pseudopotentialNonLocalProjectorTimesXTimesVectorBlock,
                    allReduceVectorType::CRconjTransX);
              }
          }
        if constexpr (dftfe::utils::MemorySpace::DEVICE == memorySpace)
          {
            d_pseudopotentialNonLocalOperator->freeDeviceVectors();
            d_pseudopotentialNonLocalOperator
              ->initialiseCellWaveFunctionPointers(cellWaveFunctionMatrix,
                                                   cellsBlockSize,
                                                   nonLocalOperationsList);
          }

        for (dftfe::uInt spinIndex = 0; spinIndex < numSpinComponents;
             ++spinIndex)
          {
            for (dftfe::uInt jvec = 0; jvec < numEigenValues;
                 jvec += wfcBlockSize)
              {
                const dftfe::uInt currentBlockSize =
                  std::min(wfcBlockSize, numEigenValues - jvec);
                flattenedArrayBlock =
                  &(d_basisOperationsPtr->getMultiVector(currentBlockSize, 0));
                if ((jvec + currentBlockSize) <=
                      bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
                    (jvec + currentBlockSize) >
                      bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
                  {
                    if constexpr (dftfe::utils::MemorySpace::HOST ==
                                  memorySpace)
                      {
                        d_pseudopotentialNonLocalOperator
                          ->initialiseOperatorActionOnX(
                            kPoint, allReduceVectorType::CconjTransX);
                        if (computeForce)
                          d_pseudopotentialNonLocalOperator
                            ->initialiseOperatorActionOnX(
                              kPoint, allReduceVectorType::DconjTransX);
                        if (computeStress)
                          {
                            d_pseudopotentialNonLocalOperator
                              ->initialiseOperatorActionOnX(
                                kPoint,
                                allReduceVectorType::DDyadicRconjTransX);
                            if (!isGammaPoint)
                              d_pseudopotentialNonLocalOperator
                                ->initialiseOperatorActionOnX(
                                  kPoint, allReduceVectorType::CRconjTransX);
                          }
                        if (wfcBlockSize != currentBlockSize)
                          {
                            d_pseudopotentialNonLocalOperator
                              ->initialiseFlattenedDataStructure(
                                currentBlockSize,
                                couplingMatrixTimesPseudopotentialNonLocalProjectorTimesVectorBlock,
                                allReduceVectorType::CconjTransX);
                            if (computeForce)
                              d_pseudopotentialNonLocalOperator
                                ->initialiseFlattenedDataStructure(
                                  currentBlockSize,
                                  pseudopotentialNonLocalProjectorTimesGradientVectorBlock,
                                  allReduceVectorType::DconjTransX);
                            if (computeStress)
                              {
                                d_pseudopotentialNonLocalOperator
                                  ->initialiseFlattenedDataStructure(
                                    currentBlockSize,
                                    pseudopotentialNonLocalProjectorTimesRDyadicGradientVectorBlock,
                                    allReduceVectorType::DDyadicRconjTransX);
                                if (!isGammaPoint)
                                  d_pseudopotentialNonLocalOperator
                                    ->initialiseFlattenedDataStructure(
                                      currentBlockSize,
                                      pseudopotentialNonLocalProjectorTimesXTimesVectorBlock,
                                      allReduceVectorType::CRconjTransX);
                              }
                          }
                      }
                    for (dftfe::uInt iEigenVec = 0;
                         iEigenVec < currentBlockSize;
                         ++iEigenVec)
                      *(sqrtPartialOccupVecHost.begin() + iEigenVec) =
                        std::sqrt(partialOccupancies[kPoint][numEigenValues *
                                                               spinIndex +
                                                             jvec + iEigenVec] *
                                  kPointWeights[kPoint] * spinPolarizedFactor);

#if defined(DFTFE_WITH_DEVICE)
                    sqrtPartialOccupVec.copyFrom(sqrtPartialOccupVecHost);
                    kCoord.copyFrom(kCoordHost);
#endif
                    d_BLASWrapperPtr->stridedCopyToBlockConstantStride(
                      currentBlockSize,
                      numEigenValues,
                      numLocalDofs,
                      jvec,
                      eigenVectors.data() +
                        numLocalDofs * numEigenValues *
                          (numSpinComponents * kPoint + spinIndex),
                      flattenedArrayBlock->data());

                    d_basisOperationsPtr->reinit(currentBlockSize,
                                                 cellsBlockSize,
                                                 0,
                                                 false);

                    d_BLASWrapperPtr->rightDiagonalScale(
                      flattenedArrayBlock->numVectors(),
                      flattenedArrayBlock->locallyOwnedSize(),
                      flattenedArrayBlock->data(),
                      sqrtPartialOccupVec.data());

                    flattenedArrayBlock->updateGhostValues();
                    d_basisOperationsPtr->distribute(*(flattenedArrayBlock));

                    for (dftfe::Int iCellBlock = 0;
                         iCellBlock < (numCellBlocks + 1);
                         iCellBlock++)
                      {
                        const dftfe::uInt currentCellsBlockSize =
                          (iCellBlock == numCellBlocks) ? remCellBlockSize :
                                                          cellsBlockSize;
                        if (currentCellsBlockSize > 0)
                          {
                            const dftfe::uInt startingCellId =
                              iCellBlock * cellsBlockSize;
                            std::pair<dftfe::uInt, dftfe::uInt> cellRange(
                              startingCellId,
                              startingCellId + currentCellsBlockSize);
                            d_basisOperationsPtr->extractToCellNodalDataKernel(
                              *(flattenedArrayBlock),
                              cellWaveFunctionMatrix.data(),
                              cellRange);
                            d_pseudopotentialNonLocalOperator
                              ->applyCconjtransOnX(
                                cellWaveFunctionMatrix.data(), cellRange);
                            if (computeForce)
                              d_pseudopotentialNonLocalOperator
                                ->applyDconjtransOnX(
                                  cellWaveFunctionMatrix.data(), cellRange);
                            if (computeStress)
                              {
                                d_pseudopotentialNonLocalOperator
                                  ->applyDDyadicRconjtransOnX(
                                    cellWaveFunctionMatrix.data(), cellRange);
                                if (!isGammaPoint)
                                  d_pseudopotentialNonLocalOperator
                                    ->applyCRconjtransOnX(
                                      cellWaveFunctionMatrix.data(), cellRange);
                              }

                          } // non-trivial cell block check
                      }     // cells block loop

                    couplingMatrixTimesPseudopotentialNonLocalProjectorTimesVectorBlock
                      .setValue(0);
                    d_pseudopotentialNonLocalOperator
                      ->applyAllReduceOnCconjtransX(
                        couplingMatrixTimesPseudopotentialNonLocalProjectorTimesVectorBlock,
                        false,
                        allReduceVectorType::CconjTransX);
                    d_pseudopotentialNonLocalOperator->applyVOnCconjtransX(
                      CouplingStructure::diagonal,
                      d_pseudopotentialClassPtr->getCouplingMatrix(),
                      couplingMatrixTimesPseudopotentialNonLocalProjectorTimesVectorBlock,
                      false);
                    if (computeForce)
                      {
                        pseudopotentialNonLocalProjectorTimesGradientVectorBlock
                          .setValue(0);
                        d_pseudopotentialNonLocalOperator
                          ->applyAllReduceOnCconjtransX(
                            pseudopotentialNonLocalProjectorTimesGradientVectorBlock,
                            false,
                            allReduceVectorType::DconjTransX);
                        if (!isGammaPoint)
                          pseudopotentialNonLocalProjectorTimesVectorBlock =
                            couplingMatrixTimesPseudopotentialNonLocalProjectorTimesVectorBlock;
                      }
                    if (computeStress)
                      {
                        pseudopotentialNonLocalProjectorTimesRDyadicGradientVectorBlock
                          .setValue(0);
                        d_pseudopotentialNonLocalOperator
                          ->applyAllReduceOnCconjtransX(
                            pseudopotentialNonLocalProjectorTimesRDyadicGradientVectorBlock,
                            false,
                            allReduceVectorType::DDyadicRconjTransX);
                        if (!isGammaPoint)
                          {
                            pseudopotentialNonLocalProjectorTimesXTimesVectorBlock
                              .setValue(0);
                            d_pseudopotentialNonLocalOperator
                              ->applyAllReduceOnCconjtransX(
                                pseudopotentialNonLocalProjectorTimesXTimesVectorBlock,
                                false,
                                allReduceVectorType::CRconjTransX);
                          }
                      }
                    if (computeForce)
                      d_pseudopotentialNonLocalOperator
                        ->computeInnerProductOverSphericalFnsWaveFns(
                          3,
                          couplingMatrixTimesPseudopotentialNonLocalProjectorTimesVectorBlock,
                          pseudopotentialNonLocalProjectorTimesGradientVectorBlock,
                          false,
                          ForceNlocContrib);
                    if (computeStress)
                      d_pseudopotentialNonLocalOperator
                        ->computeInnerProductOverSphericalFnsWaveFns(
                          9,
                          couplingMatrixTimesPseudopotentialNonLocalProjectorTimesVectorBlock,
                          pseudopotentialNonLocalProjectorTimesRDyadicGradientVectorBlock,
                          false,
                          StressNlocContrib);
                    if constexpr (std::is_same<dataTypes::number,
                                               std::complex<double>>::value)
                      if (!isGammaPoint)
                        {
                          if (computeForce)
                            {
                              std::fill(
                                generatorAtAtomsNlocContribForce.begin(),
                                generatorAtAtomsNlocContribForce.end(),
                                0.0);
                              d_pseudopotentialNonLocalOperator
                                ->computeInnerProductOverSphericalFnsWaveFns(
                                  1,
                                  couplingMatrixTimesPseudopotentialNonLocalProjectorTimesVectorBlock,
                                  pseudopotentialNonLocalProjectorTimesVectorBlock,
                                  false,
                                  generatorAtAtomsNlocContribForce);
                              for (dftfe::uInt iAtom = 0;
                                   iAtom < d_dftParams.natoms;
                                   iAtom++)
                                for (dftfe::uInt iDim = 0; iDim < 3; iDim++)
                                  ForceNlocContrib[3 * iAtom + iDim] +=
                                    dataTypes::number(
                                      kCoordHost[iDim] *
                                      std::complex<double>(0.0, 1.0) *
                                      generatorAtAtomsNlocContribForce[iAtom]);
                            }
                          if (computeStress)
                            {
                              std::fill(
                                generatorAtAtomsNlocContribStress.begin(),
                                generatorAtAtomsNlocContribStress.end(),
                                0.0);
                              d_pseudopotentialNonLocalOperator
                                ->computeInnerProductOverSphericalFnsWaveFns(
                                  3,
                                  couplingMatrixTimesPseudopotentialNonLocalProjectorTimesVectorBlock,
                                  pseudopotentialNonLocalProjectorTimesXTimesVectorBlock,
                                  false,
                                  generatorAtAtomsNlocContribStress);
                              for (dftfe::uInt iAtom = 0;
                                   iAtom < d_dftParams.natoms;
                                   iAtom++)
                                for (dftfe::uInt iDim = 0; iDim < 3; iDim++)
                                  for (dftfe::uInt jDim = 0; jDim < 3; jDim++)
                                    StressNlocContrib[3 * iDim + jDim] -=
                                      dataTypes::number(
                                        kCoordHost[iDim] *
                                        std::complex<double>(0.0, 1.0) *
                                        generatorAtAtomsNlocContribStress
                                          [3 * iAtom + jDim]);
                            }
                        }
                  }
              }
          }
      }
    if (computeForce)
      {
        std::vector<double> ForceVector(3 * d_dftParams.natoms, 0.0);
        for (dftfe::uInt iAtom = 0; iAtom < d_dftParams.natoms; iAtom++)
          for (dftfe::uInt iDim = 0; iDim < 3; iDim++)
            ForceVector[3 * iAtom + iDim] +=
              2.0 * std::real(ForceNlocContrib[3 * iAtom + iDim]);
        MPI_Allreduce(MPI_IN_PLACE,
                      ForceVector.data(),
                      3 * d_dftParams.natoms,
                      MPI_DOUBLE,
                      MPI_SUM,
                      d_mpiCommParent);
        // pcout << "Force Vector: " << ForceVector.size() << std::endl;
        // for (dftfe::uInt iAtom = 0; iAtom < d_dftParams.natoms; iAtom++)
        //   {
        //     for (dftfe::uInt iDim = 0; iDim < 3; iDim++)
        //       pcout << ForceVector[3 * iAtom + iDim] << " ";
        //     pcout << std::endl;
        //   }
        for (dftfe::uInt iAtom = 0; iAtom < d_dftParams.natoms; iAtom++)
          for (dftfe::uInt iDim = 0; iDim < 3; iDim++)
            d_forceTotal[3 * iAtom + iDim] += ForceVector[3 * iAtom + iDim];
      }
    if (computeStress)
      {
        std::vector<double> StressTensor(3 * 3, 0.0);
        for (dftfe::uInt iDim = 0; iDim < 3; iDim++)
          for (dftfe::uInt jDim = 0; jDim < 3; jDim++)
            StressTensor[3 * iDim + jDim] +=
              2.0 * std::real(-StressNlocContrib[3 * jDim + iDim]);
        MPI_Allreduce(MPI_IN_PLACE,
                      StressTensor.data(),
                      9,
                      MPI_DOUBLE,
                      MPI_SUM,
                      d_mpiCommParent);
        // pcout << "Stress Tensor: " << StressTensor.size() << std::endl;
        // for (dftfe::uInt iDim = 0; iDim < 3; iDim++)
        //   {
        //     for (dftfe::uInt jDim = 0; jDim < 3; jDim++)
        //       pcout << StressTensor[3 * iDim + jDim] << " ";
        //     pcout << std::endl;
        //   }
        for (dftfe::uInt iDim = 0; iDim < 3; iDim++)
          for (dftfe::uInt jDim = 0; jDim < 3; jDim++)
            d_stressTotal[3 * iDim + jDim] += StressTensor[3 * iDim + jDim];
      }
  }


  template <dftfe::utils::MemorySpace memorySpace>
  void
  configurationalForceClass<memorySpace>::createBinObjectsForce(
    const dealii::DoFHandler<3> &dofHandlerRhoNodal,
    const vselfBinsManager      &vselfBinsManager,
    std::vector<std::vector<dealii::DoFHandler<3>::active_cell_iterator>>
      &cellsVselfBallsDofHandler,
    std::vector<std::vector<dealii::DoFHandler<3>::active_cell_iterator>>
      &cellsVselfBallsDofHandlerForce,
    std::vector<std::map<dealii::CellId, dftfe::uInt>>
                                       &cellsVselfBallsClosestAtomIdDofHandler,
    std::map<dftfe::uInt, dftfe::uInt> &AtomIdBinIdLocalDofHandler,
    std::vector<std::map<dealii::DoFHandler<3>::active_cell_iterator,
                         std::vector<dftfe::uInt>>>
      &cellFacesVselfBallSurfacesDofHandler,
    std::vector<std::map<dealii::DoFHandler<3>::active_cell_iterator,
                         std::vector<dftfe::uInt>>>
      &cellFacesVselfBallSurfacesDofHandlerForce)
  {
    const dealii::DoFHandler<3> &dofHandler =
      d_basisOperationsPtrElectroHost->getDofHandler();
    const dealii::AffineConstraints<double> &hangingPlusPBCConstraints =
      d_basisOperationsPtrElectroHost->matrixFreeData().get_affine_constraints(
        d_basisOperationsPtrElectroHost->d_dofHandlerID);
    dealii::DoFHandler<3> dofHandlerForce;
    dofHandlerForce.clear();
    dofHandlerForce.reinit(dofHandler.get_triangulation());
    dofHandlerForce.distribute_dofs(FEForce);

    const dftfe::uInt faces_per_cell = dealii::GeometryInfo<3>::faces_per_cell;
    const dftfe::uInt dofs_per_cell  = dofHandler.get_fe().dofs_per_cell;
    const dftfe::uInt dofs_per_face  = dofHandler.get_fe().dofs_per_face;
    const dftfe::uInt nVSelfBins     = vselfBinsManager.getAtomIdsBins().size();
    // clear exisitng data
    cellsVselfBallsDofHandler.clear();
    cellsVselfBallsDofHandlerForce.clear();
    cellFacesVselfBallSurfacesDofHandler.clear();
    cellFacesVselfBallSurfacesDofHandlerForce.clear();
    cellsVselfBallsClosestAtomIdDofHandler.clear();
    AtomIdBinIdLocalDofHandler.clear();
    // resize
    cellsVselfBallsDofHandler.resize(nVSelfBins);
    cellsVselfBallsDofHandlerForce.resize(nVSelfBins);
    cellFacesVselfBallSurfacesDofHandler.resize(nVSelfBins);
    cellFacesVselfBallSurfacesDofHandlerForce.resize(nVSelfBins);
    cellsVselfBallsClosestAtomIdDofHandler.resize(nVSelfBins);

    for (dftfe::uInt iBin = 0; iBin < nVSelfBins; ++iBin)
      {
        const std::map<dealii::types::global_dof_index, dftfe::Int>
          &boundaryNodeMap = vselfBinsManager.getBoundaryFlagsBins()[iBin];
        const std::map<dealii::types::global_dof_index, dftfe::Int>
          &closestAtomBinMap = vselfBinsManager.getClosestAtomIdsBins()[iBin];
        dealii::DoFHandler<3>::active_cell_iterator cell =
          dofHandler.begin_active();
        dealii::DoFHandler<3>::active_cell_iterator endc = dofHandler.end();
        dealii::DoFHandler<3>::active_cell_iterator cellForce =
          dofHandlerForce.begin_active();
        for (; cell != endc; ++cell, ++cellForce)
          {
            if (cell->is_locally_owned())
              {
                std::vector<dftfe::uInt> dirichletFaceIds;
                std::vector<dftfe::uInt>
                  faceIdsWithAtleastOneSolvedNonHangingNode;
                std::vector<dftfe::uInt> allFaceIdsOfCell;
                dftfe::uInt              closestAtomIdSum          = 0;
                dftfe::uInt              closestAtomId             = 0;
                dftfe::uInt              nonHangingNodeIdCountCell = 0;
                for (dftfe::uInt iFace = 0; iFace < faces_per_cell; ++iFace)
                  {
                    dftfe::Int dirichletDofCount         = 0;
                    bool       isSolvedDofPresent        = false;
                    dftfe::Int nonHangingNodeIdCountFace = 0;
                    std::vector<dealii::types::global_dof_index>
                      iFaceGlobalDofIndices(dofs_per_face);
                    cell->face(iFace)->get_dof_indices(iFaceGlobalDofIndices);
                    for (dftfe::uInt iFaceDof = 0; iFaceDof < dofs_per_face;
                         ++iFaceDof)
                      {
                        const dealii::types::global_dof_index nodeId =
                          iFaceGlobalDofIndices[iFaceDof];
                        if (!hangingPlusPBCConstraints.is_constrained(nodeId))
                          {
                            Assert(boundaryNodeMap.find(nodeId) !=
                                     boundaryNodeMap.end(),
                                   dealii::ExcMessage("BUG"));
                            Assert(closestAtomBinMap.find(nodeId) !=
                                     closestAtomBinMap.end(),
                                   dealii::ExcMessage("BUG"));

                            if (boundaryNodeMap.find(nodeId)->second != -1)
                              isSolvedDofPresent = true;
                            else
                              dirichletDofCount +=
                                boundaryNodeMap.find(nodeId)->second;

                            closestAtomId =
                              closestAtomBinMap.find(nodeId)->second;
                            closestAtomIdSum += closestAtomId;
                            nonHangingNodeIdCountCell++;
                            nonHangingNodeIdCountFace++;
                          } // non-hanging node check
                        else
                          {
                            const std::vector<
                              std::pair<dealii::types::global_dof_index,
                                        double>> *rowData =
                              hangingPlusPBCConstraints.get_constraint_entries(
                                nodeId);
                            for (dftfe::uInt j = 0; j < rowData->size(); ++j)
                              {
                                if (d_dftParams
                                      .createConstraintsFromSerialDofhandler)
                                  {
                                    if (boundaryNodeMap.find(
                                          (*rowData)[j].first) ==
                                        boundaryNodeMap.end())
                                      continue;
                                  }
                                else
                                  {
                                    Assert(boundaryNodeMap.find(
                                             (*rowData)[j].first) !=
                                             boundaryNodeMap.end(),
                                           dealii::ExcMessage("BUG"));
                                  }

                                if (boundaryNodeMap.find((*rowData)[j].first)
                                      ->second != -1)
                                  isSolvedDofPresent = true;
                                else
                                  dirichletDofCount +=
                                    boundaryNodeMap.find((*rowData)[j].first)
                                      ->second;
                              }
                          }

                      } // Face dof loop

                    if (isSolvedDofPresent)
                      {
                        faceIdsWithAtleastOneSolvedNonHangingNode.push_back(
                          iFace);
                      }
                    if (dirichletDofCount < 0)
                      {
                        dirichletFaceIds.push_back(iFace);
                      }
                    allFaceIdsOfCell.push_back(iFace);

                  } // Face loop

                // fill the target objects
                if (faceIdsWithAtleastOneSolvedNonHangingNode.size() > 0)
                  {
                    if (!(closestAtomIdSum ==
                          closestAtomId * nonHangingNodeIdCountCell))
                      {
                        std::cout << "closestAtomIdSum: " << closestAtomIdSum
                                  << ", closestAtomId: " << closestAtomId
                                  << ", nonHangingNodeIdCountCell: "
                                  << nonHangingNodeIdCountCell << std::endl;
                      }
                    AssertThrow(
                      closestAtomIdSum ==
                        closestAtomId * nonHangingNodeIdCountCell,
                      dealii::ExcMessage(
                        "cell dofs on vself ball surface have different closest atom ids, remedy- increase separation between vself balls"));

                    cellsVselfBallsDofHandler[iBin].push_back(cell);
                    cellsVselfBallsDofHandlerForce[iBin].push_back(cellForce);
                    cellsVselfBallsClosestAtomIdDofHandler[iBin][cell->id()] =
                      closestAtomId;
                    AtomIdBinIdLocalDofHandler[closestAtomId] = iBin;
                    if (dirichletFaceIds.size() > 0)
                      {
                        cellFacesVselfBallSurfacesDofHandler[iBin][cell] =
                          dirichletFaceIds;
                        cellFacesVselfBallSurfacesDofHandlerForce
                          [iBin][cellForce] = dirichletFaceIds;
                      }
                  }
              } // cell locally owned
          }     // cell loop
      }         // Bin loop

    d_cellIdToActiveCellIteratorMapDofHandlerRhoNodalElectro.clear();
    dealii::DoFHandler<3>::active_cell_iterator cell =
      dofHandlerRhoNodal.begin_active();
    dealii::DoFHandler<3>::active_cell_iterator endc = dofHandlerRhoNodal.end();
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        d_cellIdToActiveCellIteratorMapDofHandlerRhoNodalElectro[cell->id()] =
          cell;
  }

  void
  computeWavefuncEshelbyContributionLocal(
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
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
    const bool                                computeForce,
    const bool                                computeStress)
  {
    const dftfe::uInt   cellsBlockSize = cellRange.second - cellRange.first;
    const dftfe::uInt   wfcBlockSize   = vecRange.second - vecRange.first;
    std::vector<double> kcoord(3, 0);
    kcoord[0] = kcoordx;
    kcoord[1] = kcoordy;
    kcoord[2] = kcoordz;
    const double absksq =
      kcoord[0] * kcoord[0] + kcoord[1] * kcoord[1] + kcoord[2] * kcoord[2];
    for (dftfe::uInt iCell = 0; iCell < cellsBlockSize; iCell++)
      for (dftfe::uInt iQuad = 0; iQuad < nQuadsPerCell; iQuad++)
        for (dftfe::uInt iWfc = 0; iWfc < wfcBlockSize; iWfc++)
          {
            const dataTypes::number psiQuad =
              wfcQuadPointData[iCell * nQuadsPerCell * wfcBlockSize +
                               iQuad * wfcBlockSize + iWfc];
            const double partOcc    = partialOccupVec[iWfc];
            const double eigenValue = eigenValuesVec[iWfc];

            std::vector<dataTypes::number> gradPsiQuad(3);
            gradPsiQuad[0] =
              gradWfcQuadPointData[iCell * 3 * nQuadsPerCell * wfcBlockSize +
                                   iQuad * wfcBlockSize + iWfc];
            gradPsiQuad[1] =
              gradWfcQuadPointData[iCell * 3 * nQuadsPerCell * wfcBlockSize +
                                   nQuadsPerCell * wfcBlockSize +
                                   iQuad * wfcBlockSize + iWfc];

            gradPsiQuad[2] =
              gradWfcQuadPointData[iCell * 3 * nQuadsPerCell * wfcBlockSize +
                                   2 * nQuadsPerCell * wfcBlockSize +
                                   iQuad * wfcBlockSize + iWfc];

            const double identityFactor =
              0.5 * partOcc *
                dftfe::utils::realPart(
                  (dftfe::utils::complexConj(gradPsiQuad[0]) * gradPsiQuad[0] +
                   dftfe::utils::complexConj(gradPsiQuad[1]) * gradPsiQuad[1] +
                   dftfe::utils::complexConj(gradPsiQuad[2]) * gradPsiQuad[2] +
                   dataTypes::number(absksq - 2.0 * eigenValue) *
                     dftfe::utils::complexConj(psiQuad) * psiQuad)) +
              partOcc *
                dftfe::utils::imagPart(dftfe::utils::complexConj(psiQuad) *
                                       (kcoord[0] * gradPsiQuad[0] +
                                        kcoord[1] * gradPsiQuad[1] +
                                        kcoord[2] * gradPsiQuad[2]));
            for (dftfe::uInt iDim = 0; iDim < 3; iDim++)
              for (dftfe::uInt jDim = 0; jDim < 3; jDim++)
                {
                  eshelbyContributions[iCell * nQuadsPerCell * 9 *
                                         wfcBlockSize +
                                       iQuad * 9 * wfcBlockSize +
                                       iDim * 3 * wfcBlockSize +
                                       jDim * wfcBlockSize + iWfc] =
                    -0.5 * partOcc *
                      dftfe::utils::realPart(
                        dftfe::utils::complexConj(gradPsiQuad[iDim]) *
                          gradPsiQuad[jDim] +
                        gradPsiQuad[iDim] *
                          dftfe::utils::complexConj(gradPsiQuad[jDim])) -
                    partOcc * dftfe::utils::imagPart(
                                dftfe::utils::complexConj(psiQuad) *
                                (gradPsiQuad[iDim] * kcoord[jDim]));

                  if (iDim == jDim)
                    eshelbyContributions[iCell * nQuadsPerCell * 9 *
                                           wfcBlockSize +
                                         iQuad * 9 * wfcBlockSize +
                                         iDim * 3 * wfcBlockSize +
                                         jDim * wfcBlockSize + iWfc] +=
                      identityFactor;
                }
#ifdef USE_COMPLEX
            if (computeStress)
              {
                for (dftfe::uInt iDim = 0; iDim < 3; iDim++)
                  for (dftfe::uInt jDim = 0; jDim < 3; jDim++)
                    {
                      eshelbyContributions[iCell * nQuadsPerCell * 9 *
                                             wfcBlockSize +
                                           iQuad * 9 * wfcBlockSize +
                                           iDim * 3 * wfcBlockSize +
                                           jDim * wfcBlockSize + iWfc] +=
                        -partOcc * dftfe::utils::imagPart(
                                     dftfe::utils::complexConj(psiQuad) *
                                     (kcoord[iDim] * gradPsiQuad[jDim])) -
                        partOcc *
                          dftfe::utils::realPart(
                            kcoord[iDim] * kcoord[jDim] *
                            dftfe::utils::complexConj(psiQuad) * psiQuad);
                    }
              }
#endif
          }
    const double scalarCoeffAlphaEshelby = 1.0;
    const double scalarCoeffBetaEshelby  = 0.0;
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      onesVec(wfcBlockSize, 1.0);

    BLASWrapperPtr->xgemm('N',
                          'N',
                          1,
                          cellsBlockSize * nQuadsPerCell * 9,
                          wfcBlockSize,
                          &scalarCoeffAlphaEshelby,
                          onesVec.data(),
                          1,
                          eshelbyContributions,
                          wfcBlockSize,
                          &scalarCoeffBetaEshelby,
                          eshelbyTensor,
                          1);
  }

  template class configurationalForceClass<dftfe::utils::MemorySpace::HOST>;
#if defined(DFTFE_WITH_DEVICE)
  template class configurationalForceClass<dftfe::utils::MemorySpace::DEVICE>;
#endif
} // namespace dftfe
