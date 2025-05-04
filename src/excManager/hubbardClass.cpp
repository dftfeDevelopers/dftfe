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
// @author Vishal Subramanian
//

#include "hubbardClass.h"
#include "AtomCenteredSphericalFunctionProjectorSpline.h"
#include "dftParameters.h"
#include "DataTypeOverloads.h"
#include "constants.h"
#include "BLASWrapper.h"
#include "AtomCenteredPseudoWavefunctionSpline.h"
#include "AuxDensityMatrixFE.h"

#include "CompositeData.h"
#include "MPIWriteOnFile.h"
#include "NodalData.h"


#if defined(DFTFE_WITH_DEVICE)
#  include "deviceKernelsGeneric.h"
#endif

namespace dftfe
{
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  hubbard<ValueType, memorySpace>::hubbard(
    const MPI_Comm &mpi_comm_parent,
    const MPI_Comm &mpi_comm_domain,
    const MPI_Comm &mpi_comm_interPool,
    const MPI_Comm &mpi_comm_interBandGroup)
    : d_mpi_comm_parent(mpi_comm_parent)
    , d_mpi_comm_domain(mpi_comm_domain)
    , d_mpi_comm_interPool(mpi_comm_interPool)
    , d_mpi_comm_interBand(mpi_comm_interBandGroup)
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(d_mpi_comm_parent) == 0))
  {
    d_hubbardEnergy                 = 0.0;
    d_expectationOfHubbardPotential = 0.0;
    d_maxOccMatSizePerAtom          = 0;
    d_useSinglePrec                 = false;
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  hubbard<ValueType,
          memorySpace>::createAtomCenteredSphericalFunctionsForProjectors()
  {
    for (auto const &[key, val] : d_hubbardSpeciesData)
      {
        dftfe::uInt Znum = val.atomicNumber;

        dftfe::uInt numberOfProjectors = val.numProj;

        dftfe::uInt numProj;
        dftfe::uInt alpha = 0;
        for (dftfe::uInt i = 0; i < numberOfProjectors; i++)
          {
            char        projRadialFunctionFileName[512];
            dftfe::uInt nQuantumNo = val.nQuantumNum[i];
            dftfe::uInt lQuantumNo = val.lQuantumNum[i];

            char waveFunctionFileName[256];
            strcpy(waveFunctionFileName,
                   (d_dftfeScratchFolderName + "/z" + std::to_string(Znum) +
                    "/psi" + std::to_string(nQuantumNo) +
                    std::to_string(lQuantumNo) + ".inp")
                     .c_str());

            d_atomicProjectorFnsMap[std::make_pair(Znum, alpha)] =
              std::make_shared<AtomCenteredPseudoWavefunctionSpline>(
                waveFunctionFileName,
                lQuantumNo,
                10.0, // NOTE: the cut off is manually set to 10.0 to emulate
                      // QE's behaviour. Remove this if better accuracy is
                      // required
                1E-12);
            alpha++;
          } // i loop

      } // for loop *it
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  hubbard<ValueType, memorySpace>::init(
    std::shared_ptr<
      dftfe::basis::FEBasisOperations<ValueType, double, memorySpace>>
      basisOperationsMemPtr,
    std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<ValueType, double, dftfe::utils::MemorySpace::HOST>>
      basisOperationsHostPtr,
    std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
      BLASWrapperMemPtr,
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
                                            BLASWrapperHostPtr,
    const dftfe::uInt                       matrixFreeVectorComponent,
    const dftfe::uInt                       densityQuadratureId,
    const dftfe::uInt                       sparsityPatternQuadratureId,
    const dftfe::uInt                       numberWaveFunctions,
    const dftfe::uInt                       numSpins,
    const dftParameters                    &dftParam,
    const std::string                      &scratchFolderName,
    const bool                              singlePrecNonLocalOperator,
    const bool                              updateNonlocalSparsity,
    const std::vector<std::vector<double>> &atomLocations,
    const std::vector<std::vector<double>> &atomLocationsFrac,
    const std::vector<dftfe::Int>          &imageIds,
    const std::vector<std::vector<double>> &imagePositions,
    const std::vector<double>              &kPointCoordinates,
    const std::vector<double>              &kPointWeights,
    const std::vector<std::vector<double>> &domainBoundaries)
  {
    MPI_Barrier(d_mpi_comm_parent);
    d_BasisOperatorMemPtr  = basisOperationsMemPtr;
    d_BLASWrapperMemPtr    = BLASWrapperMemPtr;
    d_BasisOperatorHostPtr = basisOperationsHostPtr;

    d_BLASWrapperHostPtr     = BLASWrapperHostPtr;
    d_densityQuadratureId    = densityQuadratureId;
    d_dftfeScratchFolderName = scratchFolderName;
    d_kPointWeights          = kPointWeights;

    d_numberWaveFunctions = numberWaveFunctions;
    d_dftParamsPtr        = &dftParam;

    d_verbosity = d_dftParamsPtr->verbosity;

    d_kPointCoordinates = kPointCoordinates;
    d_numKPoints        = kPointCoordinates.size() / 3;
    d_domainBoundaries  = domainBoundaries;

    d_useSinglePrec = d_dftParamsPtr->useSinglePrecCheby;


    d_cellsBlockSizeApply = memorySpace == dftfe::utils::MemorySpace::HOST ?
                              1 :
                              d_BasisOperatorMemPtr->nCells();
    d_numSpins            = numSpins;

    // Read the hubbard input data.
    readHubbardInput(atomLocations, imageIds, imagePositions);

    createAtomCenteredSphericalFunctionsForProjectors();

    d_atomicProjectorFnsContainer =
      std::make_shared<AtomCenteredSphericalFunctionContainer>();

    d_atomicProjectorFnsContainer->init(d_mapAtomToAtomicNumber,
                                        d_atomicProjectorFnsMap);

    // set up the non local operator.
    d_nonLocalOperator =
      std::make_shared<AtomicCenteredNonLocalOperator<ValueType, memorySpace>>(
        d_BLASWrapperMemPtr,
        d_BasisOperatorMemPtr,
        d_atomicProjectorFnsContainer,
        d_mpi_comm_domain,
        true,
        true,
        true);

    if (d_useSinglePrec)
      {
        d_nonLocalOperatorSinglePrec =
          std::make_shared<AtomicCenteredNonLocalOperator<
            typename dftfe::dataTypes::singlePrecType<ValueType>::type,
            memorySpace>>(d_BLASWrapperMemPtr,
                          d_BasisOperatorMemPtr,
                          d_atomicProjectorFnsContainer,
                          d_mpi_comm_domain,
                          true,
                          true,
                          true);
      }


    d_atomicProjectorFnsContainer->initaliseCoordinates(d_atomicCoords,
                                                        d_periodicImagesCoords,
                                                        d_imageIds);

    d_atomicProjectorFnsContainer->computeSparseStructure(
      d_BasisOperatorHostPtr, sparsityPatternQuadratureId, 1E-8, 0);


    d_nonLocalOperator->intitialisePartitionerKPointsAndComputeCMatrixEntries(
      updateNonlocalSparsity,
      kPointWeights,
      kPointCoordinates,
      d_BasisOperatorHostPtr,
      d_BLASWrapperHostPtr,
      densityQuadratureId);

    if (d_useSinglePrec)
      {
        d_nonLocalOperatorSinglePrec
          ->copyPartitionerKPointsAndComputeCMatrixEntries(
            updateNonlocalSparsity,
            kPointWeights,
            kPointCoordinates,
            d_BasisOperatorHostPtr,
            d_BLASWrapperHostPtr,
            densityQuadratureId,
            d_nonLocalOperator);
      }

    MPI_Barrier(d_mpi_comm_domain);
    double endRead = MPI_Wtime();

    dftUtils::createBandParallelizationIndices(
      d_mpi_comm_interBand,
      d_numberWaveFunctions,
      d_bandGroupLowHighPlusOneIndices);


    d_spinPolarizedFactor = (d_dftParamsPtr->spinPolarized == 1) ? 1.0 : 2.0;

    d_noOfSpin = (d_dftParamsPtr->spinPolarized == 1) ? 2 : 1;

    dftfe::uInt numLocalAtomsInProc =
      d_nonLocalOperator->getTotalAtomInCurrentProcessor();

    const std::vector<dftfe::uInt> atomIdsInProc =
      d_atomicProjectorFnsContainer->getAtomIdsInCurrentProcess();
    std::vector<dftfe::uInt> atomicNumber =
      d_atomicProjectorFnsContainer->getAtomicNumbers();

    d_numTotalOccMatrixEntriesPerSpin = 0;
    d_OccMatrixEntryStartForAtom.resize(0);

    for (dftfe::Int iAtom = 0; iAtom < numLocalAtomsInProc; iAtom++)
      {
        const dftfe::uInt atomId     = atomIdsInProc[iAtom];
        const dftfe::uInt Znum       = atomicNumber[atomId];
        const dftfe::uInt hubbardIds = d_mapAtomToHubbardIds[atomId];

        d_OccMatrixEntryStartForAtom.push_back(
          d_numTotalOccMatrixEntriesPerSpin);
        d_numTotalOccMatrixEntriesPerSpin +=
          d_hubbardSpeciesData[hubbardIds].numberSphericalFuncSq;
      }
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST> occIn,
      occOut, occResidual;
    occIn.resize(d_numSpins * d_numTotalOccMatrixEntriesPerSpin);
    std::fill(occIn.begin(), occIn.end(), 0.0);
    d_occupationMatrix[HubbardOccFieldType::In] = occIn;

    occOut.resize(d_numSpins * d_numTotalOccMatrixEntriesPerSpin);
    std::fill(occOut.begin(), occOut.end(), 0.0);
    d_occupationMatrix[HubbardOccFieldType::Out] = occOut;

    occResidual.resize(d_numSpins * d_numTotalOccMatrixEntriesPerSpin);
    std::fill(occResidual.begin(), occResidual.end(), 0.0);
    d_occupationMatrix[HubbardOccFieldType::Residual] = occResidual;

    d_hubbOccMatAfterMixing.resize(d_numSpins *
                                   d_numTotalOccMatrixEntriesPerSpin);
    std::fill(d_hubbOccMatAfterMixing.begin(),
              d_hubbOccMatAfterMixing.end(),
              0.0);

    setInitialOccMatrix();

    // TODO commented for now. Uncomment if necessary
    //    computeSymmetricTransforms(atomLocationsFrac,domainBoundaries);



    // This is to create a locally owned atoms.
    // This is not very efficient and better methods may exist.

    std::vector<dftfe::uInt> atomProcessorMap;
    dftfe::uInt              numAtoms = atomLocations.size();
    atomProcessorMap.resize(numAtoms);

    dftfe::Int thisRank =
      dealii::Utilities::MPI::this_mpi_process(d_mpi_comm_domain);
    const dftfe::uInt nRanks =
      dealii::Utilities::MPI::n_mpi_processes(d_mpi_comm_domain);

    for (dftfe::uInt iAtom = 0; iAtom < numAtoms; iAtom++)
      {
        atomProcessorMap[iAtom] = nRanks;
      }

    for (dftfe::Int iAtom = 0; iAtom < numLocalAtomsInProc; iAtom++)
      {
        const dftfe::uInt atomId = atomIdsInProc[iAtom];
        atomProcessorMap[atomId] = thisRank;
      }
    MPI_Allreduce(MPI_IN_PLACE,
                  &atomProcessorMap[0],
                  numAtoms,
                  dftfe::dataTypes::mpi_type_id(atomProcessorMap.data()),
                  MPI_MIN,
                  d_mpi_comm_domain);

    d_procLocalAtomId.resize(0);

    for (dftfe::Int iAtom = 0; iAtom < numLocalAtomsInProc; iAtom++)
      {
        const dftfe::uInt atomId = atomIdsInProc[iAtom];
        if (thisRank == atomProcessorMap[atomId])
          {
            d_procLocalAtomId.push_back(iAtom);
          }
      }
  }

  /*
   * computes the initial occupation matrix.
   * The general rule is that iAtom is iterator for atoms whose atomic
   * projectors has a compact support in the locally owned cells.
   */
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  hubbard<ValueType, memorySpace>::setInitialOccMatrix()
  {
    dftfe::uInt numLocalAtomsInProc =
      d_nonLocalOperator->getTotalAtomInCurrentProcessor();
    const std::vector<dftfe::uInt> atomIdsInProc =
      d_atomicProjectorFnsContainer->getAtomIdsInCurrentProcess();
    std::vector<dftfe::uInt> atomicNumber =
      d_atomicProjectorFnsContainer->getAtomicNumbers();

    for (dftfe::uInt iSpin = 0; iSpin < d_numSpins; iSpin++)
      {
        for (dftfe::Int iAtom = 0; iAtom < numLocalAtomsInProc; iAtom++)
          {
            const dftfe::uInt atomId       = atomIdsInProc[iAtom];
            const dftfe::uInt Znum         = atomicNumber[atomId];
            const dftfe::uInt hubbardIds   = d_mapAtomToHubbardIds[atomId];
            double            initOccValue = 0.0;
            if (d_numSpins == 1)
              {
                initOccValue =
                  d_hubbardSpeciesData[hubbardIds].initialOccupation /
                  (2.0 * d_hubbardSpeciesData[hubbardIds].numberSphericalFunc);
              }
            else if (d_numSpins == 2)
              {
                initOccValue =
                  d_hubbardSpeciesData[hubbardIds].initialOccupation /
                  (2.0 * d_hubbardSpeciesData[hubbardIds].numberSphericalFunc);

                dftfe::uInt majorSpin = 1000, minorSpin = 1000;
                if (d_initialAtomicSpin[atomId] > 1e-3)
                  {
                    majorSpin = 0;
                    minorSpin = 1;
                  }
                else if (d_initialAtomicSpin[atomId] < -1e-3)
                  {
                    majorSpin = 1;
                    minorSpin = 0;
                  }

                if (iSpin == majorSpin)
                  {
                    if (d_hubbardSpeciesData[hubbardIds].numberSphericalFunc <
                        d_hubbardSpeciesData[hubbardIds].initialOccupation)
                      {
                        initOccValue = 1.0;
                      }
                    else
                      {
                        initOccValue =
                          d_hubbardSpeciesData[hubbardIds].initialOccupation /
                          (d_hubbardSpeciesData[hubbardIds]
                             .numberSphericalFunc);
                      }
                  }
                else if (iSpin == minorSpin)
                  {
                    if (d_hubbardSpeciesData[hubbardIds].numberSphericalFunc <
                        d_hubbardSpeciesData[hubbardIds].initialOccupation)
                      {
                        initOccValue =
                          (d_hubbardSpeciesData[hubbardIds].initialOccupation -
                           d_hubbardSpeciesData[hubbardIds]
                             .numberSphericalFunc) /
                          (d_hubbardSpeciesData[hubbardIds]
                             .numberSphericalFunc);
                      }
                    else
                      {
                        initOccValue = 0.0;
                      }
                  }
              }
            for (dftfe::uInt iOrb = 0;
                 iOrb < d_hubbardSpeciesData[hubbardIds].numberSphericalFunc;
                 iOrb++)
              {
                d_occupationMatrix[HubbardOccFieldType::In]
                                  [iSpin * d_numTotalOccMatrixEntriesPerSpin +
                                   d_OccMatrixEntryStartForAtom[iAtom] +
                                   iOrb * d_hubbardSpeciesData[hubbardIds]
                                            .numberSphericalFunc +
                                   iOrb] = initOccValue;
              }
          }
      }
    computeCouplingMatrix();
  }


  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  const std::shared_ptr<AtomicCenteredNonLocalOperator<ValueType, memorySpace>>
  hubbard<ValueType, memorySpace>::getNonLocalOperator()
  {
    return d_nonLocalOperator;
  }

  /*
   * computes the initial occupation matrix.
   * Here iAtom is iterator for locally owned atoms
   * and a MPI_Allreduce over the mpi domain.
   */
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  hubbard<ValueType, memorySpace>::computeEnergyFromOccupationMatrix()
  {
    d_hubbardEnergy                 = 0.0;
    d_expectationOfHubbardPotential = 0.0;

    d_spinPolarizedFactor = (d_dftParamsPtr->spinPolarized == 1) ? 1.0 : 2.0;
    dftfe::uInt numOwnedAtomsInProc = d_procLocalAtomId.size();
    const std::vector<dftfe::uInt> atomIdsInProc =
      d_atomicProjectorFnsContainer->getAtomIdsInCurrentProcess();
    std::vector<dftfe::uInt> atomicNumber =
      d_atomicProjectorFnsContainer->getAtomicNumbers();
    for (dftfe::uInt iAtom = 0; iAtom < numOwnedAtomsInProc; iAtom++)
      {
        const dftfe::uInt atomId     = atomIdsInProc[d_procLocalAtomId[iAtom]];
        const dftfe::uInt Znum       = atomicNumber[atomId];
        const dftfe::uInt hubbardIds = d_mapAtomToHubbardIds[atomId];

        const dftfe::uInt numSphericalFunc =
          d_hubbardSpeciesData[hubbardIds].numberSphericalFunc;

        for (dftfe::uInt spinIndex = 0; spinIndex < d_numSpins; spinIndex++)
          {
            for (dftfe::uInt iOrb = 0; iOrb < numSphericalFunc; iOrb++)
              {
                d_hubbardEnergy +=
                  0.5 * d_spinPolarizedFactor *
                  d_hubbardSpeciesData[hubbardIds].hubbardValue *
                  dftfe::utils::realPart(
                    d_occupationMatrix
                      [HubbardOccFieldType::Out]
                      [spinIndex * d_numTotalOccMatrixEntriesPerSpin +
                       d_OccMatrixEntryStartForAtom[d_procLocalAtomId[iAtom]] +
                       iOrb * numSphericalFunc + iOrb]);


                double occMatrixSq = 0.0;

                for (dftfe::uInt jOrb = 0; jOrb < numSphericalFunc; jOrb++)
                  {
                    dftfe::uInt index1 = iOrb * numSphericalFunc + jOrb;
                    dftfe::uInt index2 = jOrb * numSphericalFunc + iOrb;

                    occMatrixSq += dftfe::utils::realPart(
                      d_occupationMatrix[HubbardOccFieldType::Out]
                                        [spinIndex *
                                           d_numTotalOccMatrixEntriesPerSpin +
                                         d_OccMatrixEntryStartForAtom
                                           [d_procLocalAtomId[iAtom]] +
                                         index1] *
                      d_occupationMatrix[HubbardOccFieldType::Out]
                                        [spinIndex *
                                           d_numTotalOccMatrixEntriesPerSpin +
                                         d_OccMatrixEntryStartForAtom
                                           [d_procLocalAtomId[iAtom]] +
                                         index2]);
                  }
                d_hubbardEnergy -=
                  0.5 * d_spinPolarizedFactor *
                  d_hubbardSpeciesData[hubbardIds].hubbardValue *
                  dftfe::utils::realPart(occMatrixSq);
                d_expectationOfHubbardPotential -=
                  0.5 * d_spinPolarizedFactor *
                  d_hubbardSpeciesData[hubbardIds].hubbardValue *
                  dftfe::utils::realPart(occMatrixSq);
              }
          }
      }

    MPI_Allreduce(MPI_IN_PLACE,
                  &d_hubbardEnergy,
                  1,
                  MPI_DOUBLE,
                  MPI_SUM,
                  d_mpi_comm_domain);

    MPI_Allreduce(MPI_IN_PLACE,
                  &d_expectationOfHubbardPotential,
                  1,
                  MPI_DOUBLE,
                  MPI_SUM,
                  d_mpi_comm_domain);

    d_expectationOfHubbardPotential += d_hubbardEnergy;

    if ((d_verbosity >= 2) || (d_dftParamsPtr->computeEnergyEverySCF))
      {
        pcout << " Hubbard energy = " << d_hubbardEnergy << "\n";
        pcout << " Hubbard energy correction = "
              << d_expectationOfHubbardPotential << "\n";
      }
  }

  // Currently this function is not compatible with band parallelisation
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  hubbard<ValueType, memorySpace>::computeOccupationMatrix(
    const dftfe::utils::MemoryStorage<ValueType, memorySpace> *X,
    const std::vector<std::vector<double>>                    &orbitalOccupancy)
  {
    dftfe::uInt numLocalAtomsInProc =
      d_nonLocalOperator->getTotalAtomInCurrentProcessor();

    const std::vector<dftfe::uInt> atomIdsInProc =
      d_atomicProjectorFnsContainer->getAtomIdsInCurrentProcess();
    std::vector<dftfe::uInt> atomicNumber =
      d_atomicProjectorFnsContainer->getAtomicNumbers();

    std::fill(d_occupationMatrix[HubbardOccFieldType::Out].begin(),
              d_occupationMatrix[HubbardOccFieldType::Out].end(),
              0.0);



    const ValueType   zero = 0;
    const dftfe::uInt cellsBlockSize =
      memorySpace == dftfe::utils::MemorySpace::DEVICE ?
        d_BasisOperatorMemPtr->nCells() :
        1;
    const dftfe::uInt totalLocallyOwnedCells = d_BasisOperatorMemPtr->nCells();
    const dftfe::uInt numCellBlocks = totalLocallyOwnedCells / cellsBlockSize;
    const dftfe::uInt remCellBlockSize =
      totalLocallyOwnedCells - numCellBlocks * cellsBlockSize;

    const dftfe::uInt BVec = d_dftParamsPtr->chebyWfcBlockSize;

    d_BasisOperatorMemPtr->reinit(BVec, cellsBlockSize, d_densityQuadratureId);
    const dftfe::uInt numQuadPoints = d_BasisOperatorMemPtr->nQuadsPerCell();

    dftfe::linearAlgebra::MultiVector<ValueType, memorySpace>
      projectorKetTimesVector;

    dftfe::linearAlgebra::MultiVector<ValueType, memorySpace>
      *flattenedArrayBlock;

    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      partialOccupVecHost(BVec, 0.0);
#if defined(DFTFE_WITH_DEVICE)
    dftfe::utils::MemoryStorage<double, memorySpace> partialOccupVec(
      partialOccupVecHost.size());
#else
    auto &partialOccupVec = partialOccupVecHost;
#endif

    const dftfe::uInt bandGroupTaskId =
      dealii::Utilities::MPI::this_mpi_process(d_mpi_comm_interBand);

    dftfe::uInt numLocalDofs       = d_BasisOperatorHostPtr->nOwnedDofs();
    dftfe::uInt numNodesPerElement = d_BasisOperatorHostPtr->nDofsPerCell();
    dftfe::uInt previousSize       = 0;
    for (dftfe::uInt kPoint = 0; kPoint < d_kPointWeights.size(); ++kPoint)
      {
        for (dftfe::uInt spinIndex = 0; spinIndex < d_noOfSpin; ++spinIndex)
          {
            d_nonLocalOperator->initialiseOperatorActionOnX(kPoint);
            for (dftfe::uInt jvec = 0; jvec < d_numberWaveFunctions;
                 jvec += BVec)
              {
                const dftfe::uInt currentBlockSize =
                  std::min(BVec, d_numberWaveFunctions - jvec);
                flattenedArrayBlock =
                  &(d_BasisOperatorMemPtr->getMultiVector(currentBlockSize, 0));
                d_nonLocalOperator->initialiseFlattenedDataStructure(
                  currentBlockSize, projectorKetTimesVector);


                previousSize = cellsBlockSize * currentBlockSize;
                if (((jvec + currentBlockSize) <=
                     d_bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId +
                                                      1]) &&
                    ((jvec + currentBlockSize) >
                     d_bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId]))
                  {
                    for (dftfe::uInt iEigenVec = 0;
                         iEigenVec < currentBlockSize;
                         ++iEigenVec)
                      {
                        partialOccupVecHost.data()[iEigenVec] = std::sqrt(
                          orbitalOccupancy[kPoint]
                                          [d_numberWaveFunctions * spinIndex +
                                           jvec + iEigenVec] *
                          d_kPointWeights[kPoint]);
                      }

                    if (memorySpace == dftfe::utils::MemorySpace::HOST)
                      for (dftfe::uInt iNode = 0; iNode < numLocalDofs; ++iNode)
                        std::memcpy(flattenedArrayBlock->data() +
                                      iNode * currentBlockSize,
                                    X->data() +
                                      numLocalDofs * d_numberWaveFunctions *
                                        (d_noOfSpin * kPoint + spinIndex) +
                                      iNode * d_numberWaveFunctions + jvec,
                                    currentBlockSize * sizeof(ValueType));
#if defined(DFTFE_WITH_DEVICE)
                    else if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
                      d_BLASWrapperMemPtr->stridedCopyToBlockConstantStride(
                        currentBlockSize,
                        d_numberWaveFunctions,
                        numLocalDofs,
                        jvec,
                        X->data() + numLocalDofs * d_numberWaveFunctions *
                                      (d_noOfSpin * kPoint + spinIndex),
                        flattenedArrayBlock->data());
#endif
                    d_BasisOperatorMemPtr->reinit(currentBlockSize,
                                                  cellsBlockSize,
                                                  d_densityQuadratureId,
                                                  false);

                    flattenedArrayBlock->updateGhostValues();
                    d_BasisOperatorMemPtr->distribute(*(flattenedArrayBlock));


                    d_nonLocalOperator->applyCconjtransOnX(
                      *(flattenedArrayBlock));
                  }
                projectorKetTimesVector.setValue(0.0);
                d_nonLocalOperator->applyAllReduceOnCconjtransX(
                  projectorKetTimesVector);
                partialOccupVec.copyFrom(partialOccupVecHost);
                d_nonLocalOperator
                  ->copyBackFromDistributedVectorToLocalDataStructure(
                    projectorKetTimesVector, partialOccupVec);
                computeHubbardOccNumberFromCTransOnX(true,
                                                     currentBlockSize,
                                                     spinIndex,
                                                     kPoint);
              }
          }
      }


    if (dealii::Utilities::MPI::n_mpi_processes(d_mpi_comm_interBand) > 1)
      {
        MPI_Allreduce(MPI_IN_PLACE,
                      d_occupationMatrix[HubbardOccFieldType::Out].data(),
                      d_numSpins * d_numTotalOccMatrixEntriesPerSpin,
                      MPI_DOUBLE,
                      MPI_SUM,
                      d_mpi_comm_interBand);
      }
    if (dealii::Utilities::MPI::n_mpi_processes(d_mpi_comm_interPool) > 1)
      {
        MPI_Allreduce(MPI_IN_PLACE,
                      d_occupationMatrix[HubbardOccFieldType::Out].data(),
                      d_numSpins * d_numTotalOccMatrixEntriesPerSpin,
                      MPI_DOUBLE,
                      MPI_SUM,
                      d_mpi_comm_interPool);
      }

    dftfe::uInt numOwnedAtomsInProc = d_procLocalAtomId.size();
    for (dftfe::uInt iAtom = 0; iAtom < numOwnedAtomsInProc; iAtom++)
      {
        const dftfe::uInt atomId     = atomIdsInProc[d_procLocalAtomId[iAtom]];
        const dftfe::uInt Znum       = atomicNumber[atomId];
        const dftfe::uInt hubbardIds = d_mapAtomToHubbardIds[atomId];

        const dftfe::uInt numSphericalFunc =
          d_hubbardSpeciesData[hubbardIds].numberSphericalFunc;

        if (d_verbosity >= 3)
          {
            for (dftfe::uInt spinIndex = 0; spinIndex < d_numSpins; spinIndex++)
              {
                for (dftfe::uInt iOrb = 0; iOrb < numSphericalFunc; iOrb++)
                  {
                    for (dftfe::uInt jOrb = 0; jOrb < numSphericalFunc; jOrb++)
                      {
                        std::cout
                          << " "
                          << d_occupationMatrix
                               [HubbardOccFieldType::Out]
                               [spinIndex * d_numTotalOccMatrixEntriesPerSpin +
                                d_OccMatrixEntryStartForAtom
                                  [d_procLocalAtomId[iAtom]] +
                                iOrb * numSphericalFunc + jOrb];
                      }
                    std::cout << "\n";
                  }
              }
          }
      }

    computeResidualOccMat();
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  hubbard<ValueType, memorySpace>::computeHubbardOccNumberFromCTransOnX(
    const bool        isOccOut,
    const dftfe::uInt vectorBlockSize,
    const dftfe::uInt spinIndex,
    const dftfe::uInt kpointIndex)
  {
    const std::vector<dftfe::uInt> atomIdsInProcessor =
      d_atomicProjectorFnsContainer->getAtomIdsInCurrentProcess();
    std::vector<dftfe::uInt> atomicNumber =
      d_atomicProjectorFnsContainer->getAtomicNumbers();
    char transB = 'N';
#ifdef USE_COMPLEX
    char transA = 'C';
#else
    char  transA          = 'T';
#endif
    const ValueType beta  = 0.0;
    const ValueType alpha = 1.0;
    for (dftfe::Int iAtom = 0; iAtom < atomIdsInProcessor.size(); iAtom++)
      {
        const dftfe::uInt atomId     = atomIdsInProcessor[iAtom];
        const dftfe::uInt Znum       = atomicNumber[atomId];
        const dftfe::uInt hubbardIds = d_mapAtomToHubbardIds[atomId];
        const dftfe::uInt numberSphericalFunctionsSq =
          d_hubbardSpeciesData[hubbardIds].numberSphericalFuncSq;

        const dftfe::uInt numberSphericalFunctions =
          d_hubbardSpeciesData[hubbardIds].numberSphericalFunc;

        const dftfe::uInt numberSphericalFunc =
          d_hubbardSpeciesData[hubbardIds].numberSphericalFunc;
        std::vector<ValueType> tempOccMat(numberSphericalFunctionsSq, 0.0);

        auto valuesOfCconjTimesX =
          d_nonLocalOperator->getCconjtansXLocalDataStructure(iAtom);
        d_BLASWrapperHostPtr->xgemm(transA,
                                    transB,
                                    numberSphericalFunc,
                                    numberSphericalFunc,
                                    vectorBlockSize,
                                    &alpha,
                                    valuesOfCconjTimesX,
                                    vectorBlockSize,
                                    valuesOfCconjTimesX,
                                    vectorBlockSize,
                                    &beta,
                                    &tempOccMat[0],
                                    numberSphericalFunc);

        std::transform(d_occupationMatrix[HubbardOccFieldType::Out].data() +
                         spinIndex * d_numTotalOccMatrixEntriesPerSpin +
                         d_OccMatrixEntryStartForAtom[iAtom],
                       d_occupationMatrix[HubbardOccFieldType::Out].data() +
                         spinIndex * d_numTotalOccMatrixEntriesPerSpin +
                         d_OccMatrixEntryStartForAtom[iAtom] +
                         numberSphericalFunctions * numberSphericalFunctions,
                       tempOccMat.data(),
                       d_occupationMatrix[HubbardOccFieldType::Out].data() +
                         spinIndex * d_numTotalOccMatrixEntriesPerSpin +
                         d_OccMatrixEntryStartForAtom[iAtom],
                       [](auto &p, auto &q) {
                         return p + dftfe::utils::realPart(q);
                       });
      }
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  hubbard<ValueType, memorySpace>::computeResidualOccMat()
  {
    for (dftfe::uInt iElem = 0;
         iElem < d_numSpins * d_numTotalOccMatrixEntriesPerSpin;
         iElem++)
      {
        d_occupationMatrix[HubbardOccFieldType::Residual][iElem] =
          d_occupationMatrix[HubbardOccFieldType::Out][iElem] -
          d_occupationMatrix[HubbardOccFieldType::In][iElem];
      }
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST> &
  hubbard<ValueType, memorySpace>::getHubbMatrixForMixing()
  {
    return d_hubbOccMatAfterMixing;
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST> &
  hubbard<ValueType, memorySpace>::getOccMatIn()
  {
    return d_occupationMatrix[HubbardOccFieldType::In];
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST> &
  hubbard<ValueType, memorySpace>::getOccMatRes()
  {
    return d_occupationMatrix[HubbardOccFieldType::Residual];
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST> &
  hubbard<ValueType, memorySpace>::getOccMatOut()
  {
    return d_occupationMatrix[HubbardOccFieldType::Out];
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  hubbard<ValueType, memorySpace>::setInOccMatrix(
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &inputOccMatrix)
  {
    for (dftfe::uInt iElem = 0;
         iElem < d_numSpins * d_numTotalOccMatrixEntriesPerSpin;
         iElem++)
      {
        d_occupationMatrix[HubbardOccFieldType::In][iElem] =
          inputOccMatrix[iElem];
      }

    computeCouplingMatrix();
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  hubbard<ValueType, memorySpace>::writeHubbOccToFile()
  {
    dftfe::uInt bandGroupTaskId =
      dealii::Utilities::MPI::this_mpi_process(d_mpi_comm_interBand);

    dftfe::uInt interPoolId =
      dealii::Utilities::MPI::this_mpi_process(d_mpi_comm_interPool);

    const std::vector<dftfe::uInt> atomIdsInProc =
      d_atomicProjectorFnsContainer->getAtomIdsInCurrentProcess();

    std::vector<dftfe::uInt> atomicNumber =
      d_atomicProjectorFnsContainer->getAtomicNumbers();

    if ((bandGroupTaskId == 0) && (interPoolId == 0))
      {
        std::vector<std::shared_ptr<dftfe::dftUtils::CompositeData>> data(0);

        dftfe::uInt numOwnedAtomsInProc = d_procLocalAtomId.size();
        for (dftfe::uInt iAtom = 0; iAtom < numOwnedAtomsInProc; iAtom++)
          {
            const dftfe::uInt atomId = atomIdsInProc[d_procLocalAtomId[iAtom]];
            const dftfe::uInt Znum   = atomicNumber[atomId];
            const dftfe::uInt hubbardIds = d_mapAtomToHubbardIds[atomId];

            const dftfe::uInt numSphericalFunc =
              d_hubbardSpeciesData[hubbardIds].numberSphericalFunc;

            std::vector<double> nodeVals(0);

            nodeVals.push_back((double)getGlobalAtomId(atomId));
            for (dftfe::uInt spinIndex = 0; spinIndex < d_numSpins; spinIndex++)
              {
                for (dftfe::uInt iOrb = 0;
                     iOrb < numSphericalFunc * numSphericalFunc;
                     iOrb++)
                  {
                    double occVal = d_occupationMatrix
                      [HubbardOccFieldType::In]
                      [spinIndex * d_numTotalOccMatrixEntriesPerSpin +
                       d_OccMatrixEntryStartForAtom[d_procLocalAtomId[iAtom]] +
                       iOrb];
                    nodeVals.push_back(occVal);
                  }
              }

            for (dftfe::uInt iOrb =
                   numSphericalFunc * numSphericalFunc * d_numSpins;
                 iOrb < d_maxOccMatSizePerAtom * d_numSpins;
                 iOrb++)
              {
                nodeVals.push_back(0.0);
              }

            data.push_back(
              std::make_shared<dftfe::dftUtils::NodalData>(nodeVals));
          }

        std::vector<dftfe::dftUtils::CompositeData *> dataRawPtrs(data.size());
        for (dftfe::uInt i = 0; i < data.size(); ++i)
          dataRawPtrs[i] = data[i].get();


        const std::string filename = "HubbardOccData.chk";

        dftUtils::MPIWriteOnFile().writeData(dataRawPtrs,
                                             filename,
                                             d_mpi_comm_domain);
      }
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  hubbard<ValueType, memorySpace>::readHubbOccFromFile()
  {
    pcout << " Reading hubbard occupation number \n";
    const std::string filename = "HubbardOccData.chk";
    std::ifstream     hubbOccInputFile(filename);

    const std::vector<dftfe::uInt> atomIdsInProc =
      d_atomicProjectorFnsContainer->getAtomIdsInCurrentProcess();

    std::map<dftfe::uInt, dftfe::uInt> mapGlobalIdToProcLocalId;

    mapGlobalIdToProcLocalId.clear();
    for (dftfe::uInt iAtom = 0; iAtom < atomIdsInProc.size(); iAtom++)
      {
        const dftfe::uInt atomId = atomIdsInProc[iAtom];
        dftfe::uInt       globalId =
          d_mapHubbardAtomToGlobalAtomId.find(atomId)->second;

        mapGlobalIdToProcLocalId[globalId] = iAtom;
      }


    const std::vector<dftfe::uInt> atomIdsInProcessor =
      d_atomicProjectorFnsContainer->getAtomIdsInCurrentProcess();

    std::vector<double> hubbOccTemp;
    hubbOccTemp.resize(d_maxOccMatSizePerAtom * d_numSpins);
    for (dftfe::uInt iGlobalAtomInd = 0; iGlobalAtomInd < d_totalNumHubbAtoms;
         iGlobalAtomInd++)
      {
        double globalAtomIndexFromFile;
        hubbOccInputFile >> globalAtomIndexFromFile;

        for (dftfe::uInt iOrb = 0; iOrb < d_numSpins * d_maxOccMatSizePerAtom;
             iOrb++)
          {
            hubbOccInputFile >> hubbOccTemp[iOrb];
          }

        if (mapGlobalIdToProcLocalId.find(globalAtomIndexFromFile) !=
            mapGlobalIdToProcLocalId.end())
          {
            dftfe::uInt iAtom =
              mapGlobalIdToProcLocalId.find(globalAtomIndexFromFile)->second;
            const dftfe::uInt atomId     = atomIdsInProcessor[iAtom];
            const dftfe::uInt hubbardIds = d_mapAtomToHubbardIds[atomId];

            const dftfe::uInt numSphericalFunc =
              d_hubbardSpeciesData[hubbardIds].numberSphericalFunc;

            for (dftfe::uInt spinIndex = 0; spinIndex < d_numSpins; spinIndex++)
              {
                for (dftfe::uInt iOrb = 0;
                     iOrb < numSphericalFunc * numSphericalFunc;
                     iOrb++)
                  {
                    d_occupationMatrix
                      [HubbardOccFieldType::In]
                      [spinIndex * d_numTotalOccMatrixEntriesPerSpin +
                       d_OccMatrixEntryStartForAtom[iAtom] + iOrb] =
                        hubbOccTemp[spinIndex * numSphericalFunc *
                                      numSphericalFunc +
                                    iOrb];
                  }
              }
          }
      }
    hubbOccInputFile.close();

    computeCouplingMatrix();
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  const dftfe::utils::MemoryStorage<ValueType, memorySpace> &
  hubbard<ValueType, memorySpace>::getCouplingMatrix(dftfe::uInt spinIndex)
  {
    return d_couplingMatrixEntries[spinIndex];
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  hubbard<ValueType, memorySpace>::computeCouplingMatrix()
  {
    d_couplingMatrixEntries.resize(d_numSpins);

    if (d_useSinglePrec)
      {
        d_couplingMatrixEntriesSinglePrec.resize(d_numSpins);
      }

    for (dftfe::uInt spinIndex = 0; spinIndex < d_numSpins; spinIndex++)
      {
        std::vector<ValueType>         Entries;
        const std::vector<dftfe::uInt> atomIdsInProcessor =
          d_atomicProjectorFnsContainer->getAtomIdsInCurrentProcess();
        std::vector<dftfe::uInt> atomicNumber =
          d_atomicProjectorFnsContainer->getAtomicNumbers();
        d_couplingMatrixEntries[spinIndex].clear();

        for (dftfe::Int iAtom = 0; iAtom < atomIdsInProcessor.size(); iAtom++)
          {
            const dftfe::uInt atomId     = atomIdsInProcessor[iAtom];
            const dftfe::uInt Znum       = atomicNumber[atomId];
            const dftfe::uInt hubbardIds = d_mapAtomToHubbardIds[atomId];
            const dftfe::uInt numberSphericalFunctions =
              d_hubbardSpeciesData[hubbardIds].numberSphericalFunc;
            const dftfe::uInt numberSphericalFunctionsSq =
              d_hubbardSpeciesData[hubbardIds].numberSphericalFuncSq;

            std::vector<ValueType> V(numberSphericalFunctions *
                                     numberSphericalFunctions);
            std::fill(V.begin(), V.end(), 0.0);

            for (dftfe::uInt iOrb = 0; iOrb < numberSphericalFunctions; iOrb++)
              {
                V[iOrb * numberSphericalFunctions + iOrb] =
                  0.5 * d_hubbardSpeciesData[hubbardIds].hubbardValue;

                for (dftfe::uInt jOrb = 0; jOrb < numberSphericalFunctions;
                     jOrb++)
                  {
                    dftfe::uInt index1 = iOrb * numberSphericalFunctions + jOrb;
                    dftfe::uInt index2 = jOrb * numberSphericalFunctions + iOrb;
                    V[iOrb * numberSphericalFunctions + jOrb] -=
                      0.5 * (d_hubbardSpeciesData[hubbardIds].hubbardValue) *
                      (d_occupationMatrix[HubbardOccFieldType::In]
                                         [spinIndex *
                                            d_numTotalOccMatrixEntriesPerSpin +
                                          d_OccMatrixEntryStartForAtom[iAtom] +
                                          index1] +
                       d_occupationMatrix[HubbardOccFieldType::In]
                                         [spinIndex *
                                            d_numTotalOccMatrixEntriesPerSpin +
                                          d_OccMatrixEntryStartForAtom[iAtom] +
                                          index2]);
                  }
              }

            for (dftfe::uInt iOrb = 0;
                 iOrb < numberSphericalFunctions * numberSphericalFunctions;
                 iOrb++)
              {
                Entries.push_back(V[iOrb]);
              }
          }

        if constexpr (memorySpace == dftfe::utils::MemorySpace::HOST)
          {
            d_couplingMatrixEntries[spinIndex].resize(Entries.size());
            d_couplingMatrixEntries[spinIndex].copyFrom(Entries);

            if (d_useSinglePrec)
              {
                std::vector<
                  typename dftfe::dataTypes::singlePrecType<ValueType>::type>
                  EntriesSinglePrec;
                EntriesSinglePrec.resize(Entries.size());

                for (dftfe::uInt index = 0; index < Entries.size(); index++)
                  {
                    EntriesSinglePrec[index] = Entries[index];
                  }

                d_couplingMatrixEntriesSinglePrec[spinIndex].resize(
                  EntriesSinglePrec.size());
                d_couplingMatrixEntriesSinglePrec[spinIndex].copyFrom(
                  EntriesSinglePrec);
              }
          }
#if defined(DFTFE_WITH_DEVICE)
        else
          {
            std::vector<ValueType> EntriesPadded;
            d_nonLocalOperator->paddingCouplingMatrix(Entries,
                                                      EntriesPadded,
                                                      CouplingStructure::dense);
            d_couplingMatrixEntries[spinIndex].resize(EntriesPadded.size());
            d_couplingMatrixEntries[spinIndex].copyFrom(EntriesPadded);

            if (d_useSinglePrec)
              {
                std::vector<
                  typename dftfe::dataTypes::singlePrecType<ValueType>::type>
                  EntriesPaddedSinglePrec;
                EntriesPaddedSinglePrec.resize(EntriesPadded.size());

                for (dftfe::uInt index = 0; index < EntriesPadded.size();
                     index++)
                  {
                    EntriesPaddedSinglePrec[index] = EntriesPadded[index];
                  }

                d_couplingMatrixEntriesSinglePrec[spinIndex].resize(
                  EntriesPaddedSinglePrec.size());
                d_couplingMatrixEntriesSinglePrec[spinIndex].copyFrom(
                  EntriesPaddedSinglePrec);
              }
          }
#endif
      }
  }


  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  hubbard<ValueType, memorySpace>::readHubbardInput(
    const std::vector<std::vector<double>> &atomLocations,
    const std::vector<dftfe::Int>          &imageIds,
    const std::vector<std::vector<double>> &imagePositions)
  {
    std::ifstream hubbardInputFile(d_dftParamsPtr->hubbardFileName);

    dftfe::uInt numberOfSpecies;
    hubbardInputFile >> numberOfSpecies;
    d_noSpecies =
      numberOfSpecies -
      1; // 0 is default species corresponding to no hubbard correction

    dftfe::uInt id, numberOfProjectors, atomicNumber;
    double      hubbardValue;
    dftfe::uInt numOfOrbitals;
    hubbardInputFile >> id >> numOfOrbitals; // reading for 0
    dftfe::Int n, l;
    double     initialOccupation;

    for (dftfe::uInt i = 1; i < numberOfSpecies; i++)
      {
        hubbardInputFile >> id >> atomicNumber >> hubbardValue >>
          numberOfProjectors >> initialOccupation;

        hubbardSpecies hubbardSpeciesObj;

        hubbardSpeciesObj.hubbardValue = hubbardValue;
        hubbardSpeciesObj.numProj      = numberOfProjectors;
        hubbardSpeciesObj.atomicNumber = atomicNumber;
        hubbardSpeciesObj.nQuantumNum.resize(numberOfProjectors);
        hubbardSpeciesObj.lQuantumNum.resize(numberOfProjectors);
        hubbardSpeciesObj.initialOccupation   = initialOccupation;
        hubbardSpeciesObj.numberSphericalFunc = 0;
        for (dftfe::uInt orbitalId = 0; orbitalId < numberOfProjectors;
             orbitalId++)
          {
            hubbardInputFile >> n >> l;
            hubbardSpeciesObj.nQuantumNum[orbitalId] = n;
            hubbardSpeciesObj.lQuantumNum[orbitalId] = l;

            hubbardSpeciesObj.numberSphericalFunc += 2 * l + 1;
          }

        hubbardSpeciesObj.numberSphericalFuncSq =
          hubbardSpeciesObj.numberSphericalFunc *
          hubbardSpeciesObj.numberSphericalFunc;
        d_hubbardSpeciesData[id - 1] = hubbardSpeciesObj;

        if (d_maxOccMatSizePerAtom < hubbardSpeciesObj.numberSphericalFuncSq)
          {
            d_maxOccMatSizePerAtom = hubbardSpeciesObj.numberSphericalFuncSq;
          }
      }

    std::vector<std::vector<dftfe::uInt>> mapAtomToImageAtom;
    mapAtomToImageAtom.resize(atomLocations.size());

    for (dftfe::uInt iAtom = 0; iAtom < atomLocations.size(); iAtom++)
      {
        mapAtomToImageAtom[iAtom].resize(0, 0);
      }
    for (dftfe::uInt imageIdIter = 0; imageIdIter < imageIds.size();
         imageIdIter++)
      {
        mapAtomToImageAtom[imageIds[imageIdIter]].push_back(imageIdIter);
      }

    std::vector<double> atomCoord;
    atomCoord.resize(3, 0.0);

    d_atomicCoords.resize(0);
    d_initialAtomicSpin.resize(0);
    d_periodicImagesCoords.resize(0);
    d_imageIds.resize(0);
    d_mapAtomToHubbardIds.resize(0);
    d_mapAtomToAtomicNumber.resize(0);
    dftfe::uInt hubbardAtomId = 0;
    dftfe::uInt atomicNum;
    d_totalNumHubbAtoms = 0;
    for (dftfe::uInt iAtom = 0; iAtom < atomLocations.size(); iAtom++)
      {
        hubbardInputFile >> atomicNum >> id;
        if (id != 0)
          {
            d_atomicCoords.push_back(atomLocations[iAtom][2]);
            d_atomicCoords.push_back(atomLocations[iAtom][3]);
            d_atomicCoords.push_back(atomLocations[iAtom][4]);
            if (atomLocations[iAtom].size() > 5)
              {
                d_initialAtomicSpin.push_back(atomLocations[iAtom][5]);
              }
            d_mapAtomToHubbardIds.push_back(id - 1);
            d_mapAtomToAtomicNumber.push_back(atomicNum);
            for (dftfe::uInt jImageAtom = 0;
                 jImageAtom < mapAtomToImageAtom[iAtom].size();
                 jImageAtom++)
              {
                atomCoord[0] =
                  imagePositions[mapAtomToImageAtom[iAtom][jImageAtom]][0];
                atomCoord[1] =
                  imagePositions[mapAtomToImageAtom[iAtom][jImageAtom]][1];
                atomCoord[2] =
                  imagePositions[mapAtomToImageAtom[iAtom][jImageAtom]][2];

                d_periodicImagesCoords.push_back(atomCoord);
                d_imageIds.push_back(hubbardAtomId);
              }

            d_mapHubbardAtomToGlobalAtomId[hubbardAtomId] = iAtom;
            hubbardAtomId++;
            d_totalNumHubbAtoms++;
          }
      }
    hubbardInputFile.close();
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  dftfe::uInt
  hubbard<ValueType, memorySpace>::getTotalNumberOfSphericalFunctionsForAtomId(
    dftfe::uInt iAtom)
  {
    const std::vector<dftfe::uInt> atomIdsInProc =
      d_atomicProjectorFnsContainer->getAtomIdsInCurrentProcess();

    const dftfe::uInt atomId     = atomIdsInProc[iAtom];
    const dftfe::uInt hubbardIds = d_mapAtomToHubbardIds[atomId];
    return d_hubbardSpeciesData[hubbardIds].numberSphericalFunc;
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  dftfe::uInt
  hubbard<ValueType, memorySpace>::getGlobalAtomId(dftfe::uInt iAtom)
  {
    const std::vector<dftfe::uInt> atomIdsInProc =
      d_atomicProjectorFnsContainer->getAtomIdsInCurrentProcess();

    const dftfe::uInt atomId = atomIdsInProc[iAtom];
    return d_mapHubbardAtomToGlobalAtomId.find(atomId)->second;
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  hubbard<ValueType, memorySpace>::initialiseOperatorActionOnX(
    dftfe::uInt kPointIndex)
  {
    d_nonLocalOperator->initialiseOperatorActionOnX(kPointIndex);

    if (d_useSinglePrec)
      {
        d_nonLocalOperatorSinglePrec->initialiseOperatorActionOnX(kPointIndex);
      }
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  hubbard<ValueType, memorySpace>::initialiseFlattenedDataStructure(
    dftfe::uInt numVectors)
  {
    d_nonLocalOperator->initialiseFlattenedDataStructure(
      numVectors, d_hubbNonLocalProjectorTimesVectorBlock);

    if (d_useSinglePrec)
      {
        d_nonLocalOperatorSinglePrec->initialiseFlattenedDataStructure(
          numVectors, d_hubbNonLocalProjectorTimesVectorBlockSinglePrec);
      }
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  hubbard<ValueType, memorySpace>::applyPotentialDueToHubbardCorrection(
    const dftfe::linearAlgebra::MultiVector<ValueType, memorySpace> &src,
    dftfe::linearAlgebra::MultiVector<ValueType, memorySpace>       &dst,
    const dftfe::uInt inputVecSize,
    const dftfe::uInt kPointIndex,
    const dftfe::uInt spinIndex)
  {
    d_nonLocalOperator->applyCconjtransOnX(src);
    d_hubbNonLocalProjectorTimesVectorBlock.setValue(0);
    d_nonLocalOperator->applyAllReduceOnCconjtransX(
      d_hubbNonLocalProjectorTimesVectorBlock);
    d_nonLocalOperator->applyVOnCconjtransX(
      CouplingStructure::dense,
      d_couplingMatrixEntries[spinIndex],
      d_hubbNonLocalProjectorTimesVectorBlock,
      true);
    d_nonLocalOperator->applyCOnVCconjtransX(dst);
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  hubbard<ValueType, memorySpace>::applyPotentialDueToHubbardCorrection(
    const dftfe::linearAlgebra::MultiVector<
      typename dataTypes::singlePrecType<ValueType>::type,
      memorySpace> &src,
    dftfe::linearAlgebra::MultiVector<
      typename dataTypes::singlePrecType<ValueType>::type,
      memorySpace>   &dst,
    const dftfe::uInt inputVecSize,
    const dftfe::uInt kPointIndex,
    const dftfe::uInt spinIndex)
  {
    d_nonLocalOperatorSinglePrec->applyCconjtransOnX(src);
    d_hubbNonLocalProjectorTimesVectorBlockSinglePrec.setValue(0);
    d_nonLocalOperatorSinglePrec->applyAllReduceOnCconjtransX(
      d_hubbNonLocalProjectorTimesVectorBlockSinglePrec);
    d_nonLocalOperatorSinglePrec->applyVOnCconjtransX(
      CouplingStructure::dense,
      d_couplingMatrixEntriesSinglePrec[spinIndex],
      d_hubbNonLocalProjectorTimesVectorBlockSinglePrec,
      true);
    d_nonLocalOperatorSinglePrec->applyCOnVCconjtransX(dst);
  }


  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  double
  hubbard<ValueType, memorySpace>::getHubbardEnergy()
  {
    return d_hubbardEnergy;
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  double
  hubbard<ValueType, memorySpace>::getExpectationOfHubbardPotential()
  {
    return d_expectationOfHubbardPotential;
  }

  template class hubbard<dataTypes::number, dftfe::utils::MemorySpace::HOST>;
#if defined(DFTFE_WITH_DEVICE)
  template class hubbard<dataTypes::number, dftfe::utils::MemorySpace::DEVICE>;
#endif
} // namespace dftfe
