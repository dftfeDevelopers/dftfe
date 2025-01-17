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
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  hubbard<ValueType,
          memorySpace>::createAtomCenteredSphericalFunctionsForProjectors()
  {
    for (auto const &[key, val] : d_hubbardSpeciesData)
      {
        unsigned int Znum = val.atomicNumber;

        unsigned int numberOfProjectors = val.numProj;

        unsigned int numProj;
        unsigned int alpha = 0;
        for (unsigned int i = 0; i < numberOfProjectors; i++)
          {
            char         projRadialFunctionFileName[512];
            unsigned int nQuantumNo = val.nQuantumNum[i];
            unsigned int lQuantumNo = val.lQuantumNum[i];

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
    const unsigned int                      matrixFreeVectorComponent,
    const unsigned int                      densityQuadratureId,
    const unsigned int                      sparsityPatternQuadratureId,
    const unsigned int                      numberWaveFunctions,
    const unsigned int                      numSpins,
    const dftParameters &                   dftParam,
    const std::string &                     scratchFolderName,
    const bool                              singlePrecNonLocalOperator,
    const bool                              updateNonlocalSparsity,
    const std::vector<std::vector<double>> &atomLocations,
    const std::vector<std::vector<double>> &atomLocationsFrac,
    const std::vector<int> &                imageIds,
    const std::vector<std::vector<double>> &imagePositions,
    const std::vector<double> &             kPointCoordinates,
    const std::vector<double> &             kPointWeights,
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


    d_cellsBlockSizeApply = memorySpace == dftfe::utils::MemorySpace::HOST ?
                              1 :
                              d_BasisOperatorMemPtr->nCells();
    d_numSpins = numSpins;

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
        d_mpi_comm_domain);


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
      densityQuadratureId);

    MPI_Barrier(d_mpi_comm_domain);
    double endRead = MPI_Wtime();

    dftUtils::createBandParallelizationIndices(
      d_mpi_comm_interBand,
      d_numberWaveFunctions,
      d_bandGroupLowHighPlusOneIndices);


    d_spinPolarizedFactor = (d_dftParamsPtr->spinPolarized == 1) ? 1.0 : 2.0;

    d_noOfSpin = (d_dftParamsPtr->spinPolarized == 1) ? 2 : 1;

    unsigned int numLocalAtomsInProc =
      d_nonLocalOperator->getTotalAtomInCurrentProcessor();

    const std::vector<unsigned int> atomIdsInProc =
      d_atomicProjectorFnsContainer->getAtomIdsInCurrentProcess();
    std::vector<unsigned int> atomicNumber =
      d_atomicProjectorFnsContainer->getAtomicNumbers();

    d_numTotalOccMatrixEntriesPerSpin = 0;
    d_OccMatrixEntryStartForAtom.resize(0);

    for (int iAtom = 0; iAtom < numLocalAtomsInProc; iAtom++)
      {
        const unsigned int atomId     = atomIdsInProc[iAtom];
        const unsigned int Znum       = atomicNumber[atomId];
        const unsigned int hubbardIds = d_mapAtomToHubbardIds[atomId];

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

    std::vector<unsigned int> atomProcessorMap;
    unsigned int              numAtoms = atomLocations.size();
    atomProcessorMap.resize(numAtoms);

    int thisRank = dealii::Utilities::MPI::this_mpi_process(d_mpi_comm_domain);
    const unsigned int nRanks =
      dealii::Utilities::MPI::n_mpi_processes(d_mpi_comm_domain);

    for (unsigned int iAtom = 0; iAtom < numAtoms; iAtom++)
      {
        atomProcessorMap[iAtom] = nRanks;
      }

    for (int iAtom = 0; iAtom < numLocalAtomsInProc; iAtom++)
      {
        const unsigned int atomId = atomIdsInProc[iAtom];
        atomProcessorMap[atomId]  = thisRank;
      }
    MPI_Allreduce(MPI_IN_PLACE,
                  &atomProcessorMap[0],
                  numAtoms,
                  MPI_UNSIGNED,
                  MPI_MIN,
                  d_mpi_comm_domain);

    d_procLocalAtomId.resize(0);

    for (int iAtom = 0; iAtom < numLocalAtomsInProc; iAtom++)
      {
        const unsigned int atomId = atomIdsInProc[iAtom];
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
    unsigned int numLocalAtomsInProc =
      d_nonLocalOperator->getTotalAtomInCurrentProcessor();
    const std::vector<unsigned int> atomIdsInProc =
      d_atomicProjectorFnsContainer->getAtomIdsInCurrentProcess();
    std::vector<unsigned int> atomicNumber =
      d_atomicProjectorFnsContainer->getAtomicNumbers();

    for (unsigned int iSpin = 0; iSpin < d_numSpins; iSpin++)
      {
        for (int iAtom = 0; iAtom < numLocalAtomsInProc; iAtom++)
          {
            const unsigned int atomId       = atomIdsInProc[iAtom];
            const unsigned int Znum         = atomicNumber[atomId];
            const unsigned int hubbardIds   = d_mapAtomToHubbardIds[atomId];
            double             initOccValue = 0.0;
            if (d_numSpins == 1)
              {
                initOccValue =
                  d_hubbardSpeciesData[hubbardIds].initialOccupation /
                  (2.0 * d_hubbardSpeciesData[hubbardIds].numberSphericalFunc);
              }
            else if (d_numSpins == 2)
              {
                if (iSpin == 0)
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
                else if (iSpin == 1)
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
            for (unsigned int iOrb = 0;
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
    unsigned int numOwnedAtomsInProc = d_procLocalAtomId.size();
    const std::vector<unsigned int> atomIdsInProc =
      d_atomicProjectorFnsContainer->getAtomIdsInCurrentProcess();
    std::vector<unsigned int> atomicNumber =
      d_atomicProjectorFnsContainer->getAtomicNumbers();
    for (unsigned int iAtom = 0; iAtom < numOwnedAtomsInProc; iAtom++)
      {
        const unsigned int atomId     = atomIdsInProc[d_procLocalAtomId[iAtom]];
        const unsigned int Znum       = atomicNumber[atomId];
        const unsigned int hubbardIds = d_mapAtomToHubbardIds[atomId];

        const unsigned int numSphericalFunc =
          d_hubbardSpeciesData[hubbardIds].numberSphericalFunc;

        for (unsigned int spinIndex = 0; spinIndex < d_numSpins; spinIndex++)
          {
            for (unsigned int iOrb = 0; iOrb < numSphericalFunc; iOrb++)
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

                for (unsigned int jOrb = 0; jOrb < numSphericalFunc; jOrb++)
                  {
                    unsigned int index1 = iOrb * numSphericalFunc + jOrb;
                    unsigned int index2 = jOrb * numSphericalFunc + iOrb;

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
    const std::vector<std::vector<double>> &                   orbitalOccupancy)
  {
    unsigned int numLocalAtomsInProc =
      d_nonLocalOperator->getTotalAtomInCurrentProcessor();

    const std::vector<unsigned int> atomIdsInProc =
      d_atomicProjectorFnsContainer->getAtomIdsInCurrentProcess();
    std::vector<unsigned int> atomicNumber =
      d_atomicProjectorFnsContainer->getAtomicNumbers();

    std::fill(d_occupationMatrix[HubbardOccFieldType::Out].begin(),
              d_occupationMatrix[HubbardOccFieldType::Out].end(),
              0.0);



    const ValueType    zero = 0;
    const unsigned int cellsBlockSize =
      memorySpace == dftfe::utils::MemorySpace::DEVICE ?
        d_BasisOperatorMemPtr->nCells() :
        1;
    const unsigned int totalLocallyOwnedCells = d_BasisOperatorMemPtr->nCells();
    const unsigned int numCellBlocks = totalLocallyOwnedCells / cellsBlockSize;
    const unsigned int remCellBlockSize =
      totalLocallyOwnedCells - numCellBlocks * cellsBlockSize;

    const unsigned int BVec =
      d_dftParamsPtr->chebyWfcBlockSize; // TODO extend to band parallelisation

    d_BasisOperatorMemPtr->reinit(BVec, cellsBlockSize, d_densityQuadratureId);
    const unsigned int numQuadPoints = d_BasisOperatorMemPtr->nQuadsPerCell();

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

    const unsigned int bandGroupTaskId =
      dealii::Utilities::MPI::this_mpi_process(d_mpi_comm_interBand);

    unsigned int numLocalDofs       = d_BasisOperatorHostPtr->nOwnedDofs();
    unsigned int numNodesPerElement = d_BasisOperatorHostPtr->nDofsPerCell();
    dftfe::utils::MemoryStorage<ValueType, memorySpace> tempCellNodalData;
    unsigned int                                        previousSize = 0;
    for (unsigned int kPoint = 0; kPoint < d_kPointWeights.size(); ++kPoint)
      {
        for (unsigned int spinIndex = 0; spinIndex < d_noOfSpin; ++spinIndex)
          {
            d_nonLocalOperator->initialiseOperatorActionOnX(kPoint);
            for (unsigned int jvec = 0; jvec < d_numberWaveFunctions;
                 jvec += BVec)
              {
                const unsigned int currentBlockSize =
                  std::min(BVec, d_numberWaveFunctions - jvec);
                flattenedArrayBlock =
                  &(d_BasisOperatorMemPtr->getMultiVector(currentBlockSize, 0));
                d_nonLocalOperator->initialiseFlattenedDataStructure(
                  currentBlockSize, projectorKetTimesVector);

                tempCellNodalData.resize(cellsBlockSize * currentBlockSize *
                                         numNodesPerElement);

                previousSize = cellsBlockSize * currentBlockSize;
                if constexpr (dftfe::utils::MemorySpace::DEVICE == memorySpace)
                  {
                    d_nonLocalOperator->initialiseCellWaveFunctionPointers(
                      tempCellNodalData);
                  }

                if (((jvec + currentBlockSize) <=
                     d_bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId +
                                                      1]) &&
                    ((jvec + currentBlockSize) >
                     d_bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId]))
                  {
                    for (unsigned int iEigenVec = 0;
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
                      for (unsigned int iNode = 0; iNode < numLocalDofs;
                           ++iNode)
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


                    for (int iblock = 0; iblock < (numCellBlocks + 1); iblock++)
                      {
                        const unsigned int currentCellsBlockSize =
                          (iblock == numCellBlocks) ? remCellBlockSize :
                                                      cellsBlockSize;
                        if (currentCellsBlockSize > 0)
                          {
                            const unsigned int startingCellId =
                              iblock * cellsBlockSize;
                            if (currentCellsBlockSize * currentBlockSize !=
                                previousSize)
                              {
                                tempCellNodalData.resize(currentCellsBlockSize *
                                                         currentBlockSize *
                                                         numNodesPerElement);
                                previousSize =
                                  currentCellsBlockSize * currentBlockSize;
                              }
                            d_BasisOperatorMemPtr->extractToCellNodalDataKernel(
                              *(flattenedArrayBlock),
                              tempCellNodalData.data(),
                              std::pair<unsigned int, unsigned int>(
                                startingCellId,
                                startingCellId + currentCellsBlockSize));

                            if (
                              d_nonLocalOperator
                                ->getTotalNonLocalElementsInCurrentProcessor() >
                              0)
                              {
                                d_nonLocalOperator->applyCconjtransOnX(
                                  tempCellNodalData.data(),
                                  std::pair<unsigned int, unsigned int>(
                                    startingCellId,
                                    startingCellId + currentCellsBlockSize));
                              }
                          }
                      }
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

    unsigned int numOwnedAtomsInProc = d_procLocalAtomId.size();
    for (unsigned int iAtom = 0; iAtom < numOwnedAtomsInProc; iAtom++)
      {
        const unsigned int atomId     = atomIdsInProc[d_procLocalAtomId[iAtom]];
        const unsigned int Znum       = atomicNumber[atomId];
        const unsigned int hubbardIds = d_mapAtomToHubbardIds[atomId];

        const unsigned int numSphericalFunc =
          d_hubbardSpeciesData[hubbardIds].numberSphericalFunc;

        if (d_verbosity >= 3)
          {
            for (unsigned int spinIndex = 0; spinIndex < d_numSpins;
                 spinIndex++)
              {
                for (unsigned int iOrb = 0; iOrb < numSphericalFunc; iOrb++)
                  {
                    for (unsigned int jOrb = 0; jOrb < numSphericalFunc; jOrb++)
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
    const bool         isOccOut,
    const unsigned int vectorBlockSize,
    const unsigned int spinIndex,
    const unsigned int kpointIndex)
  {
    const std::vector<unsigned int> atomIdsInProcessor =
      d_atomicProjectorFnsContainer->getAtomIdsInCurrentProcess();
    std::vector<unsigned int> atomicNumber =
      d_atomicProjectorFnsContainer->getAtomicNumbers();
    char transB = 'N';
#ifdef USE_COMPLEX
    char transA = 'C';
#else
    char  transA          = 'T';
#endif
    const ValueType beta  = 0.0;
    const ValueType alpha = 1.0;
    for (int iAtom = 0; iAtom < atomIdsInProcessor.size(); iAtom++)
      {
        const unsigned int atomId     = atomIdsInProcessor[iAtom];
        const unsigned int Znum       = atomicNumber[atomId];
        const unsigned int hubbardIds = d_mapAtomToHubbardIds[atomId];
        const unsigned int numberSphericalFunctionsSq =
          d_hubbardSpeciesData[hubbardIds].numberSphericalFuncSq;

        const unsigned int numberSphericalFunctions =
          d_hubbardSpeciesData[hubbardIds].numberSphericalFunc;

        const unsigned int numberSphericalFunc =
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
    for (unsigned int iElem = 0;
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
    for (unsigned int iElem = 0;
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
    unsigned int bandGroupTaskId =
      dealii::Utilities::MPI::this_mpi_process(d_mpi_comm_interBand);

    unsigned int interPoolId =
      dealii::Utilities::MPI::this_mpi_process(d_mpi_comm_interPool);

    const std::vector<unsigned int> atomIdsInProc =
      d_atomicProjectorFnsContainer->getAtomIdsInCurrentProcess();

    std::vector<unsigned int> atomicNumber =
      d_atomicProjectorFnsContainer->getAtomicNumbers();

    if ((bandGroupTaskId == 0) && (interPoolId == 0))
      {
        std::vector<std::shared_ptr<dftfe::dftUtils::CompositeData>> data(0);

        unsigned int numOwnedAtomsInProc = d_procLocalAtomId.size();
        for (unsigned int iAtom = 0; iAtom < numOwnedAtomsInProc; iAtom++)
          {
            const unsigned int atomId = atomIdsInProc[d_procLocalAtomId[iAtom]];
            const unsigned int Znum   = atomicNumber[atomId];
            const unsigned int hubbardIds = d_mapAtomToHubbardIds[atomId];

            const unsigned int numSphericalFunc =
              d_hubbardSpeciesData[hubbardIds].numberSphericalFunc;

            std::vector<double> nodeVals(0);

            nodeVals.push_back((double)getGlobalAtomId(atomId));
            for (unsigned int spinIndex = 0; spinIndex < d_numSpins;
                 spinIndex++)
              {
                for (unsigned int iOrb = 0;
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

            for (unsigned int iOrb =
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
        for (unsigned int i = 0; i < data.size(); ++i)
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

    const std::vector<unsigned int> atomIdsInProc =
      d_atomicProjectorFnsContainer->getAtomIdsInCurrentProcess();

    std::map<unsigned int, unsigned int> mapGlobalIdToProcLocalId;

    mapGlobalIdToProcLocalId.clear();
    for (unsigned int iAtom = 0; iAtom < atomIdsInProc.size(); iAtom++)
      {
        const unsigned int atomId = atomIdsInProc[iAtom];
        unsigned int       globalId =
          d_mapHubbardAtomToGlobalAtomId.find(atomId)->second;

        mapGlobalIdToProcLocalId[globalId] = iAtom;
      }


    const std::vector<unsigned int> atomIdsInProcessor =
      d_atomicProjectorFnsContainer->getAtomIdsInCurrentProcess();

    std::vector<double> hubbOccTemp;
    hubbOccTemp.resize(d_maxOccMatSizePerAtom * d_numSpins);
    for (unsigned int iGlobalAtomInd = 0; iGlobalAtomInd < d_totalNumHubbAtoms;
         iGlobalAtomInd++)
      {
        double globalAtomIndexFromFile;
        hubbOccInputFile >> globalAtomIndexFromFile;

        for (unsigned int iOrb = 0; iOrb < d_numSpins * d_maxOccMatSizePerAtom;
             iOrb++)
          {
            hubbOccInputFile >> hubbOccTemp[iOrb];
          }

        if (mapGlobalIdToProcLocalId.find(globalAtomIndexFromFile) !=
            mapGlobalIdToProcLocalId.end())
          {
            unsigned int iAtom =
              mapGlobalIdToProcLocalId.find(globalAtomIndexFromFile)->second;
            const unsigned int atomId     = atomIdsInProcessor[iAtom];
            const unsigned int hubbardIds = d_mapAtomToHubbardIds[atomId];

            const unsigned int numSphericalFunc =
              d_hubbardSpeciesData[hubbardIds].numberSphericalFunc;

            for (unsigned int spinIndex = 0; spinIndex < d_numSpins;
                 spinIndex++)
              {
                for (unsigned int iOrb = 0;
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
  hubbard<ValueType, memorySpace>::getCouplingMatrix(unsigned int spinIndex)
  {
    return d_couplingMatrixEntries[spinIndex];
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  hubbard<ValueType, memorySpace>::computeCouplingMatrix()
  {
    d_couplingMatrixEntries.resize(d_numSpins);
    for (unsigned int spinIndex = 0; spinIndex < d_numSpins; spinIndex++)
      {
        std::vector<ValueType>          Entries;
        const std::vector<unsigned int> atomIdsInProcessor =
          d_atomicProjectorFnsContainer->getAtomIdsInCurrentProcess();
        std::vector<unsigned int> atomicNumber =
          d_atomicProjectorFnsContainer->getAtomicNumbers();
        d_couplingMatrixEntries[spinIndex].clear();

        for (int iAtom = 0; iAtom < atomIdsInProcessor.size(); iAtom++)
          {
            const unsigned int atomId     = atomIdsInProcessor[iAtom];
            const unsigned int Znum       = atomicNumber[atomId];
            const unsigned int hubbardIds = d_mapAtomToHubbardIds[atomId];
            const unsigned int numberSphericalFunctions =
              d_hubbardSpeciesData[hubbardIds].numberSphericalFunc;
            const unsigned int numberSphericalFunctionsSq =
              d_hubbardSpeciesData[hubbardIds].numberSphericalFuncSq;

            std::vector<ValueType> V(numberSphericalFunctions *
                                     numberSphericalFunctions);
            std::fill(V.begin(), V.end(), 0.0);

            for (unsigned int iOrb = 0; iOrb < numberSphericalFunctions; iOrb++)
              {
                V[iOrb * numberSphericalFunctions + iOrb] =
                  0.5 * d_hubbardSpeciesData[hubbardIds].hubbardValue;

                for (unsigned int jOrb = 0; jOrb < numberSphericalFunctions;
                     jOrb++)
                  {
                    unsigned int index1 =
                      iOrb * numberSphericalFunctions + jOrb;
                    unsigned int index2 =
                      jOrb * numberSphericalFunctions + iOrb;
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

            for (unsigned int iOrb = 0;
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
          }
#endif
      }
  }


  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  hubbard<ValueType, memorySpace>::readHubbardInput(
    const std::vector<std::vector<double>> &atomLocations,
    const std::vector<int> &                imageIds,
    const std::vector<std::vector<double>> &imagePositions)
  {
    std::ifstream hubbardInputFile(d_dftParamsPtr->hubbardFileName);

    unsigned int numberOfSpecies;
    hubbardInputFile >> numberOfSpecies;
    d_noSpecies =
      numberOfSpecies -
      1; // 0 is default species corresponding to no hubbard correction

    unsigned int id, numberOfProjectors, atomicNumber;
    double       hubbardValue;
    unsigned int numOfOrbitals;
    hubbardInputFile >> id >> numOfOrbitals; // reading for 0
    int    n, l;
    double initialOccupation;

    for (unsigned int i = 1; i < numberOfSpecies; i++)
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
        for (unsigned int orbitalId = 0; orbitalId < numberOfProjectors;
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

    std::vector<std::vector<unsigned int>> mapAtomToImageAtom;
    mapAtomToImageAtom.resize(atomLocations.size());

    for (unsigned int iAtom = 0; iAtom < atomLocations.size(); iAtom++)
      {
        mapAtomToImageAtom[iAtom].resize(0, 0);
      }
    for (unsigned int imageIdIter = 0; imageIdIter < imageIds.size();
         imageIdIter++)
      {
        mapAtomToImageAtom[imageIds[imageIdIter]].push_back(imageIdIter);
      }

    std::vector<double> atomCoord;
    atomCoord.resize(3, 0.0);

    d_atomicCoords.resize(0);
    d_periodicImagesCoords.resize(0);
    d_imageIds.resize(0);
    d_mapAtomToHubbardIds.resize(0);
    d_mapAtomToAtomicNumber.resize(0);
    unsigned int hubbardAtomId = 0;
    unsigned int atomicNum;
    d_totalNumHubbAtoms = 0;
    for (unsigned int iAtom = 0; iAtom < atomLocations.size(); iAtom++)
      {
        hubbardInputFile >> atomicNum >> id;
        if (id != 0)
          {
            d_atomicCoords.push_back(atomLocations[iAtom][2]);
            d_atomicCoords.push_back(atomLocations[iAtom][3]);
            d_atomicCoords.push_back(atomLocations[iAtom][4]);
            d_mapAtomToHubbardIds.push_back(id - 1);
            d_mapAtomToAtomicNumber.push_back(atomicNum);
            for (unsigned int jImageAtom = 0;
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
  unsigned int
  hubbard<ValueType, memorySpace>::getTotalNumberOfSphericalFunctionsForAtomId(
    unsigned int iAtom)
  {
    const std::vector<unsigned int> atomIdsInProc =
      d_atomicProjectorFnsContainer->getAtomIdsInCurrentProcess();

    const unsigned int atomId     = atomIdsInProc[iAtom];
    const unsigned int hubbardIds = d_mapAtomToHubbardIds[atomId];
    return d_hubbardSpeciesData[hubbardIds].numberSphericalFunc;
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  unsigned int
  hubbard<ValueType, memorySpace>::getGlobalAtomId(unsigned int iAtom)
  {
    const std::vector<unsigned int> atomIdsInProc =
      d_atomicProjectorFnsContainer->getAtomIdsInCurrentProcess();

    const unsigned int atomId = atomIdsInProc[iAtom];
    return d_mapHubbardAtomToGlobalAtomId.find(atomId)->second;
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  hubbard<ValueType, memorySpace>::initialiseOperatorActionOnX(
    unsigned int kPointIndex)
  {
    d_nonLocalOperator->initialiseOperatorActionOnX(kPointIndex);
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  hubbard<ValueType, memorySpace>::initialiseFlattenedDataStructure(
    unsigned int numVectors)
  {
    d_nonLocalOperator->initialiseFlattenedDataStructure(
      numVectors, d_hubbNonLocalProjectorTimesVectorBlock);
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  hubbard<ValueType, memorySpace>::applyPotentialDueToHubbardCorrection(
    const dftfe::linearAlgebra::MultiVector<ValueType, memorySpace> &src,
    dftfe::linearAlgebra::MultiVector<ValueType, memorySpace> &      dst,
    const unsigned int inputVecSize,
    const unsigned int kPointIndex,
    const unsigned int spinIndex)
  {
    dst.setValue(0.0);
    if (d_nonLocalOperator->getTotalNonLocalElementsInCurrentProcessor() > 0)
      {
        const unsigned int nCells       = d_BasisOperatorMemPtr->nCells();
        const unsigned int nDofsPerCell = d_BasisOperatorMemPtr->nDofsPerCell();

        const ValueType scalarCoeffAlpha = ValueType(1.0),
                        scalarCoeffBeta  = ValueType(0.0);

        // TODO check if this will lead to performance degradation
        d_cellWaveFunctionMatrixDst.setValue(0.0);
        d_cellWaveFunctionMatrixSrc.setValue(0.0);
        Assert(
          d_cellWaveFunctionMatrixSrc.size() >=
            nCells * nDofsPerCell * inputVecSize,
          dealii::ExcMessage(
            "DFT-FE Error: d_cellWaveFunctionMatrixSrc in Hubbard is not set properly. Call initialiseCellWaveFunctionPointers()."));

        Assert(
          d_cellWaveFunctionMatrixDst.size() >=
            d_cellsBlockSizeApply * nDofsPerCell * inputVecSize,
          dealii::ExcMessage(
            "DFT-FE Error: d_cellWaveFunctionMatrixDst in Hubbard is not set properly. Call initialiseCellWaveFunctionPointers()."));

        Assert(
          d_BasisOperatorMemPtr->nVectors() == inputVecSize,
          dealii::ExcMessage(
            "DFT-FE Error: d_BasisOperatorMemPtr in Hubbard is not set with correct input size."));


        unsigned int hamiltonianIndex =
          d_dftParamsPtr->memOptMode ?
            0 :
            kPointIndex * (d_dftParamsPtr->spinPolarized + 1) + spinIndex;
        for (unsigned int iCell = 0; iCell < nCells;
             iCell += d_cellsBlockSizeApply)
          {
            std::pair<unsigned int, unsigned int> cellRange(
              iCell, std::min(iCell + d_cellsBlockSizeApply, nCells));

            d_BLASWrapperMemPtr->stridedCopyToBlock(
              inputVecSize,
              nDofsPerCell * (cellRange.second - cellRange.first),
              src.data(),
              d_cellWaveFunctionMatrixSrc.data() +
                cellRange.first * nDofsPerCell * inputVecSize,
              d_BasisOperatorMemPtr->d_flattenedCellDofIndexToProcessDofIndexMap
                  .data() +
                cellRange.first * nDofsPerCell);

            d_nonLocalOperator->applyCconjtransOnX(
              d_cellWaveFunctionMatrixSrc.data() +
                cellRange.first * nDofsPerCell * inputVecSize,
              cellRange);
          }

        d_hubbNonLocalProjectorTimesVectorBlock.setValue(0);
        d_nonLocalOperator->applyAllReduceOnCconjtransX(
          d_hubbNonLocalProjectorTimesVectorBlock);
        d_nonLocalOperator->applyVOnCconjtransX(
          CouplingStructure::dense,
          d_couplingMatrixEntries[spinIndex],
          d_hubbNonLocalProjectorTimesVectorBlock,
          true);

        for (unsigned int iCell = 0; iCell < nCells;
             iCell += d_cellsBlockSizeApply)
          {
            std::pair<unsigned int, unsigned int> cellRange(
              iCell, std::min(iCell + d_cellsBlockSizeApply, nCells));

            // d_cellWaveFunctionMatrixDst has to be reinitialised to zero
            // before applyCOnVCconjtransX() is called
            d_cellWaveFunctionMatrixDst.setValue(0.0);

            d_nonLocalOperator->applyCOnVCconjtransX(
              d_cellWaveFunctionMatrixDst.data(), cellRange);


            d_BLASWrapperMemPtr->axpyStridedBlockAtomicAdd(
              inputVecSize,
              nDofsPerCell * (cellRange.second - cellRange.first),
              d_cellWaveFunctionMatrixDst.data(),
              dst.data(),
              d_BasisOperatorMemPtr->d_flattenedCellDofIndexToProcessDofIndexMap
                  .data() +
                cellRange.first * nDofsPerCell);
          }
      }
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  hubbard<ValueType, memorySpace>::initialiseCellWaveFunctionPointers(
    unsigned int numVectors)
  {
    const unsigned int nCells       = d_BasisOperatorMemPtr->nCells();
    const unsigned int nDofsPerCell = d_BasisOperatorMemPtr->nDofsPerCell();
    unsigned int       cellWaveFuncSizeSrc = nCells * nDofsPerCell * numVectors;
    if (d_cellWaveFunctionMatrixSrc.size() < cellWaveFuncSizeSrc)
      {
        d_cellWaveFunctionMatrixSrc.resize(cellWaveFuncSizeSrc);
      }
    if constexpr (dftfe::utils::MemorySpace::DEVICE == memorySpace)
      {
        d_nonLocalOperator->initialiseCellWaveFunctionPointers(
          d_cellWaveFunctionMatrixSrc);
      }

    if (d_cellWaveFunctionMatrixDst.size() <
        d_cellsBlockSizeApply * nDofsPerCell * numVectors)
      {
        d_cellWaveFunctionMatrixDst.resize(d_cellsBlockSizeApply *
                                           nDofsPerCell * numVectors);
      }
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
