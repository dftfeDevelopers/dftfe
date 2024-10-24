// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022  The Regents of the University of Michigan and DFT-FE
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

/*
 * @author Vishal Subramanian, Bikash Kanungo
 */

#include "InterpolateCellWiseDataToPoints.h"
#include "linearAlgebraOperationsInternal.h"
#include "linearAlgebraOperations.h"
#include "FECell.h"

#ifdef DFTFE_WITH_DEVICE
#  include "deviceDirectCCLWrapper.h"
#  include "elpaScalaManager.h"
#  include "dftParameters.h"
#  include <chebyshevOrthogonalizedSubspaceIterationSolverDevice.h>
#  include <dftUtils.h>
#  include <DeviceAPICalls.h>
#  include <DeviceDataTypeOverloads.h>
#  include <DeviceKernelLauncherConstants.h>
#  include <linearAlgebraOperationsDevice.h>
#  include <vectorUtilities.h>
#  include <DeviceAPICalls.h>
#  include <DeviceDataTypeOverloads.h>
#  include <DeviceKernelLauncherConstants.h>
#  include <vectorUtilities.h>
#endif

namespace dftfe
{
  template <typename T, dftfe::utils::MemorySpace memorySpace>
  InterpolateCellWiseDataToPoints<T, memorySpace>::
    InterpolateCellWiseDataToPoints(
      const std::vector<std::shared_ptr<const dftfe::utils::Cell<3>>> &srcCells,
      std::vector<
        std::shared_ptr<InterpolateFromCellToLocalPoints<memorySpace>>>
                                              interpolateLocalObj,
      const std::vector<std::vector<double>> &targetPts,
      const std::vector<unsigned int> &       numDofsPerElem,
      const unsigned int                      verbosity,
      const MPI_Comm &                        mpiComm)
    : d_mapPoints(verbosity, mpiComm)
    , d_mpiComm(mpiComm)
  {
    d_verbosity = verbosity;
    MPI_Barrier(d_mpiComm);
    double startComp                 = MPI_Wtime();
    d_numLocalPtsSze                 = targetPts.size();
    size_type numInitialTargetPoints = targetPts.size();

    d_interpolateLocalObj = interpolateLocalObj;

    std::vector<std::vector<double>> coordinatesOfPointsInCell;

    MPI_Barrier(d_mpiComm);
    double startMapPoints = MPI_Wtime();


    size_type maxNumCells  = srcCells.size();
    size_type maxNumPoints = numInitialTargetPoints;

    if (d_verbosity > 2)
      {
        std::cout << " NumPoints in proc = " << numInitialTargetPoints
                  << " numCells per proc = " << srcCells.size() << "\n";
      }
    MPI_Allreduce(MPI_IN_PLACE,
                  &maxNumCells,
                  1,
                  dftfe::dataTypes::mpi_type_id(&maxNumCells),
                  MPI_MAX,
                  d_mpiComm);

    MPI_Allreduce(MPI_IN_PLACE,
                  &maxNumPoints,
                  1,
                  dftfe::dataTypes::mpi_type_id(&maxNumPoints),
                  MPI_MAX,
                  d_mpiComm);

    if (dealii::Utilities::MPI::this_mpi_process(d_mpiComm) == 0)
      {
        std::cout << " maxNumPoints = " << maxNumPoints
                  << " maxNumCells = " << maxNumCells << "\n";
      }
    // create the RTree and the
    d_mapPoints.init(srcCells,
                     targetPts,
                     coordinatesOfPointsInCell,
                     d_mapCellLocalToProcLocal,
                     d_localRange,
                     d_ghostGlobalIds,
                     1e-7); // TODO this is hardcoded

    MPI_Barrier(d_mpiComm);
    double endMapPoints = MPI_Wtime();

    d_numCells = srcCells.size();

    d_cellPointStartIndex.resize(d_numCells);
    d_cellShapeFuncStartIndex.resize(d_numCells);


    d_numPointsLocal = 0;

    d_numDofsPerElement = numDofsPerElem;

    d_cumulativeDofs.resize(d_numCells);
    d_numPointsInCell.resize(d_numCells);


    d_pointsFoundInProc            = 0;
    d_cumulativeDofs[0]            = 0;
    global_size_type shapeFuncSize = 0;
    for (size_type iCell = 0; iCell < d_numCells; iCell++)
      {
        d_numPointsInCell[iCell] = coordinatesOfPointsInCell[iCell].size() / 3;
        d_pointsFoundInProc += d_numPointsInCell[iCell];

        shapeFuncSize += (global_size_type)(d_numPointsInCell[iCell] *
                                            d_numDofsPerElement[iCell]);
        if (iCell > 0)
          {
            d_cellPointStartIndex[iCell] =
              d_cellPointStartIndex[iCell - 1] + d_numPointsInCell[iCell - 1];
            d_cellShapeFuncStartIndex[iCell] =
              d_cellShapeFuncStartIndex[iCell - 1] +
              d_numDofsPerElement[iCell - 1] * (d_numPointsInCell[iCell - 1]);
            d_cumulativeDofs[iCell] =
              d_cumulativeDofs[iCell - 1] + d_numDofsPerElement[iCell - 1];
          }
        else
          {
            d_cellPointStartIndex[0]     = 0;
            d_cellShapeFuncStartIndex[0] = 0;
            d_cumulativeDofs[0]          = 0;
          }
      }
    totalDofsInCells = std::accumulate(d_numDofsPerElement.begin(),
                                       d_numDofsPerElement.end(),
                                       0.0);

    MPI_Barrier(d_mpiComm);
    double startShapeFunc = MPI_Wtime();

    d_numPointsLocal = targetPts.size() + d_ghostGlobalIds.size();


    size_type numFinalTargetPoints = targetPts.size();

    for (size_type iCell = 0; iCell < d_numCells; iCell++)
      {
        d_interpolateLocalObj[iCell]->setRealCoordinatesOfLocalPoints(
          d_numPointsInCell[iCell], coordinatesOfPointsInCell[iCell]);
      }


    global_size_type numTargetPointsInput = (global_size_type)targetPts.size();
    MPI_Allreduce(MPI_IN_PLACE,
                  &numTargetPointsInput,
                  1,
                  dftfe::dataTypes::mpi_type_id(&numTargetPointsInput),
                  MPI_SUM,
                  d_mpiComm);

    global_size_type numTargetPointsFound =
      (global_size_type)d_pointsFoundInProc;
    MPI_Allreduce(MPI_IN_PLACE,
                  &numTargetPointsFound,
                  1,
                  dftfe::dataTypes::mpi_type_id(&numTargetPointsFound),
                  MPI_SUM,
                  d_mpiComm);


    global_size_type numLocalPlusGhost = (global_size_type)d_numPointsLocal;
    MPI_Allreduce(MPI_IN_PLACE,
                  &numLocalPlusGhost,
                  1,
                  dftfe::dataTypes::mpi_type_id(&numLocalPlusGhost),
                  MPI_SUM,
                  d_mpiComm);


    std::cout << std::flush;
    MPI_Barrier(d_mpiComm);

    MPI_Barrier(d_mpiComm);
    double endShapeFunc = MPI_Wtime();
    d_mpiPatternP2PPtr  = std::make_shared<
      dftfe::utils::mpi::MPIPatternP2P<dftfe::utils::MemorySpace::HOST>>(
      d_localRange, d_ghostGlobalIds, d_mpiComm);


    d_mpiP2PPtrMemSpace =
      std::make_shared<dftfe::utils::mpi::MPIPatternP2P<memorySpace>>(
        d_localRange, d_ghostGlobalIds, d_mpiComm);
    std::vector<global_size_type> cellLocalToProcLocal;
    cellLocalToProcLocal.resize(d_pointsFoundInProc);


    size_type pointIndex = 0;
    for (size_type iCell = 0; iCell < d_numCells; iCell++)
      {
        for (size_type iPoint = 0; iPoint < d_numPointsInCell[iCell]; iPoint++)
          {
            cellLocalToProcLocal[pointIndex] =
              d_mapCellLocalToProcLocal[iCell][iPoint];
            pointIndex++;
          }
      }


    d_mapPointToProcLocalMemSpace.resize(cellLocalToProcLocal.size());
    d_mapPointToProcLocalMemSpace.copyFrom(cellLocalToProcLocal);


    checkIfAllPointsAreFound(targetPts);

    MPI_Barrier(d_mpiComm);
    double endMPIPattern = MPI_Wtime();

    double nonLocalFrac =
      ((double)((double)(numLocalPlusGhost - numTargetPointsInput)) /
       numTargetPointsInput);
    if ((dealii::Utilities::MPI::this_mpi_process(d_mpiComm) == 0))
      {
        std::cout << " Total number of points provided as input = "
                  << numTargetPointsInput << "\n";
        std::cout << " Total number of points found from input = "
                  << numTargetPointsFound << "\n";
        std::cout << " Total number of points in all procs = "
                  << numLocalPlusGhost << "\n";
        dftfe::utils::throwException(
          numTargetPointsFound >= numTargetPointsInput,
          " Number of points found is less than the input points \n");
      }
    if ((dealii::Utilities::MPI::this_mpi_process(d_mpiComm) == 0) &&
        (d_verbosity > 2))
      {
        std::cout << " Fraction of non local pts = " << nonLocalFrac << "\n";
        std::cout << " Time for start Comp = " << startMapPoints - startComp
                  << "\n";
        std::cout << " Time for map Points init = "
                  << endMapPoints - startMapPoints << "\n";
        std::cout << " Time for shape func array = "
                  << startShapeFunc - endMapPoints << "\n";
        std::cout << " Time for computing shape func = "
                  << endShapeFunc - startShapeFunc << "\n";
        std::cout << " time for MPI pattern creation = "
                  << endMPIPattern - endShapeFunc << "\n";
      }
  }


  template <typename T, dftfe::utils::MemorySpace memorySpace>
  void
  InterpolateCellWiseDataToPoints<T, memorySpace>::checkIfAllPointsAreFound(
    const std::vector<std::vector<double>> &targetPts)
  {
    dftfe::utils::MemoryStorage<T, dftfe::utils::MemorySpace::HOST> outputData;

    d_mpiCommP2PPtr = std::make_shared<
      dftfe::utils::mpi::MPICommunicatorP2P<T,
                                            dftfe::utils::MemorySpace::HOST>>(
      d_mpiPatternP2PPtr, 1);
    d_mpiCommP2PPtr->setCommunicationPrecision(
      dftfe::utils::mpi::communicationPrecision::full);
    outputData.resize(d_numPointsLocal);


    outputData.setValue(0.0);
    for (size_type iElemSrc = 0; iElemSrc < d_numCells; iElemSrc++)
      {
        unsigned int numberOfPointsInSrcCell = d_numPointsInCell[iElemSrc];
        for (unsigned int iPoint = 0; iPoint < numberOfPointsInSrcCell;
             iPoint++)
          {
            outputData[d_mapCellLocalToProcLocal[iElemSrc][iPoint]] = 1.0;
          }
      }


    d_mpiCommP2PPtr->accumulateInsertLocallyOwned(outputData);

    unsigned int pointsFound = 1;

    for (unsigned int iPoint = 0; iPoint < d_numLocalPtsSze; iPoint++)
      {
        if (std::abs(outputData.data()[iPoint] - 1.0) > 1e-3)
          {
            std::cout << "rank = "
                      << dealii::Utilities::MPI::this_mpi_process(d_mpiComm)
                      << " out Val = " << outputData.data()[iPoint]
                      << " x :" << targetPts[iPoint][0]
                      << " y : " << targetPts[iPoint][1]
                      << " z : " << targetPts[iPoint][2] << "\n";
          }
        if (std::abs(outputData.data()[iPoint] - 1.0) > 1e-3)
          {
            pointsFound = 0;
            if (d_verbosity >= 5)
              {
                std::cout << "rank = "
                          << dealii::Utilities::MPI::this_mpi_process(d_mpiComm)
                          << " out Val = " << outputData.data()[iPoint]
                          << " x :" << targetPts[iPoint][0]
                          << " y : " << targetPts[iPoint][1]
                          << " z : " << targetPts[iPoint][2] << "\n";
              }
          }
      }

    MPI_Allreduce(MPI_IN_PLACE,
                  &pointsFound,
                  1,
                  dftfe::dataTypes::mpi_type_id(&pointsFound),
                  MPI_MIN,
                  d_mpiComm);

    if (pointsFound == 1)
      {
        if ((dealii::Utilities::MPI::this_mpi_process(d_mpiComm) == 0))
          {
            std::cout << " All points found successfully \n";
          }
      }
    else
      {
        AssertThrow(false,
                    dealii::ExcMessage(
                      " Some points were not found while interpolating "));
      }
  }
  template <typename T, dftfe::utils::MemorySpace memorySpace>
  void
  InterpolateCellWiseDataToPoints<T, memorySpace>::
    interpolateSrcDataToTargetPoints(
      const std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
        &                         BLASWrapperPtr,
      const distributedCPUVec<T> &inputVec,
      const unsigned int          numberOfVectors,
      const dftfe::utils::MemoryStorage<dftfe::global_size_type,
                                        dftfe::utils::MemorySpace::HOST>
        &mapVecToCells,
      dftfe::utils::MemoryStorage<T,
                                  dftfe::utils::MemorySpace::HOST>
        &  outputData, // this is not std::vector
      bool resizeData)
  {
    if (resizeData)
      {
        d_mpiCommP2PPtr =
          std::make_shared<dftfe::utils::mpi::MPICommunicatorP2P<
            T,
            dftfe::utils::MemorySpace::HOST>>(d_mpiPatternP2PPtr,
                                              numberOfVectors);
        d_mpiCommP2PPtr->setCommunicationPrecision(
          dftfe::utils::mpi::communicationPrecision::full);
        outputData.resize(d_numPointsLocal * numberOfVectors);
      }

    std::fill(outputData.begin(), outputData.end(), 0.0);
    const T            scalarCoeffAlpha = 1.0;
    const T            scalarCoeffBeta  = 0.0;
    const char         transA = 'N', transB = 'N';
    const unsigned int inc = 1;


    unsigned int iElemSrc = 0;

    std::vector<T> cellLevelOutputPoints;

    for (size_type iElemSrc = 0; iElemSrc < d_numCells; iElemSrc++)
      {
        std::vector<T> cellLevelInputVec(d_numDofsPerElement[iElemSrc] *
                                           numberOfVectors,
                                         0.0);
        unsigned int   numberOfPointsInSrcCell = d_numPointsInCell[iElemSrc];
        cellLevelOutputPoints.resize(numberOfPointsInSrcCell * numberOfVectors);

        for (unsigned int iNode = 0; iNode < d_numDofsPerElement[iElemSrc];
             iNode++)
          {
            BLASWrapperPtr->xcopy(
              numberOfVectors,
              inputVec.begin() +
                mapVecToCells[d_cumulativeDofs[iElemSrc] + iNode],
              inc,
              &cellLevelInputVec[numberOfVectors * iNode],
              inc);
          }

        d_interpolateLocalObj[iElemSrc]->interpolate(BLASWrapperPtr,
                                                     numberOfVectors,
                                                     cellLevelInputVec,
                                                     cellLevelOutputPoints);

        for (unsigned int iPoint = 0; iPoint < numberOfPointsInSrcCell;
             iPoint++)
          {
            BLASWrapperPtr->xcopy(
              numberOfVectors,
              &cellLevelOutputPoints[iPoint * numberOfVectors],
              inc,
              &outputData[d_mapCellLocalToProcLocal[iElemSrc][iPoint] *
                          numberOfVectors],
              inc);
          }
      }

    d_mpiCommP2PPtr->accumulateInsertLocallyOwned(outputData);
  }

  template <typename T, dftfe::utils::MemorySpace memorySpace>
  void
  InterpolateCellWiseDataToPoints<T, memorySpace>::
    interpolateSrcDataToTargetPoints(
      const std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
        &                                                      BLASWrapperPtr,
      const dftfe::linearAlgebra::MultiVector<T, memorySpace> &inputVec,
      const unsigned int                                       numberOfVectors,
      const dftfe::utils::MemoryStorage<dftfe::global_size_type, memorySpace>
        &mapVecToCells,
      dftfe::utils::MemoryStorage<T,
                                  memorySpace>
        &  outputData, // this is not std::vector
      bool resizeData)
  {
#if defined(DFTFE_WITH_DEVICE)
    if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
      dftfe::utils::deviceSynchronize();
#endif
    MPI_Barrier(d_mpiComm);
    double startTime = MPI_Wtime();
    if (resizeData)
      {
        d_mpiCommPtrMemSpace = std::make_unique<
          dftfe::utils::mpi::MPICommunicatorP2P<T, memorySpace>>(
          d_mpiP2PPtrMemSpace, numberOfVectors);
        d_mpiCommPtrMemSpace->setCommunicationPrecision(
          dftfe::utils::mpi::communicationPrecision::full);
        outputData.resize(d_numPointsLocal * numberOfVectors);
        d_cellLevelParentNodalMemSpace.resize(totalDofsInCells *
                                              numberOfVectors);

        std::vector<global_size_type> cellLocalToProcLocal;
        cellLocalToProcLocal.resize(d_pointsFoundInProc);
        size_type pointIndex = 0;
        for (size_type iCell = 0; iCell < d_numCells; iCell++)
          {
            for (size_type iPoint = 0; iPoint < d_numPointsInCell[iCell];
                 iPoint++)
              {
                cellLocalToProcLocal[pointIndex] =
                  d_mapCellLocalToProcLocal[iCell][iPoint] * numberOfVectors;
                pointIndex++;
              }
          }


        d_mapPointToProcLocalMemSpace.resize(cellLocalToProcLocal.size());
        d_mapPointToProcLocalMemSpace.copyFrom(cellLocalToProcLocal);


        d_tempOutputMemSpace.resize(d_pointsFoundInProc * numberOfVectors);
      }

#if defined(DFTFE_WITH_DEVICE)
    if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
      dftfe::utils::deviceSynchronize();
#endif
    MPI_Barrier(d_mpiComm);
    double endResizeTime = MPI_Wtime();
    outputData.setValue(0.0);

    BLASWrapperPtr->stridedCopyToBlock(numberOfVectors,
                                       totalDofsInCells,
                                       inputVec.data(),
                                       d_cellLevelParentNodalMemSpace.begin(),
                                       mapVecToCells.data());

    dftfe::size_type pointStartIndex = 0;
    for (dftfe::size_type iCell = 0; iCell < d_numCells; iCell++)
      {
        d_interpolateLocalObj[iCell]->interpolate(
          BLASWrapperPtr,
          numberOfVectors,
          d_cellLevelParentNodalMemSpace.data() +
            numberOfVectors * d_cumulativeDofs[iCell],
          d_tempOutputMemSpace.data() + pointStartIndex * numberOfVectors);

        pointStartIndex += d_numPointsInCell[iCell];
      }

    BLASWrapperPtr->axpyStridedBlockAtomicAdd(
      numberOfVectors,
      d_pointsFoundInProc,
      d_tempOutputMemSpace.data(),
      outputData.begin(),
      d_mapPointToProcLocalMemSpace.begin());


#if defined(DFTFE_WITH_DEVICE)
    if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
      dftfe::utils::deviceSynchronize();
#endif
    MPI_Barrier(d_mpiComm);
    double endCompTime = MPI_Wtime();
    d_mpiCommPtrMemSpace->accumulateInsertLocallyOwned(outputData);

#if defined(DFTFE_WITH_DEVICE)
    if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
      dftfe::utils::deviceSynchronize();
#endif
    MPI_Barrier(d_mpiComm);
    double endCommTime = MPI_Wtime();

    int thisRankId;
    MPI_Comm_rank(d_mpiComm, &thisRankId);
    if ((thisRankId == 0) && (d_verbosity > 2))
      {
        std::cout << " resize Time = " << endResizeTime - startTime
                  << " Comp time = " << endCompTime - endResizeTime
                  << " comm time = " << endCommTime - endCompTime << "\n";
      }
  }


  template class InterpolateCellWiseDataToPoints<
    dftfe::dataTypes::number,
    dftfe::utils::MemorySpace::HOST>;

#ifdef DFTFE_WITH_DEVICE
  template class InterpolateCellWiseDataToPoints<
    dftfe::dataTypes::number,
    dftfe::utils::MemorySpace::DEVICE>;
#endif
} // namespace dftfe
