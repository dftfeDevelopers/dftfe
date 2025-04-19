
#include <algorithm>
namespace dftfe
{
  namespace utils
  {
    namespace
    {
      template <typename T>
      void
      appendToVec(std::vector<T> &dst, const std::vector<T> &src)
      {
        dst.insert(dst.end(), src.begin(), src.end());
      }

      std::pair<dftfe::uInt, dftfe::uInt>
      getLocallyOwnedRange(const MPI_Comm   &mpiComm,
                           const dftfe::uInt myProcRank,
                           const dftfe::uInt nProcs,
                           const dftfe::uInt nLocalPoints)
      {
        std::vector<dftfe::uInt> numPointsInProcs(nProcs, 0);
        std::fill(numPointsInProcs.begin(), numPointsInProcs.end(), 0);
        numPointsInProcs[myProcRank] = nLocalPoints;
        MPI_Allreduce(MPI_IN_PLACE,
                      &numPointsInProcs[0],
                      nProcs,
                      dftfe::dataTypes::mpi_type_id(&numPointsInProcs[0]),
                      MPI_SUM,
                      mpiComm);

        dftfe::uInt locallyOwnedStart = 0, locallyOwnedEnd = 0;

        for (dftfe::uInt iProc = 0; iProc < myProcRank; iProc++)
          {
            locallyOwnedStart += (dftfe::uInt)numPointsInProcs[iProc];
          }

        locallyOwnedEnd = locallyOwnedStart + numPointsInProcs[myProcRank];
        return (std::make_pair(locallyOwnedStart, locallyOwnedEnd));
      }

      template <dftfe::uInt dim>
      void
      getProcBoundingBox(std::vector<std::shared_ptr<const Cell<dim>>> &cells,
                         std::vector<double> &lowerLeft,
                         std::vector<double> &upperRight)
      {
        lowerLeft.resize(dim);
        upperRight.resize(dim);
        const dftfe::uInt nCells = cells.size();
        // First index is dimension and second index is cell Id
        // For each cell store both the lower left and upper right
        // limit in each dimension
        std::vector<std::vector<double>> cellsLowerLeft(
          dim, std::vector<double>(nCells));
        std::vector<std::vector<double>> cellsUpperRight(
          dim, std::vector<double>(nCells));
        for (dftfe::uInt iCell = 0; iCell < nCells; ++iCell)
          {
            auto boundingBox = cells[iCell]->getBoundingBox();
            for (dftfe::uInt iDim = 0; iDim < dim; ++iDim)
              {
                cellsLowerLeft[iDim][iCell]  = boundingBox.first[iDim];
                cellsUpperRight[iDim][iCell] = boundingBox.second[iDim];
              }
          }

        // sort the cellLimits
        for (dftfe::uInt iDim = 0; iDim < dim; ++iDim)
          {
            std::sort(cellsLowerLeft[iDim].begin(), cellsLowerLeft[iDim].end());
            std::sort(cellsUpperRight[iDim].begin(),
                      cellsUpperRight[iDim].end());
            lowerLeft[iDim]  = cellsLowerLeft[iDim][0];
            upperRight[iDim] = cellsUpperRight[iDim][nCells - 1];
          }
      }


      void
      getAllProcsBoundingBoxes(const std::vector<double> &procLowerLeft,
                               const std::vector<double> &procUpperRight,
                               const dftfe::uInt          myProcRank,
                               const dftfe::uInt          nProcs,
                               const MPI_Comm            &mpiComm,
                               std::vector<double>       &allProcsBoundingBoxes)
      {
        const dftfe::uInt dim = procLowerLeft.size();
        allProcsBoundingBoxes.resize(2 * dim * nProcs);
        std::fill(allProcsBoundingBoxes.begin(),
                  allProcsBoundingBoxes.end(),
                  0.0);

        for (dftfe::uInt j = 0; j < dim; j++)
          {
            allProcsBoundingBoxes[2 * dim * myProcRank + j] = procLowerLeft[j];
            allProcsBoundingBoxes[2 * dim * myProcRank + dim + j] =
              procUpperRight[j];
          }

        MPI_Allreduce(MPI_IN_PLACE,
                      &allProcsBoundingBoxes[0],
                      2 * dim * nProcs,
                      MPI_DOUBLE,
                      MPI_SUM,
                      mpiComm);
      }


      template <dftfe::uInt dim, dftfe::uInt M>
      void
      pointsToCell(std::vector<std::shared_ptr<const Cell<dim>>> &srcCells,
                   const std::vector<std::vector<double>>        &targetPts,
                   std::vector<std::vector<dftfe::uInt>>         &cellFoundIds,
                   std::vector<std::vector<double>> &cellRealCoords,
                   std::vector<bool>                &pointsFound,
                   const double                      paramCoordsTol)
      {
        RTreePoint<dim, M> rTreePoint(targetPts);
        const dftfe::uInt  numCells = srcCells.size();
        pointsFound.resize(targetPts.size());
        std::fill(pointsFound.begin(), pointsFound.end(), false);
        cellFoundIds.resize(numCells, std::vector<dftfe::uInt>(0));
        cellRealCoords.resize(numCells, std::vector<double>(0));
        for (dftfe::uInt iCell = 0; iCell < numCells; iCell++)
          {
            auto bbCell = srcCells[iCell]->getBoundingBox();
            auto targetPointList =
              rTreePoint.getPointIdsInsideBox(bbCell.first, bbCell.second);

            for (dftfe::uInt iPoint = 0; iPoint < targetPointList.size();
                 iPoint++)
              {
                dftfe::uInt pointIndex = targetPointList[iPoint];
                if (!pointsFound[pointIndex])
                  {
                    //                    auto paramPoint =
                    //                    srcCells[iCell]->getParametricPoint(targetPts[pointIndex]);
                    bool pointInside =
                      srcCells[iCell]->isPointInside(targetPts[pointIndex],
                                                     paramCoordsTol);
                    //                    for( dftfe::uInt j = 0 ; j <dim; j++)
                    //                      {
                    //                        if((paramPoint[j] <
                    //                        -paramCoordsTol) || (paramPoint[j]
                    //                        > 1.0 + paramCoordsTol))
                    //                          {
                    //                            pointInside = false;
                    //                          }
                    //                      }
                    if (pointInside)
                      {
                        pointsFound[pointIndex] = true;
                        for (dftfe::uInt iDim = 0; iDim < dim; iDim++)
                          {
                            cellRealCoords[iCell].push_back(
                              targetPts[pointIndex][iDim]);
                          }
                        cellFoundIds[iCell].push_back(pointIndex);
                      }
                  }
              }
          }
      }


      template <dftfe::uInt dim, dftfe::uInt M>
      void
      getTargetPointsToSend(
        const std::vector<std::shared_ptr<const Cell<dim>>> &srcCells,
        const std::vector<dftfe::uInt>         &nonLocalPointLocalIds,
        const std::vector<std::vector<double>> &nonLocalPointCoordinates,
        const std::vector<double>              &allProcsBoundingBoxes,
        const dftfe::uInt                       locallyOwnedStart,
        const dftfe::uInt                       myProcRank,
        const dftfe::uInt                       nProcs,
        std::vector<dftfe::uInt>               &sendToProcIds,
        std::vector<std::vector<dftfe::uInt>>  &sendToPointsGlobalIds,
        std::vector<std::vector<double>>       &sendToPointsCoords)
      {
        sendToProcIds.resize(0);
        sendToPointsGlobalIds.resize(0, std::vector<dftfe::uInt>(0));
        sendToPointsCoords.resize(0, std::vector<double>(0));

        RTreePoint<dim, M> rTree(nonLocalPointCoordinates);
        for (dftfe::uInt iProc = 0; iProc < nProcs; iProc++)
          {
            if (iProc != myProcRank)
              {
                std::vector<double> llProc(dim, 0.0);
                std::vector<double> urProc(dim, 0.0);
                for (dftfe::uInt iDim = 0; iDim < dim; iDim++)
                  {
                    llProc[iDim] =
                      allProcsBoundingBoxes[2 * dim * iProc + iDim];
                    urProc[iDim] =
                      allProcsBoundingBoxes[2 * dim * iProc + dim + iDim];
                  }
                auto targetPointList =
                  rTree.getPointIdsInsideBox(llProc, urProc);

                dftfe::uInt numTargetPointsToSend = targetPointList.size();
                if (numTargetPointsToSend > 0)
                  {
                    std::vector<dftfe::uInt> globalIds(numTargetPointsToSend,
                                                       -1);
                    sendToProcIds.push_back(iProc);
                    std::vector<double> pointCoordinates(0);
                    for (dftfe::uInt iPoint = 0;
                         iPoint < targetPointList.size();
                         iPoint++)
                      {
                        dftfe::uInt pointIndex = targetPointList[iPoint];

                        appendToVec(pointCoordinates,
                                    nonLocalPointCoordinates[pointIndex]);
                        globalIds[iPoint] =
                          locallyOwnedStart +
                          nonLocalPointLocalIds[targetPointList[iPoint]];
                      }
                    // also have to send the coordinates and the indices.
                    sendToPointsGlobalIds.push_back(globalIds);
                    sendToPointsCoords.push_back(pointCoordinates);
                  }
              }
          }
      }

      template <dftfe::uInt dim>
      void
      receivePoints(
        const std::vector<dftfe::uInt>              &sendToProcIds,
        const std::vector<std::vector<dftfe::uInt>> &sendToPointsGlobalIds,
        const std::vector<std::vector<double>>      &sendToPointsCoords,
        std::vector<dftfe::uInt>                    &receivedPointsGlobalIds,
        std::vector<std::vector<double>>            &receivedPointsCoords,
        dftfe::uInt                                  verbosity,
        const MPI_Comm                              &mpiComm)
      {
        int thisRankId;
        MPI_Comm_rank(mpiComm, &thisRankId);
        dftfe::utils::mpi::MPIRequestersNBX mpiRequestersNBX(sendToProcIds,
                                                             mpiComm);
        std::vector<dftfe::uInt>            receiveFromProcIds =
          mpiRequestersNBX.getRequestingRankIds();

        dftfe::uInt numMaxProcsSendTo = sendToProcIds.size();
        MPI_Allreduce(MPI_IN_PLACE,
                      &numMaxProcsSendTo,
                      1,
                      dftfe::dataTypes::mpi_type_id(&numMaxProcsSendTo),
                      MPI_MAX,
                      mpiComm);

        dftfe::uInt numMaxProcsReceiveFrom = receiveFromProcIds.size();
        MPI_Allreduce(MPI_IN_PLACE,
                      &numMaxProcsReceiveFrom,
                      1,
                      dftfe::dataTypes::mpi_type_id(&numMaxProcsReceiveFrom),
                      MPI_MAX,
                      mpiComm);

        if ((thisRankId == 0) && (verbosity > 2))
          {
            std::cout << " Max number of procs to send to = "
                      << numMaxProcsSendTo << "\n";
            std::cout << " Max number of procs to receive from = "
                      << numMaxProcsReceiveFrom << "\n";
          }



        std::vector<std::vector<double>> receivedPointsCoordsProcWise(
          receiveFromProcIds.size(), std::vector<double>(0));
        std::vector<dftfe::uInt> numPointsReceived(receiveFromProcIds.size(),
                                                   -1);

        std::vector<dftfe::uInt> numPointsToSend(sendToPointsGlobalIds.size(),
                                                 -1);
        std::vector<MPI_Request> sendRequests(sendToProcIds.size());
        std::vector<MPI_Status>  sendStatuses(sendToProcIds.size());
        std::vector<MPI_Request> recvRequests(receiveFromProcIds.size());
        std::vector<MPI_Status>  recvStatuses(receiveFromProcIds.size());
        const dftfe::Int         tag = static_cast<dftfe::Int>(
          dftfe::utils::mpi::MPITags::MPI_P2P_PATTERN_TAG);
        for (dftfe::uInt i = 0; i < sendToProcIds.size(); ++i)
          {
            dftfe::uInt procId = sendToProcIds[i];
            numPointsToSend[i] = sendToPointsGlobalIds[i].size();
            MPI_Isend(&numPointsToSend[i],
                      1,
                      dftfe::dataTypes::mpi_type_id(&numPointsToSend[i]),
                      procId,
                      procId, // setting the tag to procId
                      mpiComm,
                      &sendRequests[i]);
          }

        for (dftfe::uInt i = 0; i < receiveFromProcIds.size(); ++i)
          {
            dftfe::uInt procId = receiveFromProcIds[i];
            MPI_Irecv(&numPointsReceived[i],
                      1,
                      dftfe::dataTypes::mpi_type_id(&numPointsReceived[i]),
                      procId,
                      thisRankId, // the tag is set to the receiving id
                      mpiComm,
                      &recvRequests[i]);
          }


        if (sendRequests.size() > 0)
          {
            dftfe::Int  err    = MPI_Waitall(sendToProcIds.size(),
                                         sendRequests.data(),
                                         sendStatuses.data());
            std::string errMsg = "Error occured while using MPI_Waitall. "
                                 "Error code: " +
                                 std::to_string(err);
            throwException(err == MPI_SUCCESS, errMsg);
          }

        if (recvRequests.size() > 0)
          {
            dftfe::Int  err    = MPI_Waitall(receiveFromProcIds.size(),
                                         recvRequests.data(),
                                         recvStatuses.data());
            std::string errMsg = "Error occured while using MPI_Waitall. "
                                 "Error code: " +
                                 std::to_string(err);
            throwException(err == MPI_SUCCESS, errMsg);
          }

        const dftfe::uInt numTotalPointsReceived =
          std::accumulate(numPointsReceived.begin(),
                          numPointsReceived.end(),
                          0);
        receivedPointsGlobalIds.resize(numTotalPointsReceived, -1);

        for (dftfe::uInt i = 0; i < sendToProcIds.size(); ++i)
          {
            dftfe::uInt procId        = sendToProcIds[i];
            dftfe::uInt nPointsToSend = sendToPointsGlobalIds[i].size();
            MPI_Isend(&sendToPointsGlobalIds[i][0],
                      nPointsToSend,
                      dftfe::dataTypes::mpi_type_id(
                        &sendToPointsGlobalIds[i][0]),
                      procId,
                      tag,
                      mpiComm,
                      &sendRequests[i]);
          }

        dftfe::uInt offset = 0;
        for (dftfe::uInt i = 0; i < receiveFromProcIds.size(); ++i)
          {
            dftfe::uInt procId = receiveFromProcIds[i];
            MPI_Irecv(&receivedPointsGlobalIds[offset],
                      numPointsReceived[i],
                      dftfe::dataTypes::mpi_type_id(
                        &receivedPointsGlobalIds[offset]),
                      procId,
                      tag,
                      mpiComm,
                      &recvRequests[i]);

            offset += numPointsReceived[i];
          }


        if (sendRequests.size() > 0)
          {
            dftfe::Int  err    = MPI_Waitall(sendToProcIds.size(),
                                         sendRequests.data(),
                                         sendStatuses.data());
            std::string errMsg = "Error occured while using MPI_Waitall. "
                                 "Error code: " +
                                 std::to_string(err);
            throwException(err == MPI_SUCCESS, errMsg);
          }

        if (recvRequests.size() > 0)
          {
            dftfe::Int  err    = MPI_Waitall(receiveFromProcIds.size(),
                                         recvRequests.data(),
                                         recvStatuses.data());
            std::string errMsg = "Error occured while using MPI_Waitall. "
                                 "Error code: " +
                                 std::to_string(err);
            throwException(err == MPI_SUCCESS, errMsg);
          }

        for (dftfe::uInt i = 0; i < sendToProcIds.size(); ++i)
          {
            dftfe::uInt procId        = sendToProcIds[i];
            dftfe::uInt nPointsToSend = sendToPointsGlobalIds[i].size();
            MPI_Isend(&sendToPointsCoords[i][0],
                      nPointsToSend * dim,
                      MPI_DOUBLE,
                      procId,
                      tag,
                      mpiComm,
                      &sendRequests[i]);
          }

        for (dftfe::uInt i = 0; i < receiveFromProcIds.size(); ++i)
          {
            dftfe::uInt procId = receiveFromProcIds[i];
            receivedPointsCoordsProcWise[i].resize(numPointsReceived[i] * dim);
            MPI_Irecv(&receivedPointsCoordsProcWise[i][0],
                      numPointsReceived[i] * dim,
                      MPI_DOUBLE,
                      procId,
                      tag,
                      mpiComm,
                      &recvRequests[i]);
          }

        if (sendRequests.size() > 0)
          {
            dftfe::Int  err    = MPI_Waitall(sendToProcIds.size(),
                                         sendRequests.data(),
                                         sendStatuses.data());
            std::string errMsg = "Error occured while using MPI_Waitall. "
                                 "Error code: " +
                                 std::to_string(err);
            throwException(err == MPI_SUCCESS, errMsg);
          }

        if (recvRequests.size() > 0)
          {
            dftfe::Int  err    = MPI_Waitall(receiveFromProcIds.size(),
                                         recvRequests.data(),
                                         recvStatuses.data());
            std::string errMsg = "Error occured while using MPI_Waitall. "
                                 "Error code: " +
                                 std::to_string(err);
            throwException(err == MPI_SUCCESS, errMsg);
          }

        receivedPointsCoords.resize(numTotalPointsReceived,
                                    std::vector<double>(dim, 0.0));
        dftfe::uInt count = 0;
        for (dftfe::uInt i = 0; i < receiveFromProcIds.size(); i++)
          {
            std::vector<double> &pointsCoordProc =
              receivedPointsCoordsProcWise[i];
            for (dftfe::uInt iPoint = 0; iPoint < numPointsReceived[i];
                 iPoint++)
              {
                for (dftfe::uInt iDim = 0; iDim < dim; iDim++)
                  {
                    receivedPointsCoords[count][iDim] =
                      pointsCoordProc[iPoint * dim + iDim];
                  }
                count++;
              }
          }
      }
    } // namespace

    template <dftfe::uInt dim, dftfe::uInt M>
    MapPointsToCells<dim, M>::MapPointsToCells(const dftfe::uInt verbosity,
                                               const MPI_Comm   &mpiComm)
      : d_mpiComm(mpiComm)
    {
      d_verbosity = verbosity;
      MPI_Comm_rank(d_mpiComm, &d_thisRank);
      MPI_Comm_size(d_mpiComm, &d_numMPIRank);
    }

    template <dftfe::uInt dim, dftfe::uInt M>
    void
    MapPointsToCells<dim, M>::init(
      std::vector<std::shared_ptr<const Cell<dim>>> srcCells,
      const std::vector<std::vector<double>>       &targetPts,
      std::vector<std::vector<double>>             &mapCellsToRealCoordinates,
      std::vector<std::vector<dftfe::uInt>>        &mapCellLocalToProcLocal,
      std::pair<dftfe::uInt, dftfe::uInt>          &locallyOwnedRange,
      std::vector<dftfe::uInt>                     &ghostGlobalIds,
      const double                                  paramCoordsTol)
    {
      MPI_Barrier(d_mpiComm);
      double      startComp = MPI_Wtime();
      dftfe::uInt numCells  = srcCells.size();
      dftfe::uInt numPoints = targetPts.size();
      mapCellLocalToProcLocal.resize(numCells, std::vector<dftfe::uInt>(0));
      mapCellsToRealCoordinates.resize(numCells, std::vector<double>(0));

      std::vector<std::vector<dftfe::uInt>> mapCellLocalToGlobal;
      mapCellLocalToGlobal.resize(numCells, std::vector<dftfe::uInt>(0));

      std::vector<dftfe::uInt> numLocalPointsInCell(numCells, 0);

      // Create the bounding box for each process
      // and share it across to all the processors
      // TODO what to do when there are no cells
      std::vector<double> procLowerLeft(dim, 0.0);
      std::vector<double> procUpperRight(dim, 0.0);
      if (numCells > 0)
        {
          getProcBoundingBox<dim>(srcCells, procLowerLeft, procUpperRight);
        }

      // get bounding boxes of all the processors
      std::vector<double> allProcsBoundingBoxes(0);
      getAllProcsBoundingBoxes(procLowerLeft,
                               procUpperRight,
                               d_thisRank,
                               d_numMPIRank,
                               d_mpiComm,
                               allProcsBoundingBoxes);

      locallyOwnedRange =
        getLocallyOwnedRange(d_mpiComm, d_thisRank, d_numMPIRank, numPoints);
      const dftfe::uInt locallyOwnedStart = locallyOwnedRange.first;
      const dftfe::uInt locallyOwnedEnd   = locallyOwnedRange.second;

      std::vector<bool> pointsFoundLocally(numPoints, false);
      std::vector<std::vector<dftfe::uInt>> cellLocalFoundIds;
      std::vector<std::vector<double>>      cellLocalFoundRealCoords;
      // pointsToCell finds the points from the target pts that lie inside each
      // cell
      pointsToCell<dim, M>(srcCells,
                           targetPts,
                           cellLocalFoundIds,
                           cellLocalFoundRealCoords,
                           pointsFoundLocally,
                           paramCoordsTol);

      dftfe::uInt numLocallyFoundPoints = 0;
      for (dftfe::uInt iCell = 0; iCell < numCells; iCell++)
        {
          numLocalPointsInCell[iCell] = cellLocalFoundIds[iCell].size();

          appendToVec(mapCellLocalToProcLocal[iCell], cellLocalFoundIds[iCell]);

          // initialSize should be zero
          dftfe::uInt initialSize = mapCellLocalToGlobal[iCell].size();
          dftfe::uInt finalSize = initialSize + cellLocalFoundIds[iCell].size();
          mapCellLocalToGlobal[iCell].resize(finalSize);
          for (dftfe::uInt indexVal = initialSize; indexVal < finalSize;
               indexVal++)
            {
              mapCellLocalToGlobal[iCell][indexVal] =
                cellLocalFoundIds[iCell][indexVal - initialSize] +
                locallyOwnedStart;
              numLocallyFoundPoints++;
            }

          appendToVec(mapCellsToRealCoordinates[iCell],
                      cellLocalFoundRealCoords[iCell]);
        }

      MPI_Barrier(d_mpiComm);
      double endLocalComp = MPI_Wtime();
      // get the points that are not found locally
      std::vector<dftfe::uInt>         nonLocalPointLocalIds(0);
      std::vector<std::vector<double>> nonLocalPointCoordinates(0);
      for (dftfe::uInt iPoint = 0; iPoint < numPoints; iPoint++)
        {
          if (!pointsFoundLocally[iPoint])
            {
              nonLocalPointLocalIds.push_back(iPoint);
              nonLocalPointCoordinates.push_back(
                targetPts[iPoint]); // TODO will this work ?
            }
        }

      std::vector<dftfe::uInt>              sendToProcIds(0);
      std::vector<std::vector<dftfe::uInt>> sendToPointsGlobalIds;
      std::vector<std::vector<double>>      sendToPointsCoords;

      // This function takes the points not found locally and find all the
      // bounding boxes inside which any of the non-local points lie.
      // This tells the to which processors the points have to be sent
      getTargetPointsToSend<dim, M>(srcCells,
                                    nonLocalPointLocalIds,
                                    nonLocalPointCoordinates,
                                    allProcsBoundingBoxes,
                                    locallyOwnedStart,
                                    d_thisRank,
                                    d_numMPIRank,
                                    sendToProcIds,
                                    sendToPointsGlobalIds,
                                    sendToPointsCoords);

      std::vector<dftfe::uInt>         receivedPointsGlobalIds;
      std::vector<std::vector<double>> receivedPointsCoords;

      // Receive points from other points that lie inside the bounding box
      // of this processor
      receivePoints<dim>(sendToProcIds,
                         sendToPointsGlobalIds,
                         sendToPointsCoords,
                         receivedPointsGlobalIds,
                         receivedPointsCoords,
                         d_verbosity,
                         d_mpiComm);

      MPI_Barrier(d_mpiComm);
      double endReceive = MPI_Wtime();

      std::cout << std::flush;
      MPI_Barrier(d_mpiComm);


      dftfe::uInt numTotalPointsReceived = receivedPointsCoords.size();
      std::vector<std::vector<dftfe::uInt>> cellReceivedPointsFoundIds;
      std::vector<std::vector<double>>      cellReceivedPointsFoundRealCoords;
      std::vector<bool> receivedPointsFound(numTotalPointsReceived, false);

      // Search through the points received from other processors to find which
      // of them lie within the cells of this processor
      pointsToCell<dim, M>(srcCells,
                           receivedPointsCoords,
                           cellReceivedPointsFoundIds,
                           cellReceivedPointsFoundRealCoords,
                           receivedPointsFound,
                           paramCoordsTol);


      std::cout << std::flush;
      MPI_Barrier(d_mpiComm);
      double endNonLocalComp = MPI_Wtime();

      ghostGlobalIds.resize(0);
      std::set<dftfe::uInt> ghostGlobalIdsSet;
      for (dftfe::uInt iCell = 0; iCell < numCells; iCell++)
        {
          const dftfe::uInt numPointsReceivedFound =
            cellReceivedPointsFoundIds[iCell].size();
          const dftfe::uInt mapCellLocalToGlobalCurrIndex =
            mapCellLocalToGlobal[iCell].size();
          mapCellLocalToGlobal[iCell].resize(mapCellLocalToGlobalCurrIndex +
                                             numPointsReceivedFound);
          for (dftfe::uInt i = 0; i < numPointsReceivedFound; ++i)
            {
              const dftfe::uInt pointIndex =
                cellReceivedPointsFoundIds[iCell][i];
              const dftfe::uInt globalId = receivedPointsGlobalIds[pointIndex];
              mapCellLocalToGlobal[iCell][mapCellLocalToGlobalCurrIndex + i] =
                globalId;

              ghostGlobalIdsSet.insert(globalId);
            }

          // append the list of points to each cell
          appendToVec(mapCellsToRealCoordinates[iCell],
                      cellReceivedPointsFoundRealCoords[iCell]);
        }

      MPI_Barrier(d_mpiComm);
      double endNonLocalVecComp = MPI_Wtime();

      OptimizedIndexSet<dftfe::uInt> ghostGlobalIdsOptIndexSet(
        ghostGlobalIdsSet);

      std::string errMsgInFindingPoint =
        "Error in finding ghost index in mapPointsToCells.cpp.";
      for (dftfe::uInt iCell = 0; iCell < numCells; iCell++)
        {
          const dftfe::uInt startId = numLocalPointsInCell[iCell];
          const dftfe::uInt endId   = mapCellLocalToGlobal[iCell].size();
          for (dftfe::uInt iPoint = startId; iPoint < endId; ++iPoint)
            {
              dftfe::uInt globalId = mapCellLocalToGlobal[iCell][iPoint];
              dftfe::uInt pos      = -1;
              bool        found    = true;
              ghostGlobalIdsOptIndexSet.getPosition(globalId, pos, found);
              mapCellLocalToProcLocal[iCell].push_back(numPoints + pos);
            }
        }

      dftfe::uInt ghostSetSize = ghostGlobalIdsSet.size();
      ghostGlobalIds.resize(ghostSetSize, -1);
      dftfe::uInt ghostIndex = 0;
      for (auto it = ghostGlobalIdsSet.begin(); it != ghostGlobalIdsSet.end();
           it++)
        {
          ghostGlobalIds[ghostIndex] = *it;

          ghostIndex++;
        }
      std::cout << std::flush;
      MPI_Barrier(d_mpiComm);

      double endCompAll = MPI_Wtime();

      dftfe::uInt numNonLocalPointsReceived = numTotalPointsReceived;
      MPI_Allreduce(MPI_IN_PLACE,
                    &numNonLocalPointsReceived,
                    1,
                    dftfe::dataTypes::mpi_type_id(&numNonLocalPointsReceived),
                    MPI_MAX,
                    d_mpiComm);

      if ((d_thisRank == 0) && (d_verbosity > 2))
        {
          std::cout << " Max number of non local pts received = "
                    << numNonLocalPointsReceived << "\n";
          std::cout << " Time taken for local pts = "
                    << endLocalComp - startComp << "\n";
          std::cout << " Time taken for transfer = "
                    << endReceive - endLocalComp << "\n";
          std::cout << " Time taken for non-local pts = "
                    << endNonLocalComp - endReceive << "\n";
          std::cout << " Time taken for non-local vec gen = "
                    << endNonLocalVecComp - endNonLocalComp << "\n";
          std::cout << " Time for remaining comp = "
                    << endCompAll - endNonLocalVecComp << "\n";
        }
    }
  } // end of namespace utils
} // end of namespace dftfe
