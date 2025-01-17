
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

      std::pair<global_size_type, global_size_type>
      getLocallyOwnedRange(const MPI_Comm &mpiComm,
                           const size_type myProcRank,
                           const size_type nProcs,
                           const size_type nLocalPoints)
      {
        std::vector<size_type> numPointsInProcs(nProcs, 0);
        std::fill(numPointsInProcs.begin(), numPointsInProcs.end(), 0);
        numPointsInProcs[myProcRank] = nLocalPoints;
        MPI_Allreduce(MPI_IN_PLACE,
                      &numPointsInProcs[0],
                      nProcs,
                      dftfe::dataTypes::mpi_type_id(&numPointsInProcs[0]),
                      MPI_SUM,
                      mpiComm);

        global_size_type locallyOwnedStart = 0, locallyOwnedEnd = 0;

        for (unsigned int iProc = 0; iProc < myProcRank; iProc++)
          {
            locallyOwnedStart += (global_size_type)numPointsInProcs[iProc];
          }

        locallyOwnedEnd = locallyOwnedStart + numPointsInProcs[myProcRank];
        return (std::make_pair(locallyOwnedStart, locallyOwnedEnd));
      }

      template <size_type dim>
      void
      getProcBoundingBox(std::vector<std::shared_ptr<const Cell<dim>>> &cells,
                         std::vector<double> &lowerLeft,
                         std::vector<double> &upperRight)
      {
        lowerLeft.resize(dim);
        upperRight.resize(dim);
        const size_type nCells = cells.size();
        // First index is dimension and second index is cell Id
        // For each cell store both the lower left and upper right
        // limit in each dimension
        std::vector<std::vector<double>> cellsLowerLeft(
          dim, std::vector<double>(nCells));
        std::vector<std::vector<double>> cellsUpperRight(
          dim, std::vector<double>(nCells));
        for (size_type iCell = 0; iCell < nCells; ++iCell)
          {
            auto boundingBox = cells[iCell]->getBoundingBox();
            for (size_type iDim = 0; iDim < dim; ++iDim)
              {
                cellsLowerLeft[iDim][iCell]  = boundingBox.first[iDim];
                cellsUpperRight[iDim][iCell] = boundingBox.second[iDim];
              }
          }

        // sort the cellLimits
        for (size_type iDim = 0; iDim < dim; ++iDim)
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
                               const size_type            myProcRank,
                               const size_type            nProcs,
                               const MPI_Comm &           mpiComm,
                               std::vector<double> &      allProcsBoundingBoxes)
      {
        const size_type dim = procLowerLeft.size();
        allProcsBoundingBoxes.resize(2 * dim * nProcs);
        std::fill(allProcsBoundingBoxes.begin(),
                  allProcsBoundingBoxes.end(),
                  0.0);

        for (unsigned int j = 0; j < dim; j++)
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


      template <size_type dim, size_type M>
      void
      pointsToCell(std::vector<std::shared_ptr<const Cell<dim>>> &srcCells,
                   const std::vector<std::vector<double>> &       targetPts,
                   std::vector<std::vector<size_type>> &          cellFoundIds,
                   std::vector<std::vector<double>> &cellRealCoords,
                   std::vector<bool> &               pointsFound,
                   const double                      paramCoordsTol)
      {
        RTreePoint<dim, M> rTreePoint(targetPts);
        const size_type    numCells = srcCells.size();
        pointsFound.resize(targetPts.size());
        std::fill(pointsFound.begin(), pointsFound.end(), false);
        cellFoundIds.resize(numCells, std::vector<size_type>(0));
        cellRealCoords.resize(numCells, std::vector<double>(0));
        for (size_type iCell = 0; iCell < numCells; iCell++)
          {
            auto bbCell = srcCells[iCell]->getBoundingBox();
            auto targetPointList =
              rTreePoint.getPointIdsInsideBox(bbCell.first, bbCell.second);

            for (size_type iPoint = 0; iPoint < targetPointList.size();
                 iPoint++)
              {
                size_type pointIndex = targetPointList[iPoint];
                if (!pointsFound[pointIndex])
                  {
                    //                    auto paramPoint =
                    //                    srcCells[iCell]->getParametricPoint(targetPts[pointIndex]);
                    bool pointInside =
                      srcCells[iCell]->isPointInside(targetPts[pointIndex],
                                                     paramCoordsTol);
                    //                    for( unsigned int j = 0 ; j <dim; j++)
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
                        for (size_type iDim = 0; iDim < dim; iDim++)
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


      template <size_type dim, size_type M>
      void
      getTargetPointsToSend(
        const std::vector<std::shared_ptr<const Cell<dim>>> &srcCells,
        const std::vector<size_type> &              nonLocalPointLocalIds,
        const std::vector<std::vector<double>> &    nonLocalPointCoordinates,
        const std::vector<double> &                 allProcsBoundingBoxes,
        const global_size_type                      locallyOwnedStart,
        const size_type                             myProcRank,
        const size_type                             nProcs,
        std::vector<size_type> &                    sendToProcIds,
        std::vector<std::vector<global_size_type>> &sendToPointsGlobalIds,
        std::vector<std::vector<double>> &          sendToPointsCoords)
      {
        sendToProcIds.resize(0);
        sendToPointsGlobalIds.resize(0, std::vector<global_size_type>(0));
        sendToPointsCoords.resize(0, std::vector<double>(0));

        RTreePoint<dim, M> rTree(nonLocalPointCoordinates);
        for (size_type iProc = 0; iProc < nProcs; iProc++)
          {
            if (iProc != myProcRank)
              {
                std::vector<double> llProc(dim, 0.0);
                std::vector<double> urProc(dim, 0.0);
                for (size_type iDim = 0; iDim < dim; iDim++)
                  {
                    llProc[iDim] =
                      allProcsBoundingBoxes[2 * dim * iProc + iDim];
                    urProc[iDim] =
                      allProcsBoundingBoxes[2 * dim * iProc + dim + iDim];
                  }
                auto targetPointList =
                  rTree.getPointIdsInsideBox(llProc, urProc);

                size_type numTargetPointsToSend = targetPointList.size();
                if (numTargetPointsToSend > 0)
                  {
                    std::vector<global_size_type> globalIds(
                      numTargetPointsToSend, -1);
                    sendToProcIds.push_back(iProc);
                    std::vector<double> pointCoordinates(0);
                    for (size_type iPoint = 0; iPoint < targetPointList.size();
                         iPoint++)
                      {
                        size_type pointIndex = targetPointList[iPoint];

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

      template <size_type dim>
      void
      receivePoints(
        const std::vector<size_type> &                    sendToProcIds,
        const std::vector<std::vector<global_size_type>> &sendToPointsGlobalIds,
        const std::vector<std::vector<double>> &          sendToPointsCoords,
        std::vector<global_size_type> &   receivedPointsGlobalIds,
        std::vector<std::vector<double>> &receivedPointsCoords,
        unsigned int                      verbosity,
        const MPI_Comm &                  mpiComm)
      {
        int thisRankId;
        MPI_Comm_rank(mpiComm, &thisRankId);
        dftfe::utils::mpi::MPIRequestersNBX mpiRequestersNBX(sendToProcIds,
                                                             mpiComm);
        std::vector<size_type>              receiveFromProcIds =
          mpiRequestersNBX.getRequestingRankIds();

        size_type numMaxProcsSendTo = sendToProcIds.size();
        MPI_Allreduce(MPI_IN_PLACE,
                      &numMaxProcsSendTo,
                      1,
                      dftfe::dataTypes::mpi_type_id(&numMaxProcsSendTo),
                      MPI_MAX,
                      mpiComm);

        size_type numMaxProcsReceiveFrom = receiveFromProcIds.size();
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
        std::vector<size_type> numPointsReceived(receiveFromProcIds.size(), -1);

        std::vector<size_type>   numPointsToSend(sendToPointsGlobalIds.size(),
                                               -1);
        std::vector<MPI_Request> sendRequests(sendToProcIds.size());
        std::vector<MPI_Status>  sendStatuses(sendToProcIds.size());
        std::vector<MPI_Request> recvRequests(receiveFromProcIds.size());
        std::vector<MPI_Status>  recvStatuses(receiveFromProcIds.size());
        const int                tag =
          static_cast<int>(dftfe::utils::mpi::MPITags::MPI_P2P_PATTERN_TAG);
        for (size_type i = 0; i < sendToProcIds.size(); ++i)
          {
            size_type procId   = sendToProcIds[i];
            numPointsToSend[i] = sendToPointsGlobalIds[i].size();
            MPI_Isend(&numPointsToSend[i],
                      1,
                      //                            MPI_UNSIGNED,
                      dftfe::dataTypes::mpi_type_id(&numPointsToSend[i]),
                      procId,
                      procId, // setting the tag to procId
                      mpiComm,
                      &sendRequests[i]);
          }

        for (size_type i = 0; i < receiveFromProcIds.size(); ++i)
          {
            size_type procId = receiveFromProcIds[i];
            MPI_Irecv(&numPointsReceived[i],
                      1,
                      //                            MPI_UNSIGNED,
                      dftfe::dataTypes::mpi_type_id(&numPointsReceived[i]),
                      procId,
                      thisRankId, // the tag is set to the receiving id
                      mpiComm,
                      &recvRequests[i]);
          }


        if (sendRequests.size() > 0)
          {
            int         err    = MPI_Waitall(sendToProcIds.size(),
                                  sendRequests.data(),
                                  sendStatuses.data());
            std::string errMsg = "Error occured while using MPI_Waitall. "
                                 "Error code: " +
                                 std::to_string(err);
            throwException(err == MPI_SUCCESS, errMsg);
          }

        if (recvRequests.size() > 0)
          {
            int         err    = MPI_Waitall(receiveFromProcIds.size(),
                                  recvRequests.data(),
                                  recvStatuses.data());
            std::string errMsg = "Error occured while using MPI_Waitall. "
                                 "Error code: " +
                                 std::to_string(err);
            throwException(err == MPI_SUCCESS, errMsg);
          }

        const size_type numTotalPointsReceived =
          std::accumulate(numPointsReceived.begin(),
                          numPointsReceived.end(),
                          0);
        receivedPointsGlobalIds.resize(numTotalPointsReceived, -1);

        for (size_type i = 0; i < sendToProcIds.size(); ++i)
          {
            size_type procId        = sendToProcIds[i];
            size_type nPointsToSend = sendToPointsGlobalIds[i].size();
            MPI_Isend(&sendToPointsGlobalIds[i][0],
                      nPointsToSend,
                      dftfe::dataTypes::mpi_type_id(
                        &sendToPointsGlobalIds[i][0]),
                      procId,
                      tag,
                      mpiComm,
                      &sendRequests[i]);
          }

        size_type offset = 0;
        for (size_type i = 0; i < receiveFromProcIds.size(); ++i)
          {
            size_type procId = receiveFromProcIds[i];
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
            int         err    = MPI_Waitall(sendToProcIds.size(),
                                  sendRequests.data(),
                                  sendStatuses.data());
            std::string errMsg = "Error occured while using MPI_Waitall. "
                                 "Error code: " +
                                 std::to_string(err);
            throwException(err == MPI_SUCCESS, errMsg);
          }

        if (recvRequests.size() > 0)
          {
            int         err    = MPI_Waitall(receiveFromProcIds.size(),
                                  recvRequests.data(),
                                  recvStatuses.data());
            std::string errMsg = "Error occured while using MPI_Waitall. "
                                 "Error code: " +
                                 std::to_string(err);
            throwException(err == MPI_SUCCESS, errMsg);
          }

        for (size_type i = 0; i < sendToProcIds.size(); ++i)
          {
            size_type procId        = sendToProcIds[i];
            size_type nPointsToSend = sendToPointsGlobalIds[i].size();
            MPI_Isend(&sendToPointsCoords[i][0],
                      nPointsToSend * dim,
                      MPI_DOUBLE,
                      procId,
                      tag,
                      mpiComm,
                      &sendRequests[i]);
          }

        for (size_type i = 0; i < receiveFromProcIds.size(); ++i)
          {
            size_type procId = receiveFromProcIds[i];
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
            int         err    = MPI_Waitall(sendToProcIds.size(),
                                  sendRequests.data(),
                                  sendStatuses.data());
            std::string errMsg = "Error occured while using MPI_Waitall. "
                                 "Error code: " +
                                 std::to_string(err);
            throwException(err == MPI_SUCCESS, errMsg);
          }

        if (recvRequests.size() > 0)
          {
            int         err    = MPI_Waitall(receiveFromProcIds.size(),
                                  recvRequests.data(),
                                  recvStatuses.data());
            std::string errMsg = "Error occured while using MPI_Waitall. "
                                 "Error code: " +
                                 std::to_string(err);
            throwException(err == MPI_SUCCESS, errMsg);
          }

        receivedPointsCoords.resize(numTotalPointsReceived,
                                    std::vector<double>(dim, 0.0));
        size_type count = 0;
        for (size_type i = 0; i < receiveFromProcIds.size(); i++)
          {
            std::vector<double> &pointsCoordProc =
              receivedPointsCoordsProcWise[i];
            for (size_type iPoint = 0; iPoint < numPointsReceived[i]; iPoint++)
              {
                for (size_type iDim = 0; iDim < dim; iDim++)
                  {
                    receivedPointsCoords[count][iDim] =
                      pointsCoordProc[iPoint * dim + iDim];
                  }
                count++;
              }
          }
      }
    } // namespace

    template <size_type dim, size_type M>
    MapPointsToCells<dim, M>::MapPointsToCells(const unsigned int verbosity,
                                               const MPI_Comm &   mpiComm)
      : d_mpiComm(mpiComm)
    {
      d_verbosity = verbosity;
      MPI_Comm_rank(d_mpiComm, &d_thisRank);
      MPI_Comm_size(d_mpiComm, &d_numMPIRank);
    }

    template <size_type dim, size_type M>
    void
    MapPointsToCells<dim, M>::init(
      std::vector<std::shared_ptr<const Cell<dim>>>  srcCells,
      const std::vector<std::vector<double>> &       targetPts,
      std::vector<std::vector<double>> &             mapCellsToRealCoordinates,
      std::vector<std::vector<size_type>> &          mapCellLocalToProcLocal,
      std::pair<global_size_type, global_size_type> &locallyOwnedRange,
      std::vector<global_size_type> &                ghostGlobalIds,
      const double                                   paramCoordsTol)
    {
      MPI_Barrier(d_mpiComm);
      double    startComp = MPI_Wtime();
      size_type numCells  = srcCells.size();
      size_type numPoints = targetPts.size();
      mapCellLocalToProcLocal.resize(numCells, std::vector<size_type>(0));
      mapCellsToRealCoordinates.resize(numCells, std::vector<double>(0));

      std::vector<std::vector<global_size_type>> mapCellLocalToGlobal;
      mapCellLocalToGlobal.resize(numCells, std::vector<global_size_type>(0));

      std::vector<size_type> numLocalPointsInCell(numCells, 0);

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
      const global_size_type locallyOwnedStart = locallyOwnedRange.first;
      const global_size_type locallyOwnedEnd   = locallyOwnedRange.second;

      std::vector<bool>                   pointsFoundLocally(numPoints, false);
      std::vector<std::vector<size_type>> cellLocalFoundIds;
      std::vector<std::vector<double>>    cellLocalFoundRealCoords;
      // pointsToCell finds the points from the target pts that lie inside each
      // cell
      pointsToCell<dim, M>(srcCells,
                           targetPts,
                           cellLocalFoundIds,
                           cellLocalFoundRealCoords,
                           pointsFoundLocally,
                           paramCoordsTol);

      size_type numLocallyFoundPoints = 0;
      for (size_type iCell = 0; iCell < numCells; iCell++)
        {
          numLocalPointsInCell[iCell] = cellLocalFoundIds[iCell].size();

          appendToVec(mapCellLocalToProcLocal[iCell], cellLocalFoundIds[iCell]);

          // initialSize should be zero
          size_type initialSize = mapCellLocalToGlobal[iCell].size();
          size_type finalSize   = initialSize + cellLocalFoundIds[iCell].size();
          mapCellLocalToGlobal[iCell].resize(finalSize);
          for (size_type indexVal = initialSize; indexVal < finalSize;
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
      std::vector<size_type>           nonLocalPointLocalIds(0);
      std::vector<std::vector<double>> nonLocalPointCoordinates(0);
      for (size_type iPoint = 0; iPoint < numPoints; iPoint++)
        {
          if (!pointsFoundLocally[iPoint])
            {
              nonLocalPointLocalIds.push_back(iPoint);
              nonLocalPointCoordinates.push_back(
                targetPts[iPoint]); // TODO will this work ?
            }
        }

      std::vector<size_type>                     sendToProcIds(0);
      std::vector<std::vector<global_size_type>> sendToPointsGlobalIds;
      std::vector<std::vector<double>>           sendToPointsCoords;

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

      std::vector<global_size_type>    receivedPointsGlobalIds;
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


      size_type numTotalPointsReceived = receivedPointsCoords.size();
      std::vector<std::vector<size_type>> cellReceivedPointsFoundIds;
      std::vector<std::vector<double>>    cellReceivedPointsFoundRealCoords;
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
      std::set<global_size_type> ghostGlobalIdsSet;
      for (size_type iCell = 0; iCell < numCells; iCell++)
        {
          const size_type numPointsReceivedFound =
            cellReceivedPointsFoundIds[iCell].size();
          const size_type mapCellLocalToGlobalCurrIndex =
            mapCellLocalToGlobal[iCell].size();
          mapCellLocalToGlobal[iCell].resize(mapCellLocalToGlobalCurrIndex +
                                             numPointsReceivedFound);
          for (size_type i = 0; i < numPointsReceivedFound; ++i)
            {
              const size_type pointIndex = cellReceivedPointsFoundIds[iCell][i];
              const global_size_type globalId =
                receivedPointsGlobalIds[pointIndex];
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

      OptimizedIndexSet<global_size_type> ghostGlobalIdsOptIndexSet(
        ghostGlobalIdsSet);

      std::string errMsgInFindingPoint =
        "Error in finding ghost index in mapPointsToCells.cpp.";
      for (size_type iCell = 0; iCell < numCells; iCell++)
        {
          const size_type startId = numLocalPointsInCell[iCell];
          const size_type endId   = mapCellLocalToGlobal[iCell].size();
          for (size_type iPoint = startId; iPoint < endId; ++iPoint)
            {
              size_type globalId = mapCellLocalToGlobal[iCell][iPoint];
              size_type pos      = -1;
              bool      found    = true;
              ghostGlobalIdsOptIndexSet.getPosition(globalId, pos, found);
              mapCellLocalToProcLocal[iCell].push_back(numPoints + pos);
            }
        }

      size_type ghostSetSize = ghostGlobalIdsSet.size();
      ghostGlobalIds.resize(ghostSetSize, -1);
      size_type ghostIndex = 0;
      for (auto it = ghostGlobalIdsSet.begin(); it != ghostGlobalIdsSet.end();
           it++)
        {
          ghostGlobalIds[ghostIndex] = *it;

          ghostIndex++;
        }
      std::cout << std::flush;
      MPI_Barrier(d_mpiComm);

      double endCompAll = MPI_Wtime();

      global_size_type numNonLocalPointsReceived = numTotalPointsReceived;
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
