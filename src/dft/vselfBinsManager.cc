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
// @author  Sambit Das, Phani Motamarri
//

#include <vectorUtilities.h>
#include <vselfBinsManager.h>

#include "createBinsSanityCheck.cc"
#include "solveVselfInBins.cc"
#include <dftUtils.h>
namespace dftfe
{
  namespace internal
  {
    void
    exchangeAtomToGlobalNodeIdMaps(
      const dftfe::uInt totalNumberAtoms,
      std::map<dftfe::Int, std::set<dealii::types::global_dof_index>>
                       &atomToGlobalNodeIdMap,
      const dftfe::uInt numMeshPartitions,
      const MPI_Comm   &mpi_communicator)

    {
      std::map<dftfe::Int, std::set<dealii::types::global_dof_index>>::iterator
        iter;

      for (dftfe::uInt iGlobal = 0; iGlobal < totalNumberAtoms; ++iGlobal)
        {
          //
          // for each charge, exchange its global list across all procs
          //
          iter = atomToGlobalNodeIdMap.find(iGlobal);

          std::vector<dealii::types::global_dof_index>
            localAtomToGlobalNodeIdList;

          if (iter != atomToGlobalNodeIdMap.end())
            {
              std::set<dealii::types::global_dof_index> &localGlobalNodeIdSet =
                iter->second;
              std::copy(localGlobalNodeIdSet.begin(),
                        localGlobalNodeIdSet.end(),
                        std::back_inserter(localAtomToGlobalNodeIdList));
            }

          const int numberGlobalNodeIdsOnLocalProc =
            localAtomToGlobalNodeIdList.size();

          std::vector<int> atomToGlobalNodeIdListSizes(numMeshPartitions);

          MPI_Allgather(
            &numberGlobalNodeIdsOnLocalProc,
            1,
            dftfe::dataTypes::mpi_type_id(&numberGlobalNodeIdsOnLocalProc),
            &(atomToGlobalNodeIdListSizes[0]),
            1,
            dftfe::dataTypes::mpi_type_id(atomToGlobalNodeIdListSizes.data()),
            mpi_communicator);

          const dftfe::Int newAtomToGlobalNodeIdListSize =
            std::accumulate(&(atomToGlobalNodeIdListSizes[0]),
                            &(atomToGlobalNodeIdListSizes[numMeshPartitions]),
                            0);

          std::vector<dealii::types::global_dof_index>
            globalAtomToGlobalNodeIdList(newAtomToGlobalNodeIdListSize);

          std::vector<int> mpiOffsets(numMeshPartitions);

          mpiOffsets[0] = 0;

          for (dftfe::uInt i = 1; i < numMeshPartitions; ++i)
            mpiOffsets[i] =
              atomToGlobalNodeIdListSizes[i - 1] + mpiOffsets[i - 1];

          MPI_Allgatherv(&(localAtomToGlobalNodeIdList[0]),
                         numberGlobalNodeIdsOnLocalProc,
                         DEAL_II_DOF_INDEX_MPI_TYPE,
                         &(globalAtomToGlobalNodeIdList[0]),
                         &(atomToGlobalNodeIdListSizes[0]),
                         &(mpiOffsets[0]),
                         DEAL_II_DOF_INDEX_MPI_TYPE,
                         mpi_communicator);

          //
          // over-write local interaction with items of globalInteractionList
          //
          for (dftfe::uInt i = 0; i < globalAtomToGlobalNodeIdList.size(); ++i)
            (atomToGlobalNodeIdMap[iGlobal])
              .insert(globalAtomToGlobalNodeIdList[i]);
        }
    }


    void
    exchangeInteractionMaps(
      const dftfe::uInt                           totalNumberAtoms,
      std::map<dftfe::Int, std::set<dftfe::Int>> &interactionMap,
      const dftfe::uInt                           numMeshPartitions,
      const MPI_Comm                             &mpi_communicator)

    {
      std::map<dftfe::Int, std::set<dftfe::Int>>::iterator iter;

      for (dftfe::uInt iGlobal = 0; iGlobal < totalNumberAtoms; ++iGlobal)
        {
          //
          // for each charge, exchange its global list across all procs
          //
          iter = interactionMap.find(iGlobal);

          std::vector<dftfe::Int> localAtomToInteractingAtomsList;

          if (iter != interactionMap.end())
            {
              std::set<dftfe::Int> &localInteractingAtomsSet = iter->second;
              std::copy(localInteractingAtomsSet.begin(),
                        localInteractingAtomsSet.end(),
                        std::back_inserter(localAtomToInteractingAtomsList));
            }

          const int sizeOnLocalProc = localAtomToInteractingAtomsList.size();

          std::vector<int> interactionMapListSizes(numMeshPartitions);

          MPI_Allgather(&sizeOnLocalProc,
                        1,
                        dftfe::dataTypes::mpi_type_id(&sizeOnLocalProc),
                        &(interactionMapListSizes[0]),
                        1,
                        dftfe::dataTypes::mpi_type_id(
                          interactionMapListSizes.data()),
                        mpi_communicator);

          const dftfe::Int newListSize =
            std::accumulate(&(interactionMapListSizes[0]),
                            &(interactionMapListSizes[numMeshPartitions]),
                            0);

          std::vector<dftfe::Int> globalInteractionMapList(newListSize);

          std::vector<int> mpiOffsets(numMeshPartitions);

          mpiOffsets[0] = 0;

          for (dftfe::uInt i = 1; i < numMeshPartitions; ++i)
            mpiOffsets[i] = interactionMapListSizes[i - 1] + mpiOffsets[i - 1];

          MPI_Allgatherv(&(localAtomToInteractingAtomsList[0]),
                         sizeOnLocalProc,
                         dftfe::dataTypes::mpi_type_id(
                           localAtomToInteractingAtomsList.data()),
                         &(globalInteractionMapList[0]),
                         &(interactionMapListSizes[0]),
                         &(mpiOffsets[0]),
                         dftfe::dataTypes::mpi_type_id(
                           globalInteractionMapList.data()),
                         mpi_communicator);

          //
          // over-write local interaction with items of globalInteractionList
          //
          for (dftfe::uInt i = 0; i < globalInteractionMapList.size(); ++i)
            (interactionMap[iGlobal]).insert(globalInteractionMapList[i]);
        }
    }


    //
    dftfe::uInt
    createAndCheckInteractionMap(
      std::map<dftfe::Int, std::set<dftfe::Int>> &interactionMap,
      const dealii::DoFHandler<3>                &dofHandler,
      const std::map<dealii::types::global_dof_index, dealii::Point<3>>
                                             &supportPoints,
      const std::vector<std::vector<double>> &atomLocations,
      const std::vector<std::vector<double>> &imagePositions,
      const std::vector<dftfe::Int>          &imageIds,
      const double                            radiusAtomBall,
      const dealii::BoundingBox<3>           &boundingBoxTria,
      const dftfe::uInt                       n_mpi_processes,
      const MPI_Comm                         &mpi_communicator,
      dealii::TimerOutput                    &computing_timer)
    {
      computing_timer.enter_subsection(
        "create bins: find nodes inside atom balls");
      interactionMap.clear();
      const dftfe::uInt numberImageCharges = imageIds.size();
      const dftfe::uInt numberGlobalAtoms  = atomLocations.size();
      const dftfe::uInt totalNumberAtoms =
        numberGlobalAtoms + numberImageCharges;

      const dftfe::uInt dofs_per_cell = dofHandler.get_fe().dofs_per_cell;
      const dftfe::uInt vertices_per_cell =
        dealii::GeometryInfo<3>::vertices_per_cell;

      std::map<dftfe::Int, std::set<dealii::types::global_dof_index>>
        atomToGlobalNodeIdMap;
      for (dftfe::uInt iAtom = 0; iAtom < totalNumberAtoms; ++iAtom)
        {
          std::set<dealii::types::global_dof_index> tempNodalSet;
          dealii::Point<3>                          atomCoor;

          if (iAtom < numberGlobalAtoms)
            {
              atomCoor[0] = atomLocations[iAtom][2];
              atomCoor[1] = atomLocations[iAtom][3];
              atomCoor[2] = atomLocations[iAtom][4];
            }
          else
            {
              //
              // Fill with ImageAtom Coors
              //
              atomCoor[0] = imagePositions[iAtom - numberGlobalAtoms][0];
              atomCoor[1] = imagePositions[iAtom - numberGlobalAtoms][1];
              atomCoor[2] = imagePositions[iAtom - numberGlobalAtoms][2];
            }

          dealii::Tensor<1, 3, double> tempDisp;
          tempDisp[0] = radiusAtomBall + 0.1;
          tempDisp[1] = radiusAtomBall + 0.1;
          tempDisp[2] = radiusAtomBall + 0.1;
          std::pair<dealii::Point<3, double>, dealii::Point<3, double>>
            boundaryPoints;
          boundaryPoints.first  = atomCoor - tempDisp;
          boundaryPoints.second = atomCoor + tempDisp;
          dealii::BoundingBox<3> boundingBoxAroundPoint(boundaryPoints);

          if (boundingBoxTria.get_neighbor_type(boundingBoxAroundPoint) ==
              dealii::NeighborType::not_neighbors)
            continue;

          dealii::DoFHandler<3>::active_cell_iterator cell = dofHandler
                                                               .begin_active(),
                                                      endc = dofHandler.end();
          std::vector<dealii::types::global_dof_index> cell_dof_indices(
            dofs_per_cell);

          // loop over ghost cells is need to account for interactions between
          // atom balls of diferent atoms interecting a locally owned cell and a
          // neighbouring ghost cell.
          for (; cell != endc; ++cell)
            if (cell->is_locally_owned() || cell->is_ghost())
              {
                const dealii::BoundingBox<3> &cellBoundingBox =
                  cell->bounding_box();
                if (cellBoundingBox.get_neighbor_type(boundingBoxAroundPoint) ==
                    dealii::NeighborType::not_neighbors)
                  continue;

                dftfe::Int cutOffFlag = 0;
                // cell->get_dof_indices(cell_dof_indices);

                for (dftfe::uInt iNode = 0; iNode < vertices_per_cell; ++iNode)
                  {
                    const dealii::Point<3> &feNodeGlobalCoord =
                      supportPoints.find(cell->vertex_dof_index(iNode, 0))
                        ->second;
                    const double distance =
                      atomCoor.distance(feNodeGlobalCoord);

                    if (distance < radiusAtomBall)
                      {
                        cutOffFlag = 1;
                        break;
                      }

                  } // element vertex loop

                if (cutOffFlag == 0)
                  for (const auto &face : cell->face_iterators())
                    {
                      const auto center = face->center();
                      if (atomCoor.distance(center) < radiusAtomBall)
                        {
                          cutOffFlag = 1;
                          break;
                        }
                    }

                if (cutOffFlag == 0)
                  {
                    cell->get_dof_indices(cell_dof_indices);
                    for (dftfe::uInt iNode = 0; iNode < dofs_per_cell; ++iNode)
                      {
                        const dealii::Point<3> &feNodeGlobalCoord =
                          supportPoints.find(cell_dof_indices[iNode])->second;
                        const double distance =
                          atomCoor.distance(feNodeGlobalCoord);
                        if (distance < radiusAtomBall)
                          {
                            cutOffFlag = 1;
                            break;
                          }
                      } // element dofs loop
                  }

                if (cutOffFlag == 1)
                  {
                    for (dftfe::uInt iNode = 0; iNode < vertices_per_cell;
                         ++iNode)
                      {
                        const dealii::types::global_dof_index nodeID =
                          cell->vertex_dof_index(iNode, 0);
                        tempNodalSet.insert(nodeID);
                      }
                  }

              } // cell locally owned if loop

          if (!tempNodalSet.empty())
            atomToGlobalNodeIdMap[iAtom] = tempNodalSet;

        } // atom loop

      computing_timer.leave_subsection(
        "create bins: find nodes inside atom balls");
      //
      // exchange atomToGlobalNodeIdMap across all processors
      //
      // internal::exchangeAtomToGlobalNodeIdMaps(totalNumberAtoms,
      //					   atomToGlobalNodeIdMap,
      //					   n_mpi_processes,
      //					   mpi_communicator);

      computing_timer.enter_subsection("create bins: local interaction maps");
      dftfe::uInt ilegalInteraction = 0;

      for (dftfe::uInt iAtom = 0; iAtom < totalNumberAtoms; ++iAtom)
        {
          if (atomToGlobalNodeIdMap.find(iAtom) == atomToGlobalNodeIdMap.end())
            continue;

          //
          // Add iAtom to the interactionMap corresponding to the key iAtom
          //
          if (iAtom < numberGlobalAtoms)
            interactionMap[iAtom].insert(iAtom);

          // std::cout<<"IAtom: "<<iAtom<<std::endl;

          for (dftfe::Int jAtom = iAtom - 1; jAtom > -1; jAtom--)
            {
              // std::cout<<"JAtom: "<<jAtom<<std::endl;
              if (atomToGlobalNodeIdMap.find(jAtom) ==
                  atomToGlobalNodeIdMap.end())
                continue;
              //
              // compute intersection between the atomGlobalNodeIdMap of iAtom
              // and jAtom
              //
              std::vector<dealii::types::global_dof_index> nodesIntersection;

              std::set_intersection(atomToGlobalNodeIdMap[iAtom].begin(),
                                    atomToGlobalNodeIdMap[iAtom].end(),
                                    atomToGlobalNodeIdMap[jAtom].begin(),
                                    atomToGlobalNodeIdMap[jAtom].end(),
                                    std::back_inserter(nodesIntersection));

              // std::cout<<"Size of NodeIntersection:
              // "<<nodesIntersection.size()<<std::endl;

              if (nodesIntersection.size() > 0)
                {
                  if (iAtom < numberGlobalAtoms && jAtom < numberGlobalAtoms)
                    {
                      //
                      // if both iAtom and jAtom are actual atoms in
                      // unit-cell/domain,then iAtom and jAtom are interacting
                      // atoms
                      //
                      interactionMap[iAtom].insert(jAtom);
                      interactionMap[jAtom].insert(iAtom);
                    }
                  else if (iAtom < numberGlobalAtoms &&
                           jAtom >= numberGlobalAtoms)
                    {
                      //
                      // if iAtom is actual atom in unit-cell and jAtom is
                      // imageAtom, find the actual atom for which jAtom is the
                      // image then create the interaction map between that atom
                      // and iAtom
                      //
                      const dftfe::Int masterAtomId =
                        imageIds[jAtom - numberGlobalAtoms];
                      if (masterAtomId == iAtom)
                        {
                          // std::cout<<"Atom and its own image is interacting
                          // decrease radius"<<std::endl;
                          ilegalInteraction = 1;
                          break;
                        }
                      interactionMap[iAtom].insert(masterAtomId);
                      interactionMap[masterAtomId].insert(iAtom);
                    }
                  else if (iAtom >= numberGlobalAtoms &&
                           jAtom < numberGlobalAtoms)
                    {
                      //
                      // if jAtom is actual atom in unit-cell and iAtom is
                      // imageAtom, find the actual atom for which iAtom is the
                      // image and then create interaction map between that atom
                      // and jAtom
                      //
                      const dftfe::Int masterAtomId =
                        imageIds[iAtom - numberGlobalAtoms];
                      if (masterAtomId == jAtom)
                        {
                          // std::cout<<"Atom and its own image is interacting
                          // decrease radius"<<std::endl;
                          ilegalInteraction = 1;
                          break;
                        }
                      interactionMap[masterAtomId].insert(jAtom);
                      interactionMap[jAtom].insert(masterAtomId);
                    }
                  else if (iAtom >= numberGlobalAtoms &&
                           jAtom >= numberGlobalAtoms)
                    {
                      //
                      // if both iAtom and jAtom are image atoms in unit-cell
                      // iAtom and jAtom are interacting atoms find the actual
                      // atoms for which iAtom and jAtoms are images and create
                      // interacting maps between them
                      const dftfe::Int masteriAtomId =
                        imageIds[iAtom - numberGlobalAtoms];
                      const dftfe::Int masterjAtomId =
                        imageIds[jAtom - numberGlobalAtoms];
                      if (masteriAtomId == masterjAtomId)
                        {
                          // std::cout<<"Two Image Atoms corresponding to same
                          // parent Atoms are interacting decrease
                          // radius"<<std::endl;
                          ilegalInteraction = 2;
                          break;
                        }
                      interactionMap[masteriAtomId].insert(masterjAtomId);
                      interactionMap[masterjAtomId].insert(masteriAtomId);
                    }
                }

            } // end of jAtom loop

          if (ilegalInteraction != 0)
            break;

        } // end of iAtom loop
      computing_timer.leave_subsection("create bins: local interaction maps");
      if (dealii::Utilities::MPI::sum(ilegalInteraction, mpi_communicator) > 0)
        return 1;

      /*
      computing_timer.enter_subsection("create bins: exchange interaction
      maps"); internal::exchangeInteractionMaps(totalNumberAtoms,
          interactionMap,
          n_mpi_processes,
          mpi_communicator);
      computing_timer.leave_subsection("create bins: exchange interaction
      maps");
      */
      return 0;
    }
  } // namespace internal

  // constructor

  vselfBinsManager::vselfBinsManager(const MPI_Comm      &mpi_comm_parent,
                                     const MPI_Comm      &mpi_comm_domain,
                                     const MPI_Comm      &mpi_intercomm_kpts,
                                     const dftParameters &dftParams)
    : mpi_communicator(mpi_comm_domain)
    , d_mpiCommParent(mpi_comm_parent)
    , d_mpiInterCommKpts(mpi_intercomm_kpts)
    , n_mpi_processes(dealii::Utilities::MPI::n_mpi_processes(mpi_comm_domain))
    , d_dftParams(dftParams)
    , this_mpi_process(
        dealii::Utilities::MPI::this_mpi_process(mpi_comm_domain))
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0))
    , d_storedAdaptiveBallRadius(0)
  {}


  void
  vselfBinsManager::createAtomBins(
    std::vector<const dealii::AffineConstraints<double> *> &constraintsVector,
    const dealii::AffineConstraints<double> &onlyHangingNodeConstraints,
    const dealii::DoFHandler<3>             &dofHandler,
    const dealii::AffineConstraints<double> &constraintMatrix,
    const std::vector<std::vector<double>>  &atomLocations,
    const std::vector<std::vector<double>>  &imagePositions,
    const std::vector<dftfe::Int>           &imageIds,
    const std::vector<double>               &imageCharges,
    const double                             radiusAtomBall)

  {
    dealii::ConditionalOStream pcout(std::cout,
                                     (dealii::Utilities::MPI::this_mpi_process(
                                        d_mpiCommParent) == 0));
    dealii::TimerOutput        computing_timer(mpi_communicator,
                                        pcout,
                                        d_dftParams.reproducible_output ||
                                            d_dftParams.verbosity < 4 ?
                                                 dealii::TimerOutput::never :
                                                 dealii::TimerOutput::summary,
                                        dealii::TimerOutput::wall_times);

    computing_timer.enter_subsection("create bins: initial overheads");

    d_bins.clear();
    d_binsImages.clear();
    d_boundaryFlag.clear();
    d_boundaryFlagOnlyChargeId.clear();
    d_dofClosestChargeLocationMap.clear();
    d_vselfBinField.clear();
    d_closestAtomBin.clear();
    d_vselfBinConstraintMatrices.clear();
    d_atomIdBinIdMapLocalAllImages.clear();

    d_atomLocations = atomLocations;

    const dftfe::uInt numberImageCharges = imageIds.size();
    const dftfe::uInt numberGlobalAtoms  = atomLocations.size();
    const dftfe::uInt totalNumberAtoms = numberGlobalAtoms + numberImageCharges;

    const dftfe::uInt vertices_per_cell =
      dealii::GeometryInfo<3>::vertices_per_cell;
    const dftfe::uInt dofs_per_cell = dofHandler.get_fe().dofs_per_cell;


    dealii::BoundingBox<3> boundingBoxTria(
      vectorTools::createBoundingBoxTriaLocallyOwned(dofHandler));

    std::map<dealii::types::global_dof_index, dealii::Point<3>> supportPoints;
    dealii::DoFTools::map_dofs_to_support_points(dealii::MappingQ1<3, 3>(),
                                                 dofHandler,
                                                 supportPoints);

    dealii::IndexSet locally_relevant_dofs;
    dealii::DoFTools::extract_locally_relevant_dofs(dofHandler,
                                                    locally_relevant_dofs);

    computing_timer.leave_subsection("create bins: initial overheads");

    // create interaction maps by finding the intersection of global NodeIds of
    // each atom
    std::map<dftfe::Int, std::set<dftfe::Int>> interactionMap;

    double radiusAtomBallAdaptive =
      (d_storedAdaptiveBallRadius > 1e-6) ?
        d_storedAdaptiveBallRadius :
        ((d_dftParams.meshSizeOuterBall > 0.5) ? 6.0 : 4.5);

    if (d_dftParams.smearedNuclearCharges &&
        (d_storedAdaptiveBallRadius < 1e-6))
      radiusAtomBallAdaptive = ((d_dftParams.meshSizeOuterBall > 1.5 &&
                                 d_dftParams.outerAtomBallRadius < 6.0) ||
                                (d_dftParams.meshSizeOuterBall > 2.2)) ?
                                 6.0 :
                                 4.0;

    if (std::fabs(radiusAtomBall) < 1e-6)
      {
        if (d_dftParams.verbosity >= 1)
          pcout
            << "Determining the ball radius around the atom for nuclear self-potential solve... "
            << std::endl;
        dftfe::uInt check =
          internal::createAndCheckInteractionMap(interactionMap,
                                                 dofHandler,
                                                 supportPoints,
                                                 atomLocations,
                                                 imagePositions,
                                                 imageIds,
                                                 radiusAtomBallAdaptive,
                                                 boundingBoxTria,
                                                 n_mpi_processes,
                                                 mpi_communicator,
                                                 computing_timer);
        while (check != 0 && radiusAtomBallAdaptive >= 0.1)
          {
            radiusAtomBallAdaptive -= 0.25;
            check =
              internal::createAndCheckInteractionMap(interactionMap,
                                                     dofHandler,
                                                     supportPoints,
                                                     atomLocations,
                                                     imagePositions,
                                                     imageIds,
                                                     radiusAtomBallAdaptive,
                                                     boundingBoxTria,
                                                     n_mpi_processes,
                                                     mpi_communicator,
                                                     computing_timer);
          }

        std::string message;
        if (check == 1 || check == 2)
          message =
            "DFT-FE error: Tried to adaptively determine the ball radius for nuclear self-potential solve and it has reached the minimum allowed value of 0.1, which can severly detoriate the accuracy of the KSDFT groundstate energy and forces. Please use a larger periodic super cell which can accomodate a larger ball radius.";

        AssertThrow(check == 0, dealii::ExcMessage(message));

        if (d_dftParams.verbosity >= 1 && !d_dftParams.reproducible_output)
          pcout << "...Adaptively set ball radius: " << radiusAtomBallAdaptive
                << std::endl;

        if (radiusAtomBallAdaptive < 2.5)
          if (d_dftParams.verbosity >= 1 && !d_dftParams.reproducible_output)
            pcout
              << "DFT-FE warning: Tried to adaptively determine the ball radius for nuclear self-potential solve and was found to be less than 2.5, which can detoriate the accuracy of the KSDFT groundstate energy and forces. One approach to overcome this issue is to use a larger super cell with smallest periodic dimension greater than 5.0 (twice of 2.5), assuming an orthorhombic domain. If that is not feasible, you may need more h refinement of the finite element mesh around the atoms to achieve the desired accuracy."
              << std::endl;
        MPI_Barrier(mpi_communicator);

        d_storedAdaptiveBallRadius = radiusAtomBallAdaptive;
      }
    else
      {
        if (d_dftParams.verbosity >= 1)
          pcout
            << "Setting the ball radius for nuclear self-potential solve from input parameters value: "
            << radiusAtomBall << std::endl;

        radiusAtomBallAdaptive = radiusAtomBall;
        const dftfe::uInt check =
          internal::createAndCheckInteractionMap(interactionMap,
                                                 dofHandler,
                                                 supportPoints,
                                                 atomLocations,
                                                 imagePositions,
                                                 imageIds,
                                                 radiusAtomBallAdaptive,
                                                 boundingBoxTria,
                                                 n_mpi_processes,
                                                 mpi_communicator,
                                                 computing_timer);
        d_storedAdaptiveBallRadius = radiusAtomBall;
        std::string message;
        if (check == 1)
          message =
            "DFT-FE Error: Atom and its own image is interacting. Decrease SELF POTENTIAL RADIUS.";
        else if (check == 2)
          message =
            "DFT-FE Error: Two Image Atoms corresponding to same parent Atoms are interacting. Decrease SELF POTENTIAL RADIUS.";

        AssertThrow(check == 0, dealii::ExcMessage(message));
      }

    computing_timer.enter_subsection("create bins: put in bins");
    std::map<dftfe::Int, std::set<dftfe::Int>>::iterator iter;

    //
    // start by adding atom 0 to bin 0
    //
    (d_bins[0]).insert(0);
    dftfe::Int binCount = 0;
    // iterate from atom 1 onwards
    for (dftfe::Int i = 1; i < numberGlobalAtoms; ++i)
      {
        const std::set<dftfe::Int> &interactingAtoms = interactionMap[i];
        //
        // treat spl case when no atom intersects with another. e.g. simple
        // cubic
        //
        dftfe::Int isInteraction = 1;
        if (interactingAtoms.size() == 0)
          isInteraction = 0;

        if (dealii::Utilities::MPI::sum(isInteraction, mpi_communicator) == 0)
          {
            (d_bins[binCount]).insert(i);
            continue;
          }

        bool isBinFound;
        // iterate over each existing bin and see if atom i fits into the bin
        for (iter = d_bins.begin(); iter != d_bins.end(); ++iter)
          {
            // pick out atoms in this bin
            std::set<dftfe::Int> &atomsInThisBin = iter->second;
            dftfe::Int            index = std::distance(d_bins.begin(), iter);

            isBinFound                   = true;
            dftfe::Int isBinIntersecting = 0;

            // to belong to this bin, this atom must not overlap with any other
            // atom already present in this bin
            for (std::set<dftfe::Int>::iterator iter2 =
                   interactingAtoms.begin();
                 iter2 != interactingAtoms.end();
                 ++iter2)
              {
                dftfe::Int atom = *iter2;

                if (atomsInThisBin.find(atom) != atomsInThisBin.end())
                  {
                    isBinIntersecting = 1;
                    break;
                  }
              }

            if (dealii::Utilities::MPI::sum(isBinIntersecting,
                                            mpi_communicator) > 0)
              {
                isBinFound = false;
              }

            if (isBinFound == true)
              {
                (d_bins[index]).insert(i);
                break;
              }
          }
        // if all current bins have been iterated over w/o a match then
        // create a new bin for this atom
        if (isBinFound == false)
          {
            binCount++;
            (d_bins[binCount]).insert(i);
          }
      }

    const dftfe::Int numberBins = binCount + 1;
    if (d_dftParams.verbosity >= 2)
      pcout << "number bins: " << numberBins << std::endl;

    computing_timer.leave_subsection("create bins: put in bins");

    computing_timer.enter_subsection("create bins: set boundary conditions");
    const dftfe::uInt faces_per_cell = dealii::GeometryInfo<3>::faces_per_cell;
    const dftfe::uInt dofs_per_face  = dofHandler.get_fe().dofs_per_face;

    std::vector<std::vector<dftfe::Int>> imageIdsInBins(numberBins);
    d_boundaryFlag.resize(numberBins);
    d_boundaryFlagOnlyChargeId.resize(numberBins);
    d_dofClosestChargeLocationMap.resize(numberBins);
    d_vselfBinField.resize(numberBins);
    d_closestAtomBin.resize(numberBins);
    d_vselfBinConstraintMatrices.resize(4 * numberBins);

    const dealii::IndexSet &locally_owned_dofs =
      dofHandler.locally_owned_dofs();
    dealii::IndexSet ghost_indices = locally_relevant_dofs;
    ghost_indices.subtract_set(locally_owned_dofs);

    distributedCPUVec<double> inhomogBoundaryVec =
      distributedCPUVec<double>(locally_owned_dofs,
                                ghost_indices,
                                mpi_communicator);

    std::vector<distributedCPUVec<double>> inhomogBoundaryVecVselfDerR(3);
    for (dftfe::uInt idim = 0; idim < 3; idim++)
      inhomogBoundaryVecVselfDerR[idim].reinit(inhomogBoundaryVec);

    d_constraintsOnlyHangingInfo.initialize(
      inhomogBoundaryVec.get_partitioner(), onlyHangingNodeConstraints);


    double radiusAtomBallReduced = radiusAtomBallAdaptive;

    // d_inhomoIdsColoredVecFlattened.clear();
    // d_inhomoIdsColoredVecFlattened.resize(numberBins*locally_owned_dofs.size(),1.0);

    //
    // set constraint matrices for each bin
    //
    for (dftfe::Int iBin = 0; iBin < numberBins; ++iBin)
      {
        /*
        inhomogBoundaryVec=0.0;
        for (dftfe::uInt idim=0; idim<3; idim++)
            inhomogBoundaryVecVselfDerR[idim]=0.0;
        */

        std::set<dftfe::Int>         &atomsInBinSet = d_bins[iBin];
        std::vector<dftfe::Int>       atomsInCurrentBin(atomsInBinSet.begin(),
                                                  atomsInBinSet.end());
        std::vector<dealii::Point<3>> atomPositionsInCurrentBin;

        dftfe::Int numberGlobalAtomsInBin = atomsInCurrentBin.size();

        std::vector<dftfe::Int> &imageIdsOfAtomsInCurrentBin =
          imageIdsInBins[iBin];
        std::vector<std::vector<double>> imagePositionsOfAtomsInCurrentBin;

        if (d_dftParams.verbosity >= 2)
          pcout << "bin " << iBin
                << ": number of global atoms: " << numberGlobalAtomsInBin
                << std::endl;

        for (dftfe::Int index = 0; index < numberGlobalAtomsInBin; ++index)
          {
            dftfe::Int globalChargeIdInCurrentBin = atomsInCurrentBin[index];

            d_atomIdBinIdMapLocalAllImages[globalChargeIdInCurrentBin] = iBin;

            // std:cout<<"Index: "<<index<<"Global Charge Id:
            // "<<globalChargeIdInCurrentBin<<std::endl;

            dealii::Point<3> atomPosition(
              atomLocations[globalChargeIdInCurrentBin][2],
              atomLocations[globalChargeIdInCurrentBin][3],
              atomLocations[globalChargeIdInCurrentBin][4]);
            atomPositionsInCurrentBin.push_back(atomPosition);

            for (dftfe::Int iImageAtom = 0; iImageAtom < numberImageCharges;
                 ++iImageAtom)
              {
                if (imageIds[iImageAtom] == globalChargeIdInCurrentBin)
                  {
                    imageIdsOfAtomsInCurrentBin.push_back(iImageAtom);
                    std::vector<double> imageChargeCoor =
                      imagePositions[iImageAtom];
                    imagePositionsOfAtomsInCurrentBin.push_back(
                      imageChargeCoor);
                    d_atomIdBinIdMapLocalAllImages[numberGlobalAtoms +
                                                   iImageAtom] = iBin;
                  }
              }
          }

        dftfe::Int numberImageAtomsInBin = imageIdsOfAtomsInCurrentBin.size();

        std::map<dealii::types::global_dof_index, dftfe::Int> &boundaryNodeMap =
          d_boundaryFlag[iBin];
        std::map<dealii::types::global_dof_index, dftfe::Int>
          &boundaryNodeMapOnlyChargeId = d_boundaryFlagOnlyChargeId[iBin];
        std::map<dealii::types::global_dof_index, dealii::Point<3>>
          &dofClosestChargeLocationMap = d_dofClosestChargeLocationMap[iBin];
        std::map<dealii::types::global_dof_index, double> &vSelfBinNodeMap =
          d_vselfBinField[iBin];

        //
        // create constraint matrix for current bin
        //
        d_vselfBinConstraintMatrices[4 * iBin].reinit(locally_relevant_dofs);
        for (dftfe::uInt idim = 0; idim < 3; idim++)
          d_vselfBinConstraintMatrices[4 * iBin + idim + 1].reinit(
            locally_relevant_dofs);


        std::map<dealii::types::global_dof_index,
                 dealii::Point<3>>::const_iterator iterMap;

        bool areBoundaryConditionsCorrectInCaseOfHangingNodes = false;
        while (!areBoundaryConditionsCorrectInCaseOfHangingNodes)
          {
            inhomogBoundaryVec = 0.0;
            for (dftfe::uInt idim = 0; idim < 3; idim++)
              inhomogBoundaryVecVselfDerR[idim] = 0.0;

            for (iterMap = supportPoints.begin();
                 iterMap != supportPoints.end();
                 ++iterMap)
              {
                if (locally_relevant_dofs.is_element(iterMap->first))
                  {
                    if (!onlyHangingNodeConstraints.is_constrained(
                          iterMap->first))
                      {
                        dftfe::Int              overlapFlag = 0;
                        const dealii::Point<3> &nodalCoor   = iterMap->second;
                        std::vector<double>     distanceFromNode;

                        for (dftfe::uInt iAtom = 0;
                             iAtom <
                             numberGlobalAtomsInBin + numberImageAtomsInBin;
                             ++iAtom)
                          {
                            dealii::Point<3> atomCoor;
                            if (iAtom < numberGlobalAtomsInBin)
                              {
                                atomCoor = atomPositionsInCurrentBin[iAtom];
                              }
                            else
                              {
                                atomCoor[0] = imagePositionsOfAtomsInCurrentBin
                                  [iAtom - numberGlobalAtomsInBin][0];
                                atomCoor[1] = imagePositionsOfAtomsInCurrentBin
                                  [iAtom - numberGlobalAtomsInBin][1];
                                atomCoor[2] = imagePositionsOfAtomsInCurrentBin
                                  [iAtom - numberGlobalAtomsInBin][2];
                              }


                            double distance = nodalCoor.distance(atomCoor);

                            distanceFromNode.push_back(distance);

                            if (distance < radiusAtomBallReduced)
                              {
                                overlapFlag += 1;
                                break;
                              }

                          } // atom loop

                        std::vector<double>::iterator minDistanceIter =
                          std::min_element(distanceFromNode.begin(),
                                           distanceFromNode.end());

                        std::iterator_traits<std::vector<double>::iterator>::
                          difference_type minDistanceAtomId =
                            std::distance(distanceFromNode.begin(),
                                          minDistanceIter);

                        double minDistance = *minDistanceIter;

                        dftfe::Int chargeId;
                        dftfe::Int domainChargeId;

                        if (minDistanceAtomId < numberGlobalAtomsInBin)
                          {
                            chargeId = atomsInCurrentBin[minDistanceAtomId];
                            domainChargeId = chargeId;
                          }
                        else
                          {
                            chargeId =
                              imageIdsOfAtomsInCurrentBin
                                [minDistanceAtomId - numberGlobalAtomsInBin] +
                              numberGlobalAtoms;
                            domainChargeId =
                              imageIds[imageIdsOfAtomsInCurrentBin
                                         [minDistanceAtomId -
                                          numberGlobalAtomsInBin]];
                          }

                        d_closestAtomBin[iBin][iterMap->first] = chargeId;

                        if (minDistanceAtomId < numberGlobalAtomsInBin)
                          {
                            dofClosestChargeLocationMap[iterMap->first][0] =
                              atomLocations[chargeId][2];
                            dofClosestChargeLocationMap[iterMap->first][1] =
                              atomLocations[chargeId][3];
                            dofClosestChargeLocationMap[iterMap->first][2] =
                              atomLocations[chargeId][4];
                          }
                        else
                          {
                            dofClosestChargeLocationMap[iterMap->first][0] =
                              imagePositions[chargeId - numberGlobalAtoms][0];
                            dofClosestChargeLocationMap[iterMap->first][1] =
                              imagePositions[chargeId - numberGlobalAtoms][1];
                            dofClosestChargeLocationMap[iterMap->first][2] =
                              imagePositions[chargeId - numberGlobalAtoms][2];
                          }

                        if (minDistance < radiusAtomBallReduced)
                          {
                            boundaryNodeMap[iterMap->first] = chargeId;
                            boundaryNodeMapOnlyChargeId[iterMap->first] =
                              domainChargeId;

                            double atomCharge;

                            if (minDistanceAtomId < numberGlobalAtomsInBin)
                              {
                                if (d_dftParams.isPseudopotential)
                                  atomCharge = atomLocations[chargeId][1];
                                else
                                  atomCharge = atomLocations[chargeId][0];
                              }
                            else
                              atomCharge =
                                imageCharges[imageIdsOfAtomsInCurrentBin
                                               [minDistanceAtomId -
                                                numberGlobalAtomsInBin]];

                            if (minDistance <= 1e-05)
                              vSelfBinNodeMap[iterMap->first] = 0.0;
                            else
                              vSelfBinNodeMap[iterMap->first] =
                                -atomCharge / minDistance;
                          }
                        else
                          {
                            double atomCharge;

                            if (minDistanceAtomId < numberGlobalAtomsInBin)
                              {
                                if (d_dftParams.isPseudopotential)
                                  atomCharge = atomLocations[chargeId][1];
                                else
                                  atomCharge = atomLocations[chargeId][0];
                              }
                            else
                              atomCharge =
                                imageCharges[imageIdsOfAtomsInCurrentBin
                                               [minDistanceAtomId -
                                                numberGlobalAtomsInBin]];

                            const double potentialValue =
                              -atomCharge / minDistance;

                            boundaryNodeMap[iterMap->first]             = -1;
                            boundaryNodeMapOnlyChargeId[iterMap->first] = -1;
                            vSelfBinNodeMap[iterMap->first] = potentialValue;


                            inhomogBoundaryVec[iterMap->first] = potentialValue;
                            for (dftfe::uInt idim = 0; idim < 3; idim++)
                              inhomogBoundaryVecVselfDerR[idim][iterMap
                                                                  ->first] =
                                potentialValue / minDistance *
                                (nodalCoor[idim] -
                                 dofClosestChargeLocationMap[iterMap->first]
                                                            [idim]) /
                                minDistance;

                          } // else loop

                      } // non-hanging node check

                  } // locally relevant dofs

              } // nodal loop
            // First Apply correct dirichlet boundary conditions on elements
            // with atleast one solved node
            dealii::DoFHandler<3>::active_cell_iterator cell =
                                                          dofHandler
                                                            .begin_active(),
                                                        endc = dofHandler.end();
            for (; cell != endc; ++cell)
              {
                if (cell->is_locally_owned() || cell->is_ghost())
                  {
                    std::vector<dealii::types::global_dof_index>
                      cell_dof_indices(dofs_per_cell);
                    cell->get_dof_indices(cell_dof_indices);

                    dftfe::Int closestChargeIdSolvedNode = -1;
                    bool       isDirichletNodePresent    = false;
                    bool       isSolvedNodePresent       = false;
                    dftfe::Int numSolvedNodes            = 0;
                    dftfe::Int closestChargeIdSolvedSum  = 0;
                    for (dftfe::uInt iNode = 0; iNode < dofs_per_cell; ++iNode)
                      {
                        const dealii::types::global_dof_index globalNodeId =
                          cell_dof_indices[iNode];
                        if (!onlyHangingNodeConstraints.is_constrained(
                              globalNodeId))
                          {
                            const dftfe::Int boundaryId =
                              d_boundaryFlag[iBin][globalNodeId];
                            if (boundaryId == -1)
                              {
                                isDirichletNodePresent = true;
                              }
                            else
                              {
                                isSolvedNodePresent       = true;
                                closestChargeIdSolvedNode = boundaryId;
                                numSolvedNodes++;
                                closestChargeIdSolvedSum += boundaryId;
                              }
                          }

                      } // element node loop



                    if (isDirichletNodePresent && isSolvedNodePresent)
                      {
                        double           closestAtomChargeSolved;
                        dealii::Point<3> closestAtomLocationSolved;
                        Assert(numSolvedNodes * closestChargeIdSolvedNode ==
                                 closestChargeIdSolvedSum,
                               dealii::ExcMessage("BUG"));
                        if (closestChargeIdSolvedNode < numberGlobalAtoms)
                          {
                            closestAtomLocationSolved[0] =
                              atomLocations[closestChargeIdSolvedNode][2];
                            closestAtomLocationSolved[1] =
                              atomLocations[closestChargeIdSolvedNode][3];
                            closestAtomLocationSolved[2] =
                              atomLocations[closestChargeIdSolvedNode][4];
                            if (d_dftParams.isPseudopotential)
                              closestAtomChargeSolved =
                                atomLocations[closestChargeIdSolvedNode][1];
                            else
                              closestAtomChargeSolved =
                                atomLocations[closestChargeIdSolvedNode][0];
                          }
                        else
                          {
                            const dftfe::Int imageId =
                              closestChargeIdSolvedNode - numberGlobalAtoms;
                            closestAtomChargeSolved = imageCharges[imageId];
                            closestAtomLocationSolved[0] =
                              imagePositions[imageId][0];
                            closestAtomLocationSolved[1] =
                              imagePositions[imageId][1];
                            closestAtomLocationSolved[2] =
                              imagePositions[imageId][2];
                          }

                        // FIXME: dofs touched optimization to be done
                        for (dftfe::uInt iNode = 0; iNode < dofs_per_cell;
                             ++iNode)
                          {
                            const dealii::types::global_dof_index globalNodeId =
                              cell_dof_indices[iNode];
                            const dftfe::Int boundaryId =
                              d_boundaryFlag[iBin][globalNodeId];
                            if (!onlyHangingNodeConstraints.is_constrained(
                                  globalNodeId) &&
                                boundaryId == -1)
                              {
                                // d_vselfBinConstraintMatrices[iBin].add_line(globalNodeId);
                                const dealii::Point<3> &nodalPoint =
                                  supportPoints[globalNodeId];
                                const double distance = nodalPoint.distance(
                                  closestAtomLocationSolved);
                                const double newPotentialValue =
                                  -closestAtomChargeSolved / distance;
                                // d_vselfBinConstraintMatrices[iBin].set_inhomogeneity(globalNodeId,newPotentialValue);
                                d_vselfBinField[iBin][globalNodeId] =
                                  newPotentialValue;
                                d_closestAtomBin[iBin][globalNodeId] =
                                  closestChargeIdSolvedNode;
                                inhomogBoundaryVec[globalNodeId] =
                                  newPotentialValue;

                                for (dftfe::uInt idim = 0; idim < 3; idim++)
                                  inhomogBoundaryVecVselfDerR
                                    [idim][globalNodeId] =
                                      newPotentialValue / distance *
                                      (nodalPoint[idim] -
                                       closestAtomLocationSolved[idim]) /
                                      distance;
                              } // check non hanging node and vself consraints
                                // not already set
                          }     // element node loop

                      } // check if element has atleast one dirichlet node and
                        // atleast one solved node
                  }     // cell locally owned
              }         // cell loop

            const std::map<dealii::types::global_dof_index, dftfe::Int>
              &closestAtomBinMap = d_closestAtomBin[iBin];

            bool checkPassed = true;
            cell = dofHandler.begin_active(), endc = dofHandler.end();
            for (; cell != endc; ++cell)
              if (cell->is_locally_owned())
                {
                  std::vector<dftfe::uInt>
                              faceIdsWithAtleastOneSolvedNonHangingNode;
                  dftfe::uInt closestAtomIdSum          = 0;
                  dftfe::uInt closestAtomId             = 0;
                  dftfe::uInt nonHangingNodeIdCountCell = 0;
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
                          if (!constraintMatrix.is_constrained(nodeId))
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
                            } // non-hanging node check
                          else
                            {
                              const std::vector<
                                std::pair<dealii::types::global_dof_index,
                                          double>> *rowData =
                                constraintMatrix.get_constraint_entries(nodeId);
                              for (dftfe::uInt j = 0; j < rowData->size(); ++j)
                                {
                                  if (d_dftParams
                                        .createConstraintsFromSerialDofhandler)
                                    {
                                      //
                                      // FIXME: When constraints are obtained
                                      // using serial dof handler to account for
                                      // a known parallel constraints issue in
                                      // dealii, this can cause the relevant
                                      // dofs to not have all dofs to which the
                                      // local constraints expand to
                                      //
                                      if (boundaryNodeMap.find(
                                            (*rowData)[j].first) ==
                                          boundaryNodeMap.end())
                                        {
                                          if (d_dftParams.verbosity >= 4)
                                            std::cout
                                              << "DFT-FE Warning: relevant dofs do not have all dofs to which the local constraints expand to. This is due to a known parallel constraints issue in dealii combined with our temporary serial dof handler fix."
                                              << std::endl;
                                          continue;
                                        }
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
                    } // Face loop

                  // fill the target objects
                  if (faceIdsWithAtleastOneSolvedNonHangingNode.size() > 0)
                    {
                      if (!(closestAtomIdSum ==
                            closestAtomId * nonHangingNodeIdCountCell))
                        checkPassed = false;
                    }
                } // cell locally owned

            dftfe::Int temp = 0;
            if (!checkPassed)
              temp = 1;

            temp = dealii::Utilities::MPI::sum(temp, mpi_communicator);

            if (temp == 0)
              areBoundaryConditionsCorrectInCaseOfHangingNodes = true;

            if (!areBoundaryConditionsCorrectInCaseOfHangingNodes)
              radiusAtomBallReduced -= 0.5;

            if (!d_dftParams.reproducible_output)
              AssertThrow(
                radiusAtomBallReduced >= 0.1,
                dealii::ExcMessage(
                  "DFT-FE error: Adaptively determined reduced ball radius for applying correct Dirichlet boundary condtions taking hanging nodes into account is less than minimum value of 0.1. Try increasing SELF POTENTIAL RADIUS to > 6.0. If that is not possible due to small domain sizes along the periodic directions, reduce MESH SIZE AROUND ATOM and/or increase ATOM BALL RADIUS."));
          }

        if (d_dftParams.verbosity >= 4 && !d_dftParams.reproducible_output)
          pcout
            << "Reduced ball radius for setting boundary conditions taking hanging nodes into consideration: "
            << radiusAtomBallReduced << std::endl;

        inhomogBoundaryVec.update_ghost_values();
        for (auto index : locally_relevant_dofs)
          {
            if (!onlyHangingNodeConstraints.is_constrained(index) &&
                std::abs(inhomogBoundaryVec[index]) > 1e-10)
              {
                d_vselfBinConstraintMatrices[4 * iBin].add_line(index);
                d_vselfBinConstraintMatrices[4 * iBin].set_inhomogeneity(
                  index, inhomogBoundaryVec[index]);
              }
          }


        d_vselfBinConstraintMatrices[4 * iBin].merge(
          onlyHangingNodeConstraints,
          dealii::AffineConstraints<
            double>::MergeConflictBehavior::left_object_wins);
        d_vselfBinConstraintMatrices[4 * iBin].close();
        d_vselfBinConstraintMatrices[4 * iBin].merge(
          constraintMatrix,
          dealii::AffineConstraints<
            double>::MergeConflictBehavior::left_object_wins);
        d_vselfBinConstraintMatrices[4 * iBin].close();
        constraintsVector.push_back(&(d_vselfBinConstraintMatrices[4 * iBin]));

        for (dftfe::uInt idim = 0; idim < 3; idim++)
          {
            inhomogBoundaryVecVselfDerR[idim].update_ghost_values();
            for (auto index : locally_relevant_dofs)
              {
                if (!onlyHangingNodeConstraints.is_constrained(index) &&
                    std::abs(inhomogBoundaryVecVselfDerR[idim][index]) > 1e-10)
                  {
                    d_vselfBinConstraintMatrices[4 * iBin + idim + 1].add_line(
                      index);
                    d_vselfBinConstraintMatrices[4 * iBin + idim + 1]
                      .set_inhomogeneity(
                        index, inhomogBoundaryVecVselfDerR[idim][index]);
                  }
              }

            d_vselfBinConstraintMatrices[4 * iBin + idim + 1].merge(
              onlyHangingNodeConstraints,
              dealii::AffineConstraints<
                double>::MergeConflictBehavior::left_object_wins);
            d_vselfBinConstraintMatrices[4 * iBin + idim + 1].close();
            d_vselfBinConstraintMatrices[4 * iBin + idim + 1].merge(
              constraintMatrix,
              dealii::AffineConstraints<
                double>::MergeConflictBehavior::left_object_wins);
            d_vselfBinConstraintMatrices[4 * iBin + idim + 1].close();
            constraintsVector.push_back(
              &(d_vselfBinConstraintMatrices[4 * iBin + idim + 1]));
          }

        /*
           for (dftfe::uInt i = 0; i < inhomogBoundaryVec.locally_owned_size();
        ++i)
           {
           const dealii::types::global_dof_index
        globalNodeId=inhomogBoundaryVec.get_partitioner()->local_to_global(i);
           if(
        d_vselfBinConstraintMatrices[iBin].is_inhomogeneously_constrained(globalNodeId)
           &&
        d_vselfBinConstraintMatrices[iBin].get_constraint_entries(globalNodeId)->size()==0)
           d_inhomoIdsColoredVecFlattened[i*numberBins+iBin]=0.0;
        //if(
        d_vselfBinConstraintMatrices[iBin].is_inhomogeneously_constrained(globalNodeId))
        //    d_inhomoIdsColoredVecFlattened[i*numberBins+iBin]=0.0;
        }
         */
      } // bin loop

    computing_timer.leave_subsection("create bins: set boundary conditions");

    computing_timer.enter_subsection("create bins: sanity check");
    createAtomBinsSanityCheck(dofHandler, onlyHangingNodeConstraints);
    computing_timer.leave_subsection("create bins: sanity check");

    if (!d_dftParams.floatingNuclearCharges)
      locateAtomsInBins(dofHandler);

    return;

  } //


  void
  vselfBinsManager::updateBinsBc(
    std::vector<const dealii::AffineConstraints<double> *> &constraintsVector,
    const dealii::AffineConstraints<double> &onlyHangingNodeConstraints,
    const dealii::DoFHandler<3>             &dofHandler,
    const dealii::AffineConstraints<double> &constraintMatrix,
    const std::vector<std::vector<double>>  &atomLocations,
    const std::vector<std::vector<double>>  &imagePositions,
    const std::vector<dftfe::Int>           &imageIds,
    const std::vector<double>               &imageCharges,
    const bool                               vselfPerturbationUpdateForStress)

  {
    d_dofClosestChargeLocationMap.clear();

    d_atomLocations = atomLocations;

    const dftfe::uInt numberImageCharges = imageIds.size();
    const dftfe::uInt numberGlobalAtoms  = atomLocations.size();
    const dftfe::uInt totalNumberAtoms = numberGlobalAtoms + numberImageCharges;

    const dftfe::uInt vertices_per_cell =
      dealii::GeometryInfo<3>::vertices_per_cell;
    const dftfe::uInt dofs_per_cell = dofHandler.get_fe().dofs_per_cell;

    const dealii::IndexSet &locally_owned_dofs =
      dofHandler.locally_owned_dofs();


    std::map<dealii::types::global_dof_index, dealii::Point<3>> supportPoints;
    dealii::DoFTools::map_dofs_to_support_points(dealii::MappingQ1<3, 3>(),
                                                 dofHandler,
                                                 supportPoints);

    dealii::IndexSet locally_relevant_dofs;
    dealii::DoFTools::extract_locally_relevant_dofs(dofHandler,
                                                    locally_relevant_dofs);

    dealii::IndexSet ghost_indices = locally_relevant_dofs;
    ghost_indices.subtract_set(locally_owned_dofs);

    distributedCPUVec<double> inhomogBoundaryVec =
      distributedCPUVec<double>(locally_owned_dofs,
                                ghost_indices,
                                mpi_communicator);

    std::vector<distributedCPUVec<double>> inhomogBoundaryVecVselfDerR(3);
    for (dftfe::uInt idim = 0; idim < 3; idim++)
      inhomogBoundaryVecVselfDerR[idim].reinit(inhomogBoundaryVec);

    const dftfe::Int numberBins = d_bins.size();
    d_dofClosestChargeLocationMap.resize(numberBins);
    //
    // set constraint matrices for each bin
    //
    for (dftfe::Int iBin = 0; iBin < numberBins; ++iBin)
      {
        inhomogBoundaryVec = 0.0;
        for (dftfe::uInt idim = 0; idim < 3; idim++)
          inhomogBoundaryVecVselfDerR[idim] = 0.0;

        std::map<dealii::types::global_dof_index, dealii::Point<3>>
          &dofClosestChargeLocationMap = d_dofClosestChargeLocationMap[iBin];

        const std::map<dealii::types::global_dof_index, dftfe::Int>
          &closestAtomMapCurrentBin = d_closestAtomBin[iBin];

        const std::map<dealii::types::global_dof_index, dftfe::Int>
          &boundaryFlagMapCurrentBin = d_boundaryFlag[iBin];

        dealii::DoFHandler<3>::active_cell_iterator cell =
                                                      dofHandler.begin_active(),
                                                    endc = dofHandler.end();
        for (; cell != endc; ++cell)
          {
            if (cell->is_locally_owned())
              {
                std::vector<dealii::types::global_dof_index> cell_dof_indices(
                  dofs_per_cell);
                cell->get_dof_indices(cell_dof_indices);

                for (dftfe::uInt iNode = 0; iNode < dofs_per_cell; ++iNode)
                  {
                    const dealii::types::global_dof_index globalNodeId =
                      cell_dof_indices[iNode];
                    if (!onlyHangingNodeConstraints.is_constrained(
                          globalNodeId))
                      {
                        const dftfe::Int closestAtomId =
                          closestAtomMapCurrentBin.find(globalNodeId)->second;
                        const dftfe::Int boundaryId =
                          boundaryFlagMapCurrentBin.find(globalNodeId)->second;

                        double           closestAtomCharge;
                        dealii::Point<3> closestAtomLocation;
                        if (closestAtomId < numberGlobalAtoms)
                          {
                            closestAtomLocation[0] =
                              atomLocations[closestAtomId][2];
                            closestAtomLocation[1] =
                              atomLocations[closestAtomId][3];
                            closestAtomLocation[2] =
                              atomLocations[closestAtomId][4];
                            if (d_dftParams.isPseudopotential)
                              closestAtomCharge =
                                atomLocations[closestAtomId][1];
                            else
                              closestAtomCharge =
                                atomLocations[closestAtomId][0];
                          }
                        else
                          {
                            const dftfe::Int imageId =
                              closestAtomId - numberGlobalAtoms;
                            closestAtomCharge      = imageCharges[imageId];
                            closestAtomLocation[0] = imagePositions[imageId][0];
                            closestAtomLocation[1] = imagePositions[imageId][1];
                            closestAtomLocation[2] = imagePositions[imageId][2];
                          }

                        if (!vselfPerturbationUpdateForStress)
                          dofClosestChargeLocationMap[globalNodeId] =
                            closestAtomLocation;

                        if (boundaryId == -1 &&
                            !(std::abs(inhomogBoundaryVec[globalNodeId]) >
                              1e-10))
                          {
                            const double distance =
                              supportPoints[globalNodeId].distance(
                                closestAtomLocation);
                            const double newPotentialValue =
                              -closestAtomCharge / distance;
                            inhomogBoundaryVec[globalNodeId] =
                              newPotentialValue;

                            if (!vselfPerturbationUpdateForStress)
                              {
                                // d_vselfBinField[iBin][globalNodeId] =
                                //  newPotentialValue;

                                for (dftfe::uInt idim = 0; idim < 3; idim++)
                                  inhomogBoundaryVecVselfDerR
                                    [idim][globalNodeId] =
                                      newPotentialValue / distance *
                                      (supportPoints[globalNodeId][idim] -
                                       closestAtomLocation[idim]) /
                                      distance;
                              }
                          } // check non hanging node and vself consraints not
                            // already set
                      }
                  } // element node loop

              } // cell locally owned
          }     // cell loop

        //
        // create constraint matrix for current bin
        //

        const bool hasHangingNodes =
          dofHandler.get_triangulation().has_hanging_nodes();

        // if (!hasHangingNodes)
        // std::cout<<"uniform mesh"<<std::endl;
        if (hasHangingNodes)
          {
            d_vselfBinConstraintMatrices[4 * iBin].reinit(
              locally_relevant_dofs);

            inhomogBoundaryVec.update_ghost_values();
            for (auto index : locally_relevant_dofs)
              {
                if (!onlyHangingNodeConstraints.is_constrained(index) &&
                    std::abs(inhomogBoundaryVec[index]) > 1e-10)
                  {
                    d_vselfBinConstraintMatrices[4 * iBin].add_line(index);
                    d_vselfBinConstraintMatrices[4 * iBin].set_inhomogeneity(
                      index, inhomogBoundaryVec[index]);
                  }
              }

            d_vselfBinConstraintMatrices[4 * iBin].merge(
              onlyHangingNodeConstraints,
              dealii::AffineConstraints<
                double>::MergeConflictBehavior::left_object_wins);
            d_vselfBinConstraintMatrices[4 * iBin].close();
            d_vselfBinConstraintMatrices[4 * iBin].merge(
              constraintMatrix,
              dealii::AffineConstraints<
                double>::MergeConflictBehavior::left_object_wins);
            d_vselfBinConstraintMatrices[4 * iBin].close();
            constraintsVector.push_back(
              &(d_vselfBinConstraintMatrices[4 * iBin]));

            if (!vselfPerturbationUpdateForStress)
              {
                for (dftfe::uInt idim = 0; idim < 3; idim++)
                  d_vselfBinConstraintMatrices[4 * iBin + idim + 1].reinit(
                    locally_relevant_dofs);

                for (dftfe::uInt idim = 0; idim < 3; idim++)
                  {
                    inhomogBoundaryVecVselfDerR[idim].update_ghost_values();
                    for (auto index : locally_relevant_dofs)
                      {
                        if (!onlyHangingNodeConstraints.is_constrained(index) &&
                            std::abs(inhomogBoundaryVecVselfDerR[idim][index]) >
                              1e-10)
                          {
                            d_vselfBinConstraintMatrices[4 * iBin + idim + 1]
                              .add_line(index);
                            d_vselfBinConstraintMatrices[4 * iBin + idim + 1]
                              .set_inhomogeneity(
                                index,
                                inhomogBoundaryVecVselfDerR[idim][index]);
                          }
                      }

                    d_vselfBinConstraintMatrices[4 * iBin + idim + 1].merge(
                      onlyHangingNodeConstraints,
                      dealii::AffineConstraints<
                        double>::MergeConflictBehavior::left_object_wins);
                    d_vselfBinConstraintMatrices[4 * iBin + idim + 1].close();
                    d_vselfBinConstraintMatrices[4 * iBin + idim + 1].merge(
                      constraintMatrix,
                      dealii::AffineConstraints<
                        double>::MergeConflictBehavior::left_object_wins);
                    d_vselfBinConstraintMatrices[4 * iBin + idim + 1].close();
                    constraintsVector.push_back(
                      &(d_vselfBinConstraintMatrices[4 * iBin + idim + 1]));
                  }
              }
            else
              {
                for (dftfe::uInt idim = 0; idim < 3; idim++)
                  constraintsVector.push_back(
                    &(d_vselfBinConstraintMatrices[4 * iBin + idim + 1]));
              }
          }
        else
          {
            inhomogBoundaryVec.update_ghost_values();
            d_constraintsOnlyHangingInfo.distribute(inhomogBoundaryVec);
            inhomogBoundaryVec.update_ghost_values();
            for (auto index : locally_relevant_dofs)
              {
                if (std::abs(
                      d_vselfBinConstraintMatrices[4 * iBin].get_inhomogeneity(
                        index)) > 1e-10)
                  d_vselfBinConstraintMatrices[4 * iBin].set_inhomogeneity(
                    index, inhomogBoundaryVec[index]);
              }

            constraintsVector.push_back(
              &(d_vselfBinConstraintMatrices[4 * iBin]));

            if (!vselfPerturbationUpdateForStress)
              {
                for (dftfe::uInt idim = 0; idim < 3; idim++)
                  {
                    inhomogBoundaryVecVselfDerR[idim].update_ghost_values();
                    d_constraintsOnlyHangingInfo.distribute(
                      inhomogBoundaryVecVselfDerR[idim]);
                    inhomogBoundaryVecVselfDerR[idim].update_ghost_values();
                    for (auto index : locally_relevant_dofs)
                      {
                        if (std::abs(
                              d_vselfBinConstraintMatrices[4 * iBin + idim + 1]
                                .get_inhomogeneity(index)) > 1e-10)
                          d_vselfBinConstraintMatrices[4 * iBin + idim + 1]
                            .set_inhomogeneity(
                              index, inhomogBoundaryVecVselfDerR[idim][index]);
                      }


                    constraintsVector.push_back(
                      &(d_vselfBinConstraintMatrices[4 * iBin + idim + 1]));
                  }
              }
            else
              {
                for (dftfe::uInt idim = 0; idim < 3; idim++)
                  constraintsVector.push_back(
                    &(d_vselfBinConstraintMatrices[4 * iBin + idim + 1]));
              }
          }
      } // bin loop
  }


  void
  vselfBinsManager::locateAtomsInBins(const dealii::DoFHandler<3> &dofHandler)
  {
    d_atomsInBin.clear();

    const dealii::IndexSet &locally_owned_dofs =
      dofHandler.locally_owned_dofs();

    const dftfe::uInt numberBins = d_boundaryFlag.size();
    d_atomsInBin.resize(numberBins);


    for (dftfe::Int iBin = 0; iBin < numberBins; ++iBin)
      {
        dftfe::uInt vertices_per_cell =
          dealii::GeometryInfo<3>::vertices_per_cell;
        dealii::DoFHandler<3>::active_cell_iterator cell =
                                                      dofHandler.begin_active(),
                                                    endc = dofHandler.end();

        std::set<dftfe::Int>   &atomsInBinSet = d_bins[iBin];
        std::vector<dftfe::Int> atomsInCurrentBin(atomsInBinSet.begin(),
                                                  atomsInBinSet.end());
        dftfe::uInt           numberGlobalAtomsInBin = atomsInCurrentBin.size();
        std::set<dftfe::uInt> atomsTolocate;
        for (dftfe::uInt i = 0; i < numberGlobalAtomsInBin; i++)
          atomsTolocate.insert(i);

        for (; cell != endc; ++cell)
          {
            if (cell->is_locally_owned())
              {
                for (dftfe::uInt i = 0; i < vertices_per_cell; ++i)
                  {
                    const dealii::types::global_dof_index nodeID =
                      cell->vertex_dof_index(i, 0);
                    dealii::Point<3> feNodeGlobalCoord = cell->vertex(i);
                    //
                    // loop over all atoms to locate the corresponding nodes
                    //
                    for (std::set<dftfe::uInt>::iterator it =
                           atomsTolocate.begin();
                         it != atomsTolocate.end();
                         ++it)
                      {
                        const dftfe::Int chargeId = atomsInCurrentBin[*it];
                        dealii::Point<3> atomCoord(
                          d_atomLocations[chargeId][2],
                          d_atomLocations[chargeId][3],
                          d_atomLocations[chargeId][4]);
                        if (feNodeGlobalCoord.distance(atomCoord) < 1.0e-5)
                          {
#ifdef DEBUG
                            if (d_dftParams.isPseudopotential)
                              {
                                if (d_dftParams.verbosity >= 4)
                                  std::cout << "atom core in bin " << iBin
                                            << " with valence charge "
                                            << d_atomLocations[chargeId][1]
                                            << " located with node id "
                                            << nodeID << " in processor "
                                            << this_mpi_process;
                              }
                            else
                              {
                                if (d_dftParams.verbosity >= 4)
                                  std::cout << "atom core in bin " << iBin
                                            << " with charge "
                                            << d_atomLocations[chargeId][0]
                                            << " located with node id "
                                            << nodeID << " in processor "
                                            << this_mpi_process;
                              }
#endif
                            if (locally_owned_dofs.is_element(nodeID))
                              {
                                if (d_dftParams.isPseudopotential)
                                  d_atomsInBin[iBin].insert(
                                    std::pair<dealii::types::global_dof_index,
                                              double>(
                                      nodeID, d_atomLocations[chargeId][1]));
                                else
                                  d_atomsInBin[iBin].insert(
                                    std::pair<dealii::types::global_dof_index,
                                              double>(
                                      nodeID, d_atomLocations[chargeId][0]));
#ifdef DEBUG
                                if (d_dftParams.verbosity >= 4)
                                  std::cout << " and added \n";
#endif
                              }
                            else
                              {
#ifdef DEBUG
                                if (d_dftParams.verbosity >= 4)
                                  std::cout << " but skipped \n";
#endif
                              }
                            atomsTolocate.erase(*it);
                            break;
                          } // tolerance check if loop
                      }     // atomsTolocate loop
                  }         // vertices_per_cell loop
              }             // locally owned cell if loop
          }                 // cell loop
        MPI_Barrier(mpi_communicator);
      } // iBin loop
  }


  const std::map<dftfe::Int, std::set<dftfe::Int>> &
  vselfBinsManager::getAtomIdsBins() const
  {
    return d_bins;
  }


  const std::map<dftfe::Int, std::set<dftfe::Int>> &
  vselfBinsManager::getAtomImageIdsBins() const
  {
    return d_binsImages;
  }


  const std::vector<std::map<dealii::types::global_dof_index, dftfe::Int>> &
  vselfBinsManager::getBoundaryFlagsBins() const
  {
    return d_boundaryFlag;
  }


  const std::vector<std::map<dealii::types::global_dof_index, dftfe::Int>> &
  vselfBinsManager::getBoundaryFlagsBinsOnlyChargeId() const
  {
    return d_boundaryFlagOnlyChargeId;
  }


  const std::vector<std::map<dealii::types::global_dof_index, dftfe::Int>> &
  vselfBinsManager::getClosestAtomIdsBins() const
  {
    return d_closestAtomBin;
  }


  const std::vector<
    std::map<dealii::types::global_dof_index, dealii::Point<3>>> &
  vselfBinsManager::getClosestAtomLocationsBins() const
  {
    return d_dofClosestChargeLocationMap;
  }


  const std::vector<distributedCPUVec<double>> &
  vselfBinsManager::getVselfFieldBins() const
  {
    return d_vselfFieldBins;
  }


  const std::vector<distributedCPUVec<double>> &
  vselfBinsManager::getVselfFieldDerRBins() const
  {
    return d_vselfFieldDerRBins;
  }


  const std::vector<distributedCPUVec<double>> &
  vselfBinsManager::getPerturbedVselfFieldBins() const
  {
    return d_vselfFieldPerturbedBins;
  }


  const std::map<dftfe::uInt, dftfe::uInt> &
  vselfBinsManager::getAtomIdBinIdMapLocalAllImages() const
  {
    return d_atomIdBinIdMapLocalAllImages;
  }


  double
  vselfBinsManager::getStoredAdaptiveBallRadius() const
  {
    return d_storedAdaptiveBallRadius;
  }
} // namespace dftfe
