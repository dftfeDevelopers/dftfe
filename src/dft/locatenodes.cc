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
// @author Shiva Rudraraju, Phani Motamarri, Sambit Das
//

// source file for locating core atom nodes
template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
dftClass<FEOrder, FEOrderElectro>::locateAtomCoreNodes(
  const dealii::DoFHandler<3> &                      _dofHandler,
  std::map<dealii::types::global_dof_index, double> &atomNodeIdToChargeValueMap)
{
  TimerOutput::Scope scope(computing_timer, "locate atom nodes");
  atomNodeIdToChargeValueMap.clear();
  const unsigned int vertices_per_cell = GeometryInfo<3>::vertices_per_cell;

  const bool isPseudopotential = d_dftParamsPtr->isPseudopotential;

  DoFHandler<3>::active_cell_iterator cell = _dofHandler.begin_active(),
                                      endc = _dofHandler.end();

  dealii::IndexSet locallyOwnedDofs = _dofHandler.locally_owned_dofs();

  std::map<dealii::types::global_dof_index, dealii::Point<3>> supportPoints;
  dealii::DoFTools::map_dofs_to_support_points(dealii::MappingQ1<3, 3>(),
                                               _dofHandler,
                                               supportPoints);

  // locating atom nodes
  const unsigned int     numAtoms = atomLocations.size();
  std::set<unsigned int> atomsTolocate;
  for (unsigned int i = 0; i < numAtoms; i++)
    atomsTolocate.insert(i);
  // element loop
  for (; cell != endc; ++cell)
    if (cell->is_locally_owned())
      for (unsigned int i = 0; i < vertices_per_cell; ++i)
        {
          const dealii::types::global_dof_index nodeID =
            cell->vertex_dof_index(i, 0);
          Point<3> feNodeGlobalCoord = cell->vertex(i);
          //
          // loop over all atoms to locate the corresponding nodes
          //
          for (std::set<unsigned int>::iterator it = atomsTolocate.begin();
               it != atomsTolocate.end();
               ++it)
            {
              Point<3> atomCoord(atomLocations[*it][2],
                                 atomLocations[*it][3],
                                 atomLocations[*it][4]);
              if (feNodeGlobalCoord.distance(atomCoord) < 1.0e-5)
                {
#ifdef DEBUG
                  if (isPseudopotential)
                    {
                      if (d_dftParamsPtr->verbosity >= 4)
                        {
                          std::cout
                            << "atom core with valence charge "
                            << atomLocations[*it][1] << " located with node id "
                            << nodeID << " in processor " << this_mpi_process
                            << " nodal coor " << feNodeGlobalCoord[0] << " "
                            << feNodeGlobalCoord[1] << " "
                            << feNodeGlobalCoord[2] << std::endl;
                        }
                    }
                  else
                    {
                      if (d_dftParamsPtr->verbosity >= 4)
                        {
                          std::cout
                            << "atom core with charge " << atomLocations[*it][0]
                            << " located with node id " << nodeID
                            << " in processor " << this_mpi_process
                            << " nodal coor " << feNodeGlobalCoord[0] << " "
                            << feNodeGlobalCoord[1] << " "
                            << feNodeGlobalCoord[2] << std::endl;
                        }
                    }
#endif
                  if (locallyOwnedDofs.is_element(nodeID))
                    {
                      if (isPseudopotential)
                        atomNodeIdToChargeValueMap.insert(
                          std::pair<dealii::types::global_dof_index, double>(
                            nodeID, atomLocations[*it][1]));
                      else
                        atomNodeIdToChargeValueMap.insert(
                          std::pair<dealii::types::global_dof_index, double>(
                            nodeID, atomLocations[*it][0]));
#ifdef DEBUG
                      if (d_dftParamsPtr->verbosity >= 4)
                        std::cout << " and added \n";
#endif
                    }
                  else
                    {
#ifdef DEBUG
                      if (d_dftParamsPtr->verbosity >= 4)
                        std::cout << " but skipped \n";
#endif
                    }
                  atomsTolocate.erase(*it);
                  break;
                } // tolerance check if loop
            }     // atomsTolocate loop
        }         // vertices_per_cell loop
  MPI_Barrier(mpi_communicator);

  const unsigned int totalAtomNodesFound =
    Utilities::MPI::sum(atomNodeIdToChargeValueMap.size(), mpi_communicator);
  AssertThrow(totalAtomNodesFound == numAtoms,
              ExcMessage(
                "Atleast one atom doesn't lie on a triangulation vertex"));
}

template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
dftClass<FEOrder, FEOrderElectro>::locatePeriodicPinnedNodes(
  const dealii::DoFHandler<3> &            _dofHandler,
  const dealii::AffineConstraints<double> &constraintsBase,
  dealii::AffineConstraints<double> &      constraints)
{
  // pin a node away from all atoms in case of full PBC for total electrostatic
  // potential solve
  if (!(d_dftParamsPtr->periodicX && d_dftParamsPtr->periodicY &&
        d_dftParamsPtr->periodicZ))
    return;

  TimerOutput::Scope scope(computing_timer, "locate periodic pinned node");
  const int          numberImageCharges = d_imageIds.size();
  const int          numberGlobalAtoms  = atomLocations.size();
  const int          totalNumberAtoms = numberGlobalAtoms + numberImageCharges;

  dealii::IndexSet locallyRelevantDofs;
  dealii::DoFTools::extract_locally_relevant_dofs(_dofHandler,
                                                  locallyRelevantDofs);
  dealii::IndexSet locallyOwnedDofs = _dofHandler.locally_owned_dofs();

  std::map<dealii::types::global_dof_index, dealii::Point<3>> supportPoints;
  dealii::DoFTools::map_dofs_to_support_points(dealii::MappingQ1<3, 3>(),
                                               _dofHandler,
                                               supportPoints);

  //
  // find vertex furthest from all nuclear charges
  //
  double                          maxDistance = -1.0;
  dealii::types::global_dof_index maxNode, minNode;

  std::map<dealii::types::global_dof_index, Point<3>>::iterator iterMap;
  for (iterMap = supportPoints.begin(); iterMap != supportPoints.end();
       ++iterMap)
    if (locallyOwnedDofs.is_element(iterMap->first) &&
        !(constraintsBase.is_constrained(iterMap->first) &&
          !constraintsBase.is_identity_constrained(iterMap->first)))
      {
        double minDistance             = 1e10;
        minNode                        = -1;
        Point<3> nodalPointCoordinates = iterMap->second;
        for (unsigned int iAtom = 0; iAtom < totalNumberAtoms; ++iAtom)
          {
            Point<3> atomCoor;

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
                atomCoor[0] = d_imagePositions[iAtom - numberGlobalAtoms][0];
                atomCoor[1] = d_imagePositions[iAtom - numberGlobalAtoms][1];
                atomCoor[2] = d_imagePositions[iAtom - numberGlobalAtoms][2];
              }

            double distance = atomCoor.distance(nodalPointCoordinates);

            if (distance <= minDistance)
              {
                minDistance = distance;
                minNode     = iterMap->first;
              }
          }

        if (minDistance > maxDistance)
          {
            maxDistance = minDistance;
            maxNode     = iterMap->first;
          }
      }

  double globalMaxDistance;

  MPI_Allreduce(
    &maxDistance, &globalMaxDistance, 1, MPI_DOUBLE, MPI_MAX, mpi_communicator);



  // locating pinned nodes
  std::vector<std::vector<double>> pinnedLocations;
  std::vector<double>              temp(3, 0.0);
  std::vector<double>              tempLocal(3, 0.0);
  unsigned int                     taskId = 0;

  if (std::abs(maxDistance - globalMaxDistance) < 1e-07)
    taskId = Utilities::MPI::this_mpi_process(mpi_communicator);

  unsigned int maxTaskId;

  MPI_Allreduce(&taskId, &maxTaskId, 1, MPI_INT, MPI_MAX, mpi_communicator);

  if (Utilities::MPI::this_mpi_process(mpi_communicator) == maxTaskId)
    {
#ifdef DEBUG
      if (d_dftParamsPtr->verbosity >= 4)
        std::cout << "Found Node locally on processor Id: "
                  << Utilities::MPI::this_mpi_process(mpi_communicator)
                  << std::endl;
#endif
      if (locallyOwnedDofs.is_element(maxNode))
        {
          if (constraintsBase.is_identity_constrained(maxNode))
            {
              const dealii::types::global_dof_index masterNode =
                (*constraintsBase.get_constraint_entries(maxNode))[0].first;
              Point<3> nodalPointCoordinates =
                supportPoints.find(masterNode)->second;
              tempLocal[0] = nodalPointCoordinates[0];
              tempLocal[1] = nodalPointCoordinates[1];
              tempLocal[2] = nodalPointCoordinates[2];
            }
          else
            {
              Point<3> nodalPointCoordinates =
                supportPoints.find(maxNode)->second;
              tempLocal[0] = nodalPointCoordinates[0];
              tempLocal[1] = nodalPointCoordinates[1];
              tempLocal[2] = nodalPointCoordinates[2];
            }
          // checkFlag = 1;
        }
    }


  MPI_Allreduce(
    &tempLocal[0], &temp[0], 3, MPI_DOUBLE, MPI_SUM, mpi_communicator);

  pinnedLocations.push_back(temp);


  const unsigned int dofs_per_cell         = _dofHandler.get_fe().dofs_per_cell;
  DoFHandler<3>::active_cell_iterator cell = _dofHandler.begin_active(),
                                      endc = _dofHandler.end();

  const unsigned int     numberNodes = pinnedLocations.size();
  std::set<unsigned int> nodesTolocate;
  for (unsigned int i = 0; i < numberNodes; i++)
    nodesTolocate.insert(i);

  std::vector<dealii::types::global_dof_index> cell_dof_indices(dofs_per_cell);
  for (; cell != endc; ++cell)
    if (cell->is_locally_owned() || cell->is_ghost())
      {
        cell->get_dof_indices(cell_dof_indices);

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            const dealii::types::global_dof_index nodeID = cell_dof_indices[i];
            Point<3> feNodeGlobalCoord = supportPoints[cell_dof_indices[i]];

            //
            // loop over all atoms to locate the corresponding nodes
            //
            for (std::set<unsigned int>::iterator it = nodesTolocate.begin();
                 it != nodesTolocate.end();
                 ++it)
              {
                Point<3> pinnedNodeCoord(pinnedLocations[*it][0],
                                         pinnedLocations[*it][1],
                                         pinnedLocations[*it][2]);
                if (feNodeGlobalCoord.distance(pinnedNodeCoord) < 1.0e-5)
                  {
                    if (d_dftParamsPtr->verbosity >= 4)
                      std::cout << "Pinned core with nodal coordinates ("
                                << pinnedLocations[*it][0] << " "
                                << pinnedLocations[*it][1] << " "
                                << pinnedLocations[*it][2]
                                << ") located with node id " << nodeID
                                << " in processor " << this_mpi_process
                                << std::endl;
                    if (locallyRelevantDofs.is_element(nodeID))
                      {
                        constraints.add_line(nodeID);
                        constraints.set_inhomogeneity(nodeID, 0.0);
                      }
                    nodesTolocate.erase(*it);
                    break;
                  } // tolerance check if loop

              } // atomsTolocate loop

          } // vertices_per_cell loop

      } // locally owned cell if loop

  MPI_Barrier(mpi_communicator);
}
