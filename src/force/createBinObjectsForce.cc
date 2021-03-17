// ---------------------------------------------------------------------
//
// Copyright (c) 2017 The Regents of the University of Michigan and DFT-FE
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
// @author Sambit Das(2017)
//


template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
forceClass<FEOrder, FEOrderElectro>::createBinObjectsForce(
  const DoFHandler<3> &                            dofHandler,
  const DoFHandler<3> &                            dofHandlerForce,
  const dealii::AffineConstraints<double> &        hangingPlusPBCConstraints,
  const vselfBinsManager<FEOrder, FEOrderElectro> &vselfBinsManager,
  std::vector<std::vector<DoFHandler<C_DIM>::active_cell_iterator>>
    &cellsVselfBallsDofHandler,
  std::vector<std::vector<DoFHandler<C_DIM>::active_cell_iterator>>
    &cellsVselfBallsDofHandlerForce,
  std::vector<std::map<dealii::CellId, unsigned int>>
    &                                   cellsVselfBallsClosestAtomIdDofHandler,
  std::map<unsigned int, unsigned int> &AtomIdBinIdLocalDofHandler,
  std::vector<std::map<DoFHandler<C_DIM>::active_cell_iterator,
                       std::vector<unsigned int>>>
    &cellFacesVselfBallSurfacesDofHandler,
  std::vector<std::map<DoFHandler<C_DIM>::active_cell_iterator,
                       std::vector<unsigned int>>>
    &cellFacesVselfBallSurfacesDofHandlerForce)
{
  const unsigned int faces_per_cell = GeometryInfo<C_DIM>::faces_per_cell;
  const unsigned int dofs_per_cell  = dofHandler.get_fe().dofs_per_cell;
  const unsigned int dofs_per_face  = dofHandler.get_fe().dofs_per_face;
  const unsigned int numberBins     = vselfBinsManager.getAtomIdsBins().size();
  // clear exisitng data
  cellsVselfBallsDofHandler.clear();
  cellsVselfBallsDofHandlerForce.clear();
  cellFacesVselfBallSurfacesDofHandler.clear();
  cellFacesVselfBallSurfacesDofHandlerForce.clear();
  cellsVselfBallsClosestAtomIdDofHandler.clear();
  AtomIdBinIdLocalDofHandler.clear();
  // resize
  cellsVselfBallsDofHandler.resize(numberBins);
  cellsVselfBallsDofHandlerForce.resize(numberBins);
  cellFacesVselfBallSurfacesDofHandler.resize(numberBins);
  cellFacesVselfBallSurfacesDofHandlerForce.resize(numberBins);
  cellsVselfBallsClosestAtomIdDofHandler.resize(numberBins);

  for (unsigned int iBin = 0; iBin < numberBins; ++iBin)
    {
      const std::map<dealii::types::global_dof_index, int> &boundaryNodeMap =
        vselfBinsManager.getBoundaryFlagsBins()[iBin];
      const std::map<dealii::types::global_dof_index, int> &closestAtomBinMap =
        vselfBinsManager.getClosestAtomIdsBins()[iBin];
      DoFHandler<C_DIM>::active_cell_iterator cell = dofHandler.begin_active();
      DoFHandler<C_DIM>::active_cell_iterator endc = dofHandler.end();
      DoFHandler<C_DIM>::active_cell_iterator cellForce =
        dofHandlerForce.begin_active();
      for (; cell != endc; ++cell, ++cellForce)
        {
          if (cell->is_locally_owned())
            {
              std::vector<unsigned int> dirichletFaceIds;
              std::vector<unsigned int>
                                        faceIdsWithAtleastOneSolvedNonHangingNode;
              std::vector<unsigned int> allFaceIdsOfCell;
              unsigned int              closestAtomIdSum          = 0;
              unsigned int              closestAtomId             = 0;
              unsigned int              nonHangingNodeIdCountCell = 0;
              for (unsigned int iFace = 0; iFace < faces_per_cell; ++iFace)
                {
                  int  dirichletDofCount         = 0;
                  bool isSolvedDofPresent        = false;
                  int  nonHangingNodeIdCountFace = 0;
                  std::vector<types::global_dof_index> iFaceGlobalDofIndices(
                    dofs_per_face);
                  cell->face(iFace)->get_dof_indices(iFaceGlobalDofIndices);
                  for (unsigned int iFaceDof = 0; iFaceDof < dofs_per_face;
                       ++iFaceDof)
                    {
                      const types::global_dof_index nodeId =
                        iFaceGlobalDofIndices[iFaceDof];
                      if (!hangingPlusPBCConstraints.is_constrained(nodeId))
                        {
                          Assert(boundaryNodeMap.find(nodeId) !=
                                   boundaryNodeMap.end(),
                                 ExcMessage("BUG"));
                          Assert(closestAtomBinMap.find(nodeId) !=
                                   closestAtomBinMap.end(),
                                 ExcMessage("BUG"));

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
                            std::pair<dealii::types::global_dof_index, double>>
                            *rowData =
                              hangingPlusPBCConstraints.get_constraint_entries(
                                nodeId);
                          for (unsigned int j = 0; j < rowData->size(); ++j)
                            {
                              Assert(boundaryNodeMap.find(
                                       (*rowData)[j].first) !=
                                       boundaryNodeMap.end(),
                                     ExcMessage("BUG"));

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
                    ExcMessage(
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
  DoFHandler<C_DIM>::active_cell_iterator cell =
    dftPtr->d_dofHandlerRhoNodal.begin_active();
  DoFHandler<C_DIM>::active_cell_iterator endc =
    dftPtr->d_dofHandlerRhoNodal.end();
  for (; cell != endc; ++cell)
    if (cell->is_locally_owned())
      d_cellIdToActiveCellIteratorMapDofHandlerRhoNodalElectro[cell->id()] =
        cell;
}
//
