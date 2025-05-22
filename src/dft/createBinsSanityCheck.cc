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
// @author Sambit Das
//

namespace dftfe
{
  void
  vselfBinsManager::createAtomBinsSanityCheck(
    const dealii::DoFHandler<3>             &dofHandler,
    const dealii::AffineConstraints<double> &onlyHangingNodeConstraints)
  {
    const dftfe::uInt faces_per_cell = dealii::GeometryInfo<3>::faces_per_cell;
    const dftfe::uInt dofs_per_cell  = dofHandler.get_fe().dofs_per_cell;
    const dftfe::uInt dofs_per_face  = dofHandler.get_fe().dofs_per_face;
    const dftfe::uInt numberBins     = d_bins.size();

    std::vector<dealii::types::global_dof_index> cell_dof_indices(
      dofs_per_cell);

    for (dftfe::uInt iBin = 0; iBin < numberBins; ++iBin)
      {
        std::map<dealii::types::global_dof_index, dftfe::Int> &boundaryNodeMap =
          d_boundaryFlag[iBin];
        std::map<dealii::types::global_dof_index, dftfe::Int>
          &closestAtomBinMap = d_closestAtomBin[iBin];
        dealii::DoFHandler<3>::active_cell_iterator cell =
                                                      dofHandler.begin_active(),
                                                    endc = dofHandler.end();
        for (; cell != endc; ++cell)
          if (cell->is_locally_owned())
            {
              cell->get_dof_indices(cell_dof_indices);

              bool        isSolvedNodePresent       = false;
              dftfe::uInt numSolvedNodes            = 0;
              dftfe::Int  closestChargeIdSolvedSum  = 0;
              dftfe::Int  closestChargeIdSolvedNode = -1;
              for (dftfe::uInt iNode = 0; iNode < dofs_per_cell; ++iNode)
                {
                  const dealii::types::global_dof_index globalNodeId =
                    cell_dof_indices[iNode];
                  if (!onlyHangingNodeConstraints.is_constrained(globalNodeId))
                    {
                      const dftfe::Int boundaryId =
                        d_boundaryFlag[iBin][globalNodeId];
                      if (boundaryId != -1)
                        {
                          isSolvedNodePresent       = true;
                          closestChargeIdSolvedNode = boundaryId;
                          numSolvedNodes++;
                          closestChargeIdSolvedSum += boundaryId;
                        }
                    }

                } // element node loop
              Assert(numSolvedNodes * closestChargeIdSolvedNode ==
                       closestChargeIdSolvedSum,
                     dealii::ExcMessage("BUG"));

              std::vector<dftfe::uInt> dirichletFaceIds;
              dftfe::uInt              closestAtomIdSum          = 0;
              dftfe::uInt              closestAtomId             = 0;
              dftfe::uInt              nonHangingNodeIdCountCell = 0;
              for (dftfe::uInt iFace = 0; iFace < faces_per_cell; ++iFace)
                {
                  dftfe::Int  dirichletDofCount         = 0;
                  dftfe::uInt nonHangingNodeIdCountFace = 0;
                  std::vector<dealii::types::global_dof_index>
                    iFaceGlobalDofIndices(dofs_per_face);
                  cell->face(iFace)->get_dof_indices(iFaceGlobalDofIndices);
                  for (dftfe::uInt iFaceDof = 0; iFaceDof < dofs_per_face;
                       ++iFaceDof)
                    {
                      dftfe::uInt nodeId = iFaceGlobalDofIndices[iFaceDof];
                      if (!onlyHangingNodeConstraints.is_constrained(nodeId))
                        {
                          Assert(boundaryNodeMap.find(nodeId) !=
                                   boundaryNodeMap.end(),
                                 dealii::ExcMessage("BUG"));
                          Assert(closestAtomBinMap.find(nodeId) !=
                                   closestAtomBinMap.end(),
                                 dealii::ExcMessage("BUG"));
                          dirichletDofCount += boundaryNodeMap[nodeId];
                          closestAtomId = closestAtomBinMap[nodeId];
                          closestAtomIdSum += closestAtomId;
                          nonHangingNodeIdCountCell++;
                          nonHangingNodeIdCountFace++;
                        } // non-hanging node check

                    } // Face dof loop

                  if (dirichletDofCount == -nonHangingNodeIdCountFace)
                    dirichletFaceIds.push_back(iFace);

                } // Face loop

              if (dirichletFaceIds.size() != faces_per_cell)
                {
                  // run time exception handling
                  if (!(closestAtomIdSum ==
                        closestAtomId * nonHangingNodeIdCountCell))
                    {
                      std::cout
                        << "closestAtomIdSum: " << closestAtomIdSum
                        << ", closestAtomId: " << closestAtomId
                        << ", nonHangingNodeIdCountCell: "
                        << nonHangingNodeIdCountCell
                        << " cell center: " << cell->center()
                        << " is solved node present: " << isSolvedNodePresent
                        << std::endl;
                    }
                  AssertThrow(
                    closestAtomIdSum ==
                      closestAtomId * nonHangingNodeIdCountCell,
                    dealii::ExcMessage(
                      "DFT-FE Error: dofs of cells touching vself ball have different closest atom ids, remedy- increase separation between vself balls"));
                }
            } // cell locally owned
      }       // Bin loop
  }
} // namespace dftfe
