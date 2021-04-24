// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2018 The Regents of the University of Michigan and DFT-FE
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
// @author  Sambit Das
//

template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
dftClass<FEOrder, FEOrderElectro>::applyHomogeneousDirichletBC(
  const dealii::DoFHandler<3> &            _dofHandler,
  const dealii::AffineConstraints<double> &onlyHangingNodeConstraints,
  dealii::AffineConstraints<double> &      constraintMatrix)

{
  dealii::IndexSet locallyRelevantDofs;
  dealii::DoFTools::extract_locally_relevant_dofs(_dofHandler,
                                                  locallyRelevantDofs);

  const unsigned int vertices_per_cell =
    dealii::GeometryInfo<3>::vertices_per_cell;
  const unsigned int dofs_per_cell  = _dofHandler.get_fe().dofs_per_cell;
  const unsigned int faces_per_cell = dealii::GeometryInfo<3>::faces_per_cell;
  const unsigned int dofs_per_face  = _dofHandler.get_fe().dofs_per_face;

  std::vector<dealii::types::global_dof_index> cellGlobalDofIndices(
    dofs_per_cell);
  std::vector<dealii::types::global_dof_index> iFaceGlobalDofIndices(
    dofs_per_face);

  std::vector<bool> dofs_touched(_dofHandler.n_dofs(), false);
  dealii::DoFHandler<3>::active_cell_iterator cell = _dofHandler.begin_active(),
                                              endc = _dofHandler.end();
  for (; cell != endc; ++cell)
    if (cell->is_locally_owned() || cell->is_ghost())
      {
        cell->get_dof_indices(cellGlobalDofIndices);
        for (unsigned int iFace = 0; iFace < faces_per_cell; ++iFace)
          {
            const unsigned int boundaryId = cell->face(iFace)->boundary_id();
            if (boundaryId == 0)
              {
                cell->face(iFace)->get_dof_indices(iFaceGlobalDofIndices);
                for (unsigned int iFaceDof = 0; iFaceDof < dofs_per_face;
                     ++iFaceDof)
                  {
                    const dealii::types::global_dof_index nodeId =
                      iFaceGlobalDofIndices[iFaceDof];
                    if (dofs_touched[nodeId])
                      continue;
                    dofs_touched[nodeId] = true;
                    if (!onlyHangingNodeConstraints.is_constrained(nodeId))
                      {
                        constraintMatrix.add_line(nodeId);
                        constraintMatrix.set_inhomogeneity(nodeId, 0);
                      } // non-hanging node check
                  }     // Face dof loop
              }         // non-periodic boundary id
          }             // Face loop
      }                 // cell locally owned
}
