// ---------------------------------------------------------------------
//
// Copyright (c) 2017 The Regents of the University of Michigan and DFT-FE authors.
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
// @author  Sambit Das (2017)
//

template<unsigned int FEOrder>
void dftClass<FEOrder>::applyTotalPotentialDirichletBC()
			     
{
 
  const unsigned int vertices_per_cell=GeometryInfo<3>::vertices_per_cell;
  const unsigned int dofs_per_cell = FE.dofs_per_cell; 
  const unsigned int faces_per_cell=GeometryInfo<C_DIM>::faces_per_cell;
  const unsigned int dofs_per_face=FE.dofs_per_face;
  std::vector<bool> dofs_touched(dofHandler.n_dofs(),false);   
  DoFHandler<C_DIM>::active_cell_iterator cell = dofHandler.begin_active(),endc = dofHandler.end();
  for(; cell!= endc; ++cell)
  {
   if(cell->is_locally_owned())
   {
     std::vector<types::global_dof_index> cellGlobalDofIndices(dofs_per_cell);
     cell->get_dof_indices(cellGlobalDofIndices);	
     for(unsigned int iFace = 0; iFace < faces_per_cell; ++iFace)
     {
        const unsigned int boundaryId=cell->face(iFace)->boundary_id();
	if (boundaryId==0){
	  std::vector<types::global_dof_index> iFaceGlobalDofIndices(dofs_per_face);
	  cell->face(iFace)->get_dof_indices(iFaceGlobalDofIndices);			
          for(unsigned int iFaceDof = 0; iFaceDof < dofs_per_face; ++iFaceDof){
             unsigned int nodeId=iFaceGlobalDofIndices[iFaceDof];			  
	     if (dofs_touched[nodeId])
		 continue;
	     dofs_touched[nodeId]=true;
             if(!d_noConstraints.is_constrained(nodeId))
             {
	        d_constraintsForTotalPotential.add_line(nodeId);
	        d_constraintsForTotalPotential.set_inhomogeneity(nodeId,0);
	     }//non-hanging node check	     
          }//Face dof loop
	}//non-periodic boundary id
     }//Face loop
    }//cell locally owned
  }// cell loop

  return;

}
