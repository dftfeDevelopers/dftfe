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
// @author Sambit Das(2017)
//


template<unsigned int FEOrder>
void forceClass<FEOrder>::createBinObjectsForce()
{
  const unsigned int faces_per_cell=GeometryInfo<C_DIM>::faces_per_cell;
  const unsigned int dofs_per_cell=dftPtr->FE.dofs_per_cell;
  const unsigned int dofs_per_face=dftPtr->FE.dofs_per_face;
  const int numberBins=dftPtr->d_bins.size();
  //clear exisitng data
  d_cellsVselfBallsDofHandler.clear();
  d_cellsVselfBallsDofHandlerForce.clear();
  d_cellFacesVselfBallSurfacesDofHandler.clear();
  d_cellFacesVselfBallSurfacesDofHandlerForce.clear();
  //resize
  d_cellsVselfBallsDofHandler.resize(numberBins);
  d_cellsVselfBallsDofHandlerForce.resize(numberBins);
  d_cellFacesVselfBallSurfacesDofHandler.resize(numberBins);
  d_cellFacesVselfBallSurfacesDofHandlerForce.resize(numberBins);

  for(int iBin = 0; iBin < numberBins; ++iBin)
  {

     std::map<dealii::types::global_dof_index, int> & boundaryNodeMap = dftPtr->d_boundaryFlag[iBin];
     std::map<dealii::types::global_dof_index, int> & closestAtomBinMap = dftPtr->d_closestAtomBin[iBin];
     DoFHandler<C_DIM>::active_cell_iterator cell = dftPtr->dofHandler.begin_active(),endc = dftPtr->dofHandler.end();
     DoFHandler<C_DIM>::active_cell_iterator cellForce = d_dofHandlerForce.begin_active();
     for(; cell!= endc; ++cell)
     {
	if(cell->is_locally_owned())
	{
	   std::vector<types::global_dof_index> cellGlobalDofIndices(dofs_per_cell);
	   cell->get_dof_indices(cellGlobalDofIndices);	
	   std::vector<unsigned int> dirichletFaceIds;
	   int closestAtomIdSum=0;
	   int closestAtomId=closestAtomBinMap[cellGlobalDofIndices[0]];
	   for(unsigned int iFace = 0; iFace < faces_per_cell; ++iFace)
           {
              int dirichletDofCount=0;
	      std::vector<types::global_dof_index> iFaceGlobalDofIndices(dofs_per_face);
	      cell->face(iFace)->get_dof_indices(iFaceGlobalDofIndices);		      
	      for(unsigned int iFaceDof = 0; iFaceDof < dofs_per_face; ++iFaceDof){
                 unsigned int nodeId=iFaceGlobalDofIndices[iFaceDof];		      
		 //unsigned int iCellDof=dftPtr->FE.face_to_cell_index(iFaceDof,iFace,cell->face_orientation(iFace),cell->face_flip(iFace),cell->face_rotation(iFace));// FIXME: throws error in debug mode for FEOrder > 2
                 //unsigned int nodeId2=cellGlobalDofIndices[iCellDof];		  
	         //AssertThrow(nodeId2==nodeId,ExcMessage("BUG"));	
		 dirichletDofCount+=boundaryNodeMap[nodeId];
		 closestAtomIdSum+=closestAtomBinMap[nodeId];

	      }//Face dof loop
              
	      if (dirichletDofCount== -dofs_per_face)
	         dirichletFaceIds.push_back(iFace);

	   }//Face loop
           

	   //fill the target objects
	   if (dirichletFaceIds.size()!=faces_per_cell){
	      //run time exception handling
	      AssertThrow(closestAtomIdSum==closestAtomId*dofs_per_face*faces_per_cell,ExcMessage("cell dofs on vself ball surface have different closest atom ids, remedy- increase separation between vself balls"));		   
	      d_cellsVselfBallsDofHandler[iBin].push_back(cell);
	      d_cellsVselfBallsDofHandlerForce[iBin].push_back(cellForce);
	      if (dirichletFaceIds.size()!=0){
	        d_cellFacesVselfBallSurfacesDofHandler[iBin][cell]=dirichletFaceIds;
		d_cellFacesVselfBallSurfacesDofHandlerForce[iBin][cellForce]=dirichletFaceIds;
	      }
	   }
	}//cell locally owned
	++cellForce;
     }// cell loop
  }//Bin loop

}
//
