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
  const unsigned int numberBins=dftPtr->d_bins.size();
  //clear exisitng data
  d_cellsVselfBallsDofHandler.clear();
  d_cellsVselfBallsDofHandlerForce.clear();
  d_cellFacesVselfBallSurfacesDofHandler.clear();
  d_cellFacesVselfBallSurfacesDofHandlerForce.clear();
  d_cellsVselfBallsClosestAtomIdDofHandler.clear();
  d_AtomIdBinIdLocalDofHandler.clear();
  //resize
  d_cellsVselfBallsDofHandler.resize(numberBins);
  d_cellsVselfBallsDofHandlerForce.resize(numberBins);
  d_cellFacesVselfBallSurfacesDofHandler.resize(numberBins);
  d_cellFacesVselfBallSurfacesDofHandlerForce.resize(numberBins);
  d_cellsVselfBallsClosestAtomIdDofHandler.resize(numberBins);

  for(unsigned int iBin = 0; iBin < numberBins; ++iBin)
  {

     std::map<dealii::types::global_dof_index, int> & boundaryNodeMap = dftPtr->d_boundaryFlag[iBin];
     std::map<dealii::types::global_dof_index, int> & closestAtomBinMap = dftPtr->d_closestAtomBin[iBin];
     DoFHandler<C_DIM>::active_cell_iterator cell = dftPtr->dofHandler.begin_active(),endc = dftPtr->dofHandler.end();
     DoFHandler<C_DIM>::active_cell_iterator cellForce = d_dofHandlerForce.begin_active();
     for(; cell!= endc; ++cell, ++cellForce)
     {
	if(cell->is_locally_owned())
	{
	   std::vector<unsigned int> dirichletFaceIds;
	   std::vector<unsigned int> faceIdsWithAtleastOneSolvedNonHangingNode;	 
	   std::vector<unsigned int> allFaceIdsOfCell;	
	   unsigned int closestAtomIdSum=0;
	   unsigned int closestAtomId;
	   unsigned int nonHangingNodeIdCountCell=0;
	   for(unsigned int iFace = 0; iFace < faces_per_cell; ++iFace)
           {
              int dirichletDofCount=0;
	      bool isSolvedDofPresent=false;
	      int nonHangingNodeIdCountFace=0;	      
	      std::vector<types::global_dof_index> iFaceGlobalDofIndices(dofs_per_face);
	      cell->face(iFace)->get_dof_indices(iFaceGlobalDofIndices);		      
	      for(unsigned int iFaceDof = 0; iFaceDof < dofs_per_face; ++iFaceDof)
	      {
                 const types::global_dof_index nodeId=iFaceGlobalDofIndices[iFaceDof];
		 if (!dftPtr->d_noConstraints.is_constrained(nodeId))
		 {
	            Assert(boundaryNodeMap.find(nodeId)!=boundaryNodeMap.end(),ExcMessage("BUG"));
                    Assert(closestAtomBinMap.find(nodeId)!=closestAtomBinMap.end(),ExcMessage("BUG"));

		    if (boundaryNodeMap[nodeId]!=-1)
			isSolvedDofPresent=true;
		    else
			dirichletDofCount+=boundaryNodeMap[nodeId];

		    closestAtomId=closestAtomBinMap[nodeId];
		    closestAtomIdSum+=closestAtomId;
		    nonHangingNodeIdCountCell++;
		    nonHangingNodeIdCountFace++;
	         }//non-hanging node check 

	      }//Face dof loop
              
	      if (isSolvedDofPresent)
	      {
	         faceIdsWithAtleastOneSolvedNonHangingNode.push_back(iFace);
	      }
	      if (dirichletDofCount<0)
	      {
	         dirichletFaceIds.push_back(iFace);
              } 
              allFaceIdsOfCell.push_back(iFace);		      
              
	   }//Face loop
           
	   //fill the target objects
	   if (faceIdsWithAtleastOneSolvedNonHangingNode.size()>0){
	      if (!(closestAtomIdSum==closestAtomId*nonHangingNodeIdCountCell))
	      {
		  std::cout << "closestAtomIdSum: "<<closestAtomIdSum<< ", closestAtomId: "<<closestAtomId<< ", nonHangingNodeIdCountCell: "<<nonHangingNodeIdCountCell<<std::endl;
	      }
	      AssertThrow(closestAtomIdSum==closestAtomId*nonHangingNodeIdCountCell,ExcMessage("cell dofs on vself ball surface have different closest atom ids, remedy- increase separation between vself balls"));		   
	      d_cellsVselfBallsDofHandler[iBin].push_back(cell);
	      d_cellsVselfBallsDofHandlerForce[iBin].push_back(cellForce);
	      d_cellsVselfBallsClosestAtomIdDofHandler[iBin][cell->id()]=closestAtomId;
	      d_AtomIdBinIdLocalDofHandler[closestAtomId]=iBin;
	      d_cellFacesVselfBallSurfacesDofHandler[iBin][cell]= allFaceIdsOfCell;
	      d_cellFacesVselfBallSurfacesDofHandlerForce[iBin][cellForce]= dirichletFaceIds;//allFaceIdsOfCell;
	   }
	}//cell locally owned
     }// cell loop
  }//Bin loop

}
//
