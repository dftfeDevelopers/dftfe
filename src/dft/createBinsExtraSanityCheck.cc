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
void dftClass<FEOrder>::createAtomBinsExtraSanityCheck()
{
  const unsigned int faces_per_cell=GeometryInfo<C_DIM>::faces_per_cell;
  const unsigned int dofs_per_cell=FE.dofs_per_cell;
  const unsigned int dofs_per_face=FE.dofs_per_face;
  const int numberBins=d_bins.size();

  for(int iBin = 0; iBin < numberBins; ++iBin)
  {

     std::map<dealii::types::global_dof_index, int> & boundaryNodeMap = d_boundaryFlag[iBin];
     std::map<dealii::types::global_dof_index, int> & closestAtomBinMap = d_closestAtomBin[iBin];
     DoFHandler<C_DIM>::active_cell_iterator cell = dofHandler.begin_active(),endc = dofHandler.end();
     for(; cell!= endc; ++cell)
     {
	if(cell->is_locally_owned())
	{

	  std::vector<types::global_dof_index> cell_dof_indices(dofs_per_cell);
	  cell->get_dof_indices(cell_dof_indices);	

	  bool isSolvedNodePresent=false;
	  int numSolvedNodes=0;
	  int closestChargeIdSolvedSum=0;
	  int closestChargeIdSolvedNode=-1;
	  for(unsigned int iNode = 0; iNode < dofs_per_cell; ++iNode)
	    {
	      const int globalNodeId=cell_dof_indices[iNode];
	      if(!d_noConstraints.is_constrained(globalNodeId))
	      {		  
		const int boundaryId=d_boundaryFlag[iBin][globalNodeId];
		if(boundaryId != -1)
		{
		    isSolvedNodePresent=true;
		    closestChargeIdSolvedNode=boundaryId;
		    numSolvedNodes++;
		    closestChargeIdSolvedSum+=boundaryId;
		}
		
	      }
	      
	    }//element node loop
           Assert(numSolvedNodes*closestChargeIdSolvedNode==closestChargeIdSolvedSum,ExcMessage("BUG"));

	   std::vector<unsigned int> dirichletFaceIds;
	   unsigned int closestAtomIdSum=0;
	   unsigned int closestAtomId=0;
	   unsigned int nonHangingNodeIdCountCell=0;
	   for(unsigned int iFace = 0; iFace < faces_per_cell; ++iFace)
           {
              int dirichletDofCount=0;
	      unsigned int nonHangingNodeIdCountFace=0;	      
	      std::vector<types::global_dof_index> iFaceGlobalDofIndices(dofs_per_face);
	      cell->face(iFace)->get_dof_indices(iFaceGlobalDofIndices);		      
	      for(unsigned int iFaceDof = 0; iFaceDof < dofs_per_face; ++iFaceDof){
                 unsigned int nodeId=iFaceGlobalDofIndices[iFaceDof];
		 if (!d_noConstraints.is_constrained(nodeId)){
	            Assert(boundaryNodeMap.find(nodeId)!=boundaryNodeMap.end(),ExcMessage("BUG"));
                    Assert(closestAtomBinMap.find(nodeId)!=closestAtomBinMap.end(),ExcMessage("BUG"));
		    dirichletDofCount+=boundaryNodeMap[nodeId];
		    closestAtomId=closestAtomBinMap[nodeId];
		    closestAtomIdSum+=closestAtomId;
		    nonHangingNodeIdCountCell++;
		    nonHangingNodeIdCountFace++;
	         }//non-hanging node check 

	      }//Face dof loop
              
	      if (dirichletDofCount== -nonHangingNodeIdCountFace)
	         dirichletFaceIds.push_back(iFace);

	   }//Face loop
	   
	   if (dirichletFaceIds.size()!=faces_per_cell){
	      //run time exception handling
	      if (!(closestAtomIdSum==closestAtomId*nonHangingNodeIdCountCell))
	      {
		  std::cout << "closestAtomIdSum: "<<closestAtomIdSum<< ", closestAtomId: "<<closestAtomId<< ", nonHangingNodeIdCountCell: "<<nonHangingNodeIdCountCell<< " cell center: "<< cell->center() << " is solved node present: "<< isSolvedNodePresent<< std::endl;
	      }
	      AssertThrow(closestAtomIdSum==closestAtomId*nonHangingNodeIdCountCell,ExcMessage("dofs of cells touching vself ball have different closest atom ids, remedy- increase separation between vself balls"));		   
	   }
	}//cell locally owned
     }// cell loop
  }//Bin loop

}
//
