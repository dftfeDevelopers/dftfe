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
// @author Sambit Das (2017)
//

//source file for locating core atom nodes
template<unsigned int FEOrder>
void forceClass<FEOrder>::locateAtomCoreNodesForce(){
  d_atomsForceDofs.clear();
  const std::vector<std::vector<double> > & atomLocations=dftPtr->atomLocations;	
  unsigned int vertices_per_cell=GeometryInfo<3>::vertices_per_cell;
  //
  //locating atom nodes
  unsigned int numAtoms=atomLocations.size();
  std::set<unsigned int> atomsTolocate;
  for (unsigned int i = 0; i < numAtoms; i++) atomsTolocate.insert(i);
  
  //loop over all atoms to locate the corresponding nodes
  for (std::set<unsigned int>::iterator it=atomsTolocate.begin(); it!=atomsTolocate.end(); ++it){ 
    Point<3> atomCoord(atomLocations[*it][2],atomLocations[*it][3],atomLocations[*it][4]);	  
    //element loop
    bool isFound=false;
    DoFHandler<3>::active_cell_iterator
    cell = d_dofHandlerForce.begin_active(),
    endc = d_dofHandlerForce.end();
    for (; cell!=endc; ++cell) {
      if (cell->is_locally_owned()){
        for (unsigned int i=0; i<vertices_per_cell; ++i){
	  Point<3> feNodeGlobalCoord = cell->vertex(i);

	  if(feNodeGlobalCoord.distance(atomCoord) < 1.0e-5){

	      for (unsigned int idim=0; idim < C_DIM ; idim++){
                const unsigned int forceNodeId=cell->vertex_dof_index(i,idim);		  
	        if (d_locally_owned_dofsForce.is_element(forceNodeId)){            
 	           //std::cout << "Atom nodal coordinates (" << feNodeGlobalCoord << " ,"<< atomCoord <<") associated with force node id " << forceNodeId << " , force component: "<< idim << " in processor " << this_mpi_process << " and added \n";

                 d_atomsForceDofs[std::pair<unsigned int,unsigned int>(*it,idim)]=forceNodeId;
	       }
	      }
	      isFound=true;         
	      break;
	  }//tolerance check if loop
        }//vertices_per_cell loop
      }//locally owned cell if loop
      if (isFound)
      break;
    }//cell loop
  }//atomsToLocate loop
  MPI_Barrier(mpi_communicator);

  const unsigned int totalForceNodesFound = Utilities::MPI::sum(d_atomsForceDofs.size(), mpi_communicator);
  AssertThrow(totalForceNodesFound==numAtoms*3,ExcMessage("Atleast one atom doesn't lie on force dof handler dof"));
}
