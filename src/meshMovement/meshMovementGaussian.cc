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
//
#include "../../include/meshMovementGaussian.h"

meshMovementGaussianClass::meshMovementGaussianClass()
{
}

void meshMovementGaussianClass::moveMesh(std::vector<Point<C_DIM> > controlPointLocations,
                                         std::vector<Tensor<1,C_DIM,double> > controlPointDisplacements,
                                         double controllingParameter)   
{
  d_controlPointLocations=controlPointLocations;
  d_controlPointDisplacements=controlPointDisplacements;
  d_controllingParameter=controllingParameter;	
  MPI_Barrier(mpi_communicator); 
  pcout << "Computing triangulation displacement increment caused by gaussian generator displacements..." << std::endl;
  initIncrementField();
  computeIncrement();	
  finalizeIncrementField();
  pcout << "...Computed triangulation displacement increment" << std::endl;	
  updateTriangulationVertices();
  movedMeshCheck();
}



//The triangulation nodes corresponding to control point location are constrained to only 
//their corresponding controlPointDisplacements. In other words for those nodes we don't consider overlapping 
//Gaussians
void meshMovementGaussianClass::computeIncrement()
{
  unsigned int vertices_per_cell=GeometryInfo<C_DIM>::vertices_per_cell;
  
  for (unsigned int iControl=0;iControl <d_controlPointLocations.size(); iControl++){

      std::vector<bool> vertex_touched(d_dofHandlerMoveMesh.get_tria().n_vertices(),
                                       false);      
      DoFHandler<3>::active_cell_iterator
      cell = d_dofHandlerMoveMesh.begin_active(),
      endc = d_dofHandlerMoveMesh.end();      
      for (; cell!=endc; ++cell) {
       if (!cell->is_artificial()){
        for (unsigned int i=0; i<vertices_per_cell; ++i){
            const unsigned global_vertex_no = cell->vertex_index(i);

	    if (vertex_touched[global_vertex_no])
	       continue;	    
            vertex_touched[global_vertex_no]=true;
	    Point<C_DIM> nodalCoor = cell->vertex(i);
            const double rsq=(nodalCoor-d_controlPointLocations[iControl]).norm_square();
	    bool isGaussianOverlapOtherControlPoint=false;
	    for (unsigned int jControl=0;jControl <d_controlPointLocations.size(); jControl++){
	      if (jControl !=iControl){
                 const double distanceSq=(nodalCoor-d_controlPointLocations[jControl]).norm_square();
		 if (distanceSq < 1e-6){
		    isGaussianOverlapOtherControlPoint=true;
		    break;
		 }
	      }
	    }
	    if (isGaussianOverlapOtherControlPoint)
	       continue;
	    const double gaussianWeight=std::exp(-d_controllingParameter*rsq);
	    for (unsigned int idim=0; idim < C_DIM ; idim++){
              const unsigned int globalDofIndex=cell->vertex_dof_index(i,idim);

	      if(!d_constraintsMoveMesh.is_constrained(globalDofIndex)){
	           d_incrementalDisplacement[globalDofIndex]+=gaussianWeight*d_controlPointDisplacements[iControl][idim];
	      }

	   }
         }
       }
      }
  }
}

