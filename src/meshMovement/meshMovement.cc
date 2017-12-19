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
#include "../../include/meshMovement.h"
//
//constructor
//
meshMovementClass::meshMovementClass():
  FEMoveMesh(FE_Q<3>(QGaussLobatto<1>(2)), 3), //linear shape function
  mpi_communicator(MPI_COMM_WORLD),	
  this_mpi_process (Utilities::MPI::this_mpi_process(mpi_communicator)),
  pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
{

}

void meshMovementClass::init(parallel::distributed::Triangulation<3> & triangulation)
{
  d_dofHandlerMoveMesh.initialize(triangulation,FEMoveMesh);		
  d_dofHandlerMoveMesh.distribute_dofs(FEMoveMesh);
  d_locally_owned_dofs.clear();d_locally_relevant_dofs.clear();
  d_locally_owned_dofs = d_dofHandlerMoveMesh.locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs(d_dofHandlerMoveMesh, d_locally_relevant_dofs);  

  d_constraintsHangingNodes.clear();
  DoFTools::make_hanging_node_constraints(d_dofHandlerMoveMesh, d_constraintsHangingNodes);
  d_constraintsHangingNodes.close();

  d_constraintsMoveMesh.clear();
  DoFTools::make_hanging_node_constraints(d_dofHandlerMoveMesh, d_constraintsMoveMesh);   
#ifdef ENABLE_PERIODIC_BC
  for (int i = 0; i < C_DIM; ++i)
    {
      GridTools::collect_periodic_faces(d_dofHandlerMoveMesh, /*b_id1*/ 2*i+1, /*b_id2*/ 2*i+2,/*direction*/ i, d_periodicity_vector);
    }
  DoFTools::make_periodicity_constraints<DoFHandler<C_DIM> >(d_periodicity_vector, d_constraintsMoveMesh);
  d_constraintsMoveMesh.close();
#else
  d_constraintsMoveMesh.close();
#endif	
}

void meshMovementClass::reinit(parallel::distributed::Triangulation<3> & triangulation,
		               bool isTriaRefined)
{
  if (isTriaRefined){
    d_dofHandlerMoveMesh.clear();
    d_dofHandlerMoveMesh.initialize(triangulation,FEMoveMesh);	
    d_dofHandlerMoveMesh.distribute_dofs(FEMoveMesh);	  
    d_locally_owned_dofs.clear();d_locally_relevant_dofs.clear();
    d_locally_owned_dofs = d_dofHandlerMoveMesh.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(d_dofHandlerMoveMesh, d_locally_relevant_dofs);  

    d_constraintsHangingNodes.clear();
    DoFTools::make_hanging_node_constraints(d_dofHandlerMoveMesh, d_constraintsHangingNodes);
    d_constraintsHangingNodes.close();

    d_constraintsMoveMesh.clear();
    DoFTools::make_hanging_node_constraints(d_dofHandlerMoveMesh, d_constraintsMoveMesh); 
    d_periodicity_vector.clear();
#ifdef ENABLE_PERIODIC_BC
    for (int i = 0; i < C_DIM; ++i)
    {
       GridTools::collect_periodic_faces(d_dofHandlerMoveMesh, /*b_id1*/ 2*i+1, /*b_id2*/ 2*i+2,/*direction*/ i, d_periodicity_vector);
    }
    DoFTools::make_periodicity_constraints<DoFHandler<C_DIM> >(d_periodicity_vector, d_constraintsMoveMesh);
    d_constraintsMoveMesh.close();
#else
    d_constraintsMoveMesh.close();
#endif
  }
  else
  {
    d_dofHandlerMoveMesh.distribute_dofs(FEMoveMesh);		  
  }
}

void meshMovementClass::initIncrementField()
{
  //dftPtr->matrix_free_data.initialize_dof_vector(d_incrementalDisplacement,d_forceDofHandlerIndex);	
  IndexSet  ghost_indices=d_locally_relevant_dofs;
  ghost_indices.subtract_set(d_locally_owned_dofs);
  d_incrementalDisplacement=parallel::distributed::Vector<double>::Vector(d_locally_owned_dofs                                                                                              ,ghost_indices,
                                                                          mpi_communicator);	  
  d_incrementalDisplacement=0;  	
}


void meshMovementClass::finalizeIncrementField()
{
  //d_incrementalDisplacement.compress(VectorOperation::insert);//inserts current value at owned node and sets ghosts to zero	
  d_constraintsMoveMesh.distribute(d_incrementalDisplacement);//distribute to constrained degrees of freedom (periodic and hanging nodes)
  d_incrementalDisplacement.update_ghost_values();
}

void meshMovementClass::updateTriangulationVertices()
{
  MPI_Barrier(mpi_communicator); 
  pcout << "Start moving triangulation..." << std::endl;
  std::vector<bool> vertex_moved(d_dofHandlerMoveMesh.get_tria().n_vertices(),
                                 false);
  
  // Next move vertices on locally owned cells
  DoFHandler<3>::active_cell_iterator   cell = d_dofHandlerMoveMesh.begin_active();
  DoFHandler<3>::active_cell_iterator   endc = d_dofHandlerMoveMesh.end();
  for (; cell!=endc; ++cell) {
     if (!cell->is_artificial())
     {
	for (unsigned int vertex_no=0; vertex_no<GeometryInfo<C_DIM>::vertices_per_cell;++vertex_no)
	 {
	     const unsigned global_vertex_no = cell->vertex_index(vertex_no);

	     if (vertex_moved[global_vertex_no])
	       continue;	    

	     Point<C_DIM> vertexDisplacement;
	     for (unsigned int d=0; d<C_DIM; ++d){
		const unsigned int globalDofIndex= cell->vertex_dof_index(vertex_no,d);
	     	vertexDisplacement[d]=d_incrementalDisplacement[globalDofIndex];
	     }
			
	     cell->vertex(vertex_no) += vertexDisplacement;
	     vertex_moved[global_vertex_no] = true;
	  }
      }
  }
  pcout << "...End moving triangulation" << std::endl;
  //dftPtr->triangulation.communicate_locally_moved_vertices(locally_owned_vertices);
}

void meshMovementClass::periodicSanityCheck()
{
  //sanity check to make sure periodic boundary conditions are maintained
  MPI_Barrier(mpi_communicator); 
#ifdef ENABLE_PERIODIC_BC
  pcout << "Sanity check for periodic matched faces on moved triangulation..." << std::endl;  
  for(unsigned int i=0; i< d_periodicity_vector.size(); ++i) 
  {
    if (d_periodicity_vector[i].cell[0]->is_artificial() || d_periodicity_vector[i].cell[1]->is_artificial())
       continue;

    std::vector<bool> isPeriodicFace(3);	  
    for(unsigned int idim=0; idim<3; ++idim){
        isPeriodicFace[idim]=GridTools::orthogonal_equality(d_periodicity_vector[i].cell[0]->face(d_periodicity_vector[i].face_idx[0]),d_periodicity_vector[i].cell[1]->face(d_periodicity_vector[i].face_idx[1]),idim);
    }
	      
    AssertThrow(isPeriodicFace[0]==true || isPeriodicFace[1]==true || isPeriodicFace[2]==true,ExcMessage("Previously periodic matched face pairs not matching periodically for any directions after mesh movement"));			    
  }
  MPI_Barrier(mpi_communicator);  
  pcout << "...Sanity check passed" << std::endl;
#endif
}


void meshMovementClass::findClosestVerticesToDestinationPoints(const std::vector<Point<3>> & destinationPoints,
		                                               std::vector<Point<3>> & closestTriaVertexToDestPointsLocation,
                                                               std::vector<Point<3>> & distanceClosestTriaVerticesToDestPoints)
{
  closestTriaVertexToDestPointsLocation.clear();
  distanceClosestTriaVerticesToDestPoints.clear();
  unsigned int vertices_per_cell=GeometryInfo<C_DIM>::vertices_per_cell;
  
  for (unsigned int idest=0;idest <destinationPoints.size(); idest++){

      double minDistance=1e+6;
      Point<3> closestTriaVertexLocation;

      std::vector<bool> vertex_touched(d_dofHandlerMoveMesh.get_tria().n_vertices(),
                                       false);      
      DoFHandler<3>::active_cell_iterator
      cell = d_dofHandlerMoveMesh.begin_active(),
      endc = d_dofHandlerMoveMesh.end();      
      for (; cell!=endc; ++cell) {
       if (cell->is_locally_owned()){
        for (unsigned int i=0; i<vertices_per_cell; ++i){
            const unsigned global_vertex_no = cell->vertex_index(i);

	   if (vertex_touched[global_vertex_no])
	       continue;		   
           vertex_touched[global_vertex_no]=true;

	   if(d_constraintsHangingNodes.is_constrained(cell->vertex_dof_index(i,0))
	      || !d_locally_owned_dofs.is_element(cell->vertex_dof_index(i,0))){
	          continue;
	    }

	    Point<C_DIM> nodalCoor = cell->vertex(i);

            const double distance=(nodalCoor-destinationPoints[idest]).norm();

	    if (distance < minDistance)
	        minDistance=distance;
         }
       }
      }
      const double globalMinDistance=Utilities::MPI::min(minDistance, mpi_communicator);
      
      if ((minDistance-globalMinDistance)>1e-5){
	  closestTriaVertexLocation=Point<3>(0);
      }



      Point<3> closestTriaVertexLocationGlobal;
      // accumulate value
      MPI_Allreduce(&(closestTriaVertexLocation[0]),
		    &(closestTriaVertexLocationGlobal[0]),
		    3,
		    MPI_DOUBLE,
		    MPI_SUM,
		    mpi_communicator);

      closestTriaVertexToDestPointsLocation.push_back(closestTriaVertexLocationGlobal);
      distanceClosestTriaVerticesToDestPoints.push_back(Point<3>(destinationPoints[idest]-closestTriaVertexLocationGlobal));
  }
}	
