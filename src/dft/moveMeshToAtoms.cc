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
// @author Shiva Rudraraju (2016), Phani Motamarri (2016)
//

//source file for all mesh reading/generation functions

//Generate triangulation.
template<unsigned int FEOrder>
void dftClass<FEOrder>::moveMeshToAtoms(parallel::distributed::Triangulation<3> & triangulationMove,bool isCoarserMove){
  meshMovementGaussianClass gaussianMove;
  gaussianMove.init(triangulationMove);

  if (!isCoarserMove){
    const int numberGlobalAtoms = atomLocations.size();
    const int numberImageAtoms = d_imageIds.size();

    std::vector<Point<3>> atomPoints;
    for (unsigned int iAtom=0;iAtom <numberGlobalAtoms; iAtom++){
       Point<3> atomCoor;
       atomCoor[0] = atomLocations[iAtom][2];
       atomCoor[1] = atomLocations[iAtom][3];
       atomCoor[2] = atomLocations[iAtom][4];
       atomPoints.push_back(atomCoor);
    } 

    gaussianMove.findClosestVerticesToDestinationPoints(atomPoints,
		                                     closestTriaVertexToAtomsLocation,
                                                     distanceClosestTriaVerticesToAtoms);


    //add control point locations and displacements corresponding to images
    for (unsigned int iImage=0;iImage <numberImageAtoms; iImage++){
      Point<3> imageCoor;
      Point<3> correspondingAtomCoor;
     
      imageCoor[0] = d_imagePositions[iImage][0];
      imageCoor[1] = d_imagePositions[iImage][1];
      imageCoor[2] = d_imagePositions[iImage][2];
      const int atomId=d_imageIds[iImage];
      correspondingAtomCoor[0] = atomLocations[atomId][2];
      correspondingAtomCoor[1] = atomLocations[atomId][3];
      correspondingAtomCoor[2] = atomLocations[atomId][4];
   
      closestTriaVertexToAtomsLocation.push_back(Point<3>(closestTriaVertexToAtomsLocation[atomId]+correspondingAtomCoor-imageCoor));
      distanceClosestTriaVerticesToAtoms.push_back(distanceClosestTriaVerticesToAtoms[atomId]);
     }
	
  }
 
  const double gaussianConstant=0.5;
  gaussianMove.moveMesh(closestTriaVertexToAtomsLocation,
		        distanceClosestTriaVerticesToAtoms,
			gaussianConstant);

  //print out mesh metrics
  typename parallel::distributed::Triangulation<3>::active_cell_iterator cell, endc;
  double minElemLength=1e+6;
  cell = triangulationMove.begin_active();
  endc = triangulationMove.end();
  for ( ; cell != endc; ++cell){
    if (cell->is_locally_owned()){
      if (cell->minimum_vertex_distance()<minElemLength) minElemLength = cell->minimum_vertex_distance();
    }
  }
  Utilities::MPI::min(minElemLength, mpi_communicator);
  char buffer[100];
  sprintf(buffer, "Mesh movement quality metric, h_min: %5.2e\n", minElemLength);
  pcout << buffer;  
}

	

