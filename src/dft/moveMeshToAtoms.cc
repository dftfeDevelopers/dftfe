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
void dftClass<FEOrder>::moveMeshToAtoms(Triangulation<3,3> & triangulationMove,bool reuse)
{

  meshMovementGaussianClass gaussianMove(mpi_communicator);
  gaussianMove.init(triangulationMove,d_domainBoundingVectors);


  if(!reuse)
  {
      const int numberGlobalAtoms = atomLocations.size();
      const int numberImageAtoms = d_imageIds.size();

      std::vector<Point<3>> atomPoints;
      for (unsigned int iAtom=0;iAtom <numberGlobalAtoms; iAtom++)
      {
	  Point<3> atomCoor;
	  atomCoor[0] = atomLocations[iAtom][2];
	  atomCoor[1] = atomLocations[iAtom][3];
	  atomCoor[2] = atomLocations[iAtom][4];
	  atomPoints.push_back(atomCoor);
      } 


      gaussianMove.findClosestVerticesToDestinationPoints(atomPoints,
		                                          closestTriaVertexToAtomsLocation,
                                                          dispClosestTriaVerticesToAtoms);
      /*
      for (unsigned int iAtom=0;iAtom <numberGlobalAtoms; iAtom++)
      {
  	  pcout<< " atomId: "<< iAtom << " disp: "<< dispClosestTriaVerticesToAtoms[iAtom] <<std::endl;
      }       
      */

      //add control point locations and displacements corresponding to images
      for(unsigned int iImage=0;iImage <numberImageAtoms; iImage++)
      {
	  Point<3> imageCoor;
	  Point<3> correspondingAtomCoor;
     
	  imageCoor[0] = d_imagePositions[iImage][0];
	  imageCoor[1] = d_imagePositions[iImage][1];
	  imageCoor[2] = d_imagePositions[iImage][2];
	  const int atomId=d_imageIds[iImage];
	  correspondingAtomCoor[0] = atomLocations[atomId][2];
	  correspondingAtomCoor[1] = atomLocations[atomId][3];
	  correspondingAtomCoor[2] = atomLocations[atomId][4];
  

          Point<3> temp=closestTriaVertexToAtomsLocation[atomId]+(correspondingAtomCoor-imageCoor);
          closestTriaVertexToAtomsLocation.push_back(temp);
          dispClosestTriaVerticesToAtoms.push_back(dispClosestTriaVerticesToAtoms[atomId]);
       }
  }
 
  const double gaussianConstant=0.5;
  std::pair<bool,double> meshQualityMetrics=gaussianMove.moveMesh(closestTriaVertexToAtomsLocation,
		                                                  dispClosestTriaVerticesToAtoms,
			                                          gaussianConstant);

  AssertThrow(!meshQualityMetrics.first,ExcMessage("Negative jacobian created after moving closest nodes to atoms. Suggestion: increase refinement near atoms"));
  pcout<< " Mesh quality check: maximum jacobian ratio: "<< meshQualityMetrics.second<<std::endl;
}

	

