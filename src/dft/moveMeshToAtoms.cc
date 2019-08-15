// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2018 The Regents of the University of Michigan and DFT-FE authors.
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
// @author Sambit Das
//

template<unsigned int FEOrder>
void dftClass<FEOrder>::moveMeshToAtoms(Triangulation<3,3> & triangulationMove,
		                        Triangulation<3,3> & triangulationSerial,
					bool reuseClosestTriaVertices,
					bool moveSubdivided)
{
  dealii::ConditionalOStream pcout_movemesh (std::cout, (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0));
  dealii::TimerOutput timer_movemesh(mpi_communicator,
	                             pcout_movemesh,
                                     dftParameters::reproducible_output ||
                                     dftParameters::verbosity<4 ? dealii::TimerOutput::never:
                                     dealii::TimerOutput::summary,dealii::TimerOutput::wall_times);

  meshMovementGaussianClass gaussianMove(mpi_communicator);
  gaussianMove.init(triangulationMove,
		    triangulationSerial,
		    d_domainBoundingVectors);

  const unsigned int numberGlobalAtoms = atomLocations.size();
  const unsigned int numberImageAtoms = d_imageIds.size();

  std::vector<Point<3>> atomPoints;
  for (unsigned int iAtom=0;iAtom <numberGlobalAtoms; iAtom++)
  {
      Point<3> atomCoor;
      atomCoor[0] = atomLocations[iAtom][2];
      atomCoor[1] = atomLocations[iAtom][3];
      atomCoor[2] = atomLocations[iAtom][4];
      atomPoints.push_back(atomCoor);
  }

  std::vector<Point<3>> closestTriaVertexToAtomsLocation;
  std::vector<Tensor<1,3,double> > dispClosestTriaVerticesToAtoms;

  timer_movemesh.enter_section("move mesh to atoms: find closest vertices");
  if(reuseClosestTriaVertices)
    {
      closestTriaVertexToAtomsLocation = d_closestTriaVertexToAtomsLocation;
      dispClosestTriaVerticesToAtoms = d_dispClosestTriaVerticesToAtoms;
    }
  else
    {
      gaussianMove.findClosestVerticesToDestinationPoints(atomPoints,
							  closestTriaVertexToAtomsLocation,
							  dispClosestTriaVerticesToAtoms);
    }
  timer_movemesh.exit_section("move mesh to atoms: find closest vertices");


  timer_movemesh.enter_section("move mesh to atoms: move mesh");
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


      const dealii::Point<3> temp=closestTriaVertexToAtomsLocation[atomId]+(imageCoor-correspondingAtomCoor);
      closestTriaVertexToAtomsLocation.push_back(temp);
      dispClosestTriaVerticesToAtoms.push_back(dispClosestTriaVerticesToAtoms[atomId]);
   }

  d_closestTriaVertexToAtomsLocation = closestTriaVertexToAtomsLocation;
  d_dispClosestTriaVerticesToAtoms = dispClosestTriaVerticesToAtoms;


  double minDist=1e+6;
  for (unsigned int i=0;i <numberGlobalAtoms-1; i++)
     for (unsigned int j=i+1;j <numberGlobalAtoms; j++)
     {
          const double dist=atomPoints[i].distance(atomPoints[j]);
          if (dist<minDist)
            minDist=dist;
     }
  if (dftParameters::verbosity>=2)
     pcout<<"Minimum distance between atoms: "<<minDist<<std::endl;

  const double gaussianConstant=dftParameters::reproducible_output?1/std::sqrt(0.5):std::min(0.9* minDist/2.0, 2.0);
  const std::pair<bool,double> meshQualityMetrics=gaussianMove.moveMesh(closestTriaVertexToAtomsLocation,
									dispClosestTriaVerticesToAtoms,
									gaussianConstant,
									moveSubdivided);
  timer_movemesh.exit_section("move mesh to atoms: move mesh");

  AssertThrow(!meshQualityMetrics.first,ExcMessage("Negative jacobian created after moving closest nodes to atoms. Suggestion: increase refinement near atoms"));

  d_autoMeshMaxJacobianRatio = meshQualityMetrics.second;

  if (dftParameters::verbosity>=2)
      pcout<< "Mesh quality check after mesh movement, maximum jacobian ratio: "<< meshQualityMetrics.second<<std::endl;
}



