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

namespace internal{

  extern "C"{
      //
      // lapack Ax=b
      //
      void dgesv_(int *N, int * NRHS, double* A, int * LDA, int* IPIV,
		  double *B, int * LDB, int *INFO);

  }


  std::vector<double> getFractionalCoordinates(const std::vector<double> & latticeVectors,
	                                       const Point<3> & point,                                                                                           const Point<3> & corner)
  {
      //
      // recenter vertex about corner
      //
      std::vector<double> recenteredPoint(3);
      for(unsigned int i = 0; i < 3; ++i)
        recenteredPoint[i] = point[i]-corner[i];

      std::vector<double> latticeVectorsDup = latticeVectors;

      //
      // to get the fractionalCoords, solve a linear
      // system of equations
      //
      int N = 3;
      int NRHS = 1;
      int LDA = 3;
      int IPIV[3];
      int info;

      dgesv_(&N, &NRHS, &latticeVectorsDup[0], &LDA, &IPIV[0], &recenteredPoint[0], &LDA,&info);
      AssertThrow(info == 0, ExcMessage("LU solve in finding fractional coordinates failed."));
      return recenteredPoint;
  }

  std::vector<double> wrapAtomsAcrossPeriodicBc(const Point<3> & cellCenteredCoord,
	                                        const Point<3> & corner,
					        const std::vector<double> & latticeVectors,
						const std::vector<bool> & periodicBc)
  {
     const double tol=1e-8;
     std::vector<double> fracCoord= getFractionalCoordinates(latticeVectors,
	                                                     cellCenteredCoord,                                                                                                corner);
     //wrap fractional coordinate
     for(unsigned int i = 0; i < 3; ++i)
     {
       if (periodicBc[i])
       {
         if (fracCoord[i]<-tol)
	   fracCoord[i]+=1.0;
         else if (fracCoord[i]>1.0+tol)
	   fracCoord[i]-=1.0;
         AssertThrow(fracCoord[i]>-2.0*tol && fracCoord[i]<1.0+2.0*tol,ExcMessage("Moved atom position doesnt't lie inside the cell after wrapping across periodic boundary"));
       }
     }
     return fracCoord;
  }

}

// Function to update the atom positions and mesh based on the provided displacement input.
// Depending on the maximum displacement magnitude this function decides wether to do auto remeshing
// or move mesh using Gaussian functions.
template<unsigned int FEOrder>
void dftClass<FEOrder>::updateAtomPositionsAndMoveMesh(const std::vector<Point<C_DIM> > & globalAtomsDisplacements)
{
  const int numberGlobalAtoms = atomLocations.size();
  const int numberImageCharges = d_imageIds.size();
  const int totalNumberAtoms = numberGlobalAtoms + numberImageCharges;

  std::vector<double> latticeVectorsFlattened(9,0.0);
  for (unsigned int idim=0; idim<3; idim++)
      for(unsigned int jdim=0; jdim<3; jdim++)
          latticeVectorsFlattened[3*idim+jdim]=d_domainBoundingVectors[idim][jdim];
  Point<3> corner;
  for (unsigned int idim=0; idim<3; idim++)
  {
      corner[idim]=0;
      for(unsigned int jdim=0; jdim<3; jdim++)
          corner[idim]-=d_domainBoundingVectors[jdim][idim]/2.0;
  }
  std::vector<bool> periodicBc(3,false);
  periodicBc[0]=dftParameters::periodicX;periodicBc[1]=dftParameters::periodicY;periodicBc[2]=dftParameters::periodicZ;

  std::vector<Point<C_DIM> > controlPointLocations;
  std::vector<Tensor<1,C_DIM,double> > controlPointDisplacements;

  double maxDispAtom=-1;
  for (unsigned int iAtom=0;iAtom <totalNumberAtoms; iAtom++)
  {
     Point<C_DIM> atomCoor;
     int atomId=iAtom;
     if(iAtom < numberGlobalAtoms)
     {
        atomCoor[0] = atomLocations[iAtom][2];
        atomCoor[1] = atomLocations[iAtom][3];
        atomCoor[2] = atomLocations[iAtom][4];
	const double temp=globalAtomsDisplacements[atomId].norm();

	if (dftParameters::periodicX || dftParameters::periodicY || dftParameters::periodicZ)
        {
	    Point<C_DIM> newCoord;
	    for (unsigned int idim=0; idim<C_DIM; ++idim)
		newCoord[idim]=atomCoor[idim]+globalAtomsDisplacements[atomId][idim];
	    std::vector<double> newFracCoord=internal::wrapAtomsAcrossPeriodicBc(newCoord,
										      corner,
										      latticeVectorsFlattened,
										      periodicBc);
	    //for synchrozination
	    MPI_Bcast(&(newFracCoord[0]),
		     3,
		     MPI_DOUBLE,
		     0,
		     MPI_COMM_WORLD);

	    atomLocationsFractional[iAtom][2]=newFracCoord[0];
	    atomLocationsFractional[iAtom][3]=newFracCoord[1];
	    atomLocationsFractional[iAtom][4]=newFracCoord[2];
        }
	else
	{
	    atomLocations[iAtom][2]+=globalAtomsDisplacements[atomId][0];
	    atomLocations[iAtom][3]+=globalAtomsDisplacements[atomId][1];
	    atomLocations[iAtom][4]+=globalAtomsDisplacements[atomId][2];
        }

	if (temp>maxDispAtom)
	    maxDispAtom=temp;
     }
     else
     {
	atomCoor[0] = d_imagePositions[iAtom-numberGlobalAtoms][0];
	atomCoor[1] = d_imagePositions[iAtom-numberGlobalAtoms][1];
	atomCoor[2] = d_imagePositions[iAtom-numberGlobalAtoms][2];
	atomId=d_imageIds[iAtom-numberGlobalAtoms];
     }
     controlPointLocations.push_back(atomCoor);
     controlPointDisplacements.push_back(globalAtomsDisplacements[atomId]);
  }
  MPI_Barrier(mpi_communicator);

  const bool useHybridMeshUpdateScheme=dftParameters::electrostaticsHRefinement?false:true;

  if (!useHybridMeshUpdateScheme)//always remesh
  {
          if (!dftParameters::reproducible_output)
	    pcout << "Auto remeshing and reinitialization of dft problem for new atom coordinates" << std::endl;

	  if (maxDispAtom<0.2 && dftParameters::isPseudopotential)
	  {
	    init(dftParameters::reuseWfcGeoOpt && maxDispAtom<0.1?2:(dftParameters::reuseDensityGeoOpt?1:0));
	  }
	  else
	    init(0);

	  if (!dftParameters::reproducible_output)
	    pcout << "...Reinitialization end" << std::endl;
  }
  else
  {
      meshMovementGaussianClass gaussianMove(mpi_communicator);
      gaussianMove.init(d_mesh.getParallelMeshMoved(),
                        d_mesh.getSerialMeshUnmoved(),
                        d_domainBoundingVectors);

      const double tol=1e-6;
      //Heuristic values
      const  double maxJacobianRatio=2.0;
      const double break1=5e-2;

      unsigned int useGaussian=0;
      if (maxDispAtom <(break1+tol))
	useGaussian=1;

      //for synchrozination in case the updateCase are different in different processors due to floating point comparison
      MPI_Bcast(&(useGaussian),
		1,
		MPI_INT,
		0,
		MPI_COMM_WORLD);

      if (useGaussian!=1)
      {
	  pcout << "Auto remeshing and reinitialization of dft problem for new atom coordinates as max displacement magnitude: "<<maxDispAtom<< " is greater than: "<< break1 << " Bohr..." << std::endl;
	  init(0);
	  pcout << "...Reinitialization end" << std::endl;
      }
      else
      {
	   pcout << "Trying to Move using Gaussian with same Gaussian constant for computing the forces: "<<forcePtr->getGaussianGeneratorParameter()<<" as max displacement magnitude: "<< maxDispAtom<< " is below " << break1 <<" Bohr"<<std::endl;
	  const std::pair<bool,double> meshQualityMetrics=gaussianMove.moveMesh(controlPointLocations,controlPointDisplacements,forcePtr->getGaussianGeneratorParameter());
	  unsigned int autoMesh=0;
	  if (meshQualityMetrics.first || meshQualityMetrics.second>maxJacobianRatio)
	      autoMesh=1;
	  MPI_Bcast(&(autoMesh),
		    1,
		    MPI_INT,
		    0,
		    MPI_COMM_WORLD);
	  if (autoMesh==1)
	  {
	      if (meshQualityMetrics.first)
		 pcout<< " Auto remeshing and reinitialization of dft problem for new atom coordinates due to negative jacobian after Gaussian mesh movement using Gaussian constant: "<< forcePtr->getGaussianGeneratorParameter()<<std::endl;
	      else
		 pcout<< " Auto remeshing and reinitialization of dft problem for new atom coordinates due to maximum jacobian ratio: "<< meshQualityMetrics.second<< " exceeding set bound of: "<< maxJacobianRatio<<" after Gaussian mesh movement using Gaussian constant: "<< forcePtr->getGaussianGeneratorParameter()<<std::endl;
	      init(0);
	      pcout << "...Reinitialization end" << std::endl;
	  }
	  else
	  {
	      pcout<< " Mesh quality check: maximum jacobian ratio after movement: "<< meshQualityMetrics.second<<std::endl;
	      pcout << "Now Reinitializing all moved triangulation dependent objects..." << std::endl;
	      initNoRemesh();
	      pcout << "...Reinitialization end" << std::endl;
	  }
      }
  }

}
