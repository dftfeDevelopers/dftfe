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
namespace atomsForcesUtils{

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
      for(int i = 0; i < 3; ++i)
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

      if (info != 0) {
        const std::string
          message("LU solve in finding fractional coordinates failed.");
        Assert(false,ExcMessage(message));
      }
      return recenteredPoint;
    }    
    //
    // round a given fractional coordinate to zero or 1
    //
    double roundToCell(double frac){
      double returnValue = 0;
      if(frac < 0)
	returnValue = 0;
      else if(frac >=0 && frac <= 1)
	returnValue = frac;
      else
	returnValue = 1;
	
      return returnValue;
	
    }

    //
    // cross product
    //
    std::vector<double> cross(const std::vector<double> & v1,
			      const std::vector<double> & v2){

      assert(v1.size()==3);
      assert(v2.size()==3);

      std::vector<double> returnValue(3);

      returnValue[0] = v1[1]*v2[2]-v1[2]*v2[1];
      returnValue[1]= -v1[0]*v2[2]+v2[0]*v1[2];
      returnValue[2]=  v1[0]*v2[1]-v2[0]*v1[1];
      return returnValue;
	  
    }

    //
    // given surface defined by normal = surfaceNormal and a point = xred2
    // find the point on this surface closest to an arbitrary point = xred1
    // return fractional coordinates of nearest point
    //
    std::vector<double> 
    getNearestPointOnGivenSurface(std::vector<double>  latticeVectors,
				  const std::vector<double> & xred1,
				  const std::vector<double> & xred2,
				  const std::vector<double> & surfaceNormal)

    {

      //
      // get real space coordinates for xred1 and xred2
      //
      std::vector<double> P(3,0.0);
      std::vector<double> Q(3,0.0);
      std::vector<double> R(3);

      for (int i = 0; i < 3; ++i){
	for(int j = 0; j < 3;++j){
	  P[i] += latticeVectors[3*j +i]*xred1[j]; 
	  Q[i] += latticeVectors[3*j +i]*xred2[j];
	}
	R[i] = Q[i] - P[i];
      }
	
      //
      // fine nearest point on the plane defined by surfaceNormal and xred2
      //
      double num = R[0]*surfaceNormal[0]+R[1]*surfaceNormal[1]+R[2]*surfaceNormal[2];
      double denom = surfaceNormal[0]*surfaceNormal[0]+surfaceNormal[1]*surfaceNormal[1]+surfaceNormal[2]*surfaceNormal[2];
      const double t = num/denom;

	  
      std::vector<double> nearestPtCoords(3);
      for(int i = 0; i < 3; ++i)
	nearestPtCoords[i] = P[i]+t*surfaceNormal[i];
	
      //
      // get fractional coordinates for the nearest point : solve a system
      // of equations
      int N = 3;
      int NRHS = 1;
      int LDA = 3;
      int IPIV[3];
      int info;

      
      dgesv_(&N, &NRHS, &latticeVectors[0], &LDA, &IPIV[0], &nearestPtCoords[0], &LDA,&info);

	     
      if (info != 0) {

	std::cout<<"LU solve in conversion of frac to real coords failed."<<std::endl;
	exit(-1);

      }
       
      //
      // nearestPtCoords is overwritten with the solution = frac coords
      //

      std::vector<double> returnValue(3);

      for(int i = 0; i < 3 ;++i)
	returnValue[i] = roundToCell(nearestPtCoords[i]);

      return returnValue;

    }

    //
    // input : xreduced = frac coords of image charge
    // output : min distance to any of the cel surfaces
    //
    double 
    getMinDistanceFromImageToCell(const std::vector<double> & latticeVectors,
				  const std::vector<double> & xreduced)
    {
      const double xfrac = xreduced[0];
      const double yfrac = xreduced[1];
      const double zfrac = xreduced[2];

      //
      // if interior point, then return 0 distance
      //
      if(xfrac >=0 && xfrac <=1 && yfrac >=0 && yfrac <=1 && zfrac >=0 && zfrac <=1)
	return 0;
      else
	{
	  //
	  // extract lattice vectors and define surface normals
	  //
	  const std::vector<double> a(&latticeVectors[0],&latticeVectors[0]+3);
	  const std::vector<double> b(&latticeVectors[3],&latticeVectors[3]+3);
	  const std::vector<double> c(&latticeVectors[6],&latticeVectors[6]+3);

	  std::vector<double> surface1Normal = cross(b,c);
	  std::vector<double> surface2Normal = cross(c,a);
	  std::vector<double> surface3Normal = cross(a,b);

	  std::vector<double> surfacePoint(3);
	  std::vector<double> dFrac(3);
	  std::vector<double> dReal(3);

	  //
	  //find closest distance to surface 1
	  //
	  surfacePoint[0] = 0; 
	  surfacePoint[1] = yfrac; 
	  surfacePoint[2] = zfrac;

	  std::vector<double> fracPtA = getNearestPointOnGivenSurface(latticeVectors,
								      xreduced,
								      surfacePoint,
								      surface1Normal);
	  //
	  // compute distance between fracPtA (closest point on surface A) and xreduced
	  //
	  for(int i = 0; i < 3; ++i)
	    dFrac[i] = xreduced[i] - fracPtA[i];

	  for (int i = 0; i < 3; ++i)
	    for(int j = 0; j < 3;++j)
	      dReal[i] += latticeVectors[3*j +i]*dFrac[j]; 
	  
	  double distA = dReal[0]*dReal[0]+dReal[1]*dReal[1]+dReal[2]*dReal[2];
	  distA = sqrt(distA);

	  //
	  // find closest distance to surface 2
	  //
	  surfacePoint[0] = xfrac; 
	  surfacePoint[1] = 0; 
	  surfacePoint[2] = zfrac;
	      
	  std::vector<double> fracPtB = getNearestPointOnGivenSurface(latticeVectors,
								      xreduced,
								      surfacePoint,
								      surface2Normal);

	  for(int i = 0; i < 3; ++i){
	    dFrac[i] = xreduced[i] - fracPtB[i];
	    dReal[i] = 0.0;
	  }
	  
	  for (int i = 0; i < 3; ++i)
	    for(int j = 0; j < 3;++j)
	      dReal[i] += latticeVectors[3*j +i]*dFrac[j]; 

	  double distB =  dReal[0]*dReal[0]+dReal[1]*dReal[1]+dReal[2]*dReal[2];
	  distB = sqrt(distB);

	  //
	  // find min distance to surface 3
	  //
	  surfacePoint[0] = xfrac; 
	  surfacePoint[1] = yfrac; 
	  surfacePoint[2] = 0;
	      
	  std::vector<double> fracPtC = getNearestPointOnGivenSurface(latticeVectors,
								      xreduced,
								      surfacePoint,
								      surface3Normal);

	  for(int i = 0; i < 3; ++i){
	    dFrac[i] = xreduced[i] - fracPtC[i];
	    dReal[i] = 0.0;
	  }

	  for (int i = 0; i < 3; ++i)
	    for(int j = 0; j < 3;++j)
	      dReal[i] += latticeVectors[3*j +i]*dFrac[j]; 

	  double distC = dReal[0]*dReal[0]+dReal[1]*dReal[1]+dReal[2]*dReal[2];
	  distC = sqrt(distC);

	  //
	  // fine min distance to surface 4
	  //
	  surfacePoint[0] = 1; 
	  surfacePoint[1] = yfrac; 
	  surfacePoint[2] = zfrac;

	  std::vector<double> fracPtD = getNearestPointOnGivenSurface(latticeVectors,
								      xreduced,
								      surfacePoint,
								      surface1Normal);

	  for(int i = 0; i < 3; ++i){
	    dFrac[i] = xreduced[i] - fracPtD[i];
	    dReal[i] = 0.0;
	  }

	  for (int i = 0; i < 3; ++i)
	    for(int j = 0; j < 3;++j)
	      dReal[i] += latticeVectors[3*j +i]*dFrac[j]; 

	  double distD =  dReal[0]*dReal[0]+dReal[1]*dReal[1]+dReal[2]*dReal[2];
	  distD = sqrt(distD);
	  
	  //
	  // find min distance to surface 5
	  //
	  surfacePoint[0] = xfrac; 
	  surfacePoint[1] = 1; 
	  surfacePoint[2] = zfrac;
	  
	  std::vector<double> fracPtE = getNearestPointOnGivenSurface(latticeVectors,
								      xreduced,
								      surfacePoint,	
								      surface2Normal);

	  for(int i = 0; i < 3; ++i){
	    dFrac[i] = xreduced[i] - fracPtE[i];
	    dReal[i] = 0.0;
	  }

	  for (int i = 0; i < 3; ++i)
	    for(int j = 0; j < 3;++j)
	      dReal[i] += latticeVectors[3*j +i]*dFrac[j];

	  double distE = dReal[0]*dReal[0]+dReal[1]*dReal[1]+dReal[2]*dReal[2];
	  distE = sqrt(distE);


	  //
	  // find min distance to surface 6
	  //
	  surfacePoint[0] = xfrac; 
	  surfacePoint[1] = yfrac; 
	  surfacePoint[2] = 1;
	  
	  std::vector<double> fracPtF = getNearestPointOnGivenSurface(latticeVectors,
								      xreduced,
								      surfacePoint,	
								      surface3Normal);

	  for(int i = 0; i < 3; ++i){
	    dFrac[i] = xreduced[i] - fracPtF[i];
	    dReal[i] = 0.0;
	  }

	  for (int i = 0; i < 3; ++i)
	    for(int j = 0; j < 3;++j)
	      dReal[i] += latticeVectors[3*j +i]*dFrac[j];

	  double distF = dReal[0]*dReal[0]+dReal[1]*dReal[1]+dReal[2]*dReal[2];
	  distF = sqrt(distF);
	  
	  return std::min(distF, std::min(distE, std::min( distD, std::min(distC, std::min(distB,distA)))));
	     
	}


    }
}
//Configurational force on atoms corresponding to Gaussian generator. Generator is discretized using linear FE shape functions. Configurational force on nodes due to linear FE shape functions precomputed 
template<unsigned int FEOrder>
void forceClass<FEOrder>::computeAtomsForcesGaussianGenerator(bool allowGaussianOverlapOnAtoms)
{
  unsigned int vertices_per_cell=GeometryInfo<C_DIM>::vertices_per_cell;  
  const std::vector<std::vector<double> > & atomLocations=dftPtr->atomLocations;
  const std::vector<std::vector<double> > & imagePositions=dftPtr->d_imagePositions;
  const std::vector<int > & imageIds=dftPtr->d_imageIds;
  const int numberGlobalAtoms = atomLocations.size();
  const int numberImageCharges = imageIds.size();
  const int totalNumberAtoms = numberGlobalAtoms + numberImageCharges;  
  std::vector<double> globalAtomsGaussianForcesLocalPart(numberGlobalAtoms*C_DIM,0);
  d_globalAtomsGaussianForces.clear();
  d_globalAtomsGaussianForces.resize(numberGlobalAtoms*C_DIM,0.0);
#ifdef ENABLE_PERIODIC_BC  
  std::vector<double> globalAtomsGaussianForcesKPointsLocalPart(numberGlobalAtoms*C_DIM,0);
  d_globalAtomsGaussianForcesKPoints.clear();
  d_globalAtomsGaussianForcesKPoints.resize(numberGlobalAtoms*C_DIM,0.0);  
#endif    
  std::vector<bool> vertex_touched(d_dofHandlerForce.get_triangulation().n_vertices(),
				   false);      
  DoFHandler<3>::active_cell_iterator
  cell = d_dofHandlerForce.begin_active(),
  endc = d_dofHandlerForce.end();      
  for (; cell!=endc; ++cell) 
  {
   if (cell->is_locally_owned())
   {
    for (unsigned int i=0; i<vertices_per_cell; ++i)
    {
	const unsigned global_vertex_no = cell->vertex_index(i);

	if (vertex_touched[global_vertex_no])
	   continue;	    
	vertex_touched[global_vertex_no]=true;
	Point<C_DIM> nodalCoor = cell->vertex(i);

	int overlappedAtomId=-1;
	for (unsigned int jAtom=0;jAtom <totalNumberAtoms; jAtom++)
	{
           Point<C_DIM> jAtomCoor;
           if(jAtom < numberGlobalAtoms)
           {
              jAtomCoor[0] = atomLocations[jAtom][2];
              jAtomCoor[1] = atomLocations[jAtom][3];
              jAtomCoor[2] = atomLocations[jAtom][4];
           }
           else
           {
	      jAtomCoor[0] = imagePositions[jAtom-numberGlobalAtoms][0];
	      jAtomCoor[1] = imagePositions[jAtom-numberGlobalAtoms][1];
	      jAtomCoor[2] = imagePositions[jAtom-numberGlobalAtoms][2];
            }
            const double distance=(nodalCoor-jAtomCoor).norm();
	    if (distance < 1e-5){
		overlappedAtomId=jAtom;
		break;
	    }
	}//j atom loop

        for (unsigned int iAtom=0;iAtom <totalNumberAtoms; iAtom++)
	{
             if (overlappedAtomId!=iAtom && overlappedAtomId!=-1 && !allowGaussianOverlapOnAtoms)
		 continue;
             Point<C_DIM> atomCoor;
	     int atomId=iAtom;
	     if(iAtom < numberGlobalAtoms)
	     {
		atomCoor[0] = atomLocations[iAtom][2];
		atomCoor[1] = atomLocations[iAtom][3];
		atomCoor[2] = atomLocations[iAtom][4];
	      }
	      else
	      {
		atomCoor[0] = imagePositions[iAtom-numberGlobalAtoms][0];
		atomCoor[1] = imagePositions[iAtom-numberGlobalAtoms][1];
		atomCoor[2] = imagePositions[iAtom-numberGlobalAtoms][2];
		atomId=imageIds[iAtom-numberGlobalAtoms];
	      }	
	      const double rsq=(nodalCoor-atomCoor).norm_square();	
	      const double gaussianWeight=std::exp(-d_gaussianConstant*rsq);
	      for (unsigned int idim=0; idim < C_DIM ; idim++)
	      {
	          const unsigned int globalDofIndex=cell->vertex_dof_index(i,idim);
	          if (!d_constraintsNoneForce.is_constrained(globalDofIndex) && d_locally_owned_dofsForce.is_element(globalDofIndex))
		  {
	              globalAtomsGaussianForcesLocalPart[C_DIM*atomId+idim]+=gaussianWeight*d_configForceVectorLinFE[globalDofIndex];
#ifdef ENABLE_PERIODIC_BC  
                      globalAtomsGaussianForcesKPointsLocalPart[C_DIM*atomId+idim]+=gaussianWeight*d_configForceVectorLinFEKPoints[globalDofIndex];
#endif 
		  }
	      }//idim loop
 	}//iAtom loop
     }//vertices per cell
   }//locally owned check
  }//cell loop
  

  //Sum all processor contributions and distribute to all processors
  MPI_Allreduce(&(globalAtomsGaussianForcesLocalPart[0]),
		&(d_globalAtomsGaussianForces[0]), 
		numberGlobalAtoms*C_DIM,
		MPI_DOUBLE,
		MPI_SUM,
                mpi_communicator);
#ifdef ENABLE_PERIODIC_BC 
  //Sum all processor contributions and distribute to all processors
  MPI_Allreduce(&(globalAtomsGaussianForcesKPointsLocalPart[0]),
		&(d_globalAtomsGaussianForcesKPoints[0]), 
		numberGlobalAtoms*C_DIM,
		MPI_DOUBLE,
		MPI_SUM,
                mpi_communicator);
  //Sum over k point pools and add to total Gaussian force
  for (unsigned int iAtom=0;iAtom <numberGlobalAtoms; iAtom++)
  {
      for (unsigned int idim=0; idim < C_DIM ; idim++)
      {      
          d_globalAtomsGaussianForcesKPoints[iAtom*C_DIM+idim]= Utilities::MPI::sum(d_globalAtomsGaussianForcesKPoints[iAtom*C_DIM+idim], dftPtr->interpoolcomm);
          d_globalAtomsGaussianForces[iAtom*C_DIM+idim]+=d_globalAtomsGaussianForcesKPoints[iAtom*C_DIM+idim];
      }
  }
#endif

}

template<unsigned int FEOrder>
void forceClass<FEOrder>::printAtomsForces()
{
    const int numberGlobalAtoms = dftPtr->atomLocations.size();	 
    pcout<<std::endl;
    pcout<<"Ion forces"<<std::endl;
    if (dftParameters::verbosity==1)    
       pcout<< "Negative of configurational force (Hartree/Bohr) on atoms for Gaussian generator with constant: "<< d_gaussianConstant <<std::endl;

    pcout<< "--------------------------------------------------------------------------------------------"<<std::endl;    
    //also find the atom with the maximum absolute force and print that
    double maxForce=-1.0;
    unsigned int maxForceAtomId=0;
    for (unsigned int i=0; i< numberGlobalAtoms; i++)
    {
	pcout<< "AtomId "<< std::setw(4) << i << ":  "<< std::scientific<< -d_globalAtomsGaussianForces[3*i]<<","<< -d_globalAtomsGaussianForces[3*i+1]<<","<<-d_globalAtomsGaussianForces[3*i+2]<<std::endl;  
	double absForce=0.0;
	for (unsigned int idim=0; idim< C_DIM; idim++)
        {
	    absForce+=d_globalAtomsGaussianForces[3*i+idim]*d_globalAtomsGaussianForces[3*i+idim];
	}
	Assert (absForce>=0., ExcInternalError());
	absForce=std::sqrt(absForce);
	if (absForce>maxForce)
	{
	    maxForce=absForce;
	    maxForceAtomId=i;
	}
    }
    
    pcout<< "--------------------------------------------------------------------------------------------"<<std::endl;
  
    if (dftParameters::verbosity==1)
        pcout<<" Maximum absolute force atom id: "<< maxForceAtomId << ", Force vec: "<< -d_globalAtomsGaussianForces[3*maxForceAtomId]<<","<< -d_globalAtomsGaussianForces[3*maxForceAtomId+1]<<","<<-d_globalAtomsGaussianForces[3*maxForceAtomId+2]<<std::endl;
}
