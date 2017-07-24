//
//source file for generating image atoms
//

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

template<unsigned int FEOrder>
void dftClass<FEOrder>::generateImageCharges()
{ 

  const double pspCutOff = 15.0;
  const double tol = 1e-4;

  //
  // get origin/centroid of the cell
  //
  std::vector<double> shift(3,0.0);

  for(int i = 0; i < 3; ++i)
    {
      for(int j = 0; j < 3; ++j){
	shift[i] += d_latticeVectors[j][i]/2.0; 
      }
    }

  std::vector<double> latticeVectors(9,0.0);
  int count = 0;
  for(int i = 0; i < 3; ++i)
    {
      for(int j = 0; j < 3; ++j)
	{
	  latticeVectors[count] = d_latticeVectors[i][j];
	  count++;
	}
    }
  
  d_imageIds.clear();
  for (int i = 0; i < d_imagePositions.size(); ++i)
    {
      std::vector<double> & imagePosition = d_imagePositions[i];
      imagePosition.clear();
    }


  
  for(int i = 0; i < atomLocations.size(); ++i)
    {
      const int iCharge = i;
      const double fracX = atomLocations[i][2];
      const double fracY = atomLocations[i][3];
      const double fracZ = atomLocations[i][4];

      int izmax = 3;
      int iymax = 3;
      int ixmax = 3;

      for(int iz = -2; iz < 3; ++iz)
	{
	  if(periodicZ == 0)
	    iz = izmax;
	  for(int iy = -2; iy < 3; ++iy)
	    {
	      if(periodicY == 0)
		iy = iymax;
	      for(int ix = -2; ix < 3; ++ix)
		{
		  if(periodicX == 0)
		    ix = ixmax;

		  if((periodicX*ix) != 0 || (periodicY*iy) != 0 || (periodicZ*iz) != 0)
		    {
		      const double newFracZ = periodicZ*iz + fracZ;
		      const double newFracY = periodicY*iy + fracY;
		      const double newFracX = periodicX*ix + fracX;

		      std::vector<double> newFrac(3);
		      newFrac[0] = newFracX;
		      newFrac[1] = newFracY;
		      newFrac[2] = newFracZ;

		      bool outsideCell = true;
		      bool withinCutoff = false;
		     

		      if(outsideCell)
			{

			  const double distanceFromCell = getMinDistanceFromImageToCell(latticeVectors,
											newFrac);

			  if (distanceFromCell < pspCutOff)
			    withinCutoff = true;

			}

		      std::vector<double> currentImageChargePosition(3,0.0);

		      if(outsideCell && withinCutoff){
			d_imageIds.push_back(iCharge);
		
			for (int ii = 0; ii < 3; ++ii)
			  for(int jj = 0; jj < 3;++jj)
			    currentImageChargePosition[ii] += d_latticeVectors[jj][ii]*newFrac[jj];
 
			for(int ii = 0; ii < 3; ++ii)
			  currentImageChargePosition[ii] -= shift[ii];
		
			d_imagePositions.push_back(currentImageChargePosition);

			/*if((newFracX >= -tol && newFracX <= 1+tol) &&
			  (newFracY >= -tol && newFracY <= 1+tol) &&
			  (newFracZ >= -tol && newFracZ <= 1+tol))
			  outsideCell = false;*/

		      }
		
		    }
		}
	    }
	}

    }

  const int numImageCharges = d_imageIds.size();
  pcout<<"Number Image Charges  "<<numImageCharges<<std::endl;

  for(int i = 0; i < numImageCharges; ++i)
    {
      double atomCharge;
      if(isPseudopotential)
	atomCharge = atomLocations[d_imageIds[i]][1];
      else
	atomCharge = atomLocations[d_imageIds[i]][0];

      d_imageCharges.push_back(atomCharge);

    }

  /*for(int i = 0; i < d_imagePositions.size();++i){
    std::cout<<"i "<<i<<std::endl;
    for(int  j= 0;  j<  3;++j)
      std::cout<<d_imagePositions[i][j]<<" ";
    std::cout<<'\n';
    }*/

}


