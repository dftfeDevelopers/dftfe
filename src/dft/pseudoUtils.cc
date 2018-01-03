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


#include <boost/math/special_functions/spherical_harmonic.hpp>


namespace pseudoUtils
{
     double tolerance = 1e-12;
   
     //some inline functions
     inline
     void exchangeLocalList(std::vector<unsigned int> & masterNodeIdList,
			       std::vector<unsigned int> & globalMasterNodeIdList,
			       unsigned int numMeshPartitions,
			       const MPI_Comm & mpi_communicator)
     {

	  int numberMasterNodesOnLocalProc = masterNodeIdList.size();

	  int * masterNodeIdListSizes = new int[numMeshPartitions];

	  MPI_Allgather(&numberMasterNodesOnLocalProc,
			1,
			MPI_INT,
			masterNodeIdListSizes,
			1,
			MPI_INT,
			mpi_communicator);

	  int newMasterNodeIdListSize = std::accumulate(&(masterNodeIdListSizes[0]),
							&(masterNodeIdListSizes[numMeshPartitions]),
							0);

	  globalMasterNodeIdList.resize(newMasterNodeIdListSize);

	  int * mpiOffsets = new int[numMeshPartitions];

	  mpiOffsets[0] = 0;

	  for(int i = 1; i < numMeshPartitions; ++i)
	    mpiOffsets[i] = masterNodeIdListSizes[i-1] + mpiOffsets[i-1];

	  MPI_Allgatherv(&(masterNodeIdList[0]),
			 numberMasterNodesOnLocalProc,
			 MPI_INT,
			 &(globalMasterNodeIdList[0]),
			 &(masterNodeIdListSizes[0]),
			 &(mpiOffsets[0]),
			 MPI_INT,
			 mpi_communicator);


	  delete [] masterNodeIdListSizes;
	  delete [] mpiOffsets;

	  return;
     }

     inline 
     void getRadialFunctionVal(const double radialCoordinate,
			  double &splineVal,
			  const alglib::spline1dinterpolant * spline) 
     {
  
       splineVal = alglib::spline1dcalc(*spline,
				        radialCoordinate);
       return;
     }

     inline
     void
     getSphericalHarmonicVal(const double theta, const double phi, const int l, const int m, double & sphericalHarmonicVal)
     {
      
       if(m < 0)
          sphericalHarmonicVal = sqrt(2.0)*boost::math::spherical_harmonic_i(l,-m,theta,phi);
      
       else if (m == 0)
          sphericalHarmonicVal = boost::math::spherical_harmonic_r(l,m,theta,phi);

       else if (m > 0)
          sphericalHarmonicVal = sqrt(2.0)*boost::math::spherical_harmonic_r(l,m,theta,phi);

       return;

     }
     
     inline
     void
     convertCartesianToSpherical(double *x, double & r, double & theta, double & phi)
     {

       r = sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
 
       if(r == 0)
        {
           theta = 0.0;
           phi = 0.0;
        }
       else
        {
           theta = acos(x[2]/r);
	   //
	   // check if theta = 0 or PI (i.e, whether the point is on the Z-axis)
	   // If yes, assign phi = 0.0.
	   // NOTE: In case theta = 0 or PI, phi is undetermined. The actual value 
	   // of phi doesn't matter in computing the enriched function value or 
	   // its gradient. We assign phi = 0.0 here just as a dummy value
	   //
	   if(fabs(theta - 0.0) >= tolerance && fabs(theta - M_PI) >= tolerance)
	      phi = atan2(x[1],x[0]);
	   else
	      phi = 0.0;
	}
    }
}
