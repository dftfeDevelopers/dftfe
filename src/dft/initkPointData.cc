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
// @author Shiva Rudraraju (2016), Phani Motamarri (2016), Krishnendu Ghosh (2017)
//
#include "../../include/dftParameters.h"


using namespace dftParameters ;
namespace internaldft{
  std::vector<double>  cross_product(const std::vector<double> &a,
		                     const std::vector<double> &b)
  {
    std::vector<double> crossProduct(a.size(),0.0);

    crossProduct[0] = a[1]*b[2]-a[2]*b[1];
    crossProduct[1] = a[2]*b[0]-a[0]*b[2];
    crossProduct[2] = a[0]*b[1]-a[1]*b[0];

    return crossProduct;
  }

  std::vector<std::vector<double> > getReciprocalLatticeVectors(const std::vector<std::vector<double> > & latticeVectors) 
	                           
  {
      std::vector<std::vector<double> > reciprocalLatticeVectors(3,std::vector<double> (3,0.0));
      for(unsigned int i = 0; i < 2; ++i)
	{
	  std::vector<double> cross=internaldft::cross_product(latticeVectors[i+1],
				                               latticeVectors[3 - (2*i + 1)]);

	  const double scalarConst = latticeVectors[i][0]*cross[0] + latticeVectors[i][1]*cross[1] + latticeVectors[i][2]*cross[2];

	  for (unsigned int d = 0; d < 3; ++d)
	    reciprocalLatticeVectors[i][d] = (2.*M_PI/scalarConst)*cross[d];
	}

      //
      //fill up 3rd reciprocal lattice vector
      //
      std::vector<double> cross=internaldft::cross_product(latticeVectors[0],
				                           latticeVectors[1]);

      const double scalarConst = latticeVectors[2][0]*cross[0] + latticeVectors[2][1]*cross[1] + latticeVectors[2][2]*cross[2];
      for (unsigned int d = 0; d < 3; ++d)
         reciprocalLatticeVectors[2][d] = (2*M_PI/scalarConst)*cross[d];

      return reciprocalLatticeVectors;
  }
}

template<unsigned int FEOrder>
void dftClass<FEOrder>::readkPointData()
{
  const int numberColumnskPointDataFile = 4;
  std::vector<std::vector<double> > kPointData;
  //std::vector<double> kPointReducedCoordinates;
  char kPointRuleFile[256];
  sprintf(kPointRuleFile, "%s/data/kPointList/%s", dftParameters::currentPath.c_str(), dftParameters::kPointDataFile.c_str());
  dftUtils::readFile(numberColumnskPointDataFile, kPointData, kPointRuleFile);
  d_kPointCoordinates.clear() ;
  d_kPointWeights.clear();
  d_maxkPoints = kPointData.size();
  d_kPointCoordinates.resize(d_maxkPoints*3,0.0);
  d_kPointWeights.resize(d_maxkPoints,0.0);

  kPointReducedCoordinates = d_kPointCoordinates;

  for(unsigned int i = 0; i < d_maxkPoints; ++i)
    {
      for (unsigned int d = 0; d < 3; ++d)
        kPointReducedCoordinates[3*i + d] = kPointData[i][d];
      d_kPointWeights[i] = kPointData[i][3];
    }

  pcout<<"Reduced k-Point-coordinates and weights: "<<std::endl;

  for(unsigned int i = 0; i < d_maxkPoints; ++i)
    pcout<<kPointReducedCoordinates[3*i + 0]<<" "<<kPointReducedCoordinates[3*i + 1]<<" "<<kPointReducedCoordinates[3*i + 2]<<" "<<d_kPointWeights[i]<<std::endl;


  for(unsigned int i = 0; i < d_maxkPoints; ++i)
    {
      for (unsigned int d1 = 0; d1 < 3; ++d1)
        d_kPointCoordinates[3*i + d1] = kPointReducedCoordinates[3*i+0]*d_reciprocalLatticeVectors[0][d1] +
                                        kPointReducedCoordinates[3*i+1]*d_reciprocalLatticeVectors[1][d1] +
                                        kPointReducedCoordinates[3*i+2]*d_reciprocalLatticeVectors[2][d1];
    }



}

template<unsigned int FEOrder>
void dftClass<FEOrder>::recomputeKPointCoordinates()
{
  // Get the reciprocal lattice vectors
  d_reciprocalLatticeVectors=internaldft::getReciprocalLatticeVectors(d_domainBoundingVectors);    
  // Convert from crystal to Cartesian coordinates
  for(unsigned int i = 0; i < d_maxkPoints; ++i)
    for (unsigned int d=0; d < 3; ++d)
      d_kPointCoordinates[3*i + d] = kPointReducedCoordinates[3*i+0]*d_reciprocalLatticeVectors[0][d] +
                                     kPointReducedCoordinates[3*i+1]*d_reciprocalLatticeVectors[1][d] +
                                     kPointReducedCoordinates[3*i+2]*d_reciprocalLatticeVectors[2][d];    
}

template<unsigned int FEOrder>
void dftClass<FEOrder>::generateMPGrid()
{
  std::vector<double> del(3);
  d_maxkPoints = (nkx * nky) * nkz;
  pcout<<"Total number of k-points " << d_maxkPoints << std::endl;
  //
  del[0] = 1.0/double(nkx); del[1] = 1.0/double(nky); del[2] = 1.0/double(nkz);
  if (nkx==1)
    del[0]=0.0;
  if (nky==1)
    del[1]=0.0;
  if (nkz==1)
    del[2]=0.0;
  //
  d_kPointCoordinates.resize(d_maxkPoints*3,0.0);
  d_kPointWeights.resize(d_maxkPoints,0.0);

  kPointReducedCoordinates = d_kPointCoordinates;

  for(unsigned int i = 0; i < d_maxkPoints; ++i)
    {
      kPointReducedCoordinates[3*i + 2] = del[2]*(i%nkz) + dkz;
      kPointReducedCoordinates[3*i + 1] = del[1]*(std::floor( ( i%(nkz*nky) ) / nkz) ) + dky;
      kPointReducedCoordinates[3*i + 0] = del[0]*(std::floor( (i/(nkz*nky) ) ) ) + dkx;
      for (unsigned int dir = 0; dir < 3; ++dir) {
         if(kPointReducedCoordinates[3*i + dir] > ( 0.5 + 1.0E-10 ) )
              kPointReducedCoordinates[3*i + dir] = kPointReducedCoordinates[3*i + dir] - 1.0 ;
      }
      d_kPointWeights[i] = 1.0/d_maxkPoints ;
    }

  // Get the reciprocal lattice vectors
  d_reciprocalLatticeVectors=internaldft::getReciprocalLatticeVectors(d_domainBoundingVectors);

  if (useSymm || timeReversal) {
    //
    const int numberColumnsSymmDataFile = 3;
    std::vector<std::vector<int>> symmUnderGroupTemp ;
    std::vector<std::vector<double> > symmData;
    std::vector<std::vector<std::vector<double> >> symmMatTemp, symmMatTemp2;
    const int max_size = 500;
    int rotation[max_size][3][3];
    //
    //////////////////////////////////////////  SPG CALL  ///////////////////////////////////////////////////
    if (useSymm) { 
     const int num_atom = atomLocationsFractional.size();
     double lattice[3][3], position[num_atom][3];
     int types[num_atom] ;
     const int mesh[3] = {static_cast<int>(nkx), static_cast<int>(nky), static_cast<int>(nkz)};
     int grid_address[nkx * nky * nkz][3];
     int grid_mapping_table[nkx * nky * nkz];
     //
     for (unsigned int i=0; i<3; ++i) {
        for (unsigned int j=0; j<3; ++j)
            lattice[i][j] = d_domainBoundingVectors[i][j];
     }
     std::set<unsigned int>::iterator it = atomTypes.begin();
     for (unsigned int i=0; i<num_atom; ++i){
        std::advance(it, i);
        types[i] = atomLocationsFractional[i][0];
        for (unsigned int j=0; j<3; ++j)
           position[i][j] = atomLocationsFractional[i][j+2] ;
     }
     //
     pcout<<" getting space group symmetries from spg " << std::endl;
     symmetryPtr->numSymm = spg_get_symmetry(rotation,
                     (symmetryPtr->translation),
                     max_size,
                     lattice,
                     position,
                     types,
                     num_atom,
                     1e-5);
     pcout<<" number of symmetries allowed for the lattice " <<symmetryPtr->numSymm << std::endl;
     for (unsigned int iSymm=0; iSymm<symmetryPtr->numSymm; ++iSymm){
	pcout << " Symmetry " << iSymm+1 << std::endl;
	pcout << " Rotation " << std::endl;
	for (unsigned int ipol = 0; ipol<3; ++ipol)
		 pcout << rotation[iSymm][ipol][0] << "  " <<rotation[iSymm][ipol][1] << "  " << rotation[iSymm][ipol][2] << std::endl;
	pcout << " translation " << std::endl;
	pcout << symmetryPtr->translation[iSymm][0] << "  " <<symmetryPtr->translation[iSymm][1] << "  " << symmetryPtr->translation[iSymm][2] << std::endl;
	pcout << "	" << std::endl ;
     }
    }
    //
    else {   // only time reversal symmetry; no point group symmetry       
     //
     symmetryPtr->numSymm = 1 ;
     for (unsigned int j=0; j<3; ++j) {
          for (unsigned int k=0; k<3; ++k) {
	       if (j==k) 
                   rotation[0][j][k] = 1 ;
	       else 
	           rotation[0][j][k] = 0 ;
          }
	  symmetryPtr->translation[0][j] = 0.0 ;
      }
      pcout<<" Only time reversal symmetry to be used " << std::endl;

    }

    //// adding time reversal  //////
    if(timeReversal) {
      for (unsigned int iSymm=symmetryPtr->numSymm; iSymm<2*symmetryPtr->numSymm; ++iSymm){
         for (unsigned int j=0; j<3; ++j) {
            for (unsigned int k=0; k<3; ++k)
               rotation[iSymm][j][k] = -1*rotation[iSymm-symmetryPtr->numSymm][j][k] ;
	    symmetryPtr->translation[iSymm][j] = symmetryPtr->translation[iSymm-symmetryPtr->numSymm][j] ;
         }     
      }
      symmetryPtr->numSymm = 2*symmetryPtr->numSymm;
    }
  ///
  symmMatTemp.resize(symmetryPtr->numSymm);
  symmMatTemp2.resize(symmetryPtr->numSymm);
  symmetryPtr->symmMat.resize(symmetryPtr->numSymm);
  symmUnderGroupTemp.resize(symmetryPtr->numSymm);
  for (unsigned int i=0; i<symmetryPtr->numSymm; ++i) {
     symmMatTemp[i].resize(3,std::vector<double> (3,0.0)) ;
     symmMatTemp2[i].resize(3,std::vector<double> (3,0.0)) ;
     for (unsigned int j=0; j<3; ++j) {
        for (unsigned int k=0; k<3; ++k) {
         symmMatTemp[i][j][k] = double(rotation[i][j][k]);
         symmMatTemp2[i][j][k] = double(rotation[i][j][k]);
	 if (timeReversal && i >= symmetryPtr->numSymm/2)
              symmMatTemp2[i][j][k] = - double(rotation[i][j][k]);

	}
    }
  }

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //
  std::vector<double> kPointAllCoordinates, kPointTemp(3);
  std::vector<int> discard(d_maxkPoints, 0), countedSymm(symmetryPtr->numSymm, 0), usedSymmNum(symmetryPtr->numSymm, 1);
  kPointAllCoordinates = kPointReducedCoordinates;
  const int nk = d_maxkPoints ;
  d_maxkPoints = 0;

  double translationTemp[symmetryPtr->numSymm][3];
  for (unsigned int i=0; i<(symmetryPtr->numSymm); ++i) {
    for (unsigned int j=0; j<3; ++j)
     translationTemp[i][j] = (symmetryPtr->translation)[i][j];
  }
  //
  symmetryPtr->symmMat[0] = symmMatTemp[0] ;
  //symmMat = symmMatTemp ;
  unsigned int usedSymm=1, ik = 0; // note usedSymm is initialized to 1 and not 0. Because identity is always present
  countedSymm[0] = 1 ;
  //
  while( ik < nk ) {
    //
    for (unsigned int d=0; d < 3; ++d)
      kPointReducedCoordinates[3*d_maxkPoints + d] = kPointAllCoordinates[3*ik+d];

    d_maxkPoints = d_maxkPoints + 1;
    //
    for (unsigned int iSymm = 1; iSymm < symmetryPtr->numSymm; ++iSymm) // iSymm begins from 1. because identity is always present and is taken care of.
	{
      for(unsigned int d = 0; d < 3; ++d)
        kPointTemp[d] = kPointAllCoordinates[3*ik+0]*symmMatTemp[iSymm][0][d] +
                        kPointAllCoordinates[3*ik+1]*symmMatTemp[iSymm][1][d] +
                        kPointAllCoordinates[3*ik+2]*symmMatTemp[iSymm][2][d];
        //
	kPointTemp[0] = kPointTemp[0] - dkx ;
        kPointTemp[1] = kPointTemp[1] - dky ;
        kPointTemp[2] = kPointTemp[2] - dkz ;
        for ( unsigned int dir=0;  dir < 3; ++dir) {
            while (kPointTemp[dir] > (1.0-1.0E-5) )
              kPointTemp[dir] = kPointTemp[dir] - 1.0 ;
	    while (kPointTemp[dir] < -1.0E-5 )
              kPointTemp[dir] = kPointTemp[dir] + 1.0 ;
        }
        //
        const unsigned int jk =  round(kPointTemp[0]*nkx)*nky*nkz + round(kPointTemp[1]*nky)*nkz + round( kPointTemp[2]*nkz);
        if( jk!=ik && jk<nk && discard[jk]!=1) {
           d_kPointWeights[d_maxkPoints-1] = d_kPointWeights[d_maxkPoints-1] + 1.0/nk;
           discard[jk] = 1;
	   pcout<< "    " << ik << "     " << jk << std::endl ;
           if (countedSymm[iSymm]==0) {
	       usedSymmNum[iSymm] = usedSymm ;
               (symmetryPtr->symmMat)[usedSymm] = symmMatTemp2[iSymm] ;
    		for (unsigned int j=0; j<3; ++j)
     		    (symmetryPtr->translation)[usedSymm][j] = translationTemp[iSymm][j];
	       usedSymm++ ;
	       countedSymm[iSymm] = 1;
             }
	   symmUnderGroupTemp[usedSymmNum[iSymm]].push_back(d_maxkPoints-1) ;
	   }
       }
    //
    discard[ik]=1;
    ik = ik + 1 ;
    if (ik < nk) {
        while (discard[ik]==1) {
            ik = ik + 1 ;
            if(ik == nk)
                break;
        }
    }
  }
  //
  symmetryPtr->numSymm = usedSymm;
  symmetryPtr->symmUnderGroup.resize (d_maxkPoints, std::vector<int>(symmetryPtr->numSymm,0));
  symmetryPtr->numSymmUnderGroup.resize(d_maxkPoints,1) ; // minimum should be 1, because identity is always present
  for (unsigned int i=0; i<d_maxkPoints; ++i){
      //
      symmetryPtr->symmUnderGroup[i][0] = 1;
      for (unsigned int iSymm = 1; iSymm<(symmetryPtr->numSymm); ++iSymm) {
         if(std::find(symmUnderGroupTemp[iSymm].begin(), symmUnderGroupTemp[iSymm].end(), i) != symmUnderGroupTemp[iSymm].end()) {
            symmetryPtr->symmUnderGroup[i][iSymm] = 1 ;
	    symmetryPtr->numSymmUnderGroup[i] += 1 ;
     }
   }
   pcout << " kpoint " << i << " numSymmUnderGroup " << symmetryPtr->numSymmUnderGroup[i] << std::endl;
  }
  //
  pcout<<" " << usedSymm << " symmetries used to reduce BZ "  << std::endl;
  for (unsigned int iSymm = 0; iSymm < symmetryPtr->numSymm; ++iSymm)
	{
         for ( unsigned int ipol = 0; ipol<3; ++ipol)
		 pcout << symmetryPtr->symmMat[iSymm][ipol][0] << "  " << symmetryPtr->symmMat[iSymm][ipol][1] << "  " << symmetryPtr->symmMat[iSymm][ipol][2] << std::endl;
  }
  //
  pcout<<" number of irreducible k-points " << d_maxkPoints << std::endl;
  }



  pcout<<"Reduced k-Point-coordinates and weights: "<<std::endl;
  char buffer[100];
  for(int i = 0; i < d_maxkPoints; ++i){
    sprintf(buffer, "  %5u:  %12.5f  %12.5f %12.5f %12.5f\n", i+1, kPointReducedCoordinates[3*i+0], kPointReducedCoordinates[3*i+1], kPointReducedCoordinates[3*i+2],d_kPointWeights[i]);
    pcout << buffer;
  }

 // Convert from crystal to Cartesian coordinates

  for(int i = 0; i < d_maxkPoints; ++i)
    for (unsigned int d=0; d < 3; ++d)
      d_kPointCoordinates[3*i + d] = kPointReducedCoordinates[3*i+0]*d_reciprocalLatticeVectors[0][d] +
                                     kPointReducedCoordinates[3*i+1]*d_reciprocalLatticeVectors[1][d] +
                                     kPointReducedCoordinates[3*i+2]*d_reciprocalLatticeVectors[2][d];
   //
   // Split k-points over pools
   //
   AssertThrow(d_maxkPoints>=npool,ExcMessage("Number of k-points should be higher than or equal to number of pools"));
   const unsigned int this_mpi_pool (Utilities::MPI::this_mpi_process(interpoolcomm)) ;
   std::vector<double> d_kPointCoordinatesGlobal(3*d_maxkPoints, 0.0) ;
   std::vector<double> d_kPointWeightsGlobal(d_maxkPoints, 0.0) ;
   std::vector<double> kPointReducedCoordinatesGlobal(3*d_maxkPoints, 0.0) ;   
   for(unsigned int i = 0; i < d_maxkPoints; ++i)
    {
       for (unsigned int d=0; d < 3; ++d)
       {
         d_kPointCoordinatesGlobal[3*i + d] = d_kPointCoordinates[3*i + d];
	 kPointReducedCoordinatesGlobal[3*i + d] = kPointReducedCoordinates[3*i + d];
       }
     d_kPointWeightsGlobal[i] = d_kPointWeights[i] ;
    }
   //
   const unsigned int d_maxkPointsGlobal = d_maxkPoints ;
   d_kPointCoordinates.clear() ;
   kPointReducedCoordinates.clear();
   d_kPointWeights.clear() ;
   d_maxkPoints = d_maxkPointsGlobal / npool ;
   const unsigned int rest = d_maxkPointsGlobal%npool ;
   if (this_mpi_pool < rest)
       d_maxkPoints = d_maxkPoints + 1 ;
   //
   d_kPointCoordinates.resize(3*d_maxkPoints, 0.0) ;
   kPointReducedCoordinates.resize(3*d_maxkPoints, 0.0);
   d_kPointWeights.resize(d_maxkPoints, 0.0) ;
   //
   std::vector<int> sendSizekPoints1(npool, 0), mpiOffsetskPoints1(npool, 0) ;
   std::vector<int> sendSizekPoints2(npool, 0), mpiOffsetskPoints2(npool, 0) ;
   if (this_mpi_pool==0) {
   //
   for (unsigned int i=0; i < npool; ++i) {
	sendSizekPoints1[i] = 3*(d_maxkPointsGlobal / npool) ;
	sendSizekPoints2[i] = d_maxkPointsGlobal / npool ;
	if (i < rest){
	   sendSizekPoints1[i] = sendSizekPoints1[i] + 3 ;
	   sendSizekPoints2[i] = sendSizekPoints2[i] + 1 ;
        }
    if (i > 0){
  	  mpiOffsetskPoints1[i] = mpiOffsetskPoints1[i-1] + sendSizekPoints1[i-1] ;
	  mpiOffsetskPoints2[i] = mpiOffsetskPoints2[i-1] + sendSizekPoints2[i-1] ;
    }
   }
   }
   //pcout << sendSizekPoints[0] << "  " << sendSizekPoints[1] << " " << d_maxkPoints << std::endl;
   //
   MPI_Scatterv(&(d_kPointCoordinatesGlobal[0]),&(sendSizekPoints1[0]), &(mpiOffsetskPoints1[0]), MPI_DOUBLE, &(d_kPointCoordinates[0]), 3*d_maxkPoints, MPI_DOUBLE, 0, interpoolcomm);
   MPI_Scatterv(&(d_kPointWeightsGlobal[0]),&(sendSizekPoints2[0]), &(mpiOffsetskPoints2[0]), MPI_DOUBLE, &(d_kPointWeights[0]), d_maxkPoints, MPI_DOUBLE, 0, interpoolcomm);
   MPI_Scatterv(&(kPointReducedCoordinatesGlobal[0]),&(sendSizekPoints1[0]), &(mpiOffsetskPoints1[0]), MPI_DOUBLE, &(kPointReducedCoordinates[0]), 3*d_maxkPoints, MPI_DOUBLE, 0, interpoolcomm);   

}
