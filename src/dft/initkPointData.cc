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

void cross_product(std::vector<double> &a,
		   std::vector<double> &b,
		   std::vector<double> &crossProduct)
{
  crossProduct.resize(a.size(),0.0);

  crossProduct[0] = a[1]*b[2]-a[2]*b[1];
  crossProduct[1] = a[2]*b[0]-a[0]*b[2];
  crossProduct[2] = a[0]*b[1]-a[1]*b[0];

}

template<unsigned int FEOrder>
void dftClass<FEOrder>::readkPointData()
{
  int numberColumnskPointDataFile = 4;
  std::vector<std::vector<double> > kPointData;
  std::vector<double> kPointReducedCoordinates;
  char kPointRuleFile[256];
  sprintf(kPointRuleFile, "%s/data/kPointList/%s", dftParameters::currentPath.c_str(), dftParameters::kPointDataFile.c_str());
  dftUtils::readFile(numberColumnskPointDataFile, kPointData, kPointRuleFile);

  d_maxkPoints = kPointData.size();
  d_kPointCoordinates.resize(d_maxkPoints*3,0.0);
  d_kPointWeights.resize(d_maxkPoints,0.0);

  kPointReducedCoordinates = d_kPointCoordinates;

  for(int i = 0; i < d_maxkPoints; ++i)
    {
      kPointReducedCoordinates[3*i + 0] = kPointData[i][0];
      kPointReducedCoordinates[3*i + 1] = kPointData[i][1];
      kPointReducedCoordinates[3*i + 2] = kPointData[i][2];
      d_kPointWeights[i] = kPointData[i][3];
    }
  
  pcout<<"Reduced k-Point-coordinates and weights: "<<std::endl;

  for(int i = 0; i < d_maxkPoints; ++i)
    {
      pcout<<kPointReducedCoordinates[3*i + 0]<<" "<<kPointReducedCoordinates[3*i + 1]<<" "<<kPointReducedCoordinates[3*i + 2]<<" "<<d_kPointWeights[i]<<std::endl;
    }

  d_reciprocalLatticeVectors.resize(3,
				    std::vector<double> (3,0.0));

  //
  //convert them into reciprocal lattice vectors
  //
  for(int i = 0; i < 2; ++i)
    {
      std::vector<double> cross(3,0.0);
      cross_product(d_latticeVectors[i+1],
		    d_latticeVectors[3 - (2*i + 1)],
		    cross);

      double scalarConst = d_latticeVectors[i][0]*cross[0] + d_latticeVectors[i][1]*cross[1] + d_latticeVectors[i][2]*cross[2];

      d_reciprocalLatticeVectors[i][0] = (2*M_PI/scalarConst)*cross[0];
      d_reciprocalLatticeVectors[i][1] = (2*M_PI/scalarConst)*cross[1];
      d_reciprocalLatticeVectors[i][2] = (2*M_PI/scalarConst)*cross[2];
    }

  //
  //fill up 3rd reciprocal lattice vector
  //
  std::vector<double> cross(3,0.0);
  cross_product(d_latticeVectors[0],
		d_latticeVectors[1],
		cross);

  double scalarConst = d_latticeVectors[2][0]*cross[0] + d_latticeVectors[2][1]*cross[1] + d_latticeVectors[2][2]*cross[2];

  d_reciprocalLatticeVectors[2][0] = (2*M_PI/scalarConst)*cross[0];
  d_reciprocalLatticeVectors[2][1] = (2*M_PI/scalarConst)*cross[1];
  d_reciprocalLatticeVectors[2][2] = (2*M_PI/scalarConst)*cross[2];

  for(int i = 0; i < d_maxkPoints; ++i)
    {
      d_kPointCoordinates[3*i + 0] = kPointReducedCoordinates[3*i+0]*d_reciprocalLatticeVectors[0][0] + kPointReducedCoordinates[3*i+1]*d_reciprocalLatticeVectors[1][0] + kPointReducedCoordinates[3*i+2]*d_reciprocalLatticeVectors[2][0];
      d_kPointCoordinates[3*i + 1] = kPointReducedCoordinates[3*i+0]*d_reciprocalLatticeVectors[0][1] + kPointReducedCoordinates[3*i+1]*d_reciprocalLatticeVectors[1][1] + kPointReducedCoordinates[3*i+2]*d_reciprocalLatticeVectors[2][1];
      d_kPointCoordinates[3*i + 2] = kPointReducedCoordinates[3*i+0]*d_reciprocalLatticeVectors[0][2] + kPointReducedCoordinates[3*i+1]*d_reciprocalLatticeVectors[1][2] + kPointReducedCoordinates[3*i+2]*d_reciprocalLatticeVectors[2][2];
    }

 
 
}
template<unsigned int FEOrder>
void dftClass<FEOrder>::generateMPGrid()
{
  std::vector<double> kPointReducedCoordinates, del(3);
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

  for(int i = 0; i < d_maxkPoints; ++i) 
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
  d_reciprocalLatticeVectors.resize(3,std::vector<double> (3,0.0));
  for(int i = 0; i < 2; ++i)
    {
      std::vector<double> cross(3,0.0);
      cross_product(d_latticeVectors[i+1],
		    d_latticeVectors[3 - (2*i + 1)],
		    cross);

      double scalarConst = d_latticeVectors[i][0]*cross[0] + d_latticeVectors[i][1]*cross[1] + d_latticeVectors[i][2]*cross[2];

      d_reciprocalLatticeVectors[i][0] = (2*M_PI/scalarConst)*cross[0];
      d_reciprocalLatticeVectors[i][1] = (2*M_PI/scalarConst)*cross[1];
      d_reciprocalLatticeVectors[i][2] = (2*M_PI/scalarConst)*cross[2];
    }

  //
  //fill up 3rd reciprocal lattice vector
  //
  std::vector<double> cross(3,0.0);
  cross_product(d_latticeVectors[0],
		d_latticeVectors[1],
		cross);

  double scalarConst = d_latticeVectors[2][0]*cross[0] + d_latticeVectors[2][1]*cross[1] + d_latticeVectors[2][2]*cross[2];

  d_reciprocalLatticeVectors[2][0] = (2*M_PI/scalarConst)*cross[0];
  d_reciprocalLatticeVectors[2][1] = (2*M_PI/scalarConst)*cross[1];
  d_reciprocalLatticeVectors[2][2] = (2*M_PI/scalarConst)*cross[2];
  
  if (useSymm) {
  //
  int numberColumnsSymmDataFile = 3;
  std::vector<std::vector<int>> symmUnderGroupTemp ;
  std::vector<std::vector<double> > symmData;
  std::vector<std::vector<std::vector<double> >> symmMatTemp;
  if (symmFromFile) {
  dftUtils::readFile(numberColumnsSymmDataFile, symmData, symmDataFile);
  numSymm = symmData.size()/3 ;
  pcout<<" number of symmetries read from file " << numSymm << std::endl;
  symmMatTemp.resize( numSymm );
  symmMat.resize( numSymm );
  symmUnderGroupTemp.resize(numSymm);
  for (unsigned int iSymm = 0; iSymm < numSymm; ++iSymm) {
       symmMatTemp[iSymm].resize(3,std::vector<double> (3,0.0)) ;
       symmMat[iSymm].resize(3,std::vector<double> (3,0.0)) ;
  }
  //
  for(int iSymm = 0; iSymm < numSymm; ++iSymm)
  {
    translation[iSymm][0] = 0.0 ;
    translation[iSymm][1] = 0.0 ;
    translation[iSymm][2] = 0.0 ;
    for(int j = 0; j < 3; ++j)
      {
        symmMatTemp[iSymm][j][0] = symmData[3*iSymm+j][0];
        symmMatTemp[iSymm][j][1] = symmData[3*iSymm+j][1];
        symmMatTemp[iSymm][j][2] = symmData[3*iSymm+j][2];
      }
  }
  }
  else {
  //////////////////////////////////////////  SPG CALL  ///////////////////////////////////////////////////
  
  const int num_atom = atomLocationsFractional.size();
  double lattice[3][3], position[num_atom][3];
  int types[num_atom] ;
  int mesh[3] = {nkx, nky, nkz};
  int grid_address[nkx * nky * nkz][3];
  int grid_mapping_table[nkx * nky * nkz];
  //
  int max_size = 500;
  int rotation[max_size][3][3];
  for (unsigned int i=0; i<3; ++i) {
     for (unsigned int j=0; j<3; ++j)
         lattice[i][j] = d_latticeVectors[i][j];
  }
   std::set<unsigned int>::iterator it = atomTypes.begin();  
  for (unsigned int i=0; i<num_atom; ++i){
      std::advance(it, i);
      types[i] = atomLocationsFractional[i][0];
      for (unsigned int j=0; j<3; ++j)
      position[i][j] = atomLocationsFractional[i][j+2] ;
   }
  // ***********************************  Checking on SPG ******************************************** 
  /*int max_size = 500;
  int num_atom = 2 ;
  int rotation[max_size][3][3];
  double lattice[3][3], position[num_atom][3];
  int types[num_atom] ;
  //
  lattice[0][0] = 5.131550 ; lattice[0][1]=5.131550 ; lattice[0][2]=0.000000 ;
  lattice[1][0] = 5.131550 ; lattice[1][1]=0.000000 ; lattice[1][2]=5.131550 ;
  lattice[2][0] = 0.000000 ; lattice[2][1]=5.131550 ; lattice[2][2]=5.131550 ;
  //
  position[0][0] = 0.000000; position[0][1] = 0.0000000; position[0][2] = 0.000000 ; 
  position[1][0] = 0.125000; position[1][1] = 0.1250000; position[1][2] = 0.125000 ;
  //
  types[0]=1; types[1]=1; */
  //**************************************************************************************************
  pcout<<" getting space group symmetries from spg " << std::endl;
  numSymm = spg_get_symmetry(rotation,
                     translation,
                     max_size,
                     lattice,
                     position,
                     types,
                     num_atom,
                     1e-5);
  pcout<<" number of symmetries allowed for the lattice " << numSymm << std::endl;
  for (unsigned int iSymm=0; iSymm<numSymm; ++iSymm){
	pcout << " Symmetry " << iSymm+1 << std::endl ;
	pcout << " Rotation " << std::endl;
	for ( unsigned int ipol = 0; ipol<3; ++ipol)
		 pcout << rotation[iSymm][ipol][0] << "  " <<rotation[iSymm][ipol][1] << "  " << rotation[iSymm][ipol][2] << std::endl;
	pcout << " translation " << std::endl;
	pcout << translation[iSymm][0] << "  " <<translation[iSymm][1] << "  " << translation[iSymm][2] << std::endl;	
	pcout << "	" << std::endl ;
  }
  symmMatTemp.resize( numSymm );
  symmMat.resize( numSymm );
  symmUnderGroupTemp.resize(numSymm);
  for (unsigned int i=0; i<numSymm; ++i) {
     symmMatTemp[i].resize(3,std::vector<double> (3,0.0)) ;
     for (unsigned int j=0; j<3; ++j) {
        for (unsigned int k=0; k<3; ++k)
         symmMatTemp[i][j][k] = double(rotation[i][j][k]);
    }
  }
  }
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //
  std::vector<double> kPointAllCoordinates, kPointTemp(3); 
  std::vector<int> discard(d_maxkPoints, 0), countedSymm(numSymm, 0), usedSymmNum(numSymm, 1) ;
  kPointAllCoordinates = kPointReducedCoordinates ;
  int nk = d_maxkPoints ;
  d_maxkPoints = 0;
  
  double translationTemp[numSymm][3];
  for (unsigned int i=0; i<numSymm; ++i) {
    for (unsigned int j=0; j<3; ++j)
     translationTemp[i][j] = translation[i][j];
  }
  //
  symmMat[0] = symmMatTemp[0] ;
  //symmMat = symmMatTemp ;
  unsigned int usedSymm=1, ik = 0; // note usedSymm is initialized to 1 and not 0. Because identity is always present
  countedSymm[0] = 1 ;
  //
  while( ik < nk ) {
    //
    kPointReducedCoordinates[3*d_maxkPoints + 0] = kPointAllCoordinates[3*ik+0] ;
    kPointReducedCoordinates[3*d_maxkPoints + 1] = kPointAllCoordinates[3*ik+1] ;
    kPointReducedCoordinates[3*d_maxkPoints + 2] = kPointAllCoordinates[3*ik+2] ;
    d_maxkPoints = d_maxkPoints + 1 ;
    //
    for (unsigned int iSymm = 1; iSymm < numSymm; ++iSymm) // iSymm begins from 1. because identity is always present and is taken care of.
	{
        
        kPointTemp[0] = kPointAllCoordinates[3*ik+0]*symmMatTemp[iSymm][0][0] + kPointAllCoordinates[3*ik+1]*symmMatTemp[iSymm][1][0] + kPointAllCoordinates[3*ik+2]*symmMatTemp[iSymm][2][0];
	kPointTemp[1] = kPointAllCoordinates[3*ik+0]*symmMatTemp[iSymm][0][1] + kPointAllCoordinates[3*ik+1]*symmMatTemp[iSymm][1][1] + kPointAllCoordinates[3*ik+2]*symmMatTemp[iSymm][2][1];
	kPointTemp[2] = kPointAllCoordinates[3*ik+0]*symmMatTemp[iSymm][0][2] + kPointAllCoordinates[3*ik+1]*symmMatTemp[iSymm][1][2] + kPointAllCoordinates[3*ik+2]*symmMatTemp[iSymm][2][2];
        //
	kPointTemp[0] = kPointTemp[0] - dkx ; kPointTemp[1] = kPointTemp[1] - dky ; kPointTemp[2] = kPointTemp[2] - dkz ;
        for ( unsigned int dir=0;  dir < 3; ++dir) {
            while (kPointTemp[dir] > (1.0-1.0E-5) )
              kPointTemp[dir] = kPointTemp[dir] - 1.0 ;
	    while (kPointTemp[dir] < -1.0E-5 )
              kPointTemp[dir] = kPointTemp[dir] + 1.0 ;
        }
        //           
        unsigned int jk =  round(kPointTemp[0]*nkx)*nky*nkz + round(kPointTemp[1]*nky)*nkz + round( kPointTemp[2]*nkz)  ;                   
        if( jk!=ik && jk<nk && discard[jk]!=1) {	   
           d_kPointWeights[d_maxkPoints-1] = d_kPointWeights[d_maxkPoints-1] + 1.0/nk;        
           discard[jk] = 1;
	   pcout<< "    " << ik << "     " << jk << std::endl ;
           if (countedSymm[iSymm]==0) {
	       usedSymmNum[iSymm] = usedSymm ;
               symmMat[usedSymm] = symmMatTemp[iSymm] ;
    		for (unsigned int j=0; j<3; ++j)
     		    translation[usedSymm][j] = translationTemp[iSymm][j];
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
  numSymm = usedSymm;
  symmUnderGroup.resize (d_maxkPoints, std::vector<int>(numSymm,0) ) ;
  numSymmUnderGroup.resize(d_maxkPoints,1) ; // minimum should be 1, because identity is always present
  for (unsigned int i=0; i<d_maxkPoints; ++i){
      //
      symmUnderGroup[i][0] = 1 ;
      for (unsigned int iSymm = 1; iSymm<numSymm; ++iSymm) {
         if(std::find(symmUnderGroupTemp[iSymm].begin(), symmUnderGroupTemp[iSymm].end(), i) != symmUnderGroupTemp[iSymm].end()) {
            symmUnderGroup[i][iSymm] = 1 ;
	    numSymmUnderGroup[i] += 1 ;
     }
   }
   pcout << " kpoint " << i << " numSymmUnderGroup " << numSymmUnderGroup[i] << std::endl;
  }
  //
  pcout<<" " << usedSymm << " symmetries used to reduce BZ "  << std::endl;  
  for (unsigned int iSymm = 0; iSymm < numSymm; ++iSymm) 
	{
         for ( unsigned int ipol = 0; ipol<3; ++ipol)
		 pcout << symmMat[iSymm][ipol][0] << "  " << symmMat[iSymm][ipol][1] << "  " << symmMat[iSymm][ipol][2] << std::endl;
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
    {
      d_kPointCoordinates[3*i + 0] = kPointReducedCoordinates[3*i+0]*d_reciprocalLatticeVectors[0][0] + kPointReducedCoordinates[3*i+1]*d_reciprocalLatticeVectors[1][0] + kPointReducedCoordinates[3*i+2]*d_reciprocalLatticeVectors[2][0];
      d_kPointCoordinates[3*i + 1] = kPointReducedCoordinates[3*i+0]*d_reciprocalLatticeVectors[0][1] + kPointReducedCoordinates[3*i+1]*d_reciprocalLatticeVectors[1][1] + kPointReducedCoordinates[3*i+2]*d_reciprocalLatticeVectors[2][1];
      d_kPointCoordinates[3*i + 2] = kPointReducedCoordinates[3*i+0]*d_reciprocalLatticeVectors[0][2] + kPointReducedCoordinates[3*i+1]*d_reciprocalLatticeVectors[1][2] + kPointReducedCoordinates[3*i+2]*d_reciprocalLatticeVectors[2][2];
    }

 
 
}
