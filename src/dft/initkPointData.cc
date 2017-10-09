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
  std::vector<std::vector<double> > d_reciprocalLatticeVectors;
  std::vector<double> kPointReducedCoordinates;
  char kPointRuleFile[256];
  sprintf(kPointRuleFile, "%s/data/kPointList/%s", currentPath.c_str(), kPointDataFile.c_str());
  readFile(numberColumnskPointDataFile, kPointData, kPointRuleFile);

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
  std::vector<std::vector<double> > d_reciprocalLatticeVectors;
  std::vector<double> kPointReducedCoordinates, del(3);
  d_maxkPoints = (nkx * nky) * nkz;
  pcout<<"Total number of k-points " << d_maxkPoints << std::endl;
  //
  del[0] = 1.0/float(nkx); del[1] = 1.0/float(nky); del[2] = 1.0/float(nkz);
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
      kPointReducedCoordinates[3*i + 2] = del[2]*(i%nkz) ;
      kPointReducedCoordinates[3*i + 1] = del[1]*(std::floor( ( i%(nkz*nky) ) / nkz) ) ;
      kPointReducedCoordinates[3*i + 0] = del[0]*(std::floor( (i/(nkz*nky) ) ) );
      for (unsigned int dir = 0; dir < 3; ++dir) {
         if(kPointReducedCoordinates[3*i + dir] > (0.5-del[dir]) )
              kPointReducedCoordinates[3*i + dir] = kPointReducedCoordinates[3*i + dir] - 1.0 ;
      }
      d_kPointWeights[i] = 1.0/d_maxkPoints ;
    }

  // Get the reciprocal lattice vectors
  d_reciprocalLatticeVectors.resize(3,std::vector<double> (3,0.0));
  d_reciprocalVectors.resize(3,std::vector<double> (3,0.0));
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
  
   d_reciprocalVectors = d_reciprocalLatticeVectors ;
  
  //
  int numberColumnsSymmDataFile = 3;
  std::vector<std::vector<double> > symmData;
  readFile(numberColumnsSymmDataFile, symmData, symmDataFile);
  unsigned int numSymm = symmData.size()/3 ;
  symmMat.resize( numSymm );
  for (unsigned int iSymm = 0; iSymm < numSymm; ++iSymm) 
       symmMat[iSymm].resize(3,std::vector<double> (3,0.0)) ;
  //
  for(int iSymm = 0; iSymm < numSymm; ++iSymm)
  {
    for(int j = 0; j < 3; ++j)
      {
        symmMat[iSymm][j][0] = symmData[3*iSymm+j][0];
        symmMat[iSymm][j][1] = symmData[3*iSymm+j][1];
        symmMat[iSymm][j][2] = symmData[3*iSymm+j][2];
      }
  }
  //unsigned int numSymm = 1 ;
  //symmMat.resize(numSymm);
  //for (unsigned int iSymm = 0; iSymm < numSymm; ++iSymm) 
  //     symmMat[iSymm].resize(3,std::vector<double> (3,0.0)) ;
  //symmMat[0][0][0] = 1.0 ;
  //symmMat[0][1][1] = 1.0 ;
  //symmMat[0][2][2] = 1.0 ;
  //
  std::vector<double> kPointAllCoordinates, kPointTemp(3); 
  std::vector<int> discard(d_maxkPoints, 0) ;
  std::vector<int> usedSymm(d_maxkPoints, 0) ;
  kPointAllCoordinates = kPointReducedCoordinates ;
  int nk = d_maxkPoints ;
  d_maxkPoints = 0;
  unsigned int ik = 0;
  while( ik < nk ) {
    //
    kPointReducedCoordinates[3*d_maxkPoints + 0] = kPointAllCoordinates[3*ik+0] ;
    kPointReducedCoordinates[3*d_maxkPoints + 1] = kPointAllCoordinates[3*ik+1] ;
    kPointReducedCoordinates[3*d_maxkPoints + 2] = kPointAllCoordinates[3*ik+2] ;
    d_maxkPoints = d_maxkPoints + 1 ;
    //
    for (unsigned int iSymm = 0; iSymm < numSymm; ++iSymm) 
	{
        
        kPointTemp[0] = kPointAllCoordinates[3*ik+0]*symmMat[iSymm][0][0] + kPointAllCoordinates[3*ik+1]*symmMat[iSymm][0][1] + kPointAllCoordinates[3*ik+2]*symmMat[iSymm][0][2];
	kPointTemp[1] = kPointAllCoordinates[3*ik+0]*symmMat[iSymm][1][0] + kPointAllCoordinates[3*ik+1]*symmMat[iSymm][1][1] + kPointAllCoordinates[3*ik+2]*symmMat[iSymm][1][2];
	kPointTemp[2] = kPointAllCoordinates[3*ik+0]*symmMat[iSymm][2][0] + kPointAllCoordinates[3*ik+1]*symmMat[iSymm][2][1] + kPointAllCoordinates[3*ik+2]*symmMat[iSymm][2][2];
        //
        for ( unsigned int dir=0;  dir < 3; ++dir) {
            while (kPointTemp[dir] > (1.0-del[dir]) )
              kPointTemp[dir] = kPointTemp[dir] - 1.0 ;
	    while (kPointTemp[dir] < 0.0 )
              kPointTemp[dir] = kPointTemp[dir] + 1.0 ;
        }
        //
        //double xx = std::abs( (std::round(kPointTemp(0))*nkx) - kPointTemp(0))*nkx ) ;
        //double yy = std::abs( (std::round(kPointTemp(1))*nky) - kPointTemp(1))*nky ) ;
        //double zz = std::abs( (std::round(kPointTemp(2))*nkz) - kPointTemp(2))*nkz ) ;
        //           
        unsigned int jk =  round(kPointTemp[0]*nkx)*nky*nkz + round(kPointTemp[1]*nky)*nkz + round( kPointTemp[2]*nkz)  ;                   
        if( jk!=ik && discard[jk]!=1) 
           d_kPointWeights[ik] = d_kPointWeights[ik] + 1.0/nk;   
	   usedSymm[d_maxkPoints-1][iSymm] = 1 ;         
           discard[jk] = 1;                    
       }
    //
    ik = ik + 1 ;
    if (ik < nk) {
        while (discard[ik]==1) {
            ik = ik + 1 ;
            if(ik == nk)
                break;
        }    
    }
  }
    
  pcout<<" number of irreducible k-points " << d_maxkPoints << std::endl;
   

  
  pcout<<"Reduced k-Point-coordinates and weights: "<<std::endl;

  for(int i = 0; i < d_maxkPoints; ++i)
    {
      pcout<<kPointReducedCoordinates[3*i + 0]<<" "<<kPointReducedCoordinates[3*i + 1]<<" "<<kPointReducedCoordinates[3*i + 2]<<" "<<d_kPointWeights[i]<<std::endl;
    }

 // Convert from crystal to Cartesian coordinates

  for(int i = 0; i < d_maxkPoints; ++i)
    {
      d_kPointCoordinates[3*i + 0] = kPointReducedCoordinates[3*i+0]*d_reciprocalLatticeVectors[0][0] + kPointReducedCoordinates[3*i+1]*d_reciprocalLatticeVectors[1][0] + kPointReducedCoordinates[3*i+2]*d_reciprocalLatticeVectors[2][0];
      d_kPointCoordinates[3*i + 1] = kPointReducedCoordinates[3*i+0]*d_reciprocalLatticeVectors[0][1] + kPointReducedCoordinates[3*i+1]*d_reciprocalLatticeVectors[1][1] + kPointReducedCoordinates[3*i+2]*d_reciprocalLatticeVectors[2][1];
      d_kPointCoordinates[3*i + 2] = kPointReducedCoordinates[3*i+0]*d_reciprocalLatticeVectors[0][2] + kPointReducedCoordinates[3*i+1]*d_reciprocalLatticeVectors[1][2] + kPointReducedCoordinates[3*i+2]*d_reciprocalLatticeVectors[2][2];
    }

 
 
}
