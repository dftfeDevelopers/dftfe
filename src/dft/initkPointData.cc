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
