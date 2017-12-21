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

#ifndef meshMovementGaussian_H_
#define meshMovementGaussian_H_
#include "meshMovement.h"

class meshMovementGaussianClass : public meshMovementClass
{

public:
  meshMovementGaussianClass();	
  void moveMesh(std::vector<Point<C_DIM> > controlPointLocations,
                std::vector<Tensor<1,3,double> > controlPointDisplacements,
                double controllingParameter);
private:  
  void computeIncrement();  
  //move mesh data
  std::vector<Point<C_DIM> > d_controlPointLocations;
  std::vector<Tensor<1,C_DIM,double> > d_controlPointDisplacements;
  double d_controllingParameter;  
};

#endif
