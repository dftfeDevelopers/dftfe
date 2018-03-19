//
// File:      NonLinearSolver.cc
// Package:   dft
//
// Density Functional Theory
//
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
// @author Sambit Das (2018)
//
#include "../../include/nonLinearSolver.h"

  //
  // Constructor.
  //
  nonLinearSolver::nonLinearSolver()
  {
  }

  //
  // Destructor.
  //
  nonLinearSolver::~nonLinearSolver()
  {

    //
    //
    //
    return;

  }

  //
  // Get tolerance.
  //
  double 
  nonLinearSolver::getTolerance() const
  {

    //
    //
    //
    return d_tolerance;

  }
  
  //
  // Get maximum number of iterations.
  //
  unsigned int 
  nonLinearSolver::getMaximumNumberIterations() const
  {

    //
    //
    //
    return d_maxNumberIterations;

  }


  //
  // Get debug level.
  //
  unsigned int 
  nonLinearSolver::getDebugLevel() const
  {
    
    //
    //
    //
    return d_debugLevel;

  }


