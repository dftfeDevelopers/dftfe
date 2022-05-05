// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022 The Regents of the University of Michigan and DFT-FE
// authors.
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

//
// dft header
//
#include "dftfeWrapper.h"

//
// C++ headers
//
#include <fstream>
#include <iostream>
#include <list>
#include <sstream>
#include <sys/stat.h>



int
main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  dftfe::dftfeWrapper::globalHandlesInitialize();

  std::vector<std::vector<double>> atomicPositionsCart(
    2, std::vector<double>(3, 0.0));
  atomicPositionsCart[0][0] = -2.0;
  atomicPositionsCart[0][0] = 2.0;

  std::vector<std::vector<double>> cell(3, std::vector<double>(3, 0.0));
  cell[0][0] = 20.0;
  cell[1][1] = 20.0;
  cell[2][2] = 20.0;

  std::vector<unsigned int> atomicNumbers(atomicPositionsCart.size());
  atomicNumbers[0] = 7;
  atomicNumbers[1] = 7;

  std::vector<bool> pbc(3, false);

  dftfe::dftfeWrapper dftfeWrapped(
    MPI_COMM_WORLD, false, atomicPositionsCart, atomicNumbers, cell, pbc);

  double energy = dftfeWrapped.computeDFTFreeEnergy(true, false);

  dftfe::dftfeWrapper::globalHandlesFinalize();

  MPI_Finalize();
  return 0;
}
