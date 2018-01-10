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
// @author Phani Motamarri (2017)
//

#ifndef meshGenerator_H_
#define meshGenerator_H_
#include "headers.h"
#include "constants.h"

using namespace dealii;


class meshGeneratorClass
{

 public:
  /**
   * meshGeneratorClass constructor
   */
  meshGeneratorClass();


  /**
   * meshGeneratorClass destructor
   */
  ~meshGeneratorClass();

 

  void generateSerialAndParallelMesh(std::vector<std::vector<double> > & atomLocations,
				     std::vector<std::vector<double> > & imageAtomLocations,
				     std::vector<std::vector<double> > & domainBoundingVectors);

  void generateMesh(Triangulation<3,3> & triangulation,
		    unsigned int & numberGlobalCells);

  Triangulation<3,3> & getSerialMesh();

  parallel::distributed::Triangulation<3> & getParallelMesh();

  
  
 private:
  //
  //data members
  //
  parallel::distributed::Triangulation<3> d_parallelTriangulationUnmoved;
  parallel::distributed::Triangulation<3> d_parallelTriangulationMoved;

  Triangulation<3,3> d_serialTriangulationUnmoved;
  Triangulation<3,3> d_serialTriangulationMoved;

  std::vector<std::vector<double> > d_atomPositions;
  std::vector<std::vector<double> > d_imageAtomPositions;
  std::vector<std::vector<double> > d_domainBoundingVectors;

  //
  //parallel objects
  //
  MPI_Comm mpi_communicator;
  const unsigned int this_mpi_process;
  dealii::ConditionalOStream   pcout;  

  //
  //compute-time logger
  //
  TimerOutput computing_timer;

};

#endif
