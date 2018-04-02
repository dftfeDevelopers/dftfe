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
// @author Sambit Das
//

#ifndef projectFields_H_
#define projectFields_H_
#include "headers.h"

#include <iostream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <complex>
#include <deque>

namespace vectorTools
{
    typedef dealii::parallel::distributed::Vector<double> vectorType;

    class projectFieldsFromPreviousMesh
    {
     public:
      /**
       * projectFieldsFromPreviousMesh constructor
       */
      projectFieldsFromPreviousMesh(const MPI_Comm &mpi_communicator);

      void project(const dealii::parallel::distributed::Triangulation<3> & triangulationSerPrev,
		   const dealii::parallel::distributed::Triangulation<3> & triangulationParPrev,
		   const dealii::parallel::distributed::Triangulation<3> & triangulationParCurrent,
		   const dealii::FESystem<3> & FE,
		   const dealii::ConstraintMatrix & constraintsCurrentDof,
		   const std::vector<vectorType*> & fieldsPreviousMesh,
		   std::vector<vectorType*> & fieldsCurrentMesh);

     private:

      MPI_Comm mpi_communicator;
      const unsigned int n_mpi_processes;
      const unsigned int this_mpi_process;

      ///  parallel message stream
      dealii::ConditionalOStream   pcout;
    };

}

#endif
