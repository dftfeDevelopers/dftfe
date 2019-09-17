// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2018 The Regents of the University of Michigan and DFT-FE authors.
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


#ifndef interpolateFields_H_
#define interpolateFields_H_
#include "headers.h"

#include <iostream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <complex>
#include <deque>

namespace dftfe
{
  namespace vectorTools
  {

/**
 *  @brief Interpolate solutions fields from one finite element mesh to another.
 *
 *   Unlike the dealii function  VectorTools::interpolate_to_different_mesh, this function
 *   doesn't assume that the parallel partitioning of the two meshes are same.
 *   Further the two meshes can be arbitraririly refined. The only constraint is that they
 *   must discretize the same real-space domain.
 *
 *  @author Sambit Das
 */
    class interpolateFieldsFromPreviousMesh
    {
     //typedef dealii::parallel::distributed::Vector<double> vectorType;
     public:
    /** @brief Constructor.
     *
     *  @param mpi_comm mpi_communicator of the domain decomposition
     */
      interpolateFieldsFromPreviousMesh(const MPI_Comm &mpi_comm);

    /**
     * @brief Projects a vector of parallel distributed vectors
     * from previous to current mesh.
     *
     * triangulationSerPrev and triangulationParPrev are serial and parallel versions of
     * the previous mesh triangulation. Currently we need triangulationSerPrev for the algorithm to work,
     * but the algorithm could be potentially improved to not need triangulationSerPrev. When
     * that happens, we can directly pass previous and current dofHandlers as input arguments.
     *
     * @param triangulationSerPrev  serial triangulation of previous mesh
     * @param triangulationParPrev  parallel distributed triangulation of previous mesh
     * @param triangulationParCurrent parallel distributed triangulation of current mesh
     * @param FEPrev FiniteElement object of the previous mesh
     * @param FECurrent FiniteElement object of the current mesh. FECurrent and FEPrev must have
     * the same number of components.
     * @param fieldsPreviousMesh parallel distributed fields on previous mesh to be interpolated from
     * @param fieldsCurrentMesh  parallel distributed fields on current mesh to be interpolated upon
     * @param constraintsCurrentPtr  pointer to ConstraintMatrix object of current mesh. Default value
     * is NULL in which case the distribute operation doesn't happen inside interpolate. We have this
     * function so that outside interpolate function we can use the inhouse distribute function.
     */
      void interpolate(const dealii::parallel::distributed::Triangulation<3> & triangulationSerPrev,
		   const dealii::parallel::distributed::Triangulation<3> & triangulationParPrev,
		   const dealii::parallel::distributed::Triangulation<3> & triangulationParCurrent,
		   const dealii::FESystem<3> & FEPrev,
		   const dealii::FESystem<3> & FECurrent,
		   const std::vector<vectorType*> & fieldsPreviousMesh,
		   std::vector<vectorType*> & fieldsCurrentMesh,
		   const dealii::ConstraintMatrix * constraintsCurrentPtr=NULL);

     private:

      MPI_Comm mpi_communicator;
      const unsigned int n_mpi_processes;
      const unsigned int this_mpi_process;

      ///  parallel message stream
      dealii::ConditionalOStream   pcout;
    };

  }
}
#endif
