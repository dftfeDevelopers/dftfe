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


#ifndef constraintMatrixInfo_H_
#define constraintMatrixInfo_H_

#include <vector>

#include "headers.h"

namespace dftfe
{
  //
  // Declare dftUtils functions
  //
  namespace dftUtils
  {
    /**
     *  @brief Overloads dealii's distribute and distribute_local_to_global functions associated with constraints class.
     *  Stores the dealii's constraint matrix data into STL vectors for faster
     * memory access costs
     *
     *  @author Phani Motamarri
     *
     */
    class constraintMatrixInfo
    {
    public:
      /**
       * class constructor
       */
      constraintMatrixInfo();

      /**
       * class destructor
       */
      ~constraintMatrixInfo();

      /**
       * @brief convert a given constraintMatrix to simple arrays (STL) for fast access
       *
       * @param partitioner associated with the dealii vector
       * @param constraintMatrixData dealii constraint matrix from which the data is extracted
       */
      void
      initialize(
        const std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
          &                                      partitioner,
        const dealii::AffineConstraints<double> &constraintMatrixData);

      /**
       * @brief overloaded dealii internal function "distribute" which sets the slave node
       * field values from master nodes
       *
       * @param fieldVector parallel dealii vector
       */
      void
      distribute(distributedCPUVec<double> &fieldVector) const;

      /**
       * @brief overloaded dealii internal function distribute for flattened dealii array  which sets
       * the slave node field values from master nodes
       *
       * @param blockSize number of components for a given node
       */
      template <typename T>
      void
      distribute(distributedCPUVec<T> &fieldVector,
                 const unsigned int    blockSize) const;

      template <typename T>
      void
      distribute(distributedCPUMultiVec<T> &fieldVector,
                 const unsigned int         blockSize) const;

      /**
       * @brief transfers the contributions of slave nodes to master nodes using the constraint equation
       * slave nodes are the nodes which are to the right of the constraint
       * equation and master nodes are the nodes which are left of the
       * constraint equation.
       *
       * @param fieldVector parallel dealii vector which is the result of matrix-vector product(vmult) withot taking
       * care of constraints
       * @param blockSize number of components for a given node
       */
      template <typename T>
      void
      distribute_slave_to_master(distributedCPUVec<T> &fieldVector,
                                 const unsigned int    blockSize) const;

      template <typename T>
      void
      distribute_slave_to_master(distributedCPUMultiVec<T> &fieldVector,
                                 const unsigned int         blockSize) const;


      /**
       * @brief sets field values at constrained nodes to be zero
       *
       * @param fieldVector parallel dealii vector with fields stored in a flattened format
       * @param blockSize number of field components for a given node
       */
      template <typename T>
      void
      set_zero(distributedCPUVec<T> &fieldVector,
               const unsigned int    blockSize) const;
      template <typename T>
      void
      set_zero(distributedCPUMultiVec<T> &fieldVector,
               const unsigned int         blockSize) const;

      /**
       * clear data members
       */
      void
      clear();


    private:
      std::vector<dealii::types::global_dof_index> d_rowIdsGlobal;
      std::vector<dealii::types::global_dof_index> d_rowIdsLocal;
      std::vector<dealii::types::global_dof_index> d_columnIdsLocal;
      std::vector<dealii::types::global_dof_index> d_columnIdsGlobal;
      std::vector<double>                          d_columnValues;
      std::vector<double>                          d_inhomogenities;
      std::vector<dealii::types::global_dof_index> d_rowSizes;
      std::vector<dealii::types::global_dof_index>
        d_localIndexMapUnflattenedToFlattened;
    };

  } // namespace dftUtils

} // namespace dftfe
#endif
