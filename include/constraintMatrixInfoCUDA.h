// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2018 The Regents of the University of Michigan and DFT-FE
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

#if defined(DFTFE_WITH_GPU)
#  ifndef constraintMatrixInfoCUDA_H_
#    define constraintMatrixInfoCUDA_H_

#    include <thrust/device_vector.h>

#    include <vector>

#    include "headers.h"

namespace dftfe
{
  namespace dftUtils
  {
    /**
     *  @brief Overloads dealii's distribute and distribute_local_to_global functions associated with constraints class.
     *  Stores the dealii's constraint matrix data into STL vectors for faster
     * memory access costs
     *
     *  @author Sambit Das, Phani Motamarri
     *
     */
    class constraintMatrixInfoCUDA
    {
    public:
      /**
       * class constructor
       */
      constraintMatrixInfoCUDA();

      /**
       * class destructor
       */
      ~constraintMatrixInfoCUDA();

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
       * @brief precompute map between local processor index of unflattened deallii array to the local processor index of
       * the first field associated with the multi-field flattened dealii array
       *
       * @param partitioner1 associated with unflattened dealii vector
       * @param partitioner2 associated with flattened dealii vector storing multi-fields
       */
      void
      precomputeMaps(
        const std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
          &partitioner1,
        const std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
          &                partitioner2,
        const unsigned int blockSize);


      /**
       * @brief overloaded dealii internal function distribute for flattened dealii array  which sets
       * the slave node field values from master nodes
       *
       * @param blockSize number of components for a given node
       */
      template <typename NumberType>
      void
      distribute(distributedGPUVec<NumberType> &fieldVector,
                 const unsigned int             blockSize) const;

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
      void
      distribute_slave_to_master(distributedGPUVec<double> &fieldVector,
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
      void
      distribute_slave_to_master(
        distributedGPUVec<cuDoubleComplex> &fieldVector,
        double *                            tempReal,
        double *                            tempImag,
        const unsigned int                  blockSize) const;


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
      void
      distribute_slave_to_master(
        distributedGPUVec<cuFloatComplex> &fieldVector,
        float *                            tempReal,
        float *                            tempImag,
        const unsigned int                  blockSize) const;


      /**
       * @brief sets field values at constrained nodes to be zero
       *
       * @param fieldVector parallel dealii vector with fields stored in a flattened format
       * @param blockSize number of field components for a given node
       */
      template <typename NumberType>
      void
      set_zero(distributedGPUVec<NumberType> &fieldVector,
               const unsigned int         blockSize) const;

      /**
       * clear data members
       */
      void
      clear();


    private:
      std::vector<unsigned int> d_rowIdsLocal;
      std::vector<unsigned int> d_columnIdsLocal;
      std::vector<double>       d_columnValues;
      std::vector<double>       d_inhomogenities;
      std::vector<unsigned int> d_rowSizes;
      std::vector<unsigned int> d_rowSizesAccumulated;
      std::vector<dealii::types::global_dof_index>
        d_localIndexMapUnflattenedToFlattened;

      thrust::device_vector<unsigned int> d_rowIdsLocalDevice;
      thrust::device_vector<unsigned int> d_columnIdsLocalDevice;
      thrust::device_vector<double>       d_columnValuesDevice;
      thrust::device_vector<double>       d_inhomogenitiesDevice;
      thrust::device_vector<unsigned int> d_rowSizesDevice;
      thrust::device_vector<unsigned int> d_rowSizesAccumulatedDevice;
      thrust::device_vector<dealii::types::global_dof_index>
        d_localIndexMapUnflattenedToFlattenedDevice;

      std::vector<unsigned int> d_rowIdsLocalBins;
      std::vector<unsigned int> d_columnIdsLocalBins;
      std::vector<unsigned int> d_columnIdToRowIdMapBins;
      std::vector<double>       d_columnValuesBins;
      std::vector<unsigned int> d_binColumnSizes;
      std::vector<unsigned int> d_binColumnSizesAccumulated;

      thrust::device_vector<unsigned int> d_rowIdsLocalBinsDevice;
      thrust::device_vector<unsigned int> d_columnIdsLocalBinsDevice;
      thrust::device_vector<unsigned int> d_columnIdToRowIdMapBinsDevice;
      thrust::device_vector<double>       d_columnValuesBinsDevice;

      unsigned int d_numConstrainedDofs;
    };
  } // namespace dftUtils

} // namespace dftfe
#  endif
#endif
