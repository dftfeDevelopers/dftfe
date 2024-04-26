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
#include <MemoryStorage.h>
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
    template <dftfe::utils::MemorySpace memorySpace>
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
      distribute(
        dftfe::linearAlgebra::MultiVector<T, memorySpace> &fieldVector) const;

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
      distribute_slave_to_master(
        dftfe::linearAlgebra::MultiVector<T, memorySpace> &fieldVector) const;

      /**
       * @brief Scales the constraints with the inverse diagonal mass matrix so that the scaling of the vector can be done at the cell level
       *
       * @param invSqrtMassVec the inverse diagonal mass matrix
       */
      void
      initializeScaledConstraints(
        const distributedCPUVec<double> &invSqrtMassVec);

      void
      initializeScaledConstraints(
        const dftfe::utils::MemoryStorage<double, memorySpace> &invSqrtMassVec);


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
      set_zero(
        dftfe::linearAlgebra::MultiVector<T, memorySpace> &fieldVector) const;

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

#if defined(DFTFE_WITH_DEVICE)
    /**
     *  @brief Overloads dealii's distribute and distribute_local_to_global functions associated with constraints class.
     *  Stores the dealii's constraint matrix data into STL vectors for faster
     * memory access costs
     *
     *  @author Sambit Das, Phani Motamarri
     *
     */
    template <>
    class constraintMatrixInfo<dftfe::utils::MemorySpace::DEVICE>
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
        const dealii::AffineConstraints<double> &constraintMatrixData,
        const bool                               useInhomogeneties = true);

      void
      distribute(distributedCPUVec<double> &fieldVector) const;

      template <typename T>
      void
      distribute(distributedCPUVec<T> &fieldVector,
                 const unsigned int    blockSize) const;

      /**
       * @brief overloaded dealii internal function distribute for flattened dealii array  which sets
       * the slave node field values from master nodes
       *
       * @param blockSize number of components for a given node
       */
      template <typename NumberType>
      void
      distribute(
        dftfe::linearAlgebra::MultiVector<NumberType,
                                          dftfe::utils::MemorySpace::DEVICE>
          &fieldVector) const;


      /**
       * @brief Scales the constraints with the inverse diagonal mass matrix so that the scaling of the vector can be done at the cell level
       *
       * @param invSqrtMassVec the inverse diagonal mass matrix
       */
      void
      initializeScaledConstraints(
        const dftfe::utils::MemoryStorage<double,
                                          dftfe::utils::MemorySpace::DEVICE>
          &invSqrtMassVec);


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
        distributedDeviceVec<double> &fieldVector) const;

      void
      distribute_slave_to_master(
        distributedDeviceVec<std::complex<double>> &fieldVector) const;


      template <typename T>
      void
      distribute_slave_to_master(distributedCPUVec<T> &fieldVector,
                                 const unsigned int    blockSize) const;

      void
      initializeScaledConstraints(
        const distributedCPUVec<double> &invSqrtMassVec);

      /**
       * @brief sets field values at constrained nodes to be zero
       *
       * @param fieldVector parallel dealii vector with fields stored in a flattened format
       * @param blockSize number of field components for a given node
       */
      template <typename NumberType>
      void
      set_zero(distributedDeviceVec<NumberType> &fieldVector) const;

      template <typename T>
      void
      set_zero(distributedCPUVec<T> &fieldVector,
               const unsigned int    blockSize) const;

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

      dftfe::utils::MemoryStorage<unsigned int,
                                  dftfe::utils::MemorySpace::DEVICE>
        d_rowIdsLocalDevice;
      dftfe::utils::MemoryStorage<unsigned int,
                                  dftfe::utils::MemorySpace::DEVICE>
        d_columnIdsLocalDevice;
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
        d_columnValuesDevice;
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
        d_inhomogenitiesDevice;
      dftfe::utils::MemoryStorage<unsigned int,
                                  dftfe::utils::MemorySpace::DEVICE>
        d_rowSizesDevice;
      dftfe::utils::MemoryStorage<unsigned int,
                                  dftfe::utils::MemorySpace::DEVICE>
        d_rowSizesAccumulatedDevice;
      dftfe::utils::MemoryStorage<dealii::types::global_dof_index,
                                  dftfe::utils::MemorySpace::DEVICE>
        d_localIndexMapUnflattenedToFlattenedDevice;

      std::vector<unsigned int> d_rowIdsLocalBins;
      std::vector<unsigned int> d_columnIdsLocalBins;
      std::vector<unsigned int> d_columnIdToRowIdMapBins;
      std::vector<double>       d_columnValuesBins;
      std::vector<unsigned int> d_binColumnSizes;
      std::vector<unsigned int> d_binColumnSizesAccumulated;

      dftfe::utils::MemoryStorage<unsigned int,
                                  dftfe::utils::MemorySpace::DEVICE>
        d_rowIdsLocalBinsDevice;
      dftfe::utils::MemoryStorage<unsigned int,
                                  dftfe::utils::MemorySpace::DEVICE>
        d_columnIdsLocalBinsDevice;
      dftfe::utils::MemoryStorage<unsigned int,
                                  dftfe::utils::MemorySpace::DEVICE>
        d_columnIdToRowIdMapBinsDevice;
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
        d_columnValuesBinsDevice;

      unsigned int d_numConstrainedDofs;
    };
#endif

  } // namespace dftUtils

} // namespace dftfe
#endif
