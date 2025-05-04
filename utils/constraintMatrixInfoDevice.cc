// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
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
// @author  Sambit Das, Phani Motamarri
//

#include <constraintMatrixInfo.h>
#include <dftUtils.h>
#include "constraintMatrixInfoDeviceKernels.h"

namespace dftfe
{
  // Declare dftUtils functions
  namespace dftUtils
  {
    // constructor
    //
    constraintMatrixInfo<
      dftfe::utils::MemorySpace::DEVICE>::constraintMatrixInfo()
    {}

    //
    // destructor
    //
    constraintMatrixInfo<
      dftfe::utils::MemorySpace::DEVICE>::~constraintMatrixInfo()
    {}


    //
    // store constraintMatrix row data in STL vector
    //
    void
    constraintMatrixInfo<dftfe::utils::MemorySpace::DEVICE>::initialize(
      const std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
                                              &partitioner,
      const dealii::AffineConstraints<double> &constraintMatrixData,
      const bool                               useInhomogeneties)

    {
      clear();
      const dealii::IndexSet &locally_owned_dofs =
        partitioner->locally_owned_range();
      const dealii::IndexSet &ghost_dofs = partitioner->ghost_indices();

      dealii::types::global_dof_index    count = 0;
      std::vector<std::set<dftfe::uInt>> slaveToMasterSet;
      for (dealii::IndexSet::ElementIterator it = locally_owned_dofs.begin();
           it != locally_owned_dofs.end();
           ++it)
        {
          if (constraintMatrixData.is_constrained(*it))
            {
              const dealii::types::global_dof_index lineDof = *it;
              d_rowIdsLocal.push_back(partitioner->global_to_local(lineDof));
              if (useInhomogeneties)
                d_inhomogenities.push_back(
                  constraintMatrixData.get_inhomogeneity(lineDof));
              else
                d_inhomogenities.push_back(0.0);
              const std::vector<
                std::pair<dealii::types::global_dof_index, double>> *rowData =
                constraintMatrixData.get_constraint_entries(lineDof);
              d_rowSizes.push_back(rowData->size());
              d_rowSizesAccumulated.push_back(count);
              count += rowData->size();
              std::set<dftfe::uInt> columnIds;
              for (dftfe::uInt j = 0; j < rowData->size(); ++j)
                {
                  Assert((*rowData)[j].first < partitioner->size(),
                         dealii::ExcMessage("Index out of bounds"));
                  const dftfe::uInt columnId =
                    partitioner->global_to_local((*rowData)[j].first);
                  d_columnIdsLocal.push_back(columnId);
                  d_columnValues.push_back((*rowData)[j].second);
                  columnIds.insert(columnId);
                }
              slaveToMasterSet.push_back(columnIds);
            }
        }


      for (dealii::IndexSet::ElementIterator it = ghost_dofs.begin();
           it != ghost_dofs.end();
           ++it)
        {
          if (constraintMatrixData.is_constrained(*it))
            {
              const dealii::types::global_dof_index lineDof = *it;
              d_rowIdsLocal.push_back(partitioner->global_to_local(lineDof));
              if (useInhomogeneties)
                d_inhomogenities.push_back(
                  constraintMatrixData.get_inhomogeneity(lineDof));
              else
                d_inhomogenities.push_back(0.0);
              const std::vector<
                std::pair<dealii::types::global_dof_index, double>> *rowData =
                constraintMatrixData.get_constraint_entries(lineDof);
              d_rowSizes.push_back(rowData->size());
              d_rowSizesAccumulated.push_back(count);
              count += rowData->size();
              std::set<dftfe::uInt> columnIds;
              for (dftfe::uInt j = 0; j < rowData->size(); ++j)
                {
                  Assert((*rowData)[j].first < partitioner->size(),
                         dealii::ExcMessage("Index out of bounds"));
                  const dftfe::uInt columnId =
                    partitioner->global_to_local((*rowData)[j].first);
                  d_columnIdsLocal.push_back(columnId);
                  d_columnValues.push_back((*rowData)[j].second);
                  columnIds.insert(columnId);
                }
              slaveToMasterSet.push_back(columnIds);
            }
        }

      d_rowIdsLocalDevice.resize(d_rowIdsLocal.size());
      d_rowIdsLocalDevice.copyFrom(d_rowIdsLocal);

      d_columnIdsLocalDevice.resize(d_columnIdsLocal.size());
      d_columnIdsLocalDevice.copyFrom(d_columnIdsLocal);

      d_columnValuesDevice.resize(d_columnValues.size());
      d_columnValuesDevice.copyFrom(d_columnValues);

      d_inhomogenitiesDevice.resize(d_inhomogenities.size());
      d_inhomogenitiesDevice.copyFrom(d_inhomogenities);

      d_rowSizesDevice.resize(d_rowSizes.size());
      d_rowSizesDevice.copyFrom(d_rowSizes);

      d_rowSizesAccumulatedDevice.resize(d_rowSizesAccumulated.size());
      d_rowSizesAccumulatedDevice.copyFrom(d_rowSizesAccumulated);

      d_numConstrainedDofs = d_rowIdsLocal.size();
    }

    template <typename NumberType>
    void
    constraintMatrixInfo<dftfe::utils::MemorySpace::DEVICE>::distribute(
      distributedDeviceVec<NumberType> &fieldVector) const
    {
      if (d_numConstrainedDofs == 0)
        return;

      const dftfe::uInt blockSize = fieldVector.numVectors();
      distributeDevice(blockSize,
                       fieldVector.begin(),
                       d_rowIdsLocalDevice.begin(),
                       d_numConstrainedDofs,
                       d_rowSizesDevice.begin(),
                       d_rowSizesAccumulatedDevice.begin(),
                       d_columnIdsLocalDevice.begin(),
                       d_columnValuesDevice.begin(),
                       d_inhomogenitiesDevice.begin());
    }


    void
    constraintMatrixInfo<dftfe::utils::MemorySpace::DEVICE>::
      initializeScaledConstraints(
        const dftfe::utils::MemoryStorage<double,
                                          dftfe::utils::MemorySpace::DEVICE>
          &invSqrtMassVec)
    {
      if (d_numConstrainedDofs == 0)
        return;
      scaleConstraintsDevice(invSqrtMassVec.data(),
                             d_rowIdsLocalDevice.begin(),
                             d_numConstrainedDofs,
                             d_rowSizesDevice.begin(),
                             d_rowSizesAccumulatedDevice.begin(),
                             d_columnIdsLocalDevice.begin(),
                             d_columnValuesDevice.begin());
    }
    //
    // set the constrained degrees of freedom to values so that constraints
    // are satisfied for flattened array
    //
    template <typename NumberType>
    void
    constraintMatrixInfo<dftfe::utils::MemorySpace::DEVICE>::
      distribute_slave_to_master(
        distributedDeviceVec<NumberType> &fieldVector) const
    {
      if (d_numConstrainedDofs == 0)
        return;

      const dftfe::uInt blockSize = fieldVector.numVectors();
      distributeSlaveToMasterAtomicAddDevice(
        blockSize,
        fieldVector.begin(),
        d_rowIdsLocalDevice.begin(),
        d_numConstrainedDofs,
        d_rowSizesDevice.begin(),
        d_rowSizesAccumulatedDevice.begin(),
        d_columnIdsLocalDevice.begin(),
        d_columnValuesDevice.begin());
    }

    template <typename NumberType>
    void
    constraintMatrixInfo<dftfe::utils::MemorySpace::DEVICE>::set_zero(
      distributedDeviceVec<NumberType> &fieldVector) const
    {
      if (d_numConstrainedDofs == 0)
        return;

      const dftfe::uInt blockSize          = fieldVector.numVectors();
      const dftfe::uInt numConstrainedDofs = d_rowIdsLocal.size();
      setzeroDevice(blockSize,
                    fieldVector.begin(),
                    d_rowIdsLocalDevice.begin(),
                    numConstrainedDofs);
    }

    //
    //
    // clear the data variables
    //
    void
    constraintMatrixInfo<dftfe::utils::MemorySpace::DEVICE>::clear()
    {
      d_rowIdsLocal.clear();
      d_columnIdsLocal.clear();
      d_columnValues.clear();
      d_inhomogenities.clear();
      d_rowSizes.clear();
      d_rowSizesAccumulated.clear();
      d_rowIdsLocalBins.clear();
      d_columnIdsLocalBins.clear();
      d_columnValuesBins.clear();
      d_binColumnSizesAccumulated.clear();
      d_binColumnSizes.clear();

      d_rowIdsLocalDevice.clear();
      d_columnIdsLocalDevice.clear();
      d_columnValuesDevice.clear();
      d_inhomogenitiesDevice.clear();
      d_rowSizesDevice.clear();
      d_rowSizesAccumulatedDevice.clear();
      d_rowIdsLocalBinsDevice.clear();
      d_columnIdsLocalBinsDevice.clear();
      d_columnValuesBinsDevice.clear();
    }


    void
    constraintMatrixInfo<dftfe::utils::MemorySpace::DEVICE>::distribute(
      distributedCPUVec<double> &fieldVector) const
    {
      AssertThrow(false, dftUtils::ExcNotImplementedYet());
    }

    template <typename T>
    void
    constraintMatrixInfo<dftfe::utils::MemorySpace::DEVICE>::distribute(
      distributedCPUVec<T> &fieldVector,
      const dftfe::uInt     blockSize) const
    {
      AssertThrow(false, dftUtils::ExcNotImplementedYet());
    }

    template <typename T>
    void
    constraintMatrixInfo<dftfe::utils::MemorySpace::DEVICE>::
      distribute_slave_to_master(distributedCPUVec<T> &fieldVector,
                                 const dftfe::uInt     blockSize) const
    {
      AssertThrow(false, dftUtils::ExcNotImplementedYet());
    }

    void
    constraintMatrixInfo<dftfe::utils::MemorySpace::DEVICE>::
      initializeScaledConstraints(
        const distributedCPUVec<double> &invSqrtMassVec)
    {
      AssertThrow(false, dftUtils::ExcNotImplementedYet());
    }

    template <typename T>
    void
    constraintMatrixInfo<dftfe::utils::MemorySpace::DEVICE>::set_zero(
      distributedCPUVec<T> &fieldVector,
      const dftfe::uInt     blockSize) const
    {
      AssertThrow(false, dftUtils::ExcNotImplementedYet());
    }

    template void
    constraintMatrixInfo<dftfe::utils::MemorySpace::DEVICE>::distribute(
      distributedCPUVec<dataTypes::number> &fieldVector,
      const dftfe::uInt                     blockSize) const;

    template void
    constraintMatrixInfo<dftfe::utils::MemorySpace::DEVICE>::
      distribute_slave_to_master(
        distributedCPUVec<dataTypes::number> &fieldVector,
        const dftfe::uInt                     blockSize) const;

#if defined(USE_COMPLEX)
    template void
    constraintMatrixInfo<dftfe::utils::MemorySpace::DEVICE>::
      distribute_slave_to_master(distributedCPUVec<double> &fieldVector,
                                 const dftfe::uInt          blockSize) const;
#endif

    template void
    constraintMatrixInfo<dftfe::utils::MemorySpace::DEVICE>::set_zero(
      distributedCPUVec<dataTypes::number> &fieldVector,
      const dftfe::uInt                     blockSize) const;



    template void
    constraintMatrixInfo<dftfe::utils::MemorySpace::DEVICE>::distribute(
      distributedDeviceVec<double> &fieldVector) const;

    template void
    constraintMatrixInfo<dftfe::utils::MemorySpace::DEVICE>::distribute(
      distributedDeviceVec<std::complex<double>> &fieldVector) const;

    template void
    constraintMatrixInfo<dftfe::utils::MemorySpace::DEVICE>::distribute(
      distributedDeviceVec<float> &fieldVector) const;

    template void
    constraintMatrixInfo<dftfe::utils::MemorySpace::DEVICE>::distribute(
      distributedDeviceVec<std::complex<float>> &fieldVector) const;

    template void
    constraintMatrixInfo<dftfe::utils::MemorySpace::DEVICE>::set_zero(
      distributedDeviceVec<double> &fieldVector) const;

    template void
    constraintMatrixInfo<dftfe::utils::MemorySpace::DEVICE>::set_zero(
      distributedDeviceVec<std::complex<double>> &fieldVector) const;

    template void
    constraintMatrixInfo<dftfe::utils::MemorySpace::DEVICE>::set_zero(
      distributedDeviceVec<float> &fieldVector) const;

    template void
    constraintMatrixInfo<dftfe::utils::MemorySpace::DEVICE>::set_zero(
      distributedDeviceVec<std::complex<float>> &fieldVector) const;

    template void
    constraintMatrixInfo<dftfe::utils::MemorySpace::DEVICE>::
      distribute_slave_to_master(
        distributedDeviceVec<double> &fieldVector) const;

    template void
    constraintMatrixInfo<dftfe::utils::MemorySpace::DEVICE>::
      distribute_slave_to_master(
        distributedDeviceVec<std::complex<double>> &fieldVector) const;

    template void
    constraintMatrixInfo<dftfe::utils::MemorySpace::DEVICE>::
      distribute_slave_to_master(
        distributedDeviceVec<float> &fieldVector) const;

    template void
    constraintMatrixInfo<dftfe::utils::MemorySpace::DEVICE>::
      distribute_slave_to_master(
        distributedDeviceVec<std::complex<float>> &fieldVector) const;
    template class constraintMatrixInfo<dftfe::utils::MemorySpace::DEVICE>;

  } // namespace dftUtils
} // namespace dftfe
