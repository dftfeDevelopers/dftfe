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
// @author  Sambit Das, Phani Motamarri
//

#include <constraintMatrixInfoDevice.h>
#include <deviceKernelsGeneric.h>
#include <DeviceDataTypeOverloads.h>
#include <DeviceKernelLauncherConstants.h>

namespace dftfe
{
  // Declare dftUtils functions
  namespace dftUtils
  {
    namespace
    {
      __global__ void
      distributeKernel(
        const unsigned int  contiguousBlockSize,
        double *            xVec,
        const unsigned int *constraintLocalRowIdsUnflattened,
        const unsigned int  numConstraints,
        const unsigned int *constraintRowSizes,
        const unsigned int *constraintRowSizesAccumulated,
        const unsigned int *constraintLocalColumnIdsAllRowsUnflattened,
        const double *      constraintColumnValuesAllRowsUnflattened,
        const double *      inhomogenities)
      {
        const dealii::types::global_dof_index globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const dealii::types::global_dof_index numberEntries =
          numConstraints * contiguousBlockSize;

        for (dealii::types::global_dof_index index = globalThreadId;
             index < numberEntries;
             index += blockDim.x * gridDim.x)
          {
            const unsigned int blockIndex      = index / contiguousBlockSize;
            const unsigned int intraBlockIndex = index % contiguousBlockSize;
            const unsigned int constrainedRowId =
              constraintLocalRowIdsUnflattened[blockIndex];
            const unsigned int numberColumns = constraintRowSizes[blockIndex];
            const unsigned int startingColumnNumber =
              constraintRowSizesAccumulated[blockIndex];
            const dealii::types::global_dof_index xVecStartingIdRow =
              constrainedRowId * contiguousBlockSize;
            xVec[xVecStartingIdRow + intraBlockIndex] =
              inhomogenities[blockIndex];
            for (unsigned int i = 0; i < numberColumns; ++i)
              {
                const unsigned int constrainedColumnId =
                  constraintLocalColumnIdsAllRowsUnflattened
                    [startingColumnNumber + i];
                const dealii::types::global_dof_index xVecStartingIdColumn =
                  constrainedColumnId * contiguousBlockSize;
                xVec[xVecStartingIdRow + intraBlockIndex] +=
                  constraintColumnValuesAllRowsUnflattened
                    [startingColumnNumber + i] *
                  xVec[xVecStartingIdColumn + intraBlockIndex];
              }
          }
      }


      __global__ void
      distributeKernel(
        const unsigned int  contiguousBlockSize,
        float *             xVec,
        const unsigned int *constraintLocalRowIdsUnflattened,
        const unsigned int  numConstraints,
        const unsigned int *constraintRowSizes,
        const unsigned int *constraintRowSizesAccumulated,
        const unsigned int *constraintLocalColumnIdsAllRowsUnflattened,
        const double *      constraintColumnValuesAllRowsUnflattened,
        const double *      inhomogenities)
      {
        const dealii::types::global_dof_index globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const dealii::types::global_dof_index numberEntries =
          numConstraints * contiguousBlockSize;

        for (dealii::types::global_dof_index index = globalThreadId;
             index < numberEntries;
             index += blockDim.x * gridDim.x)
          {
            const unsigned int blockIndex      = index / contiguousBlockSize;
            const unsigned int intraBlockIndex = index % contiguousBlockSize;
            const unsigned int constrainedRowId =
              constraintLocalRowIdsUnflattened[blockIndex];
            const unsigned int numberColumns = constraintRowSizes[blockIndex];
            const unsigned int startingColumnNumber =
              constraintRowSizesAccumulated[blockIndex];
            const dealii::types::global_dof_index xVecStartingIdRow =
              constrainedRowId * contiguousBlockSize;
            xVec[xVecStartingIdRow + intraBlockIndex] =
              inhomogenities[blockIndex];
            for (unsigned int i = 0; i < numberColumns; ++i)
              {
                const unsigned int constrainedColumnId =
                  constraintLocalColumnIdsAllRowsUnflattened
                    [startingColumnNumber + i];
                const dealii::types::global_dof_index xVecStartingIdColumn =
                  constrainedColumnId * contiguousBlockSize;
                xVec[xVecStartingIdRow + intraBlockIndex] +=
                  constraintColumnValuesAllRowsUnflattened
                    [startingColumnNumber + i] *
                  xVec[xVecStartingIdColumn + intraBlockIndex];
              }
          }
      }


      __global__ void
      distributeKernel(
        const unsigned int                 contiguousBlockSize,
        dftfe::utils::deviceDoubleComplex *xVec,
        const unsigned int *               constraintLocalRowIdsUnflattened,
        const unsigned int                 numConstraints,
        const unsigned int *               constraintRowSizes,
        const unsigned int *               constraintRowSizesAccumulated,
        const unsigned int *constraintLocalColumnIdsAllRowsUnflattened,
        const double *      constraintColumnValuesAllRowsUnflattened,
        const double *      inhomogenities)
      {
        const dealii::types::global_dof_index globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const dealii::types::global_dof_index numberEntries =
          numConstraints * contiguousBlockSize;

        for (dealii::types::global_dof_index index = globalThreadId;
             index < numberEntries;
             index += blockDim.x * gridDim.x)
          {
            const unsigned int blockIndex      = index / contiguousBlockSize;
            const unsigned int intraBlockIndex = index % contiguousBlockSize;
            const unsigned int constrainedRowId =
              constraintLocalRowIdsUnflattened[blockIndex];
            const unsigned int numberColumns = constraintRowSizes[blockIndex];
            const unsigned int startingColumnNumber =
              constraintRowSizesAccumulated[blockIndex];
            const dealii::types::global_dof_index xVecStartingIdRow =
              constrainedRowId * contiguousBlockSize;
            dftfe::utils::copyValue(xVec + xVecStartingIdRow + intraBlockIndex,
                                    inhomogenities[blockIndex]);
            for (unsigned int i = 0; i < numberColumns; ++i)
              {
                const unsigned int constrainedColumnId =
                  constraintLocalColumnIdsAllRowsUnflattened
                    [startingColumnNumber + i];
                const dealii::types::global_dof_index xVecStartingIdColumn =
                  constrainedColumnId * contiguousBlockSize;
                dftfe::utils::copyValue(
                  xVec + xVecStartingIdRow + intraBlockIndex,
                  dftfe::utils::add(
                    xVec[xVecStartingIdRow + intraBlockIndex],
                    dftfe::utils::makeComplex(
                      xVec[xVecStartingIdColumn + intraBlockIndex].x *
                        constraintColumnValuesAllRowsUnflattened
                          [startingColumnNumber + i],
                      xVec[xVecStartingIdColumn + intraBlockIndex].y *
                        constraintColumnValuesAllRowsUnflattened
                          [startingColumnNumber + i])));
              }
          }
      }


      __global__ void
      distributeKernel(
        const unsigned int                contiguousBlockSize,
        dftfe::utils::deviceFloatComplex *xVec,
        const unsigned int *              constraintLocalRowIdsUnflattened,
        const unsigned int                numConstraints,
        const unsigned int *              constraintRowSizes,
        const unsigned int *              constraintRowSizesAccumulated,
        const unsigned int *constraintLocalColumnIdsAllRowsUnflattened,
        const double *      constraintColumnValuesAllRowsUnflattened,
        const double *      inhomogenities)
      {
        const dealii::types::global_dof_index globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const dealii::types::global_dof_index numberEntries =
          numConstraints * contiguousBlockSize;

        for (dealii::types::global_dof_index index = globalThreadId;
             index < numberEntries;
             index += blockDim.x * gridDim.x)
          {
            const unsigned int blockIndex      = index / contiguousBlockSize;
            const unsigned int intraBlockIndex = index % contiguousBlockSize;
            const unsigned int constrainedRowId =
              constraintLocalRowIdsUnflattened[blockIndex];
            const unsigned int numberColumns = constraintRowSizes[blockIndex];
            const unsigned int startingColumnNumber =
              constraintRowSizesAccumulated[blockIndex];
            const dealii::types::global_dof_index xVecStartingIdRow =
              constrainedRowId * contiguousBlockSize;
            dftfe::utils::copyValue(xVec + xVecStartingIdRow + intraBlockIndex,
                                    inhomogenities[blockIndex]);
            for (unsigned int i = 0; i < numberColumns; ++i)
              {
                const unsigned int constrainedColumnId =
                  constraintLocalColumnIdsAllRowsUnflattened
                    [startingColumnNumber + i];
                const dealii::types::global_dof_index xVecStartingIdColumn =
                  constrainedColumnId * contiguousBlockSize;
                dftfe::utils::copyValue(
                  xVec + xVecStartingIdRow + intraBlockIndex,
                  dftfe::utils::add(
                    xVec[xVecStartingIdRow + intraBlockIndex],
                    dftfe::utils::makeComplex(
                      xVec[xVecStartingIdColumn + intraBlockIndex].x *
                        constraintColumnValuesAllRowsUnflattened
                          [startingColumnNumber + i],
                      xVec[xVecStartingIdColumn + intraBlockIndex].y *
                        constraintColumnValuesAllRowsUnflattened
                          [startingColumnNumber + i])));
              }
          }
      }

      __global__ void
      distributeSlaveToMasterKernelAtomicAdd(
        const unsigned int  contiguousBlockSize,
        double *            xVec,
        const unsigned int *constraintLocalRowIdsUnflattened,
        const unsigned int  numConstraints,
        const unsigned int *constraintRowSizes,
        const unsigned int *constraintRowSizesAccumulated,
        const unsigned int *constraintLocalColumnIdsAllRowsUnflattened,
        const double *      constraintColumnValuesAllRowsUnflattened)
      {
        const dealii::types::global_dof_index globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const dealii::types::global_dof_index numberEntries =
          numConstraints * contiguousBlockSize;

        for (dealii::types::global_dof_index index = globalThreadId;
             index < numberEntries;
             index += blockDim.x * gridDim.x)
          {
            const unsigned int blockIndex      = index / contiguousBlockSize;
            const unsigned int intraBlockIndex = index % contiguousBlockSize;
            const unsigned int constrainedRowId =
              constraintLocalRowIdsUnflattened[blockIndex];
            const unsigned int numberColumns = constraintRowSizes[blockIndex];
            const unsigned int startingColumnNumber =
              constraintRowSizesAccumulated[blockIndex];
            const dealii::types::global_dof_index xVecStartingIdRow =
              constrainedRowId * contiguousBlockSize;
            for (unsigned int i = 0; i < numberColumns; ++i)
              {
                const unsigned int constrainedColumnId =
                  constraintLocalColumnIdsAllRowsUnflattened
                    [startingColumnNumber + i];
                const dealii::types::global_dof_index xVecStartingIdColumn =
                  constrainedColumnId * contiguousBlockSize;
                atomicAdd(&(xVec[xVecStartingIdColumn + intraBlockIndex]),
                          constraintColumnValuesAllRowsUnflattened
                              [startingColumnNumber + i] *
                            xVec[xVecStartingIdRow + intraBlockIndex]);
              }
            xVec[xVecStartingIdRow + intraBlockIndex] = 0.0;
          }
      }


      __global__ void
      distributeSlaveToMasterKernelAtomicAdd(
        const unsigned int  contiguousBlockSize,
        float *             xVec,
        const unsigned int *constraintLocalRowIdsUnflattened,
        const unsigned int  numConstraints,
        const unsigned int *constraintRowSizes,
        const unsigned int *constraintRowSizesAccumulated,
        const unsigned int *constraintLocalColumnIdsAllRowsUnflattened,
        const double *      constraintColumnValuesAllRowsUnflattened)
      {
        const dealii::types::global_dof_index globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const dealii::types::global_dof_index numberEntries =
          numConstraints * contiguousBlockSize;

        for (dealii::types::global_dof_index index = globalThreadId;
             index < numberEntries;
             index += blockDim.x * gridDim.x)
          {
            const unsigned int blockIndex      = index / contiguousBlockSize;
            const unsigned int intraBlockIndex = index % contiguousBlockSize;
            const unsigned int constrainedRowId =
              constraintLocalRowIdsUnflattened[blockIndex];
            const unsigned int numberColumns = constraintRowSizes[blockIndex];
            const unsigned int startingColumnNumber =
              constraintRowSizesAccumulated[blockIndex];
            const dealii::types::global_dof_index xVecStartingIdRow =
              constrainedRowId * contiguousBlockSize;
            for (unsigned int i = 0; i < numberColumns; ++i)
              {
                const unsigned int constrainedColumnId =
                  constraintLocalColumnIdsAllRowsUnflattened
                    [startingColumnNumber + i];
                const dealii::types::global_dof_index xVecStartingIdColumn =
                  constrainedColumnId * contiguousBlockSize;
                const float tempfloatval =
                  constraintColumnValuesAllRowsUnflattened
                    [startingColumnNumber + i] *
                  xVec[xVecStartingIdRow + intraBlockIndex];
                atomicAdd(&(xVec[xVecStartingIdColumn + intraBlockIndex]),
                          tempfloatval);
              }
            xVec[xVecStartingIdRow + intraBlockIndex] = 0.0;
          }
      }


      __global__ void
      setzeroKernel(const unsigned int  contiguousBlockSize,
                    double *            xVec,
                    const unsigned int *constraintLocalRowIdsUnflattened,
                    const unsigned int  numConstraints)
      {
        const dealii::types::global_dof_index globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const dealii::types::global_dof_index numberEntries =
          numConstraints * contiguousBlockSize;

        for (dealii::types::global_dof_index index = globalThreadId;
             index < numberEntries;
             index += blockDim.x * gridDim.x)
          {
            const unsigned int blockIndex      = index / contiguousBlockSize;
            const unsigned int intraBlockIndex = index % contiguousBlockSize;
            xVec[constraintLocalRowIdsUnflattened[blockIndex] *
                   contiguousBlockSize +
                 intraBlockIndex]              = 0;
          }
      }

      __global__ void
      setzeroKernel(const unsigned int  contiguousBlockSize,
                    float *             xVec,
                    const unsigned int *constraintLocalRowIdsUnflattened,
                    const unsigned int  numConstraints)
      {
        const dealii::types::global_dof_index globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const dealii::types::global_dof_index numberEntries =
          numConstraints * contiguousBlockSize;

        for (dealii::types::global_dof_index index = globalThreadId;
             index < numberEntries;
             index += blockDim.x * gridDim.x)
          {
            const unsigned int blockIndex      = index / contiguousBlockSize;
            const unsigned int intraBlockIndex = index % contiguousBlockSize;
            xVec[constraintLocalRowIdsUnflattened[blockIndex] *
                   contiguousBlockSize +
                 intraBlockIndex]              = 0;
          }
      }

      __global__ void
      setzeroKernel(const unsigned int                 contiguousBlockSize,
                    dftfe::utils::deviceDoubleComplex *xVec,
                    const unsigned int *constraintLocalRowIdsUnflattened,
                    const unsigned int  numConstraints)
      {
        const dealii::types::global_dof_index globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const dealii::types::global_dof_index numberEntries =
          numConstraints * contiguousBlockSize;

        for (dealii::types::global_dof_index index = globalThreadId;
             index < numberEntries;
             index += blockDim.x * gridDim.x)
          {
            const unsigned int blockIndex      = index / contiguousBlockSize;
            const unsigned int intraBlockIndex = index % contiguousBlockSize;
            dftfe::utils::copyValue(
              xVec +
                constraintLocalRowIdsUnflattened[blockIndex] *
                  contiguousBlockSize +
                intraBlockIndex,
              0.0);
          }
      }


      __global__ void
      setzeroKernel(const unsigned int                contiguousBlockSize,
                    dftfe::utils::deviceFloatComplex *xVec,
                    const unsigned int *constraintLocalRowIdsUnflattened,
                    const unsigned int  numConstraints)
      {
        const dealii::types::global_dof_index globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const dealii::types::global_dof_index numberEntries =
          numConstraints * contiguousBlockSize;

        for (dealii::types::global_dof_index index = globalThreadId;
             index < numberEntries;
             index += blockDim.x * gridDim.x)
          {
            const unsigned int blockIndex      = index / contiguousBlockSize;
            const unsigned int intraBlockIndex = index % contiguousBlockSize;
            dftfe::utils::copyValue(
              xVec +
                constraintLocalRowIdsUnflattened[blockIndex] *
                  contiguousBlockSize +
                intraBlockIndex,
              0.0);
          }
      }
    } // namespace

    // constructor
    //
    constraintMatrixInfoDevice::constraintMatrixInfoDevice()
    {}

    //
    // destructor
    //
    constraintMatrixInfoDevice::~constraintMatrixInfoDevice()
    {}


    //
    // store constraintMatrix row data in STL vector
    //
    void
    constraintMatrixInfoDevice::initialize(
      const std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
        &                                      partitioner,
      const dealii::AffineConstraints<double> &constraintMatrixData)

    {
      clear();
      const dealii::IndexSet &locally_owned_dofs =
        partitioner->locally_owned_range();
      const dealii::IndexSet &ghost_dofs = partitioner->ghost_indices();

      dealii::types::global_dof_index     count = 0;
      std::vector<std::set<unsigned int>> slaveToMasterSet;
      for (dealii::IndexSet::ElementIterator it = locally_owned_dofs.begin();
           it != locally_owned_dofs.end();
           ++it)
        {
          if (constraintMatrixData.is_constrained(*it))
            {
              const dealii::types::global_dof_index lineDof = *it;
              d_rowIdsLocal.push_back(partitioner->global_to_local(lineDof));
              d_inhomogenities.push_back(
                constraintMatrixData.get_inhomogeneity(lineDof));
              const std::vector<
                std::pair<dealii::types::global_dof_index, double>> *rowData =
                constraintMatrixData.get_constraint_entries(lineDof);
              d_rowSizes.push_back(rowData->size());
              d_rowSizesAccumulated.push_back(count);
              count += rowData->size();
              std::set<unsigned int> columnIds;
              for (unsigned int j = 0; j < rowData->size(); ++j)
                {
                  Assert((*rowData)[j].first < partitioner->size(),
                         dealii::ExcMessage("Index out of bounds"));
                  const unsigned int columnId =
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
              d_inhomogenities.push_back(
                constraintMatrixData.get_inhomogeneity(lineDof));
              const std::vector<
                std::pair<dealii::types::global_dof_index, double>> *rowData =
                constraintMatrixData.get_constraint_entries(lineDof);
              d_rowSizes.push_back(rowData->size());
              d_rowSizesAccumulated.push_back(count);
              count += rowData->size();
              std::set<unsigned int> columnIds;
              for (unsigned int j = 0; j < rowData->size(); ++j)
                {
                  Assert((*rowData)[j].first < partitioner->size(),
                         dealii::ExcMessage("Index out of bounds"));
                  const unsigned int columnId =
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
    constraintMatrixInfoDevice::distribute(
      distributedDeviceVec<NumberType> &fieldVector,
      const unsigned int                blockSize) const
    {
      if (d_numConstrainedDofs == 0)
        return;
        // fieldVector.update_ghost_values();

#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
      distributeKernel<<<
        min((blockSize + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
              dftfe::utils::DEVICE_BLOCK_SIZE * d_numConstrainedDofs,
            30000),
        dftfe::utils::DEVICE_BLOCK_SIZE>>>(
        blockSize,
        dftfe::utils::makeDataTypeDeviceCompatible(fieldVector.begin()),
        d_rowIdsLocalDevice.begin(),
        d_numConstrainedDofs,
        d_rowSizesDevice.begin(),
        d_rowSizesAccumulatedDevice.begin(),
        d_columnIdsLocalDevice.begin(),
        d_columnValuesDevice.begin(),
        d_inhomogenitiesDevice.begin());
#elif DFTFE_WITH_DEVICE_LANG_HIP
      hipLaunchKernelGGL(
        distributeKernel,
        min((blockSize + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
              dftfe::utils::DEVICE_BLOCK_SIZE * d_numConstrainedDofs,
            30000),
        dftfe::utils::DEVICE_BLOCK_SIZE,
        0,
        0,
        blockSize,
        dftfe::utils::makeDataTypeDeviceCompatible(fieldVector.begin()),
        d_rowIdsLocalDevice.begin(),
        d_numConstrainedDofs,
        d_rowSizesDevice.begin(),
        d_rowSizesAccumulatedDevice.begin(),
        d_columnIdsLocalDevice.begin(),
        d_columnValuesDevice.begin(),
        d_inhomogenitiesDevice.begin());
#endif
    }



    //
    // set the constrained degrees of freedom to values so that constraints
    // are satisfied for flattened array
    //
    void
    constraintMatrixInfoDevice::distribute_slave_to_master(
      distributedDeviceVec<double> &fieldVector,
      const unsigned int            blockSize) const
    {
      if (d_numConstrainedDofs == 0)
        return;

#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
      distributeSlaveToMasterKernelAtomicAdd<<<
        min((blockSize + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
              dftfe::utils::DEVICE_BLOCK_SIZE * d_numConstrainedDofs,
            30000),
        dftfe::utils::DEVICE_BLOCK_SIZE>>>(
        blockSize,
        dftfe::utils::makeDataTypeDeviceCompatible(fieldVector.begin()),
        d_rowIdsLocalDevice.begin(),
        d_numConstrainedDofs,
        d_rowSizesDevice.begin(),
        d_rowSizesAccumulatedDevice.begin(),
        d_columnIdsLocalDevice.begin(),
        d_columnValuesDevice.begin());
#elif DFTFE_WITH_DEVICE_LANG_HIP
      hipLaunchKernelGGL(
        distributeSlaveToMasterKernelAtomicAdd,
        min((blockSize + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
              dftfe::utils::DEVICE_BLOCK_SIZE * d_numConstrainedDofs,
            30000),
        dftfe::utils::DEVICE_BLOCK_SIZE,
        0,
        0,
        blockSize,
        dftfe::utils::makeDataTypeDeviceCompatible(fieldVector.begin()),
        d_rowIdsLocalDevice.begin(),
        d_numConstrainedDofs,
        d_rowSizesDevice.begin(),
        d_rowSizesAccumulatedDevice.begin(),
        d_columnIdsLocalDevice.begin(),
        d_columnValuesDevice.begin());
#endif
    }

    //
    // set the constrained degrees of freedom to values so that constraints
    // are satisfied for flattened array
    //
    void
    constraintMatrixInfoDevice::distribute_slave_to_master(
      distributedDeviceVec<std::complex<double>> &fieldVector,
      double *                                    tempReal,
      double *                                    tempImag,
      const unsigned int                          blockSize) const
    {
      if (d_numConstrainedDofs == 0)
        return;

      dftfe::utils::deviceKernelsGeneric::copyComplexArrToRealArrsDevice(
        (fieldVector.localSize() * fieldVector.numVectors()),
        fieldVector.begin(),
        tempReal,
        tempImag);

#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
      distributeSlaveToMasterKernelAtomicAdd<<<
        min((blockSize + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
              dftfe::utils::DEVICE_BLOCK_SIZE * d_numConstrainedDofs,
            30000),
        dftfe::utils::DEVICE_BLOCK_SIZE>>>(blockSize,
                                           tempReal,
                                           d_rowIdsLocalDevice.begin(),
                                           d_numConstrainedDofs,
                                           d_rowSizesDevice.begin(),
                                           d_rowSizesAccumulatedDevice.begin(),
                                           d_columnIdsLocalDevice.begin(),
                                           d_columnValuesDevice.begin());

      distributeSlaveToMasterKernelAtomicAdd<<<
        min((blockSize + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
              dftfe::utils::DEVICE_BLOCK_SIZE * d_numConstrainedDofs,
            30000),
        dftfe::utils::DEVICE_BLOCK_SIZE>>>(blockSize,
                                           tempImag,
                                           d_rowIdsLocalDevice.begin(),
                                           d_numConstrainedDofs,
                                           d_rowSizesDevice.begin(),
                                           d_rowSizesAccumulatedDevice.begin(),
                                           d_columnIdsLocalDevice.begin(),
                                           d_columnValuesDevice.begin());
#elif DFTFE_WITH_DEVICE_LANG_HIP
      hipLaunchKernelGGL(
        distributeSlaveToMasterKernelAtomicAdd,
        min((blockSize + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
              dftfe::utils::DEVICE_BLOCK_SIZE * d_numConstrainedDofs,
            30000),
        dftfe::utils::DEVICE_BLOCK_SIZE,
        0,
        0,
        blockSize,
        tempReal,
        d_rowIdsLocalDevice.begin(),
        d_numConstrainedDofs,
        d_rowSizesDevice.begin(),
        d_rowSizesAccumulatedDevice.begin(),
        d_columnIdsLocalDevice.begin(),
        d_columnValuesDevice.begin());

      hipLaunchKernelGGL(
        distributeSlaveToMasterKernelAtomicAdd,
        min((blockSize + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
              dftfe::utils::DEVICE_BLOCK_SIZE * d_numConstrainedDofs,
            30000),
        dftfe::utils::DEVICE_BLOCK_SIZE,
        0,
        0,
        blockSize,
        tempImag,
        d_rowIdsLocalDevice.begin(),
        d_numConstrainedDofs,
        d_rowSizesDevice.begin(),
        d_rowSizesAccumulatedDevice.begin(),
        d_columnIdsLocalDevice.begin(),
        d_columnValuesDevice.begin());
#endif

      dftfe::utils::deviceKernelsGeneric::copyRealArrsToComplexArrDevice(
        (fieldVector.localSize() * fieldVector.numVectors()),
        tempReal,
        tempImag,
        fieldVector.begin());
    }

    //
    // set the constrained degrees of freedom to values so that constraints
    // are satisfied for flattened array
    //
    void
    constraintMatrixInfoDevice::distribute_slave_to_master(
      distributedDeviceVec<std::complex<float>> &fieldVector,
      float *                                    tempReal,
      float *                                    tempImag,
      const unsigned int                         blockSize) const
    {
      if (d_numConstrainedDofs == 0)
        return;

      dftfe::utils::deviceKernelsGeneric::copyComplexArrToRealArrsDevice(
        (fieldVector.localSize() * fieldVector.numVectors()),
        fieldVector.begin(),
        tempReal,
        tempImag);

#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
      distributeSlaveToMasterKernelAtomicAdd<<<
        min((blockSize + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
              dftfe::utils::DEVICE_BLOCK_SIZE * d_numConstrainedDofs,
            30000),
        dftfe::utils::DEVICE_BLOCK_SIZE>>>(blockSize,
                                           tempReal,
                                           d_rowIdsLocalDevice.begin(),
                                           d_numConstrainedDofs,
                                           d_rowSizesDevice.begin(),
                                           d_rowSizesAccumulatedDevice.begin(),
                                           d_columnIdsLocalDevice.begin(),
                                           d_columnValuesDevice.begin());

      distributeSlaveToMasterKernelAtomicAdd<<<
        min((blockSize + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
              dftfe::utils::DEVICE_BLOCK_SIZE * d_numConstrainedDofs,
            30000),
        dftfe::utils::DEVICE_BLOCK_SIZE>>>(blockSize,
                                           tempImag,
                                           d_rowIdsLocalDevice.begin(),
                                           d_numConstrainedDofs,
                                           d_rowSizesDevice.begin(),
                                           d_rowSizesAccumulatedDevice.begin(),
                                           d_columnIdsLocalDevice.begin(),
                                           d_columnValuesDevice.begin());
#elif DFTFE_WITH_DEVICE_LANG_HIP
      hipLaunchKernelGGL(
        distributeSlaveToMasterKernelAtomicAdd,
        min((blockSize + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
              dftfe::utils::DEVICE_BLOCK_SIZE * d_numConstrainedDofs,
            30000),
        dftfe::utils::DEVICE_BLOCK_SIZE,
        0,
        0,
        blockSize,
        tempReal,
        d_rowIdsLocalDevice.begin(),
        d_numConstrainedDofs,
        d_rowSizesDevice.begin(),
        d_rowSizesAccumulatedDevice.begin(),
        d_columnIdsLocalDevice.begin(),
        d_columnValuesDevice.begin());

      hipLaunchKernelGGL(
        distributeSlaveToMasterKernelAtomicAdd,
        min((blockSize + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
              dftfe::utils::DEVICE_BLOCK_SIZE * d_numConstrainedDofs,
            30000),
        dftfe::utils::DEVICE_BLOCK_SIZE,
        0,
        0,
        blockSize,
        tempImag,
        d_rowIdsLocalDevice.begin(),
        d_numConstrainedDofs,
        d_rowSizesDevice.begin(),
        d_rowSizesAccumulatedDevice.begin(),
        d_columnIdsLocalDevice.begin(),
        d_columnValuesDevice.begin());
#endif

      dftfe::utils::deviceKernelsGeneric::copyRealArrsToComplexArrDevice(
        (fieldVector.localSize() * fieldVector.numVectors()),
        tempReal,
        tempImag,
        fieldVector.begin());
    }


    template <typename NumberType>
    void
    constraintMatrixInfoDevice::set_zero(
      distributedDeviceVec<NumberType> &fieldVector,
      const unsigned int                blockSize) const
    {
      if (d_numConstrainedDofs == 0)
        return;

      const unsigned int numConstrainedDofs = d_rowIdsLocal.size();
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
      setzeroKernel<<<min((blockSize + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                            dftfe::utils::DEVICE_BLOCK_SIZE *
                            numConstrainedDofs,
                          30000),
                      dftfe::utils::DEVICE_BLOCK_SIZE>>>(
        blockSize,
        dftfe::utils::makeDataTypeDeviceCompatible(fieldVector.begin()),
        d_rowIdsLocalDevice.begin(),
        numConstrainedDofs);
#elif DFTFE_WITH_DEVICE_LANG_HIP
      hipLaunchKernelGGL(
        setzeroKernel,
        min((blockSize + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
              dftfe::utils::DEVICE_BLOCK_SIZE * numConstrainedDofs,
            30000),
        dftfe::utils::DEVICE_BLOCK_SIZE,
        0,
        0,
        blockSize,
        dftfe::utils::makeDataTypeDeviceCompatible(fieldVector.begin()),
        d_rowIdsLocalDevice.begin(),
        numConstrainedDofs);
#endif
    }

    //
    //
    // clear the data variables
    //
    void
    constraintMatrixInfoDevice::clear()
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

    template void
    constraintMatrixInfoDevice::distribute(
      distributedDeviceVec<double> &fieldVector,
      const unsigned int            blockSize) const;

    template void
    constraintMatrixInfoDevice::distribute(
      distributedDeviceVec<std::complex<double>> &fieldVector,
      const unsigned int                          blockSize) const;

    template void
    constraintMatrixInfoDevice::distribute(
      distributedDeviceVec<float> &fieldVector,
      const unsigned int           blockSize) const;

    template void
    constraintMatrixInfoDevice::distribute(
      distributedDeviceVec<std::complex<float>> &fieldVector,
      const unsigned int                         blockSize) const;

    template void
    constraintMatrixInfoDevice::set_zero(
      distributedDeviceVec<double> &fieldVector,
      const unsigned int            blockSize) const;

    template void
    constraintMatrixInfoDevice::set_zero(
      distributedDeviceVec<std::complex<double>> &fieldVector,
      const unsigned int                          blockSize) const;

    template void
    constraintMatrixInfoDevice::set_zero(
      distributedDeviceVec<float> &fieldVector,
      const unsigned int           blockSize) const;

    template void
    constraintMatrixInfoDevice::set_zero(
      distributedDeviceVec<std::complex<float>> &fieldVector,
      const unsigned int                         blockSize) const;


  } // namespace dftUtils
} // namespace dftfe
