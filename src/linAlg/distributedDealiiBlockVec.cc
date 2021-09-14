// ---------------------------------------------------------------------
//
// Copyright (c) 2019-2020 The Regents of the University of Michigan and DFT-FE
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

#include "distributedDealiiBlockVec.h"
#include "headers.h"
#include "dftUtils.h"


#if defined(DFTFE_WITH_GPU)
#  include <thrust/device_vector.h>
#  include <thrust/complex.h>
#  include <cuComplex.h>
#endif


namespace dftfe
{
  namespace distributedBlockvecInternal
  {
    template <typename T>
    void
    createDealiiVector(
      const std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
        &                   partitioner,
      const unsigned int    blockSize,
      distributedCPUVec<T> &flattenedArray)
    {
      const MPI_Comm &mpi_communicator = partitioner->get_mpi_communicator();
      //
      // Get required sizes
      //
      const unsigned int n_ghosts  = partitioner->n_ghost_indices();
      const unsigned int localSize = partitioner->local_size();
      const unsigned int totalSize = localSize + n_ghosts;
      const dealii::types::global_dof_index globalNumberDegreesOfFreedom =
        partitioner->size();

      //
      // create data for new parallel layout
      //
      dealii::IndexSet locallyOwnedFlattenedNodesSet, ghostFlattenedNodesSet;
      locallyOwnedFlattenedNodesSet.clear();
      ghostFlattenedNodesSet.clear();

      //
      // Set the maximal size of the indices upon which this object operates.
      //
      locallyOwnedFlattenedNodesSet.set_size(
        globalNumberDegreesOfFreedom *
        (dealii::types::global_dof_index)blockSize);
      ghostFlattenedNodesSet.set_size(
        globalNumberDegreesOfFreedom *
        (dealii::types::global_dof_index)blockSize);


      for (unsigned int ilocaldof = 0; ilocaldof < totalSize; ++ilocaldof)
        {
          std::vector<dealii::types::global_dof_index>
                                                       newLocallyOwnedGlobalNodeIds;
          std::vector<dealii::types::global_dof_index> newGhostGlobalNodeIds;
          const dealii::types::global_dof_index        globalIndex =
            partitioner->local_to_global(ilocaldof);
          const bool isGhost = partitioner->is_ghost_entry(globalIndex);
          if (isGhost)
            {
              for (unsigned int iwave = 0; iwave < blockSize; ++iwave)
                {
                  newGhostGlobalNodeIds.push_back(
                    (dealii::types::global_dof_index)blockSize * globalIndex +
                    (dealii::types::global_dof_index)iwave);
                }
            }
          else
            {
              for (unsigned int iwave = 0; iwave < blockSize; ++iwave)
                {
                  newLocallyOwnedGlobalNodeIds.push_back(
                    (dealii::types::global_dof_index)blockSize * globalIndex +
                    (dealii::types::global_dof_index)iwave);
                }
            }

          // insert into dealii index sets
          locallyOwnedFlattenedNodesSet.add_indices(
            newLocallyOwnedGlobalNodeIds.begin(),
            newLocallyOwnedGlobalNodeIds.end());
          ghostFlattenedNodesSet.add_indices(newGhostGlobalNodeIds.begin(),
                                             newGhostGlobalNodeIds.end());
        }

      // compress index set ranges
      locallyOwnedFlattenedNodesSet.compress();
      ghostFlattenedNodesSet.compress();

      bool print = false;
      if (print)
        {
          std::cout << "Number of Wave Functions per Block: " << blockSize
                    << std::endl;
          std::stringstream ss1;
          locallyOwnedFlattenedNodesSet.print(ss1);
          std::stringstream ss2;
          ghostFlattenedNodesSet.print(ss2);
          std::string s1(ss1.str());
          s1.pop_back();
          std::string s2(ss2.str());
          s2.pop_back();
          std::cout << "procId: "
                    << dealii::Utilities::MPI::this_mpi_process(
                         mpi_communicator)
                    << " new owned: " << s1 << " new ghost: " << s2
                    << std::endl;
        }

      //
      // sanity check
      //
      AssertThrow(
        locallyOwnedFlattenedNodesSet.is_ascending_and_one_to_one(
          mpi_communicator),
        dealii::ExcMessage(
          "Incorrect renumbering and/or partitioning of flattened wave function matrix"));

      //
      // create flattened wave function matrix
      //
      flattenedArray.reinit(locallyOwnedFlattenedNodesSet,
                            ghostFlattenedNodesSet,
                            mpi_communicator);
    }


#ifdef DFTFE_WITH_GPU
    template <typename T>
    void
    createDealiiVector(
      const std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
        &                   partitioner,
      const unsigned int    blockSize,
      distributedGPUVec<T> &flattenedArray)
    {
      const MPI_Comm &mpi_communicator = partitioner->get_mpi_communicator();
      //
      // Get required sizes
      //
      const unsigned int n_ghosts  = partitioner->n_ghost_indices();
      const unsigned int localSize = partitioner->local_size();
      const unsigned int totalSize = localSize + n_ghosts;
      const dealii::types::global_dof_index globalNumberDegreesOfFreedom =
        partitioner->size();

      //
      // create data for new parallel layout
      //
      dealii::IndexSet locallyOwnedFlattenedNodesSet, ghostFlattenedNodesSet;
      locallyOwnedFlattenedNodesSet.clear();
      ghostFlattenedNodesSet.clear();

      //
      // Set the maximal size of the indices upon which this object operates.
      //
      locallyOwnedFlattenedNodesSet.set_size(
        globalNumberDegreesOfFreedom *
        (dealii::types::global_dof_index)blockSize);
      ghostFlattenedNodesSet.set_size(
        globalNumberDegreesOfFreedom *
        (dealii::types::global_dof_index)blockSize);


      for (unsigned int ilocaldof = 0; ilocaldof < totalSize; ++ilocaldof)
        {
          std::vector<dealii::types::global_dof_index>
                                                       newLocallyOwnedGlobalNodeIds;
          std::vector<dealii::types::global_dof_index> newGhostGlobalNodeIds;
          const dealii::types::global_dof_index        globalIndex =
            partitioner->local_to_global(ilocaldof);
          const bool isGhost = partitioner->is_ghost_entry(globalIndex);
          if (isGhost)
            {
              for (unsigned int iwave = 0; iwave < blockSize; ++iwave)
                {
                  newGhostGlobalNodeIds.push_back(
                    (dealii::types::global_dof_index)blockSize * globalIndex +
                    (dealii::types::global_dof_index)iwave);
                }
            }
          else
            {
              for (unsigned int iwave = 0; iwave < blockSize; ++iwave)
                {
                  newLocallyOwnedGlobalNodeIds.push_back(
                    (dealii::types::global_dof_index)blockSize * globalIndex +
                    (dealii::types::global_dof_index)iwave);
                }
            }

          // insert into dealii index sets
          locallyOwnedFlattenedNodesSet.add_indices(
            newLocallyOwnedGlobalNodeIds.begin(),
            newLocallyOwnedGlobalNodeIds.end());
          ghostFlattenedNodesSet.add_indices(newGhostGlobalNodeIds.begin(),
                                             newGhostGlobalNodeIds.end());
        }

      // compress index set ranges
      locallyOwnedFlattenedNodesSet.compress();
      ghostFlattenedNodesSet.compress();

      bool print = false;
      if (print)
        {
          std::cout << "Number of Wave Functions per Block: " << blockSize
                    << std::endl;
          std::stringstream ss1;
          locallyOwnedFlattenedNodesSet.print(ss1);
          std::stringstream ss2;
          ghostFlattenedNodesSet.print(ss2);
          std::string s1(ss1.str());
          s1.pop_back();
          std::string s2(ss2.str());
          s2.pop_back();
          std::cout << "procId: "
                    << dealii::Utilities::MPI::this_mpi_process(
                         mpi_communicator)
                    << " new owned: " << s1 << " new ghost: " << s2
                    << std::endl;
        }

      //
      // sanity check
      //
      AssertThrow(
        locallyOwnedFlattenedNodesSet.is_ascending_and_one_to_one(
          mpi_communicator),
        dealii::ExcMessage(
          "Incorrect renumbering and/or partitioning of flattened wave function matrix"));

      //
      // create flattened wave function matrix
      //
      flattenedArray.reinit(locallyOwnedFlattenedNodesSet,
                            ghostFlattenedNodesSet,
                            mpi_communicator);
    }
#endif
  } // namespace distributedBlockvecInternal

  template <typename NumberType, typename MemorySpace>
  DistributedDealiiBlockVec<NumberType,
                            MemorySpace>::DistributedDealiiBlockVec()
    : d_vecData(NULL)
  {
    if (std::is_same<NumberType, double>::value ||
        std::is_same<NumberType, float>::value ||
        std::is_same<NumberType, std::complex<double>>::value ||
        std::is_same<NumberType, std::complex<float>>::value)
      {
        if (std::is_same<MemorySpace, dftfe::MemorySpace::Host>::value)
          d_dealiiVecData =
            (void *)(new dealii::LinearAlgebra::distributed::
                       Vector<NumberType, dealii::MemorySpace::Host>);
        else
          {
#if defined(DFTFE_WITH_GPU)
            d_dealiiVecData =
              (void *)(new dealii::LinearAlgebra::distributed::
                         Vector<NumberType, dealii::MemorySpace::CUDA>);
#endif
          }
      }
    else if (std::is_same<MemorySpace, dftfe::MemorySpace::GPU>::value &&
             (std::is_same<NumberType, cuDoubleComplex>::value ||
              std::is_same<NumberType, cuFloatComplex>::value))
      {
#if defined(DFTFE_WITH_GPU)
        if (std::is_same<NumberType, cuDoubleComplex>::value)
          {
            d_dealiiVecDataReal =
              (void *)(new dealii::LinearAlgebra::distributed::
                         Vector<double, dealii::MemorySpace::CUDA>);
            d_dealiiVecDataImag =
              (void *)(new dealii::LinearAlgebra::distributed::
                         Vector<double, dealii::MemorySpace::CUDA>);
          }
        else if (std::is_same<NumberType, cuFloatComplex>::value)
          {
            d_dealiiVecDataReal =
              (void *)(new dealii::LinearAlgebra::distributed::
                         Vector<float, dealii::MemorySpace::CUDA>);
            d_dealiiVecDataImag =
              (void *)(new dealii::LinearAlgebra::distributed::
                         Vector<float, dealii::MemorySpace::CUDA>);
          }
#endif
      }
  }

  template <typename NumberType, typename MemorySpace>
  DistributedDealiiBlockVec<NumberType,
                            MemorySpace>::~DistributedDealiiBlockVec()
  {
    if (d_vecData != NULL)
      {
        if (std::is_same<MemorySpace, dftfe::MemorySpace::Host>::value)
          free(d_vecData);
        else if (std::is_same<MemorySpace, dftfe::MemorySpace::GPU>::value)
          {
#if defined(DFTFE_WITH_GPU)
            cudaFree(&d_vecData);
#endif
          }
      }

    if (std::is_same<NumberType, double>::value ||
        std::is_same<NumberType, float>::value ||
        std::is_same<NumberType, std::complex<double>>::value ||
        std::is_same<NumberType, std::complex<float>>::value)
      {
        if (std::is_same<MemorySpace, dftfe::MemorySpace::Host>::value)
          delete (
            dealii::LinearAlgebra::distributed::
              Vector<NumberType, dealii::MemorySpace::Host> *)d_dealiiVecData;
        else
          {
#if defined(DFTFE_WITH_GPU)
            delete (
              dealii::LinearAlgebra::distributed::
                Vector<NumberType, dealii::MemorySpace::CUDA> *)d_dealiiVecData;
#endif
          }
      }
    else if (std::is_same<MemorySpace, dftfe::MemorySpace::GPU>::value &&
             (std::is_same<NumberType, cuDoubleComplex>::value ||
              std::is_same<NumberType, cuFloatComplex>::value))
      {
#if defined(DFTFE_WITH_GPU)
        if (std::is_same<NumberType, cuDoubleComplex>::value)
          {
            delete (
              dealii::LinearAlgebra::distributed::
                Vector<double, dealii::MemorySpace::CUDA> *)d_dealiiVecDataReal;
            delete (
              dealii::LinearAlgebra::distributed::
                Vector<double, dealii::MemorySpace::CUDA> *)d_dealiiVecDataImag;
          }
        else if (std::is_same<NumberType, cuFloatComplex>::value)
          {
            delete (
              dealii::LinearAlgebra::distributed::
                Vector<float, dealii::MemorySpace::CUDA> *)d_dealiiVecDataReal;
            delete (
              dealii::LinearAlgebra::distributed::
                Vector<float, dealii::MemorySpace::CUDA> *)d_dealiiVecDataImag;
          }
#endif
      }
  }

  template <typename NumberType, typename MemorySpace>
  void
  DistributedDealiiBlockVec<NumberType, MemorySpace>::reinit(
    const std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
      &                partitionerSingleVec,
    const unsigned int blockSize)
  {
    if (std::is_same<NumberType, double>::value ||
        std::is_same<NumberType, float>::value ||
        std::is_same<NumberType, std::complex<double>>::value ||
        std::is_same<NumberType, std::complex<float>>::value)
      {
        if (std::is_same<MemorySpace, dftfe::MemorySpace::Host>::value)
          distributedBlockvecInternal::createDealiiVector(
            partitionerSingleVec,
            blockSize,
            *((dealii::LinearAlgebra::distributed::
                 Vector<NumberType, dealii::MemorySpace::Host> *)
                d_dealiiVecData));
        else if (std::is_same<MemorySpace, dftfe::MemorySpace::GPU>::value)
          {
#if defined(DFTFE_WITH_GPU)
            distributedBlockvecInternal::createDealiiVector(
              partitionerSingleVec,
              blockSize,
              *((dealii::LinearAlgebra::distributed::
                   Vector<NumberType, dealii::MemorySpace::CUDA> *)
                  d_dealiiVecData));
#endif
          }
      }
    else if (std::is_same<MemorySpace, dftfe::MemorySpace::GPU>::value &&
             (std::is_same<NumberType, cuDoubleComplex>::value ||
              std::is_same<NumberType, cuFloatComplex>::value))
      {
#if defined(DFTFE_WITH_GPU)
        cudaMalloc((void **)&d_vecData, d_size * sizeof(NumberType));
        if (std::is_same<NumberType, cuDoubleComplex>::value)
          {
            distributedBlockvecInternal::createDealiiVector(
              partitionerSingleVec,
              blockSize,
              *((dealii::LinearAlgebra::distributed::
                   Vector<double, dealii::MemorySpace::Host> *)
                  d_dealiiVecDataReal));

            distributedBlockvecInternal::createDealiiVector(
              partitionerSingleVec,
              blockSize,
              *((dealii::LinearAlgebra::distributed::
                   Vector<double, dealii::MemorySpace::Host> *)
                  d_dealiiVecDataImag));
          }
        else if (std::is_same<NumberType, cuFloatComplex>::value)
          {
            distributedBlockvecInternal::createDealiiVector(
              partitionerSingleVec,
              blockSize,
              *((dealii::LinearAlgebra::distributed::
                   Vector<float, dealii::MemorySpace::Host> *)
                  d_dealiiVecDataReal));

            distributedBlockvecInternal::createDealiiVector(
              partitionerSingleVec,
              blockSize,
              *((dealii::LinearAlgebra::distributed::
                   Vector<float, dealii::MemorySpace::Host> *)
                  d_dealiiVecDataImag));
          }
        else
          {
            AssertThrow(false, dftUtils::ExcNotImplementedYet());
          }
#endif
      }
    else
      {
        AssertThrow(false, dftUtils::ExcNotImplementedYet());
      }
  }


  template <typename NumberType, typename MemorySpace>
  NumberType *
  DistributedDealiiBlockVec<NumberType, MemorySpace>::begin()
  {
    if (std::is_same<NumberType, double>::value ||
        std::is_same<NumberType, float>::value ||
        std::is_same<NumberType, std::complex<double>>::value ||
        std::is_same<NumberType, std::complex<float>>::value)
      {
        if (std::is_same<MemorySpace, dftfe::MemorySpace::Host>::value)
          return ((dealii::LinearAlgebra::distributed::
                     Vector<NumberType, dealii::MemorySpace::Host> *)
                    d_dealiiVecData)
            ->begin();
        else
          {
#if defined(DFTFE_WITH_GPU)
            return ((dealii::LinearAlgebra::distributed::
                       Vector<NumberType, dealii::MemorySpace::CUDA> *)
                      d_dealiiVecData)
              ->begin();
#endif
          }
      }
    else
      return d_vecData;
  }

  template <typename NumberType, typename MemorySpace>
  const NumberType *
  DistributedDealiiBlockVec<NumberType, MemorySpace>::begin() const
  {
    if (std::is_same<NumberType, double>::value ||
        std::is_same<NumberType, float>::value ||
        std::is_same<NumberType, std::complex<double>>::value ||
        std::is_same<NumberType, std::complex<float>>::value)
      {
        if (std::is_same<MemorySpace, dftfe::MemorySpace::Host>::value)
          return ((dealii::LinearAlgebra::distributed::
                     Vector<NumberType, dealii::MemorySpace::Host> *)
                    d_dealiiVecData)
            ->begin();
        else
          {
#if defined(DFTFE_WITH_GPU)
            return ((dealii::LinearAlgebra::distributed::
                       Vector<NumberType, dealii::MemorySpace::CUDA> *)
                      d_dealiiVecData)
              ->begin();
#endif
          }
      }
    else
      return d_vecData;
  }

  template <typename NumberType, typename MemorySpace>
  unsigned int
  DistributedDealiiBlockVec<NumberType, MemorySpace>::size() const
  {
    return d_size;
  }

  template <typename NumberType, typename MemorySpace>
  unsigned int
  DistributedDealiiBlockVec<NumberType, MemorySpace>::dofsSize() const
  {
    return d_dofsSize;
  }

  template <typename NumberType, typename MemorySpace>
  unsigned int
  DistributedDealiiBlockVec<NumberType, MemorySpace>::blockSize() const
  {
    return d_blockSize;
  }

  template <typename NumberType, typename MemorySpace>
  void
  DistributedDealiiBlockVec<NumberType, MemorySpace>::updateGhostValues()
  {
    if (std::is_same<NumberType, double>::value ||
        std::is_same<NumberType, float>::value ||
        std::is_same<NumberType, std::complex<double>>::value ||
        std::is_same<NumberType, std::complex<float>>::value)
      {
        if (std::is_same<MemorySpace, dftfe::MemorySpace::Host>::value)
          return ((dealii::LinearAlgebra::distributed::
                     Vector<NumberType, dealii::MemorySpace::Host> *)
                    d_dealiiVecData)
            ->update_ghost_values();
        else
          {
#if defined(DFTFE_WITH_GPU)
            return ((dealii::LinearAlgebra::distributed::
                       Vector<NumberType, dealii::MemorySpace::CUDA> *)
                      d_dealiiVecData)
              ->update_ghost_values();
#endif
          }
      }
    else
      {
#if defined(DFTFE_WITH_GPU)

#endif
      }
  }


  template <typename NumberType, typename MemorySpace>
  void
  DistributedDealiiBlockVec<NumberType, MemorySpace>::compressAdd()
  {
    if (std::is_same<NumberType, double>::value ||
        std::is_same<NumberType, float>::value ||
        std::is_same<NumberType, std::complex<double>>::value ||
        std::is_same<NumberType, std::complex<float>>::value)
      {
        if (std::is_same<MemorySpace, dftfe::MemorySpace::Host>::value)
          return ((dealii::LinearAlgebra::distributed::
                     Vector<NumberType, dealii::MemorySpace::Host> *)
                    d_dealiiVecData)
            ->compress(dealii::VectorOperation::add);
        else
          {
#if defined(DFTFE_WITH_GPU)
            return ((dealii::LinearAlgebra::distributed::
                       Vector<NumberType, dealii::MemorySpace::CUDA> *)
                      d_dealiiVecData)
              ->compress(dealii::VectorOperation::add);
#endif
          }
      }
    else
      {
#if defined(DFTFE_WITH_GPU)

#endif
      }
  }


  template class DistributedDealiiBlockVec<double, dftfe::MemorySpace::Host>;
  template class DistributedDealiiBlockVec<float, dftfe::MemorySpace::Host>;
  template class DistributedDealiiBlockVec<std::complex<double>,
                                           dftfe::MemorySpace::Host>;
  template class DistributedDealiiBlockVec<std::complex<float>,
                                           dftfe::MemorySpace::Host>;

#if defined(DFTFE_WITH_GPU)
  template class DistributedDealiiBlockVec<double, dftfe::MemorySpace::GPU>;
  template class DistributedDealiiBlockVec<float, dftfe::MemorySpace::GPU>;
  template class DistributedDealiiBlockVec<cuDoubleComplex,
                                           dftfe::MemorySpace::GPU>;
  template class DistributedDealiiBlockVec<cuFloatComplex,
                                           dftfe::MemorySpace::GPU>;
#endif
} // namespace dftfe
