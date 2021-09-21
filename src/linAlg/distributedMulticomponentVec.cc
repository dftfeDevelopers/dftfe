// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2021 The Regents of the University of Michigan and DFT-FE
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

#include "distributedMulticomponentVec.h"
#include "dftUtils.h"
#include <deal.II/lac/la_parallel_vector.h>

#if defined(DFTFE_WITH_GPU)
#  include "cudaHelpers.h"
#  include <thrust/device_vector.h>
#  include <thrust/complex.h>
#  include <cuComplex.h>
#endif

namespace dftfe
{
  namespace distributedMulticomponentvecInternal
  {
    template <typename T>
    void
    createDealiiVector(
      const std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
        &                              partitioner,
      const dataTypes::local_size_type numberComponents,
      dealii::LinearAlgebra::distributed::Vector<T, dealii::MemorySpace::Host>
        &flattenedArray)
    {
      const MPI_Comm &mpi_communicator = partitioner->get_mpi_communicator();
      //
      // Get required sizes
      //
      const dataTypes::local_size_type n_ghosts =
        partitioner->n_ghost_indices();
      const dataTypes::local_size_type  localSize = partitioner->local_size();
      const dataTypes::local_size_type  totalSize = localSize + n_ghosts;
      const dataTypes::global_size_type globalNumberDegreesOfFreedom =
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
        (dataTypes::global_size_type)numberComponents);
      ghostFlattenedNodesSet.set_size(
        globalNumberDegreesOfFreedom *
        (dataTypes::global_size_type)numberComponents);


      for (dataTypes::local_size_type ilocaldof = 0; ilocaldof < totalSize;
           ++ilocaldof)
        {
          std::vector<dataTypes::global_size_type> newLocallyOwnedGlobalNodeIds;
          std::vector<dataTypes::global_size_type> newGhostGlobalNodeIds;
          const dataTypes::global_size_type        globalIndex =
            partitioner->local_to_global(ilocaldof);
          const bool isGhost = partitioner->is_ghost_entry(globalIndex);
          if (isGhost)
            {
              for (dataTypes::local_size_type iwave = 0;
                   iwave < numberComponents;
                   ++iwave)
                {
                  newGhostGlobalNodeIds.push_back(
                    (dataTypes::global_size_type)numberComponents *
                      globalIndex +
                    (dataTypes::global_size_type)iwave);
                }
            }
          else
            {
              for (dataTypes::local_size_type iwave = 0;
                   iwave < numberComponents;
                   ++iwave)
                {
                  newLocallyOwnedGlobalNodeIds.push_back(
                    (dataTypes::global_size_type)numberComponents *
                      globalIndex +
                    (dataTypes::global_size_type)iwave);
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
          std::cout << "Number of Wave Functions per Block: "
                    << numberComponents << std::endl;
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
        &                              partitioner,
      const dataTypes::local_size_type numberComponents,
      dealii::LinearAlgebra::distributed::Vector<T, dealii::MemorySpace::CUDA>
        &flattenedArray)
    {
      const MPI_Comm &mpi_communicator = partitioner->get_mpi_communicator();
      //
      // Get required sizes
      //
      const dataTypes::local_size_type n_ghosts =
        partitioner->n_ghost_indices();
      const dataTypes::local_size_type  localSize = partitioner->local_size();
      const dataTypes::local_size_type  totalSize = localSize + n_ghosts;
      const dataTypes::global_size_type globalNumberDegreesOfFreedom =
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
        (dataTypes::global_size_type)numberComponents);
      ghostFlattenedNodesSet.set_size(
        globalNumberDegreesOfFreedom *
        (dataTypes::global_size_type)numberComponents);


      for (dataTypes::local_size_type ilocaldof = 0; ilocaldof < totalSize;
           ++ilocaldof)
        {
          std::vector<dataTypes::global_size_type> newLocallyOwnedGlobalNodeIds;
          std::vector<dataTypes::global_size_type> newGhostGlobalNodeIds;
          const dataTypes::global_size_type        globalIndex =
            partitioner->local_to_global(ilocaldof);
          const bool isGhost = partitioner->is_ghost_entry(globalIndex);
          if (isGhost)
            {
              for (dataTypes::local_size_type iwave = 0;
                   iwave < numberComponents;
                   ++iwave)
                {
                  newGhostGlobalNodeIds.push_back(
                    (dataTypes::global_size_type)numberComponents *
                      globalIndex +
                    (dataTypes::global_size_type)iwave);
                }
            }
          else
            {
              for (dataTypes::local_size_type iwave = 0;
                   iwave < numberComponents;
                   ++iwave)
                {
                  newLocallyOwnedGlobalNodeIds.push_back(
                    (dataTypes::global_size_type)numberComponents *
                      globalIndex +
                    (dataTypes::global_size_type)iwave);
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
          std::cout << "Number of Wave Functions per Block: "
                    << numberComponents << std::endl;
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
  } // namespace distributedMulticomponentvecInternal

  // Constructor
  template <typename NumberType, typename MemorySpace>
  DistributedMulticomponentVec<NumberType,
                               MemorySpace>::DistributedMulticomponentVec()
    : d_vecData(NULL)
    , d_dealiiVecData(NULL)
    , d_dealiiVecTempDataReal(NULL)
    , d_dealiiVecTempDataImag(NULL)
    , d_locallyOwnedSize(0)
    , d_ghostSize(0)
    , d_globalSize(0)
    , d_locallyOwnedDofsSize(0)
    , d_numberComponents(0)

  {}

  // Destructor
  template <typename NumberType, typename MemorySpace>
  DistributedMulticomponentVec<NumberType,
                               MemorySpace>::~DistributedMulticomponentVec()
  {
    this->clear();
  }

  template <typename NumberType, typename MemorySpace>
  void
  DistributedMulticomponentVec<NumberType, MemorySpace>::reinit(
    const std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
      &                              partitionerSingleVec,
    const dataTypes::local_size_type numberComponents)
  {
    this->clear();
    d_locallyOwnedDofsSize = partitionerSingleVec->locally_owned_size();
    d_numberComponents     = numberComponents;
    d_locallyOwnedSize     = d_numberComponents * d_locallyOwnedDofsSize;
    d_ghostSize  = d_numberComponents * partitionerSingleVec->n_ghost_indices();
    d_globalSize = d_numberComponents * partitionerSingleVec->size();

    if (std::is_same<NumberType, double>::value ||
        std::is_same<NumberType, float>::value ||
        std::is_same<NumberType, std::complex<double>>::value ||
        std::is_same<NumberType, std::complex<float>>::value)
      {
        if (std::is_same<MemorySpace, dftfe::MemorySpace::Host>::value)
          {
            d_dealiiVecData =
              (void *)(new dealii::LinearAlgebra::distributed::
                         Vector<NumberType, dealii::MemorySpace::Host>);

            distributedMulticomponentvecInternal::createDealiiVector(
              partitionerSingleVec,
              d_numberComponents,
              *((dealii::LinearAlgebra::distributed::
                   Vector<NumberType, dealii::MemorySpace::Host> *)
                  d_dealiiVecData));
          }
        else
          {
#if defined(DFTFE_WITH_GPU)
            d_dealiiVecData =
              (void *)(new dealii::LinearAlgebra::distributed::
                         Vector<NumberType, dealii::MemorySpace::CUDA>);

            distributedMulticomponentvecInternal::createDealiiVector(
              partitionerSingleVec,
              d_numberComponents,
              *((dealii::LinearAlgebra::distributed::
                   Vector<NumberType, dealii::MemorySpace::CUDA> *)
                  d_dealiiVecData));
#endif
          }
      }
#if defined(DFTFE_WITH_GPU)
    else if (std::is_same<MemorySpace, dftfe::MemorySpace::GPU>::value &&
             (std::is_same<NumberType, cuDoubleComplex>::value ||
              std::is_same<NumberType, cuFloatComplex>::value))
      {
        cudaMalloc((void **)&d_vecData,
                   (d_locallyOwnedSize + d_ghostSize) * sizeof(NumberType));
        cudaMemset(d_vecData,
                   0,
                   (d_locallyOwnedSize + d_ghostSize) * sizeof(NumberType));

        if (std::is_same<NumberType, cuDoubleComplex>::value)
          {
            d_dealiiVecTempDataReal =
              (void *)(new dealii::LinearAlgebra::distributed::
                         Vector<double, dealii::MemorySpace::CUDA>);
            d_dealiiVecTempDataImag =
              (void *)(new dealii::LinearAlgebra::distributed::
                         Vector<double, dealii::MemorySpace::CUDA>);

            distributedMulticomponentvecInternal::createDealiiVector(
              partitionerSingleVec,
              d_numberComponents,
              *((dealii::LinearAlgebra::distributed::
                   Vector<double, dealii::MemorySpace::CUDA> *)
                  d_dealiiVecTempDataReal));

            ((dealii::LinearAlgebra::distributed::
                Vector<double, dealii::MemorySpace::CUDA> *)
               d_dealiiVecTempDataImag)
              ->reinit(*((const dealii::LinearAlgebra::distributed::
                            Vector<double, dealii::MemorySpace::CUDA> *)
                           d_dealiiVecTempDataReal));
          }
        else if (std::is_same<NumberType, cuFloatComplex>::value)
          {
            d_dealiiVecTempDataReal =
              (void *)(new dealii::LinearAlgebra::distributed::
                         Vector<float, dealii::MemorySpace::CUDA>);
            d_dealiiVecTempDataImag =
              (void *)(new dealii::LinearAlgebra::distributed::
                         Vector<float, dealii::MemorySpace::CUDA>);

            distributedMulticomponentvecInternal::createDealiiVector(
              partitionerSingleVec,
              d_numberComponents,
              *((dealii::LinearAlgebra::distributed::
                   Vector<float, dealii::MemorySpace::CUDA> *)
                  d_dealiiVecTempDataReal));

            ((dealii::LinearAlgebra::distributed::
                Vector<float, dealii::MemorySpace::CUDA> *)
               d_dealiiVecTempDataImag)
              ->reinit(*((const dealii::LinearAlgebra::distributed::
                            Vector<float, dealii::MemorySpace::CUDA> *)
                           d_dealiiVecTempDataReal));
          }
      }
#endif
    else
      {
        AssertThrow(false, dftUtils::ExcNotImplementedYet());
      }

    this->setZero();
  }


  template <typename NumberType, typename MemorySpace>
  void
  DistributedMulticomponentVec<NumberType, MemorySpace>::reinit(
    const DistributedMulticomponentVec<NumberType, MemorySpace> &vec)
  {
    this->clear();
    d_locallyOwnedSize     = vec.locallyOwnedFlattenedSize();
    d_ghostSize            = vec.ghostFlattenedSize();
    d_numberComponents     = vec.numberComponents();
    d_locallyOwnedDofsSize = vec.locallyOwnedDofsSize();
    d_globalSize           = vec.globalSize();

    if (std::is_same<NumberType, double>::value ||
        std::is_same<NumberType, float>::value ||
        std::is_same<NumberType, std::complex<double>>::value ||
        std::is_same<NumberType, std::complex<float>>::value)
      {
        if (std::is_same<MemorySpace, dftfe::MemorySpace::Host>::value)
          {
            d_dealiiVecData =
              (void *)(new dealii::LinearAlgebra::distributed::
                         Vector<NumberType, dealii::MemorySpace::Host>);

            ((dealii::LinearAlgebra::distributed::
                Vector<NumberType, dealii::MemorySpace::Host> *)d_dealiiVecData)
              ->reinit(*((const dealii::LinearAlgebra::distributed::
                            Vector<NumberType, dealii::MemorySpace::Host> *)
                           vec.getDealiiVec()));
          }
        else
          {
#if defined(DFTFE_WITH_GPU)
            d_dealiiVecData =
              (void *)(new dealii::LinearAlgebra::distributed::
                         Vector<NumberType, dealii::MemorySpace::CUDA>);

            ((dealii::LinearAlgebra::distributed::
                Vector<NumberType, dealii::MemorySpace::CUDA> *)d_dealiiVecData)
              ->reinit(*((const dealii::LinearAlgebra::distributed::
                            Vector<NumberType, dealii::MemorySpace::CUDA> *)
                           vec.getDealiiVec()));

#endif
          }
      }
#if defined(DFTFE_WITH_GPU)
    else if (std::is_same<MemorySpace, dftfe::MemorySpace::GPU>::value &&
             (std::is_same<NumberType, cuDoubleComplex>::value ||
              std::is_same<NumberType, cuFloatComplex>::value))
      {
        cudaMalloc((void **)&d_vecData,
                   (d_locallyOwnedSize + d_ghostSize) * sizeof(NumberType));
        cudaMemset(d_vecData,
                   0,
                   (d_locallyOwnedSize + d_ghostSize) * sizeof(NumberType));

        if (std::is_same<NumberType, cuDoubleComplex>::value)
          {
            d_dealiiVecTempDataReal =
              (void *)(new dealii::LinearAlgebra::distributed::
                         Vector<double, dealii::MemorySpace::CUDA>);
            d_dealiiVecTempDataImag =
              (void *)(new dealii::LinearAlgebra::distributed::
                         Vector<double, dealii::MemorySpace::CUDA>);


            ((dealii::LinearAlgebra::distributed::
                Vector<double, dealii::MemorySpace::CUDA> *)
               d_dealiiVecTempDataReal)
              ->reinit(*((const dealii::LinearAlgebra::distributed::
                            Vector<double, dealii::MemorySpace::CUDA> *)
                           vec.getDealiiVec()));


            ((dealii::LinearAlgebra::distributed::
                Vector<double, dealii::MemorySpace::CUDA> *)
               d_dealiiVecTempDataImag)
              ->reinit(*((const dealii::LinearAlgebra::distributed::
                            Vector<double, dealii::MemorySpace::CUDA> *)
                           vec.getDealiiVec()));
          }
        else if (std::is_same<NumberType, cuFloatComplex>::value)
          {
            d_dealiiVecTempDataReal =
              (void *)(new dealii::LinearAlgebra::distributed::
                         Vector<float, dealii::MemorySpace::CUDA>);
            d_dealiiVecTempDataImag =
              (void *)(new dealii::LinearAlgebra::distributed::
                         Vector<float, dealii::MemorySpace::CUDA>);

            ((dealii::LinearAlgebra::distributed::
                Vector<float, dealii::MemorySpace::CUDA> *)
               d_dealiiVecTempDataReal)
              ->reinit(*((const dealii::LinearAlgebra::distributed::
                            Vector<float, dealii::MemorySpace::CUDA> *)
                           vec.getDealiiVec()));


            ((dealii::LinearAlgebra::distributed::
                Vector<float, dealii::MemorySpace::CUDA> *)
               d_dealiiVecTempDataImag)
              ->reinit(*((const dealii::LinearAlgebra::distributed::
                            Vector<float, dealii::MemorySpace::CUDA> *)
                           vec.getDealiiVec()));
          }
      }
#endif
    else
      {
        AssertThrow(false, dftUtils::ExcNotImplementedYet());
      }

    this->setZero();
  }


  template <typename NumberType, typename MemorySpace>
  void
  DistributedMulticomponentVec<NumberType, MemorySpace>::setZero()
  {
    if (std::is_same<MemorySpace, dftfe::MemorySpace::Host>::value)
      {
        std::memset(this->begin(), 0, d_locallyOwnedSize * sizeof(NumberType));
      }
    else if (std::is_same<MemorySpace, dftfe::MemorySpace::GPU>::value)
      {
#if defined(DFTFE_WITH_GPU)
        cudaMemset(this->begin(), 0, d_locallyOwnedSize * sizeof(NumberType));
#endif
      }
    this->zeroOutGhosts();
  }



  template <typename NumberType, typename MemorySpace>
  NumberType *
  DistributedMulticomponentVec<NumberType, MemorySpace>::begin()
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
  DistributedMulticomponentVec<NumberType, MemorySpace>::begin() const
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
  dataTypes::global_size_type
  DistributedMulticomponentVec<NumberType, MemorySpace>::globalSize() const
  {
    return d_globalSize;
  }

  template <typename NumberType, typename MemorySpace>
  dataTypes::local_size_type
  DistributedMulticomponentVec<NumberType,
                               MemorySpace>::locallyOwnedFlattenedSize() const
  {
    return d_locallyOwnedSize;
  }

  template <typename NumberType, typename MemorySpace>
  dataTypes::local_size_type
  DistributedMulticomponentVec<NumberType, MemorySpace>::ghostFlattenedSize()
    const
  {
    return d_ghostSize;
  }

  template <typename NumberType, typename MemorySpace>
  dataTypes::local_size_type
  DistributedMulticomponentVec<NumberType, MemorySpace>::locallyOwnedDofsSize()
    const
  {
    return d_locallyOwnedDofsSize;
  }

  template <typename NumberType, typename MemorySpace>
  dataTypes::local_size_type
  DistributedMulticomponentVec<NumberType, MemorySpace>::numberComponents()
    const
  {
    return d_numberComponents;
  }

  template <typename NumberType, typename MemorySpace>
  void
  DistributedMulticomponentVec<NumberType, MemorySpace>::updateGhostValues()
  {
    if (std::is_same<NumberType, double>::value ||
        std::is_same<NumberType, float>::value ||
        std::is_same<NumberType, std::complex<double>>::value ||
        std::is_same<NumberType, std::complex<float>>::value)
      {
        if (std::is_same<MemorySpace, dftfe::MemorySpace::Host>::value)
          ((dealii::LinearAlgebra::distributed::
              Vector<NumberType, dealii::MemorySpace::Host> *)d_dealiiVecData)
            ->update_ghost_values();
        else
          {
#if defined(DFTFE_WITH_GPU)
            ((dealii::LinearAlgebra::distributed::
                Vector<NumberType, dealii::MemorySpace::CUDA> *)d_dealiiVecData)
              ->update_ghost_values();
#endif
          }
      }
#if defined(DFTFE_WITH_GPU)
    else if (std::is_same<MemorySpace, dftfe::MemorySpace::GPU>::value &&
             (std::is_same<NumberType, cuDoubleComplex>::value ||
              std::is_same<NumberType, cuFloatComplex>::value))
      {
        if (std::is_same<NumberType, cuDoubleComplex>::value)
          {
            cudaUtils::copyComplexArrToRealArrsGPU(
              d_locallyOwnedSize,
              d_vecData,
              ((dealii::LinearAlgebra::distributed::
                  Vector<double, dealii::MemorySpace::CUDA> *)
                 d_dealiiVecTempDataReal)
                ->begin(),
              ((dealii::LinearAlgebra::distributed::
                  Vector<double, dealii::MemorySpace::CUDA> *)
                 d_dealiiVecTempDataImag)
                ->begin());


            ((dealii::LinearAlgebra::distributed::
                Vector<double, dealii::MemorySpace::CUDA> *)
               d_dealiiVecTempDataReal)
              ->update_ghost_values();

            ((dealii::LinearAlgebra::distributed::
                Vector<double, dealii::MemorySpace::CUDA> *)
               d_dealiiVecTempDataImag)
              ->update_ghost_values();


            cudaUtils::copyRealArrsToComplexArrGPU(
              d_ghostSize,
              ((dealii::LinearAlgebra::distributed::
                  Vector<double, dealii::MemorySpace::CUDA> *)
                 d_dealiiVecTempDataReal)
                  ->begin() +
                d_locallyOwnedSize,
              ((dealii::LinearAlgebra::distributed::
                  Vector<double, dealii::MemorySpace::CUDA> *)
                 d_dealiiVecTempDataImag)
                  ->begin() +
                d_locallyOwnedSize,
              d_vecData + d_locallyOwnedSize);
          }
        else if (std::is_same<NumberType, cuFloatComplex>::value)
          {
            cudaUtils::copyComplexArrToRealArrsGPU(
              d_locallyOwnedSize,
              d_vecData,
              ((dealii::LinearAlgebra::distributed::
                  Vector<float, dealii::MemorySpace::CUDA> *)
                 d_dealiiVecTempDataReal)
                ->begin(),
              ((dealii::LinearAlgebra::distributed::
                  Vector<float, dealii::MemorySpace::CUDA> *)
                 d_dealiiVecTempDataImag)
                ->begin());

            ((dealii::LinearAlgebra::distributed::
                Vector<float, dealii::MemorySpace::CUDA> *)
               d_dealiiVecTempDataReal)
              ->update_ghost_values();

            ((dealii::LinearAlgebra::distributed::
                Vector<float, dealii::MemorySpace::CUDA> *)
               d_dealiiVecTempDataImag)
              ->update_ghost_values();


            cudaUtils::copyRealArrsToComplexArrGPU(
              d_ghostSize,
              ((dealii::LinearAlgebra::distributed::
                  Vector<float, dealii::MemorySpace::CUDA> *)
                 d_dealiiVecTempDataReal)
                  ->begin() +
                d_locallyOwnedSize,
              ((dealii::LinearAlgebra::distributed::
                  Vector<float, dealii::MemorySpace::CUDA> *)
                 d_dealiiVecTempDataImag)
                  ->begin() +
                d_locallyOwnedSize,
              d_vecData + d_locallyOwnedSize);
          }
      }
#endif
    else
      {
        AssertThrow(false, dftUtils::ExcNotImplementedYet());
      }
  }

  template <typename NumberType, typename MemorySpace>
  void
  DistributedMulticomponentVec<NumberType,
                               MemorySpace>::updateGhostValuesStart()
  {
    if (std::is_same<NumberType, double>::value ||
        std::is_same<NumberType, float>::value ||
        std::is_same<NumberType, std::complex<double>>::value ||
        std::is_same<NumberType, std::complex<float>>::value)
      {
        if (std::is_same<MemorySpace, dftfe::MemorySpace::Host>::value)
          ((dealii::LinearAlgebra::distributed::
              Vector<NumberType, dealii::MemorySpace::Host> *)d_dealiiVecData)
            ->update_ghost_values_start();
        else
          {
#if defined(DFTFE_WITH_GPU)
            ((dealii::LinearAlgebra::distributed::
                Vector<NumberType, dealii::MemorySpace::CUDA> *)d_dealiiVecData)
              ->update_ghost_values_start();
#endif
          }
      }
#if defined(DFTFE_WITH_GPU)
    else if (std::is_same<MemorySpace, dftfe::MemorySpace::GPU>::value &&
             (std::is_same<NumberType, cuDoubleComplex>::value ||
              std::is_same<NumberType, cuFloatComplex>::value))
      {
        if (std::is_same<NumberType, cuDoubleComplex>::value)
          {
            cudaUtils::copyComplexArrToRealArrsGPU(
              d_locallyOwnedSize,
              d_vecData,
              ((dealii::LinearAlgebra::distributed::
                  Vector<double, dealii::MemorySpace::CUDA> *)
                 d_dealiiVecTempDataReal)
                ->begin(),
              ((dealii::LinearAlgebra::distributed::
                  Vector<double, dealii::MemorySpace::CUDA> *)
                 d_dealiiVecTempDataImag)
                ->begin());

            ((dealii::LinearAlgebra::distributed::
                Vector<double, dealii::MemorySpace::CUDA> *)
               d_dealiiVecTempDataReal)
              ->update_ghost_values_start();

            ((dealii::LinearAlgebra::distributed::
                Vector<double, dealii::MemorySpace::CUDA> *)
               d_dealiiVecTempDataImag)
              ->update_ghost_values_start();
          }
        else if (std::is_same<NumberType, cuFloatComplex>::value)
          {
            cudaUtils::copyComplexArrToRealArrsGPU(
              d_locallyOwnedSize,
              d_vecData,
              ((dealii::LinearAlgebra::distributed::
                  Vector<float, dealii::MemorySpace::CUDA> *)
                 d_dealiiVecTempDataReal)
                ->begin(),
              ((dealii::LinearAlgebra::distributed::
                  Vector<float, dealii::MemorySpace::CUDA> *)
                 d_dealiiVecTempDataImag)
                ->begin());

            ((dealii::LinearAlgebra::distributed::
                Vector<float, dealii::MemorySpace::CUDA> *)
               d_dealiiVecTempDataReal)
              ->update_ghost_values_start();

            ((dealii::LinearAlgebra::distributed::
                Vector<float, dealii::MemorySpace::CUDA> *)
               d_dealiiVecTempDataImag)
              ->update_ghost_values_start();
          }
      }
#endif
    else
      {
        AssertThrow(false, dftUtils::ExcNotImplementedYet());
      }
  }


  template <typename NumberType, typename MemorySpace>
  void
  DistributedMulticomponentVec<NumberType,
                               MemorySpace>::updateGhostValuesFinish()
  {
    if (std::is_same<NumberType, double>::value ||
        std::is_same<NumberType, float>::value ||
        std::is_same<NumberType, std::complex<double>>::value ||
        std::is_same<NumberType, std::complex<float>>::value)
      {
        if (std::is_same<MemorySpace, dftfe::MemorySpace::Host>::value)
          ((dealii::LinearAlgebra::distributed::
              Vector<NumberType, dealii::MemorySpace::Host> *)d_dealiiVecData)
            ->update_ghost_values_finish();
        else
          {
#if defined(DFTFE_WITH_GPU)
            ((dealii::LinearAlgebra::distributed::
                Vector<NumberType, dealii::MemorySpace::CUDA> *)d_dealiiVecData)
              ->update_ghost_values_finish();
#endif
          }
      }
#if defined(DFTFE_WITH_GPU)
    else if (std::is_same<MemorySpace, dftfe::MemorySpace::GPU>::value &&
             (std::is_same<NumberType, cuDoubleComplex>::value ||
              std::is_same<NumberType, cuFloatComplex>::value))
      {
        if (std::is_same<NumberType, cuDoubleComplex>::value)
          {
            ((dealii::LinearAlgebra::distributed::
                Vector<double, dealii::MemorySpace::CUDA> *)
               d_dealiiVecTempDataReal)
              ->update_ghost_values_finish();

            ((dealii::LinearAlgebra::distributed::
                Vector<double, dealii::MemorySpace::CUDA> *)
               d_dealiiVecTempDataImag)
              ->update_ghost_values_finish();

            cudaUtils::copyRealArrsToComplexArrGPU(
              d_ghostSize,
              ((dealii::LinearAlgebra::distributed::
                  Vector<double, dealii::MemorySpace::CUDA> *)
                 d_dealiiVecTempDataReal)
                  ->begin() +
                d_locallyOwnedSize,
              ((dealii::LinearAlgebra::distributed::
                  Vector<double, dealii::MemorySpace::CUDA> *)
                 d_dealiiVecTempDataImag)
                  ->begin() +
                d_locallyOwnedSize,
              d_vecData + d_locallyOwnedSize);
          }
        else if (std::is_same<NumberType, cuFloatComplex>::value)
          {
            ((dealii::LinearAlgebra::distributed::
                Vector<float, dealii::MemorySpace::CUDA> *)
               d_dealiiVecTempDataReal)
              ->update_ghost_values_finish();

            ((dealii::LinearAlgebra::distributed::
                Vector<float, dealii::MemorySpace::CUDA> *)
               d_dealiiVecTempDataImag)
              ->update_ghost_values_finish();

            cudaUtils::copyRealArrsToComplexArrGPU(
              d_ghostSize,
              ((dealii::LinearAlgebra::distributed::
                  Vector<float, dealii::MemorySpace::CUDA> *)
                 d_dealiiVecTempDataReal)
                  ->begin() +
                d_locallyOwnedSize,
              ((dealii::LinearAlgebra::distributed::
                  Vector<float, dealii::MemorySpace::CUDA> *)
                 d_dealiiVecTempDataImag)
                  ->begin() +
                d_locallyOwnedSize,
              d_vecData + d_locallyOwnedSize);
          }
      }
#endif
    else
      {
        AssertThrow(false, dftUtils::ExcNotImplementedYet());
      }
  }


  template <typename NumberType, typename MemorySpace>
  void
  DistributedMulticomponentVec<NumberType, MemorySpace>::compressAdd()
  {
    if (std::is_same<NumberType, double>::value ||
        std::is_same<NumberType, float>::value ||
        std::is_same<NumberType, std::complex<double>>::value ||
        std::is_same<NumberType, std::complex<float>>::value)
      {
        if (std::is_same<MemorySpace, dftfe::MemorySpace::Host>::value)
          ((dealii::LinearAlgebra::distributed::
              Vector<NumberType, dealii::MemorySpace::Host> *)d_dealiiVecData)
            ->compress(dealii::VectorOperation::add);
        else
          {
#if defined(DFTFE_WITH_GPU)
            ((dealii::LinearAlgebra::distributed::
                Vector<NumberType, dealii::MemorySpace::CUDA> *)d_dealiiVecData)
              ->compress(dealii::VectorOperation::add);
#endif
          }
      }
#if defined(DFTFE_WITH_GPU)
    else if (std::is_same<MemorySpace, dftfe::MemorySpace::GPU>::value &&
             (std::is_same<NumberType, cuDoubleComplex>::value ||
              std::is_same<NumberType, cuFloatComplex>::value))
      {
        if (std::is_same<NumberType, cuDoubleComplex>::value)
          {
            cudaUtils::copyComplexArrToRealArrsGPU(
              (d_locallyOwnedSize + d_ghostSize),
              d_vecData,
              ((dealii::LinearAlgebra::distributed::
                  Vector<double, dealii::MemorySpace::CUDA> *)
                 d_dealiiVecTempDataReal)
                ->begin(),
              ((dealii::LinearAlgebra::distributed::
                  Vector<double, dealii::MemorySpace::CUDA> *)
                 d_dealiiVecTempDataImag)
                ->begin());

            ((dealii::LinearAlgebra::distributed::
                Vector<double, dealii::MemorySpace::CUDA> *)
               d_dealiiVecTempDataReal)
              ->compress(dealii::VectorOperation::add);

            ((dealii::LinearAlgebra::distributed::
                Vector<double, dealii::MemorySpace::CUDA> *)
               d_dealiiVecTempDataImag)
              ->compress(dealii::VectorOperation::add);

            cudaUtils::copyRealArrsToComplexArrGPU(
              d_locallyOwnedSize,
              ((dealii::LinearAlgebra::distributed::
                  Vector<double, dealii::MemorySpace::CUDA> *)
                 d_dealiiVecTempDataReal)
                ->begin(),
              ((dealii::LinearAlgebra::distributed::
                  Vector<double, dealii::MemorySpace::CUDA> *)
                 d_dealiiVecTempDataImag)
                ->begin(),
              d_vecData);
          }
        else if (std::is_same<NumberType, cuFloatComplex>::value)
          {
            cudaUtils::copyComplexArrToRealArrsGPU(
              (d_locallyOwnedSize + d_ghostSize),
              d_vecData,
              ((dealii::LinearAlgebra::distributed::
                  Vector<float, dealii::MemorySpace::CUDA> *)
                 d_dealiiVecTempDataReal)
                ->begin(),
              ((dealii::LinearAlgebra::distributed::
                  Vector<float, dealii::MemorySpace::CUDA> *)
                 d_dealiiVecTempDataImag)
                ->begin());

            ((dealii::LinearAlgebra::distributed::
                Vector<float, dealii::MemorySpace::CUDA> *)
               d_dealiiVecTempDataReal)
              ->compress_finish(dealii::VectorOperation::add);

            ((dealii::LinearAlgebra::distributed::
                Vector<float, dealii::MemorySpace::CUDA> *)
               d_dealiiVecTempDataImag)
              ->compress(dealii::VectorOperation::add);

            cudaUtils::copyRealArrsToComplexArrGPU(
              d_locallyOwnedSize,
              ((dealii::LinearAlgebra::distributed::
                  Vector<float, dealii::MemorySpace::CUDA> *)
                 d_dealiiVecTempDataReal)
                ->begin(),
              ((dealii::LinearAlgebra::distributed::
                  Vector<float, dealii::MemorySpace::CUDA> *)
                 d_dealiiVecTempDataImag)
                ->begin(),
              d_vecData);
          }
      }
#endif
    else
      {
        AssertThrow(false, dftUtils::ExcNotImplementedYet());
      }
  }

  template <typename NumberType, typename MemorySpace>
  void
  DistributedMulticomponentVec<NumberType, MemorySpace>::compressAddStart()
  {
    if (std::is_same<NumberType, double>::value ||
        std::is_same<NumberType, float>::value ||
        std::is_same<NumberType, std::complex<double>>::value ||
        std::is_same<NumberType, std::complex<float>>::value)
      {
        if (std::is_same<MemorySpace, dftfe::MemorySpace::Host>::value)
          ((dealii::LinearAlgebra::distributed::
              Vector<NumberType, dealii::MemorySpace::Host> *)d_dealiiVecData)
            ->compress_start(dealii::VectorOperation::add);
        else
          {
#if defined(DFTFE_WITH_GPU)
            ((dealii::LinearAlgebra::distributed::
                Vector<NumberType, dealii::MemorySpace::CUDA> *)d_dealiiVecData)
              ->compress_start(dealii::VectorOperation::add);
#endif
          }
      }
#if defined(DFTFE_WITH_GPU)
    else if (std::is_same<MemorySpace, dftfe::MemorySpace::GPU>::value &&
             (std::is_same<NumberType, cuDoubleComplex>::value ||
              std::is_same<NumberType, cuFloatComplex>::value))
      {
        if (std::is_same<NumberType, cuDoubleComplex>::value)
          {
            cudaUtils::copyComplexArrToRealArrsGPU(
              (d_locallyOwnedSize + d_ghostSize),
              d_vecData,
              ((dealii::LinearAlgebra::distributed::
                  Vector<double, dealii::MemorySpace::CUDA> *)
                 d_dealiiVecTempDataReal)
                ->begin(),
              ((dealii::LinearAlgebra::distributed::
                  Vector<double, dealii::MemorySpace::CUDA> *)
                 d_dealiiVecTempDataImag)
                ->begin());

            ((dealii::LinearAlgebra::distributed::
                Vector<double, dealii::MemorySpace::CUDA> *)
               d_dealiiVecTempDataReal)
              ->compress_start(dealii::VectorOperation::add);

            ((dealii::LinearAlgebra::distributed::
                Vector<double, dealii::MemorySpace::CUDA> *)
               d_dealiiVecTempDataImag)
              ->compress_start(dealii::VectorOperation::add);
          }
        else if (std::is_same<NumberType, cuFloatComplex>::value)
          {
            cudaUtils::copyComplexArrToRealArrsGPU(
              (d_locallyOwnedSize + d_ghostSize),
              d_vecData,
              ((dealii::LinearAlgebra::distributed::
                  Vector<float, dealii::MemorySpace::CUDA> *)
                 d_dealiiVecTempDataReal)
                ->begin(),
              ((dealii::LinearAlgebra::distributed::
                  Vector<float, dealii::MemorySpace::CUDA> *)
                 d_dealiiVecTempDataImag)
                ->begin());

            ((dealii::LinearAlgebra::distributed::
                Vector<float, dealii::MemorySpace::CUDA> *)
               d_dealiiVecTempDataReal)
              ->compress_start(dealii::VectorOperation::add);

            ((dealii::LinearAlgebra::distributed::
                Vector<float, dealii::MemorySpace::CUDA> *)
               d_dealiiVecTempDataImag)
              ->compress_start(dealii::VectorOperation::add);
          }
      }
#endif
    else
      {
        AssertThrow(false, dftUtils::ExcNotImplementedYet());
      }
  }

  template <typename NumberType, typename MemorySpace>
  void
  DistributedMulticomponentVec<NumberType, MemorySpace>::compressAddFinish()
  {
    if (std::is_same<NumberType, double>::value ||
        std::is_same<NumberType, float>::value ||
        std::is_same<NumberType, std::complex<double>>::value ||
        std::is_same<NumberType, std::complex<float>>::value)
      {
        if (std::is_same<MemorySpace, dftfe::MemorySpace::Host>::value)
          ((dealii::LinearAlgebra::distributed::
              Vector<NumberType, dealii::MemorySpace::Host> *)d_dealiiVecData)
            ->compress_finish(dealii::VectorOperation::add);
        else
          {
#if defined(DFTFE_WITH_GPU)
            ((dealii::LinearAlgebra::distributed::
                Vector<NumberType, dealii::MemorySpace::CUDA> *)d_dealiiVecData)
              ->compress_finish(dealii::VectorOperation::add);
#endif
          }
      }
#if defined(DFTFE_WITH_GPU)
    else if (std::is_same<MemorySpace, dftfe::MemorySpace::GPU>::value &&
             (std::is_same<NumberType, cuDoubleComplex>::value ||
              std::is_same<NumberType, cuFloatComplex>::value))
      {
        if (std::is_same<NumberType, cuDoubleComplex>::value)
          {
            ((dealii::LinearAlgebra::distributed::
                Vector<double, dealii::MemorySpace::CUDA> *)
               d_dealiiVecTempDataReal)
              ->compress_finish(dealii::VectorOperation::add);

            ((dealii::LinearAlgebra::distributed::
                Vector<double, dealii::MemorySpace::CUDA> *)
               d_dealiiVecTempDataImag)
              ->compress_finish(dealii::VectorOperation::add);

            cudaUtils::copyRealArrsToComplexArrGPU(
              d_locallyOwnedSize,
              ((dealii::LinearAlgebra::distributed::
                  Vector<double, dealii::MemorySpace::CUDA> *)
                 d_dealiiVecTempDataReal)
                ->begin(),
              ((dealii::LinearAlgebra::distributed::
                  Vector<double, dealii::MemorySpace::CUDA> *)
                 d_dealiiVecTempDataImag)
                ->begin(),
              d_vecData);
          }
        else if (std::is_same<NumberType, cuFloatComplex>::value)
          {
            ((dealii::LinearAlgebra::distributed::
                Vector<float, dealii::MemorySpace::CUDA> *)
               d_dealiiVecTempDataReal)
              ->compress_finish(dealii::VectorOperation::add);

            ((dealii::LinearAlgebra::distributed::
                Vector<float, dealii::MemorySpace::CUDA> *)
               d_dealiiVecTempDataImag)
              ->compress_finish(dealii::VectorOperation::add);

            cudaUtils::copyRealArrsToComplexArrGPU(
              d_locallyOwnedSize,
              ((dealii::LinearAlgebra::distributed::
                  Vector<float, dealii::MemorySpace::CUDA> *)
                 d_dealiiVecTempDataReal)
                ->begin(),
              ((dealii::LinearAlgebra::distributed::
                  Vector<float, dealii::MemorySpace::CUDA> *)
                 d_dealiiVecTempDataImag)
                ->begin(),
              d_vecData);
          }
      }
#endif
    else
      {
        AssertThrow(false, dftUtils::ExcNotImplementedYet());
      }
  }


  template <typename NumberType, typename MemorySpace>
  void
  DistributedMulticomponentVec<NumberType, MemorySpace>::zeroOutGhosts()
  {
    if (std::is_same<NumberType, double>::value ||
        std::is_same<NumberType, float>::value ||
        std::is_same<NumberType, std::complex<double>>::value ||
        std::is_same<NumberType, std::complex<float>>::value)
      {
        if (std::is_same<MemorySpace, dftfe::MemorySpace::Host>::value)
          ((dealii::LinearAlgebra::distributed::
              Vector<NumberType, dealii::MemorySpace::Host> *)d_dealiiVecData)
            ->zero_out_ghosts();
        else
          {
#if defined(DFTFE_WITH_GPU)
            ((dealii::LinearAlgebra::distributed::
                Vector<NumberType, dealii::MemorySpace::CUDA> *)d_dealiiVecData)
              ->zero_out_ghosts();
#endif
          }
      }
#if defined(DFTFE_WITH_GPU)
    else if (std::is_same<MemorySpace, dftfe::MemorySpace::GPU>::value &&
             (std::is_same<NumberType, cuDoubleComplex>::value ||
              std::is_same<NumberType, cuFloatComplex>::value))
      {
        cudaMemset(this->begin() + d_locallyOwnedSize,
                   0,
                   d_ghostSize * sizeof(NumberType));

        if (std::is_same<NumberType, cuDoubleComplex>::value)
          {
            ((dealii::LinearAlgebra::distributed::
                Vector<double, dealii::MemorySpace::CUDA> *)
               d_dealiiVecTempDataReal)
              ->zero_out_ghosts();

            ((dealii::LinearAlgebra::distributed::
                Vector<double, dealii::MemorySpace::CUDA> *)
               d_dealiiVecTempDataImag)
              ->zero_out_ghosts();
          }
        else if (std::is_same<NumberType, cuFloatComplex>::value)
          {
            ((dealii::LinearAlgebra::distributed::
                Vector<float, dealii::MemorySpace::CUDA> *)
               d_dealiiVecTempDataReal)
              ->zero_out_ghosts();

            ((dealii::LinearAlgebra::distributed::
                Vector<float, dealii::MemorySpace::CUDA> *)
               d_dealiiVecTempDataImag)
              ->zero_out_ghosts();
          }
      }
#endif
    else
      {
        AssertThrow(false, dftUtils::ExcNotImplementedYet());
      }
  }



  template <typename NumberType, typename MemorySpace>
  void
  DistributedMulticomponentVec<NumberType, MemorySpace>::swap(
    DistributedMulticomponentVec<NumberType, MemorySpace> &vec)
  {
    NumberType *tempPtr = this->d_vecData;
    this->d_vecData     = vec.d_vecData;
    vec.d_vecData       = tempPtr;

    if (std::is_same<NumberType, double>::value ||
        std::is_same<NumberType, float>::value ||
        std::is_same<NumberType, std::complex<double>>::value ||
        std::is_same<NumberType, std::complex<float>>::value)
      {
        if (std::is_same<MemorySpace, dftfe::MemorySpace::Host>::value)
          {
            ((dealii::LinearAlgebra::distributed::
                Vector<NumberType, dealii::MemorySpace::Host> *)d_dealiiVecData)
              ->swap(*((dealii::LinearAlgebra::distributed::
                          Vector<NumberType, dealii::MemorySpace::Host> *)
                         vec.d_dealiiVecData));
          }
        else if (std::is_same<MemorySpace, dftfe::MemorySpace::GPU>::value)
          {
#if defined(DFTFE_WITH_GPU)
            ((dealii::LinearAlgebra::distributed::
                Vector<NumberType, dealii::MemorySpace::CUDA> *)d_dealiiVecData)
              ->swap(*((dealii::LinearAlgebra::distributed::
                          Vector<NumberType, dealii::MemorySpace::CUDA> *)
                         vec.d_dealiiVecData));
#endif
          }
      }
#if defined(DFTFE_WITH_GPU)
    else if (std::is_same<MemorySpace, dftfe::MemorySpace::GPU>::value &&
             (std::is_same<NumberType, cuDoubleComplex>::value ||
              std::is_same<NumberType, cuFloatComplex>::value))
      {
        if (std::is_same<NumberType, cuDoubleComplex>::value)
          {
            ((dealii::LinearAlgebra::distributed::
                Vector<double, dealii::MemorySpace::CUDA> *)
               d_dealiiVecTempDataReal)
              ->swap(*((dealii::LinearAlgebra::distributed::
                          Vector<double, dealii::MemorySpace::CUDA> *)
                         vec.d_dealiiVecTempDataReal));
            ((dealii::LinearAlgebra::distributed::
                Vector<double, dealii::MemorySpace::CUDA> *)
               d_dealiiVecTempDataImag)
              ->swap(*((dealii::LinearAlgebra::distributed::
                          Vector<double, dealii::MemorySpace::CUDA> *)
                         vec.d_dealiiVecTempDataImag));
          }
        else if (std::is_same<NumberType, cuFloatComplex>::value)
          {
            ((dealii::LinearAlgebra::distributed::
                Vector<float, dealii::MemorySpace::CUDA> *)
               d_dealiiVecTempDataReal)
              ->swap(*((dealii::LinearAlgebra::distributed::
                          Vector<float, dealii::MemorySpace::CUDA> *)
                         vec.d_dealiiVecTempDataReal));
            ((dealii::LinearAlgebra::distributed::
                Vector<float, dealii::MemorySpace::CUDA> *)
               d_dealiiVecTempDataImag)
              ->swap(*((dealii::LinearAlgebra::distributed::
                          Vector<float, dealii::MemorySpace::CUDA> *)
                         vec.d_dealiiVecTempDataImag));
          }
      }
#endif

    dataTypes::local_size_type locallyOwnedSizeTemp = vec.d_locallyOwnedSize;
    dataTypes::local_size_type ghostSizeTemp        = vec.d_ghostSize;
    dataTypes::local_size_type numberComponentsTemp = vec.d_numberComponents;
    dataTypes::local_size_type locallyOwnedDofsSizeTemp =
      vec.d_locallyOwnedDofsSize;
    dataTypes::global_size_type globalSizeTemp = vec.d_globalSize;

    d_locallyOwnedSize     = vec.d_locallyOwnedSize;
    d_ghostSize            = vec.d_ghostSize;
    d_numberComponents     = vec.d_numberComponents;
    d_locallyOwnedDofsSize = vec.d_locallyOwnedDofsSize;
    d_globalSize           = vec.d_globalSize;

    vec.d_locallyOwnedSize     = locallyOwnedSizeTemp;
    vec.d_ghostSize            = ghostSizeTemp;
    vec.d_numberComponents     = numberComponentsTemp;
    vec.d_locallyOwnedDofsSize = locallyOwnedDofsSizeTemp;
    vec.d_globalSize           = globalSizeTemp;
  }

  template <typename NumberType, typename MemorySpace>
  const void *
  DistributedMulticomponentVec<NumberType, MemorySpace>::getDealiiVec() const
  {
    void *temp;
    if (std::is_same<NumberType, double>::value ||
        std::is_same<NumberType, float>::value ||
        std::is_same<NumberType, std::complex<double>>::value ||
        std::is_same<NumberType, std::complex<float>>::value)
      {
        temp = d_dealiiVecData;
      }
#if defined(DFTFE_WITH_GPU)
    else if (std::is_same<MemorySpace, dftfe::MemorySpace::GPU>::value &&
             (std::is_same<NumberType, cuDoubleComplex>::value ||
              std::is_same<NumberType, cuFloatComplex>::value))
      {
        temp = d_dealiiVecTempDataReal;
      }
#endif

    return temp;
  }


  template <typename NumberType, typename MemorySpace>
  void
  DistributedMulticomponentVec<NumberType, MemorySpace>::clear()
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

    if ((std::is_same<NumberType, double>::value ||
         std::is_same<NumberType, float>::value ||
         std::is_same<NumberType, std::complex<double>>::value ||
         std::is_same<NumberType, std::complex<float>>::value) &&
        d_dealiiVecData != NULL)
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
#if defined(DFTFE_WITH_GPU)
    else if (std::is_same<MemorySpace, dftfe::MemorySpace::GPU>::value &&
             (std::is_same<NumberType, cuDoubleComplex>::value ||
              std::is_same<NumberType, cuFloatComplex>::value) &&
             d_dealiiVecTempDataReal != NULL)
      {
        if (std::is_same<NumberType, cuDoubleComplex>::value)
          {
            delete (dealii::LinearAlgebra::distributed::
                      Vector<double, dealii::MemorySpace::CUDA> *)
              d_dealiiVecTempDataReal;
            delete (dealii::LinearAlgebra::distributed::
                      Vector<double, dealii::MemorySpace::CUDA> *)
              d_dealiiVecTempDataImag;
          }
        else if (std::is_same<NumberType, cuFloatComplex>::value)
          {
            delete (dealii::LinearAlgebra::distributed::
                      Vector<float, dealii::MemorySpace::CUDA> *)
              d_dealiiVecTempDataReal;
            delete (dealii::LinearAlgebra::distributed::
                      Vector<float, dealii::MemorySpace::CUDA> *)
              d_dealiiVecTempDataImag;
          }
      }
#endif

    d_vecData               = NULL;
    d_dealiiVecData         = NULL;
    d_dealiiVecTempDataReal = NULL;
    d_dealiiVecTempDataImag = NULL;
    d_locallyOwnedSize      = 0;
    d_ghostSize             = 0;
    d_locallyOwnedDofsSize  = 0;
    d_globalSize            = 0;
    d_numberComponents      = 0;
  }

  template <typename NumberType, typename MemorySpace>
  const std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
  DistributedMulticomponentVec<NumberType, MemorySpace>::getDealiiPartitioner()
    const
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
            ->get_partitioner();
        else
          {
#if defined(DFTFE_WITH_GPU)
            return ((dealii::LinearAlgebra::distributed::
                       Vector<NumberType, dealii::MemorySpace::CUDA> *)
                      d_dealiiVecData)
              ->get_partitioner();
#endif
          }
      }
#if defined(DFTFE_WITH_GPU)
    else if (std::is_same<MemorySpace, dftfe::MemorySpace::GPU>::value &&
             (std::is_same<NumberType, cuDoubleComplex>::value ||
              std::is_same<NumberType, cuFloatComplex>::value))
      {
        if (std::is_same<NumberType, cuDoubleComplex>::value)
          {
            return ((dealii::LinearAlgebra::distributed::
                       Vector<double, dealii::MemorySpace::CUDA> *)
                      d_dealiiVecTempDataReal)
              ->get_partitioner();
          }
        else if (std::is_same<NumberType, cuFloatComplex>::value)
          {
            return ((dealii::LinearAlgebra::distributed::
                       Vector<float, dealii::MemorySpace::CUDA> *)
                      d_dealiiVecTempDataReal)
              ->get_partitioner();
          }
      }
#endif
  }

  template class DistributedMulticomponentVec<double, dftfe::MemorySpace::Host>;
  template class DistributedMulticomponentVec<float, dftfe::MemorySpace::Host>;
  template class DistributedMulticomponentVec<std::complex<double>,
                                              dftfe::MemorySpace::Host>;
  template class DistributedMulticomponentVec<std::complex<float>,
                                              dftfe::MemorySpace::Host>;

#if defined(DFTFE_WITH_GPU)
  template class DistributedMulticomponentVec<double, dftfe::MemorySpace::GPU>;
  template class DistributedMulticomponentVec<float, dftfe::MemorySpace::GPU>;
  template class DistributedMulticomponentVec<cuDoubleComplex,
                                              dftfe::MemorySpace::GPU>;
  template class DistributedMulticomponentVec<cuFloatComplex,
                                              dftfe::MemorySpace::GPU>;
#endif
} // namespace dftfe
