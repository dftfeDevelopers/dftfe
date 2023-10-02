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
// @author  Phani Motamarri
//
#include <constraintMatrixInfo.h>
#include <linearAlgebraOperations.h>

namespace dftfe
{
  //
  // Declare dftUtils functions
  //
  namespace dftUtils
  {
    //
    // wrapper function to call blas function daxpy or zapxy depending
    // on the data type (complex or double)
    //

    void
    callaxpy(const unsigned int *n,
             const double *      alpha,
             double *            x,
             const unsigned int *incx,
             double *            y,
             const unsigned int *incy)
    {
      daxpy_(n, alpha, x, incx, y, incy);
    }

    void
    callaxpy(const unsigned int *        n,
             const std::complex<double> *alpha,
             std::complex<double> *      x,
             const unsigned int *        incx,
             std::complex<double> *      y,
             const unsigned int *        incy)
    {
      zaxpy_(n, alpha, x, incx, y, incy);
    }



    //
    // constructor
    //
    constraintMatrixInfo::constraintMatrixInfo()
    {}

    //
    // destructor
    //
    constraintMatrixInfo::~constraintMatrixInfo()
    {}


    //
    // store constraintMatrix row data in STL vector
    //
    void
    constraintMatrixInfo::initialize(
      const std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
        &                                      partitioner,
      const dealii::AffineConstraints<double> &constraintMatrixData)

    {
      clear();

      const dealii::IndexSet &locally_owned_dofs =
        partitioner->locally_owned_range();
      const dealii::IndexSet &ghost_dofs = partitioner->ghost_indices();

      for (dealii::IndexSet::ElementIterator it = locally_owned_dofs.begin();
           it != locally_owned_dofs.end();
           ++it)
        {
          if (constraintMatrixData.is_constrained(*it))
            {
              const dealii::types::global_dof_index lineDof = *it;
              const std::vector<
                std::pair<dealii::types::global_dof_index, double>> *rowData =
                constraintMatrixData.get_constraint_entries(lineDof);

              bool isConstraintRhsExpandingOutOfIndexSet = false;
              for (unsigned int j = 0; j < rowData->size(); ++j)
                {
                  if (!(partitioner->is_ghost_entry((*rowData)[j].first) ||
                        partitioner->in_local_range((*rowData)[j].first)))
                    {
                      isConstraintRhsExpandingOutOfIndexSet = true;
                      break;
                    }
                }

              if (isConstraintRhsExpandingOutOfIndexSet)
                continue;

              d_rowIdsLocal.push_back(partitioner->global_to_local(lineDof));
              d_rowIdsGlobal.push_back(lineDof);
              d_inhomogenities.push_back(
                constraintMatrixData.get_inhomogeneity(lineDof));
              d_rowSizes.push_back(rowData->size());
              for (unsigned int j = 0; j < rowData->size(); ++j)
                {
                  // Assert((*rowData)[j].first < partitioner->size(),
                  //       dealii::ExcMessage("Index out of bounds"));
                  d_columnIdsGlobal.push_back((*rowData)[j].first);
                  d_columnIdsLocal.push_back(
                    partitioner->global_to_local((*rowData)[j].first));
                  d_columnValues.push_back((*rowData)[j].second);
                }
            }
        }


      for (dealii::IndexSet::ElementIterator it = ghost_dofs.begin();
           it != ghost_dofs.end();
           ++it)
        {
          if (constraintMatrixData.is_constrained(*it))
            {
              const dealii::types::global_dof_index lineDof = *it;

              const std::vector<
                std::pair<dealii::types::global_dof_index, double>> *rowData =
                constraintMatrixData.get_constraint_entries(lineDof);

              bool isConstraintRhsExpandingOutOfIndexSet = false;
              for (unsigned int j = 0; j < rowData->size(); ++j)
                {
                  if (!(partitioner->is_ghost_entry((*rowData)[j].first) ||
                        partitioner->in_local_range((*rowData)[j].first)))
                    {
                      isConstraintRhsExpandingOutOfIndexSet = true;
                      break;
                    }
                }

              if (isConstraintRhsExpandingOutOfIndexSet)
                continue;

              d_rowIdsLocal.push_back(partitioner->global_to_local(lineDof));
              d_rowIdsGlobal.push_back(lineDof);
              d_inhomogenities.push_back(
                constraintMatrixData.get_inhomogeneity(lineDof));
              d_rowSizes.push_back(rowData->size());
              for (unsigned int j = 0; j < rowData->size(); ++j)
                {
                  // Assert((*rowData)[j].first < partitioner->size(),
                  //       dealii::ExcMessage("Index out of bounds"));
                  d_columnIdsGlobal.push_back((*rowData)[j].first);
                  d_columnIdsLocal.push_back(
                    partitioner->global_to_local((*rowData)[j].first));
                  d_columnValues.push_back((*rowData)[j].second);
                }
            }
        }
    }

    //
    // set the constrained degrees of freedom to values so that constraints
    // are satisfied
    //
    void
    constraintMatrixInfo::distribute(
      distributedCPUVec<double> &fieldVector) const
    {
      fieldVector.update_ghost_values();
      unsigned int count = 0;
      for (unsigned int i = 0; i < d_rowIdsLocal.size(); ++i)
        {
          double new_value = d_inhomogenities[i];
          for (unsigned int j = 0; j < d_rowSizes[i]; ++j)
            {
              new_value += fieldVector.local_element(d_columnIdsLocal[count]) *
                           d_columnValues[count];
              count++;
            }
          fieldVector.local_element(d_rowIdsLocal[i]) = new_value;
        }
    }


    template <typename T>
    void
    constraintMatrixInfo::distribute(distributedCPUVec<T> &fieldVector,
                                     const unsigned int    blockSize) const
    {
      fieldVector.update_ghost_values();


      unsigned int       count = 0;
      const unsigned int inc   = 1;
      std::vector<T>     newValuesBlock(blockSize, 0.0);
      for (unsigned int i = 0; i < d_rowIdsLocal.size(); ++i)
        {
          std::fill(newValuesBlock.begin(),
                    newValuesBlock.end(),
                    d_inhomogenities[i]);

          const dealii::types::global_dof_index startingLocalDofIndexRow =
            d_rowIdsLocal[i] * blockSize;

          for (unsigned int j = 0; j < d_rowSizes[i]; ++j)
            {
              Assert(
                count < d_columnIdsGlobal.size(),
                dealii::ExcMessage(
                  "Overloaded distribute for flattened array has indices out of bounds"));

              const dealii::types::global_dof_index
                startingLocalDofIndexColumn =
                  d_columnIdsLocal[count] * blockSize;

              T alpha = d_columnValues[count];

              callaxpy(&blockSize,
                       &alpha,
                       fieldVector.begin() + startingLocalDofIndexColumn,
                       &inc,
                       &newValuesBlock[0],
                       &inc);
              count++;
            }

          std::copy(&newValuesBlock[0],
                    &newValuesBlock[0] + blockSize,
                    fieldVector.begin() + startingLocalDofIndexRow);
        }
    }


    template <typename T>
    void
    constraintMatrixInfo::distribute(distributedCPUMultiVec<T> &fieldVector,
                                     const unsigned int         blockSize) const
    {
      fieldVector.updateGhostValues();


      unsigned int       count = 0;
      const unsigned int inc   = 1;
      std::vector<T>     newValuesBlock(blockSize, 0.0);
      for (unsigned int i = 0; i < d_rowIdsLocal.size(); ++i)
        {
          std::fill(newValuesBlock.begin(),
                    newValuesBlock.end(),
                    d_inhomogenities[i]);

          const dealii::types::global_dof_index startingLocalDofIndexRow =
            d_rowIdsLocal[i] * blockSize;

          for (unsigned int j = 0; j < d_rowSizes[i]; ++j)
            {
              Assert(
                count < d_columnIdsGlobal.size(),
                dealii::ExcMessage(
                  "Overloaded distribute for flattened array has indices out of bounds"));

              const dealii::types::global_dof_index
                startingLocalDofIndexColumn =
                  d_columnIdsLocal[count] * blockSize;

              T alpha = d_columnValues[count];

              callaxpy(&blockSize,
                       &alpha,
                       fieldVector.data() + startingLocalDofIndexColumn,
                       &inc,
                       &newValuesBlock[0],
                       &inc);
              count++;
            }

          std::copy(&newValuesBlock[0],
                    &newValuesBlock[0] + blockSize,
                    fieldVector.data() + startingLocalDofIndexRow);
        }
    }



    //
    // set the constrained degrees of freedom to values so that constraints
    // are satisfied for flattened array
    //
    template <typename T>
    void
    constraintMatrixInfo::distribute_slave_to_master(
      distributedCPUVec<T> &fieldVector,
      const unsigned int    blockSize) const
    {
      unsigned int       count = 0;
      const unsigned int inc   = 1;
      for (unsigned int i = 0; i < d_rowIdsLocal.size(); ++i)
        {
          const dealii::types::global_dof_index startingLocalDofIndexRow =
            d_rowIdsLocal[i] * blockSize;
          for (unsigned int j = 0; j < d_rowSizes[i]; ++j)
            {
              const dealii::types::global_dof_index
                startingLocalDofIndexColumn =
                  d_columnIdsLocal[count] * blockSize;

              T alpha = d_columnValues[count];
              callaxpy(&blockSize,
                       &alpha,
                       fieldVector.begin() + startingLocalDofIndexRow,
                       &inc,
                       fieldVector.begin() + startingLocalDofIndexColumn,
                       &inc);


              count++;
            }

          //
          // set slave contribution to zero
          //
          std::fill(fieldVector.begin() + startingLocalDofIndexRow,
                    fieldVector.begin() + startingLocalDofIndexRow + blockSize,
                    0.0);
        }
    }

    template <typename T>
    void
    constraintMatrixInfo::distribute_slave_to_master(
      distributedCPUMultiVec<T> &fieldVector,
      const unsigned int         blockSize) const
    {
      unsigned int       count = 0;
      const unsigned int inc   = 1;
      for (unsigned int i = 0; i < d_rowIdsLocal.size(); ++i)
        {
          const dealii::types::global_dof_index startingLocalDofIndexRow =
            d_rowIdsLocal[i] * blockSize;
          for (unsigned int j = 0; j < d_rowSizes[i]; ++j)
            {
              const dealii::types::global_dof_index
                startingLocalDofIndexColumn =
                  d_columnIdsLocal[count] * blockSize;

              T alpha = d_columnValues[count];
              callaxpy(&blockSize,
                       &alpha,
                       fieldVector.data() + startingLocalDofIndexRow,
                       &inc,
                       fieldVector.data() + startingLocalDofIndexColumn,
                       &inc);


              count++;
            }

          //
          // set slave contribution to zero
          //
          std::fill(fieldVector.data() + startingLocalDofIndexRow,
                    fieldVector.data() + startingLocalDofIndexRow + blockSize,
                    0.0);
        }
    }

    template <typename T>
    void
    constraintMatrixInfo::set_zero(distributedCPUVec<T> &fieldVector,
                                   const unsigned int    blockSize) const
    {
      for (unsigned int i = 0; i < d_rowIdsLocal.size(); ++i)
        {
          const dealii::types::global_dof_index startingLocalDofIndexRow =
            d_rowIdsLocal[i] * blockSize;

          // set constrained nodes to zero
          std::fill(fieldVector.begin() + startingLocalDofIndexRow,
                    fieldVector.begin() + startingLocalDofIndexRow + blockSize,
                    0.0);
        }
    }

    template <typename T>
    void
    constraintMatrixInfo::set_zero(distributedCPUMultiVec<T> &fieldVector,
                                   const unsigned int         blockSize) const
    {
      for (unsigned int i = 0; i < d_rowIdsLocal.size(); ++i)
        {
          const dealii::types::global_dof_index startingLocalDofIndexRow =
            d_rowIdsLocal[i] * blockSize;

          // set constrained nodes to zero
          std::fill(fieldVector.data() + startingLocalDofIndexRow,
                    fieldVector.data() + startingLocalDofIndexRow + blockSize,
                    0.0);
        }
    }

    //
    //
    // clear the data variables
    //
    void
    constraintMatrixInfo::clear()
    {
      d_rowIdsGlobal.clear();
      d_rowIdsLocal.clear();
      d_columnIdsLocal.clear();
      d_columnIdsGlobal.clear();
      d_columnValues.clear();
      d_inhomogenities.clear();
      d_rowSizes.clear();
    }


    template void
    constraintMatrixInfo::distribute(
      distributedCPUVec<dataTypes::number> &fieldVector,
      const unsigned int                    blockSize) const;

    template void
    constraintMatrixInfo::distribute_slave_to_master(
      distributedCPUVec<dataTypes::number> &fieldVector,
      const unsigned int                    blockSize) const;

    template void
    constraintMatrixInfo::set_zero(
      distributedCPUVec<dataTypes::number> &fieldVector,
      const unsigned int                    blockSize) const;

    template void
    constraintMatrixInfo::distribute(
      distributedCPUMultiVec<double> &fieldVector,
      const unsigned int              blockSize) const;

    template void
    constraintMatrixInfo::distribute(
      distributedCPUMultiVec<std::complex<double>> &fieldVector,
      const unsigned int                            blockSize) const;

    template void
    constraintMatrixInfo::distribute_slave_to_master(
      distributedCPUMultiVec<dataTypes::number> &fieldVector,
      const unsigned int                         blockSize) const;

    template void
    constraintMatrixInfo::set_zero(
      distributedCPUMultiVec<dataTypes::number> &fieldVector,
      const unsigned int                         blockSize) const;

  } // namespace dftUtils

} // namespace dftfe
