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

#include "scalapackWrapper.h"
#include "scalapack.templates.h"

namespace dftfe
{
  template <typename NumberType>
  ScaLAPACKMatrix<NumberType>::ScaLAPACKMatrix(
    const size_type                                  n_rows_,
    const size_type                                  n_columns_,
    const std::shared_ptr<const dftfe::ProcessGrid> &process_grid,
    const size_type                                  row_block_size_,
    const size_type                                  column_block_size_,
    const dftfe::LAPACKSupport::Property             property_)
    : uplo('L')
    , // for non-hermitian matrices this is not needed
    first_process_row(0)
    , first_process_column(0)
    , submatrix_row(1)
    , submatrix_column(1)
  {
    reinit(n_rows_,
           n_columns_,
           process_grid,
           row_block_size_,
           column_block_size_,
           property_);
  }



  template <typename NumberType>
  ScaLAPACKMatrix<NumberType>::ScaLAPACKMatrix(
    const size_type                                  size,
    const std::shared_ptr<const dftfe::ProcessGrid> &process_grid,
    const size_type                                  block_size,
    const dftfe::LAPACKSupport::Property             property)
    : ScaLAPACKMatrix<NumberType>(size,
                                  size,
                                  process_grid,
                                  block_size,
                                  block_size,
                                  property)
  {}



  template <typename NumberType>
  void
  ScaLAPACKMatrix<NumberType>::reinit(
    const size_type                                  n_rows_,
    const size_type                                  n_columns_,
    const std::shared_ptr<const dftfe::ProcessGrid> &process_grid,
    const size_type                                  row_block_size_,
    const size_type                                  column_block_size_,
    const dftfe::LAPACKSupport::Property             property_)
  {
    Assert(row_block_size_ > 0,
           dealii::ExcMessage("Row block size has to be positive."));
    Assert(column_block_size_ > 0,
           dealii::ExcMessage("Column block size has to be positive."));
    Assert(
      row_block_size_ <= n_rows_,
      dealii::ExcMessage(
        "Row block size can not be greater than the number of rows of the matrix"));
    Assert(
      column_block_size_ <= n_columns_,
      dealii::ExcMessage(
        "Column block size can not be greater than the number of columns of the matrix"));

    state             = dftfe::LAPACKSupport::State::matrix;
    property          = property_;
    grid              = process_grid;
    n_rows            = n_rows_;
    n_columns         = n_columns_;
    row_block_size    = row_block_size_;
    column_block_size = column_block_size_;

    if (grid->mpi_process_is_active)
      {
        // Get local sizes:
        n_local_rows    = numroc_(&n_rows,
                               &row_block_size,
                               &(grid->this_process_row),
                               &first_process_row,
                               &(grid->n_process_rows));
        n_local_columns = numroc_(&n_columns,
                                  &column_block_size,
                                  &(grid->this_process_column),
                                  &first_process_column,
                                  &(grid->n_process_columns));

        // LLD_A = MAX(1,NUMROC(M_A, MB_A, MYROW, RSRC_A, NPROW)), different
        // between processes
        int lda = std::max(1, n_local_rows);

        int info = 0;
        descinit_(descriptor,
                  &n_rows,
                  &n_columns,
                  &row_block_size,
                  &column_block_size,
                  &first_process_row,
                  &first_process_column,
                  &(grid->blacs_context),
                  &lda,
                  &info);
        AssertThrow(info == 0,
                    dftfe::LAPACKSupport::ExcErrorCode("descinit_", info));

        values.clear();
        values.resize(n_local_rows * n_local_columns, NumberType(0.0));
        // this->TransposeTable<NumberType>::reinit(n_local_rows,
        // n_local_columns);
      }
    else
      {
        // set process-local variables to something telling:
        n_local_rows    = -1;
        n_local_columns = -1;
        std::fill(std::begin(descriptor), std::end(descriptor), -1);
      }
  }



  template <typename NumberType>
  void
  ScaLAPACKMatrix<NumberType>::reinit(
    const size_type                                  size,
    const std::shared_ptr<const dftfe::ProcessGrid> &process_grid,
    const size_type                                  block_size,
    const dftfe::LAPACKSupport::Property             property)
  {
    reinit(size, size, process_grid, block_size, block_size, property);
  }

  template <typename NumberType>
  void
  ScaLAPACKMatrix<NumberType>::set_property(
    const dftfe::LAPACKSupport::Property property_)
  {
    property = property_;
  }



  template <typename NumberType>
  dftfe::LAPACKSupport::Property
  ScaLAPACKMatrix<NumberType>::get_property() const
  {
    return property;
  }


  template <typename NumberType>
  dftfe::LAPACKSupport::State
  ScaLAPACKMatrix<NumberType>::get_state() const
  {
    return state;
  }


  template <typename NumberType>
  unsigned int
  ScaLAPACKMatrix<NumberType>::global_row(const unsigned int loc_row) const
  {
    Assert(n_local_rows >= 0 &&
             loc_row < static_cast<unsigned int>(n_local_rows),
           dealii::ExcIndexRange(loc_row, 0, n_local_rows));
    const int i = loc_row + 1;
    return indxl2g_(&i,
                    &row_block_size,
                    &(grid->this_process_row),
                    &first_process_row,
                    &(grid->n_process_rows)) -
           1;
  }



  template <typename NumberType>
  unsigned int
  ScaLAPACKMatrix<NumberType>::global_column(
    const unsigned int loc_column) const
  {
    Assert(n_local_columns >= 0 &&
             loc_column < static_cast<unsigned int>(n_local_columns),
           dealii::ExcIndexRange(loc_column, 0, n_local_columns));
    const int j = loc_column + 1;
    return indxl2g_(&j,
                    &column_block_size,
                    &(grid->this_process_column),
                    &first_process_column,
                    &(grid->n_process_columns)) -
           1;
  }

  template <typename NumberType>
  void
  ScaLAPACKMatrix<NumberType>::conjugate()
  {
    if (std::is_same<NumberType, std::complex<double>>::value)
      {
        if (this->grid->mpi_process_is_active)
          {
            NumberType *A_loc =
              (this->values.size() > 0) ? this->values.data() : nullptr;
            const int totalSize = n_rows * n_columns;
            const int incx      = 1;
            pplacgv(&totalSize,
                    A_loc,
                    &submatrix_row,
                    &submatrix_column,
                    descriptor,
                    &incx);
          }
      }
  }


  template <typename NumberType>
  void
  ScaLAPACKMatrix<NumberType>::copy_to(ScaLAPACKMatrix<NumberType> &dest) const
  {
    Assert(n_rows == dest.n_rows, ExcDimensionMismatch(n_rows, dest.n_rows));
    Assert(n_columns == dest.n_columns,
           ExcDimensionMismatch(n_columns, dest.n_columns));

    if (this->grid->mpi_process_is_active)
      AssertThrow(
        this->descriptor[0] == 1,
        dealii::ExcMessage(
          "Copying of ScaLAPACK matrices only implemented for dense matrices"));
    if (dest.grid->mpi_process_is_active)
      AssertThrow(
        dest.descriptor[0] == 1,
        dealii::ExcMessage(
          "Copying of ScaLAPACK matrices only implemented for dense matrices"));

    /*
     * just in case of different process grids or block-cyclic distributions
     * inter-process communication is necessary
     * if distributed matrices have the same process grid and block sizes, local
     * copying is enough
     */
    if ((this->grid != dest.grid) || (row_block_size != dest.row_block_size) ||
        (column_block_size != dest.column_block_size))
      {
        /*
         * get the MPI communicator, which is the union of the source and
         * destination MPI communicator
         */
        int       ierr = 0;
        MPI_Group group_source, group_dest, group_union;
        ierr = MPI_Comm_group(this->grid->mpi_communicator, &group_source);
        AssertThrowMPI(ierr);
        ierr = MPI_Comm_group(dest.grid->mpi_communicator, &group_dest);
        AssertThrowMPI(ierr);
        ierr = MPI_Group_union(group_source, group_dest, &group_union);
        AssertThrowMPI(ierr);
        MPI_Comm mpi_communicator_union;

        // to create a communicator representing the union of the source
        // and destination MPI communicator we need a communicator that
        // is guaranteed to contain all desired processes -- i.e.,
        // MPI_COMM_WORLD. on the other hand, as documented in the MPI
        // standard, MPI_Comm_create_group is not collective on all
        // processes in the first argument, but instead is collective on
        // only those processes listed in the group. in other words,
        // there is really no harm in passing MPI_COMM_WORLD as the
        // first argument, even if the program we are currently running
        // and that is calling this function only works on a subset of
        // processes. the same holds for the wrapper/fallback we are using here.

        const int mpi_tag =
          dealii::Utilities::MPI::internal::Tags::scalapack_copy_to2;
        ierr = dealii::Utilities::MPI::create_group(MPI_COMM_WORLD,
                                                    group_union,
                                                    mpi_tag,
                                                    &mpi_communicator_union);
        AssertThrowMPI(ierr);

        /*
         * The routine pgemr2d requires a BLACS context resembling at least the
         * union of process grids described by the BLACS contexts of matrix A
         * and
         * B
         */
        int union_blacs_context = Csys2blacs_handle(mpi_communicator_union);
        const char *order       = "Col";
        int         union_n_process_rows =
          dealii::Utilities::MPI::n_mpi_processes(mpi_communicator_union);
        int union_n_process_columns = 1;
        Cblacs_gridinit(&union_blacs_context,
                        order,
                        union_n_process_rows,
                        union_n_process_columns);

        const NumberType *loc_vals_source = nullptr;
        NumberType *      loc_vals_dest   = nullptr;

        if (this->grid->mpi_process_is_active && (this->values.size() > 0))
          {
            AssertThrow(this->values.size() > 0,
                        dealii::ExcMessage(
                          "source: process is active but local matrix empty"));
            loc_vals_source = this->values.data();
          }
        if (dest.grid->mpi_process_is_active && (dest.values.size() > 0))
          {
            AssertThrow(
              dest.values.size() > 0,
              dealii::ExcMessage(
                "destination: process is active but local matrix empty"));
            loc_vals_dest = dest.values.data();
          }
        pgemr2d(&n_rows,
                &n_columns,
                loc_vals_source,
                &submatrix_row,
                &submatrix_column,
                descriptor,
                loc_vals_dest,
                &dest.submatrix_row,
                &dest.submatrix_column,
                dest.descriptor,
                &union_blacs_context);

        Cblacs_gridexit(union_blacs_context);

        if (mpi_communicator_union != MPI_COMM_NULL)
          {
            ierr = MPI_Comm_free(&mpi_communicator_union);
            AssertThrowMPI(ierr);
          }
        ierr = MPI_Group_free(&group_source);
        AssertThrowMPI(ierr);
        ierr = MPI_Group_free(&group_dest);
        AssertThrowMPI(ierr);
        ierr = MPI_Group_free(&group_union);
        AssertThrowMPI(ierr);
      }
    else
      // process is active in the process grid
      if (this->grid->mpi_process_is_active)
      dest.values = this->values;

    dest.state    = state;
    dest.property = property;
  }

  template <typename NumberType>
  void
  ScaLAPACKMatrix<NumberType>::add(const ScaLAPACKMatrix<NumberType> &B,
                                   const NumberType                   alpha,
                                   const NumberType                   beta,
                                   const bool transpose_B)
  {
    if (transpose_B)
      {
        Assert(n_rows == B.n_columns,
               ExcDimensionMismatch(n_rows, B.n_columns));
        Assert(n_columns == B.n_rows,
               ExcDimensionMismatch(n_columns, B.n_rows));
        Assert(column_block_size == B.row_block_size,
               ExcDimensionMismatch(column_block_size, B.row_block_size));
        Assert(row_block_size == B.column_block_size,
               ExcDimensionMismatch(row_block_size, B.column_block_size));
      }
    else
      {
        Assert(n_rows == B.n_rows, ExcDimensionMismatch(n_rows, B.n_rows));
        Assert(n_columns == B.n_columns,
               ExcDimensionMismatch(n_columns, B.n_columns));
        Assert(column_block_size == B.column_block_size,
               ExcDimensionMismatch(column_block_size, B.column_block_size));
        Assert(row_block_size == B.row_block_size,
               ExcDimensionMismatch(row_block_size, B.row_block_size));
      }
    Assert(this->grid == B.grid,
           dealii::ExcMessage(
             "The matrices A and B need to have the same process grid"));

    if (this->grid->mpi_process_is_active)
      {
        char        trans_b = transpose_B ? 'T' : 'N';
        NumberType *A_loc =
          (this->values.size() > 0) ? this->values.data() : nullptr;
        const NumberType *B_loc =
          (B.values.size() > 0) ? B.values.data() : nullptr;

        pgeadd(&trans_b,
               &n_rows,
               &n_columns,
               &beta,
               B_loc,
               &B.submatrix_row,
               &B.submatrix_column,
               B.descriptor,
               &alpha,
               A_loc,
               &submatrix_row,
               &submatrix_column,
               descriptor);
      }
    state = dftfe::LAPACKSupport::matrix;
  }

  template <typename NumberType>
  void
  ScaLAPACKMatrix<NumberType>::zadd(const ScaLAPACKMatrix<NumberType> &B,
                                    const NumberType                   alpha,
                                    const NumberType                   beta,
                                    const bool conjugate_transpose_B)
  {
    if (conjugate_transpose_B)
      {
        Assert(n_rows == B.n_columns,
               ExcDimensionMismatch(n_rows, B.n_columns));
        Assert(n_columns == B.n_rows,
               ExcDimensionMismatch(n_columns, B.n_rows));
        Assert(column_block_size == B.row_block_size,
               ExcDimensionMismatch(column_block_size, B.row_block_size));
        Assert(row_block_size == B.column_block_size,
               ExcDimensionMismatch(row_block_size, B.column_block_size));
      }
    else
      {
        Assert(n_rows == B.n_rows, ExcDimensionMismatch(n_rows, B.n_rows));
        Assert(n_columns == B.n_columns,
               ExcDimensionMismatch(n_columns, B.n_columns));
        Assert(column_block_size == B.column_block_size,
               ExcDimensionMismatch(column_block_size, B.column_block_size));
        Assert(row_block_size == B.row_block_size,
               ExcDimensionMismatch(row_block_size, B.row_block_size));
      }
    Assert(this->grid == B.grid,
           dealii::ExcMessage(
             "The matrices A and B need to have the same process grid"));

    if (this->grid->mpi_process_is_active)
      {
        char trans_b =
          conjugate_transpose_B ?
            (std::is_same<NumberType, std::complex<double>>::value ? 'C' :
                                                                     'T') :
            'N';
        NumberType *A_loc =
          (this->values.size() > 0) ? this->values.data() : nullptr;
        const NumberType *B_loc =
          (B.values.size() > 0) ? B.values.data() : nullptr;

        pgeadd(&trans_b,
               &n_rows,
               &n_columns,
               &beta,
               B_loc,
               &B.submatrix_row,
               &B.submatrix_column,
               B.descriptor,
               &alpha,
               A_loc,
               &submatrix_row,
               &submatrix_column,
               descriptor);
      }
    state = dftfe::LAPACKSupport::matrix;
  }

  template <typename NumberType>
  void
  ScaLAPACKMatrix<NumberType>::copy_transposed(
    const ScaLAPACKMatrix<NumberType> &B)
  {
    add(B, 0, 1, true);
  }

  template <typename NumberType>
  void
  ScaLAPACKMatrix<NumberType>::copy_conjugate_transposed(
    const ScaLAPACKMatrix<NumberType> &B)
  {
    zadd(B, 0, 1, true);
  }

  template <typename NumberType>
  void
  ScaLAPACKMatrix<NumberType>::mult(const NumberType                   b,
                                    const ScaLAPACKMatrix<NumberType> &B,
                                    const NumberType                   c,
                                    ScaLAPACKMatrix<NumberType> &      C,
                                    const bool transpose_A,
                                    const bool transpose_B) const
  {
    Assert(this->grid == B.grid,
           dealii::ExcMessage(
             "The matrices A and B need to have the same process grid"));
    Assert(C.grid == B.grid,
           dealii::ExcMessage(
             "The matrices B and C need to have the same process grid"));

    // see for further info:
    // https://www.ibm.com/support/knowledgecenter/SSNR5K_4.2.0/com.ibm.cluster.pessl.v4r2.pssl100.doc/am6gr_lgemm.htm
    if (!transpose_A && !transpose_B)
      {
        Assert(this->n_columns == B.n_rows,
               ExcDimensionMismatch(this->n_columns, B.n_rows));
        Assert(this->n_rows == C.n_rows,
               ExcDimensionMismatch(this->n_rows, C.n_rows));
        Assert(B.n_columns == C.n_columns,
               ExcDimensionMismatch(B.n_columns, C.n_columns));
        Assert(this->row_block_size == C.row_block_size,
               ExcDimensionMismatch(this->row_block_size, C.row_block_size));
        Assert(this->column_block_size == B.row_block_size,
               ExcDimensionMismatch(this->column_block_size, B.row_block_size));
        Assert(B.column_block_size == C.column_block_size,
               ExcDimensionMismatch(B.column_block_size, C.column_block_size));
      }
    else if (transpose_A && !transpose_B)
      {
        Assert(this->n_rows == B.n_rows,
               ExcDimensionMismatch(this->n_rows, B.n_rows));
        Assert(this->n_columns == C.n_rows,
               ExcDimensionMismatch(this->n_columns, C.n_rows));
        Assert(B.n_columns == C.n_columns,
               ExcDimensionMismatch(B.n_columns, C.n_columns));
        Assert(this->column_block_size == C.row_block_size,
               ExcDimensionMismatch(this->column_block_size, C.row_block_size));
        Assert(this->row_block_size == B.row_block_size,
               ExcDimensionMismatch(this->row_block_size, B.row_block_size));
        Assert(B.column_block_size == C.column_block_size,
               ExcDimensionMismatch(B.column_block_size, C.column_block_size));
      }
    else if (!transpose_A && transpose_B)
      {
        Assert(this->n_columns == B.n_columns,
               ExcDimensionMismatch(this->n_columns, B.n_columns));
        Assert(this->n_rows == C.n_rows,
               ExcDimensionMismatch(this->n_rows, C.n_rows));
        Assert(B.n_rows == C.n_columns,
               ExcDimensionMismatch(B.n_rows, C.n_columns));
        Assert(this->row_block_size == C.row_block_size,
               ExcDimensionMismatch(this->row_block_size, C.row_block_size));
        Assert(this->column_block_size == B.column_block_size,
               ExcDimensionMismatch(this->column_block_size,
                                    B.column_block_size));
        Assert(B.row_block_size == C.column_block_size,
               ExcDimensionMismatch(B.row_block_size, C.column_block_size));
      }
    else // if (transpose_A && transpose_B)
      {
        Assert(this->n_rows == B.n_columns,
               ExcDimensionMismatch(this->n_rows, B.n_columns));
        Assert(this->n_columns == C.n_rows,
               ExcDimensionMismatch(this->n_columns, C.n_rows));
        Assert(B.n_rows == C.n_columns,
               ExcDimensionMismatch(B.n_rows, C.n_columns));
        Assert(this->column_block_size == C.row_block_size,
               ExcDimensionMismatch(this->row_block_size, C.row_block_size));
        Assert(this->row_block_size == B.column_block_size,
               ExcDimensionMismatch(this->column_block_size, B.row_block_size));
        Assert(B.row_block_size == C.column_block_size,
               ExcDimensionMismatch(B.column_block_size, C.column_block_size));
      }

    if (this->grid->mpi_process_is_active)
      {
        char trans_a = transpose_A ? 'T' : 'N';
        char trans_b = transpose_B ? 'T' : 'N';

        const NumberType *A_loc =
          (this->values.size() > 0) ? this->values.data() : nullptr;
        const NumberType *B_loc =
          (B.values.size() > 0) ? B.values.data() : nullptr;
        NumberType *C_loc = (C.values.size() > 0) ? C.values.data() : nullptr;
        int         m     = C.n_rows;
        int         n     = C.n_columns;
        int         k     = transpose_A ? this->n_rows : this->n_columns;

        pgemm(&trans_a,
              &trans_b,
              &m,
              &n,
              &k,
              &b,
              A_loc,
              &(this->submatrix_row),
              &(this->submatrix_column),
              this->descriptor,
              B_loc,
              &B.submatrix_row,
              &B.submatrix_column,
              B.descriptor,
              &c,
              C_loc,
              &C.submatrix_row,
              &C.submatrix_column,
              C.descriptor);
      }
    C.state = dftfe::LAPACKSupport::matrix;
  }

  template <typename NumberType>
  void
  ScaLAPACKMatrix<NumberType>::zmult(const NumberType                   b,
                                     const ScaLAPACKMatrix<NumberType> &B,
                                     const NumberType                   c,
                                     ScaLAPACKMatrix<NumberType> &      C,
                                     const bool conjugate_transpose_A,
                                     const bool conjugate_transpose_B) const
  {
    Assert(this->grid == B.grid,
           dealii::ExcMessage(
             "The matrices A and B need to have the same process grid"));
    Assert(C.grid == B.grid,
           dealii::ExcMessage(
             "The matrices B and C need to have the same process grid"));

    // see for further info:
    // https://www.ibm.com/support/knowledgecenter/SSNR5K_4.2.0/com.ibm.cluster.pessl.v4r2.pssl100.doc/am6gr_lgemm.htm
    if (!conjugate_transpose_A && !conjugate_transpose_B)
      {
        Assert(this->n_columns == B.n_rows,
               ExcDimensionMismatch(this->n_columns, B.n_rows));
        Assert(this->n_rows == C.n_rows,
               ExcDimensionMismatch(this->n_rows, C.n_rows));
        Assert(B.n_columns == C.n_columns,
               ExcDimensionMismatch(B.n_columns, C.n_columns));
        Assert(this->row_block_size == C.row_block_size,
               ExcDimensionMismatch(this->row_block_size, C.row_block_size));
        Assert(this->column_block_size == B.row_block_size,
               ExcDimensionMismatch(this->column_block_size, B.row_block_size));
        Assert(B.column_block_size == C.column_block_size,
               ExcDimensionMismatch(B.column_block_size, C.column_block_size));
      }
    else if (conjugate_transpose_A && !conjugate_transpose_B)
      {
        Assert(this->n_rows == B.n_rows,
               ExcDimensionMismatch(this->n_rows, B.n_rows));
        Assert(this->n_columns == C.n_rows,
               ExcDimensionMismatch(this->n_columns, C.n_rows));
        Assert(B.n_columns == C.n_columns,
               ExcDimensionMismatch(B.n_columns, C.n_columns));
        Assert(this->column_block_size == C.row_block_size,
               ExcDimensionMismatch(this->column_block_size, C.row_block_size));
        Assert(this->row_block_size == B.row_block_size,
               ExcDimensionMismatch(this->row_block_size, B.row_block_size));
        Assert(B.column_block_size == C.column_block_size,
               ExcDimensionMismatch(B.column_block_size, C.column_block_size));
      }
    else if (!conjugate_transpose_A && conjugate_transpose_B)
      {
        Assert(this->n_columns == B.n_columns,
               ExcDimensionMismatch(this->n_columns, B.n_columns));
        Assert(this->n_rows == C.n_rows,
               ExcDimensionMismatch(this->n_rows, C.n_rows));
        Assert(B.n_rows == C.n_columns,
               ExcDimensionMismatch(B.n_rows, C.n_columns));
        Assert(this->row_block_size == C.row_block_size,
               ExcDimensionMismatch(this->row_block_size, C.row_block_size));
        Assert(this->column_block_size == B.column_block_size,
               ExcDimensionMismatch(this->column_block_size,
                                    B.column_block_size));
        Assert(B.row_block_size == C.column_block_size,
               ExcDimensionMismatch(B.row_block_size, C.column_block_size));
      }
    else // if (transpose_A && transpose_B)
      {
        Assert(this->n_rows == B.n_columns,
               ExcDimensionMismatch(this->n_rows, B.n_columns));
        Assert(this->n_columns == C.n_rows,
               ExcDimensionMismatch(this->n_columns, C.n_rows));
        Assert(B.n_rows == C.n_columns,
               ExcDimensionMismatch(B.n_rows, C.n_columns));
        Assert(this->column_block_size == C.row_block_size,
               ExcDimensionMismatch(this->row_block_size, C.row_block_size));
        Assert(this->row_block_size == B.column_block_size,
               ExcDimensionMismatch(this->column_block_size, B.row_block_size));
        Assert(B.row_block_size == C.column_block_size,
               ExcDimensionMismatch(B.column_block_size, C.column_block_size));
      }

    if (this->grid->mpi_process_is_active)
      {
        char trans_a =
          conjugate_transpose_A ?
            (std::is_same<NumberType, std::complex<double>>::value ? 'C' :
                                                                     'T') :
            'N';
        char trans_b =
          conjugate_transpose_B ?
            (std::is_same<NumberType, std::complex<double>>::value ? 'C' :
                                                                     'T') :
            'N';

        const NumberType *A_loc =
          (this->values.size() > 0) ? this->values.data() : nullptr;
        const NumberType *B_loc =
          (B.values.size() > 0) ? B.values.data() : nullptr;
        NumberType *C_loc = (C.values.size() > 0) ? C.values.data() : nullptr;
        int         m     = C.n_rows;
        int         n     = C.n_columns;
        int         k = conjugate_transpose_A ? this->n_rows : this->n_columns;

        pgemm(&trans_a,
              &trans_b,
              &m,
              &n,
              &k,
              &b,
              A_loc,
              &(this->submatrix_row),
              &(this->submatrix_column),
              this->descriptor,
              B_loc,
              &B.submatrix_row,
              &B.submatrix_column,
              B.descriptor,
              &c,
              C_loc,
              &C.submatrix_row,
              &C.submatrix_column,
              C.descriptor);
      }
    C.state = dftfe::LAPACKSupport::matrix;
  }

  template <typename NumberType>
  void
  ScaLAPACKMatrix<NumberType>::mmult(ScaLAPACKMatrix<NumberType> &      C,
                                     const ScaLAPACKMatrix<NumberType> &B,
                                     const bool adding) const
  {
    if (adding)
      mult(1., B, 1., C, false, false);
    else
      mult(1., B, 0, C, false, false);
  }



  template <typename NumberType>
  void
  ScaLAPACKMatrix<NumberType>::Tmmult(ScaLAPACKMatrix<NumberType> &      C,
                                      const ScaLAPACKMatrix<NumberType> &B,
                                      const bool adding) const
  {
    if (adding)
      mult(1., B, 1., C, true, false);
    else
      mult(1., B, 0, C, true, false);
  }



  template <typename NumberType>
  void
  ScaLAPACKMatrix<NumberType>::mTmult(ScaLAPACKMatrix<NumberType> &      C,
                                      const ScaLAPACKMatrix<NumberType> &B,
                                      const bool adding) const
  {
    if (adding)
      mult(1., B, 1., C, false, true);
    else
      mult(1., B, 0, C, false, true);
  }



  template <typename NumberType>
  void
  ScaLAPACKMatrix<NumberType>::TmTmult(ScaLAPACKMatrix<NumberType> &      C,
                                       const ScaLAPACKMatrix<NumberType> &B,
                                       const bool adding) const
  {
    if (adding)
      mult(1., B, 1., C, true, true);
    else
      mult(1., B, 0, C, true, true);
  }


  template <typename NumberType>
  void
  ScaLAPACKMatrix<NumberType>::zmmult(ScaLAPACKMatrix<NumberType> &      C,
                                      const ScaLAPACKMatrix<NumberType> &B,
                                      const bool adding) const
  {
    if (adding)
      zmult(1., B, 1., C, false, false);
    else
      zmult(1., B, 0, C, false, false);
  }



  template <typename NumberType>
  void
  ScaLAPACKMatrix<NumberType>::zCmmult(ScaLAPACKMatrix<NumberType> &      C,
                                       const ScaLAPACKMatrix<NumberType> &B,
                                       const bool adding) const
  {
    if (adding)
      zmult(1., B, 1., C, true, false);
    else
      zmult(1., B, 0, C, true, false);
  }



  template <typename NumberType>
  void
  ScaLAPACKMatrix<NumberType>::zmCmult(ScaLAPACKMatrix<NumberType> &      C,
                                       const ScaLAPACKMatrix<NumberType> &B,
                                       const bool adding) const
  {
    if (adding)
      zmult(1., B, 1., C, false, true);
    else
      zmult(1., B, 0, C, false, true);
  }



  template <typename NumberType>
  void
  ScaLAPACKMatrix<NumberType>::zCmCmult(ScaLAPACKMatrix<NumberType> &      C,
                                        const ScaLAPACKMatrix<NumberType> &B,
                                        const bool adding) const
  {
    if (adding)
      zmult(1., B, 1., C, true, true);
    else
      zmult(1., B, 0, C, true, true);
  }

  template <typename NumberType>
  void
  ScaLAPACKMatrix<NumberType>::compute_cholesky_factorization()
  {
    Assert(
      n_columns == n_rows && property == LAPACKSupport::Property::hermitian,
      dealii::ExcMessage(
        "Cholesky factorization can be applied to hermitian matrices only."));
    Assert(state == LAPACKSupport::matrix,
           dealii::ExcMessage(
             "Matrix has to be in Matrix state before calling this function."));

    if (grid->mpi_process_is_active)
      {
        int         info  = 0;
        NumberType *A_loc = this->values.data();
        // pdpotrf_(&uplo,&n_columns,A_loc,&submatrix_row,&submatrix_column,descriptor,&info);
        ppotrf(&uplo,
               &n_columns,
               A_loc,
               &submatrix_row,
               &submatrix_column,
               descriptor,
               &info);
        AssertThrow(info == 0,
                    dftfe::LAPACKSupport::ExcErrorCode("ppotrf", info));
      }
    state    = dftfe::LAPACKSupport::cholesky;
    property = (uplo == 'L' ? dftfe::LAPACKSupport::lower_triangular :
                              dftfe::LAPACKSupport::upper_triangular);
  }

  template <typename NumberType>
  void
  ScaLAPACKMatrix<NumberType>::compute_lu_factorization()
  {
    Assert(state == LAPACKSupport::matrix,
           dealii::ExcMessage(
             "Matrix has to be in Matrix state before calling this function."));

    if (grid->mpi_process_is_active)
      {
        int         info  = 0;
        NumberType *A_loc = this->values.data();

        const int iarow = indxg2p_(&submatrix_row,
                                   &row_block_size,
                                   &(grid->this_process_row),
                                   &first_process_row,
                                   &(grid->n_process_rows));
        const int mp    = numroc_(&n_rows,
                               &row_block_size,
                               &(grid->this_process_row),
                               &iarow,
                               &(grid->n_process_rows));
        ipiv.resize(mp + row_block_size);

        pgetrf(&n_rows,
               &n_columns,
               A_loc,
               &submatrix_row,
               &submatrix_column,
               descriptor,
               ipiv.data(),
               &info);
        AssertThrow(info == 0,
                    dftfe::LAPACKSupport::ExcErrorCode("pgetrf", info));
      }
    state    = dftfe::LAPACKSupport::State::lu;
    property = dftfe::LAPACKSupport::Property::general;
  }

  template <typename NumberType>
  void
  ScaLAPACKMatrix<NumberType>::invert()
  {
    // Check whether matrix is hermitian and save flag.
    // If a Cholesky factorization has been applied previously,
    // the original matrix was hermitian.
    const bool is_hermitian = (property == dftfe::LAPACKSupport::hermitian ||
                               state == dftfe::LAPACKSupport::State::cholesky);

    // Check whether matrix is triangular and is in an unfactorized state.
    const bool is_triangular =
      (property == dftfe::LAPACKSupport::upper_triangular ||
       property == dftfe::LAPACKSupport::lower_triangular) &&
      (state == dftfe::LAPACKSupport::State::matrix ||
       state == dftfe::LAPACKSupport::State::inverse_matrix);

    if (is_triangular)
      {
        if (grid->mpi_process_is_active)
          {
            const char uploTriangular =
              property == dftfe::LAPACKSupport::upper_triangular ? 'U' : 'L';
            const char  diag  = 'N';
            int         info  = 0;
            NumberType *A_loc = this->values.data();
            ptrtri(&uploTriangular,
                   &diag,
                   &n_columns,
                   A_loc,
                   &submatrix_row,
                   &submatrix_column,
                   descriptor,
                   &info);
            AssertThrow(info == 0,
                        dftfe::LAPACKSupport::ExcErrorCode("ptrtri", info));
            // The inversion is stored in the same part as the triangular
            // matrix, so we don't need to re-set the property here.
          }
      }
    else
      {
        // Matrix is neither in Cholesky nor LU state.
        // Compute the required factorizations based on the property of the
        // matrix.
        if (!(state == dftfe::LAPACKSupport::State::lu ||
              state == dftfe::LAPACKSupport::State::cholesky))
          {
            if (is_hermitian)
              compute_cholesky_factorization();
            else
              compute_lu_factorization();
          }
        if (grid->mpi_process_is_active)
          {
            int         info  = 0;
            NumberType *A_loc = this->values.data();

            if (is_hermitian)
              {
                ppotri(&uplo,
                       &n_columns,
                       A_loc,
                       &submatrix_row,
                       &submatrix_column,
                       descriptor,
                       &info);
                AssertThrow(info == 0,
                            dftfe::LAPACKSupport::ExcErrorCode("ppotri", info));
                property = dftfe::LAPACKSupport::Property::hermitian;
              }
            else
              {
                int lwork = -1, liwork = -1;
                work.resize(1);
                iwork.resize(1);

                pgetri(&n_columns,
                       A_loc,
                       &submatrix_row,
                       &submatrix_column,
                       descriptor,
                       ipiv.data(),
                       work.data(),
                       &lwork,
                       iwork.data(),
                       &liwork,
                       &info);

                AssertThrow(info == 0,
                            dftfe::LAPACKSupport::ExcErrorCode("pgetri", info));
                lwork  = lworkFromWork(work);
                liwork = iwork[0];
                work.resize(lwork);
                iwork.resize(liwork);

                pgetri(&n_columns,
                       A_loc,
                       &submatrix_row,
                       &submatrix_column,
                       descriptor,
                       ipiv.data(),
                       work.data(),
                       &lwork,
                       iwork.data(),
                       &liwork,
                       &info);

                AssertThrow(info == 0,
                            dftfe::LAPACKSupport::ExcErrorCode("pgetri", info));
              }
          }
      }
    state = dftfe::LAPACKSupport::State::inverse_matrix;
  }

  template <typename NumberType>
  std::vector<double>
  ScaLAPACKMatrix<NumberType>::eigenpairs_hermitian_by_index(
    const std::pair<unsigned int, unsigned int> &index_limits,
    const bool                                   compute_eigenvectors)
  {
    // check validity of index limits
    AssertIndexRange(index_limits.first, n_rows);
    AssertIndexRange(index_limits.second, n_rows);

    std::pair<unsigned int, unsigned int> idx =
      std::make_pair(std::min(index_limits.first, index_limits.second),
                     std::max(index_limits.first, index_limits.second));

    // compute all eigenvalues/eigenvectors
    if (idx.first == 0 && idx.second == static_cast<unsigned int>(n_rows - 1))
      return eigenpairs_hermitian(compute_eigenvectors);
    else
      return eigenpairs_hermitian(compute_eigenvectors, idx);
  }


  template <typename NumberType>
  std::vector<double>
  ScaLAPACKMatrix<NumberType>::eigenpairs_hermitian(
    const bool                                   compute_eigenvectors,
    const std::pair<unsigned int, unsigned int> &eigenvalue_idx,
    const std::pair<double, double> &            eigenvalue_limits)
  {
    Assert(state == dftfe::LAPACKSupport::matrix,
           dealii::ExcMessage(
             "Matrix has to be in Matrix state before calling this function."));
    Assert(property == dftfe::LAPACKSupport::hermitian,
           dealii::ExcMessage(
             "Matrix has to be hermitian for this operation."));

    std::lock_guard<std::mutex> lock(mutex);

    const bool use_values = (std::isnan(eigenvalue_limits.first) ||
                             std::isnan(eigenvalue_limits.second)) ?
                              false :
                              true;
    const bool use_indices =
      ((eigenvalue_idx.first == dealii::numbers::invalid_unsigned_int) ||
       (eigenvalue_idx.second == dealii::numbers::invalid_unsigned_int)) ?
        false :
        true;

    Assert(
      !(use_values && use_indices),
      dealii::ExcMessage(
        "Prescribing both the index and value range for the eigenvalues is ambiguous"));

    // if computation of eigenvectors is not required use a sufficiently small
    // distributed matrix
    std::unique_ptr<ScaLAPACKMatrix<NumberType>> eigenvectors =
      compute_eigenvectors ?
        std::make_unique<ScaLAPACKMatrix<NumberType>>(n_rows,
                                                      grid,
                                                      row_block_size) :
        std::make_unique<ScaLAPACKMatrix<NumberType>>(
          grid->n_process_rows, grid->n_process_columns, grid, 1, 1);

    eigenvectors->property = property;
    // number of eigenvalues to be returned from psyevx; upon successful exit ev
    // contains the m seclected eigenvalues in ascending order set to all
    // eigenvaleus in case we will be using psyev.
    int                 m = n_rows;
    std::vector<double> ev(n_rows);

    if (grid->mpi_process_is_active)
      {
        int info = 0;
        /*
         * for jobz==N only eigenvalues are computed, for jobz='V' also the
         * eigenvectors of the matrix are computed
         */
        char jobz  = compute_eigenvectors ? 'V' : 'N';
        char range = 'A';
        // default value is to compute all eigenvalues and optionally
        // eigenvectors
        bool   all_eigenpairs = true;
        double vl             = 0.0;
        double vu             = 0.0;
        int    il = 1, iu = 1;
        // number of eigenvectors to be returned;
        // upon successful exit the first m=nz columns contain the selected
        // eigenvectors (only if jobz=='V')
        int    nz     = 0;
        double abstol = 0.0;

        // orfac decides which eigenvectors should be reorthogonalized
        // see
        // http://www.netlib.org/scalapack/explore-html/df/d1a/pdsyevx_8f_source.html
        // for explanation to keeps simple no reorthogonalized will be done by
        // setting orfac to 0
        double orfac = 0;
        // contains the indices of eigenvectors that failed to converge
        std::vector<int> ifail;
        // This array contains indices of eigenvectors corresponding to
        // a cluster of eigenvalues that could not be reorthogonalized
        // due to insufficient workspace
        // see
        // http://www.netlib.org/scalapack/explore-html/df/d1a/pdsyevx_8f_source.html
        // for explanation
        std::vector<int> iclustr;
        // This array contains the gap between eigenvalues whose
        // eigenvectors could not be reorthogonalized.
        // see
        // http://www.netlib.org/scalapack/explore-html/df/d1a/pdsyevx_8f_source.html
        // for explanation
        std::vector<double> gap(n_local_rows * n_local_columns);

        // index range for eigenvalues is not specified
        if (!use_indices)
          {
            // interval for eigenvalues is not specified and consequently all
            // eigenvalues/eigenpairs will be computed
            if (!use_values)
              {
                range          = 'A';
                all_eigenpairs = true;
              }
            else
              {
                range          = 'V';
                all_eigenpairs = false;
                vl =
                  std::min(eigenvalue_limits.first, eigenvalue_limits.second);
                vu =
                  std::max(eigenvalue_limits.first, eigenvalue_limits.second);
              }
          }
        else
          {
            range          = 'I';
            all_eigenpairs = false;
            // as Fortran starts counting/indexing from 1 unlike C/C++, where it
            // starts from 0
            il = std::min(eigenvalue_idx.first, eigenvalue_idx.second) + 1;
            iu = std::max(eigenvalue_idx.first, eigenvalue_idx.second) + 1;
          }
        NumberType *A_loc = this->values.data();
        /*
         * by setting lwork to -1 a workspace query for optimal length of work
         * is performed
         */
        int         lwork  = -1;
        int         liwork = -1;
        NumberType *eigenvectors_loc =
          (compute_eigenvectors ? eigenvectors->values.data() : nullptr);
        work.resize(1);
        iwork.resize(1);

        if (all_eigenpairs)
          {
            psyev(&jobz,
                  &uplo,
                  &n_rows,
                  A_loc,
                  &submatrix_row,
                  &submatrix_column,
                  descriptor,
                  ev.data(),
                  eigenvectors_loc,
                  &eigenvectors->submatrix_row,
                  &eigenvectors->submatrix_column,
                  eigenvectors->descriptor,
                  work.data(),
                  &lwork,
                  &info);
            AssertThrow(info == 0,
                        dftfe::LAPACKSupport::ExcErrorCode("psyev", info));
          }
        else
          {
            char cmach = compute_eigenvectors ? 'U' : 'S';
            plamch(&(this->grid->blacs_context), &cmach, abstol);
            abstol *= 2;
            ifail.resize(n_rows);
            iclustr.resize(2 * grid->n_process_rows * grid->n_process_columns);
            gap.resize(grid->n_process_rows * grid->n_process_columns);

            psyevx(&jobz,
                   &range,
                   &uplo,
                   &n_rows,
                   A_loc,
                   &submatrix_row,
                   &submatrix_column,
                   descriptor,
                   &vl,
                   &vu,
                   &il,
                   &iu,
                   &abstol,
                   &m,
                   &nz,
                   ev.data(),
                   &orfac,
                   eigenvectors_loc,
                   &eigenvectors->submatrix_row,
                   &eigenvectors->submatrix_column,
                   eigenvectors->descriptor,
                   work.data(),
                   &lwork,
                   iwork.data(),
                   &liwork,
                   ifail.data(),
                   iclustr.data(),
                   gap.data(),
                   &info);
            AssertThrow(info == 0,
                        dftfe::LAPACKSupport::ExcErrorCode("psyevx", info));
          }
        lwork = lworkFromWork(work);
        work.resize(lwork);

        if (all_eigenpairs)
          {
            psyev(&jobz,
                  &uplo,
                  &n_rows,
                  A_loc,
                  &submatrix_row,
                  &submatrix_column,
                  descriptor,
                  ev.data(),
                  eigenvectors_loc,
                  &eigenvectors->submatrix_row,
                  &eigenvectors->submatrix_column,
                  eigenvectors->descriptor,
                  work.data(),
                  &lwork,
                  &info);

            AssertThrow(info == 0,
                        dftfe::LAPACKSupport::ExcErrorCode("psyev", info));
          }
        else
          {
            liwork = iwork[0];
            AssertThrow(liwork > 0, dealii::ExcInternalError());
            iwork.resize(liwork);

            psyevx(&jobz,
                   &range,
                   &uplo,
                   &n_rows,
                   A_loc,
                   &submatrix_row,
                   &submatrix_column,
                   descriptor,
                   &vl,
                   &vu,
                   &il,
                   &iu,
                   &abstol,
                   &m,
                   &nz,
                   ev.data(),
                   &orfac,
                   eigenvectors_loc,
                   &eigenvectors->submatrix_row,
                   &eigenvectors->submatrix_column,
                   eigenvectors->descriptor,
                   work.data(),
                   &lwork,
                   iwork.data(),
                   &liwork,
                   ifail.data(),
                   iclustr.data(),
                   gap.data(),
                   &info);

            AssertThrow(info == 0,
                        dftfe::LAPACKSupport::ExcErrorCode("psyevx", info));
          }
        // if eigenvectors are queried copy eigenvectors to original matrix
        // as the temporary matrix eigenvectors has identical dimensions and
        // block-cyclic distribution we simply swap the local array
        if (compute_eigenvectors)
          this->values.swap(eigenvectors->values);

        // adapt the size of ev to fit m upon return
        while (ev.size() > static_cast<size_type>(m))
          ev.pop_back();
      }
    /*
     * send number of computed eigenvalues to inactive processes
     */
    grid->send_to_inactive(&m, 1);

    /*
     * inactive processes have to resize array of eigenvalues
     */
    if (!grid->mpi_process_is_active)
      ev.resize(m);
    /*
     * send the eigenvalues to processors not being part of the process grid
     */
    grid->send_to_inactive(ev.data(), ev.size());

    /*
     * if only eigenvalues are queried the content of the matrix will be
     * destroyed if the eigenpairs are queried matrix A on exit stores the
     * eigenvectors in the columns
     */
    if (compute_eigenvectors)
      {
        property = dftfe::LAPACKSupport::Property::general;
        state    = dftfe::LAPACKSupport::eigenvalues;
      }
    else
      state = dftfe::LAPACKSupport::unusable;

    return ev;
  }


  template <typename NumberType>
  std::vector<double>
  ScaLAPACKMatrix<NumberType>::eigenpairs_hermitian_by_index_MRRR(
    const std::pair<unsigned int, unsigned int> &index_limits,
    const bool                                   compute_eigenvectors)
  {
    // Check validity of index limits.
    AssertIndexRange(index_limits.first, static_cast<unsigned int>(n_rows));
    AssertIndexRange(index_limits.second, static_cast<unsigned int>(n_rows));

    const std::pair<unsigned int, unsigned int> idx =
      std::make_pair(std::min(index_limits.first, index_limits.second),
                     std::max(index_limits.first, index_limits.second));

    // Compute all eigenvalues/eigenvectors.
    if (idx.first == 0 && idx.second == static_cast<unsigned int>(n_rows - 1))
      return eigenpairs_hermitian_MRRR(compute_eigenvectors);
    else
      return eigenpairs_hermitian_MRRR(compute_eigenvectors, idx);
  }


  template <typename NumberType>
  std::vector<double>
  ScaLAPACKMatrix<NumberType>::eigenpairs_hermitian_MRRR(
    const bool                                   compute_eigenvectors,
    const std::pair<unsigned int, unsigned int> &eigenvalue_idx,
    const std::pair<double, double> &            eigenvalue_limits)
  {
    Assert(state == dftfe::LAPACKSupport::matrix,
           dealii::ExcMessage(
             "Matrix has to be in Matrix state before calling this function."));
    Assert(property == dftfe::LAPACKSupport::hermitian,
           dealii::ExcMessage(
             "Matrix has to be hermitian for this operation."));

    std::lock_guard<std::mutex> lock(mutex);

    const bool use_values = (std::isnan(eigenvalue_limits.first) ||
                             std::isnan(eigenvalue_limits.second)) ?
                              false :
                              true;
    const bool use_indices =
      ((eigenvalue_idx.first == dealii::numbers::invalid_unsigned_int) ||
       (eigenvalue_idx.second == dealii::numbers::invalid_unsigned_int)) ?
        false :
        true;

    Assert(
      !(use_values && use_indices),
      dealii::ExcMessage(
        "Prescribing both the index and value range for the eigenvalues is ambiguous"));

    // If computation of eigenvectors is not required, use a sufficiently small
    // distributed matrix.
    std::unique_ptr<ScaLAPACKMatrix<NumberType>> eigenvectors =
      compute_eigenvectors ?
        std::make_unique<ScaLAPACKMatrix<NumberType>>(n_rows,
                                                      grid,
                                                      row_block_size) :
        std::make_unique<ScaLAPACKMatrix<NumberType>>(
          grid->n_process_rows, grid->n_process_columns, grid, 1, 1);

    eigenvectors->property = property;
    // Number of eigenvalues to be returned from psyevr; upon successful exit ev
    // contains the m seclected eigenvalues in ascending order.
    int                 m = n_rows;
    std::vector<double> ev(n_rows);

    // Number of eigenvectors to be returned;
    // Upon successful exit the first m=nz columns contain the selected
    // eigenvectors (only if jobz=='V').
    int nz = 0;

    if (grid->mpi_process_is_active)
      {
        int info = 0;
        /*
         * For jobz==N only eigenvalues are computed, for jobz='V' also the
         * eigenvectors of the matrix are computed.
         */
        char jobz = compute_eigenvectors ? 'V' : 'N';
        // Default value is to compute all eigenvalues and optionally
        // eigenvectors.
        char   range = 'A';
        double vl    = 0.0;
        double vu    = 0.0;
        int    il = 1, iu = 1;

        // Index range for eigenvalues is not specified.
        if (!use_indices)
          {
            // Interval for eigenvalues is not specified and consequently all
            // eigenvalues/eigenpairs will be computed.
            if (!use_values)
              {
                range = 'A';
              }
            else
              {
                range = 'V';
                vl =
                  std::min(eigenvalue_limits.first, eigenvalue_limits.second);
                vu =
                  std::max(eigenvalue_limits.first, eigenvalue_limits.second);
              }
          }
        else
          {
            range = 'I';
            // As Fortran starts counting/indexing from 1 unlike C/C++, where it
            // starts from 0.
            il = std::min(eigenvalue_idx.first, eigenvalue_idx.second) + 1;
            iu = std::max(eigenvalue_idx.first, eigenvalue_idx.second) + 1;
          }
        NumberType *A_loc = this->values.data();

        /*
         * By setting lwork to -1 a workspace query for optimal length of work
         * is performed.
         */
        int         lwork  = -1;
        int         liwork = -1;
        NumberType *eigenvectors_loc =
          (compute_eigenvectors ? eigenvectors->values.data() : nullptr);
        work.resize(1);
        iwork.resize(1);

        psyevr(&jobz,
               &range,
               &uplo,
               &n_rows,
               A_loc,
               &submatrix_row,
               &submatrix_column,
               descriptor,
               &vl,
               &vu,
               &il,
               &iu,
               &m,
               &nz,
               ev.data(),
               eigenvectors_loc,
               &eigenvectors->submatrix_row,
               &eigenvectors->submatrix_column,
               eigenvectors->descriptor,
               work.data(),
               &lwork,
               iwork.data(),
               &liwork,
               &info);
        AssertThrow(info == 0,
                    dftfe::LAPACKSupport::ExcErrorCode("psyevr", info));

        lwork = lworkFromWork(work);
        work.resize(lwork);
        liwork = iwork[0];
        iwork.resize(liwork);

        psyevr(&jobz,
               &range,
               &uplo,
               &n_rows,
               A_loc,
               &submatrix_row,
               &submatrix_column,
               descriptor,
               &vl,
               &vu,
               &il,
               &iu,
               &m,
               &nz,
               ev.data(),
               eigenvectors_loc,
               &eigenvectors->submatrix_row,
               &eigenvectors->submatrix_column,
               eigenvectors->descriptor,
               work.data(),
               &lwork,
               iwork.data(),
               &liwork,
               &info);

        AssertThrow(info == 0,
                    dftfe::LAPACKSupport::ExcErrorCode("psyevr", info));

        if (compute_eigenvectors)
          AssertThrow(
            m == nz,
            dealii::ExcMessage(
              "psyevr failed to compute all eigenvectors for the selected eigenvalues"));

        // If eigenvectors are queried, copy eigenvectors to original matrix.
        // As the temporary matrix eigenvectors has identical dimensions and
        // block-cyclic distribution we simply swap the local array.
        if (compute_eigenvectors)
          this->values.swap(eigenvectors->values);

        // Adapt the size of ev to fit m upon return.
        while (ev.size() > static_cast<size_type>(m))
          ev.pop_back();
      }
    /*
     * Send number of computed eigenvalues to inactive processes.
     */
    grid->send_to_inactive(&m, 1);

    /*
     * Inactive processes have to resize array of eigenvalues.
     */
    if (!grid->mpi_process_is_active)
      ev.resize(m);
    /*
     * Send the eigenvalues to processors not being part of the process grid.
     */
    grid->send_to_inactive(ev.data(), ev.size());

    /*
     * If only eigenvalues are queried, the content of the matrix will be
     * destroyed. If the eigenpairs are queried, matrix A on exit stores the
     * eigenvectors in the columns.
     */
    if (compute_eigenvectors)
      {
        property = dftfe::LAPACKSupport::Property::general;
        state    = dftfe::LAPACKSupport::eigenvalues;
      }
    else
      state = dftfe::LAPACKSupport::unusable;

    return ev;
  }


  template class ScaLAPACKMatrix<double>;
  template class ScaLAPACKMatrix<std::complex<double>>;
} // namespace dftfe
