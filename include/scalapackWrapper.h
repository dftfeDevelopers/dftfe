// ---------------------------------------------------------------------
//
// Copyright (c) 2019-2020  The Regents of the University of Michigan and DFT-FE
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



#ifndef ScaLAPACKMatrix_H_
#define ScaLAPACKMatrix_H_

#include "headers.h"
#include "process_grid.h"
#include "lapack_support.h"


namespace dftfe
{
  /**
   * @brief Scalapack wrapper adapted from dealii library and extended implementation to complex datatype
   *
   * @author Sambit Das
   */
  template <typename NumberType>
  class ScaLAPACKMatrix
  {
  public:
    /**
     * Declare the type for container size.
     */
    using size_type = unsigned int;

    /**
     * Constructor for a rectangular matrix with @p n_rows and @p n_cols
     * and distributed using the grid @p process_grid.
     *
     * The parameters @p row_block_size and @p column_block_size are the block sizes used
     * for the block-cyclic distribution of the matrix.
     * In general, it is recommended to use powers of $2$, e.g. $16,32,64,
     * \dots$.
     */
    ScaLAPACKMatrix(
      const size_type                                  n_rows,
      const size_type                                  n_columns,
      const std::shared_ptr<const dftfe::ProcessGrid> &process_grid,
      const size_type                                  row_block_size    = 32,
      const size_type                                  column_block_size = 32,
      const dftfe::LAPACKSupport::Property             property =
        dftfe::LAPACKSupport::Property::general);

    /**
     * Constructor for a square matrix of size @p size, and distributed
     * using the process grid in @p process_grid.
     *
     * The parameter @p block_size is used for the block-cyclic distribution of the matrix.
     * An identical block size is used for the rows and columns of the matrix.
     * In general, it is recommended to use powers of $2$, e.g. $16,32,64,
     * \dots$.
     */
    ScaLAPACKMatrix(
      const size_type                                  size,
      const std::shared_ptr<const dftfe::ProcessGrid> &process_grid,
      const size_type                                  block_size = 32,
      const dftfe::LAPACKSupport::Property             property =
        dftfe::LAPACKSupport::Property::hermitian);


    /**
     * Initialize the rectangular matrix with @p n_rows and @p n_cols
     * and distributed using the grid @p process_grid.
     *
     * The parameters @p row_block_size and @p column_block_size are the block sizes used
     * for the block-cyclic distribution of the matrix.
     * In general, it is recommended to use powers of $2$, e.g. $16,32,64,
     * \dots$.
     */
    void
    reinit(const size_type                                  n_rows,
           const size_type                                  n_columns,
           const std::shared_ptr<const dftfe::ProcessGrid> &process_grid,
           const size_type                                  row_block_size = 32,
           const size_type                      column_block_size          = 32,
           const dftfe::LAPACKSupport::Property property =
             dftfe::LAPACKSupport::Property::general);

    /**
     * Initialize the square matrix of size @p size and distributed using the grid @p process_grid.
     *
     * The parameter @p block_size is used for the block-cyclic distribution of the matrix.
     * An identical block size is used for the rows and columns of the matrix.
     * In general, it is recommended to use powers of $2$, e.g. $16,32,64,
     * \dots$.
     */
    void
    reinit(const size_type                                  size,
           const std::shared_ptr<const dftfe::ProcessGrid> &process_grid,
           const size_type                                  block_size = 32,
           const dftfe::LAPACKSupport::Property             property =
             dftfe::LAPACKSupport::Property::hermitian);


    /**
     * Assign @p property to this matrix.
     */
    void
    set_property(const dftfe::LAPACKSupport::Property property);

    /**
     * Return current @p property of this matrix
     */
    dftfe::LAPACKSupport::Property
    get_property() const;

    /**
     * Return current @p state of this matrix
     */
    dftfe::LAPACKSupport::State
    get_state() const;


    /**
     * Copy the contents of the distributed matrix into a differently distributed matrix @p dest.
     * The function also works for matrices with different process grids
     * or block-cyclic distributions.
     */
    void
    copy_to(ScaLAPACKMatrix<NumberType> &dest) const;


    /**
     * Complex conjugate.
     */
    void
    conjugate();


    /**
     * The operations based on the input parameter @p transpose_B and the
     * alignment conditions are summarized in the following table:
     *
     * | transpose_B |          Block Sizes         |                    Operation                  |
     * | :---------: | :--------------------------: | :-------------------------------------------: |
     * |   false     | $MB_A=MB_B$ <br> $NB_A=NB_B$ |  $\mathbf{A} = a \mathbf{A} + b \mathbf{B}$   |
     * |   true      | $MB_A=NB_B$ <br> $NB_A=MB_B$ | $\mathbf{A} = a \mathbf{A} + b \mathbf{B}^T$  |
     *
     * The matrices $\mathbf{A}$ and $\mathbf{B}$ must have the same process
     * grid.
     */
    void
    add(const ScaLAPACKMatrix<NumberType> &B,
        const NumberType                   a           = 0.,
        const NumberType                   b           = 1.,
        const bool                         transpose_B = false);


    /**
     * The operations based on the input parameter @p conjugate_transpose_B and the
     * alignment conditions are summarized in the following table:
     *
     * | transpose_B |          Block Sizes         |                    Operation                  |
     * | :---------: | :--------------------------: | :-------------------------------------------: |
     * |   false     | $MB_A=MB_B$ <br> $NB_A=NB_B$ |  $\mathbf{A} = a \mathbf{A} + b \mathbf{B}$   |
     * |   true      | $MB_A=NB_B$ <br> $NB_A=MB_B$ | $\mathbf{A} = a \mathbf{A} + b \mathbf{B}^C$  |
     *
     * The matrices $\mathbf{A}$ and $\mathbf{B}$ must have the same process
     * grid.
     */
    void
    zadd(const ScaLAPACKMatrix<NumberType> &B,
         const NumberType                   a                     = 0.,
         const NumberType                   b                     = 1.,
         const bool                         conjugate_transpose_B = false);

    /**
     * Transposing assignment: $\mathbf{A} = \mathbf{B}^T$
     *
     * The matrices $\mathbf{A}$ and $\mathbf{B}$ must have the same process
     * grid.
     *
     * The following alignment conditions have to be fulfilled: $MB_A=NB_B$ and
     * $NB_A=MB_B$.
     */
    void
    copy_transposed(const ScaLAPACKMatrix<NumberType> &B);


    /**
     * Transposing assignment: $\mathbf{A} = \mathbf{B}^C$
     *
     * The matrices $\mathbf{A}$ and $\mathbf{B}$ must have the same process
     * grid.
     *
     * The following alignment conditions have to be fulfilled: $MB_A=NB_B$ and
     * $NB_A=MB_B$.
     */
    void
    copy_conjugate_transposed(const ScaLAPACKMatrix<NumberType> &B);

    /**
     * Matrix-matrix-multiplication:
     *
     * The operations based on the input parameters and the alignment conditions
     * are summarized in the following table:
     *
     * | transpose_A | transpose_B |                  Block Sizes                  |                             Operation                           |
     * | :---------: | :---------: | :-------------------------------------------: | :-------------------------------------------------------------: |
     * | false       |   false     | $MB_A=MB_C$ <br> $NB_A=MB_B$ <br> $NB_B=NB_C$ |   $\mathbf{C} = b \mathbf{A} \cdot \mathbf{B} + c \mathbf{C}$   |
     * | false       |   true      | $MB_A=MB_C$ <br> $NB_A=NB_B$ <br> $MB_B=NB_C$ |  $\mathbf{C} = b \mathbf{A} \cdot \mathbf{B}^T + c \mathbf{C}$  |
     * | true        |   false     | $MB_A=MB_B$ <br> $NB_A=MB_C$ <br> $NB_B=NB_C$ | $\mathbf{C} = b \mathbf{A}^T \cdot \mathbf{B} + c \mathbf{C}$   |
     * | true        |   true      | $MB_A=NB_B$ <br> $NB_A=MB_C$ <br> $MB_B=NB_C$ | $\mathbf{C} = b \mathbf{A}^T \cdot \mathbf{B}^T + c \mathbf{C}$ |
     *
     * It is assumed that $\mathbf{A}$ and $\mathbf{B}$ have compatible sizes
     * and that
     * $\mathbf{C}$ already has the right size.
     *
     * The matrices $\mathbf{A}$, $\mathbf{B}$ and $\mathbf{C}$ must have the
     * same process grid.
     */
    void
    mult(const NumberType                   b,
         const ScaLAPACKMatrix<NumberType> &B,
         const NumberType                   c,
         ScaLAPACKMatrix<NumberType> &      C,
         const bool                         transpose_A = false,
         const bool                         transpose_B = false) const;


    /**
     * Matrix-matrix-multiplication:
     *
     * The operations based on the input parameters and the alignment conditions
     * are summarized in the following table:
     *
     * | conjugate_transpose_A | conjugate_transpose_B |                  Block Sizes                  |                             Operation                           |
     * | :---------: | :---------: | :-------------------------------------------: | :-------------------------------------------------------------: |
     * | false       |   false     | $MB_A=MB_C$ <br> $NB_A=MB_B$ <br> $NB_B=NB_C$ |   $\mathbf{C} = b \mathbf{A} \cdot \mathbf{B} + c \mathbf{C}$   |
     * | false       |   true      | $MB_A=MB_C$ <br> $NB_A=NB_B$ <br> $MB_B=NB_C$ |  $\mathbf{C} = b \mathbf{A} \cdot \mathbf{B}^C + c \mathbf{C}$  |
     * | true        |   false     | $MB_A=MB_B$ <br> $NB_A=MB_C$ <br> $NB_B=NB_C$ | $\mathbf{C} = b \mathbf{A}^C \cdot \mathbf{B} + c \mathbf{C}$   |
     * | true        |   true      | $MB_A=NB_B$ <br> $NB_A=MB_C$ <br> $MB_B=NB_C$ | $\mathbf{C} = b \mathbf{A}^C \cdot \mathbf{B}^C + c \mathbf{C}$ |
     *
     * It is assumed that $\mathbf{A}$ and $\mathbf{B}$ have compatible sizes
     * and that
     * $\mathbf{C}$ already has the right size.
     *
     * The matrices $\mathbf{A}$, $\mathbf{B}$ and $\mathbf{C}$ must have the
     * same process grid.
     */
    void
    zmult(const NumberType                   b,
          const ScaLAPACKMatrix<NumberType> &B,
          const NumberType                   c,
          ScaLAPACKMatrix<NumberType> &      C,
          const bool                         conjugate_transpose_A = false,
          const bool conjugate_transpose_B = false) const;


    /**
     * Matrix-matrix-multiplication.
     *
     * The optional parameter @p adding determines whether the result is
     * stored in $\mathbf{C}$ or added to $\mathbf{C}$.
     *
     * if (@p adding) $\mathbf{C} = \mathbf{C} + \mathbf{A} \cdot \mathbf{B}$
     *
     * else $\mathbf{C} = \mathbf{A} \cdot \mathbf{B}$
     *
     * It is assumed that $\mathbf{A}$ and $\mathbf{B}$ have compatible sizes
     * and that
     * $\mathbf{C}$ already has the right size.
     *
     * The following alignment conditions have to be fulfilled: $MB_A=MB_C$,
     * $NB_A=MB_B$ and $NB_B=NB_C$.
     */
    void
    mmult(ScaLAPACKMatrix<NumberType> &      C,
          const ScaLAPACKMatrix<NumberType> &B,
          const bool                         adding = false) const;

    /**
     * Matrix-matrix-multiplication using transpose of $\mathbf{A}$.
     *
     * The optional parameter @p adding determines whether the result is
     * stored in $\mathbf{C}$ or added to $\mathbf{C}$.
     *
     * if (@p adding) $\mathbf{C} = \mathbf{C} + \mathbf{A}^T \cdot \mathbf{B}$
     *
     * else $\mathbf{C} = \mathbf{A}^T \cdot \mathbf{B}$
     *
     * It is assumed that $\mathbf{A}$ and $\mathbf{B}$ have compatible sizes
     * and that
     * $\mathbf{C}$ already has the right size.
     *
     * The following alignment conditions have to be fulfilled: $MB_A=MB_B$,
     * $NB_A=MB_C$ and $NB_B=NB_C$.
     */
    void
    Tmmult(ScaLAPACKMatrix<NumberType> &      C,
           const ScaLAPACKMatrix<NumberType> &B,
           const bool                         adding = false) const;

    /**
     * Matrix-matrix-multiplication using the transpose of $\mathbf{B}$.
     *
     * The optional parameter @p adding determines whether the result is
     * stored in $\mathbf{C}$ or added to $\mathbf{C}$.
     *
     * if (@p adding) $\mathbf{C} = \mathbf{C} + \mathbf{A} \cdot \mathbf{B}^T$
     *
     * else $\mathbf{C} = \mathbf{A} \cdot \mathbf{B}^T$
     *
     * It is assumed that $\mathbf{A}$ and $\mathbf{B}$ have compatible sizes
     * and that
     * $\mathbf{C}$ already has the right size.
     *
     * The following alignment conditions have to be fulfilled: $MB_A=MB_C$,
     * $NB_A=NB_B$ and $MB_B=NB_C$.
     */
    void
    mTmult(ScaLAPACKMatrix<NumberType> &      C,
           const ScaLAPACKMatrix<NumberType> &B,
           const bool                         adding = false) const;

    /**
     * Matrix-matrix-multiplication using transpose of $\mathbf{A}$ and
     * $\mathbf{B}$.
     *
     * The optional parameter @p adding determines whether the result is
     * stored in $\mathbf{C}$ or added to $\mathbf{C}$.
     *
     * if (@p adding) $\mathbf{C} = \mathbf{C} + \mathbf{A}^T \cdot
     * \mathbf{B}^T$
     *
     * else $\mathbf{C} = \mathbf{A}^T \cdot \mathbf{B}^T$
     *
     * It is assumed that $\mathbf{A}$ and $\mathbf{B}$ have compatible sizes
     * and that
     * $\mathbf{C}$ already has the right size.
     *
     * The following alignment conditions have to be fulfilled: $MB_A=NB_B$,
     * $NB_A=MB_C$ and $MB_B=NB_C$.
     */
    void
    TmTmult(ScaLAPACKMatrix<NumberType> &      C,
            const ScaLAPACKMatrix<NumberType> &B,
            const bool                         adding = false) const;


    /**
     * Matrix-matrix-multiplication.
     *
     * The optional parameter @p adding determines whether the result is
     * stored in $\mathbf{C}$ or added to $\mathbf{C}$.
     *
     * if (@p adding) $\mathbf{C} = \mathbf{C} + \mathbf{A} \cdot \mathbf{B}$
     *
     * else $\mathbf{C} = \mathbf{A} \cdot \mathbf{B}$
     *
     * It is assumed that $\mathbf{A}$ and $\mathbf{B}$ have compatible sizes
     * and that
     * $\mathbf{C}$ already has the right size.
     *
     * The following alignment conditions have to be fulfilled: $MB_A=MB_C$,
     * $NB_A=MB_B$ and $NB_B=NB_C$.
     */
    void
    zmmult(ScaLAPACKMatrix<NumberType> &      C,
           const ScaLAPACKMatrix<NumberType> &B,
           const bool                         adding = false) const;

    /**
     * Matrix-matrix-multiplication using conjugate transpose of $\mathbf{A}$.
     *
     * The optional parameter @p adding determines whether the result is
     * stored in $\mathbf{C}$ or added to $\mathbf{C}$.
     *
     * if (@p adding) $\mathbf{C} = \mathbf{C} + \mathbf{A}^C \cdot \mathbf{B}$
     *
     * else $\mathbf{C} = \mathbf{A}^C \cdot \mathbf{B}$
     *
     * It is assumed that $\mathbf{A}$ and $\mathbf{B}$ have compatible sizes
     * and that
     * $\mathbf{C}$ already has the right size.
     *
     * The following alignment conditions have to be fulfilled: $MB_A=MB_B$,
     * $NB_A=MB_C$ and $NB_B=NB_C$.
     */
    void
    zCmmult(ScaLAPACKMatrix<NumberType> &      C,
            const ScaLAPACKMatrix<NumberType> &B,
            const bool                         adding = false) const;

    /**
     * Matrix-matrix-multiplication using the conjugate transpose of
     * $\mathbf{B}$.
     *
     * The optional parameter @p adding determines whether the result is
     * stored in $\mathbf{C}$ or added to $\mathbf{C}$.
     *
     * if (@p adding) $\mathbf{C} = \mathbf{C} + \mathbf{A} \cdot \mathbf{B}^C$
     *
     * else $\mathbf{C} = \mathbf{A} \cdot \mathbf{B}^C$
     *
     * It is assumed that $\mathbf{A}$ and $\mathbf{B}$ have compatible sizes
     * and that
     * $\mathbf{C}$ already has the right size.
     *
     * The following alignment conditions have to be fulfilled: $MB_A=MB_C$,
     * $NB_A=NB_B$ and $MB_B=NB_C$.
     */
    void
    zmCmult(ScaLAPACKMatrix<NumberType> &      C,
            const ScaLAPACKMatrix<NumberType> &B,
            const bool                         adding = false) const;

    /**
     * Matrix-matrix-multiplication using conjugate transpose of $\mathbf{A}$
     * and
     * $\mathbf{B}$.
     *
     * The optional parameter @p adding determines whether the result is
     * stored in $\mathbf{C}$ or added to $\mathbf{C}$.
     *
     * if (@p adding) $\mathbf{C} = \mathbf{C} + \mathbf{A}^C \cdot
     * \mathbf{B}^T$
     *
     * else $\mathbf{C} = \mathbf{A}^C \cdot \mathbf{B}^C$
     *
     * It is assumed that $\mathbf{A}$ and $\mathbf{B}$ have compatible sizes
     * and that
     * $\mathbf{C}$ already has the right size.
     *
     * The following alignment conditions have to be fulfilled: $MB_A=NB_B$,
     * $NB_A=MB_C$ and $MB_B=NB_C$.
     */
    void
    zCmCmult(ScaLAPACKMatrix<NumberType> &      C,
             const ScaLAPACKMatrix<NumberType> &B,
             const bool                         adding = false) const;


    /**
     * Number of rows of the $M \times N$ matrix.
     */
    size_type
    m() const;

    /**
     * Number of columns of the $M \times N$ matrix.
     */
    size_type
    n() const;

    /**
     * Number of local rows on this MPI processes.
     */
    unsigned int
    local_m() const;

    /**
     * Number of local columns on this MPI process.
     */
    unsigned int
    local_n() const;

    /**
     * Return the global row number for the given local row @p loc_row .
     */
    unsigned int
    global_row(const unsigned int loc_row) const;

    /**
     * Return the global column number for the given local column @p loc_column.
     */
    unsigned int
    global_column(const unsigned int loc_column) const;

    /**
     * Read access to local element.
     */
    NumberType
    local_el(const unsigned int loc_row, const unsigned int loc_column) const;

    /**
     * Write access to local element.
     */
    NumberType &
    local_el(const unsigned int loc_row, const unsigned int loc_column);

    /**
     * Compute the Cholesky factorization of the matrix using ScaLAPACK
     * function <code>pXpotrf</code>. The result of the factorization is stored
     * in this object.
     */
    void
    compute_cholesky_factorization();

    /**
     * Compute the LU factorization of the matrix using ScaLAPACK
     * function <code>pXgetrf</code> and partial pivoting with row interchanges.
     * The result of the factorization is stored in this object.
     */
    void
    compute_lu_factorization();

    /**
     * Invert the matrix by first computing a Cholesky for hermitian matrices
     * or a LU factorization for general matrices and then
     * building the actual inverse using <code>pXpotri</code> or
     * <code>pXgetri</code>. If the matrix is triangular, the LU factorization
     * step is skipped, and <code>pXtrtri</code> is used directly.
     *
     * If a Cholesky or LU factorization has been applied previously,
     * <code>pXpotri</code> or <code>pXgetri</code> are called directly.
     *
     * The inverse is stored in this object.
     */
    void
    invert();

    /**
     * Computing selected eigenvalues and, optionally, the eigenvectors of the
     * real hermitian matrix $\mathbf{A} \in \mathbb{R}^{M \times M}$.
     *
     * The eigenvalues/eigenvectors are selected by prescribing a range of indices @p index_limits.
     *
     * If successful, the computed eigenvalues are arranged in ascending order.
     * The eigenvectors are stored in the columns of the matrix, thereby
     * overwriting the original content of the matrix.
     *
     * If all eigenvalues/eigenvectors have to be computed, pass the closed interval $ \left[ 0, M-1 \right] $ in @p index_limits.
     *
     * Pass the closed interval $ \left[ M-r, M-1 \right] $ if the $r$ largest
     * eigenvalues/eigenvectors are desired.
     */
    std::vector<double>
    eigenpairs_hermitian_by_index(
      const std::pair<unsigned int, unsigned int> &index_limits,
      const bool                                   compute_eigenvectors);

    /**
     * Computing selected eigenvalues and, optionally, the eigenvectors of the
     * real hermitian matrix $\mathbf{A} \in \mathbb{R}^{M \times M}$ using the
     * MRRR algorithm.
     *
     * The eigenvalues/eigenvectors are selected by prescribing a range of indices @p index_limits.
     *
     * If successful, the computed eigenvalues are arranged in ascending order.
     * The eigenvectors are stored in the columns of the matrix, thereby
     * overwriting the original content of the matrix.
     *
     * If all eigenvalues/eigenvectors have to be computed, pass the closed interval $ \left[ 0, M-1 \right] $ in @p index_limits.
     *
     * Pass the closed interval $ \left[ M-r, M-1 \right] $ if the $r$ largest
     * eigenvalues/eigenvectors are desired.
     */
    std::vector<double>
    eigenpairs_hermitian_by_index_MRRR(
      const std::pair<unsigned int, unsigned int> &index_limits,
      const bool                                   compute_eigenvectors);

  private:
    /**
     * Computing selected eigenvalues and, optionally, the eigenvectors.
     * The eigenvalues/eigenvectors are selected by either prescribing a range of indices @p index_limits
     * or a range of values @p value_limits for the eigenvalues. The function will throw an exception
     * if both ranges are prescribed (meaning that both ranges differ from the
     * default value) as this ambiguity is prohibited. If successful, the
     * computed eigenvalues are arranged in ascending order. The eigenvectors
     * are stored in the columns of the matrix, thereby overwriting the original
     * content of the matrix.
     */
    std::vector<double>
    eigenpairs_hermitian(
      const bool                                   compute_eigenvectors,
      const std::pair<unsigned int, unsigned int> &index_limits =
        std::make_pair(dealii::numbers::invalid_unsigned_int,
                       dealii::numbers::invalid_unsigned_int),
      const std::pair<double, double> &value_limits =
        std::make_pair(std::numeric_limits<double>::quiet_NaN(),
                       std::numeric_limits<double>::quiet_NaN()));

    /**
     * Computing selected eigenvalues and, optionally, the eigenvectors of the
     * real hermitian matrix $\mathbf{A} \in \mathbb{R}^{M \times M}$ using the
     * MRRR algorithm.
     * The eigenvalues/eigenvectors are selected by either prescribing a range of indices @p index_limits
     * or a range of values @p value_limits for the eigenvalues. The function will throw an exception
     * if both ranges are prescribed (meaning that both ranges differ from the
     * default value) as this ambiguity is prohibited.
     *
     * By calling this function the original content of the matrix will be
     * overwritten. If requested, the eigenvectors are stored in the columns of
     * the matrix. Also in the case that just the eigenvalues are required, the
     * content of the matrix will be overwritten.
     *
     * If successful, the computed eigenvalues are arranged in ascending order.
     *
     * @note Due to a bug in Netlib-ScaLAPACK, either all or no eigenvectors can be computed.
     * Therefore, the input @p index_limits has to be set accordingly. Using Intel-MKL this restriction is not required.
     */
    std::vector<double>
    eigenpairs_hermitian_MRRR(
      const bool                                   compute_eigenvectors,
      const std::pair<unsigned int, unsigned int> &index_limits =
        std::make_pair(dealii::numbers::invalid_unsigned_int,
                       dealii::numbers::invalid_unsigned_int),
      const std::pair<double, double> &value_limits =
        std::make_pair(std::numeric_limits<double>::quiet_NaN(),
                       std::numeric_limits<double>::quiet_NaN()));


    /**
     * local storage
     */
    std::vector<NumberType> values;

    /**
     * Since ScaLAPACK operations notoriously change the meaning of the matrix
     * entries, we record the current state after the last operation here.
     */
    dftfe::LAPACKSupport::State state;

    /**
     * Additional property of the matrix which may help to select more
     * efficient ScaLAPACK functions.
     */
    dftfe::LAPACKSupport::Property property;

    /**
     * A shared pointer to a Utilities::MPI::ProcessGrid object which contains a
     * BLACS context and a MPI communicator, as well as other necessary data
     * structures.
     */
    std::shared_ptr<const dftfe::ProcessGrid> grid;

    /**
     * Number of rows in the matrix.
     */
    int n_rows;

    /**
     * Number of columns in the matrix.
     */
    int n_columns;

    /**
     * Row block size.
     */
    int row_block_size;

    /**
     * Column block size.
     */
    int column_block_size;

    /**
     * Number of rows in the matrix owned by the current process.
     */
    int n_local_rows;

    /**
     * Number of columns in the matrix owned by the current process.
     */
    int n_local_columns;

    /**
     * ScaLAPACK description vector.
     */
    int descriptor[9];

    /**
     * Workspace array.
     */
    mutable std::vector<NumberType> work;

    /**
     * Integer workspace array.
     */
    mutable std::vector<int> iwork;

    /**
     * Integer array holding pivoting information required
     * by ScaLAPACK's matrix factorization routines.
     */
    std::vector<int> ipiv;

    /**
     * A character to define where elements are stored in case
     * ScaLAPACK operations support this.
     */
    const char uplo;

    /**
     * The process row of the process grid over which the first row
     * of the global matrix is distributed.
     */
    const int first_process_row;

    /**
     * The process column of the process grid over which the first column
     * of the global matrix is distributed.
     */
    const int first_process_column;

    /**
     * Global row index that determines where to start a submatrix.
     * Currently this equals unity, as we don't use submatrices.
     */
    const int submatrix_row;

    /**
     * Global column index that determines where to start a submatrix.
     * Currently this equals unity, as we don't use submatrices.
     */
    const int submatrix_column;

    /**
     * Thread mutex.
     */
    mutable dealii::Threads::Mutex mutex;
  };

  // ----------------------- inline functions ----------------------------

#ifndef DOXYGEN

  template <typename NumberType>
  inline NumberType
  ScaLAPACKMatrix<NumberType>::local_el(const unsigned int loc_row,
                                        const unsigned int loc_column) const
  {
    return values[loc_column * n_local_rows + loc_row];
    // return (*this)(loc_row, loc_column);
  }



  template <typename NumberType>
  inline NumberType &
  ScaLAPACKMatrix<NumberType>::local_el(const unsigned int loc_row,
                                        const unsigned int loc_column)
  {
    return values[loc_column * n_local_rows + loc_row];
    // return (*this)(loc_row, loc_column);
  }


  template <typename NumberType>
  inline unsigned int
  ScaLAPACKMatrix<NumberType>::m() const
  {
    return n_rows;
  }



  template <typename NumberType>
  inline unsigned int
  ScaLAPACKMatrix<NumberType>::n() const
  {
    return n_columns;
  }



  template <typename NumberType>
  unsigned int
  ScaLAPACKMatrix<NumberType>::local_m() const
  {
    return n_local_rows;
  }



  template <typename NumberType>
  unsigned int
  ScaLAPACKMatrix<NumberType>::local_n() const
  {
    return n_local_columns;
  }


#endif // DOXYGEN


} // namespace dftfe
#endif // ScaLAPACKMatrix_H_
