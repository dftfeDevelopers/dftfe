//
// -------------------------------------------------------------------------------------
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
// --------------------------------------------------------------------------------------
//
// @author Phani Motamarri
//
#ifndef operatorDFTClass_h
#define operatorDFTClass_h

#include "constraintMatrixInfo.h"
#include "headers.h"
#include "process_grid.h"
#include "scalapackWrapper.h"
#include "elpaScalaManager.h"

#include <vector>

namespace dftfe
{
  /**
   * @brief Base class for building the DFT operator and the action of operator on a vector
   *
   * @author Phani Motamarri, Sambit Das
   */
  class operatorDFTClass
  {
    //
    // methods
    //
  public:
    /**
     * @brief Destructor.
     */
    virtual ~operatorDFTClass() = 0;

    /**
     * @brief initialize operatorClass
     *
     */
    virtual void
    init() = 0;

    /**
     * @brief initializes parallel layouts and index maps for HX, XtHX and creates a flattened array format for X
     *
     * @param wavefunBlockSize number of wavefunction vector (block size of X).
     * @param flag controls the creation of flattened array format and index maps or only index maps
     *
     * @return X format to store a multi-vector array
     * in a flattened format with all the wavefunction values corresponding to a
     * given node being stored contiguously
     *
     */
    virtual void
    reinit(const unsigned int                    wavefunBlockSize,
           distributedCPUVec<dataTypes::number> &X,
           bool                                  flag) = 0;

    virtual void
    reinit(const unsigned int                         wavefunBlockSize,
           distributedCPUMultiVec<dataTypes::number> &X,
           bool                                       flag) = 0;

    virtual void
    reinit(const unsigned int wavefunBlockSize) = 0;

    virtual void
    reinitkPointSpinIndex(const unsigned int kPointIndex,
                          const unsigned int spinIndex) = 0;

    virtual void
    initCellWaveFunctionMatrix(
      const unsigned int                         numberWaveFunctions,
      distributedCPUMultiVec<dataTypes::number> &X,
      std::vector<dataTypes::number> &           cellWaveFunctionMatrix) = 0;


    virtual void
    fillGlobalArrayFromCellWaveFunctionMatrix(
      const unsigned int                         wavefunBlockSize,
      const std::vector<dataTypes::number> &     cellWaveFunctionMatrix,
      distributedCPUMultiVec<dataTypes::number> &X) = 0;

    virtual void
    initWithScalar(const unsigned int              numberWaveFunctions,
                   double                          scalarValue,
                   std::vector<dataTypes::number> &cellWaveFunctionMatrix) = 0;

    virtual void
    axpby(double                                scalarA,
          double                                scalarB,
          const unsigned int                    numberWaveFunctions,
          const std::vector<dataTypes::number> &cellXWaveFunctionMatrix,
          std::vector<dataTypes::number> &      cellYWaveFunctionMatrix) = 0;

    virtual void
    getInteriorSurfaceNodesMapFromGlobalArray(
      std::vector<unsigned int> &globalArrayClassificationMap) = 0;


    /**
     * @brief compute diagonal mass matrix
     *
     * @param dofHandler dofHandler associated with the current mesh
     * @param constraintMatrix constraints to be used
     * @param sqrtMassVec output the value of square root of diagonal mass matrix
     * @param invSqrtMassVec output the value of inverse square root of diagonal mass matrix
     */
    virtual void
    computeMassVector(const dealii::DoFHandler<3> &            dofHandler,
                      const dealii::AffineConstraints<double> &constraintMatrix,
                      distributedCPUVec<double> &              sqrtMassVec,
                      distributedCPUVec<double> &invSqrtMassVec) = 0;


    /**
     * @brief Compute operator times multi-field vectors
     *
     * @param X Vector containing multi-wavefunction fields (though X does not
     * change inside the function it is scaled and rescaled back to
     * avoid duplication of memory and hence is not const)
     * @param numberComponents number of wavefunctions associated with a given node
     * @param Y Vector containing multi-component fields after operator times vectors product
     */
    virtual void
    HX(distributedCPUMultiVec<dataTypes::number> &X,
       const unsigned int                         numberComponents,
       const bool                                 scaleFlag,
       const double                               scalar,
       distributedCPUMultiVec<dataTypes::number> &Y,
       const bool onlyHPrimePartForFirstOrderDensityMatResponse = false) = 0;


    virtual void
    HX(distributedCPUMultiVec<dataTypes::number> &src,
       std::vector<dataTypes::number> &           cellSrcWaveFunctionMatrix,
       const unsigned int                         numberWaveFunctions,
       const bool                                 scaleFlag,
       const double                               scalar,
       const double                               scalarA,
       const double                               scalarB,
       distributedCPUMultiVec<dataTypes::number> &dst,
       std::vector<dataTypes::number> &cellDstWaveFunctionMatrix) = 0;

    virtual void
    computeNonLocalProjectorKetTimesXTimesV(
      distributedCPUMultiVec<dataTypes::number> &src,
      distributedCPUMultiVec<dataTypes::number>
        &                projectorKetTimesVectorFlattened,
      const unsigned int numberWaveFunctions) = 0;


    /**
     * @brief Compute projection of the operator into a subspace spanned by a given orthogonal basis HProjConj=X^{T}*HConj*XConj
     *
     * @param X Vector of Vectors containing multi-wavefunction fields
     * @param numberComponents number of wavefunctions associated with a given node
     * @param ProjMatrix projected small matrix
     */
    virtual void
    XtHX(const dataTypes::number *       X,
         const unsigned int              numberComponents,
         const unsigned int              numberLocalDofs,
         std::vector<dataTypes::number> &ProjHam) = 0;

    /**
     * @brief Compute projection of the operator into a subspace spanned by a given orthogonal basis HProjConj=X^{T}*HConj*XConj
     *
     * @param X Vector of Vectors containing multi-wavefunction fields
     * @param numberComponents number of wavefunctions associated with a given node
     * @param processGrid two-dimensional processor grid corresponding to the parallel projHamPar
     * @param projHamPar parallel ScaLAPACKMatrix which stores the computed projection
     * of the operation into the given subspace
     */
    virtual void
    XtHX(const dataTypes::number *                        X,
         const unsigned int                               numberComponents,
         const unsigned int                               numberLocalDofs,
         const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
         dftfe::ScaLAPACKMatrix<dataTypes::number> &      projHamPar,
         const bool onlyHPrimePartForFirstOrderDensityMatResponse = false) = 0;


    /**
     * @brief Compute projection of the operator into a subspace spanned by a given orthogonal basis HProjConj=X^{T}*HConj*XConj
     *
     * @param X Vector of Vectors containing multi-wavefunction fields
     * @param totalNumberComponents number of wavefunctions associated with a given node
     * @param singlePrecComponents number of wavecfuntions starting from the first for
     * which the project Hamiltionian block will be computed in single
     * procession. However the cross blocks will still be computed in double
     * precision.
     * @param processGrid two-dimensional processor grid corresponding to the parallel projHamPar
     * @param projHamPar parallel ScaLAPACKMatrix which stores the computed projection
     * of the operation into the given subspace
     */
    virtual void
    XtHXMixedPrec(
      const dataTypes::number *                        X,
      const unsigned int                               totalNumberComponents,
      const unsigned int                               singlePrecComponents,
      const unsigned int                               numberLocalDofs,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      dftfe::ScaLAPACKMatrix<dataTypes::number> &      projHamPar,
      const bool onlyHPrimePartForFirstOrderDensityMatResponse = false) = 0;


    void
    setInvSqrtMassVector(distributedCPUVec<double> &X);
    distributedCPUVec<double> &
    getInvSqrtMassVector();


    /**
     * @brief Get constraint matrix eigen
     *
     * @return pointer to constraint matrix eigen
     */
    dftUtils::constraintMatrixInfo *
    getOverloadedConstraintMatrix() const;


    /**
     * @brief Get matrix free data
     *
     * @return pointer to matrix free data
     */
    const dealii::MatrixFree<3, double> *
    getMatrixFreeData() const;


    /**
     * @brief Get relevant mpi communicator
     *
     * @return mpi communicator
     */
    const MPI_Comm &
    getMPICommunicator() const;


    /**
     * @brief Get index map of flattened array to cell based numbering
     *
     * @return pointer to constraint matrix eigen
     */
    virtual const std::vector<dealii::types::global_dof_index> &
    getFlattenedArrayCellLocalProcIndexIdMap() const = 0;


    virtual distributedCPUMultiVec<dataTypes::number> &
    getParallelProjectorKetTimesBlockVector() = 0;

    virtual const std::vector<double> &
    getShapeFunctionGradValuesDensityGaussQuad() const = 0;

    virtual const std::vector<double> &
    getShapeFunctionValuesDensityGaussLobattoQuad() const = 0;

    virtual const std::vector<double> &
    getShapeFunctionValuesDensityTransposed() const = 0;

    virtual const std::vector<double> &
    getShapeFunctionValuesNLPTransposed() const = 0;

    virtual const std::vector<double> &
    getShapeFunctionGradientValuesNLPTransposed() const = 0;

    virtual const std::vector<double> &
    getInverseJacobiansNLP() const = 0;


  protected:
    /**
     * @brief default Constructor.
     */
    operatorDFTClass();


    /**
     * @brief Constructor.
     */
    operatorDFTClass(const MPI_Comm &                     mpi_comm_replica,
                     const dealii::MatrixFree<3, double> &matrix_free_data,
                     dftUtils::constraintMatrixInfo &     constraintMatrixNone);

  protected:
    //
    // Get overloaded constraint matrix object constructed using 1-component FE
    // object
    //
    dftUtils::constraintMatrixInfo *d_constraintMatrixData;
    //
    // matrix-free data
    //
    const dealii::MatrixFree<3, double> *d_matrix_free_data;

    //
    // inv sqrt mass vector
    //
    distributedCPUVec<double> d_invSqrtMassVector;

    /// index map
    std::vector<dealii::types::global_dof_index>
      d_FullflattenedArrayCellLocalProcIndexIdMap;

    /// density quad rule shape function data for FEOrder mesh with node index
    /// being the fastest index
    std::vector<double> d_densityGaussQuadShapeFunctionValues;

    std::vector<double> d_densityGaussQuadShapeFunctionGradientValues;
    /// FEOrderRhoNodal+1 Gauss Lobotto quadrature shape function data for
    /// FEOrder mesh with node index being the fastest index
    std::vector<double> d_densityGlQuadShapeFunctionValues;

    std::vector<double> d_shapeFunctionValueDensityTransposed;

    std::vector<double> d_shapeFunctionValueNLPTransposed;

    std::vector<double> d_shapeFunctionGradientValueNLPTransposed;

    std::vector<double> d_inverseJacobiansNLP;

    //
    // mpi communicator
    //
    MPI_Comm d_mpi_communicator;
  };
} // namespace dftfe
#endif
