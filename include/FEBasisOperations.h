// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022  The Regents of the University of Michigan and DFT-FE
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

#ifndef dftfeFEBasisOperations_h
#define dftfeFEBasisOperations_h

#include <MultiVector.h>
#include <headers.h>
#include <constraintMatrixInfo.h>
#include <constraintMatrixInfoDevice.h>
#include <DeviceTypeConfig.h>
#include <BLASWrapper.h>

namespace dftfe
{
  namespace basis
  {
    enum UpdateFlags
    {
      update_default = 0,

      update_values = 0x0001,

      update_gradients = 0x0002,

      update_transpose = 0x0004,

      update_quadpoints = 0x0008,

      update_inversejacobians = 0x0010,

      update_jxw = 0x0020,
    };

    inline UpdateFlags
    operator|(const UpdateFlags f1, const UpdateFlags f2)
    {
      return static_cast<UpdateFlags>(static_cast<unsigned int>(f1) |
                                      static_cast<unsigned int>(f2));
    }



    inline UpdateFlags &
    operator|=(UpdateFlags &f1, const UpdateFlags f2)
    {
      f1 = f1 | f2;
      return f1;
    }


    inline UpdateFlags operator&(const UpdateFlags f1, const UpdateFlags f2)
    {
      return static_cast<UpdateFlags>(static_cast<unsigned int>(f1) &
                                      static_cast<unsigned int>(f2));
    }


    inline UpdateFlags &
    operator&=(UpdateFlags &f1, const UpdateFlags f2)
    {
      f1 = f1 & f2;
      return f1;
    }


    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    class FEBasisOperationsBase
    {
    protected:
      mutable dftfe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace>
        tempCellNodalData, tempQuadratureGradientsData,
        tempQuadratureGradientsDataNonAffine;

      std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
        d_BLASWrapperPtr;

    public:
      /**
       * @brief Constructor, fills required data structures using deal.ii's MatrixFree and AffineConstraints objects
       * @param[in] matrixFreeData MatrixFree object.
       * @param[in] constraintsVector std::vector of AffineConstraints, should
       * be the same vector which was passed for the construction of the given
       * MatrixFree object.
       */
      FEBasisOperationsBase(
        dealii::MatrixFree<3, ValueTypeBasisData> &matrixFreeData,
        std::vector<const dealii::AffineConstraints<ValueTypeBasisData> *>
          &constraintsVector,
        std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
          BLASWrapperPtr);


      /**
       * @brief Default Destructor
       */
      ~FEBasisOperationsBase() = default;

      /**
       * @brief fills required data structures for the given dofHandlerID
       * @param[in] dofHandlerID dofHandler index to be used for getting data
       * from the MatrixFree object.
       * @param[in] quadratureID std::vector of quadratureIDs to be used, should
       * be the same IDs which were used during the construction of the given
       * MatrixFree object.
       */
      void
      init(const unsigned int &             dofHandlerID,
           const std::vector<unsigned int> &quadratureID,
           const std::vector<UpdateFlags>   updateFlags);
      /**
       * @brief fills required data structures from another FEBasisOperations object
       * @param[in] basisOperationsSrc Source FEBasisOperations object.
       */
      template <dftfe::utils::MemorySpace memorySpaceSrc>
      void
      init(const FEBasisOperationsBase<ValueTypeBasisCoeff,
                                       ValueTypeBasisData,
                                       memorySpaceSrc> &basisOperationsSrc);

      /**
       * @brief sets internal variables and optionally resizes internal temp storage for interpolation operations
       * @param[in] vecBlockSize block size to used for operations on vectors,
       * this has to be set to the exact value before any such operations are
       * called.
       * @param[in] cellBlockSize block size to used for cells, this has to be
       * set to a value greater than or equal to the required value before any
       * such operations are called
       * @param[in] quadratureID Quadrature index to be used.
       * @param[in] isResizeTempStorage whether to resize internal tempstorage.
       */
      void
      reinit(const unsigned int &vecBlockSize,
             const unsigned int &cellBlockSize,
             const unsigned int &quadratureID,
             const bool          isResizeTempStorage = true);

      // private:
#if defined(DFTFE_WITH_DEVICE)
      using constraintInfoClass =
        typename std::conditional<memorySpace ==
                                    dftfe::utils::MemorySpace::DEVICE,
                                  dftUtils::constraintMatrixInfoDevice,
                                  dftUtils::constraintMatrixInfo>::type;
#else
      using constraintInfoClass = dftUtils::constraintMatrixInfo;
#endif



      /**
       * @brief Initializes indexset maps from process level indices to cell level indices for a single vector, also initializes cell index to cellid map.
       */
      void
      initializeIndexMaps();

      /**
       * @brief Initializes indexset maps from process level indices to cell level indices for multivectors.
       */
      void
      initializeFlattenedIndexMaps();

      /**
       * @brief Initializes the constraintMatrixInfo object.
       */
      void
      initializeConstraints();

      /**
       * @brief Reinitializes the constraintMatrixInfo object.
       */
      void
      reinitializeConstraints(
        std::vector<const dealii::AffineConstraints<ValueTypeBasisData> *>
          &constraintsVector);

      /**
       * @brief Constructs the MPIPatternP2P object.
       */
      void
      initializeMPIPattern();

      /**
       * @brief Fill the shape function data and jacobian data in the ValueTypeBasisCoeff datatype.
       */
      void
      initializeShapeFunctionAndJacobianData();

      /**
       * @brief Fill the shape function data and jacobian data in the ValueTypeBasisData datatype.
       */
      void
      initializeShapeFunctionAndJacobianBasisData();

      /**
       * @brief Resizes the internal temp storage to be sufficient for the vector and cell block sizes provided in reinit.
       */
      void
      resizeTempStorage();

      /**
       * @brief Number of quadrature points per cell for the quadratureID set in reinit.
       */
      unsigned int
      nQuadsPerCell() const;

      /**
       * @brief Number of DoFs per cell for the dofHandlerID set in init.
       */
      unsigned int
      nDofsPerCell() const;

      /**
       * @brief Number of locally owned cells on the current processor.
       */
      unsigned int
      nCells() const;

      /**
       * @brief Number of DoFs on the current processor, locally owned + ghosts.
       */
      unsigned int
      nRelaventDofs() const;

      /**
       * @brief Number of locally owned DoFs on the current processor.
       */
      unsigned int
      nOwnedDofs() const;

      /**
       * @brief Shape function values at quadrature points.
       * @param[in] transpose if false the the data is indexed as [iQuad *
       * d_nDofsPerCell + iNode] and if true it is indexed as [iNode *
       * d_nQuadsPerCell + iQuad].
       */
      const dftfe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace> &
      shapeFunctionData(bool transpose = false) const;

      /**
       * @brief Shape function gradient values at quadrature points.
       * @param[in] transpose if false the the data is indexed as [iDim *
       * d_nQuadsPerCell * d_nDofsPerCell + iQuad * d_nDofsPerCell + iNode] and
       * if true it is indexed as [iDim * d_nQuadsPerCell * d_nDofsPerCell +
       * iNode * d_nQuadsPerCell + iQuad].
       */
      const dftfe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace> &
      shapeFunctionGradientData(bool transpose = false) const;

      /**
       * @brief Inverse Jacobian matrices, for cartesian cells returns the
       * diagonal elements of the inverse Jacobian matrices for each cell, for
       * affine cells returns the 3x3 inverse Jacobians for each cell otherwise
       * returns the 3x3 inverse Jacobians at each quad point for each cell.
       */
      const dftfe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace> &
      inverseJacobians() const;

      /**
       * @brief determinant of Jacobian times the quadrature weight at each
       * quad point for each cell.
       */
      const dftfe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace> &
      JxW() const;

      /**
       * @brief quad point coordinates for each cell.
       */
      const dftfe::utils::MemoryStorage<ValueTypeBasisData,
                                        dftfe::utils::MemorySpace::HOST> &
      quadPoints() const;

      /**
       * @brief Shape function values at quadrature points in ValueTypeBasisData.
       * @param[in] transpose if false the the data is indexed as [iQuad *
       * d_nDofsPerCell + iNode] and if true it is indexed as [iNode *
       * d_nQuadsPerCell + iQuad].
       */
      template <typename A = ValueTypeBasisCoeff,
                typename B = ValueTypeBasisData,
                typename std::enable_if_t<std::is_same<A, B>::value, int> = 0>
      const dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace> &
      shapeFunctionBasisData(bool transpose = false) const;
      template <typename A = ValueTypeBasisCoeff,
                typename B = ValueTypeBasisData,
                typename std::enable_if_t<!std::is_same<A, B>::value, int> = 0>
      const dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace> &
      shapeFunctionBasisData(bool transpose = false) const;

      /**
       * @brief Shape function gradient values at quadrature points in ValueTypeBasisData.
       * @param[in] transpose if false the the data is indexed as [iDim *
       * d_nQuadsPerCell * d_nDofsPerCell + iQuad * d_nDofsPerCell + iNode] and
       * if true it is indexed as [iDim * d_nQuadsPerCell * d_nDofsPerCell +
       * iNode * d_nQuadsPerCell + iQuad].
       */
      template <typename A = ValueTypeBasisCoeff,
                typename B = ValueTypeBasisData,
                typename std::enable_if_t<std::is_same<A, B>::value, int> = 0>
      const dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace> &
      shapeFunctionGradientBasisData(bool transpose = false) const;
      template <typename A = ValueTypeBasisCoeff,
                typename B = ValueTypeBasisData,
                typename std::enable_if_t<!std::is_same<A, B>::value, int> = 0>
      const dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace> &
      shapeFunctionGradientBasisData(bool transpose = false) const;

      /**
       * @brief Inverse Jacobian matrices in ValueTypeBasisData, for cartesian cells returns the
       * diagonal elements of the inverse Jacobian matrices for each cell, for
       * affine cells returns the 3x3 inverse Jacobians for each cell otherwise
       * returns the 3x3 inverse Jacobians at each quad point for each cell.
       */
      template <typename A = ValueTypeBasisCoeff,
                typename B = ValueTypeBasisData,
                typename std::enable_if_t<std::is_same<A, B>::value, int> = 0>
      const dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace> &
      inverseJacobiansBasisData() const;
      template <typename A = ValueTypeBasisCoeff,
                typename B = ValueTypeBasisData,
                typename std::enable_if_t<!std::is_same<A, B>::value, int> = 0>
      const dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace> &
      inverseJacobiansBasisData() const;

      /**
       * @brief determinant of Jacobian times the quadrature weight in ValueTypeBasisData at each
       * quad point for each cell.
       */
      template <typename A = ValueTypeBasisCoeff,
                typename B = ValueTypeBasisData,
                typename std::enable_if_t<std::is_same<A, B>::value, int> = 0>
      const dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace> &
      JxWBasisData() const;
      template <typename A = ValueTypeBasisCoeff,
                typename B = ValueTypeBasisData,
                typename std::enable_if_t<!std::is_same<A, B>::value, int> = 0>
      const dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace> &
      JxWBasisData() const;

      /**
       * @brief returns 2 if all cells on current processor are Cartesian,
       * 1 if all cells on current processor are affine and 0 otherwise.
       */
      unsigned int
      cellsTypeFlag() const;

      /**
       * @brief returns the deal.ii cellID corresponing to given cell Index.
       * @param[in] iElem cell Index
       */
      dealii::CellId
      cellID(const unsigned int iElem) const;

      /**
       * @brief returns the cell index corresponding to given deal.ii cellID.
       * @param[in] iElem cell Index
       */
      unsigned int
      cellIndex(const dealii::CellId cellid) const;

      /**
       * @brief Creates a multivector.
       * @param[in] blocksize Number of vectors in the multivector.
       * @param[out] multiVector the created multivector.
       */
      void
      createMultiVector(
        const unsigned int blocksize,
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
          &multiVector) const;

      /**
       * @brief Creates scratch multivectors.
       * @param[in] vecBlockSize Number of vectors in the multivector.
       * @param[out] numMultiVecs number of scratch multivectors needed with
       * this vecBlockSize.
       */
      void
      createScratchMultiVectors(const unsigned int vecBlockSize,
                                const unsigned int numMultiVecs = 1) const;

      /**
       * @brief Clears scratch multivectors.
       */
      void
      clearScratchMultiVectors() const;

      /**
       * @brief Gets scratch multivectors.
       * @param[in] vecBlockSize Number of vectors in the multivector.
       * @param[out] numMultiVecs index of the multivector among those with the
       * same vecBlockSize.
       */
      dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace> &
      getMultiVector(const unsigned int vecBlockSize,
                     const unsigned int index = 0) const;

      /**
       * @brief Apply constraints on given multivector.
       * @param[inout] multiVector the given multivector.
       */
      void
      distribute(dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                                   memorySpace> &multiVector,
                 unsigned int constraintIndex =
                   std::numeric_limits<unsigned int>::max()) const;



      /**
       * @brief Return the underlying deal.II matrixfree object.
       */
      const dealii::MatrixFree<3, ValueTypeBasisData> &
      matrixFreeData() const;



      std::vector<constraintInfoClass> d_constraintInfo;
      unsigned int                     d_nOMPThreads;
      std::vector<const dealii::AffineConstraints<ValueTypeBasisData> *>
        *                                              d_constraintsVector;
      const dealii::MatrixFree<3, ValueTypeBasisData> *d_matrixFreeDataPtr;
      dftfe::utils::MemoryStorage<dftfe::global_size_type,
                                  dftfe::utils::MemorySpace::HOST>
        d_cellDofIndexToProcessDofIndexMap;
      std::map<unsigned int,
               dftfe::utils::MemoryStorage<ValueTypeBasisData,
                                           dftfe::utils::MemorySpace::HOST>>
        d_quadPoints;
      dftfe::utils::MemoryStorage<dftfe::global_size_type, memorySpace>
                                             d_flattenedCellDofIndexToProcessDofIndexMap;
      std::vector<dealii::CellId>            d_cellIndexToCellIdMap;
      std::map<dealii::CellId, unsigned int> d_cellIdToCellIndexMap;
      std::map<unsigned int,
               dftfe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace>>
        d_inverseJacobianData;
      std::map<unsigned int,
               dftfe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace>>
        d_JxWData;
      std::map<unsigned int,
               dftfe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace>>
        d_shapeFunctionData;
      std::map<unsigned int,
               dftfe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace>>
        d_shapeFunctionGradientDataInternalLayout;
      std::map<unsigned int,
               dftfe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace>>
        d_shapeFunctionGradientData;
      std::map<unsigned int,
               dftfe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace>>
        d_shapeFunctionDataTranspose;
      std::map<unsigned int,
               dftfe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace>>
        d_shapeFunctionGradientDataTranspose;

      std::map<unsigned int,
               dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace>>
        d_inverseJacobianBasisData;
      std::map<unsigned int,
               dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace>>
        d_JxWBasisData;
      std::map<unsigned int,
               dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace>>
        d_shapeFunctionBasisData;
      std::map<unsigned int,
               dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace>>
        d_shapeFunctionGradientBasisData;
      std::map<unsigned int,
               dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace>>
        d_shapeFunctionBasisDataTranspose;
      std::map<unsigned int,
               dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace>>
        d_shapeFunctionGradientBasisDataTranspose;


      mutable std::map<
        unsigned int,
        std::vector<
          dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>>>
        scratchMultiVectors;

      std::vector<unsigned int> d_quadratureIDsVector;
      unsigned int              d_quadratureID;
      unsigned int              d_quadratureIndex;
      std::vector<unsigned int> d_nQuadsPerCell;
      unsigned int              d_dofHandlerID;
      unsigned int              d_nVectors;
      unsigned int              d_nCells;
      unsigned int              d_cellsBlockSize;
      unsigned int              d_nDofsPerCell;
      unsigned int              d_localSize;
      unsigned int              d_locallyOwnedSize;
      bool                      areAllCellsAffine;
      bool                      areAllCellsCartesian;
      std::vector<UpdateFlags>  d_updateFlags;

      std::shared_ptr<const utils::mpi::MPIPatternP2P<memorySpace>>
        mpiPatternP2P;
    };
    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    class FEBasisOperations : FEBasisOperationsBase<ValueTypeBasisCoeff,
                                                    ValueTypeBasisData,
                                                    memorySpace>
    {};

    template <typename ValueTypeBasisCoeff, typename ValueTypeBasisData>
    class FEBasisOperations<ValueTypeBasisCoeff,
                            ValueTypeBasisData,
                            dftfe::utils::MemorySpace::HOST>
      : public FEBasisOperationsBase<ValueTypeBasisCoeff,
                                     ValueTypeBasisData,
                                     dftfe::utils::MemorySpace::HOST>
    {
    public:
      using FEBasisOperationsBase<
        ValueTypeBasisCoeff,
        ValueTypeBasisData,
        dftfe::utils::MemorySpace::HOST>::FEBasisOperationsBase;

      using FEBasisOperationsBase<ValueTypeBasisCoeff,
                                  ValueTypeBasisData,
                                  dftfe::utils::MemorySpace::HOST>::d_nCells;
      using FEBasisOperationsBase<ValueTypeBasisCoeff,
                                  ValueTypeBasisData,
                                  dftfe::utils::MemorySpace::HOST>::d_localSize;
      using FEBasisOperationsBase<
        ValueTypeBasisCoeff,
        ValueTypeBasisData,
        dftfe::utils::MemorySpace::HOST>::d_locallyOwnedSize;
      using FEBasisOperationsBase<
        ValueTypeBasisCoeff,
        ValueTypeBasisData,
        dftfe::utils::MemorySpace::HOST>::tempCellNodalData;
      using FEBasisOperationsBase<
        ValueTypeBasisCoeff,
        ValueTypeBasisData,
        dftfe::utils::MemorySpace::HOST>::tempQuadratureGradientsData;
      using FEBasisOperationsBase<
        ValueTypeBasisCoeff,
        ValueTypeBasisData,
        dftfe::utils::MemorySpace::HOST>::tempQuadratureGradientsDataNonAffine;
      using FEBasisOperationsBase<ValueTypeBasisCoeff,
                                  ValueTypeBasisData,
                                  dftfe::utils::MemorySpace::HOST>::d_nVectors;
      using FEBasisOperationsBase<
        ValueTypeBasisCoeff,
        ValueTypeBasisData,
        dftfe::utils::MemorySpace::HOST>::d_BLASWrapperPtr;
      using FEBasisOperationsBase<
        ValueTypeBasisCoeff,
        ValueTypeBasisData,
        dftfe::utils::MemorySpace::HOST>::d_quadratureID;
      using FEBasisOperationsBase<
        ValueTypeBasisCoeff,
        ValueTypeBasisData,
        dftfe::utils::MemorySpace::HOST>::d_quadratureIndex;
      using FEBasisOperationsBase<
        ValueTypeBasisCoeff,
        ValueTypeBasisData,
        dftfe::utils::MemorySpace::HOST>::d_nQuadsPerCell;
      using FEBasisOperationsBase<
        ValueTypeBasisCoeff,
        ValueTypeBasisData,
        dftfe::utils::MemorySpace::HOST>::d_nDofsPerCell;
      using FEBasisOperationsBase<
        ValueTypeBasisCoeff,
        ValueTypeBasisData,
        dftfe::utils::MemorySpace::HOST>::areAllCellsAffine;
      using FEBasisOperationsBase<
        ValueTypeBasisCoeff,
        ValueTypeBasisData,
        dftfe::utils::MemorySpace::HOST>::areAllCellsCartesian;
      using FEBasisOperationsBase<
        ValueTypeBasisCoeff,
        ValueTypeBasisData,
        dftfe::utils::MemorySpace::HOST>::d_updateFlags;
      using FEBasisOperationsBase<
        ValueTypeBasisCoeff,
        ValueTypeBasisData,
        dftfe::utils::MemorySpace::HOST>::d_shapeFunctionData;
      using FEBasisOperationsBase<
        ValueTypeBasisCoeff,
        ValueTypeBasisData,
        dftfe::utils::MemorySpace::HOST>::d_shapeFunctionDataTranspose;
      using FEBasisOperationsBase<
        ValueTypeBasisCoeff,
        ValueTypeBasisData,
        dftfe::utils::MemorySpace::HOST>::d_shapeFunctionGradientData;
      using FEBasisOperationsBase<
        ValueTypeBasisCoeff,
        ValueTypeBasisData,
        dftfe::utils::MemorySpace::HOST>::d_shapeFunctionGradientDataTranspose;
      using FEBasisOperationsBase<ValueTypeBasisCoeff,
                                  ValueTypeBasisData,
                                  dftfe::utils::MemorySpace::HOST>::
        d_shapeFunctionGradientDataInternalLayout;
      using FEBasisOperationsBase<ValueTypeBasisCoeff,
                                  ValueTypeBasisData,
                                  dftfe::utils::MemorySpace::HOST>::d_JxWData;
      using FEBasisOperationsBase<
        ValueTypeBasisCoeff,
        ValueTypeBasisData,
        dftfe::utils::MemorySpace::HOST>::d_inverseJacobianData;
      using FEBasisOperationsBase<
        ValueTypeBasisCoeff,
        ValueTypeBasisData,
        dftfe::utils::MemorySpace::HOST>::d_cellIndexToCellIdMap;
      using FEBasisOperationsBase<
        ValueTypeBasisCoeff,
        ValueTypeBasisData,
        dftfe::utils::MemorySpace::HOST>::d_cellDofIndexToProcessDofIndexMap;
      using FEBasisOperationsBase<ValueTypeBasisCoeff,
                                  ValueTypeBasisData,
                                  dftfe::utils::MemorySpace::HOST>::
        d_flattenedCellDofIndexToProcessDofIndexMap;
      using FEBasisOperationsBase<
        ValueTypeBasisCoeff,
        ValueTypeBasisData,
        dftfe::utils::MemorySpace::HOST>::d_constraintsVector;


      /**
       * @brief Interpolate process level nodal data to cell level quadrature data.
       * @param[in] nodalData process level nodal data, the multivector should
       * already have ghost data and constraints should have been applied.
       * @param[out] quadratureValues Cell level quadrature values, indexed by
       * [iCell * d_nQuadsPerCell * d_nVectors + iQuad * d_nVectors + iVec].
       * @param[out] quadratureGradients Cell level quadrature gradients,
       * indexed by [iCell * 3 * d_nQuadsPerCell * d_nVectors + iDim *
       * d_nQuadsPerCell * d_nVectors + iQuad * d_nVectors + iVec].
       */
      void
      interpolate(
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                          dftfe::utils::MemorySpace::HOST>
          &                  nodalData,
        ValueTypeBasisCoeff *quadratureValues,
        ValueTypeBasisCoeff *quadratureGradients = NULL) const;

      // FIXME Untested function
      /**
       * @brief Integrate cell level quadrature data times shape functions to process level nodal data.
       * @param[in] quadratureValues Cell level quadrature values, indexed by
       * [iCell * d_nQuadsPerCell * d_nVectors + iQuad * d_nVectors + iVec].
       * @param[in] quadratureGradients Cell level quadrature gradients,
       * indexed by [iCell * 3 * d_nQuadsPerCell * d_nVectors + iDim *
       * d_nQuadsPerCell * d_nVectors + iQuad * d_nVectors + iVec].
       * @param[out] nodalData process level nodal data.
       */
      void
      integrateWithBasis(
        ValueTypeBasisCoeff *quadratureValues,
        ValueTypeBasisCoeff *quadratureGradients,
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                          dftfe::utils::MemorySpace::HOST>
          &nodalData) const;

      /**
       * @brief Get cell level nodal data from process level nodal data.
       * @param[in] nodalData process level nodal data, the multivector should
       * already have ghost data and constraints should have been applied.
       * @param[out] cellNodalDataPtr Cell level nodal values, indexed by
       * [iCell * d_nDofsPerCell * d_nVectors + iDoF * d_nVectors + iVec].
       */
      void
      extractToCellNodalData(
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                          dftfe::utils::MemorySpace::HOST>
          &                  nodalData,
        ValueTypeBasisCoeff *cellNodalDataPtr) const;
      // FIXME Untested function
      /**
       * @brief Accumulate cell level nodal data into process level nodal data.
       * @param[in] cellNodalDataPtr Cell level nodal values, indexed by
       * [iCell * d_nDofsPerCell * d_nVectors + iDoF * d_nVectors + iVec].
       * @param[out] nodalData process level nodal data.
       */
      void
      accumulateFromCellNodalData(
        const ValueTypeBasisCoeff *cellNodalDataPtr,
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                          dftfe::utils::MemorySpace::HOST>
          &nodalData) const;

      /**
       * @brief Interpolate process level nodal data to cell level quadrature data.
       * @param[in] nodalData process level nodal data, the multivector should
       * already have ghost data and constraints should have been applied.
       * @param[out] quadratureValues Cell level quadrature values, indexed by
       * [iCell * d_nQuadsPerCell * d_nVectors + iQuad * d_nVectors + iVec].
       * @param[out] quadratureGradients Cell level quadrature gradients,
       * indexed by [iCell * 3 * d_nQuadsPerCell * d_nVectors + iDim *
       * d_nQuadsPerCell * d_nVectors + iQuad * d_nVectors + iVec].
       * @param[in] cellRange the range of cells for which interpolation has to
       * be done.
       */
      void
      interpolateKernel(
        const dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                                dftfe::utils::MemorySpace::HOST>
          &                                         nodalData,
        ValueTypeBasisCoeff *                       quadratureValues,
        ValueTypeBasisCoeff *                       quadratureGradients,
        const std::pair<unsigned int, unsigned int> cellRange) const;

      /**
       * @brief Interpolate cell level nodal data to cell level quadrature data.
       * @param[in] nodalData cell level nodal data, the multivector should
       * already have ghost data and constraints should have been applied.
       * @param[out] quadratureValues Cell level quadrature values, indexed by
       * [iCell * d_nQuadsPerCell * d_nVectors + iQuad * d_nVectors + iVec].
       * @param[out] quadratureGradients Cell level quadrature gradients,
       * indexed by [iCell * 3 * d_nQuadsPerCell * d_nVectors + iDim *
       * d_nQuadsPerCell * d_nVectors + iQuad * d_nVectors + iVec].
       * @param[in] cellRange the range of cells for which interpolation has to
       * be done.
       */
      void
      interpolateKernel(
        const ValueTypeBasisCoeff *                 nodalData,
        ValueTypeBasisCoeff *                       quadratureValues,
        ValueTypeBasisCoeff *                       quadratureGradients,
        const std::pair<unsigned int, unsigned int> cellRange) const;

      // FIXME Untested function
      /**
       * @brief Integrate cell level quadrature data times shape functions to process level nodal data.
       * @param[in] quadratureValues Cell level quadrature values, indexed by
       * [iCell * d_nQuadsPerCell * d_nVectors + iQuad * d_nVectors + iVec].
       * @param[in] quadratureGradients Cell level quadrature gradients,
       * indexed by [iCell * 3 * d_nQuadsPerCell * d_nVectors + iDim *
       * d_nQuadsPerCell * d_nVectors + iQuad * d_nVectors + iVec].
       * @param[out] nodalData process level nodal data.
       * @param[in] cellRange the range of cells for which integration has to be
       * done.
       */
      void
      integrateWithBasisKernel(
        const ValueTypeBasisCoeff *quadratureValues,
        const ValueTypeBasisCoeff *quadratureGradients,
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                          dftfe::utils::MemorySpace::HOST>
          &                                         nodalData,
        const std::pair<unsigned int, unsigned int> cellRange) const;


      /**
       * @brief Get cell level nodal data from process level nodal data.
       * @param[in] nodalData process level nodal data, the multivector should
       * already have ghost data and constraints should have been applied.
       * @param[out] cellNodalDataPtr Cell level nodal values, indexed by
       * [iCell * d_nDofsPerCell * d_nVectors + iDoF * d_nVectors + iVec].
       * @param[in] cellRange the range of cells for which extraction has to be
       * done.
       */
      void
      extractToCellNodalDataKernel(
        const dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                                dftfe::utils::MemorySpace::HOST>
          &                                         nodalData,
        ValueTypeBasisCoeff *                       cellNodalDataPtr,
        const std::pair<unsigned int, unsigned int> cellRange) const;

      // FIXME Untested function
      /**
       * @brief Accumulate cell level nodal data into process level nodal data.
       * @param[in] cellNodalDataPtr Cell level nodal values, indexed by
       * [iCell * d_nDofsPerCell * d_nVectors + iDoF * d_nVectors + iVec].
       * @param[out] nodalData process level nodal data.
       * @param[in] cellRange the range of cells for which extraction has to be
       * done.
       */
      void
      accumulateFromCellNodalDataKernel(
        const ValueTypeBasisCoeff *cellNodalDataPtr,
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                          dftfe::utils::MemorySpace::HOST>
          &                                         nodalData,
        const std::pair<unsigned int, unsigned int> cellRange) const;
    };
#if defined(DFTFE_WITH_DEVICE)
    template <typename ValueTypeBasisCoeff, typename ValueTypeBasisData>
    class FEBasisOperations<ValueTypeBasisCoeff,
                            ValueTypeBasisData,
                            dftfe::utils::MemorySpace::DEVICE>
      : public FEBasisOperationsBase<ValueTypeBasisCoeff,
                                     ValueTypeBasisData,
                                     dftfe::utils::MemorySpace::DEVICE>
    {
    public:
      using FEBasisOperationsBase<
        ValueTypeBasisCoeff,
        ValueTypeBasisData,
        dftfe::utils::MemorySpace::DEVICE>::FEBasisOperationsBase;
      using FEBasisOperationsBase<ValueTypeBasisCoeff,
                                  ValueTypeBasisData,
                                  dftfe::utils::MemorySpace::DEVICE>::d_nCells;
      using FEBasisOperationsBase<
        ValueTypeBasisCoeff,
        ValueTypeBasisData,
        dftfe::utils::MemorySpace::DEVICE>::d_localSize;
      using FEBasisOperationsBase<
        ValueTypeBasisCoeff,
        ValueTypeBasisData,
        dftfe::utils::MemorySpace::DEVICE>::d_locallyOwnedSize;
      using FEBasisOperationsBase<
        ValueTypeBasisCoeff,
        ValueTypeBasisData,
        dftfe::utils::MemorySpace::DEVICE>::tempCellNodalData;
      using FEBasisOperationsBase<
        ValueTypeBasisCoeff,
        ValueTypeBasisData,
        dftfe::utils::MemorySpace::DEVICE>::d_BLASWrapperPtr;
      using FEBasisOperationsBase<
        ValueTypeBasisCoeff,
        ValueTypeBasisData,
        dftfe::utils::MemorySpace::DEVICE>::tempQuadratureGradientsData;
      using FEBasisOperationsBase<ValueTypeBasisCoeff,
                                  ValueTypeBasisData,
                                  dftfe::utils::MemorySpace::DEVICE>::
        tempQuadratureGradientsDataNonAffine;
      using FEBasisOperationsBase<
        ValueTypeBasisCoeff,
        ValueTypeBasisData,
        dftfe::utils::MemorySpace::DEVICE>::d_nVectors;
      using FEBasisOperationsBase<
        ValueTypeBasisCoeff,
        ValueTypeBasisData,
        dftfe::utils::MemorySpace::DEVICE>::d_cellsBlockSize;
      using FEBasisOperationsBase<
        ValueTypeBasisCoeff,
        ValueTypeBasisData,
        dftfe::utils::MemorySpace::DEVICE>::d_quadratureID;
      using FEBasisOperationsBase<
        ValueTypeBasisCoeff,
        ValueTypeBasisData,
        dftfe::utils::MemorySpace::DEVICE>::d_quadratureIndex;
      using FEBasisOperationsBase<
        ValueTypeBasisCoeff,
        ValueTypeBasisData,
        dftfe::utils::MemorySpace::DEVICE>::d_nQuadsPerCell;
      using FEBasisOperationsBase<
        ValueTypeBasisCoeff,
        ValueTypeBasisData,
        dftfe::utils::MemorySpace::DEVICE>::d_nDofsPerCell;
      using FEBasisOperationsBase<
        ValueTypeBasisCoeff,
        ValueTypeBasisData,
        dftfe::utils::MemorySpace::DEVICE>::areAllCellsAffine;
      using FEBasisOperationsBase<
        ValueTypeBasisCoeff,
        ValueTypeBasisData,
        dftfe::utils::MemorySpace::DEVICE>::areAllCellsCartesian;
      using FEBasisOperationsBase<
        ValueTypeBasisCoeff,
        ValueTypeBasisData,
        dftfe::utils::MemorySpace::DEVICE>::d_updateFlags;
      using FEBasisOperationsBase<
        ValueTypeBasisCoeff,
        ValueTypeBasisData,
        dftfe::utils::MemorySpace::DEVICE>::d_shapeFunctionData;
      using FEBasisOperationsBase<
        ValueTypeBasisCoeff,
        ValueTypeBasisData,
        dftfe::utils::MemorySpace::DEVICE>::d_shapeFunctionDataTranspose;
      using FEBasisOperationsBase<
        ValueTypeBasisCoeff,
        ValueTypeBasisData,
        dftfe::utils::MemorySpace::DEVICE>::d_shapeFunctionGradientData;
      using FEBasisOperationsBase<ValueTypeBasisCoeff,
                                  ValueTypeBasisData,
                                  dftfe::utils::MemorySpace::DEVICE>::
        d_shapeFunctionGradientDataTranspose;
      using FEBasisOperationsBase<ValueTypeBasisCoeff,
                                  ValueTypeBasisData,
                                  dftfe::utils::MemorySpace::DEVICE>::
        d_shapeFunctionGradientDataInternalLayout;
      using FEBasisOperationsBase<ValueTypeBasisCoeff,
                                  ValueTypeBasisData,
                                  dftfe::utils::MemorySpace::DEVICE>::d_JxWData;
      using FEBasisOperationsBase<
        ValueTypeBasisCoeff,
        ValueTypeBasisData,
        dftfe::utils::MemorySpace::DEVICE>::d_inverseJacobianData;
      using FEBasisOperationsBase<
        ValueTypeBasisCoeff,
        ValueTypeBasisData,
        dftfe::utils::MemorySpace::DEVICE>::d_cellIndexToCellIdMap;
      using FEBasisOperationsBase<
        ValueTypeBasisCoeff,
        ValueTypeBasisData,
        dftfe::utils::MemorySpace::DEVICE>::d_cellDofIndexToProcessDofIndexMap;
      using FEBasisOperationsBase<ValueTypeBasisCoeff,
                                  ValueTypeBasisData,
                                  dftfe::utils::MemorySpace::DEVICE>::
        d_flattenedCellDofIndexToProcessDofIndexMap;
      using FEBasisOperationsBase<
        ValueTypeBasisCoeff,
        ValueTypeBasisData,
        dftfe::utils::MemorySpace::DEVICE>::d_constraintsVector;

      // FIXME has to be removed in a future PR
      /**
       * @brief sets device blas handle for internal blas operations.
       */
      dftfe::utils::deviceBlasHandle_t *d_deviceBlasHandlePtr;
      void
      setDeviceBLASHandle(
        dftfe::utils::deviceBlasHandle_t *deviceBlasHandlePtr);

      // FIXME has to be removed in a future PR
      /**
       * @brief gets device blas handle for blas operations.
       */
      dftfe::utils::deviceBlasHandle_t &
      getDeviceBLASHandle();



      /**
       * @brief Interpolate process level nodal data to cell level quadrature data.
       * @param[in] nodalData process level nodal data, the multivector should
       * already have ghost data and constraints should have been applied.
       * @param[out] quadratureValues Cell level quadrature values, indexed by
       * [iCell * d_nQuadsPerCell * d_nVectors + iQuad * d_nVectors + iVec].
       * @param[out] quadratureGradients Cell level quadrature gradients,
       * indexed by [iCell * 3 * d_nQuadsPerCell * d_nVectors + iDim *
       * d_nQuadsPerCell * d_nVectors + iQuad * d_nVectors + iVec].
       */
      void
      interpolate(
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                          dftfe::utils::MemorySpace::DEVICE>
          &                  nodalData,
        ValueTypeBasisCoeff *quadratureValues,
        ValueTypeBasisCoeff *quadratureGradients = NULL) const;


      // FIXME Untested function
      /**
       * @brief Integrate cell level quadrature data times shape functions to process level nodal data.
       * @param[in] quadratureValues Cell level quadrature values, indexed by
       * [iCell * d_nQuadsPerCell * d_nVectors + iQuad * d_nVectors + iVec].
       * @param[in] quadratureGradients Cell level quadrature gradients,
       * indexed by [iCell * 3 * d_nQuadsPerCell * d_nVectors + iDim *
       * d_nQuadsPerCell * d_nVectors + iQuad * d_nVectors + iVec].
       * @param[out] nodalData process level nodal data.
       */
      void
      integrateWithBasis(
        ValueTypeBasisCoeff *quadratureValues,
        ValueTypeBasisCoeff *quadratureGradients,
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                          dftfe::utils::MemorySpace::DEVICE>
          &nodalData) const;

      /**
       * @brief Get cell level nodal data from process level nodal data.
       * @param[in] nodalData process level nodal data, the multivector should
       * already have ghost data and constraints should have been applied.
       * @param[out] cellNodalDataPtr Cell level nodal values, indexed by
       * [iCell * d_nDofsPerCell * d_nVectors + iDoF * d_nVectors + iVec].
       */
      void
      extractToCellNodalData(
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                          dftfe::utils::MemorySpace::DEVICE>
          &                  nodalData,
        ValueTypeBasisCoeff *cellNodalDataPtr) const;

      // FIXME Untested function
      /**
       * @brief Accumulate cell level nodal data into process level nodal data.
       * @param[in] cellNodalDataPtr Cell level nodal values, indexed by
       * [iCell * d_nDofsPerCell * d_nVectors + iDoF * d_nVectors + iVec].
       * @param[out] nodalData process level nodal data.
       */
      void
      accumulateFromCellNodalData(
        const ValueTypeBasisCoeff *cellNodalDataPtr,
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                          dftfe::utils::MemorySpace::DEVICE>
          &nodalData) const;

      /**
       * @brief Interpolate process level nodal data to cell level quadrature data.
       * @param[in] nodalData process level nodal data, the multivector should
       * already have ghost data and constraints should have been applied.
       * @param[out] quadratureValues Cell level quadrature values, indexed by
       * [iCell * d_nQuadsPerCell * d_nVectors + iQuad * d_nVectors + iVec].
       * @param[out] quadratureGradients Cell level quadrature gradients,
       * indexed by [iCell * 3 * d_nQuadsPerCell * d_nVectors + iDim *
       * d_nQuadsPerCell * d_nVectors + iQuad * d_nVectors + iVec].
       * @param[in] cellRange the range of cells for which interpolation has to
       * be done.
       */
      void
      interpolateKernel(
        const dftfe::linearAlgebra::MultiVector<
          ValueTypeBasisCoeff,
          dftfe::utils::MemorySpace::DEVICE> &      nodalData,
        ValueTypeBasisCoeff *                       quadratureValues,
        ValueTypeBasisCoeff *                       quadratureGradients,
        const std::pair<unsigned int, unsigned int> cellRange) const;

      /**
       * @brief Interpolate cell level nodal data to cell level quadrature data.
       * @param[in] nodalData cell level nodal data, the multivector should
       * already have ghost data and constraints should have been applied.
       * @param[out] quadratureValues Cell level quadrature values, indexed by
       * [iCell * d_nQuadsPerCell * d_nVectors + iQuad * d_nVectors + iVec].
       * @param[out] quadratureGradients Cell level quadrature gradients,
       * indexed by [iCell * 3 * d_nQuadsPerCell * d_nVectors + iDim *
       * d_nQuadsPerCell * d_nVectors + iQuad * d_nVectors + iVec].
       * @param[in] cellRange the range of cells for which interpolation has to
       * be done.
       */
      void
      interpolateKernel(
        const ValueTypeBasisCoeff *                 nodalData,
        ValueTypeBasisCoeff *                       quadratureValues,
        ValueTypeBasisCoeff *                       quadratureGradients,
        const std::pair<unsigned int, unsigned int> cellRange) const;

      // FIXME Untested function
      /**
       * @brief Integrate cell level quadrature data times shape functions to process level nodal data.
       * @param[in] quadratureValues Cell level quadrature values, indexed by
       * [iCell * d_nQuadsPerCell * d_nVectors + iQuad * d_nVectors + iVec].
       * @param[in] quadratureGradients Cell level quadrature gradients,
       * indexed by [iCell * 3 * d_nQuadsPerCell * d_nVectors + iDim *
       * d_nQuadsPerCell * d_nVectors + iQuad * d_nVectors + iVec].
       * @param[out] nodalData process level nodal data.
       * @param[in] cellRange the range of cells for which integration has to be
       * done.
       */
      void
      integrateWithBasisKernel(
        const ValueTypeBasisCoeff *quadratureValues,
        const ValueTypeBasisCoeff *quadratureGradients,
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                          dftfe::utils::MemorySpace::DEVICE>
          &                                         nodalData,
        const std::pair<unsigned int, unsigned int> cellRange) const;


      /**
       * @brief Get cell level nodal data from process level nodal data.
       * @param[in] nodalData process level nodal data, the multivector should
       * already have ghost data and constraints should have been applied.
       * @param[out] cellNodalDataPtr Cell level nodal values, indexed by
       * [iCell * d_nDofsPerCell * d_nVectors + iDoF * d_nVectors + iVec].
       * @param[in] cellRange the range of cells for which extraction has to be
       * done.
       */
      void
      extractToCellNodalDataKernel(
        const dftfe::linearAlgebra::MultiVector<
          ValueTypeBasisCoeff,
          dftfe::utils::MemorySpace::DEVICE> &      nodalData,
        ValueTypeBasisCoeff *                       cellNodalDataPtr,
        const std::pair<unsigned int, unsigned int> cellRange) const;

      // FIXME Untested function
      /**
       * @brief Accumulate cell level nodal data into process level nodal data.
       * @param[in] cellNodalDataPtr Cell level nodal values, indexed by
       * [iCell * d_nDofsPerCell * d_nVectors + iDoF * d_nVectors + iVec].
       * @param[out] nodalData process level nodal data.
       * @param[in] cellRange the range of cells for which extraction has to be
       * done.
       */
      void
      accumulateFromCellNodalDataKernel(
        const ValueTypeBasisCoeff *cellNodalDataPtr,
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                          dftfe::utils::MemorySpace::DEVICE>
          &                                         nodalData,
        const std::pair<unsigned int, unsigned int> cellRange) const;
    };
#endif
  } // end of namespace basis
} // end of namespace dftfe
#include "../utils/FEBasisOperations.t.cc"
#include "../utils/FEBasisOperationsHost.t.cc"
#if defined(DFTFE_WITH_DEVICE)
#  include "../utils/FEBasisOperationsDevice.t.cc"
#endif

#endif // dftfeBasisOperations_h
