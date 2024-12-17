// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2025  The Regents of the University of Michigan and DFT-FE
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
    class FEBasisOperations
    {
    protected:
      mutable dftfe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace>
        tempCellNodalData, tempQuadratureGradientsData,
        tempQuadratureGradientsDataNonAffine;
      mutable dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace>
        tempCellGradientsBlock, tempCellGradientsBlock2, tempCellValuesBlock,
        tempCellMatrixBlock;
      mutable dftfe::utils::MemoryStorage<dftfe::global_size_type, memorySpace>
        zeroIndexVec;
      mutable dftfe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace>
        tempCellValuesBlockCoeff;
      std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
        d_BLASWrapperPtr;

    public:
      /**
       * @brief Constructor
       */
      FEBasisOperations(
        std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
          BLASWrapperPtr);


      /**
       * @brief Default Destructor
       */
      ~FEBasisOperations() = default;

      /**
       * @brief Clears the FEBasisOperations internal storage.
       */
      void
      clear();

      /**
       * @brief fills required data structures for the given dofHandlerID
       * @param[in] matrixFreeData MatrixFree object.
       * @param[in] constraintsVector std::vector of AffineConstraints, should
       * be the same vector which was passed for the construction of the given
       * MatrixFree object.
       * @param[in] dofHandlerID dofHandler index to be used for getting data
       * from the MatrixFree object.
       * @param[in] quadratureID std::vector of quadratureIDs to be used, should
       * be the same IDs which were used during the construction of the given
       * MatrixFree object.
       */
      void
        init(dealii::MatrixFree<3, ValueTypeBasisData> &matrixFreeData,
             std::vector<const dealii::AffineConstraints<ValueTypeBasisData> *>
               &                              constraintsVector,
             const unsigned int &             dofHandlerID,
             const std::vector<unsigned int> &quadratureID,
             const std::vector<UpdateFlags>   updateFlags);

      /**
       * @brief fills required data structures from another FEBasisOperations object
       * @param[in] basisOperationsSrc Source FEBasisOperations object.
       */
      template <dftfe::utils::MemorySpace memorySpaceSrc>
      void
      init(const FEBasisOperations<ValueTypeBasisCoeff,
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
             const bool          isResizeTempStorageForInerpolation = true,
             const bool          isResizeTempStorageForCellMatrices = false);

      dftfe::utils::MemoryStorage<dftfe::global_size_type,
                                  dftfe::utils::MemorySpace::HOST> &
      getFlattenedMapsHost();

      // private:


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
       * @brief Computes the cell-level stiffness matrix.
       */
      void
      computeCellStiffnessMatrix(const unsigned int quadratureID,
                                 const unsigned int cellsBlockSize,
                                 const bool         basisType = false,
                                 const bool         ceoffType = true);

      void
      computeCellMassMatrix(const unsigned int quadratureID,
                            const unsigned int cellsBlockSize,
                            const bool         basisType = false,
                            const bool         ceoffType = true);

      void
      computeWeightedCellMassMatrix(
        const std::pair<unsigned int, unsigned int> cellRangeTotal,
        dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace> &weights,
        dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace>
          &weightedCellMassMatrix) const;

      void
      computeWeightedCellNjGradNiMatrix(
        const std::pair<unsigned int, unsigned int> cellRangeTotal,
        dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace> &weights,
        dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace>
          &weightedCellNjGradNiMatrix) const;

      void
      computeWeightedCellNjGradNiPlusNiGradNjMatrix(
        const std::pair<unsigned int, unsigned int> cellRangeTotal,
        dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace> &weights,
        dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace>
          &weightedCellNjGradNiPlusNiGradNjMatrix) const;

      void
      computeInverseSqrtMassVector(const bool basisType = true,
                                   const bool ceoffType = false);

      /**
       * @brief Computes the stiffness matrix
       * \grad Ni . \grad Ni.
       */
      void
      computeStiffnessVector(const bool basisType = true,
                             const bool ceoffType = false);

      /**
       * @brief Resizes the internal temp storage to be sufficient for the vector and cell block sizes provided in reinit.
       */
      void
      resizeTempStorage(const bool isResizeTempStorageForInerpolation,
                        const bool isResizeTempStorageForCellMatrices);

      /**
       * @brief Number of quadrature points per cell for the quadratureID set in reinit.
       */
      unsigned int
      nQuadsPerCell() const;

      /**
       * @brief Number of vectors set in reinit.
       */
      unsigned int
      nVectors() const;
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
      const auto &
      shapeFunctionBasisData(bool transpose = false) const
      {
        if constexpr (std::is_same<ValueTypeBasisCoeff,
                                   ValueTypeBasisData>::value)
          {
            return transpose ?
                     d_shapeFunctionDataTranspose.find(d_quadratureID)->second :
                     d_shapeFunctionData.find(d_quadratureID)->second;
          }
        else
          {
            return transpose ?
                     d_shapeFunctionBasisDataTranspose.find(d_quadratureID)
                       ->second :
                     d_shapeFunctionBasisData.find(d_quadratureID)->second;
          }
      }
      /**
       * @brief Shape function gradient values at quadrature points in ValueTypeBasisData.
       * @param[in] transpose if false the the data is indexed as [iDim *
       * d_nQuadsPerCell * d_nDofsPerCell + iQuad * d_nDofsPerCell + iNode] and
       * if true it is indexed as [iDim * d_nQuadsPerCell * d_nDofsPerCell +
       * iNode * d_nQuadsPerCell + iQuad].
       */
      const auto &
      shapeFunctionGradientBasisData(bool transpose = false) const
      {
        if constexpr (std::is_same<ValueTypeBasisCoeff,
                                   ValueTypeBasisData>::value)
          {
            return transpose ?
                     d_shapeFunctionGradientDataTranspose.find(d_quadratureID)
                       ->second :
                     d_shapeFunctionGradientData.find(d_quadratureID)->second;
          }
        else
          {
            return transpose ?
                     d_shapeFunctionGradientBasisDataTranspose
                       .find(d_quadratureID)
                       ->second :
                     d_shapeFunctionGradientBasisData.find(d_quadratureID)
                       ->second;
          }
      }

      /**
       * @brief Inverse Jacobian matrices in ValueTypeBasisData, for cartesian cells returns the
       * diagonal elements of the inverse Jacobian matrices for each cell, for
       * affine cells returns the 3x3 inverse Jacobians for each cell otherwise
       * returns the 3x3 inverse Jacobians at each quad point for each cell.
       */
      const auto &
      inverseJacobiansBasisData() const
      {
        if constexpr (std::is_same<ValueTypeBasisCoeff,
                                   ValueTypeBasisData>::value)
          {
            return d_inverseJacobianData
              .find(areAllCellsAffine ? 0 : d_quadratureID)
              ->second;
          }
        else
          {
            return d_inverseJacobianBasisData
              .find(areAllCellsAffine ? 0 : d_quadratureID)
              ->second;
          }
      }

      /**
       * @brief determinant of Jacobian times the quadrature weight in ValueTypeBasisData at each
       * quad point for each cell.
       */
      const auto &
      JxWBasisData() const
      {
        if constexpr (std::is_same<ValueTypeBasisCoeff,
                                   ValueTypeBasisData>::value)
          {
            return d_JxWData.find(d_quadratureID)->second;
          }
        else
          {
            return d_JxWBasisData.find(d_quadratureID)->second;
          }
      }

      /**
       * @brief Cell level stiffness matrix in ValueTypeBasisCoeff
       */
      const auto &
      cellStiffnessMatrix() const
      {
        if constexpr (std::is_same<ValueTypeBasisCoeff,
                                   ValueTypeBasisData>::value)
          {
            return d_cellStiffnessMatrixBasisType;
          }
        else
          {
            return d_cellStiffnessMatrixCoeffType;
          }
      }


      /**
       * @brief Cell level stiffness matrix in ValueTypeBasisData
       */
      const dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace> &
      cellStiffnessMatrixBasisData() const;


      /**
       * @brief Cell level mass matrix in ValueTypeBasisCoeff
       */
      const auto &
      cellMassMatrix() const
      {
        if constexpr (std::is_same<ValueTypeBasisCoeff,
                                   ValueTypeBasisData>::value)
          {
            return d_cellMassMatrixBasisType;
          }
        else
          {
            return d_cellMassMatrixCoeffType;
          }
      }


      /**
       * @brief Cell level mass matrix in ValueTypeBasisData
       */
      const dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace> &
      cellMassMatrixBasisData() const;


      /**
       * @brief Cell level inverse sqrt diagonal mass matrix in ValueTypeBasisCoeff
       */
      const auto &
      cellInverseSqrtMassVector() const
      {
        if constexpr (std::is_same<ValueTypeBasisCoeff,
                                   ValueTypeBasisData>::value)
          {
            return d_cellInverseSqrtMassVectorBasisType;
          }
        else
          {
            return d_cellInverseSqrtMassVectorCoeffType;
          }
      }

      /**
       * @brief Cell level inverse diagonal mass matrix in ValueTypeBasisCoeff
       */
      const auto &
      cellInverseMassVector() const
      {
        if constexpr (std::is_same<ValueTypeBasisCoeff,
                                   ValueTypeBasisData>::value)
          {
            return d_cellInverseMassVectorBasisType;
          }
        else
          {
            return d_cellInverseMassVectorCoeffType;
          }
      }

      /**
       * @brief Cell level sqrt diagonal mass matrix in ValueTypeBasisCoeff
       */
      const auto &
      cellSqrtMassVector() const
      {
        if constexpr (std::is_same<ValueTypeBasisCoeff,
                                   ValueTypeBasisData>::value)
          {
            return d_cellSqrtMassVectorBasisType;
          }
        else
          {
            return d_cellSqrtMassVectorCoeffType;
          }
      }

      /**
       * @brief Cell level diagonal mass matrix in ValueTypeBasisCoeff
       */
      const auto &
      cellMassVector() const
      {
        if constexpr (std::is_same<ValueTypeBasisCoeff,
                                   ValueTypeBasisData>::value)
          {
            return d_cellMassVectorBasisType;
          }
        else
          {
            return d_cellMassVectorCoeffType;
          }
      }


      /**
       * @brief Inverse sqrt diagonal mass matrix in ValueTypeBasisCoeff
       */
      const auto &
      inverseSqrtMassVector() const
      {
        if constexpr (std::is_same<ValueTypeBasisCoeff,
                                   ValueTypeBasisData>::value)
          {
            return d_inverseSqrtMassVectorBasisType;
          }
        else
          {
            return d_inverseSqrtMassVectorCoeffType;
          }
      }



      /**
       * @brief Inverse sqrt diagonal mass matrix in ValueTypeBasisCoeff
       */
      const auto &
      inverseMassVector() const
      {
        if constexpr (std::is_same<ValueTypeBasisCoeff,
                                   ValueTypeBasisData>::value)
          {
            return d_inverseMassVectorBasisType;
          }
        else
          {
            return d_inverseMassVectorCoeffType;
          }
      }


      /**
       * @brief sqrt diagonal mass matrix in ValueTypeBasisCoeff
       */
      const auto &
      sqrtMassVector() const
      {
        if constexpr (std::is_same<ValueTypeBasisCoeff,
                                   ValueTypeBasisData>::value)
          {
            return d_sqrtMassVectorBasisType;
          }
        else
          {
            return d_sqrtMassVectorCoeffType;
          }
      }

      /**
       * @brief diagonal mass matrix in ValueTypeBasisCoeff
       */
      const auto &
      massVector() const
      {
        if constexpr (std::is_same<ValueTypeBasisCoeff,
                                   ValueTypeBasisData>::value)
          {
            return d_massVectorBasisType;
          }
        else
          {
            return d_massVectorCoeffType;
          }
      }


      /**
       * @brief Cell level inverse sqrt diagonal mass matrix in ValueTypeBasisData
       */
      const dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace> &
      cellInverseSqrtMassVectorBasisData() const;


      /**
       * @brief Cell level inverse  diagonal mass matrix in ValueTypeBasisData
       */
      const dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace> &
      cellInverseMassVectorBasisData() const;


      /**
       * @brief Cell level sqrt diagonal mass matrix in ValueTypeBasisData
       */
      const dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace> &
      cellSqrtMassVectorBasisData() const;


      /**
       * @brief Cell level diagonal mass matrix in ValueTypeBasisData
       */
      const dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace> &
      cellMassVectorBasisData() const;

      /**
       * @brief Inverse sqrt diagonal mass matrix in ValueTypeBasisData
       */
      const dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace> &
      inverseSqrtMassVectorBasisData() const;

      /**
       * @brief Inverse diagonal mass matrix in ValueTypeBasisData
       */
      const dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace> &
      inverseMassVectorBasisData() const;

      /**
       * @brief sqrt diagonal mass matrix in ValueTypeBasisData
       */
      const dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace> &
      sqrtMassVectorBasisData() const;

      /**
       * @brief diagonal mass matrix in ValueTypeBasisData
       */
      const dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace> &
      massVectorBasisData() const;

      /**
       * @brief diagonal stiffness matrix in ValueTypeBasisData
       */
      const dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace> &
      stiffnessVectorBasisData() const;

      /**
       * @brief diagonal inverse stiffness matrix in ValueTypeBasisData
       */
      const dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace> &
      inverseStiffnessVectorBasisData() const;

      /**
       * @brief diagonal inverse sqrt stiffness matrix in ValueTypeBasisData
       */
      const dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace> &
      inverseSqrtStiffnessVectorBasisData() const;

      /**
       * @brief diagonal sqrt stiffness matrix in ValueTypeBasisData
       */
      const dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace> &
      sqrtStiffnessVectorBasisData() const;

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
       * @brief Creates single precision scratch multivectors.
       * @param[in] vecBlockSize Number of vectors in the multivector.
       * @param[out] numMultiVecs number of scratch multivectors needed with
       * this vecBlockSize.
       */
      void
      createScratchMultiVectorsSinglePrec(
        const unsigned int vecBlockSize,
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
       * @brief Gets single precision scratch multivectors.
       * @param[in] vecBlockSize Number of vectors in the multivector.
       * @param[out] numMultiVecs index of the multivector among those with the
       * same vecBlockSize.
       */
      dftfe::linearAlgebra::MultiVector<
        typename dftfe::dataTypes::singlePrecType<ValueTypeBasisCoeff>::type,
        memorySpace> &
      getMultiVectorSinglePrec(const unsigned int vecBlockSize,
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

      /**
       * @brief Return the underlying deal.II dofhandler object.
       */
      const dealii::DoFHandler<3> &
      getDofHandler() const;



      std::vector<dftUtils::constraintMatrixInfo<memorySpace>> d_constraintInfo;
      unsigned int                                             d_nOMPThreads;
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

      dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace>
        d_cellStiffnessMatrixBasisType;
      dftfe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace>
        d_cellStiffnessMatrixCoeffType;
      dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace>
        d_cellMassMatrixBasisType;
      dftfe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace>
        d_cellMassMatrixCoeffType;
      dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace>
        d_cellInverseMassVectorBasisType;
      dftfe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace>
        d_cellInverseMassVectorCoeffType;
      dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace>
        d_cellInverseSqrtMassVectorBasisType;
      dftfe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace>
        d_cellInverseSqrtMassVectorCoeffType;
      dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace>
        d_cellMassVectorBasisType;
      dftfe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace>
        d_cellMassVectorCoeffType;
      dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace>
        d_cellSqrtMassVectorBasisType;
      dftfe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace>
        d_cellSqrtMassVectorCoeffType;
      dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace>
        d_massVectorBasisType;
      dftfe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace>
        d_massVectorCoeffType;
      dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace>
        d_inverseMassVectorBasisType;
      dftfe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace>
        d_inverseMassVectorCoeffType;
      dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace>
        d_inverseSqrtMassVectorBasisType;
      dftfe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace>
        d_inverseSqrtMassVectorCoeffType;
      dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace>
        d_sqrtMassVectorBasisType;
      dftfe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace>
        d_sqrtMassVectorCoeffType;
      mutable std::map<
        unsigned int,
        std::vector<
          dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>>>
        scratchMultiVectors;

      dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace>
        d_cellStiffnessVectorBasisType;
      dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace>
        d_cellInverseStiffnessVectorBasisType;
      dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace>
        d_cellSqrtStiffnessVectorBasisType;
      dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace>
        d_cellInverseSqrtStiffnessVectorBasisType;
      dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace>
        d_inverseSqrtStiffnessVectorBasisType;
      dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace>
        d_sqrtStiffnessVectorBasisType;
      dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace>
        d_inverseStiffnessVectorBasisType;
      dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace>
        d_stiffnessVectorBasisType;

      dftfe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace>
        d_cellInverseStiffnessVectorCoeffType;
      dftfe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace>
        d_cellInverseSqrtStiffnessVectorCoeffType;
      dftfe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace>
        d_cellStiffnessVectorCoeffType;
      dftfe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace>
        d_cellSqrtStiffnessVectorCoeffType;
      dftfe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace>
        d_inverseSqrtStiffnessVectorCoeffType;
      dftfe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace>
        d_sqrtStiffnessVectorCoeffType;
      dftfe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace>
        d_stiffnessVectorCoeffType;
      dftfe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace>
        d_inverseStiffnessVectorCoeffType;


      mutable std::map<
        unsigned int,
        std::vector<dftfe::linearAlgebra::MultiVector<
          typename dftfe::dataTypes::singlePrecType<ValueTypeBasisCoeff>::type,
          memorySpace>>>
        scratchMultiVectorsSinglePrec;

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
      interpolate(dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                                    memorySpace> &nodalData,
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
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
          &nodalData,
        dftfe::utils::MemoryStorage<dftfe::global_size_type, memorySpace>
          &mapQuadIdToProcId) const;

      /**
       * @brief Get cell level nodal data from process level nodal data.
       * @param[in] nodalData process level nodal data, the multivector should
       * already have ghost data and constraints should have been applied.
       * @param[out] cellNodalDataPtr Cell level nodal values, indexed by
       * [iCell * d_nDofsPerCell * d_nVectors + iDoF * d_nVectors + iVec].
       */
      void
      extractToCellNodalData(
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
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
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
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
                                                memorySpace> &nodalData,
        ValueTypeBasisCoeff *                                 quadratureValues,
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
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
          &nodalData,
        dftfe::utils::MemoryStorage<dftfe::global_size_type, memorySpace>
          &                                         mapQuadIdToProcId,
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
                                                memorySpace> &nodalData,
        ValueTypeBasisCoeff *                                 cellNodalDataPtr,
        const std::pair<unsigned int, unsigned int>           cellRange) const;

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
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
          &                                         nodalData,
        const std::pair<unsigned int, unsigned int> cellRange) const;
    };
  } // end of namespace basis
} // end of namespace dftfe

#endif // dftfeBasisOperations_h
