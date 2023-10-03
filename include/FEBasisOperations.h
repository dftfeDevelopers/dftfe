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

namespace dftfe
{
  namespace basis
  {
    enum UpdateFlags
    {
      update_default = 0,

      update_values = 0x0001,

      update_gradients = 0x0002,

      update_transpose = 0x0004
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

    public:
      FEBasisOperationsBase(
        dealii::MatrixFree<3, ValueTypeBasisData> &matrixFreeData,
        std::vector<const dealii::AffineConstraints<ValueTypeBasisData> *>
          &constraintsVector);

      ~FEBasisOperationsBase() = default;

      void
      init(const unsigned int &             dofHandlerID,
           const std::vector<unsigned int> &quadratureID,
           const UpdateFlags                updateFlags = update_values);

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



      void
      initializeIndexMaps();
      void
      initializeFlattenedIndexMaps();

      void
      initializeConstraints();

      void
      initializeShapeFunctionAndJacobianData();

      void
      resizeTempStorage();

      unsigned int
      nQuadsPerCell() const;

      unsigned int
      nDofsPerCell() const;

      unsigned int
      nCells() const;

      unsigned int
      nRelaventDofs() const;

      unsigned int
      nOwnedDofs() const;

      const ValueTypeBasisCoeff *
      shapeFunctionData(bool transpose = false) const;

      const ValueTypeBasisCoeff *
      shapeFunctionGradientData(bool transpose = false) const;

      const ValueTypeBasisCoeff *
      inverseJacobians() const;

      const ValueTypeBasisCoeff *
      JxW() const;

      unsigned int
      cellsTypeFlag() const;


      void
      createMultiVector(
        const unsigned int dofHandlerIndex,
        const unsigned int blocksize,
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
          &multiVector) const;

      void
      distribute(
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
          &multiVector) const;



      constraintInfoClass d_constraintInfo;
      std::vector<const dealii::AffineConstraints<ValueTypeBasisData> *>
        *                                              d_constraintsVector;
      const dealii::MatrixFree<3, ValueTypeBasisData> *d_matrixFreeDataPtr;
      dftfe::utils::MemoryStorage<dftfe::global_size_type,
                                  dftfe::utils::MemorySpace::HOST>
        d_cellDofIndexToProcessDofIndexMap;
      dftfe::utils::MemoryStorage<dftfe::global_size_type, memorySpace>
                                  d_flattenedCellDofIndexToProcessDofIndexMap;
      std::vector<dealii::CellId> d_cellIndexToCellIdMap;
      std::vector<dftfe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace>>
        d_inverseJacobianData;
      std::vector<dftfe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace>>
        d_JxWData;
      std::vector<dftfe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace>>
        d_shapeFunctionData;
      std::vector<dftfe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace>>
        d_shapeFunctionGradientDataInternalLayout;
      std::vector<dftfe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace>>
        d_shapeFunctionGradientData;
      std::vector<dftfe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace>>
        d_shapeFunctionDataTranspose;
      std::vector<dftfe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace>>
        d_shapeFunctionGradientDataTranspose;

      std::vector<unsigned int> d_quadratureIDsVector;
      unsigned int              d_quadratureID;
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
      UpdateFlags               d_updateFlags;
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
        dftfe::utils::MemorySpace::HOST>::d_quadratureID;
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


      void
      interpolate(
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                          dftfe::utils::MemorySpace::HOST>
          &                  nodalData,
        ValueTypeBasisCoeff *quadratureValues,
        ValueTypeBasisCoeff *quadratureGradients = NULL) const;


      void
      integrateWithBasis(
        ValueTypeBasisCoeff *quadratureValues,
        ValueTypeBasisCoeff *quadratureGradients,
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                          dftfe::utils::MemorySpace::HOST>
          &nodalData) const;

      void
      extractToCellNodalData(
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                          dftfe::utils::MemorySpace::HOST>
          &                  nodalData,
        ValueTypeBasisCoeff *cellNodalDataPtr) const;

      void
      accumulateFromCellNodalData(
        const ValueTypeBasisCoeff *cellNodalDataPtr,
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                          dftfe::utils::MemorySpace::HOST>
          &nodalData) const;

      void
      interpolateKernel(
        const dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                                dftfe::utils::MemorySpace::HOST>
          &                                         nodalData,
        ValueTypeBasisCoeff *                       quadratureValues,
        ValueTypeBasisCoeff *                       quadratureGradients,
        const std::pair<unsigned int, unsigned int> cellRange) const;
      void
      interpolateKernel(
        const ValueTypeBasisCoeff *                 nodalData,
        ValueTypeBasisCoeff *                       quadratureValues,
        ValueTypeBasisCoeff *                       quadratureGradients,
        const std::pair<unsigned int, unsigned int> cellRange) const;

      void
      integrateWithBasisKernel(
        const ValueTypeBasisCoeff *quadratureValues,
        const ValueTypeBasisCoeff *quadratureGradients,
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                          dftfe::utils::MemorySpace::HOST>
          &                                         nodalData,
        const std::pair<unsigned int, unsigned int> cellRange) const;


      void
      extractToCellNodalDataKernel(
        const dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                                dftfe::utils::MemorySpace::HOST>
          &                                         nodalData,
        ValueTypeBasisCoeff *                       cellNodalDataPtr,
        const std::pair<unsigned int, unsigned int> cellRange) const;

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

      dftfe::utils::deviceBlasHandle_t *d_deviceBlasHandlePtr;
      void
      setDeviceBLASHandle(
        dftfe::utils::deviceBlasHandle_t *deviceBlasHandlePtr);

      dftfe::utils::deviceBlasHandle_t &
      getDeviceBLASHandle();



      void
      interpolate(
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                          dftfe::utils::MemorySpace::DEVICE>
          &                  nodalData,
        ValueTypeBasisCoeff *quadratureValues,
        ValueTypeBasisCoeff *quadratureGradients = NULL) const;


      void
      integrateWithBasis(
        ValueTypeBasisCoeff *quadratureValues,
        ValueTypeBasisCoeff *quadratureGradients,
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                          dftfe::utils::MemorySpace::DEVICE>
          &nodalData) const;

      void
      extractToCellNodalData(
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                          dftfe::utils::MemorySpace::DEVICE>
          &                  nodalData,
        ValueTypeBasisCoeff *cellNodalDataPtr) const;

      void
      accumulateFromCellNodalData(
        const ValueTypeBasisCoeff *cellNodalDataPtr,
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                          dftfe::utils::MemorySpace::DEVICE>
          &nodalData) const;

      void
      interpolateKernel(
        const dftfe::linearAlgebra::MultiVector<
          ValueTypeBasisCoeff,
          dftfe::utils::MemorySpace::DEVICE> &      nodalData,
        ValueTypeBasisCoeff *                       quadratureValues,
        ValueTypeBasisCoeff *                       quadratureGradients,
        const std::pair<unsigned int, unsigned int> cellRange) const;
      void
      interpolateKernel(
        const ValueTypeBasisCoeff *                 nodalData,
        ValueTypeBasisCoeff *                       quadratureValues,
        ValueTypeBasisCoeff *                       quadratureGradients,
        const std::pair<unsigned int, unsigned int> cellRange) const;

      void
      integrateWithBasisKernel(
        const ValueTypeBasisCoeff *quadratureValues,
        const ValueTypeBasisCoeff *quadratureGradients,
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                          dftfe::utils::MemorySpace::DEVICE>
          &                                         nodalData,
        const std::pair<unsigned int, unsigned int> cellRange) const;


      void
      extractToCellNodalDataKernel(
        const dftfe::linearAlgebra::MultiVector<
          ValueTypeBasisCoeff,
          dftfe::utils::MemorySpace::DEVICE> &      nodalData,
        ValueTypeBasisCoeff *                       cellNodalDataPtr,
        const std::pair<unsigned int, unsigned int> cellRange) const;

      void
      accumulateFromCellNodalDataKernel(
        const ValueTypeBasisCoeff *cellNodalDataPtr,
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                          dftfe::utils::MemorySpace::DEVICE>
          &                                         nodalData,
        const std::pair<unsigned int, unsigned int> cellRange) const;
    };
#endif

    template class FEBasisOperationsBase<dataTypes::number,
                                         double,
                                         dftfe::utils::MemorySpace::HOST>;
#ifdef USE_COMPLEX
    template class FEBasisOperationsBase<double,
                                         double,
                                         dftfe::utils::MemorySpace::HOST>;
#endif
#if defined(DFTFE_WITH_DEVICE)
    template class FEBasisOperationsBase<dataTypes::number,
                                         double,
                                         dftfe::utils::MemorySpace::DEVICE>;
#  ifdef USE_COMPLEX
    template class FEBasisOperationsBase<double,
                                         double,
                                         dftfe::utils::MemorySpace::DEVICE>;
#  endif
#endif

    template class FEBasisOperations<dataTypes::number,
                                     double,
                                     dftfe::utils::MemorySpace::HOST>;
#ifdef USE_COMPLEX
    template class FEBasisOperations<double,
                                     double,
                                     dftfe::utils::MemorySpace::HOST>;
#endif
#if defined(DFTFE_WITH_DEVICE)
    template class FEBasisOperations<dataTypes::number,
                                     double,
                                     dftfe::utils::MemorySpace::DEVICE>;
#  ifdef USE_COMPLEX
    template class FEBasisOperations<double,
                                     double,
                                     dftfe::utils::MemorySpace::DEVICE>;
#  endif
#endif

  } // end of namespace basis

} // end of namespace dftfe
#endif // dftfeBasisOperations_h
