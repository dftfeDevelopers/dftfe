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

/**
 * @author Gourab Panigrahi
 *
 */

#ifndef MatrixFreeWrapper_H_
#define MatrixFreeWrapper_H_
#include <variant>
#include <memory>
#include <headers.h>
#include <FEBasisOperations.h>
#include <linearAlgebraOperations.h>
#include <MatrixFree.h>

namespace dftfe
{
  /**
   * @brief Datastructure to hold different MatrixFree class objects
   *
   * @author Gourab Panigrahi
   *
   */
  using MatrixFreeObject = std::variant<
#define MatrixFreeWrapperTemplates(NDOFSPERDIM)                          \
  std::shared_ptr<dftfe::MatrixFree<double,                              \
                                    double,                              \
                                    dftfe::operatorList::Laplace,        \
                                    dftfe::utils::MemorySpace::DEVICE,   \
                                    NDOFSPERDIM,                         \
                                    NDOFSPERDIM,                         \
                                    1,                                   \
                                    1>>,                                 \
    std::shared_ptr<dftfe::MatrixFree<double,                            \
                                      double,                            \
                                      dftfe::operatorList::Helmholtz,    \
                                      dftfe::utils::MemorySpace::DEVICE, \
                                      NDOFSPERDIM,                       \
                                      NDOFSPERDIM,                       \
                                      1,                                 \
                                      1>>,
#define MatrixFreeWrapperTemplatesL(NDOFSPERDIM)                         \
  std::shared_ptr<dftfe::MatrixFree<double,                              \
                                    double,                              \
                                    dftfe::operatorList::Laplace,        \
                                    dftfe::utils::MemorySpace::DEVICE,   \
                                    NDOFSPERDIM,                         \
                                    NDOFSPERDIM,                         \
                                    1,                                   \
                                    1>>,                                 \
    std::shared_ptr<dftfe::MatrixFree<double,                            \
                                      double,                            \
                                      dftfe::operatorList::Helmholtz,    \
                                      dftfe::utils::MemorySpace::DEVICE, \
                                      NDOFSPERDIM,                       \
                                      NDOFSPERDIM,                       \
                                      1,                                 \
                                      1>>
#include "MatrixFreeWrapper.def"
#undef MatrixFreeWrapperTemplates
#undef MatrixFreeWrapperTemplatesL
    >;

  /**
   * @brief Factory function to create MatrixFree object
   *
   */
  template <typename T,
            typename TypeFEBasis,
            dftfe::operatorList       operatorID,
            dftfe::utils::MemorySpace memorySpace,
            class... Args>
  inline MatrixFreeObject
  createMatrixFreeObject(dftfe::uInt nDofsPerDim, Args &&...args)
  {
    switch (nDofsPerDim)
      {
#define MatrixFreeWrapperTemplates(NDOFSPERDIM)       \
  case NDOFSPERDIM:                                   \
    return MatrixFreeObject(                          \
      std::make_shared<dftfe::MatrixFree<T,           \
                                         TypeFEBasis, \
                                         operatorID,  \
                                         memorySpace, \
                                         NDOFSPERDIM, \
                                         NDOFSPERDIM, \
                                         1,           \
                                         1>>(std::forward<Args>(args)...));
#define MatrixFreeWrapperTemplatesL(NDOFSPERDIM)      \
  case NDOFSPERDIM:                                   \
    return MatrixFreeObject(                          \
      std::make_shared<dftfe::MatrixFree<T,           \
                                         TypeFEBasis, \
                                         operatorID,  \
                                         memorySpace, \
                                         NDOFSPERDIM, \
                                         NDOFSPERDIM, \
                                         1,           \
                                         1>>(std::forward<Args>(args)...));
#include "MatrixFreeWrapper.def"
#undef MatrixFreeWrapperTemplates
#undef MatrixFreeWrapperTemplatesL
        default:
          throw std::logic_error{"createMatrixFreeObject dispatch failed"};
      }
  }

  /**
   * @brief MatrixFreeWrapper class
   *
   */
  template <typename T,
            typename TypeFEBasis,
            dftfe::operatorList       operatorID,
            dftfe::utils::MemorySpace memorySpace>
  class MatrixFreeWrapperClass
  {
  public:
    /// Constructor
    MatrixFreeWrapperClass(
      dftfe::uInt     nDofsPerDim,
      const MPI_Comm &mpi_comm,
      std::shared_ptr<
        dftfe::basis::FEBasisOperations<TypeFEBasis,
                                        double,
                                        dftfe::utils::MemorySpace::HOST>>
        basisOperationsPtrHost,
      std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
                        BLASWrapperPtr,
      const dftfe::uInt quadratureID,
      const dftfe::uInt nVectors)
      : d_MatrixFreeObject(
          createMatrixFreeObject<T, TypeFEBasis, operatorID, memorySpace>(
            nDofsPerDim,
            mpi_comm,
            basisOperationsPtrHost,
            BLASWrapperPtr,
            quadratureID,
            nVectors))
    {}

    /**
     * @brief Initialize data structures for MatrixFree class
     *
     */
    inline void
    init()
    {
      std::visit([&](auto &t) { t->init(); }, d_MatrixFreeObject);
    }

    /**
     * @brief Initialize Helmholtz operator coefficient
     *
     */
    inline void
    initOperatorCoeffs(T coeffHelmholtz)
    {
      std::visit([&](auto &t) { t->initOperatorCoeffs(coeffHelmholtz); },
                 d_MatrixFreeObject);
    }

    /**
     * @brief Compute Laplace operator multipled by X
     *
     */
    inline void
    computeAX(T *dst, T *src)
    {
      std::visit([&](auto &t) { t->computeAX(dst, src); }, d_MatrixFreeObject);
    }

    /**
     * @brief Distribute constraints on vector src
     *
     */
    inline void
    constraintsDistribute(T *src)
    {
      std::visit([&](auto &t) { t->constraintsDistribute(src); },
                 d_MatrixFreeObject);
    }

    /**
     * @brief Distribute transpose constraints on vector src
     *
     */
    inline void
    constraintsDistributeTranspose(T *dst, T *src)
    {
      std::visit([&](auto &t) { t->constraintsDistributeTranspose(dst, src); },
                 d_MatrixFreeObject);
    }

  private:
    MatrixFreeObject d_MatrixFreeObject;
  };

} // namespace dftfe
#endif // MatrixFreeWrapper_H_
