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
   * @brief MatrixFreeWrapper class
   *
   * @author Gourab Panigrahi
   *
   */

  using MatrixFreeObject = std::variant<
#define MatrixFreeWrapperTemplates(T) \
  std::shared_ptr<                    \
    dftfe::MatrixFree<double, dftfe::utils::MemorySpace::DEVICE, T, T, 1, 1>>,
#define MatrixFreeWrapperTemplatesL(T) \
  std::shared_ptr<                     \
    dftfe::MatrixFree<double, dftfe::utils::MemorySpace::DEVICE, T, T, 1, 1>>
#include "MatrixFreeWrapper.def"
#undef MatrixFreeWrapperTemplates
#undef MatrixFreeWrapperTemplatesL
    >;


  constexpr dftfe::uInt
  encodeKey(const dftfe::uInt a, const dftfe::uInt b)
  {
    return a * 100 + b;
  }


  template <class... Args>
  inline MatrixFreeObject
  createMatrixFreeObject(dftfe::uInt floatingPointID,
                         dftfe::uInt nDofsPerDim,
                         Args &&...args)
  {
    const dftfe::uInt key = encodeKey(floatingPointID, nDofsPerDim);

    if (floatingPointID == 1)
      {
        switch (key)
          {
#define MatrixFreeWrapperTemplates(T)                                         \
  case encodeKey(1, T):                                                       \
    return MatrixFreeObject(                                                  \
      std::make_shared<                                                       \
        dftfe::                                                               \
          MatrixFree<double, dftfe::utils::MemorySpace::DEVICE, T, T, 1, 1>>( \
        std::forward<Args>(args)...));
#define MatrixFreeWrapperTemplatesL(T)                                        \
  case encodeKey(1, T):                                                       \
    return MatrixFreeObject(                                                  \
      std::make_shared<                                                       \
        dftfe::                                                               \
          MatrixFree<double, dftfe::utils::MemorySpace::DEVICE, T, T, 1, 1>>( \
        std::forward<Args>(args)...));
#include "MatrixFreeWrapper.def"
#undef MatrixFreeWrapperTemplates
#undef MatrixFreeWrapperTemplatesL
            default:
              throw std::logic_error{"createMatrixFreeObject dispatch failed"};
          }
      }
  }


  template <typename T, dftfe::utils::MemorySpace memorySpace>
  class MatrixFreeWrapperClass
  {
  public:
    /// Constructor
    MatrixFreeWrapperClass(
      dftfe::uInt     floatingPointID,
      dftfe::uInt     nDofsPerDim,
      const MPI_Comm &mpi_comm,
      std::shared_ptr<
        dftfe::basis::FEBasisOperations<dataTypes::number,
                                        double,
                                        dftfe::utils::MemorySpace::HOST>>
        basisOperationsPtrHost,
      std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
                        BLASWrapperPtr,
      const dftfe::uInt operatorID,
      const dftfe::uInt quadratureID,
      const dftfe::uInt nVectors)
      : d_MatrixFreeObject(createMatrixFreeObject(floatingPointID,
                                                  nDofsPerDim,
                                                  mpi_comm,
                                                  basisOperationsPtrHost,
                                                  BLASWrapperPtr,
                                                  operatorID,
                                                  quadratureID,
                                                  nVectors))
    {}

    void
    init()
    {
      std::visit([&](auto &t) { t->init(); }, d_MatrixFreeObject);
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

    inline void
    constraintsDistribute(T *src)
    {
      std::visit([&](auto &t) { t->constraintsDistribute(src); },
                 d_MatrixFreeObject);
    }

    inline void
    constraintsDistributeTranspose(T *dst, T *src)
    {
      std::visit([&](auto &t) { t->constraintsDistributeTranspose(dst, src); },
                 d_MatrixFreeObject);
    }

    inline void
    constraintsSetZero(T *src)
    {
      std::visit([&](auto &t) { t->constraintsSetZero(src); },
                 d_MatrixFreeObject);
    }

  private:
    MatrixFreeObject d_MatrixFreeObject;
  };

} // namespace dftfe
#endif // MatrixFreeWrapper_H_
