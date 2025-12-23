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

/*
 * @author Gourab Panigrahi
 */

#ifndef MatrixFreeHandle_H_
#define MatrixFreeHandle_H_

namespace dftfe
{
  /**
   * @brief MatrixFreeHandle struct
   *
   * @author Gourab Panigrahi
   *
   */
  // Thin non-virtual type-erased handle for MatrixFree<T...> instances
  struct MatrixFreeHandle
  {
    void *impl = nullptr;

    using InitFn           = void (*)(void *impl);
    using ComputeAXFn      = void (*)(void *impl, double *dst, double *src);
    using ComputeAXCoeffFn = void (*)(void        *impl,
                                      double      *dst,
                                      double      *src,
                                      const double coeffHelmholtz);
    using DestroyFn        = void (*)(void *impl);

    InitFn           init_fn           = nullptr;
    ComputeAXFn      computeAX_fn      = nullptr;
    ComputeAXCoeffFn computeAXcoeff_fn = nullptr;
    DestroyFn        destroy_fn        = nullptr;

    MatrixFreeHandle() = default;

    // RAII: destroy MatrixFreeTemplate impl when handle goes out of scope
    ~MatrixFreeHandle()
    {
      if (impl && destroy_fn)
        destroy_fn(impl);
    }

    // convenient callers (no virtual overhead)
    void
    init()
    {
      init_fn(impl);
    }

    inline void
    computeAX(double *dst, double *src) const
    {
      computeAX_fn(impl, dst, src);
    }

    inline void
    computeAX(double *dst, double *src, const double coeffHelmholtz) const
    {
      // use coefficient variant if provided else fall back to plain computeAX
      if (computeAXcoeff_fn)
        computeAXcoeff_fn(impl, dst, src, coeffHelmholtz);
      else
        computeAX_fn(impl, dst, src);
    }
  };

  // Helpers to create a handle from a MatrixFreeTemplate (MatrixFree<T...>)
  // pointer.
  template <typename MatrixFreeTemplate>
  void
  destroy_impl(void *impl)
  {
    delete static_cast<MatrixFreeTemplate *>(impl);
  }

  template <typename MatrixFreeTemplate>
  void
  init_impl(void *impl)
  {
    static_cast<MatrixFreeTemplate *>(impl)->init();
  }

  template <typename MatrixFreeTemplate>
  void
  computeAX_impl(void *impl, double *dst, double *src)
  {
    static_cast<MatrixFreeTemplate *>(impl)->computeAX(dst, src);
  }

  template <typename MatrixFreeTemplate>
  void
  computeAXcoeff_impl(void        *impl,
                      double      *dst,
                      double      *src,
                      const double coeffHelmholtz)
  {
    // static_cast<MatrixFreeTemplate *>(impl)->computeAX(dst, src,
    // coeffHelmholtz);
  }

  template <typename MatrixFreeTemplate>
  MatrixFreeHandle
  make_matrix_free_handle(MatrixFreeTemplate *ptr)
  {
    MatrixFreeHandle h;
    h.impl              = ptr;
    h.init_fn           = &init_impl<MatrixFreeTemplate>;
    h.computeAX_fn      = &computeAX_impl<MatrixFreeTemplate>;
    h.computeAXcoeff_fn = &computeAXcoeff_impl<MatrixFreeTemplate>;
    h.destroy_fn        = &destroy_impl<MatrixFreeTemplate>;
    return h;
  }
} // namespace dftfe
#endif // MatrixFreeHandle_H_
