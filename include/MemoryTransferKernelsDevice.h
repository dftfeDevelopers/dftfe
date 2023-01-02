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

/*
 * @author Sambit Das
 */

#ifndef dftfeMemoryTransferKernelsDevice_h
#define dftfeMemoryTransferKernelsDevice_h

#ifdef DFTFE_WITH_DEVICE
#  include <TypeConfig.h>

namespace dftfe
{
  namespace utils
  {
    namespace memoryTransferKernelsDevice
    {
      /**
       * @brief Copy array from device to host
       * @param count The memory size in bytes of the array
       */
      void
      deviceMemcpyD2H(void *dst, const void *src, std::size_t count);

      /**
       * @brief Copy array from device to device
       * @param count The memory size in bytes of the array
       */
      void
      deviceMemcpyD2D(void *dst, const void *src, std::size_t count);

      /**
       * @brief Copy array from host to device
       * @param count The memory size in bytes of the array
       */
      void
      deviceMemcpyH2D(void *dst, const void *src, std::size_t count);

    }; // namespace memoryTransferKernelsDevice
  }    // namespace utils
} // namespace dftfe

#endif // DFTFE_WITH_DEVICE
#endif // dftfeMemoryTransferKernels_h
