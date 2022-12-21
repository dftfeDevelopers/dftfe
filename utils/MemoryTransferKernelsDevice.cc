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
 * @author Sambit Das.
 */

#ifdef DFTFE_WITH_DEVICE
#  include <MemoryTransferKernelsDevice.h>
#  include <DeviceAPICalls.h>

namespace dftfe
{
  namespace utils
  {
    namespace memoryTransferKernelsDevice
    {
      void
      deviceMemcpyD2H(void *dst, const void *src, size_type count)
      {
        dftfe::utils::deviceMemcpyD2H(dst, src, count);
      }

      void
      deviceMemcpyH2D(void *dst, const void *src, size_type count)
      {
        dftfe::utils::deviceMemcpyH2D(dst, src, count);
      }

      void
      deviceMemcpyD2D(void *dst, const void *src, size_type count)
      {
        dftfe::utils::deviceMemcpyD2D(dst, src, count);
      }


    } // namespace memoryTransferKernelsDevice
  }   // namespace utils
} // namespace dftfe
#endif
