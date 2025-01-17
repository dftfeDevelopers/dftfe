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

#if defined(DFTFE_WITH_DEVICE)
#  ifndef deviceKernelsGeneric_H_
#    define deviceKernelsGeneric_H_

#    include <dftfeDataTypes.h>
#    include <MemorySpaceType.h>
#    include <headers.h>
#    include <TypeConfig.h>
#    include <DeviceTypeConfig.h>

namespace dftfe
{
  namespace utils
  {
    namespace deviceKernelsGeneric
    {
      void
      setupDevice();

      template <typename ValueTypeComplex, typename ValueTypeReal>
      void
      copyComplexArrToRealArrsDevice(const dftfe::size_type  size,
                                     const ValueTypeComplex *complexArr,
                                     ValueTypeReal *         realArr,
                                     ValueTypeReal *         imagArr);


      template <typename ValueTypeComplex, typename ValueTypeReal>
      void
      copyRealArrsToComplexArrDevice(const dftfe::size_type size,
                                     const ValueTypeReal *  realArr,
                                     const ValueTypeReal *  imagArr,
                                     ValueTypeComplex *     complexArr);



      template <typename ValueType>
      void
      sadd(ValueType *            y,
           ValueType *            x,
           const ValueType        beta,
           const dftfe::size_type size);


      // This kernel interpolates the nodal data to quad data
      // The function takes the cell level nodal data
      // and interpolates it to the quad data in each cell
      // by multiplying with the shape function
      template <typename ValueType1, typename ValueType2>
      void
      interpolateNodalDataToQuadDevice(
        const dftfe::size_type numDofsPerElem,
        const dftfe::size_type numQuadPoints,
        const dftfe::size_type numVecs,
        const ValueType2 *     parentShapeFunc,
        const ValueType1 *     mapPointToCellIndex,
        const ValueType1 *     mapPointToProcLocal,
        const ValueType1 *     mapPointToShapeFuncIndex,
        const ValueType2 *     parentNodalValues,
        ValueType2 *           quadValues);

    } // namespace deviceKernelsGeneric
  }   // namespace utils
} // namespace dftfe

#  endif
#endif
