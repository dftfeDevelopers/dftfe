//
// -------------------------------------------------------------------------------------
//
// Copyright (c) 2017-2023 The Regents of the University of Michigan and DFT-FE
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
// @author nelrufus, vishal subramanian
//

#ifndef DFTFE_COMPOSITEDATA_H
#define DFTFE_COMPOSITEDATA_H

#include <mpi.h>

namespace dftfe
{
  namespace dftUtils
  {
    class CompositeData
    {
    public:
      CompositeData() = default;

      virtual void
      getCharArray(char *data) = 0;

      virtual void
      getMPIDataType(MPI_Datatype *mpi_datatype) = 0;

      virtual int
      getNumberCharsPerCompositeData() = 0;
    };
  } // namespace dftUtils
} // namespace dftfe
#endif // DFTFE_COMPOSITEDATA_H
