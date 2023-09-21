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

#ifndef DFTFE_MPIWRITEONFILE_H
#define DFTFE_MPIWRITEONFILE_H

#include "CompositeData.h"
#include <string>
#include <vector>
namespace dftfe
{
  namespace dftUtils
  {
    class MPIWriteOnFile
    {
    public:
      static void
      writeData(const std::vector<CompositeData *> &data,
                const std::string &                 fileName,
                const MPI_Comm &                    mpiComm);
    };
  } // namespace dftUtils
} // namespace dftfe

#endif // DFTFE_MPIWRITEONFILE_H
