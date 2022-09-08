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
//

#if defined(DFTFE_WITH_MDI)
#  ifndef dftfeLibraryMDI_H_
#    define dftfeLibraryMDI_H_

/* C style library calls to DFTFE when a DFTFE shared library is
 *  used as a plugin through MolSSI Driver Interface (MDI). */

#    include <mdi.h>

extern "C"
{
  int
  MDI_Plugin_init_dftfe();
  int
  dftfe_execute_mdi_command(const char *, MDI_Comm, void *);
}
#  endif
#endif
