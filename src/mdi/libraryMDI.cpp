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
// @author Sambit Das
//

#if defined(DFTFE_WITH_MDI)
// ----------------------------------------------------------------------
// MolSSI Driver Interface functions
// these are added to DFTFE library interface when MDI package is included
// ----------------------------------------------------------------------

#  include "libraryMDI.h"
#  include "MDIEngine.h"
#  include "dftfeWrapper.h"

#  include <cstring>

/* ---------------------------------------------------------------------- */

/** Initialize an instance of DFTFE as an MDI plugin
 *
\verbatim embed:rst

This function is called by the MolSSI Driver Interface library (MDI)
when DFTFE is run as a plugin, and should not otherwise be used.

The function initializes MDI, then creates and initializes an instance
of DFTFE.  The command-line arguments ``argc`` and ``argv`` used to
initialize DFTFE are recieved from MDI.  The DFTFE instance runs an
input file, which must include the ``mdi/engine`` command; when DFTFE
executes this command, it will begin listening for commands from the
driver.  The name of the input file is obtained from the ``-in``
command-line argument, which must be provided by the MDI driver.

\endverbatim
 * \param  command    string buffer corresponding to the command to be executed
 * \param  comm       MDI communicator that can be used to communicated with the
driver.
 * \param  class_obj  pointer to an instance of an mdi/engine fix cast to ``void
*``.
 * \return 0 on no error. */

int
MDI_Plugin_init_dftfe()
{
  // initialize MDI

  int    mdi_argc;
  char **mdi_argv;
  if (MDI_Plugin_get_argc(&mdi_argc))
    MPI_Abort(MPI_COMM_WORLD, 1);
  if (MDI_Plugin_get_argv(&mdi_argv))
    MPI_Abort(MPI_COMM_WORLD, 1);
  if (MDI_Init(&mdi_argc, &mdi_argv))
    MPI_Abort(MPI_COMM_WORLD, 1);

  // get the MPI intra-communicator for this code

  MPI_Comm mpi_world_comm = MPI_COMM_WORLD;
  if (MDI_MPI_get_world_comm(&mpi_world_comm))
    MPI_Abort(MPI_COMM_WORLD, 1);

  // open DFT-FE
  // dftfe::dftfeWrapper::globalHandlesInitialize();

  // launch MDI engine in endless loop
  dftfe::MDIEngine mdiEngine(mpi_world_comm, mdi_argc, mdi_argv);

  // close DFT-FE
  dftfe::dftfeWrapper::globalHandlesFinalize();

  return 0;
}

/* ---------------------------------------------------------------------- */

/** Execute an MDI command
 *
\verbatim embed:rst

This function is called by the MolSSI Driver Interface Library (MDI)
when DFTFE is run as a plugin, and should not otherwise be used.
The function executes a single command from an external MDI driver.

\endverbatim
 * \param  command   string buffer corresponding to the command to be executed
 * \param  comm      MDI communicator that can be used to communicated with the
driver.
 * \param  class_obj pointer to an instance of an mdi/engine fix cast to ``void
*``.
 * \return 0 on no error, 1 on error. */

int
dftfe_execute_mdi_command(const char *command, MDI_Comm comm, void *class_obj)
{
  auto mdi_engine = (dftfe::MDIEngine *)class_obj;
  return mdi_engine->execute_command(command, comm);
}

#endif
