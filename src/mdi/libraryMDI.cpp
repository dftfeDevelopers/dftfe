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


int
MDI_Plugin_init_dftfe(void *plugin_state)
{
  MDI_Set_plugin_state(plugin_state);

  // Get the MPI intra-communicator for this code
  MPI_Comm mpi_world_comm = MPI_COMM_WORLD;
  MDI_MPI_get_world_comm(&mpi_world_comm);

  // open DFT-FE
  dftfe::dftfeWrapper::globalHandlesInitialize(mpi_world_comm);

  int    mdi_argc;
  char **mdi_argv;
  // launch MDI engine in endless loop
  dftfe::MDIEngine mdiEngine(mpi_world_comm, mdi_argc, mdi_argv);

  dftfe::dftfeWrapper::globalHandlesFinalize();

  return 0;
}

int
MDI_Plugin_open_dftfe(void *plugin_state)
{
  MDI_Set_plugin_state(plugin_state);

  // Get the MPI intra-communicator for this code
  MPI_Comm mpi_world_comm = MPI_COMM_WORLD;
  MDI_MPI_get_world_comm(&mpi_world_comm);

  // open DFT-FE
  dftfe::dftfeWrapper::globalHandlesInitialize(mpi_world_comm);

  int    mdi_argc;
  char **mdi_argv;
  // launch MDI engine in endless loop
  dftfe::MDIEngine mdiEngine(mpi_world_comm, mdi_argc, mdi_argv);

  return 0;
}


int
MDI_Plugin_close_dftfe()
{
  dftfe::dftfeWrapper::globalHandlesFinalize();
  return 0;
}


int
dftfe_execute_mdi_command(const char *command, MDI_Comm comm, void *class_obj)
{
  auto mdi_engine = (dftfe::MDIEngine *)class_obj;
  return mdi_engine->execute_command(command, comm);
}

#endif
