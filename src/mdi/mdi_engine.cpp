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

#include "mdi_engine.h"
#include "library_mdi.h"


#include <cstring>
#include <limits>

#include <mdi.h>

namespace dftfe
{

  // per-atom data which engine commands access

  enum { TYPE, COORD};

#define MAXELEMENT 103           // used elsewhere in MDI package

  /* ----------------------------------------------------------------------
     trigger DFT-FE to start acting as an MDI engine
     either in standalone mode or plugin mode
       MDI_Init() for standalone mode is in main.cpp
       MDI_Init() for plugin mode is in library_mdi.cpp::MDI_Plugin_init_dftfe()
     endlessly loop over receiving commands from driver and responding
     when EXIT command is received, mdi engine command exits
  ---------------------------------------------------------------------- */

  MDIEngine::MDIEngine(int argc, char *argv[])
  {

  }

  /* ----------------------------------------------------------------------
     engine is now at this MDI node
     loop over received commands so long as driver is also at this node
     return when not the case or EXIT command received
  ---------------------------------------------------------------------- */

  void MDIEngine::engine_node(const char *node)
  {
    int ierr;

    // do not process commands if engine and driver request are not the same

    strncpy(node_engine, node, MDI_COMMAND_LENGTH);

    if (strcmp(node_driver, "\0") != 0 && strcmp(node_driver, node_engine) != 0) node_match = false;

    // respond to commands from the driver

    while (!exit_command && node_match) {

      // read the next command from the driver
      // all procs call this, but only proc 0 receives the command

      ierr = MDI_Recv_command(mdicmd, mdicomm);
      if (ierr) error->all(FLERR, "MDI: Unable to receive command from driver");

      // broadcast command to the other MPI tasks

      MPI_Bcast(mdicmd, MDI_COMMAND_LENGTH, MPI_CHAR, 0, world);

      // execute the command

      execute_command(mdicmd, mdicomm);

      // check if driver request is now different than engine node

      if (strcmp(node_driver, "\0") != 0 && strcmp(node_driver, node_engine) != 0) node_match = false;
    }

    // node exit was triggered so reset node_match

    node_match = true;
  }

  /* ----------------------------------------------------------------------
     process a single driver command
     called by engine_node() in loop
     also called by MDI itself via lib::lammps_execute_mdi_command()
       when LAMMPS is running as a plugin
  ---------------------------------------------------------------------- */

  int MDIEngine::execute_command(const char *command, MDI_Comm mdicomm)
  {
  }

  /* ----------------------------------------------------------------------
     define which MDI commands the LAMMPS engine recognizes at each node
     both standard MDI commands and custom LAMMPS commands
     max length for a command is currently 11 chars
  ---------------------------------------------------------------------- */

  void MDIEngine::mdi_commands()
  {
    // default node, MDI standard commands

    MDI_Register_node("@DEFAULT");
    MDI_Register_command("@DEFAULT", "<@");
    MDI_Register_command("@DEFAULT", "<CELL");
    MDI_Register_command("@DEFAULT", "<CELL_DISPL");
    MDI_Register_command("@DEFAULT", "<CHARGES");
    MDI_Register_command("@DEFAULT", "<COORDS");
    MDI_Register_command("@DEFAULT", "<ENERGY");
    MDI_Register_command("@DEFAULT", "<FORCES");
    MDI_Register_command("@DEFAULT", "<LABELS");
    MDI_Register_command("@DEFAULT", "<MASSES");
    MDI_Register_command("@DEFAULT", "<NATOMS");
    MDI_Register_command("@DEFAULT", "<PE");
    MDI_Register_command("@DEFAULT", "<STRESS");
    MDI_Register_command("@DEFAULT", "<TYPES");
    MDI_Register_command("@DEFAULT", "<VELOCITIES");
    MDI_Register_command("@DEFAULT", ">CELL");
    MDI_Register_command("@DEFAULT", ">CELL_DISPL");
    MDI_Register_command("@DEFAULT", ">CHARGES");
    MDI_Register_command("@DEFAULT", ">COORDS");
    MDI_Register_command("@DEFAULT", ">ELEMENTS");
    MDI_Register_command("@DEFAULT", ">NATOMS");
    MDI_Register_command("@DEFAULT", ">NSTEPS");
    MDI_Register_command("@DEFAULT", ">TOLERANCE");
    MDI_Register_command("@DEFAULT", ">TYPES");
    MDI_Register_command("@DEFAULT", ">VELOCITIES");
    MDI_Register_command("@DEFAULT", "MD");
    MDI_Register_command("@DEFAULT", "OPTG");
    MDI_Register_command("@DEFAULT", "@INIT_MD");
    MDI_Register_command("@DEFAULT", "@INIT_OPTG");
    MDI_Register_command("@DEFAULT", "EXIT");

    // default node, custom commands added by LAMMPS

    MDI_Register_command("@DEFAULT", "NBYTES");
    MDI_Register_command("@DEFAULT", "COMMAND");
    MDI_Register_command("@DEFAULT", "COMMANDS");
    MDI_Register_command("@DEFAULT", "INFILE");
    MDI_Register_command("@DEFAULT", "<KE");

    // node for setting up and running a dynamics simulation

    MDI_Register_node("@INIT_MD");
    MDI_Register_command("@INIT_MD", "<@");
    MDI_Register_command("@INIT_MD", "@");
    MDI_Register_command("@INIT_MD", "@DEFAULT");
    MDI_Register_command("@INIT_MD", "@COORDS");
    MDI_Register_command("@INIT_MD", "@FORCES");
    MDI_Register_command("@INIT_MD", "@ENDSTEP");
    MDI_Register_command("@INIT_MD", "EXIT");

    // node for setting up and running a minimization

    MDI_Register_node("@INIT_OPTG");
    MDI_Register_command("@INIT_OPTG", "<@");
    MDI_Register_command("@INIT_OPTG", "@");
    MDI_Register_command("@INIT_OPTG", "@DEFAULT");
    MDI_Register_command("@INIT_OPTG", "@COORDS");
    MDI_Register_command("@INIT_OPTG", "@FORCES");
    MDI_Register_command("@INIT_OPTG", "EXIT");

    // node at POST_INTEGRATE location in timestep
    // only used if fix MDI/ENGINE is instantiated

    MDI_Register_node("@COORDS");
    MDI_Register_command("@COORDS", "<@");
    MDI_Register_command("@COORDS", "<COORDS");
    MDI_Register_command("@COORDS", "<VELOCITIES");
    MDI_Register_command("@COORDS", ">COORDS");
    MDI_Register_command("@COORDS", ">VELOCITIES");
    MDI_Register_command("@COORDS", "@");
    MDI_Register_command("@COORDS", "@DEFAULT");
    MDI_Register_command("@COORDS", "@COORDS");
    MDI_Register_command("@COORDS", "@FORCES");
    MDI_Register_command("@COORDS", "@ENDSTEP");
    MDI_Register_command("@COORDS", "EXIT");

    // node at POST_FORCE location in timestep
    // only used if fix MDI/ENGINE is instantiated
    // two register callbacks allow LAMMPS to interact more easily
    //   with drivers which don't know LAMMPS control flow

    MDI_Register_node("@FORCES");
    MDI_Register_callback("@FORCES", ">FORCES");
    MDI_Register_callback("@FORCES", ">+FORCES");
    MDI_Register_command("@FORCES", "<@");
    MDI_Register_command("@FORCES", "<COORDS");
    MDI_Register_command("@FORCES", "<ENERGY");
    MDI_Register_command("@FORCES", "<FORCES");
    MDI_Register_command("@FORCES", "<KE");
    MDI_Register_command("@FORCES", "<PE");
    MDI_Register_command("@FORCES", "<STRESS");
    MDI_Register_command("@FORCES", "<VELOCITIES");
    MDI_Register_command("@FORCES", ">FORCES");
    MDI_Register_command("@FORCES", ">+FORCES");
    MDI_Register_command("@FORCES", ">VELOCITIES");
    MDI_Register_command("@FORCES", "@");
    MDI_Register_command("@FORCES", "@DEFAULT");
    MDI_Register_command("@FORCES", "@COORDS");
    MDI_Register_command("@FORCES", "@FORCES");
    MDI_Register_command("@FORCES", "@ENDSTEP");
    MDI_Register_command("@FORCES", "EXIT");

    // node at END_OF_STEP location in timestep
    // only used if fix MDI/ENGINE is instantiated

    MDI_Register_node("@ENDSTEP");
    MDI_Register_command("@ENDSTEP", "<@");
    MDI_Register_command("@ENDSTEP", "<ENERGY");
    MDI_Register_command("@ENDSTEP", "<FORCES");
    MDI_Register_command("@ENDSTEP", "<KE");
    MDI_Register_command("@ENDSTEP", "<PE");
    MDI_Register_command("@ENDSTEP", "<STRESS");
    MDI_Register_command("@ENDSTEP", "@");
    MDI_Register_command("@ENDSTEP", "@DEFAULT");
    MDI_Register_command("@ENDSTEP", "@COORDS");
    MDI_Register_command("@ENDSTEP", "@FORCES");
    MDI_Register_command("@ENDSTEP", "@ENDSTEP");
    MDI_Register_command("@ENDSTEP", "EXIT");
  }


  /* ----------------------------------------------------------------------
     perform minimization to convergence using >TOLERANCE settings
  ---------------------------------------------------------------------- */


  /* ----------------------------------------------------------------------
     evaluate() invoked by <ENERGY, <FORCES, <PE, <STRESS
     if flag_natoms or flag_types set, create a new system
     if any receive flags set, evaulate eng/forces/stress
  ---------------------------------------------------------------------- */

  void MDIEngine::evaluate()
  {

  }

  /* ----------------------------------------------------------------------
     create a new system
     >CELL, >NATOMS, >TYPES or >ELEMENTS, >COORDS commands are required
  ---------------------------------------------------------------------- */

  void MDIEngine::create_system()
  {

  }

  /* ----------------------------------------------------------------------
     adjust simulation box
  ---------------------------------------------------------------------- */

  void MDIEngine::adjust_box()
  {

  }


  /* ----------------------------------------------------------------------
     overwrite coords
  ---------------------------------------------------------------------- */

  void MDIEngine::adjust_coords()
  {

  }

  // ----------------------------------------------------------------------
  // ----------------------------------------------------------------------/
  // MDI ">" driver commands that send data
  // ----------------------------------------------------------------------
  // ----------------------------------------------------------------------

  /* ----------------------------------------------------------------------
     >CELL command
     reset simulation box edge vectors
     in conjunction with >CELL_DISPL this can change box arbitrarily
     can be done to create a new box
     can be done incrementally during MD or OPTG
  ---------------------------------------------------------------------- */

  void MDIEngine::receive_cell()
  {

  }

  /* ----------------------------------------------------------------------
     >CELL_DISPL command
     reset simulation box lower left corner
     in conjunction with >CELL this can change box arbitrarily
     can be done to create a new box
     can be done incrementally during MD or OPTG
  ---------------------------------------------------------------------- */

  void MDIEngine::receive_cell_displ()
  {

  }


  /* ----------------------------------------------------------------------
     >COORDS command
  ---------------------------------------------------------------------- */

  void MDIEngine::receive_coords()
  {

  }

  /* ----------------------------------------------------------------------
     >ELEMENTS command
     receive elements for each atom = atomic numbers
     convert to LAMMPS atom types and store in sys_types
  ---------------------------------------------------------------------- */

  void MDIEngine::receive_elements()
  {

  }

  /* ----------------------------------------------------------------------
     >NATOMS command
     natoms cannot exceed 32-bit int for use with MDI
  ---------------------------------------------------------------------- */

  void MDIEngine::receive_natoms()
  {

  }

  /* ----------------------------------------------------------------------
     >NSTEPS command
     receive nsteps for timestepping
  ---------------------------------------------------------------------- */

  void MDIEngine::receive_nsteps()
  {

  }

  /* ----------------------------------------------------------------------
     >TOLERANCE command
     receive 4 minimization tolerance params
  ---------------------------------------------------------------------- */

  void MDIEngine::receive_tolerance()
  {

  }

  /* ----------------------------------------------------------------------
     >TYPES command
  ---------------------------------------------------------------------- */

  void MDIEngine::receive_types()
  {

  }

  /* ----------------------------------------------------------------------
     receive vector of 3 doubles for all atoms
     atoms are ordered by atomID, 1 to Natoms
     used by >FORCES command
  ---------------------------------------------------------------------- */

  void MDIEngine::receive_double3(int which)
  {

  }

  // ----------------------------------------------------------------------
  // ----------------------------------------------------------------------
  // MDI "<" driver commands that request data
  // ----------------------------------------------------------------------
  // ----------------------------------------------------------------------

  /* ----------------------------------------------------------------------
     <CELL command
     send simulation box edge vectors
  ---------------------------------------------------------------------- */

  void MDIEngine::send_cell()
  {

  }

  /* ----------------------------------------------------------------------
     <CELL_DISPL command
     send simulation box origin = lower-left corner
  ---------------------------------------------------------------------- */

  void MDIEngine::send_cell_displ()
  {

  }

  /* ----------------------------------------------------------------------
     <ENERGY command
     send total energy = PE + KE
  ---------------------------------------------------------------------- */

  void MDIEngine::send_total_energy()
  {

  }

  /* ----------------------------------------------------------------------
     <LABELS command
     convert numeric atom type to string for each atom
     atoms are ordered by atomID, 1 to Natoms
  ---------------------------------------------------------------------- */

  void MDIEngine::send_labels()
  {

  }

  /* ----------------------------------------------------------------------
     <NATOMS command
     natoms cannot exceed 32-bit int for use with MDI
  ---------------------------------------------------------------------- */

  void MDIEngine::send_natoms()
  {

  }

  /* ----------------------------------------------------------------------
     <PE command
     send potential energy
  ---------------------------------------------------------------------- */

  void MDIEngine::send_pe()
  {

  }

  /* ----------------------------------------------------------------------
     <STRESS command
     send 9-component stress tensor (no kinetic energy term)
  ---------------------------------------------------------------------- */

  void MDIEngine::send_stress()
  {

  }

  /* ----------------------------------------------------------------------
     send vector of 1 double for all atoms
     atoms are ordered by atomID, 1 to Natoms
     used by <CHARGE, <MASSES commands
  ---------------------------------------------------------------------- */

  void MDIEngine::send_double1(int which)
  {

  }

  /* ----------------------------------------------------------------------
     send vector of 1 int for all atoms
     atoms are ordered by atomID, 1 to Natoms
     use by <TYPES command
  ---------------------------------------------------------------------- */

  void MDIEngine::send_int1(int which)
  {

  }

  /* ----------------------------------------------------------------------
     <COORDS, <FORCES commands
     send vector of 3 doubles for all atoms
     atoms are ordered by atomID, 1 to Natoms
  ---------------------------------------------------------------------- */

  void MDIEngine::send_double3(int which)
  {

  }

  // ----------------------------------------------------------------------
  // ----------------------------------------------------------------------
  // responses to custom LAMMPS MDI commands
  // ----------------------------------------------------------------------
  // ----------------------------------------------------------------------

  /* ----------------------------------------------------------------------
     NBYTES command
     store received value in nbytes
     for use by a subsequent command, e.g. ones that send strings
  ---------------------------------------------------------------------- */

  void MDIEngine::nbytes_command()
  {
    int ierr = MDI_Recv(&nbytes, 1, MDI_INT, mdicomm);
    if (ierr) error->all(FLERR, "MDI: NBYTES data");
    MPI_Bcast(&nbytes, 1, MPI_INT, 0, world);
  }


  /* ----------------------------------------------------------------------
     MDI to/from DFTFE conversion factors
  ------------------------------------------------------------------------- */

  void MDIEngine::unit_conversions()
  {
  }

}
