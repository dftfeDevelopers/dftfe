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
#include "MDIEngine.h"
#include "libraryMDI.h"
#include "dftfeWrapper.h"

#include <deal.II/base/data_out_base.h>

#include <cstring>
#include <limits>



namespace dftfe
{

  // per-atom data which engine commands access
  //enum { TYPE, COORD};
  enum { DEFAULT, MD, OPT };       // top-level MDI engine modes

  /* ----------------------------------------------------------------------
     trigger DFT-FE to start acting as an MDI engine
     either in standalone mode or plugin mode
       MDI_Init() for standalone mode is in main.cpp
       MDI_Init() for plugin mode is in library_mdi.cpp::MDI_Plugin_init_dftfe()
     endlessly loop over receiving commands from driver and responding
     when EXIT command is received, mdi engine command exits
  ---------------------------------------------------------------------- */

  MDIEngine::MDIEngine(MPI_Comm dftfeMPIComm, int argc, char *argv[])
  :d_dftfeMPIComm(dftfeMPIComm)
  {
    // confirm DFTFE is being run as an engine

    int role;
    MDI_Get_role(&role);
    if (role != MDI_ENGINE)
      AssertThrow(false,dealii::ExcMessage("Must invoke DFTFE as an MDI engine to use mdi engine"));      
      //error->all(FLERR, "Must invoke DFTFE as an MDI engine to use mdi engine");

    // root = 1 for proc 0, otherwise 0
    int rank;
    MPI_Comm_rank(d_dftfeMPIComm, &rank);
    d_root = (rank == 0) ? 1 : 0;

    // MDI setup

    d_mdicmd = new char[MDI_COMMAND_LENGTH];
    d_node_engine = new char[MDI_COMMAND_LENGTH];
    strncpy(d_node_engine, "@DEFAULT", MDI_COMMAND_LENGTH);
    d_node_driver = new char[MDI_COMMAND_LENGTH];
    strncpy(d_node_driver, "\0", MDI_COMMAND_LENGTH);

    // internal state of engine

    d_flag_natoms = 0;
    d_flag_types = d_flag_coords = 0;
    d_flag_cell = d_flag_cell_displ = 0;

    d_actionflag = 0;

    // define MDI commands that DFTFE engine recognizes

    mdi_commands();

    // register the execute_command function with MDI
    // only used when engine runs in plugin mode

    MDI_Set_execute_command_func(dftfe_execute_mdi_command, this);

    // one-time operation to establish a connection with the driver

    MDI_Accept_communicator(&d_mdicomm);
    if (d_mdicomm <= 0) 
      AssertThrow(false,dealii::ExcMessage("Unable to connect to MDI driver"));
      //error->all(FLERR, "Unable to connect to MDI driver");

    // endless engine loop, responding to driver commands

    d_mode = DEFAULT;
    d_exit_command = false;

    while (true)
    {
      // top-level mdi engine only recognizes one node: DEFAULT
      engine_node("@DEFAULT");

      if (d_exit_command)
      {
        break;

      }
      else
        AssertThrow(false,dealii::ExcMessage("MDI engine exited with invalid command: " +std::string(d_mdicmd)));
        //error->all(FLERR, fmt::format("MDI engine exited with invalid command: {}", mdicmd));
    }

    // clean up

    delete[] d_mdicmd;
    delete[] d_node_engine;
    delete[] d_node_driver;
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

    strncpy(d_node_engine, node, MDI_COMMAND_LENGTH);

    if (strcmp(d_node_driver, "\0") != 0 && strcmp(d_node_driver, d_node_engine) != 0)
      d_node_match = false;

    // respond to commands from the driver

    while (!d_exit_command && d_node_match) {

      // read the next command from the driver
      // all procs call this, but only proc 0 receives the command

      ierr = MDI_Recv_command(d_mdicmd, d_mdicomm);
      if (ierr)
        AssertThrow(false,dealii::ExcMessage("MDI: Unable to receive command from driver"));
        //error->all(FLERR, "MDI: Unable to receive command from driver");

      // broadcast command to the other MPI tasks

      MPI_Bcast(d_mdicmd, MDI_COMMAND_LENGTH, MPI_CHAR, 0,d_dftfeMPIComm);

      // execute the command

      execute_command(d_mdicmd, d_mdicomm);

      // check if driver request is now different than engine node

      if (strcmp(d_node_driver, "\0") != 0 && strcmp(d_node_driver, d_node_engine) != 0) 
        d_node_match = false;
    }

    // node exit was triggered so reset node_match

    d_node_match = true;
  }

  /* ----------------------------------------------------------------------
     process a single driver command
     called by engine_node() in loop
     also called by MDI itself via lib::dftfe_execute_mdi_command()
       when DFTFE is running as a plugin
  ---------------------------------------------------------------------- */

  int MDIEngine::execute_command(const char *command, MDI_Comm mdicomm)
  {
    int ierr;

    // confirm this command is supported at this node
    // otherwise is error

    int command_exists;
    if (d_root) {
      ierr = MDI_Check_command_exists(d_node_engine, command, MDI_COMM_NULL, &command_exists);
      if (ierr)
         AssertThrow(false,dealii::ExcMessage("MDI: Cannot confirm that command "+std::string(command)+" is supported"));     
        //error->one(FLERR, "MDI: Cannot confirm that command '{}' is supported", command);
    }

    MPI_Bcast(&command_exists, 1, MPI_INT, 0, d_dftfeMPIComm);
    if (!command_exists)
      AssertThrow(false,dealii::ExcMessage("MDI: Received command "+std::string(command)+" unsupported by engine node "+std::string(d_node_engine)));        
      //error->all(FLERR, "MDI: Received command '{}' unsupported by engine node {}", command,
      //           node_engine);

    // ---------------------------------------
    // respond to MDI standard commands
    // receives first, sends second
    // ---------------------------------------

    if (strcmp(command, ">CELL") == 0) {
      receive_cell();

    } else if (strcmp(command, ">CELL_DISPL") == 0) {
      receive_cell_displ();

    } else if (strcmp(command, ">COORDS") == 0) {
      receive_coords();

    } else if (strcmp(command, ">NATOMS") == 0) {
      receive_natoms();

    } else if (strcmp(command, ">TYPES") == 0) {
      receive_types();

    // -----------------------------------------------

    } else if (strcmp(command, "<ENERGY") == 0) {
      if (!d_actionflag) evaluate();
      d_actionflag = 1;
      send_energy();

    } else if (strcmp(command, "<FORCES") == 0) {
      if (!d_actionflag) evaluate();
      d_actionflag = 1;
      send_forces();

    } else if (strcmp(command, "<STRESS") == 0) {
      if (!d_actionflag) evaluate();
      d_actionflag = 1;
      send_stress();

    // exit command

    } else if (strcmp(command, "EXIT") == 0) {
      d_exit_command = true;

    // -------------------------------------------------------
    // unknown command
    // -------------------------------------------------------

    } 
    else 
    {
      AssertThrow(false,dealii::ExcMessage("MDI: Unknown command "+std::string(command)+" received from driver"));      
      //error->all(FLERR, "MDI: Unknown command {} received from driver", command);
    }

    return 0;
    
  }

  /* ----------------------------------------------------------------------
     define which MDI commands the DFTFE engine recognizes at each node
     both standard MDI commands and custom DFTFE commands
     max length for a command is currently 11 chars
  ---------------------------------------------------------------------- */

  void MDIEngine::mdi_commands()
  {
    // default node, MDI standard commands

    MDI_Register_node("@DEFAULT");
    MDI_Register_command("@DEFAULT", "<ENERGY");
    MDI_Register_command("@DEFAULT", "<FORCES");
    MDI_Register_command("@DEFAULT", "<STRESS");
    MDI_Register_command("@DEFAULT", ">CELL");
    MDI_Register_command("@DEFAULT", ">COORDS");
    MDI_Register_command("@DEFAULT", ">NATOMS");
    MDI_Register_command("@DEFAULT", ">ELEMENTS");
    MDI_Register_command("@DEFAULT", "EXIT");
  }


  /* ----------------------------------------------------------------------
     evaluate() invoked by <ENERGY, <FORCES, <STRESS
     if flag_natoms or flag_types set, create a new system
     if any receive flags set, evaulate eng/forces/stress
  ---------------------------------------------------------------------- */

  void MDIEngine::evaluate()
  {
    int flag_create = d_flag_natoms | d_flag_types;
    int flag_other = d_flag_cell | d_flag_cell_displ | d_flag_coords;

    // create new system or incrementally update system
    // NOTE: logic here is as follows
    //   if >NATOMS or >TYPES received since last eval, create a new system
    //     using natoms, cell edge vectors, cell_displ, coords, types
    //   otherwise just update existing system
    //     using any of cell edge vectors, cell_displ, coords
    //     assume the received values are incremental changes

    if (flag_create)
      create_system();
    else if (flag_other)
    {
      if (d_flag_cell || d_flag_cell_displ)
        adjust_box();  // your method
      if (d_flag_coords)
        adjust_coords();                // your method
    }

    // evaluate energy, forces, virial
    // NOTE: here is where you trigger the QM calc to happen



    // clear flags that trigger next eval

    d_flag_natoms = d_flag_types = 0;
    d_flag_cell = d_flag_cell_displ = d_flag_coords = 0;
  }

  /* ----------------------------------------------------------------------
     create a new system
     >CELL, >NATOMS, >TYPES, >COORDS commands are required
     >CELL_DISPL command is optional
  ---------------------------------------------------------------------- */

  void MDIEngine::create_system()
  {
    // check requirements

    if (d_flag_cell == 0 || d_flag_natoms == 0 || d_flag_types == 0 || d_flag_coords == 0)
      AssertThrow(false,dealii::ExcMessage("MDI create_system requires >CELL, >NATOMS, >TYPES, >COORDS MDI commands"));      
      //error->all(FLERR,
      //           "MDI create_system requires >CELL, >NATOMS, >TYPES, >COORDS "
      //           "MDI commands");

    // create new system
    // NOTE: here is where you wipeout the old system
    //       setup a new box, atoms, types, etc
  }

  void MDIEngine::adjust_box()
  {
  }

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
     can be done incrementally during AIMD
  ---------------------------------------------------------------------- */

  void MDIEngine::receive_cell()
  {
    d_actionflag = 0;
    d_flag_cell = 1;
    int ierr = MDI_Recv(d_sys_cell, 9, MDI_DOUBLE, d_mdicomm);
    if (ierr)
      AssertThrow(false,dealii::ExcMessage("MDI: >CELL data"));
      //error->all(FLERR, "MDI: >CELL data");
    MPI_Bcast(d_sys_cell, 9, MPI_DOUBLE, 0, d_dftfeMPIComm);
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
    d_actionflag = 0;
    d_flag_cell_displ = 1;
    int ierr = MDI_Recv(d_sys_cell_displ, 3, MDI_DOUBLE, d_mdicomm);
    if (ierr)
      AssertThrow(false,dealii::ExcMessage("MDI: >CELL_DISPLS data"));
      //error->all(FLERR, "MDI: >CELL_DISPLS data");
    MPI_Bcast(d_sys_cell_displ, 3, MPI_DOUBLE, 0, d_dftfeMPIComm);
  }

  /* ----------------------------------------------------------------------
     >COORDS command
  ---------------------------------------------------------------------- */

  void MDIEngine::receive_coords()
  {
    d_actionflag = 0;
    d_flag_coords = 1;
    int n = 3 * d_sys_natoms;
    int ierr = MDI_Recv(d_sys_coords, n, MDI_DOUBLE, d_mdicomm);
    if (ierr)
      AssertThrow(false,dealii::ExcMessage("MDI: >COORDS data"));
      //error->all(FLERR, "MDI: >COORDS data");
    MPI_Bcast(d_sys_coords, n, MPI_DOUBLE, 0, d_dftfeMPIComm);
  }

  /* ----------------------------------------------------------------------
     >NATOMS command
     natoms cannot exceed 32-bit int for use with MDI
  ---------------------------------------------------------------------- */

  void MDIEngine::receive_natoms()
  {
    d_actionflag = 0;
    d_flag_natoms = 1;
    int ierr = MDI_Recv(&d_sys_natoms, 1, MDI_INT, d_mdicomm);
    if (ierr) 
      AssertThrow(false,dealii::ExcMessage("MDI: >NATOMS data"));
      //error->all(FLERR, "MDI: >NATOMS data");
    MPI_Bcast(&d_sys_natoms, 1, MPI_INT, 0, d_dftfeMPIComm);
  }

  /* ----------------------------------------------------------------------
     >TYPES command
     NOTE: these are numeric types (atom species)
           need to figure out what these mean in driver vs QM engine
  ---------------------------------------------------------------------- */

  void MDIEngine::receive_types()
  {
    d_actionflag = 0;    
    d_flag_types = 1;
    int ierr = MDI_Recv(d_sys_types, d_sys_natoms, MDI_INT, d_mdicomm);
    if (ierr) 
      AssertThrow(false,dealii::ExcMessage("MDI: >TYPES data"));
      //error->all(FLERR, "MDI: >TYPES data");
    //FIXME: check if the correct communicator is being used
    MPI_Bcast(d_sys_types, d_sys_natoms, MPI_INT, 0, d_dftfeMPIComm);
  }

  // ----------------------------------------------------------------------
  // ----------------------------------------------------------------------
  // MDI "<" driver commands that request data
  // ----------------------------------------------------------------------
  // ----------------------------------------------------------------------

  // ----------------------------------------------------------------------
  // ----------------------------------------------------------------------
  // MDI "<" driver commands that request data
  // ----------------------------------------------------------------------
  // ----------------------------------------------------------------------

  /* ----------------------------------------------------------------------
     <ENERGY command
  ---------------------------------------------------------------------- */

  void MDIEngine::send_energy()
  {
    // NOTE: energy should be QM energy
    double energy;
    int ierr = MDI_Send(&energy, 1, MDI_DOUBLE, d_mdicomm);
    if (ierr)
      AssertThrow(false,dealii::ExcMessage("MDI: <ENERGY data"));
      //error->all(FLERR, "MDI: <ENERGY data");
  }

  /* ----------------------------------------------------------------------
     <FORCES command
     send vector of 3 doubles for all atoms
     atoms are ordered by atomID, 1 to Natoms
  ---------------------------------------------------------------------- */

  void MDIEngine::send_forces()
  {
    // NOTE: forces should be vector of 3*N QM forces
    std::vector<double> forces(3*d_sys_natoms,0.0);
    int ierr = MDI_Send(&forces[0], 3 * d_sys_natoms, MDI_DOUBLE, d_mdicomm);
    if (ierr)
      AssertThrow(false,dealii::ExcMessage("MDI: <FORCES data"));
      //error->all(FLERR, "MDI: <FORCES data");
  }

  /* ----------------------------------------------------------------------
     <STRESS command
     send 6-component stress tensor (no kinetic energy term)
  ---------------------------------------------------------------------- */

  void MDIEngine::send_stress()
  {
    // NOTE: vtensor should be QM virial values (symmetric tensor)
    std::vector<double> vtensor(6,0.0);
    int ierr = MDI_Send(&vtensor[0], 6, MDI_DOUBLE, d_mdicomm);
    if (ierr)
      AssertThrow(false,dealii::ExcMessage("MDI: <STRESS data"));
      //error->all(FLERR, "MDI: <STRESS data");
  }
}
#endif
