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
#  include "MDIEngine.h"
#  include "libraryMDI.h"
#  include "linearAlgebraOperations.h"

#  include <deal.II/base/data_out_base.h>

#  include <cstring>
#  include <limits>
#  include <iostream>


namespace dftfe
{
  // per-atom data which engine commands access
  // enum { TYPE, COORD};
  enum
  {
    DEFAULT,
    MD,
    OPT
  }; // top-level MDI engine modes

  /* ----------------------------------------------------------------------
     trigger DFT-FE to start acting as an MDI engine
     either in standalone mode or plugin mode
       MDI_Init() for standalone mode is in main.cpp
       MDI_Init() for plugin mode is in library_mdi.cpp::MDI_Plugin_init_dftfe()
     endlessly loop over receiving commands from driver and responding
     when EXIT command is received, mdi engine command exits
  ---------------------------------------------------------------------- */

  MDIEngine::MDIEngine(MPI_Comm &dftfeMPIComm, int argc, char *argv[])
    : d_dftfeMPIComm(dftfeMPIComm)
  {
    // confirm DFTFE is being run as an engine

    int role;
    MDI_Get_role(&role);
    if (role != MDI_ENGINE)
      AssertThrow(false,
                  dealii::ExcMessage(
                    "Must invoke DFTFE as an MDI engine to use mdi engine"));
    // error->all(FLERR, "Must invoke DFTFE as an MDI engine to use mdi
    // engine");

    // root = 1 for proc 0, otherwise 0
    int rank;
    MPI_Comm_rank(d_dftfeMPIComm, &rank);
    d_root = (rank == 0) ? 1 : 0;

    // MDI setup

    d_mdicmd      = new char[MDI_COMMAND_LENGTH];
    d_node_engine = new char[MDI_COMMAND_LENGTH];
    strncpy(d_node_engine, "@DEFAULT", MDI_COMMAND_LENGTH);
    d_node_driver = new char[MDI_COMMAND_LENGTH];
    strncpy(d_node_driver, "\0", MDI_COMMAND_LENGTH);

    // internal state of engine

    d_flag_natoms   = 0;
    d_flag_elements = d_flag_coords = 0;
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
      AssertThrow(false, dealii::ExcMessage("Unable to connect to MDI driver"));
    // error->all(FLERR, "Unable to connect to MDI driver");

    // endless engine loop, responding to driver commands

    d_mode         = DEFAULT;
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
          AssertThrow(false,
                      dealii::ExcMessage(
                        "MDI engine exited with invalid command: " +
                        std::string(d_mdicmd)));
        // error->all(FLERR, fmt::format("MDI engine exited with invalid
        // command: {}", mdicmd));
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

  void
  MDIEngine::engine_node(const char *node)
  {
    int ierr;

    // do not process commands if engine and driver request are not the same

    strncpy(d_node_engine, node, MDI_COMMAND_LENGTH);

    if (strcmp(d_node_driver, "\0") != 0 &&
        strcmp(d_node_driver, d_node_engine) != 0)
      d_node_match = false;

    // respond to commands from the driver

    while (!d_exit_command && d_node_match)
      {
        // read the next command from the driver
        // all procs call this, but only proc 0 receives the command

        ierr = MDI_Recv_command(d_mdicmd, d_mdicomm);
        if (ierr)
          AssertThrow(false,
                      dealii::ExcMessage(
                        "MDI: Unable to receive command from driver"));
        // error->all(FLERR, "MDI: Unable to receive command from driver");

        // broadcast command to the other MPI tasks

        MPI_Bcast(d_mdicmd, MDI_COMMAND_LENGTH, MPI_CHAR, 0, d_dftfeMPIComm);

        // execute the command

        execute_command(d_mdicmd, d_mdicomm);

        // check if driver request is now different than engine node

        if (strcmp(d_node_driver, "\0") != 0 &&
            strcmp(d_node_driver, d_node_engine) != 0)
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

  int
  MDIEngine::execute_command(const char *command, MDI_Comm &mdicomm)
  {
    int ierr;

    // confirm this command is supported at this node
    // otherwise is error

    int command_exists;
    if (d_root)
      {
        ierr = MDI_Check_command_exists(d_node_engine,
                                        command,
                                        MDI_COMM_NULL,
                                        &command_exists);
        if (ierr)
          AssertThrow(false,
                      dealii::ExcMessage("MDI: Cannot confirm that command " +
                                         std::string(command) +
                                         " is supported"));
        // error->one(FLERR, "MDI: Cannot confirm that command '{}' is
        // supported", command);
      }

    MPI_Bcast(&command_exists, 1, MPI_INT, 0, d_dftfeMPIComm);
    if (!command_exists)
      AssertThrow(false,
                  dealii::ExcMessage("MDI: Received command " +
                                     std::string(command) +
                                     " unsupported by engine node " +
                                     std::string(d_node_engine)));
    // error->all(FLERR, "MDI: Received command '{}' unsupported by engine node
    // {}", command,
    //           node_engine);

    // ---------------------------------------
    // respond to MDI standard commands
    // receives first, sends second
    // ---------------------------------------
    // std::cout << "HELLO " << command << std::endl;

    if (strcmp(command, ">CELL") == 0)
      {
        receive_cell();
      }
    else if (strcmp(command, ">CELL_DISPL") == 0)
      {
        receive_cell_displ();
      }
    else if (strcmp(command, ">COORDS") == 0)
      {
        receive_coords();
      }
    else if (strcmp(command, ">NATOMS") == 0)
      {
        receive_natoms();
      }
    else if (strcmp(command, ">ELEMENTS") == 0)
      {
        receive_elements();
      }
    else if (strcmp(command, ">DIMENSIONS") == 0)
      {
        receive_dimensions();
      }
    else if (strcmp(command, "<ENERGY") == 0)
      {
        if (!d_actionflag)
          evaluate();
        d_actionflag = 1;
        send_energy();
      }
    else if (strcmp(command, "<FORCES") == 0)
      {
        if (!d_actionflag)
          evaluate();
        d_actionflag = 1;
        send_forces();
      }
    else if (strcmp(command, "<STRESS") == 0)
      {
        if (!d_actionflag)
          evaluate();
        d_actionflag = 1;
        send_stress();

        // exit command
      }
    else if (strcmp(command, "EXIT") == 0)
      {
        d_exit_command = true;
        d_dftfeWrapper.clear();
        // -------------------------------------------------------
        // unknown command
        // -------------------------------------------------------
      }
    else
      {
        AssertThrow(false,
                    dealii::ExcMessage("MDI: Unknown command " +
                                       std::string(command) +
                                       " received from driver"));
        // error->all(FLERR, "MDI: Unknown command {} received from driver",
        // command);
      }

    return 0;
  }

  /* ----------------------------------------------------------------------
     define which MDI commands the DFTFE engine recognizes at each node
     both standard MDI commands and custom DFTFE commands
     max length for a command is currently 11 chars
  ---------------------------------------------------------------------- */

  void
  MDIEngine::mdi_commands()
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
    MDI_Register_command("@DEFAULT", ">DIMENSIONS");
    MDI_Register_command("@DEFAULT", "EXIT");
  }


  /* ----------------------------------------------------------------------
     evaluate() invoked by <ENERGY, <FORCES, <STRESS
     if flag_natoms or flag_elements set, create a new system
     if any receive flags set, evaulate eng/forces/stress
  ---------------------------------------------------------------------- */

  void
  MDIEngine::evaluate()
  {
    int flag_create = d_flag_natoms | d_flag_elements | d_flag_dimensions;
    int flag_other  = d_flag_cell | d_flag_cell_displ | d_flag_coords;

    // create new system or incrementally update system
    // NOTE: logic here is as follows
    //   if >NATOMS or >ELEMENTS received since last eval, create a new system
    //     using natoms, cell edge vectors, cell_displ, coords, elements
    //   otherwise just update existing system
    //     using any of cell edge vectors, cell_displ, coords
    //     assume the received values are incremental changes

    if (flag_create)
      create_system();
    else if (flag_other)
      {
        if (d_flag_cell || d_flag_cell_displ)
          adjust_box(); // your method
        if (d_flag_coords)
          adjust_coords(); // your method
      }

    // evaluate energy, forces, virial
    // NOTE: here is where you trigger the QM calc to happen
    d_dftfeWrapper.computeDFTFreeEnergy(true, false);


    // clear flags that trigger next eval

    d_flag_natoms = d_flag_elements = 0;
    d_flag_cell = d_flag_dimensions = d_flag_cell_displ = d_flag_coords = 0;
  }

  /* ----------------------------------------------------------------------
     create a new system
     >CELL, >NATOMS, >ELEMENTS, >COORDS commands are required
     >CELL_DISPL command is optional
  ---------------------------------------------------------------------- */

  void
  MDIEngine::create_system()
  {
    // check requirements

    if (d_flag_cell == 0 || d_flag_dimensions == 0 || d_flag_natoms == 0 ||
        d_flag_elements == 0 || d_flag_coords == 0)
      AssertThrow(
        false,
        dealii::ExcMessage(
          "MDI create_system requires >CELL, >DIMENSIONS, >NATOMS, >ELEMENTS, >COORDS MDI commands"));
    // error->all(FLERR,
    //           "MDI create_system requires >CELL, >NATOMS, >ELEMENTS, >COORDS
    //           " "MDI commands");

    // create new system
    // NOTE: here is where you wipeout the old system
    //       setup a new box, atoms, elements, etc

    // in atomic units
    std::vector<std::vector<double>> cell(3, std::vector<double>(3, 0.0));
    cell[0][0] = d_sys_cell[0];
    cell[0][1] = d_sys_cell[1];
    cell[0][2] = d_sys_cell[2];
    cell[1][0] = d_sys_cell[3];
    cell[1][1] = d_sys_cell[4];
    cell[1][2] = d_sys_cell[5];
    cell[2][0] = d_sys_cell[6];
    cell[2][1] = d_sys_cell[7];
    cell[2][2] = d_sys_cell[8];

    // in atomic units
    std::vector<std::vector<double>> atomicPositionsCart(
      d_sys_natoms, std::vector<double>(3, 0.0));

    for (unsigned int i = 0; i < d_sys_natoms; ++i)
      for (unsigned int j = 0; j < 3; ++j)
        atomicPositionsCart[i][j] = d_sys_coords[3 * i + j];

    std::vector<unsigned int> atomicNumbers(d_sys_natoms, 0);
    for (unsigned int i = 0; i < d_sys_natoms; ++i)
      atomicNumbers[i] = d_sys_elements[i];

    std::vector<bool> pbc({true, true, true});

    for (int i = 0; i < 3; i++)
      {
        if (d_sys_dimensions[i] == 1)
          {
            pbc[i] = false;
          }
        else if (d_sys_dimensions[i] == 2)
          {
            pbc[i] = true;
          }
        else
          {
            AssertThrow(
              false,
              dealii::ExcMessage(
                "Incorrect DIMENSIONS vector input values from MDI driver. Should have value of either 1 or 2 for the three different cell vectors."));
          }
      }


    // constructs dftfe wrapper object
    d_dftfeWrapper.reinit(d_dftfeMPIComm,
                          true, // use GPU mode if compiled with CUDA
                          atomicPositionsCart,
                          atomicNumbers,
                          cell,
                          pbc,                                // pbc
                          std::vector<unsigned int>{1, 1, 1}, // MP grid
                          std::vector<bool>{false,
                                            false,
                                            false}); // MP grid shift
  }

  void
  MDIEngine::adjust_box()
  {
    std::vector<std::vector<double>> currentCell = d_dftfeWrapper.getCell();

    // row major storage of columns of cell vectors
    std::vector<double> currentCellFlattenedInv(9, 0.0);
    for (unsigned int i = 0; i < 3; ++i)
      for (unsigned int j = 0; j < 3; ++j)
        currentCellFlattenedInv[i + 3 * j] = currentCell[i][j];
    dftfe::linearAlgebraOperations::inverse(&currentCellFlattenedInv[0], 3);

    // row major storage of columns of cell vectors
    std::vector<double> newCellFlattened(9, 0.0);
    for (unsigned int i = 0; i < 3; ++i)
      for (unsigned int j = 0; j < 3; ++j)
        newCellFlattened[i + 3 * j] = d_sys_cell[3 * i + j];



    std::vector<double> deformationGradientFlattened(9, 0.0);

    for (unsigned int i = 0; i < 3; ++i)
      for (unsigned int j = 0; j < 3; ++j)
        for (unsigned int k = 0; k < 3; ++k)
          deformationGradientFlattened[3 * i + j] +=
            newCellFlattened[3 * i + k] * currentCellFlattenedInv[3 * k + j];


    std::vector<std::vector<double>> deformationGradient(
      3, std::vector<double>(3));
    for (unsigned int i = 0; i < 3; ++i)
      for (unsigned int j = 0; j < 3; ++j)
        deformationGradient[i][j] = deformationGradientFlattened[3 * i + j];
    d_dftfeWrapper.deformCell(deformationGradient);
  }

  void
  MDIEngine::adjust_coords()
  {
    std::vector<std::vector<double>> currentCoords =
      d_dftfeWrapper.getAtomPositionsCart();

    std::vector<std::vector<double>> atomsDisplacements(
      d_sys_natoms, std::vector<double>(3, 0.0));
    // in atomic units
    for (unsigned int i = 0; i < d_sys_natoms; ++i)
      for (unsigned int j = 0; j < 3; ++j)
        atomsDisplacements[i][j] =
          d_sys_coords[3 * i + j] - currentCoords[i][j];

    d_dftfeWrapper.updateAtomPositions(atomsDisplacements);
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

  void
  MDIEngine::receive_cell()
  {
    d_actionflag = 0;
    d_flag_cell  = 1;

    if (d_root == 1)
      {
        int ierr = MDI_Recv(d_sys_cell, 9, MDI_DOUBLE, d_mdicomm);
        if (ierr)
          AssertThrow(false, dealii::ExcMessage("MDI: >CELL data"));
      }

    MPI_Bcast(d_sys_cell, 9, MPI_DOUBLE, 0, d_dftfeMPIComm);
    // for (int i = 0; i < 9; i++)
    //  std::cout << "cell: " << d_sys_cell[i] << std::endl;
  }

  /* ----------------------------------------------------------------------
     >CELL_DISPL command
     reset simulation box lower left corner
     in conjunction with >CELL this can change box arbitrarily
     can be done to create a new box
     can be done incrementally during MD or OPTG
  ---------------------------------------------------------------------- */

  void
  MDIEngine::receive_cell_displ()
  {
    d_actionflag      = 0;
    d_flag_cell_displ = 1;

    if (d_root == 1)
      {
        int ierr = MDI_Recv(d_sys_cell_displ, 3, MDI_DOUBLE, d_mdicomm);
        if (ierr)
          AssertThrow(false, dealii::ExcMessage("MDI: >CELL_DISPLS data"));
      }

    MPI_Bcast(d_sys_cell_displ, 3, MPI_DOUBLE, 0, d_dftfeMPIComm);
  }

  /* ----------------------------------------------------------------------
     >COORDS command
  ---------------------------------------------------------------------- */

  void
  MDIEngine::receive_coords()
  {
    d_actionflag  = 0;
    d_flag_coords = 1;
    int n         = 3 * d_sys_natoms;
    d_sys_coords.resize(n);

    if (d_root == 1)
      {
        int ierr = MDI_Recv(&d_sys_coords[0], n, MDI_DOUBLE, d_mdicomm);
        if (ierr)
          AssertThrow(false, dealii::ExcMessage("MDI: >COORDS data"));
      }

    MPI_Bcast(&d_sys_coords[0], n, MPI_DOUBLE, 0, d_dftfeMPIComm);

    // for (int i = 0; i < n; i++)
    //  std::cout << "coord: " << d_sys_coords[i] << std::endl;
  }

  /* ----------------------------------------------------------------------
     >NATOMS command
     natoms cannot exceed 32-bit int for use with MDI
  ---------------------------------------------------------------------- */

  void
  MDIEngine::receive_natoms()
  {
    d_actionflag  = 0;
    d_flag_natoms = 1;
    if (d_root == 1)
      {
        int ierr = MDI_Recv(&d_sys_natoms, 1, MDI_INT, d_mdicomm);
        if (ierr)
          AssertThrow(false, dealii::ExcMessage("MDI: >NATOMS data"));
      }

    MPI_Bcast(&d_sys_natoms, 1, MPI_INT, 0, d_dftfeMPIComm);
    // std::cout << "n atoms: " << d_sys_natoms << std::endl;
  }

  /* ----------------------------------------------------------------------
     >ELEMENTS command
     NOTE: these are the atomic numbers for each atom index
  ---------------------------------------------------------------------- */

  void
  MDIEngine::receive_elements()
  {
    d_actionflag    = 0;
    d_flag_elements = 1;
    d_sys_elements.resize(d_sys_natoms);
    if (d_root == 1)
      {
        int ierr =
          MDI_Recv(&d_sys_elements[0], d_sys_natoms, MDI_INT, d_mdicomm);
        if (ierr)
          AssertThrow(false, dealii::ExcMessage("MDI: >ELEMENTS data"));
      }

    MPI_Bcast(&d_sys_elements[0], d_sys_natoms, MPI_INT, 0, d_dftfeMPIComm);

    // for (int i = 0; i < d_sys_natoms; i++)
    //  std::cout << "element: " << d_sys_elements[i] << std::endl;
  }


  void
  MDIEngine::receive_dimensions()
  {
    d_actionflag      = 0;
    d_flag_dimensions = 1;
    if (d_root == 1)
      {
        int ierr = MDI_Recv(d_sys_dimensions, 3, MDI_INT, d_mdicomm);
        if (ierr)
          AssertThrow(false, dealii::ExcMessage("MDI: >DIMENSIONS data"));
      }

    MPI_Bcast(d_sys_dimensions, 3, MPI_INT, 0, d_dftfeMPIComm);

    // for (int i = 0; i < d_sys_natoms; i++)
    //  std::cout << "element: " << d_sys_elements[i] << std::endl;
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

  void
  MDIEngine::send_energy()
  {
    // NOTE: energy should be QM energy
    const double energy = d_dftfeWrapper.getDFTFreeEnergy();
    if (d_root == 1)
      {
        int ierr = MDI_Send(&energy, 1, MDI_DOUBLE, d_mdicomm);
        if (ierr)
          AssertThrow(false, dealii::ExcMessage("MDI: <ENERGY data"));
      }
    MPI_Barrier(d_dftfeMPIComm);
  }

  /* ----------------------------------------------------------------------
     <FORCES command
     send vector of 3 doubles for all atoms
     atoms are ordered by atomID, 1 to Natoms
  ---------------------------------------------------------------------- */

  void
  MDIEngine::send_forces()
  {
    // NOTE: forces should be vector of 3*N QM forces
    std::vector<std::vector<double>> ionicForces =
      d_dftfeWrapper.getForcesAtoms();

    std::vector<double> forces(3 * d_sys_natoms, 0.0);
    for (unsigned int i = 0; i < d_sys_natoms; i++)
      {
        forces[3 * i + 0] = ionicForces[i][0];
        forces[3 * i + 1] = ionicForces[i][1];
        forces[3 * i + 2] = ionicForces[i][2];
      }
    if (d_root == 1)
      {
        int ierr =
          MDI_Send(&forces[0], 3 * d_sys_natoms, MDI_DOUBLE, d_mdicomm);
        if (ierr)
          AssertThrow(false, dealii::ExcMessage("MDI: <FORCES data"));
      }
    MPI_Barrier(d_dftfeMPIComm);
  }

  /* ----------------------------------------------------------------------
     <STRESS command
  ---------------------------------------------------------------------- */

  void
  MDIEngine::send_stress()
  {
    // NOTE: vtensor should be QM virial values (symmetric tensor)
    std::vector<std::vector<double>> stressTensor =
      d_dftfeWrapper.getCellStress();

    std::vector<double> stressTensorFlattened(9, 0.0);
    stressTensorFlattened[0] = stressTensor[0][0];
    stressTensorFlattened[1] = stressTensor[0][1];
    stressTensorFlattened[2] = stressTensor[0][2];
    stressTensorFlattened[3] = stressTensor[1][0];
    stressTensorFlattened[4] = stressTensor[1][1];
    stressTensorFlattened[5] = stressTensor[1][2];
    stressTensorFlattened[6] = stressTensor[2][0];
    stressTensorFlattened[7] = stressTensor[2][1];
    stressTensorFlattened[8] = stressTensor[2][2];

    if (d_root == 1)
      {
        int ierr =
          MDI_Send(&stressTensorFlattened[0], 9, MDI_DOUBLE, d_mdicomm);
        if (ierr)
          AssertThrow(false, dealii::ExcMessage("MDI: <STRESS data"));
      }
    MPI_Barrier(d_dftfeMPIComm);
  }
} // namespace dftfe
#endif
