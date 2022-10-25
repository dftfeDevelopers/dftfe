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
#  ifndef dftfeMDIEngine_H_
#    define dftfeMDIEngine_H_
#    include "dftfeWrapper.h"
#    include <mdi.h>
#    include <mpi.h>
#    include <string>
#    include <vector>

namespace dftfe
{
  /**
   * @brief MDIEngine interface class for dftfe
   *
   * @author Sambit Das
   */
  class MDIEngine
  {
  public:
    /**
     * @brief constructor
     */
    MDIEngine(MPI_Comm &dftfeMPIComm, int argc, char *argv[]);

    int
    execute_command(const char *command, MDI_Comm &mdicomm);

    void
    engine_node(const char *node);


  private:
    /// 1 for proc 0, otherwise 0
    int d_root;

    /// MDI communicator
    MDI_Comm d_mdicomm;

    /// MDI communicator
    MPI_Comm d_dftfeMPIComm;

    /// DFT-FE object
    dftfeWrapper d_dftfeWrapper;

    // state of MDI engine

    /// which mode engine is in ()
    int d_mode;
    /// current MDI command being processed
    char *d_mdicmd;
    /// which node engine is at
    char *d_node_engine;
    /// which node driver has requested
    char *d_node_driver;
    /// true if driver and engine node currently match
    bool d_node_match;
    /// true if EXIT command received from driver
    bool d_exit_command;

    // flags for data received by engine
    // not acted on until a request to send <ENERGY,<FORCES,<PE,<STRESS
    int d_actionflag;
    int d_flag_natoms, d_flag_elements;
    int d_flag_cell, d_flag_dimensions, d_flag_cell_displ;
    int d_flag_charges, d_flag_coords;

    int                 d_sys_natoms;
    int                 d_sys_dimensions[3];
    std::vector<int>    d_sys_elements;
    std::vector<double> d_sys_coords;
    double              d_sys_cell[9], d_sys_cell_displ[3];

    // class methods
    void
    mdi_commands();

    void
    evaluate();
    void
    create_system();
    void
    adjust_box();
    void
    adjust_coords();

    void
    receive_cell();
    void
    receive_cell_displ();
    void
    receive_coords();
    void
    receive_natoms();
    void
    receive_elements();
    void
    receive_dimensions();

    void
    send_energy();
    void
    send_forces();
    void
    send_stress();
  };
} // namespace dftfe
#  endif
#endif
