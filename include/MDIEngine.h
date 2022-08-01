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
#ifndef dftfeMDIEngine_H_
#define dftfeMDIEngine_H_

#include <mpi.h>
#include <string>
#include <vector>

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
      MDIEngine(int argc, char *argv[]);

      int execute_command(const char *command, MDI_Comm mdicomm);

      void engine_node(const char *node);


    private:

      /// 1 for proc 0, otherwise 0
      int d_root; 

      /// MDI communicator
      MDI_Comm d_mdicomm;

      // state of MDI engine

      /// which mode engine is in ()
      int d_mode;
      /// current MDI command being processed
      char * d_mdicmd;
      /// which node engine is at
      char * d_node_engine;
      /// which node driver has requested
      char * d_node_driver;
      /// true if driver and engine node currently match
      bool d_node_match;
      /// true if EXIT command received from driver
      bool d_exit_command; 


  };
} // namespace dftfe
#endif
#endif
