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
// @author Nikhil Kodali
//



#include <dftUtils.h>
#include <fileReaders.h>
#include <geometryOptimizationClass.h>


namespace dftfe
{
  geometryOptimizationClass::geometryOptimizationClass(
    dftfeWrapper &  dftfeWrapper,
    const MPI_Comm &mpi_comm_parent)
    : d_dftPtr(dftfeWrapper.getDftfeBasePtr())
    , d_mpiCommParent(mpi_comm_parent)
    , pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0))
  {}



  void
  geometryOptimizationClass::runOpt()
  {
    if (!(d_dftPtr->getParametersObject().chkType == 1 &&
          d_dftPtr->getParametersObject().restartFromChk))
      {
        d_dftPtr->solve(true, true, false, false);
      }
    geoOptIonPtr = new geoOptIon(d_dftPtr, d_mpiCommParent);

    geoOptCellPtr = new geoOptCell(d_dftPtr, d_mpiCommParent);


    if (d_dftPtr->getParametersObject().isIonOpt &&
        !d_dftPtr->getParametersObject().isCellOpt)
      {
        geoOptIonPtr->init();
        geoOptIonPtr->run();
      }
    else if (!d_dftPtr->getParametersObject().isIonOpt &&
             d_dftPtr->getParametersObject().isCellOpt)
      {
        geoOptCellPtr->init();
        geoOptCellPtr->run();
      }
    else if (d_dftPtr->getParametersObject().isIonOpt &&
             d_dftPtr->getParametersObject().isCellOpt)
      {
        // staggered ion and cell relaxation

        int ionGeoUpdates  = 100;
        int cellGeoUpdates = 100;
        int cycle          = 0;
        while (ionGeoUpdates > 0 && cellGeoUpdates > 0)
          {
            if (d_dftPtr->getParametersObject().verbosity >= 1)
              pcout
                << std::endl
                << "----------Staggered ionic and cell relaxation cycle no: "
                << cycle << " start---------" << std::endl;

            // relax ionic forces. Current forces are assumed
            // to be already computed
            geoOptIonPtr->init();
            ionGeoUpdates = geoOptIonPtr->run();

            // redo trivial solve to compute current stress
            // as stress is not computed during ionic relaxation
            // for efficiency gains
            d_dftPtr->trivialSolveForStress();

            // relax cell stress
            geoOptCellPtr->init();
            cellGeoUpdates = geoOptCellPtr->run();

            if (d_dftPtr->getParametersObject().verbosity >= 1)
              pcout
                << std::endl
                << "----------Staggered ionic and cell relaxation cycle no: "
                << cycle << " end-----------" << std::endl;

            cycle++;
          }

        if (d_dftPtr->getParametersObject().verbosity >= 1)
          pcout
            << std::endl
            << "--------- Staggered ionic and cell relaxation cycle completed in "
            << cycle << " cycles-------" << std::endl;
      }

    delete geoOptIonPtr;
    delete geoOptCellPtr;
  }

} // namespace dftfe
