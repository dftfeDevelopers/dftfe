// ---------------------------------------------------------------------
//
// Copyright (c) 2017 The Regents of the University of Michigan and DFT-FE authors.
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
/** @file restartUtils.cc
 *
 *  @author Sambit Das
 */

#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/numerics/solution_transfer.h>

namespace dftfe {


    void
    triangulationManager::saveTriangulationsSolutionVectors
				 (const dealii::DoFHandler<3> & dofHandler,
				  const std::vector< const dealii::parallel::distributed::Vector<double> * > & solutionVectors)
    {

      dealii::parallel::distributed::SolutionTransfer<3,typename dealii::parallel::distributed::Vector<double> > solTrans(dofHandler);
      solTrans.prepare_serialization(solutionVectors);

      std::string filename="testsave";
      d_parallelTriangulationMoved.save(filename.c_str());
    }

}
