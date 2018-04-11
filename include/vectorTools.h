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

/** @file vectorTools.h
 *
 *  @brief
 *
 *  @author Sambit Das
 */



#ifndef vectorTools_H_
#define vectorTools_H_
#include "headers.h"

namespace dftfe
{
    namespace vectorTools
    {

      /**
       * @brief wrapper function for L2 projection from cell quadrature data to nodal field
       *
       * @param [input]cellQuadData cell quadrature data
       * @param [input]quadrature quadrature rule corresponding to the cellQuadData
       * @param [input]dofHandler
       * @param [input]constraintMatrix
       * @param [output]nodalField nodalField must be intialized on dofHandler prior to this
       * function call
       */
      void projectQuadDataToNodalField(const std::map<dealii::CellId, std::vector<double> > * cellQuadData,
	                               const dealii::QGauss<3> & quadrature,
				       const dealii::DoFHandler<3> & dofHandler,
				       const dealii::ConstraintMatrix & constraintMatrix,
	                               dealii::parallel::distributed::Vector<double> & nodalField);
    }
}
#endif
