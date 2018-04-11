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
// @author Sambit Das (2018)
//



#include <vectorTools.h>

namespace dftfe
{
    namespace vectorTools
    {
        void projectQuadDataToNodalField(const std::map<dealii::CellId, std::vector<double> > * cellQuadData,
	                               const dealii::QGauss<3> & quadrature,
				       const dealii::DoFHandler<3> & dofHandler,
				       const dealii::ConstraintMatrix & constraintMatrix,
	                               dealii::parallel::distributed::Vector<double> & nodalField)
	{
	  //
	  //access quadrature rules and mapping data
	  //
	  const unsigned int n_q_points = quadrature.size();
	  dealii::MappingQ1<3,3> mapping;
	  struct quadDataStruct { double quadVal; };

	  //
	  //create temporary quadrature data using "CellDataStorage" class of dealii
	  //
	  dealii::CellDataStorage<typename dealii::DoFHandler<3>::active_cell_iterator,quadDataStruct> cellQuadDataDealii;


	  cellQuadDataDealii.initialize(dofHandler.begin_active(),
				 dofHandler.end(),
				 n_q_points);
	  //
	  //copy input quad values into CellDataStorage container
	  //
	  typename dealii::DoFHandler<3>::active_cell_iterator cell = dofHandler.begin_active(), endc = dofHandler.end();

	  for(; cell!=endc; ++cell)
	      if(cell->is_locally_owned())
		{
		  const std::vector<std::shared_ptr<quadDataStruct> > quadPointVector = cellQuadDataDealii.get_data(cell);
		  for(unsigned int q = 0; q < n_q_points; ++q)
		    {
		      quadPointVector[q]->quadVal = (*cellQuadData).find(cell->id())->second[q];
		    }
		}

	  //
	  //project and create a nodal field of the same mesh from the quadrature data (L2 projection from quad points to nodes)
	  //
          nodalField=0;
	  dealii::VectorTools::project<3,dealii::parallel::distributed::Vector<double>>
	                                                    (mapping,
							     dofHandler,
							     constraintMatrix,
							     quadrature,
							     [&](const typename dealii::DoFHandler<3>::active_cell_iterator & cell , const unsigned int q) -> double {return cellQuadDataDealii.get_data(cell)[q]->quadVal;},
							     nodalField);

	   nodalField.update_ghost_values();
	}
    }//vector tools namespace
}//dftfe namespace
