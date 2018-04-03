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

template <unsigned int FEOrder>
void dftClass<FEOrder>::computeGroundStateRhoNodalField()
{
  //
  //access quadrature rules and mapping data
  //
  QGauss<3>  quadrature_formula(C_num1DQuad<FEOrder>());
  const unsigned int n_q_points = quadrature_formula.size();
  MappingQ1<3,3> mapping;
  struct quadDensityData { double density; };

  //
  //create electron-density quadrature data using "CellDataStorage" class of dealii
  //
  CellDataStorage<typename DoFHandler<3>::active_cell_iterator,quadDensityData> rhoQuadData;


  rhoQuadData.initialize(dofHandler.begin_active(),
			 dofHandler.end(),
			 n_q_points);
  //
  //copy rhoValues into CellDataStorage container
  //
  typename DoFHandler<3>::active_cell_iterator cell = dofHandler.begin_active(), endc = dofHandler.end();

  for(; cell!=endc; ++cell)
    {
      if(cell->is_locally_owned())
	{
	  const std::vector<std::shared_ptr<quadDensityData> > rhoQuadPointVector = rhoQuadData.get_data(cell);
	  for(unsigned int q = 0; q < n_q_points; ++q)
	    {
	      rhoQuadPointVector[q]->density = (*rhoOutValues)[cell->id()][q];
	    }
	}
    }

  //
  //project and create a nodal field of the same mesh from the quadrature data (L2 projection from quad points to nodes)
  //

  //
  //create a new nodal field
  //
  matrix_free_data.initialize_dof_vector(d_rhoNodalFieldGroundState);

  VectorTools::project<3,parallel::distributed::Vector<double>>(mapping,
								dofHandler,
								constraintsNone,
								quadrature_formula,
								[&](const typename DoFHandler<3>::active_cell_iterator & cell , const unsigned int q) -> double {return rhoQuadData.get_data(cell)[q]->density;},
								d_rhoNodalFieldGroundState);

  d_rhoNodalFieldGroundState.update_ghost_values();
}
