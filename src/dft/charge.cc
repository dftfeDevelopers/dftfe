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
// @author Shiva Rudraraju (2016), Phani Motamarri (2016)
//

//source file for all charge calculations

//compute total charge
template <unsigned int FEOrder>
double dftClass<FEOrder>::totalCharge(const std::map<dealii::CellId, std::vector<double> > *rhoQuadValues){
  double normValue=0.0;
  QGauss<3>  quadrature_formula(C_num1DQuad<FEOrder>());
  FEValues<3> fe_values (FE, quadrature_formula, update_JxW_values);
  const unsigned int   dofs_per_cell = FE.dofs_per_cell;
  const unsigned int   n_q_points    = quadrature_formula.size();

  DoFHandler<3>::active_cell_iterator
    cell = dofHandler.begin_active(),
    endc = dofHandler.end();
  for (; cell!=endc; ++cell) {
    if (cell->is_locally_owned()){
      fe_values.reinit (cell);
      for (unsigned int q_point=0; q_point<n_q_points; ++q_point){
        normValue+=(*rhoQuadValues).find(cell->id())->second[q_point]*fe_values.JxW(q_point);
      }
    }
  }
  return Utilities::MPI::sum(normValue, mpi_communicator);
}
