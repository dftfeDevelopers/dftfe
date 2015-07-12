//source file for all charge calculations

//compute total charge
double dftClass::totalCharge(){
  double normValue=0.0;
  QGauss<3>  quadrature_formula(quadratureRule);
  FEValues<3> fe_values (FE, quadrature_formula, update_values | update_JxW_values | update_quadrature_points);
  const unsigned int   dofs_per_cell = FE.dofs_per_cell;
  const unsigned int   n_q_points    = quadrature_formula.size();
  
  DoFHandler<3>::active_cell_iterator
    cell = dofHandler.begin_active(),
    endc = dofHandler.end();
  unsigned int cellID=0;
  for (; cell!=endc; ++cell) {
    if (cell->is_locally_owned()){
      fe_values.reinit (cell);
      for (unsigned int q_point=0; q_point<n_q_points; ++q_point){
        normValue+=(*rhoInValues)(cellID,q_point)*fe_values.JxW(q_point);
      }
    cellID++;
    }
  }
  return Utilities::MPI::sum(normValue, mpi_communicator);
}
