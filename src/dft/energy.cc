//source file for all energy computations 

//compute energies
void dft::compute_energy(){
  QGauss<3>  quadrature(quadratureRule);
  FEValues<3> fe_values (FE, quadrature, update_values | update_gradients | update_JxW_values);
  const unsigned int   dofs_per_cell = FE.dofs_per_cell;
  const unsigned int   num_quad_points    = quadrature.size();
  std::vector<double> cellPhiTotRhoIn(num_quad_points);  
  std::vector<double> cellPhiTotRhoOut(num_quad_points);  
  std::vector<double> cellPhiExt(num_quad_points);
  
  // Loop through all cells.
  Vector<double>  localPhiTotRhoIn(phiTotRhoIn);
  Vector<double>  localPhiTotRhoOut(phiTotRhoOut);
  Vector<double>  localPhiExt(phiExt);
  //
  double bandEnergy=0.0;
  double partialOccupancy, temp;
  for (unsigned int i=0; i<numEigenValues; i++){
    temp=(eigenValues[i]-fermiEnergy)/(kb*TVal);
    partialOccupancy=1.0/(1.0+exp(temp));
    bandEnergy+= 2*partialOccupancy*eigenValues[i];
    if (this_mpi_process == 0) std::printf("partialOccupancy %u: %30.20e \n", i, partialOccupancy);
  }
  double potentialTimesRho=0.0, exchangeEnergy=0.0, correlationEnergy=0.0, electrostaticEnergy=0.0; 
  
  //parallel loop over all elements
  typename DoFHandler<3>::active_cell_iterator cell = dofHandler.begin_active(), endc = dofHandler.end();
  unsigned int cellID=0;
  for (; cell!=endc; ++cell) {
    if (cell->is_locally_owned()){
      // Compute values for current cell.
      fe_values.reinit (cell);
      fe_values.get_function_values(localPhiTotRhoIn, cellPhiTotRhoIn);
      fe_values.get_function_values(localPhiTotRhoOut, cellPhiTotRhoOut);
      fe_values.get_function_values(localPhiExt, cellPhiExt);
      //Get Exc
      std::vector<double> densityValueIn(num_quad_points), densityValueOut(num_quad_points);
      std::vector<double> exchangeEnergyVal(num_quad_points), corrEnergyVal(num_quad_points);
      std::vector<double> exchangePotentialVal(num_quad_points), corrPotentialVal(num_quad_points);
      for (unsigned int q_point=0; q_point<num_quad_points; ++q_point){
	densityValueIn[q_point] = (*rhoInValues)(cellID, q_point);
	densityValueOut[q_point] = (*rhoOutValues)(cellID, q_point);
      }
      xc_lda_exc(&funcX,num_quad_points,&densityValueOut[0],&exchangeEnergyVal[0]);
      xc_lda_exc(&funcC,num_quad_points,&densityValueOut[0],&corrEnergyVal[0]);
      xc_lda_vxc(&funcX,num_quad_points,&densityValueIn[0],&exchangePotentialVal[0]);
      xc_lda_vxc(&funcC,num_quad_points,&densityValueIn[0],&corrPotentialVal[0]);
      for (unsigned int q_point=0; q_point<num_quad_points; ++q_point){
	//Veff computed with rhoIn
	double Veff=cellPhiTotRhoIn[q_point]+exchangePotentialVal[q_point]+corrPotentialVal[q_point];
	//Vtot, Vext computet with rhoIn
	double Vtot=cellPhiTotRhoOut[q_point];
	double Vext=cellPhiExt[q_point];
	potentialTimesRho+=Veff*(*rhoOutValues)(cellID, q_point)*fe_values.JxW (q_point);
	exchangeEnergy+=(exchangeEnergyVal[q_point])*(*rhoOutValues)(cellID, q_point)*fe_values.JxW (q_point);
	correlationEnergy+=(corrEnergyVal[q_point])*(*rhoOutValues)(cellID, q_point)*fe_values.JxW (q_point);
	electrostaticEnergy+=0.5*(Vtot+Vext)*(*rhoOutValues)(cellID, q_point)*fe_values.JxW (q_point);
	//energy+= 0.0; //This should be the repulsion energy for multi-atom
      }
    cellID++;
    }
  } 
  double energy=-potentialTimesRho+exchangeEnergy+correlationEnergy+electrostaticEnergy;
  //sum over all processors
  double totalEnergy= Utilities::MPI::sum(energy, mpi_communicator);
  double totalpotentialTimesRho= Utilities::MPI::sum(potentialTimesRho, mpi_communicator); 
  double totalexchangeEnergy= Utilities::MPI::sum(exchangeEnergy, mpi_communicator); 
  double totalcorrelationEnergy= Utilities::MPI::sum(correlationEnergy, mpi_communicator);
  double totalelectrostaticEnergy= Utilities::MPI::sum(electrostaticEnergy, mpi_communicator); 
  //total energy
 totalEnergy+=bandEnergy;
 double totalkineticEnergy=-totalpotentialTimesRho+bandEnergy;
 if (this_mpi_process == 0) {
   std::printf("Total energy:%30.20e \n", totalEnergy);
   std::printf("Band energy:%30.20e \nKinetic energy:%30.20e \nExchange energy:%30.20e \nCorrelation energy:%30.20e \nElectrostatic energy:%30.20e \n", bandEnergy, totalkineticEnergy, totalexchangeEnergy, totalcorrelationEnergy, totalelectrostaticEnergy);
 }
}

//compute fermi energy
void dft::compute_fermienergy(){
  //initial guess for fe
  double fe;
  if (numElectrons%2==0)
    fe = eigenValues[numElectrons/2-1];
  else
    fe = eigenValues[numElectrons/2];
  
  //compute residual
  double R=1.0;
  unsigned int iter=0;
  double temp1, temp2, temp3, temp4;
  while((std::abs(R)>1.0e-12) && (iter<100)){
    temp3=0.0; temp4=0.0;
    for (unsigned int i=0; i<numEigenValues; i++){
      temp1=(eigenValues[i]-fe)/(kb*TVal);
      if (temp1<=0.0){
	temp2=1.0/(1.0+exp(temp1));
	temp3+=2.0*temp2;
	temp4+=2.0*(exp(temp1)/(kb*TVal))*temp2*temp2;
      }
      else{
	temp2=1.0/(1.0+exp(-temp1));
	temp3+=2.0*exp(-temp1)*temp2;
	temp4+=2.0*(exp(-temp1)/(kb*TVal))*temp2*temp2;       
      }
    }
    R=temp3-numElectrons;
    fe+=-R/temp4;
    iter++;
    //if (this_mpi_process == 0) std::printf("Fermi energy Residual %u:%30.20e \n", iter, std::abs(R));
  }
  if(std::abs(R)>1.0e-12){
    pcout << "Fermi Energy computation: Newton iterations failed to converge\n";
    exit(-1);
  }
  if (this_mpi_process == 0) std::printf("Fermi energy Residual:%30.20e \n", std::abs(R));

  //set Fermi energy
  fermiEnergy=fe;
  if (this_mpi_process == 0) std::printf("Fermi energy:%30.20e \n", fermiEnergy);
}
