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


//source file for all energy computations 
#include "../../include/dftParameters.h"


double FermiDiracFunctionValue(double x,
			       std::vector<std::vector<double> > & eigenValues,
			       std::vector<double> & kPointWeights,
			       double &TVal)
{

  int numberkPoints = eigenValues.size();
  int numberEigenValues = eigenValues[0].size();
  double functionValue = 0.0;
  double temp1,temp2;

  
  for(unsigned int kPoint = 0; kPoint < numberkPoints; ++kPoint)
    {
      for(unsigned int i = 0; i < numberEigenValues; i++)
	{
	  temp1 = (eigenValues[kPoint][i]-x)/(C_kb*TVal);
	  if(temp1 <= 0.0)
	    {
	      temp2  =  1.0/(1.0+exp(temp1));
	      functionValue += (2.0-dftParameters::spinPolarized)*kPointWeights[kPoint]*temp2;
	    }
	  else
	    {
	      temp2 =  1.0/(1.0+exp(-temp1));
	      functionValue += (2.0-dftParameters::spinPolarized)*kPointWeights[kPoint]*exp(-temp1)*temp2;
	    }
	}
    }

  return functionValue;

}

double FermiDiracFunctionDerivativeValue(double x,
					 std::vector<std::vector<double> > & eigenValues,
					 std::vector<double> & kPointWeights,
					 double &TVal)
{

  int numberkPoints = eigenValues.size();
  int numberEigenValues = eigenValues[0].size();
  double functionDerivative = 0.0;
  double temp1,temp2;

  for(unsigned int kPoint = 0; kPoint < numberkPoints; ++kPoint)
    {
      for(unsigned int i = 0; i < numberEigenValues; i++)
	{
	  temp1 = (eigenValues[kPoint][i]-x)/(C_kb*TVal);
	  if(temp1 <= 0.0)
	    {
	      temp2  =  1.0/(1.0 + exp(temp1));
	      functionDerivative += (2.0-dftParameters::spinPolarized)*kPointWeights[kPoint]*(exp(temp1)/(C_kb*TVal))*temp2*temp2;
	    }
	  else
	    {
	      temp2 =  1.0/(1.0 + exp(-temp1));
	      functionDerivative += (2.0-dftParameters::spinPolarized)*kPointWeights[kPoint]*(exp(-temp1)/(C_kb*TVal))*temp2*temp2; 

	    }
	}
    }

  return functionDerivative;

}


//compute energies
template<unsigned int FEOrder>
void dftClass<FEOrder>::compute_energy()
{
  QGauss<3>  quadrature(C_num1DQuad<FEOrder>());
  FEValues<3> fe_values (FE, quadrature, update_values | update_gradients | update_JxW_values);
  const unsigned int   num_quad_points    = quadrature.size();
  std::vector<double> cellPhiTotRhoIn(num_quad_points);  
  std::vector<double> cellPhiTotRhoOut(num_quad_points);  
  std::vector<double> cellPhiExt(num_quad_points);
  double TVal = dftParameters::TVal;



  //
  // Loop through all cells.
  //
  double bandEnergy=0.0, bandEnergyLocal=0.0;
  double partialOccupancy, factor;
  char buffer[100];
  for(int kPoint = 0; kPoint < d_maxkPoints; ++kPoint)
    {
      pcout << "kPoint: "<< kPoint <<std::endl;
      for (unsigned int i=0; i<numEigenValues; i++)
	{
	  factor=(eigenValues[kPoint][i]-fermiEnergy)/(C_kb*TVal);
	  //partialOccupancy=1.0/(1.0+exp(temp));
	  double partialOccupancy = (factor >= 0)?std::exp(-factor)/(1.0 + std::exp(-factor)) : 1.0/(1.0 + std::exp(factor));
	  bandEnergyLocal+= 2*partialOccupancy*d_kPointWeights[kPoint]*eigenValues[kPoint][i];
	  sprintf(buffer, "%s %u: %0.14f\n", "fractional occupancy", i, partialOccupancy); pcout << buffer;
	}
      pcout << std::endl; 
      sprintf(buffer, "number of electrons: %18.16e \n", integralRhoValue); pcout << buffer;
    }
  
   bandEnergy= Utilities::MPI::sum(bandEnergyLocal, interpoolcomm);
  
  pcout <<std::endl;
  double potentialTimesRho = 0.0, exchangeEnergy = 0.0, correlationEnergy = 0.0, electrostaticEnergyTotPot = 0.0; 

  //parallel loop over all elements
  typename DoFHandler<3>::active_cell_iterator cell = dofHandler.begin_active(), endc = dofHandler.end();

  if(dftParameters::xc_id == 4)
    {
      for (; cell!=endc; ++cell) 
	{
	  if (cell->is_locally_owned())
	    {
	      // Compute values for current cell.
	      fe_values.reinit (cell);
	      fe_values.get_function_values(poissonPtr->phiTotRhoIn,cellPhiTotRhoIn);
	      fe_values.get_function_values(poissonPtr->phiTotRhoOut,cellPhiTotRhoOut);
	      fe_values.get_function_values(poissonPtr->phiExt,cellPhiExt);
	  
	      // Get exc
	      std::vector<double> densityValueIn(num_quad_points), densityValueOut(num_quad_points);
	      std::vector<double> exchangeEnergyDensity(num_quad_points), corrEnergyDensity(num_quad_points);
	      std::vector<double> derExchEnergyWithInputDensity(num_quad_points), derCorrEnergyWithInputDensity(num_quad_points);
	      std::vector<double> derExchEnergyWithSigmaGradDenInput(num_quad_points),derCorrEnergyWithSigmaGradDenInput(num_quad_points);
	      std::vector<double> sigmaWithOutputGradDensity(num_quad_points), sigmaWithInputGradDensity(num_quad_points);
	      std::vector<double> gradRhoInDotgradRhoOut(num_quad_points);
	      for (unsigned int q_point=0; q_point<num_quad_points; ++q_point)
		{
		  densityValueIn[q_point] = (*rhoInValues)[cell->id()][q_point];
		  densityValueOut[q_point] = (*rhoOutValues)[cell->id()][q_point];
		  double gradRhoInX = ((*gradRhoInValues)[cell->id()][3*q_point + 0]);
		  double gradRhoInY = ((*gradRhoInValues)[cell->id()][3*q_point + 1]);
		  double gradRhoInZ = ((*gradRhoInValues)[cell->id()][3*q_point + 2]);
		  double gradRhoOutX = ((*gradRhoOutValues)[cell->id()][3*q_point + 0]);
		  double gradRhoOutY = ((*gradRhoOutValues)[cell->id()][3*q_point + 1]);
		  double gradRhoOutZ = ((*gradRhoOutValues)[cell->id()][3*q_point + 2]);
		  sigmaWithInputGradDensity[q_point] = gradRhoInX*gradRhoInX + gradRhoInY*gradRhoInY + gradRhoInZ*gradRhoInZ;
		  sigmaWithOutputGradDensity[q_point] = gradRhoOutX*gradRhoOutX + gradRhoOutY*gradRhoOutY + gradRhoOutZ*gradRhoOutZ;
		  gradRhoInDotgradRhoOut[q_point] = gradRhoInX*gradRhoOutX + gradRhoInY*gradRhoOutY + gradRhoInZ*gradRhoOutZ;
		}
	      xc_gga_exc(&funcX,num_quad_points,&densityValueOut[0],&sigmaWithOutputGradDensity[0],&exchangeEnergyDensity[0]);
	      xc_gga_exc(&funcC,num_quad_points,&densityValueOut[0],&sigmaWithOutputGradDensity[0],&corrEnergyDensity[0]);

	      xc_gga_vxc(&funcX,num_quad_points,&densityValueIn[0],&sigmaWithInputGradDensity[0],&derExchEnergyWithInputDensity[0],&derExchEnergyWithSigmaGradDenInput[0]);
	      xc_gga_vxc(&funcC,num_quad_points,&densityValueIn[0],&sigmaWithInputGradDensity[0],&derCorrEnergyWithInputDensity[0],&derCorrEnergyWithSigmaGradDenInput[0]);
	      for (unsigned int q_point=0; q_point<num_quad_points; ++q_point)
		{
		  // Veff computed with rhoIn
		  double Veff=cellPhiTotRhoIn[q_point]+derExchEnergyWithInputDensity[q_point]+derCorrEnergyWithInputDensity[q_point];
		  double VxcGrad = 2.0*(derExchEnergyWithSigmaGradDenInput[q_point]+derCorrEnergyWithSigmaGradDenInput[q_point])*gradRhoInDotgradRhoOut[q_point];

		  // Vtot, Vext computet with rhoIn
		  double Vtot=cellPhiTotRhoOut[q_point];
		  double Vext=cellPhiExt[q_point];

		  // quad rule
		  potentialTimesRho+=(Veff*((*rhoOutValues)[cell->id()][q_point])+VxcGrad)*fe_values.JxW (q_point);
		  exchangeEnergy+=(exchangeEnergyDensity[q_point])*((*rhoOutValues)[cell->id()][q_point])*fe_values.JxW(q_point);
		  correlationEnergy+=(corrEnergyDensity[q_point])*((*rhoOutValues)[cell->id()][q_point])*fe_values.JxW(q_point);
#ifdef ENABLE_PERIODIC_BC
		  electrostaticEnergyTotPot+=0.5*(Vtot)*((*rhoOutValues)[cell->id()][q_point])*fe_values.JxW(q_point);
#else
		  //electrostaticEnergyTotPot+=0.5*(Vtot+Vext)*((*rhoOutValues)[cell->id()][q_point])*fe_values.JxW(q_point);
		  electrostaticEnergyTotPot+=0.5*(Vtot)*((*rhoOutValues)[cell->id()][q_point])*fe_values.JxW(q_point);
#endif
		}
	    }
	} 
    }
  else
    {
      for (; cell!=endc; ++cell) 
	{
	  if (cell->is_locally_owned())
	    {
	      // Compute values for current cell.
	      fe_values.reinit (cell);
	      fe_values.get_function_values(poissonPtr->phiTotRhoIn,cellPhiTotRhoIn);
	      fe_values.get_function_values(poissonPtr->phiTotRhoOut,cellPhiTotRhoOut);
	      fe_values.get_function_values(poissonPtr->phiExt,cellPhiExt);
	  
	      // Get Exc
	      std::vector<double> densityValueIn(num_quad_points), densityValueOut(num_quad_points);
	      std::vector<double> exchangeEnergyVal(num_quad_points), corrEnergyVal(num_quad_points);
	      std::vector<double> exchangePotentialVal(num_quad_points), corrPotentialVal(num_quad_points);
	      for (unsigned int q_point=0; q_point<num_quad_points; ++q_point)
		{
		  densityValueIn[q_point] = (*rhoInValues)[cell->id()][q_point];
		  densityValueOut[q_point] = (*rhoOutValues)[cell->id()][q_point];
		}
	      xc_lda_exc(&funcX,num_quad_points,&densityValueOut[0],&exchangeEnergyVal[0]);
	      xc_lda_exc(&funcC,num_quad_points,&densityValueOut[0],&corrEnergyVal[0]);
	      xc_lda_vxc(&funcX,num_quad_points,&densityValueIn[0],&exchangePotentialVal[0]);
	      xc_lda_vxc(&funcC,num_quad_points,&densityValueIn[0],&corrPotentialVal[0]);
	      for (unsigned int q_point = 0; q_point < num_quad_points; ++q_point)
		{
		  // Veff computed with rhoIn
		  double Veff=cellPhiTotRhoIn[q_point]+exchangePotentialVal[q_point]+corrPotentialVal[q_point];
		  // Vtot, Vext computet with rhoIn
		  double Vtot=cellPhiTotRhoOut[q_point];
		  double Vext=cellPhiExt[q_point];
		  potentialTimesRho+=Veff*((*rhoOutValues)[cell->id()][q_point])*fe_values.JxW (q_point);
		  exchangeEnergy+=(exchangeEnergyVal[q_point])*((*rhoOutValues)[cell->id()][q_point])*fe_values.JxW(q_point);
		  correlationEnergy+=(corrEnergyVal[q_point])*((*rhoOutValues)[cell->id()][q_point])*fe_values.JxW(q_point);
#ifdef ENABLE_PERIODIC_BC
		  electrostaticEnergyTotPot+=0.5*(Vtot)*((*rhoOutValues)[cell->id()][q_point])*fe_values.JxW(q_point);
#else
		  //electrostaticEnergyTotPot+=0.5*(Vtot+Vext)*((*rhoOutValues)[cell->id()][q_point])*fe_values.JxW(q_point);
		  electrostaticEnergyTotPot+=0.5*(Vtot)*((*rhoOutValues)[cell->id()][q_point])*fe_values.JxW(q_point);
#endif
		}
	    }
	} 
    }


  
  double energy=-potentialTimesRho+exchangeEnergy+correlationEnergy+electrostaticEnergyTotPot;

  //
  //get nuclear electrostatic energy 0.5*sum_I*(phi_tot(RI) - VselfI(RI))
  //
  
  //
  //First evaluate sum_I*(Z_I*phi_tot(RI)) on atoms belonging to current processor
  //
  double phiContribution = 0.0,vSelfContribution=0.0;
  for (std::map<unsigned int, double>::iterator it=atoms.begin(); it!=atoms.end(); ++it)
    {
      phiContribution += (-it->second)*poissonPtr->phiTotRhoOut(it->first);//-charge*potential
    }

  //
  //Then evaluate sum_I*(Z_I*Vself_I(R_I)) on atoms belonging to current processor
  //
  for(int i = 0; i < d_localVselfs.size(); ++i)
    {
      vSelfContribution += (-d_localVselfs[i][0])*(d_localVselfs[i][1]);//-charge*potential
    }

  double nuclearElectrostaticEnergy = 0.5*(phiContribution - vSelfContribution);



  //sum over all processors
  double totalEnergy= Utilities::MPI::sum(energy, mpi_communicator);
  double totalpotentialTimesRho= Utilities::MPI::sum(potentialTimesRho, mpi_communicator); 
  double totalexchangeEnergy= Utilities::MPI::sum(exchangeEnergy, mpi_communicator); 
  double totalcorrelationEnergy= Utilities::MPI::sum(correlationEnergy, mpi_communicator);
  double totalelectrostaticEnergyPot= Utilities::MPI::sum(electrostaticEnergyTotPot, mpi_communicator);
  double totalNuclearElectrostaticEnergy = Utilities::MPI::sum(nuclearElectrostaticEnergy, mpi_communicator);


  //
  //total energy
  //
  totalEnergy+=bandEnergy;
 
 
#ifdef ENABLE_PERIODIC_BC
  totalEnergy+=totalNuclearElectrostaticEnergy;
#else
  totalEnergy+=totalNuclearElectrostaticEnergy;//repulsiveEnergy();
#endif


  double totalkineticEnergy=-totalpotentialTimesRho+bandEnergy;

  //output
  char bufferEnergy[200];
  pcout << "Energy computations\n";
  pcout << "-------------------\n";
  sprintf(bufferEnergy, "%-24s:%25.16e\n", "Band energy", bandEnergy); pcout << bufferEnergy; 
  sprintf(bufferEnergy, "%-24s:%25.16e\n", "Kinetic energy", totalkineticEnergy); pcout << bufferEnergy; 
  sprintf(bufferEnergy, "%-24s:%25.16e\n", "Exchange energy", totalexchangeEnergy); pcout << bufferEnergy; 
  sprintf(bufferEnergy, "%-24s:%25.16e\n", "Correlation energy", totalcorrelationEnergy); pcout << bufferEnergy; 
#ifdef ENABLE_PERIODIC_BC  
  sprintf(bufferEnergy, "%-24s:%25.16e\n", "Electrostatic energy", totalelectrostaticEnergyPot+totalNuclearElectrostaticEnergy); pcout << bufferEnergy; 
#else
  //double repulsive_energy= repulsiveEnergy();
  //sprintf(bufferEnergy, "%-24s:%25.16e\n", "Repulsive energy", repulsive_energy); pcout << bufferEnergy; 
  //sprintf(bufferEnergy, "%-24s:%25.16e\n", "Electrostatic energy", totalelectrostaticEnergyPot+repulsive_energy); pcout << bufferEnergy; 
  sprintf(bufferEnergy, "%-24s:%25.16e\n", "Electrostatic energy", totalelectrostaticEnergyPot+totalNuclearElectrostaticEnergy); pcout << bufferEnergy;   
#endif
  sprintf(bufferEnergy, "%-24s:%25.16e\n", "Total energy", totalEnergy); pcout << bufferEnergy; 
  sprintf(bufferEnergy, "%-24s:%25.16e\n", "Total energy per atom", totalEnergy/((double) atomLocations.size())); pcout << bufferEnergy; 
  /* 
  if (this_mpi_process == 0) {
    std::printf("Total energy:%30.20e \nTotal energy per atom:%30.20e \n", totalEnergy, totalEnergy/((double) atomLocations.size()));
    std::printf("Band energy:%30.20e \nKinetic energy:%30.20e \nExchange energy:%30.20e \nCorrelation energy:%30.20e \nElectrostatic energy Total Potential:%30.20e \nRepulsive energy:%30.20e \nNuclear Electrostatic Energy:%30.20e \n\n", bandEnergy, totalkineticEnergy, totalexchangeEnergy, totalcorrelationEnergy, totalelectrostaticEnergyPot, repulsiveEnergy(),totalNuclearElectrostaticEnergy);
  }
  */
}
 
//compute fermi energy
template<unsigned int FEOrder>
void dftClass<FEOrder>::compute_fermienergy()
{
  char bufferFermi[100];
  int count =  std::ceil(static_cast<double>(numElectrons)/(2.0-dftParameters::spinPolarized));
  double TVal = dftParameters::TVal;


  std::vector<double> eigenValuesAllkPoints;
  for(int kPoint = 0; kPoint < d_maxkPoints; ++kPoint)
    {
      for(int statesIter = 0; statesIter < eigenValues[0].size(); ++statesIter)
	{
	  eigenValuesAllkPoints.push_back(eigenValues[kPoint][statesIter]);
	}
    }

  std::sort(eigenValuesAllkPoints.begin(),eigenValuesAllkPoints.end());

  unsigned int maxNumberFermiEnergySolveIterations = 100;
  double fe;
  double R = 1.0;

#ifdef ENABLE_PERIODIC_BC
  //
  //compute Fermi-energy first by bisection method
  //  
  //double initialGuessLeft = Utilities::MPI::min(eigenValuesAllkPoints[0],interpoolcomm);
  //double initialGuessRight = Utilities::MPI::max(eigenValuesAllkPoints[eigenValuesAllkPoints.size() - 1],interpoolcomm);

  double initialGuessLeft = eigenValuesAllkPoints[0];
  double initialGuessRight = eigenValuesAllkPoints[eigenValuesAllkPoints.size() - 1];


  double xLeft,xRight;

  xRight = Utilities::MPI::max(initialGuessRight, interpoolcomm);
  xLeft =  Utilities::MPI::min(initialGuessLeft, interpoolcomm);
  

  for(int iter = 0; iter < maxNumberFermiEnergySolveIterations; ++iter)
    {
      double yRightLocal = FermiDiracFunctionValue(xRight,
					      eigenValues,
					      d_kPointWeights,
					      TVal);

      double yRight = Utilities::MPI::sum(yRightLocal, interpoolcomm);

      yRight -=  (double)numElectrons;

      double yLeftLocal =  FermiDiracFunctionValue(xLeft,
					      eigenValues,
					      d_kPointWeights,
					      TVal);

      double yLeft = Utilities::MPI::sum(yLeftLocal, interpoolcomm);

      yLeft -=  (double)numElectrons;

      if((yLeft*yRight) > 0.0)
	{
	  pcout << " Bisection Method Failed " <<std::endl;
	  exit(-1);
	}
      
      double xBisected = (xLeft + xRight)/2.0;

      double yBisectedLocal = FermiDiracFunctionValue(xBisected,
						 eigenValues,
						 d_kPointWeights,
						 TVal) ;
      double yBisected = Utilities::MPI::sum(yBisectedLocal, interpoolcomm);
      yBisected -=  (double)numElectrons;

      if((yBisected*yLeft) > 0.0)
	xLeft = xBisected;
      else
	xRight = xBisected;
					      
      if (std::abs(yBisected) <= 1.0e-09 || iter == maxNumberFermiEnergySolveIterations-1)
	{
	  fe = xBisected;
	  R  = std::abs(yBisected);
	  break;
	}

    }
  sprintf(bufferFermi, "%-50s: %25.16e\n", "Fermi energy constraint residual (bisection)", R); pcout << bufferFermi; 
#else
   fe = eigenValuesAllkPoints[d_maxkPoints*count - 1];
#endif  
  //
  //compute residual and find FermiEnergy using Newton-Raphson solve
  //
  //double R = 1.0;
  unsigned int iter = 0;
  double  functionValue, functionDerivativeValue;

  while((std::abs(R) > 1.0e-12) && (iter < maxNumberFermiEnergySolveIterations))
    {

      double functionValueLocal = FermiDiracFunctionValue(fe,
					      eigenValues,
					      d_kPointWeights,
					      TVal);
      functionValue = Utilities::MPI::sum(functionValueLocal, interpoolcomm);

      double functionDerivativeValueLocal  = FermiDiracFunctionDerivativeValue(fe,
								  eigenValues,
								  d_kPointWeights,
								  TVal);

      functionDerivativeValue = Utilities::MPI::sum(functionDerivativeValueLocal, interpoolcomm);
      

      R   =  functionValue - numElectrons;
      fe += -R/functionDerivativeValue; 
      iter++;
    }

  if(std::abs(R) > 1.0e-12)
    {
      pcout << "Fermi Energy computation: Newton iterations failed to converge\n";
      //exit(-1);
    }

  sprintf(bufferFermi, "%-50s: %25.16e\n", "Fermi energy constraint residual (Newton-Raphson)", std::abs(R)); pcout << bufferFermi; 

  //set Fermi energy
  fermiEnergy = fe;
  sprintf(bufferFermi, "%-50s: %25.16e\n\n", "Fermi energy", fermiEnergy); pcout << bufferFermi; 
}

template<unsigned int FEOrder>
double dftClass<FEOrder>::repulsiveEnergy()
{
  double energy=0.0;
  for (unsigned int n1=0; n1<atomLocations.size(); n1++){
    for (unsigned int n2=n1+1; n2<atomLocations.size(); n2++){
      double Z1,Z2;
      if(dftParameters::isPseudopotential)
	{
	  Z1=atomLocations[n1][1];
	  Z2=atomLocations[n2][1];
	}
      else
	{
	  Z1=atomLocations[n1][0];
	  Z2=atomLocations[n2][0];
	}
      Point<3> atom1(atomLocations[n1][2],atomLocations[n1][3],atomLocations[n1][4]);
      Point<3> atom2(atomLocations[n2][2],atomLocations[n2][3],atomLocations[n2][4]);
      energy+=(Z1*Z2)/atom1.distance(atom2);
    }
  }
  return energy;
}
//compute energies
template<unsigned int FEOrder>
void dftClass<FEOrder>::compute_energy_spinPolarized()
{
  QGauss<3>  quadrature(FEOrder+1);
  FEValues<3> fe_values (FE, quadrature, update_values | update_gradients | update_JxW_values);
  const unsigned int   num_quad_points    = quadrature.size();
  std::vector<double> cellPhiTotRhoIn(num_quad_points);  
  std::vector<double> cellPhiTotRhoOut(num_quad_points);  
  std::vector<double> cellPhiExt(num_quad_points);
  
  //
  // Loop through all cells.
  //
  double bandEnergy=0.0, bandEnergyLocal=0.0;
  double partialOccupancy, factor;
  char buffer[100];
  for(int kPoint = 0; kPoint < d_maxkPoints; ++kPoint)
    {
      pcout << "kPoint: "<< kPoint <<std::endl;
      for (unsigned int i=0; i<(1+dftParameters::spinPolarized)*numEigenValues; i++)
	{
	  factor=(eigenValues[kPoint][i]-fermiEnergy)/(C_kb*dftParameters::TVal);
	  //partialOccupancy=1.0/(1.0+exp(temp));
	  double partialOccupancy = (factor >= 0)?std::exp(-factor)/(1.0 + std::exp(-factor)) : 1.0/(1.0 + std::exp(factor));
	  bandEnergyLocal+= (2-dftParameters::spinPolarized)*partialOccupancy*d_kPointWeights[kPoint]*eigenValues[kPoint][i];
	  sprintf(buffer, "%s %u: %0.14f\n", "fractional occupancy", i, partialOccupancy); pcout << buffer;
	}
      pcout << std::endl; 
      sprintf(buffer, "number of electrons: %18.16e \n", integralRhoValue); pcout << buffer;
    }
  pcout <<std::endl;
  bandEnergy= Utilities::MPI::sum(bandEnergyLocal, interpoolcomm);
  double potentialTimesRho = 0.0, exchangeEnergy = 0.0, correlationEnergy = 0.0, electrostaticEnergyTotPot = 0.0; 

  

  //parallel loop over all elements
  typename DoFHandler<3>::active_cell_iterator cell = dofHandler.begin_active(), endc = dofHandler.end();

  if(dftParameters::xc_id == 4)
    {
      for (; cell!=endc; ++cell) 
	{
	  if (cell->is_locally_owned())
	    {
	      // Compute values for current cell.
	      fe_values.reinit (cell);
	      fe_values.get_function_values(poissonPtr->phiTotRhoIn,cellPhiTotRhoIn);
	      fe_values.get_function_values(poissonPtr->phiTotRhoOut,cellPhiTotRhoOut);
	      fe_values.get_function_values(poissonPtr->phiExt,cellPhiExt);
	  
	      // Get exc
	      std::vector<double> densityValueIn(2*num_quad_points), densityValueOut(2*num_quad_points);
	      std::vector<double> exchangeEnergyDensity(num_quad_points), corrEnergyDensity(num_quad_points);
	      std::vector<double> derExchEnergyWithInputDensity(2*num_quad_points), derCorrEnergyWithInputDensity(2*num_quad_points);
	      std::vector<double> derExchEnergyWithSigmaGradDenInput(3*num_quad_points),derCorrEnergyWithSigmaGradDenInput(3*num_quad_points);
	      std::vector<double> sigmaWithOutputGradDensity(3*num_quad_points), sigmaWithInputGradDensity(3*num_quad_points);
	      std::vector<double> gradRhoInDotgradRhoOut(3*num_quad_points);
	      for (unsigned int q_point=0; q_point<num_quad_points; ++q_point)
		{
		   densityValueIn[2*q_point+0] = (*rhoInValuesSpinPolarized)[cell->id()][2*q_point+0];
                  densityValueIn[2*q_point+1] = (*rhoInValuesSpinPolarized)[cell->id()][2*q_point+1];
                  densityValueOut[2*q_point+0] = (*rhoOutValuesSpinPolarized)[cell->id()][2*q_point+0];
                  densityValueOut[2*q_point+1] = (*rhoOutValuesSpinPolarized)[cell->id()][2*q_point+1];
		  //
		  double gradRhoInX1 = ((*gradRhoInValuesSpinPolarized)[cell->id()][6*q_point + 0]);
		  double gradRhoInY1 = ((*gradRhoInValuesSpinPolarized)[cell->id()][6*q_point + 1]);
		  double gradRhoInZ1 = ((*gradRhoInValuesSpinPolarized)[cell->id()][6*q_point + 2]);
		  double gradRhoOutX1 = ((*gradRhoOutValuesSpinPolarized)[cell->id()][6*q_point + 0]);
		  double gradRhoOutY1 = ((*gradRhoOutValuesSpinPolarized)[cell->id()][6*q_point + 1]);
		  double gradRhoOutZ1 = ((*gradRhoOutValuesSpinPolarized)[cell->id()][6*q_point + 2]);
		 //
		  double gradRhoInX2 = ((*gradRhoInValuesSpinPolarized)[cell->id()][6*q_point + 3]);
		  double gradRhoInY2 = ((*gradRhoInValuesSpinPolarized)[cell->id()][6*q_point + 4]);
		  double gradRhoInZ2 = ((*gradRhoInValuesSpinPolarized)[cell->id()][6*q_point + 5]);
		  double gradRhoOutX2 = ((*gradRhoOutValuesSpinPolarized)[cell->id()][6*q_point + 3]);
		  double gradRhoOutY2 = ((*gradRhoOutValuesSpinPolarized)[cell->id()][6*q_point + 4]);
		  double gradRhoOutZ2 = ((*gradRhoOutValuesSpinPolarized)[cell->id()][6*q_point + 5]);
		//
		  sigmaWithInputGradDensity[3*q_point+0] = gradRhoInX1*gradRhoInX1 + gradRhoInY1*gradRhoInY1 + gradRhoInZ1*gradRhoInZ1;
		  sigmaWithInputGradDensity[3*q_point+1] = gradRhoInX1*gradRhoInX2 + gradRhoInY1*gradRhoInY2 + gradRhoInZ1*gradRhoInZ2;
		  sigmaWithInputGradDensity[3*q_point+2] = gradRhoInX2*gradRhoInX2 + gradRhoInY2*gradRhoInY2 + gradRhoInZ2*gradRhoInZ2;
		  sigmaWithOutputGradDensity[3*q_point+0] = gradRhoOutX1*gradRhoOutX1 + gradRhoOutY1*gradRhoOutY1 + gradRhoOutZ1*gradRhoOutZ1;
		  sigmaWithOutputGradDensity[3*q_point+1] = gradRhoOutX1*gradRhoOutX2 + gradRhoOutY1*gradRhoOutY2 + gradRhoOutZ1*gradRhoOutZ2;
		  sigmaWithOutputGradDensity[3*q_point+2] = gradRhoOutX2*gradRhoOutX2 + gradRhoOutY2*gradRhoOutY2 + gradRhoOutZ2*gradRhoOutZ2;
		  gradRhoInDotgradRhoOut[3*q_point+0] = gradRhoInX1*gradRhoOutX1 + gradRhoInY1*gradRhoOutY1 + gradRhoInZ1*gradRhoOutZ1;
		  gradRhoInDotgradRhoOut[3*q_point+1] = gradRhoInX1*gradRhoOutX2 + gradRhoInY1*gradRhoOutY2 + gradRhoInZ1*gradRhoOutZ2;
                  gradRhoInDotgradRhoOut[3*q_point+2] = gradRhoInX2*gradRhoOutX2 + gradRhoInY2*gradRhoOutY2 + gradRhoInZ2*gradRhoOutZ2;
		}
	      xc_gga_exc(&funcX,num_quad_points,&densityValueOut[0],&sigmaWithOutputGradDensity[0],&exchangeEnergyDensity[0]);
	      xc_gga_exc(&funcC,num_quad_points,&densityValueOut[0],&sigmaWithOutputGradDensity[0],&corrEnergyDensity[0]);

	      xc_gga_vxc(&funcX,num_quad_points,&densityValueIn[0],&sigmaWithInputGradDensity[0],&derExchEnergyWithInputDensity[0],&derExchEnergyWithSigmaGradDenInput[0]);
	      xc_gga_vxc(&funcC,num_quad_points,&densityValueIn[0],&sigmaWithInputGradDensity[0],&derCorrEnergyWithInputDensity[0],&derCorrEnergyWithSigmaGradDenInput[0]);
	      for (unsigned int q_point=0; q_point<num_quad_points; ++q_point)
		{
		  // Veff computed with rhoIn
		  double Veff=cellPhiTotRhoIn[q_point]+derExchEnergyWithInputDensity[2*q_point+0]+derCorrEnergyWithInputDensity[2*q_point+0];
		  double VxcGrad = 2.0*(derExchEnergyWithSigmaGradDenInput[3*q_point+0]+derCorrEnergyWithSigmaGradDenInput[3*q_point+0])*gradRhoInDotgradRhoOut[3*q_point+0];
		  VxcGrad += 2.0*(derExchEnergyWithSigmaGradDenInput[3*q_point+1]+derCorrEnergyWithSigmaGradDenInput[3*q_point+1])*gradRhoInDotgradRhoOut[3*q_point+1];
		  VxcGrad += 2.0*(derExchEnergyWithSigmaGradDenInput[3*q_point+2]+derCorrEnergyWithSigmaGradDenInput[3*q_point+2] )*gradRhoInDotgradRhoOut[3*q_point+2];
		  potentialTimesRho+=(Veff*((*rhoOutValuesSpinPolarized)[cell->id()][2*q_point+0])+VxcGrad)*fe_values.JxW (q_point);
		  Veff=cellPhiTotRhoIn[q_point]+derExchEnergyWithInputDensity[2*q_point+1]+derCorrEnergyWithInputDensity[2*q_point+1];
		  potentialTimesRho+=(Veff*((*rhoOutValuesSpinPolarized)[cell->id()][2*q_point+1]))*fe_values.JxW (q_point);
		  // Vtot, Vext computet with rhoIn
		  double Vtot=cellPhiTotRhoOut[q_point];
		  double Vext=cellPhiExt[q_point];
		  // quad rule
		  exchangeEnergy+=(exchangeEnergyDensity[q_point])*((*rhoOutValues)[cell->id()][q_point])*fe_values.JxW(q_point);
		  correlationEnergy+=(corrEnergyDensity[q_point])*((*rhoOutValues)[cell->id()][q_point])*fe_values.JxW(q_point);
#ifdef ENABLE_PERIODIC_BC
		  electrostaticEnergyTotPot+=0.5*(Vtot)*((*rhoOutValues)[cell->id()][q_point])*fe_values.JxW(q_point);
#else
		  electrostaticEnergyTotPot+=0.5*(Vtot+Vext)*((*rhoOutValues)[cell->id()][q_point])*fe_values.JxW(q_point);
#endif
		}
	    }
	} 
    }
  else
    {
      for (; cell!=endc; ++cell) 
	{
	  if (cell->is_locally_owned())
	    {
	      // Compute values for current cell.
	      fe_values.reinit (cell);
	      fe_values.get_function_values(poissonPtr->phiTotRhoIn,cellPhiTotRhoIn);
	      fe_values.get_function_values(poissonPtr->phiTotRhoOut,cellPhiTotRhoOut);
	      fe_values.get_function_values(poissonPtr->phiExt,cellPhiExt);
	  
	      // Get Exc
	      std::vector<double> densityValueIn(2*num_quad_points), densityValueOut(2*num_quad_points);
	      std::vector<double> exchangeEnergyVal(num_quad_points), corrEnergyVal(num_quad_points);
	      std::vector<double> exchangePotentialVal(2*num_quad_points), corrPotentialVal(2*num_quad_points);
	      for (unsigned int q_point=0; q_point<2*num_quad_points; ++q_point)
		{
		  densityValueIn[q_point] = (*rhoInValuesSpinPolarized)[cell->id()][q_point];
                  densityValueOut[q_point] = (*rhoOutValuesSpinPolarized)[cell->id()][q_point];
		}
              //
           
	      xc_lda_exc(&funcX,num_quad_points,&densityValueOut[0],&exchangeEnergyVal[0]);
	      xc_lda_exc(&funcC,num_quad_points,&densityValueOut[0],&corrEnergyVal[0]);
	      xc_lda_vxc(&funcX,num_quad_points,&densityValueIn[0],&exchangePotentialVal[0]);
	      xc_lda_vxc(&funcC,num_quad_points,&densityValueIn[0],&corrPotentialVal[0]);

	      for (unsigned int q_point = 0; q_point < num_quad_points; ++q_point)
		{
                   // Vtot, Vext computet with rhoIn
		  double Vtot=cellPhiTotRhoOut[q_point];
		  double Vext=cellPhiExt[q_point];
		  // Veff computed with rhoIn
		  double Veff=cellPhiTotRhoIn[q_point]+exchangePotentialVal[2*q_point]+corrPotentialVal[2*q_point] ;
		  potentialTimesRho+=Veff*((*rhoOutValuesSpinPolarized)[cell->id()][2*q_point])*fe_values.JxW (q_point);
		  //
		  Veff= cellPhiTotRhoIn[q_point]+exchangePotentialVal[2*q_point+1]+corrPotentialVal[2*q_point+1] ;
		  potentialTimesRho+=Veff*((*rhoOutValuesSpinPolarized)[cell->id()][2*q_point+1])*fe_values.JxW (q_point);
		  //
		  exchangeEnergy+=(exchangeEnergyVal[q_point])*((*rhoOutValues)[cell->id()][q_point])*fe_values.JxW(q_point);
                  //exchangeEnergy+=(exchangeEnergyVal[2*q_point+1])*((*rhoOutValuesSpinPolarized)[cell->id()][2*q_point+1])*fe_values.JxW(q_point) ;
		  correlationEnergy+=(corrEnergyVal[q_point])*((*rhoOutValues)[cell->id()][q_point])*fe_values.JxW(q_point) ; 
		  //correlationEnergy+=(corrEnergyVal[2*q_point+1])*((*rhoOutValuesSpinPolarized)[cell->id()][2*q_point+1])*fe_values.JxW(q_point);
#ifdef ENABLE_PERIODIC_BC
		  electrostaticEnergyTotPot+=0.5*(Vtot)*((*rhoOutValues)[cell->id()][q_point])*fe_values.JxW(q_point);
#else
		  electrostaticEnergyTotPot+=0.5*(Vtot+Vext)*((*rhoOutValues)[cell->id()][q_point])*fe_values.JxW(q_point);
#endif
		}
	    }
	} 
    }



  
  double energy=-potentialTimesRho+exchangeEnergy+correlationEnergy+electrostaticEnergyTotPot;

  //
  //get nuclear electrostatic energy 0.5*sum_I*(phi_tot(RI) - VselfI(RI))
  //
  
  //
  //First evaluate sum_I*(Z_I*phi_tot(RI)) on atoms belonging to current processor
  //
  double phiContribution = 0.0,vSelfContribution=0.0;
  for (std::map<unsigned int, double>::iterator it=atoms.begin(); it!=atoms.end(); ++it)
    {
      phiContribution += (-it->second)*poissonPtr->phiTotRhoOut(it->first);//-charge*potential
    }

  //
  //Then evaluate sum_I*(Z_I*Vself_I(R_I)) on atoms belonging to current processor
  //
  for(int i = 0; i < d_localVselfs.size(); ++i)
    {
      vSelfContribution += (-d_localVselfs[i][0])*(d_localVselfs[i][1]);//-charge*potential
    }

  double nuclearElectrostaticEnergy = 0.5*(phiContribution - vSelfContribution);



  //sum over all processors
  double totalEnergy= Utilities::MPI::sum(energy, mpi_communicator);
  double totalpotentialTimesRho= Utilities::MPI::sum(potentialTimesRho, mpi_communicator); 
  double totalexchangeEnergy= Utilities::MPI::sum(exchangeEnergy, mpi_communicator); 
  double totalcorrelationEnergy= Utilities::MPI::sum(correlationEnergy, mpi_communicator);
  double totalelectrostaticEnergyPot= Utilities::MPI::sum(electrostaticEnergyTotPot, mpi_communicator);
  double totalNuclearElectrostaticEnergy = Utilities::MPI::sum(nuclearElectrostaticEnergy, mpi_communicator);

  //
  //total energy
  //
  totalEnergy+=bandEnergy;
 
 
#ifdef ENABLE_PERIODIC_BC
  totalEnergy+=totalNuclearElectrostaticEnergy;
#else
  totalEnergy+=repulsiveEnergy();
#endif


  double totalkineticEnergy=-totalpotentialTimesRho+bandEnergy;

  //output
  char bufferEnergy[200];
  pcout << "Energy computations\n";
  pcout << "-------------------\n";
  sprintf(bufferEnergy, "%-24s:%25.16e\n", "Band energy", bandEnergy); pcout << bufferEnergy; 
  sprintf(bufferEnergy, "%-24s:%25.16e\n", "Kinetic energy", totalkineticEnergy); pcout << bufferEnergy; 
  sprintf(bufferEnergy, "%-24s:%25.16e\n", "Exchange energy", totalexchangeEnergy); pcout << bufferEnergy; 
  sprintf(bufferEnergy, "%-24s:%25.16e\n", "Correlation energy", totalcorrelationEnergy); pcout << bufferEnergy; 
#ifdef ENABLE_PERIODIC_BC  
  sprintf(bufferEnergy, "%-24s:%25.16e\n", "Electrostatic energy", totalelectrostaticEnergyPot+totalNuclearElectrostaticEnergy); pcout << bufferEnergy; 
#else
  double repulsive_energy= repulsiveEnergy();
  sprintf(bufferEnergy, "%-24s:%25.16e\n", "Repulsive energy", repulsive_energy); pcout << bufferEnergy; 
  sprintf(bufferEnergy, "%-24s:%25.16e\n", "Electrostatic energy", totalelectrostaticEnergyPot+repulsive_energy); pcout << bufferEnergy; 
#endif
  sprintf(bufferEnergy, "%-24s:%25.16e\n", "Total energy", totalEnergy); pcout << bufferEnergy; 
  sprintf(bufferEnergy, "%-24s:%25.16e\n", "Total energy per atom", totalEnergy/((double) atomLocations.size())); pcout << bufferEnergy; 
  /* 
  if (this_mpi_process == 0) {
    std::printf("Total energy:%30.20e \nTotal energy per atom:%30.20e \n", totalEnergy, totalEnergy/((double) atomLocations.size()));
    std::printf("Band energy:%30.20e \nKinetic energy:%30.20e \nExchange energy:%30.20e \nCorrelation energy:%30.20e \nElectrostatic energy Total Potential:%30.20e \nRepulsive energy:%30.20e \nNuclear Electrostatic Energy:%30.20e \n\n", bandEnergy, totalkineticEnergy, totalexchangeEnergy, totalcorrelationEnergy, totalelectrostaticEnergyPot, repulsiveEnergy(),totalNuclearElectrostaticEnergy);
  }
  */
}
