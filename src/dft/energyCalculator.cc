// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2018 The Regents of the University of Michigan and DFT-FE authors.
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
// @author Shiva Rudraraju, Phani Motamarri, Sambit Das, Krishnendu Ghosh
//


//source file for energy computations
#include <dftParameters.h>
#include <energyCalculator.h>
#include <constants.h>
#include <dftUtils.h>

namespace dftfe
{
  namespace internal {

    void  printEnergy(const double bandEnergy,
		      const double totalkineticEnergy,
		      const double totalexchangeEnergy,
		      const double totalcorrelationEnergy,
		      const double totalElectrostaticEnergy,
		      const double totalEnergy,
		      const unsigned int numberAtoms,
		      const dealii::ConditionalOStream & pcout,
		      const bool reproducibleOutput,
		      const unsigned int verbosity)
    {

      if (reproducibleOutput)
	{
	  const double bandEnergyTrunc  = std::floor(1000000000 * (bandEnergy)) / 1000000000.0;
	  const double totalkineticEnergyTrunc  = std::floor(1000000000 * (totalkineticEnergy)) / 1000000000.0;
	  const double totalexchangeEnergyTrunc  = std::floor(1000000000 * (totalexchangeEnergy)) / 1000000000.0;
	  const double totalcorrelationEnergyTrunc  = std::floor(1000000000 * (totalcorrelationEnergy)) / 1000000000.0;
	  const double totalElectrostaticEnergyTrunc  = std::floor(1000000000 * (totalElectrostaticEnergy)) / 1000000000.0;
	  const double totalEnergyTrunc  = std::floor(1000000000 * (totalEnergy)) / 1000000000.0;
	  const double totalEnergyPerAtomTrunc  = std::floor(1000000000 * (totalEnergy/numberAtoms)) / 1000000000.0;

	  pcout <<std::endl<< "Energy computations (Hartree) "<<std::endl;
	  pcout << "-------------------"<<std::endl;
	  if (dftParameters::useMixedPrecPGS_O || dftParameters::useMixedPrecPGS_SR)
	    pcout<<std::setw(25)<<"Total energy"<<": "<<std::fixed<<std::setprecision(6)<<std::setw(20)<<totalEnergyTrunc<< std::endl;
	  else
	    pcout<<std::setw(25)<<"Total energy"<<": "<<std::fixed<<std::setprecision(8)<<std::setw(20)<<totalEnergyTrunc<< std::endl;
	}
      else
	{
	  pcout<<std::endl;
	  char bufferEnergy[200];
	  pcout << "Energy computations (Hartree)\n";
	  pcout << "-------------------------------------------------------------------------------\n";
	  sprintf(bufferEnergy, "%-52s:%25.16e\n", "Band energy", bandEnergy); pcout << bufferEnergy;
	  if (verbosity>=2)
	    {
	      sprintf(bufferEnergy, "%-52s:%25.16e\n", "Kinetic energy plus nonlocal PSP", totalkineticEnergy); pcout << bufferEnergy;
	    }

	  sprintf(bufferEnergy, "%-52s:%25.16e\n", "Exchange energy", totalexchangeEnergy); pcout << bufferEnergy;
	  sprintf(bufferEnergy, "%-52s:%25.16e\n", "Correlation energy", totalcorrelationEnergy); pcout << bufferEnergy;
	  if (verbosity>=2)
	    {
	      sprintf(bufferEnergy, "%-52s:%25.16e\n", "Local PSP Electrostatic energy", totalElectrostaticEnergy); pcout << bufferEnergy;
	    }

	  sprintf(bufferEnergy, "%-52s:%25.16e\n", "Total energy", totalEnergy); pcout << bufferEnergy;
	  sprintf(bufferEnergy, "%-52s:%25.16e\n", "Total energy per atom", totalEnergy/numberAtoms); pcout << bufferEnergy;
	  pcout << "-------------------------------------------------------------------------------\n";
	}

    }

    double localBandEnergy(const std::vector<std::vector<double> > & eigenValues,
			   const std::vector<double> & kPointWeights,
			   const double fermiEnergy,
			   const double fermiEnergyUp,
			   const double fermiEnergyDown,
			   const double TVal,
			   const unsigned int spinPolarized,
			   const dealii::ConditionalOStream & scout,
			   const MPI_Comm & interpoolcomm,
			   const unsigned int lowerBoundKindex,
			   const unsigned int verbosity)
    {
      double bandEnergyLocal=0.0;
      unsigned int numEigenValues = eigenValues[0].size()/(1+spinPolarized) ;
      //
      for (unsigned int ipool = 0 ;  ipool < dealii::Utilities::MPI::n_mpi_processes(interpoolcomm) ; ++ipool) {
	MPI_Barrier(interpoolcomm) ;
	if (ipool==dealii::Utilities::MPI::this_mpi_process(interpoolcomm)) {
	  for(unsigned int kPoint = 0; kPoint < kPointWeights.size(); ++kPoint)
	    {
	      if (verbosity > 2)
		{
		  scout<<" Printing KS eigen values (spin split if this is a spin polarized calculation ) and fractional occupancies for kPoint " << (lowerBoundKindex + kPoint) << std::endl;
	          scout << "  " << std::endl ;
		}
              for (unsigned int i=0; i<numEigenValues; i++)
		{
		  if (spinPolarized==0)
		    {
		      const double partialOccupancy=dftUtils::getPartialOccupancy
			(eigenValues[kPoint][i],
			 fermiEnergy,
			 C_kb,
			 TVal);
		      bandEnergyLocal+= 2.0*partialOccupancy*kPointWeights[kPoint]*eigenValues[kPoint][i];
		      //

		      if (verbosity>2)
		        scout << i<<" : "<< eigenValues[kPoint][i] << "       " << partialOccupancy<<std::endl;
		      //
		    }
		  if (spinPolarized==1){
		    double partialOccupancy=dftUtils::getPartialOccupancy
		      (eigenValues[kPoint][i],
		       fermiEnergy,
		       C_kb,
		       TVal);
		    double partialOccupancy2=dftUtils::getPartialOccupancy
		      (eigenValues[kPoint][i+numEigenValues],
		       fermiEnergy,
		       C_kb,
		       TVal);

		    if(dftParameters::constraintMagnetization)
		      {
			partialOccupancy = 1.0 , partialOccupancy2 = 1.0 ;
			if (eigenValues[kPoint][i+numEigenValues] > fermiEnergyDown)
			  partialOccupancy2 = 0.0 ;
			if (eigenValues[kPoint][i] > fermiEnergyUp)
			  partialOccupancy = 0.0 ;

		      }
		    bandEnergyLocal+= partialOccupancy*kPointWeights[kPoint]*eigenValues[kPoint][i];
		    bandEnergyLocal+= partialOccupancy2*kPointWeights[kPoint]*eigenValues[kPoint][i+numEigenValues];
		    //
		    if (verbosity>2)
		      scout<< i<<" : "<< eigenValues[kPoint][i] << "       " << eigenValues[kPoint][i+numEigenValues] << "       " <<
			partialOccupancy << "       " << partialOccupancy2 << std::endl;
		  }
		}  // eigen state
	      //
	      if (verbosity > 2)
		scout << "============================================================================================================" << std::endl ;
	    }  // kpoint
	} // is it current pool
	//
	MPI_Barrier(interpoolcomm) ;
	//
      } // loop over pool

      return bandEnergyLocal;
    }

    //get nuclear electrostatic energy 0.5*sum_I*(Z_I*phi_tot(RI) - Z_I*VselfI(RI))
    double nuclearElectrostaticEnergyLocal(const vectorType & phiTotRhoOut,
					   const std::vector<std::vector<double> > & localVselfs,
					   const std::map<dealii::types::global_dof_index, double> & atomElectrostaticNodeIdToChargeMap)
    {


      double phiContribution = 0.0,vSelfContribution=0.0;
      for (std::map<dealii::types::global_dof_index, double>::const_iterator it=atomElectrostaticNodeIdToChargeMap.begin(); it!=atomElectrostaticNodeIdToChargeMap.end(); ++it)
	phiContribution += (-it->second)*phiTotRhoOut(it->first);//-charge*potential

      //
      //Then evaluate sum_I*(Z_I*Vself_I(R_I)) on atoms belonging to current processor
      //
      for(unsigned int i = 0; i < localVselfs.size(); ++i)
	vSelfContribution += (-localVselfs[i][0])*(localVselfs[i][1]);//-charge*potential

      return 0.5*(phiContribution - vSelfContribution);
    }


    double computeRepulsiveEnergy(const std::vector<std::vector<double> > & atomLocationsAndCharge,
				  const bool isPseudopotential)
    {
      double energy=0.0;
      for (unsigned int n1=0; n1<atomLocationsAndCharge.size(); n1++){
	for (unsigned int n2=n1+1; n2<atomLocationsAndCharge.size(); n2++){
	  double Z1,Z2;
	  if(isPseudopotential)
	    {
	      Z1=atomLocationsAndCharge[n1][1];
	      Z2=atomLocationsAndCharge[n2][1];
	    }
	  else
	    {
	      Z1=atomLocationsAndCharge[n1][0];
	      Z2=atomLocationsAndCharge[n2][0];
	    }
	  const dealii::Point<3> atom1(atomLocationsAndCharge[n1][2],
				       atomLocationsAndCharge[n1][3],
				       atomLocationsAndCharge[n1][4]);
	  const dealii::Point<3> atom2(atomLocationsAndCharge[n2][2],
				       atomLocationsAndCharge[n2][3],
				       atomLocationsAndCharge[n2][4]);
	  energy+=(Z1*Z2)/atom1.distance(atom2);
	}
      }
      return energy;
    }

  }

  energyCalculator::energyCalculator(const MPI_Comm &mpi_comm,
				     const MPI_Comm &interpool_comm,
				     const MPI_Comm & interbandgroup_comm):
    mpi_communicator (mpi_comm),
    interpoolcomm(interpool_comm),
    interBandGroupComm(interbandgroup_comm),
    pcout (std::cout, (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0))
  {

  }

  //compute energies
  double energyCalculator::computeEnergy
  (const dealii::DoFHandler<3> & dofHandlerElectrostatic,
   const dealii::DoFHandler<3> & dofHandlerElectronic,
   const dealii::QGauss<3> & quadratureElectrostatic,
   const dealii::QGauss<3> & quadratureElectronic,
   const std::vector<std::vector<double> > & eigenValues,
   const std::vector<double> & kPointWeights,
   const double fermiEnergy,
   const xc_func_type & funcX,
   const xc_func_type & funcC,
   const vectorType & phiTotRhoIn,
   const vectorType & phiTotRhoOut,
   const vectorType & phiExt,
   const vectorType & phiExtElec,
   const std::map<dealii::CellId, std::vector<double> > & rhoInValues,
   const std::map<dealii::CellId, std::vector<double> > & rhoOutValues,
   const std::map<dealii::CellId, std::vector<double> > & rhoOutValuesElectrostatic,
   const std::map<dealii::CellId, std::vector<double> > & gradRhoInValues,
   const std::map<dealii::CellId, std::vector<double> > & gradRhoOutValues,
   const std::vector<std::vector<double> > & localVselfs,
   const std::map<dealii::CellId, std::vector<double> > & pseudoValuesElectronic,
   const std::map<dealii::CellId, std::vector<double> > & pseudoValuesElectrostatic,
   const std::map<dealii::types::global_dof_index, double> & atomElectrostaticNodeIdToChargeMap,
   const unsigned int numberGlobalAtoms,
   const unsigned int lowerBoundKindex,
   const unsigned int scfConverged,
   const bool print) const
  {
    dealii::FEValues<3> feValuesElectrostatic (dofHandlerElectrostatic.get_fe(), quadratureElectrostatic, dealii::update_values | dealii::update_JxW_values);
    dealii::FEValues<3> feValuesElectronic (dofHandlerElectronic.get_fe(), quadratureElectronic, dealii::update_values | dealii::update_JxW_values);

    const unsigned int   num_quad_points_electrostatic    = quadratureElectrostatic.size();
    const unsigned int   num_quad_points_electronic    = quadratureElectronic.size();

    const double TVal = dftParameters::TVal;
    std::vector<double> cellPhiTotRhoIn(num_quad_points_electronic);
    std::vector<double> cellPhiTotRhoOut(num_quad_points_electrostatic);
    std::vector<double> cellPhiExt(num_quad_points_electronic);
    std::vector<double> cellPhiExtElec(num_quad_points_electrostatic);

    const dealii::ConditionalOStream scout (std::cout,
					    (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0 &&
					     dealii::Utilities::MPI::this_mpi_process(interBandGroupComm)==0)) ;
    const double bandEnergy=
      dealii::Utilities::MPI::sum(internal::localBandEnergy(eigenValues,
							    kPointWeights,
							    fermiEnergy,
							    fermiEnergy,
							    fermiEnergy,
							    dftParameters::TVal,
							    dftParameters::spinPolarized,
							    scout,
							    interpoolcomm,
							    lowerBoundKindex,
							    (dftParameters::verbosity+scfConverged)), interpoolcomm);

    double excCorrPotentialTimesRho=0.0, electrostaticPotentialTimesRho=0.0, exchangeEnergy = 0.0, correlationEnergy = 0.0, electrostaticEnergyTotPot = 0.0, vSelfPotentialTimesRho = 0.0, vSelfPotentialElecTimesRho = 0.0, vPseudoMinusVselfPotentialTimesRho=0.0;

    //parallel loop over all elements
    typename dealii::DoFHandler<3>::active_cell_iterator cellElectrostatic = dofHandlerElectrostatic.begin_active(), endcElectrostatic = dofHandlerElectrostatic.end();

    typename dealii::DoFHandler<3>::active_cell_iterator cellElectronic = dofHandlerElectronic.begin_active(), endcElectronic = dofHandlerElectronic.end();

    for (; cellElectronic!=endcElectronic; ++cellElectronic)
      if (cellElectronic->is_locally_owned())
	{

	  feValuesElectronic.reinit (cellElectronic);
	  feValuesElectronic.get_function_values(phiTotRhoIn,cellPhiTotRhoIn);
	  feValuesElectronic.get_function_values(phiExt,cellPhiExt);

	  if(dftParameters::xc_id == 4)
	    {
	      // Get exc
	      std::vector<double> densityValueIn(num_quad_points_electronic),
		densityValueOut(num_quad_points_electronic);
	      std::vector<double> exchangeEnergyDensity(num_quad_points_electronic),
		corrEnergyDensity(num_quad_points_electronic);
	      std::vector<double> derExchEnergyWithInputDensity(num_quad_points_electronic),
		derCorrEnergyWithInputDensity(num_quad_points_electronic);
	      std::vector<double> derExchEnergyWithSigmaGradDenInput(num_quad_points_electronic),
		derCorrEnergyWithSigmaGradDenInput(num_quad_points_electronic);
	      std::vector<double> sigmaWithOutputGradDensity(num_quad_points_electronic),
		sigmaWithInputGradDensity(num_quad_points_electronic);
	      std::vector<double> gradRhoInDotgradRhoOut(num_quad_points_electronic);

	      for (unsigned int q_point=0; q_point<num_quad_points_electronic; ++q_point)
		{
		  densityValueIn[q_point] = rhoInValues.find(cellElectronic->id())->second[q_point];
		  densityValueOut[q_point] = rhoOutValues.find(cellElectronic->id())->second[q_point];
		  const double gradRhoInX = (gradRhoInValues.find(cellElectronic->id())->second[3*q_point + 0]);
		  const double gradRhoInY = (gradRhoInValues.find(cellElectronic->id())->second[3*q_point + 1]);
		  const double gradRhoInZ = (gradRhoInValues.find(cellElectronic->id())->second[3*q_point + 2]);
		  const double gradRhoOutX = (gradRhoOutValues.find(cellElectronic->id())->second[3*q_point + 0]);
		  const double gradRhoOutY = (gradRhoOutValues.find(cellElectronic->id())->second[3*q_point + 1]);
		  const double gradRhoOutZ = (gradRhoOutValues.find(cellElectronic->id())->second[3*q_point + 2]);
		  sigmaWithInputGradDensity[q_point] = gradRhoInX*gradRhoInX + gradRhoInY*gradRhoInY + gradRhoInZ*gradRhoInZ;
		  sigmaWithOutputGradDensity[q_point] = gradRhoOutX*gradRhoOutX + gradRhoOutY*gradRhoOutY + gradRhoOutZ*gradRhoOutZ;
		  gradRhoInDotgradRhoOut[q_point] = gradRhoInX*gradRhoOutX + gradRhoInY*gradRhoOutY + gradRhoInZ*gradRhoOutZ;
		}
	      xc_gga_exc(&funcX,num_quad_points_electronic,&densityValueOut[0],&sigmaWithOutputGradDensity[0],&exchangeEnergyDensity[0]);
	      xc_gga_exc(&funcC,num_quad_points_electronic,&densityValueOut[0],&sigmaWithOutputGradDensity[0],&corrEnergyDensity[0]);

	      xc_gga_vxc(&funcX,num_quad_points_electronic,&densityValueIn[0],&sigmaWithInputGradDensity[0],&derExchEnergyWithInputDensity[0],&derExchEnergyWithSigmaGradDenInput[0]);
	      xc_gga_vxc(&funcC,num_quad_points_electronic,&densityValueIn[0],&sigmaWithInputGradDensity[0],&derCorrEnergyWithInputDensity[0],&derCorrEnergyWithSigmaGradDenInput[0]);

	      for (unsigned int q_point = 0; q_point < num_quad_points_electronic; ++q_point)
		{
		  // Vxc computed with rhoIn
		  const double Vxc=derExchEnergyWithInputDensity[q_point]+derCorrEnergyWithInputDensity[q_point];
		  const double VxcGrad = 2.0*(derExchEnergyWithSigmaGradDenInput[q_point]+derCorrEnergyWithSigmaGradDenInput[q_point])*gradRhoInDotgradRhoOut[q_point];

		  excCorrPotentialTimesRho+=(Vxc*(rhoOutValues.find(cellElectronic->id())->second[q_point])+VxcGrad)*feValuesElectronic.JxW (q_point);

		  exchangeEnergy+=(exchangeEnergyDensity[q_point])*(rhoOutValues.find(cellElectronic->id())->second[q_point])*feValuesElectronic.JxW(q_point);

		  correlationEnergy+=(corrEnergyDensity[q_point])*(rhoOutValues.find(cellElectronic->id())->second[q_point])*feValuesElectronic.JxW(q_point);

		  electrostaticPotentialTimesRho+=(cellPhiTotRhoIn[q_point]
			                          +pseudoValuesElectronic.find(cellElectronic->id())->second[q_point]
			                          -cellPhiExt[q_point])
				  *(rhoOutValues.find(cellElectronic->id())->second[q_point])
				  *feValuesElectronic.JxW (q_point);

		  vSelfPotentialTimesRho+=cellPhiExt[q_point]*(rhoOutValues.find(cellElectronic->id())->second[q_point])*feValuesElectronic.JxW (q_point);

		}

	    }
	  else
	    {
	      // Get Exc
	      std::vector<double> densityValueIn(num_quad_points_electronic),
		densityValueOut(num_quad_points_electronic);
	      std::vector<double> exchangeEnergyVal(num_quad_points_electronic),
		corrEnergyVal(num_quad_points_electronic);
	      std::vector<double> exchangePotentialVal(num_quad_points_electronic),
		corrPotentialVal(num_quad_points_electronic);

	      for (unsigned int q_point=0; q_point<num_quad_points_electronic; ++q_point)
		{
		  densityValueIn[q_point] = rhoInValues.find(cellElectronic->id())->second[q_point];
		  densityValueOut[q_point] = rhoOutValues.find(cellElectronic->id())->second[q_point];
		}
	      xc_lda_exc(&funcX,num_quad_points_electronic,&densityValueOut[0],&exchangeEnergyVal[0]);
	      xc_lda_exc(&funcC,num_quad_points_electronic,&densityValueOut[0],&corrEnergyVal[0]);
	      xc_lda_vxc(&funcX,num_quad_points_electronic,&densityValueIn[0],&exchangePotentialVal[0]);
	      xc_lda_vxc(&funcC,num_quad_points_electronic,&densityValueIn[0],&corrPotentialVal[0]);

	      for (unsigned int q_point = 0; q_point < num_quad_points_electronic; ++q_point)
		{
		  excCorrPotentialTimesRho+=(exchangePotentialVal[q_point]+corrPotentialVal[q_point])*(rhoOutValues.find(cellElectronic->id())->second[q_point])*feValuesElectronic.JxW (q_point);

		  exchangeEnergy+=(exchangeEnergyVal[q_point])*(rhoOutValues.find(cellElectronic->id())->second[q_point])*feValuesElectronic.JxW(q_point);

		  correlationEnergy+=(corrEnergyVal[q_point])*(rhoOutValues.find(cellElectronic->id())->second[q_point])*feValuesElectronic.JxW(q_point);

		  electrostaticPotentialTimesRho+=(cellPhiTotRhoIn[q_point]
			                          +pseudoValuesElectronic.find(cellElectronic->id())->second[q_point]
			                          -cellPhiExt[q_point])
				  *(rhoOutValues.find(cellElectronic->id())->second[q_point])
				  *feValuesElectronic.JxW (q_point);

		  vSelfPotentialTimesRho+=cellPhiExt[q_point]*(rhoOutValues.find(cellElectronic->id())->second[q_point])*feValuesElectronic.JxW (q_point);

		}
	    }

	}



    for (; cellElectrostatic!=endcElectrostatic; ++cellElectrostatic)
      if (cellElectrostatic->is_locally_owned())
	{
	  // Compute values for current cell.
	  feValuesElectrostatic.reinit(cellElectrostatic);
	  feValuesElectrostatic.get_function_values(phiTotRhoOut,cellPhiTotRhoOut);
	  feValuesElectrostatic.get_function_values(phiExtElec,cellPhiExtElec);

	  for (unsigned int q_point = 0; q_point < num_quad_points_electrostatic; ++q_point)
	    {
	      electrostaticEnergyTotPot  += 0.5*(cellPhiTotRhoOut[q_point])*(rhoOutValuesElectrostatic.find(cellElectrostatic->id())->second[q_point])*feValuesElectrostatic.JxW(q_point);
	      vSelfPotentialElecTimesRho += cellPhiExtElec[q_point]*(rhoOutValuesElectrostatic.find(cellElectrostatic->id())->second[q_point])*feValuesElectrostatic.JxW (q_point);

	      vPseudoMinusVselfPotentialTimesRho+=
		     (pseudoValuesElectrostatic.find(cellElectrostatic->id())->second[q_point]
		     -cellPhiExtElec[q_point])
		     *(rhoOutValuesElectrostatic.find(cellElectrostatic->id())->second[q_point])
		     *feValuesElectrostatic.JxW (q_point);
	    }
	}

    const double potentialTimesRho=excCorrPotentialTimesRho+electrostaticPotentialTimesRho;

    double energy=-potentialTimesRho+exchangeEnergy+correlationEnergy+electrostaticEnergyTotPot
	          +vPseudoMinusVselfPotentialTimesRho;


    const double nuclearElectrostaticEnergy=internal::nuclearElectrostaticEnergyLocal(phiTotRhoOut,
										      localVselfs,
										      atomElectrostaticNodeIdToChargeMap);

    //sum over all processors
    double totalEnergy= dealii::Utilities::MPI::sum(energy, mpi_communicator);
    double totalpotentialTimesRho= dealii::Utilities::MPI::sum(potentialTimesRho, mpi_communicator);
    double totalexchangeEnergy= dealii::Utilities::MPI::sum(exchangeEnergy, mpi_communicator);
    double totalcorrelationEnergy= dealii::Utilities::MPI::sum(correlationEnergy, mpi_communicator);
    double totalelectrostaticEnergyPot= dealii::Utilities::MPI::sum(electrostaticEnergyTotPot, mpi_communicator);
    double totalNuclearElectrostaticEnergy = dealii::Utilities::MPI::sum(nuclearElectrostaticEnergy, mpi_communicator);



    //
    //total energy
    //
    totalEnergy+=bandEnergy;


    totalEnergy+=totalNuclearElectrostaticEnergy;

    const double allElectronElectrostaticEnergy=
      (totalelectrostaticEnergyPot+totalNuclearElectrostaticEnergy);

    double totalkineticEnergy=-totalpotentialTimesRho+bandEnergy;


    //output
    if (print)
      {
	internal::printEnergy(bandEnergy,
			      totalkineticEnergy,
			      totalexchangeEnergy,
			      totalcorrelationEnergy,
			      allElectronElectrostaticEnergy,
			      totalEnergy,
			      numberGlobalAtoms,
			      pcout,
			      dftParameters::reproducible_output,
			      dftParameters::verbosity);
      }

    return totalEnergy;
  }


  //compute energies
  double energyCalculator::computeEnergySpinPolarized
  (const dealii::DoFHandler<3> & dofHandlerElectrostatic,
   const dealii::DoFHandler<3> & dofHandlerElectronic,
   const dealii::QGauss<3> & quadratureElectrostatic,
   const dealii::QGauss<3> & quadratureElectronic,
   const std::vector<std::vector<double> > & eigenValues,
   const std::vector<double> & kPointWeights,
   const double fermiEnergy,
   const double fermiEnergyUp,
   const double fermiEnergyDown,
   const xc_func_type & funcX,
   const xc_func_type & funcC,
   const vectorType & phiTotRhoIn,
   const vectorType & phiTotRhoOut,
   const vectorType & phiExt,
   const vectorType & phiExtElec,
   const std::map<dealii::CellId, std::vector<double> > & rhoInValues,
   const std::map<dealii::CellId, std::vector<double> > & rhoOutValues,
   const std::map<dealii::CellId, std::vector<double> > & rhoOutValuesElectrostatic,
   const std::map<dealii::CellId, std::vector<double> > & gradRhoInValues,
   const std::map<dealii::CellId, std::vector<double> > & gradRhoOutValues,
   const std::map<dealii::CellId, std::vector<double> > & rhoInValuesSpinPolarized,
   const std::map<dealii::CellId, std::vector<double> > & rhoOutValuesSpinPolarized,
   const std::map<dealii::CellId, std::vector<double> > & gradRhoInValuesSpinPolarized,
   const std::map<dealii::CellId, std::vector<double> > & gradRhoOutValuesSpinPolarized,
   const std::vector<std::vector<double> > & localVselfs,
   const std::map<dealii::CellId, std::vector<double> > & pseudoValuesElectronic,
   const std::map<dealii::CellId, std::vector<double> > & pseudoValuesElectrostatic,
   const std::map<dealii::types::global_dof_index, double> & atomElectrostaticNodeIdToChargeMap,
   const unsigned int numberGlobalAtoms,
   const unsigned int lowerBoundKindex,
   const unsigned int scfConverged,
   const bool print) const
  {
    dealii::FEValues<3> feValuesElectrostatic (dofHandlerElectrostatic.get_fe(), quadratureElectrostatic, dealii::update_values | dealii::update_JxW_values);
    dealii::FEValues<3> feValuesElectronic (dofHandlerElectronic.get_fe(), quadratureElectronic, dealii::update_values | dealii::update_JxW_values);

    const unsigned int   num_quad_points_electrostatic    = quadratureElectrostatic.size();
    const unsigned int   num_quad_points_electronic    = quadratureElectronic.size();

    std::vector<double> cellPhiTotRhoIn(num_quad_points_electronic);
    std::vector<double> cellPhiTotRhoOut(num_quad_points_electrostatic);
    std::vector<double> cellPhiExt(num_quad_points_electronic);
    std::vector<double> cellPhiExtElec(num_quad_points_electrostatic);
    //
    const dealii::ConditionalOStream scout (std::cout, (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0)) ;
    const double bandEnergy=
      dealii::Utilities::MPI::sum(internal::localBandEnergy(eigenValues,
							    kPointWeights,
							    fermiEnergy,
							    fermiEnergyUp,
							    fermiEnergyDown,
							    dftParameters::TVal,
							    dftParameters::spinPolarized,
							    scout,
							    interpoolcomm,
							    lowerBoundKindex,
							    (dftParameters::verbosity+scfConverged)), interpoolcomm);

    double excCorrPotentialTimesRho=0.0, electrostaticPotentialTimesRho=0.0 , exchangeEnergy = 0.0, correlationEnergy = 0.0, electrostaticEnergyTotPot = 0.0, vSelfPotentialTimesRho = 0.0, vSelfPotentialElecTimesRho = 0.0, vPseudoMinusVselfPotentialTimesRho=0.0;

    //parallel loop over all elements
    typename dealii::DoFHandler<3>::active_cell_iterator cellElectrostatic = dofHandlerElectrostatic.begin_active(), endcElectrostatic = dofHandlerElectrostatic.end();

    typename dealii::DoFHandler<3>::active_cell_iterator cellElectronic = dofHandlerElectronic.begin_active(), endcElectronic = dofHandlerElectronic.end();

    for (; cellElectronic!=endcElectronic; ++cellElectronic)
      if (cellElectronic->is_locally_owned())
	{

	  feValuesElectronic.reinit (cellElectronic);
	  feValuesElectronic.get_function_values(phiTotRhoIn,cellPhiTotRhoIn);
	  feValuesElectronic.get_function_values(phiExt,cellPhiExt);

	  if(dftParameters::xc_id == 4)
	    {
	      // Get exc
	      std::vector<double> densityValueIn(2*num_quad_points_electronic),
		densityValueOut(2*num_quad_points_electronic);
	      std::vector<double> exchangeEnergyDensity(num_quad_points_electronic),
		corrEnergyDensity(num_quad_points_electronic);
	      std::vector<double> derExchEnergyWithInputDensity(2*num_quad_points_electronic),
		derCorrEnergyWithInputDensity(2*num_quad_points_electronic);
	      std::vector<double> derExchEnergyWithSigmaGradDenInput(3*num_quad_points_electronic),
		derCorrEnergyWithSigmaGradDenInput(3*num_quad_points_electronic);
	      std::vector<double> sigmaWithOutputGradDensity(3*num_quad_points_electronic),
		sigmaWithInputGradDensity(3*num_quad_points_electronic);
	      std::vector<double> gradRhoInDotgradRhoOut(3*num_quad_points_electronic);

	      for (unsigned int q_point=0; q_point<num_quad_points_electronic; ++q_point)
		{
		  densityValueIn[2*q_point+0] = rhoInValuesSpinPolarized.find(cellElectronic->id())->second[2*q_point+0];
		  densityValueIn[2*q_point+1] = rhoInValuesSpinPolarized.find(cellElectronic->id())->second[2*q_point+1];
		  densityValueOut[2*q_point+0] = rhoOutValuesSpinPolarized.find(cellElectronic->id())->second[2*q_point+0];
		  densityValueOut[2*q_point+1] = rhoOutValuesSpinPolarized.find(cellElectronic->id())->second[2*q_point+1];
		  //
		  const double gradRhoInX1 = (gradRhoInValuesSpinPolarized.find(cellElectronic->id())->second[6*q_point + 0]);
		  const double gradRhoInY1 = (gradRhoInValuesSpinPolarized.find(cellElectronic->id())->second[6*q_point + 1]);
		  const double gradRhoInZ1 = (gradRhoInValuesSpinPolarized.find(cellElectronic->id())->second[6*q_point + 2]);
		  const double gradRhoOutX1 = (gradRhoOutValuesSpinPolarized.find(cellElectronic->id())->second[6*q_point + 0]);
		  const double gradRhoOutY1 = (gradRhoOutValuesSpinPolarized.find(cellElectronic->id())->second[6*q_point + 1]);
		  const double gradRhoOutZ1 = (gradRhoOutValuesSpinPolarized.find(cellElectronic->id())->second[6*q_point + 2]);
		  //
		  const double gradRhoInX2 = (gradRhoInValuesSpinPolarized.find(cellElectronic->id())->second[6*q_point + 3]);
		  const double gradRhoInY2 = (gradRhoInValuesSpinPolarized.find(cellElectronic->id())->second[6*q_point + 4]);
		  const double gradRhoInZ2 = (gradRhoInValuesSpinPolarized.find(cellElectronic->id())->second[6*q_point + 5]);
		  const double gradRhoOutX2 = (gradRhoOutValuesSpinPolarized.find(cellElectronic->id())->second[6*q_point + 3]);
		  const double gradRhoOutY2 = (gradRhoOutValuesSpinPolarized.find(cellElectronic->id())->second[6*q_point + 4]);
		  const double gradRhoOutZ2 = (gradRhoOutValuesSpinPolarized.find(cellElectronic->id())->second[6*q_point + 5]);
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
	      xc_gga_exc(&funcX,num_quad_points_electronic,&densityValueOut[0],&sigmaWithOutputGradDensity[0],&exchangeEnergyDensity[0]);
	      xc_gga_exc(&funcC,num_quad_points_electronic,&densityValueOut[0],&sigmaWithOutputGradDensity[0],&corrEnergyDensity[0]);

	      xc_gga_vxc(&funcX,num_quad_points_electronic,&densityValueIn[0],&sigmaWithInputGradDensity[0],&derExchEnergyWithInputDensity[0],&derExchEnergyWithSigmaGradDenInput[0]);
	      xc_gga_vxc(&funcC,num_quad_points_electronic,&densityValueIn[0],&sigmaWithInputGradDensity[0],&derCorrEnergyWithInputDensity[0],&derCorrEnergyWithSigmaGradDenInput[0]);

	      for (unsigned int q_point=0; q_point<num_quad_points_electronic; ++q_point)
		{
		  // Vxc computed with rhoIn
		  double Vxc=derExchEnergyWithInputDensity[2*q_point+0]+derCorrEnergyWithInputDensity[2*q_point+0];
		  double VxcGrad = 2.0*(derExchEnergyWithSigmaGradDenInput[3*q_point+0]+derCorrEnergyWithSigmaGradDenInput[3*q_point+0])*gradRhoInDotgradRhoOut[3*q_point+0];

		  VxcGrad += 2.0*(derExchEnergyWithSigmaGradDenInput[3*q_point+1]+derCorrEnergyWithSigmaGradDenInput[3*q_point+1])*gradRhoInDotgradRhoOut[3*q_point+1];

		  VxcGrad += 2.0*(derExchEnergyWithSigmaGradDenInput[3*q_point+2]+derCorrEnergyWithSigmaGradDenInput[3*q_point+2] )*gradRhoInDotgradRhoOut[3*q_point+2];

		  excCorrPotentialTimesRho+=(Vxc*(rhoOutValuesSpinPolarized.find(cellElectronic->id())->second[2*q_point+0])+VxcGrad)*feValuesElectronic.JxW (q_point);

		  Vxc=derExchEnergyWithInputDensity[2*q_point+1]+derCorrEnergyWithInputDensity[2*q_point+1];

		  excCorrPotentialTimesRho+=(Vxc*(rhoOutValuesSpinPolarized.find(cellElectronic->id())->second[2*q_point+1]))*feValuesElectronic.JxW (q_point);

		  exchangeEnergy+=(exchangeEnergyDensity[q_point])*(rhoOutValues.find(cellElectronic->id())->second[q_point])*feValuesElectronic.JxW(q_point);

		  correlationEnergy+=(corrEnergyDensity[q_point])*(rhoOutValues.find(cellElectronic->id())->second[q_point])*feValuesElectronic.JxW(q_point);

		  electrostaticPotentialTimesRho+=(cellPhiTotRhoIn[q_point]
			                          +pseudoValuesElectronic.find(cellElectronic->id())->second[q_point]
			                          -cellPhiExt[q_point])
				  *(rhoOutValues.find(cellElectronic->id())->second[q_point])
				  *feValuesElectronic.JxW (q_point);

		  vSelfPotentialTimesRho+=cellPhiExt[q_point]*(rhoOutValues.find(cellElectronic->id())->second[q_point])*feValuesElectronic.JxW (q_point);

		}
	    }
	  else
	    {
	      // Get Exc
	      std::vector<double> densityValueIn(2*num_quad_points_electronic),
		densityValueOut(2*num_quad_points_electronic);
	      std::vector<double> exchangeEnergyVal(num_quad_points_electronic),
		corrEnergyVal(num_quad_points_electronic);
	      std::vector<double> exchangePotentialVal(2*num_quad_points_electronic),
		corrPotentialVal(2*num_quad_points_electronic);
	      for (unsigned int q_point=0; q_point<2*num_quad_points_electronic; ++q_point)
		{
		  densityValueIn[q_point] = rhoInValuesSpinPolarized.find(cellElectronic->id())->second[q_point];
		  densityValueOut[q_point] = rhoOutValuesSpinPolarized.find(cellElectronic->id())->second[q_point];
		}
	      //

	      xc_lda_exc(&funcX,num_quad_points_electronic,&densityValueOut[0],&exchangeEnergyVal[0]);
	      xc_lda_exc(&funcC,num_quad_points_electronic,&densityValueOut[0],&corrEnergyVal[0]);
	      xc_lda_vxc(&funcX,num_quad_points_electronic,&densityValueIn[0],&exchangePotentialVal[0]);
	      xc_lda_vxc(&funcC,num_quad_points_electronic,&densityValueIn[0],&corrPotentialVal[0]);

	      for (unsigned int q_point = 0; q_point < num_quad_points_electronic; ++q_point)
		{
		  // Vxc computed with rhoIn
		  double Vxc=exchangePotentialVal[2*q_point]+corrPotentialVal[2*q_point] ;
		  excCorrPotentialTimesRho+=Vxc*(rhoOutValuesSpinPolarized.find(cellElectronic->id())->second[2*q_point])*feValuesElectronic.JxW (q_point);
		  //
		  Vxc= exchangePotentialVal[2*q_point+1]+corrPotentialVal[2*q_point+1] ;
		  excCorrPotentialTimesRho+=Vxc*(rhoOutValuesSpinPolarized.find(cellElectronic->id())->second[2*q_point+1])*feValuesElectronic.JxW (q_point);
		  //
		  exchangeEnergy+=(exchangeEnergyVal[q_point])*(rhoOutValues.find(cellElectronic->id())->second[q_point])*feValuesElectronic.JxW(q_point);
		  correlationEnergy+=(corrEnergyVal[q_point])*(rhoOutValues.find(cellElectronic->id())->second[q_point])*feValuesElectronic.JxW(q_point) ;

		  electrostaticPotentialTimesRho+=(cellPhiTotRhoIn[q_point]
			                          +pseudoValuesElectronic.find(cellElectronic->id())->second[q_point]
			                          -cellPhiExt[q_point])
				  *(rhoOutValues.find(cellElectronic->id())->second[q_point])
				  *feValuesElectronic.JxW (q_point);

		  vSelfPotentialTimesRho+=cellPhiExt[q_point]*(rhoOutValues.find(cellElectronic->id())->second[q_point])*feValuesElectronic.JxW (q_point);

		}
	    }

	}


    for(; cellElectrostatic!=endcElectrostatic; ++cellElectrostatic)
      if(cellElectrostatic->is_locally_owned())
	{
	  // Compute values for current cell.
	  feValuesElectrostatic.reinit(cellElectrostatic);
	  feValuesElectrostatic.get_function_values(phiTotRhoOut,cellPhiTotRhoOut);
          feValuesElectrostatic.get_function_values(phiExtElec,cellPhiExtElec);

	  for(unsigned int q_point = 0; q_point < num_quad_points_electrostatic; ++q_point)
	    {
	      electrostaticEnergyTotPot+=0.5*(cellPhiTotRhoOut[q_point])*(rhoOutValuesElectrostatic.find(cellElectrostatic->id())->second[q_point])*feValuesElectrostatic.JxW(q_point);
	      vSelfPotentialElecTimesRho += cellPhiExtElec[q_point]*(rhoOutValuesElectrostatic.find(cellElectrostatic->id())->second[q_point])*feValuesElectrostatic.JxW (q_point);

	      vPseudoMinusVselfPotentialTimesRho+=
		     (pseudoValuesElectrostatic.find(cellElectrostatic->id())->second[q_point]
		     -cellPhiExtElec[q_point])
		     *(rhoOutValuesElectrostatic.find(cellElectrostatic->id())->second[q_point])
		     *feValuesElectrostatic.JxW (q_point);
	    }
	}



    const double potentialTimesRho=excCorrPotentialTimesRho+electrostaticPotentialTimesRho;

    double energy=-potentialTimesRho+exchangeEnergy+correlationEnergy+electrostaticEnergyTotPot
	          +vPseudoMinusVselfPotentialTimesRho;

    const double nuclearElectrostaticEnergy=internal::nuclearElectrostaticEnergyLocal(phiTotRhoOut,
										      localVselfs,
										      atomElectrostaticNodeIdToChargeMap);

    //sum over all processors
    double totalEnergy= dealii::Utilities::MPI::sum(energy, mpi_communicator);
    double totalpotentialTimesRho= dealii::Utilities::MPI::sum(potentialTimesRho, mpi_communicator);
    double totalexchangeEnergy= dealii::Utilities::MPI::sum(exchangeEnergy, mpi_communicator);
    double totalcorrelationEnergy= dealii::Utilities::MPI::sum(correlationEnergy, mpi_communicator);
    double totalelectrostaticEnergyPot= dealii::Utilities::MPI::sum(electrostaticEnergyTotPot, mpi_communicator);
    double totalNuclearElectrostaticEnergy = dealii::Utilities::MPI::sum(nuclearElectrostaticEnergy, mpi_communicator);

    //
    //total energy
    //
    totalEnergy+=bandEnergy;


    totalEnergy+=totalNuclearElectrostaticEnergy;

    const double allElectronElectrostaticEnergy=
      (totalelectrostaticEnergyPot+totalNuclearElectrostaticEnergy);


    double totalkineticEnergy=-totalpotentialTimesRho+bandEnergy;

    //output
    if (print)
      {
	internal::printEnergy(bandEnergy,
			      totalkineticEnergy,
			      totalexchangeEnergy,
			      totalcorrelationEnergy,
			      allElectronElectrostaticEnergy,
			      totalEnergy,
			      numberGlobalAtoms,
			      pcout,
			      dftParameters::reproducible_output,
			      dftParameters::verbosity);
      }

    return totalEnergy;
  }
}
