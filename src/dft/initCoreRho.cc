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
// @author Phani Motamarri
//

//
//Initialize rho by reading in single-atom electron-density and fit a spline
//
#include <dftParameters.h>


	template<unsigned int FEOrder,unsigned int FEOrderElectro>
void dftClass<FEOrder,FEOrderElectro>::initCoreRho()
{
	//clear existing data
	d_rhoCore.clear();
	d_gradRhoCore.clear();
	d_gradRhoCoreAtoms.clear();
	d_hessianRhoCore.clear();
	d_hessianRhoCoreAtoms.clear();

	//Reading single atom rho initial guess
	pcout << std::endl << "Reading data for core electron-density to be used in nonlinear core-correction....."<<std::endl;
	std::map<unsigned int, alglib::spline1dinterpolant> coreDenSpline;
	std::map<unsigned int, std::vector<std::vector<double> > > singleAtomCoreElectronDensity;
	std::map<unsigned int, double> outerMostPointCoreDen;
	std::map<unsigned int, unsigned int> atomTypeNLCCFlagMap;
	const double truncationTol = 1e-12;
	unsigned int fileReadFlag = 0;

  double maxCoreRhoTail=0.0;
	//loop over atom types
	for(std::set<unsigned int>::iterator it = atomTypes.begin(); it != atomTypes.end(); it++)
	{
		char coreDensityFile[256];
		if(dftParameters::isPseudopotential)
		{
			sprintf(coreDensityFile,"temp/z%u/coreDensity.inp",*it);
		}

		unsigned int fileReadFlag = dftUtils::readPsiFile(2, singleAtomCoreElectronDensity[*it], coreDensityFile);

		atomTypeNLCCFlagMap[*it] = fileReadFlag;

		if(dftParameters::verbosity>=4)
			pcout<<"Atomic number: "<<*it<<" NLCC flag: "<<fileReadFlag<<std::endl;

		if(fileReadFlag > 0)
		{
			unsigned int numRows = singleAtomCoreElectronDensity[*it].size()-1;
			std::vector<double> xData(numRows), yData(numRows);

			unsigned int maxRowId=0;
			for(unsigned int irow = 0; irow < numRows; ++irow)
			{
				xData[irow] = singleAtomCoreElectronDensity[*it][irow][0];
				yData[irow] = std::abs(singleAtomCoreElectronDensity[*it][irow][1]);

				if(yData[irow] > truncationTol)
					maxRowId=irow;
			}

			//interpolate rho
			alglib::real_1d_array x;
			x.setcontent(numRows,&xData[0]);
			alglib::real_1d_array y;
			y.setcontent(numRows,&yData[0]);
			alglib::ae_int_t natural_bound_type_L = 1;
			alglib::ae_int_t natural_bound_type_R = 1;
			//const double slopeL = (singleAtomCoreElectronDensity[*it][1][1]- singleAtomCoreElectronDensity[*it][0][1])/(singleAtomCoreElectronDensity[*it][1][0]-singleAtomCoreElectronDensity[*it][0][0]);
			spline1dbuildcubic(x, y, numRows, natural_bound_type_L, 0.0, natural_bound_type_R, 0.0, coreDenSpline[*it]);
			outerMostPointCoreDen[*it]= xData[maxRowId];

      if(outerMostPointCoreDen[*it]>maxCoreRhoTail)
        maxCoreRhoTail = outerMostPointCoreDen[*it];

			if(dftParameters::verbosity>=4)
				pcout<<" Atomic number: "<<*it<<" Outermost Point Core Den: "<<outerMostPointCoreDen[*it]<<std::endl;
		}
	}

  if(maxCoreRhoTail < d_coreRhoTail)
    d_coreRhoTail = maxCoreRhoTail;

  const double cellCenterCutOff=d_coreRhoTail+5.0;

	if(dftParameters::verbosity>=2)
		pcout << " d_coreRhoTail adjusted to " << d_coreRhoTail << std::endl ;

	//
	//Initialize rho
	//
	const Quadrature<3> &  quadrature_formula=matrix_free_data.get_quadrature(d_densityQuadratureId);
	FEValues<3> fe_values (FE, quadrature_formula, update_quadrature_points);
	const unsigned int n_q_points = quadrature_formula.size();

	//
	//get number of global charges
	//
	const int numberGlobalCharges = atomLocations.size();

	//
	//get number of image charges used only for periodic
	//
	const int numberImageCharges = d_imageIdsTrunc.size();

	//
	//loop over elements
	//
	typename DoFHandler<3>::active_cell_iterator cell = dofHandler.begin_active(), endc = dofHandler.end();
	for (; cell!=endc; ++cell)
	{
		if (cell->is_locally_owned())
		{
			fe_values.reinit(cell);

			std::vector<double> & rhoCoreQuadValues = d_rhoCore[cell->id()];
			rhoCoreQuadValues.resize(n_q_points,0.0);

			for (unsigned int q = 0; q < n_q_points; ++q)
			{
				const Point<3> & quadPoint=fe_values.quadrature_point(q);
				double rhoValueAtQuadPt = 0.0;

				//loop over atoms
				for (unsigned int n = 0; n < atomLocations.size(); n++)
				{
          if (atomTypeNLCCFlagMap[atomLocations[n][0]]==0)
            continue;

					Point<3> atom(atomLocations[n][2],atomLocations[n][3],atomLocations[n][4]);
					double distanceToAtom = quadPoint.distance(atom);
					if(distanceToAtom <= d_coreRhoTail)
					{
						rhoValueAtQuadPt += alglib::spline1dcalc(coreDenSpline[atomLocations[n][0]], distanceToAtom);
					}
					else
					{
						rhoValueAtQuadPt += 0.0;
					}
				}

				//loop over image charges
				for(int iImageCharge = 0; iImageCharge < numberImageCharges; ++iImageCharge)
				{
					int masterAtomId = d_imageIdsTrunc[iImageCharge];

          if (atomTypeNLCCFlagMap[atomLocations[masterAtomId][0]]==0)
            continue;

					Point<3> imageAtom(d_imagePositionsTrunc[iImageCharge][0],
							d_imagePositionsTrunc[iImageCharge][1],
							d_imagePositionsTrunc[iImageCharge][2]);

					double distanceToAtom = quadPoint.distance(imageAtom);
          
					if(distanceToAtom <= d_coreRhoTail)
					{
						rhoValueAtQuadPt += alglib::spline1dcalc(coreDenSpline[atomLocations[masterAtomId][0]], distanceToAtom);
					}
					else
					{
						rhoValueAtQuadPt += 0.0;
					}

				}

				rhoCoreQuadValues[q] = std::abs(rhoValueAtQuadPt);
			}
		}
	}

  Tensor<1,3,double> zeroTensor1;
  for (unsigned int i=0; i<3;i++)
    zeroTensor1[i]=0.0;

  Tensor<2,3,double> zeroTensor2;

  for (unsigned int i=0; i<3;i++)
    for (unsigned int j=0; j<3;j++)    
      zeroTensor2[i][j]=0.0;  

	//loop over elements
	if(dftParameters::xcFamilyType=="GGA" || dftParameters::nonLinearCoreCorrection == true)
	{
		//
		cell = dofHandler.begin_active();
		for(; cell!=endc; ++cell)
		{
			if(cell->is_locally_owned())
			{
				fe_values.reinit(cell);

				std::vector<double> & gradRhoCoreQuadValues = d_gradRhoCore[cell->id()];
				gradRhoCoreQuadValues.resize(n_q_points*3,0.0);

				std::vector<double> & hessianRhoCoreQuadValues = d_hessianRhoCore[cell->id()];
				if(dftParameters::xcFamilyType=="GGA")
					hessianRhoCoreQuadValues.resize(n_q_points*9,0.0);

				std::vector<Tensor<1,3,double> > gradRhoCoreAtom(n_q_points,zeroTensor1);
				std::vector<Tensor<2,3,double> > hessianRhoCoreAtom(n_q_points,zeroTensor2);


				//loop over atoms
				for(unsigned int iAtom = 0; iAtom < atomLocations.size(); ++iAtom)
				{
					Point<3> atom(atomLocations[iAtom][2],atomLocations[iAtom][3],atomLocations[iAtom][4]);
					bool isCoreRhoDataInCell=false;

          if (atomTypeNLCCFlagMap[atomLocations[iAtom][0]]==0)
            continue;

					if (atom.distance(cell->center())>cellCenterCutOff)
            continue;            

					//loop over quad points
					for(unsigned int q = 0; q < n_q_points; ++q)
					{
						Point<3> quadPoint = fe_values.quadrature_point(q);
            Tensor<1,3,double> diff=quadPoint-atom; 
						double distanceToAtom = quadPoint.distance(atom);

            if (dftParameters::floatingNuclearCharges && distanceToAtom<1.0e-4)
            {
              if(dftParameters::verbosity>=4)
                std::cout<<"Atom close to quad point, iatom: "<<iAtom<<std::endl;

              distanceToAtom=1.0e-4;
              diff[0]=(1.0e-4)/std::sqrt(3.0);
              diff[1]=(1.0e-4)/std::sqrt(3.0);
              diff[2]=(1.0e-4)/std::sqrt(3.0);              
            }

						double value,radialDensityFirstDerivative,radialDensitySecondDerivative;
						if(distanceToAtom <= d_coreRhoTail)
						{

							alglib::spline1ddiff(coreDenSpline[atomLocations[iAtom][0]],
									distanceToAtom,
									value,
									radialDensityFirstDerivative,
									radialDensitySecondDerivative);

							isCoreRhoDataInCell = true;

						}
						else
						{
							radialDensityFirstDerivative = 0.0;
							radialDensitySecondDerivative = 0.0;
						}

						gradRhoCoreAtom[q] = radialDensityFirstDerivative*diff/distanceToAtom;
						gradRhoCoreQuadValues[3*q + 0] += gradRhoCoreAtom[q][0];
						gradRhoCoreQuadValues[3*q + 1] += gradRhoCoreAtom[q][1];
						gradRhoCoreQuadValues[3*q + 2] += gradRhoCoreAtom[q][2];

						if(dftParameters::xcFamilyType=="GGA")
						{
							for(unsigned int iDim = 0; iDim < 3; ++iDim)
							{
								for(unsigned int jDim = 0; jDim < 3; ++jDim)
								{
									double temp = (radialDensitySecondDerivative -radialDensityFirstDerivative/distanceToAtom)*diff[iDim]*diff[jDim]/(distanceToAtom*distanceToAtom);
									if(iDim == jDim)
										temp += radialDensityFirstDerivative/distanceToAtom;

									hessianRhoCoreAtom[q][iDim][jDim] = temp;
									hessianRhoCoreQuadValues[9*q + 3*iDim + jDim] += temp;
								}
							}
						}


					}//end loop over quad points
					if(isCoreRhoDataInCell)
					{
						std::vector<double> & gradRhoCoreAtomCell = d_gradRhoCoreAtoms[iAtom][cell->id()];
						gradRhoCoreAtomCell.resize(n_q_points*3,0.0);

						std::vector<double> & hessianRhoCoreAtomCell = d_hessianRhoCoreAtoms[iAtom][cell->id()];
						if(dftParameters::xcFamilyType=="GGA")
							hessianRhoCoreAtomCell.resize(n_q_points*9,0.0);

						for(unsigned int q = 0; q < n_q_points; ++q)
						{
							gradRhoCoreAtomCell[3*q+0] = gradRhoCoreAtom[q][0];
							gradRhoCoreAtomCell[3*q+1] = gradRhoCoreAtom[q][1];
							gradRhoCoreAtomCell[3*q+2] = gradRhoCoreAtom[q][2];

							if(dftParameters::xcFamilyType=="GGA")
							{
								for(unsigned int iDim = 0; iDim < 3; ++iDim)
								{
									for(unsigned int jDim = 0; jDim < 3; ++jDim)
									{
										hessianRhoCoreAtomCell[9*q + 3*iDim + jDim] = hessianRhoCoreAtom[q][iDim][jDim];
									}
								}
							}
						}//q_point loop
					}//if loop
				}//loop over atoms

				//loop over image charges
				for(unsigned int iImageCharge = 0; iImageCharge < numberImageCharges; ++iImageCharge)
				{
					const int masterAtomId = d_imageIdsTrunc[iImageCharge];          
          if (atomTypeNLCCFlagMap[atomLocations[masterAtomId][0]]==0)
            continue;

					Point<3> imageAtom(d_imagePositionsTrunc[iImageCharge][0],
							d_imagePositionsTrunc[iImageCharge][1],
							d_imagePositionsTrunc[iImageCharge][2]);

					if (imageAtom.distance(cell->center())>cellCenterCutOff)
            continue;  

					bool isCoreRhoDataInCell = false;

					// loop over quad points
					for (unsigned int q = 0; q < n_q_points; ++q)
					{
						Point<3> quadPoint=fe_values.quadrature_point(q);
            Tensor<1,3,double> diff=quadPoint-imageAtom; 
						double distanceToAtom = quadPoint.distance(imageAtom);

            if (dftParameters::floatingNuclearCharges && distanceToAtom<1.0e-4)
            {
              distanceToAtom=1.0e-4;
              diff[0]=(1.0e-4)/std::sqrt(3.0);
              diff[1]=(1.0e-4)/std::sqrt(3.0);
              diff[2]=(1.0e-4)/std::sqrt(3.0);              
            }

						double value,radialDensityFirstDerivative,radialDensitySecondDerivative;
						if(distanceToAtom <= d_coreRhoTail)
						{

							alglib::spline1ddiff(coreDenSpline[atomLocations[masterAtomId][0]],
									distanceToAtom,
									value,
									radialDensityFirstDerivative,
									radialDensitySecondDerivative);

							isCoreRhoDataInCell = true;

						}
						else
						{
							radialDensityFirstDerivative = 0.0;
							radialDensitySecondDerivative = 0.0;
						}

						gradRhoCoreAtom[q] = radialDensityFirstDerivative*diff/distanceToAtom;
						gradRhoCoreQuadValues[3*q + 0] += gradRhoCoreAtom[q][0];
						gradRhoCoreQuadValues[3*q + 1] += gradRhoCoreAtom[q][1];
						gradRhoCoreQuadValues[3*q + 2] += gradRhoCoreAtom[q][2];

						if(dftParameters::xcFamilyType=="GGA")
						{
							for(unsigned int iDim = 0; iDim < 3; ++iDim)
							{
								for(unsigned int jDim = 0; jDim < 3; ++jDim)
								{
									double temp = (radialDensitySecondDerivative -radialDensityFirstDerivative/distanceToAtom)*diff[iDim]*diff[jDim]/(distanceToAtom*distanceToAtom);
									if(iDim == jDim)
										temp += radialDensityFirstDerivative/distanceToAtom;
									hessianRhoCoreAtom[q][iDim][jDim] = temp;
									hessianRhoCoreQuadValues[9*q + 3*iDim + jDim] += temp;
								}
							}

						}

					}//quad point loop

					if(isCoreRhoDataInCell)
					{
						std::vector<double> & gradRhoCoreAtomCell = d_gradRhoCoreAtoms[numberGlobalCharges+iImageCharge][cell->id()];
						gradRhoCoreAtomCell.resize(n_q_points*3);

						std::vector<double> & hessianRhoCoreAtomCell = d_hessianRhoCoreAtoms[numberGlobalCharges+iImageCharge][cell->id()];
						if(dftParameters::xcFamilyType=="GGA")
							hessianRhoCoreAtomCell.resize(n_q_points*9);

						for(unsigned int q = 0; q < n_q_points; ++q)
						{
							gradRhoCoreAtomCell[3*q+0] = gradRhoCoreAtom[q][0];
							gradRhoCoreAtomCell[3*q+1] = gradRhoCoreAtom[q][1];
							gradRhoCoreAtomCell[3*q+2] = gradRhoCoreAtom[q][2];

							if(dftParameters::xcFamilyType=="GGA")
							{
								for(unsigned int iDim = 0; iDim < 3; ++iDim)
								{
									for(unsigned int jDim = 0; jDim < 3; ++jDim)
									{
										hessianRhoCoreAtomCell[9*q + 3*iDim + jDim] = hessianRhoCoreAtom[q][iDim][jDim];
									}
								}
							}
						}//q_point loop
					}//if loop

				}//end of image charges

			}//cell locally owned check

		}//cell loop
	}
}



