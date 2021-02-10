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
// @author Sambit Das
//

//
//Initialize rho by reading in single-atom electron-density and fit a spline
//

	template<unsigned int FEOrder,unsigned int FEOrderElectro>
void dftClass<FEOrder,FEOrderElectro>::initAtomicRho(const bool reusePreviousScalingFactor)
{
  computing_timer.enter_section("initialize atomic density for density splitting approach in md and relaxations");    
	//clear existing data
	d_rhoAtomsValues.clear();
	d_gradRhoAtomsValues.clear();
	d_derRRhoAtomsValuesSeparate.clear();
	d_der2XRRhoAtomsValuesSeparate.clear();

	//Reading single atom rho initial guess
	pcout <<std::endl<< "Reading initial guess for electron-density....."<<std::endl;
	std::map<unsigned int, alglib::spline1dinterpolant> denSpline;
	std::map<unsigned int, std::vector<std::vector<double> > > singleAtomElectronDensity;
	std::map<unsigned int, double> outerMostPointDen;
	const double truncationTol=1e-10;//1e-8
  double maxRhoTail=0.0;
 
	//loop over atom types
	for (std::set<unsigned int>::iterator it=atomTypes.begin(); it!=atomTypes.end(); it++)
	{
		char densityFile[256];
		if(dftParameters::isPseudopotential)
		{
			sprintf(densityFile,"temp/z%u/density.inp",*it);
		}
		else
		{
			sprintf(densityFile, "%s/data/electronicStructure/allElectron/z%u/singleAtomData/density.inp", DFT_PATH, *it);
		}

		dftUtils::readFile(2, singleAtomElectronDensity[*it], densityFile);
		unsigned int numRows = singleAtomElectronDensity[*it].size()-1;
		std::vector<double> xData(numRows), yData(numRows);

		unsigned int maxRowId=0;
		for(unsigned int irow = 0; irow < numRows; ++irow)
		{
			xData[irow] = singleAtomElectronDensity[*it][irow][0];
			yData[irow] = singleAtomElectronDensity[*it][irow][1];

			if (yData[irow]>truncationTol)
				maxRowId=irow;
		}

		//interpolate rho
		alglib::real_1d_array x;
		x.setcontent(numRows,&xData[0]);
		alglib::real_1d_array y;
		y.setcontent(numRows,&yData[0]);
		alglib::ae_int_t natural_bound_type_L = 1;
		alglib::ae_int_t natural_bound_type_R = 1;
		spline1dbuildcubic(x, y, numRows, natural_bound_type_L, 0.0, natural_bound_type_R, 0.0, denSpline[*it]);
		outerMostPointDen[*it]= xData[maxRowId];

    if(outerMostPointDen[*it]>maxRhoTail)
      maxRhoTail = outerMostPointDen[*it];
	}

	//Initialize rho
  const Quadrature<3> &  quadrature_formula=matrix_free_data.get_quadrature(d_densityQuadratureId);
	const unsigned int n_q_points    = quadrature_formula.size();

	//
	//get number of image charges used only for periodic
	//
	const int numberImageCharges = d_imageIdsTrunc.size();

	//IndexSet locallyOwnedSet;
	//DoFTools::extract_locally_owned_dofs(d_dofHandlerPRefined,locallyOwnedSet);
	//std::vector<IndexSet::size_type> locallyOwnedDOFs;
	//locallyOwnedSet.fill_index_vector(locallyOwnedDOFs);
	//unsigned int numberDofs = locallyOwnedDOFs.size();
	std::map<types::global_dof_index, Point<3> > supportPointsRhoNodal;
	DoFTools::map_dofs_to_support_points(MappingQ1<3,3>(), d_dofHandlerRhoNodal, supportPointsRhoNodal);

	//d_matrixFreeDataPRefined.initialize_dof_vector(d_rhoInNodalValues);
	std::vector<distributedCPUVec<double>> singleAtomsRho(atomLocations.size()+numberImageCharges);
  std::vector<distributedCPUVec<double>> singleAtomsDerRRho(3*(atomLocations.size()+numberImageCharges));
  std::vector<distributedCPUVec<double>> singleAtomsDer2RRho(9*(atomLocations.size()+numberImageCharges));

  d_atomicRho=0.0;

  for(unsigned int iAtom = 0; iAtom < atomLocations.size()+numberImageCharges; ++iAtom)
  {
    if (iAtom==0)
      d_matrixFreeDataPRefined.initialize_dof_vector(singleAtomsRho[iAtom],d_nonPeriodicDensityDofHandlerIndexElectro);
    else
      singleAtomsRho[iAtom].reinit(singleAtomsRho[0]);

    singleAtomsRho[iAtom]=0.0;
  }

  if (dftParameters::isBOMD && dftParameters::isXLBOMD)
  {
    for(unsigned int i = 0; i < singleAtomsDerRRho.size(); ++i)
    {
      singleAtomsDerRRho[i].reinit(singleAtomsRho[0]);
      singleAtomsDerRRho[i]=0.0;
    }

    if (dftParameters::xcFamilyType=="GGA")
    {
      for(unsigned int i = 0; i < singleAtomsDer2RRho.size(); ++i)
      {
        singleAtomsDer2RRho[i].reinit(singleAtomsRho[0]);
        singleAtomsDer2RRho[i]=0.0;
      }
    }
  }

	const std::shared_ptr< const dealii::Utilities::MPI::Partitioner > & partitioner
		=singleAtomsRho[0].get_partitioner();
	const unsigned int localSize =  partitioner->local_size();
	const unsigned int n_ghosts   = partitioner->n_ghost_indices();
	const unsigned int relevantDofs = localSize + n_ghosts;

	std::vector<bool> isAtomLocal(atomLocations.size()+numberImageCharges,false);
	for(unsigned int dof = 0; dof < relevantDofs; ++dof)
	{ 
		//const dealii::types::global_dof_index dofID = locallyOwnedDOFs[dof];
		const dealii::types::global_dof_index dofID = partitioner->local_to_global(dof);
	  Point<3> nodalCoor = supportPointsRhoNodal[dofID];
		if(!d_constraintsRhoNodalOnlyHanging.is_constrained(dofID))
		{
			//loop over atoms and superimpose electron-density at a given dof from all atoms
			double rhoNodalValue = 0.0;
			for(unsigned int iAtom = 0; iAtom < atomLocations.size(); ++iAtom)
			{
				Point<3> atom(atomLocations[iAtom][2],atomLocations[iAtom][3],atomLocations[iAtom][4]);
				double distanceToAtom = nodalCoor.distance(atom);

        Tensor<1,3,double> diff=nodalCoor-atom; 

        if (dftParameters::floatingNuclearCharges && distanceToAtom<1.0e-4)
        {
          if(dftParameters::verbosity>=4)
            std::cout<<"Atom close to nodal point, iatom: "<<iAtom<<std::endl;

          distanceToAtom=1.0e-4;
          diff[0]=(1.0e-4)/std::sqrt(3.0);
          diff[1]=(1.0e-4)/std::sqrt(3.0);
          diff[2]=(1.0e-4)/std::sqrt(3.0);              
        }

				double value,radialDensityFirstDerivative,radialDensitySecondDerivative;
				if(distanceToAtom <= outerMostPointDen[atomLocations[iAtom][0]])
				{
					alglib::spline1ddiff(denSpline[atomLocations[iAtom][0]],
                               distanceToAtom,
                               value,
                               radialDensityFirstDerivative,
                               radialDensitySecondDerivative);
					rhoNodalValue += value;
					isAtomLocal[iAtom]=true;

          if (dftParameters::isBOMD && dftParameters::isXLBOMD)
          {
					  singleAtomsRho[iAtom].local_element(dof)=std::abs(value);
            singleAtomsDerRRho[iAtom*3+0].local_element(dof)=-radialDensityFirstDerivative*(diff[0]/distanceToAtom);
            singleAtomsDerRRho[iAtom*3+1].local_element(dof)=-radialDensityFirstDerivative*(diff[1]/distanceToAtom);
            singleAtomsDerRRho[iAtom*3+2].local_element(dof)=-radialDensityFirstDerivative*(diff[2]/distanceToAtom);   

            if (dftParameters::xcFamilyType=="GGA")
            {
							for(unsigned int iDim = 0; iDim < 3; ++iDim)
								for(unsigned int jDim = 0; jDim < 3; ++jDim)
								{
									double temp = (radialDensitySecondDerivative -radialDensityFirstDerivative/distanceToAtom)*(diff[iDim]/distanceToAtom)*(diff[jDim]/distanceToAtom);
									if(iDim == jDim)
										temp += radialDensityFirstDerivative/distanceToAtom;

									singleAtomsDer2RRho[iAtom*9+iDim*3+jDim].local_element(dof) = -temp;
								}
            }
          }
				}
			}

			//loop over image charges and do as above
			for(int iImageCharge = 0; iImageCharge < numberImageCharges; ++iImageCharge)
			{
				Point<3> imageAtom(d_imagePositionsTrunc[iImageCharge][0],
						d_imagePositionsTrunc[iImageCharge][1],
						d_imagePositionsTrunc[iImageCharge][2]);
				double distanceToAtom = nodalCoor.distance(imageAtom);

        Tensor<1,3,double> diff=nodalCoor-imageAtom; 

        if (dftParameters::floatingNuclearCharges && distanceToAtom<1.0e-4)
        {
          if(dftParameters::verbosity>=4)
            std::cout<<"Atom close to nodal point, iatom: "<<iImageCharge<<std::endl;

          distanceToAtom=1.0e-4;
          diff[0]=(1.0e-4)/std::sqrt(3.0);
          diff[1]=(1.0e-4)/std::sqrt(3.0);
          diff[2]=(1.0e-4)/std::sqrt(3.0);              
        }

				int masterAtomId = d_imageIdsTrunc[iImageCharge];
        double value,radialDensityFirstDerivative,radialDensitySecondDerivative;
				if(distanceToAtom <= outerMostPointDen[atomLocations[masterAtomId][0]])
				{
					alglib::spline1ddiff(denSpline[atomLocations[masterAtomId][0]],
                               distanceToAtom,
                               value,
                               radialDensityFirstDerivative,
                               radialDensitySecondDerivative);

					rhoNodalValue += value;
					isAtomLocal[iImageCharge+atomLocations.size()]=true;

          if (dftParameters::isBOMD && dftParameters::isXLBOMD)
          {
					  singleAtomsRho[iImageCharge+atomLocations.size()].local_element(dof)=std::abs(value);
            singleAtomsDerRRho[(iImageCharge+atomLocations.size())*3+0].local_element(dof)=-radialDensityFirstDerivative*(diff[0]/distanceToAtom);
            singleAtomsDerRRho[(iImageCharge+atomLocations.size())*3+1].local_element(dof)=-radialDensityFirstDerivative*(diff[1]/distanceToAtom);
            singleAtomsDerRRho[(iImageCharge+atomLocations.size())*3+2].local_element(dof)=-radialDensityFirstDerivative*(diff[2]/distanceToAtom);              
            if (dftParameters::xcFamilyType=="GGA")
            {
							for(unsigned int iDim = 0; iDim < 3; ++iDim)
								for(unsigned int jDim = 0; jDim < 3; ++jDim)
								{
									double temp = (radialDensitySecondDerivative -radialDensityFirstDerivative/distanceToAtom)*(diff[iDim]/distanceToAtom)*(diff[jDim]/distanceToAtom);
									if(iDim == jDim)
										temp += radialDensityFirstDerivative/distanceToAtom;

									singleAtomsDer2RRho[(iImageCharge+atomLocations.size())*9+iDim*3+jDim].local_element(dof) = -temp;
								}

            }
          }
          
				}
			}

			if (dof<localSize && !d_constraintsRhoNodal.is_constrained(dofID))
      {
				d_atomicRho.local_element(dof) = std::abs(rhoNodalValue);
      }
		}
	}

	d_atomicRho.update_ghost_values();

	interpolateRhoNodalDataToQuadratureDataGeneral(d_matrixFreeDataPRefined,
      d_densityDofHandlerIndexElectro,
      d_densityQuadratureIdElectro,
			d_atomicRho,
			d_rhoAtomsValues,
			d_gradRhoAtomsValues,
			d_gradRhoAtomsValues,
			dftParameters::xcFamilyType=="GGA",
      false);

  //normalize rho
  const double charge = totalCharge(d_matrixFreeDataPRefined,
      d_atomicRho);

  if (!reusePreviousScalingFactor)
  {
    const double scalingFactor = ((double)numElectrons)/charge;
    d_atomicRhoScalingFac=scalingFactor;

    //scale nodal vector with scalingFactor
    d_atomicRho *= scalingFactor;

    //for(unsigned int iAtom = 0; iAtom < atomLocations.size()+numberImageCharges; ++iAtom)
    //   singleAtomsRho[iAtom]*=scalingFactor; 

    if (dftParameters::verbosity>=3)
    {
      pcout<<"Total Charge before Normalizing nodal Rho:  "<<charge<<std::endl;
      pcout<<"Total Charge after Normalizing nodal Rho: "<< totalCharge(d_matrixFreeDataPRefined,d_atomicRho)<<std::endl;
    }
  }
  else
  {
    d_atomicRho*=d_atomicRhoScalingFac;
  }

  if (dftParameters::isBOMD && dftParameters::isXLBOMD)
  {
    /* 
	  for(unsigned int i = 0; i < singleAtomsRho.size(); ++i)
	   singleAtomsRho[i].update_ghost_values();

	  for(unsigned int i = 0; i < singleAtomsDerRRho.size(); ++i)
	   singleAtomsDerRRho[i].update_ghost_values(); 

    if (dftParameters::xcFamilyType=="GGA")
    {
      for(unsigned int i = 0; i < singleAtomsDer2RRho.size(); ++i)
       singleAtomsDer2RRho[i].update_ghost_values();   
    }
    */

    FEEvaluation<C_DIM,C_rhoNodalPolyOrder<FEOrder,FEOrderElectro>(),C_num1DQuad<C_rhoNodalPolyOrder<FEOrder,FEOrderElectro>()>(),1,double> feEvalObj(d_matrixFreeDataPRefined,d_nonPeriodicDensityDofHandlerIndexElectro,d_densityQuadratureIdElectro);
    const unsigned int numQuadPoints = feEvalObj.n_q_points;

    DoFHandler<C_DIM>::active_cell_iterator subCellPtr;
    for(unsigned int cell = 0; cell < d_matrixFreeDataPRefined.n_macro_cells(); ++cell)
    {
      feEvalObj.reinit(cell);
      for (unsigned int iatom=0; iatom<(atomLocations.size()+numberImageCharges); iatom++)
      {
        //if (!isAtomLocal[iatom])
        //  continue;

        feEvalObj.read_dof_values(singleAtomsRho[iatom]);
        feEvalObj.evaluate(true,false,false);

        bool isRhoNonZero=true;

        for(unsigned int q_point = 0; q_point < numQuadPoints; ++q_point)
        {
          VectorizedArray<double> val=feEvalObj.get_value(q_point);
          for(unsigned int iSubCell = 0; iSubCell < d_matrixFreeDataPRefined.n_components_filled(cell); ++iSubCell)
          {
            if (val[iSubCell]>truncationTol)
              isRhoNonZero=false; 
          }
        }


        if (!isRhoNonZero)
        {
          for(unsigned int iSubCell = 0; iSubCell < d_matrixFreeDataPRefined.n_components_filled(cell); ++iSubCell)
          {
            subCellPtr= d_matrixFreeDataPRefined.get_cell_iterator(cell,iSubCell,d_nonPeriodicDensityDofHandlerIndexElectro);
            dealii::CellId subCellId=subCellPtr->id();
            d_derRRhoAtomsValuesSeparate[iatom][subCellId]=std::vector<double>(3*numQuadPoints);

            if (dftParameters::xcFamilyType=="GGA")
            {
              d_der2XRRhoAtomsValuesSeparate[iatom][subCellId]=std::vector<double>(9*numQuadPoints);            
            }
          }

          for (unsigned int idim=0; idim<3;++idim)
          {
            feEvalObj.read_dof_values(singleAtomsDerRRho[3*iatom+idim]);
            feEvalObj.evaluate(true,false,false); 

            for(unsigned int iSubCell = 0; iSubCell < d_matrixFreeDataPRefined.n_components_filled(cell); ++iSubCell)
            {
              subCellPtr= d_matrixFreeDataPRefined.get_cell_iterator(cell,iSubCell,d_nonPeriodicDensityDofHandlerIndexElectro);
              dealii::CellId subCellId=subCellPtr->id();

              std::vector<double> & tempVec1 = d_derRRhoAtomsValuesSeparate[iatom].find(subCellId)->second;
              for(unsigned int q_point = 0; q_point < numQuadPoints; ++q_point)
                tempVec1[3*q_point+idim] = feEvalObj.get_value(q_point)[iSubCell];
            }
          }

          if (dftParameters::xcFamilyType=="GGA")
          {
            for (unsigned int idim=0; idim<3;++idim)
              for (unsigned int jdim=0; jdim<3;++jdim)
              {
                feEvalObj.read_dof_values(singleAtomsDer2RRho[9*iatom+3*idim+jdim]);
                feEvalObj.evaluate(true,false,false); 

                for(unsigned int iSubCell = 0; iSubCell < d_matrixFreeDataPRefined.n_components_filled(cell); ++iSubCell)
                {
                  subCellPtr= d_matrixFreeDataPRefined.get_cell_iterator(cell,iSubCell,d_nonPeriodicDensityDofHandlerIndexElectro);
                  dealii::CellId subCellId=subCellPtr->id();

                  std::vector<double> & tempVec1 = d_der2XRRhoAtomsValuesSeparate[iatom].find(subCellId)->second;
                  for(unsigned int q_point = 0; q_point < numQuadPoints; ++q_point)
                    tempVec1[9*q_point+3*idim+jdim] = feEvalObj.get_value(q_point)[iSubCell];
                }
              }
          }//GGA
        }//non-zero rho
      }//iatom loop

    }//cell loop

  }

  normalizeAtomicRhoQuadValues(reusePreviousScalingFactor);
  computing_timer.exit_section("initialize atomic density for density splitting approach in md and relaxations");  
}


//
//Normalize rho
//
	template<unsigned int FEOrder,unsigned int FEOrderElectro>
void dftClass<FEOrder,FEOrderElectro>::normalizeAtomicRhoQuadValues(const bool reusePreviousScalingFactor)
{

  const double charge = totalCharge(dofHandler,
			&d_rhoAtomsValues);
	const double scaling=reusePreviousScalingFactor?d_atomicRhoScalingFac:(((double)numElectrons)/charge);

	if (dftParameters::verbosity>=2)
		pcout<< "Total charge rho single atomic before normalizing to number of electrons: "<< charge<<std::endl;


	for (auto it1=d_rhoAtomsValues.begin(); it1!=d_rhoAtomsValues.end(); ++it1)
		for (unsigned int i=0; i<(it1->second).size(); ++i)
			(it1->second)[i]*=scaling;

  if (dftParameters::isBOMD && dftParameters::isXLBOMD)
  {
      for (auto it1=d_derRRhoAtomsValuesSeparate.begin(); it1!=d_derRRhoAtomsValuesSeparate.end(); ++it1)
        for (auto it2=it1->second.begin(); it2!=it1->second.end(); ++it2)
          for (unsigned int i=0; i<(it2->second).size(); ++i)
            (it2->second)[i]*=scaling;
  }


	if (dftParameters::xcFamilyType=="GGA")
	{
    for (auto it1=d_gradRhoAtomsValues.begin(); it1!=d_gradRhoAtomsValues.end(); ++it1)
      for (unsigned int i=0; i<(it1->second).size(); ++i)
        (it1->second)[i]*=scaling;

    if (dftParameters::isBOMD && dftParameters::isXLBOMD)
    {
      for (auto it1=d_der2XRRhoAtomsValuesSeparate.begin(); it1!=d_der2XRRhoAtomsValuesSeparate.end(); ++it1)
        for (auto it2=it1->second.begin(); it2!=it1->second.end(); ++it2)
          for (unsigned int i=0; i<(it2->second).size(); ++i)
            (it2->second)[i]*=scaling;
    }
	}

	double chargeAfterScaling = totalCharge(dofHandler,
			&d_rhoAtomsValues);

	if (dftParameters::verbosity>=2)
		pcout<<"Total charge rho single atomic after normalizing: "<< chargeAfterScaling<<std::endl;
}

//
//
//
	template<unsigned int FEOrder,unsigned int FEOrderElectro>
void  dftClass<FEOrder,FEOrderElectro>::addAtomicRhoQuadValuesGradients(std::map<dealii::CellId, std::vector<double> > & quadratureValueData,
    std::map<dealii::CellId, std::vector<double> > & quadratureGradValueData,
    const bool isConsiderGradData)
{
	const Quadrature<3> &  quadrature_formula=matrix_free_data.get_quadrature(d_densityQuadratureId);
	const unsigned int n_q_points    = quadrature_formula.size();

	DoFHandler<3>::active_cell_iterator
		cell = dofHandler.begin_active(),
		     endc = dofHandler.end();
	for (; cell!=endc; ++cell) 
		if (cell->is_locally_owned())
    {
			std::vector<double> & rhoValues=quadratureValueData.find(cell->id())->second;
      const std::vector<double> & rhoAtomicValues=d_rhoAtomsValues.find(cell->id())->second;
			for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
				rhoValues[q_point]+=rhoAtomicValues[q_point];

      if (isConsiderGradData)
      {
        std::vector<double> & gradRhoValues=quadratureGradValueData.find(cell->id())->second;  
        const std::vector<double> & gradRhoAtomicValues=d_gradRhoAtomsValues.find(cell->id())->second;        
        for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
        {
          gradRhoValues[3*q_point+0]+=gradRhoAtomicValues[3*q_point+0];    
          gradRhoValues[3*q_point+1]+=gradRhoAtomicValues[3*q_point+1];  
          gradRhoValues[3*q_point+2]+=gradRhoAtomicValues[3*q_point+2];            
        }
      }
		}  
}

//
//
	template<unsigned int FEOrder,unsigned int FEOrderElectro>
void  dftClass<FEOrder,FEOrderElectro>::subtractAtomicRhoQuadValuesGradients(std::map<dealii::CellId, std::vector<double> > & quadratureValueData,
    std::map<dealii::CellId, std::vector<double> > & quadratureGradValueData,
    const bool isConsiderGradData)
{
	const Quadrature<3> &  quadrature_formula=matrix_free_data.get_quadrature(d_densityQuadratureId);
	const unsigned int n_q_points    = quadrature_formula.size();

	DoFHandler<3>::active_cell_iterator
		cell = dofHandler.begin_active(),
		     endc = dofHandler.end();
	for (; cell!=endc; ++cell) 
		if (cell->is_locally_owned())
    {
			std::vector<double> & rhoValues=quadratureValueData.find(cell->id())->second;
      const std::vector<double> & rhoAtomicValues=d_rhoAtomsValues.find(cell->id())->second;
			for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
				rhoValues[q_point]-=rhoAtomicValues[q_point];

      if (isConsiderGradData)
      {
        std::vector<double> & gradRhoValues=quadratureGradValueData.find(cell->id())->second;  
        const std::vector<double> & gradRhoAtomicValues=d_gradRhoAtomsValues.find(cell->id())->second;        
        for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
        {
          gradRhoValues[3*q_point+0]-=gradRhoAtomicValues[3*q_point+0];    
          gradRhoValues[3*q_point+1]-=gradRhoAtomicValues[3*q_point+1];  
          gradRhoValues[3*q_point+2]-=gradRhoAtomicValues[3*q_point+2];            
        }
      }
		}  
}

//
//compute l2 projection of quad data to nodal data
//
	template <unsigned int FEOrder,unsigned int FEOrderElectro>
void dftClass<FEOrder,FEOrderElectro>::l2ProjectionQuadDensityMinusAtomicDensity(const dealii::MatrixFree<3,double> & matrixFreeDataObject,
    const dealii::AffineConstraints<double> & constraintMatrix,
		const unsigned int dofHandlerId,
		const unsigned int quadratureId,
		const std::map<dealii::CellId, std::vector<double> > & quadratureValueData,
		distributedCPUVec<double> & nodalField)
{
    std::function<double(const typename dealii::DoFHandler<3>::active_cell_iterator & cell ,
        const unsigned int q)> funcRho =
      [&](const typename dealii::DoFHandler<3>::active_cell_iterator & cell ,
          const unsigned int q)
      {return (quadratureValueData.find(cell->id())->second[q]-d_rhoAtomsValues.find(cell->id())->second[q]);};
    dealii::VectorTools::project<3,distributedCPUVec<double>> (dealii::MappingQ1<3,3>(),
        matrixFreeDataObject.get_dof_handler(dofHandlerId),
        constraintMatrix,
        matrixFreeDataObject.get_quadrature(quadratureId),
        funcRho,
        nodalField);
}
