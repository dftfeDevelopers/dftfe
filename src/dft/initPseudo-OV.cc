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
// @author Krishnendu Ghosh, Sambit Das
//

#include "stdafx.h"
#include <linalg.h>
#include <dftParameters.h>


	template<unsigned int FEOrder,unsigned int FEOrderElectro>
void dftClass<FEOrder,FEOrderElectro>::computeElementalOVProjectorKets()
{

	//
	//get the number of non-local atoms
	//
	int numberNonLocalAtoms = d_nonLocalAtomGlobalChargeIds.size();

	//
	//get number of global charges
	//
	unsigned int numberGlobalCharges  = atomLocations.size();


	//
	//get FE data structures
	//
  QIterated<3> quadrature(QGauss<1>(C_num1DQuad<C_rhoNodalPolyOrder<FEOrder,FEOrderElectro>()>()),1);
  QIterated<3> quadratureHigh(QGauss<1>(C_num1DQuadNLPSP<FEOrder>()),C_numCopies1DQuadNLPSP());

	//FEValues<3> fe_values(FE, quadrature, update_values | update_gradients | update_JxW_values);
	FEValues<3> fe_values(FE, quadratureHigh,
			update_values | update_JxW_values| update_quadrature_points);
	const unsigned int numberNodesPerElement  = FE.dofs_per_cell;
	const unsigned int numberQuadraturePoints = quadratureHigh.size();

  std::map<DoFHandler<3>::active_cell_iterator,std::vector<double>> cellIteratorQuadPointsMap;
  std::vector<double> shapeValQuads(numberNodesPerElement*numberQuadraturePoints);
  std::map<DoFHandler<3>::active_cell_iterator,std::vector<double>> cellIteratorJxWQuadsMap;

  DoFHandler<3>::active_cell_iterator
    cell = dofHandler.begin_active(),
         endc = dofHandler.end();
  int iElem=0;
  for (; cell!=endc; ++cell) 
    if (cell->is_locally_owned())
    {
      fe_values.reinit (cell);
      std::vector<double> & temp1=cellIteratorQuadPointsMap[cell];
      std::vector<double> & temp2=cellIteratorJxWQuadsMap[cell];        
      temp1.resize(numberQuadraturePoints*3);
      temp2.resize(numberQuadraturePoints);
      for (unsigned int q_point=0; q_point<numberQuadraturePoints; ++q_point)
      {
        temp1[3*q_point+0]=fe_values.quadrature_point(q_point)[0];
        temp1[3*q_point+1]=fe_values.quadrature_point(q_point)[1];
        temp1[3*q_point+2]=fe_values.quadrature_point(q_point)[2];
        temp2[q_point]=fe_values.JxW(q_point);
      }

      if (iElem==0)
      {
        for (unsigned int q_point=0; q_point<numberQuadraturePoints; ++q_point)        
          for (unsigned int inode=0; inode<numberNodesPerElement; ++inode)           
            shapeValQuads[q_point*numberNodesPerElement+inode]=fe_values.shape_value(inode,q_point);
      }
      iElem++;
    }

	//
	//get number of kPoints
	//
	const unsigned int maxkPoints = d_kPointWeights.size();

	//
	//reinit kohnShamDFTOperator for getting access to global to local element nodeIds
	//
	kohnShamDFTOperatorClass<FEOrder,FEOrderElectro> kohnShamDFTEigenOperator(this,mpi_communicator);
	distributedCPUVec<double> sqrtMassVector,invSqrtMassVector;

	if(dftParameters::cellLevelMassMatrixScaling)
	  {
	    matrix_free_data.initialize_dof_vector(invSqrtMassVector,0);
	    sqrtMassVector.reinit(invSqrtMassVector);
	    kohnShamDFTEigenOperator.computeMassVector(dofHandler,
						       constraintsNone,
						       sqrtMassVector,
						       invSqrtMassVector);

	    constraintsNone.distribute(invSqrtMassVector);
	    invSqrtMassVector.update_ghost_values();
	  }
	
	distributedCPUVec<dataTypes::number> tmpVector;
	vectorTools::createDealiiVector<dataTypes::number>(matrix_free_data.get_vector_partitioner(),
							   1,
							   tmpVector);

	//storage for precomputing index maps
	std::vector<std::vector<dealii::types::global_dof_index> > flattenedArrayMacroCellLocalProcIndexIdMap, flattenedArrayCellLocalProcIndexIdMap;
	
	vectorTools::computeCellLocalIndexSetMap(tmpVector.get_partitioner(),
						 matrix_free_data,
                                                 d_densityDofHandlerIndex,
						 1,
						 flattenedArrayMacroCellLocalProcIndexIdMap,
						 flattenedArrayCellLocalProcIndexIdMap);


	//
	//preallocate element Matrices
	//
	d_nonLocalProjectorElementMatrices.clear();
	d_nonLocalProjectorElementMatricesConjugate.clear();
	d_nonLocalProjectorElementMatricesTranspose.clear();
	d_nonLocalProjectorElementMatricesCellMassMatrixScaled.clear();
        d_nonLocalProjectorElementMatricesConjugateCellMassMatrixScaled.clear();  
	d_nonLocalProjectorElementMatricesTransposeCellMassMatrixScaled.clear();
	d_nonLocalPSP_ZetalmDeltaVl.clear();
	d_nonLocalPSP_zetalmDeltaVlProductDistImageAtoms_KPoint.clear();
	d_cellIdToNonlocalAtomIdsLocalCompactSupportMap.clear();

	d_nonLocalProjectorElementMatrices.resize(numberNonLocalAtoms);
	d_nonLocalProjectorElementMatricesConjugate.resize(numberNonLocalAtoms);
	d_nonLocalProjectorElementMatricesTranspose.resize(numberNonLocalAtoms);


	if(dftParameters::cellLevelMassMatrixScaling)
	  {
	    d_nonLocalProjectorElementMatricesTransposeCellMassMatrixScaled.resize(numberNonLocalAtoms);
	    d_nonLocalProjectorElementMatricesCellMassMatrixScaled.resize(numberNonLocalAtoms);
	  }

	std::vector<double> nonLocalProjectorBasisReal(maxkPoints*numberQuadraturePoints,0.0);
	std::vector<double> nonLocalProjectorBasisImag(maxkPoints*numberQuadraturePoints,0.0);

#ifdef USE_COMPLEX
  std::vector<double>  ZetalmDeltaVl_KPoint(maxkPoints*numberQuadraturePoints*2,0.0);
  std::vector<double> zetalmDeltaVlProductDistImageAtoms_KPoint(maxkPoints*numberQuadraturePoints*C_DIM*2,0.0);
#else
  std::vector<double> ZetalmDeltaVl(numberQuadraturePoints,0.0);
  std::vector<double> zetalmDeltaVlProductDistImageAtoms_KPoint(maxkPoints*numberQuadraturePoints*C_DIM,0.0); 
	AssertThrow(maxkPoints==1,ExcMessage("DFT-FE Error"));  
#endif

	int cumulativeWaveSplineId = 0;
	int waveFunctionId;
  unsigned int count=0;
	const unsigned int numNonLocalAtomsCurrentProcess= d_nonLocalAtomIdsInCurrentProcess.size();
	d_nonLocalPSP_ZetalmDeltaVl.resize(numNonLocalAtomsCurrentProcess);
  d_nonLocalPSP_zetalmDeltaVlProductDistImageAtoms_KPoint.resize(numNonLocalAtomsCurrentProcess);
	//
	//
	for(int iAtom = 0; iAtom < numberNonLocalAtoms; ++iAtom)
	{
		//
		//get the global charge Id of the current nonlocal atom
		//
		const int globalChargeIdNonLocalAtom =  d_nonLocalAtomGlobalChargeIds[iAtom];


		Point<3> nuclearCoordinates(atomLocations[globalChargeIdNonLocalAtom][2],atomLocations[globalChargeIdNonLocalAtom][3],atomLocations[globalChargeIdNonLocalAtom][4]);

		std::vector<int> & imageIdsList = d_globalChargeIdToImageIdMapTrunc[globalChargeIdNonLocalAtom];

		//
		//get the number of elements in the compact support of the current nonlocal atom
		//
		int numberElementsInAtomCompactSupport = d_elementOneFieldIteratorsInAtomCompactSupport[iAtom].size();


		//pcout<<"Number of elements in compact support of nonlocal atom "<<iAtom<<" is "<<numberElementsInAtomCompactSupport<<std::endl;
		//pcout<<"Image Ids List: "<<imageIdsList.size()<<std::endl;
		//pcout<<numberElementsInAtomCompactSupport<<std::endl;

		//
		//get the number of pseudowavefunctions for the current nonlocal atoms
		//
		const unsigned int numberPseudoWaveFunctions = d_numberPseudoAtomicWaveFunctions[iAtom];


		//
		//allocate element Matrices
		//
		if (numberElementsInAtomCompactSupport>0)
		{
			d_nonLocalProjectorElementMatrices[iAtom].resize(numberElementsInAtomCompactSupport);
			d_nonLocalProjectorElementMatricesConjugate[iAtom].resize(numberElementsInAtomCompactSupport);
			d_nonLocalProjectorElementMatricesTranspose[iAtom].resize(numberElementsInAtomCompactSupport);

			d_nonLocalPSP_ZetalmDeltaVl[count].resize(numberPseudoWaveFunctions);
			d_nonLocalPSP_zetalmDeltaVlProductDistImageAtoms_KPoint[count].resize(numberPseudoWaveFunctions);

			if(dftParameters::cellLevelMassMatrixScaling)
			  {
			    d_nonLocalProjectorElementMatricesConjugateCellMassMatrixScaled[iAtom].resize(numberElementsInAtomCompactSupport);
			    d_nonLocalProjectorElementMatricesCellMassMatrixScaled[iAtom].resize(numberElementsInAtomCompactSupport);
			    d_nonLocalProjectorElementMatricesTransposeCellMassMatrixScaled[iAtom].resize(numberElementsInAtomCompactSupport);
			  }
			
		}

		for(int iElemComp = 0; iElemComp < numberElementsInAtomCompactSupport; ++iElemComp)
		{

			cell = d_elementOneFieldIteratorsInAtomCompactSupport[iAtom][iElemComp];

			d_cellIdToNonlocalAtomIdsLocalCompactSupportMap[cell->id()].insert(count);      

			const std::vector<double> & quadPoints=cellIteratorQuadPointsMap[cell];  
			const std::vector<double> & jxwQuads=cellIteratorJxWQuadsMap[cell];    

			//compute values for the current elements
			//fe_values.reinit(cell);

#ifdef USE_COMPLEX
			d_nonLocalProjectorElementMatrices[iAtom][iElemComp].resize(maxkPoints,
					std::vector<std::complex<double> > (numberNodesPerElement*numberPseudoWaveFunctions,0.0));
			d_nonLocalProjectorElementMatricesConjugate[iAtom][iElemComp].resize(maxkPoints,
					std::vector<std::complex<double> > (numberNodesPerElement*numberPseudoWaveFunctions,0.0));
			d_nonLocalProjectorElementMatricesTranspose[iAtom][iElemComp].resize(maxkPoints,
					std::vector<std::complex<double> > (numberNodesPerElement*numberPseudoWaveFunctions,0.0));

			if(dftParameters::cellLevelMassMatrixScaling)
			  {
			    d_nonLocalProjectorElementMatricesConjugateCellMassMatrixScaled[iAtom][iElemComp].resize(maxkPoints,
					std::vector<std::complex<double> > (numberNodesPerElement*numberPseudoWaveFunctions,0.0));

			    d_nonLocalProjectorElementMatricesTransposeCellMassMatrixScaled[iAtom][iElemComp].resize(maxkPoints,
					std::vector<std::complex<double> > (numberNodesPerElement*numberPseudoWaveFunctions,0.0));
			  }

			std::vector<std::vector<std::complex<double> > > & nonLocalProjectorElementMatricesAtomElem=d_nonLocalProjectorElementMatrices[iAtom][iElemComp];

			std::vector<std::vector<std::complex<double> > > & nonLocalProjectorElementMatricesConjugateAtomElem=d_nonLocalProjectorElementMatricesConjugate[iAtom][iElemComp];

			std::vector<std::vector<std::complex<double> > > & nonLocalProjectorElementMatricesTransposeAtomElem=d_nonLocalProjectorElementMatricesTranspose[iAtom][iElemComp];

#else
			d_nonLocalProjectorElementMatrices[iAtom][iElemComp].resize(numberNodesPerElement*numberPseudoWaveFunctions,0.0);
			d_nonLocalProjectorElementMatricesTranspose[iAtom][iElemComp].resize(numberNodesPerElement*numberPseudoWaveFunctions,0.0);

			if(dftParameters::cellLevelMassMatrixScaling)
			  {
			    d_nonLocalProjectorElementMatricesCellMassMatrixScaled[iAtom][iElemComp].resize(numberNodesPerElement*numberPseudoWaveFunctions,0.0);
			    d_nonLocalProjectorElementMatricesTransposeCellMassMatrixScaled[iAtom][iElemComp].resize(numberNodesPerElement*numberPseudoWaveFunctions,0.0);

			  }

			std::vector<double> & nonLocalProjectorElementMatricesAtomElem
				=d_nonLocalProjectorElementMatrices[iAtom][iElemComp];


			std::vector<double> & nonLocalProjectorElementMatricesTransposeAtomElem
				=d_nonLocalProjectorElementMatricesTranspose[iAtom][iElemComp];


#endif

			int iPsp = -1;
			int lTemp = 1e5;

#ifdef USE_COMPLEX
      std::vector<double> nonLocalProjectorBasisRealTimesJxW(maxkPoints*numberPseudoWaveFunctions*numberQuadraturePoints,0.0);
      std::vector<double> nonLocalProjectorBasisImagTimesJxW(maxkPoints*numberPseudoWaveFunctions*numberQuadraturePoints,0.0);
#else
      std::vector<double> ZetalmDeltaVlTimesJxW(numberPseudoWaveFunctions*numberQuadraturePoints,0.0);
#endif

			for(int iPseudoWave = 0; iPseudoWave < numberPseudoWaveFunctions; ++iPseudoWave)
			{
#ifdef USE_COMPLEX
				d_nonLocalPSP_ZetalmDeltaVl[count][iPseudoWave][cell->id()]=std::vector<double>(maxkPoints*numberQuadraturePoints*2);
				d_nonLocalPSP_zetalmDeltaVlProductDistImageAtoms_KPoint[count][iPseudoWave][cell->id()]=std::vector<double>(maxkPoints*numberQuadraturePoints*C_DIM*2);        
#else
				d_nonLocalPSP_ZetalmDeltaVl[count][iPseudoWave][cell->id()]=std::vector<double>(numberQuadraturePoints);
				d_nonLocalPSP_zetalmDeltaVlProductDistImageAtoms_KPoint[count][iPseudoWave][cell->id()]=std::vector<double>(maxkPoints*numberQuadraturePoints*C_DIM);             
#endif

				waveFunctionId = iPseudoWave + cumulativeWaveSplineId;
				const int globalWaveSplineId = d_pseudoWaveFunctionIdToFunctionIdDetails[waveFunctionId][0];
				const int lQuantumNumber = d_pseudoWaveFunctionIdToFunctionIdDetails[waveFunctionId][1];
				const int mQuantumNumber = d_pseudoWaveFunctionIdToFunctionIdDetails[waveFunctionId][2];

				//
				//access pseudoPotential Ids
				//
				if(lQuantumNumber != lTemp)
					iPsp += 1;
				lTemp = lQuantumNumber;


				std::fill(nonLocalProjectorBasisReal.begin(),nonLocalProjectorBasisReal.end(),0.0);
				std::fill(nonLocalProjectorBasisImag.begin(),nonLocalProjectorBasisImag.end(),0.0);

#ifdef USE_COMPLEX
				std::fill(ZetalmDeltaVl_KPoint.begin(),ZetalmDeltaVl_KPoint.end(),0.0);
#else
				std::fill(ZetalmDeltaVl.begin(),ZetalmDeltaVl.end(),0.0);
#endif
        std::fill(zetalmDeltaVlProductDistImageAtoms_KPoint.begin(),zetalmDeltaVlProductDistImageAtoms_KPoint.end(),0.0);        

				double nlpValue = 0.0;
      
        for(int iImageAtomCount = 0; iImageAtomCount < imageIdsList.size(); ++iImageAtomCount)
        {

          int chargeId = imageIdsList[iImageAtomCount];

          //const Point & chargePoint = chargeId < numberGlobalCharges? d_nuclearContainer.getGlobalPoint(chargeId,meshId):
          //d_nuclearContainer.getImagePoint(chargeId-numberGlobalCharges,meshId);

          Point<3> chargePoint(0.0,0.0,0.0);

          if(chargeId < numberGlobalCharges)
          {
            chargePoint[0] = atomLocations[chargeId][2];
            chargePoint[1] = atomLocations[chargeId][3];
            chargePoint[2] = atomLocations[chargeId][4];
          }
          else
          {
            chargePoint[0] = d_imagePositionsTrunc[chargeId-numberGlobalCharges][0];
            chargePoint[1] = d_imagePositionsTrunc[chargeId-numberGlobalCharges][1];
            chargePoint[2] = d_imagePositionsTrunc[chargeId-numberGlobalCharges][2];
          }

					if (chargePoint.distance(cell->center())>d_nlPSPCutOff)
            continue;

					double x[3],pointMinusLatticeVector[3];
					double sphericalHarmonicVal, radialProjVal, projectorFunctionValue;     
          double r,theta,phi,angle;

          for(int iQuadPoint = 0; iQuadPoint < numberQuadraturePoints; ++iQuadPoint)
          {
						x[0] = quadPoints[3*iQuadPoint] - chargePoint[0];
						x[1] = quadPoints[3*iQuadPoint+1] - chargePoint[1];
						x[2] = quadPoints[3*iQuadPoint+2] - chargePoint[2];


						if((x[0]*x[0]+x[1]*x[1]+x[2]*x[2]) <= d_outerMostPointPseudoProjectorData[globalWaveSplineId]*d_outerMostPointPseudoProjectorData[globalWaveSplineId])
            {
              pseudoUtils::convertCartesianToSpherical(x,r,theta,phi);


              pseudoUtils::getRadialFunctionVal(r,
                    radialProjVal,
                    &d_pseudoWaveFunctionSplines[globalWaveSplineId]);

              pseudoUtils::getSphericalHarmonicVal(theta,phi,lQuantumNumber,mQuantumNumber,sphericalHarmonicVal);

              projectorFunctionValue = radialProjVal*sphericalHarmonicVal;

              /*if(iElemComp == 0 && iQuadPoint == 0 && iPseudoWave == 0)
                {
                std::cout<<"ChargeId : "<<chargeId<<std::endl;
                std::cout<<"Coordinates: "<<chargePoint[0]<<" "<<chargePoint[1]<<" "<<chargePoint[2]<<std::endl;
                std::cout<<"Distance : "<<r<<std::endl;
                std::cout<<"DeltaVl: "<<deltaVlValue<<std::endl;
                std::cout<<"JacTimesWeight: "<<fe_values.JxW(iQuadPoint)<<std::endl;
                }*/

              //
              //kpoint loop
              //
#ifdef USE_COMPLEX
              pointMinusLatticeVector[0] = x[0] + nuclearCoordinates[0];
              pointMinusLatticeVector[1] = x[1] + nuclearCoordinates[1];
              pointMinusLatticeVector[2] = x[2] + nuclearCoordinates[2];
              for(int kPoint = 0; kPoint < maxkPoints; ++kPoint)
              {
                angle = d_kPointCoordinates[3*kPoint+0]*pointMinusLatticeVector[0] + d_kPointCoordinates[3*kPoint+1]*pointMinusLatticeVector[1] + d_kPointCoordinates[3*kPoint+2]*pointMinusLatticeVector[2];
                nonLocalProjectorBasisReal[kPoint*numberQuadraturePoints+iQuadPoint] += cos(angle)*projectorFunctionValue;                
                nonLocalProjectorBasisImag[kPoint*numberQuadraturePoints+iQuadPoint] += -sin(angle)*projectorFunctionValue;
								
                const double tempReal=std::cos(-angle);
								const double tempImag=std::sin(-angle);
								ZetalmDeltaVl_KPoint[kPoint*numberQuadraturePoints*2+2*iQuadPoint+0] += tempReal*projectorFunctionValue;
								ZetalmDeltaVl_KPoint[kPoint*numberQuadraturePoints*2+2*iQuadPoint+1] += tempImag*projectorFunctionValue;
								for(unsigned int iDim = 0; iDim < C_DIM; ++iDim)
								{
                  zetalmDeltaVlProductDistImageAtoms_KPoint[kPoint*numberQuadraturePoints*C_DIM*2+iQuadPoint*C_DIM*2+iDim*2+0]+=tempReal*projectorFunctionValue*x[iDim];
                  zetalmDeltaVlProductDistImageAtoms_KPoint[kPoint*numberQuadraturePoints*C_DIM*2+iQuadPoint*C_DIM*2+iDim*2+1]+=tempImag*projectorFunctionValue*x[iDim];
								}
							}
#else

						  ZetalmDeltaVl[iQuadPoint] += projectorFunctionValue;
              for(unsigned int iDim = 0; iDim < C_DIM; ++iDim)
                zetalmDeltaVlProductDistImageAtoms_KPoint[iQuadPoint*C_DIM+iDim]+=projectorFunctionValue*x[iDim];
#endif              
            }//inside psp tail

					}//quad loop

				}//image atom loop

#ifdef USE_COMPLEX
				d_nonLocalPSP_ZetalmDeltaVl[count][iPseudoWave][cell->id()]=ZetalmDeltaVl_KPoint;
#else
				d_nonLocalPSP_ZetalmDeltaVl[count][iPseudoWave][cell->id()]=ZetalmDeltaVl;
#endif
        d_nonLocalPSP_zetalmDeltaVlProductDistImageAtoms_KPoint[count][iPseudoWave][cell->id()]=zetalmDeltaVlProductDistImageAtoms_KPoint;

#ifdef USE_COMPLEX
				for(int kPoint = 0; kPoint < maxkPoints; ++kPoint)   
          for(int iQuadPoint = 0; iQuadPoint < numberQuadraturePoints; ++iQuadPoint)
          {
             nonLocalProjectorBasisRealTimesJxW[kPoint*numberPseudoWaveFunctions*numberQuadraturePoints+iPseudoWave*numberQuadraturePoints+iQuadPoint]=nonLocalProjectorBasisReal[kPoint*numberQuadraturePoints+iQuadPoint]*jxwQuads[iQuadPoint];
             nonLocalProjectorBasisImagTimesJxW[kPoint*numberPseudoWaveFunctions*numberQuadraturePoints+iPseudoWave*numberQuadraturePoints+iQuadPoint]=nonLocalProjectorBasisImag[kPoint*numberQuadraturePoints+iQuadPoint]*jxwQuads[iQuadPoint];
          }
#else
        for(int iQuadPoint = 0; iQuadPoint < numberQuadraturePoints; ++iQuadPoint)
           ZetalmDeltaVlTimesJxW[iPseudoWave*numberQuadraturePoints+iQuadPoint]=ZetalmDeltaVl[iQuadPoint]*jxwQuads[iQuadPoint];
#endif
      }//pseudowave loop


      const char transA = 'N',transB = 'N';
      const double scalarCoeffAlpha = 1.0,scalarCoeffBeta = 0.0;
      const unsigned int inc = 1;
      std::vector<double> projectorMatrixReal(numberNodesPerElement*numberPseudoWaveFunctions,0.0);
      std::vector<double> projectorMatrixImag(numberNodesPerElement*numberPseudoWaveFunctions,0.0);

      for(int kPoint = 0; kPoint < maxkPoints; ++kPoint)
      {      
#ifdef USE_COMPLEX
        dgemm_(&transA,
            &transB,
            &numberNodesPerElement,
            &numberPseudoWaveFunctions,
            &numberQuadraturePoints,
            &scalarCoeffAlpha,
            &shapeValQuads[0],
            &numberNodesPerElement,
            &nonLocalProjectorBasisRealTimesJxW[kPoint*numberPseudoWaveFunctions*numberQuadraturePoints],
            &numberQuadraturePoints,
            &scalarCoeffBeta,
            &projectorMatrixReal[0],
            &numberNodesPerElement);       

        dgemm_(&transA,
            &transB,
            &numberNodesPerElement,
            &numberPseudoWaveFunctions,
            &numberQuadraturePoints,
            &scalarCoeffAlpha,
            &shapeValQuads[0],
            &numberNodesPerElement,
            &nonLocalProjectorBasisImagTimesJxW[kPoint*numberPseudoWaveFunctions*numberQuadraturePoints],
            &numberQuadraturePoints,
            &scalarCoeffBeta,
            &projectorMatrixImag[0],
            &numberNodesPerElement);    
#else
        dgemm_(&transA,
            &transB,
            &numberNodesPerElement,
            &numberPseudoWaveFunctions,
            &numberQuadraturePoints,
            &scalarCoeffAlpha,
            &shapeValQuads[0],
            &numberNodesPerElement,
            &ZetalmDeltaVlTimesJxW[0],
            &numberQuadraturePoints,
            &scalarCoeffBeta,
            &projectorMatrixReal[0],
            &numberNodesPerElement);
#endif

        for(int iPseudoWave = 0; iPseudoWave < numberPseudoWaveFunctions; ++iPseudoWave)
          for(int iNode = 0; iNode < numberNodesPerElement; ++iNode)
          {
						const double tempReal = projectorMatrixReal[iPseudoWave*numberNodesPerElement+iNode];
						const double tempImag = projectorMatrixImag[iPseudoWave*numberNodesPerElement+iNode];            
#ifdef USE_COMPLEX
						nonLocalProjectorElementMatricesAtomElem[kPoint][numberNodesPerElement*iPseudoWave + iNode].real(tempReal);
						nonLocalProjectorElementMatricesAtomElem[kPoint][numberNodesPerElement*iPseudoWave + iNode].imag(tempImag);

						nonLocalProjectorElementMatricesConjugateAtomElem[kPoint][numberNodesPerElement*iPseudoWave + iNode].real(tempReal);
						nonLocalProjectorElementMatricesConjugateAtomElem[kPoint][numberNodesPerElement*iPseudoWave + iNode].imag(-tempImag);

						nonLocalProjectorElementMatricesTransposeAtomElem[kPoint]
							[numberPseudoWaveFunctions*iNode+iPseudoWave].real(tempReal);
						nonLocalProjectorElementMatricesTransposeAtomElem[kPoint]
							[numberPseudoWaveFunctions*iNode+iPseudoWave].imag(tempImag);
#else              
							nonLocalProjectorElementMatricesAtomElem
								[numberNodesPerElement*iPseudoWave + iNode]
								= tempReal;

							nonLocalProjectorElementMatricesTransposeAtomElem
								[numberPseudoWaveFunctions*iNode+iPseudoWave]
								= tempReal;
#endif
					}//node loop
			}//k point loop


		}//element loop

		cumulativeWaveSplineId += numberPseudoWaveFunctions;
		if (numberElementsInAtomCompactSupport !=0)
			count++;

	}//atom loop

	
	//scaling nonlocal element matrices with M^{-1/2}
#ifdef USE_COMPLEX
	if(dftParameters::cellLevelMassMatrixScaling)
	  {
	    for(int iAtom = 0; iAtom < numberNonLocalAtoms; ++iAtom)
	      {
		int numberElementsInAtomCompactSupport = d_elementOneFieldIteratorsInAtomCompactSupport[iAtom].size();
		int numberPseudoWaveFunctions = d_numberPseudoAtomicWaveFunctions[iAtom];
		for(int iElem = 0; iElem < numberElementsInAtomCompactSupport; ++iElem)
		  {
		    for(int iNode = 0; iNode < numberNodesPerElement; ++iNode)
		      {
			int origElemId = d_elementIdsInAtomCompactSupport[iAtom][iElem];
			dealii::types::global_dof_index localNodeId = flattenedArrayCellLocalProcIndexIdMap[origElemId][iNode];
			double alpha = invSqrtMassVector.local_element(localNodeId);

			for(int iPseudoWave = 0; iPseudoWave < numberPseudoWaveFunctions; ++iPseudoWave)
			  {
			    for(int kPoint = 0; kPoint < maxkPoints; ++kPoint)
			      {
				//d_nonLocalProjectorElementMatricesTranspose[iAtom][iElem][kPoint][numberPseudoWaveFunctions*iNode + iPseudoWave]*=alpha;

				d_nonLocalProjectorElementMatricesTransposeCellMassMatrixScaled[iAtom][iElem][kPoint][numberPseudoWaveFunctions*iNode + iPseudoWave] = d_nonLocalProjectorElementMatricesTranspose[iAtom][iElem][kPoint][numberPseudoWaveFunctions*iNode + iPseudoWave]*alpha;

				//d_nonLocalProjectorElementMatricesCellMassMatrixScaled[iAtom][iElem][kPoint][numberNodesPerElement*iPseudoWave + iNode] = d_nonLocalProjectorElementMatricesTransposeCellMassMatrixScaled[iAtom][iElem][kPoint][numberPseudoWaveFunctions*iNode + iPseudoWave];

				d_nonLocalProjectorElementMatricesConjugateCellMassMatrixScaled[iAtom][iElem][kPoint][numberNodesPerElement*iPseudoWave + iNode] = std::conj(d_nonLocalProjectorElementMatricesTransposeCellMassMatrixScaled[iAtom][iElem][kPoint][numberPseudoWaveFunctions*iNode + iPseudoWave]);
			      }

			  }

		      }

		  }

	      }
	  }
#else
	if(dftParameters::cellLevelMassMatrixScaling)
	  {
	    for(int iAtom = 0; iAtom < numberNonLocalAtoms; ++iAtom)
	      {
		int numberElementsInAtomCompactSupport = d_elementOneFieldIteratorsInAtomCompactSupport[iAtom].size();
		int numberPseudoWaveFunctions = d_numberPseudoAtomicWaveFunctions[iAtom];
		for(int iElem = 0; iElem < numberElementsInAtomCompactSupport; ++iElem)
		  {
		    for(int iNode = 0; iNode < numberNodesPerElement; ++iNode)
		      {
			int origElemId = d_elementIdsInAtomCompactSupport[iAtom][iElem];
			dealii::types::global_dof_index localNodeId = flattenedArrayCellLocalProcIndexIdMap[origElemId][iNode];
			double alpha = invSqrtMassVector.local_element(localNodeId);

			for(int iPseudoWave = 0; iPseudoWave < numberPseudoWaveFunctions; ++iPseudoWave)
			  {
			    d_nonLocalProjectorElementMatricesTransposeCellMassMatrixScaled[iAtom][iElem][numberPseudoWaveFunctions*iNode + iPseudoWave] = d_nonLocalProjectorElementMatricesTranspose[iAtom][iElem][numberPseudoWaveFunctions*iNode + iPseudoWave]*alpha;

			    d_nonLocalProjectorElementMatricesCellMassMatrixScaled[iAtom][iElem][numberNodesPerElement*iPseudoWave + iNode] = d_nonLocalProjectorElementMatricesTransposeCellMassMatrixScaled[iAtom][iElem][numberPseudoWaveFunctions*iNode + iPseudoWave];

			  }

		      }

		  }

	      }
	  }
#endif	

	//
	//Add mpi accumulation
	//

}
	template<unsigned int FEOrder,unsigned int FEOrderElectro>
void dftClass<FEOrder,FEOrderElectro>::initNonLocalPseudoPotential_OV()
{
	d_pseudoWaveFunctionIdToFunctionIdDetails.clear();
	d_numberPseudoAtomicWaveFunctions.clear();
	d_nonLocalAtomGlobalChargeIds.clear();

	//
	//this is the data structure used to store splines corresponding to projector information of various atom types
	//
	d_pseudoWaveFunctionSplines.clear();
	d_nonLocalPseudoPotentialConstants.clear();
	d_outerMostPointPseudoProjectorData.clear();

	// Store the Map between the atomic number and the projector details
	// (i.e. map from atomicNumber to a 2D vector storing atom specific projector Id and its corresponding
	// radial and angular Ids)
	// (atomicNumber->[projectorId][Global Spline Id, l quantum number, m quantum number]
	//
	std::map<unsigned int, std::vector<std::vector<int> > > atomicNumberToWaveFunctionIdDetails;
	std::map<unsigned int, std::vector<std::vector<int> > > atomicNumberToPotentialIdMap;
	std::map<unsigned int, std::vector< std::vector<double> >> denominatorData;
	const double truncationTol = 1e-12;


	//
	// Store the number of unique splines encountered so far
	//
	unsigned int cumulativeSplineId    = 0;
	unsigned int cumulativePotSplineId = 0;
	std::map<unsigned int,std::vector<int>>  projector ;


	for(std::set<unsigned int>::iterator it = atomTypes.begin(); it != atomTypes.end(); ++it)
	{
		char pseudoAtomDataFile[256];
		sprintf(pseudoAtomDataFile, "temp/z%u/PseudoAtomDat",*it);


		unsigned int atomicNumber = *it;

		if(dftParameters::verbosity >= 2)
			pcout<<"Reading data from file: "<<pseudoAtomDataFile<<std::endl;

		//
		// open the testFunctionFileName
		//
		std::ifstream readPseudoDataFileNames(pseudoAtomDataFile);


		//
		// 2D vector to store the function Id details for the current atom type
		// [Atomic wavefunction id](global spline id, l quantum number, m quantum number)
		//
		std::vector<std::vector<int> > atomicFunctionIdDetails;

		//
		// store the number of single-atom projectors associated with the current atomic number
		//
		unsigned int numberAtomicWaveFunctions ; //numberStates;

		//
		// read number of single-atom wavefunctions
		//
		if(readPseudoDataFileNames.is_open())
			readPseudoDataFileNames >> numberAtomicWaveFunctions;


		//
		// resize atomicFunctionIdDetails
		//
		atomicFunctionIdDetails.resize(numberAtomicWaveFunctions);

		//
		// Skip the rest in the first line and proceed to next line
		//
		readPseudoDataFileNames.ignore();

		if (dftParameters::verbosity>=2)
			pcout << "Number of projectors for atom with Z: " << atomicNumber<<" is " << numberAtomicWaveFunctions << std::endl;

		//
		//string to store each line of the file
		//
		std::string readLine;

		//
		// set to store the radial(spline) function Ids
		//
		std::set<int> radFunctionIds, splineFunctionIds;
		std::vector<int> lquantum(numberAtomicWaveFunctions), mquantum(numberAtomicWaveFunctions) ;
		projector[(*it)].resize(numberAtomicWaveFunctions) ;
		//
		//
		for(unsigned int i = 0; i < numberAtomicWaveFunctions; ++i)
		{

			std::vector<int>  & radAndAngularFunctionId = atomicFunctionIdDetails[i];

			radAndAngularFunctionId.resize(3,0);

			//
			// get the next line
			//
			std::getline(readPseudoDataFileNames, readLine);
			std::istringstream lineString(readLine);

			unsigned int count = 0;
			int Id;
			double mollifierRadius;
			std::string dummyString;
			while(lineString >> dummyString)
			{
				if(count < 3)
				{

					Id = atoi(dummyString.c_str());
					//
					// insert the radial(spline) Id to the splineIds set
					//
					if(count == 1)
						radFunctionIds.insert(Id);

					if(count == 0)
					{
						splineFunctionIds.insert(Id);
						projector[(*it)][i] = Id ;
					}

					radAndAngularFunctionId[count] = Id;

				}
				//if (count==3) {
				// Id = atoi(dummyString.c_str());
				// projector[(*it)][i] = Id ;
				// }
				if(count>3)
				{
					std::cerr<<"Invalid argument in the SingleAtomData file"<<std::endl;
					exit(-1);
				}

				count++;

			}



			radAndAngularFunctionId[0] += cumulativeSplineId;

			if (dftParameters::verbosity>=2)
			{
				pcout << "Radial and Angular Functions Ids: " << radAndAngularFunctionId[0] << " " << radAndAngularFunctionId[1] << " " << radAndAngularFunctionId[2] << std::endl;
				pcout << "Projector Id: " << projector[(*it)][i] << std::endl;
			}

		}

		if (dftParameters::verbosity>=2)
			pcout << " splineFunctionIds.size() " << splineFunctionIds.size() << std::endl;

		//
		// map the atomic number to atomicNumberToFunctionIdDetails
		//
		atomicNumberToWaveFunctionIdDetails[atomicNumber] = atomicFunctionIdDetails;

		//
		// update cumulativeSplineId
		//
		cumulativeSplineId += splineFunctionIds.size();

		//
		// store the splines for the current atom type
		//
		std::vector<alglib::spline1dinterpolant> atomicSplines(splineFunctionIds.size());
		std::vector<alglib::real_1d_array> atomicRadialNodes(splineFunctionIds.size());
		std::vector<alglib::real_1d_array> atomicRadialFunctionNodalValues(splineFunctionIds.size());
		std::vector<double> outerMostRadialPointProjector(splineFunctionIds.size());

		//pcout << "Number of radial Projector wavefunctions for atomic number " << atomicNumber << " is: " << radFunctionIds.size() << std::endl;

		//
		// string to store the radial function file name
		//
		std::string tempProjRadialFunctionFileName;

		unsigned int projId = 0, numProj ;

		for(unsigned int i = 0; i < radFunctionIds.size(); ++i)
		{

			//
			// get the radial function file name (name local to the directory)
			//
			readPseudoDataFileNames >> tempProjRadialFunctionFileName;
			readPseudoDataFileNames >> numProj;


			char projRadialFunctionFileName[512];
			sprintf(projRadialFunctionFileName, "temp/z%u/%s",*it,tempProjRadialFunctionFileName.c_str());

			//
			// 2D vector to store the radial coordinate and its corresponding
			// function value
			std::vector< std::vector<double> > radialFunctionData(0);

			//
			//read the radial function file
			//
			dftUtils::readFile(numProj+1,radialFunctionData,projRadialFunctionFileName);


			int numRows = radialFunctionData.size();

			//std::cout << "Number of Rows: " << numRows << std::endl;

			for (int iProj = 0; iProj<numProj; ++iProj) 
			{
				double xData[numRows];
				double yData[numRows];

				unsigned int maxRowId = 0;
				for (int iRow = 0; iRow < numRows; ++iRow)
				{
					xData[iRow] = radialFunctionData[iRow][0];
					yData[iRow] = radialFunctionData[iRow][iProj+1];

					if(std::abs(yData[iRow])>truncationTol)
						maxRowId = iRow;

				}

				outerMostRadialPointProjector[projId] = xData[maxRowId+10];

				alglib::real_1d_array & x = atomicRadialNodes[projId];
				atomicRadialNodes[projId].setcontent(numRows, xData);

				alglib::real_1d_array & y = atomicRadialFunctionNodalValues[projId];
				atomicRadialFunctionNodalValues[projId].setcontent(numRows, yData);

				alglib::ae_int_t natural_bound_type = 1;
				alglib::spline1dbuildcubic(atomicRadialNodes[projId],
						atomicRadialFunctionNodalValues[projId],
						numRows,
						natural_bound_type,
						0.0,
						natural_bound_type,
						0.0,
						atomicSplines[projId]);

				projId++ ;
			}
		}
		d_pseudoWaveFunctionSplines.insert(d_pseudoWaveFunctionSplines.end(), atomicSplines.begin(), atomicSplines.end());
		d_outerMostPointPseudoProjectorData.insert(d_outerMostPointPseudoProjectorData.end(),outerMostRadialPointProjector.begin(),outerMostRadialPointProjector.end());


		//
		// 2D vector to store the radial coordinate and its corresponding
		// function value
		//
		std::vector<std::vector<double> > denominator(0);

		//
		//read the radial function file
		//
		std::string tempDenominatorDataFileName;
		char denominatorDataFileName[256];

		//
		//read the pseudo data file name
		//
		readPseudoDataFileNames >> tempDenominatorDataFileName ;
		sprintf(denominatorDataFileName, "temp/z%u/%s", *it, tempDenominatorDataFileName.c_str());
		dftUtils::readFile(projId,denominator,denominatorDataFileName);
		denominatorData[(*it)] = denominator ;

		readPseudoDataFileNames.close() ;
	}

	//
	// Get the number of charges present in the system
	//
	unsigned int numberGlobalCharges  = atomLocations.size();

	//
	//store information for non-local atoms
	//
	std::vector<int> nonLocalAtomGlobalChargeIds;


	for(unsigned int iCharge = 0; iCharge < numberGlobalCharges; ++iCharge)
	{

		//
		// Get the atomic number for current nucleus
		//
		unsigned int atomicNumber =  atomLocations[iCharge][0];

		//
		// Get the function id details for the current nucleus
		//
		std::vector<std::vector<int> > & atomicFunctionIdDetails =
			atomicNumberToWaveFunctionIdDetails[atomicNumber];


		//
		// Get the number of functions associated with the current nucleus
		//
		unsigned int numberAtomicWaveFunctions = atomicFunctionIdDetails.size();

		if(numberAtomicWaveFunctions > 0 )
		{
			nonLocalAtomGlobalChargeIds.push_back(iCharge);
			d_numberPseudoAtomicWaveFunctions.push_back(numberAtomicWaveFunctions);
		}


		//
		// Add the atomic wave function details to the global wave function vectors
		//
		for(unsigned iAtomWave = 0; iAtomWave < numberAtomicWaveFunctions; ++iAtomWave)
		{
			d_pseudoWaveFunctionIdToFunctionIdDetails.push_back(atomicFunctionIdDetails[iAtomWave]);
		}

	}//end of iCharge loop

	d_nonLocalAtomGlobalChargeIds = nonLocalAtomGlobalChargeIds;
	int numberNonLocalAtoms = d_nonLocalAtomGlobalChargeIds.size();

	if (dftParameters::verbosity>=2)
		pcout<<"Number of Nonlocal Atoms: " <<d_nonLocalAtomGlobalChargeIds.size()<<std::endl;

	d_nonLocalPseudoPotentialConstants.resize(numberNonLocalAtoms);

	for(int iAtom = 0; iAtom < numberNonLocalAtoms; ++iAtom)
	{

		int numberPseudoWaveFunctions = d_numberPseudoAtomicWaveFunctions[iAtom];
		d_nonLocalPseudoPotentialConstants[iAtom].resize(numberPseudoWaveFunctions,0.0);
		/*
		//
		char pseudoAtomDataFile[256];
		sprintf(pseudoAtomDataFile, "%s/data/electronicStructure/pseudoPotential/z%u/oncv/pseudoAtomData/PseudoAtomData", DFT_PATH.c_str(), atomLocations[iAtom][0]);
		//
		std::ifstream readPseudoDataFileNames(pseudoAtomDataFile);
		if(readPseudoDataFileNames.is_open()){
		while (!readPseudoDataFileNames.eof()) {
		std::getline(readPseudoDataFileNames, readLine);
		std::istringstream lineString(readLine);
		while(lineString >> tempDenominatorDataFileName)
		pcout << tempDenominatorDataFileName.c_str() << std::endl ;
		}
		}
		//std::cout << c;
		//while (!readPseudoDataFileNames.eof())
		//        readPseudoDataFileNames >> tempDenominatorDataFileName;
		pcout << tempDenominatorDataFileName.c_str() << std::endl ;
		char denominatorDataFileName[256];
		sprintf(denominatorDataFileName, "%s/data/electronicStructure/pseudoPotential/z%u/oncv/pseudoAtomData/%s", DFT_PATH.c_str(),atomLocations[iAtom][0], tempDenominatorDataFileName.c_str());

		//
		// 2D vector to store the radial coordinate and its corresponding
		// function value
		std::vector< std::vector<double> > denominatorData(0);

		//
		//read the radial function file
		//
		readFile(numberPseudoWaveFunctions,denominatorData,denominatorDataFileName);*/

		for(int iPseudoWave = 0; iPseudoWave < numberPseudoWaveFunctions; ++iPseudoWave)
		{
			d_nonLocalPseudoPotentialConstants[iAtom][iPseudoWave] = denominatorData[atomLocations[iAtom][0]][projector[atomLocations[iAtom][0]][iPseudoWave]][projector[atomLocations[iAtom][0]][iPseudoWave]];
			//d_nonLocalPseudoPotentialConstants[iAtom][iPseudoWave] = 1.0/d_nonLocalPseudoPotentialConstants[iAtom][iPseudoWave];
#ifdef DEBUG
			if (dftParameters::verbosity>=4)
				pcout<<"The value of 1/nlpConst corresponding to atom and lCount "<<iAtom<<' '<<
					iPseudoWave<<" is "<<d_nonLocalPseudoPotentialConstants[iAtom][iPseudoWave]<<std::endl;
#endif
		}


	}


	return;


}
	template<unsigned int FEOrder,unsigned int FEOrderElectro>
void dftClass<FEOrder,FEOrderElectro>::computeSparseStructureNonLocalProjectors_OV()
{

	//
	//get the number of non-local atoms
	//
	int numberNonLocalAtoms = d_nonLocalAtomGlobalChargeIds.size();
	const double nlpTolerance = 1e-8;


	//
	//pre-allocate data structures that stores the sparsity of deltaVl
	//
	d_sparsityPattern.clear();
	d_elementIteratorsInAtomCompactSupport.clear();
	d_elementIdsInAtomCompactSupport.clear();
	d_elementOneFieldIteratorsInAtomCompactSupport.clear();

	//d_sparsityPattern.resize(numberNonLocalAtoms);
	d_elementIteratorsInAtomCompactSupport.resize(numberNonLocalAtoms);
	d_elementIdsInAtomCompactSupport.resize(numberNonLocalAtoms);
	d_elementOneFieldIteratorsInAtomCompactSupport.resize(numberNonLocalAtoms);
	d_nonLocalAtomIdsInCurrentProcess.clear();

	//
	//loop over nonlocal atoms
	//
	unsigned int sparseFlag = 0;
	int cumulativeSplineId = 0;
	int waveFunctionId;


	//
	//get number of global charges
	//
	unsigned int numberGlobalCharges  = atomLocations.size();

	//
	//get FE data structures
	//
	QGauss<3>  quadrature(C_num1DQuad<3>());
	//FEValues<3> fe_values(FE, quadrature, update_values | update_gradients | update_JxW_values);
	FEValues<3> fe_values(FE, quadrature, update_quadrature_points);
	const unsigned int numberQuadraturePoints = quadrature.size();
	//const unsigned int numberElements         = triangulation.n_locally_owned_active_cells();
	typename DoFHandler<3>::active_cell_iterator cell = dofHandler.begin_active(), endc = dofHandler.end();
	int iElemCount = 0;
	for(; cell != endc; ++cell)
	{
		if(cell->is_locally_owned())
			iElemCount += 1;
	}

	const unsigned int numberElements = iElemCount;
	std::vector<int> sparsityPattern(numberElements,-1);
	for(int iAtom = 0; iAtom < numberNonLocalAtoms; ++iAtom)
	{

		//
		//temp variables
		//
		int matCount = 0;
		bool isAtomIdInProcessor=false;

		//
		//
		int numberPseudoWaveFunctions = d_numberPseudoAtomicWaveFunctions[iAtom];

		//
		//get the global charge Id of the current nonlocal atom
		//
		const int globalChargeIdNonLocalAtom =  d_nonLocalAtomGlobalChargeIds[iAtom];

		//
		//get the imageIdmap information corresponding to globalChargeIdNonLocalAtom
		//
		std::vector<int> & imageIdsList = d_globalChargeIdToImageIdMapTrunc[globalChargeIdNonLocalAtom];

		//
		//resize the data structure corresponding to sparsity pattern
		//
		//std::vector<int> sparsityPattern;(numberElements,-1);
		//d_sparsityPattern[iAtom].resize(numberElements,-1);

		if (imageIdsList.size()!=0)
		{
			std::fill(sparsityPattern.begin(),sparsityPattern.end(),-1);
			//
			//parallel loop over all elements
			//
			typename DoFHandler<3>::active_cell_iterator cell = dofHandler.begin_active(), endc = dofHandler.end();
			typename DoFHandler<3>::active_cell_iterator cellEigen = dofHandlerEigen.begin_active();

			int iElem = -1;

			for(; cell != endc; ++cell,++cellEigen)
			{
				if(cell->is_locally_owned())
				{
					iElem += 1;
					bool isSkipCell=true;
					for(int iImageAtomCount = 0; iImageAtomCount < imageIdsList.size(); ++iImageAtomCount)
					{

						int chargeId = imageIdsList[iImageAtomCount];

						Point<3> chargePoint(0.0,0.0,0.0);

						if(chargeId < numberGlobalCharges)
						{
							chargePoint[0] = atomLocations[chargeId][2];
							chargePoint[1] = atomLocations[chargeId][3];
							chargePoint[2] = atomLocations[chargeId][4];
						}
						else
						{
							chargePoint[0] = d_imagePositionsTrunc[chargeId-numberGlobalCharges][0];
							chargePoint[1] = d_imagePositionsTrunc[chargeId-numberGlobalCharges][1];
							chargePoint[2] = d_imagePositionsTrunc[chargeId-numberGlobalCharges][2];
						}

						if (chargePoint.distance(cell->center())<d_nlPSPCutOff)
						{
							isSkipCell=false;
							break;
						}
					}

					if (isSkipCell)
						continue;

					//compute the values for the current element
					fe_values.reinit(cell);

					int lTemp = 1000 ;

					for(int iPsp = 0; iPsp < numberPseudoWaveFunctions; ++iPsp)
					{
						sparseFlag = 0;
						waveFunctionId = iPsp + cumulativeSplineId;
						const int globalWaveSplineId = d_pseudoWaveFunctionIdToFunctionIdDetails[waveFunctionId][0];
						const int lQuantumNumber = d_pseudoWaveFunctionIdToFunctionIdDetails[waveFunctionId][1];
						//
						if(lQuantumNumber != lTemp) {
							lTemp = lQuantumNumber ;
							for(int iQuadPoint = 0; iQuadPoint < numberQuadraturePoints; ++iQuadPoint)
							{
								const Point<3> & quadPoint=fe_values.quadrature_point(iQuadPoint);

								for(int iImageAtomCount = 0; iImageAtomCount < imageIdsList.size(); ++iImageAtomCount)
								{

									int chargeId = imageIdsList[iImageAtomCount];

									Point<3> chargePoint(0.0,0.0,0.0);

									if(chargeId < numberGlobalCharges)
									{
										chargePoint[0] = atomLocations[chargeId][2];
										chargePoint[1] = atomLocations[chargeId][3];
										chargePoint[2] = atomLocations[chargeId][4];
									}
									else
									{
										chargePoint[0] = d_imagePositionsTrunc[chargeId-numberGlobalCharges][0];
										chargePoint[1] = d_imagePositionsTrunc[chargeId-numberGlobalCharges][1];
										chargePoint[2] = d_imagePositionsTrunc[chargeId-numberGlobalCharges][2];
									}

									double r = quadPoint.distance(chargePoint);
									double radialProjVal;

									if(r <= d_outerMostPointPseudoProjectorData[globalWaveSplineId])
										pseudoUtils::getRadialFunctionVal( r, radialProjVal, &d_pseudoWaveFunctionSplines[globalWaveSplineId] );
									else
										radialProjVal = 0.0;

									if(fabs(radialProjVal) >= nlpTolerance)
									{
										sparseFlag = 1;
										break;
									}
								}//imageAtomLoop

								if(sparseFlag == 1)
									break;

							}//quadrature loop

						}

						if(sparseFlag == 1)
							break;

					}//iPsp loop ("l" loop)

					if(sparseFlag==1) {
						sparsityPattern[iElem] = matCount;
						d_elementIteratorsInAtomCompactSupport[iAtom].push_back(cellEigen);
						d_elementIdsInAtomCompactSupport[iAtom].push_back(iElem);
						d_elementOneFieldIteratorsInAtomCompactSupport[iAtom].push_back(cell);
						matCount += 1;
						isAtomIdInProcessor=true;
					}

				}
			}//cell loop
		}
		cumulativeSplineId += numberPseudoWaveFunctions;
#ifdef DEBUG
		if (dftParameters::verbosity>=4)
			pcout<<"No.of non zero elements in the compact support of atom "<<iAtom<<" is "<<d_elementIteratorsInAtomCompactSupport[iAtom].size()<<std::endl;
#endif

		if (isAtomIdInProcessor)
		{
			d_nonLocalAtomIdsInCurrentProcess.push_back(iAtom);
			d_sparsityPattern[iAtom]=sparsityPattern;
		}

	}//atom loop

	d_nonLocalAtomIdsInElement.clear();
	d_nonLocalAtomIdsInElement.resize(numberElements);


	for(int iElem = 0; iElem < numberElements; ++iElem)
	{
		for(int iAtom = 0; iAtom < d_nonLocalAtomIdsInCurrentProcess.size(); ++iAtom)
		{
			if(d_sparsityPattern[d_nonLocalAtomIdsInCurrentProcess[iAtom]][iElem] >= 0)
				d_nonLocalAtomIdsInElement[iElem].push_back(d_nonLocalAtomIdsInCurrentProcess[iAtom]);
		}
	}

	//
	//data structures for memory optimization of projectorKetTimesVector
	//
	std::vector<unsigned int> nonLocalAtomIdsAllProcessFlattened;
	pseudoUtils::exchangeLocalList(d_nonLocalAtomIdsInCurrentProcess,
			nonLocalAtomIdsAllProcessFlattened,
			n_mpi_processes,
			mpi_communicator);

	std::vector<unsigned int> nonLocalAtomIdsSizeCurrentProcess(1); nonLocalAtomIdsSizeCurrentProcess[0]=d_nonLocalAtomIdsInCurrentProcess.size();
	std::vector<unsigned int> nonLocalAtomIdsSizesAllProcess;
	pseudoUtils::exchangeLocalList(nonLocalAtomIdsSizeCurrentProcess,
			nonLocalAtomIdsSizesAllProcess,
			n_mpi_processes,
			mpi_communicator);

	std::vector<std::vector<unsigned int> >nonLocalAtomIdsInAllProcess(n_mpi_processes);
	unsigned int count=0;
	for (unsigned int iProc=0; iProc< n_mpi_processes; iProc++)
	{
		for (unsigned int j=0; j < nonLocalAtomIdsSizesAllProcess[iProc]; j++)
		{
			nonLocalAtomIdsInAllProcess[iProc].push_back(nonLocalAtomIdsAllProcessFlattened[count]);
			count++;
		}
	}
	nonLocalAtomIdsAllProcessFlattened.clear();

	IndexSet nonLocalOwnedAtomIdsInCurrentProcess; nonLocalOwnedAtomIdsInCurrentProcess.set_size(numberNonLocalAtoms);
	nonLocalOwnedAtomIdsInCurrentProcess.add_indices(d_nonLocalAtomIdsInCurrentProcess.begin(),d_nonLocalAtomIdsInCurrentProcess.end());
	IndexSet nonLocalGhostAtomIdsInCurrentProcess(nonLocalOwnedAtomIdsInCurrentProcess);
	for (unsigned int iProc=0; iProc< n_mpi_processes; iProc++)
	{
		if (iProc < this_mpi_process)
		{
			IndexSet temp; temp.set_size(numberNonLocalAtoms);
			temp.add_indices(nonLocalAtomIdsInAllProcess[iProc].begin(),nonLocalAtomIdsInAllProcess[iProc].end());
			nonLocalOwnedAtomIdsInCurrentProcess.subtract_set(temp);
		}
	}

	nonLocalGhostAtomIdsInCurrentProcess.subtract_set(nonLocalOwnedAtomIdsInCurrentProcess);

	std::vector<unsigned int> ownedNonLocalAtomIdsSizeCurrentProcess(1); ownedNonLocalAtomIdsSizeCurrentProcess[0]=nonLocalOwnedAtomIdsInCurrentProcess.n_elements();
	std::vector<unsigned int> ownedNonLocalAtomIdsSizesAllProcess;
	pseudoUtils::exchangeLocalList(ownedNonLocalAtomIdsSizeCurrentProcess,
			ownedNonLocalAtomIdsSizesAllProcess,
			n_mpi_processes,
			mpi_communicator);
	//renumbering to make contiguous set of nonLocal atomIds
	std::map<int, int> oldToNewNonLocalAtomIds;
	std::map<int, int> newToOldNonLocalAtomIds;
	unsigned int startingCount=0;
	for (unsigned int iProc=0; iProc< n_mpi_processes; iProc++)
	{
		if (iProc < this_mpi_process)
		{
			startingCount+=ownedNonLocalAtomIdsSizesAllProcess[iProc];
		}
	}

	IndexSet nonLocalOwnedAtomIdsInCurrentProcessRenum, nonLocalGhostAtomIdsInCurrentProcessRenum;
	nonLocalOwnedAtomIdsInCurrentProcessRenum.set_size(numberNonLocalAtoms);
	nonLocalGhostAtomIdsInCurrentProcessRenum.set_size(numberNonLocalAtoms);
	for (IndexSet::ElementIterator it=nonLocalOwnedAtomIdsInCurrentProcess.begin(); it!=nonLocalOwnedAtomIdsInCurrentProcess.end(); it++)
	{
		oldToNewNonLocalAtomIds[*it]=startingCount;
		newToOldNonLocalAtomIds[startingCount]=*it;
		nonLocalOwnedAtomIdsInCurrentProcessRenum.add_index(startingCount);
		startingCount++;
	}

	pseudoUtils::exchangeNumberingMap(oldToNewNonLocalAtomIds,
			n_mpi_processes,
			mpi_communicator);
	pseudoUtils::exchangeNumberingMap(newToOldNonLocalAtomIds,
			n_mpi_processes,
			mpi_communicator);

	for (IndexSet::ElementIterator it=nonLocalGhostAtomIdsInCurrentProcess.begin(); it!=nonLocalGhostAtomIdsInCurrentProcess.end(); it++)
	{
		unsigned int newAtomId=oldToNewNonLocalAtomIds[*it];
		nonLocalGhostAtomIdsInCurrentProcessRenum.add_index(newAtomId);
	}

	if(this_mpi_process==0 && false){
		for( std::map<int, int>::const_iterator it=oldToNewNonLocalAtomIds.begin(); it!=oldToNewNonLocalAtomIds.end();it++)
			std::cout<<" old nonlocal atom id: "<<it->first <<" new nonlocal atomid: "<<it->second<<std::endl;

		std::cout<<"number of local owned non local atom ids in all processors"<< '\n';
		for (unsigned int iProc=0; iProc<n_mpi_processes; iProc++)
			std::cout<<ownedNonLocalAtomIdsSizesAllProcess[iProc]<<",";
		std::cout<<std::endl;
	}
	if (false)
	{
		std::stringstream ss1;nonLocalOwnedAtomIdsInCurrentProcess.print(ss1);
		std::stringstream ss2;nonLocalGhostAtomIdsInCurrentProcess.print(ss2);
		std::string s1(ss1.str());s1.pop_back(); std::string s2(ss2.str());s2.pop_back();
		std::cout<<"procId: "<< this_mpi_process<< " old owned: "<< s1<< " old ghost: "<< s2<<std::endl;
		std::stringstream ss3;nonLocalOwnedAtomIdsInCurrentProcessRenum.print(ss3);
		std::stringstream ss4;nonLocalGhostAtomIdsInCurrentProcessRenum.print(ss4);
		std::string s3(ss3.str());s3.pop_back(); std::string s4(ss4.str());s4.pop_back();
		std::cout<<"procId: "<< this_mpi_process<< " new owned: "<< s3<<" new ghost: "<< s4<< std::endl;
	}
	AssertThrow(nonLocalOwnedAtomIdsInCurrentProcessRenum.is_ascending_and_one_to_one(mpi_communicator),ExcMessage("Incorrect renumbering and/or partitioning of non local atom ids"));

	int numberLocallyOwnedProjectors=0;
	int numberGhostProjectors=0;
	std::vector<unsigned int> coarseNodeIdsCurrentProcess;
	for (IndexSet::ElementIterator it=nonLocalOwnedAtomIdsInCurrentProcessRenum.begin(); it!=nonLocalOwnedAtomIdsInCurrentProcessRenum.end(); it++)
	{
		coarseNodeIdsCurrentProcess.push_back(numberLocallyOwnedProjectors);
		numberLocallyOwnedProjectors += d_numberPseudoAtomicWaveFunctions[newToOldNonLocalAtomIds[*it]];

	}

	std::vector<unsigned int> ghostAtomIdNumberPseudoWaveFunctions;
	for (IndexSet::ElementIterator it=nonLocalGhostAtomIdsInCurrentProcessRenum.begin(); it!=nonLocalGhostAtomIdsInCurrentProcessRenum.end(); it++)
	{
		const unsigned temp=d_numberPseudoAtomicWaveFunctions[newToOldNonLocalAtomIds[*it]];
		numberGhostProjectors += temp;
		ghostAtomIdNumberPseudoWaveFunctions.push_back(temp);
	}

	std::vector<unsigned int> numberLocallyOwnedProjectorsCurrentProcess(1); numberLocallyOwnedProjectorsCurrentProcess[0]=numberLocallyOwnedProjectors;
	std::vector<unsigned int> numberLocallyOwnedProjectorsAllProcess;
	pseudoUtils::exchangeLocalList(numberLocallyOwnedProjectorsCurrentProcess,
			numberLocallyOwnedProjectorsAllProcess,
			n_mpi_processes,
			mpi_communicator);

	startingCount=0;
	for (unsigned int iProc=0; iProc< n_mpi_processes; iProc++)
	{
		if (iProc < this_mpi_process)
		{
			startingCount+=numberLocallyOwnedProjectorsAllProcess[iProc];
		}
	}

	d_locallyOwnedProjectorIdsCurrentProcess.clear(); d_locallyOwnedProjectorIdsCurrentProcess.set_size(std::accumulate(numberLocallyOwnedProjectorsAllProcess.begin(),numberLocallyOwnedProjectorsAllProcess.end(),0));
	std::vector<unsigned int> v(numberLocallyOwnedProjectors) ;
	std::iota (std::begin(v), std::end(v), startingCount);
	d_locallyOwnedProjectorIdsCurrentProcess.add_indices(v.begin(),v.end());

	std::vector<unsigned int> coarseNodeIdsAllProcess;
	for (unsigned int i=0; i< coarseNodeIdsCurrentProcess.size();++i)
		coarseNodeIdsCurrentProcess[i]+=startingCount;
	pseudoUtils::exchangeLocalList(coarseNodeIdsCurrentProcess,
			coarseNodeIdsAllProcess,
			n_mpi_processes,
			mpi_communicator);

	d_ghostProjectorIdsCurrentProcess.clear(); d_ghostProjectorIdsCurrentProcess.set_size(std::accumulate(numberLocallyOwnedProjectorsAllProcess.begin(),numberLocallyOwnedProjectorsAllProcess.end(),0));
	unsigned int localGhostCount=0;
	for (IndexSet::ElementIterator it=nonLocalGhostAtomIdsInCurrentProcessRenum.begin(); it!=nonLocalGhostAtomIdsInCurrentProcessRenum.end(); it++)
	{
		std::vector<unsigned int> g(ghostAtomIdNumberPseudoWaveFunctions[localGhostCount]);
		std::iota (std::begin(g), std::end(g), coarseNodeIdsAllProcess[*it]);
		d_ghostProjectorIdsCurrentProcess.add_indices(g.begin(),g.end());
		localGhostCount++;
	}
	if (false)
	{
		std::stringstream ss1;d_locallyOwnedProjectorIdsCurrentProcess.print(ss1);
		std::stringstream ss2;d_ghostProjectorIdsCurrentProcess.print(ss2);
		std::string s1(ss1.str());s1.pop_back(); std::string s2(ss2.str());s2.pop_back();
		std::cout<<"procId: "<< this_mpi_process<< " projectors owned: "<< s1<< " projectors ghost: "<< s2<<std::endl;
	}
	AssertThrow(d_locallyOwnedProjectorIdsCurrentProcess.is_ascending_and_one_to_one(mpi_communicator),ExcMessage("Incorrect numbering and/or partitioning of non local projectors"));

	d_projectorIdsNumberingMapCurrentProcess.clear();

	for (IndexSet::ElementIterator it=nonLocalOwnedAtomIdsInCurrentProcess.begin(); it!=nonLocalOwnedAtomIdsInCurrentProcess.end(); it++)
	{
		const int numberPseudoWaveFunctions=d_numberPseudoAtomicWaveFunctions[*it];

		for (unsigned int i=0; i<numberPseudoWaveFunctions;++i)
		{
			d_projectorIdsNumberingMapCurrentProcess[std::make_pair(*it,i)]=coarseNodeIdsAllProcess[oldToNewNonLocalAtomIds[*it]]+i;
		}
	}

	for (IndexSet::ElementIterator it=nonLocalGhostAtomIdsInCurrentProcess.begin(); it!=nonLocalGhostAtomIdsInCurrentProcess.end(); it++)
	{
		const int numberPseudoWaveFunctions=d_numberPseudoAtomicWaveFunctions[*it];

		for (unsigned int i=0; i<numberPseudoWaveFunctions;++i)
		{
			d_projectorIdsNumberingMapCurrentProcess[std::make_pair(*it,i)]=coarseNodeIdsAllProcess[oldToNewNonLocalAtomIds[*it]]+i;
		}
	}

	if (false){
		for (std::map<std::pair<unsigned int,unsigned int>, unsigned int>::const_iterator it=d_projectorIdsNumberingMapCurrentProcess.begin(); it!=d_projectorIdsNumberingMapCurrentProcess.end();++it)
		{
			std::cout << "procId: "<< this_mpi_process<<" ["<<it->first.first << "," << it->first.second << "] " << it->second<< std::endl;
		}
	}

#ifdef USE_COMPLEX
	distributedCPUVec<std::complex<double> > vec(d_locallyOwnedProjectorIdsCurrentProcess,
			d_ghostProjectorIdsCurrentProcess,
			mpi_communicator);
#else
	distributedCPUVec<double > vec(d_locallyOwnedProjectorIdsCurrentProcess,
			d_ghostProjectorIdsCurrentProcess,
			mpi_communicator);
#endif
	vec.update_ghost_values();
	d_projectorKetTimesVectorPar.resize(1);
	d_projectorKetTimesVectorPar[0].reinit(vec);
}
