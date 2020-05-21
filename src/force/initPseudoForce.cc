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


#include "pseudoForceUtils.cc"
#include "nonlocalProjectorKetTimesEigenVectors.cc"

	template<unsigned int FEOrder>
void forceClass<FEOrder>::computeElementalNonLocalPseudoDataForce()
{
	//
	//get the number of non-local atoms
	//
	const unsigned int numberNonLocalAtoms = dftPtr->d_nonLocalAtomGlobalChargeIds.size();

	//
	//get number of global charges
	//
	const unsigned int numberGlobalCharges  = dftPtr->atomLocations.size();


	//
	//get FE data structures
	//
	QGauss<3>  quadrature(C_num1DQuad<FEOrder>());
	FEValues<3> fe_values(dftPtr->FE, quadrature, update_quadrature_points);
	const unsigned int numberNodesPerElement  = dftPtr->FE.dofs_per_cell;
	const unsigned int numberQuadraturePoints = quadrature.size();
	const unsigned int numKPoints=dftPtr->d_kPointWeights.size();

	//
	//get number of kPoints
	//
	const unsigned int numkPoints = dftPtr->d_kPointWeights.size();

	//
	//clear existing data
	//
	d_nonLocalPSP_ZetalmDeltaVl.clear();
#ifdef USE_COMPLEX
	d_nonLocalPSP_gradZetalmDeltaVl_minusZetalmDeltaVl_KPoint.clear();
	d_nonLocalPSP_gradZetalmDeltaVl_KPoint.clear();
	d_nonLocalPSP_gradZetalmDeltaVlDyadicDistImageAtoms_KPoint.clear();
#else
	d_nonLocalPSP_gradZetalmDeltaVl.clear();
#endif
	//
	//
	int cumulativePotSplineId = 0;
	int cumulativeWaveSplineId = 0;
	int waveFunctionId;
	int pseudoPotentialId;
	unsigned int count=0;
	const unsigned int numNonLocalAtomsCurrentProcess= dftPtr->d_nonLocalAtomIdsInCurrentProcess.size();
	d_nonLocalPSP_ZetalmDeltaVl.resize(numNonLocalAtomsCurrentProcess);
#ifdef USE_COMPLEX
	d_nonLocalPSP_gradZetalmDeltaVl_minusZetalmDeltaVl_KPoint.resize(numNonLocalAtomsCurrentProcess);
	d_nonLocalPSP_gradZetalmDeltaVl_KPoint.resize(numNonLocalAtomsCurrentProcess);
	d_nonLocalPSP_gradZetalmDeltaVlDyadicDistImageAtoms_KPoint.resize(numNonLocalAtomsCurrentProcess);
#else
	d_nonLocalPSP_gradZetalmDeltaVl.resize(numNonLocalAtomsCurrentProcess);
#endif

	for(int iAtom = 0; iAtom < numberNonLocalAtoms; ++iAtom)
	{
		//
		//get the global charge Id of the current nonlocal atom
		//
		const int globalChargeIdNonLocalAtom =  dftPtr->d_nonLocalAtomGlobalChargeIds[iAtom];


		Point<3> nuclearCoordinates(dftPtr->atomLocations[globalChargeIdNonLocalAtom][2],dftPtr->atomLocations[globalChargeIdNonLocalAtom][3],dftPtr->atomLocations[globalChargeIdNonLocalAtom][4]);

		std::vector<int> & imageIdsList = dftPtr->d_globalChargeIdToImageIdMap[globalChargeIdNonLocalAtom];

		//
		//get the number of elements in the compact support of the current nonlocal atom
		//
		const unsigned int numberElementsInAtomCompactSupport = dftPtr->d_elementOneFieldIteratorsInAtomCompactSupport[iAtom].size();

		//
		//get the number of pseudowavefunctions for the current nonlocal atoms
		//
		const unsigned int numberPseudoWaveFunctions = dftPtr->d_numberPseudoAtomicWaveFunctions[iAtom];
		const unsigned int numberAngularMomentumSpecificPotentials = dftPtr->d_numberPseudoPotentials[iAtom];

		//
		//allocate
		//
		if (numberElementsInAtomCompactSupport !=0)
		{
			d_nonLocalPSP_ZetalmDeltaVl[count].resize(numberPseudoWaveFunctions);
#ifdef USE_COMPLEX
			d_nonLocalPSP_gradZetalmDeltaVl_minusZetalmDeltaVl_KPoint[count].resize(numberPseudoWaveFunctions);
			d_nonLocalPSP_gradZetalmDeltaVl_KPoint[count].resize(numberPseudoWaveFunctions);
			d_nonLocalPSP_gradZetalmDeltaVlDyadicDistImageAtoms_KPoint[count].resize(numberPseudoWaveFunctions);
#else
			d_nonLocalPSP_gradZetalmDeltaVl[count].resize(numberPseudoWaveFunctions);
#endif
		}

		for(unsigned int iElemComp = 0; iElemComp < numberElementsInAtomCompactSupport; ++iElemComp)
		{

			DoFHandler<3>::active_cell_iterator cell = dftPtr->d_elementOneFieldIteratorsInAtomCompactSupport[iAtom][iElemComp];

			//compute values for the current elements
			fe_values.reinit(cell);

			int iPsp = -1;
			int lTemp = 1e5;

			for(unsigned int iPseudoWave = 0; iPseudoWave < numberPseudoWaveFunctions; ++iPseudoWave)
			{
#ifdef USE_COMPLEX
				d_nonLocalPSP_ZetalmDeltaVl[count][iPseudoWave][cell->id()]=std::vector<double>(numkPoints*numberQuadraturePoints*2);
				d_nonLocalPSP_gradZetalmDeltaVl_minusZetalmDeltaVl_KPoint[count][iPseudoWave][cell->id()]=std::vector<double>(numkPoints*numberQuadraturePoints*C_DIM*2);
				d_nonLocalPSP_gradZetalmDeltaVl_KPoint[count][iPseudoWave][cell->id()]=std::vector<double>(numkPoints*numberQuadraturePoints*C_DIM*2);
				d_nonLocalPSP_gradZetalmDeltaVlDyadicDistImageAtoms_KPoint[count][iPseudoWave][cell->id()]=std::vector<double>(numkPoints*numberQuadraturePoints*C_DIM*C_DIM*2);
#else
				d_nonLocalPSP_ZetalmDeltaVl[count][iPseudoWave][cell->id()]=std::vector<double>(numberQuadraturePoints);
				d_nonLocalPSP_gradZetalmDeltaVl[count][iPseudoWave][cell->id()]=std::vector<double>(numberQuadraturePoints*C_DIM);
#endif

				waveFunctionId = iPseudoWave + cumulativeWaveSplineId;
				const int globalWaveSplineId = dftPtr->d_pseudoWaveFunctionIdToFunctionIdDetails[waveFunctionId][0];
				const int lQuantumNumber = dftPtr->d_pseudoWaveFunctionIdToFunctionIdDetails[waveFunctionId][1];
				const int mQuantumNumber = dftPtr->d_pseudoWaveFunctionIdToFunctionIdDetails[waveFunctionId][2];

				//
				//access pseudoPotential Ids
				//
				if(lQuantumNumber != lTemp)
					iPsp += 1;
				pseudoPotentialId = iPsp + cumulativePotSplineId;
				lTemp = lQuantumNumber;

				const int globalPotSplineId = dftPtr->d_deltaVlIdToFunctionIdDetails[pseudoPotentialId][0];
				assert(lQuantumNumber == dftPtr->d_deltaVlIdToFunctionIdDetails[pseudoPotentialId][1]);
#ifdef USE_COMPLEX
				std::vector<double>  ZetalmDeltaVl_KPoint(numkPoints*numberQuadraturePoints*2,0.0);
				std::vector<double> gradZetalmDeltaVl_KPoint(numkPoints*numberQuadraturePoints*C_DIM*2,0.0);
				std::vector<double> gradZetalmDeltaVl_minusZetalmDeltaVl_KPoint(numkPoints*numberQuadraturePoints*C_DIM*2,0.0);
				std::vector<double> gradZetalmDeltaVlDyadicDistImageAtoms_KPoint(numkPoints*numberQuadraturePoints*C_DIM*C_DIM*2,0.0);
#else
				std::vector<double> ZetalmDeltaVl(numberQuadraturePoints,0.0);
				std::vector<double> gradZetalmDeltaVl(numberQuadraturePoints*C_DIM,0.0);
#endif

				double nlpValue = 0.0;
				for(unsigned int iQuadPoint = 0; iQuadPoint < numberQuadraturePoints; ++iQuadPoint)
				{

					Point<3> quadPoint=fe_values.quadrature_point(iQuadPoint);

					for(unsigned int iImageAtomCount = 0; iImageAtomCount < imageIdsList.size(); ++iImageAtomCount)
					{

						int chargeId = imageIdsList[iImageAtomCount];

						Point<3> chargePoint(0.0,0.0,0.0);

						if(chargeId < numberGlobalCharges)
						{
							chargePoint[0] = dftPtr->atomLocations[chargeId][2];
							chargePoint[1] = dftPtr->atomLocations[chargeId][3];
							chargePoint[2] = dftPtr->atomLocations[chargeId][4];
						}
						else
						{
							chargePoint[0] = dftPtr->d_imagePositions[chargeId-numberGlobalCharges][0];
							chargePoint[1] = dftPtr->d_imagePositions[chargeId-numberGlobalCharges][1];
							chargePoint[2] = dftPtr->d_imagePositions[chargeId-numberGlobalCharges][2];
						}

						double x[3],qMinusLr[3];

						x[0] = quadPoint[0] - chargePoint[0];
						x[1] = quadPoint[1] - chargePoint[1];
						x[2] = quadPoint[2] - chargePoint[2];
						qMinusLr[0] =x[0] +nuclearCoordinates[0];
						qMinusLr[1] =x[1] +nuclearCoordinates[1];
						qMinusLr[2] =x[2] +nuclearCoordinates[2];

						//
						// get the spherical coordinates from cartesian
						//
						double r,theta,phi;
						pseudoForceUtils::convertCartesianToSpherical(x,r,theta,phi);

						double radialWaveFunVal, sphericalHarmonicVal, radialPotFunVal, pseudoWaveFunctionValue, deltaVlValue;
						std::vector<double> pseudoWaveFunctionDerivatives(3,0.0);
						std::vector<double> deltaVlDerivatives(3,0.0);
						if(r <= dftPtr->d_pspTail)//d_outerMostPointPseudoWaveFunctionsData[globalWaveSplineId])
						{
							pseudoForceUtils::getRadialFunctionVal(r,
									radialWaveFunVal,
									&dftPtr->d_pseudoWaveFunctionSplines[globalWaveSplineId]);

							pseudoForceUtils::getSphericalHarmonicVal(theta,phi,lQuantumNumber,mQuantumNumber,sphericalHarmonicVal);

							pseudoWaveFunctionValue = radialWaveFunVal*sphericalHarmonicVal;

							pseudoForceUtils::getRadialFunctionVal(r,
									radialPotFunVal,
									&dftPtr->d_deltaVlSplines[globalPotSplineId]);

							deltaVlValue = radialPotFunVal;

							pseudoForceUtils::getPseudoWaveFunctionDerivatives(r,
									theta,
									phi,
									lQuantumNumber,
									mQuantumNumber,
									pseudoWaveFunctionDerivatives,
									dftPtr->d_pseudoWaveFunctionSplines[globalWaveSplineId]);

							pseudoForceUtils::getDeltaVlDerivatives(r,
									x,
									deltaVlDerivatives,
									dftPtr->d_deltaVlSplines[globalPotSplineId]);
							std::vector<double> tempDer(3);
							for(unsigned int iDim = 0; iDim < C_DIM; ++iDim)
							{
								tempDer[iDim]=pseudoWaveFunctionDerivatives[iDim]*radialPotFunVal + pseudoWaveFunctionValue*deltaVlDerivatives[iDim];
							}
#ifdef USE_COMPLEX
							for (unsigned int ik=0; ik < numkPoints; ++ik)
							{

								const double kDotqMinusLr= dftPtr->d_kPointCoordinates[ik*C_DIM+0]*qMinusLr[0]+ dftPtr->d_kPointCoordinates[ik*C_DIM+1]*qMinusLr[1]+dftPtr->d_kPointCoordinates[ik*C_DIM+2]*qMinusLr[2];
								const double tempReal=std::cos(-kDotqMinusLr);
								const double tempImag=std::sin(-kDotqMinusLr);
								ZetalmDeltaVl_KPoint[ik*numberQuadraturePoints*2+2*iQuadPoint+0] += tempReal*deltaVlValue*pseudoWaveFunctionValue;
								ZetalmDeltaVl_KPoint[ik*numberQuadraturePoints*2+2*iQuadPoint+1] += tempImag*deltaVlValue*pseudoWaveFunctionValue;
								for(unsigned int iDim = 0; iDim < C_DIM; ++iDim)
								{
									gradZetalmDeltaVl_KPoint[ik*numberQuadraturePoints*C_DIM*2+iQuadPoint*C_DIM*2+iDim*2+0]+= tempReal*tempDer[iDim];
									gradZetalmDeltaVl_KPoint[ik*numberQuadraturePoints*C_DIM*2+iQuadPoint*C_DIM*2+iDim*2+1]+= tempImag*tempDer[iDim];
									gradZetalmDeltaVl_minusZetalmDeltaVl_KPoint[ik*numberQuadraturePoints*C_DIM*2+iQuadPoint*C_DIM*2+iDim*2+0]+= tempReal*tempDer[iDim];
									gradZetalmDeltaVl_minusZetalmDeltaVl_KPoint[ik*numberQuadraturePoints*C_DIM*2+iQuadPoint*C_DIM*2+iDim*2+1]+= tempImag*tempDer[iDim];
									gradZetalmDeltaVl_minusZetalmDeltaVl_KPoint[ik*numberQuadraturePoints*C_DIM*2+iQuadPoint*C_DIM*2+iDim*2+0]+= tempImag*deltaVlValue*pseudoWaveFunctionValue*dftPtr->d_kPointCoordinates[ik*C_DIM+iDim];
									gradZetalmDeltaVl_minusZetalmDeltaVl_KPoint[ik*numberQuadraturePoints*C_DIM*2+iQuadPoint*C_DIM*2+iDim*2+1]-= tempReal*deltaVlValue*pseudoWaveFunctionValue*dftPtr->d_kPointCoordinates[ik*C_DIM+iDim];
									for(unsigned int jDim=0; jDim < C_DIM; ++jDim)
									{
										gradZetalmDeltaVlDyadicDistImageAtoms_KPoint[ik*numberQuadraturePoints*C_DIM*C_DIM*2+iQuadPoint*C_DIM*C_DIM*2+iDim*C_DIM*2+jDim*2+0]+= tempReal*tempDer[iDim]*x[jDim];
										gradZetalmDeltaVlDyadicDistImageAtoms_KPoint[ik*numberQuadraturePoints*C_DIM*C_DIM*2+iQuadPoint*C_DIM*C_DIM*2+iDim*C_DIM*2+jDim*2+1]+= tempImag*tempDer[iDim]*x[jDim];
									}
								}
							}
#else

							ZetalmDeltaVl[iQuadPoint] += deltaVlValue*pseudoWaveFunctionValue;

							for(unsigned int iDim = 0; iDim < C_DIM; ++iDim)
								gradZetalmDeltaVl[iQuadPoint*C_DIM+iDim]+= tempDer[iDim];

#endif
						}// within psp tail check

					}//image atom loop (contribution added)

				}//end of quad loop
#ifdef USE_COMPLEX
				d_nonLocalPSP_ZetalmDeltaVl[count][iPseudoWave][cell->id()]=ZetalmDeltaVl_KPoint;
				d_nonLocalPSP_gradZetalmDeltaVl_KPoint[count][iPseudoWave][cell->id()]=gradZetalmDeltaVl_KPoint;
				d_nonLocalPSP_gradZetalmDeltaVl_minusZetalmDeltaVl_KPoint[count][iPseudoWave][cell->id()]=gradZetalmDeltaVl_minusZetalmDeltaVl_KPoint;
				d_nonLocalPSP_gradZetalmDeltaVlDyadicDistImageAtoms_KPoint[count][iPseudoWave][cell->id()]=gradZetalmDeltaVlDyadicDistImageAtoms_KPoint;
#else
				d_nonLocalPSP_ZetalmDeltaVl[count][iPseudoWave][cell->id()]=ZetalmDeltaVl;
				d_nonLocalPSP_gradZetalmDeltaVl[count][iPseudoWave][cell->id()]=gradZetalmDeltaVl;
#endif

			}//end of iPseudoWave loop


		}//element loop

		cumulativePotSplineId += numberAngularMomentumSpecificPotentials;
		cumulativeWaveSplineId += numberPseudoWaveFunctions;
		if (numberElementsInAtomCompactSupport !=0)
			count++;

	}//atom loop

}


	template<unsigned int FEOrder>
void forceClass<FEOrder>::computeNonLocalProjectorKetTimesPsiTimesV(const std::vector<distributedCPUVec<double>> &src,
		std::vector<std::vector<double> > & projectorKetTimesPsiTimesVReal,
		std::vector<std::vector<std::complex<double> > > & projectorKetTimesPsiTimesVComplex,
		const unsigned int kPointIndex)
{
	//
	//get FE data
	//
	QGauss<3>  quadrature_formula(C_num1DQuad<FEOrder>());

	const unsigned int dofs_per_cell = dftPtr->FEEigen.dofs_per_cell;

#ifdef USE_COMPLEX
	const unsigned int numberNodesPerElement = dftPtr->FEEigen.dofs_per_cell/2;//GeometryInfo<3>::vertices_per_cell;
#else
	const unsigned int numberNodesPerElement = dftPtr->FEEigen.dofs_per_cell;
#endif

	//
	//compute nonlocal projector ket times x i.e C^{T}*X
	//
	std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);
#ifdef USE_COMPLEX
	std::vector<std::vector<std::complex<double> > > & projectorKetTimesVector=projectorKetTimesPsiTimesVComplex;
#else
	std::vector<std::vector<double> > & projectorKetTimesVector=projectorKetTimesPsiTimesVReal;
#endif

	const unsigned int numberWaveFunctions = src.size();
	projectorKetTimesVector.clear();
	projectorKetTimesVector.resize(dftPtr->d_nonLocalAtomIdsInCurrentProcess.size());

	//
	//allocate memory for matrix-vector product
	//
	std::map<unsigned int, unsigned int> globalToLocalMap;
	for(int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInCurrentProcess.size(); ++iAtom)
	{
		const int atomId=dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
		globalToLocalMap[atomId]=iAtom;
		int numberSingleAtomPseudoWaveFunctions = dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
		projectorKetTimesVector[iAtom].resize(numberWaveFunctions*numberSingleAtomPseudoWaveFunctions,0.0);
	}

	//
	//some useful vectors
	//
#ifdef USE_COMPLEX
	std::vector<std::complex<double> > inputVectors(numberNodesPerElement*numberWaveFunctions,0.0);
#else
	std::vector<double> inputVectors(numberNodesPerElement*numberWaveFunctions,0.0);
#endif


	//
	//parallel loop over all elements to compute nonlocal projector ket times x i.e C^{T}*X
	//
	typename DoFHandler<3>::active_cell_iterator cell = dftPtr->dofHandlerEigen.begin_active(), endc = dftPtr->dofHandlerEigen.end();
	int iElem = -1;
	for(; cell!=endc; ++cell)
	{
		if(cell->is_locally_owned())
		{
			iElem += 1;
			cell->get_dof_indices(local_dof_indices);

			unsigned int index=0;
#ifdef USE_COMPLEX
			std::vector<double> temp(dofs_per_cell,0.0);
			for (std::vector<distributedCPUVec<double>>::const_iterator it=src.begin(); it!=src.end(); it++)
			{
				(*it).extract_subvector_to(local_dof_indices.begin(), local_dof_indices.end(), temp.begin());
				for(int idof = 0; idof < dofs_per_cell; ++idof)
				{
					//
					//This is the component index 0(real) or 1(imag).
					//
					const unsigned int ck = dftPtr->FEEigen.system_to_component_index(idof).first;
					const unsigned int iNode = dftPtr->FEEigen.system_to_component_index(idof).second;
					if(ck == 0)
						inputVectors[numberNodesPerElement*index + iNode].real(temp[idof]);
					else
						inputVectors[numberNodesPerElement*index + iNode].imag(temp[idof]);
				}
				index++;
			}


#else
			for (std::vector<distributedCPUVec<double>>::const_iterator it=src.begin(); it!=src.end(); it++)
			{
				(*it).extract_subvector_to(local_dof_indices.begin(), local_dof_indices.end(), inputVectors.begin()+numberNodesPerElement*index);
				index++;
			}
#endif

			for(int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInElement[iElem].size();++iAtom)
			{
				int atomId = dftPtr->d_nonLocalAtomIdsInElement[iElem][iAtom];
				const unsigned int numberPseudoWaveFunctions = dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
				int nonZeroElementMatrixId = dftPtr->d_sparsityPattern[atomId][iElem];
#ifdef USE_COMPLEX
				char transA = 'C';
				char transB = 'N';
				std::complex<double> alpha = 1.0;
				std::complex<double> beta = 1.0;
				zgemm_(&transA,
						&transB,
						&numberPseudoWaveFunctions,
						&numberWaveFunctions,
						&numberNodesPerElement,
						&alpha,
						&dftPtr->d_nonLocalProjectorElementMatrices[atomId][nonZeroElementMatrixId][kPointIndex][0],
						&numberNodesPerElement,
						&inputVectors[0],
						&numberNodesPerElement,
						&beta,
						&projectorKetTimesVector[globalToLocalMap[atomId]][0],
						&numberPseudoWaveFunctions);
#else
				char transA = 'T';
				char transB = 'N';
				double alpha = 1.0;
				double beta = 1.0;
				dgemm_(&transA,
						&transB,
						&numberPseudoWaveFunctions,
						&numberWaveFunctions,
						&numberNodesPerElement,
						&alpha,
						&dftPtr->d_nonLocalProjectorElementMatrices[atomId][nonZeroElementMatrixId][0],
						&numberNodesPerElement,
						&inputVectors[0],
						&numberNodesPerElement,
						&beta,
						&projectorKetTimesVector[globalToLocalMap[atomId]][0],
						&numberPseudoWaveFunctions);
#endif
			}

		}

	}//element loop

	//std::cout<<"Finished Element Loop"<<std::endl;
#ifdef USE_COMPLEX
	std::vector<distributedCPUVec<std::complex<double> > > projectorKetTimesVectorPar(numberWaveFunctions);
#else
	std::vector<distributedCPUVec<double> > projectorKetTimesVectorPar(numberWaveFunctions);
#endif
#ifdef USE_COMPLEX
	distributedCPUVec<std::complex<double> > vec(dftPtr->d_locallyOwnedProjectorIdsCurrentProcess,
			dftPtr->d_ghostProjectorIdsCurrentProcess,
			mpi_communicator);
#else
	distributedCPUVec<double > vec(dftPtr->d_locallyOwnedProjectorIdsCurrentProcess,
			dftPtr->d_ghostProjectorIdsCurrentProcess,
			mpi_communicator);
#endif
	vec.update_ghost_values();
	for (unsigned int i=0; i<numberWaveFunctions;++i)
	{
#ifdef USE_COMPLEX
		projectorKetTimesVectorPar[i].reinit(vec);
#else
		projectorKetTimesVectorPar[i].reinit(vec);
#endif
	}

	for(unsigned int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInCurrentProcess.size(); ++iAtom)
	{
		const int atomId=dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
		const unsigned int numberPseudoWaveFunctions = dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
		for(unsigned int iWave = 0; iWave < numberWaveFunctions; ++iWave)
		{
			for(unsigned int iPseudoAtomicWave = 0; iPseudoAtomicWave < numberPseudoWaveFunctions; ++iPseudoAtomicWave)
			{
				projectorKetTimesVectorPar[iWave][dftPtr->d_projectorIdsNumberingMapCurrentProcess[std::make_pair(atomId,iPseudoAtomicWave)]]
					=projectorKetTimesVector[iAtom][numberPseudoWaveFunctions*iWave + iPseudoAtomicWave];
			}
		}
	}

	for (unsigned int i=0; i<numberWaveFunctions;++i)
	{
		projectorKetTimesVectorPar[i].compress(VectorOperation::add);
		projectorKetTimesVectorPar[i].update_ghost_values();
	}

	for(unsigned int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInCurrentProcess.size(); ++iAtom)
	{
		const int atomId=dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
		const unsigned int numberPseudoWaveFunctions = dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
		for(unsigned int iWave = 0; iWave < numberWaveFunctions; ++iWave)
		{
			for(unsigned int iPseudoAtomicWave = 0; iPseudoAtomicWave < numberPseudoWaveFunctions; ++iPseudoAtomicWave)
			{
				projectorKetTimesVector[iAtom][numberPseudoWaveFunctions*iWave + iPseudoAtomicWave]
					=projectorKetTimesVectorPar[iWave][dftPtr->d_projectorIdsNumberingMapCurrentProcess[std::make_pair(atomId,iPseudoAtomicWave)]];

			}
		}
	}


	//
	//compute V*C^{T}*X
	//
	for(unsigned int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInCurrentProcess.size(); ++iAtom)
	{
		const int atomId=dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
		const unsigned int numberPseudoWaveFunctions =  dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
		for(unsigned int iWave = 0; iWave < numberWaveFunctions; ++iWave)
		{
			for(unsigned int iPseudoAtomicWave = 0; iPseudoAtomicWave < numberPseudoWaveFunctions; ++iPseudoAtomicWave)
				projectorKetTimesVector[iAtom][numberPseudoWaveFunctions*iWave + iPseudoAtomicWave] *= dftPtr->d_nonLocalPseudoPotentialConstants[atomId][iPseudoAtomicWave];
		}
	}


}
