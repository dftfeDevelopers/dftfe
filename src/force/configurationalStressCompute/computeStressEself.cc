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
// @author Sambit Das (2017)
//

#ifdef ENABLE_PERIODIC_BC 
//compute stress contribution from nuclear self energy
template<unsigned int FEOrder>
void forceClass<FEOrder>::computeStressEself()
{
  const std::vector<std::vector<double> > & atomLocations=dftPtr->atomLocations;
  const std::vector<std::vector<double> > & imagePositions=dftPtr->d_imagePositions;
  const std::vector<double> & imageCharges=dftPtr->d_imageCharges;
  //configurational stress contribution from the volume integral
  QGauss<C_DIM>  quadrature(C_num1DQuad<FEOrder>());
  FEValues<C_DIM> feForceValues (FEForce, quadrature, update_gradients | update_JxW_values);
  FEValues<C_DIM> feVselfValues (dftPtr->FE, quadrature, update_gradients);
  const unsigned int   forceDofsPerCell = FEForce.dofs_per_cell;
  Vector<double>       elementalForce (forceDofsPerCell);
  const unsigned int   numQuadPoints = quadrature.size();
  std::vector<types::global_dof_index> forceLocalDofIndices(forceDofsPerCell);
  const int numberBins=dftPtr->d_bins.size();
  std::vector<Tensor<1,C_DIM,double> > gradVselfQuad(numQuadPoints);

  for(int iBin = 0; iBin < numberBins; ++iBin)
  {
    const std::vector<DoFHandler<C_DIM>::active_cell_iterator> & cellsVselfBallDofHandler=d_cellsVselfBallsDofHandler[iBin];	   
    const std::vector<DoFHandler<C_DIM>::active_cell_iterator> & cellsVselfBallDofHandlerForce=d_cellsVselfBallsDofHandlerForce[iBin]; 
    const vectorType & iBinVselfField= dftPtr->d_vselfFieldBins[iBin];
    std::vector<DoFHandler<C_DIM>::active_cell_iterator>::const_iterator iter1;
    std::vector<DoFHandler<C_DIM>::active_cell_iterator>::const_iterator iter2;
    iter2 = cellsVselfBallDofHandlerForce.begin();
    for (iter1 = cellsVselfBallDofHandler.begin(); iter1 != cellsVselfBallDofHandler.end(); ++iter1)
    {
	DoFHandler<C_DIM>::active_cell_iterator cell=*iter1;
	DoFHandler<C_DIM>::active_cell_iterator cellForce=*iter2;
	feVselfValues.reinit(cell);
	feVselfValues.get_function_gradients(iBinVselfField,gradVselfQuad);

	feForceValues.reinit(cellForce);
	cellForce->get_dof_indices(forceLocalDofIndices);
	for (unsigned int qPoint=0; qPoint<numQuadPoints; ++qPoint)
	{ 
	     d_stress+=eshelbyTensor::getVselfBallEshelbyTensor(gradVselfQuad[qPoint])*feForceValues.JxW(qPoint);
	}//q point loop
        ++iter2;
     }//cell loop 
  }//bin loop


  //configurational stress contribution from the surface integral
  
  QGauss<C_DIM-1>  faceQuadrature(C_num1DQuad<FEOrder>());
  FEFaceValues<C_DIM> feForceFaceValues (FEForce, faceQuadrature, update_values | update_JxW_values | update_normal_vectors | update_quadrature_points);
  //FEFaceValues<C_DIM> feVselfFaceValues (FE, faceQuadrature, update_gradients);
  const unsigned int faces_per_cell=GeometryInfo<C_DIM>::faces_per_cell;
  const unsigned int   numFaceQuadPoints = faceQuadrature.size();
  const unsigned int   forceDofsPerFace = FEForce.dofs_per_face;
  Vector<double>       elementalFaceForce(forceDofsPerFace);
  std::vector<types::global_dof_index> forceFaceLocalDofIndices(forceDofsPerFace);
  std::vector<types::global_dof_index> vselfLocalDofIndices(dftPtr->FE.dofs_per_cell);
  const int numberGlobalAtoms = atomLocations.size();
	   

  for(int iBin = 0; iBin < numberBins; ++iBin)
  {
    std::map<dealii::types::global_dof_index, int> & closestAtomBinMap = dftPtr->d_closestAtomBin[iBin];
    const std::map<DoFHandler<C_DIM>::active_cell_iterator,std::vector<unsigned int > >  & cellsVselfBallSurfacesDofHandler=d_cellFacesVselfBallSurfacesDofHandler[iBin];	   
    const std::map<DoFHandler<C_DIM>::active_cell_iterator,std::vector<unsigned int > >  & cellsVselfBallSurfacesDofHandlerForce=d_cellFacesVselfBallSurfacesDofHandlerForce[iBin]; 
    const vectorType & iBinVselfField= dftPtr->d_vselfFieldBins[iBin];
    std::map<DoFHandler<C_DIM>::active_cell_iterator,std::vector<unsigned int > >::const_iterator iter1;
    std::map<DoFHandler<C_DIM>::active_cell_iterator,std::vector<unsigned int > >::const_iterator iter2;
    iter2 = cellsVselfBallSurfacesDofHandlerForce.begin();
    for (iter1 = cellsVselfBallSurfacesDofHandler.begin(); iter1 != cellsVselfBallSurfacesDofHandler.end(); ++iter1)
    {
	DoFHandler<C_DIM>::active_cell_iterator cell=iter1->first;
	cell->get_dof_indices(vselfLocalDofIndices);
        const int closestAtomId=closestAtomBinMap[vselfLocalDofIndices[0]];//is same for all faces in the cell
        double closestAtomCharge;
	Point<C_DIM> closestAtomLocation;
	if(closestAtomId < numberGlobalAtoms)
	{
           closestAtomLocation[0]=atomLocations[closestAtomId][2];
	   closestAtomLocation[1]=atomLocations[closestAtomId][3];
	   closestAtomLocation[2]=atomLocations[closestAtomId][4];
	   if(dftParameters::isPseudopotential)
	      closestAtomCharge = atomLocations[closestAtomId][1];
           else
	      closestAtomCharge = atomLocations[closestAtomId][0];
        }
	else{
           const int imageId=closestAtomId-numberGlobalAtoms;
	   closestAtomCharge = imageCharges[imageId];
           closestAtomLocation[0]=imagePositions[imageId][0];
	   closestAtomLocation[1]=imagePositions[imageId][1];
	   closestAtomLocation[2]=imagePositions[imageId][2];
        }

	DoFHandler<C_DIM>::active_cell_iterator cellForce=iter2->first;

	const std::vector<unsigned int > & dirichletFaceIds= iter1->second;
	for (unsigned int index=0; index< dirichletFaceIds.size(); index++){
           const int faceId=dirichletFaceIds[index];
	   //feVselfFaceValues.reinit(cell,faceId);
	   //std::vector<Tensor<1,C_DIM,double> > gradVselfFaceQuad(numFaceQuadPoints);
	   //feVselfFaceValues.get_function_gradients(iBinVselfField,gradVselfFaceQuad);
            
	   feForceFaceValues.reinit(cellForce,faceId);
	   cellForce->face(faceId)->get_dof_indices(forceFaceLocalDofIndices);

	   for (unsigned int qPoint=0; qPoint<numFaceQuadPoints; ++qPoint)
	   {  
	       Point<C_DIM> quadPoint=feForceFaceValues.quadrature_point(qPoint);
	       Tensor<1,C_DIM,double> dispClosestAtom=quadPoint-closestAtomLocation;
	       const double dist=dispClosestAtom.norm();
	       Tensor<1,C_DIM,double> gradVselfFaceQuadExact=closestAtomCharge*dispClosestAtom/dist/dist/dist;
             
	       //d_stress-=outer_product(eshelbyTensor::getVselfBallEshelbyTensor(gradVselfFaceQuadExact)*feForceFaceValues.normal_vector(qPoint),dispClosestAtom)*feForceFaceValues.JxW(qPoint);
	       d_stress-=outer_product(dispClosestAtom,eshelbyTensor::getVselfBallEshelbyTensor(gradVselfFaceQuadExact)*feForceFaceValues.normal_vector(qPoint))*feForceFaceValues.JxW(qPoint);	       
	       
	   }//q point loop
	}//face loop
        ++iter2;
     }//cell loop 
  }//bin loop 
}
#endif
