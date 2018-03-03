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

//compute configurational force contribution from nuclear self energy on the mesh nodes using linear shape function generators
template<unsigned int FEOrder>
void forceClass<FEOrder>::computeConfigurationalForceEselfLinFE()
{
  const std::vector<std::vector<double> > & atomLocations=dftPtr->atomLocations;
  const std::vector<std::vector<double> > & imagePositions=dftPtr->d_imagePositions;
  const std::vector<double> & imageCharges=dftPtr->d_imageCharges;
  //configurational force contribution from the volume integral
  QGauss<C_DIM>  quadrature(C_num1DQuad<FEOrder>());
  FEValues<C_DIM> feForceValues (FEForce, quadrature, update_gradients | update_JxW_values);
  FEValues<C_DIM> feVselfValues (dftPtr->FE, quadrature, update_gradients);
  const unsigned int   forceDofsPerCell = FEForce.dofs_per_cell;
  const unsigned int   forceBaseIndicesPerCell = forceDofsPerCell/FEForce.components;
  Vector<double>       elementalForce (forceDofsPerCell);
  const unsigned int   numQuadPoints = quadrature.size();
  std::vector<types::global_dof_index> forceLocalDofIndices(forceDofsPerCell);
  const int numberBins=dftPtr->d_bins.size();
  std::vector<Tensor<1,C_DIM,double> > gradVselfQuad(numQuadPoints);
  std::vector<int> baseIndexDofsVec(forceBaseIndicesPerCell*C_DIM);
  Tensor<1,C_DIM,double> baseIndexForceVec;

  for (unsigned int ibase=0; ibase<forceBaseIndicesPerCell; ++ibase)
  {
    for (unsigned int idim=0; idim<C_DIM; idim++)
       baseIndexDofsVec[C_DIM*ibase+idim]=FEForce.component_to_system_index(idim,ibase);
  }

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
	elementalForce=0.0;
	for (unsigned int ibase=0; ibase<forceBaseIndicesPerCell; ++ibase)
	{
           baseIndexForceVec=0;		
	   for (unsigned int qPoint=0; qPoint<numQuadPoints; ++qPoint)
	   { 
	     baseIndexForceVec+=eshelbyTensor::getVselfBallEshelbyTensor(gradVselfQuad[qPoint])*feForceValues.shape_grad(baseIndexDofsVec[C_DIM*ibase],qPoint)*feForceValues.JxW(qPoint);
	   }//q point loop
	   for (unsigned int idim=0; idim<C_DIM; idim++)
	      elementalForce[baseIndexDofsVec[C_DIM*ibase+idim]]=baseIndexForceVec[idim];
	}//base index loop

	d_constraintsNoneForce.distribute_local_to_global(elementalForce,forceLocalDofIndices,d_configForceVectorLinFE);
        ++iter2;
     }//cell loop 
  }//bin loop


  //configurational force contribution from the surface integral
  
  QGauss<C_DIM-1>  faceQuadrature(C_num1DQuad<FEOrder>());
  FEFaceValues<C_DIM> feForceFaceValues (FEForce, faceQuadrature, update_values | update_JxW_values | update_normal_vectors | update_quadrature_points);
  //FEFaceValues<C_DIM> feVselfFaceValues (FE, faceQuadrature, update_gradients);
  const unsigned int faces_per_cell=GeometryInfo<C_DIM>::faces_per_cell;
  const unsigned int   numFaceQuadPoints = faceQuadrature.size();
  const unsigned int   forceDofsPerFace = FEForce.dofs_per_face;
  const unsigned int   forceBaseIndicesPerFace = forceDofsPerFace/FEForce.components;
  Vector<double>       elementalFaceForce(forceDofsPerFace);
  std::vector<types::global_dof_index> forceFaceLocalDofIndices(forceDofsPerFace);
  std::vector<types::global_dof_index> vselfLocalDofIndices(dftPtr->FE.dofs_per_cell);
  std::vector<unsigned int> baseIndexFaceDofsForceVec(forceBaseIndicesPerFace*C_DIM);
  Tensor<1,C_DIM,double> baseIndexFaceForceVec;
  const int numberGlobalAtoms = atomLocations.size();
	   
  for (unsigned int iFaceDof=0; iFaceDof<forceDofsPerFace; ++iFaceDof)
  {
     std::pair<unsigned int, unsigned int> baseComponentIndexPair=FEForce.face_system_to_component_index(iFaceDof); 
     baseIndexFaceDofsForceVec[C_DIM*baseComponentIndexPair.second+baseComponentIndexPair.first]=iFaceDof;
  }
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
	   elementalFaceForce=0;

	   for (unsigned int ibase=0; ibase<forceBaseIndicesPerFace; ++ibase){
             baseIndexFaceForceVec=0;
	     //const int a=forceFaceLocalDofIndices[baseIndexFaceDofsForceVec[C_DIM*ibase]];
	     //Point<C_DIM> faceBaseDofPos=d_supportPointsForce[forceFaceLocalDofIndices[baseIndexFaceDofsForceVec[C_DIM*ibase]]];
	     for (unsigned int qPoint=0; qPoint<numFaceQuadPoints; ++qPoint)
	     {  
	       Point<C_DIM> quadPoint=feForceFaceValues.quadrature_point(qPoint);
	       Tensor<1,C_DIM,double> dispClosestAtom=quadPoint-closestAtomLocation;
	       const double dist=dispClosestAtom.norm();
	       Tensor<1,C_DIM,double> gradVselfFaceQuadExact=closestAtomCharge*dispClosestAtom/dist/dist/dist;

	       /*
	       Point<C_DIM> debugPoint1,debugPoint2; debugPoint1[0]=-4;debugPoint1[1]=-4;debugPoint1[2]=4;
	       debugPoint2=debugPoint1; debugPoint2[0]=-debugPoint2[0];
	       if (faceBaseDofPos.distance(debugPoint1)<1e-5 || faceBaseDofPos.distance(debugPoint2)<1e-5){
		 const int cellDofIndex=FEForce.face_to_cell_index(baseIndexFaceDofsForceVec[C_DIM*ibase],faceId,cellForce->face_orientation(faceId),cellForce->face_flip(faceId),cellForce->face_rotation(faceId));
		 const int b=forceLocalDofIndices[cellDofIndex];
	         std::cout<< "faceId "<< faceId <<" , " <<gradVselfFaceQuadExact<< " shapeval: "<< feForceFaceValues.shape_value(cellDofIndex,qPoint) << "a: "<<a<<" b: "<< b<< " cellDofIndex: "<< cellDofIndex << " center: "<< cellForce->center() << std::endl;
	       }
               */
             
	       baseIndexFaceForceVec-=eshelbyTensor::getVselfBallEshelbyTensor(gradVselfFaceQuadExact)*feForceFaceValues.normal_vector(qPoint)*feForceFaceValues.JxW(qPoint)*feForceFaceValues.shape_value(FEForce.face_to_cell_index(baseIndexFaceDofsForceVec[C_DIM*ibase],faceId,cellForce->face_orientation(faceId),cellForce->face_flip(faceId),cellForce->face_rotation(faceId)),qPoint);
	       
	     }//q point loop
	     for (unsigned int idim=0; idim<C_DIM; idim++){
	       elementalFaceForce[baseIndexFaceDofsForceVec[C_DIM*ibase+idim]]=baseIndexFaceForceVec[idim];
	     }
	   }//base index loop
	   d_constraintsNoneForce.distribute_local_to_global(elementalFaceForce,forceFaceLocalDofIndices,d_configForceVectorLinFE);
	}//face loop
        ++iter2;
     }//cell loop 
  }//bin loop 
}



//compute configurational force on the mesh nodes using linear shape function generators
template<unsigned int FEOrder>
void forceClass<FEOrder>::computeConfigurationalForcePhiExtLinFE()
{
  
  FEEvaluation<C_DIM,1,C_num1DQuad<FEOrder>(),C_DIM>  forceEval(dftPtr->matrix_free_data,d_forceDofHandlerIndex, 0);

  FEEvaluation<C_DIM,FEOrder,C_num1DQuad<FEOrder>(),1> eshelbyEval(dftPtr->matrix_free_data,dftPtr->phiExtDofHandlerIndex, 0);//no constraints
   
  
  for (unsigned int cell=0; cell<dftPtr->matrix_free_data.n_macro_cells(); ++cell){
    forceEval.reinit(cell);
    eshelbyEval.reinit(cell);
    eshelbyEval.read_dof_values_plain(dftPtr->poissonPtr->phiExt);
    eshelbyEval.evaluate(true,true);
    for (unsigned int q=0; q<forceEval.n_q_points; ++q){
	 VectorizedArray<double> phiExt_q =eshelbyEval.get_value(q);   
	 Tensor<1,C_DIM,VectorizedArray<double> > gradPhiExt_q =eshelbyEval.get_gradient(q);
	 forceEval.submit_gradient(eshelbyTensor::getPhiExtEshelbyTensor(phiExt_q,gradPhiExt_q),q);
    }
    forceEval.integrate (false,true);
    forceEval.distribute_local_to_global(d_configForceVectorLinFE);//also takes care of constraints

  }
} 

template<unsigned int FEOrder>
void forceClass<FEOrder>::computeConfigurationalForceEselfNoSurfaceLinFE()
{
  FEEvaluation<C_DIM,1,C_num1DQuad<FEOrder>(),C_DIM>  forceEval(dftPtr->matrix_free_data,d_forceDofHandlerIndex, 0);

  FEEvaluation<C_DIM,FEOrder,C_num1DQuad<FEOrder>(),1> eshelbyEval(dftPtr->matrix_free_data,dftPtr->phiExtDofHandlerIndex, 0);//no constraints
   
  for (unsigned int iBin=0; iBin< dftPtr->d_vselfFieldBins.size() ; iBin++){
    for (unsigned int cell=0; cell<dftPtr->matrix_free_data.n_macro_cells(); ++cell){
      forceEval.reinit(cell);
      eshelbyEval.reinit(cell);
      eshelbyEval.read_dof_values_plain(dftPtr->d_vselfFieldBins[iBin]);
      eshelbyEval.evaluate(false,true);
      for (unsigned int q=0; q<forceEval.n_q_points; ++q){
	  
	  Tensor<1,C_DIM,VectorizedArray<double> > gradVself_q =eshelbyEval.get_gradient(q);

	  forceEval.submit_gradient(eshelbyTensor::getVselfBallEshelbyTensor(gradVself_q),q);
 
      }
      forceEval.integrate (false,true);
      forceEval.distribute_local_to_global (d_configForceVectorLinFE);
    }
  }
  
   
}
