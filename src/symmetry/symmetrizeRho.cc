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
// @author Krishnendu Ghosh (2017)
//

//source file for symmetrization of electron density in periodic calculations

#include "../../include/dftParameters.h"
#include "../../include/symmetry.h"
#include "../../include/dft.h"

using namespace dftParameters ;

template<unsigned int FEOrder>
void symmetryClass<FEOrder>::computeAndSymmetrize_rhoOut()
{
  QGauss<3>  quadrature(FEOrder+1);
  const unsigned int num_quad_points = quadrature.size();

  dftPtr->rhoOutValues = new std::map<dealii::CellId,std::vector<double> >;
  dftPtr->rhoOutVals.push_back(dftPtr->rhoOutValues);
  if (spinPolarized==1)
  {
    dftPtr->rhoOutValuesSpinPolarized = new std::map<dealii::CellId,std::vector<double> >;
    dftPtr->rhoOutValsSpinPolarized.push_back(dftPtr->rhoOutValuesSpinPolarized);
  }
  if(xc_id == 4)
  {
   dftPtr->gradRhoOutValues = new std::map<dealii::CellId, std::vector<double> >;
   dftPtr->gradRhoOutVals.push_back(dftPtr->gradRhoOutValues);
   if (spinPolarized==1)
      {
         dftPtr->gradRhoOutValuesSpinPolarized = new std::map<dealii::CellId, std::vector<double> >;
         dftPtr->gradRhoOutValsSpinPolarized.push_back(dftPtr->gradRhoOutValuesSpinPolarized);
      }
  }
 
  //temp arrays
  std::vector<double> rhoOut(num_quad_points), gradRhoOut(3*num_quad_points), rhoOutSpinPolarized(2*num_quad_points),  gradRhoOutSpinPolarized(6*num_quad_points);
  
  //parallel loop over all elements
  typename DoFHandler<3>::active_cell_iterator cell = (dftPtr->dofHandlerEigen).begin_active(), endc = (dftPtr->dofHandlerEigen).end();
  for (; cell!=endc; ++cell) 
    {
      if (cell->is_locally_owned())
	{
	  (*(dftPtr->rhoOutValues))[cell->id()] = std::vector<double>(num_quad_points);
	  std::fill(rhoOut.begin(),rhoOut.end(),0.0);
          if (spinPolarized==1)
    	   {
	       	(*(dftPtr->rhoOutValuesSpinPolarized))[cell->id()] = std::vector<double>(2*num_quad_points);
		std::fill(rhoOutSpinPolarized.begin(),rhoOutSpinPolarized.end(),0.0);
	   }
	  //
	  if(xc_id == 4) {//GGA
	      (*(dftPtr->gradRhoOutValues))[cell->id()] = std::vector<double>(3*num_quad_points);
	      std::fill(gradRhoOut.begin(),gradRhoOut.end(),0.0);
	     if (spinPolarized==1) {
		(*(dftPtr->gradRhoOutValuesSpinPolarized))[cell->id()] = std::vector<double>(6*num_quad_points);
	        std::fill(gradRhoOutSpinPolarized.begin(),gradRhoOutSpinPolarized.end(),0.0);
	     }
	  }
	  //
	  for(unsigned int q_point=0; q_point<num_quad_points; ++q_point) {
	     for(unsigned int iSymm=0; iSymm<numSymm; ++iSymm) {
		unsigned int proc = std::get<0>(mappedGroup[iSymm][globalCellId[cell->id()]][q_point]) ;
		unsigned int group = std::get<1>(mappedGroup[iSymm][globalCellId[cell->id()]][q_point]) ;
		unsigned int point = std::get<2>(mappedGroup[iSymm][globalCellId[cell->id()]][q_point]) ;
		//
		if (spinPolarized==1) {
		   rhoOutSpinPolarized[2*q_point] += rhoRecvd[iSymm][globalCellId[cell->id()]][proc][2*point];
		   rhoOutSpinPolarized[2*q_point+1] += rhoRecvd[iSymm][globalCellId[cell->id()]][proc][2*point+1];
		}
		else
		   rhoOut[q_point] += rhoRecvd[iSymm][globalCellId[cell->id()]][proc][point];
		if (xc_id==4){
		   if (spinPolarized==1) {
		     gradRhoOutSpinPolarized[6*q_point+0] += gradRhoRecvd[iSymm][globalCellId[cell->id()]][proc][6*point+0];
		     gradRhoOutSpinPolarized[6*q_point+1] += gradRhoRecvd[iSymm][globalCellId[cell->id()]][proc][6*point+1];
		     gradRhoOutSpinPolarized[6*q_point+2] += gradRhoRecvd[iSymm][globalCellId[cell->id()]][proc][6*point+2];
		     gradRhoOutSpinPolarized[6*q_point+3] += gradRhoRecvd[iSymm][globalCellId[cell->id()]][proc][6*point+3];
		     gradRhoOutSpinPolarized[6*q_point+4] += gradRhoRecvd[iSymm][globalCellId[cell->id()]][proc][6*point+4];
		     gradRhoOutSpinPolarized[6*q_point+5] += gradRhoRecvd[iSymm][globalCellId[cell->id()]][proc][6*point+5];
		   }
		   else {
		     gradRhoOut[3*q_point+0] += gradRhoRecvd[iSymm][globalCellId[cell->id()]][proc][3*point+0];
		     gradRhoOut[3*q_point+1] += gradRhoRecvd[iSymm][globalCellId[cell->id()]][proc][3*point+1];
		     gradRhoOut[3*q_point+2] += gradRhoRecvd[iSymm][globalCellId[cell->id()]][proc][3*point+2];
		   } 
		  }
		}	
		if(spinPolarized==1)
		{
		   (*(dftPtr->rhoOutValuesSpinPolarized))[cell->id()][2*q_point]=rhoOutSpinPolarized[2*q_point] ;
		   (*(dftPtr->rhoOutValuesSpinPolarized))[cell->id()][2*q_point+1]=rhoOutSpinPolarized[2*q_point+1] ;
	           (*(dftPtr->rhoOutValues))[cell->id()][q_point]= rhoOutSpinPolarized[2*q_point] + rhoOutSpinPolarized[2*q_point+1];
                 }
		else
		   (*(dftPtr->rhoOutValues))[cell->id()][q_point]  = rhoOut[q_point];
		//
		if (xc_id==4) {
                  if(spinPolarized==1)
		      {
			(*dftPtr->gradRhoOutValuesSpinPolarized)[cell->id()][6*q_point + 0] = gradRhoOutSpinPolarized[6*q_point + 0];
		        (*dftPtr->gradRhoOutValuesSpinPolarized)[cell->id()][6*q_point + 1] = gradRhoOutSpinPolarized[6*q_point + 1];
		        (*dftPtr->gradRhoOutValuesSpinPolarized)[cell->id()][6*q_point + 2] = gradRhoOutSpinPolarized[6*q_point + 2];
			(*dftPtr->gradRhoOutValuesSpinPolarized)[cell->id()][6*q_point + 3] = gradRhoOutSpinPolarized[6*q_point + 3];
		        (*dftPtr->gradRhoOutValuesSpinPolarized)[cell->id()][6*q_point + 4] = gradRhoOutSpinPolarized[6*q_point + 4];
		        (*dftPtr->gradRhoOutValuesSpinPolarized)[cell->id()][6*q_point + 5] = gradRhoOutSpinPolarized[6*q_point + 5];
			//
			(*dftPtr->gradRhoOutValues)[cell->id()][3*q_point + 0] = gradRhoOutSpinPolarized[6*q_point + 0] + gradRhoOutSpinPolarized[6*q_point + 3];
		        (*dftPtr->gradRhoOutValues)[cell->id()][3*q_point + 1] = gradRhoOutSpinPolarized[6*q_point + 1] + gradRhoOutSpinPolarized[6*q_point + 4];
		        (*dftPtr->gradRhoOutValues)[cell->id()][3*q_point + 2] = gradRhoOutSpinPolarized[6*q_point + 2] + gradRhoOutSpinPolarized[6*q_point + 5];
                      }
		   else
		      {
		        (*dftPtr->gradRhoOutValues)[cell->id()][3*q_point + 0] = gradRhoOut[3*q_point + 0];
		        (*dftPtr->gradRhoOutValues)[cell->id()][3*q_point + 1] = gradRhoOut[3*q_point + 1];
		        (*dftPtr->gradRhoOutValues)[cell->id()][3*q_point + 2] = gradRhoOut[3*q_point + 2];
		      }
	       }
	    }

	}

    }

  //pop out rhoInVals and rhoOutVals if their size exceeds mixing history size
  if((dftPtr->rhoInVals).size() == mixingHistory)
    {
      (dftPtr->rhoInVals).pop_front();
      (dftPtr->rhoOutVals).pop_front();
      (dftPtr->rhoInValsSpinPolarized).pop_front();
      (dftPtr->rhoOutValsSpinPolarized).pop_front();
      (dftPtr->gradRhoInVals).pop_front();
      (dftPtr->gradRhoOutVals).pop_front();
      (dftPtr->gradRhoInValsSpinPolarized).pop_front();
      (dftPtr->gradRhoOutValsSpinPolarized).pop_front();
    }

}
template<unsigned int FEOrder>
void symmetryClass<FEOrder>::computeLocalrhoOut()
{
  QGauss<3>  quadrature(FEOrder+1);
  const unsigned int num_quad_points = quadrature.size();
  totPoints = recvdData1[0].size() ;
  double px, py, pz; 
  std::vector<Vector<double> > tempPsiAlpha, tempPsiBeta ;
  std::vector<std::vector<Tensor<1,3,double> > > tempGradPsi, tempGradPsiTempAlpha, tempGradPsiTempBeta  ;
  std::vector<Point<3>> quadPointList ;
  std::vector<double>  sendData, recvdData;
  //project eigen vectors to regular FEM space by multiplying with M^(-0.5)
  for(int kPoint = 0; kPoint < (1+spinPolarized)*(dftPtr->d_maxkPoints); ++kPoint)
    {
      for (unsigned int i = 0; i < (dftPtr->numEigenValues); ++i)
	{
	  *((dftPtr->eigenVectorsOrig)[kPoint][i])=*((dftPtr->eigenVectors)[kPoint][i]);
	  (*((dftPtr->eigenVectorsOrig)[kPoint][i])).scale(dftPtr->eigenPtr->massVector);
	  (dftPtr->constraintsNoneEigen).distribute(*((dftPtr->eigenVectorsOrig)[kPoint][i]));
	  ((dftPtr->eigenVectorsOrig)[kPoint][i])->update_ghost_values();
	}
    }
     //pcout<<"check 6: "<<std::endl;
  std::vector<double> rhoLocal, gradRhoLocal, rhoLocalSpinPolarized, gradRhoLocalSpinPolarized ;
  rhoLocal.resize(totPoints, 0.0);
  //
  if (spinPolarized==1) 
     rhoLocalSpinPolarized.resize(2*totPoints, 0.0);
  //
  if(xc_id == 4) {//GGA
     gradRhoLocal.resize(3*totPoints, 0.0);
     if (spinPolarized)
	gradRhoLocalSpinPolarized.resize(6*totPoints, 0.0);
  }
  unsigned int numPointsDone = 0, numGroupsDone = 0 ;
  for ( unsigned int proc = 0; proc < n_mpi_processes; ++proc) {
     //
     for ( unsigned int iGroup = 0; iGroup < recv_size0[proc] ; ++iGroup ) {
       //
	unsigned int numPoint = recvdData2[ numGroupsDone + iGroup];
        unsigned int cellId =   recvdData0[ numGroupsDone + iGroup];
       //
       tempPsiAlpha.resize(numPoint);
       tempPsiBeta.resize(numPoint);
       quadPointList.resize(numPoint);
       if(xc_id == 4){ //GGA
         tempGradPsi.resize(numPoint);
	 tempGradPsiTempAlpha.resize(numPoint);
	 tempGradPsiTempBeta.resize(numPoint);
        }
	for (unsigned int iList=0; iList<numPoint; ++iList) {
       //
         px = recvdData1[0][numPointsDone+iList] ;
         py = recvdData1[1][numPointsDone+iList] ;
         pz = recvdData1[2][numPointsDone+iList] ;
	//	     
         Point<3> pointTemp(px, py, pz) ;	     
         quadPointList[iList] = pointTemp;
         tempPsiAlpha[iList].reinit(2);
         tempPsiBeta[iList].reinit(2);
         if(xc_id == 4){ //GGA
	    tempGradPsi[iList].resize(2);
	    tempGradPsiTempAlpha[iList].resize(2);
	    tempGradPsiTempBeta[iList].resize(2);
	   }
	} // loop on points
       //  
      //
     Quadrature<3> quadRule (quadPointList) ;
     FEValues<3> fe_values (dftPtr->FEEigen, quadRule, update_values | update_gradients| update_JxW_values | update_quadrature_points);
     fe_values.reinit(dealIICellId[cellId]) ;
     unsigned int iSymm = recvdData3[numGroupsDone + iGroup] ;
     //
     //pcout << " check 1.1 " << std::endl;
     //
     for(int kPoint = 0; kPoint < (dftPtr->d_maxkPoints); ++kPoint)
         {
         if (symmUnderGroup[kPoint][iSymm] ==1) {
	     for(unsigned int i=0; i<(dftPtr->numEigenValues); ++i)
		      {
		       double factor=((dftPtr->eigenValues)[kPoint][i]-(dftPtr->fermiEnergy))/(C_kb*TVal);
		       double partialOccupancyAlpha = (factor >= 0)?std::exp(-factor)/(1.0 + std::exp(-factor)):1.0/(1.0 + std::exp(factor));
		       //
		       factor=((dftPtr->eigenValues)[kPoint][i+spinPolarized*(dftPtr->numEigenValues)]-(dftPtr->fermiEnergy))/(C_kb*TVal);
		       double partialOccupancyBeta = (factor >= 0)?std::exp(-factor)/(1.0 + std::exp(-factor)):1.0/(1.0 + std::exp(factor));
		       //
		       fe_values.get_function_values(*((dftPtr->eigenVectorsOrig)[(1+spinPolarized)*kPoint][i]), tempPsiAlpha);
		       if (spinPolarized==1)
			  fe_values.get_function_values(*((dftPtr->eigenVectorsOrig)[(1+spinPolarized)*kPoint+1][i]), tempPsiBeta);
		       //
		       if(xc_id == 4){
			   fe_values.get_function_gradients(*((dftPtr->eigenVectorsOrig)[(1+spinPolarized)*kPoint][i]),tempGradPsiTempAlpha);
			   if (spinPolarized==1)
			   fe_values.get_function_gradients(*((dftPtr->eigenVectorsOrig)[(1+spinPolarized)*kPoint+1][i]),tempGradPsiTempBeta);
			}
		       //
		       for (unsigned int iList=0; iList<numPoint; ++iList){
			    //
			    if (spinPolarized==1) {
			    rhoLocalSpinPolarized[2*(numPointsDone+iList)] += 1.0 / (double(numSymmUnderGroup[kPoint])) *
							partialOccupancyAlpha*(dftPtr->d_kPointWeights)[kPoint]*(tempPsiAlpha[iList](0)*tempPsiAlpha[iList](0) + tempPsiAlpha[iList](1)*tempPsiAlpha[iList](1));
			    rhoLocalSpinPolarized[2*(numPointsDone+iList)+1] += 1.0 / (double(numSymmUnderGroup[kPoint])) *
							partialOccupancyBeta*(dftPtr->d_kPointWeights)[kPoint]*(tempPsiBeta[iList](0)*tempPsiBeta[iList](0) + tempPsiBeta[iList](1)*tempPsiBeta[iList](1));
			     }
			     else
				 rhoLocal[numPointsDone+iList] += 1.0 / (double(numSymmUnderGroup[kPoint]))  *
							2.0*partialOccupancyAlpha*(dftPtr->d_kPointWeights)[kPoint]*(tempPsiAlpha[iList](0)*tempPsiAlpha[iList](0) + tempPsiAlpha[iList](1)*tempPsiAlpha[iList](1));
			     if(xc_id == 4) {//GGA
			     //
			     if (spinPolarized==1) {
			     tempGradPsi[iList][0][0]=
				(tempGradPsiTempAlpha[iList][0][0]*symmMat[iSymm][0][0] + tempGradPsiTempAlpha[iList][0][1]*symmMat[iSymm][1][0] + tempGradPsiTempAlpha[iList][0][2]*symmMat[iSymm][2][0]) ;
                 	     tempGradPsi[iList][0][1] = 
				(tempGradPsiTempAlpha[iList][0][0]*symmMat[iSymm][0][1] + tempGradPsiTempAlpha[iList][0][1]*symmMat[iSymm][1][1] + tempGradPsiTempAlpha[iList][0][2]*symmMat[iSymm][2][1]) ;
                 	     tempGradPsi[iList][0][2] =
				(tempGradPsiTempAlpha[iList][0][0]*symmMat[iSymm][0][2] + tempGradPsiTempAlpha[iList][0][1]*symmMat[iSymm][1][2] + tempGradPsiTempAlpha[iList][0][2]*symmMat[iSymm][2][2]) ; 
			     tempGradPsi[iList][1][0]=
				(tempGradPsiTempAlpha[iList][1][0]*symmMat[iSymm][0][0] + tempGradPsiTempAlpha[iList][1][1]*symmMat[iSymm][1][0] + tempGradPsiTempAlpha[iList][1][2]*symmMat[iSymm][2][0]) ;
                 	     tempGradPsi[iList][1][1] = 
				(tempGradPsiTempAlpha[iList][1][0]*symmMat[iSymm][0][1] + tempGradPsiTempAlpha[iList][1][1]*symmMat[iSymm][1][1] + tempGradPsiTempAlpha[iList][1][2]*symmMat[iSymm][2][1]) ;
                 	     tempGradPsi[iList][1][2] =
				(tempGradPsiTempAlpha[iList][1][0]*symmMat[iSymm][0][2] + tempGradPsiTempAlpha[iList][1][1]*symmMat[iSymm][1][2] + tempGradPsiTempAlpha[iList][1][2]*symmMat[iSymm][2][2]) ;


			     //
			     gradRhoLocalSpinPolarized[6*(numPointsDone+iList) + 0] += 1.0 / (double(numSymmUnderGroup[kPoint])) *
						2.0*partialOccupancyAlpha*(dftPtr->d_kPointWeights)[kPoint]*(tempPsiAlpha[iList](0)*tempGradPsi[iList][0][0] + tempPsiAlpha[iList](1)*tempGradPsi[iList][1][0]);
			     gradRhoLocalSpinPolarized[6*(numPointsDone+iList) + 1] += 1.0 / (double(numSymmUnderGroup[kPoint])) *
						2.0*partialOccupancyAlpha*(dftPtr->d_kPointWeights)[kPoint]*(tempPsiAlpha[iList](0)*tempGradPsi[iList][0][1] + tempPsiAlpha[iList](1)*tempGradPsi[iList][1][1]);
			     gradRhoLocalSpinPolarized[6*(numPointsDone+iList) + 2] += 1.0 / (double(numSymmUnderGroup[kPoint])) *
						2.0*partialOccupancyAlpha*(dftPtr->d_kPointWeights)[kPoint]*(tempPsiAlpha[iList](0)*tempGradPsi[iList][0][2] + tempPsiAlpha[iList](1)*tempGradPsi[iList][1][2]); 

		             tempGradPsi[iList][0][0]=
				(tempGradPsiTempBeta[iList][0][0]*symmMat[iSymm][0][0] + tempGradPsiTempBeta[iList][0][1]*symmMat[iSymm][1][0] + tempGradPsiTempBeta[iList][0][2]*symmMat[iSymm][2][0]) ;
                 	     tempGradPsi[iList][0][1] = 
				(tempGradPsiTempBeta[iList][0][0]*symmMat[iSymm][0][1] + tempGradPsiTempBeta[iList][0][1]*symmMat[iSymm][1][1] + tempGradPsiTempBeta[iList][0][2]*symmMat[iSymm][2][1]) ;
                 	     tempGradPsi[iList][0][2] =
				(tempGradPsiTempBeta[iList][0][0]*symmMat[iSymm][0][2] + tempGradPsiTempBeta[iList][0][1]*symmMat[iSymm][1][2] + tempGradPsiTempBeta[iList][0][2]*symmMat[iSymm][2][2]) ; 
			     tempGradPsi[iList][1][0]=
				(tempGradPsiTempBeta[iList][1][0]*symmMat[iSymm][0][0] + tempGradPsiTempBeta[iList][1][1]*symmMat[iSymm][1][0] + tempGradPsiTempBeta[iList][1][2]*symmMat[iSymm][2][0]) ;
                 	     tempGradPsi[iList][1][1] = 
				(tempGradPsiTempBeta[iList][1][0]*symmMat[iSymm][0][1] + tempGradPsiTempBeta[iList][1][1]*symmMat[iSymm][1][1] + tempGradPsiTempBeta[iList][1][2]*symmMat[iSymm][2][1]) ;
                 	     tempGradPsi[iList][1][2] =
				(tempGradPsiTempBeta[iList][1][0]*symmMat[iSymm][0][2] + tempGradPsiTempBeta[iList][1][1]*symmMat[iSymm][1][2] + tempGradPsiTempBeta[iList][1][2]*symmMat[iSymm][2][2]) ;


			     //
			     gradRhoLocalSpinPolarized[6*(numPointsDone+iList) + 3] += 1.0 / (double(numSymmUnderGroup[kPoint])) *
						2.0*partialOccupancyBeta*(dftPtr->d_kPointWeights)[kPoint]*(tempPsiBeta[iList](0)*tempGradPsi[iList][0][0] + tempPsiBeta[iList](1)*tempGradPsi[iList][1][0]);
			     gradRhoLocalSpinPolarized[6*(numPointsDone+iList) + 4] += 1.0 / (double(numSymmUnderGroup[kPoint])) *
						2.0*partialOccupancyBeta*(dftPtr->d_kPointWeights)[kPoint]*(tempPsiBeta[iList](0)*tempGradPsi[iList][0][1] + tempPsiBeta[iList](1)*tempGradPsi[iList][1][1]);
			     gradRhoLocalSpinPolarized[6*(numPointsDone+iList) + 5] += 1.0 / (double(numSymmUnderGroup[kPoint])) *
						2.0*partialOccupancyBeta*(dftPtr->d_kPointWeights)[kPoint]*(tempPsiBeta[iList](0)*tempGradPsi[iList][0][2] + tempPsiBeta[iList](1)*tempGradPsi[iList][1][2]); 
				}
			     else {
				 tempGradPsi[iList][0][0]=
				(tempGradPsiTempAlpha[iList][0][0]*symmMat[iSymm][0][0] + tempGradPsiTempAlpha[iList][0][1]*symmMat[iSymm][1][0] + tempGradPsiTempAlpha[iList][0][2]*symmMat[iSymm][2][0]) ;
                 	     tempGradPsi[iList][0][1] = 
				(tempGradPsiTempAlpha[iList][0][0]*symmMat[iSymm][0][1] + tempGradPsiTempAlpha[iList][0][1]*symmMat[iSymm][1][1] + tempGradPsiTempAlpha[iList][0][2]*symmMat[iSymm][2][1]) ;
                 	     tempGradPsi[iList][0][2] =
				(tempGradPsiTempAlpha[iList][0][0]*symmMat[iSymm][0][2] + tempGradPsiTempAlpha[iList][0][1]*symmMat[iSymm][1][2] + tempGradPsiTempAlpha[iList][0][2]*symmMat[iSymm][2][2]) ; 
			     tempGradPsi[iList][1][0]=
				(tempGradPsiTempAlpha[iList][1][0]*symmMat[iSymm][0][0] + tempGradPsiTempAlpha[iList][1][1]*symmMat[iSymm][1][0] + tempGradPsiTempAlpha[iList][1][2]*symmMat[iSymm][2][0]) ;
                 	     tempGradPsi[iList][1][1] = 
				(tempGradPsiTempAlpha[iList][1][0]*symmMat[iSymm][0][1] + tempGradPsiTempAlpha[iList][1][1]*symmMat[iSymm][1][1] + tempGradPsiTempAlpha[iList][1][2]*symmMat[iSymm][2][1]) ;
                 	     tempGradPsi[iList][1][2] =
				(tempGradPsiTempAlpha[iList][1][0]*symmMat[iSymm][0][2] + tempGradPsiTempAlpha[iList][1][1]*symmMat[iSymm][1][2] + tempGradPsiTempAlpha[iList][1][2]*symmMat[iSymm][2][2]) ;


			     //
			     gradRhoLocal[3*(numPointsDone+iList) + 0] += 1.0 / (double(numSymmUnderGroup[kPoint])) *
						2.0*2.0*partialOccupancyAlpha*(dftPtr->d_kPointWeights)[kPoint]*(tempPsiAlpha[iList](0)*tempGradPsi[iList][0][0] + tempPsiAlpha[iList](1)*tempGradPsi[iList][1][0]);
			     gradRhoLocal[3*(numPointsDone+iList) + 1] += 1.0 / (double(numSymmUnderGroup[kPoint])) *
						2.0*2.0*partialOccupancyAlpha*(dftPtr->d_kPointWeights)[kPoint]*(tempPsiAlpha[iList](0)*tempGradPsi[iList][0][1] + tempPsiAlpha[iList](1)*tempGradPsi[iList][1][1]);
			     gradRhoLocal[3*(numPointsDone+iList) + 2] += 1.0 / (double(numSymmUnderGroup[kPoint])) *
						2.0*2.0*partialOccupancyAlpha*(dftPtr->d_kPointWeights)[kPoint]*(tempPsiAlpha[iList](0)*tempGradPsi[iList][0][2] + tempPsiAlpha[iList](1)*tempGradPsi[iList][1][2]); 
				}
			     }
			} // loop on points list
		      } // loop on eigenValues
		     } // if this symm is part of the group under this kpoint
	  } // loop on k Points
       //
       numPointsDone += numPoint ;
    } // loop on group 
    //
    numGroupsDone += recv_size0[proc] ;
    //
    //
   } // loop on proc 
   //
   //std::cout << this_mpi_process << " check 1.2 " << std::endl;
   //MPI_Barrier(mpi_communicator);
    sendData.resize((1+spinPolarized)*totPoints) ;
   if (spinPolarized==1){
     
      sendData = rhoLocalSpinPolarized ;
      /*for(int i = 0; i < n_mpi_processes; i++) {
         recv_size1[i] = 2*recv_size1[i] ;
         mpi_offsets1[i] = 2*mpi_offsets1[i] ;
	}*/
     }
   else
      sendData = rhoLocal ;
   //
   //std::cout << this_mpi_process << " check 1.3 " << std::endl;
   //
   typename DoFHandler<3>::active_cell_iterator cell ;
   typename DoFHandler<3>::active_cell_iterator endc = (dftPtr->dofHandlerEigen).end();

        //
	for ( int sendProc = 0; sendProc<n_mpi_processes; ++sendProc) {
	    //
	    //
            recvdData.resize(recv_size[sendProc]);
	    //pcout << " sendData.size()  " << sendData.size() << "  "  << std::endl ;
	    //for(int i = 0; i < n_mpi_processes; i++)
	    //    pcout << recv_size1[i] << "   " << mpi_offsets1[i] << "  "  << std::endl ;
	    //
	    //MPI_Barrier(mpi_communicator);
	    MPI_Scatterv(&(sendData[0]),&(recv_size1[0]), &(mpi_offsets1[0]), MPI_DOUBLE, &(recvdData[0]), recv_size[sendProc], MPI_DOUBLE,sendProc,mpi_communicator);
	    /*for(int i = 0; i < n_mpi_processes; i++) {
		if (this_mpi_process==i)
	           std::cout << this_mpi_process << " check 1.31 " << recvSize[sendProc] << std::endl;
	        //MPI_Barrier(mpi_communicator);
	     }*/
	    //
	    cell = (dftPtr->dofHandlerEigen).begin_active();
	    unsigned int offset = 0 ;
            for (; cell!=endc; ++cell) 
                {
		  if (cell->is_locally_owned()) {
                 for (unsigned int iSymm=0; iSymm<numSymm; ++iSymm) {
			for ( unsigned int i=0; i<rhoRecvd[iSymm][globalCellId[cell->id()]][sendProc].size(); ++i)	    
		              rhoRecvd[iSymm][globalCellId[cell->id()]][sendProc][i] = recvdData [ offset + i]  ;
		  	offset += rhoRecvd[iSymm][globalCellId[cell->id()]][sendProc].size();
	           }
		 }
	        }
	     recvdData.clear();
	     //MPI_Barrier(mpi_communicator);
	     //
	     //
	  }
     //
     //std::cout << this_mpi_process << " check 1.4 " << std::endl;
     //	
	    if (xc_id==4) { //GGA
	       sendData.resize(3*(1+spinPolarized)*totPoints) ;
               if (spinPolarized==1)
		  sendData = gradRhoLocalSpinPolarized ;
	       else
	          sendData = gradRhoLocal ;
	       /*for(int i = 0; i < n_mpi_processes; i++) {
                     recv_size1[i] = 3*recv_size1[i] ;
                     mpi_offsets1[i] = 3*mpi_offsets1[i] ;
	          }*/
	     
	    //
	    //pcout << totPoints << "  " << sendData.size() << "  "  << std::endl ;
	    for ( int sendProc = 0; sendProc<n_mpi_processes; ++sendProc) {
            recvdData.resize(3*recv_size[sendProc]);
	    MPI_Scatterv(&(sendData[0]),&(recvGrad_size1[0]), &(mpiGrad_offsets1[0]), MPI_DOUBLE, &(recvdData[0]), 3*recv_size[sendProc], MPI_DOUBLE,sendProc,mpi_communicator);
	    //
	    cell = (dftPtr->dofHandlerEigen).begin_active();
	    //pcout <<  << "  " << sendData.size() << "  "  << std::endl ;
	
	    unsigned int offset = 0 ;
            for (; cell!=endc; ++cell) 
                {
		if (cell->is_locally_owned()) {
                 for (unsigned int iSymm=0; iSymm<numSymm; ++iSymm) {
			for ( unsigned int i=0; i<gradRhoRecvd[iSymm][globalCellId[cell->id()]][sendProc].size(); ++i)	    
		              gradRhoRecvd[iSymm][globalCellId[cell->id()]][sendProc][i] = recvdData [ offset + i]  ;
		  	offset += gradRhoRecvd[iSymm][globalCellId[cell->id()]][sendProc].size();
	           }
	        }
	    } 
	     recvdData.clear();
	    }

	}

	

  }

template class symmetryClass<1>;
template class symmetryClass<2>;
template class symmetryClass<3>;
template class symmetryClass<4>;
template class symmetryClass<5>;
template class symmetryClass<6>;
template class symmetryClass<7>;
template class symmetryClass<8>;
template class symmetryClass<9>;
template class symmetryClass<10>;
template class symmetryClass<11>;
template class symmetryClass<12>;
