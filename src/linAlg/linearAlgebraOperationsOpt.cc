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
// @author Phani Motamarri (2018)
//


/** @file linearAlgebraOperationsOpt.cc
 *  @brief Contains linear algebra operations
 *
 */

#include <linearAlgebraOperations.h>
#include <dftParameters.h>


namespace dftfe{

  namespace linearAlgebraOperations
  {
    //
    //chebyshev filtering of given subspace XArray
    //
    template<typename T>
    void chebyshevFilter(operatorDFTClass & operatorMatrix,
			 dealii::parallel::distributed::Vector<T> & XArray,
			 const unsigned int numberWaveFunctions,
			 const std::vector<std::vector<dealii::types::global_dof_index> > & flattenedArrayMacroCellLocalProcIndexIdMap,
			 const std::vector<std::vector<dealii::types::global_dof_index> > & flattenedArrayCellLocalProcIndexIdMap,
			 const unsigned int m,
			 const double a,
			 const double b,
			 const double a0)
    {

      double e, c, sigma, sigma1, sigma2, gamma;
      e = (b-a)/2.0; c = (b+a)/2.0;
      sigma = e/(a0-c); sigma1 = sigma; gamma = 2.0/sigma1;

      dealii::parallel::distributed::Vector<T> YArray;//,YNewArray;

      //
      //create YArray
      //
      YArray.reinit(XArray);


      //
      //initialize to zeros.
      //
      const T zeroValue = 0.0;
      YArray = zeroValue;


      //
      //call HX
      //
      bool scaleFlag = false;
      T scalar = 1.0;
      operatorMatrix.HX(XArray,
			numberWaveFunctions,
			flattenedArrayMacroCellLocalProcIndexIdMap,
			flattenedArrayCellLocalProcIndexIdMap,
			scaleFlag,
			scalar,
			YArray);


      T alpha1 = sigma1/e, alpha2 = -c;

      //
      //YArray = YArray + alpha2*XArray and YArray = alpha1*YArray
      //
      YArray.add(alpha2,XArray);
      YArray *= alpha1;

      //
      //polynomial loop
      //
      for(unsigned int degree = 2; degree < m+1; ++degree)
	{
	  sigma2 = 1.0/(gamma - sigma);
	  alpha1 = 2.0*sigma2/e, alpha2 = -(sigma*sigma2);

	  //
	  //multiply XArray with alpha2
	  //
	  XArray *= alpha2;
	  XArray.add(-c*alpha1,YArray);


	  //
	  //call HX
	  //
	  bool scaleFlag = true;
	  operatorMatrix.HX(YArray,
			    numberWaveFunctions,
			    flattenedArrayMacroCellLocalProcIndexIdMap,
			    flattenedArrayCellLocalProcIndexIdMap,
			    scaleFlag,
			    alpha1,
			    XArray);

	  //
	  //XArray = YArray
	  //
	  XArray.swap(YArray);

	  //
	  //YArray = YNewArray
	  //
	  sigma = sigma2;

	}
      
      //copy back YArray to XArray
      XArray = YArray;
    }

    template<typename T>
    void gramSchmidtOrthogonalization(operatorDFTClass & operatorMatrix,
				      dealii::parallel::distributed::Vector<T> & X,
				      const unsigned int numberVectors)
    {

      const unsigned int localVectorSize = X.local_size()/numberVectors;

      //
      //Create template PETSc vector to create BV object later
      //
      Vec templateVec;
      VecCreateMPI(operatorMatrix.getMPICommunicator(),
		   localVectorSize,
		   PETSC_DETERMINE,
		   &templateVec);
      VecSetFromOptions(templateVec);
		   
      
      //
      //Set BV options after creating BV object
      //
      BV columnSpaceOfVectors;
      BVCreate(operatorMatrix.getMPICommunicator(),&columnSpaceOfVectors);
      BVSetSizesFromVec(columnSpaceOfVectors,
			templateVec,
			numberVectors);
      BVSetFromOptions(columnSpaceOfVectors);


      //
      //create list of indices
      //
      std::vector<PetscInt> indices(localVectorSize);
      std::vector<PetscScalar> data(localVectorSize);

      PetscInt low,high;

      VecGetOwnershipRange(templateVec,
			   &low,
			   &high);
			   

      for(PetscInt index = 0;index < localVectorSize; ++index)
	indices[index] = low+index;

      //
      //Fill in data into BV object
      //
      Vec v;
      for(unsigned int iColumn = 0; iColumn < numberVectors; ++iColumn)
	{
	  BVGetColumn(columnSpaceOfVectors,
		      iColumn,
		      &v);
	  VecSet(v,0.0);
	  for(unsigned int iNode = 0; iNode < localVectorSize; ++iNode)
	    data[iNode] = X.local_element(numberVectors*iNode + iColumn);

	  VecSetValues(v,
		       localVectorSize,
		       &indices[0],
		       &data[0],
		       INSERT_VALUES);

	  VecAssemblyBegin(v);
	  VecAssemblyEnd(v);

	  BVRestoreColumn(columnSpaceOfVectors,
			  iColumn,
			  &v);
	}
      
      //
      //orthogonalize
      //
      BVOrthogonalize(columnSpaceOfVectors,NULL);
      
      //
      //Copy data back into X
      //
      Vec v1;
      PetscScalar * pointerv1;
      for(unsigned int iColumn = 0; iColumn < numberVectors; ++iColumn)
	{
	  BVGetColumn(columnSpaceOfVectors,
		      iColumn,
		      &v1);

	  VecGetArray(v1,
		      &pointerv1);

	  for(unsigned int iNode = 0; iNode < localVectorSize; ++iNode)
	    X.local_element(numberVectors*iNode + iColumn) = pointerv1[iNode];

	  VecRestoreArray(v1,
			  &pointerv1);

	  BVRestoreColumn(columnSpaceOfVectors,
			  iColumn,
			  &v1);
	}

    }




#ifdef ENABLE_PERIODIC_BC
    template void chebyshevFilter(operatorDFTClass & operatorMatrix,
				  dealii::parallel::distributed::Vector<std::complex<double> > & ,
				  const unsigned int ,
				  const std::vector<std::vector<dealii::types::global_dof_index> > & ,
				  const std::vector<std::vector<dealii::types::global_dof_index> > & ,
				  const unsigned int,
				  const double ,
				  const double ,
				  const double );


    template void gramSchmidtOrthogonalization(operatorDFTClass & operatorMatrix,
					  dealii::parallel::distributed::Vector<std::complex<double> > &,
					  const unsigned int );


#else
    template void chebyshevFilter(operatorDFTClass & operatorMatrix,
				  dealii::parallel::distributed::Vector<double> & ,
				  const unsigned int ,
				  const std::vector<std::vector<dealii::types::global_dof_index> > & ,
  				  const std::vector<std::vector<dealii::types::global_dof_index> > & ,
				  const unsigned int,
				  const double ,
				  const double ,
				  const double );

    template void gramSchmidtOrthogonalization(operatorDFTClass & operatorMatrix,
					  dealii::parallel::distributed::Vector<double> &,
					  const unsigned int );
#endif
  
  }//end of namespace

}
