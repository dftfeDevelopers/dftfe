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

      dealii::parallel::distributed::Vector<T> YArray,YNewArray;

      //
      //create YArray
      //
      YArray.reinit(XArray);
      YNewArray.reinit(XArray);

      //
      //initialize to zeros.
      //
      const T zeroValue = 0.0;
      YArray = zeroValue;
      YNewArray = zeroValue;

      //
      //call HX
      //
      operatorMatrix.HX(XArray,
			numberWaveFunctions,
			flattenedArrayMacroCellLocalProcIndexIdMap,
			flattenedArrayCellLocalProcIndexIdMap,
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
	  //call HX
	  //
	  operatorMatrix.HX(YArray,
			    numberWaveFunctions,
			    flattenedArrayMacroCellLocalProcIndexIdMap,
			    flattenedArrayCellLocalProcIndexIdMap,
			    YNewArray);

	  //
	  //YNewArray = YNewArray - c*YArray and YNewArray = alpha1*YNewArray
	  //
	  YNewArray.add(-c, YArray);
	  YNewArray *= alpha1;
	
	  //
	  //YNewArray = YNewArray + alpha2*XArray
	  //
	  YNewArray.add(alpha2,XArray);

	  //
	  //XArray = YArray
	  //
	  XArray.swap(YArray);

	  //
	  //YArray = YNewArray
	  //
	  YArray.swap(YNewArray);

	  sigma = sigma2;

	}
      
      //copy back YArray to XArray
      XArray = YArray;
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
#endif
  
  }//end of namespace

}
