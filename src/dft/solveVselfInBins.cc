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
// @author Shiva Rudraraju, Phani Motamarri, Sambit Das
//

#include <dealiiLinearSolver.h>
#include <poissonSolverProblem.h>
#include <poissonSolverProblemCellMatrixMultiVector.h>

namespace dftfe
{
    template<unsigned int FEOrder>
    void vselfBinsManager<FEOrder>::solveVselfInBins
                                    (const dealii::MatrixFree<3,double> & matrix_free_data,
	                             const unsigned int offset,
			             const dealii::AffineConstraints<double> & hangingPeriodicConstraintMatrix,
				     const std::vector<std::vector<double> > & imagePositions,
				     const std::vector<int> & imageIds,
				     const std::vector<double> &imageCharges,
	                             std::vector<std::vector<double> > & localVselfs)
    {
      localVselfs.clear();
      d_vselfFieldBins.clear();
      //d_atomIdBinIdMapLocalAllImages.clear();
      //phiExt with nuclear charge
      //
      const unsigned int numberBins = d_boundaryFlagOnlyChargeId.size();
      const unsigned int numberGlobalCharges = d_atomLocations.size();

      //set up poisson solver
      dealiiLinearSolver dealiiCGSolver(mpi_communicator,dealiiLinearSolver::CG);
      poissonSolverProblemCellMatrixMultiVector<FEOrder> vselfSolverProblem(mpi_communicator);

      std::map<dealii::types::global_dof_index, dealii::Point<3> > supportPoints;
      dealii::DoFTools::map_dofs_to_support_points(dealii::MappingQ1<3,3>(), matrix_free_data.get_dof_handler(offset), supportPoints);

      std::map<dealii::types::global_dof_index, int>::iterator iterMap;
      std::map<dealii::types::global_dof_index, double>::iterator iterMapVal;
      d_vselfFieldBins.resize(numberBins);
      for(unsigned int iBin = 0; iBin < numberBins; ++iBin)
	{
          double init_time;
          MPI_Barrier(MPI_COMM_WORLD);
          init_time = MPI_Wtime();

	  const unsigned int constraintMatrixId = iBin + offset;
	  vectorType vselfBinScratch;
	  matrix_free_data.initialize_dof_vector(vselfBinScratch,0);
	  vselfBinScratch = 0;
	  
          std::map<dealii::types::global_dof_index,dealii::Point<3> >::iterator iterNodalCoorMap;
	  std::map<dealii::types::global_dof_index, double> & vSelfBinNodeMap = d_vselfBinField[iBin];

	  //
	  //set initial guess to vSelfBinScratch
	  //
          /* 
	  for(iterNodalCoorMap = supportPoints.begin(); iterNodalCoorMap != supportPoints.end(); ++iterNodalCoorMap)
	      if(vselfBinScratch.in_local_range(iterNodalCoorMap->first))
              {
	            if(!d_vselfBinConstraintMatrices[iBin].is_constrained(iterNodalCoorMap->first))
		    {
		      iterMapVal = vSelfBinNodeMap.find(iterNodalCoorMap->first);
		      if(iterMapVal != vSelfBinNodeMap.end())
			  vselfBinScratch(iterNodalCoorMap->first) = iterMapVal->second;
		    }
               }     
          */

	  //vselfBinScratch.compress(dealii::VectorOperation::insert);
	  //d_vselfBinConstraintMatrices[iBin].distribute(vselfBinScratch);
           
          MPI_Barrier(MPI_COMM_WORLD);
          init_time = MPI_Wtime() - init_time;
          if (dftParameters::verbosity>=1)
            pcout<<" Time taken for vself field initialization for current bin: "<<init_time<<std::endl;

          double vselfinit_time;
          MPI_Barrier(MPI_COMM_WORLD);
          vselfinit_time = MPI_Wtime();
	  //
	  //call the poisson solver to compute vSelf in current bin
	  //
	  vselfSolverProblem.reinit(matrix_free_data,
				    vselfBinScratch,
                                    hangingPeriodicConstraintMatrix,
				    d_vselfBinConstraintMatrices[iBin],
				    constraintMatrixId,
				    d_atomsInBin[iBin],
                                    &d_inhomoIdsColoredVecFlattened[0],
                                    numberBins,
                                    iBin,
                                    true,
                                    false);

          MPI_Barrier(MPI_COMM_WORLD);
          vselfinit_time = MPI_Wtime() - vselfinit_time;
          if (dftParameters::verbosity>=1)
            pcout<<" Time taken for vself solver problem init for current bin: "<<vselfinit_time<<std::endl;

	  dealiiCGSolver.solve(vselfSolverProblem,
			       dftParameters::absLinearSolverTolerance,
			       dftParameters::maxLinearSolverIterations,
			       dftParameters::verbosity);

          
	  //
	  //store Vselfs for atoms in bin
	  //
	  for(std::map<dealii::types::global_dof_index, double>::iterator it = d_atomsInBin[iBin].begin(); it != d_atomsInBin[iBin].end(); ++it)
	    {
	      std::vector<double> temp(2,0.0);
	      temp[0] = it->second;//charge;
	      temp[1] = vselfBinScratch(it->first);//vself
	      if (dftParameters::verbosity>=4)
		  std::cout<< "(only for debugging: peak value of Vself: "<< temp[1] << ")" <<std::endl;

	      localVselfs.push_back(temp);
	    }
	    //
	    //store solved vselfBinScratch field
	    //
	    d_vselfFieldBins[iBin]=vselfBinScratch;
	}//bin loop
    }
}
