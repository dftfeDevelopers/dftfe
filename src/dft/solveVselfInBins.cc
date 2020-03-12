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
                    else
                      vselfBinScratch(iterNodalCoorMap->first) = 0.0;
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
                                    true,
                                    iBin==0?false:false);

          MPI_Barrier(MPI_COMM_WORLD);
          vselfinit_time = MPI_Wtime() - vselfinit_time;
          if (dftParameters::verbosity>=1)
            pcout<<" Time taken for vself solver problem init for current bin: "<<vselfinit_time<<std::endl;

	  dealiiCGSolver.solve(vselfSolverProblem,
			       dftParameters::absLinearSolverTolerance,
			       dftParameters::maxLinearSolverIterations,
			       dftParameters::verbosity);

          /*
          double sumvself_time;
          MPI_Barrier(MPI_COMM_WORLD);
          sumvself_time = MPI_Wtime();
          
	  std::set<int> & atomsInBinSet = d_bins[iBin];
	  std::vector<int> atomsInCurrentBin(atomsInBinSet.begin(),atomsInBinSet.end());
	  const unsigned int numberGlobalAtomsInBin = atomsInCurrentBin.size();

	  std::vector<int> imageIdsOfAtomsInCurrentBin;
	  std::vector<int> imageChargeIdsOfAtomsInCurrentBin;
	  for(int index = 0; index < numberGlobalAtomsInBin; ++index)
	    {
	      int globalChargeIdInCurrentBin = atomsInCurrentBin[index];
	      for(int iImageAtom = 0; iImageAtom < imageIds.size(); ++iImageAtom)
		  if(imageIds[iImageAtom] == globalChargeIdInCurrentBin)
		  {
		      imageIdsOfAtomsInCurrentBin.push_back(iImageAtom);
		      imageChargeIdsOfAtomsInCurrentBin.push_back(imageIds[iImageAtom]);
		  }
	    }

	  const unsigned int numberImageAtomsInBin = imageIdsOfAtomsInCurrentBin.size();

	  std::map<dealii::types::global_dof_index, int> & boundaryNodeMap = d_boundaryFlagOnlyChargeId[iBin];
          std::map<dealii::types::global_dof_index, dealii::Point<3>> & dofClosestChargeLocationMap
		                                            = d_dofClosestChargeLocationMap[iBin];

        
	  int inNodes =0, outNodes = 0;
	  for(iterNodalCoorMap = supportPoints.begin(); iterNodalCoorMap != supportPoints.end(); ++iterNodalCoorMap)
	      if(vselfBinScratch.in_local_range(iterNodalCoorMap->first)
		  && !phiExtConstraintMatrix.is_constrained(iterNodalCoorMap->first))
		    {
		      //
		      //get the vertex Id
		      //
		      dealii::Point<3> nodalCoor = iterNodalCoorMap->second;

		      //
		      //get the boundary flag for iVertex for current bin
		      //
		      int boundaryFlag;
		      iterMap = boundaryNodeMap.find(iterNodalCoorMap->first);
		      if(iterMap != boundaryNodeMap.end())
			{
			  boundaryFlag = iterMap->second;
			}
		      else
			{
			  std::cout<<"Could not find boundaryNode Map for the given dof:"<<std::endl;
			  exit(-1);
			}

		      //
		      //go through all atoms in a given bin
		      //
		      double vSelf=0.0;
		      const dealii::Point<3> & dofClosestChargeLocation=dofClosestChargeLocationMap[iterNodalCoorMap->first];
		      for(int iCharge = 0; iCharge < numberGlobalAtomsInBin+numberImageAtomsInBin; ++iCharge)
			{
			  //
			  //get the globalChargeId corresponding to iCharge in the current bin
			  //and add numberGlobalCharges to image atomId
			  int chargeId;
			  unsigned int atomId;
			  dealii::Point<3> atomCoor(0.0,0.0,0.0);
			  if(iCharge < numberGlobalAtomsInBin)
			  {
			    chargeId = atomsInCurrentBin[iCharge];
			    atomId=chargeId;
			    atomCoor[0] = d_atomLocations[chargeId][2];
		            atomCoor[1] = d_atomLocations[chargeId][3];
			    atomCoor[2] = d_atomLocations[chargeId][4];
			  }
			  else
			  {
			    chargeId = imageChargeIdsOfAtomsInCurrentBin[iCharge-numberGlobalAtomsInBin];
			    atomId=imageIdsOfAtomsInCurrentBin[iCharge-numberGlobalAtomsInBin]
				   +numberGlobalCharges;
			    atomCoor[0]
				= imagePositions[imageIdsOfAtomsInCurrentBin[iCharge-numberGlobalAtomsInBin]][0];
			    atomCoor[1]
				= imagePositions[imageIdsOfAtomsInCurrentBin[iCharge-numberGlobalAtomsInBin]][1];
			    atomCoor[2]
				= imagePositions[imageIdsOfAtomsInCurrentBin[iCharge-numberGlobalAtomsInBin]][2];
			  }

			  if(boundaryFlag == chargeId
			    && dofClosestChargeLocation.distance(atomCoor)<1e-5)
			    {
			      vSelf += vselfBinScratch(iterNodalCoorMap->first);
			      inNodes++;
			      d_atomIdBinIdMapLocalAllImages[atomId]=iBin;
			    }
			  else
			    {
			      double nuclearCharge;
			      if(iCharge < numberGlobalAtomsInBin)
				{
				  if(dftParameters::isPseudopotential)
				    nuclearCharge = d_atomLocations[chargeId][1];
				  else
				    nuclearCharge = d_atomLocations[chargeId][0];

				}
			      else
				{
				  nuclearCharge =
				      imageCharges[imageIdsOfAtomsInCurrentBin[iCharge-numberGlobalAtomsInBin]];
				}

			      const double r = nodalCoor.distance(atomCoor);
			      vSelf += -nuclearCharge/r;
			      outNodes++;
			    }
			}//charge loop
		        //store updated value in phiExt which is sumVself
		        phiExt(iterNodalCoorMap->first)+= vSelf;
		    }
 
          MPI_Barrier(MPI_COMM_WORLD);
          sumvself_time = MPI_Wtime() - sumvself_time;
          if (dftParameters::verbosity>=1)
            pcout<<" Time taken for sumvself for current bin: "<<sumvself_time<<std::endl;
          */
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

      //phiExt.compress(dealii::VectorOperation::insert);

      //FIXME: Should we use periodic constraints to distribute phiExt?
      //phiExtConstraintMatrix.distribute(phiExt);
      //phiExt.update_ghost_values();

      //print the norms of phiExt (in periodic case L2 norm of phiExt field does not match. check later)
      //if (dftParameters::verbosity>=4)
      //  pcout<<"L2 Norm Value of phiext: "<<phiExt.l2_norm()<<std::endl;
    }
}
