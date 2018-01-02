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
//
// @author Shiva Rudraraju (2016), Phani Motamarri (2016), Sambit Das (2017)
//

#include "initRho.cc"
#include "initPseudo.cc"

#ifdef ENABLE_PERIODIC_BC
#include "initkPointData.cc"
#endif

//
//source file for dft class initializations
//
#ifdef ENABLE_PERIODIC_BC
void exchangeMasterNodesList(std::set<unsigned int> & masterNodeIdSet,
			     std::set<unsigned int> & globalMasterNodeIdSet,
			     unsigned int numMeshPartitions,
			     const MPI_Comm & mpi_communicator)
{

  std::vector<int> localMasterNodeIdList;
  std::copy(masterNodeIdSet.begin(),
	    masterNodeIdSet.end(),
	    std::back_inserter(localMasterNodeIdList));

  int numberMasterNodesOnLocalProc = localMasterNodeIdList.size();

  int * masterNodeIdListSizes = new int[numMeshPartitions];

  MPI_Allgather(&numberMasterNodesOnLocalProc,
		1,
		MPI_INT,
		masterNodeIdListSizes,
		1,
		MPI_INT,
		mpi_communicator);

  int newMasterNodeIdListSize = std::accumulate(&(masterNodeIdListSizes[0]),
						&(masterNodeIdListSizes[numMeshPartitions]),
						0);

  std::vector<int> globalMasterNodeIdList(newMasterNodeIdListSize);

  int * mpiOffsets = new int[numMeshPartitions];

  mpiOffsets[0] = 0;

  for(int i = 1; i < numMeshPartitions; ++i)
    mpiOffsets[i] = masterNodeIdListSizes[i-1] + mpiOffsets[i-1];

  MPI_Allgatherv(&(localMasterNodeIdList[0]),
		 numberMasterNodesOnLocalProc,
		 MPI_INT,
		 &(globalMasterNodeIdList[0]),
		 &(masterNodeIdListSizes[0]),
		 &(mpiOffsets[0]),
		 MPI_INT,
		 mpi_communicator);


  for(int i = 0; i < globalMasterNodeIdList.size(); ++i)
    globalMasterNodeIdSet.insert(globalMasterNodeIdList[i]);

  delete [] masterNodeIdListSizes;
  delete [] mpiOffsets;
	    
  return;

}

void exchangeMasterNodesList(std::vector<unsigned int> & masterNodeIdList,
			     std::vector<unsigned int> & globalMasterNodeIdList,
			     unsigned int numMeshPartitions,
			     const MPI_Comm & mpi_communicator)
{

  int numberMasterNodesOnLocalProc = masterNodeIdList.size();

  int * masterNodeIdListSizes = new int[numMeshPartitions];

  MPI_Allgather(&numberMasterNodesOnLocalProc,
		1,
		MPI_INT,
		masterNodeIdListSizes,
		1,
		MPI_INT,
		mpi_communicator);

  int newMasterNodeIdListSize = std::accumulate(&(masterNodeIdListSizes[0]),
						&(masterNodeIdListSizes[numMeshPartitions]),
						0);

  globalMasterNodeIdList.resize(newMasterNodeIdListSize);

  int * mpiOffsets = new int[numMeshPartitions];

  mpiOffsets[0] = 0;

  for(int i = 1; i < numMeshPartitions; ++i)
    mpiOffsets[i] = masterNodeIdListSizes[i-1] + mpiOffsets[i-1];

  MPI_Allgatherv(&(masterNodeIdList[0]),
		 numberMasterNodesOnLocalProc,
		 MPI_INT,
		 &(globalMasterNodeIdList[0]),
		 &(masterNodeIdListSizes[0]),
		 &(mpiOffsets[0]),
		 MPI_INT,
		 mpi_communicator);


  delete [] masterNodeIdListSizes;
  delete [] mpiOffsets;
	    
  return;
}
#endif


//init
template<unsigned int FEOrder>
void dftClass<FEOrder>::initUnmovedTriangulation(){
  computing_timer.enter_section("unmoved setup");
  double domainSizeX = dftParameters::domainSizeX,domainSizeY = dftParameters::domainSizeY,domainSizeZ=dftParameters::domainSizeZ;

  //
  //initialize FE objects
  //
  dofHandler.distribute_dofs (FE);
  dofHandlerEigen.distribute_dofs (FEEigen);

  locally_owned_dofs = dofHandler.locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs(dofHandler, locally_relevant_dofs);
  DoFTools::map_dofs_to_support_points(MappingQ1<3,3>(), dofHandler, d_supportPoints);

  //selectedDofsHanging.resize(dofHandler.n_dofs(),false);
  //DoFTools::extract_hanging_node_dofs(dofHandler, selectedDofsHanging);

  locally_owned_dofsEigen = dofHandlerEigen.locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs(dofHandlerEigen, locally_relevant_dofsEigen);
  DoFTools::map_dofs_to_support_points(MappingQ1<3,3>(), dofHandlerEigen, d_supportPointsEigen);

  

  //
  //Extract real and imag DOF indices from the global vector - this will be needed in XHX operation, etc.
  //
#ifdef ENABLE_PERIODIC_BC
  FEValuesExtractors::Scalar real(0); //For Eigen
  ComponentMask componentMaskForRealDOF = FEEigen.component_mask (real);
  std::vector<bool> selectedDofsReal(locally_owned_dofsEigen.n_elements(), false);
  DoFTools::extract_dofs(dofHandlerEigen, componentMaskForRealDOF, selectedDofsReal);
  std::vector<unsigned int> local_dof_indices(locally_owned_dofsEigen.n_elements());
  locally_owned_dofsEigen.fill_index_vector(local_dof_indices);
  for (unsigned int i = 0; i < locally_owned_dofsEigen.n_elements(); i++)
    {
      if (selectedDofsReal[i]) 
	{
	  local_dof_indicesReal.push_back(local_dof_indices[i]);
	  localProc_dof_indicesReal.push_back(i);
	}
      else
	{
	  local_dof_indicesImag.push_back(local_dof_indices[i]);
	  localProc_dof_indicesImag.push_back(i);	  
	}
    }
#endif


  
  pcout << "number of elements: "
	<< triangulation.n_global_active_cells()
	<< std::endl
	<< "number of degrees of freedom: " 
	<< dofHandler.n_dofs() 
	<< std::endl;


  //
  //constraints
  //

  //
  //hanging node constraints
  //
  constraintsNone.clear(); constraintsNoneEigen.clear();
  //DoFTools::make_hanging_node_constraints(dofHandler, constraintsNone);
  //DoFTools::make_hanging_node_constraints(dofHandlerEigen, constraintsNoneEigen);


#ifdef ENABLE_PERIODIC_BC
  std::vector<GridTools::PeriodicFacePair<typename DoFHandler<3>::cell_iterator> > periodicity_vector2, periodicity_vector2Eigen;
  for (int i = 0; i < 3; ++i)
    {
      GridTools::collect_periodic_faces(dofHandler, /*b_id1*/ 2*i+1, /*b_id2*/ 2*i+2,/*direction*/ i, periodicity_vector2);
      GridTools::collect_periodic_faces(dofHandlerEigen, /*b_id1*/ 2*i+1, /*b_id2*/ 2*i+2,/*direction*/ i, periodicity_vector2Eigen);
    }
  DoFTools::make_periodicity_constraints<DoFHandler<3> >(periodicity_vector2, constraintsNone);
  DoFTools::make_periodicity_constraints<DoFHandler<3> >(periodicity_vector2Eigen, constraintsNoneEigen);

  constraintsNone.close();
  constraintsNoneEigen.close();

  //pcout << "Detected Periodic Face Pairs: " << constraintsNone.n_constraints() << std::endl;

  //pcout<<"Size of ConstraintsNone: "<< constraintsNone.n_constraints()<<std::endl;
  //constraintsNone.print(std::cout);

  //pcout<<"Size of ConstraintsNoneEigen: "<< constraintsNoneEigen.n_constraints()<<std::endl;
  //constraintsNoneEigen.print(std::cout);

  //
  //modify constraintsNone to account for the bug in higher order nodes
  //
  ConstraintMatrix constraintsTemp(constraintsNone); constraintsNone.clear(); 
  std::set<unsigned int> masterNodes;
  double periodicPrecision = 1.0e-5;

  //
  //fill all masters
  //
  for(types::global_dof_index i = 0; i <dofHandler.n_dofs(); ++i)
    {
      if(locally_relevant_dofs.is_element(i))
	{
	  if(constraintsTemp.is_constrained(i))
	    {
	      if (constraintsTemp.is_identity_constrained(i))
		{
		  unsigned int masterNode=(*constraintsTemp.get_constraint_entries(i))[0].first;
		  masterNodes.insert(masterNode);
		}
	    }
	}
    }

  
  

  //
  //fix wrong master map
  //
  for(types::global_dof_index i = 0; i <dofHandler.n_dofs(); ++i)
    {
      if(locally_relevant_dofs.is_element(i))
	{
	  if(constraintsTemp.is_constrained(i))
	    {
	      if(constraintsTemp.is_identity_constrained(i))
		{
		  Point<3> p = d_supportPoints.find(i)->second;
		  unsigned int masterNode = (*constraintsTemp.get_constraint_entries(i))[0].first;
		  unsigned int count = 0, index = 0;
		 
		  if(std::abs(std::abs(p[0])-(domainSizeX/2.0)) < periodicPrecision) 
		    {
		      count++;
		      index = 0;
		    }
		  if(std::abs(std::abs(p[1])-(domainSizeY/2.0)) < periodicPrecision) 
		    {
		      count++;
		      index = 1;
		    }
		  if(std::abs(std::abs(p[2])-(domainSizeZ/2.0)) < periodicPrecision) 
		    {
		      count++;
		      index = 2;
		    }
		 
		  if (count==1)
		    {
		      Point<3> q = d_supportPoints.find(masterNode)->second;
		      unsigned int l = 1, m = 2;
		      if (index==1){l = 0; m = 2;}
		      else if (index==2){l = 0; m = 1;} 
		      if (!((std::abs(p[l]-q[l])<periodicPrecision) and (std::abs(p[m]-q[m])<periodicPrecision)))
			{
			  bool foundNewMaster=false;
			  for (std::set<unsigned int>::iterator it = masterNodes.begin(); it != masterNodes.end(); ++it)
			    {
			      q = d_supportPoints.find(*it)->second;
			      if (((std::abs(p[l]-q[l])<periodicPrecision) and (std::abs(p[m]-q[m])<periodicPrecision)))
				{

				  //store the correct masterNodeId
				  unsigned int correctMasterDof = *it;

				  //One component
				  constraintsNone.add_line(i);
				  constraintsNone.add_entry(i, correctMasterDof, 1.0);
		  
				  foundNewMaster=true;
				  break;
				}
			    }
			  if (!foundNewMaster)
			    {
			      std::cout<< " Wrong MasterNode for slave node: "<<masterNode<<" "<<i<<std::endl;
			      std::cout<< "Error: Did not find a replacement master node for a wrong master-slave periodic pair"<<" "<<masterNode<<" slaveNode: "<<i<<std::endl;
			      exit(-1);
			    }
			}
		      else
			{

			  //One component
			  constraintsNone.add_line(i);
			  constraintsNone.add_entry(i, masterNode, 1.0);
	      
			}
		    }
		  else
		    {

		      //One component
		      constraintsNone.add_line(i);
		      constraintsNone.add_entry(i, masterNode, 1.0);

		    }
		}
	    }
	}
    }
  constraintsNone.close();
  constraintsTemp.clear();

  //
  //modify constraintsNone to account for the bug in higher order nodes for edge nodes
  //
  ConstraintMatrix constraintsTemp1(constraintsNone); constraintsNone.clear(); 
  std::vector<unsigned int> masterNodesForLocallyOwnedSlaveNodes; 

  for(types::global_dof_index i = 0; i < dofHandler.n_dofs(); ++i)
    {
      if(locally_owned_dofs.is_element(i))
	{
	  if(constraintsTemp1.is_constrained(i))
	    {
	      if (constraintsTemp1.is_identity_constrained(i))
		{
		  unsigned int masterNode = (*constraintsTemp1.get_constraint_entries(i))[0].first;
		  masterNodesForLocallyOwnedSlaveNodes.push_back(masterNode);
		}
	    }
	}
    }

  //
  //exchange all masterNodes across all processors
  //
  std::vector<unsigned int> globalMasterNodes;
  exchangeMasterNodesList(masterNodesForLocallyOwnedSlaveNodes,
			  globalMasterNodes,
			  n_mpi_processes,
			  mpi_communicator);

  //
  //find out masternodes which are on the edge
  //
  std::map<unsigned int, unsigned int> rv;
  for(int i = 0; i < globalMasterNodes.size(); ++i)
    {
      rv[globalMasterNodes[i]]++;
    }


  std::set<unsigned int> masterNodesForLocallyRelevantEdgeSlaveNodes;

  for(types::global_dof_index i = 0; i < dofHandler.n_dofs(); ++i)
    {
      if(locally_relevant_dofs.is_element(i))
	{
	  if(constraintsTemp1.is_constrained(i))
	    {
	      if (constraintsTemp1.is_identity_constrained(i))
		{
		  unsigned int masterNode = (*constraintsTemp1.get_constraint_entries(i))[0].first;
		  if(rv[masterNode] == 3)
		    {
		      masterNodesForLocallyRelevantEdgeSlaveNodes.insert(masterNode);
		    }
		}
	    }
	}
    }

  
  for(types::global_dof_index i = 0; i <dofHandler.n_dofs(); ++i)
    {
      if(locally_relevant_dofs.is_element(i))
	{
	  if(constraintsTemp1.is_constrained(i))
	    {
	      if (constraintsTemp1.is_identity_constrained(i))
		{
		  unsigned int masterNode = (*constraintsTemp1.get_constraint_entries(i))[0].first;
		  Point<3> p = d_supportPoints.find(i)->second;

		  if(masterNodesForLocallyRelevantEdgeSlaveNodes.count(masterNode))
		    {
		      Point<3> q = d_supportPoints.find(masterNode)->second;
		     
		      //
		      //Identify the normal direction to the plane formed by the periodic edge nodes
		      //
		      int normalDirOfPlaneFormedByEdgeNodes = 0;
		      if((std::abs(std::abs(q[0]) - (domainSizeX/2.0)) < periodicPrecision) && (std::abs(std::abs(q[2]) - (domainSizeZ/2.0)) < periodicPrecision))
			normalDirOfPlaneFormedByEdgeNodes = 1;
		      else if ((std::abs(std::abs(q[0]) - (domainSizeX/2.0)) < periodicPrecision) && (std::abs(std::abs(q[1]) - (domainSizeY/2.0)) < periodicPrecision))
			normalDirOfPlaneFormedByEdgeNodes = 2;

		     
		      //
		      //check if master node and corresponding edge slave node lie in the same plane
		      //
		      if(std::abs(p[normalDirOfPlaneFormedByEdgeNodes] - q[normalDirOfPlaneFormedByEdgeNodes]) < periodicPrecision)
			{
			  constraintsNone.add_line(i);
			  constraintsNone.add_entry(i, masterNode, 1.0);
			}
		      else
			{
			  bool foundNewMaster = false;

			  //
			  //match the master node lying in the same plane to this slave node
			  //
			  for(std::set<unsigned int>::iterator it = masterNodesForLocallyRelevantEdgeSlaveNodes.begin(); it != masterNodesForLocallyRelevantEdgeSlaveNodes.end(); ++it)
			    {
			      Point<3> r = d_supportPoints.find(*it)->second;
			      
			      if(std::abs(r[normalDirOfPlaneFormedByEdgeNodes] - p[normalDirOfPlaneFormedByEdgeNodes]) < periodicPrecision)
				{
				  unsigned int correctMasterDof = *it;
				  constraintsNone.add_line(i);
				  constraintsNone.add_entry(i, correctMasterDof, 1.0);
				  foundNewMaster = true;
				  break;
				}

			    }

			  if (!foundNewMaster)
			    {
			      std::cout<< " Wrong MasterNode for slave node: "<<masterNode<<" "<<i<<std::endl;
			      std::cout<< "Error: Did not find a replacement master node for a wrong master-slave periodic pair"<<" "<<masterNode<<" slaveNode: "<<i<<std::endl;
			      exit(-1);
			    }
			}
		    }
		  else
		    {
		      constraintsNone.add_line(i);
		      constraintsNone.add_entry(i, masterNode, 1.0);
		    }
		}
	    }
	}
    }
  constraintsNone.close();
  constraintsTemp1.clear();

  //
  //fix periodic match for two-component fields
  //
  ConstraintMatrix constraintsTempEigen(constraintsNoneEigen); constraintsNoneEigen.clear();

  //
  //fill temp masterNodes and slaveNodes set
  //
  std::set<unsigned int> masterNodesEigen, slaveNodesEigen;

  //
  //fill all masters
  //
  for(types::global_dof_index i = 0; i <dofHandlerEigen.n_dofs(); ++i)
    {
      if(locally_owned_dofsEigen.is_element(i))
	{
	  if(constraintsTempEigen.is_constrained(i))
	    {
	      if (constraintsTempEigen.is_identity_constrained(i))
		{
		  unsigned int masterNode = (*constraintsTempEigen.get_constraint_entries(i))[0].first;
		  masterNodesEigen.insert(masterNode);
		  slaveNodesEigen.insert(i);
		}
	    }
	}
    }

  std::set<unsigned int> globalMasterNodesEigen;
  exchangeMasterNodesList(masterNodesEigen,
			  globalMasterNodesEigen,
			  n_mpi_processes,
			  mpi_communicator);

  
  //
  //Now separate this set to real and imag sets
  //
  QGauss<3>  quadrature_formula(C_num1DQuad<FEOrder>());
  FEValues<3> fe_values (FEEigen, quadrature_formula, update_values);
  const unsigned int dofs_per_cell = FEEigen.dofs_per_cell;
  std::vector<unsigned int> local_dof_indicesEigen(dofs_per_cell);

  std::set<unsigned int> masterNodesReal, masterNodesImag, slaveNodesReal, slaveNodesImag;
  typename DoFHandler<3>::active_cell_iterator cellEigen = dofHandlerEigen.begin_active(), endcellEigen = dofHandlerEigen.end();
  for(;cellEigen!=endcellEigen;++cellEigen)
    {
      if(cellEigen->is_locally_owned())
	{
	  fe_values.reinit(cellEigen);
	  cellEigen->get_dof_indices(local_dof_indicesEigen);

	  for(unsigned int i = 0; i < dofs_per_cell; ++i)
	    {
	      const unsigned int ck = fe_values.get_fe().system_to_component_index(i).first; //This is the component index 0(real) or 1 (imag).
	      const unsigned int globalDOF = local_dof_indicesEigen[i];
	      
	      if(globalMasterNodesEigen.count(globalDOF) == 1)
		{
		  if(ck == 0)
		    {
		      masterNodesReal.insert(globalDOF);
		    }
		  else 
		    {
		      masterNodesImag.insert(globalDOF);
		    }

		}
	      else if(slaveNodesEigen.count(globalDOF) == 1)
		{

		  if(ck == 0)
		    {
		      slaveNodesReal.insert(globalDOF);
		    }
		  else 
		    {
		      slaveNodesImag.insert(globalDOF);
		    }

		}

	    }
	}

    }//end of cellEigen loop

  std::set<unsigned int> globalMasterNodesReal;
  std::set<unsigned int> globalMasterNodesImag;
  std::set<unsigned int> globalSlaveNodesReal;
  std::set<unsigned int> globalSlaveNodesImag;

  exchangeMasterNodesList(masterNodesReal,
			  globalMasterNodesReal,
			  n_mpi_processes,
			  mpi_communicator);

  exchangeMasterNodesList(masterNodesImag,
			  globalMasterNodesImag,
			  n_mpi_processes,
			  mpi_communicator);

  exchangeMasterNodesList(slaveNodesReal,
			  globalSlaveNodesReal,
			  n_mpi_processes,
			  mpi_communicator);

  exchangeMasterNodesList(slaveNodesImag,
			  globalSlaveNodesImag,
			  n_mpi_processes,
			  mpi_communicator);

  //
  //fix wrong master map
  //
  for(types::global_dof_index i = 0; i <dofHandlerEigen.n_dofs(); i++)
    {
      if(locally_relevant_dofsEigen.is_element(i))
	{
	  if(constraintsTempEigen.is_constrained(i))
	    {
	      if(constraintsTempEigen.is_identity_constrained(i))
		{
		  Point<3> p = d_supportPointsEigen.find(i)->second;
		  unsigned int masterNode=(*constraintsTempEigen.get_constraint_entries(i))[0].first;
		  unsigned int count = 0, index = 0;

		  if(std::abs(std::abs(p[0]) - (domainSizeX/2.0)) < periodicPrecision) 
		    {
		      count++;
		      index = 0;
		    }

		  if(std::abs(std::abs(p[1]) - (domainSizeY/2.0)) < periodicPrecision) 
		    {
		      count++;
		      index = 1;
		    }

		  if(std::abs(std::abs(p[2]) - (domainSizeZ/2.0)) < periodicPrecision) 
		    {
		      count++;
		      index = 2;
		    }

		  if(count == 1)
		    {
		      Point<3> q = d_supportPointsEigen.find(masterNode)->second;

		      unsigned int l = 1, m = 2;
		      if (index == 1){l = 0; m = 2;}
		      else if (index == 2){l = 0; m = 1;} 

		      if(!((std::abs(p[l]-q[l])<periodicPrecision) and (std::abs(p[m]-q[m])<periodicPrecision)))
			{
			  bool foundNewMaster=false;
			  if(globalSlaveNodesReal.count(i) == 1)
			    {
			      for(std::set<unsigned int>::iterator it = globalMasterNodesReal.begin(); it != globalMasterNodesReal.end(); ++it)
				{
				  q = d_supportPointsEigen.find(*it)->second;
				  if(((std::abs(p[l]-q[l]) < periodicPrecision) and (std::abs(p[m]-q[m]) < periodicPrecision)))
				    {
				      constraintsNoneEigen.add_line(i);
				      constraintsNoneEigen.add_entry(i, *it, 1.0);
				      foundNewMaster=true;
				      break;
				    }
				}
			    }
			  else
			    {
			      for(std::set<unsigned int>::iterator it = globalMasterNodesImag.begin(); it != globalMasterNodesImag.end(); ++it)
				{
				  q = d_supportPointsEigen.find(*it)->second;
				  if(((std::abs(p[l]-q[l]) < periodicPrecision) and (std::abs(p[m]-q[m]) < periodicPrecision)))
				    {
				      constraintsNoneEigen.add_line(i);
				      constraintsNoneEigen.add_entry(i, *it, 1.0);
				      foundNewMaster=true;
				      break;
				    }
				}
			    }

			  if(!foundNewMaster)
			    {
			      pcout << "\nError: Did not find a replacement master node for a wrong master-slave periodic pair\n";
			      exit(-1);
			    }
			}
		      else
			{
			  constraintsNoneEigen.add_line(i);
			  constraintsNoneEigen.add_entry(i, masterNode, 1.0);
			}
		    }
		  else
		    {
		      constraintsNoneEigen.add_line(i);
		      constraintsNoneEigen.add_entry(i, masterNode, 1.0);
		    }
		}
	    }
	}
    }
   
  constraintsNoneEigen.close();
  constraintsTempEigen.clear();

  //
  //modify constraintsNoneEigen to account for bug in higher order nodes for edge nodes in two-field case
  //
  ConstraintMatrix constraintsTemp1Eigen(constraintsNoneEigen); constraintsNoneEigen.clear(); 
  std::vector<unsigned int> masterNodesForLocallyOwnedSlaveNodesEigen; 

  for(types::global_dof_index i = 0; i < dofHandlerEigen.n_dofs(); ++i)
    {
      if(locally_owned_dofsEigen.is_element(i))
	{
	  if(constraintsTemp1Eigen.is_constrained(i))
	    {
	      if(constraintsTemp1Eigen.is_identity_constrained(i))
		{
		  unsigned int masterNode = (*constraintsTemp1Eigen.get_constraint_entries(i))[0].first;
		  masterNodesForLocallyOwnedSlaveNodesEigen.push_back(masterNode);
		}
	    }
	}
    }

  //
  //exchange all masterNodes across all processors
  //
  std::vector<unsigned int> globalMasterNodesVectorEigen;
  exchangeMasterNodesList(masterNodesForLocallyOwnedSlaveNodesEigen,
			  globalMasterNodesVectorEigen,
			  n_mpi_processes,
			  mpi_communicator);


  //
  //count the duplicates of each master nodeId
  //
  std::map<unsigned int, unsigned int> rvEigen;
  for(int i = 0; i < globalMasterNodesVectorEigen.size(); ++i)
    {
      rvEigen[globalMasterNodesVectorEigen[i]]++;
    }


  //
  //fill temp masterNodes and slaveNodes set
  //
  std::set<unsigned int> masterNodesEigenNew, slaveNodesEigenNew;

  //
  //fill all masters
  //
  for(types::global_dof_index i = 0; i <dofHandlerEigen.n_dofs(); ++i)
    {
      if(locally_owned_dofsEigen.is_element(i))
	{
	  if(constraintsTemp1Eigen.is_constrained(i))
	    {
	      if (constraintsTemp1Eigen.is_identity_constrained(i))
		{
		  unsigned int masterNode=(*constraintsTemp1Eigen.get_constraint_entries(i))[0].first;
		  masterNodesEigenNew.insert(masterNode);
		  slaveNodesEigenNew.insert(i);
		}
	    }
	}
    }

  std::set<unsigned int> globalMasterNodesEigenNew;
  exchangeMasterNodesList(masterNodesEigenNew,
			  globalMasterNodesEigenNew,
			  n_mpi_processes,
			  mpi_communicator);


  std::set<unsigned int> masterNodesEdgeReal, masterNodesEdgeImag, slaveNodesNewReal, slaveNodesNewImag;
  QGauss<3>  quadrature_formulaNew(C_num1DQuad<FEOrder>());
  FEValues<3> fe_valuesNew(FEEigen, quadrature_formulaNew, update_values);
  typename DoFHandler<3>::active_cell_iterator cellEigenNew = dofHandlerEigen.begin_active(), endcellEigenNew = dofHandlerEigen.end();
  for(; cellEigenNew!=endcellEigenNew; ++cellEigenNew)
    {
      if(cellEigenNew->is_locally_owned())
	{
	  fe_valuesNew.reinit(cellEigenNew);
	  cellEigenNew->get_dof_indices(local_dof_indicesEigen);
	  for(unsigned int i = 0; i < dofs_per_cell; ++i)
	    {
	      const unsigned int ck = fe_valuesNew.get_fe().system_to_component_index(i).first; //This is the component index 0(real) or 1 (imag).
	      const unsigned int globalDOF = local_dof_indicesEigen[i];
	      
	      if(globalMasterNodesEigenNew.count(globalDOF) == 1)
		{
		  if(rvEigen[globalDOF] == 3)
		    {
		      if(ck == 0)
			{
			  masterNodesEdgeReal.insert(globalDOF);
			}
		      else 
			{
			  masterNodesEdgeImag.insert(globalDOF);
			}
		    }

		}
	      else if(slaveNodesEigenNew.count(globalDOF) == 1)
		{
		  if(ck == 0)
		    {
		      slaveNodesNewReal.insert(globalDOF);
		    }
		  else 
		    {
		      slaveNodesNewImag.insert(globalDOF);
		    }

		}

	    }
	}
    }//end of cellEigen loop

  std::set<unsigned int> globalMasterNodesEdgeReal;
  std::set<unsigned int> globalMasterNodesEdgeImag;
  std::set<unsigned int> globalSlaveNodesRealNew;
  std::set<unsigned int> globalSlaveNodesImagNew;

  exchangeMasterNodesList(masterNodesEdgeReal,
			  globalMasterNodesEdgeReal,
			  n_mpi_processes,
			  mpi_communicator);

  exchangeMasterNodesList(masterNodesEdgeImag,
			  globalMasterNodesEdgeImag,
			  n_mpi_processes,
			  mpi_communicator);

  exchangeMasterNodesList(slaveNodesNewReal,
			  globalSlaveNodesRealNew,
			  n_mpi_processes,
			  mpi_communicator);

  exchangeMasterNodesList(slaveNodesNewImag,
			  globalSlaveNodesImagNew,
			  n_mpi_processes,
			  mpi_communicator);


  //
  //fix wrong master map for edge nodes
  //
  for(types::global_dof_index i = 0; i < dofHandlerEigen.n_dofs(); i++)
    {
      if(locally_relevant_dofsEigen.is_element(i))
	{
	  if(constraintsTemp1Eigen.is_constrained(i))
	    {
	      if (constraintsTemp1Eigen.is_identity_constrained(i))
		{
		  Point<3> p = d_supportPointsEigen.find(i)->second;
		  unsigned int masterNode=(*constraintsTemp1Eigen.get_constraint_entries(i))[0].first;

		  if(rvEigen[masterNode] == 3)
		    {
		      Point<3> q = d_supportPointsEigen.find(masterNode)->second;

		      //
		      //Identify the normal direction to the plane formed by the periodic edge nodes
		      //
		      int normalDirOfPlaneFormedByEdgeNodes = 0;
		      if((std::abs(std::abs(q[0]) - (domainSizeX/2.0)) < periodicPrecision) && (std::abs(std::abs(q[2]) - (domainSizeZ/2.0)) < periodicPrecision))
			normalDirOfPlaneFormedByEdgeNodes = 1;
		      else if ((std::abs(std::abs(q[0]) - (domainSizeX/2.0)) < periodicPrecision) && (std::abs(std::abs(q[1]) - (domainSizeY/2.0)) < periodicPrecision))
			normalDirOfPlaneFormedByEdgeNodes = 2;


		      //
		      //check if master node and corresponding edge slave node lie in the same plane
		      //
		      if(std::abs(p[normalDirOfPlaneFormedByEdgeNodes] - q[normalDirOfPlaneFormedByEdgeNodes]) < periodicPrecision)
			{
			  constraintsNoneEigen.add_line(i);
			  constraintsNoneEigen.add_entry(i, masterNode, 1.0);
			}
		      else
			{
			  bool foundNewMaster = false;

			  if(globalSlaveNodesRealNew.count(i) == 1)
			    {
			      for(std::set<unsigned int>::iterator it=globalMasterNodesEdgeReal.begin(); it!=globalMasterNodesEdgeReal.end(); ++it)
				{
				  Point<3> r = d_supportPointsEigen.find(*it)->second;
				  if(std::abs(r[normalDirOfPlaneFormedByEdgeNodes] - p[normalDirOfPlaneFormedByEdgeNodes]) < periodicPrecision)	
				    {
				      unsigned int correctMasterDof = *it;
				      constraintsNoneEigen.add_line(i);
				      constraintsNoneEigen.add_entry(i, correctMasterDof, 1.0);
				      foundNewMaster = true;
				      break;
				    }
				  
				}

			    }
			  else
			    {
			      for(std::set<unsigned int>::iterator it=globalMasterNodesEdgeImag.begin(); it!=globalMasterNodesEdgeImag.end(); ++it)
				{
				  Point<3> r = d_supportPointsEigen.find(*it)->second;
				  if(std::abs(r[normalDirOfPlaneFormedByEdgeNodes] - p[normalDirOfPlaneFormedByEdgeNodes]) < periodicPrecision)	
				    {
				      unsigned int correctMasterDof = *it;
				      constraintsNoneEigen.add_line(i);
				      constraintsNoneEigen.add_entry(i, correctMasterDof, 1.0);
				      foundNewMaster = true;
				      break;
				    }
				  
				}
			    }

			  if (!foundNewMaster)
			    {
			      std::cout<< " Wrong MasterNode for slave node: "<<masterNode<<" "<<i<<std::endl;
			      std::cout<< "Error: Did not find a replacement master node for a wrong master-slave periodic pair"<<" "<<masterNode<<" slaveNode: "<<i<<std::endl;
			      exit(-1);
			    }


			}

		    }
		  else
		    {
		      constraintsNoneEigen.add_line(i);
		      constraintsNoneEigen.add_entry(i, masterNode, 1.0);
		    }
		      
		}

	    }
	}
    }
  constraintsNoneEigen.close();
  constraintsTemp1Eigen.clear();
#endif

  //
  //create a constraint matrix without only hanging node constraints 
  //
  d_noConstraints.clear();d_noConstraintsEigen.clear();
  DoFTools::make_hanging_node_constraints(dofHandler, d_noConstraints);
  DoFTools::make_hanging_node_constraints(dofHandlerEigen,d_noConstraintsEigen);
  d_noConstraints.close();d_noConstraintsEigen.close();
  

  //
  //merge hanging node constraint matrix with constrains None and constraints None eigen
  //
  constraintsNone.merge(d_noConstraints,ConstraintMatrix::MergeConflictBehavior::right_object_wins);
  constraintsNoneEigen.merge(d_noConstraintsEigen,ConstraintMatrix::MergeConflictBehavior::right_object_wins);
  constraintsNone.close();
  constraintsNoneEigen.close();

  forcePtr->initUnmoved();
  computing_timer.exit_section("unmoved setup");    
}
