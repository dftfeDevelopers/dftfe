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

#include "../../include/force.h"
#include "../../include/dft.h"
#include "../../include/poisson.h"
#include "../../include/constants.h"
#include "../../include/eshelbyTensor.h"
#include "../../include/fileReaders.h"


#include "configurationalForceEEshelbyFPSPFnlLinFE.cc"
#include "FPSPLocalGammaAtomsElementalContribution.cc"
#include "configurationalForceEselfLinFE.cc"
#include "initPseudoForce.cc"
#include "createBinObjectsForce.cc"
#include "gaussianGeneratorConfForce.cc"
#include "locateAtomCoreNodesForce.cc"
#include "stress.cc"
#include "relax.cc"
#include "moveAtoms.cc"



//
//constructor
//
template<unsigned int FEOrder>
forceClass<FEOrder>::forceClass(dftClass<FEOrder>* _dftPtr):
  dftPtr(_dftPtr),
  FEForce (FE_Q<3>(QGaussLobatto<1>(2)), 3), //linear shape function
  gaussianMove(),
  mpi_communicator (MPI_COMM_WORLD),
  n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_communicator)),
  this_mpi_process (Utilities::MPI::this_mpi_process(mpi_communicator)),
  pcout(std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)),
  computing_timer (pcout, TimerOutput::never, TimerOutput::wall_times)
{

}

//
//initialize forceClass object
//
template<unsigned int FEOrder>
void forceClass<FEOrder>::initUnmoved(Triangulation<3,3> & triangulation)
{
  computing_timer.enter_section("forceClass setup");
  d_dofHandlerForce.clear();
  d_dofHandlerForce.initialize(triangulation,FEForce);
  d_dofHandlerForce.distribute_dofs(FEForce);
  d_locally_owned_dofsForce.clear();d_locally_relevant_dofsForce.clear();
  d_locally_owned_dofsForce = d_dofHandlerForce.locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs(d_dofHandlerForce, d_locally_relevant_dofsForce);  

  ///
  d_constraintsNoneForce.clear(); d_constraintsNoneForce.reinit(d_locally_relevant_dofsForce);
  DoFTools::make_hanging_node_constraints(d_dofHandlerForce, d_constraintsNoneForce);   
#ifdef ENABLE_PERIODIC_BC
  std::vector<GridTools::PeriodicFacePair<typename DoFHandler<C_DIM>::cell_iterator> > periodicity_vectorForce;
  for (int i = 0; i < C_DIM; ++i)
    {
      GridTools::collect_periodic_faces(d_dofHandlerForce, /*b_id1*/ 2*i+1, /*b_id2*/ 2*i+2,/*direction*/ i, periodicity_vectorForce);
    }
  DoFTools::make_periodicity_constraints<DoFHandler<C_DIM> >(periodicity_vectorForce, d_constraintsNoneForce);
  d_constraintsNoneForce.close();
#else
  d_constraintsNoneForce.close();
#endif
  gaussianMove.init(triangulation);
  computing_timer.exit_section("forceClass setup"); 
}

//reinitialize force class object after mesh update
template<unsigned int FEOrder>
void forceClass<FEOrder>::initMoved()
{
  d_dofHandlerForce.distribute_dofs(FEForce);
  d_supportPointsForce.clear();
  DoFTools::map_dofs_to_support_points(MappingQ1<3,3>(), d_dofHandlerForce, d_supportPointsForce);
  //
  //Extract force component dofs from the global force dofs - this will be needed in configurational force.
  //
  d_locallyOwnedSupportPointsForceX.clear();d_locallyOwnedSupportPointsForceY.clear();d_locallyOwnedSupportPointsForceZ.clear();
  FEValuesExtractors::Scalar x(0), y(1); 
  ComponentMask componentMaskX = FEForce.component_mask(x);
  ComponentMask componentMaskY = FEForce.component_mask(y);
  std::vector<bool> selectedDofsX(d_locally_owned_dofsForce.n_elements(), false);
  std::vector<bool> selectedDofsY(d_locally_owned_dofsForce.n_elements(), false);
  DoFTools::extract_dofs(d_dofHandlerForce, componentMaskX, selectedDofsX);
  DoFTools::extract_dofs(d_dofHandlerForce, componentMaskY, selectedDofsY);
  std::vector<unsigned int> local_dof_indicesForce(d_locally_owned_dofsForce.n_elements());
  d_locally_owned_dofsForce.fill_index_vector(local_dof_indicesForce);
  for (unsigned int i = 0; i < d_locally_owned_dofsForce.n_elements(); i++)
  {
      const int globalIndex=local_dof_indicesForce[i];
      if(selectedDofsX[i]) 
      {
	  d_locallyOwnedSupportPointsForceX[globalIndex]=d_supportPointsForce[globalIndex];
      }
      else if(selectedDofsY[i])
      {
	  d_locallyOwnedSupportPointsForceY[globalIndex]=d_supportPointsForce[globalIndex];
      }
      else
      {
	  d_locallyOwnedSupportPointsForceZ[globalIndex]=d_supportPointsForce[globalIndex];
      }
  }    

  createBinObjectsForce();
  locateAtomCoreNodesForce();
  //
  //initialize pseudopotential related force objects
  //
  if(dftParameters::isPseudopotential)
    {
      initLocalPseudoPotentialForce();
      //initNonLocalPseudoPotentialForce();
      //computeSparseStructureNonLocalProjectorsForce();
      //computeElementalProjectorKetsForce();
    }
}

//compute forces on atoms using a generator with a compact support
template<unsigned int FEOrder>
void forceClass<FEOrder>::computeAtomsForces(){
  computeConfigurationalForceTotalLinFE();
  computeAtomsForcesGaussianGenerator(d_allowGaussianOverlapOnAtoms);
}


template<unsigned int FEOrder>
void forceClass<FEOrder>::configForceLinFEInit()
{

  dftPtr->matrix_free_data.initialize_dof_vector(d_configForceVectorLinFE,d_forceDofHandlerIndex);
  d_configForceVectorLinFE=0;//also zeros out the ghost vectors
}

template<unsigned int FEOrder>
void forceClass<FEOrder>::configForceLinFEFinalize()
{
  d_configForceVectorLinFE.compress(VectorOperation::add);//copies the ghost element cache to the owning element
  d_configForceVectorLinFE.update_ghost_values();
  d_constraintsNoneForce.distribute(d_configForceVectorLinFE);//distribute to constrained degrees of freedom (for example periodic)
  d_configForceVectorLinFE.update_ghost_values();

}

//compute configurational force on the mesh nodes using linear shape function generators
template<unsigned int FEOrder>
void forceClass<FEOrder>::computeConfigurationalForceTotalLinFE()
{

 
  configForceLinFEInit();

  computeConfigurationalForceEEshelbyTensorFPSPFnlLinFE(); 
  //computeConfigurationalForcePhiExtLinFE();
  //computeConfigurationalForceEselfNoSurfaceLinFE();
  computeConfigurationalForceEselfLinFE();
  configForceLinFEFinalize();

  
  std::map<std::pair<unsigned int,unsigned int>, unsigned int> ::const_iterator it; 
  for (it=d_atomsForceDofs.begin(); it!=d_atomsForceDofs.end(); ++it){
	 const std::pair<unsigned int,unsigned int> & atomIdPair= it->first;
	 const unsigned int atomForceDof=it->second;
	 std::cout<<" atomId: "<< atomIdPair.first << ", force component: "<<atomIdPair.second << ", force: "<<d_configForceVectorLinFE[atomForceDof] << std::endl;
  }
   

}

template<unsigned int FEOrder>
void forceClass<FEOrder>::relax(){
}

template class forceClass<1>;
template class forceClass<2>;
template class forceClass<3>;
template class forceClass<4>;
template class forceClass<5>;
template class forceClass<6>;
template class forceClass<7>;
template class forceClass<8>;
template class forceClass<9>;
template class forceClass<10>;
template class forceClass<11>;
template class forceClass<12>;
