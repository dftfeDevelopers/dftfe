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
#include "../../include/dftUtils.h"
#include "../../include/poisson.h"
#include "../../include/constants.h"
#include "../../include/eshelbyTensor.h"
#include "../../include/eshelbyTensorSpinPolarized.h"
#include "../../include/meshGenerator.h"
#include "../../include/fileReaders.h"
#include <boost/math/special_functions/spherical_harmonic.hpp>



//This class computes and stores the configurational forces corresponding to geometry optimization.
//It uses the formulation in the paper by Motamarri et.al. (https://arxiv.org/abs/1712.05535)
//which provides an unified approach to atomic forces corresponding to internal atomic relaxation and cell stress
//corresponding to cell relaxation.

namespace  dftfe {

#include "configurationalForceCompute/configurationalForceEEshelbyFPSPFnlLinFE.cc"
#include "configurationalForceCompute/configurationalForceSpinPolarizedEEshelbyFPSPFnlLinFE.cc"
#include "configurationalForceCompute/FPSPLocalGammaAtomsElementalContribution.cc"
#include "configurationalForceCompute/FnlGammaAtomsElementalContribution.cc"
#include "configurationalForceCompute/FnlGammaAtomsElementalContributionSpinPolarized.cc"
#include "configurationalForceCompute/configurationalForceEselfLinFE.cc"
#include "configurationalForceCompute/gaussianGeneratorConfForceOpt.cc"
#include "configurationalStressCompute/stress.cc"
#include "configurationalStressCompute/computeStressEself.cc"
#include "configurationalStressCompute/computeStressEEshelbyEPSPEnlEk.cc"
#include "configurationalStressCompute/computeStressSpinPolarizedEEshelbyEPSPEnlEk.cc"
#include "configurationalStressCompute/EPSPStressContribution.cc"
#include "initPseudoForce.cc"
#include "initPseudoOVForce.cc"
#include "createBinObjectsForce.cc"
#include "locateAtomCoreNodesForce.cc"
#include "moveAtoms.cc"
//
//constructor
//
template<unsigned int FEOrder>
forceClass<FEOrder>::forceClass(dftClass<FEOrder>* _dftPtr,const MPI_Comm &mpi_comm_replica):
  dftPtr(_dftPtr),
  FEForce (FE_Q<3>(QGaussLobatto<1>(2)), 3), //linear shape function
  mpi_communicator (mpi_comm_replica),
  gaussianMovePar(mpi_comm_replica),
  n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_comm_replica)),
  this_mpi_process (Utilities::MPI::this_mpi_process(mpi_comm_replica)),
  pcout(std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0))
{

}

//
//initialize forceClass object
//
template<unsigned int FEOrder>
void forceClass<FEOrder>::initUnmoved(const Triangulation<3,3> & triangulation)
{
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
  //create unitVectorsXYZ
  std::vector<std::vector<double> > unitVectorsXYZ;
  unitVectorsXYZ.resize(3);

  for(int i = 0; i < 3; ++i)
    {
      unitVectorsXYZ[i].resize(3,0.0);
      unitVectorsXYZ[i][i] = 0.0;
    }

  std::vector<Tensor<1,3> > offsetVectors;
  //resize offset vectors
  offsetVectors.resize(3);

  for(int i = 0; i < 3; ++i)
    {
      for(int j = 0; j < 3; ++j)
	{
	  offsetVectors[i][j] = unitVectorsXYZ[i][j] - dftPtr->d_domainBoundingVectors[i][j];
	}
    }

  std::vector<GridTools::PeriodicFacePair<typename DoFHandler<C_DIM>::cell_iterator> > periodicity_vectorForce;

  const std::array<int,3> periodic = {dftParameters::periodicX, dftParameters::periodicY, dftParameters::periodicZ};
  for (int i = 0; i < std::accumulate(periodic.begin(),periodic.end(),0); ++i)
   {
      GridTools::collect_periodic_faces(d_dofHandlerForce, /*b_id1*/ 2*i+1, /*b_id2*/ 2*i+2,/*direction*/ i, periodicity_vectorForce,offsetVectors[i]);
    }

  DoFTools::make_periodicity_constraints<DoFHandler<C_DIM> >(periodicity_vectorForce, d_constraintsNoneForce);
  d_constraintsNoneForce.close();
#else
  d_constraintsNoneForce.close();
#endif
  gaussianMovePar.init(triangulation,dftPtr->d_domainBoundingVectors);
}

//reinitialize force class object after mesh update
template<unsigned int FEOrder>
void forceClass<FEOrder>::initMoved()
{
  d_dofHandlerForce.distribute_dofs(FEForce);
  d_supportPointsForce.clear();
  DoFTools::map_dofs_to_support_points(MappingQ1<3,3>(), d_dofHandlerForce, d_supportPointsForce);

  createBinObjectsForce();
  locateAtomCoreNodesForce();
  gaussianMovePar.initMoved(dftPtr->d_domainBoundingVectors);
}

//
//initialize pseudopotential data for force computation
//
template<unsigned int FEOrder>
void forceClass<FEOrder>::initPseudoData(){

  if(dftParameters::isPseudopotential)
    {
      initLocalPseudoPotentialForce();
      if (dftParameters::pseudoProjector==2)
      {
        computeElementalNonLocalPseudoOVDataForce();
      }
      else
      {
        computeElementalNonLocalPseudoDataForce();
      }
    }
}



//compute forces on atoms corresponding to a Gaussian generator
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
#ifdef ENABLE_PERIODIC_BC
  dftPtr->matrix_free_data.initialize_dof_vector(d_configForceVectorLinFEKPoints,d_forceDofHandlerIndex);
  d_configForceVectorLinFEKPoints=0;
#endif
}

template<unsigned int FEOrder>
void forceClass<FEOrder>::configForceLinFEFinalize()
{
  d_configForceVectorLinFE.compress(VectorOperation::add);//copies the ghost element cache to the owning element
  //d_configForceVectorLinFE.update_ghost_values();
  d_constraintsNoneForce.distribute(d_configForceVectorLinFE);//distribute to constrained degrees of freedom (for example periodic)
  d_configForceVectorLinFE.update_ghost_values();
#ifdef ENABLE_PERIODIC_BC
  d_configForceVectorLinFEKPoints.compress(VectorOperation::add);//copies the ghost element cache to the owning element
  //d_configForceVectorLinFEKPoints.update_ghost_values();
  d_constraintsNoneForce.distribute(d_configForceVectorLinFEKPoints);//distribute to constrained degrees of freedom (for example periodic)
  d_configForceVectorLinFEKPoints.update_ghost_values();
#endif

}

//compute configurational force on the finite element nodes corresponding to linear shape function
// generators. This function is generic to all-electron and pseudopotential as well as non-periodic and periodic
//cases. Also both LDA and GGA exchange correlation are handled. For details of the configurational
//force expressions refer to the Configurational force paper by Motamarri et.al.
//(https://arxiv.org/abs/1712.05535)
template<unsigned int FEOrder>
void forceClass<FEOrder>::computeConfigurationalForceTotalLinFE()
{


  configForceLinFEInit();
  //configurational force contribution from all terms except those from nuclear self energy
  if (dftParameters::spinPolarized)
     computeConfigurationalForceSpinPolarizedEEshelbyTensorFPSPFnlLinFE();
  else
     computeConfigurationalForceEEshelbyTensorFPSPFnlLinFE();
  //configurational force contribution from nuclear self energy. This is handled separately as it involves
  // a surface integral over the vself ball surface
  computeConfigurationalForceEselfLinFE();
  configForceLinFEFinalize();
#ifdef DEBUG
  std::map<std::pair<unsigned int,unsigned int>, unsigned int> ::const_iterator it;
  for (it=d_atomsForceDofs.begin(); it!=d_atomsForceDofs.end(); ++it)
  {
	 const std::pair<unsigned int,unsigned int> & atomIdPair= it->first;
	 const unsigned int atomForceDof=it->second;
	 std::cout<<"procid: "<< this_mpi_process<<" atomId: "<< atomIdPair.first << ", force component: "<<atomIdPair.second << ", force: "<<d_configForceVectorLinFE[atomForceDof] << std::endl;
#ifdef ENABLE_PERIODIC_BC
	 std::cout<<"procid: "<< this_mpi_process<<" atomId: "<< atomIdPair.first << ", force component: "<<atomIdPair.second << ", forceKPoints: "<<d_configForceVectorLinFEKPoints[atomForceDof] << std::endl;
#endif
  }
#endif

}


template<unsigned int FEOrder>
std::vector<double>  forceClass<FEOrder>::getAtomsForces()
{
   return  d_globalAtomsGaussianForces;
}

#ifdef ENABLE_PERIODIC_BC
template<unsigned int FEOrder>
Tensor<2,C_DIM,double>  forceClass<FEOrder>::getStress()
{
    return d_stress;
}
#endif

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

}
