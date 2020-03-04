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
// @author Sambit Das
//


#include <force.h>
#ifdef DFTFE_WITH_GPU
#include <forceCUDA.h>
#endif
#include <dft.h>
#include <dftUtils.h>
#include <constants.h>
#include <eshelbyTensor.h>
#include <eshelbyTensorSpinPolarized.h>
#include <fileReaders.h>
#include <linearAlgebraOperations.h>
#include <vectorUtilities.h>
#include <boost/math/special_functions/spherical_harmonic.hpp>


//This class computes and stores the configurational forces corresponding to geometry optimization.
//It uses the formulation in the paper by Motamarri et.al. (https://arxiv.org/abs/1712.05535)
//which provides an unified approach to atomic forces corresponding to internal atomic relaxation and cell stress
//corresponding to cell relaxation.

namespace  dftfe {

#include "configurationalForceCompute/configurationalForceEEshelbyFPSPFnlLinFE.cc"
#include "configurationalForceCompute/configurationalForceSpinPolarizedEEshelbyFPSPFnlLinFE.cc"
#include "configurationalForceCompute/FPSPLocalGammaAtomsElementalContribution.cc"
#include "configurationalForceCompute/FShadowLocalGammaAtomsElementalContribution.cc"
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

namespace internalForce
{
    void initUnmoved(const Triangulation<3,3> & triangulation,
	             const Triangulation<3,3> & serialTriangulation,
	             const std::vector<std::vector<double> >  & domainBoundingVectors,
		     const MPI_Comm & mpi_comm,
		     DoFHandler<C_DIM> & dofHandlerForce,
		     FESystem<C_DIM> & FEForce,
		     ConstraintMatrix  & constraintsForce,
		     IndexSet  & locally_owned_dofsForce,
		     IndexSet  & locally_relevant_dofsForce)
    {
      dofHandlerForce.clear();
      dofHandlerForce.initialize(triangulation,FEForce);
      dofHandlerForce.distribute_dofs(FEForce);
      locally_owned_dofsForce.clear();locally_relevant_dofsForce.clear();
      locally_owned_dofsForce = dofHandlerForce.locally_owned_dofs();
      DoFTools::extract_locally_relevant_dofs(dofHandlerForce, locally_relevant_dofsForce);

      ///
      constraintsForce.clear(); constraintsForce.reinit(locally_relevant_dofsForce);
      DoFTools::make_hanging_node_constraints(dofHandlerForce, constraintsForce);

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
	      offsetVectors[i][j] = unitVectorsXYZ[i][j] - domainBoundingVectors[i][j];
	    }
	}

      std::vector<GridTools::PeriodicFacePair<typename DoFHandler<C_DIM>::cell_iterator> > periodicity_vectorForce;

      const std::array<int,3> periodic = {dftParameters::periodicX, dftParameters::periodicY, dftParameters::periodicZ};


      std::vector<int> periodicDirectionVector;

      for(unsigned int  d= 0; d < 3; ++d)
	{
	  if(periodic[d]==1)
	    {
	      periodicDirectionVector.push_back(d);
	    }
	}

      for (int i = 0; i < std::accumulate(periodic.begin(),periodic.end(),0); ++i)
       {
	  GridTools::collect_periodic_faces(dofHandlerForce, /*b_id1*/ 2*i+1, /*b_id2*/ 2*i+2,/*direction*/ periodicDirectionVector[i], periodicity_vectorForce,offsetVectors[periodicDirectionVector[i]]);
	}

      DoFTools::make_periodicity_constraints<DoFHandler<C_DIM> >(periodicity_vectorForce, constraintsForce);
      constraintsForce.close();

      if (dftParameters::createConstraintsFromSerialDofhandler)
      {
            ConstraintMatrix  dummy;
	    vectorTools::createParallelConstraintMatrixFromSerial(serialTriangulation,
								  dofHandlerForce,
								  mpi_comm,
								  domainBoundingVectors,
								  constraintsForce,
								  dummy);
      }
    }
}

//
//constructor
//
template<unsigned int FEOrder>
forceClass<FEOrder>::forceClass(dftClass<FEOrder>* _dftPtr,const MPI_Comm &mpi_comm_replica):
  dftPtr(_dftPtr),
  FEForce (FE_Q<3>(QGaussLobatto<1>(2)), 3), //linear shape function
  mpi_communicator (mpi_comm_replica),
  n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_comm_replica)),
  this_mpi_process (Utilities::MPI::this_mpi_process(mpi_comm_replica)),
  pcout(std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0))
{

}

//
//initialize forceClass object
//
template<unsigned int FEOrder>
void forceClass<FEOrder>::initUnmoved(const Triangulation<3,3> & triangulation,
	                              const Triangulation<3,3> & serialTriangulation,
	                              const std::vector<std::vector<double> >  & domainBoundingVectors,
	                              const bool isElectrostaticsMesh,
				      const double gaussianConstant)
{
    d_gaussianConstant=dftParameters::reproducible_output?1/std::sqrt(5.0):gaussianConstant;
    if (isElectrostaticsMesh)
	internalForce::initUnmoved(triangulation,
		                   serialTriangulation,
		                   domainBoundingVectors,
				   mpi_communicator,
				   d_dofHandlerForceElectro,
				   FEForce,
				   d_constraintsNoneForceElectro,
				   d_locally_owned_dofsForceElectro,
				   d_locally_relevant_dofsForceElectro);
    else
	internalForce::initUnmoved(triangulation,
		                   serialTriangulation,
		                   domainBoundingVectors,
				   mpi_communicator,
				   d_dofHandlerForce,
				   FEForce,
				   d_constraintsNoneForce,
				   d_locally_owned_dofsForce,
				   d_locally_relevant_dofsForce);
}



//reinitialize force class object after mesh update
template<unsigned int FEOrder>
void forceClass<FEOrder>::initMoved
                    (std::vector<const DoFHandler<3> *> & dofHandlerVectorMatrixFree,
	             std::vector<const ConstraintMatrix * > & constraintsVectorMatrixFree,
	             const bool isElectrostaticsMesh,
		     const bool isElectrostaticsEigenMeshDifferent)
{
  if (isElectrostaticsMesh)
  {
     d_dofHandlerForceElectro.distribute_dofs(FEForce);

     if (isElectrostaticsEigenMeshDifferent)
     {
       dofHandlerVectorMatrixFree.push_back(&d_dofHandlerForceElectro);
       constraintsVectorMatrixFree.push_back(&d_constraintsNoneForceElectro);
       d_isElectrostaticsMeshSubdivided=true;
     }
     else
       d_isElectrostaticsMeshSubdivided=false;

     d_forceDofHandlerIndexElectro = dofHandlerVectorMatrixFree.size()-1;

     locateAtomCoreNodesForce(d_dofHandlerForceElectro,
	                      d_locally_owned_dofsForceElectro,
			      d_atomsForceDofsElectro);

  }
  else
  {
     d_dofHandlerForce.distribute_dofs(FEForce);

     dofHandlerVectorMatrixFree.push_back(&d_dofHandlerForce);
     d_forceDofHandlerIndex = dofHandlerVectorMatrixFree.size()-1;
     constraintsVectorMatrixFree.push_back(&d_constraintsNoneForce);

     locateAtomCoreNodesForce(d_dofHandlerForce,
	                      d_locally_owned_dofsForce,
			      d_atomsForceDofs);

    const unsigned int numberGlobalAtoms = dftPtr->atomLocations.size(); 
    std::vector<Point<3>> atomPoints;
    for (unsigned int iAtom=0;iAtom <numberGlobalAtoms; iAtom++)
    {
         Point<3> atomCoor;
         atomCoor[0] = dftPtr->atomLocations[iAtom][2];
         atomCoor[1] = dftPtr->atomLocations[iAtom][3];
         atomCoor[2] = dftPtr->atomLocations[iAtom][4];
         atomPoints.push_back(atomCoor);
    }

     /*
     double minDist=1e+6;
     for (unsigned int i=0;i <numberGlobalAtoms-1; i++)
	 for (unsigned int j=i+1;j <numberGlobalAtoms; j++)
	   {
	      const double dist=atomPoints[i].distance(atomPoints[j]);
	      if (dist<minDist)
	             minDist=dist;
	   }

     d_gaussianConstant=dftParameters::reproducible_output?1/std::sqrt(5.0):std::min(0.7* minDist/2.0, 0.75);
     */

  }
}

//
//initialize pseudopotential data for force computation
//
template<unsigned int FEOrder>
void forceClass<FEOrder>::initPseudoData()
{

  if(dftParameters::isPseudopotential)
        computeElementalNonLocalPseudoOVDataForce();
}

//compute forces on atoms corresponding to a Gaussian generator
template<unsigned int FEOrder>
void forceClass<FEOrder>::computeAtomsForces
		 (const MatrixFree<3,double> & matrixFreeData,
#ifdef DFTFE_WITH_GPU
                 kohnShamDFTOperatorCUDAClass<FEOrder> & kohnShamDFTEigenOperator,
#endif
		 const unsigned int eigenDofHandlerIndex,
		 const unsigned int phiExtDofHandlerIndex,
		 const unsigned int phiTotDofHandlerIndex,
		 const vectorType & phiTotRhoIn,
		 const vectorType & phiTotRhoOut,
		 const vectorType & phiExt,
		 const std::map<dealii::CellId, std::vector<double> > & pseudoVLoc,
		 const std::map<dealii::CellId, std::vector<double> > & gradPseudoVLoc,
		 const std::map<unsigned int,std::map<dealii::CellId, std::vector<double> > > & gradPseudoVLocAtoms,
		 const ConstraintMatrix  & noConstraints,
		 const vselfBinsManager<FEOrder> & vselfBinsManagerEigen,
	         const MatrixFree<3,double> & matrixFreeDataElectro,
		 const unsigned int phiTotDofHandlerIndexElectro,
		 const unsigned int phiExtDofHandlerIndexElectro,
		 const vectorType & phiTotRhoOutElectro,
		 const vectorType & phiExtElectro,
                 const std::map<dealii::CellId, std::vector<double> > & rhoOutValues,
                 const std::map<dealii::CellId, std::vector<double> > & gradRhoOutValues,
		 const std::map<dealii::CellId, std::vector<double> > & rhoOutValuesElectro,
		 const std::map<dealii::CellId, std::vector<double> > & gradRhoOutValuesElectro,
		 const std::map<dealii::CellId, std::vector<double> > & pseudoVLocElectro,
		 const std::map<dealii::CellId, std::vector<double> > & gradPseudoVLocElectro,
		 const std::map<unsigned int,std::map<dealii::CellId, std::vector<double> > > & gradPseudoVLocAtomsElectro,
	         const ConstraintMatrix  & noConstraintsElectro,
		 const vselfBinsManager<FEOrder> & vselfBinsManagerElectro,
                 const std::map<dealii::CellId, std::vector<double> > & shadowKSRhoMinValues,
                 const std::map<dealii::CellId, std::vector<double> > & shadowKSGradRhoMinValues,
                 const vectorType & phiRhoMinusApproxRho,
                 const bool shadowPotentialForce)
{
  /*
  createBinObjectsForce(matrixFreeData.get_dof_handler(phiTotDofHandlerIndex),
	                d_dofHandlerForce,
	                noConstraints,
	                vselfBinsManagerEigen,
                        d_cellsVselfBallsDofHandler,
                        d_cellsVselfBallsDofHandlerForce,
                        d_cellsVselfBallsClosestAtomIdDofHandler,
                        d_AtomIdBinIdLocalDofHandler,
                        d_cellFacesVselfBallSurfacesDofHandler,
                        d_cellFacesVselfBallSurfacesDofHandlerForce);
  */

  createBinObjectsForce(matrixFreeDataElectro.get_dof_handler(phiTotDofHandlerIndexElectro),
	                d_dofHandlerForceElectro,
	                noConstraintsElectro,
	                vselfBinsManagerElectro,
                        d_cellsVselfBallsDofHandlerElectro,
                        d_cellsVselfBallsDofHandlerForceElectro,
                        d_cellsVselfBallsClosestAtomIdDofHandlerElectro,
                        d_AtomIdBinIdLocalDofHandlerElectro,
                        d_cellFacesVselfBallSurfacesDofHandlerElectro,
                        d_cellFacesVselfBallSurfacesDofHandlerForceElectro);

  computeConfigurationalForceTotalLinFE(matrixFreeData,
#ifdef DFTFE_WITH_GPU
                                        kohnShamDFTEigenOperator,
#endif
		                        eigenDofHandlerIndex,
		                        phiExtDofHandlerIndex,
		                        phiTotDofHandlerIndex,
		                        phiTotRhoIn,
		                        phiTotRhoOut,
		                        phiExt,
					pseudoVLoc,
					gradPseudoVLoc,
					gradPseudoVLocAtoms,
		                        vselfBinsManagerEigen,
	                                matrixFreeDataElectro,
		                        phiTotDofHandlerIndexElectro,
		                        phiExtDofHandlerIndexElectro,
		                        phiTotRhoOutElectro,
		                        phiExtElectro,
                                        rhoOutValues,
                                        gradRhoOutValues,
		                        rhoOutValuesElectro,
					gradRhoOutValuesElectro,
					pseudoVLocElectro,
					gradPseudoVLocElectro,
					gradPseudoVLocAtomsElectro,
		                        vselfBinsManagerElectro,
                                        shadowKSRhoMinValues,
                                        shadowKSGradRhoMinValues,
                                        phiRhoMinusApproxRho,
                                        shadowPotentialForce);

  computeAtomsForcesGaussianGenerator(d_allowGaussianOverlapOnAtoms);
}


template<unsigned int FEOrder>
void forceClass<FEOrder>::configForceLinFEInit(const MatrixFree<3,double> & matrixFreeData,
	                                       const MatrixFree<3,double> & matrixFreeDataElectro)
{

  matrixFreeData.initialize_dof_vector(d_configForceVectorLinFE,d_forceDofHandlerIndex);
  d_configForceVectorLinFE=0;

  matrixFreeDataElectro.initialize_dof_vector(d_configForceVectorLinFEElectro,
	                                         d_forceDofHandlerIndexElectro);
  d_configForceVectorLinFEElectro=0;

#ifdef USE_COMPLEX
  matrixFreeData.initialize_dof_vector(d_configForceVectorLinFEKPoints,d_forceDofHandlerIndex);
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


  d_configForceVectorLinFEElectro.compress(VectorOperation::add);//copies the ghost element cache to the owning element
  //d_configForceVectorLinFE.update_ghost_values();
  d_constraintsNoneForceElectro.distribute(d_configForceVectorLinFEElectro);//distribute to constrained degrees of freedom (for example periodic)
  d_configForceVectorLinFEElectro.update_ghost_values();

#ifdef USE_COMPLEX
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
void forceClass<FEOrder>::computeConfigurationalForceTotalLinFE
                                     (const MatrixFree<3,double> & matrixFreeData,
#ifdef DFTFE_WITH_GPU
                                      kohnShamDFTOperatorCUDAClass<FEOrder> & kohnShamDFTEigenOperator,
#endif
				     const unsigned int eigenDofHandlerIndex,
				     const unsigned int phiExtDofHandlerIndex,
				     const unsigned int phiTotDofHandlerIndex,
				     const vectorType & phiTotRhoIn,
				     const vectorType & phiTotRhoOut,
				     const vectorType & phiExt,
		                     const std::map<dealii::CellId, std::vector<double> > & pseudoVLoc,
		                     const std::map<dealii::CellId, std::vector<double> > & gradPseudoVLoc,
		                     const std::map<unsigned int,std::map<dealii::CellId, std::vector<double> > > & gradPseudoVLocAtoms,
				     const vselfBinsManager<FEOrder> & vselfBinsManagerEigen,
				     const MatrixFree<3,double> & matrixFreeDataElectro,
				     const unsigned int phiTotDofHandlerIndexElectro,
				     const unsigned int phiExtDofHandlerIndexElectro,
				     const vectorType & phiTotRhoOutElectro,
				     const vectorType & phiExtElectro,
                                     const std::map<dealii::CellId, std::vector<double> > & rhoOutValues,
                                     const std::map<dealii::CellId, std::vector<double> > & gradRhoOutValues,
				     const std::map<dealii::CellId, std::vector<double> > & rhoOutValuesElectro,
				     const std::map<dealii::CellId, std::vector<double> > & gradRhoOutValuesElectro,
		                     const std::map<dealii::CellId, std::vector<double> > & pseudoVLocElectro,
		                     const std::map<dealii::CellId, std::vector<double> > & gradPseudoVLocElectro,
		                     const std::map<unsigned int,std::map<dealii::CellId, std::vector<double> > > & gradPseudoVLocAtomsElectro,
				     const vselfBinsManager<FEOrder> & vselfBinsManagerElectro,
                                     const std::map<dealii::CellId, std::vector<double> > & shadowKSRhoMinValues,
                                     const std::map<dealii::CellId, std::vector<double> > & shadowKSGradRhoMinValues,
                                     const vectorType & phiRhoMinusApproxRho,
                                     const bool shadowPotentialForce)
{


  configForceLinFEInit(matrixFreeData,
	               matrixFreeDataElectro);

  //configurational force contribution from all terms except those from nuclear self energy
  if (dftParameters::spinPolarized)
     computeConfigurationalForceSpinPolarizedEEshelbyTensorFPSPFnlLinFE
		                        (matrixFreeData,
		                        eigenDofHandlerIndex,
		                        phiExtDofHandlerIndex,
		                        phiTotDofHandlerIndex,
		                        phiTotRhoIn,
		                        phiTotRhoOut,
		                        phiExt,
					pseudoVLoc,
					gradPseudoVLoc,
					gradPseudoVLocAtoms,
		                        vselfBinsManagerEigen,
	                                matrixFreeDataElectro,
		                        phiTotDofHandlerIndexElectro,
		                        phiExtDofHandlerIndexElectro,
		                        phiTotRhoOutElectro,
		                        phiExtElectro,
		                        rhoOutValuesElectro,
					gradRhoOutValuesElectro,
					pseudoVLocElectro,
					gradPseudoVLocElectro,
					gradPseudoVLocAtomsElectro,
					vselfBinsManagerElectro,
                                        shadowKSRhoMinValues,
                                        shadowKSGradRhoMinValues,
                                        phiRhoMinusApproxRho,
                                        shadowPotentialForce);
  else
     computeConfigurationalForceEEshelbyTensorFPSPFnlLinFE
		                        (matrixFreeData,
#ifdef DFTFE_WITH_GPU
                                        kohnShamDFTEigenOperator,
#endif
		                        eigenDofHandlerIndex,
		                        phiExtDofHandlerIndex,
		                        phiTotDofHandlerIndex,
		                        phiTotRhoIn,
		                        phiTotRhoOut,
		                        phiExt,
					pseudoVLoc,
					gradPseudoVLoc,
					gradPseudoVLocAtoms,
		                        vselfBinsManagerEigen,
	                                matrixFreeDataElectro,
		                        phiTotDofHandlerIndexElectro,
		                        phiExtDofHandlerIndexElectro,
		                        phiTotRhoOutElectro,
		                        phiExtElectro,
                                        rhoOutValues,
                                        gradRhoOutValues,
		                        rhoOutValuesElectro,
					gradRhoOutValuesElectro,
					pseudoVLocElectro,
					gradPseudoVLocElectro,
					gradPseudoVLocAtomsElectro,
					vselfBinsManagerElectro,
                                        shadowKSRhoMinValues,
                                        shadowKSGradRhoMinValues,
                                        phiRhoMinusApproxRho,
                                        shadowPotentialForce);

  //configurational force contribution from nuclear self energy. This is handled separately as it involves
  // a surface integral over the vself ball surface
  if (dealii::Utilities::MPI::this_mpi_process(dftPtr->interBandGroupComm)==0)
    computeConfigurationalForceEselfLinFE(matrixFreeDataElectro.get_dof_handler(phiTotDofHandlerIndexElectro),
				        vselfBinsManagerElectro);
  configForceLinFEFinalize();
#ifdef DEBUG
  std::map<std::pair<unsigned int,unsigned int>, unsigned int> ::const_iterator it;
  for (it=d_atomsForceDofs.begin(); it!=d_atomsForceDofs.end(); ++it)
  {
	 const std::pair<unsigned int,unsigned int> & atomIdPair= it->first;
	 const unsigned int atomForceDof=it->second;
	 if (dftParameters::verbosity==2)
	   std::cout<<"procid: "<< this_mpi_process<<" atomId: "<< atomIdPair.first << ", force component: "<<atomIdPair.second << ", force: "<<d_configForceVectorLinFE[atomForceDof] << std::endl;
#ifdef USE_COMPLEX
	 if (dftParameters::verbosity==2)
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

template<unsigned int FEOrder>
Tensor<2,C_DIM,double>  forceClass<FEOrder>::getStress()
{
    return d_stress;
}

template<unsigned int FEOrder>
double  forceClass<FEOrder>::getGaussianGeneratorParameter() const
{
    return d_gaussianConstant;
}

template<unsigned int FEOrder>
void  forceClass<FEOrder>::updateGaussianConstant(const double newGaussianConstant)
{
    if (!dftParameters::reproducible_output)
      d_gaussianConstant=newGaussianConstant;
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

}
