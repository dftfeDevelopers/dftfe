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
// @author Shiva Rudraraju (2016), Phani Motamarri (2016)
//

#ifndef force_H_
#define force_H_
#include "headers.h"
#include "constants.h"
#include "meshMovementGaussian.h"
//#include "dft.h"

using namespace dealii;
typedef dealii::parallel::distributed::Vector<double> vectorType;
template <unsigned int T> class dftClass;
//
//Define forceClass class
//
template <unsigned int FEOrder>
class forceClass
{
  template <unsigned int T>  friend class dftClass;

public:
  forceClass(dftClass<FEOrder>* _dftPtr);
  void initUnmoved(Triangulation<3,3> & triangulation);
  void initMoved();
  void initPseudoData();
  void computeAtomsForces();
  void computeStress();
  void printAtomsForces();
  void printStress();
  void relax();   
  void updateAtomPositionsAndMoveMesh(const std::vector<Point<C_DIM> > & globalAtomsDisplacements);
private:
  vectorType d_configForceVectorLinFE;
  std::vector<unsigned int> d_globalAtomsRelaxationPermissions;
  std::vector<double> d_globalAtomsRelaxationDisplacements;
  void createBinObjectsForce();
  //configurational force functions
  void configForceLinFEInit();
  void configForceLinFEFinalize();
  void computeConfigurationalForceEEshelbyTensorFPSPFnlLinFE();  
  void computeConfigurationalForcePhiExtLinFE();
  void computeConfigurationalForceEselfLinFE();
  void computeConfigurationalForceEselfNoSurfaceLinFE();
  void computeConfigurationalForceTotalLinFE();
  void FPSPLocalGammaAtomsElementalContribution(std::map<unsigned int, std::vector<double> > & forceContributionFPSPLocalGammaAtoms,
		                                FEValues<C_DIM> & feVselfValues,
                                                FEEvaluation<C_DIM,1,C_num1DQuad<FEOrder>(),C_DIM>  & forceEval,					    
				                const unsigned int cell,
			                        const std::vector<VectorizedArray<double> > & rhoQuads);

  void distributeForceContributionFPSPLocalGammaAtoms(const std::map<unsigned int, std::vector<double> > & forceContributionFPSPLocalGammaAtoms);   

  void FnlGammaAtomsElementalContributionNonPeriodic(std::map<unsigned int, std::vector<double> > & forceContributionFnlGammaAtoms,
						     FEEvaluation<C_DIM,1,C_num1DQuad<FEOrder>(),C_DIM>  & forceEval,
						     const unsigned int cell,
						     const std::vector<std::vector<std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > > > pspnlGammaAtomQuads,
						     const std::vector<std::vector<double> >  & projectorKetTimesPsiTimesV,							       
						     const std::vector< VectorizedArray<double> > & psiQuads);
  void FnlGammaAtomsElementalContributionPeriodic(std::map<unsigned int, std::vector<double> > & forceContributionFnlGammaAtoms,
						  FEEvaluation<C_DIM,1,C_num1DQuad<FEOrder>(),C_DIM>  & forceEval,
						  const unsigned int cell,
						  const std::vector<std::vector<std::vector<std::vector<Tensor<1,2, Tensor<1,C_DIM,VectorizedArray<double> > > > > > > & pspnlGammaAtomsQuads,
						  const std::vector<std::vector<std::vector<std::complex<double> > > > & projectorKetTimesPsiTimesV,							       
						  const std::vector<Tensor<1,2,VectorizedArray<double> > > & psiQuads);  


  void distributeForceContributionFnlGammaAtoms(const std::map<unsigned int, std::vector<double> > & forceContributionFnlGammaAtoms);   
  //
  void computeAtomsForcesGaussianGenerator(bool allowGaussianOverlapOnAtoms=false);
  //void computeEnlFnlForceContribution();  
  void relaxAtomsForces();
  void relaxStress();
  void relaxAtomsForcesStress();
  void locateAtomCoreNodesForce();

  
  //////force related pseudopotential member functions and data members
  void initLocalPseudoPotentialForce();
  void computeElementalNonLocalPseudoDataForce(); 
  void computeNonLocalProjectorKetTimesPsiTimesV(const std::vector<vectorType*> &src,
                                                 std::vector<std::vector<double> > & projectorKetTimesPsiTimesVReal,
                                                 std::vector<std::vector<std::complex<double> > > & projectorKetTimesPsiTimesVComplex,
						 const unsigned int kPointIndex);
 
  //storage for precomputed nonlocal pseudopotential quadrature data
  //map<nonlocal atom id with non-zero compact support, vector< elemental quad data >(number pseudo wave functions)>
#ifdef ENABLE_PERIODIC_BC 
  std::vector<std::vector<std::map<dealii::CellId, std::vector<double > > > > d_nonLocalPSP_ZetalmDeltaVl;
  std::vector<std::vector<std::map<dealii::CellId, std::vector<double > > > > d_nonLocalPSP_gradZetalmDeltaVl_KPoint;
  std::vector<std::vector<std::map<dealii::CellId, std::vector<double > > > > d_nonLocalPSP_gradZetalmDeltaVl_minusZetalmDeltaVl_KPoint;  
#else
  std::vector<std::vector<std::map<dealii::CellId, std::vector<double > > > > d_nonLocalPSP_ZetalmDeltaVl;
  std::vector<std::vector<std::map<dealii::CellId, std::vector<double > > > > d_nonLocalPSP_gradZetalmDeltaVl;
#endif
  //storage for precompute localPseudo data
  std::map<dealii::CellId, std::vector<double> > d_gradPseudoVLoc;
  //only contains maps for atoms whose psp tail intersects the local domain
  std::map<unsigned int,std::map<dealii::CellId, std::vector<double> > > d_gradPseudoVLocAtoms;

  //meshMovementGaussianClass object  								       
  meshMovementGaussianClass gaussianMove;

  //Gaussian generator related data and functions
  const double d_gaussianConstant=5.0;
  std::vector<double> d_globalAtomsGaussianForces;
  const bool d_allowGaussianOverlapOnAtoms=false;//Dont use true except for debugging forces only without mesh movement, as gaussian ovelap on atoms for move mesh is by default set to false

  //pointer to dft class
  dftClass<FEOrder>* dftPtr;

  //dealii based FE data structres
  FESystem<C_DIM>  FEForce;
  DoFHandler<C_DIM> d_dofHandlerForce;
  unsigned int d_forceDofHandlerIndex;
  std::map<types::global_dof_index, Point<C_DIM> > d_supportPointsForce;
  std::map<types::global_dof_index, Point<C_DIM> > d_locallyOwnedSupportPointsForceX, d_locallyOwnedSupportPointsForceY, d_locallyOwnedSupportPointsForceZ ;
  IndexSet   d_locally_owned_dofsForce;
  IndexSet   d_locally_relevant_dofsForce;
  ConstraintMatrix d_constraintsNoneForce;
  //data structures

  std::map<std::pair<unsigned int,unsigned int>, unsigned int>  d_atomsForceDofs;//(<atomId,force component>, globaldof>  
  //(outermost vector over bins) local data required for configurational force computation corresponding to Eself
  std::vector<std::vector<DoFHandler<C_DIM>::active_cell_iterator> > d_cellsVselfBallsDofHandler;
  std::vector<std::vector<DoFHandler<C_DIM>::active_cell_iterator> > d_cellsVselfBallsDofHandlerForce;
  //map of vself ball cell Id  with atleast one solved dof to the closest atom Id. Vector over bins
  std::vector<std::map<dealii::CellId , unsigned int> > d_cellsVselfBallsClosestAtomIdDofHandler;
  //map of atom Id to bin Id local
  std::map<unsigned int, unsigned int> d_AtomIdBinIdLocalDofHandler;  
  //
  std::vector<std::map<DoFHandler<C_DIM>::active_cell_iterator,std::vector<unsigned int > > > d_cellFacesVselfBallSurfacesDofHandler;
  std::vector<std::map<DoFHandler<C_DIM>::active_cell_iterator,std::vector<unsigned int > > > d_cellFacesVselfBallSurfacesDofHandlerForce; 

  //parallel objects
  MPI_Comm mpi_communicator;
  const unsigned int n_mpi_processes;
  const unsigned int this_mpi_process;
  dealii::ConditionalOStream   pcout;

  //compute-time logger
  dealii::TimerOutput computing_timer;

};

#endif
