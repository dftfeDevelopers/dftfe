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
  void init();
  void reinit(bool isTriaRefined=true);
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
  void initLocalPseudoPotentialForce();
  void initNonLocalPseudoPotentialForce();
  void computeSparseStructureNonLocalProjectorsForce();
  void computeElementalProjectorKetsForce();
  void configForceLinFEInit();
  void configForceLinFEFinalize();
  void computeConfigurationalForceEEshelbyTensorFPSPNonPeriodicLinFE();
  void computeConfigurationalForceEEshelbyTensorFPSPPeriodicLinFE();  
  void computeConfigurationalForcePhiExtLinFE();
  void computeConfigurationalForceEselfLinFE();
  void computeConfigurationalForceEselfNoSurfaceLinFE();
  void computeConfigurationalForceTotalLinFE();
  void computeAtomsForcesGaussianGenerator(bool allowGaussianOverlapOnAtoms=false);
  void relaxAtomsForces();
  void relaxStress();
  void relaxAtomsForcesStress();
  void locateAtomCoreNodesForce();
  
  //force pseudopotential data
  std::map<dealii::CellId, std::vector<double> > gradPseudoVLoc;
  std::map<unsigned int,std::map<dealii::CellId, std::vector<double> > > gradPseudoVLocAtoms;  

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
  //map of active cell index of cell with atleast one solved dof to the closest atom Id
  std::map<unsigned int, unsigned int> d_cellsVselfBallsClosestAtomIdDofHandler;
  //set of atom vself bins in interecting the current processor dofHandler
  std::set<unsigned int> atomIdsBinsLocal;
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
