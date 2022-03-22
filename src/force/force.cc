// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022 The Regents of the University of Michigan and DFT-FE
// authors.
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
#  include <forceCUDA.h>
#endif
#include <boost/math/special_functions/spherical_harmonic.hpp>

#include <constants.h>
#include <dft.h>
#include <dftUtils.h>
#include <eshelbyTensor.h>
#include <eshelbyTensorSpinPolarized.h>
#include <fileReaders.h>
#include <linearAlgebraOperations.h>
#include <vectorUtilities.h>


// This class computes and stores the configurational forces corresponding to
// geometry optimization. It uses the formulation in the paper by Motamarri
// et.al. (https://arxiv.org/abs/1712.05535) which provides an unified approach
// to atomic forces corresponding to internal atomic relaxation and cell stress
// corresponding to cell relaxation.

namespace dftfe
{
#include "configurationalForceCompute/FNonlinearCoreCorrectionGammaAtomsElementalContribution.cc"
#include "configurationalForceCompute/FPSPLocalGammaAtomsElementalContribution.cc"
#include "configurationalForceCompute/FShadowLocalGammaAtomsElementalContribution.cc"
#include "configurationalForceCompute/FSmearedChargesGammaAtomsElementalContribution.cc"
#include "configurationalForceCompute/FnlGammaAtomsElementalContribution.cc"
#include "configurationalForceCompute/accumulateForceContributionGammaAtomsFloating.cc"
#include "configurationalForceCompute/computeFloatingAtomsForces.cc"
#include "configurationalForceCompute/configurationalForceEEshelbyFPSPFnlLinFE.cc"
#include "configurationalForceCompute/configurationalForceEselfLinFE.cc"
#include "configurationalForceCompute/gaussianGeneratorConfForceOpt.cc"
#include "configurationalStressCompute/ENonlinearCoreCorrectionContribution.cc"
#include "configurationalStressCompute/EPSPStressContribution.cc"
#include "configurationalStressCompute/ESmearedStressContribution.cc"
#include "configurationalStressCompute/computeStressEEshelbyEPSPEnlEk.cc"
#include "configurationalStressCompute/computeStressEself.cc"
#include "configurationalStressCompute/stress.cc"
#include "createBinObjectsForce.cc"
#include "initPseudoForce.cc"
#include "locateAtomCoreNodesForce.cc"

  namespace internalForce
  {
    void
    initUnmoved(const Triangulation<3, 3> &             triangulation,
                const Triangulation<3, 3> &             serialTriangulation,
                const std::vector<std::vector<double>> &domainBoundingVectors,
                const MPI_Comm &                        mpi_comm,
                DoFHandler<3> &                         dofHandlerForce,
                FESystem<3> &                           FEForce,
                dealii::AffineConstraints<double> &     constraintsForce,
                IndexSet &                              locally_owned_dofsForce,
                IndexSet &locally_relevant_dofsForce)
    {
      dofHandlerForce.clear();
      dofHandlerForce.initialize(triangulation, FEForce);
      dofHandlerForce.distribute_dofs(FEForce);
      locally_owned_dofsForce.clear();
      locally_relevant_dofsForce.clear();
      locally_owned_dofsForce = dofHandlerForce.locally_owned_dofs();
      DoFTools::extract_locally_relevant_dofs(dofHandlerForce,
                                              locally_relevant_dofsForce);

      ///
      constraintsForce.clear();
      constraintsForce.reinit(locally_relevant_dofsForce);
      DoFTools::make_hanging_node_constraints(dofHandlerForce,
                                              constraintsForce);

      // create unitVectorsXYZ
      std::vector<std::vector<double>> unitVectorsXYZ;
      unitVectorsXYZ.resize(3);

      for (int i = 0; i < 3; ++i)
        {
          unitVectorsXYZ[i].resize(3, 0.0);
          unitVectorsXYZ[i][i] = 0.0;
        }

      std::vector<Tensor<1, 3>> offsetVectors;
      // resize offset vectors
      offsetVectors.resize(3);

      for (int i = 0; i < 3; ++i)
        {
          for (int j = 0; j < 3; ++j)
            {
              offsetVectors[i][j] =
                unitVectorsXYZ[i][j] - domainBoundingVectors[i][j];
            }
        }

      std::vector<
        GridTools::PeriodicFacePair<typename DoFHandler<3>::cell_iterator>>
        periodicity_vectorForce;

      const std::array<int, 3> periodic = {dftParameters::periodicX,
                                           dftParameters::periodicY,
                                           dftParameters::periodicZ};


      std::vector<int> periodicDirectionVector;

      for (unsigned int d = 0; d < 3; ++d)
        {
          if (periodic[d] == 1)
            {
              periodicDirectionVector.push_back(d);
            }
        }

      for (int i = 0; i < std::accumulate(periodic.begin(), periodic.end(), 0);
           ++i)
        {
          GridTools::collect_periodic_faces(
            dofHandlerForce,
            /*b_id1*/ 2 * i + 1,
            /*b_id2*/ 2 * i + 2,
            /*direction*/ periodicDirectionVector[i],
            periodicity_vectorForce,
            offsetVectors[periodicDirectionVector[i]]);
        }

      DoFTools::make_periodicity_constraints<DoFHandler<3>>(
        periodicity_vectorForce, constraintsForce);
      constraintsForce.close();

      if (dftParameters::createConstraintsFromSerialDofhandler)
        {
          dealii::AffineConstraints<double> dummy;
          vectorTools::createParallelConstraintMatrixFromSerial(
            serialTriangulation,
            dofHandlerForce,
            mpi_comm,
            domainBoundingVectors,
            constraintsForce,
            dummy);
        }
    }
  } // namespace internalForce

  //
  // constructor
  //
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  forceClass<FEOrder, FEOrderElectro>::forceClass(
    dftClass<FEOrder, FEOrderElectro> *_dftPtr,
    const MPI_Comm &                   mpi_comm_replica)
    : dftPtr(_dftPtr)
    , FEForce(FE_Q<3>(QGaussLobatto<1>(2)), 3)
    , // linear shape function
    mpi_communicator(mpi_comm_replica)
    , n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_comm_replica))
    , this_mpi_process(Utilities::MPI::this_mpi_process(mpi_comm_replica))
    , pcout(std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0))
  {}

  //
  // initialize forceClass object
  //
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  forceClass<FEOrder, FEOrderElectro>::initUnmoved(
    const Triangulation<3, 3> &             triangulation,
    const Triangulation<3, 3> &             serialTriangulation,
    const std::vector<std::vector<double>> &domainBoundingVectors,
    const bool                              isElectrostaticsMesh)
  {
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



  // reinitialize force class object after mesh update
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  forceClass<FEOrder, FEOrderElectro>::initMoved(
    std::vector<const DoFHandler<3> *> &dofHandlerVectorMatrixFree,
    std::vector<const dealii::AffineConstraints<double> *>
      &        constraintsVectorMatrixFree,
    const bool isElectrostaticsMesh)
  {
    if (isElectrostaticsMesh)
      {
        d_dofHandlerForceElectro.distribute_dofs(FEForce);
        dofHandlerVectorMatrixFree.push_back(&d_dofHandlerForceElectro);
        constraintsVectorMatrixFree.push_back(&d_constraintsNoneForceElectro);

        d_forceDofHandlerIndexElectro = dofHandlerVectorMatrixFree.size() - 1;

        if (!dftParameters::floatingNuclearCharges)
          locateAtomCoreNodesForce(d_dofHandlerForceElectro,
                                   d_locally_owned_dofsForceElectro,
                                   d_atomsForceDofsElectro);
      }
    else
      {
        d_dofHandlerForce.distribute_dofs(FEForce);

        dofHandlerVectorMatrixFree.push_back(&d_dofHandlerForce);
        d_forceDofHandlerIndex = dofHandlerVectorMatrixFree.size() - 1;
        constraintsVectorMatrixFree.push_back(&d_constraintsNoneForce);

        if (!dftParameters::floatingNuclearCharges)
          locateAtomCoreNodesForce(d_dofHandlerForce,
                                   d_locally_owned_dofsForce,
                                   d_atomsForceDofs);

        const unsigned int    numberGlobalAtoms = dftPtr->atomLocations.size();
        std::vector<Point<3>> atomPoints;
        for (unsigned int iAtom = 0; iAtom < numberGlobalAtoms; iAtom++)
          {
            Point<3> atomCoor;
            atomCoor[0] = dftPtr->atomLocations[iAtom][2];
            atomCoor[1] = dftPtr->atomLocations[iAtom][3];
            atomCoor[2] = dftPtr->atomLocations[iAtom][4];
            atomPoints.push_back(atomCoor);
          }
      }
  }

  //
  // initialize pseudopotential data for force computation
  //
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  forceClass<FEOrder, FEOrderElectro>::initPseudoData()
  {
    // if(dftParameters::isPseudopotential)
    //	computeElementalNonLocalPseudoOVDataForce();
  }

  // compute forces on atoms corresponding to a Gaussian generator
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  forceClass<FEOrder, FEOrderElectro>::computeAtomsForces(
    const MatrixFree<3, double> &matrixFreeData,
#ifdef DFTFE_WITH_GPU
    kohnShamDFTOperatorCUDAClass<FEOrder, FEOrderElectro>
      &kohnShamDFTEigenOperator,
#endif
    const unsigned int               eigenDofHandlerIndex,
    const unsigned int               smearedChargeQuadratureId,
    const unsigned int               lpspQuadratureIdElectro,
    const MatrixFree<3, double> &    matrixFreeDataElectro,
    const unsigned int               phiTotDofHandlerIndexElectro,
    const distributedCPUVec<double> &phiTotRhoOutElectro,
    const std::map<dealii::CellId, std::vector<double>> &rhoOutValues,
    const std::map<dealii::CellId, std::vector<double>> &gradRhoOutValues,
    const std::map<dealii::CellId, std::vector<double>> &gradRhoOutValuesLpsp,
    const std::map<dealii::CellId, std::vector<double>> &rhoOutValuesElectro,
    const std::map<dealii::CellId, std::vector<double>>
      &rhoOutValuesElectroLpsp,
    const std::map<dealii::CellId, std::vector<double>>
      &gradRhoOutValuesElectro,
    const std::map<dealii::CellId, std::vector<double>>
      &gradRhoOutValuesElectroLpsp,
    const std::map<dealii::CellId, std::vector<double>> &rhoCoreValues,
    const std::map<dealii::CellId, std::vector<double>> &gradRhoCoreValues,
    const std::map<dealii::CellId, std::vector<double>> &hessianRhoCoreValues,
    const std::map<unsigned int, std::map<dealii::CellId, std::vector<double>>>
      &gradRhoCoreAtoms,
    const std::map<unsigned int, std::map<dealii::CellId, std::vector<double>>>
      &                                                  hessianRhoCoreAtoms,
    const std::map<dealii::CellId, std::vector<double>> &pseudoVLocElectro,
    const std::map<unsigned int, std::map<dealii::CellId, std::vector<double>>>
      &                                      pseudoVLocAtomsElectro,
    const dealii::AffineConstraints<double> &hangingPlusPBCConstraintsElectro,
    const vselfBinsManager<FEOrder, FEOrderElectro> &vselfBinsManagerElectro,
    const std::map<dealii::CellId, std::vector<double>> &shadowKSRhoMinValues,
    const std::map<dealii::CellId, std::vector<double>>
      &                              shadowKSGradRhoMinValues,
    const distributedCPUVec<double> &phiRhoMinusApproxRho,
    const bool                       shadowPotentialForce)
  {
    createBinObjectsForce(matrixFreeDataElectro.get_dof_handler(
                            phiTotDofHandlerIndexElectro),
                          d_dofHandlerForceElectro,
                          hangingPlusPBCConstraintsElectro,
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
                                          smearedChargeQuadratureId,
                                          lpspQuadratureIdElectro,
                                          matrixFreeDataElectro,
                                          phiTotDofHandlerIndexElectro,
                                          phiTotRhoOutElectro,
                                          rhoOutValues,
                                          gradRhoOutValues,
                                          gradRhoOutValuesLpsp,
                                          rhoOutValuesElectro,
                                          rhoOutValuesElectroLpsp,
                                          gradRhoOutValuesElectro,
                                          gradRhoOutValuesElectroLpsp,
                                          rhoCoreValues,
                                          gradRhoCoreValues,
                                          hessianRhoCoreValues,
                                          gradRhoCoreAtoms,
                                          hessianRhoCoreAtoms,
                                          pseudoVLocElectro,
                                          pseudoVLocAtomsElectro,
                                          vselfBinsManagerElectro,
                                          shadowKSRhoMinValues,
                                          shadowKSGradRhoMinValues,
                                          phiRhoMinusApproxRho,
                                          shadowPotentialForce);

    MPI_Barrier(MPI_COMM_WORLD);
    double gaussian_time = MPI_Wtime();

    if (dftParameters::floatingNuclearCharges)
      computeFloatingAtomsForces();
    else
      computeAtomsForcesGaussianGenerator(d_allowGaussianOverlapOnAtoms);

    MPI_Barrier(MPI_COMM_WORLD);
    gaussian_time = MPI_Wtime() - gaussian_time;

    if (this_mpi_process == 0 && dftParameters::verbosity >= 4)
      std::cout
        << "Time for contraction of nodal foces with gaussian generator: "
        << gaussian_time << std::endl;
  }


  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  forceClass<FEOrder, FEOrderElectro>::configForceLinFEInit(
    const MatrixFree<3, double> &matrixFreeData,
    const MatrixFree<3, double> &matrixFreeDataElectro)
  {
    matrixFreeData.initialize_dof_vector(d_configForceVectorLinFE,
                                         d_forceDofHandlerIndex);
    d_configForceVectorLinFE = 0;

    matrixFreeDataElectro.initialize_dof_vector(d_configForceVectorLinFEElectro,
                                                d_forceDofHandlerIndexElectro);
    d_configForceVectorLinFEElectro = 0;

#ifdef USE_COMPLEX
    matrixFreeData.initialize_dof_vector(d_configForceVectorLinFEKPoints,
                                         d_forceDofHandlerIndex);
    d_configForceVectorLinFEKPoints = 0;
#endif

    d_forceAtomsFloating.clear();
#ifdef USE_COMPLEX
    d_forceAtomsFloatingKPoints.clear();
#endif

    const int numberGlobalAtoms = dftPtr->atomLocations.size();
    d_forceAtomsFloating.resize(3 * numberGlobalAtoms, 0.0);
#ifdef USE_COMPLEX
    d_forceAtomsFloatingKPoints.resize(3 * numberGlobalAtoms, 0.0);
#endif
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  forceClass<FEOrder, FEOrderElectro>::configForceLinFEFinalize()
  {
    d_configForceVectorLinFE.compress(
      VectorOperation::add); // copies the ghost element cache to the owning
                             // element
    // d_configForceVectorLinFE.update_ghost_values();
    d_constraintsNoneForce.distribute(
      d_configForceVectorLinFE); // distribute to constrained degrees of freedom
                                 // (for example periodic)
    d_configForceVectorLinFE.update_ghost_values();


    d_configForceVectorLinFEElectro.compress(
      VectorOperation::add); // copies the ghost element cache to the owning
                             // element
    // d_configForceVectorLinFE.update_ghost_values();
    d_constraintsNoneForceElectro.distribute(
      d_configForceVectorLinFEElectro); // distribute to constrained degrees of
                                        // freedom (for example periodic)
    d_configForceVectorLinFEElectro.update_ghost_values();

#ifdef USE_COMPLEX
    d_configForceVectorLinFEKPoints.compress(
      VectorOperation::add); // copies the ghost element cache to the owning
                             // element
    // d_configForceVectorLinFEKPoints.update_ghost_values();
    d_constraintsNoneForce.distribute(
      d_configForceVectorLinFEKPoints); // distribute to constrained degrees of
                                        // freedom (for example periodic)
    d_configForceVectorLinFEKPoints.update_ghost_values();
#endif
  }

  // compute configurational force on the finite element nodes corresponding to
  // linear shape function
  // generators. This function is generic to all-electron and pseudopotential as
  // well as non-periodic and periodic
  // cases. Also both LDA and GGA exchange correlation are handled. For details
  // of the configurational force expressions refer to the Configurational force
  // paper by Motamarri et.al. (https://arxiv.org/abs/1712.05535)
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  forceClass<FEOrder, FEOrderElectro>::computeConfigurationalForceTotalLinFE(
    const MatrixFree<3, double> &matrixFreeData,
#ifdef DFTFE_WITH_GPU
    kohnShamDFTOperatorCUDAClass<FEOrder, FEOrderElectro>
      &kohnShamDFTEigenOperator,
#endif
    const unsigned int               eigenDofHandlerIndex,
    const unsigned int               smearedChargeQuadratureId,
    const unsigned int               lpspQuadratureIdElectro,
    const MatrixFree<3, double> &    matrixFreeDataElectro,
    const unsigned int               phiTotDofHandlerIndexElectro,
    const distributedCPUVec<double> &phiTotRhoOutElectro,
    const std::map<dealii::CellId, std::vector<double>> &rhoOutValues,
    const std::map<dealii::CellId, std::vector<double>> &gradRhoOutValues,
    const std::map<dealii::CellId, std::vector<double>> &gradRhoOutValuesLpsp,
    const std::map<dealii::CellId, std::vector<double>> &rhoOutValuesElectro,
    const std::map<dealii::CellId, std::vector<double>>
      &rhoOutValuesElectroLpsp,
    const std::map<dealii::CellId, std::vector<double>>
      &gradRhoOutValuesElectro,
    const std::map<dealii::CellId, std::vector<double>>
      &gradRhoOutValuesElectroLpsp,
    const std::map<dealii::CellId, std::vector<double>> &rhoCoreValues,
    const std::map<dealii::CellId, std::vector<double>> &gradRhoCoreValues,
    const std::map<dealii::CellId, std::vector<double>> &hessianRhoCoreValues,
    const std::map<unsigned int, std::map<dealii::CellId, std::vector<double>>>
      &gradRhoCoreAtoms,
    const std::map<unsigned int, std::map<dealii::CellId, std::vector<double>>>
      &                                                  hessianRhoCoreAtoms,
    const std::map<dealii::CellId, std::vector<double>> &pseudoVLocElectro,
    const std::map<unsigned int, std::map<dealii::CellId, std::vector<double>>>
      &                                              pseudoVLocAtomsElectro,
    const vselfBinsManager<FEOrder, FEOrderElectro> &vselfBinsManagerElectro,
    const std::map<dealii::CellId, std::vector<double>> &shadowKSRhoMinValues,
    const std::map<dealii::CellId, std::vector<double>>
      &                              shadowKSGradRhoMinValues,
    const distributedCPUVec<double> &phiRhoMinusApproxRho,
    const bool                       shadowPotentialForce)
  {
    configForceLinFEInit(matrixFreeData, matrixFreeDataElectro);

    // configurational force contribution from all terms except those from
    // nuclear self energy
    computeConfigurationalForceEEshelbyTensorFPSPFnlLinFE(
      matrixFreeData,
#ifdef DFTFE_WITH_GPU
      kohnShamDFTEigenOperator,
#endif
      eigenDofHandlerIndex,
      smearedChargeQuadratureId,
      lpspQuadratureIdElectro,
      matrixFreeDataElectro,
      phiTotDofHandlerIndexElectro,
      phiTotRhoOutElectro,
      rhoOutValues,
      gradRhoOutValues,
      gradRhoOutValuesLpsp,
      rhoOutValuesElectro,
      rhoOutValuesElectroLpsp,
      gradRhoOutValuesElectro,
      gradRhoOutValuesElectroLpsp,
      rhoCoreValues,
      gradRhoCoreValues,
      hessianRhoCoreValues,
      gradRhoCoreAtoms,
      hessianRhoCoreAtoms,
      pseudoVLocElectro,
      pseudoVLocAtomsElectro,
      vselfBinsManagerElectro,
      shadowKSRhoMinValues,
      shadowKSGradRhoMinValues,
      phiRhoMinusApproxRho,
      shadowPotentialForce);

    // configurational force contribution from nuclear self energy. This is
    // handled separately as it involves
    // a surface integral over the vself ball surface
    MPI_Barrier(MPI_COMM_WORLD);
    double vselfforce_time = MPI_Wtime();

    if (dealii::Utilities::MPI::this_mpi_process(dftPtr->interBandGroupComm) ==
        0)
      computeConfigurationalForceEselfLinFE(
        matrixFreeDataElectro.get_dof_handler(phiTotDofHandlerIndexElectro),
        vselfBinsManagerElectro,
        matrixFreeDataElectro,
        smearedChargeQuadratureId);

    configForceLinFEFinalize();

    MPI_Barrier(MPI_COMM_WORLD);
    vselfforce_time = MPI_Wtime() - vselfforce_time;

    if (this_mpi_process == 0 && dftParameters::verbosity >= 4)
      std::cout
        << "Time for configurational force computation of Eself contribution and configForceLinFEFinalize(): "
        << vselfforce_time << std::endl;

#ifdef DEBUG
    std::map<std::pair<unsigned int, unsigned int>,
             unsigned int>::const_iterator it;
    for (it = d_atomsForceDofs.begin(); it != d_atomsForceDofs.end(); ++it)
      {
        const std::pair<unsigned int, unsigned int> &atomIdPair   = it->first;
        const unsigned int                           atomForceDof = it->second;
        if (dftParameters::verbosity == 2)
          std::cout << "procid: " << this_mpi_process
                    << " atomId: " << atomIdPair.first
                    << ", force component: " << atomIdPair.second
                    << ", force: " << d_configForceVectorLinFE[atomForceDof]
                    << std::endl;
#  ifdef USE_COMPLEX
        if (dftParameters::verbosity == 2)
          std::cout << "procid: " << this_mpi_process
                    << " atomId: " << atomIdPair.first
                    << ", force component: " << atomIdPair.second
                    << ", forceKPoints: "
                    << d_configForceVectorLinFEKPoints[atomForceDof]
                    << std::endl;
#  endif
      }
#endif
  }


  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  std::vector<double>
  forceClass<FEOrder, FEOrderElectro>::getAtomsForces()
  {
    return d_globalAtomsForces;
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  Tensor<2, 3, double>
  forceClass<FEOrder, FEOrderElectro>::getStress()
  {
    return d_stress;
  }

  /*
     template<unsigned int FEOrder,unsigned int FEOrderElectro>
     double  forceClass<FEOrder>::getGaussianGeneratorParameter() const
     {
     return d_gaussianConstant;
     }

     template<unsigned int FEOrder,unsigned int FEOrderElectro>
     void  forceClass<FEOrder>::updateGaussianConstant(const double
     newGaussianConstant)
     {
     if (!dftParameters::reproducible_output)
     d_gaussianConstant=newGaussianConstant;
     }
   */

#include "force.inst.cc"
} // namespace dftfe
