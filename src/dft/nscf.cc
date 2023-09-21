// ---------------------------------------------------------------------
//
// Copyright (c) 2017 The Regents of the University of Michigan and DFT-FE
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
// @author Krishnendu Ghosh(2018)
//

#include <boost/math/distributions/normal.hpp>
#include <boost/math/special_functions/spherical_harmonic.hpp>
#include <boost/random/normal_distribution.hpp>

#include <fstream>
#include <dft.h>

namespace dftfe
{
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  dftClass<FEOrder, FEOrderElectro>::initnscf(
    kohnShamDFTOperatorClass<FEOrder, FEOrderElectro> &kohnShamDFTEigenOperator,
    poissonSolverProblem<FEOrder, FEOrderElectro> &    phiTotalSolverProblem,
    dealiiLinearSolver &                               CGSolver)
  {
    //
    const dealii::IndexSet &locallyOwnedSet = dofHandler.locally_owned_dofs();
    std::vector<dealii::IndexSet::size_type> locallyOwnedDOFs;
    locallyOwnedSet.fill_index_vector(locallyOwnedDOFs);
    unsigned int numberDofs = locallyOwnedDOFs.size();
    //
    //
    unsigned int d_maxkPoints = d_kPointWeights.size();
    //
    // clear previous scf allocations
    eigenValues.clear();
    a0.clear();
    bLow.clear();
    d_eigenVectorsFlattenedHost.clear();
    waveFunctionsVector.clear();
    numElectrons = 0;
    d_numEigenValues =
      d_numEigenValues + std::max(10, (int)(d_numEigenValues / 10));
    //
    // set size of eigenvalues and eigenvectors data structures
    eigenValues.resize(d_maxkPoints);
    a0.resize(d_maxkPoints, 0.0);
    bLow.resize(d_maxkPoints, 0.0);
    d_eigenVectorsFlattenedHost.resize(
      d_maxkPoints * d_numEigenValues *
      matrix_free_data.get_vector_partitioner()->locally_owned_size());
    //
    pcout << " check 0.1 " << std::endl;
    //
    for (unsigned int kPoint = 0; kPoint < d_maxkPoints; ++kPoint)
      eigenValues[kPoint].resize(d_numEigenValues);
    //
    if (d_dftParamsPtr->isPseudopotential)
      computeElementalOVProjectorKets();

    pcout << " check 0.2 " << std::endl;
    determineOrbitalFilling();
    pcout << " check 0.3: " << std::endl;
    readPSI();
    //
    // MPI_Barrier(d_mpiCommParent) ;
    pcout << " check 0.4 " << std::endl;
    //
    // -------------------------------------------------------------------------
    // Get SCF charge-density ------------------------------------------
    //
    double norm;
    char   buffer[100];
    norm = mixing_anderson();
    //
    if (d_dftParamsPtr->verbosity >= 1)
      pcout << "Anderson mixing: L2 norm of electron-density difference: "
            << norm << std::endl;
    d_phiTotRhoIn = d_phiTotRhoOut;
    //
    //---------------------------------------------------------------------------
    // Get SCF potential --------------------------------------------
    //
    if (d_dftParamsPtr->verbosity == 2)
      pcout << std::endl
            << "Poisson solve for total electrostatic potential (rhoIn+b): ";
    computing_timer.enter_subsection("nscf: phiTot solve");
    //
    std::map<dealii::CellId, std::vector<double>> dummy;
    phiTotalSolverProblem.reinit(
      d_matrixFreeDataPRefined,
      d_phiTotRhoIn,
      *d_constraintsVectorElectro[d_phiTotDofHandlerIndexElectro],
      d_phiTotDofHandlerIndexElectro,
      d_densityQuadratureIdElectro,
      d_phiTotAXQuadratureIdElectro,
      d_atomNodeIdToChargeMap,
      dummy,
      d_smearedChargeQuadratureIdElectro,
      *rhoInValues,
      false);

    std::map<dealii::CellId, std::vector<double>> phiInValues;

    CGSolver.solve(phiTotalSolverProblem,
                   d_dftParamsPtr->absLinearSolverTolerance,
                   d_dftParamsPtr->maxLinearSolverIterations,
                   d_dftParamsPtr->verbosity);

    std::map<dealii::CellId, std::vector<double>> dummy2;
    interpolateRhoNodalDataToQuadratureDataGeneral(
      d_matrixFreeDataPRefined,
      d_phiTotDofHandlerIndexElectro,
      d_densityQuadratureIdElectro,
      d_phiTotRhoIn,
      phiInValues,
      dummy2,
      dummy2);

    computing_timer.leave_subsection("nscf: phiTot solve");
    //
    if (d_excManagerPtr->getDensityBasedFamilyType() == densityFamilyType::LDA)
      {
        computing_timer.enter_subsection("nscf: VEff Computation");
        kohnShamDFTEigenOperator.computeVEff(rhoInValues,
                                             phiInValues,
                                             d_pseudoVLoc,
                                             d_rhoCore,
                                             d_lpspQuadratureId);
        computing_timer.leave_subsection("nscf: VEff Computation");
      }
    else if (d_excManagerPtr->getDensityBasedFamilyType() ==
             densityFamilyType::GGA)
      {
        computing_timer.enter_subsection("nscf: VEff Computation");
        kohnShamDFTEigenOperator.computeVEff(rhoInValues,
                                             gradRhoInValues,
                                             phiInValues,
                                             d_pseudoVLoc,
                                             d_rhoCore,
                                             d_gradRhoCore,
                                             d_lpspQuadratureId);
        computing_timer.leave_subsection("nscf: VEff Computation");
      }
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  dftClass<FEOrder, FEOrderElectro>::nscf(
    kohnShamDFTOperatorClass<FEOrder, FEOrderElectro> &kohnShamDFTEigenOperator,
    chebyshevOrthogonalizedSubspaceIterationSolver &   subspaceIterationSolver)
  {
    std::vector<double> residualNormWaveFunctions;
    residualNormWaveFunctions.resize(d_numEigenValues);
    //
    // if the residual norm is greater than adaptiveChebysevFilterPassesTol (a
    // heuristic value)
    // do more passes of chebysev filter till the check passes.
    //
    for (unsigned int kPoint = 0; kPoint < d_kPointWeights.size(); ++kPoint)
      {
        unsigned int count                           = 1;
        double       maxRes                          = 1e+6;
        double       adaptiveChebysevFilterPassesTol = 1.0E-3;
        //
        kohnShamDFTEigenOperator.reinitkPointSpinIndex(kPoint, 0);
        computing_timer.enter_subsection(
          "nscf: Hamiltonian Matrix Computation");
        kohnShamDFTEigenOperator.computeHamiltonianMatrix(kPoint, 0);
        computing_timer.leave_subsection(
          "nscf: Hamiltonian Matrix Computation");
        // MPI_Barrier(d_mpiCommParent);
        //
        computing_timer.enter_subsection("nscf: kohnShamEigenSpaceCompute");
        while (maxRes > adaptiveChebysevFilterPassesTol)
          {
            if (d_dftParamsPtr->verbosity >= 2)
              pcout << "Beginning Chebyshev filter pass " << count << std::endl;
            //
            kohnShamEigenSpaceComputeNSCF(
              0,
              kPoint, // using mappedkPoint to access eigenVectors and
                      // eigenValues only
              kohnShamDFTEigenOperator,
              subspaceIterationSolver,
              residualNormWaveFunctions,
              count);
            maxRes = residualNormWaveFunctions
              [d_numEigenValues - std::max(10, (int)(d_numEigenValues / 10))];
            if (d_dftParamsPtr->verbosity == 2)
              pcout
                << "Maximum residual norm of the highest empty state in the bandstructure "
                << maxRes << std::endl;
            count++;
          }
        computing_timer.leave_subsection("nscf: kohnShamEigenSpaceCompute");
      }

    // writeBands() ;
  }
#include "dft.inst.cc"
} // namespace dftfe
