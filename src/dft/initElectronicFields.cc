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
// @author  Phani Motamarri, Sambit Das
//
#include <dft.h>
#include <dftUtils.h>

namespace dftfe
{
  // init
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  dftClass<FEOrder, FEOrderElectro>::initElectronicFields()
  {
    dealii::TimerOutput::Scope scope(computing_timer, "init electronic fields");

    // reading data from pseudopotential files and fitting splines
    if (d_dftParamsPtr->isPseudopotential)
      initNonLocalPseudoPotential_OV();
    // else
    // initNonLocalPseudoPotential();

    if (d_dftParamsPtr->verbosity >= 4)
      dftUtils::printCurrentMemoryUsage(mpi_communicator,
                                        "Call to initNonLocalPseudoPotential");

    // initialize electrostatics fields
    d_matrixFreeDataPRefined.initialize_dof_vector(
      d_phiTotRhoIn, d_phiTotDofHandlerIndexElectro);
    d_phiTotRhoOut.reinit(d_phiTotRhoIn);
    d_matrixFreeDataPRefined.initialize_dof_vector(
      d_phiExt, d_phiExtDofHandlerIndexElectro);

    d_matrixFreeDataPRefined.initialize_dof_vector(
      d_rhoInNodalValues, d_densityDofHandlerIndexElectro);
    d_rhoOutNodalValues.reinit(d_rhoInNodalValues);
    d_rhoOutNodalValuesSplit.reinit(d_rhoInNodalValues);
    d_rhoOutSpin0NodalValues.reinit(d_rhoInNodalValues);
    d_rhoOutSpin1NodalValues.reinit(d_rhoInNodalValues);
    d_rhoInSpin0NodalValues.reinit(d_rhoInNodalValues);
    d_rhoInSpin1NodalValues.reinit(d_rhoInNodalValues);
    // d_atomicRho.reinit(d_rhoInNodalValues);

    d_rhoInNodalValues       = 0;
    d_rhoOutNodalValues      = 0;
    d_rhoOutNodalValuesSplit = 0;
    d_rhoOutSpin0NodalValues = 0;
    d_rhoOutSpin1NodalValues = 0;
    d_rhoInSpin0NodalValues  = 0;
    d_rhoInSpin1NodalValues  = 0;

    if ((d_dftParamsPtr->reuseDensityGeoOpt == 2 &&
         d_dftParamsPtr->solverMode == "GEOOPT") ||
        (d_dftParamsPtr->extrapolateDensity == 2 &&
         d_dftParamsPtr->solverMode == "MD"))
      {
        initAtomicRho();
      }

    //
    // initialize eigen vectors
    //
    matrix_free_data.initialize_dof_vector(d_tempEigenVec,
                                           d_eigenDofHandlerIndex);

    //
    // store constraintEigen Matrix entries into STL vector
    //
    constraintsNoneEigenDataInfo.initialize(d_tempEigenVec.get_partitioner(),
                                            constraintsNoneEigen);

    constraintsNoneDataInfo.initialize(
      matrix_free_data.get_vector_partitioner(), constraintsNone);

#ifdef DFTFE_WITH_DEVICE
    if (d_dftParamsPtr->useDevice)
      d_constraintsNoneDataInfoDevice.initialize(
        matrix_free_data.get_vector_partitioner(), constraintsNone);
#endif

    if (d_dftParamsPtr->verbosity >= 4)
      dftUtils::printCurrentMemoryUsage(
        mpi_communicator, "Overloaded constraint matrices initialized");

    //
    // initialize density and PSI/ interpolate from previous ground state
    // solution
    //
    for (unsigned int kPoint = 0;
         kPoint < (1 + d_dftParamsPtr->spinPolarized) * d_kPointWeights.size();
         ++kPoint)
      {
        d_eigenVectorsFlattenedSTL[kPoint].resize(
          d_numEigenValues *
            matrix_free_data.get_vector_partitioner()->local_size(),
          dataTypes::number(0.0));

        if (d_numEigenValuesRR != d_numEigenValues)
          {
            d_eigenVectorsRotFracDensityFlattenedSTL[kPoint].resize(
              d_numEigenValuesRR *
                matrix_free_data.get_vector_partitioner()->local_size(),
              dataTypes::number(0.0));
          }
      }

    pcout << std::endl
          << "Setting initial guess for wavefunctions...." << std::endl;

    if (d_dftParamsPtr->verbosity >= 4)
      dftUtils::printCurrentMemoryUsage(
        mpi_communicator,
        "Created flattened array eigenvectors before update ghost values");

    readPSI();

    if (d_dftParamsPtr->verbosity >= 4)
      dftUtils::printCurrentMemoryUsage(mpi_communicator,
                                        "Created flattened array eigenvectors");

    // if(!(d_dftParamsPtr->chkType==2 && d_dftParamsPtr->restartFromChk))
    //{
    initRho();
    // d_rhoOutNodalValues.reinit(d_rhoInNodalValues);
    //}

    if (d_dftParamsPtr->verbosity >= 4)
      dftUtils::printCurrentMemoryUsage(mpi_communicator, "initRho called");

#ifdef DFTFE_WITH_DEVICE
    if (d_dftParamsPtr->useDevice)
      {
        d_eigenVectorsFlattenedDevice.resize(
          d_eigenVectorsFlattenedSTL[0].size() *
          (1 + d_dftParamsPtr->spinPolarized) * d_kPointWeights.size());

        if (d_dftParamsPtr->mixingMethod == "LOW_RANK_DIELECM_PRECOND")
          d_eigenVectorsDensityMatrixPrimeFlattenedDevice.resize(
            d_eigenVectorsFlattenedSTL[0].size() *
            (1 + d_dftParamsPtr->spinPolarized) * d_kPointWeights.size());

        if (d_numEigenValuesRR != d_numEigenValues)
          d_eigenVectorsRotFracFlattenedDevice.resize(
            d_eigenVectorsRotFracDensityFlattenedSTL[0].size() *
            (1 + d_dftParamsPtr->spinPolarized) * d_kPointWeights.size());
        else
          d_eigenVectorsRotFracFlattenedDevice.resize(1);

        for (unsigned int kPoint = 0;
             kPoint <
             (1 + d_dftParamsPtr->spinPolarized) * d_kPointWeights.size();
             ++kPoint)
          {
            d_eigenVectorsFlattenedDevice
              .copyFrom<dftfe::utils::MemorySpace::HOST>(
                &d_eigenVectorsFlattenedSTL[kPoint][0],
                d_eigenVectorsFlattenedSTL[0].size(),
                0,
                kPoint * d_eigenVectorsFlattenedSTL[0].size());
          }
      }
#endif

    if (!d_dftParamsPtr->useDevice &&
        d_dftParamsPtr->mixingMethod == "LOW_RANK_DIELECM_PRECOND")
      {
        d_eigenVectorsDensityMatrixPrimeSTL = d_eigenVectorsFlattenedSTL;
      }

    if (d_dftParamsPtr->verbosity >= 2 && d_dftParamsPtr->spinPolarized == 1)
      pcout << std::endl
            << "net magnetization: "
            << totalMagnetization(rhoInValuesSpinPolarized.get()) << std::endl;
  }
#include "dft.inst.cc"
} // namespace dftfe
