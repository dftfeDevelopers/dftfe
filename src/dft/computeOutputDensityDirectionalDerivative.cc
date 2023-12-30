// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2018 The Regents of the University of Michigan and DFT-FE
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

#include <dft.h>
#include <densityFirstOrderResponseCalculator.h>

namespace dftfe
{
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  dftClass<FEOrder, FEOrderElectro>::computeOutputDensityDirectionalDerivative(
    const distributedCPUVec<double> &v,
    const distributedCPUVec<double> &vSpin0,
    const distributedCPUVec<double> &vSpin1,
    distributedCPUVec<double> &      fv,
    distributedCPUVec<double> &      fvSpin0,
    distributedCPUVec<double> &      fvSpin1)
  {
    computing_timer.enter_subsection("Output density direction derivative");

    kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>
      &kohnShamDFTEigenOperator = *d_kohnShamDFTOperatorPtr;
#ifdef DFTFE_WITH_DEVICE
    kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro>
      &kohnShamDFTEigenOperatorDevice = *d_kohnShamDFTOperatorDevicePtr;
#endif

    const dealii::Quadrature<3> &quadrature =
      matrix_free_data.get_quadrature(d_densityQuadratureId);

#ifdef DFTFE_WITH_DEVICE
    if (d_dftParamsPtr->useDevice)
      dftfe::utils::MemoryTransfer<dftfe::utils::MemorySpace::DEVICE,
                                   dftfe::utils::MemorySpace::DEVICE>::
        copy(d_eigenVectorsFlattenedDevice.size(),
             d_eigenVectorsDensityMatrixPrimeFlattenedDevice.begin(),
             d_eigenVectorsFlattenedDevice.begin());
#endif
    if (!d_dftParamsPtr->useDevice)
      d_eigenVectorsDensityMatrixPrimeHost = d_eigenVectorsFlattenedHost;


    // set up linear solver
    dealiiLinearSolver CGSolver(d_mpiCommParent,
                                mpi_communicator,
                                dealiiLinearSolver::CG);

#ifdef DFTFE_WITH_DEVICE
    // set up linear solver Device
    linearSolverCGDevice CGSolverDevice(d_mpiCommParent,
                                        mpi_communicator,
                                        linearSolverCGDevice::CG);
#endif


    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST> charge;
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST> dummy;
    v.update_ghost_values();
    interpolateDensityNodalDataToQuadratureDataGeneral(
      basisOperationsPtrElectroHost,
      d_densityDofHandlerIndexElectro,
      d_densityQuadratureIdElectro,
      v,
      charge,
      dummy,
      dummy,
      false,
      false);

    distributedCPUVec<double> electrostaticPotPrime;
    electrostaticPotPrime.reinit(d_phiTotRhoIn);
    electrostaticPotPrime = 0;

    // Reuses diagonalA and mean value constraints
    if (d_dftParamsPtr->useDevice and d_dftParamsPtr->floatingNuclearCharges and
        d_dftParamsPtr->poissonGPU and not d_dftParamsPtr->pinnedNodeForPBC)
      {
#ifdef DFTFE_WITH_DEVICE
        d_phiTotalSolverProblemDevice.reinit(
          basisOperationsPtrElectroHost,
          electrostaticPotPrime,
          *d_constraintsVectorElectro[d_phiTotDofHandlerIndexElectro],
          d_phiTotDofHandlerIndexElectro,
          d_densityQuadratureIdElectro,
          d_phiTotAXQuadratureIdElectro,
          std::map<dealii::types::global_dof_index, double>(),
          dummy,
          d_smearedChargeQuadratureIdElectro,
          charge,
          d_kohnShamDFTOperatorDevicePtr->getDeviceBlasHandle(),
          false,
          false);
#endif
      }
    else
      {
        d_phiTotalSolverProblem.reinit(
          basisOperationsPtrElectroHost,
          electrostaticPotPrime,
          *d_constraintsVectorElectro[d_phiTotDofHandlerIndexElectro],
          d_phiTotDofHandlerIndexElectro,
          d_densityQuadratureIdElectro,
          d_phiTotAXQuadratureIdElectro,
          std::map<dealii::types::global_dof_index, double>(),
          dummy,
          d_smearedChargeQuadratureIdElectro,
          charge,
          false,
          false);
      }

    if (d_dftParamsPtr->useDevice and d_dftParamsPtr->poissonGPU and
        d_dftParamsPtr->floatingNuclearCharges and
        not d_dftParamsPtr->pinnedNodeForPBC)
      {
#ifdef DFTFE_WITH_DEVICE
        CGSolverDevice.solve(
          d_phiTotalSolverProblemDevice,
          d_dftParamsPtr->absPoissonSolverToleranceLRD,
          d_dftParamsPtr->maxLinearSolverIterations,
          d_kohnShamDFTOperatorDevicePtr->getDeviceBlasHandle(),
          d_dftParamsPtr->verbosity);
#endif
      }
    else
      {
        CGSolver.solve(d_phiTotalSolverProblem,
                       d_dftParamsPtr->absPoissonSolverToleranceLRD,
                       d_dftParamsPtr->maxLinearSolverIterations,
                       d_dftParamsPtr->verbosity);
      }

    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      electrostaticPotPrimeValues;
    interpolateElectroNodalDataToQuadratureDataGeneral(
      basisOperationsPtrElectroHost,
      d_phiTotDofHandlerIndexElectro,
      d_densityQuadratureIdElectro,
      electrostaticPotPrime,
      electrostaticPotPrimeValues,
      dummy,
      false);

    // interpolate nodal data to quadrature data
    std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      rhoPrimeValues(2);
    std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      gradRhoPrimeValues(2);
    interpolateDensityNodalDataToQuadratureDataGeneral(
      basisOperationsPtrElectroHost,
      d_densityDofHandlerIndexElectro,
      d_densityQuadratureIdElectro,
      v,
      rhoPrimeValues[0],
      gradRhoPrimeValues[0],
      dummy,
      d_excManagerPtr->getDensityBasedFamilyType() == densityFamilyType::GGA);


    if (d_dftParamsPtr->spinPolarized == 1)
      {
        vSpin0.update_ghost_values();
        vSpin1.update_ghost_values();

        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
          vSpin0Values;
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
          gradvSpin0Values;

        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
          vSpin1Values;
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
          gradvSpin1Values;

        interpolateDensityNodalDataToQuadratureDataGeneral(
          basisOperationsPtrElectroHost,
          d_densityDofHandlerIndexElectro,
          d_densityQuadratureIdElectro,
          vSpin0,
          vSpin0Values,
          gradvSpin0Values,
          dummy,
          d_excManagerPtr->getDensityBasedFamilyType() ==
            densityFamilyType::GGA,
          false);

        interpolateDensityNodalDataToQuadratureDataGeneral(
          basisOperationsPtrElectroHost,
          d_densityDofHandlerIndexElectro,
          d_densityQuadratureIdElectro,
          vSpin1,
          vSpin1Values,
          gradvSpin1Values,
          dummy,
          d_excManagerPtr->getDensityBasedFamilyType() ==
            densityFamilyType::GGA,
          false);

        rhoPrimeValues[0].resize(vSpin0Values.size());
        rhoPrimeValues[1].resize(vSpin0Values.size());
        gradRhoPrimeValues[0].resize(gradvSpin0Values.size());
        gradRhoPrimeValues[1].resize(gradvSpin0Values.size());

        auto &rhoTotalPrimeQuadVals = rhoPrimeValues[0];
        auto &rhoMagPrimeQuadVals   = rhoPrimeValues[1];
        for (unsigned int i = 0; i < vSpin0Values.size(); ++i)
          {
            rhoTotalPrimeQuadVals[i] = vSpin0Values[i] + vSpin1Values[i];
            rhoMagPrimeQuadVals[i]   = vSpin0Values[i] - vSpin1Values[i];
          }

        auto &gradRhoTotalPrimeQuadVals = gradRhoPrimeValues[0];
        auto &gradRhoMagPrimeQuadVals   = gradRhoPrimeValues[1];
        for (unsigned int i = 0; i < vSpin0Values.size(); ++i)
          {
            gradRhoTotalPrimeQuadVals[i] =
              gradvSpin0Values[i] + gradvSpin1Values[i];
            gradRhoMagPrimeQuadVals[i] =
              gradvSpin0Values[i] - gradvSpin1Values[i];
          }
      }

    for (unsigned int s = 0; s < (1 + d_dftParamsPtr->spinPolarized); ++s)
      {
        if (d_excManagerPtr->getDensityBasedFamilyType() ==
            densityFamilyType::LDA)
          {
            computing_timer.enter_subsection("VEffPrime Computation");
#ifdef DFTFE_WITH_DEVICE
            if (d_dftParamsPtr->useDevice)
              {
                if (d_dftParamsPtr->spinPolarized == 1)
                  kohnShamDFTEigenOperatorDevice.computeVEffPrimeSpinPolarized(
                    d_densityInQuadValues,
                    rhoPrimeValuesS,
                    electrostaticPotPrimeValues,
                    s,
                    d_rhoCore);
                else
                  kohnShamDFTEigenOperatorDevice.computeVEffPrime(
                    d_densityInQuadValues,
                    rhoPrimeValues,
                    electrostaticPotPrimeValues,
                    d_rhoCore);
              }
#endif
            if (!d_dftParamsPtr->useDevice)
              {
                if (d_dftParamsPtr->spinPolarized == 1)
                  kohnShamDFTEigenOperator.computeVEffPrimeSpinPolarized(
                    d_densityInQuadValues,
                    rhoPrimeValues,
                    electrostaticPotPrimeValues,
                    s,
                    d_rhoCore);
                else
                  kohnShamDFTEigenOperator.computeVEffPrime(
                    d_densityInQuadValues,
                    rhoPrimeValues,
                    electrostaticPotPrimeValues,
                    d_rhoCore);
              }

            computing_timer.leave_subsection("VEffPrime Computation");
          }
        else if (d_excManagerPtr->getDensityBasedFamilyType() ==
                 densityFamilyType::GGA)
          {
            computing_timer.enter_subsection("VEffPrime Computation");
#ifdef DFTFE_WITH_DEVICE
            if (d_dftParamsPtr->useDevice)
              {
                if (d_dftParamsPtr->spinPolarized == 1)
                  kohnShamDFTEigenOperatorDevice.computeVEffPrimeSpinPolarized(
                    d_densityInQuadValues,
                    rhoPrimeValues,
                    d_gradDensityInQuadValues,
                    gradRhoPrimeValues,
                    electrostaticPotPrimeValues,
                    s,
                    d_rhoCore,
                    d_gradRhoCore);
                else
                  kohnShamDFTEigenOperatorDevice.computeVEffPrime(
                    d_densityInQuadValues,
                    rhoPrimeValues,
                    d_gradDensityInQuadValues,
                    gradRhoPrimeValues,
                    electrostaticPotPrimeValues,
                    d_rhoCore,
                    d_gradRhoCore);
              }
#endif
            if (!d_dftParamsPtr->useDevice)
              {
                if (d_dftParamsPtr->spinPolarized == 1)
                  kohnShamDFTEigenOperator.computeVEffPrimeSpinPolarized(
                    d_densityInQuadValues,
                    rhoPrimeValues,
                    d_gradDensityInQuadValues,
                    gradRhoPrimeValues,
                    electrostaticPotPrimeValues,
                    s,
                    d_rhoCore,
                    d_gradRhoCore);
                else
                  kohnShamDFTEigenOperator.computeVEffPrime(
                    d_densityInQuadValues,
                    rhoPrimeValues,
                    d_gradDensityInQuadValues,
                    gradRhoPrimeValues,
                    electrostaticPotPrimeValues,
                    d_rhoCore,
                    d_gradRhoCore);
              }

            computing_timer.leave_subsection("VEffPrime Computation");
          }

        for (unsigned int kPoint = 0; kPoint < d_kPointWeights.size(); ++kPoint)
          {
            if (kPoint == 0)
              {
#ifdef DFTFE_WITH_DEVICE
                if (d_dftParamsPtr->useDevice)
                  kohnShamDFTEigenOperatorDevice.reinitkPointSpinIndex(kPoint,
                                                                       s);
#endif
                if (!d_dftParamsPtr->useDevice)
                  kohnShamDFTEigenOperator.reinitkPointSpinIndex(kPoint, s);

                computing_timer.enter_subsection(
                  "Hamiltonian matrix prime computation");
#ifdef DFTFE_WITH_DEVICE
                if (d_dftParamsPtr->useDevice)
                  kohnShamDFTEigenOperatorDevice
                    .computeHamiltonianMatricesAllkpt(s, true);
#endif
                if (!d_dftParamsPtr->useDevice)
                  kohnShamDFTEigenOperator.computeHamiltonianMatrix(kPoint,
                                                                    s,
                                                                    true);

                computing_timer.leave_subsection(
                  "Hamiltonian matrix prime computation");
              }

#ifdef DFTFE_WITH_DEVICE
            if (d_dftParamsPtr->useDevice)
              kohnShamEigenSpaceFirstOrderDensityMatResponse(
                s,
                kPoint,
                kohnShamDFTEigenOperatorDevice,
                *d_elpaScala,
                d_subspaceIterationSolverDevice);
#endif
            if (!d_dftParamsPtr->useDevice)
              kohnShamEigenSpaceFirstOrderDensityMatResponse(
                s, kPoint, kohnShamDFTEigenOperator, *d_elpaScala);
          }
      }

    computing_timer.enter_subsection(
      "Density first order response computation");

    computeRhoNodalFirstOrderResponseFromPSIAndPSIPrime(
#ifdef DFTFE_WITH_DEVICE
      kohnShamDFTEigenOperatorDevice,
#endif
      kohnShamDFTEigenOperator,
      fv,
      fvSpin0,
      fvSpin1);

    computing_timer.leave_subsection(
      "Density first order response computation");



    computing_timer.leave_subsection("Output density direction derivative");
  }


  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  dftClass<FEOrder, FEOrderElectro>::
    computeRhoNodalFirstOrderResponseFromPSIAndPSIPrime(
#ifdef DFTFE_WITH_DEVICE
      kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro>
        &kohnShamDFTEigenOperatorDevice,
#endif
      kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>
        &                        kohnShamDFTEigenOperatorCPU,
      distributedCPUVec<double> &fv,
      distributedCPUVec<double> &fvSpin0,
      distributedCPUVec<double> &fvSpin1)
  {
    distributedCPUVec<double> fvHam, fvFermiEnergy;
    fvHam.reinit(fv);
    fvFermiEnergy.reinit(fv);
    fvHam         = 0;
    fvFermiEnergy = 0;

    distributedCPUVec<double> fvHamSpin0, fvHamSpin1, fvFermiEnergySpin0,
      fvFermiEnergySpin1;

    if (d_dftParamsPtr->spinPolarized == 1)
      {
        fvHamSpin0.reinit(fv);
        fvHamSpin1.reinit(fv);
        fvFermiEnergySpin0.reinit(fv);
        fvFermiEnergySpin1.reinit(fv);
        fvHamSpin0         = 0;
        fvHamSpin1         = 0;
        fvFermiEnergySpin0 = 0;
        fvFermiEnergySpin1 = 0;
      }

    std::map<dealii::CellId, std::vector<double>>
      rhoResponseHamPRefinedNodalData;
    std::map<dealii::CellId, std::vector<double>>
      rhoResponseFermiEnergyPRefinedNodalData;

    std::map<dealii::CellId, std::vector<double>>
      rhoResponseHamPRefinedNodalDataSpinPolarized;
    std::map<dealii::CellId, std::vector<double>>
      rhoResponseFermiEnergyPRefinedNodalDataSpinPolarized;

    // initialize variables to be used later
    const unsigned int dofs_per_cell =
      d_dofHandlerRhoNodal.get_fe().dofs_per_cell;
    typename dealii::DoFHandler<3>::active_cell_iterator
      cell = d_dofHandlerRhoNodal.begin_active(),
      endc = d_dofHandlerRhoNodal.end();
    const dealii::IndexSet &locallyOwnedDofs =
      d_dofHandlerRhoNodal.locally_owned_dofs();
    const dealii::Quadrature<3> &quadrature_formula =
      matrix_free_data.get_quadrature(d_gllQuadratureId);
    const unsigned int numQuadPoints = quadrature_formula.size();

    // get access to quadrature point coordinates and 2p DoFHandler nodal points
    const std::vector<dealii::Point<3>> &quadraturePointCoor =
      quadrature_formula.get_points();
    const std::vector<dealii::Point<3>> &supportPointNaturalCoor =
      d_dofHandlerRhoNodal.get_fe().get_unit_support_points();
    std::vector<unsigned int> renumberingMap(numQuadPoints);

    // create renumbering map between the numbering order of quadrature points
    // and lobatto support points
    for (unsigned int i = 0; i < numQuadPoints; ++i)
      {
        const dealii::Point<3> &nodalCoor = supportPointNaturalCoor[i];
        for (unsigned int j = 0; j < numQuadPoints; ++j)
          {
            const dealii::Point<3> &quadCoor = quadraturePointCoor[j];
            double                  dist     = quadCoor.distance(nodalCoor);
            if (dist <= 1e-08)
              {
                renumberingMap[i] = j;
                break;
              }
          }
      }

    // allocate the storage to compute 2p nodal values from wavefunctions
    for (; cell != endc; ++cell)
      {
        if (cell->is_locally_owned())
          {
            const dealii::CellId cellId = cell->id();
            rhoResponseHamPRefinedNodalData[cellId] =
              std::vector<double>(numQuadPoints, 0.0);
            rhoResponseFermiEnergyPRefinedNodalData[cellId] =
              std::vector<double>(numQuadPoints, 0.0);

            if (d_dftParamsPtr->spinPolarized == 1)
              {
                const dealii::CellId cellId = cell->id();
                rhoResponseHamPRefinedNodalDataSpinPolarized[cellId] =
                  std::vector<double>(2 * numQuadPoints, 0.0);
                rhoResponseFermiEnergyPRefinedNodalDataSpinPolarized[cellId] =
                  std::vector<double>(2 * numQuadPoints, 0.0);
              }
          }
      }


      // compute first order density response at nodal locations of 2p
      // DoFHandler nodes in each cell
#ifdef DFTFE_WITH_DEVICE
    if (d_dftParamsPtr->useDevice)
      {
        if (d_dftParamsPtr->singlePrecLRD)
          computeRhoFirstOrderResponseDevice<dataTypes::number,
                                             dataTypes::numberFP32>(
            d_eigenVectorsFlattenedDevice.begin(),
            d_eigenVectorsDensityMatrixPrimeFlattenedDevice.begin(),
            d_densityMatDerFermiEnergy,
            d_numEigenValues,
            matrix_free_data.get_vector_partitioner()->locally_owned_size(),
            kohnShamDFTEigenOperatorDevice,
            d_eigenDofHandlerIndex,
            dofHandler,
            matrix_free_data.n_physical_cells(),
            matrix_free_data.get_dofs_per_cell(),
            quadrature_formula.size(),
            d_kPointWeights,
            rhoResponseHamPRefinedNodalData,
            rhoResponseFermiEnergyPRefinedNodalData,
            rhoResponseHamPRefinedNodalDataSpinPolarized,
            rhoResponseFermiEnergyPRefinedNodalDataSpinPolarized,
            d_mpiCommParent,
            interpoolcomm,
            interBandGroupComm,
            *d_dftParamsPtr);
        else
          computeRhoFirstOrderResponseDevice<dataTypes::number,
                                             dataTypes::number>(
            d_eigenVectorsFlattenedDevice.begin(),
            d_eigenVectorsDensityMatrixPrimeFlattenedDevice.begin(),
            d_densityMatDerFermiEnergy,
            d_numEigenValues,
            matrix_free_data.get_vector_partitioner()->locally_owned_size(),
            kohnShamDFTEigenOperatorDevice,
            d_eigenDofHandlerIndex,
            dofHandler,
            matrix_free_data.n_physical_cells(),
            matrix_free_data.get_dofs_per_cell(),
            quadrature_formula.size(),
            d_kPointWeights,
            rhoResponseHamPRefinedNodalData,
            rhoResponseFermiEnergyPRefinedNodalData,
            rhoResponseHamPRefinedNodalDataSpinPolarized,
            rhoResponseFermiEnergyPRefinedNodalDataSpinPolarized,
            d_mpiCommParent,
            interpoolcomm,
            interBandGroupComm,
            *d_dftParamsPtr);
      }
#endif
    if (!d_dftParamsPtr->useDevice)
      {
        if (d_dftParamsPtr->singlePrecLRD)
          computeRhoFirstOrderResponseCPUMixedPrec<dataTypes::number,
                                                   dataTypes::numberFP32>(
            d_eigenVectorsFlattenedHost.data(),
            d_eigenVectorsDensityMatrixPrimeHost.data(),
            d_densityMatDerFermiEnergy,
            d_numEigenValues,
            matrix_free_data.get_vector_partitioner()->locally_owned_size(),
            kohnShamDFTEigenOperatorCPU,
            d_eigenDofHandlerIndex,
            dofHandler,
            matrix_free_data.n_physical_cells(),
            matrix_free_data.get_dofs_per_cell(),
            quadrature_formula.size(),
            d_kPointWeights,
            rhoResponseHamPRefinedNodalData,
            rhoResponseFermiEnergyPRefinedNodalData,
            rhoResponseHamPRefinedNodalDataSpinPolarized,
            rhoResponseFermiEnergyPRefinedNodalDataSpinPolarized,
            d_mpiCommParent,
            interpoolcomm,
            interBandGroupComm,
            *d_dftParamsPtr);
        else
          computeRhoFirstOrderResponseCPU(
            d_eigenVectorsFlattenedHost.data(),
            d_eigenVectorsDensityMatrixPrimeHost.data(),
            d_densityMatDerFermiEnergy,
            d_numEigenValues,
            matrix_free_data.get_vector_partitioner()->locally_owned_size(),
            kohnShamDFTEigenOperatorCPU,
            d_eigenDofHandlerIndex,
            dofHandler,
            matrix_free_data.n_physical_cells(),
            matrix_free_data.get_dofs_per_cell(),
            quadrature_formula.size(),
            d_kPointWeights,
            rhoResponseHamPRefinedNodalData,
            rhoResponseFermiEnergyPRefinedNodalData,
            rhoResponseHamPRefinedNodalDataSpinPolarized,
            rhoResponseFermiEnergyPRefinedNodalDataSpinPolarized,
            d_mpiCommParent,
            interpoolcomm,
            interBandGroupComm,
            *d_dftParamsPtr);
      }

    // copy Lobatto quadrature data to fill in 2p DoFHandler nodal data
    dealii::DoFHandler<3>::active_cell_iterator cellP = d_dofHandlerRhoNodal
                                                          .begin_active(),
                                                endcP =
                                                  d_dofHandlerRhoNodal.end();

    for (; cellP != endcP; ++cellP)
      if (cellP->is_locally_owned())
        {
          std::vector<dealii::types::global_dof_index> cell_dof_indices(
            dofs_per_cell);
          cellP->get_dof_indices(cell_dof_indices);
          const std::vector<double> &nodalValuesResponseHam =
            rhoResponseHamPRefinedNodalData.find(cellP->id())->second;

          const std::vector<double> &nodalValuesResponseFermiEnergy =
            rhoResponseFermiEnergyPRefinedNodalData.find(cellP->id())->second;

          Assert(
            nodalValuesResponseHam.size() == dofs_per_cell,
            dealii::ExcMessage(
              "Number of nodes in 2p DoFHandler does not match with data stored in rhoNodal Values variable"));

          for (unsigned int iNode = 0; iNode < dofs_per_cell; ++iNode)
            {
              const dealii::types::global_dof_index nodeID =
                cell_dof_indices[iNode];
              if (!d_constraintsRhoNodal.is_constrained(nodeID))
                {
                  if (locallyOwnedDofs.is_element(nodeID))
                    {
                      fvHam(nodeID) =
                        nodalValuesResponseHam[renumberingMap[iNode]];
                      fvFermiEnergy(nodeID) =
                        nodalValuesResponseFermiEnergy[renumberingMap[iNode]];
                    }
                }
            }
        }


    fvHam.update_ghost_values();
    fvFermiEnergy.update_ghost_values();

    const double firstOrderResponseFermiEnergy =
      -totalCharge(d_matrixFreeDataPRefined, fvHam) /
      totalCharge(d_matrixFreeDataPRefined, fvFermiEnergy);

    for (unsigned int i = 0; i < fv.local_size(); i++)
      fv.local_element(i) =
        fvHam.local_element(i) +
        firstOrderResponseFermiEnergy * fvFermiEnergy.local_element(i);

    if (d_dftParamsPtr->spinPolarized == 1)
      {
        // copy Lobatto quadrature data to fill in 2p DoFHandler nodal data
        cellP = d_dofHandlerRhoNodal.begin_active();
        endcP = d_dofHandlerRhoNodal.end();

        for (; cellP != endcP; ++cellP)
          if (cellP->is_locally_owned())
            {
              std::vector<dealii::types::global_dof_index> cell_dof_indices(
                dofs_per_cell);
              cellP->get_dof_indices(cell_dof_indices);
              const std::vector<double> &nodalValuesResponseHam =
                rhoResponseHamPRefinedNodalDataSpinPolarized.find(cellP->id())
                  ->second;

              const std::vector<double> &nodalValuesResponseFermiEnergy =
                rhoResponseFermiEnergyPRefinedNodalDataSpinPolarized
                  .find(cellP->id())
                  ->second;

              Assert(
                nodalValuesResponseHam.size() == 2 * dofs_per_cell,
                dealii::ExcMessage(
                  "Number of nodes in 2p DoFHandler does not match with data stored in rhoNodal Values variable"));

              for (unsigned int iNode = 0; iNode < dofs_per_cell; ++iNode)
                {
                  const dealii::types::global_dof_index nodeID =
                    cell_dof_indices[iNode];
                  if (!d_constraintsRhoNodal.is_constrained(nodeID))
                    {
                      if (locallyOwnedDofs.is_element(nodeID))
                        {
                          fvHamSpin0(nodeID) =
                            nodalValuesResponseHam[2 * renumberingMap[iNode]];
                          fvHamSpin1(nodeID) =
                            nodalValuesResponseHam[2 * renumberingMap[iNode] +
                                                   1];
                          fvFermiEnergySpin0(nodeID) =
                            nodalValuesResponseFermiEnergy
                              [2 * renumberingMap[iNode]];
                          fvFermiEnergySpin1(nodeID) =
                            nodalValuesResponseFermiEnergy
                              [2 * renumberingMap[iNode] + 1];
                        }
                    }
                }
            }

        for (unsigned int i = 0; i < fvHamSpin0.local_size(); i++)
          {
            fvSpin0.local_element(i) = fvHamSpin0.local_element(i) +
                                       firstOrderResponseFermiEnergy *
                                         fvFermiEnergySpin0.local_element(i);
            fvSpin1.local_element(i) = fvHamSpin1.local_element(i) +
                                       firstOrderResponseFermiEnergy *
                                         fvFermiEnergySpin1.local_element(i);
          }
      }
  }
#include "dft.inst.cc"
} // namespace dftfe
