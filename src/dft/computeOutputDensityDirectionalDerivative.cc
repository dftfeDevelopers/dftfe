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
  computingTimerStandard.enter_subsection(
    "Output density direction derivative");
  computing_timer.enter_subsection("Output density direction derivative");

  kohnShamDFTOperatorClass<FEOrder, FEOrderElectro> &kohnShamDFTEigenOperator =
    *d_kohnShamDFTOperatorPtr;
#ifdef DFTFE_WITH_GPU
  kohnShamDFTOperatorCUDAClass<FEOrder, FEOrderElectro>
    &kohnShamDFTEigenOperatorCUDA = *d_kohnShamDFTOperatorCUDAPtr;
#endif

  const Quadrature<3> &quadrature =
    matrix_free_data.get_quadrature(d_densityQuadratureId);

#ifdef DFTFE_WITH_GPU
  if (dftParameters::useGPU)
    cudaUtils::copyCUDAVecToCUDAVec(
      d_eigenVectorsFlattenedCUDA.begin(),
      d_eigenVectorsDensityMatrixPrimeFlattenedCUDA.begin(),
      d_eigenVectorsFlattenedCUDA.size());
#endif
  if (!dftParameters::useGPU)
    d_eigenVectorsDensityMatrixPrimeSTL = d_eigenVectorsFlattenedSTL;


  // set up linear solver
  dealiiLinearSolver dealiiCGSolver(mpi_communicator, dealiiLinearSolver::CG);

  std::map<dealii::CellId, std::vector<double>> charge;
  std::map<dealii::CellId, std::vector<double>> dummy;
  v.update_ghost_values();
  interpolateRhoNodalDataToQuadratureDataGeneral(
    d_matrixFreeDataPRefined,
    d_densityDofHandlerIndexElectro,
    d_densityQuadratureIdElectro,
    v,
    charge,
    dummy,
    dummy,
    false);

  distributedCPUVec<double> electrostaticPotPrime;
  electrostaticPotPrime.reinit(d_phiTotRhoIn);
  electrostaticPotPrime = 0;

  // Reuses diagonalA and mean value constraints
  d_phiTotalSolverProblem.reinit(
    d_matrixFreeDataPRefined,
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

  dealiiCGSolver.solve(d_phiTotalSolverProblem,
                       dftParameters::absPoissonSolverToleranceLRJI,
                       dftParameters::maxLinearSolverIterations,
                       dftParameters::verbosity);


  std::map<dealii::CellId, std::vector<double>> electrostaticPotPrimeValues;
  interpolateElectroNodalDataToQuadratureDataGeneral(
    d_matrixFreeDataPRefined,
    d_phiTotDofHandlerIndexElectro,
    d_densityQuadratureIdElectro,
    electrostaticPotPrime,
    electrostaticPotPrimeValues,
    dummy);

  // interpolate nodal data to quadrature data
  std::map<dealii::CellId, std::vector<double>> rhoPrimeValues;
  std::map<dealii::CellId, std::vector<double>> gradRhoPrimeValues;
  interpolateRhoNodalDataToQuadratureDataGeneral(
    d_matrixFreeDataPRefined,
    d_densityDofHandlerIndexElectro,
    d_densityQuadratureIdElectro,
    v,
    rhoPrimeValues,
    gradRhoPrimeValues,
    dummy,
    dftParameters::xcFamilyType == "GGA");

  std::map<dealii::CellId, std::vector<double>> rhoPrimeValuesSpinPolarized;
  std::map<dealii::CellId, std::vector<double>> gradRhoPrimeValuesSpinPolarized;

  if (dftParameters::spinPolarized == 1)
    {
      vSpin0.update_ghost_values();
      vSpin1.update_ghost_values();
      interpolateRhoSpinNodalDataToQuadratureDataGeneral(
        d_matrixFreeDataPRefined,
        d_densityDofHandlerIndexElectro,
        d_densityQuadratureIdElectro,
        vSpin0,
        vSpin1,
        rhoPrimeValuesSpinPolarized,
        gradRhoPrimeValuesSpinPolarized,
        dummy,
        dftParameters::xcFamilyType == "GGA");
    }

  for (unsigned int s = 0; s < (1 + dftParameters::spinPolarized); ++s)
    {
      if (dftParameters::xcFamilyType == "LDA")
        {
        }
      else if (dftParameters::xcFamilyType == "GGA")
        {
          computing_timer.enter_subsection("VEffPrime Computation");
#ifdef DFTFE_WITH_GPU
          if (dftParameters::useGPU)
            {
              if (dftParameters::spinPolarized == 1)
                kohnShamDFTEigenOperatorCUDA.computeVEffPrimeSpinPolarized(
                  *rhoInValuesSpinPolarized,
                  rhoPrimeValuesSpinPolarized,
                  *gradRhoInValuesSpinPolarized,
                  gradRhoPrimeValuesSpinPolarized,
                  electrostaticPotPrimeValues,
                  s,
                  d_rhoCore,
                  d_gradRhoCore);
              else
                kohnShamDFTEigenOperatorCUDA.computeVEffPrime(
                  *rhoInValues,
                  rhoPrimeValues,
                  *gradRhoInValues,
                  gradRhoPrimeValues,
                  electrostaticPotPrimeValues,
                  d_rhoCore,
                  d_gradRhoCore);
            }
#endif
          if (!dftParameters::useGPU)
            {
              if (dftParameters::spinPolarized == 1)
                kohnShamDFTEigenOperator.computeVEffPrimeSpinPolarized(
                  *rhoInValuesSpinPolarized,
                  rhoPrimeValuesSpinPolarized,
                  *gradRhoInValuesSpinPolarized,
                  gradRhoPrimeValuesSpinPolarized,
                  electrostaticPotPrimeValues,
                  s,
                  d_rhoCore,
                  d_gradRhoCore);
              else
                kohnShamDFTEigenOperator.computeVEffPrime(
                  *rhoInValues,
                  rhoPrimeValues,
                  *gradRhoInValues,
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
#ifdef DFTFE_WITH_GPU
              if (dftParameters::useGPU)
                kohnShamDFTEigenOperatorCUDA.reinitkPointSpinIndex(kPoint, s);
#endif
              if (!dftParameters::useGPU)
                kohnShamDFTEigenOperator.reinitkPointSpinIndex(kPoint, s);

              computing_timer.enter_subsection(
                "Hamiltonian matrix prime computation");
#ifdef DFTFE_WITH_GPU
              if (dftParameters::useGPU)
                kohnShamDFTEigenOperatorCUDA.computeHamiltonianMatrix(kPoint,
                                                                      s,
                                                                      true);
#endif
              if (!dftParameters::useGPU)
                kohnShamDFTEigenOperator.computeHamiltonianMatrix(kPoint,
                                                                  s,
                                                                  true);

              computing_timer.leave_subsection(
                "Hamiltonian matrix prime Computation");
            }

#ifdef DFTFE_WITH_GPU
          if (dftParameters::useGPU)
            kohnShamEigenSpaceFirstOrderDensityMatResponse(
              s,
              kPoint,
              kohnShamDFTEigenOperatorCUDA,
              d_elpaScala,
              d_subspaceIterationSolverCUDA);
#endif
          if (!dftParameters::useGPU)
            kohnShamEigenSpaceFirstOrderDensityMatResponse(
              s, kPoint, kohnShamDFTEigenOperator, d_elpaScala);
        }
    }

  computing_timer.enter_subsection("Density first order response computation");

  computeRhoNodalFirstOrderResponseFromPSIAndPSIPrime(
#ifdef DFTFE_WITH_GPU
    kohnShamDFTEigenOperatorCUDA,
#endif
    kohnShamDFTEigenOperator,
    fv,
    fvSpin0,
    fvSpin1);

  computing_timer.leave_subsection("Density first order response computation");



  computing_timer.leave_subsection("Output density direction derivative");
  computingTimerStandard.leave_subsection(
    "Output density direction derivative");
}


template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
dftClass<FEOrder, FEOrderElectro>::
  computeRhoNodalFirstOrderResponseFromPSIAndPSIPrime(
#ifdef DFTFE_WITH_GPU
    kohnShamDFTOperatorCUDAClass<FEOrder, FEOrderElectro>
      &kohnShamDFTEigenOperatorGPU,
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

  if (dftParameters::spinPolarized == 1)
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

  std::map<dealii::CellId, std::vector<double>> rhoResponseHamPRefinedNodalData;
  std::map<dealii::CellId, std::vector<double>>
    rhoResponseFermiEnergyPRefinedNodalData;

  std::map<dealii::CellId, std::vector<double>>
    rhoResponseHamPRefinedNodalDataSpinPolarized;
  std::map<dealii::CellId, std::vector<double>>
    rhoResponseFermiEnergyPRefinedNodalDataSpinPolarized;

  // initialize variables to be used later
  const unsigned int dofs_per_cell =
    d_dofHandlerRhoNodal.get_fe().dofs_per_cell;
  typename DoFHandler<3>::active_cell_iterator cell = d_dofHandlerRhoNodal
                                                        .begin_active(),
                                               endc =
                                                 d_dofHandlerRhoNodal.end();
  const dealii::IndexSet &locallyOwnedDofs =
    d_dofHandlerRhoNodal.locally_owned_dofs();
  const Quadrature<3> &quadrature_formula =
    matrix_free_data.get_quadrature(d_gllQuadratureId);
  const unsigned int numQuadPoints = quadrature_formula.size();

  // get access to quadrature point coordinates and 2p DoFHandler nodal points
  const std::vector<Point<3>> &quadraturePointCoor =
    quadrature_formula.get_points();
  const std::vector<Point<3>> &supportPointNaturalCoor =
    d_dofHandlerRhoNodal.get_fe().get_unit_support_points();
  std::vector<unsigned int> renumberingMap(numQuadPoints);

  // create renumbering map between the numbering order of quadrature points and
  // lobatto support points
  for (unsigned int i = 0; i < numQuadPoints; ++i)
    {
      const Point<3> &nodalCoor = supportPointNaturalCoor[i];
      for (unsigned int j = 0; j < numQuadPoints; ++j)
        {
          const Point<3> &quadCoor = quadraturePointCoor[j];
          double          dist     = quadCoor.distance(nodalCoor);
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

          if (dftParameters::spinPolarized == 1)
            {
              const dealii::CellId cellId = cell->id();
              rhoResponseHamPRefinedNodalDataSpinPolarized[cellId] =
                std::vector<double>(2 * numQuadPoints, 0.0);
              rhoResponseFermiEnergyPRefinedNodalDataSpinPolarized[cellId] =
                std::vector<double>(2 * numQuadPoints, 0.0);
            }
        }
    }


    // compute first order density response at nodal locations of 2p DoFHandler
    // nodes in each cell
#ifdef DFTFE_WITH_GPU
  if (dftParameters::useGPU)
    {
      if (dftParameters::singlePrecLRJI)
        computeRhoFirstOrderResponseGPU<dataTypes::numberGPU,
                                        dataTypes::numberFP32GPU>(
          d_eigenVectorsFlattenedCUDA.begin(),
          d_eigenVectorsDensityMatrixPrimeFlattenedCUDA.begin(),
          d_densityMatDerFermiEnergy,
          d_numEigenValues,
          d_eigenVectorsFlattenedSTL[0].size() / d_numEigenValues,
          kohnShamDFTEigenOperatorGPU,
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
          interpoolcomm,
          interBandGroupComm);
      else
        computeRhoFirstOrderResponseGPU<dataTypes::numberGPU,
                                        dataTypes::numberGPU>(
          d_eigenVectorsFlattenedCUDA.begin(),
          d_eigenVectorsDensityMatrixPrimeFlattenedCUDA.begin(),
          d_densityMatDerFermiEnergy,
          d_numEigenValues,
          d_eigenVectorsFlattenedSTL[0].size() / d_numEigenValues,
          kohnShamDFTEigenOperatorGPU,
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
          interpoolcomm,
          interBandGroupComm);
    }
#endif
  if (!dftParameters::useGPU)
    {
      if (dftParameters::singlePrecLRJI)
        computeRhoFirstOrderResponseCPUMixedPrec<dataTypes::number,
                                                 dataTypes::numberFP32>(
          d_eigenVectorsFlattenedSTL,
          d_eigenVectorsDensityMatrixPrimeSTL,
          d_densityMatDerFermiEnergy,
          d_numEigenValues,
          d_eigenVectorsFlattenedSTL[0].size() / d_numEigenValues,
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
          interpoolcomm,
          interBandGroupComm);
      else
        computeRhoFirstOrderResponseCPU(
          d_eigenVectorsFlattenedSTL,
          d_eigenVectorsDensityMatrixPrimeSTL,
          d_densityMatDerFermiEnergy,
          d_numEigenValues,
          d_eigenVectorsFlattenedSTL[0].size() / d_numEigenValues,
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
          interpoolcomm,
          interBandGroupComm);
    }

  // copy Lobatto quadrature data to fill in 2p DoFHandler nodal data
  DoFHandler<3>::active_cell_iterator cellP =
                                        d_dofHandlerRhoNodal.begin_active(),
                                      endcP = d_dofHandlerRhoNodal.end();

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
          ExcMessage(
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

  if (dftParameters::spinPolarized == 1)
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
              ExcMessage(
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
                          nodalValuesResponseHam[2 * renumberingMap[iNode] + 1];
                        fvFermiEnergySpin0(nodeID) =
                          nodalValuesResponseFermiEnergy[2 *
                                                         renumberingMap[iNode]];
                        fvFermiEnergySpin1(nodeID) =
                          nodalValuesResponseFermiEnergy
                            [2 * renumberingMap[iNode] + 1];
                      }
                  }
              }
          }

      for (unsigned int i = 0; i < fvHamSpin0.local_size(); i++)
        {
          fvSpin0.local_element(i) =
            fvHamSpin0.local_element(i) +
            firstOrderResponseFermiEnergy * fvFermiEnergySpin0.local_element(i);
          fvSpin1.local_element(i) =
            fvHamSpin1.local_element(i) +
            firstOrderResponseFermiEnergy * fvFermiEnergySpin1.local_element(i);
        }
    }
}
