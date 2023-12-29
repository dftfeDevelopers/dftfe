namespace internal
{
  double
  computeFieldTimesDensity(
    const std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>
      &                                                  basisOperationsPtr,
    const unsigned int                                   quadratureId,
    const std::map<dealii::CellId, std::vector<double>> &fieldValues,
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &densityQuadValues)
  {
    double result = 0.0;
    basisOperationsPtr->reinit(0, 0, quadratureId, false);
    const unsigned int nQuadsPerCell = basisOperationsPtr->nQuadsPerCell();
    for (unsigned int iCell = 0; iCell < basisOperationsPtr->nCells(); ++iCell)
      {
        const std::vector<double> &cellFieldValues =
          fieldValues.find(basisOperationsPtr->cellID(iCell))->second;
        for (unsigned int iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
          result += cellFieldValues[iQuad] *
                    densityQuadValues[iCell * nQuadsPerCell + iQuad] *
                    basisOperationsPtr->JxW()[iCell * nQuadsPerCell + iQuad];
      }
    return result;
  }
  double
  computeFieldTimesDensity(
    const std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>
      &                basisOperationsPtr,
    const unsigned int quadratureId,
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &fieldValues,
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &densityQuadValues)
  {
    double result = 0.0;
    basisOperationsPtr->reinit(0, 0, quadratureId, false);
    const unsigned int nQuadsPerCell = basisOperationsPtr->nQuadsPerCell();
    for (unsigned int iCell = 0; iCell < basisOperationsPtr->nCells(); ++iCell)
      {
        for (unsigned int iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
          result += fieldValues[iCell * nQuadsPerCell + iQuad] *
                    densityQuadValues[iCell * nQuadsPerCell + iQuad] *
                    basisOperationsPtr->JxW()[iCell * nQuadsPerCell + iQuad];
      }
    return result;
  }
} // namespace internal

double
energyCalculator::computeXCEnergyTermsSpinPolarized(
  const std::shared_ptr<
    dftfe::basis::
      FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>
    &                basisOperationsPtr,
  const unsigned int quadratureId,
  const excManager * excManagerPtr,
  const std::vector<
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
    &densityInValues,
  const std::vector<
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
    &densityOutValues,
  const std::vector<
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
    &gradDensityInValues,
  const std::vector<
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
    &                                                  gradDensityOutValues,
  const std::map<dealii::CellId, std::vector<double>> &rhoCoreValues,
  const std::map<dealii::CellId, std::vector<double>> &gradRhoCoreValues,
  double &                                             exchangeEnergy,
  double &                                             correlationEnergy,
  double &                                             excCorrPotentialTimesRho)
{
  basisOperationsPtr->reinit(0, 0, quadratureId, false);
  const unsigned int  nCells        = basisOperationsPtr->nCells();
  const unsigned int  nQuadsPerCell = basisOperationsPtr->nQuadsPerCell();
  std::vector<double> densityValueInXC(2 * nQuadsPerCell, 0.0);
  std::vector<double> densityValueOutXC(2 * nQuadsPerCell, 0.0);
  std::vector<double> exchangeEnergyDensity(nQuadsPerCell, 0.0);
  std::vector<double> corrEnergyDensity(nQuadsPerCell, 0.0);
  std::vector<double> derExchEnergyWithInputDensity(2 * nQuadsPerCell, 0.0);
  std::vector<double> derCorrEnergyWithInputDensity(2 * nQuadsPerCell, 0.0);
  std::vector<double> derExchEnergyWithSigmaGradDenInput,
    derCorrEnergyWithSigmaGradDenInput;
  std::vector<double> sigmaWithOutputGradDensity, sigmaWithInputGradDensity;
  std::vector<double> gradXCRhoInDotgradRhoOut;
  std::vector<std::vector<double>> gradRhoIn, gradRhoOut;
  if (excManagerPtr->getDensityBasedFamilyType() == densityFamilyType::GGA)
    {
      derExchEnergyWithSigmaGradDenInput.resize(3 * nQuadsPerCell);
      derCorrEnergyWithSigmaGradDenInput.resize(3 * nQuadsPerCell);
      sigmaWithOutputGradDensity.resize(3 * nQuadsPerCell);
      sigmaWithInputGradDensity.resize(3 * nQuadsPerCell);
      gradXCRhoInDotgradRhoOut.resize(3 * nQuadsPerCell);
    }
  auto dot3 = [](const std::array<double, 3> &a,
                 const std::array<double, 3> &b) {
    double sum = 0.0;
    for (unsigned int i = 0; i < 3; i++)
      {
        sum += a[i] * b[i];
      }
    return sum;
  };
  for (unsigned int iCell = 0; iCell < nCells; ++iCell)
    {
      auto cellId = basisOperationsPtr->cellID(iCell);
      std::map<rhoDataAttributes, const std::vector<double> *> rhoOutData;
      std::map<rhoDataAttributes, const std::vector<double> *> rhoInData;

      std::map<VeffOutputDataAttributes, std::vector<double> *>
        outputDerExchangeEnergy;
      std::map<VeffOutputDataAttributes, std::vector<double> *>
                                 outputDerCorrEnergy;
      const std::vector<double> &tempRhoCore =
        d_dftParams.nonLinearCoreCorrection ?
          rhoCoreValues.find(cellId)->second :
          NULL;
      const std::vector<double> &tempGradRhoCore =
        (d_dftParams.nonLinearCoreCorrection &&
         excManagerPtr->getDensityBasedFamilyType() == densityFamilyType::GGA) ?
          gradRhoCoreValues.find(cellElectronic->id())->second :
          NULL;
      for (unsigned int iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
        {
          densityValueInXC[2 * iQuad + 0] =
            (densityInValues[0][iCell * nQuadsPerCell + iQuad] +
             densityInValues[1][iCell * nQuadsPerCell + iQuad]) /
            2.0;
          densityValueInXC[2 * iQuad + 1] =
            (densityInValues[0][iCell * nQuadsPerCell + iQuad] -
             densityInValues[1][iCell * nQuadsPerCell + iQuad]) /
            2.0;
          densityValueOutXC[2 * iQuad + 0] =
            (densityOutValues[0][iCell * nQuadsPerCell + iQuad] +
             densityOutValues[1][iCell * nQuadsPerCell + iQuad]) /
            2.0;
          densityValueOutXC[2 * iQuad + 1] =
            (densityOutValues[0][iCell * nQuadsPerCell + iQuad] -
             densityOutValues[1][iCell * nQuadsPerCell + iQuad]) /
            2.0;
          if (d_dftParams.nonLinearCoreCorrection == true)
            {
              densityValueInXC[2 * iQuad + 0] += tempRhoCore[iQuad] / 2.0;
              densityValueInXC[2 * iQuad + 1] += tempRhoCore[iQuad] / 2.0;
              densityValueOutXC[2 * iQuad + 0] += tempRhoCore[iQuad] / 2.0;
              densityValueOutXC[2 * iQuad + 1] += tempRhoCore[iQuad] / 2.0;
            }
        }
      rhoOutData[rhoDataAttributes::values] = &densityValueOutXC;

      rhoInData[rhoDataAttributes::values] = &densityValueInXC;

      outputDerExchangeEnergy[VeffOutputDataAttributes::derEnergyWithDensity] =
        &derExchEnergyWithInputDensity;

      outputDerCorrEnergy[VeffOutputDataAttributes::derEnergyWithDensity] =
        &derCorrEnergyWithInputDensity;

      if (excManagerPtr->getDensityBasedFamilyType() == densityFamilyType::GGA)
        {
          std::array<double, 3> gradXCRhoIn1, gradXCRhoIn2, gradXCRhoOut1,
            gradXCRhoOut2, gradRhoOut1, gradRhoOut2;
          for (unsigned int iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
            {
              for (unsigned int iDim = 0; iDim < 3; ++iDim)
                {
                  gradXCRhoIn1[iDim] =
                    (gradDensityInValues[0][iCell * 3 * nQuadsPerCell +
                                            3 * iQuad + iDim] +
                     gradDensityInValues[1][iCell * 3 * nQuadsPerCell +
                                            3 * iQuad + iDim]) /
                    2.0;
                  gradXCRhoIn2[iDim] =
                    (gradDensityInValues[0][iCell * 3 * nQuadsPerCell +
                                            3 * iQuad + iDim] -
                     gradDensityInValues[1][iCell * 3 * nQuadsPerCell +
                                            3 * iQuad + iDim]) /
                    2.0;
                  gradXCRhoOut1[iDim] =
                    (gradDensityOutValues[0][iCell * 3 * nQuadsPerCell +
                                             3 * iQuad + iDim] +
                     gradDensityOutValues[1][iCell * 3 * nQuadsPerCell +
                                             3 * iQuad + iDim]) /
                    2.0;
                  gradXCRhoOut2[iDim] =
                    (gradDensityOutValues[0][iCell * 3 * nQuadsPerCell +
                                             3 * iQuad + iDim] -
                     gradDensityOutValues[1][iCell * 3 * nQuadsPerCell +
                                             3 * iQuad + iDim]) /
                    2.0;
                }
              gradRhoOut1 = gradXCRhoOut1;
              gradRhoOut2 = gradXCRhoOut2;
              if (d_dftParams.nonLinearCoreCorrection == true)
                {
                  for (unsigned int iDim = 0; iDim < 3; ++iDim)
                    {
                      gradXCRhoIn1[iDim] +=
                        tempGradRhoCore[3 * iQuad + iDim] / 2.0;
                      gradXCRhoIn2[iDim] +=
                        tempGradRhoCore[3 * iQuad + iDim] / 2.0;
                      gradXCRhoOut1[iDim] +=
                        tempGradRhoCore[3 * iQuad + iDim] / 2.0;
                      gradXCRhoOut2[iDim] +=
                        tempGradRhoCore[3 * iQuad + iDim] / 2.0;
                    }
                }
              sigmaWithInputGradDensity[3 * iQuad + 0] =
                dot3(gradXCRhoIn1, gradXCRhoIn1);
              sigmaWithInputGradDensity[3 * iQuad + 1] =
                dot3(gradXCRhoIn1, gradXCRhoIn2);
              sigmaWithInputGradDensity[3 * iQuad + 2] =
                dot3(gradXCRhoIn2, gradXCRhoIn2);
              sigmaWithOutputGradDensity[3 * iQuad + 0] =
                dot3(gradXCRhoOut1, gradXCRhoOut1);
              sigmaWithOutputGradDensity[3 * iQuad + 1] =
                dot3(gradXCRhoOut1, gradXCRhoOut2);
              sigmaWithOutputGradDensity[3 * iQuad + 2] =
                dot3(gradXCRhoOut2, gradXCRhoOut2);
              gradXCRhoInDotgradRhoOut[3 * iQuad + 0] =
                dot3(gradXCRhoIn1, gradRhoOut1);
              gradXCRhoInDotgradRhoOut[3 * iQuad + 1] =
                dot3(gradXCRhoIn1, gradRhoOut2);
              gradXCRhoInDotgradRhoOut[3 * iQuad + 2] =
                dot3(gradXCRhoIn2, gradRhoOut2);
            }
          rhoOutData[rhoDataAttributes::sigmaGradValue] =
            &sigmaWithOutputGradDensity;
          rhoInData[rhoDataAttributes::sigmaGradValue] =
            &sigmaWithInputGradDensity;
          outputDerExchangeEnergy
            [VeffOutputDataAttributes::derEnergyWithSigmaGradDensity] =
              &derExchEnergyWithSigmaGradDenInput;
          outputDerCorrEnergy
            [VeffOutputDataAttributes::derEnergyWithSigmaGradDensity] =
              &derCorrEnergyWithSigmaGradDenInput;
        }
      excManagerPtr->getExcDensityObj()->computeDensityBasedEnergyDensity(
        nQuadsPerCell, rhoOutData, exchangeEnergyDensity, corrEnergyDensity);

      excManagerPtr->getExcDensityObj()->computeDensityBasedVxc(
        nQuadsPerCell, rhoInData, outputDerExchangeEnergy, outputDerCorrEnergy);
      for (unsigned int iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
        {
          double Vxc = derExchEnergyWithInputDensity[2 * iQuad + 0] +
                       derCorrEnergyWithInputDensity[2 * iQuad + 0];
          excCorrPotentialTimesRho +=
            Vxc *
            ((densityInValues[0][iCell * nQuadsPerCell + iQuad] +
              densityInValues[1][iCell * nQuadsPerCell + iQuad]) /
             2.0) *
            basisOperationsPtr->JxW()[iCell * nQuadsPerCell + iQuad];
          Vxc = derExchEnergyWithInputDensity[2 * iQuad + 1] +
                derCorrEnergyWithInputDensity[2 * iQuad + 1];
          excCorrPotentialTimesRho +=
            Vxc *
            ((densityInValues[0][iCell * nQuadsPerCell + iQuad] -
              densityInValues[1][iCell * nQuadsPerCell + iQuad]) /
             2.0) *
            basisOperationsPtr->JxW()[iCell * nQuadsPerCell + iQuad];
          exchangeEnergy +=
            (exchangeEnergyDensity[iQuad]) *
            (densityValueOutXC[2 * iQuad] + densityValueOutXC[2 * iQuad + 1]) *
            basisOperationsPtr->JxW()[iCell * nQuadsPerCell + iQuad];

          correlationEnergy +=
            (corrEnergyDensity[iQuad]) *
            (densityValueOutXC[2 * iQuad] + densityValueOutXC[2 * iQuad + 1]) *
            basisOperationsPtr->JxW()[iCell * nQuadsPerCell + iQuad];
          if (excManagerPtr->getDensityBasedFamilyType() ==
              densityFamilyType::GGA)
            {
              double VxcGrad = 0.0;
              for (unsigned int iDim = 0; iDim < 3; ++iDim)
                VxcGrad +=
                  2.0 *
                  (derExchEnergyWithSigmaGradDenInput[3 * iQuad + iDim] +
                   derCorrEnergyWithSigmaGradDenInput[3 * iQuad + iDim]) *
                  gradXCRhoInDotgradRhoOut[3 * iQuad + iDim];
              excCorrPotentialTimesRho +=
                VxcGrad *
                basisOperationsPtr->JxW()[iCell * nQuadsPerCell + iQuad];
            }
        }
    }
}

// compute energies
double
energyCalculator::computeEnergy(
  const std::shared_ptr<
    dftfe::basis::
      FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>
    &basisOperationsPtr,
  const std::shared_ptr<
    dftfe::basis::
      FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>
    &                                     basisOperationsPtrElectro,
  const unsigned int                      densityQuadratureID,
  const unsigned int                      densityQuadratureIDElectro,
  const unsigned int                      smearedChargeQuadratureIDElectro,
  const unsigned int                      lpspQuadratureIDElectro,
  const std::vector<std::vector<double>> &eigenValues,
  const std::vector<double> &             kPointWeights,
  const double                            fermiEnergy,
  const double                            fermiEnergyUp,
  const double                            fermiEnergyDown,
  const excManager *                      excManagerPtr,
  const dispersionCorrection &            dispersionCorr,
  const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
    &phiTotRhoInValues,
  const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
    &phiTotRhoOutValues,
  const std::vector<
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
    &densityInValues,
  const std::vector<
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
    &densityOutValues,
  const std::vector<
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
    &gradDensityInValues,
  const std::vector<
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
    &gradDensityOutValues,
  const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
    &                                                  rhoOutValuesLpsp,
  const std::map<dealii::CellId, std::vector<double>> &rhoCoreValues,
  const std::map<dealii::CellId, std::vector<double>> &gradRhoCoreValues,
  const std::map<dealii::CellId, std::vector<double>> &smearedbValues,
  const std::map<dealii::CellId, std::vector<unsigned int>>
    &                                     smearedbNonTrivialAtomIds,
  const std::vector<std::vector<double>> &localVselfs,
  const std::map<dealii::CellId, std::vector<double>> &pseudoLocValues,
  const std::map<dealii::types::global_dof_index, double>
    &                atomElectrostaticNodeIdToChargeMap,
  const unsigned int numberGlobalAtoms,
  const unsigned int lowerBoundKindex,
  const unsigned int scfConverged,
  const bool         print,
  const bool         smearedNuclearCharges) const
{
  const dealii::ConditionalOStream scout(
    std::cout,
    (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0));
  const double bandEnergy = dealii::Utilities::MPI::sum(
    internal::localBandEnergy(eigenValues,
                              kPointWeights,
                              fermiEnergy,
                              fermiEnergyUp,
                              fermiEnergyDown,
                              d_dftParams.TVal,
                              d_dftParams.spinPolarized,
                              scout,
                              interpoolcomm,
                              lowerBoundKindex,
                              (d_dftParams.verbosity + scfConverged),
                              d_dftParams),
    interpoolcomm);
  double excCorrPotentialTimesRho = 0.0, electrostaticPotentialTimesRho = 0.0,
         exchangeEnergy = 0.0, correlationEnergy = 0.0,
         electrostaticEnergyTotPot = 0.0;


  electrostaticPotentialTimesRho =
    computeFieldTimesDensity(basisOperationsPtr,
                             densityQuadratureID,
                             phiTotRhoInValues,
                             densityOutValues[0]);
  if (d_dftParams.isPseudopotential || smearedNuclearCharges)
    electrostaticPotentialTimesRho +=
      computeFieldTimesDensity(basisOperationsPtrElectro,
                               lpspQuadratureIDElectro,
                               pseudoLocValues,
                               rhoOutValuesLpsp);
  electrostaticEnergyTotPot =
    0.5 * computeFieldTimesDensity(basisOperationsPtrElectro,
                                   densityQuadratureIDElectro,
                                   phiTotRhoOutValues,
                                   densityOutValues[0]);
  if (d_dftParams.isPseudopotential || smearedNuclearCharges)
    electrostaticEnergyTotPot +=
      computeFieldTimesDensity(basisOperationsPtrElectro,
                               lpspQuadratureIDElectro,
                               pseudoLocValues,
                               rhoOutValuesLpsp);
  if (d_dftParams.spinPolarized == 1)
    computeXCEnergyTermsSpinPolarized(basisOperationsPtr,
                                      densityQuadratureID,
                                      excManagerPtr,
                                      densityInValues,
                                      densityOutValues,
                                      gradDensityInValues,
                                      gradDensityOutValues,
                                      rhoCoreValues,
                                      gradRhoCoreValues,
                                      exchangeEnergy,
                                      correlationEnergy,
                                      excCorrPotentialTimesRho);
  else
    computeXCEnergyTerms(basisOperationsPtr,
                         densityQuadratureID,
                         excManagerPtr,
                         densityInValues,
                         densityOutValues,
                         gradDensityInValues,
                         gradDensityOutValues,
                         rhoCoreValues,
                         gradRhoCoreValues,
                         exchangeEnergy,
                         correlationEnergy,
                         excCorrPotentialTimesRho);
  const double potentialTimesRho =
    excCorrPotentialTimesRho + electrostaticPotentialTimesRho;

  double energy = -potentialTimesRho + exchangeEnergy + correlationEnergy +
                  electrostaticEnergyTotPot;

  const double nuclearElectrostaticEnergy =
    internal::nuclearElectrostaticEnergyLocal(
      phiTotRhoOut,
      localVselfs,
      smearedbValues,
      smearedbNonTrivialAtomIds,
      dofHandlerElectrostatic,
      quadratureElectrostatic,
      quadratureSmearedCharge,
      atomElectrostaticNodeIdToChargeMap,
      smearedNuclearCharges);

  // sum over all processors
  double totalEnergy = dealii::Utilities::MPI::sum(energy, mpi_communicator);
  double totalpotentialTimesRho =
    dealii::Utilities::MPI::sum(potentialTimesRho, mpi_communicator);
  double totalexchangeEnergy =
    dealii::Utilities::MPI::sum(exchangeEnergy, mpi_communicator);
  double totalcorrelationEnergy =
    dealii::Utilities::MPI::sum(correlationEnergy, mpi_communicator);
  double totalelectrostaticEnergyPot =
    dealii::Utilities::MPI::sum(electrostaticEnergyTotPot, mpi_communicator);
  double totalNuclearElectrostaticEnergy =
    dealii::Utilities::MPI::sum(nuclearElectrostaticEnergy, mpi_communicator);

  double d_energyDispersion = 0;
  if (d_dftParams.dc_dispersioncorrectiontype != 0)
    {
      d_energyDispersion = dispersionCorr.getEnergyCorrection();
      totalEnergy += d_energyDispersion;
    }

  //
  // total energy
  //
  totalEnergy += bandEnergy;


  totalEnergy += totalNuclearElectrostaticEnergy;

  const double allElectronElectrostaticEnergy =
    (totalelectrostaticEnergyPot + totalNuclearElectrostaticEnergy);


  double totalkineticEnergy = -totalpotentialTimesRho + bandEnergy;

  // output
  if (print)
    {
      internal::printEnergy(bandEnergy,
                            totalkineticEnergy,
                            totalexchangeEnergy,
                            totalcorrelationEnergy,
                            allElectronElectrostaticEnergy,
                            d_energyDispersion,
                            totalEnergy,
                            numberGlobalAtoms,
                            pcout,
                            d_dftParams.reproducible_output,
                            d_dftParams.isPseudopotential,
                            d_dftParams.verbosity,
                            d_dftParams);
    }

  return totalEnergy;
}
