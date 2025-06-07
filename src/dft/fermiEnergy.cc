// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
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
// @author Shiva Rudraraju, Phani Motamarri
//

#include <dft.h>

namespace dftfe
{
  namespace internal
  {
    double
    FermiDiracFunctionValue(const double                            x,
                            const std::vector<std::vector<double>> &eigenValues,
                            const std::vector<double> &kPointWeights,
                            const double              &TVal,
                            const dftParameters       &dftParams)
    {
      dftfe::Int numberkPoints     = eigenValues.size();
      dftfe::Int numberEigenValues = eigenValues[0].size();
      double     functionValue     = 0.0;
      double     temp1, temp2;


      for (dftfe::uInt kPoint = 0; kPoint < numberkPoints; ++kPoint)
        {
          for (dftfe::uInt i = 0; i < numberEigenValues; i++)
            {
              temp1 = (eigenValues[kPoint][i] - x) / (C_kb * TVal);
              if (temp1 <= 0.0)
                {
                  temp2 = 1.0 / (1.0 + exp(temp1));
                  functionValue += (2.0 - dftParams.spinPolarized) *
                                   kPointWeights[kPoint] * temp2;
                }
              else
                {
                  temp2 = 1.0 / (1.0 + exp(-temp1));
                  functionValue += (2.0 - dftParams.spinPolarized) *
                                   kPointWeights[kPoint] * exp(-temp1) * temp2;
                }
            }
        }

      return functionValue;
    }

    double
    FermiDiracFunctionDerivativeValue(
      const double                            x,
      const std::vector<std::vector<double>> &eigenValues,
      const std::vector<double>              &kPointWeights,
      const double                           &TVal,
      const dftParameters                    &dftParams)
    {
      dftfe::Int numberkPoints      = eigenValues.size();
      dftfe::Int numberEigenValues  = eigenValues[0].size();
      double     functionDerivative = 0.0;
      double     temp1, temp2;

      for (dftfe::uInt kPoint = 0; kPoint < numberkPoints; ++kPoint)
        {
          for (dftfe::uInt i = 0; i < numberEigenValues; i++)
            {
              temp1 = (eigenValues[kPoint][i] - x) / (C_kb * TVal);
              if (temp1 <= 0.0)
                {
                  temp2 = 1.0 / (1.0 + exp(temp1));
                  functionDerivative +=
                    (2.0 - dftParams.spinPolarized) * kPointWeights[kPoint] *
                    (exp(temp1) / (C_kb * TVal)) * temp2 * temp2;
                }
              else
                {
                  temp2 = 1.0 / (1.0 + exp(-temp1));
                  functionDerivative +=
                    (2.0 - dftParams.spinPolarized) * kPointWeights[kPoint] *
                    (exp(-temp1) / (C_kb * TVal)) * temp2 * temp2;
                }
            }
        }

      return functionDerivative;
    }

  } // namespace internal

  // compute fermi energy
  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::compute_fermienergy(
    const std::vector<std::vector<double>> &eigenValuesInput,
    const double                            numElectronsInput)
  {
    dftfe::Int count = std::ceil(static_cast<double>(numElectronsInput) /
                                 (2.0 - d_dftParamsPtr->spinPolarized));
    double     TVal  = d_dftParamsPtr->TVal;


    std::vector<double> eigenValuesAllkPoints;
    for (dftfe::Int kPoint = 0; kPoint < d_kPointWeights.size(); ++kPoint)
      {
        for (dftfe::Int statesIter = 0; statesIter < eigenValuesInput[0].size();
             ++statesIter)
          {
            eigenValuesAllkPoints.push_back(
              eigenValuesInput[kPoint][statesIter]);
          }
      }

    std::sort(eigenValuesAllkPoints.begin(), eigenValuesAllkPoints.end());

    dftfe::uInt maxNumberFermiEnergySolveIterations = 100;
    double      fe;
    double      R = 1.0;

#ifdef USE_COMPLEX
    //
    // compute Fermi-energy first by bisection method
    //
    // double initialGuessLeft =
    // dealii::Utilities::MPI::min(eigenValuesAllkPoints[0],interpoolcomm);
    // double initialGuessRight =
    // dealii::Utilities::MPI::max(eigenValuesAllkPoints[eigenValuesAllkPoints.size()
    // - 1],interpoolcomm);

    double initialGuessLeft = eigenValuesAllkPoints[0];
    double initialGuessRight =
      eigenValuesAllkPoints[eigenValuesAllkPoints.size() - 1];


    double xLeft, xRight;

    xRight = dealii::Utilities::MPI::max(initialGuessRight, interpoolcomm);
    xLeft  = dealii::Utilities::MPI::min(initialGuessLeft, interpoolcomm);


    for (dftfe::Int iter = 0; iter < maxNumberFermiEnergySolveIterations;
         ++iter)
      {
        double yRightLocal = internal::FermiDiracFunctionValue(
          xRight, eigenValuesInput, d_kPointWeights, TVal, *d_dftParamsPtr);

        double yRight = dealii::Utilities::MPI::sum(yRightLocal, interpoolcomm);

        yRight -= (double)numElectrons;

        double yLeftLocal = internal::FermiDiracFunctionValue(
          xLeft, eigenValuesInput, d_kPointWeights, TVal, *d_dftParamsPtr);

        double yLeft = dealii::Utilities::MPI::sum(yLeftLocal, interpoolcomm);

        yLeft -= (double)numElectrons;

        if ((yLeft * yRight) > 0.0)
          {
            pcout << " Bisection Method Failed " << std::endl;
            exit(-1);
          }

        double xBisected = (xLeft + xRight) / 2.0;

        double yBisectedLocal = internal::FermiDiracFunctionValue(
          xBisected, eigenValuesInput, d_kPointWeights, TVal, *d_dftParamsPtr);
        double yBisected =
          dealii::Utilities::MPI::sum(yBisectedLocal, interpoolcomm);
        yBisected -= (double)numElectrons;

        if ((yBisected * yLeft) > 0.0)
          xLeft = xBisected;
        else
          xRight = xBisected;

        if (std::abs(yBisected) <= 1.0e-09 ||
            iter == maxNumberFermiEnergySolveIterations - 1)
          {
            fe = xBisected;
            R  = std::abs(yBisected);
            break;
          }
      }
    if (d_dftParamsPtr->verbosity >= 4)
      pcout << "Fermi energy constraint residual (bisection): " << R
            << std::endl;
#else
    fe = eigenValuesAllkPoints[d_kPointWeights.size() * count - 1];
#endif
    //
    // compute residual and find FermiEnergy using Newton-Raphson solve
    //
    // double R = 1.0;
    dftfe::uInt  iter          = 0;
    const double newtonIterTol = 1e-10;
    double       functionValue, functionDerivativeValue;

    while ((std::abs(R) > newtonIterTol) &&
           (iter < maxNumberFermiEnergySolveIterations))
      {
        double functionValueLocal = internal::FermiDiracFunctionValue(
          fe, eigenValuesInput, d_kPointWeights, TVal, *d_dftParamsPtr);
        functionValue =
          dealii::Utilities::MPI::sum(functionValueLocal, interpoolcomm);

        double functionDerivativeValueLocal =
          internal::FermiDiracFunctionDerivativeValue(
            fe, eigenValuesInput, d_kPointWeights, TVal, *d_dftParamsPtr);

        functionDerivativeValue =
          dealii::Utilities::MPI::sum(functionDerivativeValueLocal,
                                      interpoolcomm);


        R = functionValue - numElectrons;
        fe += -R / functionDerivativeValue;
        iter++;
      }

    if (std::abs(R) > newtonIterTol)
      {
        AssertThrow(
          false,
          dealii::ExcMessage(
            "DFT-FE Error: Newton-Raphson iterations failed to converge in Fermi energy computation. Hint: Number of wavefunctions are probably insufficient- try increasing the NUMBER OF KOHN-SHAM WAVEFUNCTIONS input parameter."));
      }

    // set Fermi energy
    fermiEnergy = fe;

    if (d_dftParamsPtr->verbosity >= 4)
      pcout << "Fermi energy constraint residual (Newton-Raphson): "
            << std::abs(R) << std::endl;

    if (d_dftParamsPtr->verbosity >= 2)
      pcout << "Fermi energy                                     : "
            << fermiEnergy << std::endl;


    for (dftfe::uInt kPoint = 0; kPoint < d_kPointWeights.size(); ++kPoint)
      for (dftfe::uInt spinIndex = 0;
           spinIndex < (1 + d_dftParamsPtr->spinPolarized);
           ++spinIndex)
        for (dftfe::uInt iEigenVec = 0; iEigenVec < d_numEigenValues;
             ++iEigenVec)
          d_partialOccupancies[kPoint][d_numEigenValues * spinIndex +
                                       iEigenVec] =
            dftUtils::getPartialOccupancy(
              eigenValues[kPoint][d_numEigenValues * spinIndex + iEigenVec],
              fermiEnergy,
              C_kb,
              d_dftParamsPtr->TVal);
  }


  // compute fermi energy pure state
  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::compute_fermienergy_purestate(
    const std::vector<std::vector<double>> &eigenValuesInput,
    const double                            numElectronsInput)
  {
    dftfe::Int count =
      std::ceil(numElectronsInput / (2.0 - d_dftParamsPtr->spinPolarized));


    std::vector<double> eigenValuesAllkPoints;
    for (dftfe::Int kPoint = 0; kPoint < d_kPointWeights.size(); ++kPoint)
      {
        for (dftfe::Int statesIter = 0; statesIter < eigenValuesInput[0].size();
             ++statesIter)
          {
            eigenValuesAllkPoints.push_back(
              eigenValuesInput[kPoint][statesIter]);
          }
      }

    std::sort(eigenValuesAllkPoints.begin(), eigenValuesAllkPoints.end());

    double fermiEnergyLocal =
      count > 0 ? eigenValuesAllkPoints[count - 1] : -1.0e+15;
    //
    fermiEnergy = dealii::Utilities::MPI::max(fermiEnergyLocal, interpoolcomm);
    if (d_dftParamsPtr->verbosity >= 2)
      {
        pcout << " This is a pure state calculation " << std::endl;
        pcout << "Fermi energy                                    : "
              << fermiEnergy << std::endl;
      }

    for (dftfe::uInt kPoint = 0; kPoint < d_kPointWeights.size(); ++kPoint)
      for (dftfe::uInt spinIndex = 0;
           spinIndex < (1 + d_dftParamsPtr->spinPolarized);
           ++spinIndex)
        for (dftfe::uInt iEigenVec = 0; iEigenVec < d_numEigenValues;
             ++iEigenVec)
          {
            if (eigenValues[kPoint][d_numEigenValues * spinIndex + iEigenVec] >
                fermiEnergy)
              d_partialOccupancies[kPoint][d_numEigenValues * spinIndex +
                                           iEigenVec] = 0.0;
            else
              d_partialOccupancies[kPoint][d_numEigenValues * spinIndex +
                                           iEigenVec] = 1.0;
          }
  }


  // compute fermi energy constrained magnetization pure state
  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::compute_fermienergy_constraintMagnetization_purestate(
    const std::vector<std::vector<double>> &eigenValuesInput)
  {
    dftfe::Int countUp   = numElectronsUp;
    dftfe::Int countDown = numElectronsDown;
    //
    //
    std::vector<double> eigenValuesAllkPointsUp, eigenValuesAllkPointsDown;
    for (dftfe::Int kPoint = 0; kPoint < d_kPointWeights.size(); ++kPoint)
      {
        for (dftfe::Int statesIter = 0; statesIter < d_numEigenValues;
             ++statesIter)
          {
            eigenValuesAllkPointsUp.push_back(
              eigenValuesInput[kPoint][statesIter]);
            eigenValuesAllkPointsDown.push_back(
              eigenValuesInput[kPoint][d_numEigenValues + statesIter]);
          }
      }

    std::sort(eigenValuesAllkPointsUp.begin(), eigenValuesAllkPointsUp.end());
    std::sort(eigenValuesAllkPointsDown.begin(),
              eigenValuesAllkPointsDown.end());

    double fermiEnergyUpLocal =
      countUp > 0 ? eigenValuesAllkPointsUp[countUp - 1] : -1.0e+15;
    double fermiEnergyDownLocal =
      countDown > 0 ? eigenValuesAllkPointsDown[countDown - 1] : -1.0e+15;
    //
    fermiEnergyUp =
      dealii::Utilities::MPI::max(fermiEnergyUpLocal, interpoolcomm);
    fermiEnergyDown =
      dealii::Utilities::MPI::max(fermiEnergyDownLocal, interpoolcomm);
    //
    fermiEnergy = std::max(fermiEnergyUp, fermiEnergyDown);
    //
    if (d_dftParamsPtr->verbosity >= 2)
      {
        pcout << " This is a pure state constrained magnetization calculation "
              << std::endl;
        pcout
          << "Fermi energy for spin up                                    : "
          << fermiEnergyUp << std::endl;
        pcout
          << "Fermi energy for spin down                                    : "
          << fermiEnergyDown << std::endl;
      }

    for (dftfe::uInt kPoint = 0; kPoint < d_kPointWeights.size(); ++kPoint)
      for (dftfe::uInt spinIndex = 0; spinIndex < 2; ++spinIndex)
        {
          const double fermiEnergySpinIndex =
            spinIndex == 0 ? fermiEnergyUp : fermiEnergyDown;
          for (dftfe::uInt iEigenVec = 0; iEigenVec < d_numEigenValues;
               ++iEigenVec)
            {
              if (eigenValues[kPoint][d_numEigenValues * spinIndex +
                                      iEigenVec] > fermiEnergySpinIndex)
                d_partialOccupancies[kPoint][d_numEigenValues * spinIndex +
                                             iEigenVec] = 0.0;
              else
                d_partialOccupancies[kPoint][d_numEigenValues * spinIndex +
                                             iEigenVec] = 1.0;
            }
        }
  }

  // compute fermi energy constrained magnetization
  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::compute_fermienergy_constraintMagnetization(
    const std::vector<std::vector<double>> &eigenValuesInput)
  {
    dftfe::Int countUp   = std::ceil(numElectronsUp);
    dftfe::Int countDown = std::ceil(numElectronsDown);
    double     TVal      = d_dftParamsPtr->TVal;


    //
    //
    std::vector<std::vector<double>> eigenValuesInputUp(
      eigenValuesInput.size());
    std::vector<std::vector<double>> eigenValuesInputDown(
      eigenValuesInput.size());

    std::vector<double> eigenValuesAllkPointsUp, eigenValuesAllkPointsDown;
    for (dftfe::Int kPoint = 0; kPoint < d_kPointWeights.size(); ++kPoint)
      {
        for (dftfe::Int statesIter = 0; statesIter < d_numEigenValues;
             ++statesIter)
          {
            eigenValuesAllkPointsUp.push_back(
              eigenValuesInput[kPoint][statesIter]);
            eigenValuesAllkPointsDown.push_back(
              eigenValuesInput[kPoint][d_numEigenValues + statesIter]);

            eigenValuesInputUp[kPoint].push_back(
              eigenValuesInput[kPoint][statesIter]);
            eigenValuesInputDown[kPoint].push_back(
              eigenValuesInput[kPoint][d_numEigenValues + statesIter]);
          }
      }

    std::sort(eigenValuesAllkPointsUp.begin(), eigenValuesAllkPointsUp.end());
    std::sort(eigenValuesAllkPointsDown.begin(),
              eigenValuesAllkPointsDown.end());

    dftfe::uInt maxNumberFermiEnergySolveIterations = 100;
    double      fe;
    double      R = 1.0;

#ifdef USE_COMPLEX
    //
    // compute Fermi-energy up first by bisection method
    //
    // double initialGuessLeft =
    // dealii::Utilities::MPI::min(eigenValuesAllkPoints[0],interpoolcomm);
    // double initialGuessRight =
    // dealii::Utilities::MPI::max(eigenValuesAllkPoints[eigenValuesAllkPoints.size()
    // - 1],interpoolcomm);

    double initialGuessLeft = eigenValuesAllkPointsUp[0];
    double initialGuessRight =
      eigenValuesAllkPointsUp[eigenValuesAllkPointsUp.size() - 1];


    double xLeft, xRight;

    xRight = dealii::Utilities::MPI::max(initialGuessRight, interpoolcomm);
    xLeft  = dealii::Utilities::MPI::min(initialGuessLeft, interpoolcomm);


    for (dftfe::Int iter = 0; iter < maxNumberFermiEnergySolveIterations;
         ++iter)
      {
        double yRightLocal = internal::FermiDiracFunctionValue(
          xRight, eigenValuesInputUp, d_kPointWeights, TVal, *d_dftParamsPtr);

        double yRight = dealii::Utilities::MPI::sum(yRightLocal, interpoolcomm);

        yRight -= numElectronsUp;

        double yLeftLocal = internal::FermiDiracFunctionValue(
          xLeft, eigenValuesInputUp, d_kPointWeights, TVal, *d_dftParamsPtr);

        double yLeft = dealii::Utilities::MPI::sum(yLeftLocal, interpoolcomm);

        yLeft -= numElectronsUp;

        if ((yLeft * yRight) > 0.0)
          {
            pcout << " Bisection Method Failed " << std::endl;
            exit(-1);
          }

        double xBisected = (xLeft + xRight) / 2.0;

        double yBisectedLocal =
          internal::FermiDiracFunctionValue(xBisected,
                                            eigenValuesInputUp,
                                            d_kPointWeights,
                                            TVal,
                                            *d_dftParamsPtr);
        double yBisected =
          dealii::Utilities::MPI::sum(yBisectedLocal, interpoolcomm);
        yBisected -= numElectronsUp;

        if ((yBisected * yLeft) > 0.0)
          xLeft = xBisected;
        else
          xRight = xBisected;

        if (std::abs(yBisected) <= 1.0e-10 ||
            iter == maxNumberFermiEnergySolveIterations - 1)
          {
            fe = xBisected;
            R  = std::abs(yBisected);
            break;
          }
      }
    if (d_dftParamsPtr->verbosity >= 4)
      pcout << "Fermi energy up constraint residual (bisection): " << R
            << std::endl;
#else
    fe = eigenValuesAllkPointsUp[d_kPointWeights.size() * countUp - 1];
#endif
    //
    // compute residual and find FermiEnergy using Newton-Raphson solve
    //
    // double R = 1.0;
    dftfe::uInt  iter          = 0;
    const double newtonIterTol = 1e-10;
    double       functionValue, functionDerivativeValue;

    while ((std::abs(R) > newtonIterTol) &&
           (iter < maxNumberFermiEnergySolveIterations))
      {
        double functionValueLocal = internal::FermiDiracFunctionValue(
          fe, eigenValuesInputUp, d_kPointWeights, TVal, *d_dftParamsPtr);
        functionValue =
          dealii::Utilities::MPI::sum(functionValueLocal, interpoolcomm);

        double functionDerivativeValueLocal =
          internal::FermiDiracFunctionDerivativeValue(
            fe, eigenValuesInputUp, d_kPointWeights, TVal, *d_dftParamsPtr);

        functionDerivativeValue =
          dealii::Utilities::MPI::sum(functionDerivativeValueLocal,
                                      interpoolcomm);


        R = functionValue - numElectronsUp;
        fe += -R / functionDerivativeValue;
        iter++;
      }

    if (std::abs(R) > newtonIterTol)
      {
        AssertThrow(
          false,
          dealii::ExcMessage(
            "DFT-FE Error: Newton-Raphson iterations failed to converge in Fermi energy computation. Hint: Number of wavefunctions are probably insufficient- try increasing the NUMBER OF KOHN-SHAM WAVEFUNCTIONS input parameter."));
      }

    // set Fermi energy
    fermiEnergyUp = fe;

    if (d_dftParamsPtr->verbosity >= 4)
      pcout << "Fermi energy up constraint residual (Newton-Raphson): "
            << std::abs(R) << std::endl;

    R = 1.0;
#ifdef USE_COMPLEX
    //
    // compute Fermi-energy down first by bisection method
    //
    // double initialGuessLeft =
    // dealii::Utilities::MPI::min(eigenValuesAllkPoints[0],interpoolcomm);
    // double initialGuessRight =
    // dealii::Utilities::MPI::max(eigenValuesAllkPoints[eigenValuesAllkPoints.size()
    // - 1],interpoolcomm);

    initialGuessLeft = eigenValuesAllkPointsDown[0];
    initialGuessRight =
      eigenValuesAllkPointsDown[eigenValuesAllkPointsDown.size() - 1];


    xRight = dealii::Utilities::MPI::max(initialGuessRight, interpoolcomm);
    xLeft  = dealii::Utilities::MPI::min(initialGuessLeft, interpoolcomm);


    for (dftfe::Int iter = 0; iter < maxNumberFermiEnergySolveIterations;
         ++iter)
      {
        double yRightLocal = internal::FermiDiracFunctionValue(
          xRight, eigenValuesInputDown, d_kPointWeights, TVal, *d_dftParamsPtr);

        double yRight = dealii::Utilities::MPI::sum(yRightLocal, interpoolcomm);

        yRight -= numElectronsDown;

        double yLeftLocal = internal::FermiDiracFunctionValue(
          xLeft, eigenValuesInputDown, d_kPointWeights, TVal, *d_dftParamsPtr);

        double yLeft = dealii::Utilities::MPI::sum(yLeftLocal, interpoolcomm);

        yLeft -= numElectronsDown;

        if ((yLeft * yRight) > 0.0)
          {
            pcout << " Bisection Method Failed " << std::endl;
            exit(-1);
          }

        double xBisected = (xLeft + xRight) / 2.0;

        double yBisectedLocal =
          internal::FermiDiracFunctionValue(xBisected,
                                            eigenValuesInputDown,
                                            d_kPointWeights,
                                            TVal,
                                            *d_dftParamsPtr);
        double yBisected =
          dealii::Utilities::MPI::sum(yBisectedLocal, interpoolcomm);
        yBisected -= numElectronsDown;

        if ((yBisected * yLeft) > 0.0)
          xLeft = xBisected;
        else
          xRight = xBisected;

        if (std::abs(yBisected) <= 1.0e-10 ||
            iter == maxNumberFermiEnergySolveIterations - 1)
          {
            fe = xBisected;
            R  = std::abs(yBisected);
            break;
          }
      }
    if (d_dftParamsPtr->verbosity >= 4)
      pcout << "Fermi energy down constraint residual (bisection): " << R
            << std::endl;
#else
    fe = eigenValuesAllkPointsDown[d_kPointWeights.size() * countDown - 1];
#endif
    //
    // compute residual and find FermiEnergy using Newton-Raphson solve
    //
    iter = 0;

    while ((std::abs(R) > newtonIterTol) &&
           (iter < maxNumberFermiEnergySolveIterations))
      {
        double functionValueLocal = internal::FermiDiracFunctionValue(
          fe, eigenValuesInputDown, d_kPointWeights, TVal, *d_dftParamsPtr);
        functionValue =
          dealii::Utilities::MPI::sum(functionValueLocal, interpoolcomm);

        double functionDerivativeValueLocal =
          internal::FermiDiracFunctionDerivativeValue(
            fe, eigenValuesInputDown, d_kPointWeights, TVal, *d_dftParamsPtr);

        functionDerivativeValue =
          dealii::Utilities::MPI::sum(functionDerivativeValueLocal,
                                      interpoolcomm);


        R = functionValue - numElectronsDown;
        fe += -R / functionDerivativeValue;
        iter++;
      }

    if (std::abs(R) > newtonIterTol)
      {
        AssertThrow(
          false,
          dealii::ExcMessage(
            "DFT-FE Error: Newton-Raphson iterations failed to converge in Fermi energy computation. Hint: Number of wavefunctions are probably insufficient- try increasing the NUMBER OF KOHN-SHAM WAVEFUNCTIONS input parameter."));
      }

    // set Fermi energy
    fermiEnergyDown = fe;

    if (d_dftParamsPtr->verbosity >= 4)
      pcout << "Fermi energy down constraint residual (Newton-Raphson): "
            << std::abs(R) << std::endl;


    //
    fermiEnergy = std::max(fermiEnergyUp, fermiEnergyDown);
    //
    if (d_dftParamsPtr->verbosity >= 2)
      {
        pcout << " This is a constrained magnetization calculation "
              << std::endl;
        pcout
          << "Fermi energy for spin up                                    : "
          << fermiEnergyUp << std::endl;
        pcout
          << "Fermi energy for spin down                                    : "
          << fermiEnergyDown << std::endl;
      }

    for (dftfe::uInt kPoint = 0; kPoint < d_kPointWeights.size(); ++kPoint)
      for (dftfe::uInt spinIndex = 0; spinIndex < 2; ++spinIndex)
        {
          const double fermiEnergySpinIndex =
            spinIndex == 0 ? fermiEnergyUp : fermiEnergyDown;
          for (dftfe::uInt iEigenVec = 0; iEigenVec < d_numEigenValues;
               ++iEigenVec)
            d_partialOccupancies[kPoint][d_numEigenValues * spinIndex +
                                         iEigenVec] =
              dftUtils::getPartialOccupancy(
                eigenValues[kPoint][d_numEigenValues * spinIndex + iEigenVec],
                fermiEnergySpinIndex,
                C_kb,
                d_dftParamsPtr->TVal);
        }
  }
#include "dft.inst.cc"

} // namespace dftfe
