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

#include <exchangeCorrelationFunctionalEvaluator.h>
#include <deviceKernelsGeneric.h>
#include <DeviceAPICalls.h>
#include <DeviceDataTypeOverloads.h>
#include <DeviceTypeConfig.h>
#include <DeviceKernelLauncherHelpers.h>
#include <BLASWrapper.h>

namespace dftfe
{
  namespace
  {
#define DFTFE_FUNCTIONALEVALUATOR_LDA_X(NAME, BODY)                      \
  DFTFE_CREATE_KERNEL(void,                                              \
                      exchangeEvaluationKernel##NAME,                    \
                      DFTFE_KERNEL_ARGUMENT(                             \
                        for (dftfe::uInt index = globalThreadId;         \
                             index < numPoints;                          \
                             index += nThreadsPerBlock * nThreadBlock) { \
                          double tzk0;                                   \
                          double tvrho0, tvrho1;                         \
                          double rho0 = densityValues[2 * index + 0];    \
                          double rho1 = densityValues[2 * index + 1];    \
                          if ((rho0 + rho1) < DENS_THRESHOLD_X_##NAME)   \
                            {                                            \
                              exEnergyOut[index]         = 0.0;          \
                              pdexDensity[2 * index + 0] = 0.0;          \
                              pdexDensity[2 * index + 1] = 0.0;          \
                              continue;                                  \
                            }                                            \
                          rho0 = m_max(DENS_THRESHOLD_X_##NAME, rho0);   \
                          rho1 = m_max(DENS_THRESHOLD_X_##NAME, rho1);   \
                          BODY;                                          \
                          exEnergyOut[index]         = tzk0;             \
                          pdexDensity[2 * index + 0] = tvrho0;           \
                          pdexDensity[2 * index + 1] = tvrho1;           \
                        }),                                              \
                      const dftfe::uInt numPoints,                       \
                      const double     *densityValues,                   \
                      double           *exEnergyOut,                     \
                      double           *pdexDensity);

#define DFTFE_FUNCTIONALEVALUATOR_LDA_C(NAME, BODY)                      \
  DFTFE_CREATE_KERNEL(void,                                              \
                      correlationEvaluationKernel##NAME,                 \
                      DFTFE_KERNEL_ARGUMENT(                             \
                        for (dftfe::uInt index = globalThreadId;         \
                             index < numPoints;                          \
                             index += nThreadsPerBlock * nThreadBlock) { \
                          double tzk0;                                   \
                          double tvrho0, tvrho1;                         \
                          double rho0 = densityValues[2 * index + 0];    \
                          double rho1 = densityValues[2 * index + 1];    \
                          if ((rho0 + rho1) < DENS_THRESHOLD_C_##NAME)   \
                            {                                            \
                              corrEnergyOut[index]       = 0.0;          \
                              pdecDensity[2 * index + 0] = 0.0;          \
                              pdecDensity[2 * index + 1] = 0.0;          \
                              continue;                                  \
                            }                                            \
                          rho0 = m_max(DENS_THRESHOLD_C_##NAME, rho0);   \
                          rho1 = m_max(DENS_THRESHOLD_C_##NAME, rho1);   \
                          BODY;                                          \
                          corrEnergyOut[index]       = tzk0;             \
                          pdecDensity[2 * index + 0] = tvrho0;           \
                          pdecDensity[2 * index + 1] = tvrho1;           \
                        }),                                              \
                      const dftfe::uInt numPoints,                       \
                      const double     *densityValues,                   \
                      double           *corrEnergyOut,                   \
                      double           *pdecDensity);


#define DFTFE_FUNCTIONALEVALUATOR_GGA_X(NAME, BODY)                        \
  DFTFE_CREATE_KERNEL(                                                     \
    void,                                                                  \
    exchangeEvaluationKernel##NAME,                                        \
    DFTFE_KERNEL_ARGUMENT(for (dftfe::uInt index = globalThreadId;         \
                               index < numPoints;                          \
                               index += nThreadsPerBlock * nThreadBlock) { \
      double tzk0;                                                         \
      double tvrho0, tvrho1;                                               \
      double tvsigma0, tvsigma1, tvsigma2;                                 \
      double rho0 = densityValues[2 * index + 0];                          \
      double rho1 = densityValues[2 * index + 1];                          \
      if ((rho0 + rho1) < DENS_THRESHOLD_X_##NAME)                         \
        {                                                                  \
          exEnergyOut[index]         = 0.0;                                \
          pdexDensity[2 * index + 0] = 0.0;                                \
          pdexDensity[2 * index + 1] = 0.0;                                \
          pdexSigma[3 * index + 0]   = 0.0;                                \
          pdexSigma[3 * index + 1]   = 0.0;                                \
          pdexSigma[3 * index + 2]   = 0.0;                                \
          continue;                                                        \
        }                                                                  \
      rho0 = m_max(DENS_THRESHOLD_X_##NAME, rho0);                         \
      rho1 = m_max(DENS_THRESHOLD_X_##NAME, rho1);                         \
      double sigma0 =                                                      \
        m_max(SIGMA_THRESHOLD_X_##NAME * SIGMA_THRESHOLD_X_##NAME,         \
              sigmaValues[3 * index + 0]);                                 \
      double sigma2 =                                                      \
        m_max(SIGMA_THRESHOLD_X_##NAME * SIGMA_THRESHOLD_X_##NAME,         \
              sigmaValues[3 * index + 2]);                                 \
      double sigma1 = sigmaValues[3 * index + 1];                          \
      double s      = 0.5 * (sigma0 + sigma2);                             \
      sigma1        = (sigma1 >= -s ? sigma1 : -s);                        \
      sigma1        = (sigma1 <= s ? sigma1 : s);                          \
      BODY;                                                                \
      exEnergyOut[index]         = tzk0;                                   \
      pdexDensity[2 * index + 0] = tvrho0;                                 \
      pdexDensity[2 * index + 1] = tvrho1;                                 \
      pdexSigma[3 * index + 0]   = tvsigma0;                               \
      pdexSigma[3 * index + 1]   = tvsigma1;                               \
      pdexSigma[3 * index + 2]   = tvsigma2;                               \
    }),                                                                    \
    const dftfe::uInt numPoints,                                           \
    const double     *densityValues,                                       \
    const double     *sigmaValues,                                         \
    double           *exEnergyOut,                                         \
    double           *pdexDensity,                                         \
    double           *pdexSigma);

#define DFTFE_FUNCTIONALEVALUATOR_GGA_C(NAME, BODY)                        \
  DFTFE_CREATE_KERNEL(                                                     \
    void,                                                                  \
    correlationEvaluationKernel##NAME,                                     \
    DFTFE_KERNEL_ARGUMENT(for (dftfe::uInt index = globalThreadId;         \
                               index < numPoints;                          \
                               index += nThreadsPerBlock * nThreadBlock) { \
      double tzk0;                                                         \
      double tvrho0, tvrho1;                                               \
      double tvsigma0, tvsigma1, tvsigma2;                                 \
      double rho0 = densityValues[2 * index + 0];                          \
      double rho1 = densityValues[2 * index + 1];                          \
      if ((rho0 + rho1) < DENS_THRESHOLD_C_##NAME)                         \
        {                                                                  \
          corrEnergyOut[index]       = 0.0;                                \
          pdecDensity[2 * index + 0] = 0.0;                                \
          pdecDensity[2 * index + 1] = 0.0;                                \
          pdecSigma[3 * index + 0]   = 0.0;                                \
          pdecSigma[3 * index + 1]   = 0.0;                                \
          pdecSigma[3 * index + 2]   = 0.0;                                \
          continue;                                                        \
        }                                                                  \
      rho0 = m_max(DENS_THRESHOLD_C_##NAME, rho0);                         \
      rho1 = m_max(DENS_THRESHOLD_C_##NAME, rho1);                         \
      double sigma0 =                                                      \
        m_max(SIGMA_THRESHOLD_C_##NAME * SIGMA_THRESHOLD_C_##NAME,         \
              sigmaValues[3 * index + 0]);                                 \
      double sigma2 =                                                      \
        m_max(SIGMA_THRESHOLD_C_##NAME * SIGMA_THRESHOLD_C_##NAME,         \
              sigmaValues[3 * index + 2]);                                 \
      double sigma1 = sigmaValues[3 * index + 1];                          \
      double s      = 0.5 * (sigma0 + sigma2);                             \
      sigma1        = (sigma1 >= -s ? sigma1 : -s);                        \
      sigma1        = (sigma1 <= s ? sigma1 : s);                          \
      BODY;                                                                \
      corrEnergyOut[index]       = tzk0;                                   \
      pdecDensity[2 * index + 0] = tvrho0;                                 \
      pdecDensity[2 * index + 1] = tvrho1;                                 \
      pdecSigma[3 * index + 0]   = tvsigma0;                               \
      pdecSigma[3 * index + 1]   = tvsigma1;                               \
      pdecSigma[3 * index + 2]   = tvsigma2;                               \
    }),                                                                    \
    const dftfe::uInt numPoints,                                           \
    const double     *densityValues,                                       \
    const double     *sigmaValues,                                         \
    double           *corrEnergyOut,                                       \
    double           *pdecDensity,                                         \
    double           *pdecSigma);

#define DFTFE_FUNCTIONALEVALUATOR_MGGA_X(NAME, BODY)                       \
  DFTFE_CREATE_KERNEL(                                                     \
    void,                                                                  \
    exchangeEvaluationKernel##NAME,                                        \
    DFTFE_KERNEL_ARGUMENT(for (dftfe::uInt index = globalThreadId;         \
                               index < numPoints;                          \
                               index += nThreadsPerBlock * nThreadBlock) { \
      double tzk0;                                                         \
      double tvrho0, tvrho1;                                               \
      double tvsigma0, tvsigma1, tvsigma2;                                 \
      double tvtau0, tvtau1;                                               \
      double rho0 = densityValues[2 * index + 0];                          \
      double rho1 = densityValues[2 * index + 1];                          \
      if ((rho0 + rho1) < DENS_THRESHOLD_X_##NAME)                         \
        {                                                                  \
          exEnergyOut[index]         = 0.0;                                \
          pdexDensity[2 * index + 0] = 0.0;                                \
          pdexDensity[2 * index + 1] = 0.0;                                \
          pdexSigma[3 * index + 0]   = 0.0;                                \
          pdexSigma[3 * index + 1]   = 0.0;                                \
          pdexSigma[3 * index + 2]   = 0.0;                                \
          pdexTau[2 * index + 0]     = 0.0;                                \
          pdexTau[2 * index + 1]     = 0.0;                                \
          continue;                                                        \
        }                                                                  \
      rho0 = m_max(DENS_THRESHOLD_X_##NAME, rho0);                         \
      rho1 = m_max(DENS_THRESHOLD_X_##NAME, rho1);                         \
      double sigma0 =                                                      \
        m_max(SIGMA_THRESHOLD_X_##NAME * SIGMA_THRESHOLD_X_##NAME,         \
              sigmaValues[3 * index + 0]);                                 \
      double sigma2 =                                                      \
        m_max(SIGMA_THRESHOLD_X_##NAME * SIGMA_THRESHOLD_X_##NAME,         \
              sigmaValues[3 * index + 2]);                                 \
      double tau0;                                                         \
      double tau1;                                                         \
      if (tauNeededX)                                                      \
        {                                                                  \
          tau0 = m_max(TAU_THRESHOLD_X_##NAME, tauValues[2 * index + 0]);  \
          tau1 = m_max(TAU_THRESHOLD_X_##NAME, tauValues[2 * index + 1]);  \
          if (enforceFHCX)                                                 \
            {                                                              \
              sigma0 = m_min(sigma0, 8.0 * rho0 * tau0);                   \
              sigma2 = m_min(sigma2, 8.0 * rho1 * tau1);                   \
            }                                                              \
        }                                                                  \
      double sigma1 = sigmaValues[3 * index + 1];                          \
      double s      = 0.5 * (sigma0 + sigma2);                             \
      sigma1        = (sigma1 >= -s ? sigma1 : -s);                        \
      sigma1        = (sigma1 <= s ? sigma1 : s);                          \
      BODY;                                                                \
      exEnergyOut[index]         = tzk0;                                   \
      pdexDensity[2 * index + 0] = tvrho0;                                 \
      pdexDensity[2 * index + 1] = tvrho1;                                 \
      pdexSigma[3 * index + 0]   = tvsigma0;                               \
      pdexSigma[3 * index + 1]   = tvsigma1;                               \
      pdexSigma[3 * index + 2]   = tvsigma2;                               \
      pdexTau[2 * index + 0]     = tvtau0;                                 \
      pdexTau[2 * index + 1]     = tvtau1;                                 \
    }),                                                                    \
    const dftfe::uInt numPoints,                                           \
    const double     *densityValues,                                       \
    const double     *sigmaValues,                                         \
    const double     *tauValues,                                           \
    double           *exEnergyOut,                                         \
    double           *pdexDensity,                                         \
    double           *pdexSigma,                                           \
    double           *pdexTau,                                             \
    bool              tauNeededX,                                          \
    bool              enforceFHCX);

#define DFTFE_FUNCTIONALEVALUATOR_MGGA_C(NAME, BODY)                       \
  DFTFE_CREATE_KERNEL(                                                     \
    void,                                                                  \
    correlationEvaluationKernel##NAME,                                     \
    DFTFE_KERNEL_ARGUMENT(for (dftfe::uInt index = globalThreadId;         \
                               index < numPoints;                          \
                               index += nThreadsPerBlock * nThreadBlock) { \
      double tzk0;                                                         \
      double tvrho0, tvrho1;                                               \
      double tvsigma0, tvsigma1, tvsigma2;                                 \
      double tvtau0, tvtau1;                                               \
      double rho0 = densityValues[2 * index + 0];                          \
      double rho1 = densityValues[2 * index + 1];                          \
      if ((rho0 + rho1) < DENS_THRESHOLD_C_##NAME)                         \
        {                                                                  \
          corrEnergyOut[index]       = 0.0;                                \
          pdecDensity[2 * index + 0] = 0.0;                                \
          pdecDensity[2 * index + 1] = 0.0;                                \
          pdecSigma[3 * index + 0]   = 0.0;                                \
          pdecSigma[3 * index + 1]   = 0.0;                                \
          pdecSigma[3 * index + 2]   = 0.0;                                \
          pdecTau[2 * index + 0]     = 0.0;                                \
          pdecTau[2 * index + 1]     = 0.0;                                \
          continue;                                                        \
        }                                                                  \
      rho0 = m_max(DENS_THRESHOLD_C_##NAME, rho0);                         \
      rho1 = m_max(DENS_THRESHOLD_C_##NAME, rho1);                         \
      double sigma0 =                                                      \
        m_max(SIGMA_THRESHOLD_C_##NAME * SIGMA_THRESHOLD_C_##NAME,         \
              sigmaValues[3 * index + 0]);                                 \
      double sigma2 =                                                      \
        m_max(SIGMA_THRESHOLD_C_##NAME * SIGMA_THRESHOLD_C_##NAME,         \
              sigmaValues[3 * index + 2]);                                 \
      double tau0;                                                         \
      double tau1;                                                         \
      if (tauNeededC)                                                      \
        {                                                                  \
          tau0 = m_max(TAU_THRESHOLD_C_##NAME, tauValues[2 * index + 0]);  \
          tau1 = m_max(TAU_THRESHOLD_C_##NAME, tauValues[2 * index + 1]);  \
          if (enforceFHCC)                                                 \
            {                                                              \
              sigma0 = m_min(sigma0, 8.0 * rho0 * tau0);                   \
              sigma2 = m_min(sigma2, 8.0 * rho1 * tau1);                   \
            }                                                              \
        }                                                                  \
      double sigma1 = sigmaValues[3 * index + 1];                          \
      double s      = 0.5 * (sigma0 + sigma2);                             \
      sigma1        = (sigma1 >= -s ? sigma1 : -s);                        \
      sigma1        = (sigma1 <= s ? sigma1 : s);                          \
      BODY;                                                                \
      corrEnergyOut[index]       = tzk0;                                   \
      pdecDensity[2 * index + 0] = tvrho0;                                 \
      pdecDensity[2 * index + 1] = tvrho1;                                 \
      pdecSigma[3 * index + 0]   = tvsigma0;                               \
      pdecSigma[3 * index + 1]   = tvsigma1;                               \
      pdecSigma[3 * index + 2]   = tvsigma2;                               \
      pdecTau[2 * index + 0]     = tvtau0;                                 \
      pdecTau[2 * index + 1]     = tvtau1;                                 \
    }),                                                                    \
    const dftfe::uInt numPoints,                                           \
    const double     *densityValues,                                       \
    const double     *sigmaValues,                                         \
    const double     *tauValues,                                           \
    double           *corrEnergyOut,                                       \
    double           *pdecDensity,                                         \
    double           *pdecSigma,                                           \
    double           *pdecTau,                                             \
    bool              tauNeededC,                                          \
    bool              enforceFHCC);
  } // namespace
#include <exchangeCorrelationFunctionalEvaluation.def>
} // namespace dftfe

#undef DFTFE_FUNCTIONALEVALUATOR_LDA_X
#undef DFTFE_FUNCTIONALEVALUATOR_LDA_C
#undef DFTFE_FUNCTIONALEVALUATOR_GGA_X
#undef DFTFE_FUNCTIONALEVALUATOR_GGA_C
#undef DFTFE_FUNCTIONALEVALUATOR_MGGA_X
#undef DFTFE_FUNCTIONALEVALUATOR_MGGA_C
namespace dftfe
{
#define DFTFE_FUNCTIONALEVALUATOR_LDA_X(NAME, BODY)                         \
  template <>                                                               \
  void LDAX_##NAME(                                                         \
    dftfe::uInt numPoints,                                                  \
    const dftfe::utils::MemoryStorage<double,                               \
                                      dftfe::utils::MemorySpace::DEVICE>    \
      &densityValues,                                                       \
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>  \
      &exEnergyOut,                                                         \
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>  \
      &pdexDensity)                                                         \
  {                                                                         \
    const auto *densityValuesTemp =                                         \
      dftfe::utils::makeDataTypeDeviceCompatible(densityValues.data());     \
    auto *exEnergyOutTemp =                                                 \
      dftfe::utils::makeDataTypeDeviceCompatible(exEnergyOut.data());       \
    auto *pdexDensityTemp =                                                 \
      dftfe::utils::makeDataTypeDeviceCompatible(pdexDensity.data());       \
    DFTFE_LAUNCH_KERNEL(exchangeEvaluationKernel##NAME,                     \
                        (numPoints + dftfe::utils::DEVICE_BLOCK_SIZE - 1) / \
                          dftfe::utils::DEVICE_BLOCK_SIZE,                  \
                        dftfe::utils::DEVICE_BLOCK_SIZE,                    \
                        dftfe::linearAlgebra::BLASWrapper<                  \
                          dftfe::utils::MemorySpace::DEVICE>::d_streamId,   \
                        numPoints,                                          \
                        densityValuesTemp,                                  \
                        exEnergyOutTemp,                                    \
                        pdexDensityTemp);                                   \
  }


#define DFTFE_FUNCTIONALEVALUATOR_LDA_C(NAME, BODY)                         \
  template <>                                                               \
  void LDAC_##NAME(                                                         \
    dftfe::uInt numPoints,                                                  \
    const dftfe::utils::MemoryStorage<double,                               \
                                      dftfe::utils::MemorySpace::DEVICE>    \
      &densityValues,                                                       \
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>  \
      &corrEnergyOut,                                                       \
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>  \
      &pdecDensity)                                                         \
  {                                                                         \
    const auto *densityValuesTemp =                                         \
      dftfe::utils::makeDataTypeDeviceCompatible(densityValues.data());     \
    auto *corrEnergyOutTemp =                                               \
      dftfe::utils::makeDataTypeDeviceCompatible(corrEnergyOut.data());     \
    auto *pdecDensityTemp =                                                 \
      dftfe::utils::makeDataTypeDeviceCompatible(pdecDensity.data());       \
    DFTFE_LAUNCH_KERNEL(correlationEvaluationKernel##NAME,                  \
                        (numPoints + dftfe::utils::DEVICE_BLOCK_SIZE - 1) / \
                          dftfe::utils::DEVICE_BLOCK_SIZE,                  \
                        dftfe::utils::DEVICE_BLOCK_SIZE,                    \
                        dftfe::linearAlgebra::BLASWrapper<                  \
                          dftfe::utils::MemorySpace::DEVICE>::d_streamId,   \
                        numPoints,                                          \
                        densityValuesTemp,                                  \
                        corrEnergyOutTemp,                                  \
                        pdecDensityTemp);                                   \
  }

#define DFTFE_FUNCTIONALEVALUATOR_GGA_X(NAME, BODY)                         \
  template <>                                                               \
  void GGAX_##NAME(                                                         \
    dftfe::uInt numPoints,                                                  \
    const dftfe::utils::MemoryStorage<double,                               \
                                      dftfe::utils::MemorySpace::DEVICE>    \
      &densityValues,                                                       \
    const dftfe::utils::MemoryStorage<double,                               \
                                      dftfe::utils::MemorySpace::DEVICE>    \
      &sigmaValues,                                                         \
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>  \
      &exEnergyOut,                                                         \
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>  \
      &pdexDensity,                                                         \
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>  \
      &pdexSigma)                                                           \
  {                                                                         \
    const auto *densityValuesTemp =                                         \
      dftfe::utils::makeDataTypeDeviceCompatible(densityValues.data());     \
    const auto *sigmaValuesTemp =                                           \
      dftfe::utils::makeDataTypeDeviceCompatible(sigmaValues.data());       \
    auto *exEnergyOutTemp =                                                 \
      dftfe::utils::makeDataTypeDeviceCompatible(exEnergyOut.data());       \
    auto *pdexDensityTemp =                                                 \
      dftfe::utils::makeDataTypeDeviceCompatible(pdexDensity.data());       \
    auto *pdexSigmaTemp =                                                   \
      dftfe::utils::makeDataTypeDeviceCompatible(pdexSigma.data());         \
    DFTFE_LAUNCH_KERNEL(exchangeEvaluationKernel##NAME,                     \
                        (numPoints + dftfe::utils::DEVICE_BLOCK_SIZE - 1) / \
                          dftfe::utils::DEVICE_BLOCK_SIZE,                  \
                        dftfe::utils::DEVICE_BLOCK_SIZE,                    \
                        dftfe::linearAlgebra::BLASWrapper<                  \
                          dftfe::utils::MemorySpace::DEVICE>::d_streamId,   \
                        numPoints,                                          \
                        densityValuesTemp,                                  \
                        sigmaValuesTemp,                                    \
                        exEnergyOutTemp,                                    \
                        pdexDensityTemp,                                    \
                        pdexSigmaTemp);                                     \
  }

#define DFTFE_FUNCTIONALEVALUATOR_GGA_C(NAME, BODY)                         \
  template <>                                                               \
  void GGAC_##NAME(                                                         \
    dftfe::uInt numPoints,                                                  \
    const dftfe::utils::MemoryStorage<double,                               \
                                      dftfe::utils::MemorySpace::DEVICE>    \
      &densityValues,                                                       \
    const dftfe::utils::MemoryStorage<double,                               \
                                      dftfe::utils::MemorySpace::DEVICE>    \
      &sigmaValues,                                                         \
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>  \
      &corrEnergyOut,                                                       \
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>  \
      &pdecDensity,                                                         \
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>  \
      &pdecSigma)                                                           \
  {                                                                         \
    const auto *densityValuesTemp =                                         \
      dftfe::utils::makeDataTypeDeviceCompatible(densityValues.data());     \
    const auto *sigmaValuesTemp =                                           \
      dftfe::utils::makeDataTypeDeviceCompatible(sigmaValues.data());       \
    auto *corrEnergyOutTemp =                                               \
      dftfe::utils::makeDataTypeDeviceCompatible(corrEnergyOut.data());     \
    auto *pdecDensityTemp =                                                 \
      dftfe::utils::makeDataTypeDeviceCompatible(pdecDensity.data());       \
    auto *pdecSigmaTemp =                                                   \
      dftfe::utils::makeDataTypeDeviceCompatible(pdecSigma.data());         \
    DFTFE_LAUNCH_KERNEL(correlationEvaluationKernel##NAME,                  \
                        (numPoints + dftfe::utils::DEVICE_BLOCK_SIZE - 1) / \
                          dftfe::utils::DEVICE_BLOCK_SIZE,                  \
                        dftfe::utils::DEVICE_BLOCK_SIZE,                    \
                        dftfe::linearAlgebra::BLASWrapper<                  \
                          dftfe::utils::MemorySpace::DEVICE>::d_streamId,   \
                        numPoints,                                          \
                        densityValuesTemp,                                  \
                        sigmaValuesTemp,                                    \
                        corrEnergyOutTemp,                                  \
                        pdecDensityTemp,                                    \
                        pdecSigmaTemp);                                     \
  }

#define DFTFE_FUNCTIONALEVALUATOR_MGGA_X(NAME, BODY)                        \
  template <>                                                               \
  void MGGAX_##NAME(                                                        \
    dftfe::uInt numPoints,                                                  \
    const dftfe::utils::MemoryStorage<double,                               \
                                      dftfe::utils::MemorySpace::DEVICE>    \
      &densityValues,                                                       \
    const dftfe::utils::MemoryStorage<double,                               \
                                      dftfe::utils::MemorySpace::DEVICE>    \
      &sigmaValues,                                                         \
    const dftfe::utils::MemoryStorage<double,                               \
                                      dftfe::utils::MemorySpace::DEVICE>    \
      &tauValues,                                                           \
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>  \
      &exEnergyOut,                                                         \
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>  \
      &pdexDensity,                                                         \
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>  \
      &pdexSigma,                                                           \
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>  \
        &pdexTau,                                                           \
    bool tauNeededX,                                                        \
    bool enforceFHCX)                                                       \
  {                                                                         \
    const auto *densityValuesTemp =                                         \
      dftfe::utils::makeDataTypeDeviceCompatible(densityValues.data());     \
    const auto *sigmaValuesTemp =                                           \
      dftfe::utils::makeDataTypeDeviceCompatible(sigmaValues.data());       \
    const auto *tauValuesTemp =                                             \
      dftfe::utils::makeDataTypeDeviceCompatible(tauValues.data());         \
    auto *exEnergyOutTemp =                                                 \
      dftfe::utils::makeDataTypeDeviceCompatible(exEnergyOut.data());       \
    auto *pdexDensityTemp =                                                 \
      dftfe::utils::makeDataTypeDeviceCompatible(pdexDensity.data());       \
    auto *pdexSigmaTemp =                                                   \
      dftfe::utils::makeDataTypeDeviceCompatible(pdexSigma.data());         \
    auto *pdexTauTemp =                                                     \
      dftfe::utils::makeDataTypeDeviceCompatible(pdexTau.data());           \
    DFTFE_LAUNCH_KERNEL(exchangeEvaluationKernel##NAME,                     \
                        (numPoints + dftfe::utils::DEVICE_BLOCK_SIZE - 1) / \
                          dftfe::utils::DEVICE_BLOCK_SIZE,                  \
                        dftfe::utils::DEVICE_BLOCK_SIZE,                    \
                        dftfe::linearAlgebra::BLASWrapper<                  \
                          dftfe::utils::MemorySpace::DEVICE>::d_streamId,   \
                        numPoints,                                          \
                        densityValuesTemp,                                  \
                        sigmaValuesTemp,                                    \
                        tauValuesTemp,                                      \
                        exEnergyOutTemp,                                    \
                        pdexDensityTemp,                                    \
                        pdexSigmaTemp,                                      \
                        pdexTauTemp,                                        \
                        tauNeededX,                                         \
                        enforceFHCX);                                       \
  }

#define DFTFE_FUNCTIONALEVALUATOR_MGGA_C(NAME, BODY)                        \
  template <>                                                               \
  void MGGAC_##NAME(                                                        \
    dftfe::uInt numPoints,                                                  \
    const dftfe::utils::MemoryStorage<double,                               \
                                      dftfe::utils::MemorySpace::DEVICE>    \
      &densityValues,                                                       \
    const dftfe::utils::MemoryStorage<double,                               \
                                      dftfe::utils::MemorySpace::DEVICE>    \
      &sigmaValues,                                                         \
    const dftfe::utils::MemoryStorage<double,                               \
                                      dftfe::utils::MemorySpace::DEVICE>    \
      &tauValues,                                                           \
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>  \
      &corrEnergyOut,                                                       \
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>  \
      &pdecDensity,                                                         \
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>  \
      &pdecSigma,                                                           \
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>  \
        &pdecTau,                                                           \
    bool tauNeededC,                                                        \
    bool enforceFHCC)                                                       \
  {                                                                         \
    const auto *densityValuesTemp =                                         \
      dftfe::utils::makeDataTypeDeviceCompatible(densityValues.data());     \
    const auto *sigmaValuesTemp =                                           \
      dftfe::utils::makeDataTypeDeviceCompatible(sigmaValues.data());       \
    const auto *tauValuesTemp =                                             \
      dftfe::utils::makeDataTypeDeviceCompatible(tauValues.data());         \
    auto *corrEnergyOutTemp =                                               \
      dftfe::utils::makeDataTypeDeviceCompatible(corrEnergyOut.data());     \
    auto *pdecDensityTemp =                                                 \
      dftfe::utils::makeDataTypeDeviceCompatible(pdecDensity.data());       \
    auto *pdecSigmaTemp =                                                   \
      dftfe::utils::makeDataTypeDeviceCompatible(pdecSigma.data());         \
    auto *pdecTauTemp =                                                     \
      dftfe::utils::makeDataTypeDeviceCompatible(pdecTau.data());           \
    DFTFE_LAUNCH_KERNEL(correlationEvaluationKernel##NAME,                  \
                        (numPoints + dftfe::utils::DEVICE_BLOCK_SIZE - 1) / \
                          dftfe::utils::DEVICE_BLOCK_SIZE,                  \
                        dftfe::utils::DEVICE_BLOCK_SIZE,                    \
                        dftfe::linearAlgebra::BLASWrapper<                  \
                          dftfe::utils::MemorySpace::DEVICE>::d_streamId,   \
                        numPoints,                                          \
                        densityValuesTemp,                                  \
                        sigmaValuesTemp,                                    \
                        tauValuesTemp,                                      \
                        corrEnergyOutTemp,                                  \
                        pdecDensityTemp,                                    \
                        pdecSigmaTemp,                                      \
                        pdecTauTemp,                                        \
                        tauNeededC,                                         \
                        enforceFHCC);                                       \
  }
#include <exchangeCorrelationFunctionalEvaluation.def>
} // namespace dftfe

#undef DFTFE_FUNCTIONALEVALUATOR_LDA_X
#undef DFTFE_FUNCTIONALEVALUATOR_LDA_C
#undef DFTFE_FUNCTIONALEVALUATOR_GGA_X
#undef DFTFE_FUNCTIONALEVALUATOR_GGA_C
#undef DFTFE_FUNCTIONALEVALUATOR_MGGA_X
#undef DFTFE_FUNCTIONALEVALUATOR_MGGA_C
