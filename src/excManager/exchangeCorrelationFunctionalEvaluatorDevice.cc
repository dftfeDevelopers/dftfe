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

#include <dftfe/exchangeCorrelationFunctionalEvaluator.h>
#include <dftfe/deviceKernelsGeneric.h>
#include <dftfe/DeviceAPICalls.h>
#include <dftfe/DeviceDataTypeOverloads.h>
#include <dftfe/DeviceTypeConfig.h>
#include <dftfe/DeviceKernelLauncherHelpers.h>
#include <dftfe/BLASWrapper.h>

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

  // ============================================================
  // r2SCAN split into per-output __noinline__ device helpers to
  // cut register pressure (avoids ROCm spill miscompilation).
  // ============================================================
  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_zk(double rho0,
                   double rho1,
                   double sigma0,
                   double sigma1,
                   double sigma2,
                   double tau0,
                   double tau1)
  {
    MGGA_C_R2SCAN_ZK
    return tzk0;
  }
  // ============================================================
  // r2SCAN correlation vrho0/vrho1: decomposed into __noinline__
  // sub-helpers (each recomputes its own cone, returns a scalar) to
  // keep every function's register footprint well under the A100
  // 255-register cap (no spill). Auto-generated; verified bit-exact
  // vs the original MGGA_C_R2SCAN_VRHO0/1 macros over 20000 random
  // density points. BUDGET=40 source-live-doubles/fn.
  // ============================================================
  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho0__t252(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t14 = 0.1e1 + 0.53425e-1 * t12;
    double t15 = std::sqrt(t12);
    double t17 = 0.8969e0 * t12;
    double t18 = t12 * std::sqrt(t12);
    double t19 = 0.204775e0 * t18;
    double t20 = t2 * t2;
    double t21 = t4 * t4;
    double t22 = t20 * t21;
    double t23 = t9 * t9;
    double t26 = t22 * t6 / t23;
    double t27 = 0.123235e0 * t26;
    double t28 = 0.379785e1 * t15 + t17 + t19 + t27;
    double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
    double t32 = std::log(t31);
    double t34 = 0.621814e-1 * t14 * t32;
    double t35 = rho0 - rho1;
    double t36 = t35 * t35;
    double t37 = t36 * t36;
    double t38 = t8 * t8;
    double t39 = t38 * t38;
    double t40 = 0.1e1 / t39;
    double t41 = t37 * t40;
    double t42 = 0.1e1 / t8;
    double t43 = t35 * t42;
    double t44 = 0.1e1 + t43;
    double t45 = t44 <= ZETA_THRESHOLD_C_R2SCAN;
    double t46 = std::cbrt(ZETA_THRESHOLD_C_R2SCAN);
    double t47 = t46 * ZETA_THRESHOLD_C_R2SCAN;
    double t48 = std::cbrt(t44);
    double t49 = t48 * t44;
    double t50 = my_piecewise3(t45, t47, t49);
    double t51 = 0.1e1 - t43;
    double t52 = t51 <= ZETA_THRESHOLD_C_R2SCAN;
    double t53 = std::cbrt(t51);
    double t54 = t53 * t51;
    double t55 = my_piecewise3(t52, t47, t54);
    double t56 = t50 + t55 - 0.2e1;
    double t57 = M_CBRT2;
    double t58 = t57 - 0.1e1;
    double t60 = 0.1e1 / t58 / 0.2e1;
    double t61 = t56 * t60;
    double t63 = 0.1e1 + 0.5137e-1 * t12;
    double t65 = 0.1549425e1 * t12;
    double t66 = 0.420775e0 * t18;
    double t67 = 0.1562925e0 * t26;
    double t68 = 0.705945e1 * t15 + t65 + t66 + t67;
    double t71 = 0.1e1 + 0.32163958997385070134e2 / t68;
    double t72 = std::log(t71);
    double t76 = 0.1e1 + 0.278125e-1 * t12;
    double t78 = 0.905775e0 * t12;
    double t79 = 0.1100325e0 * t18;
    double t80 = 0.1241775e0 * t26;
    double t81 = 0.51785e1 * t15 + t78 + t79 + t80;
    double t84 = 0.1e1 + 0.29608749977793437516e2 / t81;
    double t85 = std::log(t84);
    double t86 = t76 * t85;
    double t88 = -0.310907e-1 * t63 * t72 + t34 - 0.19751673498613801407e-1 * t86;
    double t89 = t61 * t88;
    double t90 = t41 * t89;
    double t92 = 0.19751673498613801407e-1 * t61 * t86;
    double t148 = std::sqrt(0.4e1);
    double t149 = t148 * t15;
    double t151 = 0.3138525e-1 * t12;
    double t152 = 0.1e1 + 0.22225e-1 * t149 + t151;
    double t153 = t152 * t152;
    double t154 = 0.1e1 / t153;
    double t158 = 0.1e1 - 0.2363e1 * t58 * t56 * t60;
    double t159 = t154 * t158;
    double t160 = t37 * t37;
    double t161 = t160 * t37;
    double t162 = t39 * t39;
    double t163 = t162 * t39;
    double t164 = 0.1e1 / t163;
    double t166 = -t161 * t164 + 0.1e1;
    double t167 = 0.1e1 / t15;
    double t168 = t148 * t167;
    double t170 = 0.4445e-1 * t168 + 0.125541e0;
    double t171 = t166 * t170;
    double t175 = 0.1898925e1 * t149 + t17 + t19 + t27;
    double t178 = 0.1e1 + 0.16081979498692535067e2 / t175;
    double t179 = std::log(t178);
    double t180 = 0.1328816518e-1 * t179;
    double t181 = t175 * t175;
    double t182 = 0.1e1 / t181;
    double t183 = t14 * t182;
    double t185 = std::sqrt(t12);
    double t188 = 0.379785e1 * t168 + 0.35876e1 + 0.122865e1 * t185 + 0.24647e0 * t12;
    double t189 = 0.1e1 / t178;
    double t190 = t188 * t189;
    double t192 = 0.1e1 * t183 * t190;
    double t194 = 0.3529725e1 * t149 + t65 + t66 + t67;
    double t197 = 0.1e1 + 0.32163958997385070134e2 / t194;
    double t198 = std::log(t197);
    double t200 = t194 * t194;
    double t201 = 0.1e1 / t200;
    double t202 = t63 * t201;
    double t206 = 0.705945e1 * t168 + 0.61977e1 + 0.252465e1 * t185 + 0.312585e0 * t12;
    double t207 = 0.1e1 / t197;
    double t208 = t206 * t207;
    double t212 = 0.258925e1 * t149 + t78 + t79 + t80;
    double t215 = 0.1e1 + 0.29608749977793437516e2 / t212;
    double t216 = std::log(t215);
    double t218 = t212 * t212;
    double t219 = 0.1e1 / t218;
    double t220 = t76 * t219;
    double t224 = 0.51785e1 * t168 + 0.36231e1 + 0.660195e0 * t185 + 0.248355e0 * t12;
    double t225 = 0.1e1 / t215;
    double t226 = t224 * t225;
    double t229 = -0.6388517036e-2 * t198 + 0.1e1 * t202 * t208 + t180 - t192 - 0.21973736767207854065e-2 * t216 + 0.5848223622634646207e0 * t220 * t226;
    double t230 = t61 * t229;
    double t234 = t61 * t76;
    double t236 = t219 * t224 * t225;
    double t239 = 0.285764e-1 * t159 * t171 + t180 - t192 - t41 * t230 - 0.21973736767207854065e-2 * t61 * t216 + 0.5848223622634646207e0 * t234 * t236;
    double t244 = 0.1e1 + 0.4445e-1 * t15 + t151;
    double t245 = 0.1e1 / t244;
    double t246 = t245 * t158;
    double t252 = 0.5e1 * t5 * t11 * t239 - 0.45e2 * 0.001 * (-0.285764e-1 * t246 * t166 + t34 - t90 - t92);
    return t252;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho0__t354(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t23 = t9 * t9;
    double t35 = rho0 - rho1;
    double t38 = t8 * t8;
    double t42 = 0.1e1 / t8;
    double t43 = t35 * t42;
    double t44 = 0.1e1 + t43;
    double t51 = 0.1e1 - t43;
    double t95 = M_PI * M_PI;
    double t122 = sigma0 + 0.2e1 * sigma1 + sigma2;
    double t255 = M_CBRT6;
    double t256 = std::cbrt(t95);
    double t257 = t256 * t256;
    double t263 = 0.1e1 / t23 / t38;
    double t264 = t122 * t263;
    double t265 = t255 * t255;
    double t291 = std::cbrt(rho0);
    double t292 = t291 * t291;
    double t294 = 0.1e1 / t292 / rho0;
    double t295 = tau0 * t294;
    double t296 = t44 / 0.2e1;
    double t297 = std::cbrt(t296);
    double t298 = t297 * t297;
    double t299 = t298 * t296;
    double t301 = std::cbrt(rho1);
    double t302 = t301 * t301;
    double t304 = 0.1e1 / t302 / rho1;
    double t305 = tau1 * t304;
    double t306 = t51 / 0.2e1;
    double t307 = std::cbrt(t306);
    double t308 = t307 * t307;
    double t309 = t308 * t306;
    double t312 = t295 * t299 + t305 * t309 - t264 / 0.8e1;
    double t313 = t265 * t257;
    double t317 = 0.001 * t122;
    double t320 = 0.3e1 / 0.1e2 * t313 * (t299 + t309) + t317 * t263 / 0.8e1;
    double t321 = 0.1e1 / t320;
    double t322 = t312 * t321;
    double t323 = t322 <= 0.e0;
    double t324 = 0.e0 < t322;
    double t325 = my_piecewise3(t324, 0, t322);
    double t326 = 0.1e1 - t325;
    double t327 = 0.1e1 / t326;
    double t330 = std::exp(-0.64e0 * t325 * t327);
    double t331 = t322 <= 0.25e1;
    double t332 = 0.25e1 < t322;
    double t333 = my_piecewise3(t332, 0.25e1, t322);
    double t335 = t333 * t333;
    double t337 = t335 * t333;
    double t339 = t335 * t335;
    double t341 = t339 * t333;
    double t343 = t339 * t335;
    double t348 = my_piecewise3(t332, t322, 0.25e1);
    double t349 = 0.1e1 - t348;
    double t352 = std::exp(0.15e1 / t349);
    double t354 = my_piecewise5(t323, t330, t331, 0.1e1 - 0.64e0 * t333 - 0.4352e0 * t335 - 0.1535685604549e1 * t337 + 0.3061560252175e1 * t339 - 0.1915710236206e1 * t341 + 0.516884468372e0 * t343 - 0.51848879792e-1 * t339 * t337, -0.7e0 * t352);
    return t354;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho0__t371(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t15 = std::sqrt(t12);
    double t23 = t9 * t9;
    double t38 = t8 * t8;
    double t57 = M_CBRT2;
    double t95 = M_PI * M_PI;
    double t122 = sigma0 + 0.2e1 * sigma1 + sigma2;
    double t151 = 0.3138525e-1 * t12;
    double t244 = 0.1e1 + 0.4445e-1 * t15 + t151;
    double t245 = 0.1e1 / t244;
    double t255 = M_CBRT6;
    double t256 = std::cbrt(t95);
    double t257 = t256 * t256;
    double t258 = 0.1e1 / t257;
    double t259 = t255 * t258;
    double t260 = t57 * t57;
    double t263 = 0.1e1 / t23 / t38;
    double t357 = std::exp(0.1e1 * t245);
    double t358 = t357 - 0.1e1;
    double t359 = t260 * t122;
    double t360 = t359 * t263;
    double t363 = 0.1e1 + 0.21337642104376358333e-1 * t259 * t360;
    double t364 = std::sqrt(std::sqrt(t363));
    double t366 = 0.1e1 - 0.1e1 / t364;
    double t368 = t358 * t366 + 0.1e1;
    double t369 = std::log(t368);
    double t371 = -0.285764e-1 * t245 + 0.285764e-1 * t369;
    return t371;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho0__t374(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t252 = mgga_c_r2scan_vrho0__t252(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t371 = mgga_c_r2scan_vrho0__t371(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t14 = 0.1e1 + 0.53425e-1 * t12;
    double t15 = std::sqrt(t12);
    double t17 = 0.8969e0 * t12;
    double t18 = t12 * std::sqrt(t12);
    double t19 = 0.204775e0 * t18;
    double t20 = t2 * t2;
    double t21 = t4 * t4;
    double t22 = t20 * t21;
    double t23 = t9 * t9;
    double t26 = t22 * t6 / t23;
    double t27 = 0.123235e0 * t26;
    double t28 = 0.379785e1 * t15 + t17 + t19 + t27;
    double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
    double t32 = std::log(t31);
    double t34 = 0.621814e-1 * t14 * t32;
    double t35 = rho0 - rho1;
    double t36 = t35 * t35;
    double t37 = t36 * t36;
    double t38 = t8 * t8;
    double t39 = t38 * t38;
    double t40 = 0.1e1 / t39;
    double t41 = t37 * t40;
    double t42 = 0.1e1 / t8;
    double t43 = t35 * t42;
    double t44 = 0.1e1 + t43;
    double t45 = t44 <= ZETA_THRESHOLD_C_R2SCAN;
    double t46 = std::cbrt(ZETA_THRESHOLD_C_R2SCAN);
    double t47 = t46 * ZETA_THRESHOLD_C_R2SCAN;
    double t48 = std::cbrt(t44);
    double t49 = t48 * t44;
    double t50 = my_piecewise3(t45, t47, t49);
    double t51 = 0.1e1 - t43;
    double t52 = t51 <= ZETA_THRESHOLD_C_R2SCAN;
    double t53 = std::cbrt(t51);
    double t54 = t53 * t51;
    double t55 = my_piecewise3(t52, t47, t54);
    double t56 = t50 + t55 - 0.2e1;
    double t57 = M_CBRT2;
    double t58 = t57 - 0.1e1;
    double t60 = 0.1e1 / t58 / 0.2e1;
    double t61 = t56 * t60;
    double t63 = 0.1e1 + 0.5137e-1 * t12;
    double t65 = 0.1549425e1 * t12;
    double t66 = 0.420775e0 * t18;
    double t67 = 0.1562925e0 * t26;
    double t68 = 0.705945e1 * t15 + t65 + t66 + t67;
    double t71 = 0.1e1 + 0.32163958997385070134e2 / t68;
    double t72 = std::log(t71);
    double t76 = 0.1e1 + 0.278125e-1 * t12;
    double t78 = 0.905775e0 * t12;
    double t79 = 0.1100325e0 * t18;
    double t80 = 0.1241775e0 * t26;
    double t81 = 0.51785e1 * t15 + t78 + t79 + t80;
    double t84 = 0.1e1 + 0.29608749977793437516e2 / t81;
    double t85 = std::log(t84);
    double t86 = t76 * t85;
    double t88 = -0.310907e-1 * t63 * t72 + t34 - 0.19751673498613801407e-1 * t86;
    double t89 = t61 * t88;
    double t90 = t41 * t89;
    double t92 = 0.19751673498613801407e-1 * t61 * t86;
    double t93 = std::log(0.2e1);
    double t94 = 0.1e1 - t93;
    double t95 = M_PI * M_PI;
    double t97 = t94 / t95;
    double t98 = t46 * t46;
    double t99 = t48 * t48;
    double t100 = my_piecewise3(t45, t98, t99);
    double t101 = t53 * t53;
    double t102 = my_piecewise3(t52, t98, t101);
    double t104 = t100 / 0.2e1 + t102 / 0.2e1;
    double t105 = t104 * t104;
    double t106 = t105 * t104;
    double t108 = 0.1e1 / t94;
    double t109 = (-t34 + t90 + t92) * t108;
    double t110 = 0.1e1 / t106;
    double t111 = t95 * t110;
    double t113 = std::exp(-t109 * t111);
    double t114 = t113 - 0.1e1;
    double t116 = 0.1e1 + 0.25e-1 * t12;
    double t118 = 0.1e1 + 0.4445e-1 * t12;
    double t119 = 0.1e1 / t118;
    double t120 = t116 * t119;
    double t122 = sigma0 + 0.2e1 * sigma1 + sigma2;
    double t124 = 0.1e1 / t9 / t38;
    double t125 = t122 * t124;
    double t128 = 0.1e1 / t105;
    double t129 = t128 * t20;
    double t130 = 0.1e1 / t4;
    double t132 = t6 * t108;
    double t133 = 0.1e1 / t114;
    double t134 = t132 * t133;
    double t135 = t129 * t130 * t134;
    double t138 = t98 * ZETA_THRESHOLD_C_R2SCAN;
    double t139 = t99 * t44;
    double t140 = my_piecewise3(t45, t138, t139);
    double t141 = t101 * t51;
    double t142 = my_piecewise3(t52, t138, t141);
    double t144 = t140 / 0.2e1 + t142 / 0.2e1;
    double t146 = t108 / t144;
    double t147 = t110 * t133;
    double t158 = 0.1e1 - 0.2363e1 * t58 * t56 * t60;
    double t160 = t37 * t37;
    double t161 = t160 * t37;
    double t162 = t39 * t39;
    double t163 = t162 * t39;
    double t164 = 0.1e1 / t163;
    double t166 = -t161 * t164 + 0.1e1;
    double t253 = t147 * t252;
    double t254 = t146 * t253;
    double t255 = M_CBRT6;
    double t256 = std::cbrt(t95);
    double t257 = t256 * t256;
    double t258 = 0.1e1 / t257;
    double t259 = t255 * t258;
    double t260 = t57 * t57;
    double t261 = t259 * t260;
    double t263 = 0.1e1 / t23 / t38;
    double t264 = t122 * t263;
    double t265 = t255 * t255;
    double t267 = 0.1e1 / t256 / t95;
    double t268 = t265 * t267;
    double t269 = t122 * t122;
    double t270 = t57 * t269;
    double t271 = t39 * t8;
    double t273 = 0.1e1 / t9 / t271;
    double t277 = std::exp(-0.20444604078896369094e0 * t268 * t270 * t273);
    double t279 = t261 * t264 * t277;
    double t282 = 0.1e1 + 0.27439371595564631661e-1 * t120 * t125 * t57 * t135 + 0.43341108700271342816e-1 * t254 * t279;
    double t283 = std::sqrt(std::sqrt(t282));
    double t285 = 0.1e1 - 0.1e1 / t283;
    double t287 = t114 * t285 + 0.1e1;
    double t288 = std::log(t287);
    double t290 = t97 * t106 * t288;
    double t372 = t371 * t158;
    double t374 = t372 * t166 - t290 + t34 - t90 - t92;
    return t374;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho0__t404(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t14 = 0.1e1 + 0.53425e-1 * t12;
    double t15 = std::sqrt(t12);
    double t17 = 0.8969e0 * t12;
    double t18 = t12 * std::sqrt(t12);
    double t19 = 0.204775e0 * t18;
    double t20 = t2 * t2;
    double t21 = t4 * t4;
    double t22 = t20 * t21;
    double t23 = t9 * t9;
    double t26 = t22 * t6 / t23;
    double t27 = 0.123235e0 * t26;
    double t28 = 0.379785e1 * t15 + t17 + t19 + t27;
    double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
    double t167 = 0.1e1 / t15;
    double t185 = std::sqrt(t12);
    double t377 = 0.1e1 / t9 / t8;
    double t378 = t7 * t377;
    double t382 = t28 * t28;
    double t383 = 0.1e1 / t382;
    double t384 = t14 * t383;
    double t385 = t167 * t2;
    double t386 = t4 * t7;
    double t387 = t386 * t377;
    double t388 = t385 * t387;
    double t390 = t5 * t378;
    double t391 = 0.29896666666666666667e0 * t390;
    double t392 = t185 * t2;
    double t393 = t392 * t387;
    double t394 = 0.1023875e0 * t393;
    double t398 = t22 * t6 / t23 / t8;
    double t399 = 0.82156666666666666667e-1 * t398;
    double t400 = -0.632975e0 * t388 - t391 - t394 - t399;
    double t401 = 0.1e1 / t31;
    double t402 = t400 * t401;
    double t403 = t384 * t402;
    double t404 = 0.1e1 * t403;
    return t404;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho0__t408(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t14 = 0.1e1 + 0.53425e-1 * t12;
    double t15 = std::sqrt(t12);
    double t17 = 0.8969e0 * t12;
    double t18 = t12 * std::sqrt(t12);
    double t19 = 0.204775e0 * t18;
    double t20 = t2 * t2;
    double t21 = t4 * t4;
    double t22 = t20 * t21;
    double t23 = t9 * t9;
    double t26 = t22 * t6 / t23;
    double t27 = 0.123235e0 * t26;
    double t28 = 0.379785e1 * t15 + t17 + t19 + t27;
    double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
    double t32 = std::log(t31);
    double t34 = 0.621814e-1 * t14 * t32;
    double t35 = rho0 - rho1;
    double t36 = t35 * t35;
    double t38 = t8 * t8;
    double t39 = t38 * t38;
    double t40 = 0.1e1 / t39;
    double t42 = 0.1e1 / t8;
    double t43 = t35 * t42;
    double t44 = 0.1e1 + t43;
    double t45 = t44 <= ZETA_THRESHOLD_C_R2SCAN;
    double t46 = std::cbrt(ZETA_THRESHOLD_C_R2SCAN);
    double t47 = t46 * ZETA_THRESHOLD_C_R2SCAN;
    double t48 = std::cbrt(t44);
    double t49 = t48 * t44;
    double t50 = my_piecewise3(t45, t47, t49);
    double t51 = 0.1e1 - t43;
    double t52 = t51 <= ZETA_THRESHOLD_C_R2SCAN;
    double t53 = std::cbrt(t51);
    double t54 = t53 * t51;
    double t55 = my_piecewise3(t52, t47, t54);
    double t56 = t50 + t55 - 0.2e1;
    double t57 = M_CBRT2;
    double t58 = t57 - 0.1e1;
    double t60 = 0.1e1 / t58 / 0.2e1;
    double t61 = t56 * t60;
    double t63 = 0.1e1 + 0.5137e-1 * t12;
    double t65 = 0.1549425e1 * t12;
    double t66 = 0.420775e0 * t18;
    double t67 = 0.1562925e0 * t26;
    double t68 = 0.705945e1 * t15 + t65 + t66 + t67;
    double t71 = 0.1e1 + 0.32163958997385070134e2 / t68;
    double t72 = std::log(t71);
    double t76 = 0.1e1 + 0.278125e-1 * t12;
    double t78 = 0.905775e0 * t12;
    double t79 = 0.1100325e0 * t18;
    double t80 = 0.1241775e0 * t26;
    double t81 = 0.51785e1 * t15 + t78 + t79 + t80;
    double t84 = 0.1e1 + 0.29608749977793437516e2 / t81;
    double t85 = std::log(t84);
    double t86 = t76 * t85;
    double t88 = -0.310907e-1 * t63 * t72 + t34 - 0.19751673498613801407e-1 * t86;
    double t89 = t61 * t88;
    double t405 = t36 * t35;
    double t406 = t405 * t40;
    double t407 = t406 * t89;
    double t408 = 0.4e1 * t407;
    return t408;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho0__t424(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t8 = rho0 + rho1;
    double t35 = rho0 - rho1;
    double t38 = t8 * t8;
    double t42 = 0.1e1 / t8;
    double t43 = t35 * t42;
    double t44 = 0.1e1 + t43;
    double t45 = t44 <= ZETA_THRESHOLD_C_R2SCAN;
    double t48 = std::cbrt(t44);
    double t51 = 0.1e1 - t43;
    double t52 = t51 <= ZETA_THRESHOLD_C_R2SCAN;
    double t53 = std::cbrt(t51);
    double t57 = M_CBRT2;
    double t58 = t57 - 0.1e1;
    double t60 = 0.1e1 / t58 / 0.2e1;
    double t413 = 0.1e1 / t38;
    double t414 = t35 * t413;
    double t415 = t42 - t414;
    double t418 = my_piecewise3(t45, 0, 0.4e1 / 0.3e1 * t48 * t415);
    double t419 = -t415;
    double t422 = my_piecewise3(t52, 0, 0.4e1 / 0.3e1 * t53 * t419);
    double t423 = t418 + t422;
    double t424 = t423 * t60;
    return t424;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho0__t459(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t404 = mgga_c_r2scan_vrho0__t404(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t15 = std::sqrt(t12);
    double t17 = 0.8969e0 * t12;
    double t18 = t12 * std::sqrt(t12);
    double t19 = 0.204775e0 * t18;
    double t20 = t2 * t2;
    double t21 = t4 * t4;
    double t22 = t20 * t21;
    double t23 = t9 * t9;
    double t26 = t22 * t6 / t23;
    double t27 = 0.123235e0 * t26;
    double t28 = 0.379785e1 * t15 + t17 + t19 + t27;
    double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
    double t32 = std::log(t31);
    double t35 = rho0 - rho1;
    double t36 = t35 * t35;
    double t37 = t36 * t36;
    double t38 = t8 * t8;
    double t39 = t38 * t38;
    double t40 = 0.1e1 / t39;
    double t41 = t37 * t40;
    double t42 = 0.1e1 / t8;
    double t43 = t35 * t42;
    double t44 = 0.1e1 + t43;
    double t45 = t44 <= ZETA_THRESHOLD_C_R2SCAN;
    double t46 = std::cbrt(ZETA_THRESHOLD_C_R2SCAN);
    double t47 = t46 * ZETA_THRESHOLD_C_R2SCAN;
    double t48 = std::cbrt(t44);
    double t49 = t48 * t44;
    double t50 = my_piecewise3(t45, t47, t49);
    double t51 = 0.1e1 - t43;
    double t52 = t51 <= ZETA_THRESHOLD_C_R2SCAN;
    double t53 = std::cbrt(t51);
    double t54 = t53 * t51;
    double t55 = my_piecewise3(t52, t47, t54);
    double t56 = t50 + t55 - 0.2e1;
    double t57 = M_CBRT2;
    double t58 = t57 - 0.1e1;
    double t60 = 0.1e1 / t58 / 0.2e1;
    double t61 = t56 * t60;
    double t63 = 0.1e1 + 0.5137e-1 * t12;
    double t65 = 0.1549425e1 * t12;
    double t66 = 0.420775e0 * t18;
    double t67 = 0.1562925e0 * t26;
    double t68 = 0.705945e1 * t15 + t65 + t66 + t67;
    double t71 = 0.1e1 + 0.32163958997385070134e2 / t68;
    double t72 = std::log(t71);
    double t76 = 0.1e1 + 0.278125e-1 * t12;
    double t78 = 0.905775e0 * t12;
    double t79 = 0.1100325e0 * t18;
    double t80 = 0.1241775e0 * t26;
    double t81 = 0.51785e1 * t15 + t78 + t79 + t80;
    double t84 = 0.1e1 + 0.29608749977793437516e2 / t81;
    double t85 = std::log(t84);
    double t167 = 0.1e1 / t15;
    double t185 = std::sqrt(t12);
    double t377 = 0.1e1 / t9 / t8;
    double t378 = t7 * t377;
    double t380 = t5 * t378 * t32;
    double t381 = 0.11073470983333333333e-2 * t380;
    double t385 = t167 * t2;
    double t386 = t4 * t7;
    double t387 = t386 * t377;
    double t388 = t385 * t387;
    double t390 = t5 * t378;
    double t392 = t185 * t2;
    double t393 = t392 * t387;
    double t398 = t22 * t6 / t23 / t8;
    double t430 = t68 * t68;
    double t431 = 0.1e1 / t430;
    double t432 = t63 * t431;
    double t434 = 0.516475e0 * t390;
    double t435 = 0.2103875e0 * t393;
    double t436 = 0.104195e0 * t398;
    double t437 = -0.1176575e1 * t388 - t434 - t435 - t436;
    double t438 = 0.1e1 / t71;
    double t439 = t437 * t438;
    double t445 = t81 * t81;
    double t446 = 0.1e1 / t445;
    double t447 = t76 * t446;
    double t449 = 0.301925e0 * t390;
    double t450 = 0.5501625e-1 * t393;
    double t451 = 0.82785e-1 * t398;
    double t452 = -0.86308333333333333334e0 * t388 - t449 - t450 - t451;
    double t453 = 0.1e1 / t84;
    double t454 = t452 * t453;
    double t457 = 0.53237641966666666666e-3 * t5 * t378 * t72 + 0.1e1 * t432 * t439 - t381 - t404 + 0.18311447306006545054e-3 * t5 * t378 * t85 + 0.5848223622634646207e0 * t447 * t454;
    double t458 = t61 * t457;
    double t459 = t41 * t458;
    return t459;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho0__t481(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t8 = rho0 + rho1;
    double t35 = rho0 - rho1;
    double t38 = t8 * t8;
    double t42 = 0.1e1 / t8;
    double t43 = t35 * t42;
    double t44 = 0.1e1 + t43;
    double t45 = t44 <= ZETA_THRESHOLD_C_R2SCAN;
    double t48 = std::cbrt(t44);
    double t51 = 0.1e1 - t43;
    double t52 = t51 <= ZETA_THRESHOLD_C_R2SCAN;
    double t53 = std::cbrt(t51);
    double t413 = 0.1e1 / t38;
    double t414 = t35 * t413;
    double t415 = t42 - t414;
    double t419 = -t415;
    double t472 = 0.1e1 / t48;
    double t473 = t472 * t415;
    double t475 = my_piecewise3(t45, 0, 0.2e1 / 0.3e1 * t473);
    double t476 = 0.1e1 / t53;
    double t477 = t476 * t419;
    double t479 = my_piecewise3(t52, 0, 0.2e1 / 0.3e1 * t477);
    double t481 = t475 / 0.2e1 + t479 / 0.2e1;
    return t481;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho0__t484(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t252 = mgga_c_r2scan_vrho0__t252(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t481 = mgga_c_r2scan_vrho0__t481(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t14 = 0.1e1 + 0.53425e-1 * t12;
    double t15 = std::sqrt(t12);
    double t17 = 0.8969e0 * t12;
    double t18 = t12 * std::sqrt(t12);
    double t19 = 0.204775e0 * t18;
    double t20 = t2 * t2;
    double t21 = t4 * t4;
    double t22 = t20 * t21;
    double t23 = t9 * t9;
    double t26 = t22 * t6 / t23;
    double t27 = 0.123235e0 * t26;
    double t28 = 0.379785e1 * t15 + t17 + t19 + t27;
    double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
    double t32 = std::log(t31);
    double t34 = 0.621814e-1 * t14 * t32;
    double t35 = rho0 - rho1;
    double t36 = t35 * t35;
    double t37 = t36 * t36;
    double t38 = t8 * t8;
    double t39 = t38 * t38;
    double t40 = 0.1e1 / t39;
    double t41 = t37 * t40;
    double t42 = 0.1e1 / t8;
    double t43 = t35 * t42;
    double t44 = 0.1e1 + t43;
    double t45 = t44 <= ZETA_THRESHOLD_C_R2SCAN;
    double t46 = std::cbrt(ZETA_THRESHOLD_C_R2SCAN);
    double t47 = t46 * ZETA_THRESHOLD_C_R2SCAN;
    double t48 = std::cbrt(t44);
    double t49 = t48 * t44;
    double t50 = my_piecewise3(t45, t47, t49);
    double t51 = 0.1e1 - t43;
    double t52 = t51 <= ZETA_THRESHOLD_C_R2SCAN;
    double t53 = std::cbrt(t51);
    double t54 = t53 * t51;
    double t55 = my_piecewise3(t52, t47, t54);
    double t56 = t50 + t55 - 0.2e1;
    double t57 = M_CBRT2;
    double t58 = t57 - 0.1e1;
    double t60 = 0.1e1 / t58 / 0.2e1;
    double t61 = t56 * t60;
    double t63 = 0.1e1 + 0.5137e-1 * t12;
    double t65 = 0.1549425e1 * t12;
    double t66 = 0.420775e0 * t18;
    double t67 = 0.1562925e0 * t26;
    double t68 = 0.705945e1 * t15 + t65 + t66 + t67;
    double t71 = 0.1e1 + 0.32163958997385070134e2 / t68;
    double t72 = std::log(t71);
    double t76 = 0.1e1 + 0.278125e-1 * t12;
    double t78 = 0.905775e0 * t12;
    double t79 = 0.1100325e0 * t18;
    double t80 = 0.1241775e0 * t26;
    double t81 = 0.51785e1 * t15 + t78 + t79 + t80;
    double t84 = 0.1e1 + 0.29608749977793437516e2 / t81;
    double t85 = std::log(t84);
    double t86 = t76 * t85;
    double t88 = -0.310907e-1 * t63 * t72 + t34 - 0.19751673498613801407e-1 * t86;
    double t89 = t61 * t88;
    double t90 = t41 * t89;
    double t92 = 0.19751673498613801407e-1 * t61 * t86;
    double t93 = std::log(0.2e1);
    double t94 = 0.1e1 - t93;
    double t95 = M_PI * M_PI;
    double t97 = t94 / t95;
    double t98 = t46 * t46;
    double t99 = t48 * t48;
    double t100 = my_piecewise3(t45, t98, t99);
    double t101 = t53 * t53;
    double t102 = my_piecewise3(t52, t98, t101);
    double t104 = t100 / 0.2e1 + t102 / 0.2e1;
    double t105 = t104 * t104;
    double t106 = t105 * t104;
    double t108 = 0.1e1 / t94;
    double t109 = (-t34 + t90 + t92) * t108;
    double t110 = 0.1e1 / t106;
    double t111 = t95 * t110;
    double t113 = std::exp(-t109 * t111);
    double t114 = t113 - 0.1e1;
    double t116 = 0.1e1 + 0.25e-1 * t12;
    double t118 = 0.1e1 + 0.4445e-1 * t12;
    double t119 = 0.1e1 / t118;
    double t120 = t116 * t119;
    double t122 = sigma0 + 0.2e1 * sigma1 + sigma2;
    double t124 = 0.1e1 / t9 / t38;
    double t125 = t122 * t124;
    double t128 = 0.1e1 / t105;
    double t129 = t128 * t20;
    double t130 = 0.1e1 / t4;
    double t132 = t6 * t108;
    double t133 = 0.1e1 / t114;
    double t134 = t132 * t133;
    double t135 = t129 * t130 * t134;
    double t138 = t98 * ZETA_THRESHOLD_C_R2SCAN;
    double t139 = t99 * t44;
    double t140 = my_piecewise3(t45, t138, t139);
    double t141 = t101 * t51;
    double t142 = my_piecewise3(t52, t138, t141);
    double t144 = t140 / 0.2e1 + t142 / 0.2e1;
    double t146 = t108 / t144;
    double t147 = t110 * t133;
    double t253 = t147 * t252;
    double t254 = t146 * t253;
    double t255 = M_CBRT6;
    double t256 = std::cbrt(t95);
    double t257 = t256 * t256;
    double t258 = 0.1e1 / t257;
    double t259 = t255 * t258;
    double t260 = t57 * t57;
    double t261 = t259 * t260;
    double t263 = 0.1e1 / t23 / t38;
    double t264 = t122 * t263;
    double t265 = t255 * t255;
    double t267 = 0.1e1 / t256 / t95;
    double t268 = t265 * t267;
    double t269 = t122 * t122;
    double t270 = t57 * t269;
    double t271 = t39 * t8;
    double t273 = 0.1e1 / t9 / t271;
    double t277 = std::exp(-0.20444604078896369094e0 * t268 * t270 * t273);
    double t279 = t261 * t264 * t277;
    double t282 = 0.1e1 + 0.27439371595564631661e-1 * t120 * t125 * t57 * t135 + 0.43341108700271342816e-1 * t254 * t279;
    double t283 = std::sqrt(std::sqrt(t282));
    double t285 = 0.1e1 - 0.1e1 / t283;
    double t287 = t114 * t285 + 0.1e1;
    double t288 = std::log(t287);
    double t471 = t105 * t288;
    double t483 = t97 * t471 * t481;
    double t484 = 0.3e1 * t483;
    return t484;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho0__t486(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t404 = mgga_c_r2scan_vrho0__t404(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t408 = mgga_c_r2scan_vrho0__t408(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t424 = mgga_c_r2scan_vrho0__t424(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t459 = mgga_c_r2scan_vrho0__t459(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t14 = 0.1e1 + 0.53425e-1 * t12;
    double t15 = std::sqrt(t12);
    double t17 = 0.8969e0 * t12;
    double t18 = t12 * std::sqrt(t12);
    double t19 = 0.204775e0 * t18;
    double t20 = t2 * t2;
    double t21 = t4 * t4;
    double t22 = t20 * t21;
    double t23 = t9 * t9;
    double t26 = t22 * t6 / t23;
    double t27 = 0.123235e0 * t26;
    double t28 = 0.379785e1 * t15 + t17 + t19 + t27;
    double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
    double t32 = std::log(t31);
    double t34 = 0.621814e-1 * t14 * t32;
    double t35 = rho0 - rho1;
    double t36 = t35 * t35;
    double t37 = t36 * t36;
    double t38 = t8 * t8;
    double t39 = t38 * t38;
    double t40 = 0.1e1 / t39;
    double t41 = t37 * t40;
    double t42 = 0.1e1 / t8;
    double t43 = t35 * t42;
    double t44 = 0.1e1 + t43;
    double t45 = t44 <= ZETA_THRESHOLD_C_R2SCAN;
    double t46 = std::cbrt(ZETA_THRESHOLD_C_R2SCAN);
    double t47 = t46 * ZETA_THRESHOLD_C_R2SCAN;
    double t48 = std::cbrt(t44);
    double t49 = t48 * t44;
    double t50 = my_piecewise3(t45, t47, t49);
    double t51 = 0.1e1 - t43;
    double t52 = t51 <= ZETA_THRESHOLD_C_R2SCAN;
    double t53 = std::cbrt(t51);
    double t54 = t53 * t51;
    double t55 = my_piecewise3(t52, t47, t54);
    double t56 = t50 + t55 - 0.2e1;
    double t57 = M_CBRT2;
    double t58 = t57 - 0.1e1;
    double t60 = 0.1e1 / t58 / 0.2e1;
    double t61 = t56 * t60;
    double t63 = 0.1e1 + 0.5137e-1 * t12;
    double t65 = 0.1549425e1 * t12;
    double t66 = 0.420775e0 * t18;
    double t67 = 0.1562925e0 * t26;
    double t68 = 0.705945e1 * t15 + t65 + t66 + t67;
    double t71 = 0.1e1 + 0.32163958997385070134e2 / t68;
    double t72 = std::log(t71);
    double t76 = 0.1e1 + 0.278125e-1 * t12;
    double t78 = 0.905775e0 * t12;
    double t79 = 0.1100325e0 * t18;
    double t80 = 0.1241775e0 * t26;
    double t81 = 0.51785e1 * t15 + t78 + t79 + t80;
    double t84 = 0.1e1 + 0.29608749977793437516e2 / t81;
    double t85 = std::log(t84);
    double t86 = t76 * t85;
    double t88 = -0.310907e-1 * t63 * t72 + t34 - 0.19751673498613801407e-1 * t86;
    double t89 = t61 * t88;
    double t93 = std::log(0.2e1);
    double t94 = 0.1e1 - t93;
    double t108 = 0.1e1 / t94;
    double t167 = 0.1e1 / t15;
    double t185 = std::sqrt(t12);
    double t234 = t61 * t76;
    double t271 = t39 * t8;
    double t377 = 0.1e1 / t9 / t8;
    double t378 = t7 * t377;
    double t380 = t5 * t378 * t32;
    double t381 = 0.11073470983333333333e-2 * t380;
    double t385 = t167 * t2;
    double t386 = t4 * t7;
    double t387 = t386 * t377;
    double t388 = t385 * t387;
    double t390 = t5 * t378;
    double t392 = t185 * t2;
    double t393 = t392 * t387;
    double t398 = t22 * t6 / t23 / t8;
    double t409 = 0.1e1 / t271;
    double t410 = t37 * t409;
    double t411 = t410 * t89;
    double t412 = 0.4e1 * t411;
    double t425 = t424 * t88;
    double t426 = t41 * t425;
    double t445 = t81 * t81;
    double t446 = 0.1e1 / t445;
    double t449 = 0.301925e0 * t390;
    double t450 = 0.5501625e-1 * t393;
    double t451 = 0.82785e-1 * t398;
    double t452 = -0.86308333333333333334e0 * t388 - t449 - t450 - t451;
    double t453 = 0.1e1 / t84;
    double t460 = t424 * t86;
    double t461 = 0.19751673498613801407e-1 * t460;
    double t462 = t61 * t2;
    double t464 = t386 * t377 * t85;
    double t465 = t462 * t464;
    double t466 = 0.18311447306006545054e-3 * t465;
    double t468 = t446 * t452 * t453;
    double t469 = t234 * t468;
    double t470 = 0.5848223622634646207e0 * t469;
    double t486 = (t381 + t404 + t408 - t412 + t426 + t459 + t461 - t466 - t470) * t108;
    return t486;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho0__t495(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t481 = mgga_c_r2scan_vrho0__t481(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t486 = mgga_c_r2scan_vrho0__t486(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t14 = 0.1e1 + 0.53425e-1 * t12;
    double t15 = std::sqrt(t12);
    double t17 = 0.8969e0 * t12;
    double t18 = t12 * std::sqrt(t12);
    double t19 = 0.204775e0 * t18;
    double t20 = t2 * t2;
    double t21 = t4 * t4;
    double t22 = t20 * t21;
    double t23 = t9 * t9;
    double t26 = t22 * t6 / t23;
    double t27 = 0.123235e0 * t26;
    double t28 = 0.379785e1 * t15 + t17 + t19 + t27;
    double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
    double t32 = std::log(t31);
    double t34 = 0.621814e-1 * t14 * t32;
    double t35 = rho0 - rho1;
    double t36 = t35 * t35;
    double t37 = t36 * t36;
    double t38 = t8 * t8;
    double t39 = t38 * t38;
    double t40 = 0.1e1 / t39;
    double t41 = t37 * t40;
    double t42 = 0.1e1 / t8;
    double t43 = t35 * t42;
    double t44 = 0.1e1 + t43;
    double t45 = t44 <= ZETA_THRESHOLD_C_R2SCAN;
    double t46 = std::cbrt(ZETA_THRESHOLD_C_R2SCAN);
    double t47 = t46 * ZETA_THRESHOLD_C_R2SCAN;
    double t48 = std::cbrt(t44);
    double t49 = t48 * t44;
    double t50 = my_piecewise3(t45, t47, t49);
    double t51 = 0.1e1 - t43;
    double t52 = t51 <= ZETA_THRESHOLD_C_R2SCAN;
    double t53 = std::cbrt(t51);
    double t54 = t53 * t51;
    double t55 = my_piecewise3(t52, t47, t54);
    double t56 = t50 + t55 - 0.2e1;
    double t57 = M_CBRT2;
    double t58 = t57 - 0.1e1;
    double t60 = 0.1e1 / t58 / 0.2e1;
    double t61 = t56 * t60;
    double t63 = 0.1e1 + 0.5137e-1 * t12;
    double t65 = 0.1549425e1 * t12;
    double t66 = 0.420775e0 * t18;
    double t67 = 0.1562925e0 * t26;
    double t68 = 0.705945e1 * t15 + t65 + t66 + t67;
    double t71 = 0.1e1 + 0.32163958997385070134e2 / t68;
    double t72 = std::log(t71);
    double t76 = 0.1e1 + 0.278125e-1 * t12;
    double t78 = 0.905775e0 * t12;
    double t79 = 0.1100325e0 * t18;
    double t80 = 0.1241775e0 * t26;
    double t81 = 0.51785e1 * t15 + t78 + t79 + t80;
    double t84 = 0.1e1 + 0.29608749977793437516e2 / t81;
    double t85 = std::log(t84);
    double t86 = t76 * t85;
    double t88 = -0.310907e-1 * t63 * t72 + t34 - 0.19751673498613801407e-1 * t86;
    double t89 = t61 * t88;
    double t90 = t41 * t89;
    double t92 = 0.19751673498613801407e-1 * t61 * t86;
    double t93 = std::log(0.2e1);
    double t94 = 0.1e1 - t93;
    double t95 = M_PI * M_PI;
    double t98 = t46 * t46;
    double t99 = t48 * t48;
    double t100 = my_piecewise3(t45, t98, t99);
    double t101 = t53 * t53;
    double t102 = my_piecewise3(t52, t98, t101);
    double t104 = t100 / 0.2e1 + t102 / 0.2e1;
    double t105 = t104 * t104;
    double t106 = t105 * t104;
    double t108 = 0.1e1 / t94;
    double t109 = (-t34 + t90 + t92) * t108;
    double t110 = 0.1e1 / t106;
    double t111 = t95 * t110;
    double t113 = std::exp(-t109 * t111);
    double t488 = t105 * t105;
    double t489 = 0.1e1 / t488;
    double t490 = t95 * t489;
    double t491 = t490 * t481;
    double t494 = 0.3e1 * t109 * t491 - t486 * t111;
    double t495 = t494 * t113;
    return t495;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho0__t516(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t14 = 0.1e1 + 0.53425e-1 * t12;
    double t15 = std::sqrt(t12);
    double t17 = 0.8969e0 * t12;
    double t18 = t12 * std::sqrt(t12);
    double t19 = 0.204775e0 * t18;
    double t20 = t2 * t2;
    double t21 = t4 * t4;
    double t22 = t20 * t21;
    double t23 = t9 * t9;
    double t26 = t22 * t6 / t23;
    double t27 = 0.123235e0 * t26;
    double t28 = 0.379785e1 * t15 + t17 + t19 + t27;
    double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
    double t32 = std::log(t31);
    double t34 = 0.621814e-1 * t14 * t32;
    double t35 = rho0 - rho1;
    double t36 = t35 * t35;
    double t37 = t36 * t36;
    double t38 = t8 * t8;
    double t39 = t38 * t38;
    double t40 = 0.1e1 / t39;
    double t41 = t37 * t40;
    double t42 = 0.1e1 / t8;
    double t43 = t35 * t42;
    double t44 = 0.1e1 + t43;
    double t45 = t44 <= ZETA_THRESHOLD_C_R2SCAN;
    double t46 = std::cbrt(ZETA_THRESHOLD_C_R2SCAN);
    double t47 = t46 * ZETA_THRESHOLD_C_R2SCAN;
    double t48 = std::cbrt(t44);
    double t49 = t48 * t44;
    double t50 = my_piecewise3(t45, t47, t49);
    double t51 = 0.1e1 - t43;
    double t52 = t51 <= ZETA_THRESHOLD_C_R2SCAN;
    double t53 = std::cbrt(t51);
    double t54 = t53 * t51;
    double t55 = my_piecewise3(t52, t47, t54);
    double t56 = t50 + t55 - 0.2e1;
    double t57 = M_CBRT2;
    double t58 = t57 - 0.1e1;
    double t60 = 0.1e1 / t58 / 0.2e1;
    double t61 = t56 * t60;
    double t63 = 0.1e1 + 0.5137e-1 * t12;
    double t65 = 0.1549425e1 * t12;
    double t66 = 0.420775e0 * t18;
    double t67 = 0.1562925e0 * t26;
    double t68 = 0.705945e1 * t15 + t65 + t66 + t67;
    double t71 = 0.1e1 + 0.32163958997385070134e2 / t68;
    double t72 = std::log(t71);
    double t76 = 0.1e1 + 0.278125e-1 * t12;
    double t78 = 0.905775e0 * t12;
    double t79 = 0.1100325e0 * t18;
    double t80 = 0.1241775e0 * t26;
    double t81 = 0.51785e1 * t15 + t78 + t79 + t80;
    double t84 = 0.1e1 + 0.29608749977793437516e2 / t81;
    double t85 = std::log(t84);
    double t86 = t76 * t85;
    double t88 = -0.310907e-1 * t63 * t72 + t34 - 0.19751673498613801407e-1 * t86;
    double t89 = t61 * t88;
    double t90 = t41 * t89;
    double t92 = 0.19751673498613801407e-1 * t61 * t86;
    double t93 = std::log(0.2e1);
    double t94 = 0.1e1 - t93;
    double t95 = M_PI * M_PI;
    double t98 = t46 * t46;
    double t99 = t48 * t48;
    double t100 = my_piecewise3(t45, t98, t99);
    double t101 = t53 * t53;
    double t102 = my_piecewise3(t52, t98, t101);
    double t104 = t100 / 0.2e1 + t102 / 0.2e1;
    double t105 = t104 * t104;
    double t106 = t105 * t104;
    double t108 = 0.1e1 / t94;
    double t109 = (-t34 + t90 + t92) * t108;
    double t110 = 0.1e1 / t106;
    double t111 = t95 * t110;
    double t113 = std::exp(-t109 * t111);
    double t114 = t113 - 0.1e1;
    double t116 = 0.1e1 + 0.25e-1 * t12;
    double t118 = 0.1e1 + 0.4445e-1 * t12;
    double t122 = sigma0 + 0.2e1 * sigma1 + sigma2;
    double t128 = 0.1e1 / t105;
    double t133 = 0.1e1 / t114;
    double t500 = t38 * t8;
    double t502 = 0.1e1 / t23 / t500;
    double t505 = t57 * t128;
    double t506 = t108 * t133;
    double t507 = t505 * t506;
    double t510 = t118 * t118;
    double t511 = 0.1e1 / t510;
    double t512 = t116 * t511;
    double t513 = t122 * t502;
    double t514 = t512 * t513;
    double t516 = 0.48787202696913915093e-2 * t514 * t507;
    return t516;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho0__t562(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t23 = t9 * t9;
    double t35 = rho0 - rho1;
    double t38 = t8 * t8;
    double t39 = t38 * t38;
    double t42 = 0.1e1 / t8;
    double t43 = t35 * t42;
    double t44 = 0.1e1 + t43;
    double t45 = t44 <= ZETA_THRESHOLD_C_R2SCAN;
    double t48 = std::cbrt(t44);
    double t51 = 0.1e1 - t43;
    double t52 = t51 <= ZETA_THRESHOLD_C_R2SCAN;
    double t53 = std::cbrt(t51);
    double t57 = M_CBRT2;
    double t95 = M_PI * M_PI;
    double t99 = t48 * t48;
    double t101 = t53 * t53;
    double t122 = sigma0 + 0.2e1 * sigma1 + sigma2;
    double t255 = M_CBRT6;
    double t256 = std::cbrt(t95);
    double t257 = t256 * t256;
    double t258 = 0.1e1 / t257;
    double t260 = t57 * t57;
    double t263 = 0.1e1 / t23 / t38;
    double t265 = t255 * t255;
    double t267 = 0.1e1 / t256 / t95;
    double t268 = t265 * t267;
    double t269 = t122 * t122;
    double t270 = t57 * t269;
    double t271 = t39 * t8;
    double t273 = 0.1e1 / t9 / t271;
    double t277 = std::exp(-0.20444604078896369094e0 * t268 * t270 * t273);
    double t413 = 0.1e1 / t38;
    double t414 = t35 * t413;
    double t415 = t42 - t414;
    double t419 = -t415;
    double t550 = t258 * t260;
    double t551 = t550 * t122;
    double t552 = t263 * t277;
    double t555 = my_piecewise3(t45, 0, 0.5e1 / 0.3e1 * t99 * t415);
    double t558 = my_piecewise3(t52, 0, 0.5e1 / 0.3e1 * t101 * t419);
    double t560 = t555 / 0.2e1 + t558 / 0.2e1;
    double t561 = t552 * t560;
    double t562 = t551 * t561;
    return t562;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho0__t594(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t15 = std::sqrt(t12);
    double t35 = rho0 - rho1;
    double t36 = t35 * t35;
    double t37 = t36 * t36;
    double t38 = t8 * t8;
    double t39 = t38 * t38;
    double t42 = 0.1e1 / t8;
    double t43 = t35 * t42;
    double t44 = 0.1e1 + t43;
    double t45 = t44 <= ZETA_THRESHOLD_C_R2SCAN;
    double t46 = std::cbrt(ZETA_THRESHOLD_C_R2SCAN);
    double t47 = t46 * ZETA_THRESHOLD_C_R2SCAN;
    double t48 = std::cbrt(t44);
    double t49 = t48 * t44;
    double t50 = my_piecewise3(t45, t47, t49);
    double t51 = 0.1e1 - t43;
    double t52 = t51 <= ZETA_THRESHOLD_C_R2SCAN;
    double t53 = std::cbrt(t51);
    double t54 = t53 * t51;
    double t55 = my_piecewise3(t52, t47, t54);
    double t56 = t50 + t55 - 0.2e1;
    double t57 = M_CBRT2;
    double t58 = t57 - 0.1e1;
    double t60 = 0.1e1 / t58 / 0.2e1;
    double t148 = std::sqrt(0.4e1);
    double t149 = t148 * t15;
    double t151 = 0.3138525e-1 * t12;
    double t152 = 0.1e1 + 0.22225e-1 * t149 + t151;
    double t153 = t152 * t152;
    double t158 = 0.1e1 - 0.2363e1 * t58 * t56 * t60;
    double t160 = t37 * t37;
    double t161 = t160 * t37;
    double t162 = t39 * t39;
    double t163 = t162 * t39;
    double t164 = 0.1e1 / t163;
    double t166 = -t161 * t164 + 0.1e1;
    double t167 = 0.1e1 / t15;
    double t168 = t148 * t167;
    double t170 = 0.4445e-1 * t168 + 0.125541e0;
    double t171 = t166 * t170;
    double t377 = 0.1e1 / t9 / t8;
    double t378 = t7 * t377;
    double t390 = t5 * t378;
    double t583 = 0.1e1 / t153 / t152;
    double t584 = t583 * t158;
    double t585 = std::sqrt(std::cbrt(0.4e1));
    double t586 = t585 * t167;
    double t587 = t5 * t377;
    double t588 = t586 * t587;
    double t590 = 0.1046175e-1 * t390;
    double t591 = -0.14816666666666666667e-1 * t588 - t590;
    double t594 = 0.571528e-1 * t584 * t171 * t591;
    return t594;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho0__t596(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t15 = std::sqrt(t12);
    double t35 = rho0 - rho1;
    double t38 = t8 * t8;
    double t42 = 0.1e1 / t8;
    double t43 = t35 * t42;
    double t44 = 0.1e1 + t43;
    double t45 = t44 <= ZETA_THRESHOLD_C_R2SCAN;
    double t48 = std::cbrt(t44);
    double t51 = 0.1e1 - t43;
    double t52 = t51 <= ZETA_THRESHOLD_C_R2SCAN;
    double t53 = std::cbrt(t51);
    double t57 = M_CBRT2;
    double t58 = t57 - 0.1e1;
    double t148 = std::sqrt(0.4e1);
    double t149 = t148 * t15;
    double t151 = 0.3138525e-1 * t12;
    double t152 = 0.1e1 + 0.22225e-1 * t149 + t151;
    double t153 = t152 * t152;
    double t154 = 0.1e1 / t153;
    double t413 = 0.1e1 / t38;
    double t414 = t35 * t413;
    double t415 = t42 - t414;
    double t418 = my_piecewise3(t45, 0, 0.4e1 / 0.3e1 * t48 * t415);
    double t419 = -t415;
    double t422 = my_piecewise3(t52, 0, 0.4e1 / 0.3e1 * t53 * t419);
    double t423 = t418 + t422;
    double t595 = t154 * t58;
    double t596 = t595 * t423;
    return t596;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho0__t607(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t8 = rho0 + rho1;
    double t35 = rho0 - rho1;
    double t36 = t35 * t35;
    double t37 = t36 * t36;
    double t38 = t8 * t8;
    double t39 = t38 * t38;
    double t160 = t37 * t37;
    double t161 = t160 * t37;
    double t162 = t39 * t39;
    double t163 = t162 * t39;
    double t164 = 0.1e1 / t163;
    double t271 = t39 * t8;
    double t405 = t36 * t35;
    double t601 = t160 * t405;
    double t602 = t601 * t164;
    double t603 = t162 * t271;
    double t604 = 0.1e1 / t603;
    double t605 = t161 * t604;
    double t607 = -0.12e2 * t602 + 0.12e2 * t605;
    return t607;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho0__t608(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t607 = mgga_c_r2scan_vrho0__t607(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t15 = std::sqrt(t12);
    double t148 = std::sqrt(0.4e1);
    double t167 = 0.1e1 / t15;
    double t168 = t148 * t167;
    double t170 = 0.4445e-1 * t168 + 0.125541e0;
    double t608 = t607 * t170;
    return t608;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho0__t619(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t15 = std::sqrt(t12);
    double t35 = rho0 - rho1;
    double t36 = t35 * t35;
    double t37 = t36 * t36;
    double t38 = t8 * t8;
    double t39 = t38 * t38;
    double t42 = 0.1e1 / t8;
    double t43 = t35 * t42;
    double t44 = 0.1e1 + t43;
    double t45 = t44 <= ZETA_THRESHOLD_C_R2SCAN;
    double t46 = std::cbrt(ZETA_THRESHOLD_C_R2SCAN);
    double t47 = t46 * ZETA_THRESHOLD_C_R2SCAN;
    double t48 = std::cbrt(t44);
    double t49 = t48 * t44;
    double t50 = my_piecewise3(t45, t47, t49);
    double t51 = 0.1e1 - t43;
    double t52 = t51 <= ZETA_THRESHOLD_C_R2SCAN;
    double t53 = std::cbrt(t51);
    double t54 = t53 * t51;
    double t55 = my_piecewise3(t52, t47, t54);
    double t56 = t50 + t55 - 0.2e1;
    double t57 = M_CBRT2;
    double t58 = t57 - 0.1e1;
    double t60 = 0.1e1 / t58 / 0.2e1;
    double t148 = std::sqrt(0.4e1);
    double t149 = t148 * t15;
    double t151 = 0.3138525e-1 * t12;
    double t152 = 0.1e1 + 0.22225e-1 * t149 + t151;
    double t153 = t152 * t152;
    double t154 = 0.1e1 / t153;
    double t158 = 0.1e1 - 0.2363e1 * t58 * t56 * t60;
    double t159 = t154 * t158;
    double t160 = t37 * t37;
    double t161 = t160 * t37;
    double t162 = t39 * t39;
    double t163 = t162 * t39;
    double t164 = 0.1e1 / t163;
    double t166 = -t161 * t164 + 0.1e1;
    double t377 = 0.1e1 / t9 / t8;
    double t585 = std::sqrt(std::cbrt(0.4e1));
    double t611 = t166 * t585;
    double t612 = t159 * t611;
    double t614 = 0.1e1 / t15 / t12;
    double t615 = t614 * t2;
    double t616 = t4 * t377;
    double t617 = t615 * t616;
    double t619 = 0.84681398666666666666e-3 * t612 * t617;
    return t619;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho0__t624(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t15 = std::sqrt(t12);
    double t17 = 0.8969e0 * t12;
    double t18 = t12 * std::sqrt(t12);
    double t19 = 0.204775e0 * t18;
    double t20 = t2 * t2;
    double t21 = t4 * t4;
    double t22 = t20 * t21;
    double t23 = t9 * t9;
    double t26 = t22 * t6 / t23;
    double t27 = 0.123235e0 * t26;
    double t148 = std::sqrt(0.4e1);
    double t149 = t148 * t15;
    double t167 = 0.1e1 / t15;
    double t175 = 0.1898925e1 * t149 + t17 + t19 + t27;
    double t178 = 0.1e1 + 0.16081979498692535067e2 / t175;
    double t181 = t175 * t175;
    double t182 = 0.1e1 / t181;
    double t185 = std::sqrt(t12);
    double t189 = 0.1e1 / t178;
    double t377 = 0.1e1 / t9 / t8;
    double t378 = t7 * t377;
    double t386 = t4 * t7;
    double t387 = t386 * t377;
    double t390 = t5 * t378;
    double t391 = 0.29896666666666666667e0 * t390;
    double t392 = t185 * t2;
    double t393 = t392 * t387;
    double t394 = 0.1023875e0 * t393;
    double t398 = t22 * t6 / t23 / t8;
    double t399 = 0.82156666666666666667e-1 * t398;
    double t585 = std::sqrt(std::cbrt(0.4e1));
    double t586 = t585 * t167;
    double t587 = t5 * t377;
    double t588 = t586 * t587;
    double t621 = -0.126595e1 * t588 - t391 - t394 - t399;
    double t624 = 0.2137e0 * t182 * t621 * t189;
    return t624;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho0__t635(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t14 = 0.1e1 + 0.53425e-1 * t12;
    double t15 = std::sqrt(t12);
    double t17 = 0.8969e0 * t12;
    double t18 = t12 * std::sqrt(t12);
    double t19 = 0.204775e0 * t18;
    double t20 = t2 * t2;
    double t21 = t4 * t4;
    double t22 = t20 * t21;
    double t23 = t9 * t9;
    double t26 = t22 * t6 / t23;
    double t27 = 0.123235e0 * t26;
    double t148 = std::sqrt(0.4e1);
    double t149 = t148 * t15;
    double t167 = 0.1e1 / t15;
    double t168 = t148 * t167;
    double t175 = 0.1898925e1 * t149 + t17 + t19 + t27;
    double t178 = 0.1e1 + 0.16081979498692535067e2 / t175;
    double t181 = t175 * t175;
    double t185 = std::sqrt(t12);
    double t188 = 0.379785e1 * t168 + 0.35876e1 + 0.122865e1 * t185 + 0.24647e0 * t12;
    double t189 = 0.1e1 / t178;
    double t190 = t188 * t189;
    double t377 = 0.1e1 / t9 / t8;
    double t378 = t7 * t377;
    double t386 = t4 * t7;
    double t387 = t386 * t377;
    double t390 = t5 * t378;
    double t391 = 0.29896666666666666667e0 * t390;
    double t392 = t185 * t2;
    double t393 = t392 * t387;
    double t394 = 0.1023875e0 * t393;
    double t398 = t22 * t6 / t23 / t8;
    double t399 = 0.82156666666666666667e-1 * t398;
    double t585 = std::sqrt(std::cbrt(0.4e1));
    double t586 = t585 * t167;
    double t587 = t5 * t377;
    double t588 = t586 * t587;
    double t621 = -0.126595e1 * t588 - t391 - t394 - t399;
    double t630 = t181 * t175;
    double t631 = 0.1e1 / t630;
    double t632 = t14 * t631;
    double t635 = 0.2e1 * t632 * t190 * t621;
    return t635;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho0__t647(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t14 = 0.1e1 + 0.53425e-1 * t12;
    double t15 = std::sqrt(t12);
    double t17 = 0.8969e0 * t12;
    double t18 = t12 * std::sqrt(t12);
    double t19 = 0.204775e0 * t18;
    double t20 = t2 * t2;
    double t21 = t4 * t4;
    double t22 = t20 * t21;
    double t23 = t9 * t9;
    double t26 = t22 * t6 / t23;
    double t27 = 0.123235e0 * t26;
    double t148 = std::sqrt(0.4e1);
    double t149 = t148 * t15;
    double t175 = 0.1898925e1 * t149 + t17 + t19 + t27;
    double t178 = 0.1e1 + 0.16081979498692535067e2 / t175;
    double t181 = t175 * t175;
    double t182 = 0.1e1 / t181;
    double t183 = t14 * t182;
    double t189 = 0.1e1 / t178;
    double t377 = 0.1e1 / t9 / t8;
    double t378 = t7 * t377;
    double t386 = t4 * t7;
    double t387 = t386 * t377;
    double t390 = t5 * t378;
    double t585 = std::sqrt(std::cbrt(0.4e1));
    double t587 = t5 * t377;
    double t614 = 0.1e1 / t15 / t12;
    double t636 = t585 * t614;
    double t637 = t636 * t587;
    double t639 = 0.1e1 / std::sqrt(t12);
    double t640 = t639 * t2;
    double t641 = t640 * t387;
    double t644 = 0.25319e1 * t637 - 0.204775e0 * t641 - 0.82156666666666666667e-1 * t390;
    double t645 = t644 * t189;
    double t647 = 0.1e1 * t183 * t645;
    return t647;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho0__t656(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t14 = 0.1e1 + 0.53425e-1 * t12;
    double t15 = std::sqrt(t12);
    double t17 = 0.8969e0 * t12;
    double t18 = t12 * std::sqrt(t12);
    double t19 = 0.204775e0 * t18;
    double t20 = t2 * t2;
    double t21 = t4 * t4;
    double t22 = t20 * t21;
    double t23 = t9 * t9;
    double t26 = t22 * t6 / t23;
    double t27 = 0.123235e0 * t26;
    double t148 = std::sqrt(0.4e1);
    double t149 = t148 * t15;
    double t167 = 0.1e1 / t15;
    double t168 = t148 * t167;
    double t175 = 0.1898925e1 * t149 + t17 + t19 + t27;
    double t178 = 0.1e1 + 0.16081979498692535067e2 / t175;
    double t181 = t175 * t175;
    double t185 = std::sqrt(t12);
    double t188 = 0.379785e1 * t168 + 0.35876e1 + 0.122865e1 * t185 + 0.24647e0 * t12;
    double t377 = 0.1e1 / t9 / t8;
    double t378 = t7 * t377;
    double t386 = t4 * t7;
    double t387 = t386 * t377;
    double t390 = t5 * t378;
    double t391 = 0.29896666666666666667e0 * t390;
    double t392 = t185 * t2;
    double t393 = t392 * t387;
    double t394 = 0.1023875e0 * t393;
    double t398 = t22 * t6 / t23 / t8;
    double t399 = 0.82156666666666666667e-1 * t398;
    double t585 = std::sqrt(std::cbrt(0.4e1));
    double t586 = t585 * t167;
    double t587 = t5 * t377;
    double t588 = t586 * t587;
    double t621 = -0.126595e1 * t588 - t391 - t394 - t399;
    double t648 = t181 * t181;
    double t649 = 0.1e1 / t648;
    double t650 = t14 * t649;
    double t651 = t178 * t178;
    double t652 = 0.1e1 / t651;
    double t653 = t188 * t652;
    double t656 = 0.16081979498692535067e2 * t650 * t653 * t621;
    return t656;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho0__t658(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t14 = 0.1e1 + 0.53425e-1 * t12;
    double t15 = std::sqrt(t12);
    double t17 = 0.8969e0 * t12;
    double t18 = t12 * std::sqrt(t12);
    double t19 = 0.204775e0 * t18;
    double t20 = t2 * t2;
    double t21 = t4 * t4;
    double t22 = t20 * t21;
    double t23 = t9 * t9;
    double t26 = t22 * t6 / t23;
    double t27 = 0.123235e0 * t26;
    double t35 = rho0 - rho1;
    double t36 = t35 * t35;
    double t38 = t8 * t8;
    double t39 = t38 * t38;
    double t40 = 0.1e1 / t39;
    double t42 = 0.1e1 / t8;
    double t43 = t35 * t42;
    double t44 = 0.1e1 + t43;
    double t45 = t44 <= ZETA_THRESHOLD_C_R2SCAN;
    double t46 = std::cbrt(ZETA_THRESHOLD_C_R2SCAN);
    double t47 = t46 * ZETA_THRESHOLD_C_R2SCAN;
    double t48 = std::cbrt(t44);
    double t49 = t48 * t44;
    double t50 = my_piecewise3(t45, t47, t49);
    double t51 = 0.1e1 - t43;
    double t52 = t51 <= ZETA_THRESHOLD_C_R2SCAN;
    double t53 = std::cbrt(t51);
    double t54 = t53 * t51;
    double t55 = my_piecewise3(t52, t47, t54);
    double t56 = t50 + t55 - 0.2e1;
    double t57 = M_CBRT2;
    double t58 = t57 - 0.1e1;
    double t60 = 0.1e1 / t58 / 0.2e1;
    double t61 = t56 * t60;
    double t63 = 0.1e1 + 0.5137e-1 * t12;
    double t65 = 0.1549425e1 * t12;
    double t66 = 0.420775e0 * t18;
    double t67 = 0.1562925e0 * t26;
    double t76 = 0.1e1 + 0.278125e-1 * t12;
    double t78 = 0.905775e0 * t12;
    double t79 = 0.1100325e0 * t18;
    double t80 = 0.1241775e0 * t26;
    double t148 = std::sqrt(0.4e1);
    double t149 = t148 * t15;
    double t167 = 0.1e1 / t15;
    double t168 = t148 * t167;
    double t175 = 0.1898925e1 * t149 + t17 + t19 + t27;
    double t178 = 0.1e1 + 0.16081979498692535067e2 / t175;
    double t179 = std::log(t178);
    double t180 = 0.1328816518e-1 * t179;
    double t181 = t175 * t175;
    double t182 = 0.1e1 / t181;
    double t183 = t14 * t182;
    double t185 = std::sqrt(t12);
    double t188 = 0.379785e1 * t168 + 0.35876e1 + 0.122865e1 * t185 + 0.24647e0 * t12;
    double t189 = 0.1e1 / t178;
    double t190 = t188 * t189;
    double t192 = 0.1e1 * t183 * t190;
    double t194 = 0.3529725e1 * t149 + t65 + t66 + t67;
    double t197 = 0.1e1 + 0.32163958997385070134e2 / t194;
    double t198 = std::log(t197);
    double t200 = t194 * t194;
    double t201 = 0.1e1 / t200;
    double t202 = t63 * t201;
    double t206 = 0.705945e1 * t168 + 0.61977e1 + 0.252465e1 * t185 + 0.312585e0 * t12;
    double t207 = 0.1e1 / t197;
    double t208 = t206 * t207;
    double t212 = 0.258925e1 * t149 + t78 + t79 + t80;
    double t215 = 0.1e1 + 0.29608749977793437516e2 / t212;
    double t216 = std::log(t215);
    double t218 = t212 * t212;
    double t219 = 0.1e1 / t218;
    double t220 = t76 * t219;
    double t224 = 0.51785e1 * t168 + 0.36231e1 + 0.660195e0 * t185 + 0.248355e0 * t12;
    double t225 = 0.1e1 / t215;
    double t226 = t224 * t225;
    double t229 = -0.6388517036e-2 * t198 + 0.1e1 * t202 * t208 + t180 - t192 - 0.21973736767207854065e-2 * t216 + 0.5848223622634646207e0 * t220 * t226;
    double t230 = t61 * t229;
    double t405 = t36 * t35;
    double t406 = t405 * t40;
    double t658 = 0.4e1 * t406 * t230;
    return t658;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho0__t660(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t14 = 0.1e1 + 0.53425e-1 * t12;
    double t15 = std::sqrt(t12);
    double t17 = 0.8969e0 * t12;
    double t18 = t12 * std::sqrt(t12);
    double t19 = 0.204775e0 * t18;
    double t20 = t2 * t2;
    double t21 = t4 * t4;
    double t22 = t20 * t21;
    double t23 = t9 * t9;
    double t26 = t22 * t6 / t23;
    double t27 = 0.123235e0 * t26;
    double t35 = rho0 - rho1;
    double t36 = t35 * t35;
    double t37 = t36 * t36;
    double t38 = t8 * t8;
    double t39 = t38 * t38;
    double t42 = 0.1e1 / t8;
    double t43 = t35 * t42;
    double t44 = 0.1e1 + t43;
    double t45 = t44 <= ZETA_THRESHOLD_C_R2SCAN;
    double t46 = std::cbrt(ZETA_THRESHOLD_C_R2SCAN);
    double t47 = t46 * ZETA_THRESHOLD_C_R2SCAN;
    double t48 = std::cbrt(t44);
    double t49 = t48 * t44;
    double t50 = my_piecewise3(t45, t47, t49);
    double t51 = 0.1e1 - t43;
    double t52 = t51 <= ZETA_THRESHOLD_C_R2SCAN;
    double t53 = std::cbrt(t51);
    double t54 = t53 * t51;
    double t55 = my_piecewise3(t52, t47, t54);
    double t56 = t50 + t55 - 0.2e1;
    double t57 = M_CBRT2;
    double t58 = t57 - 0.1e1;
    double t60 = 0.1e1 / t58 / 0.2e1;
    double t61 = t56 * t60;
    double t63 = 0.1e1 + 0.5137e-1 * t12;
    double t65 = 0.1549425e1 * t12;
    double t66 = 0.420775e0 * t18;
    double t67 = 0.1562925e0 * t26;
    double t76 = 0.1e1 + 0.278125e-1 * t12;
    double t78 = 0.905775e0 * t12;
    double t79 = 0.1100325e0 * t18;
    double t80 = 0.1241775e0 * t26;
    double t148 = std::sqrt(0.4e1);
    double t149 = t148 * t15;
    double t167 = 0.1e1 / t15;
    double t168 = t148 * t167;
    double t175 = 0.1898925e1 * t149 + t17 + t19 + t27;
    double t178 = 0.1e1 + 0.16081979498692535067e2 / t175;
    double t179 = std::log(t178);
    double t180 = 0.1328816518e-1 * t179;
    double t181 = t175 * t175;
    double t182 = 0.1e1 / t181;
    double t183 = t14 * t182;
    double t185 = std::sqrt(t12);
    double t188 = 0.379785e1 * t168 + 0.35876e1 + 0.122865e1 * t185 + 0.24647e0 * t12;
    double t189 = 0.1e1 / t178;
    double t190 = t188 * t189;
    double t192 = 0.1e1 * t183 * t190;
    double t194 = 0.3529725e1 * t149 + t65 + t66 + t67;
    double t197 = 0.1e1 + 0.32163958997385070134e2 / t194;
    double t198 = std::log(t197);
    double t200 = t194 * t194;
    double t201 = 0.1e1 / t200;
    double t202 = t63 * t201;
    double t206 = 0.705945e1 * t168 + 0.61977e1 + 0.252465e1 * t185 + 0.312585e0 * t12;
    double t207 = 0.1e1 / t197;
    double t208 = t206 * t207;
    double t212 = 0.258925e1 * t149 + t78 + t79 + t80;
    double t215 = 0.1e1 + 0.29608749977793437516e2 / t212;
    double t216 = std::log(t215);
    double t218 = t212 * t212;
    double t219 = 0.1e1 / t218;
    double t220 = t76 * t219;
    double t224 = 0.51785e1 * t168 + 0.36231e1 + 0.660195e0 * t185 + 0.248355e0 * t12;
    double t225 = 0.1e1 / t215;
    double t226 = t224 * t225;
    double t229 = -0.6388517036e-2 * t198 + 0.1e1 * t202 * t208 + t180 - t192 - 0.21973736767207854065e-2 * t216 + 0.5848223622634646207e0 * t220 * t226;
    double t230 = t61 * t229;
    double t271 = t39 * t8;
    double t409 = 0.1e1 / t271;
    double t410 = t37 * t409;
    double t660 = 0.4e1 * t410 * t230;
    return t660;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho0__t661(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t424 = mgga_c_r2scan_vrho0__t424(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t14 = 0.1e1 + 0.53425e-1 * t12;
    double t15 = std::sqrt(t12);
    double t17 = 0.8969e0 * t12;
    double t18 = t12 * std::sqrt(t12);
    double t19 = 0.204775e0 * t18;
    double t20 = t2 * t2;
    double t21 = t4 * t4;
    double t22 = t20 * t21;
    double t23 = t9 * t9;
    double t26 = t22 * t6 / t23;
    double t27 = 0.123235e0 * t26;
    double t63 = 0.1e1 + 0.5137e-1 * t12;
    double t65 = 0.1549425e1 * t12;
    double t66 = 0.420775e0 * t18;
    double t67 = 0.1562925e0 * t26;
    double t76 = 0.1e1 + 0.278125e-1 * t12;
    double t78 = 0.905775e0 * t12;
    double t79 = 0.1100325e0 * t18;
    double t80 = 0.1241775e0 * t26;
    double t148 = std::sqrt(0.4e1);
    double t149 = t148 * t15;
    double t167 = 0.1e1 / t15;
    double t168 = t148 * t167;
    double t175 = 0.1898925e1 * t149 + t17 + t19 + t27;
    double t178 = 0.1e1 + 0.16081979498692535067e2 / t175;
    double t179 = std::log(t178);
    double t180 = 0.1328816518e-1 * t179;
    double t181 = t175 * t175;
    double t182 = 0.1e1 / t181;
    double t183 = t14 * t182;
    double t185 = std::sqrt(t12);
    double t188 = 0.379785e1 * t168 + 0.35876e1 + 0.122865e1 * t185 + 0.24647e0 * t12;
    double t189 = 0.1e1 / t178;
    double t190 = t188 * t189;
    double t192 = 0.1e1 * t183 * t190;
    double t194 = 0.3529725e1 * t149 + t65 + t66 + t67;
    double t197 = 0.1e1 + 0.32163958997385070134e2 / t194;
    double t198 = std::log(t197);
    double t200 = t194 * t194;
    double t201 = 0.1e1 / t200;
    double t202 = t63 * t201;
    double t206 = 0.705945e1 * t168 + 0.61977e1 + 0.252465e1 * t185 + 0.312585e0 * t12;
    double t207 = 0.1e1 / t197;
    double t208 = t206 * t207;
    double t212 = 0.258925e1 * t149 + t78 + t79 + t80;
    double t215 = 0.1e1 + 0.29608749977793437516e2 / t212;
    double t216 = std::log(t215);
    double t218 = t212 * t212;
    double t219 = 0.1e1 / t218;
    double t220 = t76 * t219;
    double t224 = 0.51785e1 * t168 + 0.36231e1 + 0.660195e0 * t185 + 0.248355e0 * t12;
    double t225 = 0.1e1 / t215;
    double t226 = t224 * t225;
    double t229 = -0.6388517036e-2 * t198 + 0.1e1 * t202 * t208 + t180 - t192 - 0.21973736767207854065e-2 * t216 + 0.5848223622634646207e0 * t220 * t226;
    double t661 = t424 * t229;
    return t661;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho0__t695(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t15 = std::sqrt(t12);
    double t20 = t2 * t2;
    double t21 = t4 * t4;
    double t22 = t20 * t21;
    double t23 = t9 * t9;
    double t167 = 0.1e1 / t15;
    double t185 = std::sqrt(t12);
    double t377 = 0.1e1 / t9 / t8;
    double t378 = t7 * t377;
    double t386 = t4 * t7;
    double t387 = t386 * t377;
    double t390 = t5 * t378;
    double t392 = t185 * t2;
    double t393 = t392 * t387;
    double t398 = t22 * t6 / t23 / t8;
    double t449 = 0.301925e0 * t390;
    double t450 = 0.5501625e-1 * t393;
    double t451 = 0.82785e-1 * t398;
    double t585 = std::sqrt(std::cbrt(0.4e1));
    double t586 = t585 * t167;
    double t587 = t5 * t377;
    double t588 = t586 * t587;
    double t695 = -0.17261666666666666667e1 * t588 - t449 - t450 - t451;
    return t695;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho0__t725(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t624 = mgga_c_r2scan_vrho0__t624(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t635 = mgga_c_r2scan_vrho0__t635(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t647 = mgga_c_r2scan_vrho0__t647(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t656 = mgga_c_r2scan_vrho0__t656(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t695 = mgga_c_r2scan_vrho0__t695(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t15 = std::sqrt(t12);
    double t17 = 0.8969e0 * t12;
    double t18 = t12 * std::sqrt(t12);
    double t19 = 0.204775e0 * t18;
    double t20 = t2 * t2;
    double t21 = t4 * t4;
    double t22 = t20 * t21;
    double t23 = t9 * t9;
    double t26 = t22 * t6 / t23;
    double t27 = 0.123235e0 * t26;
    double t63 = 0.1e1 + 0.5137e-1 * t12;
    double t65 = 0.1549425e1 * t12;
    double t66 = 0.420775e0 * t18;
    double t67 = 0.1562925e0 * t26;
    double t76 = 0.1e1 + 0.278125e-1 * t12;
    double t78 = 0.905775e0 * t12;
    double t79 = 0.1100325e0 * t18;
    double t80 = 0.1241775e0 * t26;
    double t148 = std::sqrt(0.4e1);
    double t149 = t148 * t15;
    double t167 = 0.1e1 / t15;
    double t168 = t148 * t167;
    double t175 = 0.1898925e1 * t149 + t17 + t19 + t27;
    double t178 = 0.1e1 + 0.16081979498692535067e2 / t175;
    double t181 = t175 * t175;
    double t182 = 0.1e1 / t181;
    double t185 = std::sqrt(t12);
    double t188 = 0.379785e1 * t168 + 0.35876e1 + 0.122865e1 * t185 + 0.24647e0 * t12;
    double t189 = 0.1e1 / t178;
    double t190 = t188 * t189;
    double t194 = 0.3529725e1 * t149 + t65 + t66 + t67;
    double t197 = 0.1e1 + 0.32163958997385070134e2 / t194;
    double t200 = t194 * t194;
    double t201 = 0.1e1 / t200;
    double t202 = t63 * t201;
    double t206 = 0.705945e1 * t168 + 0.61977e1 + 0.252465e1 * t185 + 0.312585e0 * t12;
    double t207 = 0.1e1 / t197;
    double t208 = t206 * t207;
    double t212 = 0.258925e1 * t149 + t78 + t79 + t80;
    double t215 = 0.1e1 + 0.29608749977793437516e2 / t212;
    double t218 = t212 * t212;
    double t219 = 0.1e1 / t218;
    double t220 = t76 * t219;
    double t224 = 0.51785e1 * t168 + 0.36231e1 + 0.660195e0 * t185 + 0.248355e0 * t12;
    double t225 = 0.1e1 / t215;
    double t226 = t224 * t225;
    double t377 = 0.1e1 / t9 / t8;
    double t378 = t7 * t377;
    double t386 = t4 * t7;
    double t387 = t386 * t377;
    double t390 = t5 * t378;
    double t392 = t185 * t2;
    double t393 = t392 * t387;
    double t398 = t22 * t6 / t23 / t8;
    double t434 = 0.516475e0 * t390;
    double t435 = 0.2103875e0 * t393;
    double t436 = 0.104195e0 * t398;
    double t585 = std::sqrt(std::cbrt(0.4e1));
    double t586 = t585 * t167;
    double t587 = t5 * t377;
    double t588 = t586 * t587;
    double t614 = 0.1e1 / t15 / t12;
    double t625 = t5 * t7;
    double t626 = t377 * t182;
    double t629 = 0.17808333333333333333e-1 * t625 * t626 * t190;
    double t636 = t585 * t614;
    double t637 = t636 * t587;
    double t639 = 0.1e1 / std::sqrt(t12);
    double t640 = t639 * t2;
    double t641 = t640 * t387;
    double t664 = -0.235315e1 * t588 - t434 - t435 - t436;
    double t668 = t377 * t201;
    double t672 = t200 * t194;
    double t673 = 0.1e1 / t672;
    double t674 = t63 * t673;
    double t681 = 0.47063e1 * t637 - 0.42077500000000000001e0 * t641 - 0.104195e0 * t390;
    double t682 = t681 * t207;
    double t685 = t200 * t200;
    double t686 = 0.1e1 / t685;
    double t687 = t63 * t686;
    double t688 = t197 * t197;
    double t689 = 0.1e1 / t688;
    double t690 = t206 * t689;
    double t697 = t219 * t695 * t225;
    double t699 = t377 * t219;
    double t703 = t218 * t212;
    double t704 = 0.1e1 / t703;
    double t705 = t76 * t704;
    double t706 = t226 * t695;
    double t712 = 0.34523333333333333333e1 * t637 - 0.1100325e0 * t641 - 0.82785e-1 * t390;
    double t713 = t712 * t225;
    double t716 = t218 * t218;
    double t717 = 0.1e1 / t716;
    double t718 = t76 * t717;
    double t719 = t215 * t215;
    double t720 = 0.1e1 / t719;
    double t721 = t224 * t720;
    double t722 = t721 * t695;
    double t725 = 0.20548e0 * t201 * t664 * t207 - 0.17123333333333333333e-1 * t625 * t668 * t208 - 0.2e1 * t674 * t208 * t664 + 0.1e1 * t202 * t682 + 0.32163958997385070134e2 * t687 * t690 * t664 - t624 + t629 + t635 - t647 - t656 + 0.65061487801810439052e-1 * t697 - 0.54217906501508699211e-2 * t625 * t699 * t226 - 0.11696447245269292414e1 * t705 * t706 + 0.5848223622634646207e0 * t220 * t713 + 0.17315859105681463759e2 * t718 * t722;
    return t725;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho0__t727(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t725 = mgga_c_r2scan_vrho0__t725(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t8 = rho0 + rho1;
    double t35 = rho0 - rho1;
    double t36 = t35 * t35;
    double t37 = t36 * t36;
    double t38 = t8 * t8;
    double t39 = t38 * t38;
    double t40 = 0.1e1 / t39;
    double t41 = t37 * t40;
    double t42 = 0.1e1 / t8;
    double t43 = t35 * t42;
    double t44 = 0.1e1 + t43;
    double t45 = t44 <= ZETA_THRESHOLD_C_R2SCAN;
    double t46 = std::cbrt(ZETA_THRESHOLD_C_R2SCAN);
    double t47 = t46 * ZETA_THRESHOLD_C_R2SCAN;
    double t48 = std::cbrt(t44);
    double t49 = t48 * t44;
    double t50 = my_piecewise3(t45, t47, t49);
    double t51 = 0.1e1 - t43;
    double t52 = t51 <= ZETA_THRESHOLD_C_R2SCAN;
    double t53 = std::cbrt(t51);
    double t54 = t53 * t51;
    double t55 = my_piecewise3(t52, t47, t54);
    double t56 = t50 + t55 - 0.2e1;
    double t57 = M_CBRT2;
    double t58 = t57 - 0.1e1;
    double t60 = 0.1e1 / t58 / 0.2e1;
    double t61 = t56 * t60;
    double t726 = t61 * t725;
    double t727 = t41 * t726;
    return t727;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho0__t747(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t15 = std::sqrt(t12);
    double t18 = t12 * std::sqrt(t12);
    double t20 = t2 * t2;
    double t21 = t4 * t4;
    double t22 = t20 * t21;
    double t23 = t9 * t9;
    double t26 = t22 * t6 / t23;
    double t35 = rho0 - rho1;
    double t42 = 0.1e1 / t8;
    double t43 = t35 * t42;
    double t44 = 0.1e1 + t43;
    double t45 = t44 <= ZETA_THRESHOLD_C_R2SCAN;
    double t46 = std::cbrt(ZETA_THRESHOLD_C_R2SCAN);
    double t47 = t46 * ZETA_THRESHOLD_C_R2SCAN;
    double t48 = std::cbrt(t44);
    double t49 = t48 * t44;
    double t50 = my_piecewise3(t45, t47, t49);
    double t51 = 0.1e1 - t43;
    double t52 = t51 <= ZETA_THRESHOLD_C_R2SCAN;
    double t53 = std::cbrt(t51);
    double t54 = t53 * t51;
    double t55 = my_piecewise3(t52, t47, t54);
    double t56 = t50 + t55 - 0.2e1;
    double t57 = M_CBRT2;
    double t58 = t57 - 0.1e1;
    double t60 = 0.1e1 / t58 / 0.2e1;
    double t61 = t56 * t60;
    double t76 = 0.1e1 + 0.278125e-1 * t12;
    double t78 = 0.905775e0 * t12;
    double t79 = 0.1100325e0 * t18;
    double t80 = 0.1241775e0 * t26;
    double t148 = std::sqrt(0.4e1);
    double t149 = t148 * t15;
    double t212 = 0.258925e1 * t149 + t78 + t79 + t80;
    double t215 = 0.1e1 + 0.29608749977793437516e2 / t212;
    double t218 = t212 * t212;
    double t219 = 0.1e1 / t218;
    double t225 = 0.1e1 / t215;
    double t234 = t61 * t76;
    double t377 = 0.1e1 / t9 / t8;
    double t378 = t7 * t377;
    double t386 = t4 * t7;
    double t387 = t386 * t377;
    double t390 = t5 * t378;
    double t585 = std::sqrt(std::cbrt(0.4e1));
    double t587 = t5 * t377;
    double t614 = 0.1e1 / t15 / t12;
    double t636 = t585 * t614;
    double t637 = t636 * t587;
    double t639 = 0.1e1 / std::sqrt(t12);
    double t640 = t639 * t2;
    double t641 = t640 * t387;
    double t712 = 0.34523333333333333333e1 * t637 - 0.1100325e0 * t641 - 0.82785e-1 * t390;
    double t745 = t219 * t712 * t225;
    double t747 = 0.5848223622634646207e0 * t234 * t745;
    return t747;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho0__t753(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t424 = mgga_c_r2scan_vrho0__t424(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t594 = mgga_c_r2scan_vrho0__t594(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t596 = mgga_c_r2scan_vrho0__t596(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t608 = mgga_c_r2scan_vrho0__t608(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t619 = mgga_c_r2scan_vrho0__t619(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t624 = mgga_c_r2scan_vrho0__t624(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t635 = mgga_c_r2scan_vrho0__t635(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t647 = mgga_c_r2scan_vrho0__t647(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t656 = mgga_c_r2scan_vrho0__t656(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t658 = mgga_c_r2scan_vrho0__t658(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t660 = mgga_c_r2scan_vrho0__t660(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t661 = mgga_c_r2scan_vrho0__t661(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t695 = mgga_c_r2scan_vrho0__t695(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t727 = mgga_c_r2scan_vrho0__t727(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t747 = mgga_c_r2scan_vrho0__t747(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t15 = std::sqrt(t12);
    double t17 = 0.8969e0 * t12;
    double t18 = t12 * std::sqrt(t12);
    double t19 = 0.204775e0 * t18;
    double t20 = t2 * t2;
    double t21 = t4 * t4;
    double t22 = t20 * t21;
    double t23 = t9 * t9;
    double t26 = t22 * t6 / t23;
    double t27 = 0.123235e0 * t26;
    double t35 = rho0 - rho1;
    double t36 = t35 * t35;
    double t37 = t36 * t36;
    double t38 = t8 * t8;
    double t39 = t38 * t38;
    double t40 = 0.1e1 / t39;
    double t41 = t37 * t40;
    double t42 = 0.1e1 / t8;
    double t43 = t35 * t42;
    double t44 = 0.1e1 + t43;
    double t45 = t44 <= ZETA_THRESHOLD_C_R2SCAN;
    double t46 = std::cbrt(ZETA_THRESHOLD_C_R2SCAN);
    double t47 = t46 * ZETA_THRESHOLD_C_R2SCAN;
    double t48 = std::cbrt(t44);
    double t49 = t48 * t44;
    double t50 = my_piecewise3(t45, t47, t49);
    double t51 = 0.1e1 - t43;
    double t52 = t51 <= ZETA_THRESHOLD_C_R2SCAN;
    double t53 = std::cbrt(t51);
    double t54 = t53 * t51;
    double t55 = my_piecewise3(t52, t47, t54);
    double t56 = t50 + t55 - 0.2e1;
    double t57 = M_CBRT2;
    double t58 = t57 - 0.1e1;
    double t60 = 0.1e1 / t58 / 0.2e1;
    double t61 = t56 * t60;
    double t76 = 0.1e1 + 0.278125e-1 * t12;
    double t78 = 0.905775e0 * t12;
    double t79 = 0.1100325e0 * t18;
    double t80 = 0.1241775e0 * t26;
    double t148 = std::sqrt(0.4e1);
    double t149 = t148 * t15;
    double t151 = 0.3138525e-1 * t12;
    double t152 = 0.1e1 + 0.22225e-1 * t149 + t151;
    double t153 = t152 * t152;
    double t154 = 0.1e1 / t153;
    double t158 = 0.1e1 - 0.2363e1 * t58 * t56 * t60;
    double t159 = t154 * t158;
    double t160 = t37 * t37;
    double t161 = t160 * t37;
    double t162 = t39 * t39;
    double t163 = t162 * t39;
    double t164 = 0.1e1 / t163;
    double t166 = -t161 * t164 + 0.1e1;
    double t167 = 0.1e1 / t15;
    double t168 = t148 * t167;
    double t170 = 0.4445e-1 * t168 + 0.125541e0;
    double t175 = 0.1898925e1 * t149 + t17 + t19 + t27;
    double t178 = 0.1e1 + 0.16081979498692535067e2 / t175;
    double t181 = t175 * t175;
    double t182 = 0.1e1 / t181;
    double t185 = std::sqrt(t12);
    double t188 = 0.379785e1 * t168 + 0.35876e1 + 0.122865e1 * t185 + 0.24647e0 * t12;
    double t189 = 0.1e1 / t178;
    double t190 = t188 * t189;
    double t212 = 0.258925e1 * t149 + t78 + t79 + t80;
    double t215 = 0.1e1 + 0.29608749977793437516e2 / t212;
    double t216 = std::log(t215);
    double t218 = t212 * t212;
    double t219 = 0.1e1 / t218;
    double t224 = 0.51785e1 * t168 + 0.36231e1 + 0.660195e0 * t185 + 0.248355e0 * t12;
    double t225 = 0.1e1 / t215;
    double t234 = t61 * t76;
    double t236 = t219 * t224 * t225;
    double t377 = 0.1e1 / t9 / t8;
    double t378 = t7 * t377;
    double t597 = t60 * t166;
    double t598 = t597 * t170;
    double t625 = t5 * t7;
    double t626 = t377 * t182;
    double t629 = 0.17808333333333333333e-1 * t625 * t626 * t190;
    double t697 = t219 * t695 * t225;
    double t703 = t218 * t212;
    double t704 = 0.1e1 / t703;
    double t716 = t218 * t218;
    double t717 = 0.1e1 / t716;
    double t719 = t215 * t215;
    double t720 = 0.1e1 / t719;
    double t731 = 0.65061487801810439052e-1 * t61 * t697;
    double t732 = t424 * t76;
    double t735 = t61 * t5;
    double t736 = t378 * t236;
    double t738 = 0.54217906501508699211e-2 * t735 * t736;
    double t739 = t704 * t224;
    double t740 = t225 * t695;
    double t741 = t739 * t740;
    double t743 = 0.11696447245269292414e1 * t234 * t741;
    double t748 = t717 * t224;
    double t749 = t720 * t695;
    double t750 = t748 * t749;
    double t752 = 0.17315859105681463759e2 * t234 * t750;
    double t753 = -t594 - 0.675260332e-1 * t596 * t598 + 0.285764e-1 * t159 * t608 + t619 - t624 + t629 + t635 - t647 - t656 - t658 + t660 - t41 * t661 - t727 - 0.21973736767207854065e-2 * t424 * t216 + t731 + 0.5848223622634646207e0 * t732 * t236 - t738 - t743 + t747 + t752;
    return t753;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho0__t771(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t404 = mgga_c_r2scan_vrho0__t404(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t408 = mgga_c_r2scan_vrho0__t408(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t424 = mgga_c_r2scan_vrho0__t424(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t459 = mgga_c_r2scan_vrho0__t459(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t607 = mgga_c_r2scan_vrho0__t607(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t14 = 0.1e1 + 0.53425e-1 * t12;
    double t15 = std::sqrt(t12);
    double t17 = 0.8969e0 * t12;
    double t18 = t12 * std::sqrt(t12);
    double t19 = 0.204775e0 * t18;
    double t20 = t2 * t2;
    double t21 = t4 * t4;
    double t22 = t20 * t21;
    double t23 = t9 * t9;
    double t26 = t22 * t6 / t23;
    double t27 = 0.123235e0 * t26;
    double t28 = 0.379785e1 * t15 + t17 + t19 + t27;
    double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
    double t32 = std::log(t31);
    double t34 = 0.621814e-1 * t14 * t32;
    double t35 = rho0 - rho1;
    double t36 = t35 * t35;
    double t37 = t36 * t36;
    double t38 = t8 * t8;
    double t39 = t38 * t38;
    double t40 = 0.1e1 / t39;
    double t41 = t37 * t40;
    double t42 = 0.1e1 / t8;
    double t43 = t35 * t42;
    double t44 = 0.1e1 + t43;
    double t45 = t44 <= ZETA_THRESHOLD_C_R2SCAN;
    double t46 = std::cbrt(ZETA_THRESHOLD_C_R2SCAN);
    double t47 = t46 * ZETA_THRESHOLD_C_R2SCAN;
    double t48 = std::cbrt(t44);
    double t49 = t48 * t44;
    double t50 = my_piecewise3(t45, t47, t49);
    double t51 = 0.1e1 - t43;
    double t52 = t51 <= ZETA_THRESHOLD_C_R2SCAN;
    double t53 = std::cbrt(t51);
    double t54 = t53 * t51;
    double t55 = my_piecewise3(t52, t47, t54);
    double t56 = t50 + t55 - 0.2e1;
    double t57 = M_CBRT2;
    double t58 = t57 - 0.1e1;
    double t60 = 0.1e1 / t58 / 0.2e1;
    double t61 = t56 * t60;
    double t63 = 0.1e1 + 0.5137e-1 * t12;
    double t65 = 0.1549425e1 * t12;
    double t66 = 0.420775e0 * t18;
    double t67 = 0.1562925e0 * t26;
    double t68 = 0.705945e1 * t15 + t65 + t66 + t67;
    double t71 = 0.1e1 + 0.32163958997385070134e2 / t68;
    double t72 = std::log(t71);
    double t76 = 0.1e1 + 0.278125e-1 * t12;
    double t78 = 0.905775e0 * t12;
    double t79 = 0.1100325e0 * t18;
    double t80 = 0.1241775e0 * t26;
    double t81 = 0.51785e1 * t15 + t78 + t79 + t80;
    double t84 = 0.1e1 + 0.29608749977793437516e2 / t81;
    double t85 = std::log(t84);
    double t86 = t76 * t85;
    double t88 = -0.310907e-1 * t63 * t72 + t34 - 0.19751673498613801407e-1 * t86;
    double t89 = t61 * t88;
    double t151 = 0.3138525e-1 * t12;
    double t158 = 0.1e1 - 0.2363e1 * t58 * t56 * t60;
    double t160 = t37 * t37;
    double t161 = t160 * t37;
    double t162 = t39 * t39;
    double t163 = t162 * t39;
    double t164 = 0.1e1 / t163;
    double t166 = -t161 * t164 + 0.1e1;
    double t167 = 0.1e1 / t15;
    double t185 = std::sqrt(t12);
    double t234 = t61 * t76;
    double t244 = 0.1e1 + 0.4445e-1 * t15 + t151;
    double t245 = 0.1e1 / t244;
    double t246 = t245 * t158;
    double t271 = t39 * t8;
    double t377 = 0.1e1 / t9 / t8;
    double t378 = t7 * t377;
    double t380 = t5 * t378 * t32;
    double t381 = 0.11073470983333333333e-2 * t380;
    double t385 = t167 * t2;
    double t386 = t4 * t7;
    double t387 = t386 * t377;
    double t388 = t385 * t387;
    double t390 = t5 * t378;
    double t392 = t185 * t2;
    double t393 = t392 * t387;
    double t398 = t22 * t6 / t23 / t8;
    double t409 = 0.1e1 / t271;
    double t410 = t37 * t409;
    double t411 = t410 * t89;
    double t412 = 0.4e1 * t411;
    double t425 = t424 * t88;
    double t426 = t41 * t425;
    double t445 = t81 * t81;
    double t446 = 0.1e1 / t445;
    double t449 = 0.301925e0 * t390;
    double t450 = 0.5501625e-1 * t393;
    double t451 = 0.82785e-1 * t398;
    double t452 = -0.86308333333333333334e0 * t388 - t449 - t450 - t451;
    double t453 = 0.1e1 / t84;
    double t460 = t424 * t86;
    double t461 = 0.19751673498613801407e-1 * t460;
    double t462 = t61 * t2;
    double t464 = t386 * t377 * t85;
    double t465 = t462 * t464;
    double t466 = 0.18311447306006545054e-3 * t465;
    double t468 = t446 * t452 * t453;
    double t469 = t234 * t468;
    double t470 = 0.5848223622634646207e0 * t469;
    double t590 = 0.1046175e-1 * t390;
    double t757 = t244 * t244;
    double t758 = 0.1e1 / t757;
    double t759 = t758 * t158;
    double t761 = -0.74083333333333333333e-2 * t388 - t590;
    double t764 = 0.285764e-1 * t759 * t166 * t761;
    double t765 = t245 * t58;
    double t766 = t424 * t166;
    double t771 = t764 + 0.675260332e-1 * t765 * t766 - 0.285764e-1 * t246 * t607 - t381 - t404 - t408 + t412 - t426 - t459 - t461 + t466 + t470;
    return t771;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho0__t774(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t753 = mgga_c_r2scan_vrho0__t753(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t771 = mgga_c_r2scan_vrho0__t771(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t14 = 0.1e1 + 0.53425e-1 * t12;
    double t15 = std::sqrt(t12);
    double t17 = 0.8969e0 * t12;
    double t18 = t12 * std::sqrt(t12);
    double t19 = 0.204775e0 * t18;
    double t20 = t2 * t2;
    double t21 = t4 * t4;
    double t22 = t20 * t21;
    double t23 = t9 * t9;
    double t26 = t22 * t6 / t23;
    double t27 = 0.123235e0 * t26;
    double t35 = rho0 - rho1;
    double t36 = t35 * t35;
    double t37 = t36 * t36;
    double t38 = t8 * t8;
    double t39 = t38 * t38;
    double t40 = 0.1e1 / t39;
    double t41 = t37 * t40;
    double t42 = 0.1e1 / t8;
    double t43 = t35 * t42;
    double t44 = 0.1e1 + t43;
    double t45 = t44 <= ZETA_THRESHOLD_C_R2SCAN;
    double t46 = std::cbrt(ZETA_THRESHOLD_C_R2SCAN);
    double t47 = t46 * ZETA_THRESHOLD_C_R2SCAN;
    double t48 = std::cbrt(t44);
    double t49 = t48 * t44;
    double t50 = my_piecewise3(t45, t47, t49);
    double t51 = 0.1e1 - t43;
    double t52 = t51 <= ZETA_THRESHOLD_C_R2SCAN;
    double t53 = std::cbrt(t51);
    double t54 = t53 * t51;
    double t55 = my_piecewise3(t52, t47, t54);
    double t56 = t50 + t55 - 0.2e1;
    double t57 = M_CBRT2;
    double t58 = t57 - 0.1e1;
    double t60 = 0.1e1 / t58 / 0.2e1;
    double t61 = t56 * t60;
    double t63 = 0.1e1 + 0.5137e-1 * t12;
    double t65 = 0.1549425e1 * t12;
    double t66 = 0.420775e0 * t18;
    double t67 = 0.1562925e0 * t26;
    double t76 = 0.1e1 + 0.278125e-1 * t12;
    double t78 = 0.905775e0 * t12;
    double t79 = 0.1100325e0 * t18;
    double t80 = 0.1241775e0 * t26;
    double t148 = std::sqrt(0.4e1);
    double t149 = t148 * t15;
    double t151 = 0.3138525e-1 * t12;
    double t152 = 0.1e1 + 0.22225e-1 * t149 + t151;
    double t153 = t152 * t152;
    double t154 = 0.1e1 / t153;
    double t158 = 0.1e1 - 0.2363e1 * t58 * t56 * t60;
    double t159 = t154 * t158;
    double t160 = t37 * t37;
    double t161 = t160 * t37;
    double t162 = t39 * t39;
    double t163 = t162 * t39;
    double t164 = 0.1e1 / t163;
    double t166 = -t161 * t164 + 0.1e1;
    double t167 = 0.1e1 / t15;
    double t168 = t148 * t167;
    double t170 = 0.4445e-1 * t168 + 0.125541e0;
    double t171 = t166 * t170;
    double t175 = 0.1898925e1 * t149 + t17 + t19 + t27;
    double t178 = 0.1e1 + 0.16081979498692535067e2 / t175;
    double t179 = std::log(t178);
    double t180 = 0.1328816518e-1 * t179;
    double t181 = t175 * t175;
    double t182 = 0.1e1 / t181;
    double t183 = t14 * t182;
    double t185 = std::sqrt(t12);
    double t188 = 0.379785e1 * t168 + 0.35876e1 + 0.122865e1 * t185 + 0.24647e0 * t12;
    double t189 = 0.1e1 / t178;
    double t190 = t188 * t189;
    double t192 = 0.1e1 * t183 * t190;
    double t194 = 0.3529725e1 * t149 + t65 + t66 + t67;
    double t197 = 0.1e1 + 0.32163958997385070134e2 / t194;
    double t198 = std::log(t197);
    double t200 = t194 * t194;
    double t201 = 0.1e1 / t200;
    double t202 = t63 * t201;
    double t206 = 0.705945e1 * t168 + 0.61977e1 + 0.252465e1 * t185 + 0.312585e0 * t12;
    double t207 = 0.1e1 / t197;
    double t208 = t206 * t207;
    double t212 = 0.258925e1 * t149 + t78 + t79 + t80;
    double t215 = 0.1e1 + 0.29608749977793437516e2 / t212;
    double t216 = std::log(t215);
    double t218 = t212 * t212;
    double t219 = 0.1e1 / t218;
    double t220 = t76 * t219;
    double t224 = 0.51785e1 * t168 + 0.36231e1 + 0.660195e0 * t185 + 0.248355e0 * t12;
    double t225 = 0.1e1 / t215;
    double t226 = t224 * t225;
    double t229 = -0.6388517036e-2 * t198 + 0.1e1 * t202 * t208 + t180 - t192 - 0.21973736767207854065e-2 * t216 + 0.5848223622634646207e0 * t220 * t226;
    double t230 = t61 * t229;
    double t234 = t61 * t76;
    double t236 = t219 * t224 * t225;
    double t239 = 0.285764e-1 * t159 * t171 + t180 - t192 - t41 * t230 - 0.21973736767207854065e-2 * t61 * t216 + 0.5848223622634646207e0 * t234 * t236;
    double t377 = 0.1e1 / t9 / t8;
    double t378 = t7 * t377;
    double t581 = 0.5e1 / 0.3e1 * t5 * t378 * t239;
    double t774 = 0.5e1 * t5 * t11 * t753 - 0.45e2 * 0.001 * t771 - t581;
    return t774;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho0__t775(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t774 = mgga_c_r2scan_vrho0__t774(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t14 = 0.1e1 + 0.53425e-1 * t12;
    double t15 = std::sqrt(t12);
    double t17 = 0.8969e0 * t12;
    double t18 = t12 * std::sqrt(t12);
    double t19 = 0.204775e0 * t18;
    double t20 = t2 * t2;
    double t21 = t4 * t4;
    double t22 = t20 * t21;
    double t23 = t9 * t9;
    double t26 = t22 * t6 / t23;
    double t27 = 0.123235e0 * t26;
    double t28 = 0.379785e1 * t15 + t17 + t19 + t27;
    double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
    double t32 = std::log(t31);
    double t34 = 0.621814e-1 * t14 * t32;
    double t35 = rho0 - rho1;
    double t36 = t35 * t35;
    double t37 = t36 * t36;
    double t38 = t8 * t8;
    double t39 = t38 * t38;
    double t40 = 0.1e1 / t39;
    double t41 = t37 * t40;
    double t42 = 0.1e1 / t8;
    double t43 = t35 * t42;
    double t44 = 0.1e1 + t43;
    double t45 = t44 <= ZETA_THRESHOLD_C_R2SCAN;
    double t46 = std::cbrt(ZETA_THRESHOLD_C_R2SCAN);
    double t47 = t46 * ZETA_THRESHOLD_C_R2SCAN;
    double t48 = std::cbrt(t44);
    double t49 = t48 * t44;
    double t50 = my_piecewise3(t45, t47, t49);
    double t51 = 0.1e1 - t43;
    double t52 = t51 <= ZETA_THRESHOLD_C_R2SCAN;
    double t53 = std::cbrt(t51);
    double t54 = t53 * t51;
    double t55 = my_piecewise3(t52, t47, t54);
    double t56 = t50 + t55 - 0.2e1;
    double t57 = M_CBRT2;
    double t58 = t57 - 0.1e1;
    double t60 = 0.1e1 / t58 / 0.2e1;
    double t61 = t56 * t60;
    double t63 = 0.1e1 + 0.5137e-1 * t12;
    double t65 = 0.1549425e1 * t12;
    double t66 = 0.420775e0 * t18;
    double t67 = 0.1562925e0 * t26;
    double t68 = 0.705945e1 * t15 + t65 + t66 + t67;
    double t71 = 0.1e1 + 0.32163958997385070134e2 / t68;
    double t72 = std::log(t71);
    double t76 = 0.1e1 + 0.278125e-1 * t12;
    double t78 = 0.905775e0 * t12;
    double t79 = 0.1100325e0 * t18;
    double t80 = 0.1241775e0 * t26;
    double t81 = 0.51785e1 * t15 + t78 + t79 + t80;
    double t84 = 0.1e1 + 0.29608749977793437516e2 / t81;
    double t85 = std::log(t84);
    double t86 = t76 * t85;
    double t88 = -0.310907e-1 * t63 * t72 + t34 - 0.19751673498613801407e-1 * t86;
    double t89 = t61 * t88;
    double t90 = t41 * t89;
    double t92 = 0.19751673498613801407e-1 * t61 * t86;
    double t93 = std::log(0.2e1);
    double t94 = 0.1e1 - t93;
    double t95 = M_PI * M_PI;
    double t98 = t46 * t46;
    double t99 = t48 * t48;
    double t100 = my_piecewise3(t45, t98, t99);
    double t101 = t53 * t53;
    double t102 = my_piecewise3(t52, t98, t101);
    double t104 = t100 / 0.2e1 + t102 / 0.2e1;
    double t105 = t104 * t104;
    double t106 = t105 * t104;
    double t108 = 0.1e1 / t94;
    double t109 = (-t34 + t90 + t92) * t108;
    double t110 = 0.1e1 / t106;
    double t111 = t95 * t110;
    double t113 = std::exp(-t109 * t111);
    double t114 = t113 - 0.1e1;
    double t133 = 0.1e1 / t114;
    double t147 = t110 * t133;
    double t775 = t147 * t774;
    return t775;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho0__t776(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t775 = mgga_c_r2scan_vrho0__t775(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t8 = rho0 + rho1;
    double t35 = rho0 - rho1;
    double t42 = 0.1e1 / t8;
    double t43 = t35 * t42;
    double t44 = 0.1e1 + t43;
    double t45 = t44 <= ZETA_THRESHOLD_C_R2SCAN;
    double t46 = std::cbrt(ZETA_THRESHOLD_C_R2SCAN);
    double t48 = std::cbrt(t44);
    double t51 = 0.1e1 - t43;
    double t52 = t51 <= ZETA_THRESHOLD_C_R2SCAN;
    double t53 = std::cbrt(t51);
    double t93 = std::log(0.2e1);
    double t94 = 0.1e1 - t93;
    double t98 = t46 * t46;
    double t99 = t48 * t48;
    double t101 = t53 * t53;
    double t108 = 0.1e1 / t94;
    double t138 = t98 * ZETA_THRESHOLD_C_R2SCAN;
    double t139 = t99 * t44;
    double t140 = my_piecewise3(t45, t138, t139);
    double t141 = t101 * t51;
    double t142 = my_piecewise3(t52, t138, t141);
    double t144 = t140 / 0.2e1 + t142 / 0.2e1;
    double t146 = t108 / t144;
    double t776 = t146 * t775;
    return t776;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho0__t791(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t252 = mgga_c_r2scan_vrho0__t252(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t14 = 0.1e1 + 0.53425e-1 * t12;
    double t15 = std::sqrt(t12);
    double t17 = 0.8969e0 * t12;
    double t18 = t12 * std::sqrt(t12);
    double t19 = 0.204775e0 * t18;
    double t20 = t2 * t2;
    double t21 = t4 * t4;
    double t22 = t20 * t21;
    double t23 = t9 * t9;
    double t26 = t22 * t6 / t23;
    double t27 = 0.123235e0 * t26;
    double t28 = 0.379785e1 * t15 + t17 + t19 + t27;
    double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
    double t32 = std::log(t31);
    double t34 = 0.621814e-1 * t14 * t32;
    double t35 = rho0 - rho1;
    double t36 = t35 * t35;
    double t37 = t36 * t36;
    double t38 = t8 * t8;
    double t39 = t38 * t38;
    double t40 = 0.1e1 / t39;
    double t41 = t37 * t40;
    double t42 = 0.1e1 / t8;
    double t43 = t35 * t42;
    double t44 = 0.1e1 + t43;
    double t45 = t44 <= ZETA_THRESHOLD_C_R2SCAN;
    double t46 = std::cbrt(ZETA_THRESHOLD_C_R2SCAN);
    double t47 = t46 * ZETA_THRESHOLD_C_R2SCAN;
    double t48 = std::cbrt(t44);
    double t49 = t48 * t44;
    double t50 = my_piecewise3(t45, t47, t49);
    double t51 = 0.1e1 - t43;
    double t52 = t51 <= ZETA_THRESHOLD_C_R2SCAN;
    double t53 = std::cbrt(t51);
    double t54 = t53 * t51;
    double t55 = my_piecewise3(t52, t47, t54);
    double t56 = t50 + t55 - 0.2e1;
    double t57 = M_CBRT2;
    double t58 = t57 - 0.1e1;
    double t60 = 0.1e1 / t58 / 0.2e1;
    double t61 = t56 * t60;
    double t63 = 0.1e1 + 0.5137e-1 * t12;
    double t65 = 0.1549425e1 * t12;
    double t66 = 0.420775e0 * t18;
    double t67 = 0.1562925e0 * t26;
    double t68 = 0.705945e1 * t15 + t65 + t66 + t67;
    double t71 = 0.1e1 + 0.32163958997385070134e2 / t68;
    double t72 = std::log(t71);
    double t76 = 0.1e1 + 0.278125e-1 * t12;
    double t78 = 0.905775e0 * t12;
    double t79 = 0.1100325e0 * t18;
    double t80 = 0.1241775e0 * t26;
    double t81 = 0.51785e1 * t15 + t78 + t79 + t80;
    double t84 = 0.1e1 + 0.29608749977793437516e2 / t81;
    double t85 = std::log(t84);
    double t86 = t76 * t85;
    double t88 = -0.310907e-1 * t63 * t72 + t34 - 0.19751673498613801407e-1 * t86;
    double t89 = t61 * t88;
    double t90 = t41 * t89;
    double t92 = 0.19751673498613801407e-1 * t61 * t86;
    double t93 = std::log(0.2e1);
    double t94 = 0.1e1 - t93;
    double t95 = M_PI * M_PI;
    double t98 = t46 * t46;
    double t99 = t48 * t48;
    double t100 = my_piecewise3(t45, t98, t99);
    double t101 = t53 * t53;
    double t102 = my_piecewise3(t52, t98, t101);
    double t104 = t100 / 0.2e1 + t102 / 0.2e1;
    double t105 = t104 * t104;
    double t106 = t105 * t104;
    double t108 = 0.1e1 / t94;
    double t109 = (-t34 + t90 + t92) * t108;
    double t110 = 0.1e1 / t106;
    double t111 = t95 * t110;
    double t113 = std::exp(-t109 * t111);
    double t114 = t113 - 0.1e1;
    double t122 = sigma0 + 0.2e1 * sigma1 + sigma2;
    double t133 = 0.1e1 / t114;
    double t138 = t98 * ZETA_THRESHOLD_C_R2SCAN;
    double t139 = t99 * t44;
    double t140 = my_piecewise3(t45, t138, t139);
    double t141 = t101 * t51;
    double t142 = my_piecewise3(t52, t138, t141);
    double t144 = t140 / 0.2e1 + t142 / 0.2e1;
    double t146 = t108 / t144;
    double t147 = t110 * t133;
    double t162 = t39 * t39;
    double t255 = M_CBRT6;
    double t256 = std::cbrt(t95);
    double t265 = t255 * t255;
    double t267 = 0.1e1 / t256 / t95;
    double t268 = t265 * t267;
    double t269 = t122 * t122;
    double t270 = t57 * t269;
    double t271 = t39 * t8;
    double t273 = 0.1e1 / t9 / t271;
    double t277 = std::exp(-0.20444604078896369094e0 * t268 * t270 * t273);
    double t783 = t146 * t147;
    double t784 = t269 * t122;
    double t785 = t252 * t784;
    double t786 = t162 * t8;
    double t787 = 0.1e1 / t786;
    double t788 = t787 * t277;
    double t791 = 0.58218257753910989057e-2 * t783 * t785 * t788;
    return t791;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho0__t792(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t252 = mgga_c_r2scan_vrho0__t252(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t481 = mgga_c_r2scan_vrho0__t481(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t495 = mgga_c_r2scan_vrho0__t495(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t516 = mgga_c_r2scan_vrho0__t516(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t562 = mgga_c_r2scan_vrho0__t562(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t776 = mgga_c_r2scan_vrho0__t776(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t791 = mgga_c_r2scan_vrho0__t791(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t14 = 0.1e1 + 0.53425e-1 * t12;
    double t15 = std::sqrt(t12);
    double t17 = 0.8969e0 * t12;
    double t18 = t12 * std::sqrt(t12);
    double t19 = 0.204775e0 * t18;
    double t20 = t2 * t2;
    double t21 = t4 * t4;
    double t22 = t20 * t21;
    double t23 = t9 * t9;
    double t26 = t22 * t6 / t23;
    double t27 = 0.123235e0 * t26;
    double t28 = 0.379785e1 * t15 + t17 + t19 + t27;
    double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
    double t32 = std::log(t31);
    double t34 = 0.621814e-1 * t14 * t32;
    double t35 = rho0 - rho1;
    double t36 = t35 * t35;
    double t37 = t36 * t36;
    double t38 = t8 * t8;
    double t39 = t38 * t38;
    double t40 = 0.1e1 / t39;
    double t41 = t37 * t40;
    double t42 = 0.1e1 / t8;
    double t43 = t35 * t42;
    double t44 = 0.1e1 + t43;
    double t45 = t44 <= ZETA_THRESHOLD_C_R2SCAN;
    double t46 = std::cbrt(ZETA_THRESHOLD_C_R2SCAN);
    double t47 = t46 * ZETA_THRESHOLD_C_R2SCAN;
    double t48 = std::cbrt(t44);
    double t49 = t48 * t44;
    double t50 = my_piecewise3(t45, t47, t49);
    double t51 = 0.1e1 - t43;
    double t52 = t51 <= ZETA_THRESHOLD_C_R2SCAN;
    double t53 = std::cbrt(t51);
    double t54 = t53 * t51;
    double t55 = my_piecewise3(t52, t47, t54);
    double t56 = t50 + t55 - 0.2e1;
    double t57 = M_CBRT2;
    double t58 = t57 - 0.1e1;
    double t60 = 0.1e1 / t58 / 0.2e1;
    double t61 = t56 * t60;
    double t63 = 0.1e1 + 0.5137e-1 * t12;
    double t65 = 0.1549425e1 * t12;
    double t66 = 0.420775e0 * t18;
    double t67 = 0.1562925e0 * t26;
    double t68 = 0.705945e1 * t15 + t65 + t66 + t67;
    double t71 = 0.1e1 + 0.32163958997385070134e2 / t68;
    double t72 = std::log(t71);
    double t76 = 0.1e1 + 0.278125e-1 * t12;
    double t78 = 0.905775e0 * t12;
    double t79 = 0.1100325e0 * t18;
    double t80 = 0.1241775e0 * t26;
    double t81 = 0.51785e1 * t15 + t78 + t79 + t80;
    double t84 = 0.1e1 + 0.29608749977793437516e2 / t81;
    double t85 = std::log(t84);
    double t86 = t76 * t85;
    double t88 = -0.310907e-1 * t63 * t72 + t34 - 0.19751673498613801407e-1 * t86;
    double t89 = t61 * t88;
    double t90 = t41 * t89;
    double t92 = 0.19751673498613801407e-1 * t61 * t86;
    double t93 = std::log(0.2e1);
    double t94 = 0.1e1 - t93;
    double t95 = M_PI * M_PI;
    double t98 = t46 * t46;
    double t99 = t48 * t48;
    double t100 = my_piecewise3(t45, t98, t99);
    double t101 = t53 * t53;
    double t102 = my_piecewise3(t52, t98, t101);
    double t104 = t100 / 0.2e1 + t102 / 0.2e1;
    double t105 = t104 * t104;
    double t106 = t105 * t104;
    double t108 = 0.1e1 / t94;
    double t109 = (-t34 + t90 + t92) * t108;
    double t110 = 0.1e1 / t106;
    double t111 = t95 * t110;
    double t113 = std::exp(-t109 * t111);
    double t114 = t113 - 0.1e1;
    double t116 = 0.1e1 + 0.25e-1 * t12;
    double t118 = 0.1e1 + 0.4445e-1 * t12;
    double t119 = 0.1e1 / t118;
    double t120 = t116 * t119;
    double t122 = sigma0 + 0.2e1 * sigma1 + sigma2;
    double t124 = 0.1e1 / t9 / t38;
    double t128 = 0.1e1 / t105;
    double t129 = t128 * t20;
    double t130 = 0.1e1 / t4;
    double t132 = t6 * t108;
    double t133 = 0.1e1 / t114;
    double t134 = t132 * t133;
    double t135 = t129 * t130 * t134;
    double t138 = t98 * ZETA_THRESHOLD_C_R2SCAN;
    double t139 = t99 * t44;
    double t140 = my_piecewise3(t45, t138, t139);
    double t141 = t101 * t51;
    double t142 = my_piecewise3(t52, t138, t141);
    double t144 = t140 / 0.2e1 + t142 / 0.2e1;
    double t146 = t108 / t144;
    double t147 = t110 * t133;
    double t253 = t147 * t252;
    double t254 = t146 * t253;
    double t255 = M_CBRT6;
    double t256 = std::cbrt(t95);
    double t257 = t256 * t256;
    double t258 = 0.1e1 / t257;
    double t259 = t255 * t258;
    double t260 = t57 * t57;
    double t261 = t259 * t260;
    double t263 = 0.1e1 / t23 / t38;
    double t264 = t122 * t263;
    double t265 = t255 * t255;
    double t267 = 0.1e1 / t256 / t95;
    double t268 = t265 * t267;
    double t269 = t122 * t122;
    double t270 = t57 * t269;
    double t271 = t39 * t8;
    double t273 = 0.1e1 / t9 / t271;
    double t277 = std::exp(-0.20444604078896369094e0 * t268 * t270 * t273);
    double t279 = t261 * t264 * t277;
    double t488 = t105 * t105;
    double t489 = 0.1e1 / t488;
    double t500 = t38 * t8;
    double t502 = 0.1e1 / t23 / t500;
    double t503 = t502 * t119;
    double t505 = t57 * t128;
    double t506 = t108 * t133;
    double t507 = t505 * t506;
    double t509 = 0.27439371595564631661e-2 * t503 * t122 * t507;
    double t513 = t122 * t502;
    double t518 = 0.1e1 / t9 / t500;
    double t523 = 0.64025200389650807209e-1 * t120 * t122 * t518 * t57 * t135;
    double t524 = t120 * t122;
    double t525 = t124 * t57;
    double t526 = t525 * t110;
    double t527 = t524 * t526;
    double t528 = t20 * t130;
    double t529 = t528 * t6;
    double t530 = t506 * t481;
    double t531 = t529 * t530;
    double t534 = t525 * t128;
    double t535 = t524 * t534;
    double t536 = t114 * t114;
    double t537 = 0.1e1 / t536;
    double t538 = t108 * t537;
    double t539 = t538 * t495;
    double t540 = t529 * t539;
    double t543 = t144 * t144;
    double t545 = t108 / t543;
    double t546 = t545 * t110;
    double t547 = t133 * t252;
    double t548 = t547 * t255;
    double t549 = t546 * t548;
    double t550 = t258 * t260;
    double t551 = t550 * t122;
    double t552 = t263 * t277;
    double t565 = t146 * t489;
    double t566 = t565 * t548;
    double t567 = t552 * t481;
    double t568 = t551 * t567;
    double t571 = t146 * t110;
    double t572 = t537 * t252;
    double t573 = t572 * t255;
    double t574 = t571 * t573;
    double t576 = t551 * t552 * t495;
    double t780 = t261 * t513 * t277;
    double t782 = 0.11557628986739024751e0 * t254 * t780;
    double t792 = -t509 + t516 - t523 - 0.54878743191129263322e-1 * t527 * t531 - 0.27439371595564631661e-1 * t535 * t540 - 0.43341108700271342816e-1 * t549 * t562 - 0.13002332610081402845e0 * t566 * t568 - 0.43341108700271342816e-1 * t574 * t576 + 0.43341108700271342816e-1 * t776 * t279 - t782 + t791;
    return t792;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho0__t795(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t252 = mgga_c_r2scan_vrho0__t252(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t495 = mgga_c_r2scan_vrho0__t495(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t792 = mgga_c_r2scan_vrho0__t792(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t14 = 0.1e1 + 0.53425e-1 * t12;
    double t15 = std::sqrt(t12);
    double t17 = 0.8969e0 * t12;
    double t18 = t12 * std::sqrt(t12);
    double t19 = 0.204775e0 * t18;
    double t20 = t2 * t2;
    double t21 = t4 * t4;
    double t22 = t20 * t21;
    double t23 = t9 * t9;
    double t26 = t22 * t6 / t23;
    double t27 = 0.123235e0 * t26;
    double t28 = 0.379785e1 * t15 + t17 + t19 + t27;
    double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
    double t32 = std::log(t31);
    double t34 = 0.621814e-1 * t14 * t32;
    double t35 = rho0 - rho1;
    double t36 = t35 * t35;
    double t37 = t36 * t36;
    double t38 = t8 * t8;
    double t39 = t38 * t38;
    double t40 = 0.1e1 / t39;
    double t41 = t37 * t40;
    double t42 = 0.1e1 / t8;
    double t43 = t35 * t42;
    double t44 = 0.1e1 + t43;
    double t45 = t44 <= ZETA_THRESHOLD_C_R2SCAN;
    double t46 = std::cbrt(ZETA_THRESHOLD_C_R2SCAN);
    double t47 = t46 * ZETA_THRESHOLD_C_R2SCAN;
    double t48 = std::cbrt(t44);
    double t49 = t48 * t44;
    double t50 = my_piecewise3(t45, t47, t49);
    double t51 = 0.1e1 - t43;
    double t52 = t51 <= ZETA_THRESHOLD_C_R2SCAN;
    double t53 = std::cbrt(t51);
    double t54 = t53 * t51;
    double t55 = my_piecewise3(t52, t47, t54);
    double t56 = t50 + t55 - 0.2e1;
    double t57 = M_CBRT2;
    double t58 = t57 - 0.1e1;
    double t60 = 0.1e1 / t58 / 0.2e1;
    double t61 = t56 * t60;
    double t63 = 0.1e1 + 0.5137e-1 * t12;
    double t65 = 0.1549425e1 * t12;
    double t66 = 0.420775e0 * t18;
    double t67 = 0.1562925e0 * t26;
    double t68 = 0.705945e1 * t15 + t65 + t66 + t67;
    double t71 = 0.1e1 + 0.32163958997385070134e2 / t68;
    double t72 = std::log(t71);
    double t76 = 0.1e1 + 0.278125e-1 * t12;
    double t78 = 0.905775e0 * t12;
    double t79 = 0.1100325e0 * t18;
    double t80 = 0.1241775e0 * t26;
    double t81 = 0.51785e1 * t15 + t78 + t79 + t80;
    double t84 = 0.1e1 + 0.29608749977793437516e2 / t81;
    double t85 = std::log(t84);
    double t86 = t76 * t85;
    double t88 = -0.310907e-1 * t63 * t72 + t34 - 0.19751673498613801407e-1 * t86;
    double t89 = t61 * t88;
    double t90 = t41 * t89;
    double t92 = 0.19751673498613801407e-1 * t61 * t86;
    double t93 = std::log(0.2e1);
    double t94 = 0.1e1 - t93;
    double t95 = M_PI * M_PI;
    double t98 = t46 * t46;
    double t99 = t48 * t48;
    double t100 = my_piecewise3(t45, t98, t99);
    double t101 = t53 * t53;
    double t102 = my_piecewise3(t52, t98, t101);
    double t104 = t100 / 0.2e1 + t102 / 0.2e1;
    double t105 = t104 * t104;
    double t106 = t105 * t104;
    double t108 = 0.1e1 / t94;
    double t109 = (-t34 + t90 + t92) * t108;
    double t110 = 0.1e1 / t106;
    double t111 = t95 * t110;
    double t113 = std::exp(-t109 * t111);
    double t114 = t113 - 0.1e1;
    double t116 = 0.1e1 + 0.25e-1 * t12;
    double t118 = 0.1e1 + 0.4445e-1 * t12;
    double t119 = 0.1e1 / t118;
    double t120 = t116 * t119;
    double t122 = sigma0 + 0.2e1 * sigma1 + sigma2;
    double t124 = 0.1e1 / t9 / t38;
    double t125 = t122 * t124;
    double t128 = 0.1e1 / t105;
    double t129 = t128 * t20;
    double t130 = 0.1e1 / t4;
    double t132 = t6 * t108;
    double t133 = 0.1e1 / t114;
    double t134 = t132 * t133;
    double t135 = t129 * t130 * t134;
    double t138 = t98 * ZETA_THRESHOLD_C_R2SCAN;
    double t139 = t99 * t44;
    double t140 = my_piecewise3(t45, t138, t139);
    double t141 = t101 * t51;
    double t142 = my_piecewise3(t52, t138, t141);
    double t144 = t140 / 0.2e1 + t142 / 0.2e1;
    double t146 = t108 / t144;
    double t147 = t110 * t133;
    double t253 = t147 * t252;
    double t254 = t146 * t253;
    double t255 = M_CBRT6;
    double t256 = std::cbrt(t95);
    double t257 = t256 * t256;
    double t258 = 0.1e1 / t257;
    double t259 = t255 * t258;
    double t260 = t57 * t57;
    double t261 = t259 * t260;
    double t263 = 0.1e1 / t23 / t38;
    double t264 = t122 * t263;
    double t265 = t255 * t255;
    double t267 = 0.1e1 / t256 / t95;
    double t268 = t265 * t267;
    double t269 = t122 * t122;
    double t270 = t57 * t269;
    double t271 = t39 * t8;
    double t273 = 0.1e1 / t9 / t271;
    double t277 = std::exp(-0.20444604078896369094e0 * t268 * t270 * t273);
    double t279 = t261 * t264 * t277;
    double t282 = 0.1e1 + 0.27439371595564631661e-1 * t120 * t125 * t57 * t135 + 0.43341108700271342816e-1 * t254 * t279;
    double t283 = std::sqrt(std::sqrt(t282));
    double t285 = 0.1e1 - 0.1e1 / t283;
    double t496 = t495 * t285;
    double t498 = 0.1e1 / t283 / t282;
    double t499 = t114 * t498;
    double t795 = t496 + t499 * t792 / 0.4e1;
    return t795;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho0__t799(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t252 = mgga_c_r2scan_vrho0__t252(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t795 = mgga_c_r2scan_vrho0__t795(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t14 = 0.1e1 + 0.53425e-1 * t12;
    double t15 = std::sqrt(t12);
    double t17 = 0.8969e0 * t12;
    double t18 = t12 * std::sqrt(t12);
    double t19 = 0.204775e0 * t18;
    double t20 = t2 * t2;
    double t21 = t4 * t4;
    double t22 = t20 * t21;
    double t23 = t9 * t9;
    double t26 = t22 * t6 / t23;
    double t27 = 0.123235e0 * t26;
    double t28 = 0.379785e1 * t15 + t17 + t19 + t27;
    double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
    double t32 = std::log(t31);
    double t34 = 0.621814e-1 * t14 * t32;
    double t35 = rho0 - rho1;
    double t36 = t35 * t35;
    double t37 = t36 * t36;
    double t38 = t8 * t8;
    double t39 = t38 * t38;
    double t40 = 0.1e1 / t39;
    double t41 = t37 * t40;
    double t42 = 0.1e1 / t8;
    double t43 = t35 * t42;
    double t44 = 0.1e1 + t43;
    double t45 = t44 <= ZETA_THRESHOLD_C_R2SCAN;
    double t46 = std::cbrt(ZETA_THRESHOLD_C_R2SCAN);
    double t47 = t46 * ZETA_THRESHOLD_C_R2SCAN;
    double t48 = std::cbrt(t44);
    double t49 = t48 * t44;
    double t50 = my_piecewise3(t45, t47, t49);
    double t51 = 0.1e1 - t43;
    double t52 = t51 <= ZETA_THRESHOLD_C_R2SCAN;
    double t53 = std::cbrt(t51);
    double t54 = t53 * t51;
    double t55 = my_piecewise3(t52, t47, t54);
    double t56 = t50 + t55 - 0.2e1;
    double t57 = M_CBRT2;
    double t58 = t57 - 0.1e1;
    double t60 = 0.1e1 / t58 / 0.2e1;
    double t61 = t56 * t60;
    double t63 = 0.1e1 + 0.5137e-1 * t12;
    double t65 = 0.1549425e1 * t12;
    double t66 = 0.420775e0 * t18;
    double t67 = 0.1562925e0 * t26;
    double t68 = 0.705945e1 * t15 + t65 + t66 + t67;
    double t71 = 0.1e1 + 0.32163958997385070134e2 / t68;
    double t72 = std::log(t71);
    double t76 = 0.1e1 + 0.278125e-1 * t12;
    double t78 = 0.905775e0 * t12;
    double t79 = 0.1100325e0 * t18;
    double t80 = 0.1241775e0 * t26;
    double t81 = 0.51785e1 * t15 + t78 + t79 + t80;
    double t84 = 0.1e1 + 0.29608749977793437516e2 / t81;
    double t85 = std::log(t84);
    double t86 = t76 * t85;
    double t88 = -0.310907e-1 * t63 * t72 + t34 - 0.19751673498613801407e-1 * t86;
    double t89 = t61 * t88;
    double t90 = t41 * t89;
    double t92 = 0.19751673498613801407e-1 * t61 * t86;
    double t93 = std::log(0.2e1);
    double t94 = 0.1e1 - t93;
    double t95 = M_PI * M_PI;
    double t97 = t94 / t95;
    double t98 = t46 * t46;
    double t99 = t48 * t48;
    double t100 = my_piecewise3(t45, t98, t99);
    double t101 = t53 * t53;
    double t102 = my_piecewise3(t52, t98, t101);
    double t104 = t100 / 0.2e1 + t102 / 0.2e1;
    double t105 = t104 * t104;
    double t106 = t105 * t104;
    double t108 = 0.1e1 / t94;
    double t109 = (-t34 + t90 + t92) * t108;
    double t110 = 0.1e1 / t106;
    double t111 = t95 * t110;
    double t113 = std::exp(-t109 * t111);
    double t114 = t113 - 0.1e1;
    double t116 = 0.1e1 + 0.25e-1 * t12;
    double t118 = 0.1e1 + 0.4445e-1 * t12;
    double t119 = 0.1e1 / t118;
    double t120 = t116 * t119;
    double t122 = sigma0 + 0.2e1 * sigma1 + sigma2;
    double t124 = 0.1e1 / t9 / t38;
    double t125 = t122 * t124;
    double t128 = 0.1e1 / t105;
    double t129 = t128 * t20;
    double t130 = 0.1e1 / t4;
    double t132 = t6 * t108;
    double t133 = 0.1e1 / t114;
    double t134 = t132 * t133;
    double t135 = t129 * t130 * t134;
    double t138 = t98 * ZETA_THRESHOLD_C_R2SCAN;
    double t139 = t99 * t44;
    double t140 = my_piecewise3(t45, t138, t139);
    double t141 = t101 * t51;
    double t142 = my_piecewise3(t52, t138, t141);
    double t144 = t140 / 0.2e1 + t142 / 0.2e1;
    double t146 = t108 / t144;
    double t147 = t110 * t133;
    double t253 = t147 * t252;
    double t254 = t146 * t253;
    double t255 = M_CBRT6;
    double t256 = std::cbrt(t95);
    double t257 = t256 * t256;
    double t258 = 0.1e1 / t257;
    double t259 = t255 * t258;
    double t260 = t57 * t57;
    double t261 = t259 * t260;
    double t263 = 0.1e1 / t23 / t38;
    double t264 = t122 * t263;
    double t265 = t255 * t255;
    double t267 = 0.1e1 / t256 / t95;
    double t268 = t265 * t267;
    double t269 = t122 * t122;
    double t270 = t57 * t269;
    double t271 = t39 * t8;
    double t273 = 0.1e1 / t9 / t271;
    double t277 = std::exp(-0.20444604078896369094e0 * t268 * t270 * t273);
    double t279 = t261 * t264 * t277;
    double t282 = 0.1e1 + 0.27439371595564631661e-1 * t120 * t125 * t57 * t135 + 0.43341108700271342816e-1 * t254 * t279;
    double t283 = std::sqrt(std::sqrt(t282));
    double t285 = 0.1e1 - 0.1e1 / t283;
    double t287 = t114 * t285 + 0.1e1;
    double t797 = 0.1e1 / t287;
    double t799 = t97 * t106 * t795 * t797;
    return t799;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho0__t861(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t374 = mgga_c_r2scan_vrho0__t374(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t23 = t9 * t9;
    double t35 = rho0 - rho1;
    double t38 = t8 * t8;
    double t42 = 0.1e1 / t8;
    double t43 = t35 * t42;
    double t44 = 0.1e1 + t43;
    double t51 = 0.1e1 - t43;
    double t95 = M_PI * M_PI;
    double t122 = sigma0 + 0.2e1 * sigma1 + sigma2;
    double t255 = M_CBRT6;
    double t256 = std::cbrt(t95);
    double t257 = t256 * t256;
    double t263 = 0.1e1 / t23 / t38;
    double t264 = t122 * t263;
    double t265 = t255 * t255;
    double t291 = std::cbrt(rho0);
    double t292 = t291 * t291;
    double t294 = 0.1e1 / t292 / rho0;
    double t295 = tau0 * t294;
    double t296 = t44 / 0.2e1;
    double t297 = std::cbrt(t296);
    double t298 = t297 * t297;
    double t299 = t298 * t296;
    double t301 = std::cbrt(rho1);
    double t302 = t301 * t301;
    double t304 = 0.1e1 / t302 / rho1;
    double t305 = tau1 * t304;
    double t306 = t51 / 0.2e1;
    double t307 = std::cbrt(t306);
    double t308 = t307 * t307;
    double t309 = t308 * t306;
    double t312 = t295 * t299 + t305 * t309 - t264 / 0.8e1;
    double t313 = t265 * t257;
    double t317 = 0.001 * t122;
    double t320 = 0.3e1 / 0.1e2 * t313 * (t299 + t309) + t317 * t263 / 0.8e1;
    double t321 = 0.1e1 / t320;
    double t322 = t312 * t321;
    double t323 = t322 <= 0.e0;
    double t324 = 0.e0 < t322;
    double t325 = my_piecewise3(t324, 0, t322);
    double t326 = 0.1e1 - t325;
    double t327 = 0.1e1 / t326;
    double t330 = std::exp(-0.64e0 * t325 * t327);
    double t331 = t322 <= 0.25e1;
    double t332 = 0.25e1 < t322;
    double t333 = my_piecewise3(t332, 0.25e1, t322);
    double t335 = t333 * t333;
    double t337 = t335 * t333;
    double t339 = t335 * t335;
    double t341 = t339 * t333;
    double t343 = t339 * t335;
    double t348 = my_piecewise3(t332, t322, 0.25e1);
    double t349 = 0.1e1 - t348;
    double t352 = std::exp(0.15e1 / t349);
    double t413 = 0.1e1 / t38;
    double t414 = t35 * t413;
    double t415 = t42 - t414;
    double t500 = t38 * t8;
    double t502 = 0.1e1 / t23 / t500;
    double t513 = t122 * t502;
    double t800 = rho0 * rho0;
    double t802 = 0.1e1 / t292 / t800;
    double t803 = tau0 * t802;
    double t806 = t415 / 0.2e1;
    double t807 = t298 * t806;
    double t810 = -t806;
    double t811 = t308 * t810;
    double t814 = t513 / 0.3e1;
    double t815 = -0.5e1 / 0.3e1 * t803 * t299 + 0.5e1 / 0.3e1 * t295 * t807 + 0.5e1 / 0.3e1 * t305 * t811 + t814;
    double t817 = t320 * t320;
    double t818 = 0.1e1 / t817;
    double t819 = t312 * t818;
    double t825 = t317 * t502 / 0.3e1;
    double t826 = 0.3e1 / 0.1e2 * t313 * (0.5e1 / 0.3e1 * t807 + 0.5e1 / 0.3e1 * t811) - t825;
    double t828 = t815 * t321 - t819 * t826;
    double t829 = my_piecewise3(t324, 0, t828);
    double t832 = t326 * t326;
    double t833 = 0.1e1 / t832;
    double t834 = t325 * t833;
    double t837 = -0.64e0 * t829 * t327 - 0.64e0 * t834 * t829;
    double t838 = t837 * t330;
    double t839 = my_piecewise3(t332, 0, t828);
    double t841 = t333 * t839;
    double t843 = t335 * t839;
    double t845 = t337 * t839;
    double t847 = t339 * t839;
    double t849 = t341 * t839;
    double t854 = t349 * t349;
    double t855 = 0.1e1 / t854;
    double t856 = my_piecewise3(t332, t828, 0);
    double t860 = my_piecewise5(t323, t838, t331, -0.64e0 * t839 - 0.8704e0 * t841 - 0.4607056813647e1 * t843 + 0.122462410087e2 * t845 - 0.957855118103e1 * t847 + 0.3101306810232e1 * t849 - 0.362942158544e0 * t343 * t839, -0.105e1 * t855 * t856 * t352);
    double t861 = t860 * t374;
    return t861;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho0__t880(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t15 = std::sqrt(t12);
    double t23 = t9 * t9;
    double t35 = rho0 - rho1;
    double t36 = t35 * t35;
    double t37 = t36 * t36;
    double t38 = t8 * t8;
    double t39 = t38 * t38;
    double t42 = 0.1e1 / t8;
    double t43 = t35 * t42;
    double t44 = 0.1e1 + t43;
    double t45 = t44 <= ZETA_THRESHOLD_C_R2SCAN;
    double t46 = std::cbrt(ZETA_THRESHOLD_C_R2SCAN);
    double t47 = t46 * ZETA_THRESHOLD_C_R2SCAN;
    double t48 = std::cbrt(t44);
    double t49 = t48 * t44;
    double t50 = my_piecewise3(t45, t47, t49);
    double t51 = 0.1e1 - t43;
    double t52 = t51 <= ZETA_THRESHOLD_C_R2SCAN;
    double t53 = std::cbrt(t51);
    double t54 = t53 * t51;
    double t55 = my_piecewise3(t52, t47, t54);
    double t56 = t50 + t55 - 0.2e1;
    double t57 = M_CBRT2;
    double t58 = t57 - 0.1e1;
    double t60 = 0.1e1 / t58 / 0.2e1;
    double t95 = M_PI * M_PI;
    double t122 = sigma0 + 0.2e1 * sigma1 + sigma2;
    double t151 = 0.3138525e-1 * t12;
    double t158 = 0.1e1 - 0.2363e1 * t58 * t56 * t60;
    double t160 = t37 * t37;
    double t161 = t160 * t37;
    double t162 = t39 * t39;
    double t163 = t162 * t39;
    double t164 = 0.1e1 / t163;
    double t166 = -t161 * t164 + 0.1e1;
    double t167 = 0.1e1 / t15;
    double t244 = 0.1e1 + 0.4445e-1 * t15 + t151;
    double t245 = 0.1e1 / t244;
    double t255 = M_CBRT6;
    double t256 = std::cbrt(t95);
    double t257 = t256 * t256;
    double t258 = 0.1e1 / t257;
    double t259 = t255 * t258;
    double t260 = t57 * t57;
    double t263 = 0.1e1 / t23 / t38;
    double t357 = std::exp(0.1e1 * t245);
    double t358 = t357 - 0.1e1;
    double t359 = t260 * t122;
    double t360 = t359 * t263;
    double t363 = 0.1e1 + 0.21337642104376358333e-1 * t259 * t360;
    double t364 = std::sqrt(std::sqrt(t363));
    double t366 = 0.1e1 - 0.1e1 / t364;
    double t368 = t358 * t366 + 0.1e1;
    double t377 = 0.1e1 / t9 / t8;
    double t378 = t7 * t377;
    double t385 = t167 * t2;
    double t386 = t4 * t7;
    double t387 = t386 * t377;
    double t388 = t385 * t387;
    double t390 = t5 * t378;
    double t500 = t38 * t8;
    double t502 = 0.1e1 / t23 / t500;
    double t513 = t122 * t502;
    double t550 = t258 * t260;
    double t590 = 0.1046175e-1 * t390;
    double t757 = t244 * t244;
    double t758 = 0.1e1 / t757;
    double t761 = -0.74083333333333333333e-2 * t388 - t590;
    double t862 = t758 * t761;
    double t864 = t357 * t366;
    double t868 = 0.1e1 / t364 / t363;
    double t869 = t358 * t868;
    double t870 = t869 * t255;
    double t874 = -0.1e1 * t862 * t864 - 0.14225094736250905555e-1 * t870 * t550 * t513;
    double t875 = 0.1e1 / t368;
    double t878 = 0.285764e-1 * t862 + 0.285764e-1 * t874 * t875;
    double t879 = t878 * t158;
    double t880 = t879 * t166;
    return t880;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho0__t887(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t354 = mgga_c_r2scan_vrho0__t354(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t371 = mgga_c_r2scan_vrho0__t371(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t404 = mgga_c_r2scan_vrho0__t404(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t408 = mgga_c_r2scan_vrho0__t408(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t424 = mgga_c_r2scan_vrho0__t424(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t459 = mgga_c_r2scan_vrho0__t459(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t484 = mgga_c_r2scan_vrho0__t484(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t607 = mgga_c_r2scan_vrho0__t607(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t799 = mgga_c_r2scan_vrho0__t799(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t861 = mgga_c_r2scan_vrho0__t861(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t880 = mgga_c_r2scan_vrho0__t880(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t14 = 0.1e1 + 0.53425e-1 * t12;
    double t15 = std::sqrt(t12);
    double t17 = 0.8969e0 * t12;
    double t18 = t12 * std::sqrt(t12);
    double t19 = 0.204775e0 * t18;
    double t20 = t2 * t2;
    double t21 = t4 * t4;
    double t22 = t20 * t21;
    double t23 = t9 * t9;
    double t26 = t22 * t6 / t23;
    double t27 = 0.123235e0 * t26;
    double t28 = 0.379785e1 * t15 + t17 + t19 + t27;
    double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
    double t32 = std::log(t31);
    double t34 = 0.621814e-1 * t14 * t32;
    double t35 = rho0 - rho1;
    double t36 = t35 * t35;
    double t37 = t36 * t36;
    double t38 = t8 * t8;
    double t39 = t38 * t38;
    double t40 = 0.1e1 / t39;
    double t41 = t37 * t40;
    double t42 = 0.1e1 / t8;
    double t43 = t35 * t42;
    double t44 = 0.1e1 + t43;
    double t45 = t44 <= ZETA_THRESHOLD_C_R2SCAN;
    double t46 = std::cbrt(ZETA_THRESHOLD_C_R2SCAN);
    double t47 = t46 * ZETA_THRESHOLD_C_R2SCAN;
    double t48 = std::cbrt(t44);
    double t49 = t48 * t44;
    double t50 = my_piecewise3(t45, t47, t49);
    double t51 = 0.1e1 - t43;
    double t52 = t51 <= ZETA_THRESHOLD_C_R2SCAN;
    double t53 = std::cbrt(t51);
    double t54 = t53 * t51;
    double t55 = my_piecewise3(t52, t47, t54);
    double t56 = t50 + t55 - 0.2e1;
    double t57 = M_CBRT2;
    double t58 = t57 - 0.1e1;
    double t60 = 0.1e1 / t58 / 0.2e1;
    double t61 = t56 * t60;
    double t63 = 0.1e1 + 0.5137e-1 * t12;
    double t65 = 0.1549425e1 * t12;
    double t66 = 0.420775e0 * t18;
    double t67 = 0.1562925e0 * t26;
    double t68 = 0.705945e1 * t15 + t65 + t66 + t67;
    double t71 = 0.1e1 + 0.32163958997385070134e2 / t68;
    double t72 = std::log(t71);
    double t76 = 0.1e1 + 0.278125e-1 * t12;
    double t78 = 0.905775e0 * t12;
    double t79 = 0.1100325e0 * t18;
    double t80 = 0.1241775e0 * t26;
    double t81 = 0.51785e1 * t15 + t78 + t79 + t80;
    double t84 = 0.1e1 + 0.29608749977793437516e2 / t81;
    double t85 = std::log(t84);
    double t86 = t76 * t85;
    double t88 = -0.310907e-1 * t63 * t72 + t34 - 0.19751673498613801407e-1 * t86;
    double t89 = t61 * t88;
    double t158 = 0.1e1 - 0.2363e1 * t58 * t56 * t60;
    double t160 = t37 * t37;
    double t161 = t160 * t37;
    double t162 = t39 * t39;
    double t163 = t162 * t39;
    double t164 = 0.1e1 / t163;
    double t166 = -t161 * t164 + 0.1e1;
    double t167 = 0.1e1 / t15;
    double t185 = std::sqrt(t12);
    double t234 = t61 * t76;
    double t271 = t39 * t8;
    double t372 = t371 * t158;
    double t377 = 0.1e1 / t9 / t8;
    double t378 = t7 * t377;
    double t380 = t5 * t378 * t32;
    double t381 = 0.11073470983333333333e-2 * t380;
    double t385 = t167 * t2;
    double t386 = t4 * t7;
    double t387 = t386 * t377;
    double t388 = t385 * t387;
    double t390 = t5 * t378;
    double t392 = t185 * t2;
    double t393 = t392 * t387;
    double t398 = t22 * t6 / t23 / t8;
    double t409 = 0.1e1 / t271;
    double t410 = t37 * t409;
    double t411 = t410 * t89;
    double t412 = 0.4e1 * t411;
    double t425 = t424 * t88;
    double t426 = t41 * t425;
    double t445 = t81 * t81;
    double t446 = 0.1e1 / t445;
    double t449 = 0.301925e0 * t390;
    double t450 = 0.5501625e-1 * t393;
    double t451 = 0.82785e-1 * t398;
    double t452 = -0.86308333333333333334e0 * t388 - t449 - t450 - t451;
    double t453 = 0.1e1 / t84;
    double t460 = t424 * t86;
    double t461 = 0.19751673498613801407e-1 * t460;
    double t462 = t61 * t2;
    double t464 = t386 * t377 * t85;
    double t465 = t462 * t464;
    double t466 = 0.18311447306006545054e-3 * t465;
    double t468 = t446 * t452 * t453;
    double t469 = t234 * t468;
    double t470 = 0.5848223622634646207e0 * t469;
    double t766 = t424 * t166;
    double t881 = t371 * t58;
    double t885 = t880 - 0.2363e1 * t881 * t766 + t372 * t607 - t381 - t404 - t408 + t412 - t426 - t459 - t461 + t466 + t470 - t484 - t799;
    double t886 = t354 * t885;
    double t887 = t381 + t404 + t408 - t412 + t426 + t459 + t461 - t466 - t470 + t484 + t799 + t861 + t886;
    return t887;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho0(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t252 = mgga_c_r2scan_vrho0__t252(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t354 = mgga_c_r2scan_vrho0__t354(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t374 = mgga_c_r2scan_vrho0__t374(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t887 = mgga_c_r2scan_vrho0__t887(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t14 = 0.1e1 + 0.53425e-1 * t12;
    double t15 = std::sqrt(t12);
    double t17 = 0.8969e0 * t12;
    double t18 = t12 * std::sqrt(t12);
    double t19 = 0.204775e0 * t18;
    double t20 = t2 * t2;
    double t21 = t4 * t4;
    double t22 = t20 * t21;
    double t23 = t9 * t9;
    double t26 = t22 * t6 / t23;
    double t27 = 0.123235e0 * t26;
    double t28 = 0.379785e1 * t15 + t17 + t19 + t27;
    double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
    double t32 = std::log(t31);
    double t34 = 0.621814e-1 * t14 * t32;
    double t35 = rho0 - rho1;
    double t36 = t35 * t35;
    double t37 = t36 * t36;
    double t38 = t8 * t8;
    double t39 = t38 * t38;
    double t40 = 0.1e1 / t39;
    double t41 = t37 * t40;
    double t42 = 0.1e1 / t8;
    double t43 = t35 * t42;
    double t44 = 0.1e1 + t43;
    double t45 = t44 <= ZETA_THRESHOLD_C_R2SCAN;
    double t46 = std::cbrt(ZETA_THRESHOLD_C_R2SCAN);
    double t47 = t46 * ZETA_THRESHOLD_C_R2SCAN;
    double t48 = std::cbrt(t44);
    double t49 = t48 * t44;
    double t50 = my_piecewise3(t45, t47, t49);
    double t51 = 0.1e1 - t43;
    double t52 = t51 <= ZETA_THRESHOLD_C_R2SCAN;
    double t53 = std::cbrt(t51);
    double t54 = t53 * t51;
    double t55 = my_piecewise3(t52, t47, t54);
    double t56 = t50 + t55 - 0.2e1;
    double t57 = M_CBRT2;
    double t58 = t57 - 0.1e1;
    double t60 = 0.1e1 / t58 / 0.2e1;
    double t61 = t56 * t60;
    double t63 = 0.1e1 + 0.5137e-1 * t12;
    double t65 = 0.1549425e1 * t12;
    double t66 = 0.420775e0 * t18;
    double t67 = 0.1562925e0 * t26;
    double t68 = 0.705945e1 * t15 + t65 + t66 + t67;
    double t71 = 0.1e1 + 0.32163958997385070134e2 / t68;
    double t72 = std::log(t71);
    double t76 = 0.1e1 + 0.278125e-1 * t12;
    double t78 = 0.905775e0 * t12;
    double t79 = 0.1100325e0 * t18;
    double t80 = 0.1241775e0 * t26;
    double t81 = 0.51785e1 * t15 + t78 + t79 + t80;
    double t84 = 0.1e1 + 0.29608749977793437516e2 / t81;
    double t85 = std::log(t84);
    double t86 = t76 * t85;
    double t88 = -0.310907e-1 * t63 * t72 + t34 - 0.19751673498613801407e-1 * t86;
    double t89 = t61 * t88;
    double t90 = t41 * t89;
    double t92 = 0.19751673498613801407e-1 * t61 * t86;
    double t93 = std::log(0.2e1);
    double t94 = 0.1e1 - t93;
    double t95 = M_PI * M_PI;
    double t97 = t94 / t95;
    double t98 = t46 * t46;
    double t99 = t48 * t48;
    double t100 = my_piecewise3(t45, t98, t99);
    double t101 = t53 * t53;
    double t102 = my_piecewise3(t52, t98, t101);
    double t104 = t100 / 0.2e1 + t102 / 0.2e1;
    double t105 = t104 * t104;
    double t106 = t105 * t104;
    double t108 = 0.1e1 / t94;
    double t109 = (-t34 + t90 + t92) * t108;
    double t110 = 0.1e1 / t106;
    double t111 = t95 * t110;
    double t113 = std::exp(-t109 * t111);
    double t114 = t113 - 0.1e1;
    double t116 = 0.1e1 + 0.25e-1 * t12;
    double t118 = 0.1e1 + 0.4445e-1 * t12;
    double t119 = 0.1e1 / t118;
    double t120 = t116 * t119;
    double t122 = sigma0 + 0.2e1 * sigma1 + sigma2;
    double t124 = 0.1e1 / t9 / t38;
    double t125 = t122 * t124;
    double t128 = 0.1e1 / t105;
    double t129 = t128 * t20;
    double t130 = 0.1e1 / t4;
    double t132 = t6 * t108;
    double t133 = 0.1e1 / t114;
    double t134 = t132 * t133;
    double t135 = t129 * t130 * t134;
    double t138 = t98 * ZETA_THRESHOLD_C_R2SCAN;
    double t139 = t99 * t44;
    double t140 = my_piecewise3(t45, t138, t139);
    double t141 = t101 * t51;
    double t142 = my_piecewise3(t52, t138, t141);
    double t144 = t140 / 0.2e1 + t142 / 0.2e1;
    double t146 = t108 / t144;
    double t147 = t110 * t133;
    double t253 = t147 * t252;
    double t254 = t146 * t253;
    double t255 = M_CBRT6;
    double t256 = std::cbrt(t95);
    double t257 = t256 * t256;
    double t258 = 0.1e1 / t257;
    double t259 = t255 * t258;
    double t260 = t57 * t57;
    double t261 = t259 * t260;
    double t263 = 0.1e1 / t23 / t38;
    double t264 = t122 * t263;
    double t265 = t255 * t255;
    double t267 = 0.1e1 / t256 / t95;
    double t268 = t265 * t267;
    double t269 = t122 * t122;
    double t270 = t57 * t269;
    double t271 = t39 * t8;
    double t273 = 0.1e1 / t9 / t271;
    double t277 = std::exp(-0.20444604078896369094e0 * t268 * t270 * t273);
    double t279 = t261 * t264 * t277;
    double t282 = 0.1e1 + 0.27439371595564631661e-1 * t120 * t125 * t57 * t135 + 0.43341108700271342816e-1 * t254 * t279;
    double t283 = std::sqrt(std::sqrt(t282));
    double t285 = 0.1e1 - 0.1e1 / t283;
    double t287 = t114 * t285 + 0.1e1;
    double t288 = std::log(t287);
    double t290 = t97 * t106 * t288;
    double t375 = t354 * t374;
    double tvrho0 = t8 * t887 + t290 - t34 + t375 + t90 + t92;
    return tvrho0;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho1__t252(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t14 = 0.1e1 + 0.53425e-1 * t12;
    double t15 = std::sqrt(t12);
    double t17 = 0.8969e0 * t12;
    double t18 = t12 * std::sqrt(t12);
    double t19 = 0.204775e0 * t18;
    double t20 = t2 * t2;
    double t21 = t4 * t4;
    double t22 = t20 * t21;
    double t23 = t9 * t9;
    double t26 = t22 * t6 / t23;
    double t27 = 0.123235e0 * t26;
    double t28 = 0.379785e1 * t15 + t17 + t19 + t27;
    double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
    double t32 = std::log(t31);
    double t34 = 0.621814e-1 * t14 * t32;
    double t35 = rho0 - rho1;
    double t36 = t35 * t35;
    double t37 = t36 * t36;
    double t38 = t8 * t8;
    double t39 = t38 * t38;
    double t40 = 0.1e1 / t39;
    double t41 = t37 * t40;
    double t42 = 0.1e1 / t8;
    double t43 = t35 * t42;
    double t44 = 0.1e1 + t43;
    double t45 = t44 <= ZETA_THRESHOLD_C_R2SCAN;
    double t46 = std::cbrt(ZETA_THRESHOLD_C_R2SCAN);
    double t47 = t46 * ZETA_THRESHOLD_C_R2SCAN;
    double t48 = std::cbrt(t44);
    double t49 = t48 * t44;
    double t50 = my_piecewise3(t45, t47, t49);
    double t51 = 0.1e1 - t43;
    double t52 = t51 <= ZETA_THRESHOLD_C_R2SCAN;
    double t53 = std::cbrt(t51);
    double t54 = t53 * t51;
    double t55 = my_piecewise3(t52, t47, t54);
    double t56 = t50 + t55 - 0.2e1;
    double t57 = M_CBRT2;
    double t58 = t57 - 0.1e1;
    double t60 = 0.1e1 / t58 / 0.2e1;
    double t61 = t56 * t60;
    double t63 = 0.1e1 + 0.5137e-1 * t12;
    double t65 = 0.1549425e1 * t12;
    double t66 = 0.420775e0 * t18;
    double t67 = 0.1562925e0 * t26;
    double t68 = 0.705945e1 * t15 + t65 + t66 + t67;
    double t71 = 0.1e1 + 0.32163958997385070134e2 / t68;
    double t72 = std::log(t71);
    double t76 = 0.1e1 + 0.278125e-1 * t12;
    double t78 = 0.905775e0 * t12;
    double t79 = 0.1100325e0 * t18;
    double t80 = 0.1241775e0 * t26;
    double t81 = 0.51785e1 * t15 + t78 + t79 + t80;
    double t84 = 0.1e1 + 0.29608749977793437516e2 / t81;
    double t85 = std::log(t84);
    double t86 = t76 * t85;
    double t88 = -0.310907e-1 * t63 * t72 + t34 - 0.19751673498613801407e-1 * t86;
    double t89 = t61 * t88;
    double t90 = t41 * t89;
    double t92 = 0.19751673498613801407e-1 * t61 * t86;
    double t148 = std::sqrt(0.4e1);
    double t149 = t148 * t15;
    double t151 = 0.3138525e-1 * t12;
    double t152 = 0.1e1 + 0.22225e-1 * t149 + t151;
    double t153 = t152 * t152;
    double t154 = 0.1e1 / t153;
    double t158 = 0.1e1 - 0.2363e1 * t58 * t56 * t60;
    double t159 = t154 * t158;
    double t160 = t37 * t37;
    double t161 = t160 * t37;
    double t162 = t39 * t39;
    double t163 = t162 * t39;
    double t164 = 0.1e1 / t163;
    double t166 = -t161 * t164 + 0.1e1;
    double t167 = 0.1e1 / t15;
    double t168 = t148 * t167;
    double t170 = 0.4445e-1 * t168 + 0.125541e0;
    double t171 = t166 * t170;
    double t175 = 0.1898925e1 * t149 + t17 + t19 + t27;
    double t178 = 0.1e1 + 0.16081979498692535067e2 / t175;
    double t179 = std::log(t178);
    double t180 = 0.1328816518e-1 * t179;
    double t181 = t175 * t175;
    double t182 = 0.1e1 / t181;
    double t183 = t14 * t182;
    double t185 = std::sqrt(t12);
    double t188 = 0.379785e1 * t168 + 0.35876e1 + 0.122865e1 * t185 + 0.24647e0 * t12;
    double t189 = 0.1e1 / t178;
    double t190 = t188 * t189;
    double t192 = 0.1e1 * t183 * t190;
    double t194 = 0.3529725e1 * t149 + t65 + t66 + t67;
    double t197 = 0.1e1 + 0.32163958997385070134e2 / t194;
    double t198 = std::log(t197);
    double t200 = t194 * t194;
    double t201 = 0.1e1 / t200;
    double t202 = t63 * t201;
    double t206 = 0.705945e1 * t168 + 0.61977e1 + 0.252465e1 * t185 + 0.312585e0 * t12;
    double t207 = 0.1e1 / t197;
    double t208 = t206 * t207;
    double t212 = 0.258925e1 * t149 + t78 + t79 + t80;
    double t215 = 0.1e1 + 0.29608749977793437516e2 / t212;
    double t216 = std::log(t215);
    double t218 = t212 * t212;
    double t219 = 0.1e1 / t218;
    double t220 = t76 * t219;
    double t224 = 0.51785e1 * t168 + 0.36231e1 + 0.660195e0 * t185 + 0.248355e0 * t12;
    double t225 = 0.1e1 / t215;
    double t226 = t224 * t225;
    double t229 = -0.6388517036e-2 * t198 + 0.1e1 * t202 * t208 + t180 - t192 - 0.21973736767207854065e-2 * t216 + 0.5848223622634646207e0 * t220 * t226;
    double t230 = t61 * t229;
    double t234 = t61 * t76;
    double t236 = t219 * t224 * t225;
    double t239 = 0.285764e-1 * t159 * t171 + t180 - t192 - t41 * t230 - 0.21973736767207854065e-2 * t61 * t216 + 0.5848223622634646207e0 * t234 * t236;
    double t244 = 0.1e1 + 0.4445e-1 * t15 + t151;
    double t245 = 0.1e1 / t244;
    double t246 = t245 * t158;
    double t252 = 0.5e1 * t5 * t11 * t239 - 0.45e2 * 0.001 * (-0.285764e-1 * t246 * t166 + t34 - t90 - t92);
    return t252;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho1__t371(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t15 = std::sqrt(t12);
    double t23 = t9 * t9;
    double t38 = t8 * t8;
    double t57 = M_CBRT2;
    double t95 = M_PI * M_PI;
    double t122 = sigma0 + 0.2e1 * sigma1 + sigma2;
    double t151 = 0.3138525e-1 * t12;
    double t244 = 0.1e1 + 0.4445e-1 * t15 + t151;
    double t245 = 0.1e1 / t244;
    double t255 = M_CBRT6;
    double t256 = std::cbrt(t95);
    double t257 = t256 * t256;
    double t258 = 0.1e1 / t257;
    double t259 = t255 * t258;
    double t260 = t57 * t57;
    double t263 = 0.1e1 / t23 / t38;
    double t357 = std::exp(0.1e1 * t245);
    double t358 = t357 - 0.1e1;
    double t359 = t260 * t122;
    double t360 = t359 * t263;
    double t363 = 0.1e1 + 0.21337642104376358333e-1 * t259 * t360;
    double t364 = std::sqrt(std::sqrt(t363));
    double t366 = 0.1e1 - 0.1e1 / t364;
    double t368 = t358 * t366 + 0.1e1;
    double t369 = std::log(t368);
    double t371 = -0.285764e-1 * t245 + 0.285764e-1 * t369;
    return t371;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho1__t374(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t252 = mgga_c_r2scan_vrho1__t252(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t371 = mgga_c_r2scan_vrho1__t371(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t14 = 0.1e1 + 0.53425e-1 * t12;
    double t15 = std::sqrt(t12);
    double t17 = 0.8969e0 * t12;
    double t18 = t12 * std::sqrt(t12);
    double t19 = 0.204775e0 * t18;
    double t20 = t2 * t2;
    double t21 = t4 * t4;
    double t22 = t20 * t21;
    double t23 = t9 * t9;
    double t26 = t22 * t6 / t23;
    double t27 = 0.123235e0 * t26;
    double t28 = 0.379785e1 * t15 + t17 + t19 + t27;
    double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
    double t32 = std::log(t31);
    double t34 = 0.621814e-1 * t14 * t32;
    double t35 = rho0 - rho1;
    double t36 = t35 * t35;
    double t37 = t36 * t36;
    double t38 = t8 * t8;
    double t39 = t38 * t38;
    double t40 = 0.1e1 / t39;
    double t41 = t37 * t40;
    double t42 = 0.1e1 / t8;
    double t43 = t35 * t42;
    double t44 = 0.1e1 + t43;
    double t45 = t44 <= ZETA_THRESHOLD_C_R2SCAN;
    double t46 = std::cbrt(ZETA_THRESHOLD_C_R2SCAN);
    double t47 = t46 * ZETA_THRESHOLD_C_R2SCAN;
    double t48 = std::cbrt(t44);
    double t49 = t48 * t44;
    double t50 = my_piecewise3(t45, t47, t49);
    double t51 = 0.1e1 - t43;
    double t52 = t51 <= ZETA_THRESHOLD_C_R2SCAN;
    double t53 = std::cbrt(t51);
    double t54 = t53 * t51;
    double t55 = my_piecewise3(t52, t47, t54);
    double t56 = t50 + t55 - 0.2e1;
    double t57 = M_CBRT2;
    double t58 = t57 - 0.1e1;
    double t60 = 0.1e1 / t58 / 0.2e1;
    double t61 = t56 * t60;
    double t63 = 0.1e1 + 0.5137e-1 * t12;
    double t65 = 0.1549425e1 * t12;
    double t66 = 0.420775e0 * t18;
    double t67 = 0.1562925e0 * t26;
    double t68 = 0.705945e1 * t15 + t65 + t66 + t67;
    double t71 = 0.1e1 + 0.32163958997385070134e2 / t68;
    double t72 = std::log(t71);
    double t76 = 0.1e1 + 0.278125e-1 * t12;
    double t78 = 0.905775e0 * t12;
    double t79 = 0.1100325e0 * t18;
    double t80 = 0.1241775e0 * t26;
    double t81 = 0.51785e1 * t15 + t78 + t79 + t80;
    double t84 = 0.1e1 + 0.29608749977793437516e2 / t81;
    double t85 = std::log(t84);
    double t86 = t76 * t85;
    double t88 = -0.310907e-1 * t63 * t72 + t34 - 0.19751673498613801407e-1 * t86;
    double t89 = t61 * t88;
    double t90 = t41 * t89;
    double t92 = 0.19751673498613801407e-1 * t61 * t86;
    double t93 = std::log(0.2e1);
    double t94 = 0.1e1 - t93;
    double t95 = M_PI * M_PI;
    double t97 = t94 / t95;
    double t98 = t46 * t46;
    double t99 = t48 * t48;
    double t100 = my_piecewise3(t45, t98, t99);
    double t101 = t53 * t53;
    double t102 = my_piecewise3(t52, t98, t101);
    double t104 = t100 / 0.2e1 + t102 / 0.2e1;
    double t105 = t104 * t104;
    double t106 = t105 * t104;
    double t108 = 0.1e1 / t94;
    double t109 = (-t34 + t90 + t92) * t108;
    double t110 = 0.1e1 / t106;
    double t111 = t95 * t110;
    double t113 = std::exp(-t109 * t111);
    double t114 = t113 - 0.1e1;
    double t116 = 0.1e1 + 0.25e-1 * t12;
    double t118 = 0.1e1 + 0.4445e-1 * t12;
    double t119 = 0.1e1 / t118;
    double t120 = t116 * t119;
    double t122 = sigma0 + 0.2e1 * sigma1 + sigma2;
    double t124 = 0.1e1 / t9 / t38;
    double t125 = t122 * t124;
    double t128 = 0.1e1 / t105;
    double t129 = t128 * t20;
    double t130 = 0.1e1 / t4;
    double t132 = t6 * t108;
    double t133 = 0.1e1 / t114;
    double t134 = t132 * t133;
    double t135 = t129 * t130 * t134;
    double t138 = t98 * ZETA_THRESHOLD_C_R2SCAN;
    double t139 = t99 * t44;
    double t140 = my_piecewise3(t45, t138, t139);
    double t141 = t101 * t51;
    double t142 = my_piecewise3(t52, t138, t141);
    double t144 = t140 / 0.2e1 + t142 / 0.2e1;
    double t146 = t108 / t144;
    double t147 = t110 * t133;
    double t158 = 0.1e1 - 0.2363e1 * t58 * t56 * t60;
    double t160 = t37 * t37;
    double t161 = t160 * t37;
    double t162 = t39 * t39;
    double t163 = t162 * t39;
    double t164 = 0.1e1 / t163;
    double t166 = -t161 * t164 + 0.1e1;
    double t253 = t147 * t252;
    double t254 = t146 * t253;
    double t255 = M_CBRT6;
    double t256 = std::cbrt(t95);
    double t257 = t256 * t256;
    double t258 = 0.1e1 / t257;
    double t259 = t255 * t258;
    double t260 = t57 * t57;
    double t261 = t259 * t260;
    double t263 = 0.1e1 / t23 / t38;
    double t264 = t122 * t263;
    double t265 = t255 * t255;
    double t267 = 0.1e1 / t256 / t95;
    double t268 = t265 * t267;
    double t269 = t122 * t122;
    double t270 = t57 * t269;
    double t271 = t39 * t8;
    double t273 = 0.1e1 / t9 / t271;
    double t277 = std::exp(-0.20444604078896369094e0 * t268 * t270 * t273);
    double t279 = t261 * t264 * t277;
    double t282 = 0.1e1 + 0.27439371595564631661e-1 * t120 * t125 * t57 * t135 + 0.43341108700271342816e-1 * t254 * t279;
    double t283 = std::sqrt(std::sqrt(t282));
    double t285 = 0.1e1 - 0.1e1 / t283;
    double t287 = t114 * t285 + 0.1e1;
    double t288 = std::log(t287);
    double t290 = t97 * t106 * t288;
    double t372 = t371 * t158;
    double t374 = t372 * t166 - t290 + t34 - t90 - t92;
    return t374;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho1__t1040(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t374 = mgga_c_r2scan_vrho1__t374(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t23 = t9 * t9;
    double t35 = rho0 - rho1;
    double t38 = t8 * t8;
    double t42 = 0.1e1 / t8;
    double t43 = t35 * t42;
    double t44 = 0.1e1 + t43;
    double t51 = 0.1e1 - t43;
    double t95 = M_PI * M_PI;
    double t122 = sigma0 + 0.2e1 * sigma1 + sigma2;
    double t255 = M_CBRT6;
    double t256 = std::cbrt(t95);
    double t257 = t256 * t256;
    double t263 = 0.1e1 / t23 / t38;
    double t264 = t122 * t263;
    double t265 = t255 * t255;
    double t291 = std::cbrt(rho0);
    double t292 = t291 * t291;
    double t294 = 0.1e1 / t292 / rho0;
    double t295 = tau0 * t294;
    double t296 = t44 / 0.2e1;
    double t297 = std::cbrt(t296);
    double t298 = t297 * t297;
    double t299 = t298 * t296;
    double t301 = std::cbrt(rho1);
    double t302 = t301 * t301;
    double t304 = 0.1e1 / t302 / rho1;
    double t305 = tau1 * t304;
    double t306 = t51 / 0.2e1;
    double t307 = std::cbrt(t306);
    double t308 = t307 * t307;
    double t309 = t308 * t306;
    double t312 = t295 * t299 + t305 * t309 - t264 / 0.8e1;
    double t313 = t265 * t257;
    double t317 = 0.001 * t122;
    double t320 = 0.3e1 / 0.1e2 * t313 * (t299 + t309) + t317 * t263 / 0.8e1;
    double t321 = 0.1e1 / t320;
    double t322 = t312 * t321;
    double t323 = t322 <= 0.e0;
    double t324 = 0.e0 < t322;
    double t325 = my_piecewise3(t324, 0, t322);
    double t326 = 0.1e1 - t325;
    double t327 = 0.1e1 / t326;
    double t330 = std::exp(-0.64e0 * t325 * t327);
    double t331 = t322 <= 0.25e1;
    double t332 = 0.25e1 < t322;
    double t333 = my_piecewise3(t332, 0.25e1, t322);
    double t335 = t333 * t333;
    double t337 = t335 * t333;
    double t339 = t335 * t335;
    double t341 = t339 * t333;
    double t343 = t339 * t335;
    double t348 = my_piecewise3(t332, t322, 0.25e1);
    double t349 = 0.1e1 - t348;
    double t352 = std::exp(0.15e1 / t349);
    double t413 = 0.1e1 / t38;
    double t414 = t35 * t413;
    double t500 = t38 * t8;
    double t502 = 0.1e1 / t23 / t500;
    double t513 = t122 * t502;
    double t814 = t513 / 0.3e1;
    double t817 = t320 * t320;
    double t818 = 0.1e1 / t817;
    double t819 = t312 * t818;
    double t825 = t317 * t502 / 0.3e1;
    double t832 = t326 * t326;
    double t833 = 0.1e1 / t832;
    double t834 = t325 * t833;
    double t854 = t349 * t349;
    double t855 = 0.1e1 / t854;
    double t889 = -t42 - t414;
    double t990 = t889 / 0.2e1;
    double t991 = t298 * t990;
    double t994 = rho1 * rho1;
    double t996 = 0.1e1 / t302 / t994;
    double t997 = tau1 * t996;
    double t1000 = -t990;
    double t1001 = t308 * t1000;
    double t1004 = 0.5e1 / 0.3e1 * t295 * t991 - 0.5e1 / 0.3e1 * t997 * t309 + 0.5e1 / 0.3e1 * t305 * t1001 + t814;
    double t1010 = 0.3e1 / 0.1e2 * t313 * (0.5e1 / 0.3e1 * t991 + 0.5e1 / 0.3e1 * t1001) - t825;
    double t1012 = t1004 * t321 - t819 * t1010;
    double t1013 = my_piecewise3(t324, 0, t1012);
    double t1018 = -0.64e0 * t1013 * t327 - 0.64e0 * t834 * t1013;
    double t1019 = t1018 * t330;
    double t1020 = my_piecewise3(t332, 0, t1012);
    double t1022 = t333 * t1020;
    double t1024 = t335 * t1020;
    double t1026 = t337 * t1020;
    double t1028 = t339 * t1020;
    double t1030 = t341 * t1020;
    double t1035 = my_piecewise3(t332, t1012, 0);
    double t1039 = my_piecewise5(t323, t1019, t331, -0.64e0 * t1020 - 0.8704e0 * t1022 - 0.4607056813647e1 * t1024 + 0.122462410087e2 * t1026 - 0.957855118103e1 * t1028 + 0.3101306810232e1 * t1030 - 0.362942158544e0 * t343 * t1020, -0.105e1 * t855 * t1035 * t352);
    double t1040 = t1039 * t374;
    return t1040;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho1__t354(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t23 = t9 * t9;
    double t35 = rho0 - rho1;
    double t38 = t8 * t8;
    double t42 = 0.1e1 / t8;
    double t43 = t35 * t42;
    double t44 = 0.1e1 + t43;
    double t51 = 0.1e1 - t43;
    double t95 = M_PI * M_PI;
    double t122 = sigma0 + 0.2e1 * sigma1 + sigma2;
    double t255 = M_CBRT6;
    double t256 = std::cbrt(t95);
    double t257 = t256 * t256;
    double t263 = 0.1e1 / t23 / t38;
    double t264 = t122 * t263;
    double t265 = t255 * t255;
    double t291 = std::cbrt(rho0);
    double t292 = t291 * t291;
    double t294 = 0.1e1 / t292 / rho0;
    double t295 = tau0 * t294;
    double t296 = t44 / 0.2e1;
    double t297 = std::cbrt(t296);
    double t298 = t297 * t297;
    double t299 = t298 * t296;
    double t301 = std::cbrt(rho1);
    double t302 = t301 * t301;
    double t304 = 0.1e1 / t302 / rho1;
    double t305 = tau1 * t304;
    double t306 = t51 / 0.2e1;
    double t307 = std::cbrt(t306);
    double t308 = t307 * t307;
    double t309 = t308 * t306;
    double t312 = t295 * t299 + t305 * t309 - t264 / 0.8e1;
    double t313 = t265 * t257;
    double t317 = 0.001 * t122;
    double t320 = 0.3e1 / 0.1e2 * t313 * (t299 + t309) + t317 * t263 / 0.8e1;
    double t321 = 0.1e1 / t320;
    double t322 = t312 * t321;
    double t323 = t322 <= 0.e0;
    double t324 = 0.e0 < t322;
    double t325 = my_piecewise3(t324, 0, t322);
    double t326 = 0.1e1 - t325;
    double t327 = 0.1e1 / t326;
    double t330 = std::exp(-0.64e0 * t325 * t327);
    double t331 = t322 <= 0.25e1;
    double t332 = 0.25e1 < t322;
    double t333 = my_piecewise3(t332, 0.25e1, t322);
    double t335 = t333 * t333;
    double t337 = t335 * t333;
    double t339 = t335 * t335;
    double t341 = t339 * t333;
    double t343 = t339 * t335;
    double t348 = my_piecewise3(t332, t322, 0.25e1);
    double t349 = 0.1e1 - t348;
    double t352 = std::exp(0.15e1 / t349);
    double t354 = my_piecewise5(t323, t330, t331, 0.1e1 - 0.64e0 * t333 - 0.4352e0 * t335 - 0.1535685604549e1 * t337 + 0.3061560252175e1 * t339 - 0.1915710236206e1 * t341 + 0.516884468372e0 * t343 - 0.51848879792e-1 * t339 * t337, -0.7e0 * t352);
    return t354;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho1__t404(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t14 = 0.1e1 + 0.53425e-1 * t12;
    double t15 = std::sqrt(t12);
    double t17 = 0.8969e0 * t12;
    double t18 = t12 * std::sqrt(t12);
    double t19 = 0.204775e0 * t18;
    double t20 = t2 * t2;
    double t21 = t4 * t4;
    double t22 = t20 * t21;
    double t23 = t9 * t9;
    double t26 = t22 * t6 / t23;
    double t27 = 0.123235e0 * t26;
    double t28 = 0.379785e1 * t15 + t17 + t19 + t27;
    double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
    double t167 = 0.1e1 / t15;
    double t185 = std::sqrt(t12);
    double t377 = 0.1e1 / t9 / t8;
    double t378 = t7 * t377;
    double t382 = t28 * t28;
    double t383 = 0.1e1 / t382;
    double t384 = t14 * t383;
    double t385 = t167 * t2;
    double t386 = t4 * t7;
    double t387 = t386 * t377;
    double t388 = t385 * t387;
    double t390 = t5 * t378;
    double t391 = 0.29896666666666666667e0 * t390;
    double t392 = t185 * t2;
    double t393 = t392 * t387;
    double t394 = 0.1023875e0 * t393;
    double t398 = t22 * t6 / t23 / t8;
    double t399 = 0.82156666666666666667e-1 * t398;
    double t400 = -0.632975e0 * t388 - t391 - t394 - t399;
    double t401 = 0.1e1 / t31;
    double t402 = t400 * t401;
    double t403 = t384 * t402;
    double t404 = 0.1e1 * t403;
    return t404;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho1__t408(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t14 = 0.1e1 + 0.53425e-1 * t12;
    double t15 = std::sqrt(t12);
    double t17 = 0.8969e0 * t12;
    double t18 = t12 * std::sqrt(t12);
    double t19 = 0.204775e0 * t18;
    double t20 = t2 * t2;
    double t21 = t4 * t4;
    double t22 = t20 * t21;
    double t23 = t9 * t9;
    double t26 = t22 * t6 / t23;
    double t27 = 0.123235e0 * t26;
    double t28 = 0.379785e1 * t15 + t17 + t19 + t27;
    double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
    double t32 = std::log(t31);
    double t34 = 0.621814e-1 * t14 * t32;
    double t35 = rho0 - rho1;
    double t36 = t35 * t35;
    double t38 = t8 * t8;
    double t39 = t38 * t38;
    double t40 = 0.1e1 / t39;
    double t42 = 0.1e1 / t8;
    double t43 = t35 * t42;
    double t44 = 0.1e1 + t43;
    double t45 = t44 <= ZETA_THRESHOLD_C_R2SCAN;
    double t46 = std::cbrt(ZETA_THRESHOLD_C_R2SCAN);
    double t47 = t46 * ZETA_THRESHOLD_C_R2SCAN;
    double t48 = std::cbrt(t44);
    double t49 = t48 * t44;
    double t50 = my_piecewise3(t45, t47, t49);
    double t51 = 0.1e1 - t43;
    double t52 = t51 <= ZETA_THRESHOLD_C_R2SCAN;
    double t53 = std::cbrt(t51);
    double t54 = t53 * t51;
    double t55 = my_piecewise3(t52, t47, t54);
    double t56 = t50 + t55 - 0.2e1;
    double t57 = M_CBRT2;
    double t58 = t57 - 0.1e1;
    double t60 = 0.1e1 / t58 / 0.2e1;
    double t61 = t56 * t60;
    double t63 = 0.1e1 + 0.5137e-1 * t12;
    double t65 = 0.1549425e1 * t12;
    double t66 = 0.420775e0 * t18;
    double t67 = 0.1562925e0 * t26;
    double t68 = 0.705945e1 * t15 + t65 + t66 + t67;
    double t71 = 0.1e1 + 0.32163958997385070134e2 / t68;
    double t72 = std::log(t71);
    double t76 = 0.1e1 + 0.278125e-1 * t12;
    double t78 = 0.905775e0 * t12;
    double t79 = 0.1100325e0 * t18;
    double t80 = 0.1241775e0 * t26;
    double t81 = 0.51785e1 * t15 + t78 + t79 + t80;
    double t84 = 0.1e1 + 0.29608749977793437516e2 / t81;
    double t85 = std::log(t84);
    double t86 = t76 * t85;
    double t88 = -0.310907e-1 * t63 * t72 + t34 - 0.19751673498613801407e-1 * t86;
    double t89 = t61 * t88;
    double t405 = t36 * t35;
    double t406 = t405 * t40;
    double t407 = t406 * t89;
    double t408 = 0.4e1 * t407;
    return t408;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho1__t459(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t404 = mgga_c_r2scan_vrho1__t404(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t15 = std::sqrt(t12);
    double t17 = 0.8969e0 * t12;
    double t18 = t12 * std::sqrt(t12);
    double t19 = 0.204775e0 * t18;
    double t20 = t2 * t2;
    double t21 = t4 * t4;
    double t22 = t20 * t21;
    double t23 = t9 * t9;
    double t26 = t22 * t6 / t23;
    double t27 = 0.123235e0 * t26;
    double t28 = 0.379785e1 * t15 + t17 + t19 + t27;
    double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
    double t32 = std::log(t31);
    double t35 = rho0 - rho1;
    double t36 = t35 * t35;
    double t37 = t36 * t36;
    double t38 = t8 * t8;
    double t39 = t38 * t38;
    double t40 = 0.1e1 / t39;
    double t41 = t37 * t40;
    double t42 = 0.1e1 / t8;
    double t43 = t35 * t42;
    double t44 = 0.1e1 + t43;
    double t45 = t44 <= ZETA_THRESHOLD_C_R2SCAN;
    double t46 = std::cbrt(ZETA_THRESHOLD_C_R2SCAN);
    double t47 = t46 * ZETA_THRESHOLD_C_R2SCAN;
    double t48 = std::cbrt(t44);
    double t49 = t48 * t44;
    double t50 = my_piecewise3(t45, t47, t49);
    double t51 = 0.1e1 - t43;
    double t52 = t51 <= ZETA_THRESHOLD_C_R2SCAN;
    double t53 = std::cbrt(t51);
    double t54 = t53 * t51;
    double t55 = my_piecewise3(t52, t47, t54);
    double t56 = t50 + t55 - 0.2e1;
    double t57 = M_CBRT2;
    double t58 = t57 - 0.1e1;
    double t60 = 0.1e1 / t58 / 0.2e1;
    double t61 = t56 * t60;
    double t63 = 0.1e1 + 0.5137e-1 * t12;
    double t65 = 0.1549425e1 * t12;
    double t66 = 0.420775e0 * t18;
    double t67 = 0.1562925e0 * t26;
    double t68 = 0.705945e1 * t15 + t65 + t66 + t67;
    double t71 = 0.1e1 + 0.32163958997385070134e2 / t68;
    double t72 = std::log(t71);
    double t76 = 0.1e1 + 0.278125e-1 * t12;
    double t78 = 0.905775e0 * t12;
    double t79 = 0.1100325e0 * t18;
    double t80 = 0.1241775e0 * t26;
    double t81 = 0.51785e1 * t15 + t78 + t79 + t80;
    double t84 = 0.1e1 + 0.29608749977793437516e2 / t81;
    double t85 = std::log(t84);
    double t167 = 0.1e1 / t15;
    double t185 = std::sqrt(t12);
    double t377 = 0.1e1 / t9 / t8;
    double t378 = t7 * t377;
    double t380 = t5 * t378 * t32;
    double t381 = 0.11073470983333333333e-2 * t380;
    double t385 = t167 * t2;
    double t386 = t4 * t7;
    double t387 = t386 * t377;
    double t388 = t385 * t387;
    double t390 = t5 * t378;
    double t392 = t185 * t2;
    double t393 = t392 * t387;
    double t398 = t22 * t6 / t23 / t8;
    double t430 = t68 * t68;
    double t431 = 0.1e1 / t430;
    double t432 = t63 * t431;
    double t434 = 0.516475e0 * t390;
    double t435 = 0.2103875e0 * t393;
    double t436 = 0.104195e0 * t398;
    double t437 = -0.1176575e1 * t388 - t434 - t435 - t436;
    double t438 = 0.1e1 / t71;
    double t439 = t437 * t438;
    double t445 = t81 * t81;
    double t446 = 0.1e1 / t445;
    double t447 = t76 * t446;
    double t449 = 0.301925e0 * t390;
    double t450 = 0.5501625e-1 * t393;
    double t451 = 0.82785e-1 * t398;
    double t452 = -0.86308333333333333334e0 * t388 - t449 - t450 - t451;
    double t453 = 0.1e1 / t84;
    double t454 = t452 * t453;
    double t457 = 0.53237641966666666666e-3 * t5 * t378 * t72 + 0.1e1 * t432 * t439 - t381 - t404 + 0.18311447306006545054e-3 * t5 * t378 * t85 + 0.5848223622634646207e0 * t447 * t454;
    double t458 = t61 * t457;
    double t459 = t41 * t458;
    return t459;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho1__t880(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t15 = std::sqrt(t12);
    double t23 = t9 * t9;
    double t35 = rho0 - rho1;
    double t36 = t35 * t35;
    double t37 = t36 * t36;
    double t38 = t8 * t8;
    double t39 = t38 * t38;
    double t42 = 0.1e1 / t8;
    double t43 = t35 * t42;
    double t44 = 0.1e1 + t43;
    double t45 = t44 <= ZETA_THRESHOLD_C_R2SCAN;
    double t46 = std::cbrt(ZETA_THRESHOLD_C_R2SCAN);
    double t47 = t46 * ZETA_THRESHOLD_C_R2SCAN;
    double t48 = std::cbrt(t44);
    double t49 = t48 * t44;
    double t50 = my_piecewise3(t45, t47, t49);
    double t51 = 0.1e1 - t43;
    double t52 = t51 <= ZETA_THRESHOLD_C_R2SCAN;
    double t53 = std::cbrt(t51);
    double t54 = t53 * t51;
    double t55 = my_piecewise3(t52, t47, t54);
    double t56 = t50 + t55 - 0.2e1;
    double t57 = M_CBRT2;
    double t58 = t57 - 0.1e1;
    double t60 = 0.1e1 / t58 / 0.2e1;
    double t95 = M_PI * M_PI;
    double t122 = sigma0 + 0.2e1 * sigma1 + sigma2;
    double t151 = 0.3138525e-1 * t12;
    double t158 = 0.1e1 - 0.2363e1 * t58 * t56 * t60;
    double t160 = t37 * t37;
    double t161 = t160 * t37;
    double t162 = t39 * t39;
    double t163 = t162 * t39;
    double t164 = 0.1e1 / t163;
    double t166 = -t161 * t164 + 0.1e1;
    double t167 = 0.1e1 / t15;
    double t244 = 0.1e1 + 0.4445e-1 * t15 + t151;
    double t245 = 0.1e1 / t244;
    double t255 = M_CBRT6;
    double t256 = std::cbrt(t95);
    double t257 = t256 * t256;
    double t258 = 0.1e1 / t257;
    double t259 = t255 * t258;
    double t260 = t57 * t57;
    double t263 = 0.1e1 / t23 / t38;
    double t357 = std::exp(0.1e1 * t245);
    double t358 = t357 - 0.1e1;
    double t359 = t260 * t122;
    double t360 = t359 * t263;
    double t363 = 0.1e1 + 0.21337642104376358333e-1 * t259 * t360;
    double t364 = std::sqrt(std::sqrt(t363));
    double t366 = 0.1e1 - 0.1e1 / t364;
    double t368 = t358 * t366 + 0.1e1;
    double t377 = 0.1e1 / t9 / t8;
    double t378 = t7 * t377;
    double t385 = t167 * t2;
    double t386 = t4 * t7;
    double t387 = t386 * t377;
    double t388 = t385 * t387;
    double t390 = t5 * t378;
    double t500 = t38 * t8;
    double t502 = 0.1e1 / t23 / t500;
    double t513 = t122 * t502;
    double t550 = t258 * t260;
    double t590 = 0.1046175e-1 * t390;
    double t757 = t244 * t244;
    double t758 = 0.1e1 / t757;
    double t761 = -0.74083333333333333333e-2 * t388 - t590;
    double t862 = t758 * t761;
    double t864 = t357 * t366;
    double t868 = 0.1e1 / t364 / t363;
    double t869 = t358 * t868;
    double t870 = t869 * t255;
    double t874 = -0.1e1 * t862 * t864 - 0.14225094736250905555e-1 * t870 * t550 * t513;
    double t875 = 0.1e1 / t368;
    double t878 = 0.285764e-1 * t862 + 0.285764e-1 * t874 * t875;
    double t879 = t878 * t158;
    double t880 = t879 * t166;
    return t880;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho1__t898(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t8 = rho0 + rho1;
    double t35 = rho0 - rho1;
    double t38 = t8 * t8;
    double t42 = 0.1e1 / t8;
    double t43 = t35 * t42;
    double t44 = 0.1e1 + t43;
    double t45 = t44 <= ZETA_THRESHOLD_C_R2SCAN;
    double t48 = std::cbrt(t44);
    double t51 = 0.1e1 - t43;
    double t52 = t51 <= ZETA_THRESHOLD_C_R2SCAN;
    double t53 = std::cbrt(t51);
    double t57 = M_CBRT2;
    double t58 = t57 - 0.1e1;
    double t60 = 0.1e1 / t58 / 0.2e1;
    double t413 = 0.1e1 / t38;
    double t414 = t35 * t413;
    double t889 = -t42 - t414;
    double t892 = my_piecewise3(t45, 0, 0.4e1 / 0.3e1 * t48 * t889);
    double t893 = -t889;
    double t896 = my_piecewise3(t52, 0, 0.4e1 / 0.3e1 * t53 * t893);
    double t897 = t892 + t896;
    double t898 = t897 * t60;
    return t898;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho1__t910(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t8 = rho0 + rho1;
    double t35 = rho0 - rho1;
    double t38 = t8 * t8;
    double t42 = 0.1e1 / t8;
    double t43 = t35 * t42;
    double t44 = 0.1e1 + t43;
    double t45 = t44 <= ZETA_THRESHOLD_C_R2SCAN;
    double t48 = std::cbrt(t44);
    double t51 = 0.1e1 - t43;
    double t52 = t51 <= ZETA_THRESHOLD_C_R2SCAN;
    double t53 = std::cbrt(t51);
    double t413 = 0.1e1 / t38;
    double t414 = t35 * t413;
    double t472 = 0.1e1 / t48;
    double t476 = 0.1e1 / t53;
    double t889 = -t42 - t414;
    double t893 = -t889;
    double t903 = t472 * t889;
    double t905 = my_piecewise3(t45, 0, 0.2e1 / 0.3e1 * t903);
    double t906 = t476 * t893;
    double t908 = my_piecewise3(t52, 0, 0.2e1 / 0.3e1 * t906);
    double t910 = t905 / 0.2e1 + t908 / 0.2e1;
    return t910;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho1__t913(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t252 = mgga_c_r2scan_vrho1__t252(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t910 = mgga_c_r2scan_vrho1__t910(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t14 = 0.1e1 + 0.53425e-1 * t12;
    double t15 = std::sqrt(t12);
    double t17 = 0.8969e0 * t12;
    double t18 = t12 * std::sqrt(t12);
    double t19 = 0.204775e0 * t18;
    double t20 = t2 * t2;
    double t21 = t4 * t4;
    double t22 = t20 * t21;
    double t23 = t9 * t9;
    double t26 = t22 * t6 / t23;
    double t27 = 0.123235e0 * t26;
    double t28 = 0.379785e1 * t15 + t17 + t19 + t27;
    double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
    double t32 = std::log(t31);
    double t34 = 0.621814e-1 * t14 * t32;
    double t35 = rho0 - rho1;
    double t36 = t35 * t35;
    double t37 = t36 * t36;
    double t38 = t8 * t8;
    double t39 = t38 * t38;
    double t40 = 0.1e1 / t39;
    double t41 = t37 * t40;
    double t42 = 0.1e1 / t8;
    double t43 = t35 * t42;
    double t44 = 0.1e1 + t43;
    double t45 = t44 <= ZETA_THRESHOLD_C_R2SCAN;
    double t46 = std::cbrt(ZETA_THRESHOLD_C_R2SCAN);
    double t47 = t46 * ZETA_THRESHOLD_C_R2SCAN;
    double t48 = std::cbrt(t44);
    double t49 = t48 * t44;
    double t50 = my_piecewise3(t45, t47, t49);
    double t51 = 0.1e1 - t43;
    double t52 = t51 <= ZETA_THRESHOLD_C_R2SCAN;
    double t53 = std::cbrt(t51);
    double t54 = t53 * t51;
    double t55 = my_piecewise3(t52, t47, t54);
    double t56 = t50 + t55 - 0.2e1;
    double t57 = M_CBRT2;
    double t58 = t57 - 0.1e1;
    double t60 = 0.1e1 / t58 / 0.2e1;
    double t61 = t56 * t60;
    double t63 = 0.1e1 + 0.5137e-1 * t12;
    double t65 = 0.1549425e1 * t12;
    double t66 = 0.420775e0 * t18;
    double t67 = 0.1562925e0 * t26;
    double t68 = 0.705945e1 * t15 + t65 + t66 + t67;
    double t71 = 0.1e1 + 0.32163958997385070134e2 / t68;
    double t72 = std::log(t71);
    double t76 = 0.1e1 + 0.278125e-1 * t12;
    double t78 = 0.905775e0 * t12;
    double t79 = 0.1100325e0 * t18;
    double t80 = 0.1241775e0 * t26;
    double t81 = 0.51785e1 * t15 + t78 + t79 + t80;
    double t84 = 0.1e1 + 0.29608749977793437516e2 / t81;
    double t85 = std::log(t84);
    double t86 = t76 * t85;
    double t88 = -0.310907e-1 * t63 * t72 + t34 - 0.19751673498613801407e-1 * t86;
    double t89 = t61 * t88;
    double t90 = t41 * t89;
    double t92 = 0.19751673498613801407e-1 * t61 * t86;
    double t93 = std::log(0.2e1);
    double t94 = 0.1e1 - t93;
    double t95 = M_PI * M_PI;
    double t97 = t94 / t95;
    double t98 = t46 * t46;
    double t99 = t48 * t48;
    double t100 = my_piecewise3(t45, t98, t99);
    double t101 = t53 * t53;
    double t102 = my_piecewise3(t52, t98, t101);
    double t104 = t100 / 0.2e1 + t102 / 0.2e1;
    double t105 = t104 * t104;
    double t106 = t105 * t104;
    double t108 = 0.1e1 / t94;
    double t109 = (-t34 + t90 + t92) * t108;
    double t110 = 0.1e1 / t106;
    double t111 = t95 * t110;
    double t113 = std::exp(-t109 * t111);
    double t114 = t113 - 0.1e1;
    double t116 = 0.1e1 + 0.25e-1 * t12;
    double t118 = 0.1e1 + 0.4445e-1 * t12;
    double t119 = 0.1e1 / t118;
    double t120 = t116 * t119;
    double t122 = sigma0 + 0.2e1 * sigma1 + sigma2;
    double t124 = 0.1e1 / t9 / t38;
    double t125 = t122 * t124;
    double t128 = 0.1e1 / t105;
    double t129 = t128 * t20;
    double t130 = 0.1e1 / t4;
    double t132 = t6 * t108;
    double t133 = 0.1e1 / t114;
    double t134 = t132 * t133;
    double t135 = t129 * t130 * t134;
    double t138 = t98 * ZETA_THRESHOLD_C_R2SCAN;
    double t139 = t99 * t44;
    double t140 = my_piecewise3(t45, t138, t139);
    double t141 = t101 * t51;
    double t142 = my_piecewise3(t52, t138, t141);
    double t144 = t140 / 0.2e1 + t142 / 0.2e1;
    double t146 = t108 / t144;
    double t147 = t110 * t133;
    double t253 = t147 * t252;
    double t254 = t146 * t253;
    double t255 = M_CBRT6;
    double t256 = std::cbrt(t95);
    double t257 = t256 * t256;
    double t258 = 0.1e1 / t257;
    double t259 = t255 * t258;
    double t260 = t57 * t57;
    double t261 = t259 * t260;
    double t263 = 0.1e1 / t23 / t38;
    double t264 = t122 * t263;
    double t265 = t255 * t255;
    double t267 = 0.1e1 / t256 / t95;
    double t268 = t265 * t267;
    double t269 = t122 * t122;
    double t270 = t57 * t269;
    double t271 = t39 * t8;
    double t273 = 0.1e1 / t9 / t271;
    double t277 = std::exp(-0.20444604078896369094e0 * t268 * t270 * t273);
    double t279 = t261 * t264 * t277;
    double t282 = 0.1e1 + 0.27439371595564631661e-1 * t120 * t125 * t57 * t135 + 0.43341108700271342816e-1 * t254 * t279;
    double t283 = std::sqrt(std::sqrt(t282));
    double t285 = 0.1e1 - 0.1e1 / t283;
    double t287 = t114 * t285 + 0.1e1;
    double t288 = std::log(t287);
    double t471 = t105 * t288;
    double t912 = t97 * t471 * t910;
    double t913 = 0.3e1 * t912;
    return t913;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho1__t955(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t8 = rho0 + rho1;
    double t35 = rho0 - rho1;
    double t36 = t35 * t35;
    double t37 = t36 * t36;
    double t38 = t8 * t8;
    double t39 = t38 * t38;
    double t160 = t37 * t37;
    double t161 = t160 * t37;
    double t162 = t39 * t39;
    double t163 = t162 * t39;
    double t164 = 0.1e1 / t163;
    double t271 = t39 * t8;
    double t405 = t36 * t35;
    double t601 = t160 * t405;
    double t602 = t601 * t164;
    double t603 = t162 * t271;
    double t604 = 0.1e1 / t603;
    double t605 = t161 * t604;
    double t955 = 0.12e2 * t602 + 0.12e2 * t605;
    return t955;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho1__t915(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t404 = mgga_c_r2scan_vrho1__t404(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t408 = mgga_c_r2scan_vrho1__t408(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t459 = mgga_c_r2scan_vrho1__t459(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t898 = mgga_c_r2scan_vrho1__t898(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t14 = 0.1e1 + 0.53425e-1 * t12;
    double t15 = std::sqrt(t12);
    double t17 = 0.8969e0 * t12;
    double t18 = t12 * std::sqrt(t12);
    double t19 = 0.204775e0 * t18;
    double t20 = t2 * t2;
    double t21 = t4 * t4;
    double t22 = t20 * t21;
    double t23 = t9 * t9;
    double t26 = t22 * t6 / t23;
    double t27 = 0.123235e0 * t26;
    double t28 = 0.379785e1 * t15 + t17 + t19 + t27;
    double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
    double t32 = std::log(t31);
    double t34 = 0.621814e-1 * t14 * t32;
    double t35 = rho0 - rho1;
    double t36 = t35 * t35;
    double t37 = t36 * t36;
    double t38 = t8 * t8;
    double t39 = t38 * t38;
    double t40 = 0.1e1 / t39;
    double t41 = t37 * t40;
    double t42 = 0.1e1 / t8;
    double t43 = t35 * t42;
    double t44 = 0.1e1 + t43;
    double t45 = t44 <= ZETA_THRESHOLD_C_R2SCAN;
    double t46 = std::cbrt(ZETA_THRESHOLD_C_R2SCAN);
    double t47 = t46 * ZETA_THRESHOLD_C_R2SCAN;
    double t48 = std::cbrt(t44);
    double t49 = t48 * t44;
    double t50 = my_piecewise3(t45, t47, t49);
    double t51 = 0.1e1 - t43;
    double t52 = t51 <= ZETA_THRESHOLD_C_R2SCAN;
    double t53 = std::cbrt(t51);
    double t54 = t53 * t51;
    double t55 = my_piecewise3(t52, t47, t54);
    double t56 = t50 + t55 - 0.2e1;
    double t57 = M_CBRT2;
    double t58 = t57 - 0.1e1;
    double t60 = 0.1e1 / t58 / 0.2e1;
    double t61 = t56 * t60;
    double t63 = 0.1e1 + 0.5137e-1 * t12;
    double t65 = 0.1549425e1 * t12;
    double t66 = 0.420775e0 * t18;
    double t67 = 0.1562925e0 * t26;
    double t68 = 0.705945e1 * t15 + t65 + t66 + t67;
    double t71 = 0.1e1 + 0.32163958997385070134e2 / t68;
    double t72 = std::log(t71);
    double t76 = 0.1e1 + 0.278125e-1 * t12;
    double t78 = 0.905775e0 * t12;
    double t79 = 0.1100325e0 * t18;
    double t80 = 0.1241775e0 * t26;
    double t81 = 0.51785e1 * t15 + t78 + t79 + t80;
    double t84 = 0.1e1 + 0.29608749977793437516e2 / t81;
    double t85 = std::log(t84);
    double t86 = t76 * t85;
    double t88 = -0.310907e-1 * t63 * t72 + t34 - 0.19751673498613801407e-1 * t86;
    double t89 = t61 * t88;
    double t93 = std::log(0.2e1);
    double t94 = 0.1e1 - t93;
    double t108 = 0.1e1 / t94;
    double t167 = 0.1e1 / t15;
    double t185 = std::sqrt(t12);
    double t234 = t61 * t76;
    double t271 = t39 * t8;
    double t377 = 0.1e1 / t9 / t8;
    double t378 = t7 * t377;
    double t380 = t5 * t378 * t32;
    double t381 = 0.11073470983333333333e-2 * t380;
    double t385 = t167 * t2;
    double t386 = t4 * t7;
    double t387 = t386 * t377;
    double t388 = t385 * t387;
    double t390 = t5 * t378;
    double t392 = t185 * t2;
    double t393 = t392 * t387;
    double t398 = t22 * t6 / t23 / t8;
    double t409 = 0.1e1 / t271;
    double t410 = t37 * t409;
    double t411 = t410 * t89;
    double t412 = 0.4e1 * t411;
    double t445 = t81 * t81;
    double t446 = 0.1e1 / t445;
    double t449 = 0.301925e0 * t390;
    double t450 = 0.5501625e-1 * t393;
    double t451 = 0.82785e-1 * t398;
    double t452 = -0.86308333333333333334e0 * t388 - t449 - t450 - t451;
    double t453 = 0.1e1 / t84;
    double t462 = t61 * t2;
    double t464 = t386 * t377 * t85;
    double t465 = t462 * t464;
    double t466 = 0.18311447306006545054e-3 * t465;
    double t468 = t446 * t452 * t453;
    double t469 = t234 * t468;
    double t470 = 0.5848223622634646207e0 * t469;
    double t899 = t898 * t88;
    double t900 = t41 * t899;
    double t901 = t898 * t86;
    double t902 = 0.19751673498613801407e-1 * t901;
    double t915 = (t381 + t404 - t408 - t412 + t900 + t459 + t902 - t466 - t470) * t108;
    return t915;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho1__t921(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t910 = mgga_c_r2scan_vrho1__t910(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t915 = mgga_c_r2scan_vrho1__t915(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t14 = 0.1e1 + 0.53425e-1 * t12;
    double t15 = std::sqrt(t12);
    double t17 = 0.8969e0 * t12;
    double t18 = t12 * std::sqrt(t12);
    double t19 = 0.204775e0 * t18;
    double t20 = t2 * t2;
    double t21 = t4 * t4;
    double t22 = t20 * t21;
    double t23 = t9 * t9;
    double t26 = t22 * t6 / t23;
    double t27 = 0.123235e0 * t26;
    double t28 = 0.379785e1 * t15 + t17 + t19 + t27;
    double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
    double t32 = std::log(t31);
    double t34 = 0.621814e-1 * t14 * t32;
    double t35 = rho0 - rho1;
    double t36 = t35 * t35;
    double t37 = t36 * t36;
    double t38 = t8 * t8;
    double t39 = t38 * t38;
    double t40 = 0.1e1 / t39;
    double t41 = t37 * t40;
    double t42 = 0.1e1 / t8;
    double t43 = t35 * t42;
    double t44 = 0.1e1 + t43;
    double t45 = t44 <= ZETA_THRESHOLD_C_R2SCAN;
    double t46 = std::cbrt(ZETA_THRESHOLD_C_R2SCAN);
    double t47 = t46 * ZETA_THRESHOLD_C_R2SCAN;
    double t48 = std::cbrt(t44);
    double t49 = t48 * t44;
    double t50 = my_piecewise3(t45, t47, t49);
    double t51 = 0.1e1 - t43;
    double t52 = t51 <= ZETA_THRESHOLD_C_R2SCAN;
    double t53 = std::cbrt(t51);
    double t54 = t53 * t51;
    double t55 = my_piecewise3(t52, t47, t54);
    double t56 = t50 + t55 - 0.2e1;
    double t57 = M_CBRT2;
    double t58 = t57 - 0.1e1;
    double t60 = 0.1e1 / t58 / 0.2e1;
    double t61 = t56 * t60;
    double t63 = 0.1e1 + 0.5137e-1 * t12;
    double t65 = 0.1549425e1 * t12;
    double t66 = 0.420775e0 * t18;
    double t67 = 0.1562925e0 * t26;
    double t68 = 0.705945e1 * t15 + t65 + t66 + t67;
    double t71 = 0.1e1 + 0.32163958997385070134e2 / t68;
    double t72 = std::log(t71);
    double t76 = 0.1e1 + 0.278125e-1 * t12;
    double t78 = 0.905775e0 * t12;
    double t79 = 0.1100325e0 * t18;
    double t80 = 0.1241775e0 * t26;
    double t81 = 0.51785e1 * t15 + t78 + t79 + t80;
    double t84 = 0.1e1 + 0.29608749977793437516e2 / t81;
    double t85 = std::log(t84);
    double t86 = t76 * t85;
    double t88 = -0.310907e-1 * t63 * t72 + t34 - 0.19751673498613801407e-1 * t86;
    double t89 = t61 * t88;
    double t90 = t41 * t89;
    double t92 = 0.19751673498613801407e-1 * t61 * t86;
    double t93 = std::log(0.2e1);
    double t94 = 0.1e1 - t93;
    double t95 = M_PI * M_PI;
    double t98 = t46 * t46;
    double t99 = t48 * t48;
    double t100 = my_piecewise3(t45, t98, t99);
    double t101 = t53 * t53;
    double t102 = my_piecewise3(t52, t98, t101);
    double t104 = t100 / 0.2e1 + t102 / 0.2e1;
    double t105 = t104 * t104;
    double t106 = t105 * t104;
    double t108 = 0.1e1 / t94;
    double t109 = (-t34 + t90 + t92) * t108;
    double t110 = 0.1e1 / t106;
    double t111 = t95 * t110;
    double t113 = std::exp(-t109 * t111);
    double t488 = t105 * t105;
    double t489 = 0.1e1 / t488;
    double t490 = t95 * t489;
    double t917 = t490 * t910;
    double t920 = 0.3e1 * t109 * t917 - t915 * t111;
    double t921 = t920 * t113;
    return t921;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho1__t516(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t14 = 0.1e1 + 0.53425e-1 * t12;
    double t15 = std::sqrt(t12);
    double t17 = 0.8969e0 * t12;
    double t18 = t12 * std::sqrt(t12);
    double t19 = 0.204775e0 * t18;
    double t20 = t2 * t2;
    double t21 = t4 * t4;
    double t22 = t20 * t21;
    double t23 = t9 * t9;
    double t26 = t22 * t6 / t23;
    double t27 = 0.123235e0 * t26;
    double t28 = 0.379785e1 * t15 + t17 + t19 + t27;
    double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
    double t32 = std::log(t31);
    double t34 = 0.621814e-1 * t14 * t32;
    double t35 = rho0 - rho1;
    double t36 = t35 * t35;
    double t37 = t36 * t36;
    double t38 = t8 * t8;
    double t39 = t38 * t38;
    double t40 = 0.1e1 / t39;
    double t41 = t37 * t40;
    double t42 = 0.1e1 / t8;
    double t43 = t35 * t42;
    double t44 = 0.1e1 + t43;
    double t45 = t44 <= ZETA_THRESHOLD_C_R2SCAN;
    double t46 = std::cbrt(ZETA_THRESHOLD_C_R2SCAN);
    double t47 = t46 * ZETA_THRESHOLD_C_R2SCAN;
    double t48 = std::cbrt(t44);
    double t49 = t48 * t44;
    double t50 = my_piecewise3(t45, t47, t49);
    double t51 = 0.1e1 - t43;
    double t52 = t51 <= ZETA_THRESHOLD_C_R2SCAN;
    double t53 = std::cbrt(t51);
    double t54 = t53 * t51;
    double t55 = my_piecewise3(t52, t47, t54);
    double t56 = t50 + t55 - 0.2e1;
    double t57 = M_CBRT2;
    double t58 = t57 - 0.1e1;
    double t60 = 0.1e1 / t58 / 0.2e1;
    double t61 = t56 * t60;
    double t63 = 0.1e1 + 0.5137e-1 * t12;
    double t65 = 0.1549425e1 * t12;
    double t66 = 0.420775e0 * t18;
    double t67 = 0.1562925e0 * t26;
    double t68 = 0.705945e1 * t15 + t65 + t66 + t67;
    double t71 = 0.1e1 + 0.32163958997385070134e2 / t68;
    double t72 = std::log(t71);
    double t76 = 0.1e1 + 0.278125e-1 * t12;
    double t78 = 0.905775e0 * t12;
    double t79 = 0.1100325e0 * t18;
    double t80 = 0.1241775e0 * t26;
    double t81 = 0.51785e1 * t15 + t78 + t79 + t80;
    double t84 = 0.1e1 + 0.29608749977793437516e2 / t81;
    double t85 = std::log(t84);
    double t86 = t76 * t85;
    double t88 = -0.310907e-1 * t63 * t72 + t34 - 0.19751673498613801407e-1 * t86;
    double t89 = t61 * t88;
    double t90 = t41 * t89;
    double t92 = 0.19751673498613801407e-1 * t61 * t86;
    double t93 = std::log(0.2e1);
    double t94 = 0.1e1 - t93;
    double t95 = M_PI * M_PI;
    double t98 = t46 * t46;
    double t99 = t48 * t48;
    double t100 = my_piecewise3(t45, t98, t99);
    double t101 = t53 * t53;
    double t102 = my_piecewise3(t52, t98, t101);
    double t104 = t100 / 0.2e1 + t102 / 0.2e1;
    double t105 = t104 * t104;
    double t106 = t105 * t104;
    double t108 = 0.1e1 / t94;
    double t109 = (-t34 + t90 + t92) * t108;
    double t110 = 0.1e1 / t106;
    double t111 = t95 * t110;
    double t113 = std::exp(-t109 * t111);
    double t114 = t113 - 0.1e1;
    double t116 = 0.1e1 + 0.25e-1 * t12;
    double t118 = 0.1e1 + 0.4445e-1 * t12;
    double t122 = sigma0 + 0.2e1 * sigma1 + sigma2;
    double t128 = 0.1e1 / t105;
    double t133 = 0.1e1 / t114;
    double t500 = t38 * t8;
    double t502 = 0.1e1 / t23 / t500;
    double t505 = t57 * t128;
    double t506 = t108 * t133;
    double t507 = t505 * t506;
    double t510 = t118 * t118;
    double t511 = 0.1e1 / t510;
    double t512 = t116 * t511;
    double t513 = t122 * t502;
    double t514 = t512 * t513;
    double t516 = 0.48787202696913915093e-2 * t514 * t507;
    return t516;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho1__t791(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t252 = mgga_c_r2scan_vrho1__t252(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t14 = 0.1e1 + 0.53425e-1 * t12;
    double t15 = std::sqrt(t12);
    double t17 = 0.8969e0 * t12;
    double t18 = t12 * std::sqrt(t12);
    double t19 = 0.204775e0 * t18;
    double t20 = t2 * t2;
    double t21 = t4 * t4;
    double t22 = t20 * t21;
    double t23 = t9 * t9;
    double t26 = t22 * t6 / t23;
    double t27 = 0.123235e0 * t26;
    double t28 = 0.379785e1 * t15 + t17 + t19 + t27;
    double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
    double t32 = std::log(t31);
    double t34 = 0.621814e-1 * t14 * t32;
    double t35 = rho0 - rho1;
    double t36 = t35 * t35;
    double t37 = t36 * t36;
    double t38 = t8 * t8;
    double t39 = t38 * t38;
    double t40 = 0.1e1 / t39;
    double t41 = t37 * t40;
    double t42 = 0.1e1 / t8;
    double t43 = t35 * t42;
    double t44 = 0.1e1 + t43;
    double t45 = t44 <= ZETA_THRESHOLD_C_R2SCAN;
    double t46 = std::cbrt(ZETA_THRESHOLD_C_R2SCAN);
    double t47 = t46 * ZETA_THRESHOLD_C_R2SCAN;
    double t48 = std::cbrt(t44);
    double t49 = t48 * t44;
    double t50 = my_piecewise3(t45, t47, t49);
    double t51 = 0.1e1 - t43;
    double t52 = t51 <= ZETA_THRESHOLD_C_R2SCAN;
    double t53 = std::cbrt(t51);
    double t54 = t53 * t51;
    double t55 = my_piecewise3(t52, t47, t54);
    double t56 = t50 + t55 - 0.2e1;
    double t57 = M_CBRT2;
    double t58 = t57 - 0.1e1;
    double t60 = 0.1e1 / t58 / 0.2e1;
    double t61 = t56 * t60;
    double t63 = 0.1e1 + 0.5137e-1 * t12;
    double t65 = 0.1549425e1 * t12;
    double t66 = 0.420775e0 * t18;
    double t67 = 0.1562925e0 * t26;
    double t68 = 0.705945e1 * t15 + t65 + t66 + t67;
    double t71 = 0.1e1 + 0.32163958997385070134e2 / t68;
    double t72 = std::log(t71);
    double t76 = 0.1e1 + 0.278125e-1 * t12;
    double t78 = 0.905775e0 * t12;
    double t79 = 0.1100325e0 * t18;
    double t80 = 0.1241775e0 * t26;
    double t81 = 0.51785e1 * t15 + t78 + t79 + t80;
    double t84 = 0.1e1 + 0.29608749977793437516e2 / t81;
    double t85 = std::log(t84);
    double t86 = t76 * t85;
    double t88 = -0.310907e-1 * t63 * t72 + t34 - 0.19751673498613801407e-1 * t86;
    double t89 = t61 * t88;
    double t90 = t41 * t89;
    double t92 = 0.19751673498613801407e-1 * t61 * t86;
    double t93 = std::log(0.2e1);
    double t94 = 0.1e1 - t93;
    double t95 = M_PI * M_PI;
    double t98 = t46 * t46;
    double t99 = t48 * t48;
    double t100 = my_piecewise3(t45, t98, t99);
    double t101 = t53 * t53;
    double t102 = my_piecewise3(t52, t98, t101);
    double t104 = t100 / 0.2e1 + t102 / 0.2e1;
    double t105 = t104 * t104;
    double t106 = t105 * t104;
    double t108 = 0.1e1 / t94;
    double t109 = (-t34 + t90 + t92) * t108;
    double t110 = 0.1e1 / t106;
    double t111 = t95 * t110;
    double t113 = std::exp(-t109 * t111);
    double t114 = t113 - 0.1e1;
    double t122 = sigma0 + 0.2e1 * sigma1 + sigma2;
    double t133 = 0.1e1 / t114;
    double t138 = t98 * ZETA_THRESHOLD_C_R2SCAN;
    double t139 = t99 * t44;
    double t140 = my_piecewise3(t45, t138, t139);
    double t141 = t101 * t51;
    double t142 = my_piecewise3(t52, t138, t141);
    double t144 = t140 / 0.2e1 + t142 / 0.2e1;
    double t146 = t108 / t144;
    double t147 = t110 * t133;
    double t162 = t39 * t39;
    double t255 = M_CBRT6;
    double t256 = std::cbrt(t95);
    double t265 = t255 * t255;
    double t267 = 0.1e1 / t256 / t95;
    double t268 = t265 * t267;
    double t269 = t122 * t122;
    double t270 = t57 * t269;
    double t271 = t39 * t8;
    double t273 = 0.1e1 / t9 / t271;
    double t277 = std::exp(-0.20444604078896369094e0 * t268 * t270 * t273);
    double t783 = t146 * t147;
    double t784 = t269 * t122;
    double t785 = t252 * t784;
    double t786 = t162 * t8;
    double t787 = 0.1e1 / t786;
    double t788 = t787 * t277;
    double t791 = 0.58218257753910989057e-2 * t783 * t785 * t788;
    return t791;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho1__t940(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t23 = t9 * t9;
    double t35 = rho0 - rho1;
    double t38 = t8 * t8;
    double t39 = t38 * t38;
    double t42 = 0.1e1 / t8;
    double t43 = t35 * t42;
    double t44 = 0.1e1 + t43;
    double t45 = t44 <= ZETA_THRESHOLD_C_R2SCAN;
    double t48 = std::cbrt(t44);
    double t51 = 0.1e1 - t43;
    double t52 = t51 <= ZETA_THRESHOLD_C_R2SCAN;
    double t53 = std::cbrt(t51);
    double t57 = M_CBRT2;
    double t95 = M_PI * M_PI;
    double t99 = t48 * t48;
    double t101 = t53 * t53;
    double t122 = sigma0 + 0.2e1 * sigma1 + sigma2;
    double t255 = M_CBRT6;
    double t256 = std::cbrt(t95);
    double t257 = t256 * t256;
    double t258 = 0.1e1 / t257;
    double t260 = t57 * t57;
    double t263 = 0.1e1 / t23 / t38;
    double t265 = t255 * t255;
    double t267 = 0.1e1 / t256 / t95;
    double t268 = t265 * t267;
    double t269 = t122 * t122;
    double t270 = t57 * t269;
    double t271 = t39 * t8;
    double t273 = 0.1e1 / t9 / t271;
    double t277 = std::exp(-0.20444604078896369094e0 * t268 * t270 * t273);
    double t413 = 0.1e1 / t38;
    double t414 = t35 * t413;
    double t550 = t258 * t260;
    double t551 = t550 * t122;
    double t552 = t263 * t277;
    double t889 = -t42 - t414;
    double t893 = -t889;
    double t933 = my_piecewise3(t45, 0, 0.5e1 / 0.3e1 * t99 * t889);
    double t936 = my_piecewise3(t52, 0, 0.5e1 / 0.3e1 * t101 * t893);
    double t938 = t933 / 0.2e1 + t936 / 0.2e1;
    double t939 = t552 * t938;
    double t940 = t551 * t939;
    return t940;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho1__t594(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t15 = std::sqrt(t12);
    double t35 = rho0 - rho1;
    double t36 = t35 * t35;
    double t37 = t36 * t36;
    double t38 = t8 * t8;
    double t39 = t38 * t38;
    double t42 = 0.1e1 / t8;
    double t43 = t35 * t42;
    double t44 = 0.1e1 + t43;
    double t45 = t44 <= ZETA_THRESHOLD_C_R2SCAN;
    double t46 = std::cbrt(ZETA_THRESHOLD_C_R2SCAN);
    double t47 = t46 * ZETA_THRESHOLD_C_R2SCAN;
    double t48 = std::cbrt(t44);
    double t49 = t48 * t44;
    double t50 = my_piecewise3(t45, t47, t49);
    double t51 = 0.1e1 - t43;
    double t52 = t51 <= ZETA_THRESHOLD_C_R2SCAN;
    double t53 = std::cbrt(t51);
    double t54 = t53 * t51;
    double t55 = my_piecewise3(t52, t47, t54);
    double t56 = t50 + t55 - 0.2e1;
    double t57 = M_CBRT2;
    double t58 = t57 - 0.1e1;
    double t60 = 0.1e1 / t58 / 0.2e1;
    double t148 = std::sqrt(0.4e1);
    double t149 = t148 * t15;
    double t151 = 0.3138525e-1 * t12;
    double t152 = 0.1e1 + 0.22225e-1 * t149 + t151;
    double t153 = t152 * t152;
    double t158 = 0.1e1 - 0.2363e1 * t58 * t56 * t60;
    double t160 = t37 * t37;
    double t161 = t160 * t37;
    double t162 = t39 * t39;
    double t163 = t162 * t39;
    double t164 = 0.1e1 / t163;
    double t166 = -t161 * t164 + 0.1e1;
    double t167 = 0.1e1 / t15;
    double t168 = t148 * t167;
    double t170 = 0.4445e-1 * t168 + 0.125541e0;
    double t171 = t166 * t170;
    double t377 = 0.1e1 / t9 / t8;
    double t378 = t7 * t377;
    double t390 = t5 * t378;
    double t583 = 0.1e1 / t153 / t152;
    double t584 = t583 * t158;
    double t585 = std::sqrt(std::cbrt(0.4e1));
    double t586 = t585 * t167;
    double t587 = t5 * t377;
    double t588 = t586 * t587;
    double t590 = 0.1046175e-1 * t390;
    double t591 = -0.14816666666666666667e-1 * t588 - t590;
    double t594 = 0.571528e-1 * t584 * t171 * t591;
    return t594;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho1__t619(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t15 = std::sqrt(t12);
    double t35 = rho0 - rho1;
    double t36 = t35 * t35;
    double t37 = t36 * t36;
    double t38 = t8 * t8;
    double t39 = t38 * t38;
    double t42 = 0.1e1 / t8;
    double t43 = t35 * t42;
    double t44 = 0.1e1 + t43;
    double t45 = t44 <= ZETA_THRESHOLD_C_R2SCAN;
    double t46 = std::cbrt(ZETA_THRESHOLD_C_R2SCAN);
    double t47 = t46 * ZETA_THRESHOLD_C_R2SCAN;
    double t48 = std::cbrt(t44);
    double t49 = t48 * t44;
    double t50 = my_piecewise3(t45, t47, t49);
    double t51 = 0.1e1 - t43;
    double t52 = t51 <= ZETA_THRESHOLD_C_R2SCAN;
    double t53 = std::cbrt(t51);
    double t54 = t53 * t51;
    double t55 = my_piecewise3(t52, t47, t54);
    double t56 = t50 + t55 - 0.2e1;
    double t57 = M_CBRT2;
    double t58 = t57 - 0.1e1;
    double t60 = 0.1e1 / t58 / 0.2e1;
    double t148 = std::sqrt(0.4e1);
    double t149 = t148 * t15;
    double t151 = 0.3138525e-1 * t12;
    double t152 = 0.1e1 + 0.22225e-1 * t149 + t151;
    double t153 = t152 * t152;
    double t154 = 0.1e1 / t153;
    double t158 = 0.1e1 - 0.2363e1 * t58 * t56 * t60;
    double t159 = t154 * t158;
    double t160 = t37 * t37;
    double t161 = t160 * t37;
    double t162 = t39 * t39;
    double t163 = t162 * t39;
    double t164 = 0.1e1 / t163;
    double t166 = -t161 * t164 + 0.1e1;
    double t377 = 0.1e1 / t9 / t8;
    double t585 = std::sqrt(std::cbrt(0.4e1));
    double t611 = t166 * t585;
    double t612 = t159 * t611;
    double t614 = 0.1e1 / t15 / t12;
    double t615 = t614 * t2;
    double t616 = t4 * t377;
    double t617 = t615 * t616;
    double t619 = 0.84681398666666666666e-3 * t612 * t617;
    return t619;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho1__t624(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t15 = std::sqrt(t12);
    double t17 = 0.8969e0 * t12;
    double t18 = t12 * std::sqrt(t12);
    double t19 = 0.204775e0 * t18;
    double t20 = t2 * t2;
    double t21 = t4 * t4;
    double t22 = t20 * t21;
    double t23 = t9 * t9;
    double t26 = t22 * t6 / t23;
    double t27 = 0.123235e0 * t26;
    double t148 = std::sqrt(0.4e1);
    double t149 = t148 * t15;
    double t167 = 0.1e1 / t15;
    double t175 = 0.1898925e1 * t149 + t17 + t19 + t27;
    double t178 = 0.1e1 + 0.16081979498692535067e2 / t175;
    double t181 = t175 * t175;
    double t182 = 0.1e1 / t181;
    double t185 = std::sqrt(t12);
    double t189 = 0.1e1 / t178;
    double t377 = 0.1e1 / t9 / t8;
    double t378 = t7 * t377;
    double t386 = t4 * t7;
    double t387 = t386 * t377;
    double t390 = t5 * t378;
    double t391 = 0.29896666666666666667e0 * t390;
    double t392 = t185 * t2;
    double t393 = t392 * t387;
    double t394 = 0.1023875e0 * t393;
    double t398 = t22 * t6 / t23 / t8;
    double t399 = 0.82156666666666666667e-1 * t398;
    double t585 = std::sqrt(std::cbrt(0.4e1));
    double t586 = t585 * t167;
    double t587 = t5 * t377;
    double t588 = t586 * t587;
    double t621 = -0.126595e1 * t588 - t391 - t394 - t399;
    double t624 = 0.2137e0 * t182 * t621 * t189;
    return t624;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho1__t635(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t14 = 0.1e1 + 0.53425e-1 * t12;
    double t15 = std::sqrt(t12);
    double t17 = 0.8969e0 * t12;
    double t18 = t12 * std::sqrt(t12);
    double t19 = 0.204775e0 * t18;
    double t20 = t2 * t2;
    double t21 = t4 * t4;
    double t22 = t20 * t21;
    double t23 = t9 * t9;
    double t26 = t22 * t6 / t23;
    double t27 = 0.123235e0 * t26;
    double t148 = std::sqrt(0.4e1);
    double t149 = t148 * t15;
    double t167 = 0.1e1 / t15;
    double t168 = t148 * t167;
    double t175 = 0.1898925e1 * t149 + t17 + t19 + t27;
    double t178 = 0.1e1 + 0.16081979498692535067e2 / t175;
    double t181 = t175 * t175;
    double t185 = std::sqrt(t12);
    double t188 = 0.379785e1 * t168 + 0.35876e1 + 0.122865e1 * t185 + 0.24647e0 * t12;
    double t189 = 0.1e1 / t178;
    double t190 = t188 * t189;
    double t377 = 0.1e1 / t9 / t8;
    double t378 = t7 * t377;
    double t386 = t4 * t7;
    double t387 = t386 * t377;
    double t390 = t5 * t378;
    double t391 = 0.29896666666666666667e0 * t390;
    double t392 = t185 * t2;
    double t393 = t392 * t387;
    double t394 = 0.1023875e0 * t393;
    double t398 = t22 * t6 / t23 / t8;
    double t399 = 0.82156666666666666667e-1 * t398;
    double t585 = std::sqrt(std::cbrt(0.4e1));
    double t586 = t585 * t167;
    double t587 = t5 * t377;
    double t588 = t586 * t587;
    double t621 = -0.126595e1 * t588 - t391 - t394 - t399;
    double t630 = t181 * t175;
    double t631 = 0.1e1 / t630;
    double t632 = t14 * t631;
    double t635 = 0.2e1 * t632 * t190 * t621;
    return t635;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho1__t647(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t14 = 0.1e1 + 0.53425e-1 * t12;
    double t15 = std::sqrt(t12);
    double t17 = 0.8969e0 * t12;
    double t18 = t12 * std::sqrt(t12);
    double t19 = 0.204775e0 * t18;
    double t20 = t2 * t2;
    double t21 = t4 * t4;
    double t22 = t20 * t21;
    double t23 = t9 * t9;
    double t26 = t22 * t6 / t23;
    double t27 = 0.123235e0 * t26;
    double t148 = std::sqrt(0.4e1);
    double t149 = t148 * t15;
    double t175 = 0.1898925e1 * t149 + t17 + t19 + t27;
    double t178 = 0.1e1 + 0.16081979498692535067e2 / t175;
    double t181 = t175 * t175;
    double t182 = 0.1e1 / t181;
    double t183 = t14 * t182;
    double t189 = 0.1e1 / t178;
    double t377 = 0.1e1 / t9 / t8;
    double t378 = t7 * t377;
    double t386 = t4 * t7;
    double t387 = t386 * t377;
    double t390 = t5 * t378;
    double t585 = std::sqrt(std::cbrt(0.4e1));
    double t587 = t5 * t377;
    double t614 = 0.1e1 / t15 / t12;
    double t636 = t585 * t614;
    double t637 = t636 * t587;
    double t639 = 0.1e1 / std::sqrt(t12);
    double t640 = t639 * t2;
    double t641 = t640 * t387;
    double t644 = 0.25319e1 * t637 - 0.204775e0 * t641 - 0.82156666666666666667e-1 * t390;
    double t645 = t644 * t189;
    double t647 = 0.1e1 * t183 * t645;
    return t647;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho1__t656(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t14 = 0.1e1 + 0.53425e-1 * t12;
    double t15 = std::sqrt(t12);
    double t17 = 0.8969e0 * t12;
    double t18 = t12 * std::sqrt(t12);
    double t19 = 0.204775e0 * t18;
    double t20 = t2 * t2;
    double t21 = t4 * t4;
    double t22 = t20 * t21;
    double t23 = t9 * t9;
    double t26 = t22 * t6 / t23;
    double t27 = 0.123235e0 * t26;
    double t148 = std::sqrt(0.4e1);
    double t149 = t148 * t15;
    double t167 = 0.1e1 / t15;
    double t168 = t148 * t167;
    double t175 = 0.1898925e1 * t149 + t17 + t19 + t27;
    double t178 = 0.1e1 + 0.16081979498692535067e2 / t175;
    double t181 = t175 * t175;
    double t185 = std::sqrt(t12);
    double t188 = 0.379785e1 * t168 + 0.35876e1 + 0.122865e1 * t185 + 0.24647e0 * t12;
    double t377 = 0.1e1 / t9 / t8;
    double t378 = t7 * t377;
    double t386 = t4 * t7;
    double t387 = t386 * t377;
    double t390 = t5 * t378;
    double t391 = 0.29896666666666666667e0 * t390;
    double t392 = t185 * t2;
    double t393 = t392 * t387;
    double t394 = 0.1023875e0 * t393;
    double t398 = t22 * t6 / t23 / t8;
    double t399 = 0.82156666666666666667e-1 * t398;
    double t585 = std::sqrt(std::cbrt(0.4e1));
    double t586 = t585 * t167;
    double t587 = t5 * t377;
    double t588 = t586 * t587;
    double t621 = -0.126595e1 * t588 - t391 - t394 - t399;
    double t648 = t181 * t181;
    double t649 = 0.1e1 / t648;
    double t650 = t14 * t649;
    double t651 = t178 * t178;
    double t652 = 0.1e1 / t651;
    double t653 = t188 * t652;
    double t656 = 0.16081979498692535067e2 * t650 * t653 * t621;
    return t656;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho1__t658(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t14 = 0.1e1 + 0.53425e-1 * t12;
    double t15 = std::sqrt(t12);
    double t17 = 0.8969e0 * t12;
    double t18 = t12 * std::sqrt(t12);
    double t19 = 0.204775e0 * t18;
    double t20 = t2 * t2;
    double t21 = t4 * t4;
    double t22 = t20 * t21;
    double t23 = t9 * t9;
    double t26 = t22 * t6 / t23;
    double t27 = 0.123235e0 * t26;
    double t35 = rho0 - rho1;
    double t36 = t35 * t35;
    double t38 = t8 * t8;
    double t39 = t38 * t38;
    double t40 = 0.1e1 / t39;
    double t42 = 0.1e1 / t8;
    double t43 = t35 * t42;
    double t44 = 0.1e1 + t43;
    double t45 = t44 <= ZETA_THRESHOLD_C_R2SCAN;
    double t46 = std::cbrt(ZETA_THRESHOLD_C_R2SCAN);
    double t47 = t46 * ZETA_THRESHOLD_C_R2SCAN;
    double t48 = std::cbrt(t44);
    double t49 = t48 * t44;
    double t50 = my_piecewise3(t45, t47, t49);
    double t51 = 0.1e1 - t43;
    double t52 = t51 <= ZETA_THRESHOLD_C_R2SCAN;
    double t53 = std::cbrt(t51);
    double t54 = t53 * t51;
    double t55 = my_piecewise3(t52, t47, t54);
    double t56 = t50 + t55 - 0.2e1;
    double t57 = M_CBRT2;
    double t58 = t57 - 0.1e1;
    double t60 = 0.1e1 / t58 / 0.2e1;
    double t61 = t56 * t60;
    double t63 = 0.1e1 + 0.5137e-1 * t12;
    double t65 = 0.1549425e1 * t12;
    double t66 = 0.420775e0 * t18;
    double t67 = 0.1562925e0 * t26;
    double t76 = 0.1e1 + 0.278125e-1 * t12;
    double t78 = 0.905775e0 * t12;
    double t79 = 0.1100325e0 * t18;
    double t80 = 0.1241775e0 * t26;
    double t148 = std::sqrt(0.4e1);
    double t149 = t148 * t15;
    double t167 = 0.1e1 / t15;
    double t168 = t148 * t167;
    double t175 = 0.1898925e1 * t149 + t17 + t19 + t27;
    double t178 = 0.1e1 + 0.16081979498692535067e2 / t175;
    double t179 = std::log(t178);
    double t180 = 0.1328816518e-1 * t179;
    double t181 = t175 * t175;
    double t182 = 0.1e1 / t181;
    double t183 = t14 * t182;
    double t185 = std::sqrt(t12);
    double t188 = 0.379785e1 * t168 + 0.35876e1 + 0.122865e1 * t185 + 0.24647e0 * t12;
    double t189 = 0.1e1 / t178;
    double t190 = t188 * t189;
    double t192 = 0.1e1 * t183 * t190;
    double t194 = 0.3529725e1 * t149 + t65 + t66 + t67;
    double t197 = 0.1e1 + 0.32163958997385070134e2 / t194;
    double t198 = std::log(t197);
    double t200 = t194 * t194;
    double t201 = 0.1e1 / t200;
    double t202 = t63 * t201;
    double t206 = 0.705945e1 * t168 + 0.61977e1 + 0.252465e1 * t185 + 0.312585e0 * t12;
    double t207 = 0.1e1 / t197;
    double t208 = t206 * t207;
    double t212 = 0.258925e1 * t149 + t78 + t79 + t80;
    double t215 = 0.1e1 + 0.29608749977793437516e2 / t212;
    double t216 = std::log(t215);
    double t218 = t212 * t212;
    double t219 = 0.1e1 / t218;
    double t220 = t76 * t219;
    double t224 = 0.51785e1 * t168 + 0.36231e1 + 0.660195e0 * t185 + 0.248355e0 * t12;
    double t225 = 0.1e1 / t215;
    double t226 = t224 * t225;
    double t229 = -0.6388517036e-2 * t198 + 0.1e1 * t202 * t208 + t180 - t192 - 0.21973736767207854065e-2 * t216 + 0.5848223622634646207e0 * t220 * t226;
    double t230 = t61 * t229;
    double t405 = t36 * t35;
    double t406 = t405 * t40;
    double t658 = 0.4e1 * t406 * t230;
    return t658;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho1__t660(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t14 = 0.1e1 + 0.53425e-1 * t12;
    double t15 = std::sqrt(t12);
    double t17 = 0.8969e0 * t12;
    double t18 = t12 * std::sqrt(t12);
    double t19 = 0.204775e0 * t18;
    double t20 = t2 * t2;
    double t21 = t4 * t4;
    double t22 = t20 * t21;
    double t23 = t9 * t9;
    double t26 = t22 * t6 / t23;
    double t27 = 0.123235e0 * t26;
    double t35 = rho0 - rho1;
    double t36 = t35 * t35;
    double t37 = t36 * t36;
    double t38 = t8 * t8;
    double t39 = t38 * t38;
    double t42 = 0.1e1 / t8;
    double t43 = t35 * t42;
    double t44 = 0.1e1 + t43;
    double t45 = t44 <= ZETA_THRESHOLD_C_R2SCAN;
    double t46 = std::cbrt(ZETA_THRESHOLD_C_R2SCAN);
    double t47 = t46 * ZETA_THRESHOLD_C_R2SCAN;
    double t48 = std::cbrt(t44);
    double t49 = t48 * t44;
    double t50 = my_piecewise3(t45, t47, t49);
    double t51 = 0.1e1 - t43;
    double t52 = t51 <= ZETA_THRESHOLD_C_R2SCAN;
    double t53 = std::cbrt(t51);
    double t54 = t53 * t51;
    double t55 = my_piecewise3(t52, t47, t54);
    double t56 = t50 + t55 - 0.2e1;
    double t57 = M_CBRT2;
    double t58 = t57 - 0.1e1;
    double t60 = 0.1e1 / t58 / 0.2e1;
    double t61 = t56 * t60;
    double t63 = 0.1e1 + 0.5137e-1 * t12;
    double t65 = 0.1549425e1 * t12;
    double t66 = 0.420775e0 * t18;
    double t67 = 0.1562925e0 * t26;
    double t76 = 0.1e1 + 0.278125e-1 * t12;
    double t78 = 0.905775e0 * t12;
    double t79 = 0.1100325e0 * t18;
    double t80 = 0.1241775e0 * t26;
    double t148 = std::sqrt(0.4e1);
    double t149 = t148 * t15;
    double t167 = 0.1e1 / t15;
    double t168 = t148 * t167;
    double t175 = 0.1898925e1 * t149 + t17 + t19 + t27;
    double t178 = 0.1e1 + 0.16081979498692535067e2 / t175;
    double t179 = std::log(t178);
    double t180 = 0.1328816518e-1 * t179;
    double t181 = t175 * t175;
    double t182 = 0.1e1 / t181;
    double t183 = t14 * t182;
    double t185 = std::sqrt(t12);
    double t188 = 0.379785e1 * t168 + 0.35876e1 + 0.122865e1 * t185 + 0.24647e0 * t12;
    double t189 = 0.1e1 / t178;
    double t190 = t188 * t189;
    double t192 = 0.1e1 * t183 * t190;
    double t194 = 0.3529725e1 * t149 + t65 + t66 + t67;
    double t197 = 0.1e1 + 0.32163958997385070134e2 / t194;
    double t198 = std::log(t197);
    double t200 = t194 * t194;
    double t201 = 0.1e1 / t200;
    double t202 = t63 * t201;
    double t206 = 0.705945e1 * t168 + 0.61977e1 + 0.252465e1 * t185 + 0.312585e0 * t12;
    double t207 = 0.1e1 / t197;
    double t208 = t206 * t207;
    double t212 = 0.258925e1 * t149 + t78 + t79 + t80;
    double t215 = 0.1e1 + 0.29608749977793437516e2 / t212;
    double t216 = std::log(t215);
    double t218 = t212 * t212;
    double t219 = 0.1e1 / t218;
    double t220 = t76 * t219;
    double t224 = 0.51785e1 * t168 + 0.36231e1 + 0.660195e0 * t185 + 0.248355e0 * t12;
    double t225 = 0.1e1 / t215;
    double t226 = t224 * t225;
    double t229 = -0.6388517036e-2 * t198 + 0.1e1 * t202 * t208 + t180 - t192 - 0.21973736767207854065e-2 * t216 + 0.5848223622634646207e0 * t220 * t226;
    double t230 = t61 * t229;
    double t271 = t39 * t8;
    double t409 = 0.1e1 / t271;
    double t410 = t37 * t409;
    double t660 = 0.4e1 * t410 * t230;
    return t660;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho1__t695(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t15 = std::sqrt(t12);
    double t20 = t2 * t2;
    double t21 = t4 * t4;
    double t22 = t20 * t21;
    double t23 = t9 * t9;
    double t167 = 0.1e1 / t15;
    double t185 = std::sqrt(t12);
    double t377 = 0.1e1 / t9 / t8;
    double t378 = t7 * t377;
    double t386 = t4 * t7;
    double t387 = t386 * t377;
    double t390 = t5 * t378;
    double t392 = t185 * t2;
    double t393 = t392 * t387;
    double t398 = t22 * t6 / t23 / t8;
    double t449 = 0.301925e0 * t390;
    double t450 = 0.5501625e-1 * t393;
    double t451 = 0.82785e-1 * t398;
    double t585 = std::sqrt(std::cbrt(0.4e1));
    double t586 = t585 * t167;
    double t587 = t5 * t377;
    double t588 = t586 * t587;
    double t695 = -0.17261666666666666667e1 * t588 - t449 - t450 - t451;
    return t695;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho1__t725(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t624 = mgga_c_r2scan_vrho1__t624(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t635 = mgga_c_r2scan_vrho1__t635(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t647 = mgga_c_r2scan_vrho1__t647(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t656 = mgga_c_r2scan_vrho1__t656(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t695 = mgga_c_r2scan_vrho1__t695(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t15 = std::sqrt(t12);
    double t17 = 0.8969e0 * t12;
    double t18 = t12 * std::sqrt(t12);
    double t19 = 0.204775e0 * t18;
    double t20 = t2 * t2;
    double t21 = t4 * t4;
    double t22 = t20 * t21;
    double t23 = t9 * t9;
    double t26 = t22 * t6 / t23;
    double t27 = 0.123235e0 * t26;
    double t63 = 0.1e1 + 0.5137e-1 * t12;
    double t65 = 0.1549425e1 * t12;
    double t66 = 0.420775e0 * t18;
    double t67 = 0.1562925e0 * t26;
    double t76 = 0.1e1 + 0.278125e-1 * t12;
    double t78 = 0.905775e0 * t12;
    double t79 = 0.1100325e0 * t18;
    double t80 = 0.1241775e0 * t26;
    double t148 = std::sqrt(0.4e1);
    double t149 = t148 * t15;
    double t167 = 0.1e1 / t15;
    double t168 = t148 * t167;
    double t175 = 0.1898925e1 * t149 + t17 + t19 + t27;
    double t178 = 0.1e1 + 0.16081979498692535067e2 / t175;
    double t181 = t175 * t175;
    double t182 = 0.1e1 / t181;
    double t185 = std::sqrt(t12);
    double t188 = 0.379785e1 * t168 + 0.35876e1 + 0.122865e1 * t185 + 0.24647e0 * t12;
    double t189 = 0.1e1 / t178;
    double t190 = t188 * t189;
    double t194 = 0.3529725e1 * t149 + t65 + t66 + t67;
    double t197 = 0.1e1 + 0.32163958997385070134e2 / t194;
    double t200 = t194 * t194;
    double t201 = 0.1e1 / t200;
    double t202 = t63 * t201;
    double t206 = 0.705945e1 * t168 + 0.61977e1 + 0.252465e1 * t185 + 0.312585e0 * t12;
    double t207 = 0.1e1 / t197;
    double t208 = t206 * t207;
    double t212 = 0.258925e1 * t149 + t78 + t79 + t80;
    double t215 = 0.1e1 + 0.29608749977793437516e2 / t212;
    double t218 = t212 * t212;
    double t219 = 0.1e1 / t218;
    double t220 = t76 * t219;
    double t224 = 0.51785e1 * t168 + 0.36231e1 + 0.660195e0 * t185 + 0.248355e0 * t12;
    double t225 = 0.1e1 / t215;
    double t226 = t224 * t225;
    double t377 = 0.1e1 / t9 / t8;
    double t378 = t7 * t377;
    double t386 = t4 * t7;
    double t387 = t386 * t377;
    double t390 = t5 * t378;
    double t392 = t185 * t2;
    double t393 = t392 * t387;
    double t398 = t22 * t6 / t23 / t8;
    double t434 = 0.516475e0 * t390;
    double t435 = 0.2103875e0 * t393;
    double t436 = 0.104195e0 * t398;
    double t585 = std::sqrt(std::cbrt(0.4e1));
    double t586 = t585 * t167;
    double t587 = t5 * t377;
    double t588 = t586 * t587;
    double t614 = 0.1e1 / t15 / t12;
    double t625 = t5 * t7;
    double t626 = t377 * t182;
    double t629 = 0.17808333333333333333e-1 * t625 * t626 * t190;
    double t636 = t585 * t614;
    double t637 = t636 * t587;
    double t639 = 0.1e1 / std::sqrt(t12);
    double t640 = t639 * t2;
    double t641 = t640 * t387;
    double t664 = -0.235315e1 * t588 - t434 - t435 - t436;
    double t668 = t377 * t201;
    double t672 = t200 * t194;
    double t673 = 0.1e1 / t672;
    double t674 = t63 * t673;
    double t681 = 0.47063e1 * t637 - 0.42077500000000000001e0 * t641 - 0.104195e0 * t390;
    double t682 = t681 * t207;
    double t685 = t200 * t200;
    double t686 = 0.1e1 / t685;
    double t687 = t63 * t686;
    double t688 = t197 * t197;
    double t689 = 0.1e1 / t688;
    double t690 = t206 * t689;
    double t697 = t219 * t695 * t225;
    double t699 = t377 * t219;
    double t703 = t218 * t212;
    double t704 = 0.1e1 / t703;
    double t705 = t76 * t704;
    double t706 = t226 * t695;
    double t712 = 0.34523333333333333333e1 * t637 - 0.1100325e0 * t641 - 0.82785e-1 * t390;
    double t713 = t712 * t225;
    double t716 = t218 * t218;
    double t717 = 0.1e1 / t716;
    double t718 = t76 * t717;
    double t719 = t215 * t215;
    double t720 = 0.1e1 / t719;
    double t721 = t224 * t720;
    double t722 = t721 * t695;
    double t725 = 0.20548e0 * t201 * t664 * t207 - 0.17123333333333333333e-1 * t625 * t668 * t208 - 0.2e1 * t674 * t208 * t664 + 0.1e1 * t202 * t682 + 0.32163958997385070134e2 * t687 * t690 * t664 - t624 + t629 + t635 - t647 - t656 + 0.65061487801810439052e-1 * t697 - 0.54217906501508699211e-2 * t625 * t699 * t226 - 0.11696447245269292414e1 * t705 * t706 + 0.5848223622634646207e0 * t220 * t713 + 0.17315859105681463759e2 * t718 * t722;
    return t725;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho1__t727(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t725 = mgga_c_r2scan_vrho1__t725(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t8 = rho0 + rho1;
    double t35 = rho0 - rho1;
    double t36 = t35 * t35;
    double t37 = t36 * t36;
    double t38 = t8 * t8;
    double t39 = t38 * t38;
    double t40 = 0.1e1 / t39;
    double t41 = t37 * t40;
    double t42 = 0.1e1 / t8;
    double t43 = t35 * t42;
    double t44 = 0.1e1 + t43;
    double t45 = t44 <= ZETA_THRESHOLD_C_R2SCAN;
    double t46 = std::cbrt(ZETA_THRESHOLD_C_R2SCAN);
    double t47 = t46 * ZETA_THRESHOLD_C_R2SCAN;
    double t48 = std::cbrt(t44);
    double t49 = t48 * t44;
    double t50 = my_piecewise3(t45, t47, t49);
    double t51 = 0.1e1 - t43;
    double t52 = t51 <= ZETA_THRESHOLD_C_R2SCAN;
    double t53 = std::cbrt(t51);
    double t54 = t53 * t51;
    double t55 = my_piecewise3(t52, t47, t54);
    double t56 = t50 + t55 - 0.2e1;
    double t57 = M_CBRT2;
    double t58 = t57 - 0.1e1;
    double t60 = 0.1e1 / t58 / 0.2e1;
    double t61 = t56 * t60;
    double t726 = t61 * t725;
    double t727 = t41 * t726;
    return t727;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho1__t747(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t15 = std::sqrt(t12);
    double t18 = t12 * std::sqrt(t12);
    double t20 = t2 * t2;
    double t21 = t4 * t4;
    double t22 = t20 * t21;
    double t23 = t9 * t9;
    double t26 = t22 * t6 / t23;
    double t35 = rho0 - rho1;
    double t42 = 0.1e1 / t8;
    double t43 = t35 * t42;
    double t44 = 0.1e1 + t43;
    double t45 = t44 <= ZETA_THRESHOLD_C_R2SCAN;
    double t46 = std::cbrt(ZETA_THRESHOLD_C_R2SCAN);
    double t47 = t46 * ZETA_THRESHOLD_C_R2SCAN;
    double t48 = std::cbrt(t44);
    double t49 = t48 * t44;
    double t50 = my_piecewise3(t45, t47, t49);
    double t51 = 0.1e1 - t43;
    double t52 = t51 <= ZETA_THRESHOLD_C_R2SCAN;
    double t53 = std::cbrt(t51);
    double t54 = t53 * t51;
    double t55 = my_piecewise3(t52, t47, t54);
    double t56 = t50 + t55 - 0.2e1;
    double t57 = M_CBRT2;
    double t58 = t57 - 0.1e1;
    double t60 = 0.1e1 / t58 / 0.2e1;
    double t61 = t56 * t60;
    double t76 = 0.1e1 + 0.278125e-1 * t12;
    double t78 = 0.905775e0 * t12;
    double t79 = 0.1100325e0 * t18;
    double t80 = 0.1241775e0 * t26;
    double t148 = std::sqrt(0.4e1);
    double t149 = t148 * t15;
    double t212 = 0.258925e1 * t149 + t78 + t79 + t80;
    double t215 = 0.1e1 + 0.29608749977793437516e2 / t212;
    double t218 = t212 * t212;
    double t219 = 0.1e1 / t218;
    double t225 = 0.1e1 / t215;
    double t234 = t61 * t76;
    double t377 = 0.1e1 / t9 / t8;
    double t378 = t7 * t377;
    double t386 = t4 * t7;
    double t387 = t386 * t377;
    double t390 = t5 * t378;
    double t585 = std::sqrt(std::cbrt(0.4e1));
    double t587 = t5 * t377;
    double t614 = 0.1e1 / t15 / t12;
    double t636 = t585 * t614;
    double t637 = t636 * t587;
    double t639 = 0.1e1 / std::sqrt(t12);
    double t640 = t639 * t2;
    double t641 = t640 * t387;
    double t712 = 0.34523333333333333333e1 * t637 - 0.1100325e0 * t641 - 0.82785e-1 * t390;
    double t745 = t219 * t712 * t225;
    double t747 = 0.5848223622634646207e0 * t234 * t745;
    return t747;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho1__t951(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t15 = std::sqrt(t12);
    double t35 = rho0 - rho1;
    double t38 = t8 * t8;
    double t42 = 0.1e1 / t8;
    double t43 = t35 * t42;
    double t44 = 0.1e1 + t43;
    double t45 = t44 <= ZETA_THRESHOLD_C_R2SCAN;
    double t48 = std::cbrt(t44);
    double t51 = 0.1e1 - t43;
    double t52 = t51 <= ZETA_THRESHOLD_C_R2SCAN;
    double t53 = std::cbrt(t51);
    double t57 = M_CBRT2;
    double t58 = t57 - 0.1e1;
    double t148 = std::sqrt(0.4e1);
    double t149 = t148 * t15;
    double t151 = 0.3138525e-1 * t12;
    double t152 = 0.1e1 + 0.22225e-1 * t149 + t151;
    double t153 = t152 * t152;
    double t154 = 0.1e1 / t153;
    double t413 = 0.1e1 / t38;
    double t414 = t35 * t413;
    double t595 = t154 * t58;
    double t889 = -t42 - t414;
    double t892 = my_piecewise3(t45, 0, 0.4e1 / 0.3e1 * t48 * t889);
    double t893 = -t889;
    double t896 = my_piecewise3(t52, 0, 0.4e1 / 0.3e1 * t53 * t893);
    double t897 = t892 + t896;
    double t951 = t595 * t897;
    return t951;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho1__t956(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t955 = mgga_c_r2scan_vrho1__t955(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t15 = std::sqrt(t12);
    double t148 = std::sqrt(0.4e1);
    double t167 = 0.1e1 / t15;
    double t168 = t148 * t167;
    double t170 = 0.4445e-1 * t168 + 0.125541e0;
    double t956 = t955 * t170;
    return t956;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho1__t959(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t898 = mgga_c_r2scan_vrho1__t898(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t14 = 0.1e1 + 0.53425e-1 * t12;
    double t15 = std::sqrt(t12);
    double t17 = 0.8969e0 * t12;
    double t18 = t12 * std::sqrt(t12);
    double t19 = 0.204775e0 * t18;
    double t20 = t2 * t2;
    double t21 = t4 * t4;
    double t22 = t20 * t21;
    double t23 = t9 * t9;
    double t26 = t22 * t6 / t23;
    double t27 = 0.123235e0 * t26;
    double t63 = 0.1e1 + 0.5137e-1 * t12;
    double t65 = 0.1549425e1 * t12;
    double t66 = 0.420775e0 * t18;
    double t67 = 0.1562925e0 * t26;
    double t76 = 0.1e1 + 0.278125e-1 * t12;
    double t78 = 0.905775e0 * t12;
    double t79 = 0.1100325e0 * t18;
    double t80 = 0.1241775e0 * t26;
    double t148 = std::sqrt(0.4e1);
    double t149 = t148 * t15;
    double t167 = 0.1e1 / t15;
    double t168 = t148 * t167;
    double t175 = 0.1898925e1 * t149 + t17 + t19 + t27;
    double t178 = 0.1e1 + 0.16081979498692535067e2 / t175;
    double t179 = std::log(t178);
    double t180 = 0.1328816518e-1 * t179;
    double t181 = t175 * t175;
    double t182 = 0.1e1 / t181;
    double t183 = t14 * t182;
    double t185 = std::sqrt(t12);
    double t188 = 0.379785e1 * t168 + 0.35876e1 + 0.122865e1 * t185 + 0.24647e0 * t12;
    double t189 = 0.1e1 / t178;
    double t190 = t188 * t189;
    double t192 = 0.1e1 * t183 * t190;
    double t194 = 0.3529725e1 * t149 + t65 + t66 + t67;
    double t197 = 0.1e1 + 0.32163958997385070134e2 / t194;
    double t198 = std::log(t197);
    double t200 = t194 * t194;
    double t201 = 0.1e1 / t200;
    double t202 = t63 * t201;
    double t206 = 0.705945e1 * t168 + 0.61977e1 + 0.252465e1 * t185 + 0.312585e0 * t12;
    double t207 = 0.1e1 / t197;
    double t208 = t206 * t207;
    double t212 = 0.258925e1 * t149 + t78 + t79 + t80;
    double t215 = 0.1e1 + 0.29608749977793437516e2 / t212;
    double t216 = std::log(t215);
    double t218 = t212 * t212;
    double t219 = 0.1e1 / t218;
    double t220 = t76 * t219;
    double t224 = 0.51785e1 * t168 + 0.36231e1 + 0.660195e0 * t185 + 0.248355e0 * t12;
    double t225 = 0.1e1 / t215;
    double t226 = t224 * t225;
    double t229 = -0.6388517036e-2 * t198 + 0.1e1 * t202 * t208 + t180 - t192 - 0.21973736767207854065e-2 * t216 + 0.5848223622634646207e0 * t220 * t226;
    double t959 = t898 * t229;
    return t959;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho1__t966(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t594 = mgga_c_r2scan_vrho1__t594(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t619 = mgga_c_r2scan_vrho1__t619(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t624 = mgga_c_r2scan_vrho1__t624(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t635 = mgga_c_r2scan_vrho1__t635(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t647 = mgga_c_r2scan_vrho1__t647(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t656 = mgga_c_r2scan_vrho1__t656(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t658 = mgga_c_r2scan_vrho1__t658(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t660 = mgga_c_r2scan_vrho1__t660(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t695 = mgga_c_r2scan_vrho1__t695(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t727 = mgga_c_r2scan_vrho1__t727(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t747 = mgga_c_r2scan_vrho1__t747(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t898 = mgga_c_r2scan_vrho1__t898(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t951 = mgga_c_r2scan_vrho1__t951(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t956 = mgga_c_r2scan_vrho1__t956(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t959 = mgga_c_r2scan_vrho1__t959(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t15 = std::sqrt(t12);
    double t17 = 0.8969e0 * t12;
    double t18 = t12 * std::sqrt(t12);
    double t19 = 0.204775e0 * t18;
    double t20 = t2 * t2;
    double t21 = t4 * t4;
    double t22 = t20 * t21;
    double t23 = t9 * t9;
    double t26 = t22 * t6 / t23;
    double t27 = 0.123235e0 * t26;
    double t35 = rho0 - rho1;
    double t36 = t35 * t35;
    double t37 = t36 * t36;
    double t38 = t8 * t8;
    double t39 = t38 * t38;
    double t40 = 0.1e1 / t39;
    double t41 = t37 * t40;
    double t42 = 0.1e1 / t8;
    double t43 = t35 * t42;
    double t44 = 0.1e1 + t43;
    double t45 = t44 <= ZETA_THRESHOLD_C_R2SCAN;
    double t46 = std::cbrt(ZETA_THRESHOLD_C_R2SCAN);
    double t47 = t46 * ZETA_THRESHOLD_C_R2SCAN;
    double t48 = std::cbrt(t44);
    double t49 = t48 * t44;
    double t50 = my_piecewise3(t45, t47, t49);
    double t51 = 0.1e1 - t43;
    double t52 = t51 <= ZETA_THRESHOLD_C_R2SCAN;
    double t53 = std::cbrt(t51);
    double t54 = t53 * t51;
    double t55 = my_piecewise3(t52, t47, t54);
    double t56 = t50 + t55 - 0.2e1;
    double t57 = M_CBRT2;
    double t58 = t57 - 0.1e1;
    double t60 = 0.1e1 / t58 / 0.2e1;
    double t61 = t56 * t60;
    double t76 = 0.1e1 + 0.278125e-1 * t12;
    double t78 = 0.905775e0 * t12;
    double t79 = 0.1100325e0 * t18;
    double t80 = 0.1241775e0 * t26;
    double t148 = std::sqrt(0.4e1);
    double t149 = t148 * t15;
    double t151 = 0.3138525e-1 * t12;
    double t152 = 0.1e1 + 0.22225e-1 * t149 + t151;
    double t153 = t152 * t152;
    double t154 = 0.1e1 / t153;
    double t158 = 0.1e1 - 0.2363e1 * t58 * t56 * t60;
    double t159 = t154 * t158;
    double t160 = t37 * t37;
    double t161 = t160 * t37;
    double t162 = t39 * t39;
    double t163 = t162 * t39;
    double t164 = 0.1e1 / t163;
    double t166 = -t161 * t164 + 0.1e1;
    double t167 = 0.1e1 / t15;
    double t168 = t148 * t167;
    double t170 = 0.4445e-1 * t168 + 0.125541e0;
    double t175 = 0.1898925e1 * t149 + t17 + t19 + t27;
    double t178 = 0.1e1 + 0.16081979498692535067e2 / t175;
    double t181 = t175 * t175;
    double t182 = 0.1e1 / t181;
    double t185 = std::sqrt(t12);
    double t188 = 0.379785e1 * t168 + 0.35876e1 + 0.122865e1 * t185 + 0.24647e0 * t12;
    double t189 = 0.1e1 / t178;
    double t190 = t188 * t189;
    double t212 = 0.258925e1 * t149 + t78 + t79 + t80;
    double t215 = 0.1e1 + 0.29608749977793437516e2 / t212;
    double t216 = std::log(t215);
    double t218 = t212 * t212;
    double t219 = 0.1e1 / t218;
    double t224 = 0.51785e1 * t168 + 0.36231e1 + 0.660195e0 * t185 + 0.248355e0 * t12;
    double t225 = 0.1e1 / t215;
    double t234 = t61 * t76;
    double t236 = t219 * t224 * t225;
    double t377 = 0.1e1 / t9 / t8;
    double t378 = t7 * t377;
    double t597 = t60 * t166;
    double t598 = t597 * t170;
    double t625 = t5 * t7;
    double t626 = t377 * t182;
    double t629 = 0.17808333333333333333e-1 * t625 * t626 * t190;
    double t697 = t219 * t695 * t225;
    double t703 = t218 * t212;
    double t704 = 0.1e1 / t703;
    double t716 = t218 * t218;
    double t717 = 0.1e1 / t716;
    double t719 = t215 * t215;
    double t720 = 0.1e1 / t719;
    double t731 = 0.65061487801810439052e-1 * t61 * t697;
    double t735 = t61 * t5;
    double t736 = t378 * t236;
    double t738 = 0.54217906501508699211e-2 * t735 * t736;
    double t739 = t704 * t224;
    double t740 = t225 * t695;
    double t741 = t739 * t740;
    double t743 = 0.11696447245269292414e1 * t234 * t741;
    double t748 = t717 * t224;
    double t749 = t720 * t695;
    double t750 = t748 * t749;
    double t752 = 0.17315859105681463759e2 * t234 * t750;
    double t963 = t898 * t76;
    double t966 = -t594 - 0.675260332e-1 * t951 * t598 + 0.285764e-1 * t159 * t956 + t619 - t624 + t629 + t635 - t647 - t656 + t658 + t660 - t41 * t959 - t727 - 0.21973736767207854065e-2 * t898 * t216 + t731 + 0.5848223622634646207e0 * t963 * t236 - t738 - t743 + t747 + t752;
    return t966;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho1__t975(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t404 = mgga_c_r2scan_vrho1__t404(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t408 = mgga_c_r2scan_vrho1__t408(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t459 = mgga_c_r2scan_vrho1__t459(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t898 = mgga_c_r2scan_vrho1__t898(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t955 = mgga_c_r2scan_vrho1__t955(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t14 = 0.1e1 + 0.53425e-1 * t12;
    double t15 = std::sqrt(t12);
    double t17 = 0.8969e0 * t12;
    double t18 = t12 * std::sqrt(t12);
    double t19 = 0.204775e0 * t18;
    double t20 = t2 * t2;
    double t21 = t4 * t4;
    double t22 = t20 * t21;
    double t23 = t9 * t9;
    double t26 = t22 * t6 / t23;
    double t27 = 0.123235e0 * t26;
    double t28 = 0.379785e1 * t15 + t17 + t19 + t27;
    double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
    double t32 = std::log(t31);
    double t34 = 0.621814e-1 * t14 * t32;
    double t35 = rho0 - rho1;
    double t36 = t35 * t35;
    double t37 = t36 * t36;
    double t38 = t8 * t8;
    double t39 = t38 * t38;
    double t40 = 0.1e1 / t39;
    double t41 = t37 * t40;
    double t42 = 0.1e1 / t8;
    double t43 = t35 * t42;
    double t44 = 0.1e1 + t43;
    double t45 = t44 <= ZETA_THRESHOLD_C_R2SCAN;
    double t46 = std::cbrt(ZETA_THRESHOLD_C_R2SCAN);
    double t47 = t46 * ZETA_THRESHOLD_C_R2SCAN;
    double t48 = std::cbrt(t44);
    double t49 = t48 * t44;
    double t50 = my_piecewise3(t45, t47, t49);
    double t51 = 0.1e1 - t43;
    double t52 = t51 <= ZETA_THRESHOLD_C_R2SCAN;
    double t53 = std::cbrt(t51);
    double t54 = t53 * t51;
    double t55 = my_piecewise3(t52, t47, t54);
    double t56 = t50 + t55 - 0.2e1;
    double t57 = M_CBRT2;
    double t58 = t57 - 0.1e1;
    double t60 = 0.1e1 / t58 / 0.2e1;
    double t61 = t56 * t60;
    double t63 = 0.1e1 + 0.5137e-1 * t12;
    double t65 = 0.1549425e1 * t12;
    double t66 = 0.420775e0 * t18;
    double t67 = 0.1562925e0 * t26;
    double t68 = 0.705945e1 * t15 + t65 + t66 + t67;
    double t71 = 0.1e1 + 0.32163958997385070134e2 / t68;
    double t72 = std::log(t71);
    double t76 = 0.1e1 + 0.278125e-1 * t12;
    double t78 = 0.905775e0 * t12;
    double t79 = 0.1100325e0 * t18;
    double t80 = 0.1241775e0 * t26;
    double t81 = 0.51785e1 * t15 + t78 + t79 + t80;
    double t84 = 0.1e1 + 0.29608749977793437516e2 / t81;
    double t85 = std::log(t84);
    double t86 = t76 * t85;
    double t88 = -0.310907e-1 * t63 * t72 + t34 - 0.19751673498613801407e-1 * t86;
    double t89 = t61 * t88;
    double t151 = 0.3138525e-1 * t12;
    double t158 = 0.1e1 - 0.2363e1 * t58 * t56 * t60;
    double t160 = t37 * t37;
    double t161 = t160 * t37;
    double t162 = t39 * t39;
    double t163 = t162 * t39;
    double t164 = 0.1e1 / t163;
    double t166 = -t161 * t164 + 0.1e1;
    double t167 = 0.1e1 / t15;
    double t185 = std::sqrt(t12);
    double t234 = t61 * t76;
    double t244 = 0.1e1 + 0.4445e-1 * t15 + t151;
    double t245 = 0.1e1 / t244;
    double t246 = t245 * t158;
    double t271 = t39 * t8;
    double t377 = 0.1e1 / t9 / t8;
    double t378 = t7 * t377;
    double t380 = t5 * t378 * t32;
    double t381 = 0.11073470983333333333e-2 * t380;
    double t385 = t167 * t2;
    double t386 = t4 * t7;
    double t387 = t386 * t377;
    double t388 = t385 * t387;
    double t390 = t5 * t378;
    double t392 = t185 * t2;
    double t393 = t392 * t387;
    double t398 = t22 * t6 / t23 / t8;
    double t409 = 0.1e1 / t271;
    double t410 = t37 * t409;
    double t411 = t410 * t89;
    double t412 = 0.4e1 * t411;
    double t445 = t81 * t81;
    double t446 = 0.1e1 / t445;
    double t449 = 0.301925e0 * t390;
    double t450 = 0.5501625e-1 * t393;
    double t451 = 0.82785e-1 * t398;
    double t452 = -0.86308333333333333334e0 * t388 - t449 - t450 - t451;
    double t453 = 0.1e1 / t84;
    double t462 = t61 * t2;
    double t464 = t386 * t377 * t85;
    double t465 = t462 * t464;
    double t466 = 0.18311447306006545054e-3 * t465;
    double t468 = t446 * t452 * t453;
    double t469 = t234 * t468;
    double t470 = 0.5848223622634646207e0 * t469;
    double t590 = 0.1046175e-1 * t390;
    double t757 = t244 * t244;
    double t758 = 0.1e1 / t757;
    double t759 = t758 * t158;
    double t761 = -0.74083333333333333333e-2 * t388 - t590;
    double t764 = 0.285764e-1 * t759 * t166 * t761;
    double t765 = t245 * t58;
    double t899 = t898 * t88;
    double t900 = t41 * t899;
    double t901 = t898 * t86;
    double t902 = 0.19751673498613801407e-1 * t901;
    double t970 = t898 * t166;
    double t975 = t764 + 0.675260332e-1 * t765 * t970 - 0.285764e-1 * t246 * t955 - t381 - t404 + t408 + t412 - t900 - t459 - t902 + t466 + t470;
    return t975;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho1__t978(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t966 = mgga_c_r2scan_vrho1__t966(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t975 = mgga_c_r2scan_vrho1__t975(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t14 = 0.1e1 + 0.53425e-1 * t12;
    double t15 = std::sqrt(t12);
    double t17 = 0.8969e0 * t12;
    double t18 = t12 * std::sqrt(t12);
    double t19 = 0.204775e0 * t18;
    double t20 = t2 * t2;
    double t21 = t4 * t4;
    double t22 = t20 * t21;
    double t23 = t9 * t9;
    double t26 = t22 * t6 / t23;
    double t27 = 0.123235e0 * t26;
    double t35 = rho0 - rho1;
    double t36 = t35 * t35;
    double t37 = t36 * t36;
    double t38 = t8 * t8;
    double t39 = t38 * t38;
    double t40 = 0.1e1 / t39;
    double t41 = t37 * t40;
    double t42 = 0.1e1 / t8;
    double t43 = t35 * t42;
    double t44 = 0.1e1 + t43;
    double t45 = t44 <= ZETA_THRESHOLD_C_R2SCAN;
    double t46 = std::cbrt(ZETA_THRESHOLD_C_R2SCAN);
    double t47 = t46 * ZETA_THRESHOLD_C_R2SCAN;
    double t48 = std::cbrt(t44);
    double t49 = t48 * t44;
    double t50 = my_piecewise3(t45, t47, t49);
    double t51 = 0.1e1 - t43;
    double t52 = t51 <= ZETA_THRESHOLD_C_R2SCAN;
    double t53 = std::cbrt(t51);
    double t54 = t53 * t51;
    double t55 = my_piecewise3(t52, t47, t54);
    double t56 = t50 + t55 - 0.2e1;
    double t57 = M_CBRT2;
    double t58 = t57 - 0.1e1;
    double t60 = 0.1e1 / t58 / 0.2e1;
    double t61 = t56 * t60;
    double t63 = 0.1e1 + 0.5137e-1 * t12;
    double t65 = 0.1549425e1 * t12;
    double t66 = 0.420775e0 * t18;
    double t67 = 0.1562925e0 * t26;
    double t76 = 0.1e1 + 0.278125e-1 * t12;
    double t78 = 0.905775e0 * t12;
    double t79 = 0.1100325e0 * t18;
    double t80 = 0.1241775e0 * t26;
    double t148 = std::sqrt(0.4e1);
    double t149 = t148 * t15;
    double t151 = 0.3138525e-1 * t12;
    double t152 = 0.1e1 + 0.22225e-1 * t149 + t151;
    double t153 = t152 * t152;
    double t154 = 0.1e1 / t153;
    double t158 = 0.1e1 - 0.2363e1 * t58 * t56 * t60;
    double t159 = t154 * t158;
    double t160 = t37 * t37;
    double t161 = t160 * t37;
    double t162 = t39 * t39;
    double t163 = t162 * t39;
    double t164 = 0.1e1 / t163;
    double t166 = -t161 * t164 + 0.1e1;
    double t167 = 0.1e1 / t15;
    double t168 = t148 * t167;
    double t170 = 0.4445e-1 * t168 + 0.125541e0;
    double t171 = t166 * t170;
    double t175 = 0.1898925e1 * t149 + t17 + t19 + t27;
    double t178 = 0.1e1 + 0.16081979498692535067e2 / t175;
    double t179 = std::log(t178);
    double t180 = 0.1328816518e-1 * t179;
    double t181 = t175 * t175;
    double t182 = 0.1e1 / t181;
    double t183 = t14 * t182;
    double t185 = std::sqrt(t12);
    double t188 = 0.379785e1 * t168 + 0.35876e1 + 0.122865e1 * t185 + 0.24647e0 * t12;
    double t189 = 0.1e1 / t178;
    double t190 = t188 * t189;
    double t192 = 0.1e1 * t183 * t190;
    double t194 = 0.3529725e1 * t149 + t65 + t66 + t67;
    double t197 = 0.1e1 + 0.32163958997385070134e2 / t194;
    double t198 = std::log(t197);
    double t200 = t194 * t194;
    double t201 = 0.1e1 / t200;
    double t202 = t63 * t201;
    double t206 = 0.705945e1 * t168 + 0.61977e1 + 0.252465e1 * t185 + 0.312585e0 * t12;
    double t207 = 0.1e1 / t197;
    double t208 = t206 * t207;
    double t212 = 0.258925e1 * t149 + t78 + t79 + t80;
    double t215 = 0.1e1 + 0.29608749977793437516e2 / t212;
    double t216 = std::log(t215);
    double t218 = t212 * t212;
    double t219 = 0.1e1 / t218;
    double t220 = t76 * t219;
    double t224 = 0.51785e1 * t168 + 0.36231e1 + 0.660195e0 * t185 + 0.248355e0 * t12;
    double t225 = 0.1e1 / t215;
    double t226 = t224 * t225;
    double t229 = -0.6388517036e-2 * t198 + 0.1e1 * t202 * t208 + t180 - t192 - 0.21973736767207854065e-2 * t216 + 0.5848223622634646207e0 * t220 * t226;
    double t230 = t61 * t229;
    double t234 = t61 * t76;
    double t236 = t219 * t224 * t225;
    double t239 = 0.285764e-1 * t159 * t171 + t180 - t192 - t41 * t230 - 0.21973736767207854065e-2 * t61 * t216 + 0.5848223622634646207e0 * t234 * t236;
    double t377 = 0.1e1 / t9 / t8;
    double t378 = t7 * t377;
    double t581 = 0.5e1 / 0.3e1 * t5 * t378 * t239;
    double t978 = 0.5e1 * t5 * t11 * t966 - 0.45e2 * 0.001 * t975 - t581;
    return t978;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho1__t979(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t978 = mgga_c_r2scan_vrho1__t978(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t14 = 0.1e1 + 0.53425e-1 * t12;
    double t15 = std::sqrt(t12);
    double t17 = 0.8969e0 * t12;
    double t18 = t12 * std::sqrt(t12);
    double t19 = 0.204775e0 * t18;
    double t20 = t2 * t2;
    double t21 = t4 * t4;
    double t22 = t20 * t21;
    double t23 = t9 * t9;
    double t26 = t22 * t6 / t23;
    double t27 = 0.123235e0 * t26;
    double t28 = 0.379785e1 * t15 + t17 + t19 + t27;
    double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
    double t32 = std::log(t31);
    double t34 = 0.621814e-1 * t14 * t32;
    double t35 = rho0 - rho1;
    double t36 = t35 * t35;
    double t37 = t36 * t36;
    double t38 = t8 * t8;
    double t39 = t38 * t38;
    double t40 = 0.1e1 / t39;
    double t41 = t37 * t40;
    double t42 = 0.1e1 / t8;
    double t43 = t35 * t42;
    double t44 = 0.1e1 + t43;
    double t45 = t44 <= ZETA_THRESHOLD_C_R2SCAN;
    double t46 = std::cbrt(ZETA_THRESHOLD_C_R2SCAN);
    double t47 = t46 * ZETA_THRESHOLD_C_R2SCAN;
    double t48 = std::cbrt(t44);
    double t49 = t48 * t44;
    double t50 = my_piecewise3(t45, t47, t49);
    double t51 = 0.1e1 - t43;
    double t52 = t51 <= ZETA_THRESHOLD_C_R2SCAN;
    double t53 = std::cbrt(t51);
    double t54 = t53 * t51;
    double t55 = my_piecewise3(t52, t47, t54);
    double t56 = t50 + t55 - 0.2e1;
    double t57 = M_CBRT2;
    double t58 = t57 - 0.1e1;
    double t60 = 0.1e1 / t58 / 0.2e1;
    double t61 = t56 * t60;
    double t63 = 0.1e1 + 0.5137e-1 * t12;
    double t65 = 0.1549425e1 * t12;
    double t66 = 0.420775e0 * t18;
    double t67 = 0.1562925e0 * t26;
    double t68 = 0.705945e1 * t15 + t65 + t66 + t67;
    double t71 = 0.1e1 + 0.32163958997385070134e2 / t68;
    double t72 = std::log(t71);
    double t76 = 0.1e1 + 0.278125e-1 * t12;
    double t78 = 0.905775e0 * t12;
    double t79 = 0.1100325e0 * t18;
    double t80 = 0.1241775e0 * t26;
    double t81 = 0.51785e1 * t15 + t78 + t79 + t80;
    double t84 = 0.1e1 + 0.29608749977793437516e2 / t81;
    double t85 = std::log(t84);
    double t86 = t76 * t85;
    double t88 = -0.310907e-1 * t63 * t72 + t34 - 0.19751673498613801407e-1 * t86;
    double t89 = t61 * t88;
    double t90 = t41 * t89;
    double t92 = 0.19751673498613801407e-1 * t61 * t86;
    double t93 = std::log(0.2e1);
    double t94 = 0.1e1 - t93;
    double t95 = M_PI * M_PI;
    double t98 = t46 * t46;
    double t99 = t48 * t48;
    double t100 = my_piecewise3(t45, t98, t99);
    double t101 = t53 * t53;
    double t102 = my_piecewise3(t52, t98, t101);
    double t104 = t100 / 0.2e1 + t102 / 0.2e1;
    double t105 = t104 * t104;
    double t106 = t105 * t104;
    double t108 = 0.1e1 / t94;
    double t109 = (-t34 + t90 + t92) * t108;
    double t110 = 0.1e1 / t106;
    double t111 = t95 * t110;
    double t113 = std::exp(-t109 * t111);
    double t114 = t113 - 0.1e1;
    double t133 = 0.1e1 / t114;
    double t147 = t110 * t133;
    double t979 = t147 * t978;
    return t979;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho1__t980(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t979 = mgga_c_r2scan_vrho1__t979(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t8 = rho0 + rho1;
    double t35 = rho0 - rho1;
    double t42 = 0.1e1 / t8;
    double t43 = t35 * t42;
    double t44 = 0.1e1 + t43;
    double t45 = t44 <= ZETA_THRESHOLD_C_R2SCAN;
    double t46 = std::cbrt(ZETA_THRESHOLD_C_R2SCAN);
    double t48 = std::cbrt(t44);
    double t51 = 0.1e1 - t43;
    double t52 = t51 <= ZETA_THRESHOLD_C_R2SCAN;
    double t53 = std::cbrt(t51);
    double t93 = std::log(0.2e1);
    double t94 = 0.1e1 - t93;
    double t98 = t46 * t46;
    double t99 = t48 * t48;
    double t101 = t53 * t53;
    double t108 = 0.1e1 / t94;
    double t138 = t98 * ZETA_THRESHOLD_C_R2SCAN;
    double t139 = t99 * t44;
    double t140 = my_piecewise3(t45, t138, t139);
    double t141 = t101 * t51;
    double t142 = my_piecewise3(t52, t138, t141);
    double t144 = t140 / 0.2e1 + t142 / 0.2e1;
    double t146 = t108 / t144;
    double t980 = t146 * t979;
    return t980;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho1__t983(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t252 = mgga_c_r2scan_vrho1__t252(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t516 = mgga_c_r2scan_vrho1__t516(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t791 = mgga_c_r2scan_vrho1__t791(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t910 = mgga_c_r2scan_vrho1__t910(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t921 = mgga_c_r2scan_vrho1__t921(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t940 = mgga_c_r2scan_vrho1__t940(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t980 = mgga_c_r2scan_vrho1__t980(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t14 = 0.1e1 + 0.53425e-1 * t12;
    double t15 = std::sqrt(t12);
    double t17 = 0.8969e0 * t12;
    double t18 = t12 * std::sqrt(t12);
    double t19 = 0.204775e0 * t18;
    double t20 = t2 * t2;
    double t21 = t4 * t4;
    double t22 = t20 * t21;
    double t23 = t9 * t9;
    double t26 = t22 * t6 / t23;
    double t27 = 0.123235e0 * t26;
    double t28 = 0.379785e1 * t15 + t17 + t19 + t27;
    double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
    double t32 = std::log(t31);
    double t34 = 0.621814e-1 * t14 * t32;
    double t35 = rho0 - rho1;
    double t36 = t35 * t35;
    double t37 = t36 * t36;
    double t38 = t8 * t8;
    double t39 = t38 * t38;
    double t40 = 0.1e1 / t39;
    double t41 = t37 * t40;
    double t42 = 0.1e1 / t8;
    double t43 = t35 * t42;
    double t44 = 0.1e1 + t43;
    double t45 = t44 <= ZETA_THRESHOLD_C_R2SCAN;
    double t46 = std::cbrt(ZETA_THRESHOLD_C_R2SCAN);
    double t47 = t46 * ZETA_THRESHOLD_C_R2SCAN;
    double t48 = std::cbrt(t44);
    double t49 = t48 * t44;
    double t50 = my_piecewise3(t45, t47, t49);
    double t51 = 0.1e1 - t43;
    double t52 = t51 <= ZETA_THRESHOLD_C_R2SCAN;
    double t53 = std::cbrt(t51);
    double t54 = t53 * t51;
    double t55 = my_piecewise3(t52, t47, t54);
    double t56 = t50 + t55 - 0.2e1;
    double t57 = M_CBRT2;
    double t58 = t57 - 0.1e1;
    double t60 = 0.1e1 / t58 / 0.2e1;
    double t61 = t56 * t60;
    double t63 = 0.1e1 + 0.5137e-1 * t12;
    double t65 = 0.1549425e1 * t12;
    double t66 = 0.420775e0 * t18;
    double t67 = 0.1562925e0 * t26;
    double t68 = 0.705945e1 * t15 + t65 + t66 + t67;
    double t71 = 0.1e1 + 0.32163958997385070134e2 / t68;
    double t72 = std::log(t71);
    double t76 = 0.1e1 + 0.278125e-1 * t12;
    double t78 = 0.905775e0 * t12;
    double t79 = 0.1100325e0 * t18;
    double t80 = 0.1241775e0 * t26;
    double t81 = 0.51785e1 * t15 + t78 + t79 + t80;
    double t84 = 0.1e1 + 0.29608749977793437516e2 / t81;
    double t85 = std::log(t84);
    double t86 = t76 * t85;
    double t88 = -0.310907e-1 * t63 * t72 + t34 - 0.19751673498613801407e-1 * t86;
    double t89 = t61 * t88;
    double t90 = t41 * t89;
    double t92 = 0.19751673498613801407e-1 * t61 * t86;
    double t93 = std::log(0.2e1);
    double t94 = 0.1e1 - t93;
    double t95 = M_PI * M_PI;
    double t98 = t46 * t46;
    double t99 = t48 * t48;
    double t100 = my_piecewise3(t45, t98, t99);
    double t101 = t53 * t53;
    double t102 = my_piecewise3(t52, t98, t101);
    double t104 = t100 / 0.2e1 + t102 / 0.2e1;
    double t105 = t104 * t104;
    double t106 = t105 * t104;
    double t108 = 0.1e1 / t94;
    double t109 = (-t34 + t90 + t92) * t108;
    double t110 = 0.1e1 / t106;
    double t111 = t95 * t110;
    double t113 = std::exp(-t109 * t111);
    double t114 = t113 - 0.1e1;
    double t116 = 0.1e1 + 0.25e-1 * t12;
    double t118 = 0.1e1 + 0.4445e-1 * t12;
    double t119 = 0.1e1 / t118;
    double t120 = t116 * t119;
    double t122 = sigma0 + 0.2e1 * sigma1 + sigma2;
    double t124 = 0.1e1 / t9 / t38;
    double t128 = 0.1e1 / t105;
    double t129 = t128 * t20;
    double t130 = 0.1e1 / t4;
    double t132 = t6 * t108;
    double t133 = 0.1e1 / t114;
    double t134 = t132 * t133;
    double t135 = t129 * t130 * t134;
    double t138 = t98 * ZETA_THRESHOLD_C_R2SCAN;
    double t139 = t99 * t44;
    double t140 = my_piecewise3(t45, t138, t139);
    double t141 = t101 * t51;
    double t142 = my_piecewise3(t52, t138, t141);
    double t144 = t140 / 0.2e1 + t142 / 0.2e1;
    double t146 = t108 / t144;
    double t147 = t110 * t133;
    double t253 = t147 * t252;
    double t254 = t146 * t253;
    double t255 = M_CBRT6;
    double t256 = std::cbrt(t95);
    double t257 = t256 * t256;
    double t258 = 0.1e1 / t257;
    double t259 = t255 * t258;
    double t260 = t57 * t57;
    double t261 = t259 * t260;
    double t263 = 0.1e1 / t23 / t38;
    double t264 = t122 * t263;
    double t265 = t255 * t255;
    double t267 = 0.1e1 / t256 / t95;
    double t268 = t265 * t267;
    double t269 = t122 * t122;
    double t270 = t57 * t269;
    double t271 = t39 * t8;
    double t273 = 0.1e1 / t9 / t271;
    double t277 = std::exp(-0.20444604078896369094e0 * t268 * t270 * t273);
    double t279 = t261 * t264 * t277;
    double t488 = t105 * t105;
    double t489 = 0.1e1 / t488;
    double t500 = t38 * t8;
    double t502 = 0.1e1 / t23 / t500;
    double t503 = t502 * t119;
    double t505 = t57 * t128;
    double t506 = t108 * t133;
    double t507 = t505 * t506;
    double t509 = 0.27439371595564631661e-2 * t503 * t122 * t507;
    double t513 = t122 * t502;
    double t518 = 0.1e1 / t9 / t500;
    double t523 = 0.64025200389650807209e-1 * t120 * t122 * t518 * t57 * t135;
    double t524 = t120 * t122;
    double t525 = t124 * t57;
    double t526 = t525 * t110;
    double t527 = t524 * t526;
    double t528 = t20 * t130;
    double t529 = t528 * t6;
    double t534 = t525 * t128;
    double t535 = t524 * t534;
    double t536 = t114 * t114;
    double t537 = 0.1e1 / t536;
    double t538 = t108 * t537;
    double t543 = t144 * t144;
    double t545 = t108 / t543;
    double t546 = t545 * t110;
    double t547 = t133 * t252;
    double t548 = t547 * t255;
    double t549 = t546 * t548;
    double t550 = t258 * t260;
    double t551 = t550 * t122;
    double t552 = t263 * t277;
    double t565 = t146 * t489;
    double t566 = t565 * t548;
    double t571 = t146 * t110;
    double t572 = t537 * t252;
    double t573 = t572 * t255;
    double t574 = t571 * t573;
    double t780 = t261 * t513 * t277;
    double t782 = 0.11557628986739024751e0 * t254 * t780;
    double t923 = t506 * t910;
    double t924 = t529 * t923;
    double t927 = t538 * t921;
    double t928 = t529 * t927;
    double t943 = t552 * t910;
    double t944 = t551 * t943;
    double t948 = t551 * t552 * t921;
    double t983 = -t509 + t516 - t523 - 0.54878743191129263322e-1 * t527 * t924 - 0.27439371595564631661e-1 * t535 * t928 - 0.43341108700271342816e-1 * t549 * t940 - 0.13002332610081402845e0 * t566 * t944 - 0.43341108700271342816e-1 * t574 * t948 + 0.43341108700271342816e-1 * t980 * t279 - t782 + t791;
    return t983;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho1__t986(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t252 = mgga_c_r2scan_vrho1__t252(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t921 = mgga_c_r2scan_vrho1__t921(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t983 = mgga_c_r2scan_vrho1__t983(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t14 = 0.1e1 + 0.53425e-1 * t12;
    double t15 = std::sqrt(t12);
    double t17 = 0.8969e0 * t12;
    double t18 = t12 * std::sqrt(t12);
    double t19 = 0.204775e0 * t18;
    double t20 = t2 * t2;
    double t21 = t4 * t4;
    double t22 = t20 * t21;
    double t23 = t9 * t9;
    double t26 = t22 * t6 / t23;
    double t27 = 0.123235e0 * t26;
    double t28 = 0.379785e1 * t15 + t17 + t19 + t27;
    double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
    double t32 = std::log(t31);
    double t34 = 0.621814e-1 * t14 * t32;
    double t35 = rho0 - rho1;
    double t36 = t35 * t35;
    double t37 = t36 * t36;
    double t38 = t8 * t8;
    double t39 = t38 * t38;
    double t40 = 0.1e1 / t39;
    double t41 = t37 * t40;
    double t42 = 0.1e1 / t8;
    double t43 = t35 * t42;
    double t44 = 0.1e1 + t43;
    double t45 = t44 <= ZETA_THRESHOLD_C_R2SCAN;
    double t46 = std::cbrt(ZETA_THRESHOLD_C_R2SCAN);
    double t47 = t46 * ZETA_THRESHOLD_C_R2SCAN;
    double t48 = std::cbrt(t44);
    double t49 = t48 * t44;
    double t50 = my_piecewise3(t45, t47, t49);
    double t51 = 0.1e1 - t43;
    double t52 = t51 <= ZETA_THRESHOLD_C_R2SCAN;
    double t53 = std::cbrt(t51);
    double t54 = t53 * t51;
    double t55 = my_piecewise3(t52, t47, t54);
    double t56 = t50 + t55 - 0.2e1;
    double t57 = M_CBRT2;
    double t58 = t57 - 0.1e1;
    double t60 = 0.1e1 / t58 / 0.2e1;
    double t61 = t56 * t60;
    double t63 = 0.1e1 + 0.5137e-1 * t12;
    double t65 = 0.1549425e1 * t12;
    double t66 = 0.420775e0 * t18;
    double t67 = 0.1562925e0 * t26;
    double t68 = 0.705945e1 * t15 + t65 + t66 + t67;
    double t71 = 0.1e1 + 0.32163958997385070134e2 / t68;
    double t72 = std::log(t71);
    double t76 = 0.1e1 + 0.278125e-1 * t12;
    double t78 = 0.905775e0 * t12;
    double t79 = 0.1100325e0 * t18;
    double t80 = 0.1241775e0 * t26;
    double t81 = 0.51785e1 * t15 + t78 + t79 + t80;
    double t84 = 0.1e1 + 0.29608749977793437516e2 / t81;
    double t85 = std::log(t84);
    double t86 = t76 * t85;
    double t88 = -0.310907e-1 * t63 * t72 + t34 - 0.19751673498613801407e-1 * t86;
    double t89 = t61 * t88;
    double t90 = t41 * t89;
    double t92 = 0.19751673498613801407e-1 * t61 * t86;
    double t93 = std::log(0.2e1);
    double t94 = 0.1e1 - t93;
    double t95 = M_PI * M_PI;
    double t98 = t46 * t46;
    double t99 = t48 * t48;
    double t100 = my_piecewise3(t45, t98, t99);
    double t101 = t53 * t53;
    double t102 = my_piecewise3(t52, t98, t101);
    double t104 = t100 / 0.2e1 + t102 / 0.2e1;
    double t105 = t104 * t104;
    double t106 = t105 * t104;
    double t108 = 0.1e1 / t94;
    double t109 = (-t34 + t90 + t92) * t108;
    double t110 = 0.1e1 / t106;
    double t111 = t95 * t110;
    double t113 = std::exp(-t109 * t111);
    double t114 = t113 - 0.1e1;
    double t116 = 0.1e1 + 0.25e-1 * t12;
    double t118 = 0.1e1 + 0.4445e-1 * t12;
    double t119 = 0.1e1 / t118;
    double t120 = t116 * t119;
    double t122 = sigma0 + 0.2e1 * sigma1 + sigma2;
    double t124 = 0.1e1 / t9 / t38;
    double t125 = t122 * t124;
    double t128 = 0.1e1 / t105;
    double t129 = t128 * t20;
    double t130 = 0.1e1 / t4;
    double t132 = t6 * t108;
    double t133 = 0.1e1 / t114;
    double t134 = t132 * t133;
    double t135 = t129 * t130 * t134;
    double t138 = t98 * ZETA_THRESHOLD_C_R2SCAN;
    double t139 = t99 * t44;
    double t140 = my_piecewise3(t45, t138, t139);
    double t141 = t101 * t51;
    double t142 = my_piecewise3(t52, t138, t141);
    double t144 = t140 / 0.2e1 + t142 / 0.2e1;
    double t146 = t108 / t144;
    double t147 = t110 * t133;
    double t253 = t147 * t252;
    double t254 = t146 * t253;
    double t255 = M_CBRT6;
    double t256 = std::cbrt(t95);
    double t257 = t256 * t256;
    double t258 = 0.1e1 / t257;
    double t259 = t255 * t258;
    double t260 = t57 * t57;
    double t261 = t259 * t260;
    double t263 = 0.1e1 / t23 / t38;
    double t264 = t122 * t263;
    double t265 = t255 * t255;
    double t267 = 0.1e1 / t256 / t95;
    double t268 = t265 * t267;
    double t269 = t122 * t122;
    double t270 = t57 * t269;
    double t271 = t39 * t8;
    double t273 = 0.1e1 / t9 / t271;
    double t277 = std::exp(-0.20444604078896369094e0 * t268 * t270 * t273);
    double t279 = t261 * t264 * t277;
    double t282 = 0.1e1 + 0.27439371595564631661e-1 * t120 * t125 * t57 * t135 + 0.43341108700271342816e-1 * t254 * t279;
    double t283 = std::sqrt(std::sqrt(t282));
    double t285 = 0.1e1 - 0.1e1 / t283;
    double t498 = 0.1e1 / t283 / t282;
    double t499 = t114 * t498;
    double t986 = t921 * t285 + t499 * t983 / 0.4e1;
    return t986;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho1__t989(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t252 = mgga_c_r2scan_vrho1__t252(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t986 = mgga_c_r2scan_vrho1__t986(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t14 = 0.1e1 + 0.53425e-1 * t12;
    double t15 = std::sqrt(t12);
    double t17 = 0.8969e0 * t12;
    double t18 = t12 * std::sqrt(t12);
    double t19 = 0.204775e0 * t18;
    double t20 = t2 * t2;
    double t21 = t4 * t4;
    double t22 = t20 * t21;
    double t23 = t9 * t9;
    double t26 = t22 * t6 / t23;
    double t27 = 0.123235e0 * t26;
    double t28 = 0.379785e1 * t15 + t17 + t19 + t27;
    double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
    double t32 = std::log(t31);
    double t34 = 0.621814e-1 * t14 * t32;
    double t35 = rho0 - rho1;
    double t36 = t35 * t35;
    double t37 = t36 * t36;
    double t38 = t8 * t8;
    double t39 = t38 * t38;
    double t40 = 0.1e1 / t39;
    double t41 = t37 * t40;
    double t42 = 0.1e1 / t8;
    double t43 = t35 * t42;
    double t44 = 0.1e1 + t43;
    double t45 = t44 <= ZETA_THRESHOLD_C_R2SCAN;
    double t46 = std::cbrt(ZETA_THRESHOLD_C_R2SCAN);
    double t47 = t46 * ZETA_THRESHOLD_C_R2SCAN;
    double t48 = std::cbrt(t44);
    double t49 = t48 * t44;
    double t50 = my_piecewise3(t45, t47, t49);
    double t51 = 0.1e1 - t43;
    double t52 = t51 <= ZETA_THRESHOLD_C_R2SCAN;
    double t53 = std::cbrt(t51);
    double t54 = t53 * t51;
    double t55 = my_piecewise3(t52, t47, t54);
    double t56 = t50 + t55 - 0.2e1;
    double t57 = M_CBRT2;
    double t58 = t57 - 0.1e1;
    double t60 = 0.1e1 / t58 / 0.2e1;
    double t61 = t56 * t60;
    double t63 = 0.1e1 + 0.5137e-1 * t12;
    double t65 = 0.1549425e1 * t12;
    double t66 = 0.420775e0 * t18;
    double t67 = 0.1562925e0 * t26;
    double t68 = 0.705945e1 * t15 + t65 + t66 + t67;
    double t71 = 0.1e1 + 0.32163958997385070134e2 / t68;
    double t72 = std::log(t71);
    double t76 = 0.1e1 + 0.278125e-1 * t12;
    double t78 = 0.905775e0 * t12;
    double t79 = 0.1100325e0 * t18;
    double t80 = 0.1241775e0 * t26;
    double t81 = 0.51785e1 * t15 + t78 + t79 + t80;
    double t84 = 0.1e1 + 0.29608749977793437516e2 / t81;
    double t85 = std::log(t84);
    double t86 = t76 * t85;
    double t88 = -0.310907e-1 * t63 * t72 + t34 - 0.19751673498613801407e-1 * t86;
    double t89 = t61 * t88;
    double t90 = t41 * t89;
    double t92 = 0.19751673498613801407e-1 * t61 * t86;
    double t93 = std::log(0.2e1);
    double t94 = 0.1e1 - t93;
    double t95 = M_PI * M_PI;
    double t97 = t94 / t95;
    double t98 = t46 * t46;
    double t99 = t48 * t48;
    double t100 = my_piecewise3(t45, t98, t99);
    double t101 = t53 * t53;
    double t102 = my_piecewise3(t52, t98, t101);
    double t104 = t100 / 0.2e1 + t102 / 0.2e1;
    double t105 = t104 * t104;
    double t106 = t105 * t104;
    double t108 = 0.1e1 / t94;
    double t109 = (-t34 + t90 + t92) * t108;
    double t110 = 0.1e1 / t106;
    double t111 = t95 * t110;
    double t113 = std::exp(-t109 * t111);
    double t114 = t113 - 0.1e1;
    double t116 = 0.1e1 + 0.25e-1 * t12;
    double t118 = 0.1e1 + 0.4445e-1 * t12;
    double t119 = 0.1e1 / t118;
    double t120 = t116 * t119;
    double t122 = sigma0 + 0.2e1 * sigma1 + sigma2;
    double t124 = 0.1e1 / t9 / t38;
    double t125 = t122 * t124;
    double t128 = 0.1e1 / t105;
    double t129 = t128 * t20;
    double t130 = 0.1e1 / t4;
    double t132 = t6 * t108;
    double t133 = 0.1e1 / t114;
    double t134 = t132 * t133;
    double t135 = t129 * t130 * t134;
    double t138 = t98 * ZETA_THRESHOLD_C_R2SCAN;
    double t139 = t99 * t44;
    double t140 = my_piecewise3(t45, t138, t139);
    double t141 = t101 * t51;
    double t142 = my_piecewise3(t52, t138, t141);
    double t144 = t140 / 0.2e1 + t142 / 0.2e1;
    double t146 = t108 / t144;
    double t147 = t110 * t133;
    double t253 = t147 * t252;
    double t254 = t146 * t253;
    double t255 = M_CBRT6;
    double t256 = std::cbrt(t95);
    double t257 = t256 * t256;
    double t258 = 0.1e1 / t257;
    double t259 = t255 * t258;
    double t260 = t57 * t57;
    double t261 = t259 * t260;
    double t263 = 0.1e1 / t23 / t38;
    double t264 = t122 * t263;
    double t265 = t255 * t255;
    double t267 = 0.1e1 / t256 / t95;
    double t268 = t265 * t267;
    double t269 = t122 * t122;
    double t270 = t57 * t269;
    double t271 = t39 * t8;
    double t273 = 0.1e1 / t9 / t271;
    double t277 = std::exp(-0.20444604078896369094e0 * t268 * t270 * t273);
    double t279 = t261 * t264 * t277;
    double t282 = 0.1e1 + 0.27439371595564631661e-1 * t120 * t125 * t57 * t135 + 0.43341108700271342816e-1 * t254 * t279;
    double t283 = std::sqrt(std::sqrt(t282));
    double t285 = 0.1e1 - 0.1e1 / t283;
    double t287 = t114 * t285 + 0.1e1;
    double t797 = 0.1e1 / t287;
    double t989 = t97 * t106 * t986 * t797;
    return t989;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho1__t1046(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t1040 = mgga_c_r2scan_vrho1__t1040(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t354 = mgga_c_r2scan_vrho1__t354(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t371 = mgga_c_r2scan_vrho1__t371(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t404 = mgga_c_r2scan_vrho1__t404(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t408 = mgga_c_r2scan_vrho1__t408(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t459 = mgga_c_r2scan_vrho1__t459(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t880 = mgga_c_r2scan_vrho1__t880(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t898 = mgga_c_r2scan_vrho1__t898(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t913 = mgga_c_r2scan_vrho1__t913(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t955 = mgga_c_r2scan_vrho1__t955(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t989 = mgga_c_r2scan_vrho1__t989(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t14 = 0.1e1 + 0.53425e-1 * t12;
    double t15 = std::sqrt(t12);
    double t17 = 0.8969e0 * t12;
    double t18 = t12 * std::sqrt(t12);
    double t19 = 0.204775e0 * t18;
    double t20 = t2 * t2;
    double t21 = t4 * t4;
    double t22 = t20 * t21;
    double t23 = t9 * t9;
    double t26 = t22 * t6 / t23;
    double t27 = 0.123235e0 * t26;
    double t28 = 0.379785e1 * t15 + t17 + t19 + t27;
    double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
    double t32 = std::log(t31);
    double t34 = 0.621814e-1 * t14 * t32;
    double t35 = rho0 - rho1;
    double t36 = t35 * t35;
    double t37 = t36 * t36;
    double t38 = t8 * t8;
    double t39 = t38 * t38;
    double t40 = 0.1e1 / t39;
    double t41 = t37 * t40;
    double t42 = 0.1e1 / t8;
    double t43 = t35 * t42;
    double t44 = 0.1e1 + t43;
    double t45 = t44 <= ZETA_THRESHOLD_C_R2SCAN;
    double t46 = std::cbrt(ZETA_THRESHOLD_C_R2SCAN);
    double t47 = t46 * ZETA_THRESHOLD_C_R2SCAN;
    double t48 = std::cbrt(t44);
    double t49 = t48 * t44;
    double t50 = my_piecewise3(t45, t47, t49);
    double t51 = 0.1e1 - t43;
    double t52 = t51 <= ZETA_THRESHOLD_C_R2SCAN;
    double t53 = std::cbrt(t51);
    double t54 = t53 * t51;
    double t55 = my_piecewise3(t52, t47, t54);
    double t56 = t50 + t55 - 0.2e1;
    double t57 = M_CBRT2;
    double t58 = t57 - 0.1e1;
    double t60 = 0.1e1 / t58 / 0.2e1;
    double t61 = t56 * t60;
    double t63 = 0.1e1 + 0.5137e-1 * t12;
    double t65 = 0.1549425e1 * t12;
    double t66 = 0.420775e0 * t18;
    double t67 = 0.1562925e0 * t26;
    double t68 = 0.705945e1 * t15 + t65 + t66 + t67;
    double t71 = 0.1e1 + 0.32163958997385070134e2 / t68;
    double t72 = std::log(t71);
    double t76 = 0.1e1 + 0.278125e-1 * t12;
    double t78 = 0.905775e0 * t12;
    double t79 = 0.1100325e0 * t18;
    double t80 = 0.1241775e0 * t26;
    double t81 = 0.51785e1 * t15 + t78 + t79 + t80;
    double t84 = 0.1e1 + 0.29608749977793437516e2 / t81;
    double t85 = std::log(t84);
    double t86 = t76 * t85;
    double t88 = -0.310907e-1 * t63 * t72 + t34 - 0.19751673498613801407e-1 * t86;
    double t89 = t61 * t88;
    double t158 = 0.1e1 - 0.2363e1 * t58 * t56 * t60;
    double t160 = t37 * t37;
    double t161 = t160 * t37;
    double t162 = t39 * t39;
    double t163 = t162 * t39;
    double t164 = 0.1e1 / t163;
    double t166 = -t161 * t164 + 0.1e1;
    double t167 = 0.1e1 / t15;
    double t185 = std::sqrt(t12);
    double t234 = t61 * t76;
    double t271 = t39 * t8;
    double t372 = t371 * t158;
    double t377 = 0.1e1 / t9 / t8;
    double t378 = t7 * t377;
    double t380 = t5 * t378 * t32;
    double t381 = 0.11073470983333333333e-2 * t380;
    double t385 = t167 * t2;
    double t386 = t4 * t7;
    double t387 = t386 * t377;
    double t388 = t385 * t387;
    double t390 = t5 * t378;
    double t392 = t185 * t2;
    double t393 = t392 * t387;
    double t398 = t22 * t6 / t23 / t8;
    double t409 = 0.1e1 / t271;
    double t410 = t37 * t409;
    double t411 = t410 * t89;
    double t412 = 0.4e1 * t411;
    double t445 = t81 * t81;
    double t446 = 0.1e1 / t445;
    double t449 = 0.301925e0 * t390;
    double t450 = 0.5501625e-1 * t393;
    double t451 = 0.82785e-1 * t398;
    double t452 = -0.86308333333333333334e0 * t388 - t449 - t450 - t451;
    double t453 = 0.1e1 / t84;
    double t462 = t61 * t2;
    double t464 = t386 * t377 * t85;
    double t465 = t462 * t464;
    double t466 = 0.18311447306006545054e-3 * t465;
    double t468 = t446 * t452 * t453;
    double t469 = t234 * t468;
    double t470 = 0.5848223622634646207e0 * t469;
    double t881 = t371 * t58;
    double t899 = t898 * t88;
    double t900 = t41 * t899;
    double t901 = t898 * t86;
    double t902 = 0.19751673498613801407e-1 * t901;
    double t970 = t898 * t166;
    double t1044 = t880 - 0.2363e1 * t881 * t970 + t372 * t955 - t381 - t404 + t408 + t412 - t900 - t459 - t902 + t466 + t470 - t913 - t989;
    double t1045 = t354 * t1044;
    double t1046 = t381 + t404 - t408 - t412 + t900 + t459 + t902 - t466 - t470 + t913 + t989 + t1040 + t1045;
    return t1046;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho1__t1046v(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t1046 = mgga_c_r2scan_vrho1__t1046(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t1046v = t1046;
    return t1046v;
  }

  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vrho1(double rho0, double rho1, double sigma0, double sigma1, double sigma2, double tau0, double tau1)
  {
    double t1046v = mgga_c_r2scan_vrho1__t1046v(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t252 = mgga_c_r2scan_vrho1__t252(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t354 = mgga_c_r2scan_vrho1__t354(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t374 = mgga_c_r2scan_vrho1__t374(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
    double t2 = M_CBRT3;
    double t3 = 0.1e1 / M_PI;
    double t4 = std::cbrt(t3);
    double t5 = t2 * t4;
    double t6 = M_CBRT4;
    double t7 = t6 * t6;
    double t8 = rho0 + rho1;
    double t9 = std::cbrt(t8);
    double t11 = t7 / t9;
    double t12 = t5 * t11;
    double t14 = 0.1e1 + 0.53425e-1 * t12;
    double t15 = std::sqrt(t12);
    double t17 = 0.8969e0 * t12;
    double t18 = t12 * std::sqrt(t12);
    double t19 = 0.204775e0 * t18;
    double t20 = t2 * t2;
    double t21 = t4 * t4;
    double t22 = t20 * t21;
    double t23 = t9 * t9;
    double t26 = t22 * t6 / t23;
    double t27 = 0.123235e0 * t26;
    double t28 = 0.379785e1 * t15 + t17 + t19 + t27;
    double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
    double t32 = std::log(t31);
    double t34 = 0.621814e-1 * t14 * t32;
    double t35 = rho0 - rho1;
    double t36 = t35 * t35;
    double t37 = t36 * t36;
    double t38 = t8 * t8;
    double t39 = t38 * t38;
    double t40 = 0.1e1 / t39;
    double t41 = t37 * t40;
    double t42 = 0.1e1 / t8;
    double t43 = t35 * t42;
    double t44 = 0.1e1 + t43;
    double t45 = t44 <= ZETA_THRESHOLD_C_R2SCAN;
    double t46 = std::cbrt(ZETA_THRESHOLD_C_R2SCAN);
    double t47 = t46 * ZETA_THRESHOLD_C_R2SCAN;
    double t48 = std::cbrt(t44);
    double t49 = t48 * t44;
    double t50 = my_piecewise3(t45, t47, t49);
    double t51 = 0.1e1 - t43;
    double t52 = t51 <= ZETA_THRESHOLD_C_R2SCAN;
    double t53 = std::cbrt(t51);
    double t54 = t53 * t51;
    double t55 = my_piecewise3(t52, t47, t54);
    double t56 = t50 + t55 - 0.2e1;
    double t57 = M_CBRT2;
    double t58 = t57 - 0.1e1;
    double t60 = 0.1e1 / t58 / 0.2e1;
    double t61 = t56 * t60;
    double t63 = 0.1e1 + 0.5137e-1 * t12;
    double t65 = 0.1549425e1 * t12;
    double t66 = 0.420775e0 * t18;
    double t67 = 0.1562925e0 * t26;
    double t68 = 0.705945e1 * t15 + t65 + t66 + t67;
    double t71 = 0.1e1 + 0.32163958997385070134e2 / t68;
    double t72 = std::log(t71);
    double t76 = 0.1e1 + 0.278125e-1 * t12;
    double t78 = 0.905775e0 * t12;
    double t79 = 0.1100325e0 * t18;
    double t80 = 0.1241775e0 * t26;
    double t81 = 0.51785e1 * t15 + t78 + t79 + t80;
    double t84 = 0.1e1 + 0.29608749977793437516e2 / t81;
    double t85 = std::log(t84);
    double t86 = t76 * t85;
    double t88 = -0.310907e-1 * t63 * t72 + t34 - 0.19751673498613801407e-1 * t86;
    double t89 = t61 * t88;
    double t90 = t41 * t89;
    double t92 = 0.19751673498613801407e-1 * t61 * t86;
    double t93 = std::log(0.2e1);
    double t94 = 0.1e1 - t93;
    double t95 = M_PI * M_PI;
    double t97 = t94 / t95;
    double t98 = t46 * t46;
    double t99 = t48 * t48;
    double t100 = my_piecewise3(t45, t98, t99);
    double t101 = t53 * t53;
    double t102 = my_piecewise3(t52, t98, t101);
    double t104 = t100 / 0.2e1 + t102 / 0.2e1;
    double t105 = t104 * t104;
    double t106 = t105 * t104;
    double t108 = 0.1e1 / t94;
    double t109 = (-t34 + t90 + t92) * t108;
    double t110 = 0.1e1 / t106;
    double t111 = t95 * t110;
    double t113 = std::exp(-t109 * t111);
    double t114 = t113 - 0.1e1;
    double t116 = 0.1e1 + 0.25e-1 * t12;
    double t118 = 0.1e1 + 0.4445e-1 * t12;
    double t119 = 0.1e1 / t118;
    double t120 = t116 * t119;
    double t122 = sigma0 + 0.2e1 * sigma1 + sigma2;
    double t124 = 0.1e1 / t9 / t38;
    double t125 = t122 * t124;
    double t128 = 0.1e1 / t105;
    double t129 = t128 * t20;
    double t130 = 0.1e1 / t4;
    double t132 = t6 * t108;
    double t133 = 0.1e1 / t114;
    double t134 = t132 * t133;
    double t135 = t129 * t130 * t134;
    double t138 = t98 * ZETA_THRESHOLD_C_R2SCAN;
    double t139 = t99 * t44;
    double t140 = my_piecewise3(t45, t138, t139);
    double t141 = t101 * t51;
    double t142 = my_piecewise3(t52, t138, t141);
    double t144 = t140 / 0.2e1 + t142 / 0.2e1;
    double t146 = t108 / t144;
    double t147 = t110 * t133;
    double t253 = t147 * t252;
    double t254 = t146 * t253;
    double t255 = M_CBRT6;
    double t256 = std::cbrt(t95);
    double t257 = t256 * t256;
    double t258 = 0.1e1 / t257;
    double t259 = t255 * t258;
    double t260 = t57 * t57;
    double t261 = t259 * t260;
    double t263 = 0.1e1 / t23 / t38;
    double t264 = t122 * t263;
    double t265 = t255 * t255;
    double t267 = 0.1e1 / t256 / t95;
    double t268 = t265 * t267;
    double t269 = t122 * t122;
    double t270 = t57 * t269;
    double t271 = t39 * t8;
    double t273 = 0.1e1 / t9 / t271;
    double t277 = std::exp(-0.20444604078896369094e0 * t268 * t270 * t273);
    double t279 = t261 * t264 * t277;
    double t282 = 0.1e1 + 0.27439371595564631661e-1 * t120 * t125 * t57 * t135 + 0.43341108700271342816e-1 * t254 * t279;
    double t283 = std::sqrt(std::sqrt(t282));
    double t285 = 0.1e1 - 0.1e1 / t283;
    double t287 = t114 * t285 + 0.1e1;
    double t288 = std::log(t287);
    double t290 = t97 * t106 * t288;
    double t375 = t354 * t374;
    double tvrho1 = t8 * t1046v + t290 - t34 + t375 + t90 + t92;
    return tvrho1;
  }
  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vsigma0(double rho0,
                        double rho1,
                        double sigma0,
                        double sigma1,
                        double sigma2,
                        double tau0,
                        double tau1)
  {
    MGGA_C_R2SCAN_VSIGMA0
    return tvsigma0;
  }
  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vsigma1(double rho0,
                        double rho1,
                        double sigma0,
                        double sigma1,
                        double sigma2,
                        double tau0,
                        double tau1)
  {
    MGGA_C_R2SCAN_VSIGMA1
    return tvsigma1;
  }
  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vtau0(double rho0,
                      double rho1,
                      double sigma0,
                      double sigma1,
                      double sigma2,
                      double tau0,
                      double tau1)
  {
    MGGA_C_R2SCAN_VTAU0
    return tvtau0;
  }
  static __device__ __attribute__((noinline)) double
  mgga_c_r2scan_vtau1(double rho0,
                      double rho1,
                      double sigma0,
                      double sigma1,
                      double sigma2,
                      double tau0,
                      double tau1)
  {
    MGGA_C_R2SCAN_VTAU1
    return tvtau1;
  }
  static __device__ __attribute__((noinline)) double
  mgga_x_r2scan_zk(double rho0,
                   double rho1,
                   double sigma0,
                   double sigma1,
                   double sigma2,
                   double tau0,
                   double tau1)
  {
    MGGA_X_R2SCAN_ZK
    return tzk0;
  }
  static __device__ __attribute__((noinline)) double
  mgga_x_r2scan_vrho0(double rho0,
                      double rho1,
                      double sigma0,
                      double sigma1,
                      double sigma2,
                      double tau0,
                      double tau1)
  {
    MGGA_X_R2SCAN_VRHO0
    return tvrho0;
  }
  static __device__ __attribute__((noinline)) double
  mgga_x_r2scan_vrho1(double rho0,
                      double rho1,
                      double sigma0,
                      double sigma1,
                      double sigma2,
                      double tau0,
                      double tau1)
  {
    MGGA_X_R2SCAN_VRHO1
    return tvrho1;
  }
  static __device__ __attribute__((noinline)) double
  mgga_x_r2scan_vsigma0(double rho0,
                        double rho1,
                        double sigma0,
                        double sigma1,
                        double sigma2,
                        double tau0,
                        double tau1)
  {
    MGGA_X_R2SCAN_VSIGMA0
    return tvsigma0;
  }
  static __device__ __attribute__((noinline)) double
  mgga_x_r2scan_vsigma2(double rho0,
                        double rho1,
                        double sigma0,
                        double sigma1,
                        double sigma2,
                        double tau0,
                        double tau1)
  {
    MGGA_X_R2SCAN_VSIGMA2
    return tvsigma2;
  }
  static __device__ __attribute__((noinline)) double
  mgga_x_r2scan_vtau0(double rho0,
                      double rho1,
                      double sigma0,
                      double sigma1,
                      double sigma2,
                      double tau0,
                      double tau1)
  {
    MGGA_X_R2SCAN_VTAU0
    return tvtau0;
  }
  static __device__ __attribute__((noinline)) double
  mgga_x_r2scan_vtau1(double rho0,
                      double rho1,
                      double sigma0,
                      double sigma1,
                      double sigma2,
                      double tau0,
                      double tau1)
  {
    MGGA_X_R2SCAN_VTAU1
    return tvtau1;
  }

#undef MGGA_C_R2SCAN
#define MGGA_C_R2SCAN                                                      \
  tzk0 = mgga_c_r2scan_zk(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1); \
  tvrho0 =                                                                 \
    mgga_c_r2scan_vrho0(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);   \
  tvrho1 =                                                                 \
    mgga_c_r2scan_vrho1(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);   \
  tvsigma0 =                                                               \
    mgga_c_r2scan_vsigma0(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1); \
  tvsigma1 =                                                               \
    mgga_c_r2scan_vsigma1(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1); \
  tvsigma2 = tvsigma0;                                                     \
  tvtau0 =                                                                 \
    mgga_c_r2scan_vtau0(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);   \
  tvtau1 = mgga_c_r2scan_vtau1(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);

#undef MGGA_X_R2SCAN
#define MGGA_X_R2SCAN                                                      \
  tzk0 = mgga_x_r2scan_zk(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1); \
  tvrho0 =                                                                 \
    mgga_x_r2scan_vrho0(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);   \
  tvrho1 =                                                                 \
    mgga_x_r2scan_vrho1(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);   \
  tvsigma0 =                                                               \
    mgga_x_r2scan_vsigma0(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1); \
  tvsigma1 = 0.0;                                                          \
  tvsigma2 =                                                               \
    mgga_x_r2scan_vsigma2(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1); \
  tvtau0 =                                                                 \
    mgga_x_r2scan_vtau0(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);   \
  tvtau1 = mgga_x_r2scan_vtau1(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);

#include <dftfe/exchangeCorrelationFunctionalEvaluation.def>
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
#include <dftfe/exchangeCorrelationFunctionalEvaluation.def>
} // namespace dftfe

#undef DFTFE_FUNCTIONALEVALUATOR_LDA_X
#undef DFTFE_FUNCTIONALEVALUATOR_LDA_C
#undef DFTFE_FUNCTIONALEVALUATOR_GGA_X
#undef DFTFE_FUNCTIONALEVALUATOR_GGA_C
#undef DFTFE_FUNCTIONALEVALUATOR_MGGA_X
#undef DFTFE_FUNCTIONALEVALUATOR_MGGA_C
