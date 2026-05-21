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
// @author Srinibas Nandi

// The below code snippet has been adapted from Libxc 7.0.0

#ifndef GGA_X_PBE
#define GGA_X_PBE                                                            \
  double t1, t2, t3, t5, t6, t7, t10, t11;                                   \
  double t14, t15, t16, t18, t19, t20, t21, t22;                             \
  double t23, t25, t26, t27, t28, t29, t30, t31;                             \
  double t32, t33, t34, t35, t36, t37, t39, t43;                             \
  double t48, t52, t53, t54, t56, t57, t58, t59;                             \
  double t61, t62, t63, t64, t65, t66, t68, t72;                             \
  double t77, t81;                                                           \
                                                                             \
  double t82, t83, t84, t86, t89, t90, t94, t95;                             \
  double t96, t99, t100, t101, t102, t103, t105, t106;                       \
  double t107, t109, t111, t115, t116, t118, t121, t122;                     \
  double t126, t129, t131, t135, t138, t139, t144;                           \
  double t146, t149, t150, t154, t155, t156, t158, t159;                     \
  double t160, t162, t164, t168, t171, t173, t176;                           \
  double t178, t181;                                                         \
                                                                             \
  struct gga_x_pbe                                                           \
  {                                                                          \
    double kappa = 0.8040;                                                   \
    double mu    = 0.2195149727645171;                                       \
  } params;                                                                  \
                                                                             \
                                                                             \
  t1   = rho0 <= DENS_THRESHOLD_X_PBE;                                       \
  t2   = M_CBRT3;                                                            \
  t3   = M_CBRTPI;                                                           \
  t5   = t2 / t3;                                                            \
  t6   = rho0 + rho1;                                                        \
  t7   = 0.1e1 / t6;                                                         \
  t10  = 0.2e1 * rho0 * t7 <= ZETA_THRESHOLD_X_PBE;                          \
  t11  = ZETA_THRESHOLD_X_PBE - 0.1e1;                                       \
  t14  = 0.2e1 * rho1 * t7 <= ZETA_THRESHOLD_X_PBE;                          \
  t15  = -t11;                                                               \
  t16  = rho0 - rho1;                                                        \
  t18  = my_piecewise5(t10, t11, t14, t15, t16 * t7);                        \
  t19  = 0.1e1 + t18;                                                        \
  t20  = t19 <= ZETA_THRESHOLD_X_PBE;                                        \
  t21  = std::pow(ZETA_THRESHOLD_X_PBE, 1.0 / 3.0);                          \
  t22  = t21 * ZETA_THRESHOLD_X_PBE;                                         \
  t23  = std::pow(t19, 1.0 / 3.0);                                           \
  t25  = my_piecewise3(t20, t22, t23 * t19);                                 \
  t26  = std::pow(t6, 1.0 / 3.0);                                            \
  t27  = t25 * t26;                                                          \
  t28  = M_CBRT6;                                                            \
  t29  = params.mu * t28;                                                    \
  t30  = M_PI * M_PI;                                                        \
  t31  = std::pow(t30, 1.0 / 3.0);                                           \
  t32  = t31 * t31;                                                          \
  t33  = 0.1e1 / t32;                                                        \
  t34  = t33 * sigma0;                                                       \
  t35  = rho0 * rho0;                                                        \
  t36  = std::pow(rho0, 1.0 / 3.0);                                          \
  t37  = t36 * t36;                                                          \
  t39  = 0.1e1 / t37 / t35;                                                  \
  t43  = params.kappa + t29 * t34 * t39 / 0.24e2;                            \
  t48  = 0.1e1 + params.kappa * (0.1e1 - params.kappa / t43);                \
  t52  = my_piecewise3(t1, 0, -0.3e1 / 0.8e1 * t5 * t27 * t48);              \
  t53  = rho1 <= DENS_THRESHOLD_X_PBE;                                       \
  t54  = -t16;                                                               \
  t56  = my_piecewise5(t14, t11, t10, t15, t54 * t7);                        \
  t57  = 0.1e1 + t56;                                                        \
  t58  = t57 <= ZETA_THRESHOLD_X_PBE;                                        \
  t59  = std::pow(t57, 1.0 / 3.0);                                           \
  t61  = my_piecewise3(t58, t22, t59 * t57);                                 \
  t62  = t61 * t26;                                                          \
  t63  = t33 * sigma2;                                                       \
  t64  = rho1 * rho1;                                                        \
  t65  = std::pow(rho1, 1.0 / 3.0);                                          \
  t66  = t65 * t65;                                                          \
  t68  = 0.1e1 / t66 / t64;                                                  \
  t72  = params.kappa + t29 * t63 * t68 / 0.24e2;                            \
  t77  = 0.1e1 + params.kappa * (0.1e1 - params.kappa / t72);                \
  t81  = my_piecewise3(t53, 0, -0.3e1 / 0.8e1 * t5 * t62 * t77);             \
  tzk0 = t52 + t81;                                                          \
                                                                             \
                                                                             \
  t82  = t6 * t6;                                                            \
  t83  = 0.1e1 / t82;                                                        \
  t84  = t16 * t83;                                                          \
  t86  = my_piecewise5(t10, 0, t14, 0, t7 - t84);                            \
  t89  = my_piecewise3(t20, 0, 0.4e1 / 0.3e1 * t23 * t86);                   \
  t90  = t89 * t26;                                                          \
  t94  = t26 * t26;                                                          \
  t95  = 0.1e1 / t94;                                                        \
  t96  = t25 * t95;                                                          \
  t99  = t5 * t96 * t48 / 0.8e1;                                             \
  t100 = params.kappa * params.kappa;                                        \
  t101 = t27 * t100;                                                         \
  t102 = t5 * t101;                                                          \
  t103 = t43 * t43;                                                          \
  t105 = 0.1e1 / t103 * params.mu;                                           \
  t106 = t105 * t28;                                                         \
  t107 = t35 * rho0;                                                         \
  t109 = 0.1e1 / t37 / t107;                                                 \
  t111 = t106 * t34 * t109;                                                  \
  t115 = my_piecewise3(                                                      \
    t1, 0, -0.3e1 / 0.8e1 * t5 * t90 * t48 - t99 + t102 * t111 / 0.24e2);    \
  t116   = t54 * t83;                                                        \
  t118   = my_piecewise5(t14, 0, t10, 0, -t7 - t116);                        \
  t121   = my_piecewise3(t58, 0, 0.4e1 / 0.3e1 * t59 * t118);                \
  t122   = t121 * t26;                                                       \
  t126   = t61 * t95;                                                        \
  t129   = t5 * t126 * t77 / 0.8e1;                                          \
  t131   = my_piecewise3(t53, 0, -0.3e1 / 0.8e1 * t5 * t122 * t77 - t129);   \
  tvrho0 = t52 + t81 + t6 * (t115 + t131);                                   \
                                                                             \
                                                                             \
  t135 = my_piecewise5(t10, 0, t14, 0, -t7 - t84);                           \
  t138 = my_piecewise3(t20, 0, 0.4e1 / 0.3e1 * t23 * t135);                  \
  t139 = t138 * t26;                                                         \
  t144 = my_piecewise3(t1, 0, -0.3e1 / 0.8e1 * t5 * t139 * t48 - t99);       \
  t146 = my_piecewise5(t14, 0, t10, 0, t7 - t116);                           \
  t149 = my_piecewise3(t58, 0, 0.4e1 / 0.3e1 * t59 * t146);                  \
  t150 = t149 * t26;                                                         \
  t154 = t62 * t100;                                                         \
  t155 = t5 * t154;                                                          \
  t156 = t72 * t72;                                                          \
  t158 = 0.1e1 / t156 * params.mu;                                           \
  t159 = t158 * t28;                                                         \
  t160 = t64 * rho1;                                                         \
  t162 = 0.1e1 / t66 / t160;                                                 \
  t164 = t159 * t63 * t162;                                                  \
  t168 = my_piecewise3(                                                      \
    t53, 0, -0.3e1 / 0.8e1 * t5 * t150 * t77 - t129 + t155 * t164 / 0.24e2); \
  tvrho1 = t52 + t81 + t6 * (t144 + t168);                                   \
                                                                             \
                                                                             \
  t171     = t28 * t33;                                                      \
  t173     = t105 * t171 * t39;                                              \
  t176     = my_piecewise3(t1, 0, -t102 * t173 / 0.64e2);                    \
  tvsigma0 = t6 * t176;                                                      \
                                                                             \
                                                                             \
  tvsigma1 = 0.e0;                                                           \
                                                                             \
                                                                             \
  t178     = t158 * t171 * t68;                                              \
  t181     = my_piecewise3(t53, 0, -t155 * t178 / 0.64e2);                   \
  tvsigma2 = t6 * t181;

#endif
