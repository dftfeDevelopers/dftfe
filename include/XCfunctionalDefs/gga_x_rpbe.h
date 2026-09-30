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

#ifndef GGA_X_RPBE
#define GGA_X_RPBE                                                           \
  double t1, t2, t3, t5, t6, t7, t10, t11;                                   \
  double t14, t15, t16, t18, t19, t20, t21, t22;                             \
  double t23, t25, t26, t27, t28, t29, t30, t31;                             \
  double t32, t33, t34, t35, t36, t37, t39, t41;                             \
  double t45, t48, t52, t53, t54, t56, t57, t58;                             \
  double t59, t61, t62, t63, t64, t65, t67, t72;                             \
  double t75, t79;                                                           \
                                                                             \
  double t80, t81, t82, t84, t87, t88, t92, t93;                             \
  double t94, t97, t99, t100, t101, t103, t106, t110;                        \
  double t111, t113, t116, t117, t121, t124, t126;                           \
  double t130, t133, t134, t139, t141, t144, t145, t150;                     \
  double t151, t153, t156, t160, t163, t166, t169;                           \
  double t170, t173, t176;                                                   \
                                                                             \
  struct gga_x_rpbe_params                                                   \
  {                                                                          \
    double rpbe_kappa = 0.804;                                               \
    double rpbe_mu    = 0.2195149727645170912;                               \
  } params;                                                                  \
                                                                             \
  t1   = rho0 <= DENS_THRESHOLD_X_RPBE;                                      \
  t2   = M_CBRT3;                                                            \
  t3   = M_CBRTPI;                                                           \
  t5   = t2 / t3;                                                            \
  t6   = rho0 + rho1;                                                        \
  t7   = 0.1e1 / t6;                                                         \
  t10  = 0.2e1 * rho0 * t7 <= ZETA_THRESHOLD_X_RPBE;                         \
  t11  = ZETA_THRESHOLD_X_RPBE - 0.1e1;                                      \
  t14  = 0.2e1 * rho1 * t7 <= ZETA_THRESHOLD_X_RPBE;                         \
  t15  = -t11;                                                               \
  t16  = rho0 - rho1;                                                        \
  t18  = my_piecewise5(t10, t11, t14, t15, t16 * t7);                        \
  t19  = 0.1e1 + t18;                                                        \
  t20  = t19 <= ZETA_THRESHOLD_X_RPBE;                                       \
  t21  = POW_1_3(ZETA_THRESHOLD_X_RPBE);                                     \
  t22  = t21 * ZETA_THRESHOLD_X_RPBE;                                        \
  t23  = POW_1_3(t19);                                                       \
  t25  = my_piecewise3(t20, t22, t23 * t19);                                 \
  t26  = POW_1_3(t6);                                                        \
  t27  = t25 * t26;                                                          \
  t28  = M_CBRT6;                                                            \
  t29  = params.rpbe_mu * t28;                                               \
  t30  = M_PI * M_PI;                                                        \
  t31  = POW_1_3(t30);                                                       \
  t32  = t31 * t31;                                                          \
  t33  = 0.1e1 / t32;                                                        \
  t34  = t29 * t33;                                                          \
  t35  = rho0 * rho0;                                                        \
  t36  = POW_1_3(rho0);                                                      \
  t37  = t36 * t36;                                                          \
  t39  = 0.1e1 / t37 / t35;                                                  \
  t41  = 0.1e1 / params.rpbe_kappa;                                          \
  t45  = exp(-t34 * sigma0 * t39 * t41 / 0.24e2);                            \
  t48  = 0.1e1 + params.rpbe_kappa * (0.1e1 - t45);                          \
  t52  = my_piecewise3(t1, 0, -0.3e1 / 0.8e1 * t5 * t27 * t48);              \
  t53  = rho1 <= DENS_THRESHOLD_X_RPBE;                                      \
  t54  = -t16;                                                               \
  t56  = my_piecewise5(t14, t11, t10, t15, t54 * t7);                        \
  t57  = 0.1e1 + t56;                                                        \
  t58  = t57 <= ZETA_THRESHOLD_X_RPBE;                                       \
  t59  = POW_1_3(t57);                                                       \
  t61  = my_piecewise3(t58, t22, t59 * t57);                                 \
  t62  = t61 * t26;                                                          \
  t63  = rho1 * rho1;                                                        \
  t64  = POW_1_3(rho1);                                                      \
  t65  = t64 * t64;                                                          \
  t67  = 0.1e1 / t65 / t63;                                                  \
  t72  = exp(-t34 * sigma2 * t67 * t41 / 0.24e2);                            \
  t75  = 0.1e1 + params.rpbe_kappa * (0.1e1 - t72);                          \
  t79  = my_piecewise3(t53, 0, -0.3e1 / 0.8e1 * t5 * t62 * t75);             \
  tzk0 = t52 + t79;                                                          \
                                                                             \
  t80  = t6 * t6;                                                            \
  t81  = 0.1e1 / t80;                                                        \
  t82  = t16 * t81;                                                          \
  t84  = my_piecewise5(t10, 0, t14, 0, t7 - t82);                            \
  t87  = my_piecewise3(t20, 0, 0.4e1 / 0.3e1 * t23 * t84);                   \
  t88  = t87 * t26;                                                          \
  t92  = t26 * t26;                                                          \
  t93  = 0.1e1 / t92;                                                        \
  t94  = t25 * t93;                                                          \
  t97  = t5 * t94 * t48 / 0.8e1;                                             \
  t99  = t5 * t27 * params.rpbe_mu;                                          \
  t100 = t28 * t33;                                                          \
  t101 = t35 * rho0;                                                         \
  t103 = 0.1e1 / t37 / t101;                                                 \
  t106 = t100 * sigma0 * t103 * t45;                                         \
  t110 = my_piecewise3(                                                      \
    t1, 0, -0.3e1 / 0.8e1 * t5 * t88 * t48 - t97 + t99 * t106 / 0.24e2);     \
  t111   = t54 * t81;                                                        \
  t113   = my_piecewise5(t14, 0, t10, 0, -t7 - t111);                        \
  t116   = my_piecewise3(t58, 0, 0.4e1 / 0.3e1 * t59 * t113);                \
  t117   = t116 * t26;                                                       \
  t121   = t61 * t93;                                                        \
  t124   = t5 * t121 * t75 / 0.8e1;                                          \
  t126   = my_piecewise3(t53, 0, -0.3e1 / 0.8e1 * t5 * t117 * t75 - t124);   \
  tvrho0 = t52 + t79 + t6 * (t110 + t126);                                   \
                                                                             \
  t130 = my_piecewise5(t10, 0, t14, 0, -t7 - t82);                           \
  t133 = my_piecewise3(t20, 0, 0.4e1 / 0.3e1 * t23 * t130);                  \
  t134 = t133 * t26;                                                         \
  t139 = my_piecewise3(t1, 0, -0.3e1 / 0.8e1 * t5 * t134 * t48 - t97);       \
  t141 = my_piecewise5(t14, 0, t10, 0, t7 - t111);                           \
  t144 = my_piecewise3(t58, 0, 0.4e1 / 0.3e1 * t59 * t141);                  \
  t145 = t144 * t26;                                                         \
  t150 = t5 * t62 * params.rpbe_mu;                                          \
  t151 = t63 * rho1;                                                         \
  t153 = 0.1e1 / t65 / t151;                                                 \
  t156 = t100 * sigma2 * t153 * t72;                                         \
  t160 = my_piecewise3(                                                      \
    t53, 0, -0.3e1 / 0.8e1 * t5 * t145 * t75 - t124 + t150 * t156 / 0.24e2); \
  tvrho1 = t52 + t79 + t6 * (t139 + t160);                                   \
                                                                             \
  t163     = t5 * t27;                                                       \
  t166     = t29 * t33 * t39 * t45;                                          \
  t169     = my_piecewise3(t1, 0, -t163 * t166 / 0.64e2);                    \
  tvsigma0 = t6 * t169;                                                      \
                                                                             \
                                                                             \
  tvsigma1 = 0.e0;                                                           \
                                                                             \
                                                                             \
  t170     = t5 * t62;                                                       \
  t173     = t29 * t33 * t67 * t72;                                          \
  t176     = my_piecewise3(t53, 0, -t170 * t173 / 0.64e2);                   \
  tvsigma2 = t6 * t176;

#endif
