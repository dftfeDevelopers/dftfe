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

#ifndef LDA_X_SLATER
#define LDA_X_SLATER                                                          \
  double t1, t2, t3, t5, t6, t7, t8, t10;                                     \
  double t11, t12, t13, t14, t15, t19, t20, t24;                              \
  double t25, t26, t27, t29, t30, t31, t35, t39;                              \
  double t40;                                                                 \
                                                                              \
  double t41, t44, t45, t48, t49, t50, t51, t53;                              \
  double t58, t62, t63, t66, t68, t69, t72, t73;                              \
  double t74, t75, t76, t77, t78, t79, t83, t89;                              \
  double t91, t92, t95, t96, t97, t101, t106;                                 \
  double t107, t110, t112, t117, t122, t123;                                  \
                                                                              \
  struct lda_x_params                                                         \
  {                                                                           \
    double alpha = 1;                                                         \
  } params;                                                                   \
  t1   = rho0 <= DENS_THRESHOLD_X_SLATER;                                     \
  t2   = M_CBRT3;                                                             \
  t3   = M_CBRTPI;                                                            \
  t5   = t2 / t3;                                                             \
  t6   = rho0 + rho1;                                                         \
  t7   = 0.1e1 / t6;                                                          \
  t8   = rho0 * t7;                                                           \
  t10  = 0.2e1 * t8 <= ZETA_THRESHOLD_X_SLATER;                               \
  t11  = std::pow(ZETA_THRESHOLD_X_SLATER, 1.0 / 3.0);                        \
  t12  = t11 * ZETA_THRESHOLD_X_SLATER;                                       \
  t13  = M_CBRT2;                                                             \
  t14  = t13 * rho0;                                                          \
  t15  = std::pow(t8, 1.0 / 3.0);                                             \
  t19  = my_piecewise3(t10, t12, 0.2e1 * t14 * t7 * t15);                     \
  t20  = std::pow(t6, 1.0 / 3.0);                                             \
  t24  = my_piecewise3(t1, 0, -0.3e1 / 0.8e1 * t5 * t19 * t20);               \
  t25  = params.alpha * t24;                                                  \
  t26  = rho1 <= DENS_THRESHOLD_X_SLATER;                                     \
  t27  = rho1 * t7;                                                           \
  t29  = 0.2e1 * t27 <= ZETA_THRESHOLD_X_SLATER;                              \
  t30  = t13 * rho1;                                                          \
  t31  = std::pow(t27, 1.0 / 3.0);                                            \
  t35  = my_piecewise3(t29, t12, 0.2e1 * t30 * t7 * t31);                     \
  t39  = my_piecewise3(t26, 0, -0.3e1 / 0.8e1 * t5 * t35 * t20);              \
  t40  = params.alpha * t39;                                                  \
  tzk0 = t25 + t40;                                                           \
                                                                              \
  t41 = t13 * t7;                                                             \
  t44 = t6 * t6;                                                              \
  t45 = 0.1e1 / t44;                                                          \
  t48 = 0.2e1 * t14 * t45 * t15;                                              \
  t49 = t15 * t15;                                                            \
  t50 = 0.1e1 / t49;                                                          \
  t51 = t7 * t50;                                                             \
  t53 = -rho0 * t45 + t7;                                                     \
  t58 =                                                                       \
    my_piecewise3(t10,                                                        \
                  0,                                                          \
                  0.2e1 * t41 * t15 - t48 + 0.2e1 / 0.3e1 * t14 * t51 * t53); \
  t62    = t20 * t20;                                                         \
  t63    = 0.1e1 / t62;                                                       \
  t66    = t5 * t19 * t63 / 0.8e1;                                            \
  t68    = my_piecewise3(t1, 0, -0.3e1 / 0.8e1 * t5 * t58 * t20 - t66);       \
  t69    = params.alpha * t68;                                                \
  t72    = 0.2e1 * t30 * t45 * t31;                                           \
  t73    = rho1 * rho1;                                                       \
  t74    = t13 * t73;                                                         \
  t75    = t44 * t6;                                                          \
  t76    = 0.1e1 / t75;                                                       \
  t77    = t31 * t31;                                                         \
  t78    = 0.1e1 / t77;                                                       \
  t79    = t76 * t78;                                                         \
  t83    = my_piecewise3(t29, 0, -t72 - 0.2e1 / 0.3e1 * t74 * t79);           \
  t89    = t5 * t35 * t63 / 0.8e1;                                            \
  t91    = my_piecewise3(t26, 0, -0.3e1 / 0.8e1 * t5 * t83 * t20 - t89);      \
  t92    = params.alpha * t91;                                                \
  tvrho0 = t25 + t40 + t6 * (t69 + t92);                                      \
                                                                              \
  t95  = rho0 * rho0;                                                         \
  t96  = t13 * t95;                                                           \
  t97  = t76 * t50;                                                           \
  t101 = my_piecewise3(t10, 0, -t48 - 0.2e1 / 0.3e1 * t96 * t97);             \
  t106 = my_piecewise3(t1, 0, -0.3e1 / 0.8e1 * t5 * t101 * t20 - t66);        \
  t107 = params.alpha * t106;                                                 \
  t110 = t7 * t78;                                                            \
  t112 = -rho1 * t45 + t7;                                                    \
  t117 = my_piecewise3(                                                       \
    t29, 0, 0.2e1 * t41 * t31 - t72 + 0.2e1 / 0.3e1 * t30 * t110 * t112);     \
  t122   = my_piecewise3(t26, 0, -0.3e1 / 0.8e1 * t5 * t117 * t20 - t89);     \
  t123   = params.alpha * t122;                                               \
  tvrho1 = t25 + t40 + t6 * (t107 + t123);

#endif
