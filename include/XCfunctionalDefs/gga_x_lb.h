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

#ifndef GGA_X_LB
#define GGA_X_LB                                                   \
  double t1, t4, t5, t6, t9, t10, t11, t13;                        \
  double t14, t15, t16, t17, t18, t20, t21, t23;                   \
  double t24, t25, t28, t29, t33, t34, t37, t38;                   \
  double t39, t40, t42, t43, t44, t45, t46;                        \
  double t47, t49, t50, t52, t53, t54, t57, t58;                   \
  double t62, t63, t66, t67;                                       \
                                                                   \
                                                                   \
  struct gga_x_lb                                                  \
  {                                                                \
    double alpha = 1.0;                                            \
    double beta  = 0.05;                                           \
    double gamma = 1.0;                                            \
  } params;                                                        \
                                                                   \
  t1     = M_CBRT3;                                                \
  t4     = POW_1_3(0.1e1 / M_PI);                                  \
  t5     = M_CBRT4;                                                \
  t6     = t5 * t5;                                                \
  t9     = params.alpha * t1 * t4 * t6 / 0.2e1;                    \
  t10    = sqrt(sigma0);                                           \
  t11    = POW_1_3(rho0);                                          \
  t13    = 0.1e1 / t11 / rho0;                                     \
  t14    = t10 * t13;                                              \
  t15    = t14 < 0.3e3;                                            \
  t16    = params.beta * sigma0;                                   \
  t17    = rho0 * rho0;                                            \
  t18    = t11 * t11;                                              \
  t20    = 0.1e1 / t18 / t17;                                      \
  t21    = params.beta * t10;                                      \
  t23    = params.gamma * t10 * t13;                               \
  t24    = log(t23 + sqrt(t23 * t23 + 0.1e1));                     \
  t25    = t13 * t24;                                              \
  t28    = 0.3e1 * t21 * t25 + 0.1e1;                              \
  t29    = 0.1e1 / t28;                                            \
  t33    = log(0.2e1 * t23);                                       \
  t34    = 0.1e1 / t33;                                            \
  t37    = my_piecewise3(t15, t16 * t20 * t29, t14 * t34 / 0.3e1); \
  t38    = -t9 - t37;                                              \
  tvrho0 = t38 * t11;                                              \
                                                                   \
  t39    = sqrt(sigma2);                                           \
  t40    = POW_1_3(rho1);                                          \
  t42    = 0.1e1 / t40 / rho1;                                     \
  t43    = t39 * t42;                                              \
  t44    = t43 < 0.3e3;                                            \
  t45    = params.beta * sigma2;                                   \
  t46    = rho1 * rho1;                                            \
  t47    = t40 * t40;                                              \
  t49    = 0.1e1 / t47 / t46;                                      \
  t50    = params.beta * t39;                                      \
  t52    = params.gamma * t39 * t42;                               \
  t53    = log(t52 + sqrt(t52 * t52 + 0.1e1));                     \
  t54    = t42 * t53;                                              \
  t57    = 0.3e1 * t50 * t54 + 0.1e1;                              \
  t58    = 0.1e1 / t57;                                            \
  t62    = log(0.2e1 * t52);                                       \
  t63    = 0.1e1 / t62;                                            \
  t66    = my_piecewise3(t44, t45 * t49 * t58, t43 * t63 / 0.3e1); \
  t67    = -t9 - t66;                                              \
  tvrho1 = t67 * t40;

#endif
