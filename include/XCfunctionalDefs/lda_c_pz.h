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

#ifndef LDA_C_PZ
#define LDA_C_PZ                                                       \
  double t1, t2, t3, t5, t6, t7, t8, t9;                               \
  double t10, t11, t12, t13, t14, t15, t16, t20;                       \
  double t21, t22, t25, t28, t29, t33, t34, t35;                       \
  double t39, t43, t44, t45, t49, t52, t55, t59;                       \
  double t60, t64, t68, t69, t70, t71, t72, t73;                       \
  double t74, t75, t76, t77, t79, t80, t81, t82;                       \
  double t84, t85, t87, t90, t91;                                      \
                                                                       \
  double t92, t94, t95, t97, t99, t100, t104, t109;                    \
  double t117, t118, t120, t122, t126, t137, t138, t140;               \
  double t141, t142, t143, t144, t147, t148, t151, t152;               \
  double t154, t157, t160, t161, t164, t165, t167;                     \
                                                                       \
  struct lda_c_pz_params                                               \
  {                                                                    \
    double gamma0 = -0.1423;                                           \
    double gamma1 = -0.0843;                                           \
    double beta10 = 1.0529;                                            \
    double beta11 = 1.3981;                                            \
    double beta20 = 0.3334;                                            \
    double beta21 = 0.2611;                                            \
    double a0     = 0.0311;                                            \
    double a1     = 0.01555;                                           \
    double b0     = -0.048;                                            \
    double b1     = -0.0269;                                           \
    double c0     = 0.0020;                                            \
    double c1     = 0.0007;                                            \
    double d0     = -0.0116;                                           \
    double d1     = -0.0048;                                           \
  } params;                                                            \
                                                                       \
  t1   = M_CBRT3;                                                      \
  t2   = 0.1e1 / M_PI;                                                 \
  t3   = POW_1_3(t2);                                                  \
  t5   = M_CBRT4;                                                      \
  t6   = t5 * t5;                                                      \
  t7   = rho0 + rho1;                                                  \
  t8   = POW_1_3(t7);                                                  \
  t9   = 0.1e1 / t8;                                                   \
  t10  = t6 * t9;                                                      \
  t11  = t1 * t3 * t10;                                                \
  t12  = t11 / 0.4e1;                                                  \
  t13  = 0.1e1 <= t12;                                                 \
  t14  = params.gamma0;                                                \
  t15  = params.beta10;                                                \
  t16  = sqrt(t11);                                                    \
  t20  = params.beta20 * t1;                                           \
  t21  = t3 * t6;                                                      \
  t22  = t21 * t9;                                                     \
  t25  = 0.1e1 + t15 * t16 / 0.2e1 + t20 * t22 / 0.4e1;                \
  t28  = params.a0;                                                    \
  t29  = log(t12);                                                     \
  t33  = params.c0 * t1;                                               \
  t34  = t33 * t3;                                                     \
  t35  = t10 * t29;                                                    \
  t39  = params.d0 * t1;                                               \
  t43  = my_piecewise3(t13,                                            \
                      t14 / t25,                                      \
                      t28 * t29 + params.b0 + t34 * t35 / 0.4e1 +     \
                        t39 * t22 / 0.4e1);                           \
  t44  = params.gamma1;                                                \
  t45  = params.beta11;                                                \
  t49  = params.beta21 * t1;                                           \
  t52  = 0.1e1 + t45 * t16 / 0.2e1 + t49 * t22 / 0.4e1;                \
  t55  = params.a1;                                                    \
  t59  = params.c1 * t1;                                               \
  t60  = t59 * t3;                                                     \
  t64  = params.d1 * t1;                                               \
  t68  = my_piecewise3(t13,                                            \
                      t44 / t52,                                      \
                      t55 * t29 + params.b1 + t60 * t35 / 0.4e1 +     \
                        t64 * t22 / 0.4e1);                           \
  t69  = t68 - t43;                                                    \
  t70  = rho0 - rho1;                                                  \
  t71  = 0.1e1 / t7;                                                   \
  t72  = t70 * t71;                                                    \
  t73  = 0.1e1 + t72;                                                  \
  t74  = t73 <= ZETA_THRESHOLD_C_PZ;                                   \
  t75  = POW_1_3(ZETA_THRESHOLD_C_PZ);                                 \
  t76  = t75 * ZETA_THRESHOLD_C_PZ;                                    \
  t77  = POW_1_3(t73);                                                 \
  t79  = my_piecewise3(t74, t76, t77 * t73);                           \
  t80  = 0.1e1 - t72;                                                  \
  t81  = t80 <= ZETA_THRESHOLD_C_PZ;                                   \
  t82  = POW_1_3(t80);                                                 \
  t84  = my_piecewise3(t81, t76, t82 * t80);                           \
  t85  = t79 + t84 - 0.2e1;                                            \
  t87  = M_CBRT2;                                                      \
  t90  = 0.1e1 / (0.2e1 * t87 - 0.2e1);                                \
  t91  = t69 * t85 * t90;                                              \
  tzk0 = t43 + t91;                                                    \
                                                                       \
  t92    = t25 * t25;                                                  \
  t94    = t14 / t92;                                                  \
  t95    = 0.1e1 / t16;                                                \
  t97    = t15 * t95 * t1;                                             \
  t99    = 0.1e1 / t8 / t7;                                            \
  t100   = t21 * t99;                                                  \
  t104   = -t20 * t100 / 0.12e2 - t97 * t100 / 0.12e2;                 \
  t109   = t6 * t99 * t29;                                             \
  t117   = my_piecewise3(t13,                                          \
                       -t94 * t104,                                  \
                       -t28 * t71 / 0.3e1 - t34 * t109 / 0.12e2 -    \
                         t33 * t100 / 0.12e2 - t39 * t100 / 0.12e2); \
  t118   = t52 * t52;                                                  \
  t120   = t44 / t118;                                                 \
  t122   = t45 * t95 * t1;                                             \
  t126   = -t122 * t100 / 0.12e2 - t49 * t100 / 0.12e2;                \
  t137   = my_piecewise3(t13,                                          \
                       -t120 * t126,                                 \
                       -t55 * t71 / 0.3e1 - t60 * t109 / 0.12e2 -    \
                         t59 * t100 / 0.12e2 - t64 * t100 / 0.12e2); \
  t138   = t137 - t117;                                                \
  t140   = t138 * t85 * t90;                                           \
  t141   = t7 * t7;                                                    \
  t142   = 0.1e1 / t141;                                               \
  t143   = t70 * t142;                                                 \
  t144   = t71 - t143;                                                 \
  t147   = my_piecewise3(t74, 0, 0.4e1 / 0.3e1 * t77 * t144);          \
  t148   = -t144;                                                      \
  t151   = my_piecewise3(t81, 0, 0.4e1 / 0.3e1 * t82 * t148);          \
  t152   = t147 + t151;                                                \
  t154   = t69 * t152 * t90;                                           \
  tvrho0 = t43 + t91 + t7 * (t117 + t140 + t154);                      \
                                                                       \
  t157   = -t71 - t143;                                                \
  t160   = my_piecewise3(t74, 0, 0.4e1 / 0.3e1 * t77 * t157);          \
  t161   = -t157;                                                      \
  t164   = my_piecewise3(t81, 0, 0.4e1 / 0.3e1 * t82 * t161);          \
  t165   = t160 + t164;                                                \
  t167   = t69 * t165 * t90;                                           \
  tvrho1 = t43 + t91 + t7 * (t117 + t140 + t167);

#endif
