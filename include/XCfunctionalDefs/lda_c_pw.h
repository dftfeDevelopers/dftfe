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

#ifndef LDA_C_PW
#define LDA_C_PW                                                               \
  double t1, t2, t3, t4, t5, t6, t7, t8;                                       \
  double t9, t10, t11, t12, t13, t16, t18, t19;                                \
  double t20, t22, t23, t27, t30, t31, t35, t37;                               \
  double t38, t39, t40, t44, t45, t46, t47, t48;                               \
  double t49, t50, t51, t52, t53, t54, t55, t56;                               \
  double t57, t58, t59, t60, t61, t63, t64, t65;                               \
  double t66, t68, t69, t70, t73, t74, t75, t76;                               \
  double t77, t80, t82, t83, t87, t90, t95, t96;                               \
  double t97, t98, t102, t103, t105, t106, t107, t110;                         \
  double t112, t113, t117, t120, t125, t126, t127, t128;                       \
  double t132, t133, t134, t135, t138, t139, t140, t143;                       \
  double t145;                                                                 \
                                                                               \
  double t147, t149, t152, t153, t154, t155, t156, t157;                       \
  double t159, t160, t165, t167, t173, t174, t175, t176;                       \
  double t177, t178, t179, t180, t181, t182, t183, t184;                       \
  double t185, t186, t187, t188, t191, t192, t195, t197;                       \
  double t198, t199, t201, t206, t207, t208, t210, t216;                       \
  double t222, t223, t224, t226, t227, t228, t232, t233;                       \
  double t234, t236, t242, t248, t249, t251, t253, t254;                       \
  double t255, t257, t258, t259, t260, t263, t264, t265;                       \
  double t266, t268, t269, t270, t273, t276, t277;                             \
  double t280, t282, t283, t284, t286, t287;                                   \
                                                                               \
  struct lda_c_pw_params                                                       \
  {                                                                            \
    double pp[3]{1, 1, 1};                                                     \
    double a[3]{0.031091, 0.015545, 0.016887};                                 \
    double alpha1[3]{0.21370, 0.20548, 0.11125};                               \
    double beta1[3]{7.5957, 14.1189, 10.357};                                  \
    double beta2[3]{3.5876, 6.1977, 3.6231};                                   \
    double beta3[3]{1.6382, 3.3662, 0.88026};                                  \
    double beta4[3]{0.49294, 0.62517, 0.49671};                                \
    double fz20 = 1.709921;                                                    \
  } params;                                                                    \
                                                                               \
  t1   = params.a[0];                                                          \
  t2   = params.alpha1[0];                                                     \
  t3   = M_CBRT3;                                                              \
  t4   = t2 * t3;                                                              \
  t5   = 0.1e1 / M_PI;                                                         \
  t6   = std::pow(t5, 1.0 / 3.0);                                              \
  t7   = M_CBRT4;                                                              \
  t8   = t7 * t7;                                                              \
  t9   = t6 * t8;                                                              \
  t10  = rho0 + rho1;                                                          \
  t11  = std::pow(t10, 1.0 / 3.0);                                             \
  t12  = 0.1e1 / t11;                                                          \
  t13  = t9 * t12;                                                             \
  t16  = 0.1e1 + t4 * t13 / 0.4e1;                                             \
  t18  = 0.1e1 / t1;                                                           \
  t19  = params.beta1[0];                                                      \
  t20  = t3 * t6;                                                              \
  t22  = t20 * t8 * t12;                                                       \
  t23  = sqrt(t22);                                                            \
  t27  = params.beta2[0] * t3;                                                 \
  t30  = params.beta3[0];                                                      \
  t31  = std::pow(t22, 3.0 / 2.0);                                             \
  t35  = t22 / 0.4e1;                                                          \
  t37  = params.pp[0] + 0.1e1;                                                 \
  t38  = std::pow(t35, t37);                                                   \
  t39  = params.beta4[0] * t38;                                                \
  t40  = t19 * t23 / 0.2e1 + t27 * t13 / 0.4e1 + 0.125e0 * t30 * t31 + t39;    \
  t44  = 0.1e1 + t18 / t40 / 0.2e1;                                            \
  t45  = std::log(t44);                                                        \
  t46  = t1 * t16 * t45;                                                       \
  t47  = 0.2e1 * t46;                                                          \
  t48  = rho0 - rho1;                                                          \
  t49  = t48 * t48;                                                            \
  t50  = t49 * t49;                                                            \
  t51  = t10 * t10;                                                            \
  t52  = t51 * t51;                                                            \
  t53  = 0.1e1 / t52;                                                          \
  t54  = t50 * t53;                                                            \
  t55  = 0.1e1 / t10;                                                          \
  t56  = t48 * t55;                                                            \
  t57  = 0.1e1 + t56;                                                          \
  t58  = t57 <= ZETA_THRESHOLD_C_PW;                                           \
  t59  = std::pow(ZETA_THRESHOLD_C_PW, 1.0 / 3.0);                             \
  t60  = t59 * ZETA_THRESHOLD_C_PW;                                            \
  t61  = std::pow(t57, 1.0 / 3.0);                                             \
  t63  = my_piecewise3(t58, t60, t61 * t57);                                   \
  t64  = 0.1e1 - t56;                                                          \
  t65  = t64 <= ZETA_THRESHOLD_C_PW;                                           \
  t66  = std::pow(t64, 1.0 / 3.0);                                             \
  t68  = my_piecewise3(t65, t60, t66 * t64);                                   \
  t69  = t63 + t68 - 0.2e1;                                                    \
  t70  = M_CBRT2;                                                              \
  t73  = 0.1e1 / (0.2e1 * t70 - 0.2e1);                                        \
  t74  = t69 * t73;                                                            \
  t75  = params.a[1];                                                          \
  t76  = params.alpha1[1];                                                     \
  t77  = t76 * t3;                                                             \
  t80  = 0.1e1 + t77 * t13 / 0.4e1;                                            \
  t82  = 0.1e1 / t75;                                                          \
  t83  = params.beta1[1];                                                      \
  t87  = params.beta2[1] * t3;                                                 \
  t90  = params.beta3[1];                                                      \
  t95  = params.pp[1] + 0.1e1;                                                 \
  t96  = std::pow(t35, t95);                                                   \
  t97  = params.beta4[1] * t96;                                                \
  t98  = t83 * t23 / 0.2e1 + t87 * t13 / 0.4e1 + 0.125e0 * t90 * t31 + t97;    \
  t102 = 0.1e1 + t82 / t98 / 0.2e1;                                            \
  t103 = std::log(t102);                                                       \
  t105 = params.a[2];                                                          \
  t106 = params.alpha1[2];                                                     \
  t107 = t106 * t3;                                                            \
  t110 = 0.1e1 + t107 * t13 / 0.4e1;                                           \
  t112 = 0.1e1 / t105;                                                         \
  t113 = params.beta1[2];                                                      \
  t117 = params.beta2[2] * t3;                                                 \
  t120 = params.beta3[2];                                                      \
  t125 = params.pp[2] + 0.1e1;                                                 \
  t126 = std::pow(t35, t125);                                                  \
  t127 = params.beta4[2] * t126;                                               \
  t128 =                                                                       \
    t113 * t23 / 0.2e1 + t117 * t13 / 0.4e1 + 0.125e0 * t120 * t31 + t127;     \
  t132 = 0.1e1 + t112 / t128 / 0.2e1;                                          \
  t133 = std::log(t132);                                                       \
  t134 = 0.1e1 / params.fz20;                                                  \
  t135 = t133 * t134;                                                          \
  t138 = -0.2e1 * t75 * t80 * t103 - 0.2e1 * t105 * t110 * t135 + 0.2e1 * t46; \
  t139 = t74 * t138;                                                           \
  t140 = t54 * t139;                                                           \
  t143 = t110 * t133 * t134;                                                   \
  t145 = 0.2e1 * t74 * t105 * t143;                                            \
  tzk0 = -t47 + t140 + t145;                                                   \
                                                                               \
  t147 = t1 * t2 * t3;                                                         \
  t149 = 0.1e1 / t11 / t10;                                                    \
  t152 = t147 * t9 * t149 * t45;                                               \
  t153 = t152 / 0.6e1;                                                         \
  t154 = t40 * t40;                                                            \
  t155 = 0.1e1 / t154;                                                         \
  t156 = t16 * t155;                                                           \
  t157 = 0.1e1 / t23;                                                          \
  t159 = t19 * t157 * t3;                                                      \
  t160 = t9 * t149;                                                            \
  t165 = sqrt(t22);                                                            \
  t167 = t30 * t165 * t3;                                                      \
  t173 = -t159 * t160 / 0.12e2 - t27 * t160 / 0.12e2 -                         \
         0.625e-1 * t167 * t160 - t39 * t37 * t55 / 0.3e1;                     \
  t174 = 0.1e1 / t44;                                                          \
  t175 = t173 * t174;                                                          \
  t176 = t156 * t175;                                                          \
  t177 = t49 * t48;                                                            \
  t178 = t177 * t53;                                                           \
  t179 = t178 * t139;                                                          \
  t180 = 0.4e1 * t179;                                                         \
  t181 = t52 * t10;                                                            \
  t182 = 0.1e1 / t181;                                                         \
  t183 = t50 * t182;                                                           \
  t184 = t183 * t139;                                                          \
  t185 = 0.4e1 * t184;                                                         \
  t186 = 0.1e1 / t51;                                                          \
  t187 = t48 * t186;                                                           \
  t188 = t55 - t187;                                                           \
  t191 = my_piecewise3(t58, 0, 0.4e1 / 0.3e1 * t61 * t188);                    \
  t192 = -t188;                                                                \
  t195 = my_piecewise3(t65, 0, 0.4e1 / 0.3e1 * t66 * t192);                    \
  t197 = (t191 + t195) * t73;                                                  \
  t198 = t197 * t138;                                                          \
  t199 = t54 * t198;                                                           \
  t201 = t75 * t76 * t3;                                                       \
  t206 = t98 * t98;                                                            \
  t207 = 0.1e1 / t206;                                                         \
  t208 = t80 * t207;                                                           \
  t210 = t83 * t157 * t3;                                                      \
  t216 = t90 * t165 * t3;                                                      \
  t222 = -t210 * t160 / 0.12e2 - t87 * t160 / 0.12e2 -                         \
         0.625e-1 * t216 * t160 - t97 * t95 * t55 / 0.3e1;                     \
  t223 = 0.1e1 / t102;                                                         \
  t224 = t222 * t223;                                                          \
  t226 = t105 * t106;                                                          \
  t227 = t226 * t20;                                                           \
  t228 = t8 * t149;                                                            \
  t232 = t128 * t128;                                                          \
  t233 = 0.1e1 / t232;                                                         \
  t234 = t110 * t233;                                                          \
  t236 = t113 * t157 * t3;                                                     \
  t242 = t120 * t165 * t3;                                                     \
  t248 = -t236 * t160 / 0.12e2 - t117 * t160 / 0.12e2 -                        \
         0.625e-1 * t242 * t160 - t127 * t125 * t55 / 0.3e1;                   \
  t249 = 0.1e1 / t132;                                                         \
  t251 = t248 * t249 * t134;                                                   \
  t253 = t201 * t9 * t149 * t103 / 0.6e1 + t208 * t224 - t153 - t176 +         \
         t227 * t228 * t135 / 0.6e1 + t234 * t251;                             \
  t254 = t74 * t253;                                                           \
  t255 = t54 * t254;                                                           \
  t257 = t197 * t105 * t143;                                                   \
  t258 = 0.2e1 * t257;                                                         \
  t259 = t226 * t3;                                                            \
  t260 = t74 * t259;                                                           \
  t263 = t9 * t149 * t133 * t134;                                              \
  t264 = t260 * t263;                                                          \
  t265 = t264 / 0.6e1;                                                         \
  t266 = t74 * t110;                                                           \
  t268 = t249 * t134;                                                          \
  t269 = t233 * t248 * t268;                                                   \
  t270 = t266 * t269;                                                          \
  tvrho0 =                                                                     \
    -t47 + t140 + t145 +                                                       \
    t10 * (t153 + t176 + t180 - t185 + t199 + t255 + t258 - t265 - t270);      \
                                                                               \
  t273 = -t55 - t187;                                                          \
  t276 = my_piecewise3(t58, 0, 0.4e1 / 0.3e1 * t61 * t273);                    \
  t277 = -t273;                                                                \
  t280 = my_piecewise3(t65, 0, 0.4e1 / 0.3e1 * t66 * t277);                    \
  t282 = (t276 + t280) * t73;                                                  \
  t283 = t282 * t138;                                                          \
  t284 = t54 * t283;                                                           \
  t286 = t282 * t105 * t143;                                                   \
  t287 = 0.2e1 * t286;                                                         \
  tvrho1 =                                                                     \
    -t47 + t140 + t145 +                                                       \
    t10 * (t153 + t176 - t180 - t185 + t284 + t255 + t287 - t265 - t270);
#endif
