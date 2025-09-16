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

#ifndef MGGA_X_SCAN
#define MGGA_X_SCAN                                                            \
  double t2, t3, t4, t6, t7, t8, t11, t12;                                     \
  double t15, t16, t17, t19, t20, t21, t22, t23;                               \
  double t24, t26, t27, t28, t29, t30, t31, t32;                               \
  double t33, t34, t35, t36, t37, t38, t39, t40;                               \
  double t41, t45, t46, t47, t48, t49, t50, t51;                               \
  double t52, t53, t55, t56, t57, t58, t59, t62;                               \
  double t66, t67, t70, t71, t77, t78, t80, t82;                               \
  double t85, t86, t87, t92, t93, t94, t97, t98;                               \
  double t99, t100, t101, t102, t103, t105, t106, t107;                        \
  double t110, t113, t114, t115, t116, t119, t121, t122;                       \
  double t123, t126, t127, t128, t129, t130, t131, t132;                       \
  double t133, t135, t136, t140, t141, t142, t145, t146;                       \
  double t147, t149, t150, t151, t152, t154, t155, t156;                       \
  double t157, t158, t159, t160, t161, t162, t164, t165;                       \
  double t166, t168, t169, t170, t171, t174, t180, t181;                       \
  double t187, t188, t190, t192, t195, t196, t197, t202;                       \
  double t203, t204, t205, t206, t207, t208, t209, t211;                       \
  double t212, t213, t214, t215, t218, t220, t221, t222;                       \
  double t225, t226, t227, t228, t229, t231, t232, t236;                       \
  double t237, t238, t241;                                                     \
                                                                               \
  double t242, t243, t244, t246, t249, t250, t253, t254;                       \
  double t255, t256, t258, t259, t260, t262, t263, t265;                       \
  double t266, t269, t271, t276, t277, t278, t279, t280;                       \
  double t281, t282, t283, t294, t296, t299, t303, t306;                       \
  double t307, t309, t311, t312, t315, t316, t317, t319;                       \
  double t320, t321, t322, t323, t324, t325, t329, t330;                       \
  double t333, t334, t335, t338, t339, t340, t341, t342;                       \
  double t344, t345, t346, t348, t351, t355, t356, t358;                       \
  double t361, t362, t365, t366, t368, t370, t374;                             \
  double t377, t378, t382, t384, t387, t388, t391, t393;                       \
  double t394, t396, t397, t400, t402, t407, t408, t409;                       \
  double t410, t421, t423, t426, t430, t433, t434, t436;                       \
  double t438, t439, t442, t443, t444, t446, t447, t448;                       \
  double t449, t450, t451, t455, t456, t459, t460, t461;                       \
  double t464, t466, t467, t468, t470, t473, t477;                             \
  double t481, t487, t495, t496, t499, t501, t504, t505;                       \
  double t507, t508, t509, t511, t513, t514, t515, t516;                       \
  double t520, t521, t524, t525, t526, t529, t532, t536;                       \
  double t538, t544, t552, t553, t556, t558;                                   \
  double t561, t562, t564, t565, t566, t568, t570, t571;                       \
  double t572, t573, t577, t578, t581, t582, t583, t586;                       \
  double t589, t593, tvlapl0, tvlapl1, t594, t599, t600;                       \
  double t606, t607, t608, t612, t613, t614, t615, t619;                       \
  double t620, t623, t624, t625, t628, t629, t634;                             \
  double t635, t641, t642, t643, t647, t648, t649, t650;                       \
  double t654, t655, t658, t659, t660, t663;                                   \
                                                                               \
  struct mgga_x_scan_params                                                    \
  {                                                                            \
    double c1 = 0.667;                                                         \
    double c2 = 0.8;                                                           \
    double d  = 1.24;                                                          \
    double k1 = 0.065;                                                         \
  } params;                                                                    \
                                                                               \
  t2   = rho0 <= DENS_THRESHOLD_X_SCAN;                                        \
  t3   = M_CBRT3;                                                              \
  t4   = M_CBRTPI;                                                             \
  t6   = t3 / t4;                                                              \
  t7   = rho0 + rho1;                                                          \
  t8   = 0.1e1 / t7;                                                           \
  t11  = 0.2e1 * rho0 * t8 <= ZETA_THRESHOLD_X_SCAN;                           \
  t12  = ZETA_THRESHOLD_X_SCAN - 0.1e1;                                        \
  t15  = 0.2e1 * rho1 * t8 <= ZETA_THRESHOLD_X_SCAN;                           \
  t16  = -t12;                                                                 \
  t17  = rho0 - rho1;                                                          \
  t19  = my_piecewise5(t11, t12, t15, t16, t17 * t8);                          \
  t20  = 0.1e1 + t19;                                                          \
  t21  = t20 <= ZETA_THRESHOLD_X_SCAN;                                         \
  t22  = POW_1_3(ZETA_THRESHOLD_X_SCAN);                                       \
  t23  = t22 * ZETA_THRESHOLD_X_SCAN;                                          \
  t24  = POW_1_3(t20);                                                         \
  t26  = my_piecewise3(t21, t23, t24 * t20);                                   \
  t27  = t6 * t26;                                                             \
  t28  = POW_1_3(t7);                                                          \
  t29  = M_CBRT6;                                                              \
  t30  = M_PI * M_PI;                                                          \
  t31  = POW_1_3(t30);                                                         \
  t32  = t31 * t31;                                                            \
  t33  = 0.1e1 / t32;                                                          \
  t34  = t29 * t33;                                                            \
  t35  = rho0 * rho0;                                                          \
  t36  = POW_1_3(rho0);                                                        \
  t37  = t36 * t36;                                                            \
  t38  = t37 * t35;                                                            \
  t39  = 0.1e1 / t38;                                                          \
  t40  = sigma0 * t39;                                                         \
  t41  = t34 * t40;                                                            \
  t45  = 0.1e3 / 0.6561e4 / params.k1 - 0.73e2 / 0.648e3;                      \
  t46  = t29 * t29;                                                            \
  t47  = t45 * t46;                                                            \
  t48  = t31 * t30;                                                            \
  t49  = 0.1e1 / t48;                                                          \
  t50  = t47 * t49;                                                            \
  t51  = sigma0 * sigma0;                                                      \
  t52  = t35 * t35;                                                            \
  t53  = t52 * rho0;                                                           \
  t55  = 0.1e1 / t36 / t53;                                                    \
  t56  = t51 * t55;                                                            \
  t57  = t45 * t29;                                                            \
  t58  = t33 * sigma0;                                                         \
  t59  = t58 * t39;                                                            \
  t62  = exp(-0.27e2 / 0.8e2 * t57 * t59);                                     \
  t66  = sqrt(0.146e3);                                                        \
  t67  = t66 * t29;                                                            \
  t70  = t37 * rho0;                                                           \
  t71  = 0.1e1 / t70;                                                          \
  t77  = 0.5e1 / 0.9e1 * (tau0 * t71 - t40 / 0.8e1) * t29 * t33;               \
  t78  = 0.1e1 - t77;                                                          \
  t80  = t78 * t78;                                                            \
  t82  = exp(-t80 / 0.2e1);                                                    \
  t85  = 0.7e1 / 0.1296e5 * t67 * t59 + t66 * t78 * t82 / 0.1e3;               \
  t86  = t85 * t85;                                                            \
  t87  = params.k1 + 0.5e1 / 0.972e3 * t41 + t50 * t56 * t62 / 0.576e3 + t86;  \
  t92  = 0.1e1 + params.k1 * (0.1e1 - params.k1 / t87);                        \
  t93  = t77 <= 0.1e1;                                                         \
  t94  = log(DBL_EPSILON);                                                     \
  t97  = t94 / (-t94 + params.c1);                                             \
  t98  = -t97 < t77;                                                           \
  t99  = t77 < -t97;                                                           \
  t100 = my_piecewise3(t99, t77, -t97);                                        \
  t101 = params.c1 * t100;                                                     \
  t102 = 0.1e1 - t100;                                                         \
  t103 = 0.1e1 / t102;                                                         \
  t105 = exp(-t101 * t103);                                                    \
  t106 = my_piecewise3(t98, 0, t105);                                          \
  t107 = fabs(params.d);                                                       \
  t110 = log(DBL_EPSILON / t107);                                              \
  t113 = (-t110 + params.c2) / t110;                                           \
  t114 = t77 < -t113;                                                          \
  t115 = my_piecewise3(t114, -t113, t77);                                      \
  t116 = 0.1e1 - t115;                                                         \
  t119 = exp(params.c2 / t116);                                                \
  t121 = my_piecewise3(t114, 0, -params.d * t119);                             \
  t122 = my_piecewise3(t93, t106, t121);                                       \
  t123 = 0.1e1 - t122;                                                         \
  t126 = t92 * t123 + 0.1174e1 * t122;                                         \
  t127 = t28 * t126;                                                           \
  t128 = sqrt(0.3e1);                                                          \
  t129 = 0.1e1 / t31;                                                          \
  t130 = t46 * t129;                                                           \
  t131 = sqrt(sigma0);                                                         \
  t132 = t36 * rho0;                                                           \
  t133 = 0.1e1 / t132;                                                         \
  t135 = t130 * t131 * t133;                                                   \
  t136 = sqrt(t135);                                                           \
  t140 = exp(-0.98958e1 * t128 / t136);                                        \
  t141 = 0.1e1 - t140;                                                         \
  t142 = t127 * t141;                                                          \
  t145 = my_piecewise3(t2, 0, -0.3e1 / 0.8e1 * t27 * t142);                    \
  t146 = rho1 <= DENS_THRESHOLD_X_SCAN;                                        \
  t147 = -t17;                                                                 \
  t149 = my_piecewise5(t15, t12, t11, t16, t147 * t8);                         \
  t150 = 0.1e1 + t149;                                                         \
  t151 = t150 <= ZETA_THRESHOLD_X_SCAN;                                        \
  t152 = POW_1_3(t150);                                                        \
  t154 = my_piecewise3(t151, t23, t152 * t150);                                \
  t155 = t6 * t154;                                                            \
  t156 = rho1 * rho1;                                                          \
  t157 = POW_1_3(rho1);                                                        \
  t158 = t157 * t157;                                                          \
  t159 = t158 * t156;                                                          \
  t160 = 0.1e1 / t159;                                                         \
  t161 = sigma2 * t160;                                                        \
  t162 = t34 * t161;                                                           \
  t164 = sigma2 * sigma2;                                                      \
  t165 = t156 * t156;                                                          \
  t166 = t165 * rho1;                                                          \
  t168 = 0.1e1 / t157 / t166;                                                  \
  t169 = t164 * t168;                                                          \
  t170 = t33 * sigma2;                                                         \
  t171 = t170 * t160;                                                          \
  t174 = exp(-0.27e2 / 0.8e2 * t57 * t171);                                    \
  t180 = t158 * rho1;                                                          \
  t181 = 0.1e1 / t180;                                                         \
  t187 = 0.5e1 / 0.9e1 * (tau1 * t181 - t161 / 0.8e1) * t29 * t33;             \
  t188 = 0.1e1 - t187;                                                         \
  t190 = t188 * t188;                                                          \
  t192 = exp(-t190 / 0.2e1);                                                   \
  t195 = 0.7e1 / 0.1296e5 * t67 * t171 + t66 * t188 * t192 / 0.1e3;            \
  t196 = t195 * t195;                                                          \
  t197 =                                                                       \
    params.k1 + 0.5e1 / 0.972e3 * t162 + t50 * t169 * t174 / 0.576e3 + t196;   \
  t202 = 0.1e1 + params.k1 * (0.1e1 - params.k1 / t197);                       \
  t203 = t187 <= 0.1e1;                                                        \
  t204 = -t97 < t187;                                                          \
  t205 = t187 < -t97;                                                          \
  t206 = my_piecewise3(t205, t187, -t97);                                      \
  t207 = params.c1 * t206;                                                     \
  t208 = 0.1e1 - t206;                                                         \
  t209 = 0.1e1 / t208;                                                         \
  t211 = exp(-t207 * t209);                                                    \
  t212 = my_piecewise3(t204, 0, t211);                                         \
  t213 = t187 < -t113;                                                         \
  t214 = my_piecewise3(t213, -t113, t187);                                     \
  t215 = 0.1e1 - t214;                                                         \
  t218 = exp(params.c2 / t215);                                                \
  t220 = my_piecewise3(t213, 0, -params.d * t218);                             \
  t221 = my_piecewise3(t203, t212, t220);                                      \
  t222 = 0.1e1 - t221;                                                         \
  t225 = t202 * t222 + 0.1174e1 * t221;                                        \
  t226 = t28 * t225;                                                           \
  t227 = sqrt(sigma2);                                                         \
  t228 = t157 * rho1;                                                          \
  t229 = 0.1e1 / t228;                                                         \
  t231 = t130 * t227 * t229;                                                   \
  t232 = sqrt(t231);                                                           \
  t236 = exp(-0.98958e1 * t128 / t232);                                        \
  t237 = 0.1e1 - t236;                                                         \
  t238 = t226 * t237;                                                          \
  t241 = my_piecewise3(t146, 0, -0.3e1 / 0.8e1 * t155 * t238);                 \
  tzk0 = t145 + t241;                                                          \
                                                                               \
  t242 = t7 * t7;                                                              \
  t243 = 0.1e1 / t242;                                                         \
  t244 = t17 * t243;                                                           \
  t246 = my_piecewise5(t11, 0, t15, 0, t8 - t244);                             \
  t249 = my_piecewise3(t21, 0, 0.4e1 / 0.3e1 * t24 * t246);                    \
  t250 = t6 * t249;                                                            \
  t253 = t28 * t28;                                                            \
  t254 = 0.1e1 / t253;                                                         \
  t255 = t254 * t126;                                                          \
  t256 = t255 * t141;                                                          \
  t258 = t27 * t256 / 0.8e1;                                                   \
  t259 = params.k1 * params.k1;                                                \
  t260 = t87 * t87;                                                            \
  t262 = t259 / t260;                                                          \
  t263 = t35 * rho0;                                                           \
  t265 = 0.1e1 / t37 / t263;                                                   \
  t266 = sigma0 * t265;                                                        \
  t269 = t52 * t35;                                                            \
  t271 = 0.1e1 / t36 / t269;                                                   \
  t276 = t45 * t45;                                                            \
  t277 = t30 * t30;                                                            \
  t278 = 0.1e1 / t277;                                                         \
  t279 = t276 * t278;                                                          \
  t280 = t51 * sigma0;                                                         \
  t281 = t52 * t52;                                                            \
  t282 = t281 * rho0;                                                          \
  t283 = 0.1e1 / t282;                                                         \
  t294 = -0.5e1 / 0.3e1 * tau0 * t39 + t266 / 0.3e1;                           \
  t296 = t34 * t82;                                                            \
  t299 = t66 * t80;                                                            \
  t303 = -0.7e1 / 0.486e4 * t67 * t58 * t265 - t66 * t294 * t296 / 0.18e3 +    \
         t299 * t294 * t296 / 0.18e3;                                          \
  t306 = -0.1e2 / 0.729e3 * t34 * t266 - t50 * t51 * t271 * t62 / 0.108e3 +    \
         0.3e1 / 0.32e3 * t279 * t280 * t283 * t62 + 0.2e1 * t85 * t303;       \
  t307   = t306 * t123;                                                        \
  t309   = t294 * t29;                                                         \
  t311   = 0.5e1 / 0.9e1 * t309 * t33;                                         \
  t312   = my_piecewise3(t99, t311, 0);                                        \
  t315   = t102 * t102;                                                        \
  t316   = 0.1e1 / t315;                                                       \
  t317   = t316 * t312;                                                        \
  t319   = -params.c1 * t312 * t103 - t101 * t317;                             \
  t320   = t319 * t105;                                                        \
  t321   = my_piecewise3(t98, 0, t320);                                        \
  t322   = params.d * params.c2;                                               \
  t323   = t116 * t116;                                                        \
  t324   = 0.1e1 / t323;                                                       \
  t325   = my_piecewise3(t114, 0, t311);                                       \
  t329   = my_piecewise3(t114, 0, -t322 * t324 * t325 * t119);                 \
  t330   = my_piecewise3(t93, t321, t329);                                     \
  t333   = t262 * t307 - t92 * t330 + 0.1174e1 * t330;                         \
  t334   = t28 * t333;                                                         \
  t335   = t334 * t141;                                                        \
  t338   = pow(0.3e1, 0.1e1 / 0.6e1);                                          \
  t339   = t338 * t338;                                                        \
  t340   = t339 * t339;                                                        \
  t341   = t340 * t338;                                                        \
  t342   = t341 * t26;                                                         \
  t344   = 0.1e1 / t136 / t135;                                                \
  t345   = t127 * t344;                                                        \
  t346   = t342 * t345;                                                        \
  t348   = 0.1e1 / t36 / t35;                                                  \
  t351   = t130 * t131 * t348 * t140;                                          \
  t355   = my_piecewise3(t2,                                                   \
                       0,                                                    \
                       -0.3e1 / 0.8e1 * t250 * t142 - t258 -                 \
                         0.3e1 / 0.8e1 * t27 * t335 -                        \
                         0.16891736332904387511e1 * t346 * t351);            \
  t356   = t147 * t243;                                                        \
  t358   = my_piecewise5(t15, 0, t11, 0, -t8 - t356);                          \
  t361   = my_piecewise3(t151, 0, 0.4e1 / 0.3e1 * t152 * t358);                \
  t362   = t6 * t361;                                                          \
  t365   = t254 * t225;                                                        \
  t366   = t365 * t237;                                                        \
  t368   = t155 * t366 / 0.8e1;                                                \
  t370   = my_piecewise3(t146, 0, -0.3e1 / 0.8e1 * t362 * t238 - t368);        \
  tvrho0 = t145 + t241 + t7 * (t355 + t370);                                   \
                                                                               \
  t374 = my_piecewise5(t11, 0, t15, 0, -t8 - t244);                            \
  t377 = my_piecewise3(t21, 0, 0.4e1 / 0.3e1 * t24 * t374);                    \
  t378 = t6 * t377;                                                            \
  t382 = my_piecewise3(t2, 0, -0.3e1 / 0.8e1 * t378 * t142 - t258);            \
  t384 = my_piecewise5(t15, 0, t11, 0, t8 - t356);                             \
  t387 = my_piecewise3(t151, 0, 0.4e1 / 0.3e1 * t152 * t384);                  \
  t388 = t6 * t387;                                                            \
  t391 = t197 * t197;                                                          \
  t393 = t259 / t391;                                                          \
  t394 = t156 * rho1;                                                          \
  t396 = 0.1e1 / t158 / t394;                                                  \
  t397 = sigma2 * t396;                                                        \
  t400 = t165 * t156;                                                          \
  t402 = 0.1e1 / t157 / t400;                                                  \
  t407 = t164 * sigma2;                                                        \
  t408 = t165 * t165;                                                          \
  t409 = t408 * rho1;                                                          \
  t410 = 0.1e1 / t409;                                                         \
  t421 = -0.5e1 / 0.3e1 * tau1 * t160 + t397 / 0.3e1;                          \
  t423 = t34 * t192;                                                           \
  t426 = t66 * t190;                                                           \
  t430 = -0.7e1 / 0.486e4 * t67 * t170 * t396 - t66 * t421 * t423 / 0.18e3 +   \
         t426 * t421 * t423 / 0.18e3;                                          \
  t433 = -0.1e2 / 0.729e3 * t34 * t397 - t50 * t164 * t402 * t174 / 0.108e3 +  \
         0.3e1 / 0.32e3 * t279 * t407 * t410 * t174 + 0.2e1 * t195 * t430;     \
  t434   = t433 * t222;                                                        \
  t436   = t421 * t29;                                                         \
  t438   = 0.5e1 / 0.9e1 * t436 * t33;                                         \
  t439   = my_piecewise3(t205, t438, 0);                                       \
  t442   = t208 * t208;                                                        \
  t443   = 0.1e1 / t442;                                                       \
  t444   = t443 * t439;                                                        \
  t446   = -params.c1 * t439 * t209 - t207 * t444;                             \
  t447   = t446 * t211;                                                        \
  t448   = my_piecewise3(t204, 0, t447);                                       \
  t449   = t215 * t215;                                                        \
  t450   = 0.1e1 / t449;                                                       \
  t451   = my_piecewise3(t213, 0, t438);                                       \
  t455   = my_piecewise3(t213, 0, -t322 * t450 * t451 * t218);                 \
  t456   = my_piecewise3(t203, t448, t455);                                    \
  t459   = t393 * t434 - t202 * t456 + 0.1174e1 * t456;                        \
  t460   = t28 * t459;                                                         \
  t461   = t460 * t237;                                                        \
  t464   = t341 * t154;                                                        \
  t466   = 0.1e1 / t232 / t231;                                                \
  t467   = t226 * t466;                                                        \
  t468   = t464 * t467;                                                        \
  t470   = 0.1e1 / t157 / t156;                                                \
  t473   = t130 * t227 * t470 * t236;                                          \
  t477   = my_piecewise3(t146,                                                 \
                       0,                                                    \
                       -0.3e1 / 0.8e1 * t388 * t238 - t368 -                 \
                         0.3e1 / 0.8e1 * t155 * t461 -                       \
                         0.16891736332904387511e1 * t468 * t473);            \
  tvrho1 = t145 + t241 + t7 * (t382 + t477);                                   \
                                                                               \
                                                                               \
  t481 = t39 * t29 * t33;                                                      \
  t487 = 0.1e1 / t281;                                                         \
  t495 = t66 * t39;                                                            \
  t496 = t495 * t296;                                                          \
  t499 = t299 * t39 * t296;                                                    \
  t501 = 0.7e1 / 0.1296e5 * t67 * t33 * t39 + t496 / 0.144e4 - t499 / 0.144e4; \
  t504 = 0.5e1 / 0.972e3 * t481 + t50 * sigma0 * t55 * t62 / 0.288e3 -         \
         0.9e1 / 0.256e4 * t279 * t51 * t487 * t62 + 0.2e1 * t85 * t501;       \
  t505     = t504 * t123;                                                      \
  t507     = 0.5e1 / 0.72e2 * t481;                                            \
  t508     = my_piecewise3(t99, -t507, 0);                                     \
  t509     = params.c1 * t508;                                                 \
  t511     = t316 * t508;                                                      \
  t513     = -t101 * t511 - t509 * t103;                                       \
  t514     = t513 * t105;                                                      \
  t515     = my_piecewise3(t98, 0, t514);                                      \
  t516     = my_piecewise3(t114, 0, -t507);                                    \
  t520     = my_piecewise3(t114, 0, -t322 * t324 * t516 * t119);               \
  t521     = my_piecewise3(t93, t515, t520);                                   \
  t524     = t262 * t505 - t92 * t521 + 0.1174e1 * t521;                       \
  t525     = t28 * t524;                                                       \
  t526     = t525 * t141;                                                      \
  t529     = 0.1e1 / t131;                                                     \
  t532     = t130 * t529 * t133 * t140;                                        \
  t536     = my_piecewise3(t2,                                                 \
                       0,                                                  \
                       -0.3e1 / 0.8e1 * t27 * t526 +                       \
                         0.63344011248391453166e0 * t346 * t532);          \
  tvsigma0 = t7 * t536;                                                        \
                                                                               \
  tvsigma1 = 0.e0;                                                             \
                                                                               \
                                                                               \
  t538 = t160 * t29 * t33;                                                     \
  t544 = 0.1e1 / t408;                                                         \
  t552 = t66 * t160;                                                           \
  t553 = t552 * t423;                                                          \
  t556 = t426 * t160 * t423;                                                   \
  t558 =                                                                       \
    0.7e1 / 0.1296e5 * t67 * t33 * t160 + t553 / 0.144e4 - t556 / 0.144e4;     \
  t561 = 0.5e1 / 0.972e3 * t538 + t50 * sigma2 * t168 * t174 / 0.288e3 -       \
         0.9e1 / 0.256e4 * t279 * t164 * t544 * t174 + 0.2e1 * t195 * t558;    \
  t562     = t561 * t222;                                                      \
  t564     = 0.5e1 / 0.72e2 * t538;                                            \
  t565     = my_piecewise3(t205, -t564, 0);                                    \
  t566     = params.c1 * t565;                                                 \
  t568     = t443 * t565;                                                      \
  t570     = -t207 * t568 - t566 * t209;                                       \
  t571     = t570 * t211;                                                      \
  t572     = my_piecewise3(t204, 0, t571);                                     \
  t573     = my_piecewise3(t213, 0, -t564);                                    \
  t577     = my_piecewise3(t213, 0, -t322 * t450 * t573 * t218);               \
  t578     = my_piecewise3(t203, t572, t577);                                  \
  t581     = t393 * t562 - t202 * t578 + 0.1174e1 * t578;                      \
  t582     = t28 * t581;                                                       \
  t583     = t582 * t237;                                                      \
  t586     = 0.1e1 / t227;                                                     \
  t589     = t130 * t586 * t229 * t236;                                        \
  t593     = my_piecewise3(t146,                                               \
                       0,                                                  \
                       -0.3e1 / 0.8e1 * t155 * t583 +                      \
                         0.63344011248391453166e0 * t468 * t589);          \
  tvsigma2 = t7 * t593;                                                        \
                                                                               \
  tvlapl0 = 0.e0;                                                              \
  tvlapl1 = 0.e0;                                                              \
                                                                               \
  t594   = t66 * t71;                                                          \
  t599   = t299 * t71 * t296 / 0.18e3 - t594 * t296 / 0.18e3;                  \
  t600   = t85 * t599;                                                         \
  t606   = 0.5e1 / 0.9e1 * t71 * t29 * t33;                                    \
  t607   = my_piecewise3(t99, t606, 0);                                        \
  t608   = params.c1 * t607;                                                   \
  t612   = -t101 * t316 * t607 - t608 * t103;                                  \
  t613   = t612 * t105;                                                        \
  t614   = my_piecewise3(t98, 0, t613);                                        \
  t615   = my_piecewise3(t114, 0, t606);                                       \
  t619   = my_piecewise3(t114, 0, -t322 * t324 * t615 * t119);                 \
  t620   = my_piecewise3(t93, t614, t619);                                     \
  t623   = 0.2e1 * t262 * t600 * t123 - t92 * t620 + 0.1174e1 * t620;          \
  t624   = t28 * t623;                                                         \
  t625   = t624 * t141;                                                        \
  t628   = my_piecewise3(t2, 0, -0.3e1 / 0.8e1 * t27 * t625);                  \
  tvtau0 = t7 * t628;                                                          \
                                                                               \
  t629   = t66 * t181;                                                         \
  t634   = t426 * t181 * t423 / 0.18e3 - t629 * t423 / 0.18e3;                 \
  t635   = t195 * t634;                                                        \
  t641   = 0.5e1 / 0.9e1 * t181 * t29 * t33;                                   \
  t642   = my_piecewise3(t205, t641, 0);                                       \
  t643   = params.c1 * t642;                                                   \
  t647   = -t207 * t443 * t642 - t643 * t209;                                  \
  t648   = t647 * t211;                                                        \
  t649   = my_piecewise3(t204, 0, t648);                                       \
  t650   = my_piecewise3(t213, 0, t641);                                       \
  t654   = my_piecewise3(t213, 0, -t322 * t450 * t650 * t218);                 \
  t655   = my_piecewise3(t203, t649, t654);                                    \
  t658   = 0.2e1 * t393 * t635 * t222 - t202 * t655 + 0.1174e1 * t655;         \
  t659   = t28 * t658;                                                         \
  t660   = t659 * t237;                                                        \
  t663   = my_piecewise3(t146, 0, -0.3e1 / 0.8e1 * t155 * t660);               \
  tvtau1 = t7 * t663;
#endif
