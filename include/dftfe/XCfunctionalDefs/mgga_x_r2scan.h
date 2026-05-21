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

#ifndef MGGA_X_R2SCAN
#define MGGA_X_R2SCAN                                                              \
  double t2, t3, t4, t6, t7, t8, t11, t12;                                         \
  double t15, t16, t17, t19, t20, t21, t22, t23;                                   \
  double t24, t26, t27, t28, t30, t31, t32, t33;                                   \
  double t34, t35, t36, t37, t38, t39, t40, t41;                                   \
  double t42, t44, t45, t46, t47, t48, t52, t56;                                   \
  double t57, t58, t59, t60, t61, t62, t66, t70;                                   \
  double t71, t72, t74, t76, t78, t79, t82, t83;                                   \
  double t84, t85, t86, t87, t88, t89, t90, t92;                                   \
  double t93, t94, t95, t97, t99, t101, t103, t105;                                \
  double t110, t111, t114, t116, t117, t119, t120, t121;                           \
  double t122, t123, t124, t125, t126, t128, t129, t133;                           \
  double t134, t135, t138, t139, t140, t142, t143, t144;                           \
  double t145, t147, t148, t149, t150, t151, t152, t153;                           \
  double t155, t156, t160, t164, t165, t166, t167, t168;                           \
  double t172, t176, t177, t178, t180, t182, t183, t186;                           \
  double t187, t188, t189, t190, t191, t192, t193, t194;                           \
  double t196, t197, t198, t199, t201, t203, t205, t207;                           \
  double t209, t214, t215, t218, t220, t221, t223, t224;                           \
  double t225, t226, t227, t229, t230, t234, t235, t236;                           \
  double t239;                                                                     \
                                                                                   \
  double t240, t241, t242, t244, t247, t248, t251, t252;                           \
  double t253, t254, t256, t257, t258, t259, t260, t261;                           \
  double t262, t263, t264, t265, t267, t270, t272, t276;                           \
  double t282, t284, t285, t286, t287, t290, t291, t294;                           \
  double t295, t296, t298, t299, t300, t302, t304, t306;                           \
  double t308, t310, t315, t316, t317, t318, t322, t324;                           \
  double t325, t327, t328, t329, t332, t333, t334, t335;                           \
  double t336, t338, t339, t340, t342, t344, t345, t349;                           \
  double t350, t352, t355, t356, t359, t360, t362, t364;                           \
  double t368, t371, t372, t376, t378, t381, t382;                                 \
  double t385, t386, t387, t388, t389, t390, t391, t392;                           \
  double t394, t397, t399, t403, t409, t411, t412, t413;                           \
  double t414, t417, t418, t421, t422, t423, t425, t426;                           \
  double t427, t429, t431, t433, t435, t437, t442, t443;                           \
  double t444, t448, t450, t451, t453, t454, t455, t458;                           \
  double t460, t461, t462, t464, t466, t467, t471;                                 \
  double t474, t475, t477, t483, t485, t486, t489, t490;                           \
  double t491, t493, t495, t496, t497, t499, t501, t503;                           \
  double t505, t507, t512, t516, t518, t520, t521, t522;                           \
  double t525, t527, t528, t532, t533, t534;                                       \
  double t536, t542, t544, t545, t548, t549, t550, t552;                           \
  double t554, t555, t556, t558, t560, t562, t564, t566;                           \
  double t571, t575, t577, t579, t580, t581, t584, t586;                           \
  double t587, t591, tvlapl0, tvlapl1, t592, t593, t594;                           \
  double t596, t598, t599, t600, t602, t604, t606, t608;                           \
  double t610, t615, t619, t620, t621, t622, t625;                                 \
  double t626, t627, t628, t630, t632, t633, t634, t636;                           \
  double t638, t640, t642, t644, t649, t653, t654, t655;                           \
  double t656, t659;                                                               \
                                                                                   \
  struct mgga_x_r2scan_params                                                      \
  {                                                                                \
    double c1  = 0.667;                                                            \
    double c2  = 0.8;                                                              \
    double d   = 1.24;                                                             \
    double k1  = 0.065;                                                            \
    double eta = 0.001;                                                            \
    double dp2 = 0.361;                                                            \
  } params;                                                                        \
                                                                                   \
  t2   = rho0 <= DENS_THRESHOLD_X_R2SCAN;                                          \
  t3   = M_CBRT3;                                                                  \
  t4   = M_CBRTPI;                                                                 \
  t6   = t3 / t4;                                                                  \
  t7   = rho0 + rho1;                                                              \
  t8   = 0.1e1 / t7;                                                               \
  t11  = 0.2e1 * rho0 * t8 <= ZETA_THRESHOLD_X_R2SCAN;                             \
  t12  = ZETA_THRESHOLD_X_R2SCAN - 0.1e1;                                          \
  t15  = 0.2e1 * rho1 * t8 <= ZETA_THRESHOLD_X_R2SCAN;                             \
  t16  = -t12;                                                                     \
  t17  = rho0 - rho1;                                                              \
  t19  = my_piecewise5(t11, t12, t15, t16, t17 * t8);                              \
  t20  = 0.1e1 + t19;                                                              \
  t21  = t20 <= ZETA_THRESHOLD_X_R2SCAN;                                           \
  t22  = std::pow(ZETA_THRESHOLD_X_R2SCAN, 1.0 / 3.0);                             \
  t23  = t22 * ZETA_THRESHOLD_X_R2SCAN;                                            \
  t24  = std::pow(t20, 1.0 / 3.0);                                                 \
  t26  = my_piecewise3(t21, t23, t24 * t20);                                       \
  t27  = t6 * t26;                                                                 \
  t28  = std::pow(t7, 1.0 / 3.0);                                                  \
  t30  = 0.2e2 / 0.27e2 + 0.5e1 / 0.3e1 * params.eta;                              \
  t31  = M_CBRT6;                                                                  \
  t32  = t31 * t31;                                                                \
  t33  = M_PI * M_PI;                                                              \
  t34  = std::pow(t33, 1.0 / 3.0);                                                 \
  t35  = t34 * t33;                                                                \
  t36  = 0.1e1 / t35;                                                              \
  t37  = t32 * t36;                                                                \
  t38  = sigma0 * sigma0;                                                          \
  t39  = rho0 * rho0;                                                              \
  t40  = t39 * t39;                                                                \
  t41  = t40 * rho0;                                                               \
  t42  = std::pow(rho0, 1.0 / 3.0);                                                \
  t44  = 0.1e1 / t42 / t41;                                                        \
  t45  = t38 * t44;                                                                \
  t46  = params.dp2 * params.dp2;                                                  \
  t47  = t46 * t46;                                                                \
  t48  = 0.1e1 / t47;                                                              \
  t52  = std::exp(-t37 * t45 * t48 / 0.576e3);                                     \
  t56  = (-0.162742215233874e0 * t30 * t52 + 0.1e2 / 0.81e2) * t31;                \
  t57  = t34 * t34;                                                                \
  t58  = 0.1e1 / t57;                                                              \
  t59  = t58 * sigma0;                                                             \
  t60  = t42 * t42;                                                                \
  t61  = t60 * t39;                                                                \
  t62  = 0.1e1 / t61;                                                              \
  t66  = params.k1 + t56 * t59 * t62 / 0.24e2;                                     \
  t70  = params.k1 * (0.1e1 - params.k1 / t66);                                    \
  t71  = t60 * rho0;                                                               \
  t72  = 0.1e1 / t71;                                                              \
  t74  = sigma0 * t62;                                                             \
  t76  = tau0 * t72 - t74 / 0.8e1;                                                 \
  t78  = 0.3e1 / 0.1e2 * t32 * t57;                                                \
  t79  = params.eta * sigma0;                                                      \
  t82  = t78 + t79 * t62 / 0.8e1;                                                  \
  t83  = 0.1e1 / t82;                                                              \
  t84  = t76 * t83;                                                                \
  t85  = t84 <= 0.e0;                                                              \
  t86  = 0.e0 < t84;                                                               \
  t87  = my_piecewise3(t86, 0, t84);                                               \
  t88  = params.c1 * t87;                                                          \
  t89  = 0.1e1 - t87;                                                              \
  t90  = 0.1e1 / t89;                                                              \
  t92  = std::exp(-t88 * t90);                                                     \
  t93  = t84 <= 0.25e1;                                                            \
  t94  = 0.25e1 < t84;                                                             \
  t95  = my_piecewise3(t94, 0.25e1, t84);                                          \
  t97  = t95 * t95;                                                                \
  t99  = t97 * t95;                                                                \
  t101 = t97 * t97;                                                                \
  t103 = t101 * t95;                                                               \
  t105 = t101 * t97;                                                               \
  t110 = my_piecewise3(t94, t84, 0.25e1);                                          \
  t111 = 0.1e1 - t110;                                                             \
  t114 = std::exp(params.c2 / t111);                                               \
  t116 = my_piecewise5(t85,                                                        \
                       t92,                                                        \
                       t93,                                                        \
                       0.1e1 - 0.667e0 * t95 - 0.4445555e0 * t97 -                 \
                         0.663086601049e0 * t99 + 0.145129704449e1 * t101 -        \
                         0.887998041597e0 * t103 + 0.234528941479e0 * t105 -       \
                         0.23185843322e-1 * t101 * t99,                            \
                       -params.d * t114);                                          \
  t117 = 0.174e0 - t70;                                                            \
  t119 = t116 * t117 + t70 + 0.1e1;                                                \
  t120 = t28 * t119;                                                               \
  t121 = std::sqrt(0.3e1);                                                         \
  t122 = 0.1e1 / t34;                                                              \
  t123 = t32 * t122;                                                               \
  t124 = std::sqrt(sigma0);                                                        \
  t125 = t42 * rho0;                                                               \
  t126 = 0.1e1 / t125;                                                             \
  t128 = t123 * t124 * t126;                                                       \
  t129 = std::sqrt(t128);                                                          \
  t133 = std::exp(-0.98958e1 * t121 / t129);                                       \
  t134 = 0.1e1 - t133;                                                             \
  t135 = t120 * t134;                                                              \
  t138 = my_piecewise3(t2, 0, -0.3e1 / 0.8e1 * t27 * t135);                        \
  t139 = rho1 <= DENS_THRESHOLD_X_R2SCAN;                                          \
  t140 = -t17;                                                                     \
  t142 = my_piecewise5(t15, t12, t11, t16, t140 * t8);                             \
  t143 = 0.1e1 + t142;                                                             \
  t144 = t143 <= ZETA_THRESHOLD_X_R2SCAN;                                          \
  t145 = std::pow(t143, 1.0 / 3.0);                                                \
  t147 = my_piecewise3(t144, t23, t145 * t143);                                    \
  t148 = t6 * t147;                                                                \
  t149 = sigma2 * sigma2;                                                          \
  t150 = rho1 * rho1;                                                              \
  t151 = t150 * t150;                                                              \
  t152 = t151 * rho1;                                                              \
  t153 = std::pow(rho1, 1.0 / 3.0);                                                \
  t155 = 0.1e1 / t153 / t152;                                                      \
  t156 = t149 * t155;                                                              \
  t160 = std::exp(-t37 * t156 * t48 / 0.576e3);                                    \
  t164 = (-0.162742215233874e0 * t30 * t160 + 0.1e2 / 0.81e2) * t31;               \
  t165 = t58 * sigma2;                                                             \
  t166 = t153 * t153;                                                              \
  t167 = t166 * t150;                                                              \
  t168 = 0.1e1 / t167;                                                             \
  t172 = params.k1 + t164 * t165 * t168 / 0.24e2;                                  \
  t176 = params.k1 * (0.1e1 - params.k1 / t172);                                   \
  t177 = t166 * rho1;                                                              \
  t178 = 0.1e1 / t177;                                                             \
  t180 = sigma2 * t168;                                                            \
  t182 = tau1 * t178 - t180 / 0.8e1;                                               \
  t183 = params.eta * sigma2;                                                      \
  t186 = t78 + t183 * t168 / 0.8e1;                                                \
  t187 = 0.1e1 / t186;                                                             \
  t188 = t182 * t187;                                                              \
  t189 = t188 <= 0.e0;                                                             \
  t190 = 0.e0 < t188;                                                              \
  t191 = my_piecewise3(t190, 0, t188);                                             \
  t192 = params.c1 * t191;                                                         \
  t193 = 0.1e1 - t191;                                                             \
  t194 = 0.1e1 / t193;                                                             \
  t196 = std::exp(-t192 * t194);                                                   \
  t197 = t188 <= 0.25e1;                                                           \
  t198 = 0.25e1 < t188;                                                            \
  t199 = my_piecewise3(t198, 0.25e1, t188);                                        \
  t201 = t199 * t199;                                                              \
  t203 = t201 * t199;                                                              \
  t205 = t201 * t201;                                                              \
  t207 = t205 * t199;                                                              \
  t209 = t205 * t201;                                                              \
  t214 = my_piecewise3(t198, t188, 0.25e1);                                        \
  t215 = 0.1e1 - t214;                                                             \
  t218 = std::exp(params.c2 / t215);                                               \
  t220 = my_piecewise5(t189,                                                       \
                       t196,                                                       \
                       t197,                                                       \
                       0.1e1 - 0.667e0 * t199 - 0.4445555e0 * t201 -               \
                         0.663086601049e0 * t203 + 0.145129704449e1 * t205 -       \
                         0.887998041597e0 * t207 + 0.234528941479e0 * t209 -       \
                         0.23185843322e-1 * t205 * t203,                           \
                       -params.d * t218);                                          \
  t221 = 0.174e0 - t176;                                                           \
  t223 = t220 * t221 + t176 + 0.1e1;                                               \
  t224 = t28 * t223;                                                               \
  t225 = std::sqrt(sigma2);                                                        \
  t226 = t153 * rho1;                                                              \
  t227 = 0.1e1 / t226;                                                             \
  t229 = t123 * t225 * t227;                                                       \
  t230 = std::sqrt(t229);                                                          \
  t234 = std::exp(-0.98958e1 * t121 / t230);                                       \
  t235 = 0.1e1 - t234;                                                             \
  t236 = t224 * t235;                                                              \
  t239 = my_piecewise3(t139, 0, -0.3e1 / 0.8e1 * t148 * t236);                     \
  tzk0 = t138 + t239;                                                              \
                                                                                   \
  t240   = t7 * t7;                                                                \
  t241   = 0.1e1 / t240;                                                           \
  t242   = t17 * t241;                                                             \
  t244   = my_piecewise5(t11, 0, t15, 0, t8 - t242);                               \
  t247   = my_piecewise3(t21, 0, 0.4e1 / 0.3e1 * t24 * t244);                      \
  t248   = t6 * t247;                                                              \
  t251   = t28 * t28;                                                              \
  t252   = 0.1e1 / t251;                                                           \
  t253   = t252 * t119;                                                            \
  t254   = t253 * t134;                                                            \
  t256   = t27 * t254 / 0.8e1;                                                     \
  t257   = params.k1 * params.k1;                                                  \
  t258   = t66 * t66;                                                              \
  t259   = 0.1e1 / t258;                                                           \
  t260   = t257 * t259;                                                            \
  t261   = t38 * sigma0;                                                           \
  t262   = t30 * t261;                                                             \
  t263   = t40 * t40;                                                              \
  t264   = t263 * rho0;                                                            \
  t265   = 0.1e1 / t264;                                                           \
  t267   = t265 * t48 * t52;                                                       \
  t270   = t39 * rho0;                                                             \
  t272   = 0.1e1 / t60 / t270;                                                     \
  t276   = -0.38673812353679841857e-5 * t262 * t267 - t56 * t59 * t272 / 0.9e1;    \
  t282   = -0.5e1 / 0.3e1 * tau0 * t62 + sigma0 * t272 / 0.3e1;                    \
  t284   = t82 * t82;                                                              \
  t285   = 0.1e1 / t284;                                                           \
  t286   = t76 * t285;                                                             \
  t287   = t79 * t272;                                                             \
  t290   = t282 * t83 + t286 * t287 / 0.3e1;                                       \
  t291   = my_piecewise3(t86, 0, t290);                                            \
  t294   = t89 * t89;                                                              \
  t295   = 0.1e1 / t294;                                                           \
  t296   = t295 * t291;                                                            \
  t298   = -t291 * t90 * params.c1 - t296 * t88;                                   \
  t299   = t298 * t92;                                                             \
  t300   = my_piecewise3(t94, 0, t290);                                            \
  t302   = t95 * t300;                                                             \
  t304   = t97 * t300;                                                             \
  t306   = t99 * t300;                                                             \
  t308   = t101 * t300;                                                            \
  t310   = t103 * t300;                                                            \
  t315   = params.d * params.c2;                                                   \
  t316   = t111 * t111;                                                            \
  t317   = 0.1e1 / t316;                                                           \
  t318   = my_piecewise3(t94, t290, 0);                                            \
  t322   = my_piecewise5(t85,                                                      \
                       t299,                                                     \
                       t93,                                                      \
                       -0.667e0 * t300 - 0.889111e0 * t302 -                     \
                         0.1989259803147e1 * t304 + 0.580518817796e1 * t306 -    \
                         0.4439990207985e1 * t308 + 0.1407173648874e1 * t310 -   \
                         0.162300903254e0 * t105 * t300,                         \
                       -t315 * t317 * t318 * t114);                              \
  t324   = t116 * t257;                                                            \
  t325   = t259 * t276;                                                            \
  t327   = t117 * t322 + t260 * t276 - t324 * t325;                                \
  t328   = t28 * t327;                                                             \
  t329   = t328 * t134;                                                            \
  t332   = std::pow(0.3e1, 0.1e1 / 0.6e1);                                         \
  t333   = t332 * t332;                                                            \
  t334   = t333 * t333;                                                            \
  t335   = t334 * t332;                                                            \
  t336   = t335 * t26;                                                             \
  t338   = 0.1e1 / t129 / t128;                                                    \
  t339   = t120 * t338;                                                            \
  t340   = t336 * t339;                                                            \
  t342   = 0.1e1 / t42 / t39;                                                      \
  t344   = t124 * t342 * t133;                                                     \
  t345   = t123 * t344;                                                            \
  t349   = my_piecewise3(t2,                                                       \
                       0,                                                        \
                       -0.3e1 / 0.8e1 * t248 * t135 - t256 -                     \
                         0.3e1 / 0.8e1 * t27 * t329 -                            \
                         0.16891736332904387511e1 * t340 * t345);                \
  t350   = t140 * t241;                                                            \
  t352   = my_piecewise5(t15, 0, t11, 0, -t8 - t350);                              \
  t355   = my_piecewise3(t144, 0, 0.4e1 / 0.3e1 * t145 * t352);                    \
  t356   = t6 * t355;                                                              \
  t359   = t252 * t223;                                                            \
  t360   = t359 * t235;                                                            \
  t362   = t148 * t360 / 0.8e1;                                                    \
  t364   = my_piecewise3(t139, 0, -0.3e1 / 0.8e1 * t356 * t236 - t362);            \
  tvrho0 = t138 + t239 + t7 * (t349 + t364);                                       \
                                                                                   \
  t368 = my_piecewise5(t11, 0, t15, 0, -t8 - t242);                                \
  t371 = my_piecewise3(t21, 0, 0.4e1 / 0.3e1 * t24 * t368);                        \
  t372 = t6 * t371;                                                                \
  t376 = my_piecewise3(t2, 0, -0.3e1 / 0.8e1 * t372 * t135 - t256);                \
  t378 = my_piecewise5(t15, 0, t11, 0, t8 - t350);                                 \
  t381 = my_piecewise3(t144, 0, 0.4e1 / 0.3e1 * t145 * t378);                      \
  t382 = t6 * t381;                                                                \
  t385 = t172 * t172;                                                              \
  t386 = 0.1e1 / t385;                                                             \
  t387 = t257 * t386;                                                              \
  t388 = t149 * sigma2;                                                            \
  t389 = t30 * t388;                                                               \
  t390 = t151 * t151;                                                              \
  t391 = t390 * rho1;                                                              \
  t392 = 0.1e1 / t391;                                                             \
  t394 = t392 * t48 * t160;                                                        \
  t397 = t150 * rho1;                                                              \
  t399 = 0.1e1 / t166 / t397;                                                      \
  t403 =                                                                           \
    -0.38673812353679841857e-5 * t389 * t394 - t164 * t165 * t399 / 0.9e1;         \
  t409   = -0.5e1 / 0.3e1 * tau1 * t168 + sigma2 * t399 / 0.3e1;                   \
  t411   = t186 * t186;                                                            \
  t412   = 0.1e1 / t411;                                                           \
  t413   = t182 * t412;                                                            \
  t414   = t183 * t399;                                                            \
  t417   = t409 * t187 + t413 * t414 / 0.3e1;                                      \
  t418   = my_piecewise3(t190, 0, t417);                                           \
  t421   = t193 * t193;                                                            \
  t422   = 0.1e1 / t421;                                                           \
  t423   = t422 * t418;                                                            \
  t425   = -t194 * t418 * params.c1 - t192 * t423;                                 \
  t426   = t425 * t196;                                                            \
  t427   = my_piecewise3(t198, 0, t417);                                           \
  t429   = t199 * t427;                                                            \
  t431   = t201 * t427;                                                            \
  t433   = t203 * t427;                                                            \
  t435   = t205 * t427;                                                            \
  t437   = t207 * t427;                                                            \
  t442   = t215 * t215;                                                            \
  t443   = 0.1e1 / t442;                                                           \
  t444   = my_piecewise3(t198, t417, 0);                                           \
  t448   = my_piecewise5(t189,                                                     \
                       t426,                                                     \
                       t197,                                                     \
                       -0.667e0 * t427 - 0.889111e0 * t429 -                     \
                         0.1989259803147e1 * t431 + 0.580518817796e1 * t433 -    \
                         0.4439990207985e1 * t435 + 0.1407173648874e1 * t437 -   \
                         0.162300903254e0 * t209 * t427,                         \
                       -t315 * t443 * t444 * t218);                              \
  t450   = t220 * t257;                                                            \
  t451   = t386 * t403;                                                            \
  t453   = t221 * t448 + t387 * t403 - t450 * t451;                                \
  t454   = t28 * t453;                                                             \
  t455   = t454 * t235;                                                            \
  t458   = t335 * t147;                                                            \
  t460   = 0.1e1 / t230 / t229;                                                    \
  t461   = t224 * t460;                                                            \
  t462   = t458 * t461;                                                            \
  t464   = 0.1e1 / t153 / t150;                                                    \
  t466   = t225 * t464 * t234;                                                     \
  t467   = t123 * t466;                                                            \
  t471   = my_piecewise3(t139,                                                     \
                       0,                                                        \
                       -0.3e1 / 0.8e1 * t382 * t236 - t362 -                     \
                         0.3e1 / 0.8e1 * t148 * t455 -                           \
                         0.16891736332904387511e1 * t462 * t467);                \
  tvrho1 = t138 + t239 + t7 * (t376 + t471);                                       \
                                                                                   \
  t474     = t30 * t38;                                                            \
  t475     = 0.1e1 / t263;                                                         \
  t477     = t475 * t48 * t52;                                                     \
  t483     = 0.14502679632629940697e-5 * t474 * t477 + t56 * t58 * t62 / 0.24e2;   \
  t485     = t62 * t83;                                                            \
  t486     = params.eta * t62;                                                     \
  t489     = -t286 * t486 / 0.8e1 - t485 / 0.8e1;                                  \
  t490     = my_piecewise3(t86, 0, t489);                                          \
  t491     = params.c1 * t490;                                                     \
  t493     = t295 * t490;                                                          \
  t495     = -t491 * t90 - t493 * t88;                                             \
  t496     = t495 * t92;                                                           \
  t497     = my_piecewise3(t94, 0, t489);                                          \
  t499     = t95 * t497;                                                           \
  t501     = t97 * t497;                                                           \
  t503     = t99 * t497;                                                           \
  t505     = t101 * t497;                                                          \
  t507     = t103 * t497;                                                          \
  t512     = my_piecewise3(t94, t489, 0);                                          \
  t516     = my_piecewise5(t85,                                                    \
                       t496,                                                   \
                       t93,                                                    \
                       -0.667e0 * t497 - 0.889111e0 * t499 -                   \
                         0.1989259803147e1 * t501 + 0.580518817796e1 * t503 -  \
                         0.4439990207985e1 * t505 + 0.1407173648874e1 * t507 - \
                         0.162300903254e0 * t105 * t497,                       \
                       -t315 * t317 * t512 * t114);                            \
  t518     = t259 * t483;                                                          \
  t520     = t117 * t516 + t260 * t483 - t324 * t518;                              \
  t521     = t28 * t520;                                                           \
  t522     = t521 * t134;                                                          \
  t525     = 0.1e1 / t124;                                                         \
  t527     = t525 * t126 * t133;                                                   \
  t528     = t123 * t527;                                                          \
  t532     = my_piecewise3(t2,                                                     \
                       0,                                                      \
                       -0.3e1 / 0.8e1 * t27 * t522 +                           \
                         0.63344011248391453166e0 * t340 * t528);              \
  tvsigma0 = t7 * t532;                                                            \
                                                                                   \
  tvsigma1 = 0.e0;                                                                 \
                                                                                   \
  t533 = t30 * t149;                                                               \
  t534 = 0.1e1 / t390;                                                             \
  t536 = t534 * t48 * t160;                                                        \
  t542 = 0.14502679632629940697e-5 * t533 * t536 + t164 * t58 * t168 / 0.24e2;     \
  t544 = t168 * t187;                                                              \
  t545 = params.eta * t168;                                                        \
  t548 = -t413 * t545 / 0.8e1 - t544 / 0.8e1;                                      \
  t549 = my_piecewise3(t190, 0, t548);                                             \
  t550 = params.c1 * t549;                                                         \
  t552 = t422 * t549;                                                              \
  t554 = -t192 * t552 - t194 * t550;                                               \
  t555 = t554 * t196;                                                              \
  t556 = my_piecewise3(t198, 0, t548);                                             \
  t558 = t199 * t556;                                                              \
  t560 = t201 * t556;                                                              \
  t562 = t203 * t556;                                                              \
  t564 = t205 * t556;                                                              \
  t566 = t207 * t556;                                                              \
  t571 = my_piecewise3(t198, t548, 0);                                             \
  t575 = my_piecewise5(t189,                                                       \
                       t555,                                                       \
                       t197,                                                       \
                       -0.667e0 * t556 - 0.889111e0 * t558 -                       \
                         0.1989259803147e1 * t560 + 0.580518817796e1 * t562 -      \
                         0.4439990207985e1 * t564 + 0.1407173648874e1 * t566 -     \
                         0.162300903254e0 * t209 * t556,                           \
                       -t315 * t443 * t571 * t218);                                \
  t577 = t386 * t542;                                                              \
  t579 = t221 * t575 + t387 * t542 - t450 * t577;                                  \
  t580 = t28 * t579;                                                               \
  t581 = t580 * t235;                                                              \
  t584 = 0.1e1 / t225;                                                             \
  t586 = t584 * t227 * t234;                                                       \
  t587 = t123 * t586;                                                              \
  t591 = my_piecewise3(t139,                                                       \
                       0,                                                          \
                       -0.3e1 / 0.8e1 * t148 * t581 +                              \
                         0.63344011248391453166e0 * t462 * t587);                  \
  tvsigma2 = t7 * t591;                                                            \
                                                                                   \
  tvlapl0 = 0.e0;                                                                  \
  tvlapl1 = 0.e0;                                                                  \
                                                                                   \
  t592   = t72 * t83;                                                              \
  t593   = my_piecewise3(t86, 0, t592);                                            \
  t594   = params.c1 * t593;                                                       \
  t596   = t295 * t593;                                                            \
  t598   = -t594 * t90 - t596 * t88;                                               \
  t599   = t598 * t92;                                                             \
  t600   = my_piecewise3(t94, 0, t592);                                            \
  t602   = t95 * t600;                                                             \
  t604   = t97 * t600;                                                             \
  t606   = t99 * t600;                                                             \
  t608   = t101 * t600;                                                            \
  t610   = t103 * t600;                                                            \
  t615   = my_piecewise3(t94, t592, 0);                                            \
  t619   = my_piecewise5(t85,                                                      \
                       t599,                                                     \
                       t93,                                                      \
                       -0.667e0 * t600 - 0.889111e0 * t602 -                     \
                         0.1989259803147e1 * t604 + 0.580518817796e1 * t606 -    \
                         0.4439990207985e1 * t608 + 0.1407173648874e1 * t610 -   \
                         0.162300903254e0 * t105 * t600,                         \
                       -t315 * t317 * t615 * t114);                              \
  t620   = t28 * t619;                                                             \
  t621   = t117 * t134;                                                            \
  t622   = t620 * t621;                                                            \
  t625   = my_piecewise3(t2, 0, -0.3e1 / 0.8e1 * t27 * t622);                      \
  tvtau0 = t7 * t625;                                                              \
                                                                                   \
  t626   = t178 * t187;                                                            \
  t627   = my_piecewise3(t190, 0, t626);                                           \
  t628   = params.c1 * t627;                                                       \
  t630   = t422 * t627;                                                            \
  t632   = -t192 * t630 - t194 * t628;                                             \
  t633   = t632 * t196;                                                            \
  t634   = my_piecewise3(t198, 0, t626);                                           \
  t636   = t199 * t634;                                                            \
  t638   = t201 * t634;                                                            \
  t640   = t203 * t634;                                                            \
  t642   = t205 * t634;                                                            \
  t644   = t207 * t634;                                                            \
  t649   = my_piecewise3(t198, t626, 0);                                           \
  t653   = my_piecewise5(t189,                                                     \
                       t633,                                                     \
                       t197,                                                     \
                       -0.667e0 * t634 - 0.889111e0 * t636 -                     \
                         0.1989259803147e1 * t638 + 0.580518817796e1 * t640 -    \
                         0.4439990207985e1 * t642 + 0.1407173648874e1 * t644 -   \
                         0.162300903254e0 * t209 * t634,                         \
                       -t315 * t443 * t649 * t218);                              \
  t654   = t28 * t653;                                                             \
  t655   = t221 * t235;                                                            \
  t656   = t654 * t655;                                                            \
  t659   = my_piecewise3(t139, 0, -0.3e1 / 0.8e1 * t148 * t656);                   \
  tvtau1 = t7 * t659;

#endif
