// ============================================================
// AUTO-EXTRACTED device-helper fragment (SCAN).
// NOT a standalone header. It is #included INSIDE `namespace dftfe`
// by exchangeCorrelationFunctionalEvaluatorDevice.cc (and is reusable by
// the CPU evaluator TU). It depends on, and must be included after:
//   - DFTFE_DEVICE_NOINLINE              (qualifier macro)
//   - the MGGA_X/C_SCAN_* computation macros (XCfunctionalDefs headers)
//   - the tzk0/tvrho0/... locals provided by the evaluator BODY macros
// The trailing MGGA_X/C_SCAN redefinitions remap those macros to call
// the helpers below, and must be in effect before the .def is included.
// ============================================================
#ifndef DFTFE_MGGA_SCAN_DEVICE_HELPERS_H
#define DFTFE_MGGA_SCAN_DEVICE_HELPERS_H

// ============================================================
// SCAN split into per-output __noinline__ device helpers
// ============================================================
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_zk(double rho0,
               double rho1,
               double sigma0,
               double sigma1,
               double sigma2,
               double tau0,
               double tau1)
{
  MGGA_C_SCAN_ZK
  return tzk0;
}
// ---- mgga_c_scan_vrho0: recursively decomposed into bounded-cone
// __noinline__ sub-helpers ----
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho0__t90(double, double, double, double, double, double, double);
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho0__t141(double, double, double, double, double, double, double);
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho0__t147(double, double, double, double, double, double, double);
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho0__t210(double, double, double, double, double, double, double);
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho0__t238(double, double, double, double, double, double, double);
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho0__t247(double, double, double, double, double, double, double);
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho0__t248(double, double, double, double, double, double, double);
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho0__t283(double, double, double, double, double, double, double);
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho0__t288(double, double, double, double, double, double, double);
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho0__t302(double, double, double, double, double, double, double);
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho0__t335(double, double, double, double, double, double, double);
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho0__t337(double, double, double, double, double, double, double);
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho0__t342(double, double, double, double, double, double, double);
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho0__t347(double, double, double, double, double, double, double);
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho0__t361(double, double, double, double, double, double, double);
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho0__t363(double, double, double, double, double, double, double);
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho0__t381(double, double, double, double, double, double, double);
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho0__t391(double, double, double, double, double, double, double);
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho0__t402(double, double, double, double, double, double, double);
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho0__t411(double, double, double, double, double, double, double);
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho0__t427(double, double, double, double, double, double, double);
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho0__t429(double, double, double, double, double, double, double);
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho0__t478(double, double, double, double, double, double, double);
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho0__t503(double, double, double, double, double, double, double);
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho0__t518(double, double, double, double, double, double, double);
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho0(double, double, double, double, double, double, double);

DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho0__t90(double rho0,
                       double rho1,
                       double sigma0,
                       double sigma1,
                       double sigma2,
                       double tau0,
                       double tau1)
{
  double t2  = M_CBRT3;
  double t3  = 0.1e1 / M_PI;
  double t4  = POW_1_3(t3);
  double t5  = t2 * t4;
  double t6  = M_CBRT4;
  double t7  = t6 * t6;
  double t8  = rho0 + rho1;
  double t9  = POW_1_3(t8);
  double t12 = t5 * t7 / t9;
  double t14 = 0.1e1 + 0.53425e-1 * t12;
  double t15 = sqrt(t12);
  double t18 = POW_3_2(t12);
  double t20 = t2 * t2;
  double t21 = t4 * t4;
  double t22 = t20 * t21;
  double t23 = t9 * t9;
  double t26 = t22 * t6 / t23;
  double t28 =
    0.379785e1 * t15 + 0.8969e0 * t12 + 0.204775e0 * t18 + 0.123235e0 * t26;
  double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
  double t32 = log(t31);
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
  double t45 = t44 <= ZETA_THRESHOLD_C_SCAN;
  double t46 = POW_1_3(ZETA_THRESHOLD_C_SCAN);
  double t47 = t46 * ZETA_THRESHOLD_C_SCAN;
  double t48 = POW_1_3(t44);
  double t49 = t48 * t44;
  double t50 = my_piecewise3(t45, t47, t49);
  double t51 = 0.1e1 - t43;
  double t52 = t51 <= ZETA_THRESHOLD_C_SCAN;
  double t53 = POW_1_3(t51);
  double t54 = t53 * t51;
  double t55 = my_piecewise3(t52, t47, t54);
  double t56 = t50 + t55 - 0.2e1;
  double t57 = M_CBRT2;
  double t58 = t57 - 0.1e1;
  double t60 = 0.1e1 / t58 / 0.2e1;
  double t61 = t56 * t60;
  double t63 = 0.1e1 + 0.5137e-1 * t12;
  double t68 =
    0.705945e1 * t15 + 0.1549425e1 * t12 + 0.420775e0 * t18 + 0.1562925e0 * t26;
  double t71 = 0.1e1 + 0.32163958997385070134e2 / t68;
  double t72 = log(t71);
  double t76 = 0.1e1 + 0.278125e-1 * t12;
  double t81 =
    0.51785e1 * t15 + 0.905775e0 * t12 + 0.1100325e0 * t18 + 0.1241775e0 * t26;
  double t84 = 0.1e1 + 0.29608749977793437516e2 / t81;
  double t85 = log(t84);
  double t86 = t76 * t85;
  double t88 = -0.310907e-1 * t63 * t72 + t34 - 0.19751673498613801407e-1 * t86;
  double t89 = t61 * t88;
  double t90 = t41 * t89;
  return t90;
}
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho0__t141(double rho0,
                        double rho1,
                        double sigma0,
                        double sigma1,
                        double sigma2,
                        double tau0,
                        double tau1)
{
  double t90 =
    mgga_c_scan_vrho0__t90(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t2  = M_CBRT3;
  double t3  = 0.1e1 / M_PI;
  double t4  = POW_1_3(t3);
  double t5  = t2 * t4;
  double t6  = M_CBRT4;
  double t7  = t6 * t6;
  double t8  = rho0 + rho1;
  double t9  = POW_1_3(t8);
  double t12 = t5 * t7 / t9;
  double t14 = 0.1e1 + 0.53425e-1 * t12;
  double t15 = sqrt(t12);
  double t18 = POW_3_2(t12);
  double t20 = t2 * t2;
  double t21 = t4 * t4;
  double t22 = t20 * t21;
  double t23 = t9 * t9;
  double t26 = t22 * t6 / t23;
  double t28 =
    0.379785e1 * t15 + 0.8969e0 * t12 + 0.204775e0 * t18 + 0.123235e0 * t26;
  double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
  double t32 = log(t31);
  double t34 = 0.621814e-1 * t14 * t32;
  double t35 = rho0 - rho1;
  double t38 = t8 * t8;
  double t42 = 0.1e1 / t8;
  double t43 = t35 * t42;
  double t44 = 0.1e1 + t43;
  double t45 = t44 <= ZETA_THRESHOLD_C_SCAN;
  double t46 = POW_1_3(ZETA_THRESHOLD_C_SCAN);
  double t47 = t46 * ZETA_THRESHOLD_C_SCAN;
  double t48 = POW_1_3(t44);
  double t49 = t48 * t44;
  double t50 = my_piecewise3(t45, t47, t49);
  double t51 = 0.1e1 - t43;
  double t52 = t51 <= ZETA_THRESHOLD_C_SCAN;
  double t53 = POW_1_3(t51);
  double t54 = t53 * t51;
  double t55 = my_piecewise3(t52, t47, t54);
  double t56 = t50 + t55 - 0.2e1;
  double t57 = M_CBRT2;
  double t58 = t57 - 0.1e1;
  double t60 = 0.1e1 / t58 / 0.2e1;
  double t61 = t56 * t60;
  double t76 = 0.1e1 + 0.278125e-1 * t12;
  double t81 =
    0.51785e1 * t15 + 0.905775e0 * t12 + 0.1100325e0 * t18 + 0.1241775e0 * t26;
  double t84  = 0.1e1 + 0.29608749977793437516e2 / t81;
  double t85  = log(t84);
  double t86  = t76 * t85;
  double t92  = 0.19751673498613801407e-1 * t61 * t86;
  double t93  = log(0.2e1);
  double t94  = 0.1e1 - t93;
  double t95  = M_PI * M_PI;
  double t98  = t46 * t46;
  double t99  = t48 * t48;
  double t100 = my_piecewise3(t45, t98, t99);
  double t101 = t53 * t53;
  double t102 = my_piecewise3(t52, t98, t101);
  double t104 = t100 / 0.2e1 + t102 / 0.2e1;
  double t105 = t104 * t104;
  double t106 = t105 * t104;
  double t108 = 0.1e1 + 0.25e-1 * t12;
  double t110 = 0.1e1 + 0.4445e-1 * t12;
  double t111 = 0.1e1 / t110;
  double t112 = t108 * t111;
  double t113 = 0.1e1 / t94;
  double t115 = (-t34 + t90 + t92) * t113;
  double t116 = 0.1e1 / t106;
  double t117 = t95 * t116;
  double t119 = exp(-t115 * t117);
  double t120 = t119 - 0.1e1;
  double t121 = 0.1e1 / t120;
  double t122 = t113 * t121;
  double t124 = sigma0 + 0.2e1 * sigma1 + sigma2;
  double t125 = t122 * t124;
  double t126 = t112 * t125;
  double t128 = 0.1e1 / t9 / t38;
  double t129 = t128 * t57;
  double t130 = 0.1e1 / t105;
  double t132 = 0.1e1 / t4;
  double t133 = t20 * t132;
  double t134 = t133 * t6;
  double t138 = 0.1e1 + 0.27439371595564631661e-1 * t126 * t129 * t130 * t134;
  double t139 = POW_1_4(t138);
  double t141 = 0.1e1 - 0.1e1 / t139;
  return t141;
}
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho0__t147(double rho0,
                        double rho1,
                        double sigma0,
                        double sigma1,
                        double sigma2,
                        double tau0,
                        double tau1)
{
  double t90 =
    mgga_c_scan_vrho0__t90(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t141 =
    mgga_c_scan_vrho0__t141(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t2  = M_CBRT3;
  double t3  = 0.1e1 / M_PI;
  double t4  = POW_1_3(t3);
  double t5  = t2 * t4;
  double t6  = M_CBRT4;
  double t7  = t6 * t6;
  double t8  = rho0 + rho1;
  double t9  = POW_1_3(t8);
  double t12 = t5 * t7 / t9;
  double t14 = 0.1e1 + 0.53425e-1 * t12;
  double t15 = sqrt(t12);
  double t18 = POW_3_2(t12);
  double t20 = t2 * t2;
  double t21 = t4 * t4;
  double t22 = t20 * t21;
  double t23 = t9 * t9;
  double t26 = t22 * t6 / t23;
  double t28 =
    0.379785e1 * t15 + 0.8969e0 * t12 + 0.204775e0 * t18 + 0.123235e0 * t26;
  double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
  double t32 = log(t31);
  double t34 = 0.621814e-1 * t14 * t32;
  double t35 = rho0 - rho1;
  double t42 = 0.1e1 / t8;
  double t43 = t35 * t42;
  double t44 = 0.1e1 + t43;
  double t45 = t44 <= ZETA_THRESHOLD_C_SCAN;
  double t46 = POW_1_3(ZETA_THRESHOLD_C_SCAN);
  double t47 = t46 * ZETA_THRESHOLD_C_SCAN;
  double t48 = POW_1_3(t44);
  double t49 = t48 * t44;
  double t50 = my_piecewise3(t45, t47, t49);
  double t51 = 0.1e1 - t43;
  double t52 = t51 <= ZETA_THRESHOLD_C_SCAN;
  double t53 = POW_1_3(t51);
  double t54 = t53 * t51;
  double t55 = my_piecewise3(t52, t47, t54);
  double t56 = t50 + t55 - 0.2e1;
  double t57 = M_CBRT2;
  double t58 = t57 - 0.1e1;
  double t60 = 0.1e1 / t58 / 0.2e1;
  double t61 = t56 * t60;
  double t76 = 0.1e1 + 0.278125e-1 * t12;
  double t81 =
    0.51785e1 * t15 + 0.905775e0 * t12 + 0.1100325e0 * t18 + 0.1241775e0 * t26;
  double t84  = 0.1e1 + 0.29608749977793437516e2 / t81;
  double t85  = log(t84);
  double t86  = t76 * t85;
  double t92  = 0.19751673498613801407e-1 * t61 * t86;
  double t93  = log(0.2e1);
  double t94  = 0.1e1 - t93;
  double t95  = M_PI * M_PI;
  double t97  = t94 / t95;
  double t98  = t46 * t46;
  double t99  = t48 * t48;
  double t100 = my_piecewise3(t45, t98, t99);
  double t101 = t53 * t53;
  double t102 = my_piecewise3(t52, t98, t101);
  double t104 = t100 / 0.2e1 + t102 / 0.2e1;
  double t105 = t104 * t104;
  double t106 = t105 * t104;
  double t113 = 0.1e1 / t94;
  double t115 = (-t34 + t90 + t92) * t113;
  double t116 = 0.1e1 / t106;
  double t117 = t95 * t116;
  double t119 = exp(-t115 * t117);
  double t120 = t119 - 0.1e1;
  double t144 = 0.1e1 + 0.1e1 * t141 * t120;
  double t145 = log(t144);
  double t147 = t97 * t106 * t145;
  return t147;
}
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho0__t210(double rho0,
                        double rho1,
                        double sigma0,
                        double sigma1,
                        double sigma2,
                        double tau0,
                        double tau1)
{
  double t8   = rho0 + rho1;
  double t9   = POW_1_3(t8);
  double t23  = t9 * t9;
  double t35  = rho0 - rho1;
  double t38  = t8 * t8;
  double t42  = 0.1e1 / t8;
  double t43  = t35 * t42;
  double t44  = 0.1e1 + t43;
  double t51  = 0.1e1 - t43;
  double t95  = M_PI * M_PI;
  double t124 = sigma0 + 0.2e1 * sigma1 + sigma2;
  double t148 = POW_1_3(rho0);
  double t149 = t148 * t148;
  double t151 = 0.1e1 / t149 / rho0;
  double t152 = tau0 * t151;
  double t153 = t44 / 0.2e1;
  double t154 = POW_1_3(t153);
  double t155 = t154 * t154;
  double t156 = t155 * t153;
  double t158 = POW_1_3(rho1);
  double t159 = t158 * t158;
  double t161 = 0.1e1 / t159 / rho1;
  double t162 = tau1 * t161;
  double t163 = t51 / 0.2e1;
  double t164 = POW_1_3(t163);
  double t165 = t164 * t164;
  double t166 = t165 * t163;
  double t169 = 0.1e1 / t23 / t38;
  double t173 = M_CBRT6;
  double t174 = (t152 * t156 + t162 * t166 - t124 * t169 / 0.8e1) * t173;
  double t175 = POW_1_3(t95);
  double t176 = t175 * t175;
  double t177 = 0.1e1 / t176;
  double t178 = t156 + t166;
  double t179 = 0.1e1 / t178;
  double t180 = t177 * t179;
  double t182 = 0.5e1 / 0.9e1 * t174 * t180;
  double t183 = t182 <= 0.1e1;
  double t184 = log(DBL_EPSILON);
  double t187 = t184 / (-t184 + 0.64e0);
  double t188 = -t187 < t182;
  double t189 = t182 < -t187;
  double t190 = my_piecewise3(t189, t182, -t187);
  double t191 = 0.1e1 - t190;
  double t192 = 0.1e1 / t191;
  double t195 = exp(-0.64e0 * t190 * t192);
  double t196 = my_piecewise3(t188, 0, t195);
  double t198 = log(0.14285714285714285714e1 * DBL_EPSILON);
  double t201 = (-t198 + 0.15e1) / t198;
  double t202 = t182 < -t201;
  double t203 = my_piecewise3(t202, -t201, t182);
  double t204 = 0.1e1 - t203;
  double t207 = exp(0.15e1 / t204);
  double t209 = my_piecewise3(t202, 0, -0.7e0 * t207);
  double t210 = my_piecewise3(t183, t196, t209);
  return t210;
}
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho0__t238(double rho0,
                        double rho1,
                        double sigma0,
                        double sigma1,
                        double sigma2,
                        double tau0,
                        double tau1)
{
  double t2   = M_CBRT3;
  double t3   = 0.1e1 / M_PI;
  double t4   = POW_1_3(t3);
  double t5   = t2 * t4;
  double t6   = M_CBRT4;
  double t7   = t6 * t6;
  double t8   = rho0 + rho1;
  double t9   = POW_1_3(t8);
  double t12  = t5 * t7 / t9;
  double t15  = sqrt(t12);
  double t23  = t9 * t9;
  double t35  = rho0 - rho1;
  double t38  = t8 * t8;
  double t42  = 0.1e1 / t8;
  double t43  = t35 * t42;
  double t44  = 0.1e1 + t43;
  double t45  = t44 <= ZETA_THRESHOLD_C_SCAN;
  double t46  = POW_1_3(ZETA_THRESHOLD_C_SCAN);
  double t47  = t46 * ZETA_THRESHOLD_C_SCAN;
  double t48  = POW_1_3(t44);
  double t49  = t48 * t44;
  double t50  = my_piecewise3(t45, t47, t49);
  double t51  = 0.1e1 - t43;
  double t52  = t51 <= ZETA_THRESHOLD_C_SCAN;
  double t53  = POW_1_3(t51);
  double t54  = t53 * t51;
  double t55  = my_piecewise3(t52, t47, t54);
  double t56  = t50 + t55 - 0.2e1;
  double t57  = M_CBRT2;
  double t58  = t57 - 0.1e1;
  double t60  = 0.1e1 / t58 / 0.2e1;
  double t95  = M_PI * M_PI;
  double t124 = sigma0 + 0.2e1 * sigma1 + sigma2;
  double t169 = 0.1e1 / t23 / t38;
  double t173 = M_CBRT6;
  double t175 = POW_1_3(t95);
  double t176 = t175 * t175;
  double t177 = 0.1e1 / t176;
  double t213 = 0.1e1 + 0.4445e-1 * t15 + 0.3138525e-1 * t12;
  double t214 = 0.1e1 / t213;
  double t217 = exp(0.1e1 * t214);
  double t218 = t217 - 0.1e1;
  double t219 = t173 * t177;
  double t220 = t57 * t57;
  double t221 = t220 * t124;
  double t225 = 0.1e1 + 0.21337642104376358333e-1 * t219 * t221 * t169;
  double t226 = POW_1_4(t225);
  double t228 = 0.1e1 - 0.1e1 / t226;
  double t230 = t218 * t228 + 0.1e1;
  double t231 = log(t230);
  double t233 = -0.285764e-1 * t214 + 0.285764e-1 * t231;
  double t237 = 0.1e1 - 0.2363e1 * t58 * t56 * t60;
  double t238 = t233 * t237;
  return t238;
}
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho0__t247(double rho0,
                        double rho1,
                        double sigma0,
                        double sigma1,
                        double sigma2,
                        double tau0,
                        double tau1)
{
  double t90 =
    mgga_c_scan_vrho0__t90(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t147 =
    mgga_c_scan_vrho0__t147(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t238 =
    mgga_c_scan_vrho0__t238(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t2  = M_CBRT3;
  double t3  = 0.1e1 / M_PI;
  double t4  = POW_1_3(t3);
  double t5  = t2 * t4;
  double t6  = M_CBRT4;
  double t7  = t6 * t6;
  double t8  = rho0 + rho1;
  double t9  = POW_1_3(t8);
  double t12 = t5 * t7 / t9;
  double t14 = 0.1e1 + 0.53425e-1 * t12;
  double t15 = sqrt(t12);
  double t18 = POW_3_2(t12);
  double t20 = t2 * t2;
  double t21 = t4 * t4;
  double t22 = t20 * t21;
  double t23 = t9 * t9;
  double t26 = t22 * t6 / t23;
  double t28 =
    0.379785e1 * t15 + 0.8969e0 * t12 + 0.204775e0 * t18 + 0.123235e0 * t26;
  double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
  double t32 = log(t31);
  double t34 = 0.621814e-1 * t14 * t32;
  double t35 = rho0 - rho1;
  double t36 = t35 * t35;
  double t37 = t36 * t36;
  double t38 = t8 * t8;
  double t39 = t38 * t38;
  double t42 = 0.1e1 / t8;
  double t43 = t35 * t42;
  double t44 = 0.1e1 + t43;
  double t45 = t44 <= ZETA_THRESHOLD_C_SCAN;
  double t46 = POW_1_3(ZETA_THRESHOLD_C_SCAN);
  double t47 = t46 * ZETA_THRESHOLD_C_SCAN;
  double t48 = POW_1_3(t44);
  double t49 = t48 * t44;
  double t50 = my_piecewise3(t45, t47, t49);
  double t51 = 0.1e1 - t43;
  double t52 = t51 <= ZETA_THRESHOLD_C_SCAN;
  double t53 = POW_1_3(t51);
  double t54 = t53 * t51;
  double t55 = my_piecewise3(t52, t47, t54);
  double t56 = t50 + t55 - 0.2e1;
  double t57 = M_CBRT2;
  double t58 = t57 - 0.1e1;
  double t60 = 0.1e1 / t58 / 0.2e1;
  double t61 = t56 * t60;
  double t76 = 0.1e1 + 0.278125e-1 * t12;
  double t81 =
    0.51785e1 * t15 + 0.905775e0 * t12 + 0.1100325e0 * t18 + 0.1241775e0 * t26;
  double t84  = 0.1e1 + 0.29608749977793437516e2 / t81;
  double t85  = log(t84);
  double t86  = t76 * t85;
  double t92  = 0.19751673498613801407e-1 * t61 * t86;
  double t239 = t37 * t37;
  double t240 = t239 * t37;
  double t241 = t39 * t39;
  double t242 = t241 * t39;
  double t243 = 0.1e1 / t242;
  double t245 = -t240 * t243 + 0.1e1;
  double t247 = t238 * t245 - t147 + t34 - t90 - t92;
  return t247;
}
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho0__t248(double rho0,
                        double rho1,
                        double sigma0,
                        double sigma1,
                        double sigma2,
                        double tau0,
                        double tau1)
{
  double t210 =
    mgga_c_scan_vrho0__t210(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t247 =
    mgga_c_scan_vrho0__t247(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t248 = t210 * t247;
  return t248;
}
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho0__t283(double rho0,
                        double rho1,
                        double sigma0,
                        double sigma1,
                        double sigma2,
                        double tau0,
                        double tau1)
{
  double t2  = M_CBRT3;
  double t3  = 0.1e1 / M_PI;
  double t4  = POW_1_3(t3);
  double t5  = t2 * t4;
  double t6  = M_CBRT4;
  double t7  = t6 * t6;
  double t8  = rho0 + rho1;
  double t9  = POW_1_3(t8);
  double t12 = t5 * t7 / t9;
  double t14 = 0.1e1 + 0.53425e-1 * t12;
  double t15 = sqrt(t12);
  double t18 = POW_3_2(t12);
  double t20 = t2 * t2;
  double t21 = t4 * t4;
  double t22 = t20 * t21;
  double t23 = t9 * t9;
  double t26 = t22 * t6 / t23;
  double t28 =
    0.379785e1 * t15 + 0.8969e0 * t12 + 0.204775e0 * t18 + 0.123235e0 * t26;
  double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
  double t32 = log(t31);
  double t34 = 0.621814e-1 * t14 * t32;
  double t35 = rho0 - rho1;
  double t36 = t35 * t35;
  double t38 = t8 * t8;
  double t39 = t38 * t38;
  double t40 = 0.1e1 / t39;
  double t42 = 0.1e1 / t8;
  double t43 = t35 * t42;
  double t44 = 0.1e1 + t43;
  double t45 = t44 <= ZETA_THRESHOLD_C_SCAN;
  double t46 = POW_1_3(ZETA_THRESHOLD_C_SCAN);
  double t47 = t46 * ZETA_THRESHOLD_C_SCAN;
  double t48 = POW_1_3(t44);
  double t49 = t48 * t44;
  double t50 = my_piecewise3(t45, t47, t49);
  double t51 = 0.1e1 - t43;
  double t52 = t51 <= ZETA_THRESHOLD_C_SCAN;
  double t53 = POW_1_3(t51);
  double t54 = t53 * t51;
  double t55 = my_piecewise3(t52, t47, t54);
  double t56 = t50 + t55 - 0.2e1;
  double t57 = M_CBRT2;
  double t58 = t57 - 0.1e1;
  double t60 = 0.1e1 / t58 / 0.2e1;
  double t61 = t56 * t60;
  double t63 = 0.1e1 + 0.5137e-1 * t12;
  double t68 =
    0.705945e1 * t15 + 0.1549425e1 * t12 + 0.420775e0 * t18 + 0.1562925e0 * t26;
  double t71 = 0.1e1 + 0.32163958997385070134e2 / t68;
  double t72 = log(t71);
  double t76 = 0.1e1 + 0.278125e-1 * t12;
  double t81 =
    0.51785e1 * t15 + 0.905775e0 * t12 + 0.1100325e0 * t18 + 0.1241775e0 * t26;
  double t84 = 0.1e1 + 0.29608749977793437516e2 / t81;
  double t85 = log(t84);
  double t86 = t76 * t85;
  double t88 = -0.310907e-1 * t63 * t72 + t34 - 0.19751673498613801407e-1 * t86;
  double t89 = t61 * t88;
  double t280 = t36 * t35;
  double t281 = t280 * t40;
  double t282 = t281 * t89;
  double t283 = 0.4e1 * t282;
  return t283;
}
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho0__t288(double rho0,
                        double rho1,
                        double sigma0,
                        double sigma1,
                        double sigma2,
                        double tau0,
                        double tau1)
{
  double t2  = M_CBRT3;
  double t3  = 0.1e1 / M_PI;
  double t4  = POW_1_3(t3);
  double t5  = t2 * t4;
  double t6  = M_CBRT4;
  double t7  = t6 * t6;
  double t8  = rho0 + rho1;
  double t9  = POW_1_3(t8);
  double t12 = t5 * t7 / t9;
  double t14 = 0.1e1 + 0.53425e-1 * t12;
  double t15 = sqrt(t12);
  double t18 = POW_3_2(t12);
  double t20 = t2 * t2;
  double t21 = t4 * t4;
  double t22 = t20 * t21;
  double t23 = t9 * t9;
  double t26 = t22 * t6 / t23;
  double t28 =
    0.379785e1 * t15 + 0.8969e0 * t12 + 0.204775e0 * t18 + 0.123235e0 * t26;
  double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
  double t32 = log(t31);
  double t34 = 0.621814e-1 * t14 * t32;
  double t35 = rho0 - rho1;
  double t36 = t35 * t35;
  double t37 = t36 * t36;
  double t38 = t8 * t8;
  double t39 = t38 * t38;
  double t42 = 0.1e1 / t8;
  double t43 = t35 * t42;
  double t44 = 0.1e1 + t43;
  double t45 = t44 <= ZETA_THRESHOLD_C_SCAN;
  double t46 = POW_1_3(ZETA_THRESHOLD_C_SCAN);
  double t47 = t46 * ZETA_THRESHOLD_C_SCAN;
  double t48 = POW_1_3(t44);
  double t49 = t48 * t44;
  double t50 = my_piecewise3(t45, t47, t49);
  double t51 = 0.1e1 - t43;
  double t52 = t51 <= ZETA_THRESHOLD_C_SCAN;
  double t53 = POW_1_3(t51);
  double t54 = t53 * t51;
  double t55 = my_piecewise3(t52, t47, t54);
  double t56 = t50 + t55 - 0.2e1;
  double t57 = M_CBRT2;
  double t58 = t57 - 0.1e1;
  double t60 = 0.1e1 / t58 / 0.2e1;
  double t61 = t56 * t60;
  double t63 = 0.1e1 + 0.5137e-1 * t12;
  double t68 =
    0.705945e1 * t15 + 0.1549425e1 * t12 + 0.420775e0 * t18 + 0.1562925e0 * t26;
  double t71 = 0.1e1 + 0.32163958997385070134e2 / t68;
  double t72 = log(t71);
  double t76 = 0.1e1 + 0.278125e-1 * t12;
  double t81 =
    0.51785e1 * t15 + 0.905775e0 * t12 + 0.1100325e0 * t18 + 0.1241775e0 * t26;
  double t84 = 0.1e1 + 0.29608749977793437516e2 / t81;
  double t85 = log(t84);
  double t86 = t76 * t85;
  double t88 = -0.310907e-1 * t63 * t72 + t34 - 0.19751673498613801407e-1 * t86;
  double t89 = t61 * t88;
  double t284 = t39 * t8;
  double t285 = 0.1e1 / t284;
  double t286 = t37 * t285;
  double t287 = t286 * t89;
  double t288 = 0.4e1 * t287;
  return t288;
}
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho0__t302(double rho0,
                        double rho1,
                        double sigma0,
                        double sigma1,
                        double sigma2,
                        double tau0,
                        double tau1)
{
  double t2  = M_CBRT3;
  double t3  = 0.1e1 / M_PI;
  double t4  = POW_1_3(t3);
  double t5  = t2 * t4;
  double t6  = M_CBRT4;
  double t7  = t6 * t6;
  double t8  = rho0 + rho1;
  double t9  = POW_1_3(t8);
  double t12 = t5 * t7 / t9;
  double t14 = 0.1e1 + 0.53425e-1 * t12;
  double t15 = sqrt(t12);
  double t18 = POW_3_2(t12);
  double t20 = t2 * t2;
  double t21 = t4 * t4;
  double t22 = t20 * t21;
  double t23 = t9 * t9;
  double t26 = t22 * t6 / t23;
  double t28 =
    0.379785e1 * t15 + 0.8969e0 * t12 + 0.204775e0 * t18 + 0.123235e0 * t26;
  double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
  double t32 = log(t31);
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
  double t45 = t44 <= ZETA_THRESHOLD_C_SCAN;
  double t48 = POW_1_3(t44);
  double t51 = 0.1e1 - t43;
  double t52 = t51 <= ZETA_THRESHOLD_C_SCAN;
  double t53 = POW_1_3(t51);
  double t57 = M_CBRT2;
  double t58 = t57 - 0.1e1;
  double t60 = 0.1e1 / t58 / 0.2e1;
  double t63 = 0.1e1 + 0.5137e-1 * t12;
  double t68 =
    0.705945e1 * t15 + 0.1549425e1 * t12 + 0.420775e0 * t18 + 0.1562925e0 * t26;
  double t71 = 0.1e1 + 0.32163958997385070134e2 / t68;
  double t72 = log(t71);
  double t76 = 0.1e1 + 0.278125e-1 * t12;
  double t81 =
    0.51785e1 * t15 + 0.905775e0 * t12 + 0.1100325e0 * t18 + 0.1241775e0 * t26;
  double t84 = 0.1e1 + 0.29608749977793437516e2 / t81;
  double t85 = log(t84);
  double t86 = t76 * t85;
  double t88 = -0.310907e-1 * t63 * t72 + t34 - 0.19751673498613801407e-1 * t86;
  double t289 = 0.1e1 / t38;
  double t290 = t35 * t289;
  double t291 = t42 - t290;
  double t294 = my_piecewise3(t45, 0, 0.4e1 / 0.3e1 * t48 * t291);
  double t295 = -t291;
  double t298 = my_piecewise3(t52, 0, 0.4e1 / 0.3e1 * t53 * t295);
  double t299 = t294 + t298;
  double t300 = t299 * t60;
  double t301 = t300 * t88;
  double t302 = t41 * t301;
  return t302;
}
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho0__t335(double rho0,
                        double rho1,
                        double sigma0,
                        double sigma1,
                        double sigma2,
                        double tau0,
                        double tau1)
{
  double t2  = M_CBRT3;
  double t3  = 0.1e1 / M_PI;
  double t4  = POW_1_3(t3);
  double t5  = t2 * t4;
  double t6  = M_CBRT4;
  double t7  = t6 * t6;
  double t8  = rho0 + rho1;
  double t9  = POW_1_3(t8);
  double t12 = t5 * t7 / t9;
  double t14 = 0.1e1 + 0.53425e-1 * t12;
  double t15 = sqrt(t12);
  double t18 = POW_3_2(t12);
  double t20 = t2 * t2;
  double t21 = t4 * t4;
  double t22 = t20 * t21;
  double t23 = t9 * t9;
  double t26 = t22 * t6 / t23;
  double t28 =
    0.379785e1 * t15 + 0.8969e0 * t12 + 0.204775e0 * t18 + 0.123235e0 * t26;
  double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
  double t32 = log(t31);
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
  double t45 = t44 <= ZETA_THRESHOLD_C_SCAN;
  double t46 = POW_1_3(ZETA_THRESHOLD_C_SCAN);
  double t47 = t46 * ZETA_THRESHOLD_C_SCAN;
  double t48 = POW_1_3(t44);
  double t49 = t48 * t44;
  double t50 = my_piecewise3(t45, t47, t49);
  double t51 = 0.1e1 - t43;
  double t52 = t51 <= ZETA_THRESHOLD_C_SCAN;
  double t53 = POW_1_3(t51);
  double t54 = t53 * t51;
  double t55 = my_piecewise3(t52, t47, t54);
  double t56 = t50 + t55 - 0.2e1;
  double t57 = M_CBRT2;
  double t58 = t57 - 0.1e1;
  double t60 = 0.1e1 / t58 / 0.2e1;
  double t61 = t56 * t60;
  double t63 = 0.1e1 + 0.5137e-1 * t12;
  double t68 =
    0.705945e1 * t15 + 0.1549425e1 * t12 + 0.420775e0 * t18 + 0.1562925e0 * t26;
  double t71 = 0.1e1 + 0.32163958997385070134e2 / t68;
  double t72 = log(t71);
  double t76 = 0.1e1 + 0.278125e-1 * t12;
  double t81 =
    0.51785e1 * t15 + 0.905775e0 * t12 + 0.1100325e0 * t18 + 0.1241775e0 * t26;
  double t84  = 0.1e1 + 0.29608749977793437516e2 / t81;
  double t85  = log(t84);
  double t250 = 0.1e1 / t9 / t8;
  double t251 = t7 * t250;
  double t253 = t5 * t251 * t32;
  double t254 = 0.11073470983333333333e-2 * t253;
  double t255 = t28 * t28;
  double t256 = 0.1e1 / t255;
  double t257 = t14 * t256;
  double t259 = 0.1e1 / t15 * t2;
  double t260 = t4 * t7;
  double t261 = t260 * t250;
  double t262 = t259 * t261;
  double t264 = t5 * t251;
  double t266 = sqrt(t12);
  double t267 = t266 * t2;
  double t268 = t267 * t261;
  double t273 = t22 * t6 / t23 / t8;
  double t275 = -0.632975e0 * t262 - 0.29896666666666666667e0 * t264 -
                0.1023875e0 * t268 - 0.82156666666666666667e-1 * t273;
  double t276 = 0.1e1 / t31;
  double t277 = t275 * t276;
  double t278 = t257 * t277;
  double t279 = 0.1e1 * t278;
  double t306 = t68 * t68;
  double t307 = 0.1e1 / t306;
  double t308 = t63 * t307;
  double t313 = -0.1176575e1 * t262 - 0.516475e0 * t264 - 0.2103875e0 * t268 -
                0.104195e0 * t273;
  double t314 = 0.1e1 / t71;
  double t315 = t313 * t314;
  double t321 = t81 * t81;
  double t322 = 0.1e1 / t321;
  double t323 = t76 * t322;
  double t328 = -0.86308333333333333334e0 * t262 - 0.301925e0 * t264 -
                0.5501625e-1 * t268 - 0.82785e-1 * t273;
  double t329 = 0.1e1 / t84;
  double t330 = t328 * t329;
  double t333 = 0.53237641966666666666e-3 * t5 * t251 * t72 +
                0.1e1 * t308 * t315 - t254 - t279 +
                0.18311447306006545054e-3 * t5 * t251 * t85 +
                0.5848223622634646207e0 * t323 * t330;
  double t334 = t61 * t333;
  double t335 = t41 * t334;
  return t335;
}
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho0__t337(double rho0,
                        double rho1,
                        double sigma0,
                        double sigma1,
                        double sigma2,
                        double tau0,
                        double tau1)
{
  double t2  = M_CBRT3;
  double t3  = 0.1e1 / M_PI;
  double t4  = POW_1_3(t3);
  double t5  = t2 * t4;
  double t6  = M_CBRT4;
  double t7  = t6 * t6;
  double t8  = rho0 + rho1;
  double t9  = POW_1_3(t8);
  double t12 = t5 * t7 / t9;
  double t15 = sqrt(t12);
  double t18 = POW_3_2(t12);
  double t20 = t2 * t2;
  double t21 = t4 * t4;
  double t22 = t20 * t21;
  double t23 = t9 * t9;
  double t26 = t22 * t6 / t23;
  double t35 = rho0 - rho1;
  double t38 = t8 * t8;
  double t42 = 0.1e1 / t8;
  double t43 = t35 * t42;
  double t44 = 0.1e1 + t43;
  double t45 = t44 <= ZETA_THRESHOLD_C_SCAN;
  double t48 = POW_1_3(t44);
  double t51 = 0.1e1 - t43;
  double t52 = t51 <= ZETA_THRESHOLD_C_SCAN;
  double t53 = POW_1_3(t51);
  double t57 = M_CBRT2;
  double t58 = t57 - 0.1e1;
  double t60 = 0.1e1 / t58 / 0.2e1;
  double t76 = 0.1e1 + 0.278125e-1 * t12;
  double t81 =
    0.51785e1 * t15 + 0.905775e0 * t12 + 0.1100325e0 * t18 + 0.1241775e0 * t26;
  double t84  = 0.1e1 + 0.29608749977793437516e2 / t81;
  double t85  = log(t84);
  double t86  = t76 * t85;
  double t289 = 0.1e1 / t38;
  double t290 = t35 * t289;
  double t291 = t42 - t290;
  double t294 = my_piecewise3(t45, 0, 0.4e1 / 0.3e1 * t48 * t291);
  double t295 = -t291;
  double t298 = my_piecewise3(t52, 0, 0.4e1 / 0.3e1 * t53 * t295);
  double t299 = t294 + t298;
  double t300 = t299 * t60;
  double t336 = t300 * t86;
  double t337 = 0.19751673498613801407e-1 * t336;
  return t337;
}
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho0__t342(double rho0,
                        double rho1,
                        double sigma0,
                        double sigma1,
                        double sigma2,
                        double tau0,
                        double tau1)
{
  double t2  = M_CBRT3;
  double t3  = 0.1e1 / M_PI;
  double t4  = POW_1_3(t3);
  double t5  = t2 * t4;
  double t6  = M_CBRT4;
  double t7  = t6 * t6;
  double t8  = rho0 + rho1;
  double t9  = POW_1_3(t8);
  double t12 = t5 * t7 / t9;
  double t15 = sqrt(t12);
  double t18 = POW_3_2(t12);
  double t20 = t2 * t2;
  double t21 = t4 * t4;
  double t22 = t20 * t21;
  double t23 = t9 * t9;
  double t26 = t22 * t6 / t23;
  double t35 = rho0 - rho1;
  double t42 = 0.1e1 / t8;
  double t43 = t35 * t42;
  double t44 = 0.1e1 + t43;
  double t45 = t44 <= ZETA_THRESHOLD_C_SCAN;
  double t46 = POW_1_3(ZETA_THRESHOLD_C_SCAN);
  double t47 = t46 * ZETA_THRESHOLD_C_SCAN;
  double t48 = POW_1_3(t44);
  double t49 = t48 * t44;
  double t50 = my_piecewise3(t45, t47, t49);
  double t51 = 0.1e1 - t43;
  double t52 = t51 <= ZETA_THRESHOLD_C_SCAN;
  double t53 = POW_1_3(t51);
  double t54 = t53 * t51;
  double t55 = my_piecewise3(t52, t47, t54);
  double t56 = t50 + t55 - 0.2e1;
  double t57 = M_CBRT2;
  double t58 = t57 - 0.1e1;
  double t60 = 0.1e1 / t58 / 0.2e1;
  double t61 = t56 * t60;
  double t81 =
    0.51785e1 * t15 + 0.905775e0 * t12 + 0.1100325e0 * t18 + 0.1241775e0 * t26;
  double t84  = 0.1e1 + 0.29608749977793437516e2 / t81;
  double t85  = log(t84);
  double t250 = 0.1e1 / t9 / t8;
  double t260 = t4 * t7;
  double t338 = t61 * t2;
  double t340 = t260 * t250 * t85;
  double t341 = t338 * t340;
  double t342 = 0.18311447306006545054e-3 * t341;
  return t342;
}
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho0__t347(double rho0,
                        double rho1,
                        double sigma0,
                        double sigma1,
                        double sigma2,
                        double tau0,
                        double tau1)
{
  double t2  = M_CBRT3;
  double t3  = 0.1e1 / M_PI;
  double t4  = POW_1_3(t3);
  double t5  = t2 * t4;
  double t6  = M_CBRT4;
  double t7  = t6 * t6;
  double t8  = rho0 + rho1;
  double t9  = POW_1_3(t8);
  double t12 = t5 * t7 / t9;
  double t15 = sqrt(t12);
  double t18 = POW_3_2(t12);
  double t20 = t2 * t2;
  double t21 = t4 * t4;
  double t22 = t20 * t21;
  double t23 = t9 * t9;
  double t26 = t22 * t6 / t23;
  double t35 = rho0 - rho1;
  double t42 = 0.1e1 / t8;
  double t43 = t35 * t42;
  double t44 = 0.1e1 + t43;
  double t45 = t44 <= ZETA_THRESHOLD_C_SCAN;
  double t46 = POW_1_3(ZETA_THRESHOLD_C_SCAN);
  double t47 = t46 * ZETA_THRESHOLD_C_SCAN;
  double t48 = POW_1_3(t44);
  double t49 = t48 * t44;
  double t50 = my_piecewise3(t45, t47, t49);
  double t51 = 0.1e1 - t43;
  double t52 = t51 <= ZETA_THRESHOLD_C_SCAN;
  double t53 = POW_1_3(t51);
  double t54 = t53 * t51;
  double t55 = my_piecewise3(t52, t47, t54);
  double t56 = t50 + t55 - 0.2e1;
  double t57 = M_CBRT2;
  double t58 = t57 - 0.1e1;
  double t60 = 0.1e1 / t58 / 0.2e1;
  double t61 = t56 * t60;
  double t76 = 0.1e1 + 0.278125e-1 * t12;
  double t81 =
    0.51785e1 * t15 + 0.905775e0 * t12 + 0.1100325e0 * t18 + 0.1241775e0 * t26;
  double t84  = 0.1e1 + 0.29608749977793437516e2 / t81;
  double t250 = 0.1e1 / t9 / t8;
  double t251 = t7 * t250;
  double t259 = 0.1e1 / t15 * t2;
  double t260 = t4 * t7;
  double t261 = t260 * t250;
  double t262 = t259 * t261;
  double t264 = t5 * t251;
  double t266 = sqrt(t12);
  double t267 = t266 * t2;
  double t268 = t267 * t261;
  double t273 = t22 * t6 / t23 / t8;
  double t321 = t81 * t81;
  double t322 = 0.1e1 / t321;
  double t328 = -0.86308333333333333334e0 * t262 - 0.301925e0 * t264 -
                0.5501625e-1 * t268 - 0.82785e-1 * t273;
  double t329 = 0.1e1 / t84;
  double t343 = t61 * t76;
  double t345 = t322 * t328 * t329;
  double t346 = t343 * t345;
  double t347 = 0.5848223622634646207e0 * t346;
  return t347;
}
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho0__t361(double rho0,
                        double rho1,
                        double sigma0,
                        double sigma1,
                        double sigma2,
                        double tau0,
                        double tau1)
{
  double t90 =
    mgga_c_scan_vrho0__t90(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t141 =
    mgga_c_scan_vrho0__t141(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t2  = M_CBRT3;
  double t3  = 0.1e1 / M_PI;
  double t4  = POW_1_3(t3);
  double t5  = t2 * t4;
  double t6  = M_CBRT4;
  double t7  = t6 * t6;
  double t8  = rho0 + rho1;
  double t9  = POW_1_3(t8);
  double t12 = t5 * t7 / t9;
  double t14 = 0.1e1 + 0.53425e-1 * t12;
  double t15 = sqrt(t12);
  double t18 = POW_3_2(t12);
  double t20 = t2 * t2;
  double t21 = t4 * t4;
  double t22 = t20 * t21;
  double t23 = t9 * t9;
  double t26 = t22 * t6 / t23;
  double t28 =
    0.379785e1 * t15 + 0.8969e0 * t12 + 0.204775e0 * t18 + 0.123235e0 * t26;
  double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
  double t32 = log(t31);
  double t34 = 0.621814e-1 * t14 * t32;
  double t35 = rho0 - rho1;
  double t38 = t8 * t8;
  double t42 = 0.1e1 / t8;
  double t43 = t35 * t42;
  double t44 = 0.1e1 + t43;
  double t45 = t44 <= ZETA_THRESHOLD_C_SCAN;
  double t46 = POW_1_3(ZETA_THRESHOLD_C_SCAN);
  double t47 = t46 * ZETA_THRESHOLD_C_SCAN;
  double t48 = POW_1_3(t44);
  double t49 = t48 * t44;
  double t50 = my_piecewise3(t45, t47, t49);
  double t51 = 0.1e1 - t43;
  double t52 = t51 <= ZETA_THRESHOLD_C_SCAN;
  double t53 = POW_1_3(t51);
  double t54 = t53 * t51;
  double t55 = my_piecewise3(t52, t47, t54);
  double t56 = t50 + t55 - 0.2e1;
  double t57 = M_CBRT2;
  double t58 = t57 - 0.1e1;
  double t60 = 0.1e1 / t58 / 0.2e1;
  double t61 = t56 * t60;
  double t76 = 0.1e1 + 0.278125e-1 * t12;
  double t81 =
    0.51785e1 * t15 + 0.905775e0 * t12 + 0.1100325e0 * t18 + 0.1241775e0 * t26;
  double t84  = 0.1e1 + 0.29608749977793437516e2 / t81;
  double t85  = log(t84);
  double t86  = t76 * t85;
  double t92  = 0.19751673498613801407e-1 * t61 * t86;
  double t93  = log(0.2e1);
  double t94  = 0.1e1 - t93;
  double t95  = M_PI * M_PI;
  double t97  = t94 / t95;
  double t98  = t46 * t46;
  double t99  = t48 * t48;
  double t100 = my_piecewise3(t45, t98, t99);
  double t101 = t53 * t53;
  double t102 = my_piecewise3(t52, t98, t101);
  double t104 = t100 / 0.2e1 + t102 / 0.2e1;
  double t105 = t104 * t104;
  double t106 = t105 * t104;
  double t113 = 0.1e1 / t94;
  double t115 = (-t34 + t90 + t92) * t113;
  double t116 = 0.1e1 / t106;
  double t117 = t95 * t116;
  double t119 = exp(-t115 * t117);
  double t120 = t119 - 0.1e1;
  double t144 = 0.1e1 + 0.1e1 * t141 * t120;
  double t145 = log(t144);
  double t289 = 0.1e1 / t38;
  double t290 = t35 * t289;
  double t291 = t42 - t290;
  double t295 = -t291;
  double t348 = t105 * t145;
  double t349 = 0.1e1 / t48;
  double t352 = my_piecewise3(t45, 0, 0.2e1 / 0.3e1 * t349 * t291);
  double t353 = 0.1e1 / t53;
  double t356 = my_piecewise3(t52, 0, 0.2e1 / 0.3e1 * t353 * t295);
  double t358 = t352 / 0.2e1 + t356 / 0.2e1;
  double t360 = t97 * t348 * t358;
  double t361 = 0.3e1 * t360;
  return t361;
}
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho0__t363(double rho0,
                        double rho1,
                        double sigma0,
                        double sigma1,
                        double sigma2,
                        double tau0,
                        double tau1)
{
  double t90 =
    mgga_c_scan_vrho0__t90(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t2  = M_CBRT3;
  double t3  = 0.1e1 / M_PI;
  double t4  = POW_1_3(t3);
  double t5  = t2 * t4;
  double t6  = M_CBRT4;
  double t7  = t6 * t6;
  double t8  = rho0 + rho1;
  double t9  = POW_1_3(t8);
  double t12 = t5 * t7 / t9;
  double t14 = 0.1e1 + 0.53425e-1 * t12;
  double t15 = sqrt(t12);
  double t18 = POW_3_2(t12);
  double t20 = t2 * t2;
  double t21 = t4 * t4;
  double t22 = t20 * t21;
  double t23 = t9 * t9;
  double t26 = t22 * t6 / t23;
  double t28 =
    0.379785e1 * t15 + 0.8969e0 * t12 + 0.204775e0 * t18 + 0.123235e0 * t26;
  double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
  double t32 = log(t31);
  double t34 = 0.621814e-1 * t14 * t32;
  double t35 = rho0 - rho1;
  double t38 = t8 * t8;
  double t42 = 0.1e1 / t8;
  double t43 = t35 * t42;
  double t44 = 0.1e1 + t43;
  double t45 = t44 <= ZETA_THRESHOLD_C_SCAN;
  double t46 = POW_1_3(ZETA_THRESHOLD_C_SCAN);
  double t47 = t46 * ZETA_THRESHOLD_C_SCAN;
  double t48 = POW_1_3(t44);
  double t49 = t48 * t44;
  double t50 = my_piecewise3(t45, t47, t49);
  double t51 = 0.1e1 - t43;
  double t52 = t51 <= ZETA_THRESHOLD_C_SCAN;
  double t53 = POW_1_3(t51);
  double t54 = t53 * t51;
  double t55 = my_piecewise3(t52, t47, t54);
  double t56 = t50 + t55 - 0.2e1;
  double t57 = M_CBRT2;
  double t58 = t57 - 0.1e1;
  double t60 = 0.1e1 / t58 / 0.2e1;
  double t61 = t56 * t60;
  double t76 = 0.1e1 + 0.278125e-1 * t12;
  double t81 =
    0.51785e1 * t15 + 0.905775e0 * t12 + 0.1100325e0 * t18 + 0.1241775e0 * t26;
  double t84  = 0.1e1 + 0.29608749977793437516e2 / t81;
  double t85  = log(t84);
  double t86  = t76 * t85;
  double t92  = 0.19751673498613801407e-1 * t61 * t86;
  double t93  = log(0.2e1);
  double t94  = 0.1e1 - t93;
  double t95  = M_PI * M_PI;
  double t98  = t46 * t46;
  double t99  = t48 * t48;
  double t100 = my_piecewise3(t45, t98, t99);
  double t101 = t53 * t53;
  double t102 = my_piecewise3(t52, t98, t101);
  double t104 = t100 / 0.2e1 + t102 / 0.2e1;
  double t105 = t104 * t104;
  double t106 = t105 * t104;
  double t108 = 0.1e1 + 0.25e-1 * t12;
  double t110 = 0.1e1 + 0.4445e-1 * t12;
  double t111 = 0.1e1 / t110;
  double t112 = t108 * t111;
  double t113 = 0.1e1 / t94;
  double t115 = (-t34 + t90 + t92) * t113;
  double t116 = 0.1e1 / t106;
  double t117 = t95 * t116;
  double t119 = exp(-t115 * t117);
  double t120 = t119 - 0.1e1;
  double t121 = 0.1e1 / t120;
  double t122 = t113 * t121;
  double t124 = sigma0 + 0.2e1 * sigma1 + sigma2;
  double t125 = t122 * t124;
  double t126 = t112 * t125;
  double t128 = 0.1e1 / t9 / t38;
  double t129 = t128 * t57;
  double t130 = 0.1e1 / t105;
  double t132 = 0.1e1 / t4;
  double t133 = t20 * t132;
  double t134 = t133 * t6;
  double t138 = 0.1e1 + 0.27439371595564631661e-1 * t126 * t129 * t130 * t134;
  double t139 = POW_1_4(t138);
  double t363 = 0.1e1 / t139 / t138;
  return t363;
}
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho0__t381(double rho0,
                        double rho1,
                        double sigma0,
                        double sigma1,
                        double sigma2,
                        double tau0,
                        double tau1)
{
  double t90 =
    mgga_c_scan_vrho0__t90(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t2  = M_CBRT3;
  double t3  = 0.1e1 / M_PI;
  double t4  = POW_1_3(t3);
  double t5  = t2 * t4;
  double t6  = M_CBRT4;
  double t7  = t6 * t6;
  double t8  = rho0 + rho1;
  double t9  = POW_1_3(t8);
  double t12 = t5 * t7 / t9;
  double t14 = 0.1e1 + 0.53425e-1 * t12;
  double t15 = sqrt(t12);
  double t18 = POW_3_2(t12);
  double t20 = t2 * t2;
  double t21 = t4 * t4;
  double t22 = t20 * t21;
  double t23 = t9 * t9;
  double t26 = t22 * t6 / t23;
  double t28 =
    0.379785e1 * t15 + 0.8969e0 * t12 + 0.204775e0 * t18 + 0.123235e0 * t26;
  double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
  double t32 = log(t31);
  double t34 = 0.621814e-1 * t14 * t32;
  double t35 = rho0 - rho1;
  double t38 = t8 * t8;
  double t42 = 0.1e1 / t8;
  double t43 = t35 * t42;
  double t44 = 0.1e1 + t43;
  double t45 = t44 <= ZETA_THRESHOLD_C_SCAN;
  double t46 = POW_1_3(ZETA_THRESHOLD_C_SCAN);
  double t47 = t46 * ZETA_THRESHOLD_C_SCAN;
  double t48 = POW_1_3(t44);
  double t49 = t48 * t44;
  double t50 = my_piecewise3(t45, t47, t49);
  double t51 = 0.1e1 - t43;
  double t52 = t51 <= ZETA_THRESHOLD_C_SCAN;
  double t53 = POW_1_3(t51);
  double t54 = t53 * t51;
  double t55 = my_piecewise3(t52, t47, t54);
  double t56 = t50 + t55 - 0.2e1;
  double t57 = M_CBRT2;
  double t58 = t57 - 0.1e1;
  double t60 = 0.1e1 / t58 / 0.2e1;
  double t61 = t56 * t60;
  double t76 = 0.1e1 + 0.278125e-1 * t12;
  double t81 =
    0.51785e1 * t15 + 0.905775e0 * t12 + 0.1100325e0 * t18 + 0.1241775e0 * t26;
  double t84  = 0.1e1 + 0.29608749977793437516e2 / t81;
  double t85  = log(t84);
  double t86  = t76 * t85;
  double t92  = 0.19751673498613801407e-1 * t61 * t86;
  double t93  = log(0.2e1);
  double t94  = 0.1e1 - t93;
  double t95  = M_PI * M_PI;
  double t98  = t46 * t46;
  double t99  = t48 * t48;
  double t100 = my_piecewise3(t45, t98, t99);
  double t101 = t53 * t53;
  double t102 = my_piecewise3(t52, t98, t101);
  double t104 = t100 / 0.2e1 + t102 / 0.2e1;
  double t105 = t104 * t104;
  double t106 = t105 * t104;
  double t108 = 0.1e1 + 0.25e-1 * t12;
  double t110 = 0.1e1 + 0.4445e-1 * t12;
  double t113 = 0.1e1 / t94;
  double t115 = (-t34 + t90 + t92) * t113;
  double t116 = 0.1e1 / t106;
  double t117 = t95 * t116;
  double t119 = exp(-t115 * t117);
  double t120 = t119 - 0.1e1;
  double t121 = 0.1e1 / t120;
  double t122 = t113 * t121;
  double t124 = sigma0 + 0.2e1 * sigma1 + sigma2;
  double t130 = 0.1e1 / t105;
  double t364 = t38 * t8;
  double t366 = 0.1e1 / t23 / t364;
  double t370 = t57 * t130;
  double t374 = t110 * t110;
  double t375 = 0.1e1 / t374;
  double t376 = t108 * t375;
  double t377 = t376 * t122;
  double t378 = t124 * t366;
  double t381 = 0.48787202696913915093e-2 * t377 * t378 * t370;
  return t381;
}
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho0__t391(double rho0,
                        double rho1,
                        double sigma0,
                        double sigma1,
                        double sigma2,
                        double tau0,
                        double tau1)
{
  double t283 =
    mgga_c_scan_vrho0__t283(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t288 =
    mgga_c_scan_vrho0__t288(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t302 =
    mgga_c_scan_vrho0__t302(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t335 =
    mgga_c_scan_vrho0__t335(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t337 =
    mgga_c_scan_vrho0__t337(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t342 =
    mgga_c_scan_vrho0__t342(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t347 =
    mgga_c_scan_vrho0__t347(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t2  = M_CBRT3;
  double t3  = 0.1e1 / M_PI;
  double t4  = POW_1_3(t3);
  double t5  = t2 * t4;
  double t6  = M_CBRT4;
  double t7  = t6 * t6;
  double t8  = rho0 + rho1;
  double t9  = POW_1_3(t8);
  double t12 = t5 * t7 / t9;
  double t14 = 0.1e1 + 0.53425e-1 * t12;
  double t15 = sqrt(t12);
  double t18 = POW_3_2(t12);
  double t20 = t2 * t2;
  double t21 = t4 * t4;
  double t22 = t20 * t21;
  double t23 = t9 * t9;
  double t26 = t22 * t6 / t23;
  double t28 =
    0.379785e1 * t15 + 0.8969e0 * t12 + 0.204775e0 * t18 + 0.123235e0 * t26;
  double t31  = 0.1e1 + 0.16081979498692535067e2 / t28;
  double t32  = log(t31);
  double t93  = log(0.2e1);
  double t94  = 0.1e1 - t93;
  double t113 = 0.1e1 / t94;
  double t250 = 0.1e1 / t9 / t8;
  double t251 = t7 * t250;
  double t253 = t5 * t251 * t32;
  double t254 = 0.11073470983333333333e-2 * t253;
  double t255 = t28 * t28;
  double t256 = 0.1e1 / t255;
  double t257 = t14 * t256;
  double t259 = 0.1e1 / t15 * t2;
  double t260 = t4 * t7;
  double t261 = t260 * t250;
  double t262 = t259 * t261;
  double t264 = t5 * t251;
  double t266 = sqrt(t12);
  double t267 = t266 * t2;
  double t268 = t267 * t261;
  double t273 = t22 * t6 / t23 / t8;
  double t275 = -0.632975e0 * t262 - 0.29896666666666666667e0 * t264 -
                0.1023875e0 * t268 - 0.82156666666666666667e-1 * t273;
  double t276 = 0.1e1 / t31;
  double t277 = t275 * t276;
  double t278 = t257 * t277;
  double t279 = 0.1e1 * t278;
  double t391 =
    (t254 + t279 + t283 - t288 + t302 + t335 + t337 - t342 - t347) * t113;
  return t391;
}
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho0__t402(double rho0,
                        double rho1,
                        double sigma0,
                        double sigma1,
                        double sigma2,
                        double tau0,
                        double tau1)
{
  double t90 =
    mgga_c_scan_vrho0__t90(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t391 =
    mgga_c_scan_vrho0__t391(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t2  = M_CBRT3;
  double t3  = 0.1e1 / M_PI;
  double t4  = POW_1_3(t3);
  double t5  = t2 * t4;
  double t6  = M_CBRT4;
  double t7  = t6 * t6;
  double t8  = rho0 + rho1;
  double t9  = POW_1_3(t8);
  double t12 = t5 * t7 / t9;
  double t14 = 0.1e1 + 0.53425e-1 * t12;
  double t15 = sqrt(t12);
  double t18 = POW_3_2(t12);
  double t20 = t2 * t2;
  double t21 = t4 * t4;
  double t22 = t20 * t21;
  double t23 = t9 * t9;
  double t26 = t22 * t6 / t23;
  double t28 =
    0.379785e1 * t15 + 0.8969e0 * t12 + 0.204775e0 * t18 + 0.123235e0 * t26;
  double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
  double t32 = log(t31);
  double t34 = 0.621814e-1 * t14 * t32;
  double t35 = rho0 - rho1;
  double t38 = t8 * t8;
  double t42 = 0.1e1 / t8;
  double t43 = t35 * t42;
  double t44 = 0.1e1 + t43;
  double t45 = t44 <= ZETA_THRESHOLD_C_SCAN;
  double t46 = POW_1_3(ZETA_THRESHOLD_C_SCAN);
  double t47 = t46 * ZETA_THRESHOLD_C_SCAN;
  double t48 = POW_1_3(t44);
  double t49 = t48 * t44;
  double t50 = my_piecewise3(t45, t47, t49);
  double t51 = 0.1e1 - t43;
  double t52 = t51 <= ZETA_THRESHOLD_C_SCAN;
  double t53 = POW_1_3(t51);
  double t54 = t53 * t51;
  double t55 = my_piecewise3(t52, t47, t54);
  double t56 = t50 + t55 - 0.2e1;
  double t57 = M_CBRT2;
  double t58 = t57 - 0.1e1;
  double t60 = 0.1e1 / t58 / 0.2e1;
  double t61 = t56 * t60;
  double t76 = 0.1e1 + 0.278125e-1 * t12;
  double t81 =
    0.51785e1 * t15 + 0.905775e0 * t12 + 0.1100325e0 * t18 + 0.1241775e0 * t26;
  double t84  = 0.1e1 + 0.29608749977793437516e2 / t81;
  double t85  = log(t84);
  double t86  = t76 * t85;
  double t92  = 0.19751673498613801407e-1 * t61 * t86;
  double t93  = log(0.2e1);
  double t94  = 0.1e1 - t93;
  double t95  = M_PI * M_PI;
  double t98  = t46 * t46;
  double t99  = t48 * t48;
  double t100 = my_piecewise3(t45, t98, t99);
  double t101 = t53 * t53;
  double t102 = my_piecewise3(t52, t98, t101);
  double t104 = t100 / 0.2e1 + t102 / 0.2e1;
  double t105 = t104 * t104;
  double t106 = t105 * t104;
  double t113 = 0.1e1 / t94;
  double t115 = (-t34 + t90 + t92) * t113;
  double t116 = 0.1e1 / t106;
  double t117 = t95 * t116;
  double t119 = exp(-t115 * t117);
  double t130 = 0.1e1 / t105;
  double t132 = 0.1e1 / t4;
  double t289 = 0.1e1 / t38;
  double t290 = t35 * t289;
  double t291 = t42 - t290;
  double t295 = -t291;
  double t349 = 0.1e1 / t48;
  double t352 = my_piecewise3(t45, 0, 0.2e1 / 0.3e1 * t349 * t291);
  double t353 = 0.1e1 / t53;
  double t356 = my_piecewise3(t52, 0, 0.2e1 / 0.3e1 * t353 * t295);
  double t358 = t352 / 0.2e1 + t356 / 0.2e1;
  double t370 = t57 * t130;
  double t388 = t370 * t20;
  double t389 = t132 * t6;
  double t393 = t105 * t105;
  double t394 = 0.1e1 / t393;
  double t395 = t95 * t394;
  double t396 = t395 * t358;
  double t399 = 0.3e1 * t115 * t396 - t391 * t117;
  double t400 = t399 * t119;
  double t402 = t388 * t389 * t400;
  return t402;
}
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho0__t411(double rho0,
                        double rho1,
                        double sigma0,
                        double sigma1,
                        double sigma2,
                        double tau0,
                        double tau1)
{
  double t90 =
    mgga_c_scan_vrho0__t90(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t2  = M_CBRT3;
  double t3  = 0.1e1 / M_PI;
  double t4  = POW_1_3(t3);
  double t5  = t2 * t4;
  double t6  = M_CBRT4;
  double t7  = t6 * t6;
  double t8  = rho0 + rho1;
  double t9  = POW_1_3(t8);
  double t12 = t5 * t7 / t9;
  double t14 = 0.1e1 + 0.53425e-1 * t12;
  double t15 = sqrt(t12);
  double t18 = POW_3_2(t12);
  double t20 = t2 * t2;
  double t21 = t4 * t4;
  double t22 = t20 * t21;
  double t23 = t9 * t9;
  double t26 = t22 * t6 / t23;
  double t28 =
    0.379785e1 * t15 + 0.8969e0 * t12 + 0.204775e0 * t18 + 0.123235e0 * t26;
  double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
  double t32 = log(t31);
  double t34 = 0.621814e-1 * t14 * t32;
  double t35 = rho0 - rho1;
  double t38 = t8 * t8;
  double t42 = 0.1e1 / t8;
  double t43 = t35 * t42;
  double t44 = 0.1e1 + t43;
  double t45 = t44 <= ZETA_THRESHOLD_C_SCAN;
  double t46 = POW_1_3(ZETA_THRESHOLD_C_SCAN);
  double t47 = t46 * ZETA_THRESHOLD_C_SCAN;
  double t48 = POW_1_3(t44);
  double t49 = t48 * t44;
  double t50 = my_piecewise3(t45, t47, t49);
  double t51 = 0.1e1 - t43;
  double t52 = t51 <= ZETA_THRESHOLD_C_SCAN;
  double t53 = POW_1_3(t51);
  double t54 = t53 * t51;
  double t55 = my_piecewise3(t52, t47, t54);
  double t56 = t50 + t55 - 0.2e1;
  double t57 = M_CBRT2;
  double t58 = t57 - 0.1e1;
  double t60 = 0.1e1 / t58 / 0.2e1;
  double t61 = t56 * t60;
  double t76 = 0.1e1 + 0.278125e-1 * t12;
  double t81 =
    0.51785e1 * t15 + 0.905775e0 * t12 + 0.1100325e0 * t18 + 0.1241775e0 * t26;
  double t84  = 0.1e1 + 0.29608749977793437516e2 / t81;
  double t85  = log(t84);
  double t86  = t76 * t85;
  double t92  = 0.19751673498613801407e-1 * t61 * t86;
  double t93  = log(0.2e1);
  double t94  = 0.1e1 - t93;
  double t95  = M_PI * M_PI;
  double t98  = t46 * t46;
  double t99  = t48 * t48;
  double t100 = my_piecewise3(t45, t98, t99);
  double t101 = t53 * t53;
  double t102 = my_piecewise3(t52, t98, t101);
  double t104 = t100 / 0.2e1 + t102 / 0.2e1;
  double t105 = t104 * t104;
  double t106 = t105 * t104;
  double t108 = 0.1e1 + 0.25e-1 * t12;
  double t110 = 0.1e1 + 0.4445e-1 * t12;
  double t111 = 0.1e1 / t110;
  double t112 = t108 * t111;
  double t113 = 0.1e1 / t94;
  double t115 = (-t34 + t90 + t92) * t113;
  double t116 = 0.1e1 / t106;
  double t117 = t95 * t116;
  double t119 = exp(-t115 * t117);
  double t120 = t119 - 0.1e1;
  double t121 = 0.1e1 / t120;
  double t122 = t113 * t121;
  double t124 = sigma0 + 0.2e1 * sigma1 + sigma2;
  double t125 = t122 * t124;
  double t126 = t112 * t125;
  double t130 = 0.1e1 / t105;
  double t132 = 0.1e1 / t4;
  double t133 = t20 * t132;
  double t134 = t133 * t6;
  double t364 = t38 * t8;
  double t406 = 0.1e1 / t9 / t364;
  double t407 = t406 * t57;
  double t411 = 0.64025200389650807209e-1 * t126 * t407 * t130 * t134;
  return t411;
}
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho0__t427(double rho0,
                        double rho1,
                        double sigma0,
                        double sigma1,
                        double sigma2,
                        double tau0,
                        double tau1)
{
  double t90 =
    mgga_c_scan_vrho0__t90(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t141 =
    mgga_c_scan_vrho0__t141(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t363 =
    mgga_c_scan_vrho0__t363(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t381 =
    mgga_c_scan_vrho0__t381(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t391 =
    mgga_c_scan_vrho0__t391(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t402 =
    mgga_c_scan_vrho0__t402(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t411 =
    mgga_c_scan_vrho0__t411(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t2  = M_CBRT3;
  double t3  = 0.1e1 / M_PI;
  double t4  = POW_1_3(t3);
  double t5  = t2 * t4;
  double t6  = M_CBRT4;
  double t7  = t6 * t6;
  double t8  = rho0 + rho1;
  double t9  = POW_1_3(t8);
  double t12 = t5 * t7 / t9;
  double t14 = 0.1e1 + 0.53425e-1 * t12;
  double t15 = sqrt(t12);
  double t18 = POW_3_2(t12);
  double t20 = t2 * t2;
  double t21 = t4 * t4;
  double t22 = t20 * t21;
  double t23 = t9 * t9;
  double t26 = t22 * t6 / t23;
  double t28 =
    0.379785e1 * t15 + 0.8969e0 * t12 + 0.204775e0 * t18 + 0.123235e0 * t26;
  double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
  double t32 = log(t31);
  double t34 = 0.621814e-1 * t14 * t32;
  double t35 = rho0 - rho1;
  double t38 = t8 * t8;
  double t42 = 0.1e1 / t8;
  double t43 = t35 * t42;
  double t44 = 0.1e1 + t43;
  double t45 = t44 <= ZETA_THRESHOLD_C_SCAN;
  double t46 = POW_1_3(ZETA_THRESHOLD_C_SCAN);
  double t47 = t46 * ZETA_THRESHOLD_C_SCAN;
  double t48 = POW_1_3(t44);
  double t49 = t48 * t44;
  double t50 = my_piecewise3(t45, t47, t49);
  double t51 = 0.1e1 - t43;
  double t52 = t51 <= ZETA_THRESHOLD_C_SCAN;
  double t53 = POW_1_3(t51);
  double t54 = t53 * t51;
  double t55 = my_piecewise3(t52, t47, t54);
  double t56 = t50 + t55 - 0.2e1;
  double t57 = M_CBRT2;
  double t58 = t57 - 0.1e1;
  double t60 = 0.1e1 / t58 / 0.2e1;
  double t61 = t56 * t60;
  double t76 = 0.1e1 + 0.278125e-1 * t12;
  double t81 =
    0.51785e1 * t15 + 0.905775e0 * t12 + 0.1100325e0 * t18 + 0.1241775e0 * t26;
  double t84  = 0.1e1 + 0.29608749977793437516e2 / t81;
  double t85  = log(t84);
  double t86  = t76 * t85;
  double t92  = 0.19751673498613801407e-1 * t61 * t86;
  double t93  = log(0.2e1);
  double t94  = 0.1e1 - t93;
  double t95  = M_PI * M_PI;
  double t98  = t46 * t46;
  double t99  = t48 * t48;
  double t100 = my_piecewise3(t45, t98, t99);
  double t101 = t53 * t53;
  double t102 = my_piecewise3(t52, t98, t101);
  double t104 = t100 / 0.2e1 + t102 / 0.2e1;
  double t105 = t104 * t104;
  double t106 = t105 * t104;
  double t108 = 0.1e1 + 0.25e-1 * t12;
  double t110 = 0.1e1 + 0.4445e-1 * t12;
  double t111 = 0.1e1 / t110;
  double t112 = t108 * t111;
  double t113 = 0.1e1 / t94;
  double t115 = (-t34 + t90 + t92) * t113;
  double t116 = 0.1e1 / t106;
  double t117 = t95 * t116;
  double t119 = exp(-t115 * t117);
  double t120 = t119 - 0.1e1;
  double t121 = 0.1e1 / t120;
  double t124 = sigma0 + 0.2e1 * sigma1 + sigma2;
  double t128 = 0.1e1 / t9 / t38;
  double t130 = 0.1e1 / t105;
  double t132 = 0.1e1 / t4;
  double t289 = 0.1e1 / t38;
  double t290 = t35 * t289;
  double t291 = t42 - t290;
  double t295 = -t291;
  double t349 = 0.1e1 / t48;
  double t352 = my_piecewise3(t45, 0, 0.2e1 / 0.3e1 * t349 * t291);
  double t353 = 0.1e1 / t53;
  double t356 = my_piecewise3(t52, 0, 0.2e1 / 0.3e1 * t353 * t295);
  double t358 = t352 / 0.2e1 + t356 / 0.2e1;
  double t364 = t38 * t8;
  double t366 = 0.1e1 / t23 / t364;
  double t367 = t366 * t111;
  double t369 = t121 * t124;
  double t370 = t57 * t130;
  double t371 = t369 * t370;
  double t373 = 0.27439371595564631661e-2 * t367 * t113 * t371;
  double t382 = t112 * t113;
  double t383 = t120 * t120;
  double t384 = 0.1e1 / t383;
  double t385 = t384 * t124;
  double t387 = t382 * t385 * t128;
  double t389 = t132 * t6;
  double t393 = t105 * t105;
  double t394 = 0.1e1 / t393;
  double t395 = t95 * t394;
  double t396 = t395 * t358;
  double t399 = 0.3e1 * t115 * t396 - t391 * t117;
  double t413 = t382 * t369 * t128;
  double t414 = t57 * t116;
  double t415 = t414 * t20;
  double t417 = t415 * t389 * t358;
  double t420 = -t373 + t381 - 0.27439371595564631661e-1 * t387 * t402 - t411 -
                0.54878743191129263322e-1 * t413 * t417;
  double t421 = t363 * t420;
  double t427 = 0.25e0 * t421 * t120 + 0.1e1 * t141 * t399 * t119;
  return t427;
}
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho0__t429(double rho0,
                        double rho1,
                        double sigma0,
                        double sigma1,
                        double sigma2,
                        double tau0,
                        double tau1)
{
  double t90 =
    mgga_c_scan_vrho0__t90(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t141 =
    mgga_c_scan_vrho0__t141(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t2  = M_CBRT3;
  double t3  = 0.1e1 / M_PI;
  double t4  = POW_1_3(t3);
  double t5  = t2 * t4;
  double t6  = M_CBRT4;
  double t7  = t6 * t6;
  double t8  = rho0 + rho1;
  double t9  = POW_1_3(t8);
  double t12 = t5 * t7 / t9;
  double t14 = 0.1e1 + 0.53425e-1 * t12;
  double t15 = sqrt(t12);
  double t18 = POW_3_2(t12);
  double t20 = t2 * t2;
  double t21 = t4 * t4;
  double t22 = t20 * t21;
  double t23 = t9 * t9;
  double t26 = t22 * t6 / t23;
  double t28 =
    0.379785e1 * t15 + 0.8969e0 * t12 + 0.204775e0 * t18 + 0.123235e0 * t26;
  double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
  double t32 = log(t31);
  double t34 = 0.621814e-1 * t14 * t32;
  double t35 = rho0 - rho1;
  double t42 = 0.1e1 / t8;
  double t43 = t35 * t42;
  double t44 = 0.1e1 + t43;
  double t45 = t44 <= ZETA_THRESHOLD_C_SCAN;
  double t46 = POW_1_3(ZETA_THRESHOLD_C_SCAN);
  double t47 = t46 * ZETA_THRESHOLD_C_SCAN;
  double t48 = POW_1_3(t44);
  double t49 = t48 * t44;
  double t50 = my_piecewise3(t45, t47, t49);
  double t51 = 0.1e1 - t43;
  double t52 = t51 <= ZETA_THRESHOLD_C_SCAN;
  double t53 = POW_1_3(t51);
  double t54 = t53 * t51;
  double t55 = my_piecewise3(t52, t47, t54);
  double t56 = t50 + t55 - 0.2e1;
  double t57 = M_CBRT2;
  double t58 = t57 - 0.1e1;
  double t60 = 0.1e1 / t58 / 0.2e1;
  double t61 = t56 * t60;
  double t76 = 0.1e1 + 0.278125e-1 * t12;
  double t81 =
    0.51785e1 * t15 + 0.905775e0 * t12 + 0.1100325e0 * t18 + 0.1241775e0 * t26;
  double t84  = 0.1e1 + 0.29608749977793437516e2 / t81;
  double t85  = log(t84);
  double t86  = t76 * t85;
  double t92  = 0.19751673498613801407e-1 * t61 * t86;
  double t93  = log(0.2e1);
  double t94  = 0.1e1 - t93;
  double t95  = M_PI * M_PI;
  double t98  = t46 * t46;
  double t99  = t48 * t48;
  double t100 = my_piecewise3(t45, t98, t99);
  double t101 = t53 * t53;
  double t102 = my_piecewise3(t52, t98, t101);
  double t104 = t100 / 0.2e1 + t102 / 0.2e1;
  double t105 = t104 * t104;
  double t106 = t105 * t104;
  double t113 = 0.1e1 / t94;
  double t115 = (-t34 + t90 + t92) * t113;
  double t116 = 0.1e1 / t106;
  double t117 = t95 * t116;
  double t119 = exp(-t115 * t117);
  double t120 = t119 - 0.1e1;
  double t144 = 0.1e1 + 0.1e1 * t141 * t120;
  double t429 = 0.1e1 / t144;
  return t429;
}
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho0__t478(double rho0,
                        double rho1,
                        double sigma0,
                        double sigma1,
                        double sigma2,
                        double tau0,
                        double tau1)
{
  double t247 =
    mgga_c_scan_vrho0__t247(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t8   = rho0 + rho1;
  double t9   = POW_1_3(t8);
  double t23  = t9 * t9;
  double t35  = rho0 - rho1;
  double t38  = t8 * t8;
  double t42  = 0.1e1 / t8;
  double t43  = t35 * t42;
  double t44  = 0.1e1 + t43;
  double t51  = 0.1e1 - t43;
  double t95  = M_PI * M_PI;
  double t124 = sigma0 + 0.2e1 * sigma1 + sigma2;
  double t148 = POW_1_3(rho0);
  double t149 = t148 * t148;
  double t151 = 0.1e1 / t149 / rho0;
  double t152 = tau0 * t151;
  double t153 = t44 / 0.2e1;
  double t154 = POW_1_3(t153);
  double t155 = t154 * t154;
  double t156 = t155 * t153;
  double t158 = POW_1_3(rho1);
  double t159 = t158 * t158;
  double t161 = 0.1e1 / t159 / rho1;
  double t162 = tau1 * t161;
  double t163 = t51 / 0.2e1;
  double t164 = POW_1_3(t163);
  double t165 = t164 * t164;
  double t166 = t165 * t163;
  double t169 = 0.1e1 / t23 / t38;
  double t173 = M_CBRT6;
  double t174 = (t152 * t156 + t162 * t166 - t124 * t169 / 0.8e1) * t173;
  double t175 = POW_1_3(t95);
  double t176 = t175 * t175;
  double t177 = 0.1e1 / t176;
  double t178 = t156 + t166;
  double t179 = 0.1e1 / t178;
  double t180 = t177 * t179;
  double t182 = 0.5e1 / 0.9e1 * t174 * t180;
  double t183 = t182 <= 0.1e1;
  double t184 = log(DBL_EPSILON);
  double t187 = t184 / (-t184 + 0.64e0);
  double t188 = -t187 < t182;
  double t189 = t182 < -t187;
  double t190 = my_piecewise3(t189, t182, -t187);
  double t191 = 0.1e1 - t190;
  double t192 = 0.1e1 / t191;
  double t195 = exp(-0.64e0 * t190 * t192);
  double t198 = log(0.14285714285714285714e1 * DBL_EPSILON);
  double t201 = (-t198 + 0.15e1) / t198;
  double t202 = t182 < -t201;
  double t203 = my_piecewise3(t202, -t201, t182);
  double t204 = 0.1e1 - t203;
  double t207 = exp(0.15e1 / t204);
  double t289 = 0.1e1 / t38;
  double t290 = t35 * t289;
  double t291 = t42 - t290;
  double t364 = t38 * t8;
  double t366 = 0.1e1 / t23 / t364;
  double t378 = t124 * t366;
  double t432 = rho0 * rho0;
  double t434 = 0.1e1 / t149 / t432;
  double t435 = tau0 * t434;
  double t438 = t291 / 0.2e1;
  double t439 = t155 * t438;
  double t442 = -t438;
  double t443 = t165 * t442;
  double t446 = t378 / 0.3e1;
  double t448 = (-0.5e1 / 0.3e1 * t435 * t156 + 0.5e1 / 0.3e1 * t152 * t439 +
                 0.5e1 / 0.3e1 * t162 * t443 + t446) *
                t173;
  double t450 = t178 * t178;
  double t451 = 0.1e1 / t450;
  double t452 = t177 * t451;
  double t454 = 0.5e1 / 0.3e1 * t439 + 0.5e1 / 0.3e1 * t443;
  double t455 = t452 * t454;
  double t458 = -0.5e1 / 0.9e1 * t174 * t455 + 0.5e1 / 0.9e1 * t448 * t180;
  double t459 = my_piecewise3(t189, t458, 0);
  double t462 = t191 * t191;
  double t463 = 0.1e1 / t462;
  double t464 = t190 * t463;
  double t467 = -0.64e0 * t459 * t192 - 0.64e0 * t464 * t459;
  double t468 = t467 * t195;
  double t469 = my_piecewise3(t188, 0, t468);
  double t470 = t204 * t204;
  double t471 = 0.1e1 / t470;
  double t472 = my_piecewise3(t202, 0, t458);
  double t476 = my_piecewise3(t202, 0, -0.105e1 * t471 * t472 * t207);
  double t477 = my_piecewise3(t183, t469, t476);
  double t478 = t477 * t247;
  return t478;
}
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho0__t503(double rho0,
                        double rho1,
                        double sigma0,
                        double sigma1,
                        double sigma2,
                        double tau0,
                        double tau1)
{
  double t2   = M_CBRT3;
  double t3   = 0.1e1 / M_PI;
  double t4   = POW_1_3(t3);
  double t5   = t2 * t4;
  double t6   = M_CBRT4;
  double t7   = t6 * t6;
  double t8   = rho0 + rho1;
  double t9   = POW_1_3(t8);
  double t12  = t5 * t7 / t9;
  double t15  = sqrt(t12);
  double t23  = t9 * t9;
  double t35  = rho0 - rho1;
  double t36  = t35 * t35;
  double t37  = t36 * t36;
  double t38  = t8 * t8;
  double t39  = t38 * t38;
  double t42  = 0.1e1 / t8;
  double t43  = t35 * t42;
  double t44  = 0.1e1 + t43;
  double t45  = t44 <= ZETA_THRESHOLD_C_SCAN;
  double t46  = POW_1_3(ZETA_THRESHOLD_C_SCAN);
  double t47  = t46 * ZETA_THRESHOLD_C_SCAN;
  double t48  = POW_1_3(t44);
  double t49  = t48 * t44;
  double t50  = my_piecewise3(t45, t47, t49);
  double t51  = 0.1e1 - t43;
  double t52  = t51 <= ZETA_THRESHOLD_C_SCAN;
  double t53  = POW_1_3(t51);
  double t54  = t53 * t51;
  double t55  = my_piecewise3(t52, t47, t54);
  double t56  = t50 + t55 - 0.2e1;
  double t57  = M_CBRT2;
  double t58  = t57 - 0.1e1;
  double t60  = 0.1e1 / t58 / 0.2e1;
  double t95  = M_PI * M_PI;
  double t124 = sigma0 + 0.2e1 * sigma1 + sigma2;
  double t169 = 0.1e1 / t23 / t38;
  double t173 = M_CBRT6;
  double t175 = POW_1_3(t95);
  double t176 = t175 * t175;
  double t177 = 0.1e1 / t176;
  double t213 = 0.1e1 + 0.4445e-1 * t15 + 0.3138525e-1 * t12;
  double t214 = 0.1e1 / t213;
  double t217 = exp(0.1e1 * t214);
  double t218 = t217 - 0.1e1;
  double t219 = t173 * t177;
  double t220 = t57 * t57;
  double t221 = t220 * t124;
  double t225 = 0.1e1 + 0.21337642104376358333e-1 * t219 * t221 * t169;
  double t226 = POW_1_4(t225);
  double t228 = 0.1e1 - 0.1e1 / t226;
  double t230 = t218 * t228 + 0.1e1;
  double t237 = 0.1e1 - 0.2363e1 * t58 * t56 * t60;
  double t239 = t37 * t37;
  double t240 = t239 * t37;
  double t241 = t39 * t39;
  double t242 = t241 * t39;
  double t243 = 0.1e1 / t242;
  double t245 = -t240 * t243 + 0.1e1;
  double t250 = 0.1e1 / t9 / t8;
  double t251 = t7 * t250;
  double t259 = 0.1e1 / t15 * t2;
  double t260 = t4 * t7;
  double t261 = t260 * t250;
  double t262 = t259 * t261;
  double t264 = t5 * t251;
  double t364 = t38 * t8;
  double t366 = 0.1e1 / t23 / t364;
  double t378 = t124 * t366;
  double t479 = t213 * t213;
  double t480 = 0.1e1 / t479;
  double t483 = -0.74083333333333333333e-2 * t262 - 0.1046175e-1 * t264;
  double t484 = t480 * t483;
  double t486 = t217 * t228;
  double t490 = 0.1e1 / t226 / t225;
  double t491 = t218 * t490;
  double t492 = t491 * t173;
  double t493 = t177 * t220;
  double t497 =
    -0.1e1 * t484 * t486 - 0.14225094736250905555e-1 * t492 * t493 * t378;
  double t498 = 0.1e1 / t230;
  double t501 = 0.285764e-1 * t484 + 0.285764e-1 * t497 * t498;
  double t502 = t501 * t237;
  double t503 = t502 * t245;
  return t503;
}
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho0__t518(double rho0,
                        double rho1,
                        double sigma0,
                        double sigma1,
                        double sigma2,
                        double tau0,
                        double tau1)
{
  double t210 =
    mgga_c_scan_vrho0__t210(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t238 =
    mgga_c_scan_vrho0__t238(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t283 =
    mgga_c_scan_vrho0__t283(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t288 =
    mgga_c_scan_vrho0__t288(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t302 =
    mgga_c_scan_vrho0__t302(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t335 =
    mgga_c_scan_vrho0__t335(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t337 =
    mgga_c_scan_vrho0__t337(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t342 =
    mgga_c_scan_vrho0__t342(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t347 =
    mgga_c_scan_vrho0__t347(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t361 =
    mgga_c_scan_vrho0__t361(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t427 =
    mgga_c_scan_vrho0__t427(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t429 =
    mgga_c_scan_vrho0__t429(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t478 =
    mgga_c_scan_vrho0__t478(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t503 =
    mgga_c_scan_vrho0__t503(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t2  = M_CBRT3;
  double t3  = 0.1e1 / M_PI;
  double t4  = POW_1_3(t3);
  double t5  = t2 * t4;
  double t6  = M_CBRT4;
  double t7  = t6 * t6;
  double t8  = rho0 + rho1;
  double t9  = POW_1_3(t8);
  double t12 = t5 * t7 / t9;
  double t14 = 0.1e1 + 0.53425e-1 * t12;
  double t15 = sqrt(t12);
  double t18 = POW_3_2(t12);
  double t20 = t2 * t2;
  double t21 = t4 * t4;
  double t22 = t20 * t21;
  double t23 = t9 * t9;
  double t26 = t22 * t6 / t23;
  double t28 =
    0.379785e1 * t15 + 0.8969e0 * t12 + 0.204775e0 * t18 + 0.123235e0 * t26;
  double t31  = 0.1e1 + 0.16081979498692535067e2 / t28;
  double t32  = log(t31);
  double t35  = rho0 - rho1;
  double t36  = t35 * t35;
  double t37  = t36 * t36;
  double t38  = t8 * t8;
  double t39  = t38 * t38;
  double t42  = 0.1e1 / t8;
  double t43  = t35 * t42;
  double t44  = 0.1e1 + t43;
  double t45  = t44 <= ZETA_THRESHOLD_C_SCAN;
  double t46  = POW_1_3(ZETA_THRESHOLD_C_SCAN);
  double t48  = POW_1_3(t44);
  double t51  = 0.1e1 - t43;
  double t52  = t51 <= ZETA_THRESHOLD_C_SCAN;
  double t53  = POW_1_3(t51);
  double t57  = M_CBRT2;
  double t58  = t57 - 0.1e1;
  double t60  = 0.1e1 / t58 / 0.2e1;
  double t93  = log(0.2e1);
  double t94  = 0.1e1 - t93;
  double t95  = M_PI * M_PI;
  double t97  = t94 / t95;
  double t98  = t46 * t46;
  double t99  = t48 * t48;
  double t100 = my_piecewise3(t45, t98, t99);
  double t101 = t53 * t53;
  double t102 = my_piecewise3(t52, t98, t101);
  double t104 = t100 / 0.2e1 + t102 / 0.2e1;
  double t105 = t104 * t104;
  double t106 = t105 * t104;
  double t124 = sigma0 + 0.2e1 * sigma1 + sigma2;
  double t169 = 0.1e1 / t23 / t38;
  double t173 = M_CBRT6;
  double t175 = POW_1_3(t95);
  double t176 = t175 * t175;
  double t177 = 0.1e1 / t176;
  double t213 = 0.1e1 + 0.4445e-1 * t15 + 0.3138525e-1 * t12;
  double t214 = 0.1e1 / t213;
  double t217 = exp(0.1e1 * t214);
  double t218 = t217 - 0.1e1;
  double t219 = t173 * t177;
  double t220 = t57 * t57;
  double t221 = t220 * t124;
  double t225 = 0.1e1 + 0.21337642104376358333e-1 * t219 * t221 * t169;
  double t226 = POW_1_4(t225);
  double t228 = 0.1e1 - 0.1e1 / t226;
  double t230 = t218 * t228 + 0.1e1;
  double t231 = log(t230);
  double t233 = -0.285764e-1 * t214 + 0.285764e-1 * t231;
  double t239 = t37 * t37;
  double t240 = t239 * t37;
  double t241 = t39 * t39;
  double t242 = t241 * t39;
  double t243 = 0.1e1 / t242;
  double t245 = -t240 * t243 + 0.1e1;
  double t250 = 0.1e1 / t9 / t8;
  double t251 = t7 * t250;
  double t253 = t5 * t251 * t32;
  double t254 = 0.11073470983333333333e-2 * t253;
  double t255 = t28 * t28;
  double t256 = 0.1e1 / t255;
  double t257 = t14 * t256;
  double t259 = 0.1e1 / t15 * t2;
  double t260 = t4 * t7;
  double t261 = t260 * t250;
  double t262 = t259 * t261;
  double t264 = t5 * t251;
  double t266 = sqrt(t12);
  double t267 = t266 * t2;
  double t268 = t267 * t261;
  double t273 = t22 * t6 / t23 / t8;
  double t275 = -0.632975e0 * t262 - 0.29896666666666666667e0 * t264 -
                0.1023875e0 * t268 - 0.82156666666666666667e-1 * t273;
  double t276 = 0.1e1 / t31;
  double t277 = t275 * t276;
  double t278 = t257 * t277;
  double t279 = 0.1e1 * t278;
  double t280 = t36 * t35;
  double t284 = t39 * t8;
  double t289 = 0.1e1 / t38;
  double t290 = t35 * t289;
  double t291 = t42 - t290;
  double t294 = my_piecewise3(t45, 0, 0.4e1 / 0.3e1 * t48 * t291);
  double t295 = -t291;
  double t298 = my_piecewise3(t52, 0, 0.4e1 / 0.3e1 * t53 * t295);
  double t299 = t294 + t298;
  double t300 = t299 * t60;
  double t431 = t97 * t106 * t427 * t429;
  double t504 = t233 * t58;
  double t505 = t300 * t245;
  double t508 = t239 * t280;
  double t509 = t508 * t243;
  double t510 = t241 * t284;
  double t511 = 0.1e1 / t510;
  double t512 = t240 * t511;
  double t514 = -0.12e2 * t509 + 0.12e2 * t512;
  double t516 = t503 - 0.2363e1 * t504 * t505 + t238 * t514 - t254 - t279 -
                t283 + t288 - t302 - t335 - t337 + t342 + t347 - t361 - t431;
  double t517 = t210 * t516;
  double t518 = t254 + t279 + t283 - t288 + t302 + t335 + t337 - t342 - t347 +
                t361 + t431 + t478 + t517;
  return t518;
}
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho0(double rho0,
                  double rho1,
                  double sigma0,
                  double sigma1,
                  double sigma2,
                  double tau0,
                  double tau1)
{
  double t90 =
    mgga_c_scan_vrho0__t90(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t147 =
    mgga_c_scan_vrho0__t147(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t248 =
    mgga_c_scan_vrho0__t248(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t518 =
    mgga_c_scan_vrho0__t518(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t2  = M_CBRT3;
  double t3  = 0.1e1 / M_PI;
  double t4  = POW_1_3(t3);
  double t5  = t2 * t4;
  double t6  = M_CBRT4;
  double t7  = t6 * t6;
  double t8  = rho0 + rho1;
  double t9  = POW_1_3(t8);
  double t12 = t5 * t7 / t9;
  double t14 = 0.1e1 + 0.53425e-1 * t12;
  double t15 = sqrt(t12);
  double t18 = POW_3_2(t12);
  double t20 = t2 * t2;
  double t21 = t4 * t4;
  double t22 = t20 * t21;
  double t23 = t9 * t9;
  double t26 = t22 * t6 / t23;
  double t28 =
    0.379785e1 * t15 + 0.8969e0 * t12 + 0.204775e0 * t18 + 0.123235e0 * t26;
  double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
  double t32 = log(t31);
  double t34 = 0.621814e-1 * t14 * t32;
  double t35 = rho0 - rho1;
  double t42 = 0.1e1 / t8;
  double t43 = t35 * t42;
  double t44 = 0.1e1 + t43;
  double t45 = t44 <= ZETA_THRESHOLD_C_SCAN;
  double t46 = POW_1_3(ZETA_THRESHOLD_C_SCAN);
  double t47 = t46 * ZETA_THRESHOLD_C_SCAN;
  double t48 = POW_1_3(t44);
  double t49 = t48 * t44;
  double t50 = my_piecewise3(t45, t47, t49);
  double t51 = 0.1e1 - t43;
  double t52 = t51 <= ZETA_THRESHOLD_C_SCAN;
  double t53 = POW_1_3(t51);
  double t54 = t53 * t51;
  double t55 = my_piecewise3(t52, t47, t54);
  double t56 = t50 + t55 - 0.2e1;
  double t57 = M_CBRT2;
  double t58 = t57 - 0.1e1;
  double t60 = 0.1e1 / t58 / 0.2e1;
  double t61 = t56 * t60;
  double t76 = 0.1e1 + 0.278125e-1 * t12;
  double t81 =
    0.51785e1 * t15 + 0.905775e0 * t12 + 0.1100325e0 * t18 + 0.1241775e0 * t26;
  double t84    = 0.1e1 + 0.29608749977793437516e2 / t81;
  double t85    = log(t84);
  double t86    = t76 * t85;
  double t92    = 0.19751673498613801407e-1 * t61 * t86;
  double tvrho0 = t8 * t518 + t147 + t248 - t34 + t90 + t92;
  return tvrho0;
}
// ---- mgga_c_scan_vrho1: recursively decomposed into bounded-cone
// __noinline__ sub-helpers ----
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho1__t90(double, double, double, double, double, double, double);
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho1__t141(double, double, double, double, double, double, double);
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho1__t147(double, double, double, double, double, double, double);
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho1__t210(double, double, double, double, double, double, double);
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho1__t238(double, double, double, double, double, double, double);
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho1__t247(double, double, double, double, double, double, double);
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho1__t248(double, double, double, double, double, double, double);
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho1__t283(double, double, double, double, double, double, double);
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho1__t288(double, double, double, double, double, double, double);
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho1__t335(double, double, double, double, double, double, double);
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho1__t342(double, double, double, double, double, double, double);
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho1__t347(double, double, double, double, double, double, double);
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho1__t503(double, double, double, double, double, double, double);
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho1__t531(double, double, double, double, double, double, double);
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho1__t544(double, double, double, double, double, double, double);
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho1__t429(double, double, double, double, double, double, double);
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho1__t546(double, double, double, double, double, double, double);
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho1__t363(double, double, double, double, double, double, double);
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho1__t381(double, double, double, double, double, double, double);
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho1__t411(double, double, double, double, double, double, double);
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho1__t554(double, double, double, double, double, double, double);
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho1__t562(double, double, double, double, double, double, double);
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho1__t571(double, double, double, double, double, double, double);
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho1__t609(double, double, double, double, double, double, double);
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho1__t618(double, double, double, double, double, double, double);
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho1(double, double, double, double, double, double, double);

DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho1__t90(double rho0,
                       double rho1,
                       double sigma0,
                       double sigma1,
                       double sigma2,
                       double tau0,
                       double tau1)
{
  double t2  = M_CBRT3;
  double t3  = 0.1e1 / M_PI;
  double t4  = POW_1_3(t3);
  double t5  = t2 * t4;
  double t6  = M_CBRT4;
  double t7  = t6 * t6;
  double t8  = rho0 + rho1;
  double t9  = POW_1_3(t8);
  double t12 = t5 * t7 / t9;
  double t14 = 0.1e1 + 0.53425e-1 * t12;
  double t15 = sqrt(t12);
  double t18 = POW_3_2(t12);
  double t20 = t2 * t2;
  double t21 = t4 * t4;
  double t22 = t20 * t21;
  double t23 = t9 * t9;
  double t26 = t22 * t6 / t23;
  double t28 =
    0.379785e1 * t15 + 0.8969e0 * t12 + 0.204775e0 * t18 + 0.123235e0 * t26;
  double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
  double t32 = log(t31);
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
  double t45 = t44 <= ZETA_THRESHOLD_C_SCAN;
  double t46 = POW_1_3(ZETA_THRESHOLD_C_SCAN);
  double t47 = t46 * ZETA_THRESHOLD_C_SCAN;
  double t48 = POW_1_3(t44);
  double t49 = t48 * t44;
  double t50 = my_piecewise3(t45, t47, t49);
  double t51 = 0.1e1 - t43;
  double t52 = t51 <= ZETA_THRESHOLD_C_SCAN;
  double t53 = POW_1_3(t51);
  double t54 = t53 * t51;
  double t55 = my_piecewise3(t52, t47, t54);
  double t56 = t50 + t55 - 0.2e1;
  double t57 = M_CBRT2;
  double t58 = t57 - 0.1e1;
  double t60 = 0.1e1 / t58 / 0.2e1;
  double t61 = t56 * t60;
  double t63 = 0.1e1 + 0.5137e-1 * t12;
  double t68 =
    0.705945e1 * t15 + 0.1549425e1 * t12 + 0.420775e0 * t18 + 0.1562925e0 * t26;
  double t71 = 0.1e1 + 0.32163958997385070134e2 / t68;
  double t72 = log(t71);
  double t76 = 0.1e1 + 0.278125e-1 * t12;
  double t81 =
    0.51785e1 * t15 + 0.905775e0 * t12 + 0.1100325e0 * t18 + 0.1241775e0 * t26;
  double t84 = 0.1e1 + 0.29608749977793437516e2 / t81;
  double t85 = log(t84);
  double t86 = t76 * t85;
  double t88 = -0.310907e-1 * t63 * t72 + t34 - 0.19751673498613801407e-1 * t86;
  double t89 = t61 * t88;
  double t90 = t41 * t89;
  return t90;
}
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho1__t141(double rho0,
                        double rho1,
                        double sigma0,
                        double sigma1,
                        double sigma2,
                        double tau0,
                        double tau1)
{
  double t90 =
    mgga_c_scan_vrho1__t90(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t2  = M_CBRT3;
  double t3  = 0.1e1 / M_PI;
  double t4  = POW_1_3(t3);
  double t5  = t2 * t4;
  double t6  = M_CBRT4;
  double t7  = t6 * t6;
  double t8  = rho0 + rho1;
  double t9  = POW_1_3(t8);
  double t12 = t5 * t7 / t9;
  double t14 = 0.1e1 + 0.53425e-1 * t12;
  double t15 = sqrt(t12);
  double t18 = POW_3_2(t12);
  double t20 = t2 * t2;
  double t21 = t4 * t4;
  double t22 = t20 * t21;
  double t23 = t9 * t9;
  double t26 = t22 * t6 / t23;
  double t28 =
    0.379785e1 * t15 + 0.8969e0 * t12 + 0.204775e0 * t18 + 0.123235e0 * t26;
  double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
  double t32 = log(t31);
  double t34 = 0.621814e-1 * t14 * t32;
  double t35 = rho0 - rho1;
  double t38 = t8 * t8;
  double t42 = 0.1e1 / t8;
  double t43 = t35 * t42;
  double t44 = 0.1e1 + t43;
  double t45 = t44 <= ZETA_THRESHOLD_C_SCAN;
  double t46 = POW_1_3(ZETA_THRESHOLD_C_SCAN);
  double t47 = t46 * ZETA_THRESHOLD_C_SCAN;
  double t48 = POW_1_3(t44);
  double t49 = t48 * t44;
  double t50 = my_piecewise3(t45, t47, t49);
  double t51 = 0.1e1 - t43;
  double t52 = t51 <= ZETA_THRESHOLD_C_SCAN;
  double t53 = POW_1_3(t51);
  double t54 = t53 * t51;
  double t55 = my_piecewise3(t52, t47, t54);
  double t56 = t50 + t55 - 0.2e1;
  double t57 = M_CBRT2;
  double t58 = t57 - 0.1e1;
  double t60 = 0.1e1 / t58 / 0.2e1;
  double t61 = t56 * t60;
  double t76 = 0.1e1 + 0.278125e-1 * t12;
  double t81 =
    0.51785e1 * t15 + 0.905775e0 * t12 + 0.1100325e0 * t18 + 0.1241775e0 * t26;
  double t84  = 0.1e1 + 0.29608749977793437516e2 / t81;
  double t85  = log(t84);
  double t86  = t76 * t85;
  double t92  = 0.19751673498613801407e-1 * t61 * t86;
  double t93  = log(0.2e1);
  double t94  = 0.1e1 - t93;
  double t95  = M_PI * M_PI;
  double t98  = t46 * t46;
  double t99  = t48 * t48;
  double t100 = my_piecewise3(t45, t98, t99);
  double t101 = t53 * t53;
  double t102 = my_piecewise3(t52, t98, t101);
  double t104 = t100 / 0.2e1 + t102 / 0.2e1;
  double t105 = t104 * t104;
  double t106 = t105 * t104;
  double t108 = 0.1e1 + 0.25e-1 * t12;
  double t110 = 0.1e1 + 0.4445e-1 * t12;
  double t111 = 0.1e1 / t110;
  double t112 = t108 * t111;
  double t113 = 0.1e1 / t94;
  double t115 = (-t34 + t90 + t92) * t113;
  double t116 = 0.1e1 / t106;
  double t117 = t95 * t116;
  double t119 = exp(-t115 * t117);
  double t120 = t119 - 0.1e1;
  double t121 = 0.1e1 / t120;
  double t122 = t113 * t121;
  double t124 = sigma0 + 0.2e1 * sigma1 + sigma2;
  double t125 = t122 * t124;
  double t126 = t112 * t125;
  double t128 = 0.1e1 / t9 / t38;
  double t129 = t128 * t57;
  double t130 = 0.1e1 / t105;
  double t132 = 0.1e1 / t4;
  double t133 = t20 * t132;
  double t134 = t133 * t6;
  double t138 = 0.1e1 + 0.27439371595564631661e-1 * t126 * t129 * t130 * t134;
  double t139 = POW_1_4(t138);
  double t141 = 0.1e1 - 0.1e1 / t139;
  return t141;
}
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho1__t147(double rho0,
                        double rho1,
                        double sigma0,
                        double sigma1,
                        double sigma2,
                        double tau0,
                        double tau1)
{
  double t90 =
    mgga_c_scan_vrho1__t90(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t141 =
    mgga_c_scan_vrho1__t141(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t2  = M_CBRT3;
  double t3  = 0.1e1 / M_PI;
  double t4  = POW_1_3(t3);
  double t5  = t2 * t4;
  double t6  = M_CBRT4;
  double t7  = t6 * t6;
  double t8  = rho0 + rho1;
  double t9  = POW_1_3(t8);
  double t12 = t5 * t7 / t9;
  double t14 = 0.1e1 + 0.53425e-1 * t12;
  double t15 = sqrt(t12);
  double t18 = POW_3_2(t12);
  double t20 = t2 * t2;
  double t21 = t4 * t4;
  double t22 = t20 * t21;
  double t23 = t9 * t9;
  double t26 = t22 * t6 / t23;
  double t28 =
    0.379785e1 * t15 + 0.8969e0 * t12 + 0.204775e0 * t18 + 0.123235e0 * t26;
  double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
  double t32 = log(t31);
  double t34 = 0.621814e-1 * t14 * t32;
  double t35 = rho0 - rho1;
  double t42 = 0.1e1 / t8;
  double t43 = t35 * t42;
  double t44 = 0.1e1 + t43;
  double t45 = t44 <= ZETA_THRESHOLD_C_SCAN;
  double t46 = POW_1_3(ZETA_THRESHOLD_C_SCAN);
  double t47 = t46 * ZETA_THRESHOLD_C_SCAN;
  double t48 = POW_1_3(t44);
  double t49 = t48 * t44;
  double t50 = my_piecewise3(t45, t47, t49);
  double t51 = 0.1e1 - t43;
  double t52 = t51 <= ZETA_THRESHOLD_C_SCAN;
  double t53 = POW_1_3(t51);
  double t54 = t53 * t51;
  double t55 = my_piecewise3(t52, t47, t54);
  double t56 = t50 + t55 - 0.2e1;
  double t57 = M_CBRT2;
  double t58 = t57 - 0.1e1;
  double t60 = 0.1e1 / t58 / 0.2e1;
  double t61 = t56 * t60;
  double t76 = 0.1e1 + 0.278125e-1 * t12;
  double t81 =
    0.51785e1 * t15 + 0.905775e0 * t12 + 0.1100325e0 * t18 + 0.1241775e0 * t26;
  double t84  = 0.1e1 + 0.29608749977793437516e2 / t81;
  double t85  = log(t84);
  double t86  = t76 * t85;
  double t92  = 0.19751673498613801407e-1 * t61 * t86;
  double t93  = log(0.2e1);
  double t94  = 0.1e1 - t93;
  double t95  = M_PI * M_PI;
  double t97  = t94 / t95;
  double t98  = t46 * t46;
  double t99  = t48 * t48;
  double t100 = my_piecewise3(t45, t98, t99);
  double t101 = t53 * t53;
  double t102 = my_piecewise3(t52, t98, t101);
  double t104 = t100 / 0.2e1 + t102 / 0.2e1;
  double t105 = t104 * t104;
  double t106 = t105 * t104;
  double t113 = 0.1e1 / t94;
  double t115 = (-t34 + t90 + t92) * t113;
  double t116 = 0.1e1 / t106;
  double t117 = t95 * t116;
  double t119 = exp(-t115 * t117);
  double t120 = t119 - 0.1e1;
  double t144 = 0.1e1 + 0.1e1 * t141 * t120;
  double t145 = log(t144);
  double t147 = t97 * t106 * t145;
  return t147;
}
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho1__t210(double rho0,
                        double rho1,
                        double sigma0,
                        double sigma1,
                        double sigma2,
                        double tau0,
                        double tau1)
{
  double t8   = rho0 + rho1;
  double t9   = POW_1_3(t8);
  double t23  = t9 * t9;
  double t35  = rho0 - rho1;
  double t38  = t8 * t8;
  double t42  = 0.1e1 / t8;
  double t43  = t35 * t42;
  double t44  = 0.1e1 + t43;
  double t51  = 0.1e1 - t43;
  double t95  = M_PI * M_PI;
  double t124 = sigma0 + 0.2e1 * sigma1 + sigma2;
  double t148 = POW_1_3(rho0);
  double t149 = t148 * t148;
  double t151 = 0.1e1 / t149 / rho0;
  double t152 = tau0 * t151;
  double t153 = t44 / 0.2e1;
  double t154 = POW_1_3(t153);
  double t155 = t154 * t154;
  double t156 = t155 * t153;
  double t158 = POW_1_3(rho1);
  double t159 = t158 * t158;
  double t161 = 0.1e1 / t159 / rho1;
  double t162 = tau1 * t161;
  double t163 = t51 / 0.2e1;
  double t164 = POW_1_3(t163);
  double t165 = t164 * t164;
  double t166 = t165 * t163;
  double t169 = 0.1e1 / t23 / t38;
  double t173 = M_CBRT6;
  double t174 = (t152 * t156 + t162 * t166 - t124 * t169 / 0.8e1) * t173;
  double t175 = POW_1_3(t95);
  double t176 = t175 * t175;
  double t177 = 0.1e1 / t176;
  double t178 = t156 + t166;
  double t179 = 0.1e1 / t178;
  double t180 = t177 * t179;
  double t182 = 0.5e1 / 0.9e1 * t174 * t180;
  double t183 = t182 <= 0.1e1;
  double t184 = log(DBL_EPSILON);
  double t187 = t184 / (-t184 + 0.64e0);
  double t188 = -t187 < t182;
  double t189 = t182 < -t187;
  double t190 = my_piecewise3(t189, t182, -t187);
  double t191 = 0.1e1 - t190;
  double t192 = 0.1e1 / t191;
  double t195 = exp(-0.64e0 * t190 * t192);
  double t196 = my_piecewise3(t188, 0, t195);
  double t198 = log(0.14285714285714285714e1 * DBL_EPSILON);
  double t201 = (-t198 + 0.15e1) / t198;
  double t202 = t182 < -t201;
  double t203 = my_piecewise3(t202, -t201, t182);
  double t204 = 0.1e1 - t203;
  double t207 = exp(0.15e1 / t204);
  double t209 = my_piecewise3(t202, 0, -0.7e0 * t207);
  double t210 = my_piecewise3(t183, t196, t209);
  return t210;
}
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho1__t238(double rho0,
                        double rho1,
                        double sigma0,
                        double sigma1,
                        double sigma2,
                        double tau0,
                        double tau1)
{
  double t2   = M_CBRT3;
  double t3   = 0.1e1 / M_PI;
  double t4   = POW_1_3(t3);
  double t5   = t2 * t4;
  double t6   = M_CBRT4;
  double t7   = t6 * t6;
  double t8   = rho0 + rho1;
  double t9   = POW_1_3(t8);
  double t12  = t5 * t7 / t9;
  double t15  = sqrt(t12);
  double t23  = t9 * t9;
  double t35  = rho0 - rho1;
  double t38  = t8 * t8;
  double t42  = 0.1e1 / t8;
  double t43  = t35 * t42;
  double t44  = 0.1e1 + t43;
  double t45  = t44 <= ZETA_THRESHOLD_C_SCAN;
  double t46  = POW_1_3(ZETA_THRESHOLD_C_SCAN);
  double t47  = t46 * ZETA_THRESHOLD_C_SCAN;
  double t48  = POW_1_3(t44);
  double t49  = t48 * t44;
  double t50  = my_piecewise3(t45, t47, t49);
  double t51  = 0.1e1 - t43;
  double t52  = t51 <= ZETA_THRESHOLD_C_SCAN;
  double t53  = POW_1_3(t51);
  double t54  = t53 * t51;
  double t55  = my_piecewise3(t52, t47, t54);
  double t56  = t50 + t55 - 0.2e1;
  double t57  = M_CBRT2;
  double t58  = t57 - 0.1e1;
  double t60  = 0.1e1 / t58 / 0.2e1;
  double t95  = M_PI * M_PI;
  double t124 = sigma0 + 0.2e1 * sigma1 + sigma2;
  double t169 = 0.1e1 / t23 / t38;
  double t173 = M_CBRT6;
  double t175 = POW_1_3(t95);
  double t176 = t175 * t175;
  double t177 = 0.1e1 / t176;
  double t213 = 0.1e1 + 0.4445e-1 * t15 + 0.3138525e-1 * t12;
  double t214 = 0.1e1 / t213;
  double t217 = exp(0.1e1 * t214);
  double t218 = t217 - 0.1e1;
  double t219 = t173 * t177;
  double t220 = t57 * t57;
  double t221 = t220 * t124;
  double t225 = 0.1e1 + 0.21337642104376358333e-1 * t219 * t221 * t169;
  double t226 = POW_1_4(t225);
  double t228 = 0.1e1 - 0.1e1 / t226;
  double t230 = t218 * t228 + 0.1e1;
  double t231 = log(t230);
  double t233 = -0.285764e-1 * t214 + 0.285764e-1 * t231;
  double t237 = 0.1e1 - 0.2363e1 * t58 * t56 * t60;
  double t238 = t233 * t237;
  return t238;
}
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho1__t247(double rho0,
                        double rho1,
                        double sigma0,
                        double sigma1,
                        double sigma2,
                        double tau0,
                        double tau1)
{
  double t90 =
    mgga_c_scan_vrho1__t90(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t147 =
    mgga_c_scan_vrho1__t147(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t238 =
    mgga_c_scan_vrho1__t238(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t2  = M_CBRT3;
  double t3  = 0.1e1 / M_PI;
  double t4  = POW_1_3(t3);
  double t5  = t2 * t4;
  double t6  = M_CBRT4;
  double t7  = t6 * t6;
  double t8  = rho0 + rho1;
  double t9  = POW_1_3(t8);
  double t12 = t5 * t7 / t9;
  double t14 = 0.1e1 + 0.53425e-1 * t12;
  double t15 = sqrt(t12);
  double t18 = POW_3_2(t12);
  double t20 = t2 * t2;
  double t21 = t4 * t4;
  double t22 = t20 * t21;
  double t23 = t9 * t9;
  double t26 = t22 * t6 / t23;
  double t28 =
    0.379785e1 * t15 + 0.8969e0 * t12 + 0.204775e0 * t18 + 0.123235e0 * t26;
  double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
  double t32 = log(t31);
  double t34 = 0.621814e-1 * t14 * t32;
  double t35 = rho0 - rho1;
  double t36 = t35 * t35;
  double t37 = t36 * t36;
  double t38 = t8 * t8;
  double t39 = t38 * t38;
  double t42 = 0.1e1 / t8;
  double t43 = t35 * t42;
  double t44 = 0.1e1 + t43;
  double t45 = t44 <= ZETA_THRESHOLD_C_SCAN;
  double t46 = POW_1_3(ZETA_THRESHOLD_C_SCAN);
  double t47 = t46 * ZETA_THRESHOLD_C_SCAN;
  double t48 = POW_1_3(t44);
  double t49 = t48 * t44;
  double t50 = my_piecewise3(t45, t47, t49);
  double t51 = 0.1e1 - t43;
  double t52 = t51 <= ZETA_THRESHOLD_C_SCAN;
  double t53 = POW_1_3(t51);
  double t54 = t53 * t51;
  double t55 = my_piecewise3(t52, t47, t54);
  double t56 = t50 + t55 - 0.2e1;
  double t57 = M_CBRT2;
  double t58 = t57 - 0.1e1;
  double t60 = 0.1e1 / t58 / 0.2e1;
  double t61 = t56 * t60;
  double t76 = 0.1e1 + 0.278125e-1 * t12;
  double t81 =
    0.51785e1 * t15 + 0.905775e0 * t12 + 0.1100325e0 * t18 + 0.1241775e0 * t26;
  double t84  = 0.1e1 + 0.29608749977793437516e2 / t81;
  double t85  = log(t84);
  double t86  = t76 * t85;
  double t92  = 0.19751673498613801407e-1 * t61 * t86;
  double t239 = t37 * t37;
  double t240 = t239 * t37;
  double t241 = t39 * t39;
  double t242 = t241 * t39;
  double t243 = 0.1e1 / t242;
  double t245 = -t240 * t243 + 0.1e1;
  double t247 = t238 * t245 - t147 + t34 - t90 - t92;
  return t247;
}
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho1__t248(double rho0,
                        double rho1,
                        double sigma0,
                        double sigma1,
                        double sigma2,
                        double tau0,
                        double tau1)
{
  double t210 =
    mgga_c_scan_vrho1__t210(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t247 =
    mgga_c_scan_vrho1__t247(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t248 = t210 * t247;
  return t248;
}
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho1__t283(double rho0,
                        double rho1,
                        double sigma0,
                        double sigma1,
                        double sigma2,
                        double tau0,
                        double tau1)
{
  double t2  = M_CBRT3;
  double t3  = 0.1e1 / M_PI;
  double t4  = POW_1_3(t3);
  double t5  = t2 * t4;
  double t6  = M_CBRT4;
  double t7  = t6 * t6;
  double t8  = rho0 + rho1;
  double t9  = POW_1_3(t8);
  double t12 = t5 * t7 / t9;
  double t14 = 0.1e1 + 0.53425e-1 * t12;
  double t15 = sqrt(t12);
  double t18 = POW_3_2(t12);
  double t20 = t2 * t2;
  double t21 = t4 * t4;
  double t22 = t20 * t21;
  double t23 = t9 * t9;
  double t26 = t22 * t6 / t23;
  double t28 =
    0.379785e1 * t15 + 0.8969e0 * t12 + 0.204775e0 * t18 + 0.123235e0 * t26;
  double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
  double t32 = log(t31);
  double t34 = 0.621814e-1 * t14 * t32;
  double t35 = rho0 - rho1;
  double t36 = t35 * t35;
  double t38 = t8 * t8;
  double t39 = t38 * t38;
  double t40 = 0.1e1 / t39;
  double t42 = 0.1e1 / t8;
  double t43 = t35 * t42;
  double t44 = 0.1e1 + t43;
  double t45 = t44 <= ZETA_THRESHOLD_C_SCAN;
  double t46 = POW_1_3(ZETA_THRESHOLD_C_SCAN);
  double t47 = t46 * ZETA_THRESHOLD_C_SCAN;
  double t48 = POW_1_3(t44);
  double t49 = t48 * t44;
  double t50 = my_piecewise3(t45, t47, t49);
  double t51 = 0.1e1 - t43;
  double t52 = t51 <= ZETA_THRESHOLD_C_SCAN;
  double t53 = POW_1_3(t51);
  double t54 = t53 * t51;
  double t55 = my_piecewise3(t52, t47, t54);
  double t56 = t50 + t55 - 0.2e1;
  double t57 = M_CBRT2;
  double t58 = t57 - 0.1e1;
  double t60 = 0.1e1 / t58 / 0.2e1;
  double t61 = t56 * t60;
  double t63 = 0.1e1 + 0.5137e-1 * t12;
  double t68 =
    0.705945e1 * t15 + 0.1549425e1 * t12 + 0.420775e0 * t18 + 0.1562925e0 * t26;
  double t71 = 0.1e1 + 0.32163958997385070134e2 / t68;
  double t72 = log(t71);
  double t76 = 0.1e1 + 0.278125e-1 * t12;
  double t81 =
    0.51785e1 * t15 + 0.905775e0 * t12 + 0.1100325e0 * t18 + 0.1241775e0 * t26;
  double t84 = 0.1e1 + 0.29608749977793437516e2 / t81;
  double t85 = log(t84);
  double t86 = t76 * t85;
  double t88 = -0.310907e-1 * t63 * t72 + t34 - 0.19751673498613801407e-1 * t86;
  double t89 = t61 * t88;
  double t280 = t36 * t35;
  double t281 = t280 * t40;
  double t282 = t281 * t89;
  double t283 = 0.4e1 * t282;
  return t283;
}
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho1__t288(double rho0,
                        double rho1,
                        double sigma0,
                        double sigma1,
                        double sigma2,
                        double tau0,
                        double tau1)
{
  double t2  = M_CBRT3;
  double t3  = 0.1e1 / M_PI;
  double t4  = POW_1_3(t3);
  double t5  = t2 * t4;
  double t6  = M_CBRT4;
  double t7  = t6 * t6;
  double t8  = rho0 + rho1;
  double t9  = POW_1_3(t8);
  double t12 = t5 * t7 / t9;
  double t14 = 0.1e1 + 0.53425e-1 * t12;
  double t15 = sqrt(t12);
  double t18 = POW_3_2(t12);
  double t20 = t2 * t2;
  double t21 = t4 * t4;
  double t22 = t20 * t21;
  double t23 = t9 * t9;
  double t26 = t22 * t6 / t23;
  double t28 =
    0.379785e1 * t15 + 0.8969e0 * t12 + 0.204775e0 * t18 + 0.123235e0 * t26;
  double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
  double t32 = log(t31);
  double t34 = 0.621814e-1 * t14 * t32;
  double t35 = rho0 - rho1;
  double t36 = t35 * t35;
  double t37 = t36 * t36;
  double t38 = t8 * t8;
  double t39 = t38 * t38;
  double t42 = 0.1e1 / t8;
  double t43 = t35 * t42;
  double t44 = 0.1e1 + t43;
  double t45 = t44 <= ZETA_THRESHOLD_C_SCAN;
  double t46 = POW_1_3(ZETA_THRESHOLD_C_SCAN);
  double t47 = t46 * ZETA_THRESHOLD_C_SCAN;
  double t48 = POW_1_3(t44);
  double t49 = t48 * t44;
  double t50 = my_piecewise3(t45, t47, t49);
  double t51 = 0.1e1 - t43;
  double t52 = t51 <= ZETA_THRESHOLD_C_SCAN;
  double t53 = POW_1_3(t51);
  double t54 = t53 * t51;
  double t55 = my_piecewise3(t52, t47, t54);
  double t56 = t50 + t55 - 0.2e1;
  double t57 = M_CBRT2;
  double t58 = t57 - 0.1e1;
  double t60 = 0.1e1 / t58 / 0.2e1;
  double t61 = t56 * t60;
  double t63 = 0.1e1 + 0.5137e-1 * t12;
  double t68 =
    0.705945e1 * t15 + 0.1549425e1 * t12 + 0.420775e0 * t18 + 0.1562925e0 * t26;
  double t71 = 0.1e1 + 0.32163958997385070134e2 / t68;
  double t72 = log(t71);
  double t76 = 0.1e1 + 0.278125e-1 * t12;
  double t81 =
    0.51785e1 * t15 + 0.905775e0 * t12 + 0.1100325e0 * t18 + 0.1241775e0 * t26;
  double t84 = 0.1e1 + 0.29608749977793437516e2 / t81;
  double t85 = log(t84);
  double t86 = t76 * t85;
  double t88 = -0.310907e-1 * t63 * t72 + t34 - 0.19751673498613801407e-1 * t86;
  double t89 = t61 * t88;
  double t284 = t39 * t8;
  double t285 = 0.1e1 / t284;
  double t286 = t37 * t285;
  double t287 = t286 * t89;
  double t288 = 0.4e1 * t287;
  return t288;
}
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho1__t335(double rho0,
                        double rho1,
                        double sigma0,
                        double sigma1,
                        double sigma2,
                        double tau0,
                        double tau1)
{
  double t2  = M_CBRT3;
  double t3  = 0.1e1 / M_PI;
  double t4  = POW_1_3(t3);
  double t5  = t2 * t4;
  double t6  = M_CBRT4;
  double t7  = t6 * t6;
  double t8  = rho0 + rho1;
  double t9  = POW_1_3(t8);
  double t12 = t5 * t7 / t9;
  double t14 = 0.1e1 + 0.53425e-1 * t12;
  double t15 = sqrt(t12);
  double t18 = POW_3_2(t12);
  double t20 = t2 * t2;
  double t21 = t4 * t4;
  double t22 = t20 * t21;
  double t23 = t9 * t9;
  double t26 = t22 * t6 / t23;
  double t28 =
    0.379785e1 * t15 + 0.8969e0 * t12 + 0.204775e0 * t18 + 0.123235e0 * t26;
  double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
  double t32 = log(t31);
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
  double t45 = t44 <= ZETA_THRESHOLD_C_SCAN;
  double t46 = POW_1_3(ZETA_THRESHOLD_C_SCAN);
  double t47 = t46 * ZETA_THRESHOLD_C_SCAN;
  double t48 = POW_1_3(t44);
  double t49 = t48 * t44;
  double t50 = my_piecewise3(t45, t47, t49);
  double t51 = 0.1e1 - t43;
  double t52 = t51 <= ZETA_THRESHOLD_C_SCAN;
  double t53 = POW_1_3(t51);
  double t54 = t53 * t51;
  double t55 = my_piecewise3(t52, t47, t54);
  double t56 = t50 + t55 - 0.2e1;
  double t57 = M_CBRT2;
  double t58 = t57 - 0.1e1;
  double t60 = 0.1e1 / t58 / 0.2e1;
  double t61 = t56 * t60;
  double t63 = 0.1e1 + 0.5137e-1 * t12;
  double t68 =
    0.705945e1 * t15 + 0.1549425e1 * t12 + 0.420775e0 * t18 + 0.1562925e0 * t26;
  double t71 = 0.1e1 + 0.32163958997385070134e2 / t68;
  double t72 = log(t71);
  double t76 = 0.1e1 + 0.278125e-1 * t12;
  double t81 =
    0.51785e1 * t15 + 0.905775e0 * t12 + 0.1100325e0 * t18 + 0.1241775e0 * t26;
  double t84  = 0.1e1 + 0.29608749977793437516e2 / t81;
  double t85  = log(t84);
  double t250 = 0.1e1 / t9 / t8;
  double t251 = t7 * t250;
  double t253 = t5 * t251 * t32;
  double t254 = 0.11073470983333333333e-2 * t253;
  double t255 = t28 * t28;
  double t256 = 0.1e1 / t255;
  double t257 = t14 * t256;
  double t259 = 0.1e1 / t15 * t2;
  double t260 = t4 * t7;
  double t261 = t260 * t250;
  double t262 = t259 * t261;
  double t264 = t5 * t251;
  double t266 = sqrt(t12);
  double t267 = t266 * t2;
  double t268 = t267 * t261;
  double t273 = t22 * t6 / t23 / t8;
  double t275 = -0.632975e0 * t262 - 0.29896666666666666667e0 * t264 -
                0.1023875e0 * t268 - 0.82156666666666666667e-1 * t273;
  double t276 = 0.1e1 / t31;
  double t277 = t275 * t276;
  double t278 = t257 * t277;
  double t279 = 0.1e1 * t278;
  double t306 = t68 * t68;
  double t307 = 0.1e1 / t306;
  double t308 = t63 * t307;
  double t313 = -0.1176575e1 * t262 - 0.516475e0 * t264 - 0.2103875e0 * t268 -
                0.104195e0 * t273;
  double t314 = 0.1e1 / t71;
  double t315 = t313 * t314;
  double t321 = t81 * t81;
  double t322 = 0.1e1 / t321;
  double t323 = t76 * t322;
  double t328 = -0.86308333333333333334e0 * t262 - 0.301925e0 * t264 -
                0.5501625e-1 * t268 - 0.82785e-1 * t273;
  double t329 = 0.1e1 / t84;
  double t330 = t328 * t329;
  double t333 = 0.53237641966666666666e-3 * t5 * t251 * t72 +
                0.1e1 * t308 * t315 - t254 - t279 +
                0.18311447306006545054e-3 * t5 * t251 * t85 +
                0.5848223622634646207e0 * t323 * t330;
  double t334 = t61 * t333;
  double t335 = t41 * t334;
  return t335;
}
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho1__t342(double rho0,
                        double rho1,
                        double sigma0,
                        double sigma1,
                        double sigma2,
                        double tau0,
                        double tau1)
{
  double t2  = M_CBRT3;
  double t3  = 0.1e1 / M_PI;
  double t4  = POW_1_3(t3);
  double t5  = t2 * t4;
  double t6  = M_CBRT4;
  double t7  = t6 * t6;
  double t8  = rho0 + rho1;
  double t9  = POW_1_3(t8);
  double t12 = t5 * t7 / t9;
  double t15 = sqrt(t12);
  double t18 = POW_3_2(t12);
  double t20 = t2 * t2;
  double t21 = t4 * t4;
  double t22 = t20 * t21;
  double t23 = t9 * t9;
  double t26 = t22 * t6 / t23;
  double t35 = rho0 - rho1;
  double t42 = 0.1e1 / t8;
  double t43 = t35 * t42;
  double t44 = 0.1e1 + t43;
  double t45 = t44 <= ZETA_THRESHOLD_C_SCAN;
  double t46 = POW_1_3(ZETA_THRESHOLD_C_SCAN);
  double t47 = t46 * ZETA_THRESHOLD_C_SCAN;
  double t48 = POW_1_3(t44);
  double t49 = t48 * t44;
  double t50 = my_piecewise3(t45, t47, t49);
  double t51 = 0.1e1 - t43;
  double t52 = t51 <= ZETA_THRESHOLD_C_SCAN;
  double t53 = POW_1_3(t51);
  double t54 = t53 * t51;
  double t55 = my_piecewise3(t52, t47, t54);
  double t56 = t50 + t55 - 0.2e1;
  double t57 = M_CBRT2;
  double t58 = t57 - 0.1e1;
  double t60 = 0.1e1 / t58 / 0.2e1;
  double t61 = t56 * t60;
  double t81 =
    0.51785e1 * t15 + 0.905775e0 * t12 + 0.1100325e0 * t18 + 0.1241775e0 * t26;
  double t84  = 0.1e1 + 0.29608749977793437516e2 / t81;
  double t85  = log(t84);
  double t250 = 0.1e1 / t9 / t8;
  double t260 = t4 * t7;
  double t338 = t61 * t2;
  double t340 = t260 * t250 * t85;
  double t341 = t338 * t340;
  double t342 = 0.18311447306006545054e-3 * t341;
  return t342;
}
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho1__t347(double rho0,
                        double rho1,
                        double sigma0,
                        double sigma1,
                        double sigma2,
                        double tau0,
                        double tau1)
{
  double t2  = M_CBRT3;
  double t3  = 0.1e1 / M_PI;
  double t4  = POW_1_3(t3);
  double t5  = t2 * t4;
  double t6  = M_CBRT4;
  double t7  = t6 * t6;
  double t8  = rho0 + rho1;
  double t9  = POW_1_3(t8);
  double t12 = t5 * t7 / t9;
  double t15 = sqrt(t12);
  double t18 = POW_3_2(t12);
  double t20 = t2 * t2;
  double t21 = t4 * t4;
  double t22 = t20 * t21;
  double t23 = t9 * t9;
  double t26 = t22 * t6 / t23;
  double t35 = rho0 - rho1;
  double t42 = 0.1e1 / t8;
  double t43 = t35 * t42;
  double t44 = 0.1e1 + t43;
  double t45 = t44 <= ZETA_THRESHOLD_C_SCAN;
  double t46 = POW_1_3(ZETA_THRESHOLD_C_SCAN);
  double t47 = t46 * ZETA_THRESHOLD_C_SCAN;
  double t48 = POW_1_3(t44);
  double t49 = t48 * t44;
  double t50 = my_piecewise3(t45, t47, t49);
  double t51 = 0.1e1 - t43;
  double t52 = t51 <= ZETA_THRESHOLD_C_SCAN;
  double t53 = POW_1_3(t51);
  double t54 = t53 * t51;
  double t55 = my_piecewise3(t52, t47, t54);
  double t56 = t50 + t55 - 0.2e1;
  double t57 = M_CBRT2;
  double t58 = t57 - 0.1e1;
  double t60 = 0.1e1 / t58 / 0.2e1;
  double t61 = t56 * t60;
  double t76 = 0.1e1 + 0.278125e-1 * t12;
  double t81 =
    0.51785e1 * t15 + 0.905775e0 * t12 + 0.1100325e0 * t18 + 0.1241775e0 * t26;
  double t84  = 0.1e1 + 0.29608749977793437516e2 / t81;
  double t250 = 0.1e1 / t9 / t8;
  double t251 = t7 * t250;
  double t259 = 0.1e1 / t15 * t2;
  double t260 = t4 * t7;
  double t261 = t260 * t250;
  double t262 = t259 * t261;
  double t264 = t5 * t251;
  double t266 = sqrt(t12);
  double t267 = t266 * t2;
  double t268 = t267 * t261;
  double t273 = t22 * t6 / t23 / t8;
  double t321 = t81 * t81;
  double t322 = 0.1e1 / t321;
  double t328 = -0.86308333333333333334e0 * t262 - 0.301925e0 * t264 -
                0.5501625e-1 * t268 - 0.82785e-1 * t273;
  double t329 = 0.1e1 / t84;
  double t343 = t61 * t76;
  double t345 = t322 * t328 * t329;
  double t346 = t343 * t345;
  double t347 = 0.5848223622634646207e0 * t346;
  return t347;
}
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho1__t503(double rho0,
                        double rho1,
                        double sigma0,
                        double sigma1,
                        double sigma2,
                        double tau0,
                        double tau1)
{
  double t2   = M_CBRT3;
  double t3   = 0.1e1 / M_PI;
  double t4   = POW_1_3(t3);
  double t5   = t2 * t4;
  double t6   = M_CBRT4;
  double t7   = t6 * t6;
  double t8   = rho0 + rho1;
  double t9   = POW_1_3(t8);
  double t12  = t5 * t7 / t9;
  double t15  = sqrt(t12);
  double t23  = t9 * t9;
  double t35  = rho0 - rho1;
  double t36  = t35 * t35;
  double t37  = t36 * t36;
  double t38  = t8 * t8;
  double t39  = t38 * t38;
  double t42  = 0.1e1 / t8;
  double t43  = t35 * t42;
  double t44  = 0.1e1 + t43;
  double t45  = t44 <= ZETA_THRESHOLD_C_SCAN;
  double t46  = POW_1_3(ZETA_THRESHOLD_C_SCAN);
  double t47  = t46 * ZETA_THRESHOLD_C_SCAN;
  double t48  = POW_1_3(t44);
  double t49  = t48 * t44;
  double t50  = my_piecewise3(t45, t47, t49);
  double t51  = 0.1e1 - t43;
  double t52  = t51 <= ZETA_THRESHOLD_C_SCAN;
  double t53  = POW_1_3(t51);
  double t54  = t53 * t51;
  double t55  = my_piecewise3(t52, t47, t54);
  double t56  = t50 + t55 - 0.2e1;
  double t57  = M_CBRT2;
  double t58  = t57 - 0.1e1;
  double t60  = 0.1e1 / t58 / 0.2e1;
  double t95  = M_PI * M_PI;
  double t124 = sigma0 + 0.2e1 * sigma1 + sigma2;
  double t169 = 0.1e1 / t23 / t38;
  double t173 = M_CBRT6;
  double t175 = POW_1_3(t95);
  double t176 = t175 * t175;
  double t177 = 0.1e1 / t176;
  double t213 = 0.1e1 + 0.4445e-1 * t15 + 0.3138525e-1 * t12;
  double t214 = 0.1e1 / t213;
  double t217 = exp(0.1e1 * t214);
  double t218 = t217 - 0.1e1;
  double t219 = t173 * t177;
  double t220 = t57 * t57;
  double t221 = t220 * t124;
  double t225 = 0.1e1 + 0.21337642104376358333e-1 * t219 * t221 * t169;
  double t226 = POW_1_4(t225);
  double t228 = 0.1e1 - 0.1e1 / t226;
  double t230 = t218 * t228 + 0.1e1;
  double t237 = 0.1e1 - 0.2363e1 * t58 * t56 * t60;
  double t239 = t37 * t37;
  double t240 = t239 * t37;
  double t241 = t39 * t39;
  double t242 = t241 * t39;
  double t243 = 0.1e1 / t242;
  double t245 = -t240 * t243 + 0.1e1;
  double t250 = 0.1e1 / t9 / t8;
  double t251 = t7 * t250;
  double t259 = 0.1e1 / t15 * t2;
  double t260 = t4 * t7;
  double t261 = t260 * t250;
  double t262 = t259 * t261;
  double t264 = t5 * t251;
  double t364 = t38 * t8;
  double t366 = 0.1e1 / t23 / t364;
  double t378 = t124 * t366;
  double t479 = t213 * t213;
  double t480 = 0.1e1 / t479;
  double t483 = -0.74083333333333333333e-2 * t262 - 0.1046175e-1 * t264;
  double t484 = t480 * t483;
  double t486 = t217 * t228;
  double t490 = 0.1e1 / t226 / t225;
  double t491 = t218 * t490;
  double t492 = t491 * t173;
  double t493 = t177 * t220;
  double t497 =
    -0.1e1 * t484 * t486 - 0.14225094736250905555e-1 * t492 * t493 * t378;
  double t498 = 0.1e1 / t230;
  double t501 = 0.285764e-1 * t484 + 0.285764e-1 * t497 * t498;
  double t502 = t501 * t237;
  double t503 = t502 * t245;
  return t503;
}
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho1__t531(double rho0,
                        double rho1,
                        double sigma0,
                        double sigma1,
                        double sigma2,
                        double tau0,
                        double tau1)
{
  double t2  = M_CBRT3;
  double t3  = 0.1e1 / M_PI;
  double t4  = POW_1_3(t3);
  double t5  = t2 * t4;
  double t6  = M_CBRT4;
  double t7  = t6 * t6;
  double t8  = rho0 + rho1;
  double t9  = POW_1_3(t8);
  double t12 = t5 * t7 / t9;
  double t14 = 0.1e1 + 0.53425e-1 * t12;
  double t15 = sqrt(t12);
  double t18 = POW_3_2(t12);
  double t20 = t2 * t2;
  double t21 = t4 * t4;
  double t22 = t20 * t21;
  double t23 = t9 * t9;
  double t26 = t22 * t6 / t23;
  double t28 =
    0.379785e1 * t15 + 0.8969e0 * t12 + 0.204775e0 * t18 + 0.123235e0 * t26;
  double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
  double t32 = log(t31);
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
  double t45 = t44 <= ZETA_THRESHOLD_C_SCAN;
  double t48 = POW_1_3(t44);
  double t51 = 0.1e1 - t43;
  double t52 = t51 <= ZETA_THRESHOLD_C_SCAN;
  double t53 = POW_1_3(t51);
  double t57 = M_CBRT2;
  double t58 = t57 - 0.1e1;
  double t60 = 0.1e1 / t58 / 0.2e1;
  double t63 = 0.1e1 + 0.5137e-1 * t12;
  double t68 =
    0.705945e1 * t15 + 0.1549425e1 * t12 + 0.420775e0 * t18 + 0.1562925e0 * t26;
  double t71 = 0.1e1 + 0.32163958997385070134e2 / t68;
  double t72 = log(t71);
  double t76 = 0.1e1 + 0.278125e-1 * t12;
  double t81 =
    0.51785e1 * t15 + 0.905775e0 * t12 + 0.1100325e0 * t18 + 0.1241775e0 * t26;
  double t84 = 0.1e1 + 0.29608749977793437516e2 / t81;
  double t85 = log(t84);
  double t86 = t76 * t85;
  double t88 = -0.310907e-1 * t63 * t72 + t34 - 0.19751673498613801407e-1 * t86;
  double t289 = 0.1e1 / t38;
  double t290 = t35 * t289;
  double t520 = -t42 - t290;
  double t523 = my_piecewise3(t45, 0, 0.4e1 / 0.3e1 * t48 * t520);
  double t524 = -t520;
  double t527 = my_piecewise3(t52, 0, 0.4e1 / 0.3e1 * t53 * t524);
  double t528 = t523 + t527;
  double t529 = t528 * t60;
  double t530 = t529 * t88;
  double t531 = t41 * t530;
  return t531;
}
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho1__t544(double rho0,
                        double rho1,
                        double sigma0,
                        double sigma1,
                        double sigma2,
                        double tau0,
                        double tau1)
{
  double t90 =
    mgga_c_scan_vrho1__t90(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t141 =
    mgga_c_scan_vrho1__t141(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t2  = M_CBRT3;
  double t3  = 0.1e1 / M_PI;
  double t4  = POW_1_3(t3);
  double t5  = t2 * t4;
  double t6  = M_CBRT4;
  double t7  = t6 * t6;
  double t8  = rho0 + rho1;
  double t9  = POW_1_3(t8);
  double t12 = t5 * t7 / t9;
  double t14 = 0.1e1 + 0.53425e-1 * t12;
  double t15 = sqrt(t12);
  double t18 = POW_3_2(t12);
  double t20 = t2 * t2;
  double t21 = t4 * t4;
  double t22 = t20 * t21;
  double t23 = t9 * t9;
  double t26 = t22 * t6 / t23;
  double t28 =
    0.379785e1 * t15 + 0.8969e0 * t12 + 0.204775e0 * t18 + 0.123235e0 * t26;
  double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
  double t32 = log(t31);
  double t34 = 0.621814e-1 * t14 * t32;
  double t35 = rho0 - rho1;
  double t38 = t8 * t8;
  double t42 = 0.1e1 / t8;
  double t43 = t35 * t42;
  double t44 = 0.1e1 + t43;
  double t45 = t44 <= ZETA_THRESHOLD_C_SCAN;
  double t46 = POW_1_3(ZETA_THRESHOLD_C_SCAN);
  double t47 = t46 * ZETA_THRESHOLD_C_SCAN;
  double t48 = POW_1_3(t44);
  double t49 = t48 * t44;
  double t50 = my_piecewise3(t45, t47, t49);
  double t51 = 0.1e1 - t43;
  double t52 = t51 <= ZETA_THRESHOLD_C_SCAN;
  double t53 = POW_1_3(t51);
  double t54 = t53 * t51;
  double t55 = my_piecewise3(t52, t47, t54);
  double t56 = t50 + t55 - 0.2e1;
  double t57 = M_CBRT2;
  double t58 = t57 - 0.1e1;
  double t60 = 0.1e1 / t58 / 0.2e1;
  double t61 = t56 * t60;
  double t76 = 0.1e1 + 0.278125e-1 * t12;
  double t81 =
    0.51785e1 * t15 + 0.905775e0 * t12 + 0.1100325e0 * t18 + 0.1241775e0 * t26;
  double t84  = 0.1e1 + 0.29608749977793437516e2 / t81;
  double t85  = log(t84);
  double t86  = t76 * t85;
  double t92  = 0.19751673498613801407e-1 * t61 * t86;
  double t93  = log(0.2e1);
  double t94  = 0.1e1 - t93;
  double t95  = M_PI * M_PI;
  double t97  = t94 / t95;
  double t98  = t46 * t46;
  double t99  = t48 * t48;
  double t100 = my_piecewise3(t45, t98, t99);
  double t101 = t53 * t53;
  double t102 = my_piecewise3(t52, t98, t101);
  double t104 = t100 / 0.2e1 + t102 / 0.2e1;
  double t105 = t104 * t104;
  double t106 = t105 * t104;
  double t113 = 0.1e1 / t94;
  double t115 = (-t34 + t90 + t92) * t113;
  double t116 = 0.1e1 / t106;
  double t117 = t95 * t116;
  double t119 = exp(-t115 * t117);
  double t120 = t119 - 0.1e1;
  double t144 = 0.1e1 + 0.1e1 * t141 * t120;
  double t145 = log(t144);
  double t289 = 0.1e1 / t38;
  double t290 = t35 * t289;
  double t348 = t105 * t145;
  double t349 = 0.1e1 / t48;
  double t353 = 0.1e1 / t53;
  double t520 = -t42 - t290;
  double t524 = -t520;
  double t536 = my_piecewise3(t45, 0, 0.2e1 / 0.3e1 * t349 * t520);
  double t539 = my_piecewise3(t52, 0, 0.2e1 / 0.3e1 * t353 * t524);
  double t541 = t536 / 0.2e1 + t539 / 0.2e1;
  double t543 = t97 * t348 * t541;
  double t544 = 0.3e1 * t543;
  return t544;
}
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho1__t429(double rho0,
                        double rho1,
                        double sigma0,
                        double sigma1,
                        double sigma2,
                        double tau0,
                        double tau1)
{
  double t90 =
    mgga_c_scan_vrho1__t90(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t141 =
    mgga_c_scan_vrho1__t141(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t2  = M_CBRT3;
  double t3  = 0.1e1 / M_PI;
  double t4  = POW_1_3(t3);
  double t5  = t2 * t4;
  double t6  = M_CBRT4;
  double t7  = t6 * t6;
  double t8  = rho0 + rho1;
  double t9  = POW_1_3(t8);
  double t12 = t5 * t7 / t9;
  double t14 = 0.1e1 + 0.53425e-1 * t12;
  double t15 = sqrt(t12);
  double t18 = POW_3_2(t12);
  double t20 = t2 * t2;
  double t21 = t4 * t4;
  double t22 = t20 * t21;
  double t23 = t9 * t9;
  double t26 = t22 * t6 / t23;
  double t28 =
    0.379785e1 * t15 + 0.8969e0 * t12 + 0.204775e0 * t18 + 0.123235e0 * t26;
  double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
  double t32 = log(t31);
  double t34 = 0.621814e-1 * t14 * t32;
  double t35 = rho0 - rho1;
  double t42 = 0.1e1 / t8;
  double t43 = t35 * t42;
  double t44 = 0.1e1 + t43;
  double t45 = t44 <= ZETA_THRESHOLD_C_SCAN;
  double t46 = POW_1_3(ZETA_THRESHOLD_C_SCAN);
  double t47 = t46 * ZETA_THRESHOLD_C_SCAN;
  double t48 = POW_1_3(t44);
  double t49 = t48 * t44;
  double t50 = my_piecewise3(t45, t47, t49);
  double t51 = 0.1e1 - t43;
  double t52 = t51 <= ZETA_THRESHOLD_C_SCAN;
  double t53 = POW_1_3(t51);
  double t54 = t53 * t51;
  double t55 = my_piecewise3(t52, t47, t54);
  double t56 = t50 + t55 - 0.2e1;
  double t57 = M_CBRT2;
  double t58 = t57 - 0.1e1;
  double t60 = 0.1e1 / t58 / 0.2e1;
  double t61 = t56 * t60;
  double t76 = 0.1e1 + 0.278125e-1 * t12;
  double t81 =
    0.51785e1 * t15 + 0.905775e0 * t12 + 0.1100325e0 * t18 + 0.1241775e0 * t26;
  double t84  = 0.1e1 + 0.29608749977793437516e2 / t81;
  double t85  = log(t84);
  double t86  = t76 * t85;
  double t92  = 0.19751673498613801407e-1 * t61 * t86;
  double t93  = log(0.2e1);
  double t94  = 0.1e1 - t93;
  double t95  = M_PI * M_PI;
  double t98  = t46 * t46;
  double t99  = t48 * t48;
  double t100 = my_piecewise3(t45, t98, t99);
  double t101 = t53 * t53;
  double t102 = my_piecewise3(t52, t98, t101);
  double t104 = t100 / 0.2e1 + t102 / 0.2e1;
  double t105 = t104 * t104;
  double t106 = t105 * t104;
  double t113 = 0.1e1 / t94;
  double t115 = (-t34 + t90 + t92) * t113;
  double t116 = 0.1e1 / t106;
  double t117 = t95 * t116;
  double t119 = exp(-t115 * t117);
  double t120 = t119 - 0.1e1;
  double t144 = 0.1e1 + 0.1e1 * t141 * t120;
  double t429 = 0.1e1 / t144;
  return t429;
}
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho1__t546(double rho0,
                        double rho1,
                        double sigma0,
                        double sigma1,
                        double sigma2,
                        double tau0,
                        double tau1)
{
  double t283 =
    mgga_c_scan_vrho1__t283(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t288 =
    mgga_c_scan_vrho1__t288(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t335 =
    mgga_c_scan_vrho1__t335(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t342 =
    mgga_c_scan_vrho1__t342(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t347 =
    mgga_c_scan_vrho1__t347(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t531 =
    mgga_c_scan_vrho1__t531(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t2  = M_CBRT3;
  double t3  = 0.1e1 / M_PI;
  double t4  = POW_1_3(t3);
  double t5  = t2 * t4;
  double t6  = M_CBRT4;
  double t7  = t6 * t6;
  double t8  = rho0 + rho1;
  double t9  = POW_1_3(t8);
  double t12 = t5 * t7 / t9;
  double t14 = 0.1e1 + 0.53425e-1 * t12;
  double t15 = sqrt(t12);
  double t18 = POW_3_2(t12);
  double t20 = t2 * t2;
  double t21 = t4 * t4;
  double t22 = t20 * t21;
  double t23 = t9 * t9;
  double t26 = t22 * t6 / t23;
  double t28 =
    0.379785e1 * t15 + 0.8969e0 * t12 + 0.204775e0 * t18 + 0.123235e0 * t26;
  double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
  double t32 = log(t31);
  double t35 = rho0 - rho1;
  double t38 = t8 * t8;
  double t42 = 0.1e1 / t8;
  double t43 = t35 * t42;
  double t44 = 0.1e1 + t43;
  double t45 = t44 <= ZETA_THRESHOLD_C_SCAN;
  double t48 = POW_1_3(t44);
  double t51 = 0.1e1 - t43;
  double t52 = t51 <= ZETA_THRESHOLD_C_SCAN;
  double t53 = POW_1_3(t51);
  double t57 = M_CBRT2;
  double t58 = t57 - 0.1e1;
  double t60 = 0.1e1 / t58 / 0.2e1;
  double t76 = 0.1e1 + 0.278125e-1 * t12;
  double t81 =
    0.51785e1 * t15 + 0.905775e0 * t12 + 0.1100325e0 * t18 + 0.1241775e0 * t26;
  double t84  = 0.1e1 + 0.29608749977793437516e2 / t81;
  double t85  = log(t84);
  double t86  = t76 * t85;
  double t93  = log(0.2e1);
  double t94  = 0.1e1 - t93;
  double t113 = 0.1e1 / t94;
  double t250 = 0.1e1 / t9 / t8;
  double t251 = t7 * t250;
  double t253 = t5 * t251 * t32;
  double t254 = 0.11073470983333333333e-2 * t253;
  double t255 = t28 * t28;
  double t256 = 0.1e1 / t255;
  double t257 = t14 * t256;
  double t259 = 0.1e1 / t15 * t2;
  double t260 = t4 * t7;
  double t261 = t260 * t250;
  double t262 = t259 * t261;
  double t264 = t5 * t251;
  double t266 = sqrt(t12);
  double t267 = t266 * t2;
  double t268 = t267 * t261;
  double t273 = t22 * t6 / t23 / t8;
  double t275 = -0.632975e0 * t262 - 0.29896666666666666667e0 * t264 -
                0.1023875e0 * t268 - 0.82156666666666666667e-1 * t273;
  double t276 = 0.1e1 / t31;
  double t277 = t275 * t276;
  double t278 = t257 * t277;
  double t279 = 0.1e1 * t278;
  double t289 = 0.1e1 / t38;
  double t290 = t35 * t289;
  double t520 = -t42 - t290;
  double t523 = my_piecewise3(t45, 0, 0.4e1 / 0.3e1 * t48 * t520);
  double t524 = -t520;
  double t527 = my_piecewise3(t52, 0, 0.4e1 / 0.3e1 * t53 * t524);
  double t528 = t523 + t527;
  double t529 = t528 * t60;
  double t532 = t529 * t86;
  double t533 = 0.19751673498613801407e-1 * t532;
  double t546 =
    (t254 + t279 - t283 - t288 + t531 + t335 + t533 - t342 - t347) * t113;
  return t546;
}
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho1__t363(double rho0,
                        double rho1,
                        double sigma0,
                        double sigma1,
                        double sigma2,
                        double tau0,
                        double tau1)
{
  double t90 =
    mgga_c_scan_vrho1__t90(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t2  = M_CBRT3;
  double t3  = 0.1e1 / M_PI;
  double t4  = POW_1_3(t3);
  double t5  = t2 * t4;
  double t6  = M_CBRT4;
  double t7  = t6 * t6;
  double t8  = rho0 + rho1;
  double t9  = POW_1_3(t8);
  double t12 = t5 * t7 / t9;
  double t14 = 0.1e1 + 0.53425e-1 * t12;
  double t15 = sqrt(t12);
  double t18 = POW_3_2(t12);
  double t20 = t2 * t2;
  double t21 = t4 * t4;
  double t22 = t20 * t21;
  double t23 = t9 * t9;
  double t26 = t22 * t6 / t23;
  double t28 =
    0.379785e1 * t15 + 0.8969e0 * t12 + 0.204775e0 * t18 + 0.123235e0 * t26;
  double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
  double t32 = log(t31);
  double t34 = 0.621814e-1 * t14 * t32;
  double t35 = rho0 - rho1;
  double t38 = t8 * t8;
  double t42 = 0.1e1 / t8;
  double t43 = t35 * t42;
  double t44 = 0.1e1 + t43;
  double t45 = t44 <= ZETA_THRESHOLD_C_SCAN;
  double t46 = POW_1_3(ZETA_THRESHOLD_C_SCAN);
  double t47 = t46 * ZETA_THRESHOLD_C_SCAN;
  double t48 = POW_1_3(t44);
  double t49 = t48 * t44;
  double t50 = my_piecewise3(t45, t47, t49);
  double t51 = 0.1e1 - t43;
  double t52 = t51 <= ZETA_THRESHOLD_C_SCAN;
  double t53 = POW_1_3(t51);
  double t54 = t53 * t51;
  double t55 = my_piecewise3(t52, t47, t54);
  double t56 = t50 + t55 - 0.2e1;
  double t57 = M_CBRT2;
  double t58 = t57 - 0.1e1;
  double t60 = 0.1e1 / t58 / 0.2e1;
  double t61 = t56 * t60;
  double t76 = 0.1e1 + 0.278125e-1 * t12;
  double t81 =
    0.51785e1 * t15 + 0.905775e0 * t12 + 0.1100325e0 * t18 + 0.1241775e0 * t26;
  double t84  = 0.1e1 + 0.29608749977793437516e2 / t81;
  double t85  = log(t84);
  double t86  = t76 * t85;
  double t92  = 0.19751673498613801407e-1 * t61 * t86;
  double t93  = log(0.2e1);
  double t94  = 0.1e1 - t93;
  double t95  = M_PI * M_PI;
  double t98  = t46 * t46;
  double t99  = t48 * t48;
  double t100 = my_piecewise3(t45, t98, t99);
  double t101 = t53 * t53;
  double t102 = my_piecewise3(t52, t98, t101);
  double t104 = t100 / 0.2e1 + t102 / 0.2e1;
  double t105 = t104 * t104;
  double t106 = t105 * t104;
  double t108 = 0.1e1 + 0.25e-1 * t12;
  double t110 = 0.1e1 + 0.4445e-1 * t12;
  double t111 = 0.1e1 / t110;
  double t112 = t108 * t111;
  double t113 = 0.1e1 / t94;
  double t115 = (-t34 + t90 + t92) * t113;
  double t116 = 0.1e1 / t106;
  double t117 = t95 * t116;
  double t119 = exp(-t115 * t117);
  double t120 = t119 - 0.1e1;
  double t121 = 0.1e1 / t120;
  double t122 = t113 * t121;
  double t124 = sigma0 + 0.2e1 * sigma1 + sigma2;
  double t125 = t122 * t124;
  double t126 = t112 * t125;
  double t128 = 0.1e1 / t9 / t38;
  double t129 = t128 * t57;
  double t130 = 0.1e1 / t105;
  double t132 = 0.1e1 / t4;
  double t133 = t20 * t132;
  double t134 = t133 * t6;
  double t138 = 0.1e1 + 0.27439371595564631661e-1 * t126 * t129 * t130 * t134;
  double t139 = POW_1_4(t138);
  double t363 = 0.1e1 / t139 / t138;
  return t363;
}
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho1__t381(double rho0,
                        double rho1,
                        double sigma0,
                        double sigma1,
                        double sigma2,
                        double tau0,
                        double tau1)
{
  double t90 =
    mgga_c_scan_vrho1__t90(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t2  = M_CBRT3;
  double t3  = 0.1e1 / M_PI;
  double t4  = POW_1_3(t3);
  double t5  = t2 * t4;
  double t6  = M_CBRT4;
  double t7  = t6 * t6;
  double t8  = rho0 + rho1;
  double t9  = POW_1_3(t8);
  double t12 = t5 * t7 / t9;
  double t14 = 0.1e1 + 0.53425e-1 * t12;
  double t15 = sqrt(t12);
  double t18 = POW_3_2(t12);
  double t20 = t2 * t2;
  double t21 = t4 * t4;
  double t22 = t20 * t21;
  double t23 = t9 * t9;
  double t26 = t22 * t6 / t23;
  double t28 =
    0.379785e1 * t15 + 0.8969e0 * t12 + 0.204775e0 * t18 + 0.123235e0 * t26;
  double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
  double t32 = log(t31);
  double t34 = 0.621814e-1 * t14 * t32;
  double t35 = rho0 - rho1;
  double t38 = t8 * t8;
  double t42 = 0.1e1 / t8;
  double t43 = t35 * t42;
  double t44 = 0.1e1 + t43;
  double t45 = t44 <= ZETA_THRESHOLD_C_SCAN;
  double t46 = POW_1_3(ZETA_THRESHOLD_C_SCAN);
  double t47 = t46 * ZETA_THRESHOLD_C_SCAN;
  double t48 = POW_1_3(t44);
  double t49 = t48 * t44;
  double t50 = my_piecewise3(t45, t47, t49);
  double t51 = 0.1e1 - t43;
  double t52 = t51 <= ZETA_THRESHOLD_C_SCAN;
  double t53 = POW_1_3(t51);
  double t54 = t53 * t51;
  double t55 = my_piecewise3(t52, t47, t54);
  double t56 = t50 + t55 - 0.2e1;
  double t57 = M_CBRT2;
  double t58 = t57 - 0.1e1;
  double t60 = 0.1e1 / t58 / 0.2e1;
  double t61 = t56 * t60;
  double t76 = 0.1e1 + 0.278125e-1 * t12;
  double t81 =
    0.51785e1 * t15 + 0.905775e0 * t12 + 0.1100325e0 * t18 + 0.1241775e0 * t26;
  double t84  = 0.1e1 + 0.29608749977793437516e2 / t81;
  double t85  = log(t84);
  double t86  = t76 * t85;
  double t92  = 0.19751673498613801407e-1 * t61 * t86;
  double t93  = log(0.2e1);
  double t94  = 0.1e1 - t93;
  double t95  = M_PI * M_PI;
  double t98  = t46 * t46;
  double t99  = t48 * t48;
  double t100 = my_piecewise3(t45, t98, t99);
  double t101 = t53 * t53;
  double t102 = my_piecewise3(t52, t98, t101);
  double t104 = t100 / 0.2e1 + t102 / 0.2e1;
  double t105 = t104 * t104;
  double t106 = t105 * t104;
  double t108 = 0.1e1 + 0.25e-1 * t12;
  double t110 = 0.1e1 + 0.4445e-1 * t12;
  double t113 = 0.1e1 / t94;
  double t115 = (-t34 + t90 + t92) * t113;
  double t116 = 0.1e1 / t106;
  double t117 = t95 * t116;
  double t119 = exp(-t115 * t117);
  double t120 = t119 - 0.1e1;
  double t121 = 0.1e1 / t120;
  double t122 = t113 * t121;
  double t124 = sigma0 + 0.2e1 * sigma1 + sigma2;
  double t130 = 0.1e1 / t105;
  double t364 = t38 * t8;
  double t366 = 0.1e1 / t23 / t364;
  double t370 = t57 * t130;
  double t374 = t110 * t110;
  double t375 = 0.1e1 / t374;
  double t376 = t108 * t375;
  double t377 = t376 * t122;
  double t378 = t124 * t366;
  double t381 = 0.48787202696913915093e-2 * t377 * t378 * t370;
  return t381;
}
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho1__t411(double rho0,
                        double rho1,
                        double sigma0,
                        double sigma1,
                        double sigma2,
                        double tau0,
                        double tau1)
{
  double t90 =
    mgga_c_scan_vrho1__t90(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t2  = M_CBRT3;
  double t3  = 0.1e1 / M_PI;
  double t4  = POW_1_3(t3);
  double t5  = t2 * t4;
  double t6  = M_CBRT4;
  double t7  = t6 * t6;
  double t8  = rho0 + rho1;
  double t9  = POW_1_3(t8);
  double t12 = t5 * t7 / t9;
  double t14 = 0.1e1 + 0.53425e-1 * t12;
  double t15 = sqrt(t12);
  double t18 = POW_3_2(t12);
  double t20 = t2 * t2;
  double t21 = t4 * t4;
  double t22 = t20 * t21;
  double t23 = t9 * t9;
  double t26 = t22 * t6 / t23;
  double t28 =
    0.379785e1 * t15 + 0.8969e0 * t12 + 0.204775e0 * t18 + 0.123235e0 * t26;
  double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
  double t32 = log(t31);
  double t34 = 0.621814e-1 * t14 * t32;
  double t35 = rho0 - rho1;
  double t38 = t8 * t8;
  double t42 = 0.1e1 / t8;
  double t43 = t35 * t42;
  double t44 = 0.1e1 + t43;
  double t45 = t44 <= ZETA_THRESHOLD_C_SCAN;
  double t46 = POW_1_3(ZETA_THRESHOLD_C_SCAN);
  double t47 = t46 * ZETA_THRESHOLD_C_SCAN;
  double t48 = POW_1_3(t44);
  double t49 = t48 * t44;
  double t50 = my_piecewise3(t45, t47, t49);
  double t51 = 0.1e1 - t43;
  double t52 = t51 <= ZETA_THRESHOLD_C_SCAN;
  double t53 = POW_1_3(t51);
  double t54 = t53 * t51;
  double t55 = my_piecewise3(t52, t47, t54);
  double t56 = t50 + t55 - 0.2e1;
  double t57 = M_CBRT2;
  double t58 = t57 - 0.1e1;
  double t60 = 0.1e1 / t58 / 0.2e1;
  double t61 = t56 * t60;
  double t76 = 0.1e1 + 0.278125e-1 * t12;
  double t81 =
    0.51785e1 * t15 + 0.905775e0 * t12 + 0.1100325e0 * t18 + 0.1241775e0 * t26;
  double t84  = 0.1e1 + 0.29608749977793437516e2 / t81;
  double t85  = log(t84);
  double t86  = t76 * t85;
  double t92  = 0.19751673498613801407e-1 * t61 * t86;
  double t93  = log(0.2e1);
  double t94  = 0.1e1 - t93;
  double t95  = M_PI * M_PI;
  double t98  = t46 * t46;
  double t99  = t48 * t48;
  double t100 = my_piecewise3(t45, t98, t99);
  double t101 = t53 * t53;
  double t102 = my_piecewise3(t52, t98, t101);
  double t104 = t100 / 0.2e1 + t102 / 0.2e1;
  double t105 = t104 * t104;
  double t106 = t105 * t104;
  double t108 = 0.1e1 + 0.25e-1 * t12;
  double t110 = 0.1e1 + 0.4445e-1 * t12;
  double t111 = 0.1e1 / t110;
  double t112 = t108 * t111;
  double t113 = 0.1e1 / t94;
  double t115 = (-t34 + t90 + t92) * t113;
  double t116 = 0.1e1 / t106;
  double t117 = t95 * t116;
  double t119 = exp(-t115 * t117);
  double t120 = t119 - 0.1e1;
  double t121 = 0.1e1 / t120;
  double t122 = t113 * t121;
  double t124 = sigma0 + 0.2e1 * sigma1 + sigma2;
  double t125 = t122 * t124;
  double t126 = t112 * t125;
  double t130 = 0.1e1 / t105;
  double t132 = 0.1e1 / t4;
  double t133 = t20 * t132;
  double t134 = t133 * t6;
  double t364 = t38 * t8;
  double t406 = 0.1e1 / t9 / t364;
  double t407 = t406 * t57;
  double t411 = 0.64025200389650807209e-1 * t126 * t407 * t130 * t134;
  return t411;
}
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho1__t554(double rho0,
                        double rho1,
                        double sigma0,
                        double sigma1,
                        double sigma2,
                        double tau0,
                        double tau1)
{
  double t90 =
    mgga_c_scan_vrho1__t90(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t546 =
    mgga_c_scan_vrho1__t546(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t2  = M_CBRT3;
  double t3  = 0.1e1 / M_PI;
  double t4  = POW_1_3(t3);
  double t5  = t2 * t4;
  double t6  = M_CBRT4;
  double t7  = t6 * t6;
  double t8  = rho0 + rho1;
  double t9  = POW_1_3(t8);
  double t12 = t5 * t7 / t9;
  double t14 = 0.1e1 + 0.53425e-1 * t12;
  double t15 = sqrt(t12);
  double t18 = POW_3_2(t12);
  double t20 = t2 * t2;
  double t21 = t4 * t4;
  double t22 = t20 * t21;
  double t23 = t9 * t9;
  double t26 = t22 * t6 / t23;
  double t28 =
    0.379785e1 * t15 + 0.8969e0 * t12 + 0.204775e0 * t18 + 0.123235e0 * t26;
  double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
  double t32 = log(t31);
  double t34 = 0.621814e-1 * t14 * t32;
  double t35 = rho0 - rho1;
  double t38 = t8 * t8;
  double t42 = 0.1e1 / t8;
  double t43 = t35 * t42;
  double t44 = 0.1e1 + t43;
  double t45 = t44 <= ZETA_THRESHOLD_C_SCAN;
  double t46 = POW_1_3(ZETA_THRESHOLD_C_SCAN);
  double t47 = t46 * ZETA_THRESHOLD_C_SCAN;
  double t48 = POW_1_3(t44);
  double t49 = t48 * t44;
  double t50 = my_piecewise3(t45, t47, t49);
  double t51 = 0.1e1 - t43;
  double t52 = t51 <= ZETA_THRESHOLD_C_SCAN;
  double t53 = POW_1_3(t51);
  double t54 = t53 * t51;
  double t55 = my_piecewise3(t52, t47, t54);
  double t56 = t50 + t55 - 0.2e1;
  double t57 = M_CBRT2;
  double t58 = t57 - 0.1e1;
  double t60 = 0.1e1 / t58 / 0.2e1;
  double t61 = t56 * t60;
  double t76 = 0.1e1 + 0.278125e-1 * t12;
  double t81 =
    0.51785e1 * t15 + 0.905775e0 * t12 + 0.1100325e0 * t18 + 0.1241775e0 * t26;
  double t84  = 0.1e1 + 0.29608749977793437516e2 / t81;
  double t85  = log(t84);
  double t86  = t76 * t85;
  double t92  = 0.19751673498613801407e-1 * t61 * t86;
  double t93  = log(0.2e1);
  double t94  = 0.1e1 - t93;
  double t95  = M_PI * M_PI;
  double t98  = t46 * t46;
  double t99  = t48 * t48;
  double t100 = my_piecewise3(t45, t98, t99);
  double t101 = t53 * t53;
  double t102 = my_piecewise3(t52, t98, t101);
  double t104 = t100 / 0.2e1 + t102 / 0.2e1;
  double t105 = t104 * t104;
  double t106 = t105 * t104;
  double t113 = 0.1e1 / t94;
  double t115 = (-t34 + t90 + t92) * t113;
  double t116 = 0.1e1 / t106;
  double t117 = t95 * t116;
  double t119 = exp(-t115 * t117);
  double t130 = 0.1e1 / t105;
  double t132 = 0.1e1 / t4;
  double t289 = 0.1e1 / t38;
  double t290 = t35 * t289;
  double t349 = 0.1e1 / t48;
  double t353 = 0.1e1 / t53;
  double t370 = t57 * t130;
  double t388 = t370 * t20;
  double t389 = t132 * t6;
  double t393 = t105 * t105;
  double t394 = 0.1e1 / t393;
  double t395 = t95 * t394;
  double t520 = -t42 - t290;
  double t524 = -t520;
  double t536 = my_piecewise3(t45, 0, 0.2e1 / 0.3e1 * t349 * t520);
  double t539 = my_piecewise3(t52, 0, 0.2e1 / 0.3e1 * t353 * t524);
  double t541 = t536 / 0.2e1 + t539 / 0.2e1;
  double t548 = t395 * t541;
  double t551 = 0.3e1 * t115 * t548 - t546 * t117;
  double t552 = t551 * t119;
  double t554 = t388 * t389 * t552;
  return t554;
}
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho1__t562(double rho0,
                        double rho1,
                        double sigma0,
                        double sigma1,
                        double sigma2,
                        double tau0,
                        double tau1)
{
  double t90 =
    mgga_c_scan_vrho1__t90(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t363 =
    mgga_c_scan_vrho1__t363(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t381 =
    mgga_c_scan_vrho1__t381(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t411 =
    mgga_c_scan_vrho1__t411(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t554 =
    mgga_c_scan_vrho1__t554(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t2  = M_CBRT3;
  double t3  = 0.1e1 / M_PI;
  double t4  = POW_1_3(t3);
  double t5  = t2 * t4;
  double t6  = M_CBRT4;
  double t7  = t6 * t6;
  double t8  = rho0 + rho1;
  double t9  = POW_1_3(t8);
  double t12 = t5 * t7 / t9;
  double t14 = 0.1e1 + 0.53425e-1 * t12;
  double t15 = sqrt(t12);
  double t18 = POW_3_2(t12);
  double t20 = t2 * t2;
  double t21 = t4 * t4;
  double t22 = t20 * t21;
  double t23 = t9 * t9;
  double t26 = t22 * t6 / t23;
  double t28 =
    0.379785e1 * t15 + 0.8969e0 * t12 + 0.204775e0 * t18 + 0.123235e0 * t26;
  double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
  double t32 = log(t31);
  double t34 = 0.621814e-1 * t14 * t32;
  double t35 = rho0 - rho1;
  double t38 = t8 * t8;
  double t42 = 0.1e1 / t8;
  double t43 = t35 * t42;
  double t44 = 0.1e1 + t43;
  double t45 = t44 <= ZETA_THRESHOLD_C_SCAN;
  double t46 = POW_1_3(ZETA_THRESHOLD_C_SCAN);
  double t47 = t46 * ZETA_THRESHOLD_C_SCAN;
  double t48 = POW_1_3(t44);
  double t49 = t48 * t44;
  double t50 = my_piecewise3(t45, t47, t49);
  double t51 = 0.1e1 - t43;
  double t52 = t51 <= ZETA_THRESHOLD_C_SCAN;
  double t53 = POW_1_3(t51);
  double t54 = t53 * t51;
  double t55 = my_piecewise3(t52, t47, t54);
  double t56 = t50 + t55 - 0.2e1;
  double t57 = M_CBRT2;
  double t58 = t57 - 0.1e1;
  double t60 = 0.1e1 / t58 / 0.2e1;
  double t61 = t56 * t60;
  double t76 = 0.1e1 + 0.278125e-1 * t12;
  double t81 =
    0.51785e1 * t15 + 0.905775e0 * t12 + 0.1100325e0 * t18 + 0.1241775e0 * t26;
  double t84  = 0.1e1 + 0.29608749977793437516e2 / t81;
  double t85  = log(t84);
  double t86  = t76 * t85;
  double t92  = 0.19751673498613801407e-1 * t61 * t86;
  double t93  = log(0.2e1);
  double t94  = 0.1e1 - t93;
  double t95  = M_PI * M_PI;
  double t98  = t46 * t46;
  double t99  = t48 * t48;
  double t100 = my_piecewise3(t45, t98, t99);
  double t101 = t53 * t53;
  double t102 = my_piecewise3(t52, t98, t101);
  double t104 = t100 / 0.2e1 + t102 / 0.2e1;
  double t105 = t104 * t104;
  double t106 = t105 * t104;
  double t108 = 0.1e1 + 0.25e-1 * t12;
  double t110 = 0.1e1 + 0.4445e-1 * t12;
  double t111 = 0.1e1 / t110;
  double t112 = t108 * t111;
  double t113 = 0.1e1 / t94;
  double t115 = (-t34 + t90 + t92) * t113;
  double t116 = 0.1e1 / t106;
  double t117 = t95 * t116;
  double t119 = exp(-t115 * t117);
  double t120 = t119 - 0.1e1;
  double t121 = 0.1e1 / t120;
  double t124 = sigma0 + 0.2e1 * sigma1 + sigma2;
  double t128 = 0.1e1 / t9 / t38;
  double t130 = 0.1e1 / t105;
  double t132 = 0.1e1 / t4;
  double t289 = 0.1e1 / t38;
  double t290 = t35 * t289;
  double t349 = 0.1e1 / t48;
  double t353 = 0.1e1 / t53;
  double t364 = t38 * t8;
  double t366 = 0.1e1 / t23 / t364;
  double t367 = t366 * t111;
  double t369 = t121 * t124;
  double t370 = t57 * t130;
  double t371 = t369 * t370;
  double t373 = 0.27439371595564631661e-2 * t367 * t113 * t371;
  double t382 = t112 * t113;
  double t383 = t120 * t120;
  double t384 = 0.1e1 / t383;
  double t385 = t384 * t124;
  double t387 = t382 * t385 * t128;
  double t389 = t132 * t6;
  double t413 = t382 * t369 * t128;
  double t414 = t57 * t116;
  double t415 = t414 * t20;
  double t520 = -t42 - t290;
  double t524 = -t520;
  double t536 = my_piecewise3(t45, 0, 0.2e1 / 0.3e1 * t349 * t520);
  double t539 = my_piecewise3(t52, 0, 0.2e1 / 0.3e1 * t353 * t524);
  double t541 = t536 / 0.2e1 + t539 / 0.2e1;
  double t558 = t415 * t389 * t541;
  double t561 = -t373 + t381 - 0.27439371595564631661e-1 * t387 * t554 - t411 -
                0.54878743191129263322e-1 * t413 * t558;
  double t562 = t363 * t561;
  return t562;
}
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho1__t571(double rho0,
                        double rho1,
                        double sigma0,
                        double sigma1,
                        double sigma2,
                        double tau0,
                        double tau1)
{
  double t90 =
    mgga_c_scan_vrho1__t90(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t141 =
    mgga_c_scan_vrho1__t141(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t429 =
    mgga_c_scan_vrho1__t429(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t546 =
    mgga_c_scan_vrho1__t546(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t562 =
    mgga_c_scan_vrho1__t562(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t2  = M_CBRT3;
  double t3  = 0.1e1 / M_PI;
  double t4  = POW_1_3(t3);
  double t5  = t2 * t4;
  double t6  = M_CBRT4;
  double t7  = t6 * t6;
  double t8  = rho0 + rho1;
  double t9  = POW_1_3(t8);
  double t12 = t5 * t7 / t9;
  double t14 = 0.1e1 + 0.53425e-1 * t12;
  double t15 = sqrt(t12);
  double t18 = POW_3_2(t12);
  double t20 = t2 * t2;
  double t21 = t4 * t4;
  double t22 = t20 * t21;
  double t23 = t9 * t9;
  double t26 = t22 * t6 / t23;
  double t28 =
    0.379785e1 * t15 + 0.8969e0 * t12 + 0.204775e0 * t18 + 0.123235e0 * t26;
  double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
  double t32 = log(t31);
  double t34 = 0.621814e-1 * t14 * t32;
  double t35 = rho0 - rho1;
  double t38 = t8 * t8;
  double t42 = 0.1e1 / t8;
  double t43 = t35 * t42;
  double t44 = 0.1e1 + t43;
  double t45 = t44 <= ZETA_THRESHOLD_C_SCAN;
  double t46 = POW_1_3(ZETA_THRESHOLD_C_SCAN);
  double t47 = t46 * ZETA_THRESHOLD_C_SCAN;
  double t48 = POW_1_3(t44);
  double t49 = t48 * t44;
  double t50 = my_piecewise3(t45, t47, t49);
  double t51 = 0.1e1 - t43;
  double t52 = t51 <= ZETA_THRESHOLD_C_SCAN;
  double t53 = POW_1_3(t51);
  double t54 = t53 * t51;
  double t55 = my_piecewise3(t52, t47, t54);
  double t56 = t50 + t55 - 0.2e1;
  double t57 = M_CBRT2;
  double t58 = t57 - 0.1e1;
  double t60 = 0.1e1 / t58 / 0.2e1;
  double t61 = t56 * t60;
  double t76 = 0.1e1 + 0.278125e-1 * t12;
  double t81 =
    0.51785e1 * t15 + 0.905775e0 * t12 + 0.1100325e0 * t18 + 0.1241775e0 * t26;
  double t84  = 0.1e1 + 0.29608749977793437516e2 / t81;
  double t85  = log(t84);
  double t86  = t76 * t85;
  double t92  = 0.19751673498613801407e-1 * t61 * t86;
  double t93  = log(0.2e1);
  double t94  = 0.1e1 - t93;
  double t95  = M_PI * M_PI;
  double t97  = t94 / t95;
  double t98  = t46 * t46;
  double t99  = t48 * t48;
  double t100 = my_piecewise3(t45, t98, t99);
  double t101 = t53 * t53;
  double t102 = my_piecewise3(t52, t98, t101);
  double t104 = t100 / 0.2e1 + t102 / 0.2e1;
  double t105 = t104 * t104;
  double t106 = t105 * t104;
  double t113 = 0.1e1 / t94;
  double t115 = (-t34 + t90 + t92) * t113;
  double t116 = 0.1e1 / t106;
  double t117 = t95 * t116;
  double t119 = exp(-t115 * t117);
  double t120 = t119 - 0.1e1;
  double t289 = 0.1e1 / t38;
  double t290 = t35 * t289;
  double t349 = 0.1e1 / t48;
  double t353 = 0.1e1 / t53;
  double t393 = t105 * t105;
  double t394 = 0.1e1 / t393;
  double t395 = t95 * t394;
  double t520 = -t42 - t290;
  double t524 = -t520;
  double t536 = my_piecewise3(t45, 0, 0.2e1 / 0.3e1 * t349 * t520);
  double t539 = my_piecewise3(t52, 0, 0.2e1 / 0.3e1 * t353 * t524);
  double t541 = t536 / 0.2e1 + t539 / 0.2e1;
  double t548 = t395 * t541;
  double t551 = 0.3e1 * t115 * t548 - t546 * t117;
  double t565 = t141 * t551;
  double t568 = 0.25e0 * t562 * t120 + 0.1e1 * t565 * t119;
  double t571 = t97 * t106 * t568 * t429;
  return t571;
}
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho1__t609(double rho0,
                        double rho1,
                        double sigma0,
                        double sigma1,
                        double sigma2,
                        double tau0,
                        double tau1)
{
  double t247 =
    mgga_c_scan_vrho1__t247(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t8   = rho0 + rho1;
  double t9   = POW_1_3(t8);
  double t23  = t9 * t9;
  double t35  = rho0 - rho1;
  double t38  = t8 * t8;
  double t42  = 0.1e1 / t8;
  double t43  = t35 * t42;
  double t44  = 0.1e1 + t43;
  double t51  = 0.1e1 - t43;
  double t95  = M_PI * M_PI;
  double t124 = sigma0 + 0.2e1 * sigma1 + sigma2;
  double t148 = POW_1_3(rho0);
  double t149 = t148 * t148;
  double t151 = 0.1e1 / t149 / rho0;
  double t152 = tau0 * t151;
  double t153 = t44 / 0.2e1;
  double t154 = POW_1_3(t153);
  double t155 = t154 * t154;
  double t156 = t155 * t153;
  double t158 = POW_1_3(rho1);
  double t159 = t158 * t158;
  double t161 = 0.1e1 / t159 / rho1;
  double t162 = tau1 * t161;
  double t163 = t51 / 0.2e1;
  double t164 = POW_1_3(t163);
  double t165 = t164 * t164;
  double t166 = t165 * t163;
  double t169 = 0.1e1 / t23 / t38;
  double t173 = M_CBRT6;
  double t174 = (t152 * t156 + t162 * t166 - t124 * t169 / 0.8e1) * t173;
  double t175 = POW_1_3(t95);
  double t176 = t175 * t175;
  double t177 = 0.1e1 / t176;
  double t178 = t156 + t166;
  double t179 = 0.1e1 / t178;
  double t180 = t177 * t179;
  double t182 = 0.5e1 / 0.9e1 * t174 * t180;
  double t183 = t182 <= 0.1e1;
  double t184 = log(DBL_EPSILON);
  double t187 = t184 / (-t184 + 0.64e0);
  double t188 = -t187 < t182;
  double t189 = t182 < -t187;
  double t190 = my_piecewise3(t189, t182, -t187);
  double t191 = 0.1e1 - t190;
  double t192 = 0.1e1 / t191;
  double t195 = exp(-0.64e0 * t190 * t192);
  double t198 = log(0.14285714285714285714e1 * DBL_EPSILON);
  double t201 = (-t198 + 0.15e1) / t198;
  double t202 = t182 < -t201;
  double t203 = my_piecewise3(t202, -t201, t182);
  double t204 = 0.1e1 - t203;
  double t207 = exp(0.15e1 / t204);
  double t289 = 0.1e1 / t38;
  double t290 = t35 * t289;
  double t364 = t38 * t8;
  double t366 = 0.1e1 / t23 / t364;
  double t378 = t124 * t366;
  double t446 = t378 / 0.3e1;
  double t450 = t178 * t178;
  double t451 = 0.1e1 / t450;
  double t452 = t177 * t451;
  double t462 = t191 * t191;
  double t463 = 0.1e1 / t462;
  double t464 = t190 * t463;
  double t470 = t204 * t204;
  double t471 = 0.1e1 / t470;
  double t520 = -t42 - t290;
  double t572 = t520 / 0.2e1;
  double t573 = t155 * t572;
  double t576 = rho1 * rho1;
  double t578 = 0.1e1 / t159 / t576;
  double t579 = tau1 * t578;
  double t582 = -t572;
  double t583 = t165 * t582;
  double t587 = (0.5e1 / 0.3e1 * t152 * t573 - 0.5e1 / 0.3e1 * t579 * t166 +
                 0.5e1 / 0.3e1 * t162 * t583 + t446) *
                t173;
  double t590 = 0.5e1 / 0.3e1 * t573 + 0.5e1 / 0.3e1 * t583;
  double t591 = t452 * t590;
  double t594 = -0.5e1 / 0.9e1 * t174 * t591 + 0.5e1 / 0.9e1 * t587 * t180;
  double t595 = my_piecewise3(t189, t594, 0);
  double t600 = -0.64e0 * t595 * t192 - 0.64e0 * t464 * t595;
  double t601 = t600 * t195;
  double t602 = my_piecewise3(t188, 0, t601);
  double t603 = my_piecewise3(t202, 0, t594);
  double t607 = my_piecewise3(t202, 0, -0.105e1 * t471 * t603 * t207);
  double t608 = my_piecewise3(t183, t602, t607);
  double t609 = t608 * t247;
  return t609;
}
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho1__t618(double rho0,
                        double rho1,
                        double sigma0,
                        double sigma1,
                        double sigma2,
                        double tau0,
                        double tau1)
{
  double t210 =
    mgga_c_scan_vrho1__t210(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t238 =
    mgga_c_scan_vrho1__t238(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t283 =
    mgga_c_scan_vrho1__t283(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t288 =
    mgga_c_scan_vrho1__t288(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t335 =
    mgga_c_scan_vrho1__t335(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t342 =
    mgga_c_scan_vrho1__t342(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t347 =
    mgga_c_scan_vrho1__t347(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t503 =
    mgga_c_scan_vrho1__t503(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t531 =
    mgga_c_scan_vrho1__t531(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t544 =
    mgga_c_scan_vrho1__t544(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t571 =
    mgga_c_scan_vrho1__t571(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t609 =
    mgga_c_scan_vrho1__t609(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t2  = M_CBRT3;
  double t3  = 0.1e1 / M_PI;
  double t4  = POW_1_3(t3);
  double t5  = t2 * t4;
  double t6  = M_CBRT4;
  double t7  = t6 * t6;
  double t8  = rho0 + rho1;
  double t9  = POW_1_3(t8);
  double t12 = t5 * t7 / t9;
  double t14 = 0.1e1 + 0.53425e-1 * t12;
  double t15 = sqrt(t12);
  double t18 = POW_3_2(t12);
  double t20 = t2 * t2;
  double t21 = t4 * t4;
  double t22 = t20 * t21;
  double t23 = t9 * t9;
  double t26 = t22 * t6 / t23;
  double t28 =
    0.379785e1 * t15 + 0.8969e0 * t12 + 0.204775e0 * t18 + 0.123235e0 * t26;
  double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
  double t32 = log(t31);
  double t35 = rho0 - rho1;
  double t36 = t35 * t35;
  double t37 = t36 * t36;
  double t38 = t8 * t8;
  double t39 = t38 * t38;
  double t42 = 0.1e1 / t8;
  double t43 = t35 * t42;
  double t44 = 0.1e1 + t43;
  double t45 = t44 <= ZETA_THRESHOLD_C_SCAN;
  double t48 = POW_1_3(t44);
  double t51 = 0.1e1 - t43;
  double t52 = t51 <= ZETA_THRESHOLD_C_SCAN;
  double t53 = POW_1_3(t51);
  double t57 = M_CBRT2;
  double t58 = t57 - 0.1e1;
  double t60 = 0.1e1 / t58 / 0.2e1;
  double t76 = 0.1e1 + 0.278125e-1 * t12;
  double t81 =
    0.51785e1 * t15 + 0.905775e0 * t12 + 0.1100325e0 * t18 + 0.1241775e0 * t26;
  double t84  = 0.1e1 + 0.29608749977793437516e2 / t81;
  double t85  = log(t84);
  double t86  = t76 * t85;
  double t95  = M_PI * M_PI;
  double t124 = sigma0 + 0.2e1 * sigma1 + sigma2;
  double t169 = 0.1e1 / t23 / t38;
  double t173 = M_CBRT6;
  double t175 = POW_1_3(t95);
  double t176 = t175 * t175;
  double t177 = 0.1e1 / t176;
  double t213 = 0.1e1 + 0.4445e-1 * t15 + 0.3138525e-1 * t12;
  double t214 = 0.1e1 / t213;
  double t217 = exp(0.1e1 * t214);
  double t218 = t217 - 0.1e1;
  double t219 = t173 * t177;
  double t220 = t57 * t57;
  double t221 = t220 * t124;
  double t225 = 0.1e1 + 0.21337642104376358333e-1 * t219 * t221 * t169;
  double t226 = POW_1_4(t225);
  double t228 = 0.1e1 - 0.1e1 / t226;
  double t230 = t218 * t228 + 0.1e1;
  double t231 = log(t230);
  double t233 = -0.285764e-1 * t214 + 0.285764e-1 * t231;
  double t239 = t37 * t37;
  double t240 = t239 * t37;
  double t241 = t39 * t39;
  double t242 = t241 * t39;
  double t243 = 0.1e1 / t242;
  double t245 = -t240 * t243 + 0.1e1;
  double t250 = 0.1e1 / t9 / t8;
  double t251 = t7 * t250;
  double t253 = t5 * t251 * t32;
  double t254 = 0.11073470983333333333e-2 * t253;
  double t255 = t28 * t28;
  double t256 = 0.1e1 / t255;
  double t257 = t14 * t256;
  double t259 = 0.1e1 / t15 * t2;
  double t260 = t4 * t7;
  double t261 = t260 * t250;
  double t262 = t259 * t261;
  double t264 = t5 * t251;
  double t266 = sqrt(t12);
  double t267 = t266 * t2;
  double t268 = t267 * t261;
  double t273 = t22 * t6 / t23 / t8;
  double t275 = -0.632975e0 * t262 - 0.29896666666666666667e0 * t264 -
                0.1023875e0 * t268 - 0.82156666666666666667e-1 * t273;
  double t276 = 0.1e1 / t31;
  double t277 = t275 * t276;
  double t278 = t257 * t277;
  double t279 = 0.1e1 * t278;
  double t280 = t36 * t35;
  double t284 = t39 * t8;
  double t289 = 0.1e1 / t38;
  double t290 = t35 * t289;
  double t504 = t233 * t58;
  double t508 = t239 * t280;
  double t509 = t508 * t243;
  double t510 = t241 * t284;
  double t511 = 0.1e1 / t510;
  double t512 = t240 * t511;
  double t520 = -t42 - t290;
  double t523 = my_piecewise3(t45, 0, 0.4e1 / 0.3e1 * t48 * t520);
  double t524 = -t520;
  double t527 = my_piecewise3(t52, 0, 0.4e1 / 0.3e1 * t53 * t524);
  double t528 = t523 + t527;
  double t529 = t528 * t60;
  double t532 = t529 * t86;
  double t533 = 0.19751673498613801407e-1 * t532;
  double t610 = t529 * t245;
  double t614 = 0.12e2 * t509 + 0.12e2 * t512;
  double t616 = t503 - 0.2363e1 * t504 * t610 + t238 * t614 - t254 - t279 +
                t283 + t288 - t531 - t335 - t533 + t342 + t347 - t544 - t571;
  double t617 = t210 * t616;
  double t618 = t254 + t279 - t283 - t288 + t531 + t335 + t533 - t342 - t347 +
                t544 + t571 + t609 + t617;
  return t618;
}
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vrho1(double rho0,
                  double rho1,
                  double sigma0,
                  double sigma1,
                  double sigma2,
                  double tau0,
                  double tau1)
{
  double t90 =
    mgga_c_scan_vrho1__t90(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t147 =
    mgga_c_scan_vrho1__t147(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t248 =
    mgga_c_scan_vrho1__t248(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t618 =
    mgga_c_scan_vrho1__t618(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t2  = M_CBRT3;
  double t3  = 0.1e1 / M_PI;
  double t4  = POW_1_3(t3);
  double t5  = t2 * t4;
  double t6  = M_CBRT4;
  double t7  = t6 * t6;
  double t8  = rho0 + rho1;
  double t9  = POW_1_3(t8);
  double t12 = t5 * t7 / t9;
  double t14 = 0.1e1 + 0.53425e-1 * t12;
  double t15 = sqrt(t12);
  double t18 = POW_3_2(t12);
  double t20 = t2 * t2;
  double t21 = t4 * t4;
  double t22 = t20 * t21;
  double t23 = t9 * t9;
  double t26 = t22 * t6 / t23;
  double t28 =
    0.379785e1 * t15 + 0.8969e0 * t12 + 0.204775e0 * t18 + 0.123235e0 * t26;
  double t31 = 0.1e1 + 0.16081979498692535067e2 / t28;
  double t32 = log(t31);
  double t34 = 0.621814e-1 * t14 * t32;
  double t35 = rho0 - rho1;
  double t42 = 0.1e1 / t8;
  double t43 = t35 * t42;
  double t44 = 0.1e1 + t43;
  double t45 = t44 <= ZETA_THRESHOLD_C_SCAN;
  double t46 = POW_1_3(ZETA_THRESHOLD_C_SCAN);
  double t47 = t46 * ZETA_THRESHOLD_C_SCAN;
  double t48 = POW_1_3(t44);
  double t49 = t48 * t44;
  double t50 = my_piecewise3(t45, t47, t49);
  double t51 = 0.1e1 - t43;
  double t52 = t51 <= ZETA_THRESHOLD_C_SCAN;
  double t53 = POW_1_3(t51);
  double t54 = t53 * t51;
  double t55 = my_piecewise3(t52, t47, t54);
  double t56 = t50 + t55 - 0.2e1;
  double t57 = M_CBRT2;
  double t58 = t57 - 0.1e1;
  double t60 = 0.1e1 / t58 / 0.2e1;
  double t61 = t56 * t60;
  double t76 = 0.1e1 + 0.278125e-1 * t12;
  double t81 =
    0.51785e1 * t15 + 0.905775e0 * t12 + 0.1100325e0 * t18 + 0.1241775e0 * t26;
  double t84    = 0.1e1 + 0.29608749977793437516e2 / t81;
  double t85    = log(t84);
  double t86    = t76 * t85;
  double t92    = 0.19751673498613801407e-1 * t61 * t86;
  double tvrho1 = t8 * t618 + t147 + t248 - t34 + t90 + t92;
  return tvrho1;
}
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vsigma0(double rho0,
                    double rho1,
                    double sigma0,
                    double sigma1,
                    double sigma2,
                    double tau0,
                    double tau1)
{
  MGGA_C_SCAN_VSIGMA0
  return tvsigma0;
}
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vsigma1(double rho0,
                    double rho1,
                    double sigma0,
                    double sigma1,
                    double sigma2,
                    double tau0,
                    double tau1)
{
  MGGA_C_SCAN_VSIGMA1
  return tvsigma1;
}
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vtau0(double rho0,
                  double rho1,
                  double sigma0,
                  double sigma1,
                  double sigma2,
                  double tau0,
                  double tau1)
{
  MGGA_C_SCAN_VTAU0
  return tvtau0;
}
DFTFE_DEVICE_NOINLINE double
mgga_c_scan_vtau1(double rho0,
                  double rho1,
                  double sigma0,
                  double sigma1,
                  double sigma2,
                  double tau0,
                  double tau1)
{
  MGGA_C_SCAN_VTAU1
  return tvtau1;
}
DFTFE_DEVICE_NOINLINE double
mgga_x_scan_zk(double rho0,
               double rho1,
               double sigma0,
               double sigma1,
               double sigma2,
               double tau0,
               double tau1)
{
  MGGA_X_SCAN_ZK
  return tzk0;
}
// ---- mgga_x_scan_vrho0: recursively decomposed into bounded-cone
// __noinline__ sub-helpers ----
DFTFE_DEVICE_NOINLINE double
mgga_x_scan_vrho0__t241(double, double, double, double, double, double, double);
DFTFE_DEVICE_NOINLINE double
mgga_x_scan_vrho0__t335(double, double, double, double, double, double, double);
DFTFE_DEVICE_NOINLINE double
mgga_x_scan_vrho0__t355(double, double, double, double, double, double, double);
DFTFE_DEVICE_NOINLINE double
mgga_x_scan_vrho0__t370(double, double, double, double, double, double, double);
DFTFE_DEVICE_NOINLINE double
mgga_x_scan_vrho0(double, double, double, double, double, double, double);

DFTFE_DEVICE_NOINLINE double
mgga_x_scan_vrho0__t241(double rho0,
                        double rho1,
                        double sigma0,
                        double sigma1,
                        double sigma2,
                        double tau0,
                        double tau1)
{
  struct mgga_x_scan_params
  {
    double c1 = 0.667;
    double c2 = 0.8;
    double d  = 1.24;
    double k1 = 0.065;
  } params;
  double t3   = M_CBRT3;
  double t4   = M_CBRTPI;
  double t6   = t3 / t4;
  double t7   = rho0 + rho1;
  double t8   = 0.1e1 / t7;
  double t11  = 0.2e1 * rho0 * t8 <= ZETA_THRESHOLD_X_SCAN;
  double t12  = ZETA_THRESHOLD_X_SCAN - 0.1e1;
  double t15  = 0.2e1 * rho1 * t8 <= ZETA_THRESHOLD_X_SCAN;
  double t16  = -t12;
  double t17  = rho0 - rho1;
  double t22  = POW_1_3(ZETA_THRESHOLD_X_SCAN);
  double t23  = t22 * ZETA_THRESHOLD_X_SCAN;
  double t28  = POW_1_3(t7);
  double t29  = M_CBRT6;
  double t30  = M_PI * M_PI;
  double t31  = POW_1_3(t30);
  double t32  = t31 * t31;
  double t33  = 0.1e1 / t32;
  double t34  = t29 * t33;
  double t45  = 0.1e3 / 0.6561e4 / params.k1 - 0.73e2 / 0.648e3;
  double t46  = t29 * t29;
  double t47  = t45 * t46;
  double t48  = t31 * t30;
  double t49  = 0.1e1 / t48;
  double t50  = t47 * t49;
  double t57  = t45 * t29;
  double t66  = sqrt(0.146e3);
  double t67  = t66 * t29;
  double t94  = log(DBL_EPSILON);
  double t97  = t94 / (-t94 + params.c1);
  double t107 = fabs(params.d);
  double t110 = log(DBL_EPSILON / t107);
  double t113 = (-t110 + params.c2) / t110;
  double t128 = sqrt(0.3e1);
  double t129 = 0.1e1 / t31;
  double t130 = t46 * t129;
  double t146 = rho1 <= DENS_THRESHOLD_X_SCAN;
  double t147 = -t17;
  double t149 = my_piecewise5(t15, t12, t11, t16, t147 * t8);
  double t150 = 0.1e1 + t149;
  double t151 = t150 <= ZETA_THRESHOLD_X_SCAN;
  double t152 = POW_1_3(t150);
  double t154 = my_piecewise3(t151, t23, t152 * t150);
  double t155 = t6 * t154;
  double t156 = rho1 * rho1;
  double t157 = POW_1_3(rho1);
  double t158 = t157 * t157;
  double t159 = t158 * t156;
  double t160 = 0.1e1 / t159;
  double t161 = sigma2 * t160;
  double t162 = t34 * t161;
  double t164 = sigma2 * sigma2;
  double t165 = t156 * t156;
  double t166 = t165 * rho1;
  double t168 = 0.1e1 / t157 / t166;
  double t169 = t164 * t168;
  double t170 = t33 * sigma2;
  double t171 = t170 * t160;
  double t174 = exp(-0.27e2 / 0.8e2 * t57 * t171);
  double t180 = t158 * rho1;
  double t181 = 0.1e1 / t180;
  double t187 = 0.5e1 / 0.9e1 * (tau1 * t181 - t161 / 0.8e1) * t29 * t33;
  double t188 = 0.1e1 - t187;
  double t190 = t188 * t188;
  double t192 = exp(-t190 / 0.2e1);
  double t195 = 0.7e1 / 0.1296e5 * t67 * t171 + t66 * t188 * t192 / 0.1e3;
  double t196 = t195 * t195;
  double t197 =
    params.k1 + 0.5e1 / 0.972e3 * t162 + t50 * t169 * t174 / 0.576e3 + t196;
  double t202 = 0.1e1 + params.k1 * (0.1e1 - params.k1 / t197);
  double t203 = t187 <= 0.1e1;
  double t204 = -t97 < t187;
  double t205 = t187 < -t97;
  double t206 = my_piecewise3(t205, t187, -t97);
  double t207 = params.c1 * t206;
  double t208 = 0.1e1 - t206;
  double t209 = 0.1e1 / t208;
  double t211 = exp(-t207 * t209);
  double t212 = my_piecewise3(t204, 0, t211);
  double t213 = t187 < -t113;
  double t214 = my_piecewise3(t213, -t113, t187);
  double t215 = 0.1e1 - t214;
  double t218 = exp(params.c2 / t215);
  double t220 = my_piecewise3(t213, 0, -params.d * t218);
  double t221 = my_piecewise3(t203, t212, t220);
  double t222 = 0.1e1 - t221;
  double t225 = t202 * t222 + 0.1174e1 * t221;
  double t226 = t28 * t225;
  double t227 = sqrt(sigma2);
  double t228 = t157 * rho1;
  double t229 = 0.1e1 / t228;
  double t231 = t130 * t227 * t229;
  double t232 = sqrt(t231);
  double t236 = exp(-0.98958e1 * t128 / t232);
  double t237 = 0.1e1 - t236;
  double t238 = t226 * t237;
  double t241 = my_piecewise3(t146, 0, -0.3e1 / 0.8e1 * t155 * t238);
  return t241;
}
DFTFE_DEVICE_NOINLINE double
mgga_x_scan_vrho0__t335(double rho0,
                        double rho1,
                        double sigma0,
                        double sigma1,
                        double sigma2,
                        double tau0,
                        double tau1)
{
  struct mgga_x_scan_params
  {
    double c1 = 0.667;
    double c2 = 0.8;
    double d  = 1.24;
    double k1 = 0.065;
  } params;
  double t7  = rho0 + rho1;
  double t28 = POW_1_3(t7);
  double t29 = M_CBRT6;
  double t30 = M_PI * M_PI;
  double t31 = POW_1_3(t30);
  double t32 = t31 * t31;
  double t33 = 0.1e1 / t32;
  double t34 = t29 * t33;
  double t35 = rho0 * rho0;
  double t36 = POW_1_3(rho0);
  double t37 = t36 * t36;
  double t38 = t37 * t35;
  double t39 = 0.1e1 / t38;
  double t40 = sigma0 * t39;
  double t41 = t34 * t40;
  double t45 = 0.1e3 / 0.6561e4 / params.k1 - 0.73e2 / 0.648e3;
  double t46 = t29 * t29;
  double t47 = t45 * t46;
  double t48 = t31 * t30;
  double t49 = 0.1e1 / t48;
  double t50 = t47 * t49;
  double t51 = sigma0 * sigma0;
  double t52 = t35 * t35;
  double t53 = t52 * rho0;
  double t55 = 0.1e1 / t36 / t53;
  double t56 = t51 * t55;
  double t57 = t45 * t29;
  double t58 = t33 * sigma0;
  double t59 = t58 * t39;
  double t62 = exp(-0.27e2 / 0.8e2 * t57 * t59);
  double t66 = sqrt(0.146e3);
  double t67 = t66 * t29;
  double t70 = t37 * rho0;
  double t71 = 0.1e1 / t70;
  double t77 = 0.5e1 / 0.9e1 * (tau0 * t71 - t40 / 0.8e1) * t29 * t33;
  double t78 = 0.1e1 - t77;
  double t80 = t78 * t78;
  double t82 = exp(-t80 / 0.2e1);
  double t85 = 0.7e1 / 0.1296e5 * t67 * t59 + t66 * t78 * t82 / 0.1e3;
  double t86 = t85 * t85;
  double t87 =
    params.k1 + 0.5e1 / 0.972e3 * t41 + t50 * t56 * t62 / 0.576e3 + t86;
  double t92  = 0.1e1 + params.k1 * (0.1e1 - params.k1 / t87);
  double t93  = t77 <= 0.1e1;
  double t94  = log(DBL_EPSILON);
  double t97  = t94 / (-t94 + params.c1);
  double t98  = -t97 < t77;
  double t99  = t77 < -t97;
  double t100 = my_piecewise3(t99, t77, -t97);
  double t101 = params.c1 * t100;
  double t102 = 0.1e1 - t100;
  double t103 = 0.1e1 / t102;
  double t105 = exp(-t101 * t103);
  double t106 = my_piecewise3(t98, 0, t105);
  double t107 = fabs(params.d);
  double t110 = log(DBL_EPSILON / t107);
  double t113 = (-t110 + params.c2) / t110;
  double t114 = t77 < -t113;
  double t115 = my_piecewise3(t114, -t113, t77);
  double t116 = 0.1e1 - t115;
  double t119 = exp(params.c2 / t116);
  double t121 = my_piecewise3(t114, 0, -params.d * t119);
  double t122 = my_piecewise3(t93, t106, t121);
  double t123 = 0.1e1 - t122;
  double t128 = sqrt(0.3e1);
  double t129 = 0.1e1 / t31;
  double t130 = t46 * t129;
  double t131 = sqrt(sigma0);
  double t132 = t36 * rho0;
  double t133 = 0.1e1 / t132;
  double t135 = t130 * t131 * t133;
  double t136 = sqrt(t135);
  double t140 = exp(-0.98958e1 * t128 / t136);
  double t141 = 0.1e1 - t140;
  double t259 = params.k1 * params.k1;
  double t260 = t87 * t87;
  double t262 = t259 / t260;
  double t263 = t35 * rho0;
  double t265 = 0.1e1 / t37 / t263;
  double t266 = sigma0 * t265;
  double t269 = t52 * t35;
  double t271 = 0.1e1 / t36 / t269;
  double t276 = t45 * t45;
  double t277 = t30 * t30;
  double t278 = 0.1e1 / t277;
  double t279 = t276 * t278;
  double t280 = t51 * sigma0;
  double t281 = t52 * t52;
  double t282 = t281 * rho0;
  double t283 = 0.1e1 / t282;
  double t294 = -0.5e1 / 0.3e1 * tau0 * t39 + t266 / 0.3e1;
  double t296 = t34 * t82;
  double t299 = t66 * t80;
  double t303 = -0.7e1 / 0.486e4 * t67 * t58 * t265 -
                t66 * t294 * t296 / 0.18e3 + t299 * t294 * t296 / 0.18e3;
  double t306 = -0.1e2 / 0.729e3 * t34 * t266 -
                t50 * t51 * t271 * t62 / 0.108e3 +
                0.3e1 / 0.32e3 * t279 * t280 * t283 * t62 + 0.2e1 * t85 * t303;
  double t307 = t306 * t123;
  double t309 = t294 * t29;
  double t311 = 0.5e1 / 0.9e1 * t309 * t33;
  double t312 = my_piecewise3(t99, t311, 0);
  double t315 = t102 * t102;
  double t316 = 0.1e1 / t315;
  double t317 = t316 * t312;
  double t319 = -params.c1 * t312 * t103 - t101 * t317;
  double t320 = t319 * t105;
  double t321 = my_piecewise3(t98, 0, t320);
  double t322 = params.d * params.c2;
  double t323 = t116 * t116;
  double t324 = 0.1e1 / t323;
  double t325 = my_piecewise3(t114, 0, t311);
  double t329 = my_piecewise3(t114, 0, -t322 * t324 * t325 * t119);
  double t330 = my_piecewise3(t93, t321, t329);
  double t333 = t262 * t307 - t92 * t330 + 0.1174e1 * t330;
  double t334 = t28 * t333;
  double t335 = t334 * t141;
  return t335;
}
DFTFE_DEVICE_NOINLINE double
mgga_x_scan_vrho0__t355(double rho0,
                        double rho1,
                        double sigma0,
                        double sigma1,
                        double sigma2,
                        double tau0,
                        double tau1)
{
  struct mgga_x_scan_params
  {
    double c1 = 0.667;
    double c2 = 0.8;
    double d  = 1.24;
    double k1 = 0.065;
  } params;
  double t335 =
    mgga_x_scan_vrho0__t335(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t2  = rho0 <= DENS_THRESHOLD_X_SCAN;
  double t3  = M_CBRT3;
  double t4  = M_CBRTPI;
  double t6  = t3 / t4;
  double t7  = rho0 + rho1;
  double t8  = 0.1e1 / t7;
  double t11 = 0.2e1 * rho0 * t8 <= ZETA_THRESHOLD_X_SCAN;
  double t12 = ZETA_THRESHOLD_X_SCAN - 0.1e1;
  double t15 = 0.2e1 * rho1 * t8 <= ZETA_THRESHOLD_X_SCAN;
  double t16 = -t12;
  double t17 = rho0 - rho1;
  double t19 = my_piecewise5(t11, t12, t15, t16, t17 * t8);
  double t20 = 0.1e1 + t19;
  double t21 = t20 <= ZETA_THRESHOLD_X_SCAN;
  double t22 = POW_1_3(ZETA_THRESHOLD_X_SCAN);
  double t23 = t22 * ZETA_THRESHOLD_X_SCAN;
  double t24 = POW_1_3(t20);
  double t26 = my_piecewise3(t21, t23, t24 * t20);
  double t27 = t6 * t26;
  double t28 = POW_1_3(t7);
  double t29 = M_CBRT6;
  double t30 = M_PI * M_PI;
  double t31 = POW_1_3(t30);
  double t32 = t31 * t31;
  double t33 = 0.1e1 / t32;
  double t34 = t29 * t33;
  double t35 = rho0 * rho0;
  double t36 = POW_1_3(rho0);
  double t37 = t36 * t36;
  double t38 = t37 * t35;
  double t39 = 0.1e1 / t38;
  double t40 = sigma0 * t39;
  double t41 = t34 * t40;
  double t45 = 0.1e3 / 0.6561e4 / params.k1 - 0.73e2 / 0.648e3;
  double t46 = t29 * t29;
  double t47 = t45 * t46;
  double t48 = t31 * t30;
  double t49 = 0.1e1 / t48;
  double t50 = t47 * t49;
  double t51 = sigma0 * sigma0;
  double t52 = t35 * t35;
  double t53 = t52 * rho0;
  double t55 = 0.1e1 / t36 / t53;
  double t56 = t51 * t55;
  double t57 = t45 * t29;
  double t58 = t33 * sigma0;
  double t59 = t58 * t39;
  double t62 = exp(-0.27e2 / 0.8e2 * t57 * t59);
  double t66 = sqrt(0.146e3);
  double t67 = t66 * t29;
  double t70 = t37 * rho0;
  double t71 = 0.1e1 / t70;
  double t77 = 0.5e1 / 0.9e1 * (tau0 * t71 - t40 / 0.8e1) * t29 * t33;
  double t78 = 0.1e1 - t77;
  double t80 = t78 * t78;
  double t82 = exp(-t80 / 0.2e1);
  double t85 = 0.7e1 / 0.1296e5 * t67 * t59 + t66 * t78 * t82 / 0.1e3;
  double t86 = t85 * t85;
  double t87 =
    params.k1 + 0.5e1 / 0.972e3 * t41 + t50 * t56 * t62 / 0.576e3 + t86;
  double t92  = 0.1e1 + params.k1 * (0.1e1 - params.k1 / t87);
  double t93  = t77 <= 0.1e1;
  double t94  = log(DBL_EPSILON);
  double t97  = t94 / (-t94 + params.c1);
  double t98  = -t97 < t77;
  double t99  = t77 < -t97;
  double t100 = my_piecewise3(t99, t77, -t97);
  double t101 = params.c1 * t100;
  double t102 = 0.1e1 - t100;
  double t103 = 0.1e1 / t102;
  double t105 = exp(-t101 * t103);
  double t106 = my_piecewise3(t98, 0, t105);
  double t107 = fabs(params.d);
  double t110 = log(DBL_EPSILON / t107);
  double t113 = (-t110 + params.c2) / t110;
  double t114 = t77 < -t113;
  double t115 = my_piecewise3(t114, -t113, t77);
  double t116 = 0.1e1 - t115;
  double t119 = exp(params.c2 / t116);
  double t121 = my_piecewise3(t114, 0, -params.d * t119);
  double t122 = my_piecewise3(t93, t106, t121);
  double t123 = 0.1e1 - t122;
  double t126 = t92 * t123 + 0.1174e1 * t122;
  double t127 = t28 * t126;
  double t128 = sqrt(0.3e1);
  double t129 = 0.1e1 / t31;
  double t130 = t46 * t129;
  double t131 = sqrt(sigma0);
  double t132 = t36 * rho0;
  double t133 = 0.1e1 / t132;
  double t135 = t130 * t131 * t133;
  double t136 = sqrt(t135);
  double t140 = exp(-0.98958e1 * t128 / t136);
  double t141 = 0.1e1 - t140;
  double t142 = t127 * t141;
  double t242 = t7 * t7;
  double t243 = 0.1e1 / t242;
  double t244 = t17 * t243;
  double t246 = my_piecewise5(t11, 0, t15, 0, t8 - t244);
  double t249 = my_piecewise3(t21, 0, 0.4e1 / 0.3e1 * t24 * t246);
  double t250 = t6 * t249;
  double t253 = t28 * t28;
  double t254 = 0.1e1 / t253;
  double t255 = t254 * t126;
  double t256 = t255 * t141;
  double t258 = t27 * t256 / 0.8e1;
  double t338 = pow(0.3e1, 0.1e1 / 0.6e1);
  double t339 = t338 * t338;
  double t340 = t339 * t339;
  double t341 = t340 * t338;
  double t342 = t341 * t26;
  double t344 = 0.1e1 / t136 / t135;
  double t345 = t127 * t344;
  double t346 = t342 * t345;
  double t348 = 0.1e1 / t36 / t35;
  double t351 = t130 * t131 * t348 * t140;
  double t355 = my_piecewise3(t2,
                              0,
                              -0.3e1 / 0.8e1 * t250 * t142 - t258 -
                                0.3e1 / 0.8e1 * t27 * t335 -
                                0.16891736332904387511e1 * t346 * t351);
  return t355;
}
DFTFE_DEVICE_NOINLINE double
mgga_x_scan_vrho0__t370(double rho0,
                        double rho1,
                        double sigma0,
                        double sigma1,
                        double sigma2,
                        double tau0,
                        double tau1)
{
  struct mgga_x_scan_params
  {
    double c1 = 0.667;
    double c2 = 0.8;
    double d  = 1.24;
    double k1 = 0.065;
  } params;
  double t3   = M_CBRT3;
  double t4   = M_CBRTPI;
  double t6   = t3 / t4;
  double t7   = rho0 + rho1;
  double t8   = 0.1e1 / t7;
  double t11  = 0.2e1 * rho0 * t8 <= ZETA_THRESHOLD_X_SCAN;
  double t12  = ZETA_THRESHOLD_X_SCAN - 0.1e1;
  double t15  = 0.2e1 * rho1 * t8 <= ZETA_THRESHOLD_X_SCAN;
  double t16  = -t12;
  double t17  = rho0 - rho1;
  double t22  = POW_1_3(ZETA_THRESHOLD_X_SCAN);
  double t23  = t22 * ZETA_THRESHOLD_X_SCAN;
  double t28  = POW_1_3(t7);
  double t29  = M_CBRT6;
  double t30  = M_PI * M_PI;
  double t31  = POW_1_3(t30);
  double t32  = t31 * t31;
  double t33  = 0.1e1 / t32;
  double t34  = t29 * t33;
  double t45  = 0.1e3 / 0.6561e4 / params.k1 - 0.73e2 / 0.648e3;
  double t46  = t29 * t29;
  double t47  = t45 * t46;
  double t48  = t31 * t30;
  double t49  = 0.1e1 / t48;
  double t50  = t47 * t49;
  double t57  = t45 * t29;
  double t66  = sqrt(0.146e3);
  double t67  = t66 * t29;
  double t94  = log(DBL_EPSILON);
  double t97  = t94 / (-t94 + params.c1);
  double t107 = fabs(params.d);
  double t110 = log(DBL_EPSILON / t107);
  double t113 = (-t110 + params.c2) / t110;
  double t128 = sqrt(0.3e1);
  double t129 = 0.1e1 / t31;
  double t130 = t46 * t129;
  double t146 = rho1 <= DENS_THRESHOLD_X_SCAN;
  double t147 = -t17;
  double t149 = my_piecewise5(t15, t12, t11, t16, t147 * t8);
  double t150 = 0.1e1 + t149;
  double t151 = t150 <= ZETA_THRESHOLD_X_SCAN;
  double t152 = POW_1_3(t150);
  double t154 = my_piecewise3(t151, t23, t152 * t150);
  double t155 = t6 * t154;
  double t156 = rho1 * rho1;
  double t157 = POW_1_3(rho1);
  double t158 = t157 * t157;
  double t159 = t158 * t156;
  double t160 = 0.1e1 / t159;
  double t161 = sigma2 * t160;
  double t162 = t34 * t161;
  double t164 = sigma2 * sigma2;
  double t165 = t156 * t156;
  double t166 = t165 * rho1;
  double t168 = 0.1e1 / t157 / t166;
  double t169 = t164 * t168;
  double t170 = t33 * sigma2;
  double t171 = t170 * t160;
  double t174 = exp(-0.27e2 / 0.8e2 * t57 * t171);
  double t180 = t158 * rho1;
  double t181 = 0.1e1 / t180;
  double t187 = 0.5e1 / 0.9e1 * (tau1 * t181 - t161 / 0.8e1) * t29 * t33;
  double t188 = 0.1e1 - t187;
  double t190 = t188 * t188;
  double t192 = exp(-t190 / 0.2e1);
  double t195 = 0.7e1 / 0.1296e5 * t67 * t171 + t66 * t188 * t192 / 0.1e3;
  double t196 = t195 * t195;
  double t197 =
    params.k1 + 0.5e1 / 0.972e3 * t162 + t50 * t169 * t174 / 0.576e3 + t196;
  double t202 = 0.1e1 + params.k1 * (0.1e1 - params.k1 / t197);
  double t203 = t187 <= 0.1e1;
  double t204 = -t97 < t187;
  double t205 = t187 < -t97;
  double t206 = my_piecewise3(t205, t187, -t97);
  double t207 = params.c1 * t206;
  double t208 = 0.1e1 - t206;
  double t209 = 0.1e1 / t208;
  double t211 = exp(-t207 * t209);
  double t212 = my_piecewise3(t204, 0, t211);
  double t213 = t187 < -t113;
  double t214 = my_piecewise3(t213, -t113, t187);
  double t215 = 0.1e1 - t214;
  double t218 = exp(params.c2 / t215);
  double t220 = my_piecewise3(t213, 0, -params.d * t218);
  double t221 = my_piecewise3(t203, t212, t220);
  double t222 = 0.1e1 - t221;
  double t225 = t202 * t222 + 0.1174e1 * t221;
  double t226 = t28 * t225;
  double t227 = sqrt(sigma2);
  double t228 = t157 * rho1;
  double t229 = 0.1e1 / t228;
  double t231 = t130 * t227 * t229;
  double t232 = sqrt(t231);
  double t236 = exp(-0.98958e1 * t128 / t232);
  double t237 = 0.1e1 - t236;
  double t238 = t226 * t237;
  double t242 = t7 * t7;
  double t243 = 0.1e1 / t242;
  double t253 = t28 * t28;
  double t254 = 0.1e1 / t253;
  double t356 = t147 * t243;
  double t358 = my_piecewise5(t15, 0, t11, 0, -t8 - t356);
  double t361 = my_piecewise3(t151, 0, 0.4e1 / 0.3e1 * t152 * t358);
  double t362 = t6 * t361;
  double t365 = t254 * t225;
  double t366 = t365 * t237;
  double t368 = t155 * t366 / 0.8e1;
  double t370 = my_piecewise3(t146, 0, -0.3e1 / 0.8e1 * t362 * t238 - t368);
  return t370;
}
DFTFE_DEVICE_NOINLINE double
mgga_x_scan_vrho0(double rho0,
                  double rho1,
                  double sigma0,
                  double sigma1,
                  double sigma2,
                  double tau0,
                  double tau1)
{
  struct mgga_x_scan_params
  {
    double c1 = 0.667;
    double c2 = 0.8;
    double d  = 1.24;
    double k1 = 0.065;
  } params;
  double t241 =
    mgga_x_scan_vrho0__t241(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t355 =
    mgga_x_scan_vrho0__t355(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t370 =
    mgga_x_scan_vrho0__t370(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t2  = rho0 <= DENS_THRESHOLD_X_SCAN;
  double t3  = M_CBRT3;
  double t4  = M_CBRTPI;
  double t6  = t3 / t4;
  double t7  = rho0 + rho1;
  double t8  = 0.1e1 / t7;
  double t11 = 0.2e1 * rho0 * t8 <= ZETA_THRESHOLD_X_SCAN;
  double t12 = ZETA_THRESHOLD_X_SCAN - 0.1e1;
  double t15 = 0.2e1 * rho1 * t8 <= ZETA_THRESHOLD_X_SCAN;
  double t16 = -t12;
  double t17 = rho0 - rho1;
  double t19 = my_piecewise5(t11, t12, t15, t16, t17 * t8);
  double t20 = 0.1e1 + t19;
  double t21 = t20 <= ZETA_THRESHOLD_X_SCAN;
  double t22 = POW_1_3(ZETA_THRESHOLD_X_SCAN);
  double t23 = t22 * ZETA_THRESHOLD_X_SCAN;
  double t24 = POW_1_3(t20);
  double t26 = my_piecewise3(t21, t23, t24 * t20);
  double t27 = t6 * t26;
  double t28 = POW_1_3(t7);
  double t29 = M_CBRT6;
  double t30 = M_PI * M_PI;
  double t31 = POW_1_3(t30);
  double t32 = t31 * t31;
  double t33 = 0.1e1 / t32;
  double t34 = t29 * t33;
  double t35 = rho0 * rho0;
  double t36 = POW_1_3(rho0);
  double t37 = t36 * t36;
  double t38 = t37 * t35;
  double t39 = 0.1e1 / t38;
  double t40 = sigma0 * t39;
  double t41 = t34 * t40;
  double t45 = 0.1e3 / 0.6561e4 / params.k1 - 0.73e2 / 0.648e3;
  double t46 = t29 * t29;
  double t47 = t45 * t46;
  double t48 = t31 * t30;
  double t49 = 0.1e1 / t48;
  double t50 = t47 * t49;
  double t51 = sigma0 * sigma0;
  double t52 = t35 * t35;
  double t53 = t52 * rho0;
  double t55 = 0.1e1 / t36 / t53;
  double t56 = t51 * t55;
  double t57 = t45 * t29;
  double t58 = t33 * sigma0;
  double t59 = t58 * t39;
  double t62 = exp(-0.27e2 / 0.8e2 * t57 * t59);
  double t66 = sqrt(0.146e3);
  double t67 = t66 * t29;
  double t70 = t37 * rho0;
  double t71 = 0.1e1 / t70;
  double t77 = 0.5e1 / 0.9e1 * (tau0 * t71 - t40 / 0.8e1) * t29 * t33;
  double t78 = 0.1e1 - t77;
  double t80 = t78 * t78;
  double t82 = exp(-t80 / 0.2e1);
  double t85 = 0.7e1 / 0.1296e5 * t67 * t59 + t66 * t78 * t82 / 0.1e3;
  double t86 = t85 * t85;
  double t87 =
    params.k1 + 0.5e1 / 0.972e3 * t41 + t50 * t56 * t62 / 0.576e3 + t86;
  double t92    = 0.1e1 + params.k1 * (0.1e1 - params.k1 / t87);
  double t93    = t77 <= 0.1e1;
  double t94    = log(DBL_EPSILON);
  double t97    = t94 / (-t94 + params.c1);
  double t98    = -t97 < t77;
  double t99    = t77 < -t97;
  double t100   = my_piecewise3(t99, t77, -t97);
  double t101   = params.c1 * t100;
  double t102   = 0.1e1 - t100;
  double t103   = 0.1e1 / t102;
  double t105   = exp(-t101 * t103);
  double t106   = my_piecewise3(t98, 0, t105);
  double t107   = fabs(params.d);
  double t110   = log(DBL_EPSILON / t107);
  double t113   = (-t110 + params.c2) / t110;
  double t114   = t77 < -t113;
  double t115   = my_piecewise3(t114, -t113, t77);
  double t116   = 0.1e1 - t115;
  double t119   = exp(params.c2 / t116);
  double t121   = my_piecewise3(t114, 0, -params.d * t119);
  double t122   = my_piecewise3(t93, t106, t121);
  double t123   = 0.1e1 - t122;
  double t126   = t92 * t123 + 0.1174e1 * t122;
  double t127   = t28 * t126;
  double t128   = sqrt(0.3e1);
  double t129   = 0.1e1 / t31;
  double t130   = t46 * t129;
  double t131   = sqrt(sigma0);
  double t132   = t36 * rho0;
  double t133   = 0.1e1 / t132;
  double t135   = t130 * t131 * t133;
  double t136   = sqrt(t135);
  double t140   = exp(-0.98958e1 * t128 / t136);
  double t141   = 0.1e1 - t140;
  double t142   = t127 * t141;
  double t145   = my_piecewise3(t2, 0, -0.3e1 / 0.8e1 * t27 * t142);
  double tvrho0 = t145 + t241 + t7 * (t355 + t370);
  return tvrho0;
}
// ---- mgga_x_scan_vrho1: recursively decomposed into bounded-cone
// __noinline__ sub-helpers ----
DFTFE_DEVICE_NOINLINE double
mgga_x_scan_vrho1__t241(double, double, double, double, double, double, double);
DFTFE_DEVICE_NOINLINE double
mgga_x_scan_vrho1__t382(double, double, double, double, double, double, double);
DFTFE_DEVICE_NOINLINE double
mgga_x_scan_vrho1__t461(double, double, double, double, double, double, double);
DFTFE_DEVICE_NOINLINE double
mgga_x_scan_vrho1__t477(double, double, double, double, double, double, double);
DFTFE_DEVICE_NOINLINE double
mgga_x_scan_vrho1(double, double, double, double, double, double, double);

DFTFE_DEVICE_NOINLINE double
mgga_x_scan_vrho1__t241(double rho0,
                        double rho1,
                        double sigma0,
                        double sigma1,
                        double sigma2,
                        double tau0,
                        double tau1)
{
  struct mgga_x_scan_params
  {
    double c1 = 0.667;
    double c2 = 0.8;
    double d  = 1.24;
    double k1 = 0.065;
  } params;
  double t3   = M_CBRT3;
  double t4   = M_CBRTPI;
  double t6   = t3 / t4;
  double t7   = rho0 + rho1;
  double t8   = 0.1e1 / t7;
  double t11  = 0.2e1 * rho0 * t8 <= ZETA_THRESHOLD_X_SCAN;
  double t12  = ZETA_THRESHOLD_X_SCAN - 0.1e1;
  double t15  = 0.2e1 * rho1 * t8 <= ZETA_THRESHOLD_X_SCAN;
  double t16  = -t12;
  double t17  = rho0 - rho1;
  double t22  = POW_1_3(ZETA_THRESHOLD_X_SCAN);
  double t23  = t22 * ZETA_THRESHOLD_X_SCAN;
  double t28  = POW_1_3(t7);
  double t29  = M_CBRT6;
  double t30  = M_PI * M_PI;
  double t31  = POW_1_3(t30);
  double t32  = t31 * t31;
  double t33  = 0.1e1 / t32;
  double t34  = t29 * t33;
  double t45  = 0.1e3 / 0.6561e4 / params.k1 - 0.73e2 / 0.648e3;
  double t46  = t29 * t29;
  double t47  = t45 * t46;
  double t48  = t31 * t30;
  double t49  = 0.1e1 / t48;
  double t50  = t47 * t49;
  double t57  = t45 * t29;
  double t66  = sqrt(0.146e3);
  double t67  = t66 * t29;
  double t94  = log(DBL_EPSILON);
  double t97  = t94 / (-t94 + params.c1);
  double t107 = fabs(params.d);
  double t110 = log(DBL_EPSILON / t107);
  double t113 = (-t110 + params.c2) / t110;
  double t128 = sqrt(0.3e1);
  double t129 = 0.1e1 / t31;
  double t130 = t46 * t129;
  double t146 = rho1 <= DENS_THRESHOLD_X_SCAN;
  double t147 = -t17;
  double t149 = my_piecewise5(t15, t12, t11, t16, t147 * t8);
  double t150 = 0.1e1 + t149;
  double t151 = t150 <= ZETA_THRESHOLD_X_SCAN;
  double t152 = POW_1_3(t150);
  double t154 = my_piecewise3(t151, t23, t152 * t150);
  double t155 = t6 * t154;
  double t156 = rho1 * rho1;
  double t157 = POW_1_3(rho1);
  double t158 = t157 * t157;
  double t159 = t158 * t156;
  double t160 = 0.1e1 / t159;
  double t161 = sigma2 * t160;
  double t162 = t34 * t161;
  double t164 = sigma2 * sigma2;
  double t165 = t156 * t156;
  double t166 = t165 * rho1;
  double t168 = 0.1e1 / t157 / t166;
  double t169 = t164 * t168;
  double t170 = t33 * sigma2;
  double t171 = t170 * t160;
  double t174 = exp(-0.27e2 / 0.8e2 * t57 * t171);
  double t180 = t158 * rho1;
  double t181 = 0.1e1 / t180;
  double t187 = 0.5e1 / 0.9e1 * (tau1 * t181 - t161 / 0.8e1) * t29 * t33;
  double t188 = 0.1e1 - t187;
  double t190 = t188 * t188;
  double t192 = exp(-t190 / 0.2e1);
  double t195 = 0.7e1 / 0.1296e5 * t67 * t171 + t66 * t188 * t192 / 0.1e3;
  double t196 = t195 * t195;
  double t197 =
    params.k1 + 0.5e1 / 0.972e3 * t162 + t50 * t169 * t174 / 0.576e3 + t196;
  double t202 = 0.1e1 + params.k1 * (0.1e1 - params.k1 / t197);
  double t203 = t187 <= 0.1e1;
  double t204 = -t97 < t187;
  double t205 = t187 < -t97;
  double t206 = my_piecewise3(t205, t187, -t97);
  double t207 = params.c1 * t206;
  double t208 = 0.1e1 - t206;
  double t209 = 0.1e1 / t208;
  double t211 = exp(-t207 * t209);
  double t212 = my_piecewise3(t204, 0, t211);
  double t213 = t187 < -t113;
  double t214 = my_piecewise3(t213, -t113, t187);
  double t215 = 0.1e1 - t214;
  double t218 = exp(params.c2 / t215);
  double t220 = my_piecewise3(t213, 0, -params.d * t218);
  double t221 = my_piecewise3(t203, t212, t220);
  double t222 = 0.1e1 - t221;
  double t225 = t202 * t222 + 0.1174e1 * t221;
  double t226 = t28 * t225;
  double t227 = sqrt(sigma2);
  double t228 = t157 * rho1;
  double t229 = 0.1e1 / t228;
  double t231 = t130 * t227 * t229;
  double t232 = sqrt(t231);
  double t236 = exp(-0.98958e1 * t128 / t232);
  double t237 = 0.1e1 - t236;
  double t238 = t226 * t237;
  double t241 = my_piecewise3(t146, 0, -0.3e1 / 0.8e1 * t155 * t238);
  return t241;
}
DFTFE_DEVICE_NOINLINE double
mgga_x_scan_vrho1__t382(double rho0,
                        double rho1,
                        double sigma0,
                        double sigma1,
                        double sigma2,
                        double tau0,
                        double tau1)
{
  struct mgga_x_scan_params
  {
    double c1 = 0.667;
    double c2 = 0.8;
    double d  = 1.24;
    double k1 = 0.065;
  } params;
  double t2  = rho0 <= DENS_THRESHOLD_X_SCAN;
  double t3  = M_CBRT3;
  double t4  = M_CBRTPI;
  double t6  = t3 / t4;
  double t7  = rho0 + rho1;
  double t8  = 0.1e1 / t7;
  double t11 = 0.2e1 * rho0 * t8 <= ZETA_THRESHOLD_X_SCAN;
  double t12 = ZETA_THRESHOLD_X_SCAN - 0.1e1;
  double t15 = 0.2e1 * rho1 * t8 <= ZETA_THRESHOLD_X_SCAN;
  double t16 = -t12;
  double t17 = rho0 - rho1;
  double t19 = my_piecewise5(t11, t12, t15, t16, t17 * t8);
  double t20 = 0.1e1 + t19;
  double t21 = t20 <= ZETA_THRESHOLD_X_SCAN;
  double t22 = POW_1_3(ZETA_THRESHOLD_X_SCAN);
  double t23 = t22 * ZETA_THRESHOLD_X_SCAN;
  double t24 = POW_1_3(t20);
  double t26 = my_piecewise3(t21, t23, t24 * t20);
  double t27 = t6 * t26;
  double t28 = POW_1_3(t7);
  double t29 = M_CBRT6;
  double t30 = M_PI * M_PI;
  double t31 = POW_1_3(t30);
  double t32 = t31 * t31;
  double t33 = 0.1e1 / t32;
  double t34 = t29 * t33;
  double t35 = rho0 * rho0;
  double t36 = POW_1_3(rho0);
  double t37 = t36 * t36;
  double t38 = t37 * t35;
  double t39 = 0.1e1 / t38;
  double t40 = sigma0 * t39;
  double t41 = t34 * t40;
  double t45 = 0.1e3 / 0.6561e4 / params.k1 - 0.73e2 / 0.648e3;
  double t46 = t29 * t29;
  double t47 = t45 * t46;
  double t48 = t31 * t30;
  double t49 = 0.1e1 / t48;
  double t50 = t47 * t49;
  double t51 = sigma0 * sigma0;
  double t52 = t35 * t35;
  double t53 = t52 * rho0;
  double t55 = 0.1e1 / t36 / t53;
  double t56 = t51 * t55;
  double t57 = t45 * t29;
  double t58 = t33 * sigma0;
  double t59 = t58 * t39;
  double t62 = exp(-0.27e2 / 0.8e2 * t57 * t59);
  double t66 = sqrt(0.146e3);
  double t67 = t66 * t29;
  double t70 = t37 * rho0;
  double t71 = 0.1e1 / t70;
  double t77 = 0.5e1 / 0.9e1 * (tau0 * t71 - t40 / 0.8e1) * t29 * t33;
  double t78 = 0.1e1 - t77;
  double t80 = t78 * t78;
  double t82 = exp(-t80 / 0.2e1);
  double t85 = 0.7e1 / 0.1296e5 * t67 * t59 + t66 * t78 * t82 / 0.1e3;
  double t86 = t85 * t85;
  double t87 =
    params.k1 + 0.5e1 / 0.972e3 * t41 + t50 * t56 * t62 / 0.576e3 + t86;
  double t92  = 0.1e1 + params.k1 * (0.1e1 - params.k1 / t87);
  double t93  = t77 <= 0.1e1;
  double t94  = log(DBL_EPSILON);
  double t97  = t94 / (-t94 + params.c1);
  double t98  = -t97 < t77;
  double t99  = t77 < -t97;
  double t100 = my_piecewise3(t99, t77, -t97);
  double t101 = params.c1 * t100;
  double t102 = 0.1e1 - t100;
  double t103 = 0.1e1 / t102;
  double t105 = exp(-t101 * t103);
  double t106 = my_piecewise3(t98, 0, t105);
  double t107 = fabs(params.d);
  double t110 = log(DBL_EPSILON / t107);
  double t113 = (-t110 + params.c2) / t110;
  double t114 = t77 < -t113;
  double t115 = my_piecewise3(t114, -t113, t77);
  double t116 = 0.1e1 - t115;
  double t119 = exp(params.c2 / t116);
  double t121 = my_piecewise3(t114, 0, -params.d * t119);
  double t122 = my_piecewise3(t93, t106, t121);
  double t123 = 0.1e1 - t122;
  double t126 = t92 * t123 + 0.1174e1 * t122;
  double t127 = t28 * t126;
  double t128 = sqrt(0.3e1);
  double t129 = 0.1e1 / t31;
  double t130 = t46 * t129;
  double t131 = sqrt(sigma0);
  double t132 = t36 * rho0;
  double t133 = 0.1e1 / t132;
  double t135 = t130 * t131 * t133;
  double t136 = sqrt(t135);
  double t140 = exp(-0.98958e1 * t128 / t136);
  double t141 = 0.1e1 - t140;
  double t142 = t127 * t141;
  double t242 = t7 * t7;
  double t243 = 0.1e1 / t242;
  double t244 = t17 * t243;
  double t253 = t28 * t28;
  double t254 = 0.1e1 / t253;
  double t255 = t254 * t126;
  double t256 = t255 * t141;
  double t258 = t27 * t256 / 0.8e1;
  double t374 = my_piecewise5(t11, 0, t15, 0, -t8 - t244);
  double t377 = my_piecewise3(t21, 0, 0.4e1 / 0.3e1 * t24 * t374);
  double t378 = t6 * t377;
  double t382 = my_piecewise3(t2, 0, -0.3e1 / 0.8e1 * t378 * t142 - t258);
  return t382;
}
DFTFE_DEVICE_NOINLINE double
mgga_x_scan_vrho1__t461(double rho0,
                        double rho1,
                        double sigma0,
                        double sigma1,
                        double sigma2,
                        double tau0,
                        double tau1)
{
  struct mgga_x_scan_params
  {
    double c1 = 0.667;
    double c2 = 0.8;
    double d  = 1.24;
    double k1 = 0.065;
  } params;
  double t7   = rho0 + rho1;
  double t28  = POW_1_3(t7);
  double t29  = M_CBRT6;
  double t30  = M_PI * M_PI;
  double t31  = POW_1_3(t30);
  double t32  = t31 * t31;
  double t33  = 0.1e1 / t32;
  double t34  = t29 * t33;
  double t45  = 0.1e3 / 0.6561e4 / params.k1 - 0.73e2 / 0.648e3;
  double t46  = t29 * t29;
  double t47  = t45 * t46;
  double t48  = t31 * t30;
  double t49  = 0.1e1 / t48;
  double t50  = t47 * t49;
  double t57  = t45 * t29;
  double t66  = sqrt(0.146e3);
  double t67  = t66 * t29;
  double t94  = log(DBL_EPSILON);
  double t97  = t94 / (-t94 + params.c1);
  double t107 = fabs(params.d);
  double t110 = log(DBL_EPSILON / t107);
  double t113 = (-t110 + params.c2) / t110;
  double t128 = sqrt(0.3e1);
  double t129 = 0.1e1 / t31;
  double t130 = t46 * t129;
  double t156 = rho1 * rho1;
  double t157 = POW_1_3(rho1);
  double t158 = t157 * t157;
  double t159 = t158 * t156;
  double t160 = 0.1e1 / t159;
  double t161 = sigma2 * t160;
  double t162 = t34 * t161;
  double t164 = sigma2 * sigma2;
  double t165 = t156 * t156;
  double t166 = t165 * rho1;
  double t168 = 0.1e1 / t157 / t166;
  double t169 = t164 * t168;
  double t170 = t33 * sigma2;
  double t171 = t170 * t160;
  double t174 = exp(-0.27e2 / 0.8e2 * t57 * t171);
  double t180 = t158 * rho1;
  double t181 = 0.1e1 / t180;
  double t187 = 0.5e1 / 0.9e1 * (tau1 * t181 - t161 / 0.8e1) * t29 * t33;
  double t188 = 0.1e1 - t187;
  double t190 = t188 * t188;
  double t192 = exp(-t190 / 0.2e1);
  double t195 = 0.7e1 / 0.1296e5 * t67 * t171 + t66 * t188 * t192 / 0.1e3;
  double t196 = t195 * t195;
  double t197 =
    params.k1 + 0.5e1 / 0.972e3 * t162 + t50 * t169 * t174 / 0.576e3 + t196;
  double t202 = 0.1e1 + params.k1 * (0.1e1 - params.k1 / t197);
  double t203 = t187 <= 0.1e1;
  double t204 = -t97 < t187;
  double t205 = t187 < -t97;
  double t206 = my_piecewise3(t205, t187, -t97);
  double t207 = params.c1 * t206;
  double t208 = 0.1e1 - t206;
  double t209 = 0.1e1 / t208;
  double t211 = exp(-t207 * t209);
  double t212 = my_piecewise3(t204, 0, t211);
  double t213 = t187 < -t113;
  double t214 = my_piecewise3(t213, -t113, t187);
  double t215 = 0.1e1 - t214;
  double t218 = exp(params.c2 / t215);
  double t220 = my_piecewise3(t213, 0, -params.d * t218);
  double t221 = my_piecewise3(t203, t212, t220);
  double t222 = 0.1e1 - t221;
  double t227 = sqrt(sigma2);
  double t228 = t157 * rho1;
  double t229 = 0.1e1 / t228;
  double t231 = t130 * t227 * t229;
  double t232 = sqrt(t231);
  double t236 = exp(-0.98958e1 * t128 / t232);
  double t237 = 0.1e1 - t236;
  double t259 = params.k1 * params.k1;
  double t276 = t45 * t45;
  double t277 = t30 * t30;
  double t278 = 0.1e1 / t277;
  double t279 = t276 * t278;
  double t322 = params.d * params.c2;
  double t391 = t197 * t197;
  double t393 = t259 / t391;
  double t394 = t156 * rho1;
  double t396 = 0.1e1 / t158 / t394;
  double t397 = sigma2 * t396;
  double t400 = t165 * t156;
  double t402 = 0.1e1 / t157 / t400;
  double t407 = t164 * sigma2;
  double t408 = t165 * t165;
  double t409 = t408 * rho1;
  double t410 = 0.1e1 / t409;
  double t421 = -0.5e1 / 0.3e1 * tau1 * t160 + t397 / 0.3e1;
  double t423 = t34 * t192;
  double t426 = t66 * t190;
  double t430 = -0.7e1 / 0.486e4 * t67 * t170 * t396 -
                t66 * t421 * t423 / 0.18e3 + t426 * t421 * t423 / 0.18e3;
  double t433 =
    -0.1e2 / 0.729e3 * t34 * t397 - t50 * t164 * t402 * t174 / 0.108e3 +
    0.3e1 / 0.32e3 * t279 * t407 * t410 * t174 + 0.2e1 * t195 * t430;
  double t434 = t433 * t222;
  double t436 = t421 * t29;
  double t438 = 0.5e1 / 0.9e1 * t436 * t33;
  double t439 = my_piecewise3(t205, t438, 0);
  double t442 = t208 * t208;
  double t443 = 0.1e1 / t442;
  double t444 = t443 * t439;
  double t446 = -params.c1 * t439 * t209 - t207 * t444;
  double t447 = t446 * t211;
  double t448 = my_piecewise3(t204, 0, t447);
  double t449 = t215 * t215;
  double t450 = 0.1e1 / t449;
  double t451 = my_piecewise3(t213, 0, t438);
  double t455 = my_piecewise3(t213, 0, -t322 * t450 * t451 * t218);
  double t456 = my_piecewise3(t203, t448, t455);
  double t459 = t393 * t434 - t202 * t456 + 0.1174e1 * t456;
  double t460 = t28 * t459;
  double t461 = t460 * t237;
  return t461;
}
DFTFE_DEVICE_NOINLINE double
mgga_x_scan_vrho1__t477(double rho0,
                        double rho1,
                        double sigma0,
                        double sigma1,
                        double sigma2,
                        double tau0,
                        double tau1)
{
  struct mgga_x_scan_params
  {
    double c1 = 0.667;
    double c2 = 0.8;
    double d  = 1.24;
    double k1 = 0.065;
  } params;
  double t461 =
    mgga_x_scan_vrho1__t461(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t3   = M_CBRT3;
  double t4   = M_CBRTPI;
  double t6   = t3 / t4;
  double t7   = rho0 + rho1;
  double t8   = 0.1e1 / t7;
  double t11  = 0.2e1 * rho0 * t8 <= ZETA_THRESHOLD_X_SCAN;
  double t12  = ZETA_THRESHOLD_X_SCAN - 0.1e1;
  double t15  = 0.2e1 * rho1 * t8 <= ZETA_THRESHOLD_X_SCAN;
  double t16  = -t12;
  double t17  = rho0 - rho1;
  double t22  = POW_1_3(ZETA_THRESHOLD_X_SCAN);
  double t23  = t22 * ZETA_THRESHOLD_X_SCAN;
  double t28  = POW_1_3(t7);
  double t29  = M_CBRT6;
  double t30  = M_PI * M_PI;
  double t31  = POW_1_3(t30);
  double t32  = t31 * t31;
  double t33  = 0.1e1 / t32;
  double t34  = t29 * t33;
  double t45  = 0.1e3 / 0.6561e4 / params.k1 - 0.73e2 / 0.648e3;
  double t46  = t29 * t29;
  double t47  = t45 * t46;
  double t48  = t31 * t30;
  double t49  = 0.1e1 / t48;
  double t50  = t47 * t49;
  double t57  = t45 * t29;
  double t66  = sqrt(0.146e3);
  double t67  = t66 * t29;
  double t94  = log(DBL_EPSILON);
  double t97  = t94 / (-t94 + params.c1);
  double t107 = fabs(params.d);
  double t110 = log(DBL_EPSILON / t107);
  double t113 = (-t110 + params.c2) / t110;
  double t128 = sqrt(0.3e1);
  double t129 = 0.1e1 / t31;
  double t130 = t46 * t129;
  double t146 = rho1 <= DENS_THRESHOLD_X_SCAN;
  double t147 = -t17;
  double t149 = my_piecewise5(t15, t12, t11, t16, t147 * t8);
  double t150 = 0.1e1 + t149;
  double t151 = t150 <= ZETA_THRESHOLD_X_SCAN;
  double t152 = POW_1_3(t150);
  double t154 = my_piecewise3(t151, t23, t152 * t150);
  double t155 = t6 * t154;
  double t156 = rho1 * rho1;
  double t157 = POW_1_3(rho1);
  double t158 = t157 * t157;
  double t159 = t158 * t156;
  double t160 = 0.1e1 / t159;
  double t161 = sigma2 * t160;
  double t162 = t34 * t161;
  double t164 = sigma2 * sigma2;
  double t165 = t156 * t156;
  double t166 = t165 * rho1;
  double t168 = 0.1e1 / t157 / t166;
  double t169 = t164 * t168;
  double t170 = t33 * sigma2;
  double t171 = t170 * t160;
  double t174 = exp(-0.27e2 / 0.8e2 * t57 * t171);
  double t180 = t158 * rho1;
  double t181 = 0.1e1 / t180;
  double t187 = 0.5e1 / 0.9e1 * (tau1 * t181 - t161 / 0.8e1) * t29 * t33;
  double t188 = 0.1e1 - t187;
  double t190 = t188 * t188;
  double t192 = exp(-t190 / 0.2e1);
  double t195 = 0.7e1 / 0.1296e5 * t67 * t171 + t66 * t188 * t192 / 0.1e3;
  double t196 = t195 * t195;
  double t197 =
    params.k1 + 0.5e1 / 0.972e3 * t162 + t50 * t169 * t174 / 0.576e3 + t196;
  double t202 = 0.1e1 + params.k1 * (0.1e1 - params.k1 / t197);
  double t203 = t187 <= 0.1e1;
  double t204 = -t97 < t187;
  double t205 = t187 < -t97;
  double t206 = my_piecewise3(t205, t187, -t97);
  double t207 = params.c1 * t206;
  double t208 = 0.1e1 - t206;
  double t209 = 0.1e1 / t208;
  double t211 = exp(-t207 * t209);
  double t212 = my_piecewise3(t204, 0, t211);
  double t213 = t187 < -t113;
  double t214 = my_piecewise3(t213, -t113, t187);
  double t215 = 0.1e1 - t214;
  double t218 = exp(params.c2 / t215);
  double t220 = my_piecewise3(t213, 0, -params.d * t218);
  double t221 = my_piecewise3(t203, t212, t220);
  double t222 = 0.1e1 - t221;
  double t225 = t202 * t222 + 0.1174e1 * t221;
  double t226 = t28 * t225;
  double t227 = sqrt(sigma2);
  double t228 = t157 * rho1;
  double t229 = 0.1e1 / t228;
  double t231 = t130 * t227 * t229;
  double t232 = sqrt(t231);
  double t236 = exp(-0.98958e1 * t128 / t232);
  double t237 = 0.1e1 - t236;
  double t238 = t226 * t237;
  double t242 = t7 * t7;
  double t243 = 0.1e1 / t242;
  double t253 = t28 * t28;
  double t254 = 0.1e1 / t253;
  double t338 = pow(0.3e1, 0.1e1 / 0.6e1);
  double t339 = t338 * t338;
  double t340 = t339 * t339;
  double t341 = t340 * t338;
  double t356 = t147 * t243;
  double t365 = t254 * t225;
  double t366 = t365 * t237;
  double t368 = t155 * t366 / 0.8e1;
  double t384 = my_piecewise5(t15, 0, t11, 0, t8 - t356);
  double t387 = my_piecewise3(t151, 0, 0.4e1 / 0.3e1 * t152 * t384);
  double t388 = t6 * t387;
  double t464 = t341 * t154;
  double t466 = 0.1e1 / t232 / t231;
  double t467 = t226 * t466;
  double t468 = t464 * t467;
  double t470 = 0.1e1 / t157 / t156;
  double t473 = t130 * t227 * t470 * t236;
  double t477 = my_piecewise3(t146,
                              0,
                              -0.3e1 / 0.8e1 * t388 * t238 - t368 -
                                0.3e1 / 0.8e1 * t155 * t461 -
                                0.16891736332904387511e1 * t468 * t473);
  return t477;
}
DFTFE_DEVICE_NOINLINE double
mgga_x_scan_vrho1(double rho0,
                  double rho1,
                  double sigma0,
                  double sigma1,
                  double sigma2,
                  double tau0,
                  double tau1)
{
  struct mgga_x_scan_params
  {
    double c1 = 0.667;
    double c2 = 0.8;
    double d  = 1.24;
    double k1 = 0.065;
  } params;
  double t241 =
    mgga_x_scan_vrho1__t241(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t382 =
    mgga_x_scan_vrho1__t382(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t477 =
    mgga_x_scan_vrho1__t477(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
  double t2  = rho0 <= DENS_THRESHOLD_X_SCAN;
  double t3  = M_CBRT3;
  double t4  = M_CBRTPI;
  double t6  = t3 / t4;
  double t7  = rho0 + rho1;
  double t8  = 0.1e1 / t7;
  double t11 = 0.2e1 * rho0 * t8 <= ZETA_THRESHOLD_X_SCAN;
  double t12 = ZETA_THRESHOLD_X_SCAN - 0.1e1;
  double t15 = 0.2e1 * rho1 * t8 <= ZETA_THRESHOLD_X_SCAN;
  double t16 = -t12;
  double t17 = rho0 - rho1;
  double t19 = my_piecewise5(t11, t12, t15, t16, t17 * t8);
  double t20 = 0.1e1 + t19;
  double t21 = t20 <= ZETA_THRESHOLD_X_SCAN;
  double t22 = POW_1_3(ZETA_THRESHOLD_X_SCAN);
  double t23 = t22 * ZETA_THRESHOLD_X_SCAN;
  double t24 = POW_1_3(t20);
  double t26 = my_piecewise3(t21, t23, t24 * t20);
  double t27 = t6 * t26;
  double t28 = POW_1_3(t7);
  double t29 = M_CBRT6;
  double t30 = M_PI * M_PI;
  double t31 = POW_1_3(t30);
  double t32 = t31 * t31;
  double t33 = 0.1e1 / t32;
  double t34 = t29 * t33;
  double t35 = rho0 * rho0;
  double t36 = POW_1_3(rho0);
  double t37 = t36 * t36;
  double t38 = t37 * t35;
  double t39 = 0.1e1 / t38;
  double t40 = sigma0 * t39;
  double t41 = t34 * t40;
  double t45 = 0.1e3 / 0.6561e4 / params.k1 - 0.73e2 / 0.648e3;
  double t46 = t29 * t29;
  double t47 = t45 * t46;
  double t48 = t31 * t30;
  double t49 = 0.1e1 / t48;
  double t50 = t47 * t49;
  double t51 = sigma0 * sigma0;
  double t52 = t35 * t35;
  double t53 = t52 * rho0;
  double t55 = 0.1e1 / t36 / t53;
  double t56 = t51 * t55;
  double t57 = t45 * t29;
  double t58 = t33 * sigma0;
  double t59 = t58 * t39;
  double t62 = exp(-0.27e2 / 0.8e2 * t57 * t59);
  double t66 = sqrt(0.146e3);
  double t67 = t66 * t29;
  double t70 = t37 * rho0;
  double t71 = 0.1e1 / t70;
  double t77 = 0.5e1 / 0.9e1 * (tau0 * t71 - t40 / 0.8e1) * t29 * t33;
  double t78 = 0.1e1 - t77;
  double t80 = t78 * t78;
  double t82 = exp(-t80 / 0.2e1);
  double t85 = 0.7e1 / 0.1296e5 * t67 * t59 + t66 * t78 * t82 / 0.1e3;
  double t86 = t85 * t85;
  double t87 =
    params.k1 + 0.5e1 / 0.972e3 * t41 + t50 * t56 * t62 / 0.576e3 + t86;
  double t92    = 0.1e1 + params.k1 * (0.1e1 - params.k1 / t87);
  double t93    = t77 <= 0.1e1;
  double t94    = log(DBL_EPSILON);
  double t97    = t94 / (-t94 + params.c1);
  double t98    = -t97 < t77;
  double t99    = t77 < -t97;
  double t100   = my_piecewise3(t99, t77, -t97);
  double t101   = params.c1 * t100;
  double t102   = 0.1e1 - t100;
  double t103   = 0.1e1 / t102;
  double t105   = exp(-t101 * t103);
  double t106   = my_piecewise3(t98, 0, t105);
  double t107   = fabs(params.d);
  double t110   = log(DBL_EPSILON / t107);
  double t113   = (-t110 + params.c2) / t110;
  double t114   = t77 < -t113;
  double t115   = my_piecewise3(t114, -t113, t77);
  double t116   = 0.1e1 - t115;
  double t119   = exp(params.c2 / t116);
  double t121   = my_piecewise3(t114, 0, -params.d * t119);
  double t122   = my_piecewise3(t93, t106, t121);
  double t123   = 0.1e1 - t122;
  double t126   = t92 * t123 + 0.1174e1 * t122;
  double t127   = t28 * t126;
  double t128   = sqrt(0.3e1);
  double t129   = 0.1e1 / t31;
  double t130   = t46 * t129;
  double t131   = sqrt(sigma0);
  double t132   = t36 * rho0;
  double t133   = 0.1e1 / t132;
  double t135   = t130 * t131 * t133;
  double t136   = sqrt(t135);
  double t140   = exp(-0.98958e1 * t128 / t136);
  double t141   = 0.1e1 - t140;
  double t142   = t127 * t141;
  double t145   = my_piecewise3(t2, 0, -0.3e1 / 0.8e1 * t27 * t142);
  double tvrho1 = t145 + t241 + t7 * (t382 + t477);
  return tvrho1;
}
DFTFE_DEVICE_NOINLINE double
mgga_x_scan_vsigma0(double rho0,
                    double rho1,
                    double sigma0,
                    double sigma1,
                    double sigma2,
                    double tau0,
                    double tau1)
{
  MGGA_X_SCAN_VSIGMA0
  return tvsigma0;
}
DFTFE_DEVICE_NOINLINE double
mgga_x_scan_vsigma2(double rho0,
                    double rho1,
                    double sigma0,
                    double sigma1,
                    double sigma2,
                    double tau0,
                    double tau1)
{
  MGGA_X_SCAN_VSIGMA2
  return tvsigma2;
}
DFTFE_DEVICE_NOINLINE double
mgga_x_scan_vtau0(double rho0,
                  double rho1,
                  double sigma0,
                  double sigma1,
                  double sigma2,
                  double tau0,
                  double tau1)
{
  MGGA_X_SCAN_VTAU0
  return tvtau0;
}
DFTFE_DEVICE_NOINLINE double
mgga_x_scan_vtau1(double rho0,
                  double rho1,
                  double sigma0,
                  double sigma1,
                  double sigma2,
                  double tau0,
                  double tau1)
{
  MGGA_X_SCAN_VTAU1
  return tvtau1;
}

#undef MGGA_C_SCAN
#define MGGA_C_SCAN                                                             \
  tzk0   = mgga_c_scan_zk(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);      \
  tvrho0 = mgga_c_scan_vrho0(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);   \
  tvrho1 = mgga_c_scan_vrho1(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);   \
  tvsigma0 =                                                                    \
    mgga_c_scan_vsigma0(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);        \
  tvsigma1 =                                                                    \
    mgga_c_scan_vsigma1(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);        \
  tvsigma2 = tvsigma0;                                                          \
  tvtau0   = mgga_c_scan_vtau0(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1); \
  tvtau1   = mgga_c_scan_vtau1(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);

#undef MGGA_X_SCAN
#define MGGA_X_SCAN                                                           \
  tzk0   = mgga_x_scan_zk(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);    \
  tvrho0 = mgga_x_scan_vrho0(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1); \
  tvrho1 = mgga_x_scan_vrho1(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1); \
  tvsigma0 =                                                                  \
    mgga_x_scan_vsigma0(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);      \
  tvsigma1 = 0.0;                                                             \
  tvsigma2 =                                                                  \
    mgga_x_scan_vsigma2(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);      \
  tvtau0 = mgga_x_scan_vtau0(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1); \
  tvtau1 = mgga_x_scan_vtau1(rho0, rho1, sigma0, sigma1, sigma2, tau0, tau1);
#endif // DFTFE_MGGA_SCAN_DEVICE_HELPERS_H
