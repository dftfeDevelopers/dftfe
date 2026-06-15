// ============================================================
// PBE correlation split into per-output __noinline__ device helpers to
// cut register pressure / scratch spill on GPU (mirrors the r2SCAN/SCAN
// decomposition; bit-identical to the monolithic GGA_C_PBE).
//
// NOT a standalone header. It is #included INSIDE `namespace dftfe`
// by exchangeCorrelationFunctionalEvaluatorDevice.cc (and is reusable by
// the CPU evaluator TU). It depends on, and must be included after:
//   - DFTFE_DEVICE_NOINLINE              (qualifier macro)
//   - the monolithic GGA_C_PBE macro     (XCfunctionalDefs/gga_c_pbe.h)
//   - the tzk0/tvrho0/... locals provided by the evaluator BODY macro
//
// Each helper embeds the full GGA_C_PBE and returns a single output; the
// -O3 dead-code eliminator prunes the other outputs' exclusive cones, so
// each helper's live-register peak is bounded by its own output's cone
// rather than the (wider) combined peak of all six outputs.
//
// The trailing GGA_C_PBE redefinition remaps the macro to call the helpers
// and must be in effect before the .def is included.
// ============================================================
#ifndef DFTFE_GGA_PBE_CORRELATION_DEVICE_HELPERS_H
#define DFTFE_GGA_PBE_CORRELATION_DEVICE_HELPERS_H

#include <dftfe/XCfunctionalDefs/xc_params.h>
#include <dftfe/XCfunctionalDefs/gga_c_pbe.h>

DFTFE_DEVICE_NOINLINE double
gga_c_pbe_zk(double rho0,
             double rho1,
             double sigma0,
             double sigma1,
             double sigma2)
{
  double tzk0, tvrho0, tvrho1, tvsigma0, tvsigma1, tvsigma2;
  GGA_C_PBE
  return tzk0;
}

DFTFE_DEVICE_NOINLINE double
gga_c_pbe_vrho0(double rho0,
                double rho1,
                double sigma0,
                double sigma1,
                double sigma2)
{
  double tzk0, tvrho0, tvrho1, tvsigma0, tvsigma1, tvsigma2;
  GGA_C_PBE
  return tvrho0;
}

DFTFE_DEVICE_NOINLINE double
gga_c_pbe_vrho1(double rho0,
                double rho1,
                double sigma0,
                double sigma1,
                double sigma2)
{
  double tzk0, tvrho0, tvrho1, tvsigma0, tvsigma1, tvsigma2;
  GGA_C_PBE
  return tvrho1;
}

DFTFE_DEVICE_NOINLINE double
gga_c_pbe_vsigma0(double rho0,
                  double rho1,
                  double sigma0,
                  double sigma1,
                  double sigma2)
{
  double tzk0, tvrho0, tvrho1, tvsigma0, tvsigma1, tvsigma2;
  GGA_C_PBE
  return tvsigma0;
}

DFTFE_DEVICE_NOINLINE double
gga_c_pbe_vsigma1(double rho0,
                  double rho1,
                  double sigma0,
                  double sigma1,
                  double sigma2)
{
  double tzk0, tvrho0, tvrho1, tvsigma0, tvsigma1, tvsigma2;
  GGA_C_PBE
  return tvsigma1;
}

#undef GGA_C_PBE
#define GGA_C_PBE                                                   \
  tzk0     = gga_c_pbe_zk(rho0, rho1, sigma0, sigma1, sigma2);      \
  tvrho0   = gga_c_pbe_vrho0(rho0, rho1, sigma0, sigma1, sigma2);   \
  tvrho1   = gga_c_pbe_vrho1(rho0, rho1, sigma0, sigma1, sigma2);   \
  tvsigma0 = gga_c_pbe_vsigma0(rho0, rho1, sigma0, sigma1, sigma2); \
  tvsigma1 = gga_c_pbe_vsigma1(rho0, rho1, sigma0, sigma1, sigma2); \
  tvsigma2 = tvsigma0;

#endif // DFTFE_GGA_PBE_CORRELATION_DEVICE_HELPERS_H
