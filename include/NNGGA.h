#ifndef NNGGA_H
#define NNGGA_H
#ifdef DFTFE_WITH_TORCH
#  include <string>
#  include <torch/torch.h>
#  include <excDensityPositivityCheckTypes.h>
namespace dftfe
{
  class NNGGA
  {
  public:
    NNGGA(std::string                          modelFileName,
          const bool                           isSpinPolarized = false,
          const excDensityPositivityCheckTypes densityPositivityCheckType =
            excDensityPositivityCheckTypes::MAKE_POSITIVE);
    ~NNGGA();
    void
    evaluateexc(const double *     rho,
                const double *     sigma,
                const unsigned int numPoints,
                double *           exc);
    void
    evaluatevxc(const double *     rho,
                const double *     sigma,
                const unsigned int numPoints,
                double *           exc,
                double *           dexc);

  private:
    std::string                          d_modelFilename;
    std::string                          d_ptcFilename;
    torch::jit::script::Module *         d_model;
    const bool                           d_isSpinPolarized;
    double                         d_rhoTol;
    double                         d_sThreshold;
    const excDensityPositivityCheckTypes d_densityPositivityCheckType;
  };
} // namespace dftfe
#endif
#endif // NNGGA_H
