#ifndef NNGGA_H
#define NNGGA_H

#include <string>
#include <torch/script.h>

namespace dftfe
{
  class NNGGA
  {
  public:
    NNGGA(std::string modelFileName, const bool isSpinPolarized = false);
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
    std::string                 d_modelFileName;
    torch::jit::script::Module *d_model;
    bool                        d_isSpinPolarized;
  };
} // namespace dftfe

#endif // NNGGA_H
