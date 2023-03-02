#ifndef NNLDA_H
#define NNLDA_H

#include <string>
#include <torch/script.h>

namespace dftfe
{
  class NNLDA
  {
  public:
    NNLDA(std::string modelFileName, const bool isSpinPolarized = false);
    ~NNLDA();
    void
    evaluateexc(const double *rho, const unsigned int numPoints, double *exc);
    void
    evaluatevxc(const double *     rho,
                const unsigned int numPoints,
                double *           exc,
                double *           vxc);

  private:
    std::string                 d_modelFileName;
    torch::jit::script::Module *d_model;
    bool                        d_isSpinPolarized;
  };
} // namespace dftfe

#endif // NNLDA_H
