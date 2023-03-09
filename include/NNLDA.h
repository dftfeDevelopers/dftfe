#ifndef NNLDA_H
#define NNLDA_H
#ifdef DFTFE_WITH_TORCH
#  include <string>
#  include <torch/torch.h>
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
#endif
#endif // NNLDA_H
