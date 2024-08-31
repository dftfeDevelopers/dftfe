#include <iostream>
#include <stdexcept>
#include "AtomicBasisData.h"

namespace dftfe
{
  void
  AtomicBasisData::evalBasisData(const std::vector<double> &quadpts,
                                 const AtomicBasis &        atomicBasis,
                                 const unsigned int         maxDerOrder)
  {
    int nQuad  = quadpts.size() / 3;
    int nBasis = atomicBasis.getNumBasis();

    switch (maxDerOrder)
      {
        case 0:
          d_basisValues = std::vector<double>(nQuad * nBasis, 0.0);

          for (unsigned int iBasis = 0; iBasis < nBasis; iBasis++)
            {
              auto basisVals = atomicBasis.getBasisValue(iBasis, quadpts);

              for (unsigned int iQuad = 0; iQuad < nQuad; iQuad++)
                {
                  d_basisValues[iQuad * nBasis + iBasis] = basisVals[iQuad];
                }
            }
          break;

        case 1:
          d_basisValues     = std::vector<double>(nQuad * nBasis, 0.0);
          d_basisGradValues = std::vector<double>(nQuad * nBasis * 3, 0.0);

          for (unsigned int iBasis = 0; iBasis < nBasis; iBasis++)
            {
              auto basisVals = atomicBasis.getBasisValue(iBasis, quadpts);
              auto basisGradVals =
                atomicBasis.getBasisGradient(iBasis, quadpts);

              for (unsigned int iQuad = 0; iQuad < nQuad; iQuad++)
                {
                  d_basisValues[iQuad * nBasis + iBasis] = basisVals[iQuad];
                  for (unsigned int iDim = 0; iDim < 3; iDim++)
                    {
                      d_basisGradValues[iQuad * nBasis * 3 + iBasis * 3 +
                                        iDim] = basisGradVals[iQuad * 3 + iDim];
                    }
                }
            }
          break;

        case 2:
          d_basisValues          = std::vector<double>(nQuad * nBasis, 0.0);
          d_basisGradValues      = std::vector<double>(nQuad * nBasis * 3, 0.0);
          d_basisLaplacianValues = std::vector<double>(nQuad * nBasis, 0.0);

          for (unsigned int iBasis = 0; iBasis < nBasis; iBasis++)
            {
              auto basisVals = atomicBasis.getBasisValue(iBasis, quadpts);
              auto basisGradVals =
                atomicBasis.getBasisGradient(iBasis, quadpts);
              auto basisLapVals =
                atomicBasis.getBasisLaplacian(iBasis, quadpts);

              for (unsigned int iQuad = 0; iQuad < nQuad; iQuad++)
                {
                  d_basisValues[iQuad * nBasis + iBasis] = basisVals[iQuad];
                  for (unsigned int iDim = 0; iDim < 3; iDim++)
                    {
                      d_basisGradValues[iQuad * nBasis * 3 + iBasis * 3 +
                                        iDim] = basisGradVals[iQuad * 3 + iDim];
                    }
                  d_basisLaplacianValues[iQuad * nBasis + iBasis] =
                    basisLapVals[iQuad];
                }
            }
          break;

        default:
          throw std::runtime_error("\n\n maxDerOrder should be 0, 1 or 2 \n\n");
      }
  }

  const std::vector<double> &
  AtomicBasisData::getBasisValues() const
  {
    return d_basisValues;
  }

  const std::vector<double> &
  AtomicBasisData::getBasisGradValues() const
  {
    return d_basisGradValues;
  }

  const std::vector<double> &
  AtomicBasisData::getBasisLaplacianValues() const
  {
    return d_basisLaplacianValues;
  }

} // namespace dftfe
