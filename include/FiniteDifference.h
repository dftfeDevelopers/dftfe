#ifndef DFTFE_FiniteDifference_H
#define DFTFE_FiniteDifference_H

#include <vector>
#include <TypeConfig.h>

namespace dftfe
{
  namespace utils
  {
    class FiniteDifference
    {
    public:
      static std::vector<double>
      getStencilGridOneVariableCentral(const dftfe::uInt totalStencilSize,
                                       const double      h);



      // stencil index is the fastest index in stencilDataAllQuadPoints
      // memory for firstOrderDerivative is assumed to be allocated
      static void
      firstOrderDerivativeOneVariableCentral(
        const dftfe::uInt totalStencilSize,
        const double      h,
        const dftfe::uInt numQuadPoints,
        const double     *stencilDataAllQuadPoints,
        double           *firstOrderDerivative);

      static void
      firstOrderDerivativeOneVariableCentral(
        const dftfe::uInt totalStencilSize,
        const double     *h,
        const dftfe::uInt numQuadPoints,
        const double     *stencilDataAllQuadPoints,
        double           *firstOrderDerivative);


      // stencil index is the fastest index in stencilDataAllQuadPoints
      // memory for secondOrderDerivative is assumed to be allocated
      static void
      secondOrderDerivativeOneVariableCentral(
        const dftfe::uInt totalStencilSize,
        const double      h,
        const dftfe::uInt numQuadPoints,
        const double     *stencilDataAllQuadPoints,
        double           *secondOrderDerivative);

      static void
      secondOrderDerivativeOneVariableCentral(
        const dftfe::uInt totalStencilSize,
        const double     *h,
        const dftfe::uInt numQuadPoints,
        const double     *stencilDataAllQuadPoints,
        double           *secondOrderDerivative);
    };
  } // namespace utils
} // namespace dftfe

#endif // DFTFE_FiniteDifference_H
