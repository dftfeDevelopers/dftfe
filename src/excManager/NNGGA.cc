#ifdef DFTFE_WITH_TORCH
#  include <torch/script.h>
#  include <NNGGA.h>
#  include <iostream>
#  include <vector>
#  include <algorithm>
#  include <iterator>
#  include <Exceptions.h>

#  define GGA_CS 0.1616204596739955
namespace dftfe
{
  namespace
  {
    struct CastToFloat
    {
      float
      operator()(double value) const
      {
        return static_cast<float>(value);
      }
    };

    struct CastToDouble
    {
      double
      operator()(float value) const
      {
        return static_cast<double>(value);
      }
    };
    template <typename IN1, typename IN2, typename OUT>
    inline void
    interleave(IN1          it1,
               IN1          end1,
               IN2          it2,
               IN2          end2,
               unsigned int nbatch1,
               unsigned int nbatch2,
               OUT          out)
    {
      // interleave until at least one container is done
      while (it1 != end1 && it2 != end2)
        {
          // insert from container 1
          unsigned int ibatch1 = 0;
          while (ibatch1 < nbatch1 && it1 != end1)
            {
              *out = *it1;
              out++;
              it1++;
              ibatch1++;
            }

          // insert from container 2
          unsigned int ibatch2 = 0;
          while (ibatch2 < nbatch2 && it2 != end2)
            {
              *out = *it2;
              out++;
              it2++;
              ibatch2++;
            }
        }

      if (it1 != end1) // check and finish container 1
        {
          std::copy(it1, end1, out);
        }
      if (it2 != end2) // check and finish container 2
        {
          std::copy(it2, end2, out);
        }
    }

    std::string
    trim(const std::string &str, const std::string &whitespace = " \t")
    {
      std::size_t strBegin = str.find_first_not_of(whitespace);
      if (strBegin == std::string::npos)
        return ""; // no content
      std::size_t strEnd   = str.find_last_not_of(whitespace);
      std::size_t strRange = strEnd - strBegin + 1;
      return str.substr(strBegin, strRange);
    }

    std::map<std::string, std::string>
    getKeyValuePairs(const std::string filename, const std::string delimiter)
    {
      std::map<std::string, std::string> returnValue;
      std::ifstream                      readFile;
      std::string                        readLine;
      readFile.open(filename.c_str());
      utils::throwException<utils::InvalidArgument>(readFile.is_open(),
                                                    "Unable to open file " +
                                                      filename);
      while (std::getline(readFile, readLine))
        {
          auto pos = readLine.find_last_of(delimiter);
          if (pos != std::string::npos)
            {
              std::string key   = trim(readLine.substr(0, pos));
              std::string value = trim(readLine.substr(pos + 1));
              returnValue[key]  = value;
            }
        }

      readFile.close();
      return returnValue;
    }

    void
    excSpinUnpolarized(
      const double *                       rho,
      const double *                       sigma,
      const unsigned int                   numPoints,
      double *                             exc,
      torch::jit::script::Module *         model,
      const excDensityPositivityCheckTypes densityPositivityCheckType,
      const double                         rhoTol,
      const double                         sThreshold)
    {
      std::vector<double> rhoModified(numPoints, 0.0);
      if (densityPositivityCheckType ==
          excDensityPositivityCheckTypes::EXCEPTION_POSITIVE)
        for (unsigned int i = 0; i < numPoints; ++i)
          {
            std::string errMsg =
              "Negative electron-density encountered during xc evaluations";
            dftfe::utils::throwException(rho[i] > 0, errMsg);
          }
      else if (densityPositivityCheckType ==
               excDensityPositivityCheckTypes::MAKE_POSITIVE)
        for (unsigned int i = 0; i < numPoints; ++i)
          {
            rhoModified[i] =
              std::max(rho[i], 0.0); // rhoTol will be added subsequently
          }
      else
        for (unsigned int i = 0; i < numPoints; ++i)
          {
            rhoModified[i] = rho[i];
          }

      std::vector<double> modGradRhoTotal(numPoints, 0.0);
      for (unsigned int i = 0; i < numPoints; ++i)
        modGradRhoTotal[i] = std::sqrt(sigma[i]);

      for (unsigned int i = 0; i < numPoints; ++i)
        {
          const double rho4By3 = std::pow(rhoModified[i], 4.0 / 3.0);
          const double s       = GGA_CS * modGradRhoTotal[i] / rho4By3;
          if (s < sThreshold)
            {
              modGradRhoTotal[i] = sThreshold * rho4By3 / GGA_CS;
            }
        }

      std::vector<float> rhoFloat(2 * numPoints);
      interleave(&rhoModified[0],
                 &rhoModified[0] + numPoints,
                 &(modGradRhoTotal[0]),
                 &(modGradRhoTotal[0]) + numPoints,
                 1,
                 1,
                 &rhoFloat[0]);
      auto options =
        torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true);
      torch::Tensor rhoTensor =
        torch::from_blob(&rhoFloat[0], {numPoints, 2}, options).clone();
      rhoTensor += rhoTol;
      std::vector<torch::jit::IValue> input(0);
      input.push_back(rhoTensor);
      auto excTensor = model->forward(input).toTensor();
      for (unsigned int i = 0; i < numPoints; ++i)
        exc[i] = static_cast<double>(excTensor[i][0].item<float>()) /
                 (rhoModified[i] + rhoTol);
    }

    void
    excSpinPolarized(
      const double *                       rho,
      const double *                       sigma,
      const unsigned int                   numPoints,
      double *                             exc,
      torch::jit::script::Module *         model,
      const excDensityPositivityCheckTypes densityPositivityCheckType,
      const double                         rhoTol,
      const double                         sThreshold)
    {
      std::vector<double> rhoModified(2 * numPoints, 0.0);
      if (densityPositivityCheckType ==
          excDensityPositivityCheckTypes::EXCEPTION_POSITIVE)
        for (unsigned int i = 0; i < 2 * numPoints; ++i)
          {
            std::string errMsg =
              "Negative electron-density encountered during xc evaluations";
            dftfe::utils::throwException(rho[i] > 0, errMsg);
          }
      else if (densityPositivityCheckType ==
               excDensityPositivityCheckTypes::MAKE_POSITIVE)
        for (unsigned int i = 0; i < 2 * numPoints; ++i)
          {
            rhoModified[i] =
              std::max(rho[i], 0.0); // rhoTol will be added subsequently
          }
      else
        for (unsigned int i = 0; i < 2 * numPoints; ++i)
          {
            rhoModified[i] = rho[i];
          }

      std::vector<double> modGradRhoTotal(numPoints, 0.0);
      for (unsigned int i = 0; i < numPoints; ++i)
        modGradRhoTotal[i] =
          std::sqrt(sigma[3 * i] + 2.0 * sigma[3 * i + 1] + sigma[3 * i + 2]);

      for (unsigned int i = 0; i < numPoints; ++i)
        {
          const double rhoTotal = rhoModified[2 * i] + rhoModified[2 * i + 1];
          const double rho4By3  = std::pow(rhoTotal, 4.0 / 3.0);
          const double s        = GGA_CS * modGradRhoTotal[i] / rho4By3;
          if (s < sThreshold)
            {
              modGradRhoTotal[i] = sThreshold * rho4By3 / GGA_CS;
            }
        }


      std::vector<float> rhoFloat(3 * numPoints);
      interleave(&rhoModified[0],
                 &rhoModified[0] + 2 * numPoints,
                 &(modGradRhoTotal[0]),
                 &(modGradRhoTotal[0]) + numPoints,
                 2,
                 1,
                 &rhoFloat[0]);

      auto options =
        torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true);
      torch::Tensor rhoTensor =
        torch::from_blob(&rhoFloat[0], {numPoints, 3}, options).clone();
      rhoTensor += rhoTol;
      std::vector<torch::jit::IValue> input(0);
      input.push_back(rhoTensor);
      auto excTensor = model->forward(input).toTensor();
      for (unsigned int i = 0; i < numPoints; ++i)
        exc[i] = static_cast<double>(excTensor[i][0].item<float>()) /
                 (rhoModified[2 * i] + rhoModified[2 * i + 1] + 2 * rhoTol);
    }

    void
    dexcSpinUnpolarized(
      const double *                       rho,
      const double *                       sigma,
      const unsigned int                   numPoints,
      double *                             exc,
      double *                             dexc,
      torch::jit::script::Module *         model,
      const excDensityPositivityCheckTypes densityPositivityCheckType,
      const double                         rhoTol,
      const double                         sThreshold)
    {
      std::vector<double> rhoModified(numPoints, 0.0);
      if (densityPositivityCheckType ==
          excDensityPositivityCheckTypes::EXCEPTION_POSITIVE)
        for (unsigned int i = 0; i < numPoints; ++i)
          {
            std::string errMsg =
              "Negative electron-density encountered during xc evaluations";
            dftfe::utils::throwException(rho[i] > 0, errMsg);
          }
      else if (densityPositivityCheckType ==
               excDensityPositivityCheckTypes::MAKE_POSITIVE)
        for (unsigned int i = 0; i < numPoints; ++i)
          {
            rhoModified[i] =
              std::max(rho[i], 0.0); // rhoTol will be added subsequently
          }
      else
        for (unsigned int i = 0; i < numPoints; ++i)
          {
            rhoModified[i] = rho[i];
          }

      std::vector<double> modGradRhoTotal(numPoints, 0.0);
      for (unsigned int i = 0; i < numPoints; ++i)
        modGradRhoTotal[i] = std::sqrt(sigma[i]);

      for (unsigned int i = 0; i < numPoints; ++i)
        {
          const double rho4By3 = std::pow(rhoModified[i], 4.0 / 3.0);
          const double s       = GGA_CS * modGradRhoTotal[i] / rho4By3;
          if (s < sThreshold)
            {
              modGradRhoTotal[i] = sThreshold * rho4By3 / GGA_CS;
            }
        }


      std::vector<float> rhoFloat(2 * numPoints);
      interleave(&rhoModified[0],
                 &rhoModified[0] + numPoints,
                 &(modGradRhoTotal[0]),
                 &(modGradRhoTotal[0]) + numPoints,
                 1,
                 1,
                 &rhoFloat[0]);

      auto options =
        torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true);
      torch::Tensor rhoTensor =
        torch::from_blob(&rhoFloat[0], {numPoints, 2}, options).clone();
      rhoTensor += rhoTol;
      std::vector<torch::jit::IValue> input(0);
      input.push_back(rhoTensor);
      auto excTensor   = model->forward(input).toTensor();
      auto grad_output = torch::ones_like(excTensor);
      auto vxcTensor   = torch::autograd::grad({excTensor},
                                             {rhoTensor},
                                             /*grad_outputs=*/{grad_output},
                                             /*create_graph=*/true)[0];
      for (unsigned int i = 0; i < numPoints; ++i)
        {
          exc[i] = static_cast<double>(excTensor[i][0].item<float>()) /
                   (rhoModified[i] + rhoTol);
          dexc[2 * i + 0] = static_cast<double>(vxcTensor[i][0].item<float>());
          dexc[2 * i + 1] = static_cast<double>(vxcTensor[i][1].item<float>()) /
                            (2.0 * (modGradRhoTotal[i] + rhoTol));
        }
    }

    void
    dexcSpinPolarized(
      const double *                       rho,
      const double *                       sigma,
      const unsigned int                   numPoints,
      double *                             exc,
      double *                             dexc,
      torch::jit::script::Module *         model,
      const excDensityPositivityCheckTypes densityPositivityCheckType,
      const double                         rhoTol,
      const double                         sThreshold)
    {
      std::vector<double> rhoModified(2 * numPoints, 0.0);
      if (densityPositivityCheckType ==
          excDensityPositivityCheckTypes::EXCEPTION_POSITIVE)
        for (unsigned int i = 0; i < 2 * numPoints; ++i)
          {
            std::string errMsg =
              "Negative electron-density encountered during xc evaluations";
            dftfe::utils::throwException(rho[i] > 0, errMsg);
          }
      else if (densityPositivityCheckType ==
               excDensityPositivityCheckTypes::MAKE_POSITIVE)
        for (unsigned int i = 0; i < 2 * numPoints; ++i)
          {
            rhoModified[i] =
              std::max(rho[i], 0.0); // rhoTol will be added subsequently
          }
      else
        for (unsigned int i = 0; i < 2 * numPoints; ++i)
          {
            rhoModified[i] = rho[i];
          }

      std::vector<double> modGradRhoTotal(numPoints, 0.0);
      for (unsigned int i = 0; i < numPoints; ++i)
        modGradRhoTotal[i] =
          std::sqrt(sigma[3 * i] + 2.0 * sigma[3 * i + 1] + sigma[3 * i + 2]);

      for (unsigned int i = 0; i < numPoints; ++i)
        {
          const double rhoTotal = rhoModified[2 * i] + rhoModified[2 * i + 1];
          const double rho4By3  = std::pow(rhoTotal, 4.0 / 3.0);
          const double s        = GGA_CS * modGradRhoTotal[i] / rho4By3;
          if (s < sThreshold)
            {
              modGradRhoTotal[i] = sThreshold * rho4By3 / GGA_CS;
            }
        }

      std::vector<float> rhoFloat(3 * numPoints);
      interleave(&rhoModified[0],
                 &rhoModified[0] + 2 * numPoints,
                 &(modGradRhoTotal[0]),
                 &(modGradRhoTotal[0]) + numPoints,
                 2,
                 1,
                 &rhoFloat[0]);

      auto options =
        torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true);
      torch::Tensor rhoTensor =
        torch::from_blob(&rhoFloat[0], {numPoints, 3}, options).clone();
      rhoTensor += rhoTol;
      std::vector<torch::jit::IValue> input(0);
      input.push_back(rhoTensor);
      auto excTensor   = model->forward(input).toTensor();
      auto grad_output = torch::ones_like(excTensor);
      auto vxcTensor   = torch::autograd::grad({excTensor},
                                             {rhoTensor},
                                             /*grad_outputs=*/{grad_output},
                                             /*create_graph=*/true)[0];
      for (unsigned int i = 0; i < numPoints; ++i)
        {
          exc[i] = static_cast<double>(excTensor[i][0].item<float>()) /
                   (rhoModified[2 * i] + rhoModified[2 * i + 1] + 2 * rhoTol);
          dexc[5 * i + 0] = static_cast<double>(vxcTensor[i][0].item<float>());
          dexc[5 * i + 1] = static_cast<double>(vxcTensor[i][1].item<float>());
          dexc[5 * i + 2] = static_cast<double>(vxcTensor[i][2].item<float>()) /
                            (2.0 * (modGradRhoTotal[i] + rhoTol));
          dexc[5 * i + 3] = static_cast<double>(vxcTensor[i][2].item<float>()) /
                            (modGradRhoTotal[i] + rhoTol);
          dexc[5 * i + 4] = static_cast<double>(vxcTensor[i][2].item<float>()) /
                            (2.0 * (modGradRhoTotal[i] + rhoTol));
        }
    }

  } // namespace

  NNGGA::NNGGA(std::string                          modelFilename,
               const bool                           isSpinPolarized /*=false*/,
               const excDensityPositivityCheckTypes densityPositivityCheckType)
    : d_modelFilename(modelFilename)
    , d_isSpinPolarized(isSpinPolarized)
    , d_densityPositivityCheckType(densityPositivityCheckType)
  {
    std::map<std::string, std::string> modelKeyValues =
      getKeyValuePairs(d_modelFilename, "=");

    std::vector<std::string> keysToFind = {"PTC_FILE",
                                           "RHO_TOL",
                                           "S_THRESHOLD"};

    // check if all required keys are found
    for (unsigned int i = 0; i < keysToFind.size(); ++i)
      {
        bool found = false;
        for (auto it = modelKeyValues.begin(); it != modelKeyValues.end(); ++it)
          {
            if (keysToFind[i] == it->first)
              found = true;
          }
        utils::throwException(found,
                              "Unable to find the key " + keysToFind[i] +
                                " in file " + d_modelFilename);
      }

    d_ptcFilename = modelKeyValues["PTC_FILE"];
    d_rhoTol      = std::stod(modelKeyValues["RHO_TOL"]);
    d_sThreshold  = std::stod(modelKeyValues["S_THRESHOLD"]);

    d_model  = new torch::jit::script::Module;
    *d_model = torch::jit::load(d_ptcFilename);
    // Explicitly load model onto CPU, you can use kGPU if you are on Linux
    // and have libtorch version with CUDA support (and a GPU)
    d_model->to(torch::kCPU);
  }

  NNGGA::~NNGGA()
  {
    delete d_model;
  }

  void
  NNGGA::evaluateexc(const double *     rho,
                     const double *     sigma,
                     const unsigned int numPoints,
                     double *           exc)
  {
    if (!d_isSpinPolarized)
      excSpinUnpolarized(rho,
                         sigma,
                         numPoints,
                         exc,
                         d_model,
                         d_densityPositivityCheckType,
                         d_rhoTol,
                         d_sThreshold);
    else
      excSpinPolarized(rho,
                       sigma,
                       numPoints,
                       exc,
                       d_model,
                       d_densityPositivityCheckType,
                       d_rhoTol,
                       d_sThreshold);
  }

  void
  NNGGA::evaluatevxc(const double *     rho,
                     const double *     sigma,
                     const unsigned int numPoints,
                     double *           exc,
                     double *           dexc)
  {
    if (!d_isSpinPolarized)
      dexcSpinUnpolarized(rho,
                          sigma,
                          numPoints,
                          exc,
                          dexc,
                          d_model,
                          d_densityPositivityCheckType,
                          d_rhoTol,
                          d_sThreshold);
    else
      dexcSpinPolarized(rho,
                        sigma,
                        numPoints,
                        exc,
                        dexc,
                        d_model,
                        d_densityPositivityCheckType,
                        d_rhoTol,
                        d_sThreshold);
  }

} // namespace dftfe
#endif
