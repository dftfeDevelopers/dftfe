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

#include "AtomPseudoWavefunctions.h"
#include "vector"
namespace dftfe
{
  AtomPseudoWavefunctions::AtomPseudoWavefunctions(std::string  filename,
                                                   unsigned int n,
                                                   unsigned int l)
  {
    d_nQuantumNumber = n;
    d_lQuantumNumber = l;
    std::ifstream pspFile(filename);
    double        radValue = 0.0;
    double        orbValue = 0.0;

    std::vector<double> radVec;
    radVec.reserve(5000);
    std::vector<double> orbVec;
    orbVec.reserve(5000);

    unsigned int numMeshSize = 0;
    if (pspFile.is_open())
      {
        while (pspFile.good())
          {
            pspFile >> radValue >> orbValue;
            radVec.push_back(radValue);
            orbVec.push_back(orbValue / radValue);
            numMeshSize++;
          }
      }
    else
      {
        std::cout << " Unable to open " << filename << " file\n";
        AssertThrow(false,
                    dealii::ExcMessage(
                      "Error opening file in AtomPseudoWavefunctions"));
      }

    numMeshSize--; // this is to ensure the last data is not read twice

    d_rMin    = radVec[1];
    orbVec[0] = orbVec[1];
    std::cout << "Value of the Datas at : " << radVec[0] << " is " << orbVec[0]
              << std::endl;
    std::cout << "numMeshSize = : " << numMeshSize << std::endl;
    std::cout << "Value of final Datas at : " << radVec[numMeshSize - 1]
              << " is " << orbVec[numMeshSize - 1] << std::endl;
    alglib::real_1d_array x;
    x.setcontent(numMeshSize, &radVec[0]);
    alglib::real_1d_array y;
    y.setcontent(numMeshSize, &orbVec[0]);
    alglib::ae_int_t natural_bound_typeL = 0;
    alglib::ae_int_t natural_bound_typeR = 1;
    alglib::spline1dbuildcubic(x,
                               y,
                               numMeshSize,
                               natural_bound_typeL,
                               0.0,
                               natural_bound_typeR,
                               0.0,
                               d_radialSplineObject);

    d_cutOff = radVec[numMeshSize - 1];
    pspFile.close();
  }

  double
  AtomPseudoWavefunctions::getRadialValue(double r) const
  {
    if (r >= d_cutOff)
      return 0.0;

    if (r <= d_rMin)
      r = d_rMin;

    double v = alglib::spline1dcalc(d_radialSplineObject, r);

    return v;
  }

  unsigned int
  AtomPseudoWavefunctions::getQuantumNumbern() const
  {
    return d_nQuantumNumber;
  }

  double
  AtomPseudoWavefunctions::getrMinVal() const
  {
    return d_rMin;
  }
} // end of namespace dftfe
