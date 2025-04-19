// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2025  The Regents of the University of Michigan and DFT-FE
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
// @author Phani Motamarri
//

#ifndef PERIODICTABLE_H
#define PERIODICTABLE_H

#include <map>
#include <string>
#include <vector>
#include <TypeConfig.h>

namespace dftfe
{
  //
  // Declare pseudoUtils function
  //

  /** @file PeriodicTable.h
   *  @brief stores a map between atomic number and atomic symbol and atomic mass
   *
   *
   *  @author Phani Motamarri
   */
  namespace pseudoUtils
  {
    struct Element
    {
      dftfe::Int  z;
      std::string symbol;
      std::string config;
      double      mass;
      Element(dftfe::Int zz, std::string s, std::string c, double m)
        : z(zz)
        , symbol(s)
        , config(c)
        , mass(m)
      {}
    };

    class PeriodicTable
    {
    private:
      std::vector<Element>              ptable;
      std::map<std::string, dftfe::Int> zmap;

    public:
      PeriodicTable(void);
      dftfe::Int
      z(std::string symbol) const;
      std::string
      symbol(dftfe::Int zval) const;
      std::string
      configuration(dftfe::Int zval) const;
      std::string
      configuration(std::string symbol) const;
      double
      mass(dftfe::Int zval) const;
      double
      mass(std::string symbol) const;
      dftfe::Int
      size(void) const;
    };
  } // namespace pseudoUtils
} // namespace dftfe
#endif
