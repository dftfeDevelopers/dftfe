// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022  The Regents of the University of Michigan and DFT-FE
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


#include <libxml/parser.h>
#include <libxml/tree.h>
#include <stdio.h>
#include <string.h>

#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stack>
#include <string>
#include <tuple>
#include <vector>

#ifndef xmlTodftfeParser_h
#  define xmlTodftfeParser_h

namespace dftfe
{
  //
  // Declare pseudoUtils function
  //


  namespace pseudoUtils
  {
    /**
     *  @brief converts pseudopotential file from xml format to dftfe format
     *
     *  This class parses the xmlfile and identifies appropriate tags and
     * converts into file formats which can be read by dftfe code
     *
     *  @author Shukan Parekh, Phani Motamarri
     */
    class xmlTodftfeParser
    {
    private:
      xmlDoc *                 doc;
      xmlNode *                root;
      double                   mesh_spacing;
      std::vector<std::string> local_potential;
      std::vector<std::string> density;
      std::vector<std::string> coreDensity;
      std::vector<std::string> mesh;
      std::vector<std::tuple<size_t, size_t, std::vector<std::string>>>
        projectors;
      std::vector<std::tuple<size_t, std::string, std::vector<std::string>>>
        PSwfc;

      std::vector<std::tuple<size_t, size_t, size_t, double>> d_ij;

      std::ofstream loc_pot;
      std::ofstream dense;
      std::ofstream denom;
      std::ofstream l1;
      std::ofstream l2;
      std::ofstream l3;
      std::ofstream ad_file;
      std::ofstream pseudo;

    public:
      /**
       * class constructor
       */
      xmlTodftfeParser();

      /**
       * class destructor
       */
      ~xmlTodftfeParser();

      /**
       * @brief parse a given xml pseudopotential file
       *
       * @param filePath location of the xml file
       */
      bool
      parseFile(const std::string &filePath);

      /**
       * @brief output the parsed xml pseudopotential file into dat files required by dftfe code
       *
       * @param filePath location to write the data
       */
      bool
      outputData(const std::string &filePath);
    };

  } // namespace pseudoUtils

} // namespace dftfe
#endif /* parser_h */
