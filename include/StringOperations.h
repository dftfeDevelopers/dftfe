/******************************************************************************
 * Copyright (c) 2021.                                                        *
 * The Regents of the University of Michigan and DFT-EFE developers.          *
 *                                                                            *
 * This file is part of the DFT-EFE code.                                     *
 *                                                                            *
 * DFT-EFE is free software: you can redistribute it and/or modify            *
 *   it under the terms of the Lesser GNU General Public License as           *
 *   published by the Free Software Foundation, either version 3 of           *
 *   the License, or (at your option) any later version.                      *
 *                                                                            *
 * DFT-EFE is distributed in the hope that it will be useful, but             *
 *   WITHOUT ANY WARRANTY; without even the implied warranty                  *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                     *
 *   See the Lesser GNU General Public License for more details.              *
 *                                                                            *
 * You should have received a copy of the GNU Lesser General Public           *
 *   License at the top level of DFT-EFE distribution.  If not, see           *
 *   <https://www.gnu.org/licenses/>.                                         *
 ******************************************************************************/
/*
 * @author Bikash Kanungo
 */

#ifndef DFTFE_STRINGOPERATIONS_H
#define DFTFE_STRINGOPERATIONS_H
namespace dftfe
{
  namespace utils
  {
    namespace stringOps
    {
      bool
      strToInt(const std::string s, int &i);

      bool
      strToDouble(const std::string s, double &x);

      void
      trim(std::string &s);

      std::string
      trimCopy(const std::string &s);

      /*
       *@brief Function to split a string into vector of strings based on a delimiter(s).
       *@param[in] inpStr The string to split
       *@param[in] delimiter A string containing all the characters that are to
       treated as delimiters. Default delimiter is space (" ") Examples:
        1. delimiter = " " will use a space as a delimiter
        2. delimiter = ", " will use both comma, space as a delimiter
        @param[in] skipAdjacentDelimiters Boolean to specify whether to merge
       adjacent delimiters. To elaborate, if set to false, then two adjacent
       delimiters will be treated as containing an empty string which will be
       added to the output. If set to true, adjacent delimiters will be treated
       as a single delimiter Default: true
        @param[in] skipLeadTrailWhiteSpace Boolean to specify whether to skip
       any leading and trailing whitespaces. If set to false, leading and
       trailing whitespaces are treated as non-empty strings and included in the
       output. If set to true, leading and trailing whitespaces are not
       included. Default = true
       *@return A vector of strings obtained by spliting \p inpStr
       */
      std::vector<std::string>
      split(const std::string &inpStr,
            std::string        delimiter               = " ",
            bool               skipAdjacentDelimiters  = true,
            bool               skipLeadTrailWhiteSpace = true);
    } // end of namespace stringOps
  }   // end of namespace utils
} // end of namespace dftfe
#endif // DFTFE_STRINGOPERATIONS_H
