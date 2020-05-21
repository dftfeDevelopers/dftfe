// --------------------------------------------------------------------------------------
//
// 
// This header file is for upfToxml.cc adapted from upf2qso.C which is a part of Qbox 
// (https://github.com/qboxcode/qbox-public/blob/master/util/upf2qso/src/upf2qso.C)
//  
// Qbox is distributed under the terms of the GNU General Public License
// as published by the Free Software Foundation, either version 2 of
// the License, or (at your option) any later version.
// See the file COPYING in the root directory of this distribution
// or <http://www.gnu.org/licenses/>.
//
// This .h file is added by Phani Motamarri to integrate with DFT-FE code
// -------------------------------------------------------------------------------------
//
//

#ifndef upfToxml_h
#define upfToxml_h
#include <string>
#include "string.h"
#include <headers.h>
#include <dftParameters.h>

namespace dftfe
{
	//
	//Declare pseudoUtils function
	//

	/** @file upfxml.h
	 *  @brief converts pseudopotential file from upf to xml format
	 *
	 *  The functionality reads the upfile and identifies appropriate tags and converts
	 *  into xml file format
	 *
	 *  @author Phani Motamarri
	 */
	namespace pseudoUtils
	{
		/**
		 * @brief read a given upf pseudopotential file name in upf format  and convert to xml format
		 * 
		 * @param inputFile filePath location of the upf file
		 * @param outputFile filePath location of xml file
		 *
		 * @return int errorCode indicating success or failure of conversion
		 */
		int upfToxml(const std::string &inputFile,
				const std::string &outputFile);
	}
}
#endif 
