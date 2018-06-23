// --------------------------------------------------------------------------------------
//
// Copyright (c) 2017-2018 The Regents of the University of Michigan and DFT-FE authors.
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
// -------------------------------------------------------------------------------------
//
// @author Phani Motamarri
//

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <cassert>
#include <cstdlib>
#include <stdexcept>
#include <PeriodicTable.h>
#include <upfToxml.h>
#include <boost/algorithm/string/trim.hpp>

////////////////////////////////////////////////////////////////////////////////
namespace dftfe
{
  namespace pseudoUtils
  {
    std::string find_start_element(std::string name,
				   std::ifstream & upfFile)
    {
      // return the contents of the tag at start of element "name"
      std::string buf, token;
      std::string search_str = "<" + name;
      do
	{
	  upfFile >> token;
	}
      while(!upfFile.eof() && token.find(search_str) == std::string::npos);
      if(upfFile.eof())
	{
	  std::cerr << " EOF reached before start element " << name << std::endl;
	  throw std::invalid_argument(name);
	}

      buf = token;
      if ( buf[buf.length()-1] == '>' )
	return buf;

      // read until ">" is found
      bool found = false;
      char ch;
      do
	{
	  upfFile.get(ch);
	  found = ch == '>';
	  buf += ch;
	}
      while ( !upfFile.eof() && !found );
      if ( upfFile.eof() )
	{
	  std::cerr << " EOF reached before > " << name << std::endl;
	  throw std::invalid_argument(name);
	}
      return buf;
    }

    ////////////////////////////////////////////////////////////////////////////////
    void find_end_element(std::string name,
			  std::ifstream & upfFile)
    {
      std::string buf, token;
      std::string search_str = "</" + name + ">";
      do
	{
	  upfFile >> token;
	  if ( token.find(search_str) != std::string::npos ) return;
	}
      while ( !upfFile.eof() );
      std::cerr << " EOF reached before end element " << name << std::endl;
      throw std::invalid_argument(name);
    }

    ////////////////////////////////////////////////////////////////////////////////
    void seek_str(std::string tag,
		  std::ifstream & upfFile)
    {
      // Read tokens from stdin until tag is found.
      // Throw an exception if tag not found before eof()
      bool done = false;
      std::string token;
      int count = 0;

      do
	{
	  upfFile >> token;
	  if ( token.find(tag) != std::string::npos ) return;
	}
      while ( !upfFile.eof() );

      std::cerr << " EOF reached before " << tag << std::endl;
      throw std::invalid_argument(tag);
    }

    ////////////////////////////////////////////////////////////////////////////////
    std::string get_attr(std::string buf, std::string attr)
    {
      bool done = false;
      //std::string s, search_string = " " + attr + "=";
      std::string s, search_string = attr + "=";

      // find attribute name in buf
      std::string::size_type p = buf.find(search_string);
      if(p != std::string::npos )
	{
	  // process attribute
	  std::string::size_type b = buf.find_first_of("\"",p);
	  std::string::size_type e = buf.find_first_of("\"",b+1);
	  if ( b == std::string::npos || e == std::string::npos )
	    {
	      std::cerr << " get_attr: attribute not found: " << attr << std::endl;
	      throw std::invalid_argument(attr);
	    }
	  return buf.substr(b+1,e-b-1);
	}
      else
	{
	  std::cerr << " get_attr: attribute not found: " << attr << std::endl;
	  throw std::invalid_argument(attr);
	}
      return s;
    }

   
    int upfToxml(const std::string & inputFileName, 
		 const std::string & outputFileName)
    {

       dealii::ConditionalOStream   pcout(std::cout, (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0));

      std::ofstream xmlFile(outputFileName.c_str()); 
      std::ifstream upfFile(inputFileName.c_str());
      if(dftParameters::verbosity>=2)
	pcout << "Converting upf pseudopotential file to xml format" <<std::endl;


      PeriodicTable pt;
      std::string buf,s;
      std::istringstream is;

      // determine UPF version
      int upf_version = 0;

      // The first line of the UPF potential file contains either of the following:
      // <PP_INFO>  (for UPF version 1)
      // <UPF version="2.0.1"> (for UPF version 2)
      std::string::size_type p;
      getline(upfFile,buf);
      p = buf.find("<PP_INFO>");
      if( p != std::string::npos )
	upf_version = 1;
      else
	{
	  p = buf.find("<UPF version=\"2.0.1\">");
	  if ( p != std::string::npos )
	    upf_version = 2;
	}
      if(upf_version == 0 )
	{
	  std::cerr << " Format of UPF file not recognized " << std::endl;
	  std::cerr << " First line of file: " << buf << std::endl;
	  return 1;
	}

      //std::cerr << " UPF version: " << upf_version << std::endl;
   
      if(upf_version == 1)
	{
	  std::cerr << " Format of UPF file not recognized " << std::endl;
	  std::cerr << " First line of file: " << buf << std::endl;
	  return 1;
	}
      if(upf_version == 2)
	{
	  // process UPF version 2 potential
	  seek_str("<PP_INFO>",upfFile);
	  std::string upf_pp_info;
	  bool done = false;
	  while (!done)
	    {
	      getline(upfFile,buf);
	      is.clear();
	      is.str(buf);
	      is >> s;
	      done = ( s == "</PP_INFO>" );
	      if ( !done )
		{
		  upf_pp_info += buf + '\n';
		}
	    }

	  // remove all '<' and '>' characters from the PP_INFO field
	  // for XML compatibility
	  p = upf_pp_info.find_first_of("<>");
	  while ( p != std::string::npos )
	    {
	      upf_pp_info[p] = ' ';
	      p = upf_pp_info.find_first_of("<>");
	    }

	  std::string tag = find_start_element("PP_HEADER",upfFile);

	  // get attribute "element"
	  std::string upf_symbol = get_attr(tag,"element");
	  boost::algorithm::trim(upf_symbol);
	  //std::cerr << " upf_symbol: " << upf_symbol << std::endl;

	  // get atomic number and mass
	  const int atomic_number = pt.z(upf_symbol);
	  const double mass = pt.mass(upf_symbol);

	  // check if potential is norm-conserving or semi-local
	  std::string pseudo_type = get_attr(tag,"pseudo_type");
	  //std::cerr << " pseudo_type = " << pseudo_type << std::endl;
	  if ( pseudo_type!="NC" && pseudo_type!="SL" )
	    {
	      std::cerr << " pseudo_type must be NC or SL" << std::endl;
	      return 1;
	    }

	  // NLCC flag
	  std::string upf_nlcc_flag = get_attr(tag,"core_correction");
	  if ( upf_nlcc_flag == "T" )
	    {
	      std::cerr << " Potential includes a non-linear core correction" << std::endl;
	    }
	  //std::cerr << " upf_nlcc_flag = " << upf_nlcc_flag << std::endl;

	  // XC functional (add in description)
	  std::string upf_functional = get_attr(tag,"functional");
	  // add XC functional information to description
	  upf_pp_info += "functional = " + upf_functional + '\n';
	  //std::cerr << " upf_functional = " << upf_functional << std::endl;

	  // valence charge
	  double upf_zval = 0.0;
	  std::string buf = get_attr(tag,"z_valence");
	  is.clear();
	  is.str(buf);
	  is >> upf_zval;
	  //std::cerr << " upf_zval = " << upf_zval << std::endl;

	  // max angular momentum
	  int upf_lmax;
	  buf = get_attr(tag,"l_max");
	  is.clear();
	  is.str(buf);
	  is >> upf_lmax;
	  //std::cerr << " upf_lmax = " << upf_lmax << std::endl;

	  // local angular momentum
	  int upf_llocal;
	  buf = get_attr(tag,"l_local");
	  is.clear();
	  is.str(buf);
	  is >> upf_llocal;
	  //std::cerr << " upf_llocal = " << upf_llocal << std::endl;

	  // number of points in mesh
	  int upf_mesh_size;
	  buf = get_attr(tag,"mesh_size");
	  is.clear();
	  is.str(buf);
	  is >> upf_mesh_size;
	  //std::cerr << " upf_mesh_size = " << upf_mesh_size << std::endl;

	  // number of wavefunctions
	  int upf_nwf;
	  buf = get_attr(tag,"number_of_wfc");
	  is.clear();
	  is.str(buf);
	  is >> upf_nwf;
	  //std::cerr << " upf_nwf = " << upf_nwf << std::endl;

	  // number of projectors
	  int upf_nproj;
	  buf = get_attr(tag,"number_of_proj");
	  is.clear();
	  is.str(buf);
	  is >> upf_nproj;
	  //std::cerr << " upf_nproj = " << upf_nproj << std::endl;

	  std::vector<int> upf_l(upf_nwf);

	  // read mesh
	  find_start_element("PP_MESH",upfFile);
	  find_start_element("PP_R",upfFile);
	  std::vector<double> upf_r(upf_mesh_size);
	  for ( int i = 0; i < upf_mesh_size; i++ )
	    upfFile >> upf_r[i];
	  find_end_element("PP_R",upfFile);
	  find_start_element("PP_RAB",upfFile);
	  std::vector<double> upf_rab(upf_mesh_size);
	  for ( int i = 0; i < upf_mesh_size; i++ )
	    upfFile >> upf_rab[i];
	  find_end_element("PP_RAB",upfFile);
	  find_end_element("PP_MESH",upfFile);
    
	  // NLCC
	  std::vector<double> upf_nlcc;
	  if ( upf_nlcc_flag == "T" )
	    {
	      find_start_element("PP_NLCC",upfFile);
	      upf_nlcc.resize(upf_mesh_size);
	      for ( int i = 0; i < upf_mesh_size; i++ )
		upfFile >> upf_nlcc[i];
	      find_end_element("PP_NLCC",upfFile);
	    }

	  find_start_element("PP_LOCAL",upfFile);
	  std::vector<double> upf_vloc(upf_mesh_size);
	  for ( int i = 0; i < upf_mesh_size; i++ )
	    upfFile >> upf_vloc[i];
	  find_end_element("PP_LOCAL",upfFile);
      
    
	  find_start_element("PP_NONLOCAL",upfFile);
	  std::vector<std::vector<double> > upf_vnl;
	  upf_vnl.resize(upf_nproj);
	  std::vector<int> upf_proj_l(upf_nproj);

	  std::ostringstream os;
	  for(int j = 0; j < upf_nproj; j++)
	    {
	      int index, angular_momentum;
	      os.str("");
	      os << j+1;
	      std::string element_name = "PP_BETA." + os.str();
	      tag = find_start_element(element_name,upfFile);
	      //std::cerr << tag << std::endl;

	      buf = get_attr(tag,"index");
	      is.clear();
	      is.str(buf);
	      is >> index;
	      //std::cerr << " index = " << index << std::endl;

	      buf = get_attr(tag,"angular_momentum");
	      is.clear();
	      is.str(buf);
	      is >> angular_momentum;
	      //std::cerr << " angular_momentum = " << angular_momentum << std::endl;

	      assert(angular_momentum <= upf_lmax);
	      upf_proj_l[index-1] = angular_momentum;

	      upf_vnl[j].resize(upf_mesh_size);
	      for ( int i = 0; i < upf_mesh_size; i++ )
		upfFile >> upf_vnl[j][i];

	      find_end_element(element_name,upfFile);
	    }

	  // compute number of projectors for each l
	  // nproj_l[l] is the number of projectors having angular momentum l
	  std::vector<int> nproj_l(upf_lmax+1);
	  for ( int l = 0; l <= upf_lmax; l++ )
	    {
	      nproj_l[l] = 0;
	      for ( int ip = 0; ip < upf_nproj; ip++ )
		if ( upf_proj_l[ip] == l ) nproj_l[l]++;
	    }

	  tag = find_start_element("PP_DIJ",upfFile);
	  int size;
	  buf = get_attr(tag,"size");
	  is.clear();
	  is.str(buf);
	  is >> size;
	  //std::cerr << "PP_DIJ size = " << size << std::endl;

	  if ( size != upf_nproj*upf_nproj )
	    {
	      std::cerr << " Number of non-zero Dij differs from number of projectors"
			<< std::endl;
	      return 1;
	    }
	  int upf_ndij = size;

	  std::vector<double> upf_d(upf_ndij);
	  for ( int i = 0; i < upf_ndij; i++ )
	    {
	      upfFile >> upf_d[i];
	    }
	  int imax = sqrt(size+1.e-5);
	  assert(imax*imax==size);

	  // Check if Dij has non-diagonal elements
	  // non-diagonal elements are not supported
	  for ( int m = 0; m < imax; m++ )
	    for ( int n = 0; n < imax; n++ )
	      if ( (m != n) && (upf_d[n*imax+m] != 0.0) )
		{
		  std::cerr << " Non-local Dij has off-diagonal elements" << std::endl;
		  std::cerr << " m=" << m << " n=" << n << std::endl;
		  return 1;
		}

	  find_end_element("PP_DIJ",upfFile);

	  find_end_element("PP_NONLOCAL",upfFile);
      
	  find_start_element("PP_RHOATOM",upfFile);
	  std::vector<double> upf_den(upf_mesh_size);
	  for ( int i = 0; i < upf_mesh_size; i++ )
	    upfFile >> upf_den[i];
	  find_end_element("PP_RHOATOM",upfFile);


	  // make table iproj[l] mapping l to iproj
	  // vnl(l) is in vnl[iproj[l]] if iproj[l] > -1
	  // vlocal if iproj[llocal] = -1
	  std::vector<int> iproj(upf_lmax+2);
	  for ( int l = 0; l <= upf_lmax+1; l++ )
	    iproj[l] = -1;
	  for ( int j = 0; j < upf_nproj; j++ )
	    iproj[upf_proj_l[j]] = j;

	  // determine angular momentum of local potential in UPF file
	  // upf_llocal is the angular momentum of the local potential
	  // increase lmax if there are more projectors than wavefunctions
	  int qso_lmax = upf_lmax;
	  if (upf_lmax < upf_llocal)
	    {
	      qso_lmax = upf_lmax+1;
	    }

    

	  if(pseudo_type == "NC" || pseudo_type == "SL")
	    {
	      //std::cerr << " NC potential" << std::endl;
	      // output original data in file upf.dat
	      /*std::ofstream upf("upf.dat");
	      upf << "# vloc" << std::endl;
	      for ( int i = 0; i < upf_vloc.size(); i++ )
		upf << upf_r[i] << " " << upf_vloc[i] << std::endl;
	      upf << std::endl << std::endl;
	      for ( int j = 0; j < upf_nproj; j++ )
		{
		  upf << "# proj j=" << j << std::endl;
		  for ( int i = 0; i < upf_vnl[j].size(); i++ )
		    upf << upf_r[i] << " " << upf_vnl[j][i] << std::endl;
		  upf << std::endl << std::endl;
		}

	      upf << "# dij " << std::endl;
	      for ( int j = 0; j < upf_d.size(); j++ )
		{
		  upf << j << " " << upf_d[j] << std::endl;
		}
		upf.close();*/

	      // print summary
	      /*std::cerr << "PP_INFO:" << std::endl << upf_pp_info << std::endl;
	      std::cerr << "Element: " << upf_symbol << std::endl;
	      std::cerr << "NLCC: " << upf_nlcc_flag << std::endl;
	      // std::cerr << "XC: " << upf_xcf[0] << " " << upf_xcf[1] << " "
	      //      << upf_xcf[2] << " " << upf_xcf[3] << std::endl;
	      std::cerr << "Zv: " << upf_zval << std::endl;
	      std::cerr << "lmax: " << qso_lmax << std::endl;
	      std::cerr << "nproj: " << upf_nproj << std::endl;
	      std::cerr << "mesh_size: " << upf_mesh_size << std::endl;*/

	      // interpolate functions on linear mesh
	      int nplin = upf_mesh_size;
	    
	      // interpolate vloc
	      // factor 0.5: convert from Ry in UPF to Hartree in XML
	      std::vector<double> vloc_lin(nplin);
	      for(int i = 0; i < nplin; i++ )
		{
		  vloc_lin[i] = 0.5 * upf_vloc[i];
		}

	      // interpolate vnl[j], j=0, nproj-1
	      std::vector<std::vector<double> > vnl_lin;
	      vnl_lin.resize(upf_nproj);
	      for(int j = 0; j < vnl_lin.size(); j++ )
		{
		  vnl_lin[j].resize(nplin);
		}

	      if(pseudo_type == "NC")
		{
		  for( int j = 0; j < upf_nproj; j++ )
		    {
		      for( int i = 0; i < nplin; i++ )
			{
			  if(upf_r[i] > 1e-10)
			    vnl_lin[j][i] = upf_vnl[j][i]/upf_r[i];
			  else
			    vnl_lin[j][i] = upf_vnl[j][0];
			}

		    }
		}
	      else
		{
		  for(int j = 0; j < upf_nproj; j++)
		    {
		      for(int i = 0; i < nplin; i++)
			{
			  if(upf_r[i] > 1e-10)
			    vnl_lin[j][i] = upf_vnl[j][i]/(2.0*upf_r[i]);
			  else
			    vnl_lin[j][i] = upf_vnl[j][0];
			}

		    }
		}

	      // write local potential and projectors in gnuplot format on file vlin.dat
	      /*std::ofstream vlin("vlin.dat");
	      vlin << "# vlocal" << std::endl;
	      for ( int i = 0; i < nplin; i++ )
		vlin << vloc_lin[i] << std::endl;
	      vlin << std::endl << std::endl;
	      for ( int iproj = 0; iproj < vnl_lin.size(); iproj++ )
		{
		  vlin << "# projector, l=" << upf_proj_l[iproj] << std::endl;
		  for ( int i = 0; i < nplin; i++ )
		    vlin << upf_r[i] << " " << vnl_lin[iproj][i] << std::endl;
		  vlin << std::endl << std::endl;
		  }*/

	      // Generate XML file

	      // output potential in XMLformat
	      xmlFile << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>" << std::endl;
	      xmlFile << "<fpmd:species xmlns:fpmd=\"http://www.quantum-simulation.org/ns/fpmd/fpmd-1.0\"" << std::endl;
	      xmlFile << "  xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"" << std::endl;
	      xmlFile << "  xsi:schemaLocation=\"http://www.quantum-simulation.org/ns/fpmd/fpmd-1.0"  << std::endl;
	      xmlFile << "  species.xsd\">" << std::endl;
	      xmlFile << "<description>" << std::endl;
	      xmlFile << "Translated from UPF format by upf2qso" << std::endl;
	      xmlFile << upf_pp_info;
	      xmlFile << "</description>" << std::endl;
	      xmlFile << "<symbol>" << upf_symbol << "</symbol>" << std::endl;
	      xmlFile << "<atomic_number>" << atomic_number << "</atomic_number>" << std::endl;
	      xmlFile << "<mass>" << mass << "</mass>" << std::endl;
	      xmlFile << "<norm_conserving_semilocal_pseudopotential>" << std::endl;
	      xmlFile << "<valence_charge>" << upf_zval << "</valence_charge>" << std::endl;
	      //xmlFile << "<mesh_spacing>" << mesh_spacing << "</mesh_spacing>" << std::endl;

	      xmlFile.setf(std::ios::scientific,std::ios::floatfield);
	      

	      // local potential
	      xmlFile << "<local_potential size=\"" << nplin << "\">" << std::endl;
	      int count = 0;
	      for ( int i = 0; i < nplin; i++ ){
		count += 1;
		xmlFile << std::setprecision(14) << vloc_lin[i] << "   ";
		if(count % 4 ==0) xmlFile << std::endl;
	      }
	      xmlFile << "</local_potential>" << std::endl;

	      // projectors
	      int ip = 0;
	      for(int l = 0; l <= upf_lmax; l++ )
		{
		  for(int i = 0; i < nproj_l[l]; i++ )
		    {
		      xmlFile << "<projector l=\"" << l << "\" i=\""
			      << i+1 << "\" size=\"" << nplin << "\">"
			      << std::endl;
		      int count2 = 0;
		      for(int j = 0; j < nplin; j++ )
			{
			  count2 += 1;
			  xmlFile << std::setprecision(14) << vnl_lin[ip][j] << "   ";
			  if(count2 % 4 == 0) xmlFile << std::endl;
			}
		      ip++;
		      xmlFile << "</projector>" << std::endl;
		    }
		}

	      // d_ij
	      if(pseudo_type == "NC")
		{
		  int ibase = 0;
		  int jbase = 0;
		  for(int l = 0; l <= upf_lmax; l++ )
		    {
		      for(int i = 0; i < upf_nproj; i++ )
			for(int j = 0; j < upf_nproj; j++ )
			  {
			    if((upf_proj_l[i] == l) && (upf_proj_l[j] == l))
			      {
				int ij = i + j*upf_nproj;
				xmlFile << "<d_ij l=\"" << l << "\""
					<< " i=\"" << i-ibase+1 << "\" j=\"" << j-jbase+1
					<< "\">" << std::setprecision(14) << 0.5*upf_d[ij] << "</d_ij>"
					<< std::endl;
			      }
			  }
		      ibase += nproj_l[l];
		      jbase += nproj_l[l];
		    }
		}
	      else
		{
		  int ibase = 0;
		  int jbase = 0;
		  for(int l = 0; l <= upf_lmax; l++ )
		    {
		      for(int i = 0; i < upf_nproj; i++ )
			for(int j = 0; j < upf_nproj; j++ )
			  {
			    if((upf_proj_l[i] == l) && (upf_proj_l[j] == l))
			      {
				int ij = i + j*upf_nproj;
				xmlFile << "<d_ij l=\"" << l << "\""
					<< " i=\"" << i-ibase+1 << "\" j=\"" << j-jbase+1
					<< "\">" << std::setprecision(14) << 2.0*upf_d[ij] << "</d_ij>"
					<< std::endl;
			      }
			  }
		      ibase += nproj_l[l];
		      jbase += nproj_l[l];
		    }
		}

	      xmlFile << "<atom_density size=\"" << nplin << "\">" << std::endl;
	      int count3 = 0;
	      for(int i = 0; i < nplin; i++ ){
		count3 += 1;
		xmlFile << std::setprecision(14) << upf_den[i] << "   ";
		if(count3 % 4 == 0) xmlFile << std::endl;
	      }
	      xmlFile << "</atom_density>" << std::endl;


	      xmlFile << "<mesh size=\"" << nplin << "\">" << std::endl;
	      int count4 = 0;
	      for(int i = 0; i < nplin; i++)
		{
		  count4 += 1;
		  xmlFile << std::setprecision(14) << upf_r[i] << "   ";
		  if (count4 % 4 == 0) xmlFile << std::endl;
		}
	      xmlFile << "</mesh>" << std::endl;
	      xmlFile << "</norm_conserving_semilocal_pseudopotential>" << std::endl;
	      xmlFile << "</fpmd:species>" << std::endl;
	    }
	  xmlFile.close(); 
	} // version 1 or 2
      return 0;
    }
  }
}
