// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022 The Regents of the University of Michigan and DFT-FE
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
// @author Kartikey Srivastava, Kartick Ramakrishnan
//
#include <iostream>
#include <vector>
#include <libxml/parser.h>
#include <libxml/tree.h>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <algorithm>
#include <iterator>
#include <iomanip>
#include <stdexcept>
#include <cmath>


namespace dftfe
{
  namespace pseudoUtils
  {
    std::vector<double>
    XmlTagReaderMain(std::vector<std::string> tag_name,
                     std::string              file_path_in)
    {
      xmlDocPtr  doc;
      xmlNodePtr cur;
      doc = xmlParseFile(file_path_in.c_str());
      cur = xmlDocGetRootElement(doc);
      // Finding the tag

      for (int i = 0; i < tag_name.size(); i++)
        {
          cur                 = cur->children;
          const xmlChar *temp = (const xmlChar *)tag_name[i].c_str();
          while (cur != NULL)
            {
              if ((!xmlStrcmp(cur->name, temp)))
                {
                  break;
                }
              cur = cur->next;
            }
        }
      // If tag not found
      if (cur == NULL)
        throw std::invalid_argument("Tag not found");
      else
        {
          // Extracting main data
          xmlChar *key;
          key = xmlNodeListGetString(doc, cur->xmlChildrenNode, 1);
          std::string         main_str = (char *)key;
          std::vector<double> main;
          std::stringstream   ss;
          ss << main_str;
          double temp_str;
          while (!ss.eof())
            {
              ss >> temp_str;
              main.push_back(temp_str);
            }
          main.pop_back();
          return main;
        }
    }

    void
    XmlTagReaderAttr(std::vector<std::string>  tag_name,
                     std::string               file_path_in,
                     std::vector<std::string> *attr_type,
                     std::vector<std::string> *attr_value)
    {
      xmlDocPtr  doc;
      xmlNodePtr cur;
      doc = xmlParseFile(file_path_in.c_str());
      cur = xmlDocGetRootElement(doc);

      // Finding the tag
      for (int i = 0; i < tag_name.size(); i++)
        {
          cur                 = cur->children;
          const xmlChar *temp = (const xmlChar *)tag_name[i].c_str();
          while (cur != NULL)
            {
              if ((!xmlStrcmp(cur->name, temp)))
                {
                  break;
                }
              cur = cur->next;
            }
        }

      // If tag not found
      if (cur == NULL)
        throw std::invalid_argument("Tag not found");
      else
        {
          // Extracting Attribute data
          xmlAttr *attribute = cur->properties;
          if (attribute == NULL)
            {
              throw std::invalid_argument("Tag does not have attributes");
            }
          else
            {
              for (xmlAttrPtr attr = cur->properties; NULL != attr;
                   attr            = attr->next)
                {
                  (*attr_type).push_back((char *)(attr->name));
                  xmlChar *value = xmlNodeListGetString(doc, attr->children, 1);
                  (*attr_value).push_back((char *)value);
                }
            }
        }
    }
    int
    xmlNodeChildCount(std::vector<std::string> tag_name,
                      std::string              file_path_in)
    {
      xmlDocPtr  doc;
      xmlNodePtr cur;
      doc = xmlParseFile(file_path_in.c_str());
      cur = xmlDocGetRootElement(doc);

      // Finding the tag
      for (int i = 0; i < tag_name.size(); i++)
        {
          cur                 = cur->children;
          const xmlChar *temp = (const xmlChar *)tag_name[i].c_str();
          while (cur != NULL)
            {
              if ((!xmlStrcmp(cur->name, temp)))
                {
                  break;
                }
              cur = cur->next;
            }
        }
      // Counting children of current node
      int child_count = xmlChildElementCount(cur);
      return child_count;
    }


    //                                                                DFT-FE
    //                                                                Formatting
    //                                                                (UPF file)
    void
    xmltoSummaryFile(std::string file_path_in, std::string file_path_out)
    {
      // List of momentum values
      std::vector<std::string> tag_name_parent;
      tag_name_parent.push_back("PP_NONLOCAL");
      std::vector<int> ang_mom_list;
      for (int i = 1; i < xmlNodeChildCount(tag_name_parent, file_path_in); i++)
        {
          std::string pp_beta_str = "PP_BETA.";
          pp_beta_str += std::to_string(i);
          std::vector<std::string> tag_name;
          tag_name.push_back("PP_NONLOCAL");
          tag_name.push_back(pp_beta_str);
          std::vector<std::string> attr_type;
          std::vector<std::string> attr_value;
          XmlTagReaderAttr(tag_name, file_path_in, &attr_type, &attr_value);
          unsigned int index     = 0;
          std::string  to_search = "angular_momentum";
          auto it = std::find(attr_type.begin(), attr_type.end(), to_search);
          if (it == attr_type.end())
            {
              throw std::invalid_argument(
                "angular momentum attribute not found");
            }
          else
            {
              index = std::distance(attr_type.begin(), it);
              ang_mom_list.push_back(std::stoi(attr_value[index]));
            }
        }
      // Unique angular momentum values
      std::vector<int> ang_mom_unique_list;
      auto             is_unique =
        std::adjacent_find(ang_mom_list.begin(), ang_mom_list.end()) ==
        ang_mom_list.end();
      if (!is_unique)
        {
          ang_mom_unique_list = ang_mom_list;
          std::sort(ang_mom_unique_list.begin(), ang_mom_unique_list.end());
          auto it =
            ang_mom_unique_list.erase(std::unique(ang_mom_unique_list.begin(),
                                                  ang_mom_unique_list.end()));
          ang_mom_unique_list.resize(distance(ang_mom_unique_list.begin(), it));
        }
      else
        {
          ang_mom_unique_list = ang_mom_list;
        }

      // Multiplicity of unique angular momentum values
      std::vector<int> ang_mom_multiplicity_list;
      for (int i = 0; i < ang_mom_unique_list.size(); i++)
        {
          int count = 0;
          for (int j = 0; j < ang_mom_list.size(); j++)
            {
              if (ang_mom_list[j] == ang_mom_unique_list[i])
                {
                  count++;
                }
            }
          ang_mom_multiplicity_list.push_back(count);
        }
      int                           row_index = 0;
      int                           index     = 0;
      std::vector<std::vector<int>> out_proj_arr;
      for (int i = 0; i < ang_mom_unique_list.size(); i++)
        {
          int l = ang_mom_unique_list[i];
          for (int j = 0; j < ang_mom_multiplicity_list[i]; j++)
            {
              int m = -l;
              for (int k = 0; k < 2 * l + 1; k++)
                {
                  out_proj_arr.push_back((std::vector<int>()));
                  out_proj_arr[row_index].push_back(index);
                  out_proj_arr[row_index].push_back(l);
                  out_proj_arr[row_index].push_back(m);
                  m++;
                  row_index++;
                }
              index++;
            }
        }
      // Writing the supplementary

      std::fstream file;
      file_path_out.append("/PseudoAtomDat");
      file.open(file_path_out, std::ios::out);
      if (file.is_open())
        {
          // Total projector
          file << out_proj_arr.size() << std::endl;
          // Projector data
          int m = out_proj_arr.size();
          int n = out_proj_arr[0].size();
          for (int i = 0; i < m; i++)
            {
              for (int j = 0; j < n; j++)
                file << out_proj_arr[i][j] << " ";
              file << std::endl;
            }

          for (int i = 0; i < ang_mom_unique_list.size(); i++)
            {
              std::string proj_str = "proj_l";
              proj_str += std::to_string(ang_mom_unique_list[i]);
              file << proj_str << ".dat" << std::endl;
              file << ang_mom_multiplicity_list[i] << std::endl;
            }
          // Name for D_ij file
          file << "denom.dat" << std::endl;

          // Orbitals
          std::vector<std::string> pswfc_tag;
          pswfc_tag.push_back("PP_PSWFC");
          for (int i = 1; i <= xmlNodeChildCount(pswfc_tag, file_path_in); i++)
            {
              // Reading chi data
              std::string pp_chi_str = "PP_CHI.";
              pp_chi_str += std::to_string(i);
              std::vector<std::string> chi_tag;
              chi_tag.push_back("PP_PSWFC");
              chi_tag.push_back(pp_chi_str);
              std::vector<std::string> attr_type;
              std::vector<std::string> attr_value;
              XmlTagReaderAttr(chi_tag, file_path_in, &attr_type, &attr_value);
              unsigned int index     = 0;
              std::string  to_search = "label";
              auto         it =
                std::find(attr_type.begin(), attr_type.end(), to_search);
              if (it == attr_type.end())
                {
                  throw std::invalid_argument(
                    "orbital label attribute not found");
                }
              else
                {
                  index = std::distance(attr_type.begin(), it);
                }
              std::string orbital_string = attr_value[index];
              for (auto &w : orbital_string)
                {
                  w = tolower(w);
                }
              file << orbital_string + ".dat" << std::endl;
            }
        }
      file.close();
    }

    void
    xmltoProjectorFile(std::string file_path_in, std::string file_path_out)
    {
      // List of momentum values
      std::vector<std::string> tag_name_parent;
      tag_name_parent.push_back("PP_NONLOCAL");
      std::vector<int> ang_mom_list;
      for (int i = 1; i < xmlNodeChildCount(tag_name_parent, file_path_in); i++)
        {
          std::string pp_beta_str = "PP_BETA.";
          pp_beta_str += std::to_string(i);
          std::vector<std::string> tag_name;
          tag_name.push_back("PP_NONLOCAL");
          tag_name.push_back(pp_beta_str);
          std::vector<std::string> attr_type;
          std::vector<std::string> attr_value;
          XmlTagReaderAttr(tag_name, file_path_in, &attr_type, &attr_value);
          unsigned int index     = 0;
          std::string  to_search = "angular_momentum";
          auto it = std::find(attr_type.begin(), attr_type.end(), to_search);
          if (it == attr_type.end())
            {
              throw std::invalid_argument(
                "angular momentum attribute not found");
            }
          else
            {
              index = std::distance(attr_type.begin(), it);
              ang_mom_list.push_back(std::stoi(attr_value[index]));
            }
        }

      // Unique angular momentum values
      std::vector<int> ang_mom_unique_list;
      auto             is_unique =
        std::adjacent_find(ang_mom_list.begin(), ang_mom_list.end()) ==
        ang_mom_list.end();
      if (!is_unique)
        {
          ang_mom_unique_list = ang_mom_list;
          std::sort(ang_mom_unique_list.begin(), ang_mom_unique_list.end());
          auto it =
            ang_mom_unique_list.erase(std::unique(ang_mom_unique_list.begin(),
                                                  ang_mom_unique_list.end()));
          ang_mom_unique_list.resize(distance(ang_mom_unique_list.begin(), it));
        }
      else
        {
          ang_mom_unique_list = ang_mom_list;
        }

      // Multiplicity of unique angular momentum values
      std::vector<int> ang_mom_multiplicity_list;
      for (int i = 0; i < ang_mom_unique_list.size(); i++)
        {
          int count = 0;
          for (int j = 0; j < ang_mom_list.size(); j++)
            {
              if (ang_mom_list[j] == ang_mom_unique_list[i])
                {
                  count++;
                }
            }
        }
      // Beta index for same angular momentum
      std::vector<std::vector<int>> beta_index;
      for (int i = 0; i < ang_mom_unique_list.size(); i++)
        {
          beta_index.push_back((std::vector<int>()));
          for (int j = 0; j < ang_mom_list.size(); j++)
            {
              if (ang_mom_list[j] == ang_mom_unique_list[i])
                {
                  beta_index[i].push_back(j + 1);
                }
            }
        }

      // Extracting radial coordinates
      std::vector<double>      radial_coord;
      std::vector<std::string> radial_tag;
      radial_tag.push_back("PP_MESH");
      radial_tag.push_back("PP_R");
      radial_coord = XmlTagReaderMain(radial_tag, file_path_in);

      // Extracting projector data according to angular momentum
      for (int i = 0; i < ang_mom_unique_list.size(); i++)
        {
          std::vector<std::vector<double>> beta_values;
          std::string                      proj_str = "/proj_l";
          proj_str += std::to_string(ang_mom_unique_list[i]);
          proj_str += ".dat";
          for (int j = 0; j < beta_index[i].size(); j++)
            {
              std::string pp_beta_str = "PP_BETA.";
              pp_beta_str += std::to_string(beta_index[i][j]);
              std::vector<std::string> beta_tag;
              beta_tag.push_back("PP_NONLOCAL");
              beta_tag.push_back(pp_beta_str);
              beta_values.push_back(std::vector<double>());
              beta_values[j] = XmlTagReaderMain(beta_tag, file_path_in);
              std::vector<double> trial =
                XmlTagReaderMain(beta_tag, file_path_in);
            }

          std::fstream file;
          file.open(file_path_out + proj_str, std::ios::out);
          file << std::setprecision(15);
          if (file.is_open())
            {
              for (int l = 0; l < radial_coord.size(); l++)
                {
                  if (l == 0)
                    {
                      file << radial_coord[0] << " ";
                      for (int m = 0; m < beta_values.size(); m++)
                        {
                          if (m != (beta_values.size() - 1))
                            file << beta_values[m][1] / radial_coord[1] << " ";
                          else
                            file << beta_values[m][1] / radial_coord[1]
                                 << std::endl;
                        }
                    }
                  else
                    {
                      file << radial_coord[l] << " ";
                      for (int m = 0; m < beta_values.size(); m++)
                        {
                          if (m != (beta_values.size() - 1))
                            file << beta_values[m][l] / radial_coord[l] << " ";
                          else
                            file << beta_values[m][l] / radial_coord[l]
                                 << std::endl;
                        }
                    }
                }
            }
          file.close();
        }
    }

    void
    xmltoLocalPotential(std::string file_path_in, std::string file_path_out)
    {
      // Extracting radial coordinates
      std::vector<double>      radial_coord;
      std::vector<std::string> radial_tag;
      radial_tag.push_back("PP_MESH");
      radial_tag.push_back("PP_R");
      radial_coord = XmlTagReaderMain(radial_tag, file_path_in);

      // Extracting local potential data
      std::vector<double>      local_pot_values;
      std::vector<std::string> local_por_tag;
      local_por_tag.push_back("PP_LOCAL");
      local_pot_values = XmlTagReaderMain(local_por_tag, file_path_in);

      // Writing the local potential data
      std::fstream file;
      file.open(file_path_out + "/locPot.dat", std::ios::out);
      file << std::setprecision(12);
      if (file.is_open())
        {
          for (int l = 0; l < radial_coord.size(); l++)
            {
              file << radial_coord[l] << " " << local_pot_values[l] / 2
                   << std::endl;
            }
        }
      file.close();
    }

    void
    xmltoDenomFile(std::string file_path_in, std::string file_path_out)
    {
      // Extracting Diagonal Matrix
      std::vector<double>      diagonal_mat;
      std::vector<std::string> diagonal_tag;
      diagonal_tag.push_back("PP_NONLOCAL");
      diagonal_tag.push_back("PP_DIJ");
      diagonal_mat = XmlTagReaderMain(diagonal_tag, file_path_in);

      std::vector<std::string> tag_name_parent;
      tag_name_parent.push_back("PP_NONLOCAL");
      int n = xmlNodeChildCount(tag_name_parent, file_path_in) - 1;

      // Writing the denom.dat
      std::fstream file;
      file.open(file_path_out + "/denom.dat", std::ios::out);
      file << std::setprecision(12);
      if (file.is_open())
        {
          for (int l = 0; l < diagonal_mat.size(); l++)
            {
              if (l != 0 & (l % n == 0))
                file << std::endl;
              file << diagonal_mat[l] / 2 << " ";
            }
        }
      file.close();
    }
    void
    xmltoCoreDensityFile(std::string file_path_in, std::string file_path_out)
    {
      std::vector<std::string> header_tag;
      std::vector<std::string> attr_type;
      std::vector<std::string> attr_value;
      header_tag.push_back("PP_HEADER");
      XmlTagReaderAttr(header_tag, file_path_in, &attr_type, &attr_value);
      unsigned int index     = 0;
      std::string  to_search = "core_correction";
      auto it = std::find(attr_type.begin(), attr_type.end(), to_search);
      if (it == attr_type.end())
        {
          throw std::invalid_argument("core correction attribute not found");
        }
      else
        {
          index = std::distance(attr_type.begin(), it);
        }

      if (attr_value[index] == "T")
        {
          // Extracting radial coordinates
          std::vector<double>      radial_coord;
          std::vector<std::string> radial_tag;
          radial_tag.push_back("PP_MESH");
          radial_tag.push_back("PP_R");
          radial_coord = XmlTagReaderMain(radial_tag, file_path_in);

          // Extracting non local core correction
          std::vector<double>      nlcc_values;
          std::vector<std::string> nlcc_tag;
          nlcc_tag.push_back("PP_NLCC");
          nlcc_values = XmlTagReaderMain(nlcc_tag, file_path_in);

          // Writing coreDensity.inp
          std::fstream file;
          file.open(file_path_out + "/coreDensity.inp", std::ios::out);
          file << std::setprecision(12);
          if (file.is_open())
            {
              for (int l = 0; l < radial_coord.size(); l++)
                {
                  file << radial_coord[l] << " " << nlcc_values[l] << std::endl;
                }
            }
          file.close();
        }
      else
        throw std::invalid_argument("core_correction set false");
    }

    void
    xmltoDensityFile(std::string file_path_in, std::string file_path_out)
    {
      // Extracting radial coordinates
      std::vector<double>      radial_coord;
      std::vector<std::string> radial_tag;
      radial_tag.push_back("PP_MESH");
      radial_tag.push_back("PP_R");
      radial_coord = XmlTagReaderMain(radial_tag, file_path_in);

      // Extracting valence density
      std::vector<double>      rhoatom_values;
      std::vector<std::string> rhoatom_tag;
      rhoatom_tag.push_back("PP_RHOATOM");
      rhoatom_values = XmlTagReaderMain(rhoatom_tag, file_path_in);

      // Writing density.inp
      double       pi = 2 * acos(0.0);
      std::fstream file;
      file.open(file_path_out + "/density.inp", std::ios::out);
      file << std::setprecision(15);
      if (file.is_open())
        {
          for (int l = 0; l < radial_coord.size(); l++)
            {
              if (l == 0)
                file << radial_coord[0] << " " << rhoatom_values[0]
                     << std::endl;
              else
                file << radial_coord[l] << " "
                     << rhoatom_values[l] /
                          (4 * pi * std::pow(radial_coord[l], 2))
                     << std::endl;
            }
        }
      file.close();
    }
    void
    xmltoOrbitalFile(std::string file_path_in, std::string file_path_out)
    {
      // Extracting radial coordinates
      std::vector<double>      radial_coord;
      std::vector<std::string> radial_tag;
      radial_tag.push_back("PP_MESH");
      radial_tag.push_back("PP_R");
      radial_coord = XmlTagReaderMain(radial_tag, file_path_in);

      std::vector<std::string> pswfc_tag;
      pswfc_tag.push_back("PP_PSWFC");
      for (int i = 1; i <= xmlNodeChildCount(pswfc_tag, file_path_in); i++)
        {
          // Reading chi data
          std::string pp_chi_str = "PP_CHI.";
          pp_chi_str += std::to_string(i);
          std::vector<std::string> chi_tag;
          chi_tag.push_back("PP_PSWFC");
          chi_tag.push_back(pp_chi_str);
          std::vector<double> chi_values =
            XmlTagReaderMain(chi_tag, file_path_in);
          std::vector<std::string> attr_type;
          std::vector<std::string> attr_value;
          XmlTagReaderAttr(chi_tag, file_path_in, &attr_type, &attr_value);
          unsigned int index     = 0;
          std::string  to_search = "label";
          auto it = std::find(attr_type.begin(), attr_type.end(), to_search);
          if (it == attr_type.end())
            {
              throw std::invalid_argument("orbital label attribute not found");
            }
          else
            {
              index = std::distance(attr_type.begin(), it);
            }
          std::string orbital_string = attr_value[index];
          for (auto &w : orbital_string)
            {
              w = tolower(w);
            }
          std::fstream file;
          file.open(file_path_out + "/" + orbital_string + ".dat",
                    std::ios::out);
          file << std::setprecision(12);
          if (file.is_open())
            {
              for (int l = 0; l < chi_values.size(); l++)
                {
                  file << radial_coord[l] << " " << chi_values[l] << std::endl;
                }
            }
          file.close();
        }
    }

    int
    pseudoPotentialToDftfeParser(const std::string file_path_in,
                                 const std::string file_path_out,
                                 const int         verbosity,
                                 unsigned int &    nlccFlag,
                                 unsigned int &    socFlag,
                                 unsigned int &    pawFlag)
    {
      xmltoSummaryFile(file_path_in, file_path_out);
      xmltoProjectorFile(file_path_in, file_path_out);
      xmltoLocalPotential(file_path_in, file_path_out);
      xmltoDenomFile(file_path_in, file_path_out);
      xmltoDensityFile(file_path_in, file_path_out);
      xmltoOrbitalFile(file_path_in, file_path_out);


      std::vector<std::string> header_tag;
      std::vector<std::string> attr_type;
      std::vector<std::string> attr_value;
      header_tag.push_back("PP_HEADER");
      XmlTagReaderAttr(header_tag, file_path_in, &attr_type, &attr_value);
      // NLCC
      unsigned int index     = 0;
      std::string  to_search = "core_correction";
      auto it = std::find(attr_type.begin(), attr_type.end(), to_search);
      if (it == attr_type.end())
        {
          throw std::invalid_argument("core correction attribute not found");
        }
      else
        {
          index = std::distance(attr_type.begin(), it);
        }
      if (attr_value[index] == "T")
        {
          nlccFlag = 1;
          xmltoCoreDensityFile(file_path_in, file_path_out);
        }
      else
        nlccFlag = 0;

      // SOC
      to_search = "has_so";
      it        = std::find(attr_type.begin(), attr_type.end(), to_search);
      if (it == attr_type.end())
        {
          throw std::invalid_argument(
            "spin orbit coupling attribute not found");
        }
      else
        {
          index = std::distance(attr_type.begin(), it);
        }
      if (attr_value[index] == "T")
        socFlag = 1;
      else
        socFlag = 0;

      // PAW
      to_search = "is_paw";
      it        = std::find(attr_type.begin(), attr_type.end(), to_search);
      if (it == attr_type.end())
        {
          throw std::invalid_argument("PAW attribute not found");
        }
      else
        {
          index = std::distance(attr_type.begin(), it);
        }
      if (attr_value[index] == "T")
        pawFlag = 1;
      else
        pawFlag = 0;
      return (0);
    }


  } // namespace pseudoUtils
} // namespace dftfe
