//
//  xmlTodftfeParser.cc
//
//  Created by Shukan Parekh on 5/12/18.
//  Copyright © 2018 Shukan Parekh. All rights reserved.
//

#include <xmlTodftfeParser.h>
#include <fstream>
#include <iomanip>
#include <algorithm>

namespace dftfe{
  //
  //Declare pseudoUtils function
  //
namespace pseudoUtils
{

  //
  //constructor
  //
  xmlTodftfeParser::xmlTodftfeParser()
  {


  }

  //
  //destructor
  //
  xmlTodftfeParser::~xmlTodftfeParser()
  {


  }


bool xmlTodftfeParser::parseFile(std::string & filePath){
    
    std::cout << std::scientific << std::setprecision(18);
     
    doc = xmlReadFile(filePath.c_str(), NULL, 0);
    
    // Check for valid doc
    if(doc == NULL)
        return false;
    
    // Parse doc
    root = xmlDocGetRootElement(doc);
    
    xmlNode *cur_node = root;
    std::stack<xmlNode*> traversal;
    bool traversed = false;
    bool dense_present = false;
    
    while(!traversed){
        if(cur_node != nullptr){
            traversal.push(cur_node);
            cur_node = cur_node->children;
        }else{
            if(!traversal.empty()){
                cur_node = traversal.top();
                traversal.pop();
                
                if(cur_node->type == XML_ELEMENT_NODE){
                    if(xmlStrcmp(cur_node->name, (const xmlChar *)"mesh_spacing") == 0){
                        std::string mesh_spacing_str(reinterpret_cast<char*>(xmlNodeGetContent(cur_node)));
                        mesh_spacing_str.erase(std::remove_if(mesh_spacing_str.begin(), mesh_spacing_str.end(), ::isspace), mesh_spacing_str.end());
                        mesh_spacing = stod(mesh_spacing_str);
                    }else if(xmlStrcmp(cur_node->name, (const xmlChar *)"local_potential") == 0){
                        std::string local_potential_str(reinterpret_cast<char*>(xmlNodeGetContent(cur_node)));
                        
                        std::string::size_type n = 0;
                        while((n = local_potential_str.find( "   ", n)) != std::string::npos){
                            local_potential_str.replace( n, 3, "," );
                            n++;
                        }
                        
                        n = local_potential_str.find("\n,", 0);
                        if(n != std::string::npos)
                            local_potential_str.replace( n, 2, "" );
                        
                        n = 0;
                        while((n = local_potential_str.find( "\n", n)) != std::string::npos){
                            local_potential_str.replace( n, 1, "" );
                            n++;
                        }
                        
                        std::istringstream ss(local_potential_str);
                        std::string token;
                        while(std::getline(ss, token, ','))
                            local_potential.push_back(token);
                        
                    }else if(xmlStrcmp(cur_node->name, (const xmlChar *)"atom_density") == 0){
                        std::string density_str(reinterpret_cast<char*>(xmlNodeGetContent(cur_node)));
                        dense_present = true;
                        if (!dense_present){
                            std::cerr << "Density header not present" << std::endl;
                            break;
                        }
                        std::string::size_type n = 0;
                        while((n = density_str.find( "   ", n)) != std::string::npos){
                            density_str.replace( n, 3, "," );
                            n++;
                        }
                        
                        n = density_str.find("\n,", 0);
                        if(n != std::string::npos)
                            density_str.replace( n, 2, "" );
                        
                        n = 0;
                        while((n = density_str.find( "\n", n)) != std::string::npos){
                            density_str.replace( n, 1, "" );
                            n++;
                        }
                        
                        std::istringstream ss(density_str);
                        std::string token;
                        while(std::getline(ss, token, ','))
                            density.push_back(token);
                        
                    }else if(xmlStrcmp(cur_node->name, (const xmlChar *)"mesh") == 0){
                        std::string mesh_str(reinterpret_cast<char*>(xmlNodeGetContent(cur_node)));
                        std::string::size_type n = 0;
                        while((n = mesh_str.find( "   ", n)) != std::string::npos){
                            mesh_str.replace( n, 3, "," );
                            n++;
                        }

                        n = mesh_str.find("\n,", 0);
                        if(n != std::string::npos)
                            mesh_str.replace( n, 2, "" );

                        n = 0;
                        while((n = mesh_str.find( "\n", n)) != std::string::npos){
                            mesh_str.replace( n, 1, "" );
                            n++;
                        }

                        std::istringstream ss(mesh_str);
                        std::string token;
                        while(std::getline(ss, token, ','))
                            mesh.push_back(token);

                    }else if(xmlStrcmp(cur_node->name, (const xmlChar *)"projector") == 0){
                        std::tuple<size_t, size_t, std::vector<std::string>> projector_temp;
                        xmlAttr* attribute = cur_node->properties;
                        int counter = 0;
                        
                        while(attribute != nullptr){
                            xmlChar* value = xmlNodeListGetString(doc, attribute->children, 1);
                            std::string value_str(reinterpret_cast<char*>(value));
                            
                            if(counter == 0)
                                std::get<0>(projector_temp) = stoi(value_str);
                            else if(counter == 1)
                                std::get<1>(projector_temp) = stoi(value_str);
                            
                            attribute = attribute->next;
                            counter++;
                        }
                        
                        std::string projector_content(reinterpret_cast<char*>(xmlNodeGetContent(cur_node)));
                        
                        std::string::size_type n = 0;
                        while((n = projector_content.find( "   ", n)) != std::string::npos){
                            projector_content.replace( n, 3, "," );
                            n++;
                        }
                        
                        n = projector_content.find("\n,", 0);
                        if(n != std::string::npos)
                            projector_content.replace( n, 2, "" );
                        
                        n = 0;
                        while((n = projector_content.find( "\n", n)) != std::string::npos){
                            projector_content.replace( n, 1, "" );
                            n++;
                        }
                        
                        std::istringstream ss(projector_content);
                        std::string token;
                        while(std::getline(ss, token, ','))
                            std::get<2>(projector_temp).push_back(token);
                        
                        projectors.push_back(projector_temp);
                    }else if(xmlStrcmp(cur_node->name, (const xmlChar *)"d_ij") == 0){
                        std::tuple<size_t, size_t, size_t, double> d_ij_temp;
                        xmlAttr* attribute = cur_node->properties;
                        int counter = 0;
                        
                        while(attribute != nullptr){
                            xmlChar* value = xmlNodeListGetString(doc, attribute->children, 1);
                            std::string value_str(reinterpret_cast<char*>(value));
                            
                            if(counter == 0)
                                std::get<0>(d_ij_temp) = stoi(value_str);
                            else if(counter == 1)
                                std::get<1>(d_ij_temp) = stoi(value_str);
                            else if(counter == 2)
                                std::get<2>(d_ij_temp) = stoi(value_str);
                            
                            attribute = attribute->next;
                            counter++;
                        }
                        
                        std::string d_ij_content_str(reinterpret_cast<char*>(xmlNodeGetContent(cur_node)));
                        std::get<3>(d_ij_temp) = stod(d_ij_content_str);
                        
                        d_ij.push_back(d_ij_temp);
                    }
                }
                
                cur_node = cur_node->next;
            }else
                traversed = true;
        }
    }
    
    return true;
}



bool xmlTodftfeParser::outputData(std::string & baseOutputPath)
{
    // Open filestreams
    std::ofstream loc_pot;
    loc_pot.open(baseOutputPath + "/" + "locPot.dat");
    
    std::ofstream ad_file;
    ad_file.open(baseOutputPath + "/" + "density.inp");
    
    std::ofstream l0;
    l0.open(baseOutputPath + "/" + "proj_l0.dat");
    
    std::ofstream l1;
    l1.open(baseOutputPath + "/" + "proj_l1.dat");
    
    std::ofstream l2;
    l2.open(baseOutputPath + "/" + "proj_l2.dat");
    
    std::ofstream l3;
    l3.open(baseOutputPath + "/" + "proj_l3.dat");
    
    std::ofstream denom;
    denom.open(baseOutputPath + "/" + "denom.dat");
    
    std::ofstream pseudo;
    pseudo.open(baseOutputPath + "/" + "PseudoAtomDat");
   
 
    //set precision
    loc_pot.precision(12);
    ad_file.precision(12);
    l0.precision(12);
    l1.precision(12);
    l2.precision(12);
    l3.precision(12);
    denom.precision(12);
    pseudo.precision(12);

    
    // Set up vectors for the projection values;
    std::vector<std::vector<std::string>> l0_vec;
    std::vector<std::vector<std::string>> l1_vec;
    std::vector<std::vector<std::string>> l2_vec;
    std::vector<std::vector<std::string>> l3_vec;
    
    double r0 = 0.000000000000;
    double r1 = 0.000000000000;
    double r2 = 0.000000000000;
    double r3 = 0.000000000000;
    double r4 = 0.000000000000;
    double r5 = 0.000000000000;
    double rd = 0.000000000000;
    double rp = 0.000000000000;
    //double dr = mesh_spacing;
    
    // Output dijs
    /*for(auto i : d_ij) {
        std::cout<<"d_ij: l: "<<std::get<0>(i)<<", i: "<<std::get<1>(i)<<", j: "<<std::get<2>(i)<<", value: "<<std::get<3>(i)<<std::endl;
	}*/
    // Output projectors
    for(auto i : projectors)
      {
	//std::cout<<"Projector: l: "<<std::get<0>(i)<<", i: "<<std::get<1>(i)<<", value: "<<std::endl;
        
        // Pushing back projector vectors into lx_vec
        if (std::get<0>(i) == 0) l0_vec.push_back(std::get<2>(i));
        else if (std::get<0>(i) == 1) l1_vec.push_back(std::get<2>(i));
        else if (std::get<0>(i) == 2) l2_vec.push_back(std::get<2>(i));
        else l3_vec.push_back(std::get<2>(i));
        
        //for(auto j : std::get<2>(i)) {
	//  std::cout<<j<<std::endl;
        //}
      } 

    // Output local potentials
    //std::cout<<"Potential:"<<std::endl;
    //for(auto i : local_potential){
    //  r0 += dr;
    //}
   
    
    // Output mesh_spacing
    int jl = 0;
    int jd = 0;
    int j0 = 0;
    int j1 = 0;
    int j2 = 0;
    int j3 = 0;

    // Populate the local potential file
    for(auto i : local_potential)
      {
        loc_pot << std::fixed << std::setprecision(14) << mesh[jl]/* rp*/ << " " << i << std::endl;
        jl += 1; 
        //rp += dr;
      }
    
    // Populate the atom density file
    for(auto i : density)
      {
	double di = atof(i.c_str());
	double radius = atof(mesh[jd].c_str());
	if (radius == 0.0)
	  {
	    ad_file << std::fixed << std::setprecision(14) << mesh[jd] << " " << 0.00 << std::endl;
	  }
	else 
	  { 
	    di = di/(4*M_PI*radius*radius);
            ad_file << std::fixed << std::setprecision(14) << mesh[jd] /*rd*/ << " " << di << std::endl;
	  }
	jd += 1;
	//rd += dr;
      }
    
    // Entering values into the projector file
    if(l0_vec.size() > 0) 
      {
        size_t size = l0_vec[0].size();
        for (int i = 0; i < size; ++i) 
	  {
            l0 << mesh[j0] << " ";
            for (int j = 0; j < l0_vec.size(); ++j) {
	      l0 << std::fixed << std::setprecision(14) << l0_vec[j][i] << " ";
            }
            l0 << std::endl;
            //r0 += dr;
	    j0 += 1;
	  }
      }
    
    if(l1_vec.size() > 0) 
      {
        size_t size = l1_vec[0].size();
        for(int i = 0; i < size; ++i) 
	  {
            l1 << mesh[j1] << " ";
            for (int j = 0; j < l1_vec.size(); ++j) 
	      {
                l1 << std::fixed << std::setprecision(14) << l1_vec[j][i] << " ";
	      }
            l1 << std::endl;
            //r2 += dr;
	    j1 += 1;
	  }
      }
    
    if(l2_vec.size() > 0) 
      {
        size_t size = l2_vec[0].size();
        for (int i = 0; i < size; ++i) 
	  {
            l2 << mesh[j2]<< " ";
            for (int j = 0; j < l2_vec.size(); ++j) 
	      {
                l2 << std::fixed << std::setprecision(14) << l2_vec[j][i] << " ";
	      }
            l2 << std::endl;
            //r3 += dr;
	    j2 += 1;
	  }
      }
    
    if(l3_vec.size() > 0) 
      {
        size_t size = l3_vec[0].size();
        for (int i = 0; i < size; ++i) 
	  {
            l3 << mesh[j3] << " ";
            for (int j = 0; j < l3_vec.size(); ++j) 
	      {
                l3 << std::fixed << std::setprecision(14) << l3_vec[j][i]  << " ";
	      }
            l3 << std::endl;
            //r4 += dr;
	    j3 += 1;
	  }
      }
    
    std::string file0 = "proj_l0.dat";
    std::string file1 = "proj_l1.dat";
    std::string file2 = "proj_l2.dat";
    std::string file3 = "proj_l3.dat";
    
    // Populate the PseudoAtomData file
    int lead_no = 0;
    int l0_count = 0;
    int l1_count = 0;
    int l2_count = 0;
    int l3_count = 0;
    
    for(auto i : projectors)
      {
        if (std::get<0>(i) == 0) l0_count += 1;
        else if (std::get<0>(i) == 1) l1_count += 1;
        else if (std::get<0>(i) == 2) l2_count += 1;
        else if (std::get<0>(i) == 3) l3_count += 1;
      }
    //for(auto i : projectors){
    //	std::cout << "projector size" << std::endl;
    //	std::cout << std::get<2>(i).size() << std::endl;
    //}  
    lead_no = (l0_count*1) + (l1_count*3) + (l2_count*5) + (l3_count*7);

    pseudo << lead_no << std::endl;
    
    int serial_no = 0;

    for (int i = 0; i < l0_count; ++i) {
        pseudo << serial_no << " " << "0" << " " << "0" << std::endl;
        serial_no += 1;
    }
    
    
    for (int i = 0; i < l1_count; ++i) {
        int m = -1;
        for (int j = 0; j < 3; ++j) {
            pseudo << serial_no << " " << "1" << " " << m << std::endl;
            m += 1;
        }
        serial_no += 1;
    }
    
    
    for (int i = 0; i < l2_count; ++i) {
        int m = -2;
        for (int j = 0; j < 5; ++j) {
            pseudo << serial_no << " " << "2" << " " << m << std::endl;
            m += 1;
        }
        serial_no += 1;
    }
    
    
    for (int i = 0; i < l3_count; ++i) {
        int m = -3;
        for (int j = 0; j < 7; ++j) {
            pseudo << serial_no << " " << "1" << " " << m << std::endl;
            m += 1;
        }
        serial_no += 1;
    }
    
    if(l0_count > 0) pseudo << file0 << std::endl << l0_count << std::endl;
    if(l1_count > 0) pseudo << file1 << std::endl << l1_count << std::endl;
    if(l2_count > 0) pseudo << file2 << std::endl << l2_count << std::endl;
    if(l3_count > 0) pseudo << file3 << std::endl << l3_count << std::endl;
    
    pseudo << "denom.dat" << std::endl;
    
    //Populate the denom.dat file;
    
    std::vector<int> mult;
    if(l0_count > 0) mult.push_back(l0_count);
    if(l1_count > 0) mult.push_back(l1_count);
    if(l2_count > 0) mult.push_back(l2_count);
    if(l3_count > 0) mult.push_back(l3_count);
    
    int grid_size = 1;
    
    for (int i = 0; i < mult.size(); ++i) 
      {
        grid_size = grid_size*mult[i];
      }
    
    std::vector<double> denom_vec;
    
    // Populate denom_vec
    for(auto i : d_ij) 
      {
        double check = std::get<3>(i);
        if (check != 0.00)
	  {
            denom_vec.push_back(std::get<3>(i));
	  }
      }
    
    //for (int i = 0; i < denom_vec.size(); ++i) {
    //  std::cout << denom_vec[i] << std::endl;
    //}
    
    // Creating the matrix
    int index = 0;
    
    for (int i = 0; i < denom_vec.size(); ++i) 
      {
        for (int j = 0; j < denom_vec.size(); ++j) 
	  {
            if (i == j) 
	      {
                denom << denom_vec[index] << " ";
                //std::cout << denom_vec[index] << " ";
                index += 1;
	      }
            else
	      {
                denom << "0.0000" << " ";
                //std::cout << "0.0000" << " ";
	      }
	  }
        denom << std::endl;
      }
    
    //Clear the vectors
    mult.resize(0);
    denom_vec.resize(0);
    density.resize(0);
    
    
    //projector values
    //for(auto i : projectors) {
    //	std::cout << std::get<2>(i)[0] << std::endl;
    //}    


    // Close filestreams
    loc_pot.close();
    l0.close();
    l1.close();
    l2.close();
    l3.close();
    denom.close();
    pseudo.close();
    ad_file.close();

    // Clear member variables 
    local_potential.clear();
    density.clear();
    mesh.clear();
    projectors.clear();
    d_ij.clear();
    
    
    return false;
}

}

}
