// ---------------------------------------------------------------------
//
// Copyright (c) 2017 The Regents of the University of Michigan and DFT-FE authors.
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
// @author Sambit Das (2018)
//

#include "../../include/geoOptIon.h"
#include "../../include/cgPRPNonLinearSolver.h"
#include "../../include/force.h"
#include "../../include/meshGenerator.h"
#include "../../include/dft.h"
#include "../../include/fileReaders.h"

namespace geoOptLocal
{
  void writeMesh(std::string meshFileName, const Triangulation<3,3> & triangulation)
 {
      FESystem<3> FE(FE_Q<3>(QGaussLobatto<1>(2)), 1);
      DoFHandler<3> dofHandler; dofHandler.initialize(triangulation,FE);		
      dofHandler.distribute_dofs(FE);
      DataOut<3> data_out;
      data_out.attach_dof_handler(dofHandler);
      data_out.build_patches ();
      std::ofstream output(meshFileName);
      data_out.write_vtu (output);
      //data_out.write_dx(output);
 }
}

//
//constructor
//
template<unsigned int FEOrder>
geoOptIon<FEOrder>::geoOptIon(dftClass<FEOrder>* _dftPtr):
  dftPtr(_dftPtr),
  mpi_communicator (MPI_COMM_WORLD),
  n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_communicator)),
  this_mpi_process (Utilities::MPI::this_mpi_process(mpi_communicator)),
  pcout(std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0))
{

}

//
//
template<unsigned int FEOrder>
void geoOptIon<FEOrder>::init()
{
   const int numberGlobalAtoms=dftPtr->atomLocations.size();
   std::vector<std::vector<int> > tempRelaxFlagsData;
   dftUtils::readRelaxationFlagsFile(3,tempRelaxFlagsData,"relaxationFlags.inp");
   AssertThrow(tempRelaxFlagsData.size()==numberGlobalAtoms,ExcMessage("Incorrect number of entries in relaxationFlags file"));
   d_relaxationFlags.clear();
   for (unsigned int i=0; i< numberGlobalAtoms; ++i)
   {
       for (unsigned int j=0; j< 3; ++j)
          d_relaxationFlags.push_back(tempRelaxFlagsData[i][j]);
   }
   //print relaxation flags
   if (this_mpi_process==0)
   {
     std::cout<<" --------------Ion relaxation flags----------------"<<std::endl;
     for (unsigned int i=0; i< numberGlobalAtoms; ++i)
     {
	 std::cout<<tempRelaxFlagsData[i][0] << "  "<< tempRelaxFlagsData[i][1] << "  "<<tempRelaxFlagsData[i][2]<<std::endl;
     }
     std::cout<<" --------------------------------------------------"<<std::endl;
   }
}

template<unsigned int FEOrder>
void geoOptIon<FEOrder>::run()
{
   //dftPtr->solve();
   std::string meshFileName="mesh_geo0.vtu";//"mesh_geo0.dx"//"mesh_geo0.vtu";
   geoOptLocal::writeMesh(meshFileName,dftPtr->d_mesh.getSerialMesh());

   const double tol=5e-5;//Hatree/Bohr
   const int  maxIter=100;
   const double lineSearchTol=2e-3;//5e-2;
   const int maxLineSearchIter=10;
   int debugLevel=this_mpi_process==0?1:0;
   int maxRestarts=2; int restartCount=0;
   d_totalUpdateCalls=0;
   cgPRPNonLinearSolver cgSolver(tol,maxIter,debugLevel,lineSearchTol,maxLineSearchIter);
   if (this_mpi_process==0)
   {
       std::cout<<" Starting CG Ion Relaxation... "<<std::endl;
       std::cout<<"   ---CG Parameters--------------  "<<std::endl;
       std::cout<<"      stopping tol: "<< tol<<std::endl;
       std::cout<<"      maxIter: "<< maxIter<<std::endl;
       std::cout<<"      lineSearch tol: "<< lineSearchTol<<std::endl;
       std::cout<<"      lineSearch maxIter: "<< maxLineSearchIter<<std::endl;
       std::cout<<"   ------------------------------  "<<std::endl;

   }
   if  (getNumberUnknowns()>0)
   {
       nonLinearSolver::ReturnValueType cgReturn=cgSolver.solve(*this);
       if (cgReturn == nonLinearSolver::RESTART && restartCount<maxRestarts )
       {
	   if (this_mpi_process==0)
	      std::cout<< " ...Restarting CG, restartCount: "<<restartCount<<std::endl;
	   cgReturn=cgSolver.solve(*this); 
	   restartCount++;
       }       
       
       if (cgReturn == nonLinearSolver::SUCCESS )
       {
	   if (this_mpi_process==0)
	      std::cout<< " ...CG Ion Relaxation completed, total number of geometry updates: "<<d_totalUpdateCalls<<std::endl;
       }
       else if (cgReturn == nonLinearSolver::FAILURE)
       {
	   if (this_mpi_process==0)
	     std::cout<< " ...CG Ion Relaxation failed "<<std::endl;

       }
       else if (cgReturn == nonLinearSolver::MAX_ITER_REACHED)
       {
	   if (this_mpi_process==0)
	     std::cout<< " ...Maximum CG iterations reached "<<std::endl;

       }        
       else if (cgReturn == nonLinearSolver::RESTART)
       {
	   if (this_mpi_process==0)
	     std::cout<< " ...Maximum restarts reached "<<std::endl;

       }       

   }
}


template<unsigned int FEOrder>
int geoOptIon<FEOrder>::getNumberUnknowns() const
{
   int count=0;
   for (unsigned int i=0; i< d_relaxationFlags.size(); ++i)
   {
      count+=d_relaxationFlags[i];
   }
   return count;
}



template<unsigned int FEOrder>
double geoOptIon<FEOrder>::value() const
{
}

template<unsigned int FEOrder>
void geoOptIon<FEOrder>::value(std::vector<double> & functionValue)
{
}

template<unsigned int FEOrder>
void geoOptIon<FEOrder>::gradient(std::vector<double> & gradient)
{
   gradient.clear();
   const int numberGlobalAtoms=dftPtr->atomLocations.size();
   std::vector<double> tempGradient= dftPtr->forcePtr->getAtomsForces();
   AssertThrow(tempGradient.size()==numberGlobalAtoms*3,ExcMessage("Atom forces have wrong size"));
   for (unsigned int i=0; i< numberGlobalAtoms; ++i)
   {
      for (unsigned int j=0; j< 3; ++j)
      {
          if (d_relaxationFlags[3*i+j]==1) 
	  {
              gradient.push_back(tempGradient[3*i+j]);
	      if  (this_mpi_process==0)
	         std::cout<<" atomId: "<<i << " direction: "<< j << " force: "<<-tempGradient[3*i+j]<<std::endl;
	  }	  
      }
   }

  d_maximumAtomForceToBeRelaxed=-1.0;

   for (unsigned int i=0;i<gradient.size();++i)
   {
       const double temp=std::sqrt(gradient[i]*gradient[i]);
       if (temp>d_maximumAtomForceToBeRelaxed)
	  d_maximumAtomForceToBeRelaxed=temp;
   }
   
}


template<unsigned int FEOrder>
void geoOptIon<FEOrder>::precondition(std::vector<double>       & s,
			              const std::vector<double> & gradient) const
{
}

template<unsigned int FEOrder>
void geoOptIon<FEOrder>::update(const std::vector<double> & solution)
{
   const int numberGlobalAtoms=dftPtr->atomLocations.size();
   std::vector<Point<3> > globalAtomsDisplacements(numberGlobalAtoms);
   int count=0;
   if (this_mpi_process==0)
      std::cout<<" ----CG atom displacement update-----" << std::endl;
   for (unsigned int i=0; i< numberGlobalAtoms; ++i)
   {
      for (unsigned int j=0; j< 3; ++j)
      {
	  globalAtomsDisplacements[i][j]=0.0;
	  if (this_mpi_process==0)
	  {
            if (d_relaxationFlags[3*i+j]==1) 
	    {
               globalAtomsDisplacements[i][j]=solution[count];
	       if (this_mpi_process==0)
                 std::cout << " atomId: "<<i << " ,direction: "<< j << " ,displacement: "<< solution[count]<<std::endl;
	       count++;
	    }
	  }
      }

      MPI_Bcast(&(globalAtomsDisplacements[i][0]),
	        3,
	        MPI_DOUBLE,
	        0,
	        mpi_communicator);	
   }
   for (unsigned int i=0; i< numberGlobalAtoms; ++i)
   {
       std::cout<< "processor: "<< this_mpi_process << " atomId: "<< i<< " disp: "<<globalAtomsDisplacements[i] << std::endl; 
   }
   if (this_mpi_process==0)
      std::cout<<" -----------------------------" << std::endl;

   if (this_mpi_process==0)
       std::cout<< "  Maximum force to be relaxed: "<<  d_maximumAtomForceToBeRelaxed <<std::endl;
   dftPtr->forcePtr->updateAtomPositionsAndMoveMesh(globalAtomsDisplacements);
   d_totalUpdateCalls+=1;
   //write new mesh
   std::string meshFileName="mesh_geo";
   meshFileName+=std::to_string(d_totalUpdateCalls);
   meshFileName+=".vtu";//".dx";//".vtu";
   geoOptLocal::writeMesh(meshFileName,dftPtr->d_mesh.getSerialMesh());

   dftPtr->solve();
   
}

template<unsigned int FEOrder>
void geoOptIon<FEOrder>::solution(std::vector<double> & solution)
{
}

template<unsigned int FEOrder>
std::vector<int>  geoOptIon<FEOrder>::getUnknownCountFlag() const
{
}


template class geoOptIon<1>;
template class geoOptIon<2>;
template class geoOptIon<3>;
template class geoOptIon<4>;
template class geoOptIon<5>;
template class geoOptIon<6>;
template class geoOptIon<7>;
template class geoOptIon<8>;
template class geoOptIon<9>;
template class geoOptIon<10>;
template class geoOptIon<11>;
template class geoOptIon<12>;
