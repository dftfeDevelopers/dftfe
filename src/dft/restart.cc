// ---------------------------------------------------------------------
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
// ---------------------------------------------------------------------
//
// @author Sambit Das
//

//source file for restart functionality in dftClass

//
//
template<unsigned int FEOrder>
void dftClass<FEOrder>::saveTriaInfoAndRhoData()
{
     pcout<< "Checkpointing tria info and rho data in progress..." << std::endl;
     std::vector<const std::map<dealii::CellId, std::vector<double> > *>  cellQuadDataContainerIn;


     for(auto it = rhoInVals.cbegin(); it != rhoInVals.cend(); it++)
	 cellQuadDataContainerIn.push_back(&(*it));

     for(auto it = rhoOutVals.cbegin(); it != rhoOutVals.cend(); it++)
	 cellQuadDataContainerIn.push_back(&(*it));

     if (dftParameters::xc_id==4)
     {
         for(auto it = gradRhoInVals.cbegin(); it != gradRhoInVals.cend(); it++)
	    cellQuadDataContainerIn.push_back(&(*it));

         for(auto it = gradRhoOutVals.cbegin(); it != gradRhoOutVals.cend(); it++)
	    cellQuadDataContainerIn.push_back(&(*it));
     }

     if(dftParameters::spinPolarized==1)
     {
         for(auto it = rhoInValsSpinPolarized.cbegin(); it != rhoInValsSpinPolarized.cend(); it++)
	    cellQuadDataContainerIn.push_back(&(*it));

         for(auto it = rhoOutValsSpinPolarized.cbegin(); it != rhoOutValsSpinPolarized.cend(); it++)
	    cellQuadDataContainerIn.push_back(&(*it));

     }

     if (dftParameters::xc_id==4 && dftParameters::spinPolarized==1)
     {
         for(auto it = gradRhoInValsSpinPolarized.cbegin(); it != gradRhoInValsSpinPolarized.cend(); it++)
	    cellQuadDataContainerIn.push_back(&(*it));

         for(auto it = gradRhoOutValsSpinPolarized.cbegin(); it != gradRhoOutValsSpinPolarized.cend(); it++)
	    cellQuadDataContainerIn.push_back(&(*it));

     }

     d_mesh.saveTriangulationsCellQuadData(cellQuadDataContainerIn,
	                                   interpoolcomm,
					   interBandGroupComm);

     //write size of current mixing history into an additional .txt file
     const std::string extraInfoFileName="rhoDataExtraInfo.chk";
     if (std::ifstream(extraInfoFileName) && Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
	 dftUtils::moveFile(extraInfoFileName, extraInfoFileName+".old");
     std::ofstream extraInfoFile(extraInfoFileName);
     if (extraInfoFile.is_open())
     {
        extraInfoFile <<rhoInVals.size();
        extraInfoFile.close();
     }

     pcout<< "...checkpointing done." << std::endl;
}

//
//
template<unsigned int FEOrder>
void dftClass<FEOrder>::loadTriaInfoAndRhoData()
{
     pcout<< "Reading tria info and rho data from checkpoint in progress..." << std::endl;
     //read mixing history size of the rhoData to be read in the next step
     unsigned int mixingHistorySize;
     const std::string extraInfoFileName="rhoDataExtraInfo.chk";
     dftUtils::verifyCheckpointFileExists(extraInfoFileName);
     std::ifstream extraInfoFile(extraInfoFileName);
     if (extraInfoFile.is_open())
     {
       extraInfoFile >> mixingHistorySize;
       extraInfoFile.close();
     }
     else AssertThrow(false,ExcMessage("Unable to find rhoDataExtraInfo.txt"));

     Assert(mixingHistorySize>1,ExcInternalError());

     QGauss<3>  quadrature(C_num1DQuad<FEOrder>());
     const unsigned int num_quad_points = quadrature.size();

     //Fill input data for the load function call
     std::vector<unsigned int>  cellDataSizeContainer;
     std::vector<std::map<dealii::CellId, std::vector<double> > > cellQuadDataContainerOut;

     for(unsigned int i=0; i< mixingHistorySize; i++)
     {
	 cellDataSizeContainer.push_back(num_quad_points);
	 cellQuadDataContainerOut.push_back(std::map<dealii::CellId, std::vector<double> >());
     }

     for(unsigned int i=0; i< mixingHistorySize; i++)
     {
	 cellDataSizeContainer.push_back(num_quad_points);
	 cellQuadDataContainerOut.push_back(std::map<dealii::CellId, std::vector<double> >());
     }

     if (dftParameters::xc_id==4)
     {
       for(unsigned int i=0; i< mixingHistorySize; i++)
       {
	 cellDataSizeContainer.push_back(3*num_quad_points);
	 cellQuadDataContainerOut.push_back(std::map<dealii::CellId, std::vector<double> >());
       }
       for(unsigned int i=0; i< mixingHistorySize; i++)
       {
	 cellDataSizeContainer.push_back(3*num_quad_points);
	 cellQuadDataContainerOut.push_back(std::map<dealii::CellId, std::vector<double> >());
       }
     }

     if(dftParameters::spinPolarized==1)
     {
       for(unsigned int i=0; i< mixingHistorySize; i++)
       {
	 cellDataSizeContainer.push_back(2*num_quad_points);
	 cellQuadDataContainerOut.push_back(std::map<dealii::CellId, std::vector<double> >());
       }
       for(unsigned int i=0; i< mixingHistorySize; i++)
       {
	 cellDataSizeContainer.push_back(2*num_quad_points);
	 cellQuadDataContainerOut.push_back(std::map<dealii::CellId, std::vector<double> >());
       }
     }

     if (dftParameters::xc_id==4 && dftParameters::spinPolarized==1)
     {
       for(unsigned int i=0; i< mixingHistorySize; i++)
       {
	 cellDataSizeContainer.push_back(6*num_quad_points);
	 cellQuadDataContainerOut.push_back(std::map<dealii::CellId, std::vector<double> >());
       }
       for(unsigned int i=0; i< mixingHistorySize; i++)
       {
	 cellDataSizeContainer.push_back(6*num_quad_points);
	 cellQuadDataContainerOut.push_back(std::map<dealii::CellId, std::vector<double> >());
       }
     }

     //read rho data from checkpoint file
     d_mesh.loadTriangulationsCellQuadData(cellQuadDataContainerOut,
	                                   cellDataSizeContainer);

     //Fill appropriate data structure using the read rho data
     clearRhoData();
     unsigned int count=0;
     for(unsigned int i=0; i< mixingHistorySize; i++)
     {
	 rhoInVals.push_back(cellQuadDataContainerOut[count]);
	 count++;
     }
     rhoInValues=&(rhoInVals.back());
     for(unsigned int i=0; i< mixingHistorySize; i++)
     {
	 rhoOutVals.push_back(cellQuadDataContainerOut[count]);
	 count++;
     }
     rhoOutValues=&(rhoOutVals.back());

     if (dftParameters::xc_id==4)
     {
	 for(unsigned int i=0; i< mixingHistorySize; i++)
	 {
	     gradRhoInVals.push_back(cellQuadDataContainerOut[count]);
	     count++;
	 }
	 gradRhoInValues= &(gradRhoInVals.back());
	 for(unsigned int i=0; i< mixingHistorySize; i++)
	 {
	     gradRhoOutVals.push_back(cellQuadDataContainerOut[count]);
	     count++;
	 }
	 gradRhoOutValues= &(gradRhoOutVals.back());
     }

     if(dftParameters::spinPolarized==1)
     {
	 for(unsigned int i=0; i< mixingHistorySize; i++)
	 {
	     rhoInValsSpinPolarized.push_back(cellQuadDataContainerOut[count]);
	     count++;
	 }
	 rhoInValuesSpinPolarized=&(rhoInValsSpinPolarized.back());
	 for(unsigned int i=0; i< mixingHistorySize; i++)
	 {
	     rhoOutValsSpinPolarized.push_back(cellQuadDataContainerOut[count]);
	     count++;
	 }
	 rhoOutValuesSpinPolarized=&(rhoOutValsSpinPolarized.back());
     }

     if (dftParameters::xc_id==4 && dftParameters::spinPolarized==1)
     {
	 for(unsigned int i=0; i< mixingHistorySize; i++)
	 {
	     gradRhoInValsSpinPolarized.push_back(cellQuadDataContainerOut[count]);
	     count++;
	 }
	 gradRhoInValuesSpinPolarized=&(gradRhoInValsSpinPolarized.back());
	 for(unsigned int i=0; i< mixingHistorySize; i++)
	 {
	     gradRhoOutValsSpinPolarized.push_back(cellQuadDataContainerOut[count]);
	     count++;
	 }
	 gradRhoOutValuesSpinPolarized=&(gradRhoOutValsSpinPolarized.back());
     }
     pcout<< "...Reading from checkpoint done." << std::endl;
}

template<unsigned int FEOrder>
void dftClass<FEOrder>::writeDomainAndAtomCoordinates() const
{
     dftUtils::writeDataIntoFile(d_domainBoundingVectors,
			        "domainBoundingVectors.chk");

#ifdef USE_COMPLEX
     dftUtils::writeDataIntoFile(atomLocationsFractional,
			        "atomsFracCoord.chk");
#else
    if (dftParameters::periodicX || dftParameters::periodicY || dftParameters::periodicZ)
       dftUtils::writeDataIntoFile(atomLocationsFractional,
			           "atomsFracCoord.chk");
    else
       dftUtils::writeDataIntoFile(atomLocations,
			         "atomsCartCoord.chk");
#endif
}
