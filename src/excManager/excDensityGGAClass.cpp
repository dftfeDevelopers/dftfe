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
// @author Vishal Subramanian, Sambit Das
//

#include "excDensityGGAClass.h"
#include "NNGGA.h"
#include "Exceptions.h"
#include <dftfeDataTypes.h>
#if defined(DFTFE_WITH_DEVICE)
#  include <DeviceAPICalls.h>
#  include <excManagerDeviceKernels.h>
#endif
#ifdef _OPENMP
#  include <omp.h>
#else
#  define omp_get_thread_num() 0
#endif
namespace dftfe
{
  template <dftfe::utils::MemorySpace memorySpace>
  excDensityGGAClass<memorySpace>::excDensityGGAClass(
    std::vector<std::shared_ptr<xc_func_type>> &funcXPtr,
    std::vector<std::shared_ptr<xc_func_type>> &funcCPtr,
    const dftfe::Int                            numThreads)
    : ExcSSDFunctionalBaseClass<memorySpace>(
        ExcFamilyType::GGA,
        densityFamilyType::GGA,
        std::vector<DensityDescriptorDataAttributes>{
          DensityDescriptorDataAttributes::valuesSpinUp,
          DensityDescriptorDataAttributes::valuesSpinDown,
          DensityDescriptorDataAttributes::gradValuesSpinUp,
          DensityDescriptorDataAttributes::gradValuesSpinDown})
  {
    d_funcXPtr   = funcXPtr;
    d_funcCPtr   = funcCPtr;
    d_NNGGAPtr   = nullptr;
    d_numThreads = numThreads;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  excDensityGGAClass<memorySpace>::excDensityGGAClass(
    std::vector<std::shared_ptr<xc_func_type>> &funcXPtr,
    std::vector<std::shared_ptr<xc_func_type>> &funcCPtr,
    std::string                                 modelXCInputFile,
    const dftfe::Int                            numThreads)
    : ExcSSDFunctionalBaseClass<memorySpace>(
        ExcFamilyType::GGA,
        densityFamilyType::GGA,
        std::vector<DensityDescriptorDataAttributes>{
          DensityDescriptorDataAttributes::valuesSpinUp,
          DensityDescriptorDataAttributes::valuesSpinDown,
          DensityDescriptorDataAttributes::gradValuesSpinUp,
          DensityDescriptorDataAttributes::gradValuesSpinDown})
  {
    d_funcXPtr = funcXPtr;
    d_funcCPtr = funcCPtr;
#ifdef DFTFE_WITH_TORCH
    d_NNGGAPtr = new NNGGA(modelXCInputFile, true);
#endif
    d_numThreads = numThreads;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  excDensityGGAClass<memorySpace>::~excDensityGGAClass()
  {
    if (d_NNGGAPtr != nullptr)
      delete d_NNGGAPtr;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  excDensityGGAClass<memorySpace>::checkInputOutputDataAttributesConsistency(
    const std::vector<xcRemainderOutputDataAttributes> &outputDataAttributes)
    const
  {
    const std::vector<xcRemainderOutputDataAttributes>
      allowedOutputDataAttributes = {
        xcRemainderOutputDataAttributes::e,
        xcRemainderOutputDataAttributes::pdeDensitySpinUp,
        xcRemainderOutputDataAttributes::pdeDensitySpinDown,
        xcRemainderOutputDataAttributes::pdeSigma};

    for (size_t i = 0; i < outputDataAttributes.size(); i++)
      {
        bool isFound = false;
        for (size_t j = 0; j < allowedOutputDataAttributes.size(); j++)
          {
            if (outputDataAttributes[i] == allowedOutputDataAttributes[j])
              isFound = true;
          }


        std::string errMsg =
          "xcRemainderOutputDataAttributes do not matched allowed choices for the family type.";
        dftfe::utils::throwException(isFound, errMsg);
      }
  }



  template <dftfe::utils::MemorySpace memorySpace>
  void
  excDensityGGAClass<memorySpace>::computeRhoTauDependentXCData(
    AuxDensityMatrix<memorySpace>             &auxDensityMatrix,
    const std::pair<dftfe::uInt, dftfe::uInt> &quadIndexRange,
    std::unordered_map<
      xcRemainderOutputDataAttributes,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &xDataOut,
    std::unordered_map<
      xcRemainderOutputDataAttributes,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &cDataOut) const
  {
    // double time1 = MPI_Wtime();
    const dftfe::uInt nquad = quadIndexRange.second - quadIndexRange.first;
    std::vector<xcRemainderOutputDataAttributes> outputDataAttributes;
    for (const auto &element : xDataOut)
      outputDataAttributes.push_back(element.first);

    checkInputOutputDataAttributesConsistency(outputDataAttributes);


    std::unordered_map<
      DensityDescriptorDataAttributes,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      densityDescriptorData;

    for (size_t i = 0; i < this->d_densityDescriptorAttributesList.size(); i++)
      {
        if (this->d_densityDescriptorAttributesList[i] ==
              DensityDescriptorDataAttributes::valuesSpinUp ||
            this->d_densityDescriptorAttributesList[i] ==
              DensityDescriptorDataAttributes::valuesSpinDown)
          densityDescriptorData[this->d_densityDescriptorAttributesList[i]] =
            dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::HOST>(nquad,
                                                                         0);
        else if (this->d_densityDescriptorAttributesList[i] ==
                   DensityDescriptorDataAttributes::gradValuesSpinUp ||
                 this->d_densityDescriptorAttributesList[i] ==
                   DensityDescriptorDataAttributes::gradValuesSpinDown)
          densityDescriptorData[this->d_densityDescriptorAttributesList[i]] =
            dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::HOST>(
              3 * nquad, 0);
      }

    auxDensityMatrix.applyLocalOperations(quadIndexRange,
                                          densityDescriptorData);


    auto &densityValuesSpinUp =
      densityDescriptorData.find(DensityDescriptorDataAttributes::valuesSpinUp)
        ->second;
    auto &densityValuesSpinDown =
      densityDescriptorData
        .find(DensityDescriptorDataAttributes::valuesSpinDown)
        ->second;
    auto &gradValuesSpinUp =
      densityDescriptorData
        .find(DensityDescriptorDataAttributes::gradValuesSpinUp)
        ->second;
    auto &gradValuesSpinDown =
      densityDescriptorData
        .find(DensityDescriptorDataAttributes::gradValuesSpinDown)
        ->second;



    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      densityValues(2 * nquad, 0);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      sigmaValues(3 * nquad, 0);

    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      exValues(nquad, 0);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      ecValues(nquad, 0);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      pdexDensityValuesNonNN(2 * nquad, 0);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      pdecDensityValuesNonNN(2 * nquad, 0);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      pdexDensitySpinUpValues(nquad, 0);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      pdexDensitySpinDownValues(nquad, 0);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      pdecDensitySpinUpValues(nquad, 0);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      pdecDensitySpinDownValues(nquad, 0);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      pdexSigmaValues(3 * nquad, 0);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      pdecSigmaValues(3 * nquad, 0);

    // time1 = MPI_Wtime() - time1;
    // std::cout << "Time taken for auxDensityMatrix.applyLocalOperations: "
    //           << time1 << " seconds" << std::endl;
    // double time2 = MPI_Wtime();            
    dftfe::internal::fillRhoSigmaVector(nquad,
                                        densityValuesSpinUp,
                                        densityValuesSpinDown,
                                        gradValuesSpinUp,
                                        gradValuesSpinDown,
                                        densityValues,
                                        sigmaValues);
    //   time2 = MPI_Wtime() - time2;
    // std::cout << "Time taken for fillRhoSigmaVector: " << time2
    //           << " seconds" << std::endl;  
    //   double time3 = MPI_Wtime();                                        
      std::vector<dftfe::uInt> nsize(d_numThreads,dftfe::uInt(nquad/d_numThreads));
      std::vector<dftfe::uInt> shift(d_numThreads,0);
      dftfe::uInt nRem = nquad - nsize[0] * d_numThreads;
      dftfe::uInt totalShift = 0;
      for (dftfe::uInt i = 0; i < d_numThreads; i++)
        {
          shift[i] = totalShift; 
          if(nRem > 0)
          { nsize[i] ++;
            nRem--;
          }
          totalShift += nsize[i];  

        }
    #pragma omp parallel num_threads(d_numThreads)    
    {
      //std::cout<<"Thread Id and num of Quad Points: "<<omp_get_thread_num()<<" "<<nsize[omp_get_thread_num()]<<std::endl;
    //   std::vector<double> densityValuesLocal(2 * nsize[omp_get_thread_num()], 0.5);
    //   std::vector<double> sigmaValuesLocal(3 * nsize[omp_get_thread_num()], 0.5);
    //   std::vector<double> exValuesLocal(nsize[omp_get_thread_num()], 0.5);
    //   std::vector<double> ecValuesLocal(nsize[omp_get_thread_num()], 0.5);
    //   std::vector<double> pdexDensityValuesNonNNLocal(2 * nsize[omp_get_thread_num()], 0.5);
    //   std::vector<double> pdecDensityValuesNonNNLocal(2 * nsize[omp_get_thread_num()], 0.5);
    //   std::vector<double> pdexSigmaValuesLocal(3 * nsize[omp_get_thread_num()], 0.5);
    //   std::vector<double> pdecSigmaValuesLocal(3 * nsize[omp_get_thread_num()], 0.5);
    //   xc_gga_exc_vxc(d_funcXPtr[omp_get_thread_num()].get(),
    //                nsize[omp_get_thread_num()],
    //                densityValuesLocal.data(),
    //                sigmaValuesLocal.data(),
    //                exValuesLocal.data(),
    //                pdexDensityValuesNonNNLocal.data(),
    //                pdexSigmaValuesLocal.data());
    // xc_gga_exc_vxc(d_funcCPtr[omp_get_thread_num()].get(),
    //                nsize[omp_get_thread_num()],
    //                densityValuesLocal.data(),
    //                sigmaValuesLocal.data(),
    //                ecValuesLocal.data(),
    //                pdecDensityValuesNonNNLocal.data(),
    //                pdecSigmaValuesLocal.data());      
      xc_gga_exc_vxc(d_funcXPtr[omp_get_thread_num()].get(),
                   nsize[omp_get_thread_num()],
                   densityValues.data()+shift[omp_get_thread_num()]*2,
                   sigmaValues.data()+shift[omp_get_thread_num()]*3,
                   exValues.data()+shift[omp_get_thread_num()]*1,
                   pdexDensityValuesNonNN.data()+shift[omp_get_thread_num()]*2,
                   pdexSigmaValues.data()+shift[omp_get_thread_num()]*2);
    xc_gga_exc_vxc(d_funcCPtr[omp_get_thread_num()].get(),
                   nsize[omp_get_thread_num()],
                   densityValues.data()+shift[omp_get_thread_num()]*2,
                   sigmaValues.data()+shift[omp_get_thread_num()]*3,
                   ecValues.data()+shift[omp_get_thread_num()]*1,
                   pdecDensityValuesNonNN.data()+shift[omp_get_thread_num()]*2,
                   pdecSigmaValues.data()+shift[omp_get_thread_num()]*2);
    }
    // time3 = MPI_Wtime() - time3;
    // std::cout << "Time taken for xc_gga_exc_vxc: " << time3
    //           << " seconds" << std::endl;
    // double time4 = MPI_Wtime();          
    for (size_t i = 0; i < nquad; i++)
      {
        exValues[i] =
          exValues[i] * (densityValues[2 * i + 0] + densityValues[2 * i + 1]);
        ecValues[i] =
          ecValues[i] * (densityValues[2 * i + 0] + densityValues[2 * i + 1]);
        pdexDensitySpinUpValues[i]   = pdexDensityValuesNonNN[2 * i + 0];
        pdexDensitySpinDownValues[i] = pdexDensityValuesNonNN[2 * i + 1];
        pdecDensitySpinUpValues[i]   = pdecDensityValuesNonNN[2 * i + 0];
        pdecDensitySpinDownValues[i] = pdecDensityValuesNonNN[2 * i + 1];
      }
    //   time4 = MPI_Wtime() - time4;
    // std::cout << "Time taken for post processing: " << time4
    //           << " seconds" << std::endl;

#ifdef DFTFE_WITH_TORCH
    if (d_NNGGAPtr != nullptr)
      {
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                     excValuesFromNN(nquad, 0);
        const size_t numDescriptors = 5;
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
          pdexcDescriptorValuesFromNN(numDescriptors * nquad, 0);
        d_NNGGAPtr->evaluatevxc(&(densityValues[0]),
                                &sigmaValues[0],
                                nquad,
                                &excValuesFromNN[0],
                                &pdexcDescriptorValuesFromNN[0]);
        for (size_t i = 0; i < nquad; i++)
          {
            exValues[i] += excValuesFromNN[i] * (densityValues[2 * i + 0] +
                                                 densityValues[2 * i + 1]);
            pdexDensitySpinUpValues[i] +=
              pdexcDescriptorValuesFromNN[numDescriptors * i + 0];
            pdexDensitySpinDownValues[i] +=
              pdexcDescriptorValuesFromNN[numDescriptors * i + 1];
            pdexSigmaValues[3 * i + 0] +=
              pdexcDescriptorValuesFromNN[numDescriptors * i + 2];
            pdexSigmaValues[3 * i + 1] +=
              pdexcDescriptorValuesFromNN[numDescriptors * i + 3];
            pdexSigmaValues[3 * i + 2] +=
              pdexcDescriptorValuesFromNN[numDescriptors * i + 4];
          }
      }
#endif

    for (size_t i = 0; i < outputDataAttributes.size(); i++)
      {
        if (outputDataAttributes[i] == xcRemainderOutputDataAttributes::e)
          {
            xDataOut.find(outputDataAttributes[i])->second = exValues;

            cDataOut.find(outputDataAttributes[i])->second = ecValues;
          }
        else if (outputDataAttributes[i] ==
                 xcRemainderOutputDataAttributes::pdeDensitySpinUp)
          {
            xDataOut.find(outputDataAttributes[i])->second =
              pdexDensitySpinUpValues;

            cDataOut.find(outputDataAttributes[i])->second =
              pdecDensitySpinUpValues;
          }
        else if (outputDataAttributes[i] ==
                 xcRemainderOutputDataAttributes::pdeDensitySpinDown)
          {
            xDataOut.find(outputDataAttributes[i])->second =
              pdexDensitySpinDownValues;

            cDataOut.find(outputDataAttributes[i])->second =
              pdecDensitySpinDownValues;
          }
        else if (outputDataAttributes[i] ==
                 xcRemainderOutputDataAttributes::pdeSigma)
          {
            xDataOut.find(outputDataAttributes[i])->second = pdexSigmaValues;

            cDataOut.find(outputDataAttributes[i])->second = pdecSigmaValues;
          }
      }
  }


  template <dftfe::utils::MemorySpace memorySpace>
  void
  excDensityGGAClass<memorySpace>::applyWaveFunctionDependentFuncDerWrtPsi(
    const dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace>
                                                                      &src,
    dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &dst,
    const dftfe::uInt inputVecSize,
    const dftfe::uInt kPointIndex,
    const dftfe::uInt spinIndex)
  {}

  template <dftfe::utils::MemorySpace memorySpace>
  void
  excDensityGGAClass<memorySpace>::applyWaveFunctionDependentFuncDerWrtPsi(
    const dftfe::linearAlgebra::MultiVector<dataTypes::numberFP32, memorySpace>
                                                                          &src,
    dftfe::linearAlgebra::MultiVector<dataTypes::numberFP32, memorySpace> &dst,
    const dftfe::uInt inputVecSize,
    const dftfe::uInt kPointIndex,
    const dftfe::uInt spinIndex)
  {}


  template <dftfe::utils::MemorySpace memorySpace>
  void
  excDensityGGAClass<memorySpace>::updateWaveFunctionDependentFuncDerWrtPsi(
    const std::shared_ptr<AuxDensityMatrix<memorySpace>> &auxDensityMatrixPtr,
    const std::vector<double>                            &kPointWeights)
  {}
  template <dftfe::utils::MemorySpace memorySpace>
  void
  excDensityGGAClass<memorySpace>::computeWaveFunctionDependentExcEnergy(
    const std::shared_ptr<AuxDensityMatrix<memorySpace>> &auxDensityMatrix,
    const std::vector<double>                            &kPointWeights)
  {}

  template <dftfe::utils::MemorySpace memorySpace>
  double
  excDensityGGAClass<memorySpace>::getWaveFunctionDependentExcEnergy()
  {
    return 0.0;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  double
  excDensityGGAClass<
    memorySpace>::getExpectationOfWaveFunctionDependentExcFuncDerWrtPsi()
  {
    return 0.0;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  excDensityGGAClass<memorySpace>::reinitKPointDependentVariables(
    dftfe::uInt kPointIndex)
  {}

  template class excDensityGGAClass<dftfe::utils::MemorySpace::HOST>;
#ifdef DFTFE_WITH_DEVICE
  template class excDensityGGAClass<dftfe::utils::MemorySpace::DEVICE>;
#endif
} // namespace dftfe
