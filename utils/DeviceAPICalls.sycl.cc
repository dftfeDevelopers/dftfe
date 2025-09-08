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



#ifdef DFTFE_WITH_DEVICE_LANG_SYCL
#  include <DeviceAPICalls.h>
#  include <stdio.h>
#  include <vector>
#  include <DeviceDataTypeOverloads.h>
#  include <DeviceKernelLauncherHelpers.h>
#  include <DeviceTypeConfigHalfPrec.sycl.h>
#  include <Exceptions.h>

namespace dftfe
{
  namespace utils
  {
    namespace
    {
      template <typename ValueType>
      void
      setValueKernel(sycl::nd_item<1> ind,
                     ValueType       *devPtr,
                     ValueType        value,
                     std::size_t      size)
      {
        const std::size_t globalThreadId = ind.get_global_id(0);
        std::size_t       n_workgroups   = ind.get_group_range(0);
        std::size_t       n_workitems    = ind.get_local_range(0);

        for (std::size_t idx = globalThreadId; idx < size;
             idx += n_workgroups * n_workitems)
          {
            devPtr[idx] = value;
          }
      }
    } // namespace

    deviceError_t
    deviceReset()
    {
      dftfe::utils::queueRegistry.clear();
      dftfe::utils::usedStreamIds.clear();
      dftfe::utils::usedStreamIds.insert(dftfe::utils::defaultStream);
      dftfe::utils::queueRegistry[dftfe::utils::defaultStream] =
        sycl::queue(dftfe::utils::syclContext,
                    dftfe::utils::syclDevice,
                    sycl::property::queue::in_order{});
      return dftfe::utils::deviceSuccess;
    }


    deviceError_t
    deviceMemGetInfo(std::size_t *free, std::size_t *total)
    {
      try
        {
          *free = dftfe::utils::queueRegistry.find(dftfe::utils::defaultStream)
                    ->second.get_device()
                    .get_info<sycl::info::device::local_mem_size>();
          *total =
            dftfe::utils::queueRegistry.find(dftfe::utils::defaultStream)
              ->second.get_device()
              .get_info<sycl::ext::intel::info::device::free_memory>();
        }
      catch (const deviceError_t &e)
        {
          return e;
        }
      return dftfe::utils::deviceSuccess;
    }

    deviceError_t
    getDeviceCount(int *count)
    {
      *count = sycl::device::get_devices().size();
      return dftfe::utils::deviceSuccess;
    }

    deviceError_t
    getDevice(int *deviceId)
    {
      *deviceId = dftfe::utils::syclDeviceId;
      return dftfe::utils::deviceSuccess;
    }

    deviceError_t
    setDevice(int deviceId)
    {
      dftfe::utils::syclDeviceId = deviceId;
      dftfe::utils::syclDevice =
        dftfe::utils::allSyclGPUDevices[dftfe::utils::syclDeviceId];
      dftfe::utils::syclContext = sycl::context(dftfe::utils::syclDevice);
      return dftfe::utils::deviceSuccess;
    }

    deviceError_t
    deviceMalloc(void **devPtr, std::size_t size)
    {
      try
        {
          *devPtr = sycl::malloc_device(size,
                                        dftfe::utils::queueRegistry
                                          .find(dftfe::utils::defaultStream)
                                          ->second);
        }
      catch (const dftfe::utils::deviceError_t &e)
        {
          return e;
        }
      return dftfe::utils::deviceSuccess;
    }

    deviceError_t
    deviceMemset(void *devPtr, int value, std::size_t count)
    {
      dftfe::utils::queueRegistry.find(dftfe::utils::defaultStream)
        ->second.memset(devPtr, value, count);
      dftfe::utils::queueRegistry.find(dftfe::utils::defaultStream)
        ->second.wait_and_throw();
      return dftfe::utils::deviceSuccess;
    }

    template <typename ValueType>
    void
    deviceSetValue(ValueType *devPtr, ValueType value, std::size_t size)
    {
      std::size_t total_workitems =
        (size / dftfe::utils::DEVICE_BLOCK_SIZE + 1) *
        dftfe::utils::DEVICE_BLOCK_SIZE;
      deviceEvent_t event =
        dftfe::utils::queueRegistry.find(dftfe::utils::defaultStream)
          ->second.parallel_for(
            sycl::nd_range<1>(total_workitems, dftfe::utils::DEVICE_BLOCK_SIZE),
            [=](sycl::nd_item<1> ind) {
              setValueKernel(ind,
                             makeDataTypeDeviceCompatible(devPtr),
                             makeDataTypeDeviceCompatible(value),
                             size);
            });
      DEVICE_API_CHECK(event);
    }

    template void
    deviceSetValue(bool *devPtr, bool value, std::size_t size);

    template void
    deviceSetValue(int *devPtr, int value, std::size_t size);

    template void
    deviceSetValue(long int *devPtr, long int value, std::size_t size);

    template void
    deviceSetValue(unsigned int *devPtr, unsigned int value, std::size_t size);

    template void
    deviceSetValue(unsigned long int *devPtr,
                   unsigned long int  value,
                   std::size_t        size);

    template void
    deviceSetValue(double *devPtr, double value, std::size_t size);

    template void
    deviceSetValue(float *devPtr, float value, std::size_t size);

    template void
    deviceSetValue(std::complex<float> *devPtr,
                   std::complex<float>  value,
                   std::size_t          size);

    template void
    deviceSetValue(std::complex<double> *devPtr,
                   std::complex<double>  value,
                   std::size_t           size);

    template void
    deviceSetValue(uint16_t *devPtr, uint16_t value, std::size_t size);

    template void
    deviceSetValue(std::complex<uint16_t> *devPtr,
                   std::complex<uint16_t>  value,
                   std::size_t             size);


    deviceError_t
    deviceFree(void *devPtr)
    {
      deviceSynchronize();
      try
        {
          sycl::free(devPtr,
                     dftfe::utils::queueRegistry
                       .find(dftfe::utils::defaultStream)
                       ->second);
        }
      catch (const dftfe::utils::deviceError_t &e)
        {
          return e;
        }
      return dftfe::utils::deviceSuccess;
    }

    deviceError_t
    deviceHostMalloc(void **hostPtr, std::size_t size)
    {
      try
        {
          *hostPtr = std::malloc(size);
        }
      catch (const dftfe::utils::deviceError_t &e)
        {
          return e;
        }
      return dftfe::utils::deviceSuccess;
    }

    deviceError_t
    deviceHostFree(void *hostPtr)
    {
      try
        {
          std::free(hostPtr);
        }
      catch (const dftfe::utils::deviceError_t &e)
        {
          return e;
        }
      return dftfe::utils::deviceSuccess;
    }

    deviceError_t
    deviceMemcpyD2H(void *dst, const void *src, std::size_t count)
    {
      deviceSynchronize();
      dftfe::utils::queueRegistry.find(dftfe::utils::defaultStream)
        ->second.memcpy(dst, src, count)
        .wait_and_throw();
      return dftfe::utils::deviceSuccess;
    }

    deviceError_t
    deviceMemcpyD2D(void *dst, const void *src, std::size_t count)
    {
      deviceSynchronize();
      try
        {
          dftfe::utils::queueRegistry.find(dftfe::utils::defaultStream)
            ->second.memcpy(dst, src, count)
            .wait_and_throw();
        }
      catch (const dftfe::utils::deviceError_t &e)
        {
          return e;
        }
      return dftfe::utils::deviceSuccess;
    }
    deviceError_t
    deviceMemcpyH2D(void *dst, const void *src, std::size_t count)
    {
      deviceSynchronize();
      dftfe::utils::queueRegistry.find(dftfe::utils::defaultStream)
        ->second.memcpy(dst, src, count)
        .wait_and_throw();
      return dftfe::utils::deviceSuccess;
    }

    deviceError_t
    deviceMemcpyD2H_2D(void       *dst,
                       std::size_t dpitch,
                       const void *src,
                       std::size_t spitch,
                       std::size_t width,
                       std::size_t height)
    {
      // dftfe::utils::deviceStream_t queue{sycl::gpu_selector_v};
      // deviceEvent_t event = queue.sycl::_V1::queue::ext_oneapi_memcpy2d(dst,
      // dpitch, src, spitch, width, height); DEVICE_API_CHECK(event);
      return dftfe::utils::deviceSuccess;
    }


    deviceError_t
    deviceMemcpyD2D_2D(void       *dst,
                       std::size_t dpitch,
                       const void *src,
                       std::size_t spitch,
                       std::size_t width,
                       std::size_t height)
    {
      // dftfe::utils::deviceStream_t queue{sycl::gpu_selector_v};
      // deviceEvent_t event = queue.sycl::_V1::queue::ext_oneapi_memcpy2d(dst,
      // dpitch, src, spitch, width, height); DEVICE_API_CHECK(event);
      return dftfe::utils::deviceSuccess;
    }

    deviceError_t
    deviceMemcpyH2D_2D(void       *dst,
                       std::size_t dpitch,
                       const void *src,
                       std::size_t spitch,
                       std::size_t width,
                       std::size_t height)
    {
      // dftfe::utils::deviceStream_t queue{sycl::gpu_selector_v};
      // deviceEvent_t event = queue.sycl::_V1::queue::ext_oneapi_memcpy2d(dst,
      // dpitch, src, spitch, width, height); DEVICE_API_CHECK(event);
      return dftfe::utils::deviceSuccess;
    }

    deviceError_t
    deviceSynchronize()
    {
      for (dftfe::uInt iStream : dftfe::utils::usedStreamIds)
        dftfe::utils::queueRegistry.find(iStream)->second.wait_and_throw();
      return dftfe::utils::deviceSuccess;
    }

    deviceError_t
    deviceMemcpyAsyncD2H(void          *dst,
                         const void    *src,
                         std::size_t    count,
                         deviceStream_t stream)
    {
      try
        {
          dftfe::utils::queueRegistry.find(stream)->second.memcpy(dst,
                                                                  src,
                                                                  count);
        }
      catch (const dftfe::utils::deviceError_t &e)
        {
          return e;
        }
      return dftfe::utils::deviceSuccess;
    }

    deviceError_t
    deviceMemcpyAsyncD2D(void          *dst,
                         const void    *src,
                         std::size_t    count,
                         deviceStream_t stream)
    {
      try
        {
          dftfe::utils::queueRegistry.find(stream)->second.memcpy(dst,
                                                                  src,
                                                                  count);
        }
      catch (const dftfe::utils::deviceError_t &e)
        {
          return e;
        }
      return dftfe::utils::deviceSuccess;
    }

    deviceError_t
    deviceMemcpyAsyncH2D(void          *dst,
                         const void    *src,
                         std::size_t    count,
                         deviceStream_t stream)
    {
      try
        {
          dftfe::utils::queueRegistry.find(stream)->second.memcpy(dst,
                                                                  src,
                                                                  count);
        }
      catch (const dftfe::utils::deviceError_t &e)
        {
          return e;
        }
      return dftfe::utils::deviceSuccess;
    }

    deviceError_t
    deviceStreamCreate(deviceStream_t &pStream, const bool nonBlocking)
    {
      pStream = 0;
      while (dftfe::utils::usedStreamIds.find(pStream) !=
             dftfe::utils::usedStreamIds.end())
        pStream++;
      dftfe::utils::usedStreamIds.insert(pStream);
      dftfe::utils::queueRegistry[pStream] =
        sycl::queue(dftfe::utils::syclContext,
                    dftfe::utils::syclDevice,
                    sycl::property::queue::in_order{});

      return dftfe::utils::deviceSuccess;
    }

    deviceError_t
    deviceStreamDestroy(deviceStream_t &stream)
    {
      if (stream == dftfe::utils::defaultStream)
        throw std::invalid_argument("Trying to destroy the default stream");
      dftfe::utils::queueRegistry.find(stream)->second.wait_and_throw();
      dftfe::utils::queueRegistry.erase(stream);
      dftfe::utils::usedStreamIds.erase(stream);
      stream = 0;
      return dftfe::utils::deviceSuccess;
    }

    deviceError_t
    deviceStreamSynchronize(deviceStream_t &stream)
    {
      dftfe::utils::queueRegistry.find(stream)->second.wait_and_throw();
      return dftfe::utils::deviceSuccess;
    }

    deviceError_t
    deviceEventCreate(deviceEvent_t &pEvent)
    {
      return dftfe::utils::deviceSuccess;
    }

    deviceError_t
    deviceEventDestroy(deviceEvent_t &event)
    {
      return dftfe::utils::deviceSuccess;
    }

    deviceError_t
    deviceEventRecord(deviceEvent_t &event, deviceStream_t stream)
    {
      event = dftfe::utils::queueRegistry.find(stream)
                ->second.ext_oneapi_submit_barrier();
      return dftfe::utils::deviceSuccess;
    }

    deviceError_t
    deviceEventSynchronize(deviceEvent_t &event)
    {
      event.wait_and_throw();
      return dftfe::utils::deviceSuccess;
    }

    deviceError_t
    deviceStreamWaitEvent(deviceStream_t &stream,
                          deviceEvent_t  &event,
                          unsigned int    flags)
    {
      dftfe::utils::queueRegistry.find(stream)
        ->second.ext_oneapi_submit_barrier({event});
      return dftfe::utils::deviceSuccess;
    }

  } // namespace utils
} // namespace dftfe
#endif
