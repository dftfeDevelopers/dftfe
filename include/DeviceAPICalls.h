#ifdef DFTFE_WITH_DEVICE

#  ifndef dftfeDeviceAPICalls_H
#    define dftfeDeviceAPICalls_H

#    include <TypeConfig.h>
#    include <DeviceTypeConfig.h>
namespace dftfe
{
  namespace utils
  {
    deviceError_t
    deviceReset();

    deviceError_t
    deviceMemGetInfo(std::size_t *free, std::size_t *total);

    deviceError_t
    getDeviceCount(int *count);

    deviceError_t
    getDevice(int *deviceId);

    deviceError_t
    setDevice(int deviceId);

    deviceError_t
    deviceMalloc(void **devPtr, std::size_t size);

    deviceError_t
    deviceMemset(void *devPtr, int value, std::size_t count);

    /**
     * @brief
     * @param devPtr
     * @param value
     * @param size
     */
    template <typename ValueType>
    void
    deviceSetValue(ValueType *devPtr, ValueType value, std::size_t size);

    deviceError_t
    deviceFree(void *devPtr);

    deviceError_t
    deviceHostMalloc(void **hostPtr, std::size_t size);

    deviceError_t
    deviceHostFree(void *hostPtr);

    /**
     * @brief Copy array from device to host
     * @param count The memory size in bytes of the array
     */
    deviceError_t
    deviceMemcpyD2H(void *dst, const void *src, std::size_t count);

    /**
     * @brief Copy array from device to device
     * @param count The memory size in bytes of the array
     */
    deviceError_t
    deviceMemcpyD2D(void *dst, const void *src, std::size_t count);

    /**
     * @brief Copy array from host to device
     * @param count The memory size in bytes of the array
     */
    deviceError_t
    deviceMemcpyH2D(void *dst, const void *src, std::size_t count);

    /**
     * @brief Copy 2D array from device to host
     * @param count The memory size in bytes of the array
     */
    deviceError_t
    deviceMemcpyD2H_2D(void *      dst,
                       std::size_t dpitch,
                       const void *src,
                       std::size_t spitch,
                       std::size_t width,
                       std::size_t height);

    /**
     * @brief Copy 2D array from device to device
     * @param count The memory size in bytes of the array
     */
    deviceError_t
    deviceMemcpyD2D_2D(void *      dst,
                       std::size_t dpitch,
                       const void *src,
                       std::size_t spitch,
                       std::size_t width,
                       std::size_t height);

    /**
     * @brief Copy 2D array from host to device
     * @param count The memory size in bytes of the array
     */
    deviceError_t
    deviceMemcpyH2D_2D(void *      dst,
                       std::size_t dpitch,
                       const void *src,
                       std::size_t spitch,
                       std::size_t width,
                       std::size_t height);

    /**
     * @brief HOST-DEVICE synchronization
     */
    deviceError_t
    deviceSynchronize();

    /**
     * @brief Copy array from device to host
     * @param count The memory size in bytes of the array
     */
    deviceError_t
    deviceMemcpyAsyncD2H(void *         dst,
                         const void *   src,
                         std::size_t    count,
                         deviceStream_t stream = 0);

    /**
     * @brief Copy array from device to device
     * @param count The memory size in bytes of the array
     */
    deviceError_t
    deviceMemcpyAsyncD2D(void *         dst,
                         const void *   src,
                         std::size_t    count,
                         deviceStream_t stream = 0);

    /**
     * @brief Copy array from host to device
     * @param count The memory size in bytes of the array
     */
    deviceError_t
    deviceMemcpyAsyncH2D(void *         dst,
                         const void *   src,
                         std::size_t    count,
                         deviceStream_t stream = 0);


    deviceError_t
    deviceStreamCreate(deviceStream_t *pStream);

    deviceError_t
    deviceStreamDestroy(deviceStream_t stream);

    deviceError_t
    deviceStreamSynchronize(deviceStream_t stream);

    deviceError_t
    deviceEventCreate(deviceEvent_t *pEvent);

    deviceError_t
    deviceEventDestroy(deviceEvent_t event);

    deviceError_t
    deviceEventRecord(deviceEvent_t event, deviceStream_t stream = 0);

    deviceError_t
    deviceEventSynchronize(deviceEvent_t event);

    deviceError_t
    deviceStreamWaitEvent(deviceStream_t stream,
                          deviceEvent_t  event,
                          unsigned int   flags = 0);

  } // namespace utils
} // namespace dftfe

#  endif // dftfeDeviceAPICalls_H
#endif   // DFTFE_WITH_DEVICE
