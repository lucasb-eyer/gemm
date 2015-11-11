using Base.LinAlg.BLAS
using CUDArt
using CUBLAS

device(0)

const d_A = CudaArray(map(Float32, randn(10240,10240)))
const d_B = CudaArray(map(Float32, randn(10240,10240)))

d_C = gemm('N', 'N', d_A, d_B)
device_synchronize()
free(d_C)
device_synchronize()

tmin = Inf
for i=1:10
    tic()
    d_C = gemm('N', 'N', d_A, d_B)
    to_host(d_C)  # This is here for fairness to others which always transfer the result back to the host.
    device_synchronize()
    tmin = min(tmin, toc())

    # Note: this may or may not be considered cheating.
    free(d_C)
    device_synchronize()
end

print(tmin)

device_reset(0)
