# Benchmarking GEMM in frameworks I care about.

The benchmarked operation is the multiplication of two 10240x10240 float32 matrices.
They are multiplied 10 times and the fastest of the 10 runs is reported, in seconds.

I'm trying to compile every framework with the most recent [OpenBLAS](http://www.openblas.net) version compiled for my CPU.
So ideally, the performance should be the same everywhere.

The GPU is an NVIDIA GeForce GTX980, using cuBLAS.

If that's not feasible, I'm using the framework's shipped GEMM.

## Running the benchmarks

These are commands for the [fish shell](http://fishshell.com), adapt accordingly for yours.

### Julia

Compiled with OpenBLAS build.

On the CPU:

```
julia julia_matmul.jl
```

On the GPU:

```
julia julia_matmul_cuda.jl
```

### Theano

Compiled with OpenBLAS build through NumPy.

On the CPU:

```
env THEANO_FLAGS="device=cpu,floatX=float32" python theano_matmul.py
```

On the GPU:

```
env THEANO_FLAGS="device=gpu,floatX=float32" python theano_matmul.py
```

### Tensorflow

NOT compiled with OpenBLAS build, since that doesn't seem possible. It seems to be using Eigen3.

On the CPU:

```
python tensorflow_matmul.py "/cpu:0"
```

On the GPU:

```
python tensorflow_matmul.py "/gpu:0"
```

## Results

- Julia CPU: 4.3072s
- Julia GPU: 0.5368s
- Theano CPU: 4.3492s
- Theano GPU: 0.6838s (0.6000s without `np.array`)
- Tensorflow CPU: 13.3393s
- Tensorflow GPU: 0.6979s  (0.6106s without `np.array`)

## Additional output

### Tensorflow CPU

```
I tensorflow/core/common_runtime/local_device.cc:25] Local device intra op parallelism threads: 12
I tensorflow/core/common_runtime/gpu/gpu_init.cc:88] Found device 0 with properties: 
name: GeForce GTX 980
major: 5 minor: 2 memoryClockRate (GHz) 1.2405
pciBusID 0000:03:00.0
Total memory: 4.00GiB
Free memory: 3.91GiB
I tensorflow/core/common_runtime/gpu/gpu_init.cc:88] Found device 1 with properties: 
name: GeForce GTX 470
major: 2 minor: 0 memoryClockRate (GHz) 1.215
pciBusID 0000:02:00.0
Total memory: 1.25GiB
Free memory: 958.66MiB
I tensorflow/core/common_runtime/gpu/gpu_init.cc:45] cannot enable peer access from device ordinal 0 to device ordinal 1
I tensorflow/core/common_runtime/gpu/gpu_init.cc:45] cannot enable peer access from device ordinal 1 to device ordinal 0
I tensorflow/core/common_runtime/gpu/gpu_init.cc:112] DMA: 0 1 
I tensorflow/core/common_runtime/gpu/gpu_init.cc:122] 0:   Y N 
I tensorflow/core/common_runtime/gpu/gpu_init.cc:122] 1:   N Y 
I tensorflow/core/common_runtime/gpu/gpu_device.cc:643] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 980, pci bus id: 0000:03:00.0)
I tensorflow/core/common_runtime/gpu/gpu_device.cc:611] Ignoring gpu device (device: 1, name: GeForce GTX 470, pci bus id: 0000:02:00.0) with Cuda compute capability 2.0. The minimum required Cuda capability is 3.5.
I tensorflow/core/common_runtime/gpu/gpu_region_allocator.cc:47] Setting region size to 3881496576
I tensorflow/core/common_runtime/local_session.cc:45] Local session inter op parallelism threads: 12
```

### Tensorflow GPU

```
I tensorflow/core/common_runtime/local_device.cc:25] Local device intra op parallelism threads: 12
I tensorflow/core/common_runtime/gpu/gpu_init.cc:88] Found device 0 with properties: 
name: GeForce GTX 980
ma:CtrlP
r: 5 minor: 2 memoryClockRate (GHz) 1.2405
pciBusID 0000:03:00.0
Total memory: 4.00GiB
Free memory: 3.91GiB
I tensorflow/core/common_runtime/gpu/gpu_init.cc:88] Found device 1 with properties: 
name: GeForce GTX 470
ma:CtrlP
r: 2 minor: 0 memoryClockRate (GHz) 1.215
pciBusID 0000:02:00.0
Total memory: 1.25GiB
Free memory: 958.66MiB
I tensorflow/core/common_runtime/gpu/gpu_init.cc:45] cannot enable peer access from device ordinal 0 to device ordinal 1
I tensorflow/core/common_runtime/gpu/gpu_init.cc:45] cannot enable peer access from device ordinal 1 to device ordinal 0
I tensorflow/core/common_runtime/gpu/gpu_init.cc:112] DMA: 0 1 
I tensorflow/core/common_runtime/gpu/gpu_init.cc:122] 0:   Y N 
I tensorflow/core/common_runtime/gpu/gpu_init.cc:122] 1:   N Y 
I tensorflow/core/common_runtime/gpu/gpu_device.cc:643] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 980, pci bus id: 0000:03:00.0)
I tensorflow/core/common_runtime/gpu/gpu_device.cc:611] Ignoring gpu device (device: 1, name: GeForce GTX 470, pci bus id: 0000:02:00.0) with Cuda compute capability 2.0. The minimum required Cuda capability is 3.5.
I tensorflow/core/common_runtime/gpu/gpu_region_allocator.cc:47] Setting region size to 3881496576
I tensorflow/core/common_runtime/local_session.cc:45] Local session inter op parallelism threads: 12
```
