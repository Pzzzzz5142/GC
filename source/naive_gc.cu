#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

template <typename scalar_t>
__device__ __forceinline__ scalar_t sigmoid(scalar_t z)
{
    return 1.0 / (1.0 + exp(-z));
}

template <typename scalar_t>
__global__ void convolution_1D_basic_kernel(scalar_t *N, scalar_t *M, scalar_t *P, scalar_t padding_value,
                                            int mask_width, int width)
{
    scalar_t sum = 0;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int idx = blockIdx.y * width + i;
    if (i < width)
    {
        int mask_offset = mask_width / 2;
        for (int j = 0; j < mask_width; j++)
        {
            auto mask_pos = i + j - mask_offset;
            if (mask_pos < 0 || mask_pos >= width)
                sum += padding_value;
            else
                sum += M[j] * N[mask_pos - i + idx];
        }
        P[idx] = sum;
    }
}

torch::Tensor gc_cuda_forward(torch::Tensor input, torch::Tensor mask)
{
    auto output = torch::zeros_like(input);
    auto mask_width = mask.size(2);
    auto width = input.size(2);
    const auto batch_size = input.size(0);
    const auto state_size = input.size(2);
    const int threads = 1024;
    const dim3 blocks((state_size + threads - 1) / threads, batch_size);

    AT_DISPATCH_FLOATING_TYPES(mask.type(), "convolution_1D_basic_kernel", ([&]
                                                                            { convolution_1D_basic_kernel<scalar_t><<<blocks, threads>>>(
                                                                                  input.data<scalar_t>(),
                                                                                  mask.data<scalar_t>(),
                                                                                  output.data<scalar_t>(),
                                                                                  0,
                                                                                  mask_width,
                                                                                  width); }));
    return output;
}