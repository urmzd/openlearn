// conv2d.cu — the same 2D convolution as conv2d.c, mapped onto the GPU
// (README §4). One thread computes one output pixel; the k×k filter window is
// the inner loop. This naive map is what cuDNN/CUTLASS optimize with tiling,
// shared-memory staging, im2col+tensor-core GEMM, and Winograd — but the
// arithmetic each thread does is exactly this sum.
//
// Build & run (needs an NVIDIA GPU + CUDA toolkit):
//   nvcc -O2 -o conv2d conv2d.cu && ./conv2d

#include <cstdio>
#include <cuda_runtime.h>

#define CK(call) do { cudaError_t e = (call); if (e) { \
    fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    return 1; } } while (0)

// in: H×W, ker: K×K, out: OH×OW, valid convolution, stride s.
__global__ void conv2d_kernel(const float *in, const float *ker, float bias,
                              float *out, int H, int W, int K, int s,
                              int OH, int OW) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // output col
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // output row
    if (i >= OH || j >= OW) return;

    float acc = bias;
    for (int u = 0; u < K; u++)
        for (int v = 0; v < K; v++)
            acc += in[(i * s + u) * W + (j * s + v)] * ker[u * K + v];
    out[i * OW + j] = acc;
}

int main(void) {
    const int H = 5, W = 5, K = 3, s = 1;
    const int OH = (H - K) / s + 1, OW = (W - K) / s + 1;

    float h_in[H * W] = {
        0,0,1,1,1, 0,0,1,1,1, 0,0,1,1,1, 0,0,1,1,1, 0,0,1,1,1,
    };
    float h_ker[K * K] = {-1,0,1, -2,0,2, -1,0,1};  // Sobel-x
    float h_out[OH * OW];

    float *d_in, *d_ker, *d_out;
    CK(cudaMalloc(&d_in,  sizeof h_in));
    CK(cudaMalloc(&d_ker, sizeof h_ker));
    CK(cudaMalloc(&d_out, sizeof h_out));
    CK(cudaMemcpy(d_in,  h_in,  sizeof h_in,  cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_ker, h_ker, sizeof h_ker, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((OW + block.x - 1) / block.x, (OH + block.y - 1) / block.y);
    conv2d_kernel<<<grid, block>>>(d_in, d_ker, 0.0f, d_out, H, W, K, s, OH, OW);
    CK(cudaGetLastError());
    CK(cudaMemcpy(h_out, d_out, sizeof h_out, cudaMemcpyDeviceToHost));

    printf("vertical-edge feature map (%dx%d):\n", OH, OW);
    for (int i = 0; i < OH; i++) {
        for (int j = 0; j < OW; j++) printf("%6.1f", h_out[i * OW + j]);
        putchar('\n');
    }
    cudaFree(d_in); cudaFree(d_ker); cudaFree(d_out);
    return 0;
}
