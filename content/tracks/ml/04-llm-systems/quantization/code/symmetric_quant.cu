// symmetric_quant.cu — per-tensor symmetric INT8 quantization in CUDA.
//
// Demonstrates the three pieces every weight-only / W8A8 quantizer needs:
//   1. absmax reduction          (find the per-tensor scale  s = absmax / 127)
//   2. quantize  x -> int8        (q = clamp(round(x / s), -127, 127))
//   3. INT8 dot product via dp4a  (the matmul that makes quantization pay off),
//      with the scale folded back in the epilogue: out = (s_a * s_b) * acc.
//
// This is the math from README section 1 ("symmetric") and section 4
// ("dequant-fused GEMM") in runnable form. Real kernels (Marlin, CUTLASS)
// add tiling, vectorized loads, and tensor-core mma — the arithmetic is this.
//
// Build & run:
//   nvcc -arch=sm_80 -o symquant symmetric_quant.cu && ./symquant
// (dp4a needs sm_61+; tensor-core INT8 mma needs sm_75+.)

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    return 1; } } while (0)

// ---- 1. absmax reduction: scale = max(|x|) / 127 ---------------------------
__global__ void absmax_kernel(const float* __restrict__ x, int n, float* out) {
    __shared__ float s[256];
    int tid = threadIdx.x;
    float m = 0.0f;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += gridDim.x * blockDim.x)
        m = fmaxf(m, fabsf(x[i]));
    s[tid] = m;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] = fmaxf(s[tid], s[tid + stride]);
        __syncthreads();
    }
    if (tid == 0) atomicMax((int*)out, __float_as_int(s[0])); // floats >= 0, so int bits order-preserving
}

// ---- 2. quantize: q = clamp(round(x / s), -127, 127) -----------------------
__global__ void quantize_kernel(const float* __restrict__ x, int8_t* __restrict__ q,
                                int n, float inv_scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int32_t v = __float2int_rn(x[i] * inv_scale);   // round to nearest, ties to even
    v = v < -127 ? -127 : (v > 127 ? 127 : v);
    q[i] = (int8_t)v;
}

// ---- 3. INT8 dot product with dp4a, dequantized in the epilogue ------------
// acc = sum_i a_q[i] * b_q[i]  (int32), then out = (s_a * s_b) * acc.
__global__ void int8_dot_kernel(const int8_t* __restrict__ a, const int8_t* __restrict__ b,
                                int n, float s_a, float s_b, float* out) {
    __shared__ int sdata[256];
    int tid = threadIdx.x;
    int acc = 0;
    // process 4 int8 at a time with __dp4a (a.b for 4-wide int8 vectors)
    const int32_t* a4 = reinterpret_cast<const int32_t*>(a);
    const int32_t* b4 = reinterpret_cast<const int32_t*>(b);
    int n4 = n / 4;
    for (int i = blockIdx.x * blockDim.x + tid; i < n4; i += gridDim.x * blockDim.x)
        acc = __dp4a(a4[i], b4[i], acc);
    sdata[tid] = acc;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, s_a * s_b * (float)sdata[0]);  // dequant in epilogue
}

int main() {
    const int n = 4096;                       // multiple of 4 for dp4a packing
    float *ha = (float*)malloc(n * sizeof(float));
    float *hb = (float*)malloc(n * sizeof(float));
    double ref = 0.0;
    for (int i = 0; i < n; i++) {
        ha[i] = sinf(0.01f * i) * 3.0f;        // some structure + an outlier below
        hb[i] = cosf(0.02f * i);
    }
    ha[7] = 25.0f;                             // outlier: inflates the per-tensor scale
    for (int i = 0; i < n; i++) ref += (double)ha[i] * hb[i];

    float *da, *db; int8_t *qa, *qb; float *dmax, *dout;
    CUDA_CHECK(cudaMalloc(&da, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&qa, n)); CUDA_CHECK(cudaMalloc(&qb, n));
    CUDA_CHECK(cudaMalloc(&dmax, sizeof(float))); CUDA_CHECK(cudaMalloc(&dout, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(da, ha, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, hb, n * sizeof(float), cudaMemcpyHostToDevice));

    auto scale_of = [&](float* d) -> float {
        float zero = 0.0f;
        CUDA_CHECK(cudaMemcpy(dmax, &zero, sizeof(float), cudaMemcpyHostToDevice));
        absmax_kernel<<<32, 256>>>(d, n, dmax);
        float amax; CUDA_CHECK(cudaMemcpy(&amax, dmax, sizeof(float), cudaMemcpyDeviceToHost));
        return amax / 127.0f;
    };
    float s_a = scale_of(da), s_b = scale_of(db);

    quantize_kernel<<<(n + 255) / 256, 256>>>(da, qa, n, 1.0f / s_a);
    quantize_kernel<<<(n + 255) / 256, 256>>>(db, qb, n, 1.0f / s_b);

    float zero = 0.0f;
    CUDA_CHECK(cudaMemcpy(dout, &zero, sizeof(float), cudaMemcpyHostToDevice));
    int8_dot_kernel<<<32, 256>>>(qa, qb, n, s_a, s_b, dout);
    float got; CUDA_CHECK(cudaMemcpy(&got, dout, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("scale_a=%.5f  scale_b=%.5f\n", s_a, s_b);
    printf("fp32 dot = %.4f\n", ref);
    printf("int8 dot = %.4f   rel.err = %.4f%%\n", got, 100.0 * fabs(got - ref) / fabs(ref));
    printf("(note: the outlier at a[7] inflates scale_a -> larger step -> more error.\n"
           " Per-channel/per-group scales or SmoothQuant would fix this.)\n");

    cudaFree(da); cudaFree(db); cudaFree(qa); cudaFree(qb); cudaFree(dmax); cudaFree(dout);
    free(ha); free(hb);
    return 0;
}
