// Independent CPU oracle and evaluator are outside the model's writable workspace.
// One CUDA kernel, row-major float32, full precision accumulation, no preprocessing.
extern "C" __global__ void matmul(const float* a, const float* b, float* c, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) sum += a[row*n + k] * b[k*n + col];
        c[row*n + col] = sum;
    }
}
