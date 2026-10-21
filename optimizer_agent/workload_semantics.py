"""Closed mathematical contracts shared by research and native Codex tools."""

MEMORY_POLICY = {
    "allocations": "A, B and C are distinct, disjoint device allocations; no aliasing",
    "inputs": "A and B are read-only and checked bitwise after each launch",
    "input_domain": "Dense finite float32 values in [-1,1]; no NaN/Inf, sparsity or sentinel-input assumptions",
    "output": "C is output-only, poisoned before every launch; fully overwrite it without reading prior C",
    "storage": "Contiguous exact-sized float32 buffers; no padding, halo or auxiliary allocation",
    "sizes": "A/B/C each n*n elements",
    "shape_scalar": "n is positive signed int32 with physically admissible allocations; handle non-divisible tails and use sufficiently wide address arithmetic",
    "arithmetic": "Full float32 arithmetic; ordinary float32 FMA is allowed; double, TF32, half and bfloat16 arithmetic are excluded",
    "launches": "Independent invocations; no persistent data, prior-output accumulation or cross-launch synchronization",
}

DESCRIPTIONS = {
    "matmul": ("GEMM", "A/B/C each n*n elements, contiguous row-major n by n",
        "c[row*n+col] = sum(k=0..n-1, a[row*n+k]*b[k*n+col])"),
    "reduction": ("sum reduction", "A/B each n elements; C exactly one element; B is unused",
        "c[0] = sum(i=0..n-1, a[i])"),
    "scan": ("inclusive prefix sum", "A/B/C each n elements; B is unused",
        "c[i] = sum(j=0..i, a[j]) for 0 <= i < n (inclusive scan)"),
    "stencil": ("three-point stencil", "A/B/C each n elements; B is unused",
        "c[i] = a[i-1] + a[i] + a[i+1], treating a[j]=0 for j<0 or j>=n"),
    "convolution": ("three-tap convolution", "A and C each n elements; B exactly 3 elements",
        "c[i] = a[i-1]*b[0] + a[i]*b[1] + a[i+1]*b[2], treating a[j]=0 outside [0,n); coefficients are not reversed"),
    "normalization": ("RMS normalization", "A/B/C each n elements; B is unused",
        "c[i] = a[i] / sqrt((sum(j=0..n-1, a[j]*a[j]))/n + 1e-5); this is RMS normalization, not mean subtraction"),
}


def operation_spec(operation):
    if operation not in DESCRIPTIONS:
        raise ValueError(f"Unknown workload {operation!r}; choose {tuple(DESCRIPTIONS)}")
    title, sizes, mathematics = DESCRIPTIONS[operation]
    return {"operation": operation, "title": title,
            "signature": f'extern "C" __global__ void {operation}(const float* a, const float* b, float* c, int n)',
            "mathematics": mathematics,
            "layout": "contiguous row-major square matrices" if operation == "matmul" else "contiguous one-dimensional arrays",
            "memory_policy": {**MEMORY_POLICY, "sizes": sizes},
            "numerical_tolerance": {"atol": 2e-4, "rtol": 2e-4}}
