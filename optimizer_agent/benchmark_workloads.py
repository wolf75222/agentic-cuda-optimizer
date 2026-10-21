"""Small reproducible single-kernel suite with independently computed CPU oracles.

Final cases are created only after generation has ended. This module, evaluator
and seeds are never placed in a candidate-generation workspace.
"""
from dataclasses import asdict
from pathlib import Path
import numpy as np

from agent_workload import OutputComparison
from evaluation import Candidate, EvaluationCase, LaunchRule
from use_harness import BufferArgument, BufferDirection, KernelRequest, ScalarArgument


OPERATIONS = ("matmul", "reduction", "scan", "stencil", "convolution", "normalization")


SOURCES = {
    "reduction": '''extern "C" __global__ void reduction(const float* a,const float* b,float* c,int n) {
        if(blockIdx.x==0 && threadIdx.x==0) { float s=0; for(int k=0;k<n;k++) s+=a[k]; c[0]=s; }
    }''',
    "scan": '''extern "C" __global__ void scan(const float* a,const float* b,float* c,int n) {
        int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) {float s=0;for(int k=0;k<=i;k++)s+=a[k];c[i]=s;}
    }''',
    "stencil": '''extern "C" __global__ void stencil(const float* a,const float* b,float* c,int n) {
        int i=blockIdx.x*blockDim.x+threadIdx.x;
        if(i<n)c[i]=(i>0?a[i-1]:0.0f)+a[i]+(i+1<n?a[i+1]:0.0f);
    }''',
    "convolution": '''extern "C" __global__ void convolution(const float* a,const float* b,float* c,int n) {
        int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float s=0;for(int k=0;k<3;k++){int j=i+k-1;if(j>=0&&j<n)s+=a[j]*b[k];}c[i]=s;}
    }''',
    "normalization": '''extern "C" __global__ void normalization(const float* a,const float* b,float* c,int n) {
        int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float s=0;for(int k=0;k<n;k++)s+=a[k]*a[k];c[i]=a[i]*rsqrtf(s/n+1.e-5f);}
    }''',
}


class WorkloadSuite:
    def __init__(self, operation, directory, config):
        if operation not in OPERATIONS:
            raise ValueError(f"Unknown workload {operation!r}; choose {OPERATIONS}")
        self.operation = operation
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.config = config

    @property
    def signature(self):
        return f'extern "C" __global__ void {self.operation}(const float* a, const float* b, float* c, int n)'

    def baseline(self, source_path):
        path = self.directory / "baseline.cu"
        if not path.exists():
            text = Path(source_path).read_text() if self.operation == "matmul" else SOURCES[self.operation]
            path.write_text(text)
        rule = LaunchRule(kind="square_2d", block=(16, 16, 1)) if self.operation == "matmul" else LaunchRule(kind="elementwise", block=(128, 1, 1))
        return Candidate("baseline", path, self.operation, rule)

    def cases(self, final=False):
        sizes = self.config.final_sizes if final else self.config.search_sizes
        # Tail and small-size validation cases are separate from ranking cases.
        if not final:
            sizes = tuple(dict.fromkeys((1, 17, 63) + sizes))
        seed_offset = 1000003 if final else 0
        cases = []
        for index, n in enumerate(sizes):
            seed = self.config.seed + seed_offset + index
            rng = np.random.default_rng(seed)
            count = n*n if self.operation == "matmul" else n
            a = rng.uniform(-1, 1, count).astype("<f4")
            b_count = n*n if self.operation == "matmul" else (3 if self.operation == "convolution" else n)
            b = rng.uniform(-1, 1, b_count).astype("<f4")
            case_id = f'{"final" if final else "search"}-{n}'
            case_dir = self.directory / case_id
            case_dir.mkdir(exist_ok=True)
            a.tofile(case_dir / "a.bin")
            b.tofile(case_dir / "b.bin")
            out_count = 1 if self.operation == "reduction" else count
            request = KernelRequest(
                source="", kernel=self.operation, mode="validate", device=0,
                grid=[1, 1, 1], block=[128, 1, 1], shared_memory_bytes=0,
                warmup=self.config.warmup, iterations=self.config.launches_per_series,
                arguments=[BufferArgument(a.nbytes, str(case_dir / "a.bin"), BufferDirection.Input),
                           BufferArgument(b.nbytes, str(case_dir / "b.bin"), BufferDirection.Input),
                           BufferArgument(out_count*4, None, BufferDirection.Output),
                           ScalarArgument("int32", n)])
            cases.append(EvaluationCase(case_id, request,
                         [OutputComparison(0, "float32", 2e-4, 2e-4)],
                         {"n": n}, seed, 1.0, n == max(sizes)))
        return cases

    def oracle(self, case):
        n = case.metadata["n"]
        a, b = (np.fromfile(case.request.arguments[i].input_file, dtype="<f4") for i in (0, 1))
        if self.operation == "matmul":
            result = (a.astype(np.float64).reshape(n, n) @ b.astype(np.float64).reshape(n, n)).astype("<f4").ravel()
        elif self.operation == "reduction":
            result = np.array([np.sum(a, dtype=np.float64)], dtype="<f4")
        elif self.operation == "scan":
            result = np.cumsum(a, dtype=np.float64).astype("<f4")
        elif self.operation == "stencil":
            p = np.pad(a.astype(np.float64), (1, 1))
            result = (p[:-2] + p[1:-1] + p[2:]).astype("<f4")
        elif self.operation == "convolution":
            p = np.pad(a.astype(np.float64), (1, 1))
            result = (p[:-2]*float(b[0]) + p[1:-1]*float(b[1]) + p[2:]*float(b[2])).astype("<f4")
        else:
            result = (a.astype(np.float64) / np.sqrt(np.mean(a.astype(np.float64)**2) + 1e-5)).astype("<f4")
        return [result]
