#include <CL/cl.h>

#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <assert.h>

#define swap(t, l, r)\
do { \
    t tmp = l;\
    l = r;\
    r = tmp;\
} while(0)

typedef struct {
    size_t sz;
    float* data;
} Matrix;

void transposeMatrix(Matrix m) {
    for (size_t i = 0; i < m.sz; i++) {
        for (size_t j = 0; j < m.sz; j++) {
            swap(float, m.data[i * m.sz + j], m.data[j * m.sz + i]);
        }
    }
}

void printMatrix(Matrix m) {
    for (size_t i = 0; i < m.sz; i++) {
        for (size_t j = 0; j < m.sz; j++) {
            printf("%g ", m.data[i * m.sz + j]);
        }
        puts("");
    }
}

void gemm(const float alpha, const float beta, Matrix A, Matrix B, Matrix C,
         const cl_device_id device, const cl_context context) {
    assert(A.sz == B.sz && B.sz == C.sz);

    char const* gemm_source =
    "__kernel void gemm1(const float alpha, const float beta, const ulong K,\n"
    "                   __global float const* A, __global float const* B, __global float* C) {\n"
    "   // A = MxK; B = KxN; C = MxN\n"
    "   const size_t r = get_global_id(1);\n"
    "   const size_t c = get_global_id(0);\n"
    "   const size_t M = get_global_size(1);\n"
    "   const size_t N = get_global_size(0);\n"
    "   const size_t idx = r * N + c;\n"
    "   C[idx] = 1.0f;\n"
    "   return;\n"
    "   C[idx] *= beta;\n"
    "   float dp = 0.0f;\n"
    "   for (size_t i = 0; i < K; i++) {\n"
    "       dp += A[r * K + i] * B[c * K + i];\n"
    "   }\n"
    "   C[idx] += alpha * dp;\n"
    "   C[idx] = 1;\n"
    "}\n"
    "\n"
    "kernel void gemm(global float const* A, global float* C) {\n"
    "   const size_t i = get_global_id(0) * get_global_size(1) + get_global_id(1);\n"
    "   C[i] = A[i];\n"
    "}\n";

    cl_int err = CL_SUCCESS;
    const cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    if (err) {
        fprintf(stderr, "Failed to create queue\n");
        return;
    }

    const cl_program program = clCreateProgramWithSource(context, 1, &gemm_source, NULL, NULL);
    if (clBuildProgram(program, 1, &device, NULL, NULL, NULL)) {
        fprintf(stderr, "Failed to build gemm program:\n");
        size_t message_buffer_size = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &message_buffer_size);
        char* const message = calloc(message_buffer_size, sizeof(*message));
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, message_buffer_size, message, NULL);
        printf("%s\n", message);
        free(message);
        return;
    }

    err = CL_SUCCESS;
    const cl_kernel gemm_kernel = clCreateKernel(program, "gemm", &err);
    if (err) {
        fprintf(stderr, "Failed to gemm kernel\n");
        return;
    }

    puts("A:");
    printMatrix(A);
    puts("B:");
    printMatrix(B);
    puts("C before gemm:");
    printMatrix(C);

    transposeMatrix(B);
    err = CL_SUCCESS;
    const cl_mem A_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float[A.sz * A.sz]), A.data, &err);
    if (err) {
        fprintf(stderr, "Failed to create buffer for A\n");
        return;
    }
    err = CL_SUCCESS;
    const cl_mem B_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float[B.sz * B.sz]), B.data, &err);
    if (err) {
        fprintf(stderr, "Failed to create buffer for B\n");
        return;
    }
    err = CL_SUCCESS;
    const cl_mem C_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float[C.sz * C.sz]), C.data, &err);
    if (err) {
        fprintf(stderr, "Failed to create buffer for C\n");
        return;
    }

#if 0
    if (clSetKernelArg(gemm_kernel, 0, sizeof(alpha), &alpha)) {
        fprintf(stderr, "Failed to set kernel argument\n");
        return;
    }
    if (clSetKernelArg(gemm_kernel, 1, sizeof(beta), &beta)) {
        fprintf(stderr, "Failed to set kernel argument\n");
        return;
    }
    if (clSetKernelArg(gemm_kernel, 2, sizeof(A.sz), &A.sz)) {
        fprintf(stderr, "Failed to set kernel argument\n");
        return;
    }
    if (clSetKernelArg(gemm_kernel, 4, sizeof(void*), &B_buf)) {
        fprintf(stderr, "Failed to set kernel argument\n");
        return;
    }
#endif
    if (clSetKernelArg(gemm_kernel, 0, sizeof(void*), &A_buf)) {
        fprintf(stderr, "Failed to set kernel argument\n");
        return;
    }
    if (clSetKernelArg(gemm_kernel, 1, sizeof(void*), &C_buf)) {
        fprintf(stderr, "Failed to set kernel argument\n");
        return;
    }

    {
        const size_t global_size[2] = {C.sz, C.sz};
        const size_t local_size[2] = {1, 1};
        if (clEnqueueNDRangeKernel(queue, gemm_kernel, 2, NULL, global_size, local_size, 0, NULL, NULL)) {
            fprintf(stderr, "Failed to enqueue work\n");
            return;
        }
    }
    clFinish(queue);
    
    puts("C after gemm:");
    printMatrix(C);

    transposeMatrix(B);
}

int main() {

    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    const cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);

    float A_data[] = {2.0f, 0.0f, 0.0f, 1.0f};
    float B_data[] = {1.0f, 0.0f, 0.0f, 2.0f};
    float C_data[] = {0.0f, 1.0f, 0.0f, 0.0f};

    Matrix A = {
        2,
        A_data,
    };

    Matrix B = {
        2,
        B_data,
    };

    Matrix C = {
        2,
        C_data,
    };

    gemm(1, 1, A, B, C, device, context);
    
    return 0;
}
