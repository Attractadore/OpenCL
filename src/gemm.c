#include <CL/cl.h>

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define swap(t, l, r) \
    do {              \
        t tmp = l;    \
        l = r;        \
        r = tmp;      \
    } while (0)

typedef unsigned long long nanos;

nanos timeNanos() {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ts.tv_sec * 1000000000 + ts.tv_nsec;
}

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

Matrix allocateMatrix(const size_t sz) {
    Matrix m = {
        .sz = sz,
        .data = calloc(sz * sz, sizeof(*m.data)),
    };
    if (!m.data) {
        m.sz = 0;
    }
    return m;
}

void freeMatrix(const Matrix m) {
    free(m.data);
}

Matrix generateMatrix(const size_t sz) {
    const Matrix m = allocateMatrix(sz);
    if (!m.data) {
        return m;
    }
    srand(time(NULL));
    for (size_t i = 0; i < sz; i++) {
        for (size_t j = 0; j < sz; j++) {
            m.data[i * sz + j] = rand() % 100;
        }
    }
    return m;
}

void gemm(const float alpha, const float beta, Matrix A, Matrix B, Matrix C, const cl_device_id device, const cl_context context) {
    assert(A.sz == B.sz && B.sz == C.sz);

    char const* gemm_source =
        "__kernel void gemm(const float alpha, const float beta, const ulong K,\n"
        "                   __global float const* A, __global float const* B, __global float* C) {\n"
        "   // A = MxK; B = KxN; C = MxN\n"
        "   const size_t r = get_global_id(1);\n"
        "   const size_t c = get_global_id(0);\n"
        "   const size_t M = get_global_size(1);\n"
        "   const size_t N = get_global_size(0);\n"
        "   const size_t idx = r * N + c;\n"
        "   C[idx] *= beta;\n"
        "   float dp = 0.0f;\n"
        "   for (size_t i = 0; i < K; i++) {\n"
        "       dp += A[r * K + i] * B[c * K + i];\n"
        "   }\n"
        "   C[idx] += alpha * dp;\n"
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
        fprintf(stderr, "Failed to create gemm kernel\n");
        return;
    }

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
    const cl_mem C_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float[C.sz * C.sz]), C.data, &err);
    if (err) {
        fprintf(stderr, "Failed to create buffer for C\n");
        return;
    }

    err = clSetKernelArg(gemm_kernel, 0, sizeof(alpha), &alpha) ||
          clSetKernelArg(gemm_kernel, 1, sizeof(beta), &beta) ||
          clSetKernelArg(gemm_kernel, 2, sizeof(A.sz), &A.sz) ||
          clSetKernelArg(gemm_kernel, 3, sizeof(void*), &A_buf) ||
          clSetKernelArg(gemm_kernel, 4, sizeof(void*), &B_buf) ||
          clSetKernelArg(gemm_kernel, 5, sizeof(void*), &C_buf);
    if (err) {
        fprintf(stderr, "Failed to set kernel argument\n");
        return;
    }

    {
        const size_t global_size[2] = {C.sz, C.sz};
        const size_t local_size[2] = {8, 8};
        if (clEnqueueNDRangeKernel(queue, gemm_kernel, 2, NULL, global_size, local_size, 0, NULL, NULL)) {
            fprintf(stderr, "Failed to enqueue work\n");
            return;
        }
    }
    clFinish(queue);

    transposeMatrix(B);
}

cl_device_id* getDevices(size_t* const num_devices_p) {
    cl_uint num_platforms = 0;
    if (clGetPlatformIDs(0, NULL, &num_platforms)) {
        return NULL;
    }
    cl_platform_id* const platforms = calloc(num_platforms, sizeof(*platforms));
    if (!platforms) {
        return NULL;
    }
    if (clGetPlatformIDs(num_platforms, platforms, NULL)) {
        free(platforms);
    }
    size_t num_devices = 0;
    for (size_t i = 0; i < num_platforms; i++) {
        cl_uint num_platform_devices = 0;
        if (clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_platform_devices)) {
            free(platforms);
            return NULL;
        }
        num_devices += num_platform_devices;
    }

    cl_device_id* const devices = calloc(num_devices, sizeof(*devices));
    if (!devices) {
        free(platforms);
        return NULL;
    }
    cl_device_id* current_devices = devices;
    for (size_t i = 0; i < num_platforms; i++) {
        cl_uint num_platform_devices = 0;
        if (clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, num_devices - (current_devices - devices), current_devices, &num_platform_devices)) {
            free(devices);
            free(platforms);
            return NULL;
        }
        current_devices += num_platform_devices;
    }

    free(platforms);

    *num_devices_p = num_devices;

    return devices;
}

void printDeviceName(const cl_device_id device) {
    size_t buffer_size = 0;
    if (clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &buffer_size)) {
        return;
    }
    char* const buffer = calloc(buffer_size, sizeof(char));
    if (!buffer) {
        return;
    }
    if (clGetDeviceInfo(device, CL_DEVICE_NAME, buffer_size, buffer, NULL)) {
        free(buffer);
        return;
    }
    printf("%s\n", buffer);
}

int main() {
    const size_t job_size = 1024;
    Matrix A = generateMatrix(job_size);
    Matrix B = generateMatrix(job_size);
    size_t num_devices = 0;
    cl_device_id* const devices = getDevices(&num_devices);
    if (!A.data || !B.data || !devices) {
        freeMatrix(A);
        freeMatrix(B);
        free(devices);
        return -1;
    }

    for (size_t i = 0; i < num_devices; i++) {
        printDeviceName(devices[i]);
        const cl_context context = clCreateContext(NULL, 1, devices + i, NULL, NULL, NULL);
        Matrix C = allocateMatrix(job_size);
        if (!C.data) {
            freeMatrix(A);
            freeMatrix(B);
            free(devices);
            return -1;
        }

        const nanos start = timeNanos();

        gemm(1, 1, A, B, C, devices[i], context);

        const nanos end = timeNanos();

        printf("GEMM in %gs\n", (end - start) / 1e9);

        freeMatrix(C);
    }

    freeMatrix(A);
    freeMatrix(B);
    free(devices);

    return 0;
}
