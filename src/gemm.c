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

char* readFileAsString(char const* file_name) {
    FILE* file = fopen(file_name, "rb");
    if (!file) {
        goto err;
    }

    if (fseek(file, 0, SEEK_END)) {
        goto close_file;
    }
    const size_t end = ftell(file);
    if (fseek(file, 0, SEEK_SET)) {
        goto close_file;
    }
    const size_t beg = ftell(file);
    if (beg == -1L || end == -1L) {
        goto close_file;
    }
    const size_t file_size = end - beg;

    char* buffer = calloc(file_size + 1, sizeof(*buffer));
    if (!buffer) {
        goto close_file;
        return NULL;
    }

    if (fread(buffer, sizeof(*buffer), file_size, file) != file_size) {
        goto free_buffer;
    }

    fclose(file);

    return buffer;

free_buffer:
    free(buffer);
close_file:
    fclose(file);
err:
    return NULL;
}

#define MSG_READ_SOURCE_FAILED "Failed to read gemm kernel source"
#define MSG_QUEUE_CREATE_FAILED "Failed to create queue"
#define MSG_BUILD_SOURCE_FAILED "Failed to build gemm program"
#define MSG_CREATE_KERNEL_FAILED "Failed to create gemm kernel"
#define MSG_CREATE_BUFFER_FAILED "Failed to create buffer"
#define MSG_SET_KERNEL_ARGUMENTS_FAILED "Failed to set gemm kernel arguments"
#define MSG_ENQUEUE_KERNEL_FAILED "Failed to enqueue gemm kernel"
#define MSG_READ_RESULT_FAILED "Failed to read result"

void gemm(const float alpha, const float beta, Matrix A, Matrix B, Matrix C, const cl_device_id device, const cl_context context) {
    assert(A.sz == B.sz && B.sz == C.sz);

    char const* gemm_source = readFileAsString("gemm.clc");
    char const* err_msg = NULL;
    if (!gemm_source) {
        err_msg = MSG_READ_SOURCE_FAILED;
        goto err;
    }

    cl_int err = CL_SUCCESS;
    const cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    if (err) {
        err_msg = MSG_QUEUE_CREATE_FAILED;
        goto free_source;
    }

    const cl_program program = clCreateProgramWithSource(context, 1, &gemm_source, NULL, NULL);
    if (clBuildProgram(program, 1, &device, NULL, NULL, NULL)) {
        err_msg = MSG_BUILD_SOURCE_FAILED;
        size_t message_buffer_size = 0;
        if (clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &message_buffer_size)) {
            goto free_source;
        }
        char* const message = calloc(message_buffer_size, sizeof(*message));
        if (!message) {
            goto free_source;
        }
        if (clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, message_buffer_size, message, NULL)) {
            free(message);
            goto free_source;
        }
        fprintf(stderr, MSG_BUILD_SOURCE_FAILED ":\n%s", message);
        free(gemm_source);
        free(message);
        return;
    }

    err = CL_SUCCESS;
    const cl_kernel gemm_kernel = clCreateKernel(program, "gemm", &err);
    if (err) {
        goto free_source;
    }

    err = CL_SUCCESS;
    const cl_mem A_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float[A.sz * A.sz]), A.data, &err);
    if (err) {
        err_msg = MSG_CREATE_BUFFER_FAILED " for A";
        goto free_source;
    }
    transposeMatrix(B);
    err = CL_SUCCESS;
    const cl_mem B_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float[B.sz * B.sz]), B.data, &err);
    if (err) {
        err_msg = MSG_CREATE_BUFFER_FAILED " for B";
        goto transpose_B;
    }
    err = CL_SUCCESS;
    const cl_mem C_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float[C.sz * C.sz]), C.data, &err);
    if (err) {
        err_msg = MSG_CREATE_BUFFER_FAILED " for C";
        goto transpose_B;
    }

    err = clSetKernelArg(gemm_kernel, 0, sizeof(alpha), &alpha) ||
          clSetKernelArg(gemm_kernel, 1, sizeof(beta), &beta) ||
          clSetKernelArg(gemm_kernel, 2, sizeof(A.sz), &A.sz) ||
          clSetKernelArg(gemm_kernel, 3, sizeof(void*), &A_buf) ||
          clSetKernelArg(gemm_kernel, 4, sizeof(void*), &B_buf) ||
          clSetKernelArg(gemm_kernel, 5, sizeof(void*), &C_buf);
    if (err) {
        err_msg = MSG_SET_KERNEL_ARGUMENTS_FAILED;
        goto transpose_B;
    }

    {
        const size_t global_size[2] = {C.sz, C.sz};
        const size_t local_size[2] = {8, 8};
        cl_event e;
        if (clEnqueueNDRangeKernel(queue, gemm_kernel, 2, NULL, global_size, local_size, 0, NULL, &e)) {
            err_msg = MSG_ENQUEUE_KERNEL_FAILED;
            goto transpose_B;
        }

        if (clEnqueueReadBuffer(queue, C_buf, 1, 0, sizeof(float[C.sz * C.sz]), C.data, 1, &e, NULL)) {
            err_msg = MSG_READ_RESULT_FAILED;
            goto finish_work;
        }
    }

    clFinish(queue);
    transposeMatrix(B);
    return;

finish_work:
    clFinish(queue);
transpose_B:
    transposeMatrix(B);
free_source:
    free(gemm_source);
err:
    fprintf(stderr, "%s\n", err_msg);
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
