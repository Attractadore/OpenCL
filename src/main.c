#include <CL/cl.h>

#include <stdio.h>
#include <stdlib.h>

void printDeviceInfo(const cl_device_id device) {
    size_t device_name_size = 0;
    char* device_name = NULL;
    if (clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &device_name_size)) {
        goto msg;
    }
    device_name = calloc(device_name_size, sizeof(*device_name));
    if (!device_name) {
        goto msg;
    }
    if (clGetDeviceInfo(device, CL_DEVICE_NAME, device_name_size, device_name, NULL)) {
        goto clean;
    }
    printf("Device name: %s\n", device_name);
    free(device_name);
    return;

clean:
    free(device_name);
msg:
    fprintf(stderr, "Failed to query device info\n");
}

void printPlatformDeviceInfos(const cl_platform_id platform) {
    cl_uint num_platform_devices = 0;
    cl_device_id* platform_device_ids = NULL;
    if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_platform_devices)) {
        goto msg;
    }
    platform_device_ids = calloc(num_platform_devices, sizeof(*platform_device_ids));
    if (!platform_device_ids) {
        goto msg;
    }
    if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_platform_devices, platform_device_ids, NULL)) {
        goto clean;
    }
    printf("Found %zu device(s) for platform:\n", (size_t) num_platform_devices);
    for (size_t i = 0; i < num_platform_devices; i++) {
        printf("Info for device %zu:\n", i);
        printDeviceInfo(platform_device_ids[i]);
    }
    free(platform_device_ids);
    return;

clean:
    free(platform_device_ids);
msg:
    fprintf(stderr, "Failed to query platform devices\n");
}

void printPlatformInfo(const cl_platform_id platform) {
    size_t platform_name_size = 0;
    char* platform_name = NULL;
    if (clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, NULL, &platform_name_size)) {
        goto msg;
    }
    platform_name = calloc(platform_name_size, sizeof(*platform_name));
    if (!platform_name) {
        goto msg;
    }
    if (clGetPlatformInfo(platform, CL_PLATFORM_NAME, platform_name_size, platform_name, NULL)) {
        goto clean;
    }
    printf("Platform name: %s\n", platform_name);
    printPlatformDeviceInfos(platform);
    free(platform_name);
    return;

clean:
    free(platform_name);
msg:
    fprintf(stderr, "Failed to query platform info\n");
}

void printPlatformInfos() {
    cl_uint num_platforms = 0;
    cl_platform_id* platform_ids = NULL;
    if (clGetPlatformIDs(0, NULL, &num_platforms)) {
        goto msg;
    }
    platform_ids = calloc(num_platforms, sizeof(*platform_ids));
    if (!platform_ids) {
        goto msg;
    }
    if (clGetPlatformIDs(num_platforms, platform_ids, NULL)) {
        goto clean;
    }
    printf("Found %zu OpenCL platform(s):\n", (size_t) num_platforms);
    for (size_t i = 0; i < num_platforms; i++) {
        printf("Info for platform %zu:\n", i);
        printPlatformInfo(platform_ids[i]);
        puts("");
    }
    free(platform_ids);
    return;

clean:
    free(platform_ids);
msg:
    fprintf(stderr, "Failed to query available OpenCL platforms\n");
}

int main() {
    printPlatformInfos();
    return 0;
}
