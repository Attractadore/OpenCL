#include <CL/cl.h>

#include <stdio.h>
#include <stdlib.h>

cl_int handleCLError(cl_int err) {
    switch (err) {
        case CL_SUCCESS:
            break;
        default:
            abort();
    }
    return err;
}

#define clGetDeviceInfo(device, param_name, param_value_size, param_value, param_value_size_ret) \
    handleCLError(clGetDeviceInfo(device, param_name, param_value_size, param_value, param_value_size_ret))
#define clGetDeviceIDs(platform, device_type, num_entries, devices, num_devices) \
    handleCLError(clGetDeviceIDs(platform, device_type, num_entries, devices, num_devices))
#define clGetPlatformInfo(platform, param_name, param_value_size, param_value, param_value_size_ret) \
    handleCLError(clGetPlatformInfo(platform, param_name, param_value_size, param_value, param_value_size_ret))
#define clGetPlatformIDs(num_entries, platforms, num_platforms) \
    handleCLError(clGetPlatformIDs(num_entries, platforms, num_platforms))

void printDeviceInfo(const cl_device_id device) {
    size_t device_name_size = 0;
    clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &device_name_size);
    size_t device_version_size = 0;
    clGetDeviceInfo(device, CL_DEVICE_VERSION, 0, NULL, &device_version_size);
    size_t max_work_item_sizes_size = 0;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, 0, NULL, &max_work_item_sizes_size);

    const size_t device_info_size = max_work_item_sizes_size + device_name_size + device_version_size;
    char* const device_info = calloc(device_info_size, sizeof(*device_info));
    if (!device_info) {
        fprintf(stderr, "Failed to query device info\n");
        return;
    }
    size_t* const max_work_item_sizes = (size_t*) device_info;
    char* const device_name = (char*) max_work_item_sizes + max_work_item_sizes_size;
    char* const device_version = device_name + device_name_size;

    clGetDeviceInfo(device, CL_DEVICE_NAME, device_name_size, device_name, NULL);
    clGetDeviceInfo(device, CL_DEVICE_VERSION, device_version_size, device_version, NULL);
    cl_uint max_compute_units = 0;
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(max_compute_units), &max_compute_units, NULL);
    cl_uint max_work_item_dimensions = 0;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(max_work_item_dimensions), &max_work_item_dimensions, NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, max_work_item_sizes_size, max_work_item_sizes, NULL);
    size_t max_workgroup_size = 0;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_workgroup_size), &max_workgroup_size, NULL);
    cl_uint num_subgroups = 0;
    clGetDeviceInfo(device, CL_DEVICE_MAX_NUM_SUB_GROUPS, sizeof(num_subgroups), &num_subgroups, NULL);
    cl_bool compiler_available = 0;
    clGetDeviceInfo(device, CL_DEVICE_COMPILER_AVAILABLE, sizeof(compiler_available), &compiler_available, NULL);
    cl_bool linker_available = 0;
    clGetDeviceInfo(device, CL_DEVICE_LINKER_AVAILABLE, sizeof(linker_available), &linker_available, NULL);

    printf("Device name: %s\n", device_name);
    printf("Device version: %s\n", device_version);
    printf("Device compute units: %u\n", max_compute_units);
    printf("Device maximum work-item dimensions: %u\n", max_work_item_dimensions);
    printf("Device maximum work-item sizes: ");
    for (size_t i = 0; i < max_work_item_dimensions; i++) {
        printf("%zu, ", max_work_item_sizes[i]);
    }
    printf("\n");
    printf("Device max work-group size: %zu\n", max_workgroup_size);
    printf("Device subgroups: %u\n", num_subgroups);
    printf("Device compiler available: %s\n", (compiler_available) ? ("true") : ("false"));
    printf("Device linker available: %s\n", (linker_available) ? ("true") : ("false"));

    free(device_info);
}

void printPlatformDeviceInfos(const cl_platform_id platform) {
    cl_uint num_platform_devices = 0;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_platform_devices);

    cl_device_id* const platform_device_ids = calloc(num_platform_devices, sizeof(*platform_device_ids));
    if (!platform_device_ids) {
        fprintf(stderr, "Failed to query platform devices\n");
        return;
    }

    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_platform_devices, platform_device_ids, NULL);

    printf("Found %zu device(s) for platform:\n", (size_t) num_platform_devices);
    for (size_t i = 0; i < num_platform_devices; i++) {
        printf("Info for device %zu:\n", i);
        printDeviceInfo(platform_device_ids[i]);
    }

    free(platform_device_ids);
}

void printPlatformInfo(const cl_platform_id platform) {
    size_t platform_name_size = 0;
    clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, NULL, &platform_name_size);
    size_t platform_version_size = 0;
    clGetPlatformInfo(platform, CL_PLATFORM_VERSION, 0, NULL, &platform_version_size);
    size_t platform_profile_size = 0;
    clGetPlatformInfo(platform, CL_PLATFORM_PROFILE, 0, NULL, &platform_profile_size);
    size_t platform_vendor_size = 0;
    clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, NULL, &platform_vendor_size);
    size_t platform_extensions_size = 0;
    clGetPlatformInfo(platform, CL_PLATFORM_EXTENSIONS, 0, NULL, &platform_extensions_size);

    const size_t platform_info_size = platform_name_size + platform_version_size + platform_profile_size + platform_vendor_size + platform_extensions_size;
    char* const platform_info = calloc(platform_info_size, sizeof(*platform_info));
    if (!platform_info) {
        fprintf(stderr, "Failed to query platform info\n");
        return;
    }
    char* const platform_name = platform_info;
    char* const platform_version = platform_name + platform_name_size;
    char* const platform_profile = platform_version + platform_version_size;
    char* const platform_vendor = platform_profile + platform_profile_size;
    char* const platform_extensions = platform_vendor + platform_vendor_size;

    clGetPlatformInfo(platform, CL_PLATFORM_NAME, platform_name_size, platform_name, NULL);
    clGetPlatformInfo(platform, CL_PLATFORM_VERSION, platform_version_size, platform_version, NULL);
    clGetPlatformInfo(platform, CL_PLATFORM_PROFILE, platform_profile_size, platform_profile, NULL);
    clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, platform_vendor_size, platform_vendor, NULL);
    clGetPlatformInfo(platform, CL_PLATFORM_EXTENSIONS, platform_extensions_size, platform_extensions, NULL);
    for (char* p = platform_extensions; *p; p++) {
        if (*p == ' ') {
            *p = '\n';
        }
    }

    printf("Platform name: %s\n", platform_name);
    printf("Platform version: %s\n", platform_version);
    printf("Platform vendor: %s\n", platform_vendor);
    printf("Platform profile: %s\n", platform_profile);
    printf("Platform extensions:\n%s", platform_extensions);
    printPlatformDeviceInfos(platform);

    free(platform_info);
}

void printPlatformInfos() {
    cl_uint num_platforms = 0;
    clGetPlatformIDs(0, NULL, &num_platforms);

    cl_platform_id* const platform_ids = calloc(num_platforms, sizeof(*platform_ids));
    if (!platform_ids) {
        fprintf(stderr, "Failed to query available OpenCL platforms\n");
        return;
    }

    clGetPlatformIDs(num_platforms, platform_ids, NULL);

    printf("Found %zu OpenCL platform(s):\n", (size_t) num_platforms);
    for (size_t i = 0; i < num_platforms; i++) {
        printf("Info for platform %zu:\n", i);
        printPlatformInfo(platform_ids[i]);
        printf("\n");
    }

    free(platform_ids);
}

int main() {
    printPlatformInfos();
    return 0;
}
