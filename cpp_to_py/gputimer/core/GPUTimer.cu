

#include "GPUTimer.h"
#include "gputimer/db/GTDatabase.h"
#include "gputiming.h"
#include "utils.cuh"

namespace gt {

// GPULutAllocator *d_allocator;

void GPUTimer::initialize() {
    cudaMalloc(&pinCap, num_pins * (NUM_ATTR + 2) * sizeof(float));
    cudaMalloc(&pinWireCap, num_pins * NUM_ATTR * sizeof(float));

    cudaMalloc(&testRelatedAT, num_tests * NUM_ATTR * sizeof(float));
    cudaMalloc(&testRAT, num_tests * NUM_ATTR * sizeof(float));
    cudaMalloc(&testConstraint, num_tests * NUM_ATTR * sizeof(float));
    cudaMalloc(&pinRootRes, num_pins * NUM_ATTR * sizeof(float));
    cudaMalloc(&arcSlew, num_arcs * 2 * NUM_ATTR * sizeof(float));

    cudaMalloc(&net_is_clock, num_nets * sizeof(int));
    cudaMalloc(&level_list, num_pins * sizeof(int));
    cudaMalloc(&pin_outs, num_POs * sizeof(index_type));

    cudaMemcpy(pinCap, gtdb.pin_capacitance.data(), num_pins * (NUM_ATTR + 2) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(net_is_clock, gtdb.net_is_clock.data(), num_nets * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(pin_outs, gtdb.pin_outs.data(), gtdb.pin_outs.size() * sizeof(index_type), cudaMemcpyHostToDevice);

    // auto GPUTiming = GPULutAllocator();
    // GPUTiming.AllocateBatch(gtdb.timings);
    // GPUTiming.CopyToGPU();
    // printf("%d luts\n", GPUTiming.num_luts);
    // // GPULutAllocator *d_allocator;
    // cudaMalloc((void **)&d_allocator, sizeof(GPULutAllocator));
    // cudaMemcpy(d_allocator, &GPUTiming, sizeof(GPULutAllocator), cudaMemcpyHostToDevice);
    // GPUTiming.CopyToGPU(d_allocator);

    allocator = new GPULutAllocator();
    allocator->AllocateBatch(gtdb.liberty_timing_arcs);
    allocator->CopyToGPU();
    cudaMalloc((void **)&d_allocator, sizeof(GPULutAllocator));
    cudaMemcpy(d_allocator, allocator, sizeof(GPULutAllocator), cudaMemcpyHostToDevice);
    allocator->CopyToGPU(d_allocator);

    printf("GPUTimer initialized\n");

    cudaMalloc(&__pinSlew__, num_pins * NUM_ATTR * sizeof(float));
    cudaMalloc(&__pinLoad__, num_pins * NUM_ATTR * sizeof(float));
    cudaMalloc(&__pinRAT__, num_pins * NUM_ATTR * sizeof(float));
    cudaMalloc(&__pinAT__, num_pins * NUM_ATTR * sizeof(float));

    device_copy_batch<float><<<BLOCK_NUMBER(num_pins * NUM_ATTR), BLOCK_SIZE>>>(pinSlew, __pinSlew__, num_pins * NUM_ATTR);
    device_copy_batch<float><<<BLOCK_NUMBER(num_pins * NUM_ATTR), BLOCK_SIZE>>>(pinLoad, __pinLoad__, num_pins * NUM_ATTR);
    device_copy_batch<float><<<BLOCK_NUMBER(num_pins * NUM_ATTR), BLOCK_SIZE>>>(pinRAT, __pinRAT__, num_pins * NUM_ATTR);
    device_copy_batch<float><<<BLOCK_NUMBER(num_pins * NUM_ATTR), BLOCK_SIZE>>>(pinAT, __pinAT__, num_pins * NUM_ATTR);
}

GPUTimer::~GPUTimer() {
    logger.info("destruct GPUTimer");

    cudaFree(pinCap);
    cudaFree(pinWireCap);

    cudaFree(testRelatedAT);
    cudaFree(testRAT);
    cudaFree(testConstraint);
    cudaFree(pinRootRes);
    cudaFree(arcSlew);

    cudaFree(net_is_clock);
    cudaFree(level_list);
    cudaFree(pin_outs);

    cudaFree(__pinSlew__);
    cudaFree(__pinLoad__);
    cudaFree(__pinRAT__);
    cudaFree(__pinAT__);

    // cudaFree(d_allocator);
    // destruct GPULutAllocator
    // gtdb.~GTDatabase();
    // allocator->free(d_allocator);
    allocator->~GPULutAllocator();
    cudaFree(d_allocator);
    // d_allocator->~GPULutAllocator();
    // delete allocator;
}

void GPUTimer::update_states() {
    cudaMemset(pinImpulse, 0, num_pins * NUM_ATTR * sizeof(float));
    cudaMemset(pinRootRes, 0, num_pins * NUM_ATTR * sizeof(float));
    cudaMemset(pinRootDelay, 0, num_pins * NUM_ATTR * sizeof(float));
    cudaMemset(pinWireCap, 0, num_pins * NUM_ATTR * sizeof(float));

    reset_val<float><<<BLOCK_NUMBER(2 * num_arcs * NUM_ATTR), BLOCK_SIZE>>>(arcDelay, 2 * num_arcs * NUM_ATTR);
    reset_val<float><<<BLOCK_NUMBER(2 * num_arcs * NUM_ATTR), BLOCK_SIZE>>>(arcSlew, 2 * num_arcs * NUM_ATTR);
    reset_val<float><<<BLOCK_NUMBER(num_tests * NUM_ATTR), BLOCK_SIZE>>>(testRelatedAT, num_tests * NUM_ATTR);
    reset_val<float><<<BLOCK_NUMBER(num_tests * NUM_ATTR), BLOCK_SIZE>>>(testRAT, num_tests * NUM_ATTR);
    reset_val<float><<<BLOCK_NUMBER(num_tests * NUM_ATTR), BLOCK_SIZE>>>(testConstraint, num_tests * NUM_ATTR);

    reset_val<index_type><<<BLOCK_NUMBER(num_pins * NUM_ATTR), BLOCK_SIZE>>>(at_prefix_pin, num_pins * NUM_ATTR);
    reset_val<index_type><<<BLOCK_NUMBER(num_pins * NUM_ATTR), BLOCK_SIZE>>>(at_prefix_arc, num_pins * NUM_ATTR);
    reset_val<index_type><<<BLOCK_NUMBER(num_pins * NUM_ATTR), BLOCK_SIZE>>>(at_prefix_attr, num_pins * NUM_ATTR);

    device_copy_batch<float><<<BLOCK_NUMBER(num_pins * NUM_ATTR), BLOCK_SIZE>>>(__pinSlew__, pinSlew, num_pins * NUM_ATTR);
    device_copy_batch<float><<<BLOCK_NUMBER(num_pins * NUM_ATTR), BLOCK_SIZE>>>(__pinLoad__, pinLoad, num_pins * NUM_ATTR);
    device_copy_batch<float><<<BLOCK_NUMBER(num_pins * NUM_ATTR), BLOCK_SIZE>>>(__pinRAT__, pinRAT, num_pins * NUM_ATTR);
    device_copy_batch<float><<<BLOCK_NUMBER(num_pins * NUM_ATTR), BLOCK_SIZE>>>(__pinAT__, pinAT, num_pins * NUM_ATTR);
    cudaDeviceSynchronize();
}

}  // namespace gt
