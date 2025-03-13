

#include "GPUTimer.h"
#include "gputiming.h"
#include "utils.cuh"

namespace gt {

// GPULutAllocator *d_allocator;

void GPUTimer::initialize() {
    // cudaMalloc(&pinSlew, num_pins * NUM_ATTR * sizeof(float));
    // cudaMalloc(&pinLoad, num_pins * NUM_ATTR * sizeof(float));
    cudaMalloc(&pinCap, num_pins * (NUM_ATTR + 2) * sizeof(float));
    cudaMalloc(&pinWireCap, num_pins * NUM_ATTR * sizeof(float));
    // cudaMalloc(&pinRat, num_pins * NUM_ATTR * sizeof(float));
    // cudaMalloc(&pinAt, num_pins * NUM_ATTR * sizeof(float));

    cudaMalloc(&testRelatedAT, num_tests * NUM_ATTR * sizeof(float));
    cudaMalloc(&testRAT, num_tests * NUM_ATTR * sizeof(float));
    cudaMalloc(&testConstraint, num_tests * NUM_ATTR * sizeof(float));
    // cudaMalloc(&pinImpulse, num_pins * NUM_ATTR * sizeof(float));
    // cudaMalloc(&pinRootDelay, num_pins * NUM_ATTR * sizeof(float));
    cudaMalloc(&pinRootRes, num_pins * NUM_ATTR * sizeof(float));
    // cudaMalloc(&arcDelay, num_arcs * 2 * NUM_ATTR * sizeof(float));
    cudaMalloc(&arcSlew, num_arcs * 2 * NUM_ATTR * sizeof(float));

    // cudaMalloc(&pin_f_arc_list_end, (num_pins + 1) * sizeof(index_type));
    // cudaMalloc(&pin_f_arc_list, num_arcs * sizeof(index_type));
    // cudaMalloc(&pin_b_arc_list_end, (num_pins + 1) * sizeof(index_type));
    // cudaMalloc(&pin_b_arc_list, num_arcs * sizeof(index_type));
    // cudaMalloc(&arc_from_pin, num_arcs * sizeof(index_type));
    // cudaMalloc(&arc_to_pin, num_arcs * sizeof(index_type));
    // cudaMalloc(&pin_num_fanin, num_pins * sizeof(int));
    // cudaMalloc(&pin_fanout_list_end, (num_pins + 1) * sizeof(index_type));
    // cudaMalloc(&pin_fanout_list, num_fanout_pins * sizeof(index_type));
    
    // cudaMalloc(&arc_types, num_arcs * sizeof(int));
    // cudaMalloc(&arc_timings, 2 * num_arcs * sizeof(int));
    // cudaMalloc(&arc_tests, num_arcs * sizeof(int));

    cudaMalloc(&test_to_arc, num_tests * sizeof(int));
    cudaMalloc(&net_is_clock, num_nets * sizeof(int));
    cudaMalloc(&level_list, num_pins * sizeof(int));
    cudaMalloc(&pin_outs, num_POs * sizeof(index_type));

    cudaMemcpy(pinCap, gtdb.pin_capacitance.data(), num_pins * (NUM_ATTR + 2) * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(pin_f_arc_list_end, gtdb.pin_f_arc_list_end.data(), (num_pins + 1) * sizeof(index_type), cudaMemcpyHostToDevice);
    // cudaMemcpy(pin_f_arc_list, gtdb.pin_f_arc_list.data(), num_arcs * sizeof(index_type), cudaMemcpyHostToDevice);
    // cudaMemcpy(pin_b_arc_list_end, gtdb.pin_b_arc_list_end.data(), (num_pins + 1) * sizeof(index_type), cudaMemcpyHostToDevice);
    // cudaMemcpy(pin_b_arc_list, gtdb.pin_b_arc_list.data(), num_arcs * sizeof(index_type), cudaMemcpyHostToDevice);
    // cudaMemcpy(arc_from_pin, gtdb.arc_from_pin.data(), num_arcs * sizeof(index_type), cudaMemcpyHostToDevice);
    // cudaMemcpy(arc_to_pin, gtdb.arc_to_pin.data(), num_arcs * sizeof(index_type), cudaMemcpyHostToDevice);
    // cudaMemcpy(pin_num_fanin, gtdb.pin_num_fanin.data(), num_pins * sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(pin_fanout_list_end, gtdb.pin_fanout_list_end.data(), (num_pins + 1) * sizeof(index_type), cudaMemcpyHostToDevice);
    // cudaMemcpy(pin_fanout_list, gtdb.pin_fanout_list.data(), num_fanout_pins * sizeof(index_type), cudaMemcpyHostToDevice);

    // cudaMemcpy(arc_types, gtdb.arc_types.data(), num_arcs * sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(arc_timings, gtdb.arc_timings.data(), 2 * num_arcs * sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(arc_tests, gtdb.arc_tests.data(), num_arcs * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(test_to_arc, gtdb.test_to_arc.data(), num_tests * sizeof(int), cudaMemcpyHostToDevice);
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
    allocator->AllocateBatch(gtdb.timings);;
    allocator->CopyToGPU();
    cudaMalloc((void **)&d_allocator, sizeof(GPULutAllocator));
    cudaMemcpy(d_allocator, allocator, sizeof(GPULutAllocator), cudaMemcpyHostToDevice);
    allocator->CopyToGPU(d_allocator);
    
    printf("GPUTimer initialized\n");

    cudaMalloc(&__pinSlew__, num_pins * NUM_ATTR * sizeof(float));
    cudaMalloc(&__pinLoad__, num_pins * NUM_ATTR * sizeof(float));
    cudaMalloc(&__pinRat__, num_pins * NUM_ATTR * sizeof(float));
    cudaMalloc(&__pinAt__, num_pins * NUM_ATTR * sizeof(float));

    device_copy_batch<float><<<BLOCK_NUMBER(num_pins * NUM_ATTR), BLOCK_SIZE>>>(pinSlew, __pinSlew__, num_pins * NUM_ATTR);
    device_copy_batch<float><<<BLOCK_NUMBER(num_pins * NUM_ATTR), BLOCK_SIZE>>>(pinLoad, __pinLoad__, num_pins * NUM_ATTR);
    device_copy_batch<float><<<BLOCK_NUMBER(num_pins * NUM_ATTR), BLOCK_SIZE>>>(pinRat, __pinRat__, num_pins * NUM_ATTR);
    device_copy_batch<float><<<BLOCK_NUMBER(num_pins * NUM_ATTR), BLOCK_SIZE>>>(pinAt, __pinAt__, num_pins * NUM_ATTR);

    // cudaMemcpy(__pinSlew__, gtdb.pinSlew.data(), num_pins * NUM_ATTR * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(__pinLoad__, gtdb.pinLoad.data(), num_pins * NUM_ATTR * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(__pinRat__, gtdb.pinRat.data(), num_pins * NUM_ATTR * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(__pinAt__, gtdb.pinAt.data(), num_pins * NUM_ATTR * sizeof(float), cudaMemcpyHostToDevice);
}

GPUTimer::~GPUTimer() {
    logger.info("destruct GPUTimer");

    // cudaFree(pinSlew);
    // cudaFree(pinLoad);
    cudaFree(pinCap);
    cudaFree(pinWireCap);
    // cudaFree(pinRat);
    // cudaFree(pinAt);

    cudaFree(testRelatedAT);
    cudaFree(testRAT);
    cudaFree(testConstraint);
    // cudaFree(pinImpulse);
    // cudaFree(pinRootDelay);
    cudaFree(pinRootRes);
    // cudaFree(arcDelay);
    cudaFree(arcSlew);

    // cudaFree(pin_f_arc_list_end);
    // cudaFree(pin_f_arc_list);
    // cudaFree(pin_b_arc_list_end);
    // cudaFree(pin_b_arc_list);
    // cudaFree(arc_from_pin);
    // cudaFree(arc_to_pin);
    // cudaFree(pin_num_fanin);
    // cudaFree(pin_fanout_list_end);
    // cudaFree(pin_fanout_list);

    // cudaFree(arc_types);
    // cudaFree(arc_timings);
    // cudaFree(arc_tests);

    cudaFree(test_to_arc);
    cudaFree(net_is_clock);
    cudaFree(level_list);
    cudaFree(pin_outs);

    cudaFree(__pinSlew__);
    cudaFree(__pinLoad__);
    cudaFree(__pinRat__);
    cudaFree(__pinAt__);

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
    device_copy_batch<float><<<BLOCK_NUMBER(num_pins * NUM_ATTR), BLOCK_SIZE>>>(__pinRat__, pinRat, num_pins * NUM_ATTR);
    device_copy_batch<float><<<BLOCK_NUMBER(num_pins * NUM_ATTR), BLOCK_SIZE>>>(__pinAt__, pinAt, num_pins * NUM_ATTR);
    cudaDeviceSynchronize();
}


__global__ void update_endpoints_kernel0(float *pinAt, float *testRAT, int *test_to_arc, index_type *arc_from_pin, index_type *arc_to_pin, float *endpoints0, int num_tests) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int test_idx = idx >> 2;
    const int i = idx & 0b11;
    const int el = i >> 1;
    const int rf = i & 1;
    if (test_idx < num_tests) {
        const int arc_id = test_to_arc[test_idx];
        const int from_pin_id = arc_from_pin[arc_id];
        const int to_pin_id = arc_to_pin[arc_id];
        // printf("test idx %d arc id %d from %d to %d\n", test_idx, arc_id, from_pin_id, to_pin_id);
        // printf("test pin idx %d at %.5f rat %.5f\n", to_pin_id, pinAt[to_pin_id * NUM_ATTR + i], testRAT[test_idx *
        // NUM_ATTR + i]);
        if (isnan(pinAt[to_pin_id * NUM_ATTR + i]) || isnan(testRAT[test_idx * NUM_ATTR + i])) return;
        if (el == 0) {
            endpoints0[test_idx * NUM_ATTR + i] = pinAt[to_pin_id * NUM_ATTR + i] - testRAT[test_idx * NUM_ATTR + i];
        } else {
            endpoints0[test_idx * NUM_ATTR + i] = testRAT[test_idx * NUM_ATTR + i] - pinAt[to_pin_id * NUM_ATTR + i];
        }
    }
}

__global__ void update_endpoints_kernel1(float *pinAt, float *pinRat, index_type *pin_outs, float *endpoints1, int num_POs) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int po_idx = idx >> 2;
    const int i = idx & 0b11;
    const int el = i >> 1;
    if (po_idx < num_POs) {
        const int pin_idx = pin_outs[po_idx];
        // printf("po pin idx %d at %.5f rat %.5f\n", pin_idx, pinAt[pin_idx * NUM_ATTR + i], pinRat[pin_idx * NUM_ATTR
        // + i]);
        if (isnan(pinAt[pin_idx * NUM_ATTR + i]) || isnan(pinRat[pin_idx * NUM_ATTR + i])) return;
        if (el == 0) {
            endpoints1[po_idx * NUM_ATTR + i] = pinAt[pin_idx * NUM_ATTR + i] - pinRat[pin_idx * NUM_ATTR + i];
        } else {
            endpoints1[po_idx * NUM_ATTR + i] = pinRat[pin_idx * NUM_ATTR + i] - pinAt[pin_idx * NUM_ATTR + i];
        }
    }
}

void GPUTimer::update_endpoints() {
    torch::Tensor endpoints0 = torch::zeros({num_tests, NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::kCUDA)).contiguous();
    torch::Tensor endpoints1 = torch::zeros({num_POs, NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::kCUDA)).contiguous();
    torch::fill_(endpoints0, nanf(""));
    torch::fill_(endpoints1, nanf(""));

    update_endpoints_kernel0<<<BLOCK_NUMBER(num_tests * NUM_ATTR), BLOCK_SIZE>>>(pinAt, testRAT, test_to_arc, arc_from_pin, arc_to_pin, endpoints0.data_ptr<float>(), num_tests);
    update_endpoints_kernel1<<<BLOCK_NUMBER(num_POs * NUM_ATTR), BLOCK_SIZE>>>(pinAt, pinRat, pin_outs, endpoints1.data_ptr<float>(), num_POs);

    endpoints = torch::cat({endpoints0, endpoints1}, 0).contiguous();
}

}  // namespace gt
