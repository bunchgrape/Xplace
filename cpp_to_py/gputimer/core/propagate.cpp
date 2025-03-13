
#include "GPUTimer.h"

namespace gt {

void update_timing_cuda(index_type *level_list,
                        vector<int> level_list_end_cpu,
                        index_type *pin_f_arc_list_end,
                        index_type *pin_f_arc_list,
                        index_type *arc_to_pin,
                        index_type *pin_b_arc_list_end,
                        index_type *pin_b_arc_list,
                        index_type *arc_from_pin,
                        int *arc_types,
                        int *arc_tests,
                        float *pinSlew,
                        float *pinLoad,
                        float *pinImpulse,
                        float *pinRootDelay,
                        float *pinAt,
                        float *pinRat,
                        float *testRelatedAT,
                        float *testRAT,
                        float *testConstraint,
                        float *arcDelay,
                        int *arc_timings,
                        index_type *at_prefix_pin,
                        index_type *at_prefix_arc,
                        index_type *at_prefix_attr,
                        float clock_period,
                        GPULutAllocator *d_allocator,
                        int num_pins,
                        bool deterministic);


void GPUTimer::update_timing() {
    update_timing_cuda(level_list,
                       level_list_end_cpu,
                       pin_f_arc_list_end,
                       pin_f_arc_list,
                       arc_to_pin,
                       pin_b_arc_list_end,
                       pin_b_arc_list,
                       arc_from_pin,
                       arc_types,
                       arc_tests,
                       pinSlew,
                       pinLoad,
                       pinImpulse,
                       pinRootDelay,
                       pinAt,
                       pinRat,
                       testRelatedAT,
                       testRAT,
                       testConstraint,
                       arcDelay,
                       arc_timings,
                       at_prefix_pin,
                       at_prefix_arc,
                       at_prefix_attr,
                       clock_period,
                       d_allocator,
                       num_pins,
                       true);
}

}  // namespace gt