/*
 * IRIS Localization and Mapping (LaMa)
 *
 * Copyright (c) 2021, Eurico Pedrosa, University of Aveiro - Portugal
 * Copyright (c) 2021, Ubiquity Robotics
 * All rights reserved.
 * License: New BSD
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the copyright holder nor the names of its
 *       contributors may be used to endorse or promote products derived from
 *       this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include "types.h"

namespace lama {


// Adaptive size of a sample set.
//
// This class uses the KL-Divergence to determine when a distribution
// has been adequately sampled.
//
// Fox, Dieter. "KLD-sampling: Adaptive particle filters."
// Advances in neural information processing systems. 2002.
struct KLDSampling {

    // Population size control parameters.
    //
    // The max error between the true distribution and the estimate destribution.
    double pop_error;
    // The upper standard normal quantile for (1 - p), where p is the probability
    // that the error of the estimated destribution will be less then @pop_error.
    double pop_z;
    // Minimum number of samples.
    uint32_t samples_min;
    // Maximum number of samples.
    uint32_t samples_max;

    // Adaptive number of samples
    uint32_t kld_samples;

    // cache for the number of samples given the number of bins.
    DynamicArray<uint32_t> cache;

    // Forced minimum number of samples.
    // The number of samples can never be lower than this.
    static const uint32_t forced_samples_min = 10;

    // Bins where the samples fall.
    DynamicArray< Array<double, 3> > bins;

    // Initiate the sampler with the necessary parameters.
    void init(uint32_t min, uint32_t max, double error = 0.01, double z = 3);

    // Initiate a new sampling round.
    void reset();

    // Calculate the resample limit, i.e. the adequate number of samples).
    uint32_t resample_limit(const Array<double, 3>& sample);

};

} // namespace lama

