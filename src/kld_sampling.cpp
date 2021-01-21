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

#include <cmath>

#include "lama/kld_sampling.h"

void lama::KLDSampling::init(uint32_t min, uint32_t max, double error, double z)
{
    samples_min = std::max(min, forced_samples_min);
    samples_max = std::max(samples_min, max);

    pop_error = error;
    pop_z     = z;

    cache.clear();
    cache.resize(samples_max, 0);
}

void lama::KLDSampling::reset()
{
    bins.clear();
    kld_samples = samples_min;
}

uint32_t lama::KLDSampling::resample_limit(const Array<double, 3>& sample)
{
    Array<double, 3> current_bin;
    current_bin[0] = std::floor(sample[0]/0.5);
    current_bin[1] = std::floor(sample[1]/0.5);
    current_bin[2] = std::floor(sample[2]/(10*M_PI/180.0));

    const uint32_t num_bins = bins.size();
    for (uint32_t i = 0; i < num_bins; ++i)
        if (current_bin == bins[i]){
            // sample already in a bin
            return kld_samples;
        }// end if (current_bin == bins[i])

    // save the new bin
    bins.push_back(current_bin);

    if (cache[num_bins] > 0)
        return cache[num_bins];

    if (num_bins + 1 > 2){
        double a = 1.0;
        double b = 2.0 / (9.0 * ((double) num_bins));
        double c = std::sqrt(2.0 / (9.0 * ((double)num_bins))) * pop_z;
        double x = a - b + c;

        uint32_t n = (uint32_t) std::ceil(num_bins / (2.0 * pop_error) * x * x * x);

        n = std::min(std::max(n, samples_min), samples_max);
        if (n > kld_samples)
            kld_samples = n;
    }// end if (support_samples > 2)

    return kld_samples;
}

uint32_t lama::KLDSampling::resample_limit(uint32_t k)
{
    if (cache[k-1] > 0)
        return cache[k-1];

    if (k > 2){
        double a = 1.0;
        double b = 2.0 / (9.0 * ((double) (k-1)));
        double c = std::sqrt(2.0 / (9.0 * ((double)(k-1)))) * pop_z;
        double x = a - b + c;

        uint32_t n = (uint32_t) std::ceil((k-1) / (2.0 * pop_error) * x * x * x);

        n = std::min(std::max(n, samples_min), samples_max);
        if (n > kld_samples)
            kld_samples = n;
    }// end if (support_samples > 2)

    return kld_samples;
}
