/*
 * IRIS Localization and Mapping (LaMa)
 *
 * Copyright (c) 2020, Eurico Pedrosa, University of Aveiro - Portugal
 * Copyright (c) 2020, Ubiquity Robotics
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

#pragma once

#include <memory>
#include <vector>

#include "lama/landmark2d.h"

namespace lama {

struct ThreadPool;

class LandmarkPFSlam2D {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    struct MapLandmark {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        Vector3d mu;
        Matrix3d sigma;
    };

    struct Particle {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        // The weight of the particle
        double weight = 0.0;

        double normalized_weight = 0.0;

        double weight_sum = 0.0;

        // The pose of this particle in the map
        Pose2D pose;

        // The landmark map.
        Dictionary<uint32_t, MapLandmark> landmarks;

        // history
        //DynamicArray<Pose2D> poses;
    };

    // All SLAM options are in one place for easy access and passing.
    struct Options {
        Options(){}

        /// The number of particles to use
        uint32_t particles;
        /// How much the rotation affects rotation.
        double srr = 0.1;
        /// How much the translation affects rotation.
        double str = 0.2;
        /// How much the translation affects translation.
        double stt = 0.1;
        /// How much the rotation affects translation.
        double srt = 0.2;
        /// Measurement confidence.
        double meas_sigma = 0.05;
        /// Use this to smooth the measurements likelihood.
        double meas_sigma_gain = 3;
        /// The ammount of displacement that the system must
        /// gather before any update takes place.
        double trans_thresh = 0.5;
        /// The ammout of rotation that the system must
        /// gather before any update takes place.
        double rot_thresh = 0.5;
        /// Number of working threads.
        /// -1 for none, 0 for auto, >0 user define number of workers.
        int32_t threads = -1;
        /// Pseudo random generator seed.
        /// Use 0 to generate a new seed.
        uint32_t seed = 0;
        /// Do compatibility test* using the Mahalanobis distance.
        bool do_compatibility_test = true;
    };

    LandmarkPFSlam2D(const Options& options = Options());
    virtual ~LandmarkPFSlam2D();

    bool update(const DynamicArray<Landmark2D>& landmarks, const Pose2D& odometry, double timestamp);

    size_t getBestParticleIdx() const;

    // Set the prior pose for all particle.
    // Make sure you call this function before any update occurs.
    void setPrior(const Pose2D& prior);

    // Get the pose of the best particle.
    Pose2D getPose() const;

    inline const std::vector<Particle>& getParticles() const
    { return particles_[current_particle_set_]; }

    inline double getNeff() const
    { return neff_; }


private:

    void drawFromMotion(const Pose2D& delta, const Pose2D& old_pose, Pose2D& pose);

    void updateParticleLandmarks(Particle* particle, const DynamicArray<Landmark2D>& landmarks);

    void normalize();
    void resample();

private:
    Options options_;

    DynamicArray<Particle> particles_[2];
    uint8_t current_particle_set_;

    Pose2D odom_;
    bool has_first_odom_;

    double acc_trans_;
    double acc_rot_;
    double neff_;

    ThreadPool* thread_pool_;
};

} /* lama */

