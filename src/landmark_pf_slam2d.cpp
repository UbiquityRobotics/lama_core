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

#include <iostream>

#include "lama/random.h"
#include "lama/thread_pool.h"

#include "lama/landmark_pf_slam2d.h"


lama::LandmarkPFSlam2D::LandmarkPFSlam2D(const Options& options)
    : options_(options)
{
    acc_trans_ = 0.0;
    acc_rot_   = 0.0;

    if (options_.threads <= 1){
        thread_pool_ = 0;
    } else {
        thread_pool_ = new ThreadPool;
        thread_pool_->init(options_.threads);
    }

    // handle rng seed
    if (options_.seed == 0)
        options_.seed = random::genSeed();

    random::setSeed(options_.seed);

    // initialize particles
    const uint32_t num_particles = options_.particles;
    particles_[0].resize(num_particles);
    current_particle_set_ = 0;

    for (auto& p : particles_[0])
        p.weight = 1.0 / num_particles;

    neff_ = num_particles;
    has_first_odom_ = false;
}

lama::LandmarkPFSlam2D::~LandmarkPFSlam2D()
{
    delete thread_pool_;
}

bool lama::LandmarkPFSlam2D::update(const DynamicArray<Landmark2D>& landmarks, const Pose2D& odometry, double timestamp)
{
    // 1. Predict from odometry
    Pose2D odelta = odom_ - odometry;
    odom_ = odometry;

    if (not has_first_odom_){
        has_first_odom_ = true;
        return true;
    }

    const uint32_t num_particles = options_.particles;
    for (uint32_t i = 0; i < num_particles; ++i)
        drawFromMotion(odelta, particles_[current_particle_set_][i].pose, particles_[current_particle_set_][i].pose);

    // only continue if the necessary motion was gathered.
    acc_trans_ += odelta.xy().norm();
    acc_rot_   += std::fabs(odelta.rotation());
    if (acc_trans_ <= options_.trans_thresh &&
            acc_rot_ <= options_.rot_thresh)
        return false;

    // do we have landmarks?
    if (landmarks.size() == 0)
        return false;

    acc_trans_ = 0;
    acc_rot_   = 0;

    if (thread_pool_){
        for (uint32_t i = 0; i < num_particles; ++i)
            thread_pool_->enqueue([this, i, &landmarks](){
                    updateParticleLandmarks(&particles_[current_particle_set_][i], landmarks);
                    });

        thread_pool_->wait();

    } else {

        for (uint32_t i = 0; i < num_particles; ++i){
            updateParticleLandmarks(&particles_[current_particle_set_][i], landmarks);
        } // end for

    } // end if

    normalize();

    // resample if needed
    if (neff_ < (options_.particles*0.5) )
        resample();

    return true;
}

size_t lama::LandmarkPFSlam2D::getBestParticleIdx() const
{
    const uint32_t num_particles = options_.particles;

    size_t best_idx = 0;
    double best_ws  = particles_[current_particle_set_][0].weight_sum;

    for (uint32_t i = 1; i < num_particles; ++i){

        if (best_ws < particles_[current_particle_set_][i].weight_sum){
            best_ws = particles_[current_particle_set_][i].weight_sum;
            best_idx = i;
        }
    }

    return best_idx;
}

void lama::LandmarkPFSlam2D::setPrior(const Pose2D& prior)
{
    for(auto& particle : particles_[current_particle_set_])
        particle.pose = prior;
}

lama::Pose2D lama::LandmarkPFSlam2D::getPose() const
{
    size_t pidx = getBestParticleIdx();
    return particles_[current_particle_set_][pidx].pose;
}

void lama::LandmarkPFSlam2D::drawFromMotion(const Pose2D& delta, const Pose2D& old_pose, Pose2D& pose)
{
    double sigma, x, y, yaw;
    double sxy = 0.3 * options_.srr;

    sigma = options_.srr * std::fabs(delta.x())   +
            options_.str * std::fabs(delta.rotation()) +
            sxy * std::fabs(delta.y());

    x = delta.x() + random::normal(sigma);

    sigma = options_.srr * std::fabs(delta.y())   +
            options_.str * std::fabs(delta.rotation()) +
            sxy * std::fabs(delta.x());

    y = delta.y() + random::normal(sigma);

    sigma = options_.stt * std::fabs(delta.rotation()) +
            options_.srt * delta.xy().norm();

    yaw = delta.rotation() + random::normal(sigma);
    yaw = std::fmod(yaw, 2*M_PI);
    if (yaw > M_PI)
        yaw -= 2*M_PI;

    pose += Pose2D(x, y, yaw);
}

void lama::LandmarkPFSlam2D::updateParticleLandmarks(Particle* particle, const DynamicArray<Landmark2D>& landmarks)
{
    for (const auto& landmark : landmarks){

        uint32_t id = landmark.id;
        auto map_lm = particle->landmarks.find(id);

        if ( map_lm == particle->landmarks.end() ){
            // This is the first time the landmarks is observed.
            MapLandmark new_lm;

            // Landmark in map coordinates
            new_lm.mu = landmark.toCartesian(particle->pose);

            // Calculate covariance
            Vector3d h; Matrix3d H;
            landmark.predict(particle->pose, new_lm.mu, h, H);

            Matrix3d Hi = H.inverse();
            new_lm.sigma = Hi * landmark.sigma * Hi.transpose();

            // add the landmarks to the map
            particle->landmarks[id] = new_lm;
        } else {

            // abbreviation
            Matrix3d& sig = map_lm->second.sigma;

            // Predicted landmark
            Vector3d h; Matrix3d H;
            landmark.predict(particle->pose, map_lm->second.mu, h, H);

            // inovation
            Vector3d diff = landmark.diff(h);

            // Compatibility test with the Mahalanobis distance.
            bool is_compatible = true;
            if (options_.do_compatibility_test){
                double d2 = diff.dot(sig.inverse() * diff); // squared Mahalanobis distance
                constexpr double nsigma  = 11.34 * 11.34;   // 3dof sigma
                if ( d2 > nsigma ){
                    is_compatible = false; // do not update
                }
            }//end if compatibility test

            // Kalman gain
            Matrix3d Q = H * sig * H.transpose() + landmark.sigma;
            Matrix3d K = sig * H.transpose() * Q.inverse();

            // Update landmark state
            map_lm->second.sigma = map_lm->second.sigma - K * H * sig;
            if (is_compatible)
                map_lm->second.mu = map_lm->second.mu + K * diff;

            // Calculate weight
            double w = -0.5 * diff.transpose() * Q.inverse() * diff - std::log(std::sqrt(2.0 * M_PI * Q.determinant()));
            particle->weight += w;
            particle->weight_sum += w;
        }

    }// end for

}

void lama::LandmarkPFSlam2D::normalize()
{
    double gain = 1.0 / (options_.meas_sigma_gain * options_.particles);
    double max_l  = particles_[current_particle_set_][0].weight;
    const uint32_t num_particles = options_.particles;
    for (uint32_t i = 1; i < num_particles; ++i)
        if (max_l < particles_[current_particle_set_][i].weight)
            max_l = particles_[current_particle_set_][i].weight;

    double sum = 0;
    for (uint32_t i = 0; i < num_particles; ++i){

        particles_[current_particle_set_][i].normalized_weight = std::exp(gain*(particles_[current_particle_set_][i].weight - max_l));
        sum += particles_[current_particle_set_][i].normalized_weight;
    }

    neff_ = 0;
    for (uint32_t i = 0; i < num_particles; ++i){
        particles_[current_particle_set_][i].normalized_weight /= sum;
        neff_ += particles_[current_particle_set_][i].normalized_weight * particles_[current_particle_set_][i].normalized_weight;
    }

    neff_ = 1.0 / neff_;
}

void lama::LandmarkPFSlam2D::resample()
{
    const uint32_t num_particles = options_.particles;
    std::vector<int32_t> sample_idx(num_particles);

    double interval = 1.0 / (double)num_particles;

    double target = interval * random::uniform();
    double   cw  = 0.0;
    uint32_t n   = 0;
    for (size_t i = 0; i < num_particles; ++i){
        cw += particles_[current_particle_set_][i].normalized_weight;

        while( cw > target){
            sample_idx[n++]=i;
            target += interval;
        }
    }

    // generate a new set of particles

    uint8_t ps = 1 - current_particle_set_;
    particles_[ps].resize(num_particles);

    for (size_t i = 0; i < num_particles; ++i) {
        uint32_t idx = sample_idx[i];;

        // copy the particle, assumes a copy operator
        particles_[ps][i] = particles_[current_particle_set_][ idx ];
        particles_[ps][i].weight  = 0.0;
    }

    particles_[current_particle_set_].clear();
    current_particle_set_ = ps;
}

