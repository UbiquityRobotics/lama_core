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

    for (auto& p : particles_[0]){
        p.weight = 1.0 / num_particles;
        p.map = SimpleLandmark2DMapPtr(new SimpleLandmark2DMap);
    }

    kld_.init(num_particles, options_.max_particles, 0.04);

    neff_ = num_particles;
    has_first_odom_ = false;
}

lama::LandmarkPFSlam2D::~LandmarkPFSlam2D()
{
    delete thread_pool_;
}

bool lama::LandmarkPFSlam2D::update(const DynamicArray<Landmark>& landmarks, const Pose2D& odometry, double timestamp)
{
    // 1. Predict from odometry
    Pose2D odelta = odom_ - odometry;
    odom_ = odometry;

    if (not has_first_odom_){
        has_first_odom_ = true;
        return true;
    }

    const uint32_t num_particles = particles_[current_particle_set_].size();
    for (uint32_t i = 0; i < num_particles; ++i)
        drawFromMotion(odelta, particles_[current_particle_set_][i].pose, particles_[current_particle_set_][i].pose);

    // only continue if the necessary motion was gathered.
    acc_trans_ += odelta.xy().norm();
    acc_rot_   += std::fabs(odelta.rotation());
    if (acc_trans_ <= options_.trans_thresh &&
            acc_rot_ <= options_.rot_thresh)
        return false;

    // do we have landmarks?
    if (landmarks.size() == 0){

        // Force a resample if the accumulated motion is too large.
        // The objective is to increase the number of particles due to
        // accumulated error by the motion (i.e. odometry).
        if ((acc_trans_ > 1.0) || (acc_rot_ > M_PI * 0.5)){
            normalize();
            resample(false);

            clusterStats();

            acc_trans_ = options_.trans_thresh;
            acc_rot_   = options_.rot_thresh;

            return true;
        }

        return false;
    }

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
    if (neff_ < (num_particles*0.5) ){
        resample();
    } else {
        kdtree_.reset(num_particles);
        for (auto& p : particles_[current_particle_set_])
            kdtree_.insert(p.pose);
    }

    // Calculate clusters
    clusterStats();

    return true;
}

size_t lama::LandmarkPFSlam2D::getBestParticleIdx() const
{
    const uint32_t num_particles = particles_[current_particle_set_].size();

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
    if (clusters_.size() == 0)
        return particles_[current_particle_set_][0].pose;

    double best = clusters_[0].weight;
    size_t idx  = 0;
    for (size_t i = 1; i < clusters_.size(); ++i)
        if (clusters_[i].weight > best){
            best = clusters_[i].weight;
            idx = i;
        }

    return clusters_[idx].pose;
}

Eigen::Matrix3d lama::LandmarkPFSlam2D::getCovar() const
{
    if (clusters_.size() == 0){
        Matrix3d covar = Matrix3d::Identity() * 999;
        return covar;
    }

    double best = clusters_[0].weight;
    size_t idx  = 0;
    for (size_t i = 1; i < clusters_.size(); ++i)
        if (clusters_[i].weight > best){
            best = clusters_[i].weight;
            idx = i;
        }

    return clusters_[idx].covar;
}

void lama::LandmarkPFSlam2D::drawFromMotion(const Pose2D& delta, const Pose2D& old_pose, Pose2D& pose)
{
    double sigma, x, y, yaw;
    double sxy = 0.3 * options_.stt;

    sigma = options_.stt * std::fabs(delta.x())   +
            options_.srt * std::fabs(delta.rotation()) +
            sxy * std::fabs(delta.y());

    x = delta.x() + random::normal(sigma);

    sigma = options_.stt * std::fabs(delta.y())   +
            options_.srt * std::fabs(delta.rotation()) +
            sxy * std::fabs(delta.x());

    y = delta.y() + random::normal(sigma);

    sigma = options_.srr * std::fabs(delta.rotation()) +
            options_.str * delta.xy().norm();

    yaw = delta.rotation() + random::normal(sigma);
    yaw = std::fmod(yaw, 2*M_PI);
    if (yaw > M_PI)
        yaw -= 2*M_PI;

    pose += Pose2D(x, y, yaw);
}

void lama::LandmarkPFSlam2D::updateParticleLandmarks(Particle* particle, const DynamicArray<Landmark>& landmarks)
{
    // Get the particle pose in 3D
    auto xyr = particle->pose.xyr();
    Pose3D pose(Vector3d(xyr(0), xyr(1), 0.0), xyr(2));

    for (const auto& landmark : landmarks){

        uint32_t id = landmark.id;
        auto* lm = particle->map->get(id);

        if ( lm == nullptr ){
            // This is the first time the landmarks is observed.
            lm = particle->map->alloc(id);

            // Landmark in map coordinates
            lm->state = pose + Pose3D(landmark.measurement);

            // Calculate covariance
            Matrix6d H;
            H.setIdentity();
            H.topLeftCorner<3,3>() = pose.state.rotationMatrix().transpose();

            Matrix6d Hi = H.inverse();
            lm->covar = Hi * landmark.covar * Hi.transpose();
        } else {

            // abbreviation
            Matrix6d& sig = lm->covar;

            // Predicted landmark
            auto h = pose - lm->state;

            Matrix6d H; H.setIdentity();
            H.topLeftCorner<3,3>() = pose.state.rotationMatrix().transpose();

            // Update landmark covariance
            Matrix6d Q = H * sig * H.transpose() + landmark.covar;
            Matrix6d Qi = Q.inverse();

            // inovation
            auto inov = h - landmark.measurement;
            Vector6d diff;
            diff << landmark.measurement.head<3>() - h.xyz(),
                    inov.state.so3().log();

            // Compatibility test with the Mahalanobis distance.
            bool is_compatible = true;
            if (options_.do_compatibility_test){
                double d2 = diff.transpose() * Qi * diff; // squared Mahalanobis distance
                constexpr double nsigma  = 16.8119 * 16.8119;   // 3dof 99% sigma
                if ( d2 > nsigma ){
                    is_compatible = false; // do not update
                }
            }//end if compatibility test

            // Kalman gain
            Matrix6d K = sig * H.transpose() * Qi;

            // Update landmark state
            if (is_compatible){
                lm->covar = sig - K * H * sig;

                Vector6d s = K * diff;
                lm->state.state.translation() += s.head<3>();
                lm->state.state.so3() = lm->state.state.so3() * SO3d::exp(s.tail<3>());
            }

            // Calculate weight (or likelihood)
            // The likelihood is the log of a non normalized multivariant gaussian pdf.
            // The normalizer is not needed. The final result is the same and we save some caluculation.
            double w = -0.5 * diff.transpose() * Q.inverse() * diff;

            particle->weight += w;
            particle->weight_sum += w;
        }

    }// end for

}

void lama::LandmarkPFSlam2D::normalize()
{
    const uint32_t num_particles = particles_[current_particle_set_].size();

    double gain = 1.0 / (options_.meas_sigma_gain * num_particles);
    double max_l  = particles_[current_particle_set_][0].weight;
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

void lama::LandmarkPFSlam2D::resample(bool reset_weight)
{
    const uint32_t num_particles = particles_[current_particle_set_].size();
    DynamicArray<double> c(num_particles + 1);

    c[0] = 0.0;
    for (size_t i = 0; i < num_particles; ++i)
        c[i+1] = c[i] + particles_[current_particle_set_][i].normalized_weight;

    uint8_t ps = 1 - current_particle_set_;
    particles_[ps].reserve(num_particles);

    kdtree_.reset(kld_.samples_max);
    kld_.reset();
    for (size_t i = 0; i < kld_.samples_max; ++i){

        double r = random::uniform();
        uint32_t idx;
        for (idx = 0; idx < num_particles; ++idx)
            if ((c[idx] <= r) && (r < c[idx+1]))
                break;

        particles_[ps].emplace_back( Particle{} );
        particles_[ps].back() = particles_[current_particle_set_][idx];

        if (reset_weight)
            particles_[ps].back().weight = 1.0;

        particles_[ps].back().map = SimpleLandmark2DMapPtr(new SimpleLandmark2DMap(*(particles_[current_particle_set_][idx].map)));

        auto& pose = particles_[ps].back().pose;

        kdtree_.insert(pose);
        auto kld_samples = kld_.resample_limit(kdtree_.num_leafs);

        if (particles_[ps].size() >= kld_samples) break;
    }

    particles_[current_particle_set_].clear();
    current_particle_set_ = ps;
}

void lama::LandmarkPFSlam2D::clusterStats()
{
    kdtree_.cluster();

    clusters_.clear();
    clusters_.resize(kdtree_.num_clusters);

    for (auto& particle : particles_[current_particle_set_]){

        auto cidx = kdtree_.getCluster(particle.pose);
        if ( cidx == -1 )
            continue;

        Cluster& cluster = clusters_[cidx];

        // accumulate the weight
        cluster.weight += particle.normalized_weight;

        // compute mean
        cluster.m[0] += particle.normalized_weight * particle.pose.x();
        cluster.m[1] += particle.normalized_weight * particle.pose.y();
        cluster.m[2] += particle.normalized_weight * std::cos(particle.pose.rotation());
        cluster.m[3] += particle.normalized_weight * std::sin(particle.pose.rotation());

        // compute covar of linear compomenets
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                cluster.c(i,j) += particle.normalized_weight * particle.pose.xyr()(i) * particle.pose.xyr()(j);

    }// end for

    // Normalize each cluster
    for (auto& cluster : clusters_){

        // First the mean
        Vector3d mean;
        mean << cluster.m[0] / cluster.weight,
                cluster.m[1] / cluster.weight,
                std::atan2(cluster.m[3], cluster.m[2]);

        cluster.pose = Pose2D(mean);

        // Now the covariance for the linear components
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                cluster.covar(i,j) = cluster.c(i,j) / cluster.weight - mean(i) * mean(j);

        // Circular covariance
        cluster.covar(2,2) = -2 * std::log(cluster.m.tail<2>().norm());
    }
}

