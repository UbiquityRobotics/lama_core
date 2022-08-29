/*
 * IRIS Localization and Mapping (LaMa)
 *
 * Copyright (c) 2019-today, Eurico Pedrosa, University of Aveiro - Portugal
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
 *     * Neither the name of the University of Aveiro nor the names of its
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

#include <fstream>
#include <iostream>

#include "lama/print.h"
#include "lama/thread_pool.h"

#include "lama/random.h"

#include "lama/nlls/gauss_newton.h"
#include "lama/nlls/levenberg_marquardt.h"

#include "lama/hybrid_pf_slam2d.h"
#include "lama/match_surface_2d.h"

#include "lama/gnss.h"

#include "lama/sdm/export.h"

std::string lama::HybridPFSlam2D::Summary::report() const
{
    std::string report = format("\n LaMa PF Slam2D - Report\n"
                                " =======================\n");

    Eigen::Map<const VectorXd> v_t(&time[0], time.size());
    Eigen::Map<const VectorXd> v_ts(&time_solving[0], time_solving.size());
    Eigen::Map<const VectorXd> v_tn(&time_normalizing[0], time_normalizing.size());
    Eigen::Map<const VectorXd> v_tr(&time_resampling[0], time_resampling.size());
    Eigen::Map<const VectorXd> v_tm(&time_mapping[0], time_mapping.size());
    Eigen::Map<const VectorXd> v_mem(&memory[0], memory.size());

    auto time_span = v_t.sum();
    auto time_mean = v_t.mean();

    auto stampdiff = timestamp.back() - timestamp.front();

    report += format(" Number of updates     %ld\n"
                     " Number of resamples   %ld\n"
                     " Max memory usage      %.2f MiB\n"
                     " Problem time span     %d minute(s) and %d second(s)\n"
                     " Execution time span   %d minute(s) and %d second(s)\n"
                     " Execution frequency   %.2f Hz\n"
                     " Realtime factor       %.2fx\n",
                     time.size(), time_resampling.size(), v_mem.maxCoeff() / 1024.0 / 1024.0,
                     ((uint32_t)stampdiff) / 60, ((uint32_t)stampdiff) % 60,
                     ((uint32_t)time_span) / 60, ((uint32_t)time_span) % 60,
                     (1.0 / time_mean), stampdiff / time_span );


    auto v_t_std  = std::sqrt((v_t.array() - v_t.mean()).square().sum()/(v_t.size()-1));
    auto v_ts_std = std::sqrt((v_ts.array() - v_ts.mean()).square().sum()/(v_ts.size()-1));
    auto v_tn_std = std::sqrt((v_tn.array() - v_tn.mean()).square().sum()/(v_tn.size()-1));
    auto v_tr_std = std::sqrt((v_tr.array() - v_tr.mean()).square().sum()/(v_tr.size()-1));
    auto v_tm_std = std::sqrt((v_tm.array() - v_tm.mean()).square().sum()/(v_tm.size()-1));

    report += format("\n Execution time (mean ± std [min, max]) in milliseconds\n"
                     " --------------------------------------------------------\n"
                     " Update          %f ± %f [%f, %f]\n"
                     "   Optimization  %f ± %f [%f, %f]\n"
                     "   Normalizing   %f ± %f [%f, %f]\n"
                     "   Resampling    %f ± %f [%f, %f]\n"
                     "   Mapping       %f ± %f [%f, %f]\n",
                     v_t.mean() * 1000.0, v_t_std * 1000.0,
                     v_t.minCoeff() * 1000.0, v_t.maxCoeff() * 1000.0,
                     v_ts.mean() * 1000.0, v_ts_std * 1000.0,
                     v_ts.minCoeff() * 1000.0, v_ts.maxCoeff() * 1000.0,
                     v_tn.mean() * 1000.0, v_tn_std * 1000.0,
                     v_tn.minCoeff() * 1000.0, v_tn.maxCoeff() * 1000.0,
                     v_tr.mean() * 1000.0, v_tr_std * 1000.0,
                     v_tr.minCoeff() * 1000.0, v_tr.maxCoeff() * 1000.0,
                     v_tm.mean() * 1000.0, v_tm_std * 1000.0,
                     v_tm.minCoeff() * 1000.0, v_tm.maxCoeff() * 1000.0);

    return report;
}

lama::HybridPFSlam2D::HybridPFSlam2D(const Options& options)
    : options_(options)
{
    solver_options_.max_iterations = options.max_iter;
    solver_options_.strategy       = makeStrategy(options.strategy, Vector2d::Zero());
    /* solver_options_.robust_cost    = makeRobust("cauchy", 0.25); */
    solver_options_.robust_cost.reset(new CauchyWeight(0.15));

    neff_ = options.particles;
    has_first_scan_ = false;
    has_first_landmarks_ = false;
    has_first_gnss_ = false;
    do_mapping_ = true;
    truncated_ray_ = options.truncated_ray;
    truncated_range_ = options.truncated_range;

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

    // Initialize the all particle.
    const uint32_t num_particles = options_.particles;
    particles_[0].resize(num_particles);
    current_particle_set_ = 0;

    if (options_.keep_pose_history)
        particles_[0][0].poses.push_back(pose_);
    particles_[0][0].pose = pose_;

    particles_[0][0].weight     = 0.0;
    particles_[0][0].weight_sum = 0.0;
    particles_[0][0].dm = DynamicDistanceMapPtr(new DynamicDistanceMap(options_.resolution, options_.patch_size));
    particles_[0][0].dm->setMaxDistance(options_.l2_max);
    particles_[0][0].dm->useCompression(options_.use_compression,  options_.cache_size, options_.calgorithm);

    particles_[0][0].occ = FrequencyOccupancyMapPtr(new FrequencyOccupancyMap(options_.resolution, options_.patch_size));
    particles_[0][0].occ->useCompression(options_.use_compression, options_.cache_size, options_.calgorithm);
    particles_[0][0].lm = SimpleLandmark2DMapPtr(new SimpleLandmark2DMap());

    for (uint32_t i = 1; i < num_particles; ++i){
        if (options_.keep_pose_history)
            particles_[0][i].poses.push_back(pose_);
        particles_[0][i].pose = pose_;

        particles_[0][i].weight     = 0.0;
        particles_[0][i].weight_sum = 0.0;

        particles_[0][i].dm  = DynamicDistanceMapPtr(new DynamicDistanceMap(*particles_[0][0].dm));
        particles_[0][i].occ = FrequencyOccupancyMapPtr(new FrequencyOccupancyMap(*particles_[0][0].occ));
        particles_[0][i].lm  = SimpleLandmark2DMapPtr( new SimpleLandmark2DMap(*(particles_[0][0].lm)) );
    }

    if (options.create_summary)
        summary = new Summary();

    kld_.init(options.particles, options_.max_particles, 0.04);

    // Global localization stuff
    do_global_localization_ = false;
    gloc_particles_ = options.gloc_particles;
}

lama::HybridPFSlam2D::~HybridPFSlam2D()
{
    delete thread_pool_;
}


void lama::HybridPFSlam2D::setPrior(const Pose2D& prior)
{
    // pose_ = prior;
    setPose(prior);
}

void lama::HybridPFSlam2D::setPose(const Pose2D& initialpose)
{
    // Get the expected pose of the particle filter
    Pose2D pose = getPose();
    // We will not be applying the initialpose directly.
    // To maintain the distribution of the filter we will be applying
    // the offset between the current expected pose and the desired initialpose.
    // This will effectively change the expected value to the initialpose.
    Pose2D offset = pose - initialpose;

    // Apply the offset to all particles.
    for (auto& particle : particles_[current_particle_set_]){
        particle.pose += offset;

        if (options_.keep_pose_history)
            particle.poses.back() = particle.pose;
    }
}

uint64_t lama::HybridPFSlam2D::getMemoryUsage() const
{
    uint64_t total = 0;

    const uint32_t num_particles = options_.particles;
    for (uint32_t i = 0; i < num_particles; ++i){
        total += particles_[current_particle_set_][i].dm->memory();
        total += particles_[current_particle_set_][i].occ->memory();
    }

    return total;
}

uint64_t lama::HybridPFSlam2D::getMemoryUsage(uint64_t& occmem, uint64_t& dmmem) const
{
    occmem = 0;
    dmmem  = 0;

    const uint32_t num_particles = options_.particles;
    for (uint32_t i = 0; i < num_particles; ++i){
        occmem += particles_[current_particle_set_][0].occ->memory();
        dmmem  += particles_[current_particle_set_][0].dm->memory();
    }

    return occmem + dmmem;
}

bool lama::HybridPFSlam2D::update(const PointCloudXYZ::Ptr& surface, const DynamicArray<Landmark>& landmarks, const GNSS& gnss,
                                  const Pose2D& odometry, double timestamp)
{
    Timer timer(true);
    Timer local_timer;

    // Predict from odometry if data already arived.
    Pose2D odelta;
    if (has_first_scan_ || has_first_landmarks_ || has_first_gnss_){

        odelta = odom_ - odometry;
        acc_trans_ += odelta.xy().norm();
        acc_rot_   += std::fabs(odelta.rotation());

        uint32_t num_particles = particles_[current_particle_set_].size();
        for (uint32_t i = 0; i < num_particles; ++i)
            drawFromMotion(odelta, particles_[current_particle_set_][i].pose, particles_[current_particle_set_][i].pose);
    }
    odom_ = odometry;

    // TODO: minimum number of points as a parameter
    bool invalid_surface   = surface->points.size() < 50;
    bool invalid_landmarks = landmarks.empty();
    bool invalid_gnss      = gnss.status == -1;

    if (invalid_surface && invalid_landmarks && invalid_gnss){

        // Force a resample if the accumulated motion is too large.
        // The objective is to increase the number of particles due to
        // accumulated error by the motion (i.e. odometry).
        if ((acc_trans_ > 1.0) || (acc_rot_ > M_PI * 0.5)){

            normalize();
            resample(false);

            clusterStats();

            // FORCE AN UPDATE
            acc_trans_ = options_.trans_thresh;
            acc_rot_   = options_.rot_thresh;

            return true;
        }

        // else, nothing to do here
        return false;
    }

    // The first time a data source arrives it it needs to be handled differently.
    if (!has_first_scan_ or !has_first_landmarks_ or !has_first_gnss_){
        bool is_first_data = handleFirstData(surface, landmarks, gnss);

        // If this is the very first time that any data
        // type arrives then there is nothing else to do.
        if (is_first_data)
            return true;
    }// end if

    // only continue if the necessary motion was gathered.
    if (!do_global_localization_ &&
        acc_trans_ <= options_.trans_thresh &&
        acc_rot_   <= options_.rot_thresh){
        return false;
    }

    // reset motion gathering
    acc_trans_ = 0;
    acc_rot_   = 0;

    current_surface_ = surface;
    local_timer.reset();

    // GNSS stuff
    Pose2D   gnss_prior;
    Matrix2d gnss_covar;
    if (!invalid_gnss){

        double gx, gy; // global x and y
        std::string zone;
        gnss.toUTM(gx, gy, zone);

        // gnss_pose keesp the previous gnss pose
        auto dx = gx - gnss_pose_.x();
        auto dy = gy - gnss_pose_.y();
        auto dxy_norm = hypot(dx, dy);

        double heading;

        if (dxy_norm < 0.05){
            // there may be some inplace rotation..
            // use the previous orientation and add a prediction from the odometry
            heading = (SO2d(gnss_pose_.state.so2()) * odelta.state.so2()).log();

            if (gnss_needs_heading_)
                // invalidate our current estimate because it is not enough
                invalid_gnss = true;
        } else {
            // get its absolute heading (assume differencial drive)
            heading = atan2(dy, dx);
            if (odelta.x() < 0.0){
                SO2d so2(heading + M_PI); // use so2 to normalize the angle

                // make sure the difference is not too big,
                auto diff = (gnss_pose_.state.so2().inverse() * so2).log();
                if (std::fabs(diff) < M_PI*0.5)
                    heading = so2.log();
            }

            if (gnss_needs_heading_){
                gnss_ref_pose_.state.so2() = SO2d(heading);
                gnss_needs_heading_ = false;
            }// end if
        }// end if

        gnss_pose_ = Pose2D(gx, gy, heading);
        gnss_prior = gnss_offset_ + (gnss_ref_pose_ -  gnss_pose_);

        gnss_covar = gnss.covar;
        gnss_covar(0,0) += options_.gnss_min_var;
        gnss_covar(1,1) += options_.gnss_min_var;

        // GNSS is a global position, we can use it directly to set
        // our localization when global localization is triggered.
        if (do_global_localization_ && !gnss_needs_heading_){
            setPose(gnss_prior);
            do_global_localization_ = false;
        }
    }

    // Handle global localization with lidar and landmarks before updating
    if (do_global_localization_){
        do_global_localization_ = ! globalLocalization(surface, landmarks);
    }

    // Apply scan matching and calculate landmarks likelihood.
    // If mapping is enabled, the landmarks will be added to
    // the map if they do not exist.
    uint32_t num_particles = particles_[current_particle_set_].size();
    if (thread_pool_){

        for (uint32_t i = 0; i < num_particles; ++i)
            thread_pool_->enqueue([this, i, &landmarks, &gnss, &gnss_prior, &invalid_surface, &invalid_landmarks, &invalid_gnss](){
                Particle* p = &particles_[current_particle_set_][i];

                if (!invalid_surface)   scanMatch(p);
                if (!invalid_landmarks) updateParticleLandmarks(p, landmarks);
                if (!invalid_gnss)      updateParticleGNSS(p, gnss_prior.xy(), gnss.covar);

                if (this->options_.keep_pose_history)
                    p->poses.push_back(p->pose);
            });

        thread_pool_->wait();
    } else {
        for (uint32_t i = 0; i < num_particles; ++i){
            Particle* p = &particles_[current_particle_set_][i];

            if (!invalid_surface)   scanMatch(p);
            if (!invalid_landmarks) updateParticleLandmarks(p, landmarks);
            if (!invalid_gnss)      updateParticleGNSS(p, gnss_prior.xy(), gnss.covar);

            if (options_.keep_pose_history)
                p->poses.push_back(p->pose);
        } // end for
    } // end if

    if (summary)
        probeSTime(local_timer.elapsed());

    // 3. Normalize weights and calculate neff
    local_timer.reset();

    normalize();

    if (summary)
        probeNTime(local_timer.elapsed());

    // Resample if needed
    // Force a resample if we have an invalid surface followed by a valid surface.
    // The objective is to reduce the number of samples before the map update happens.
    bool force_resample = !invalid_surface && !valid_surface_;
    valid_surface_ = !invalid_surface;

    if (force_resample || neff_ < (num_particles*0.5)){
        local_timer.reset();

        resample();
        num_particles = particles_[current_particle_set_].size();

        if (summary)
            probeRTime(local_timer.elapsed());
    } else {
        kdtree_.reset(num_particles);
        for (auto& p : particles_[current_particle_set_])
            kdtree_.insert(p.pose);
    }

    clusterStats();

    // 5. Update maps
    local_timer.reset();

    if (do_mapping_){
        // the number of particles may have change
        num_particles = particles_[current_particle_set_].size();
        if (thread_pool_){
            for (uint32_t i = 0; i < num_particles; ++i)
                thread_pool_->enqueue([this, i](){
                    updateParticleMaps(&(particles_[current_particle_set_][i]));
                });

            thread_pool_->wait();
        } else {
            for (uint32_t i = 0; i < num_particles; ++i){
                updateParticleMaps(&particles_[current_particle_set_][i]);
            }
        }// end if
    }// end if (do_mapping_)

    if (summary){
        probeMTime(local_timer.elapsed());
        probeTime(timer.elapsed());
        probeStamp(timestamp);
        probeMem();
    }

    if (options_.gnss_injection && !invalid_gnss){
        // With a 1% chance, inject a "exact" gps coordinate into the particle set
        auto r = random::uniform();
        if (neff_ < 2 or r < options_.gnss_injection_prob){
            auto idx    = getBestParticleIdx();
            Particle* p = &particles_[current_particle_set_][idx];
            p->pose = gnss_prior;
            if (options_.keep_pose_history)
                p->poses.back() = gnss_prior;
        }// end if
    }// end if !invalid_gnss

    return true;
}

size_t lama::HybridPFSlam2D::getBestParticleIdx() const
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

lama::Pose2D lama::HybridPFSlam2D::getPose() const
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

Eigen::Matrix3d lama::HybridPFSlam2D::getCovar() const
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

void lama::HybridPFSlam2D::saveOccImage(const std::string& name) const
{
    size_t pidx = getBestParticleIdx();
    sdm::export_to_png(*particles_[current_particle_set_][pidx].occ, name);
}

bool lama::HybridPFSlam2D::setMaps(FrequencyOccupancyMap* map, SimpleLandmark2DMap* lm_map)
{
    if (map == nullptr or lm_map == nullptr)
        return false;

    const uint32_t num_particles = particles_[current_particle_set_].size();
    uint8_t ps = 1 - current_particle_set_;
    particles_[ps].resize(num_particles);

    // Make all particle have the same map
    if (options_.keep_pose_history)
        particles_[ps][0].poses.push_back(pose_);
    particles_[ps][0].pose = pose_;

    particles_[ps][0].weight     = 0.0;
    particles_[ps][0].weight_sum = 0.0;
    particles_[ps][0].dm = DynamicDistanceMapPtr(new DynamicDistanceMap(options_.resolution, options_.patch_size));
    particles_[ps][0].dm->setMaxDistance(options_.l2_max);
    particles_[ps][0].dm->useCompression(options_.use_compression,  options_.cache_size, options_.calgorithm);

    map->visit_all_cells([&](auto& coords){
        if (map->isOccupied(coords))
            particles_[ps][0].dm->addObstacle(coords);
    });
    particles_[ps][0].dm->update();

    particles_[ps][0].occ = FrequencyOccupancyMapPtr(map);
    //particles_[ps][0].occ->useCompression(options_.use_compression, options_.cache_size, options_.calgorithm);
    particles_[ps][0].lm = SimpleLandmark2DMapPtr(lm_map);

    for (uint32_t i = 1; i < num_particles; ++i){
        if (options_.keep_pose_history)
            particles_[ps][i].poses.push_back(pose_);
        particles_[ps][i].pose = pose_;

        particles_[ps][i].weight     = 0.0;
        particles_[ps][i].weight_sum = 0.0;

        particles_[ps][i].occ = FrequencyOccupancyMapPtr(new FrequencyOccupancyMap(*particles_[ps][0].occ));
        particles_[ps][i].dm  = DynamicDistanceMapPtr(new DynamicDistanceMap(*particles_[ps][0].dm));
        particles_[ps][i].lm  = SimpleLandmark2DMapPtr( new SimpleLandmark2DMap(*(particles_[ps][0].lm)) );
    }

    particles_[current_particle_set_].clear();
    current_particle_set_ = ps;
    return true;
}

bool lama::HybridPFSlam2D::UTMtoLL(double x, double y, double& latitude, double& longitude)
{
    if (not has_first_gnss_)
        return false;

    Pose2D global = gnss_ref_pose_ + (gnss_offset_ - Pose2D(x,y,0));

    GNSS gnss;
    gnss.fromUTM(global.x(), global.y(), gnss_zone_);

    latitude = gnss.latitude;
    longitude = gnss.longitude;

    return true;
}

bool lama::HybridPFSlam2D::globalLocalization(const PointCloudXYZ::Ptr& surface, const DynamicArray<Landmark>& landmarks)
{
    // TODO: minimum number of points as a parameter
    bool invalid_surface   = surface->points.size() < 50;
    bool invalid_landmarks = landmarks.empty();

    if (invalid_surface && invalid_landmarks)
        return false; // global localization not done

    // First, try with the landmarks
    if (!invalid_landmarks){
        auto landmark_map  = getLandmarkMap();

        for (auto& landmark : landmarks){

            uint32_t id = landmark.id;
            auto lm = landmark_map->get(id);
            if ( lm == nullptr ) continue;

            // We use the first available landmark to estimate the correct pose of the robot.
            // But the landmark should be reasonably close
            Pose3D measurement(landmark.measurement);
            if (measurement.xyz().head<2>().norm() > 3)
                continue;

            Pose3D loc3d( lm->state.state * measurement.state.inverse() );

            // NOTE: The yaw + roll is a hack to solve the ambiguity of the euler angles.
            Pose2D newloc(loc3d.x(), loc3d.y(), loc3d.yaw() + std::fabs(loc3d.roll()) );

            // RMSE validation if a surface is available
            if (!invalid_surface){
                auto distance_map  = getDistanceMap();

                Affine3d moving_tf = Translation3d(surface->sensor_origin_) * surface->sensor_orientation_;
                Affine3d fixed_tf = Translation3d(Vector3d(newloc.x(),newloc.y(),0.0)) * AngleAxisd(newloc.rotation(), Vector3d::UnitZ());

                Affine3d tf = fixed_tf * moving_tf;

                const size_t num_points = surface->points.size();
                double rmse = 0.0;
                for (size_t i = 0; i < num_points; ++i){
                    Vector3d hit = tf * surface->points[i];
                    double dist = distance_map->distance(distance_map->w2m(hit));
                    rmse += dist*dist;
                } // end for
                rmse = std::sqrt( rmse / num_points);

                if (rmse > 0.1) continue;
            }

            setPose(newloc);
            return true;

        }// end for

    }// end if


    // Second, try with the laser
    if (invalid_surface)
        return false;

    const size_t num_points = surface->points.size();

    // Use the best occupancy map
    auto occupancy_map = getOccupancyMap();
    auto distance_map  = getDistanceMap();

    Vector3d min, max;
    occupancy_map->bounds(min, max);

    Vector3d diff = max - min;

    Pose2D best_pose = getPose();
    double best_likelihood = -std::numeric_limits<double>::max();
    double best_rmse = 0;

    for (uint32_t i = 0; i < gloc_particles_; ++i){

        double x, y, a;
        int j;
        for (j = 0; j < 1000; ++j){
            x = min[0] + random::uniform() * diff[0];
            y = min[1] + random::uniform() * diff[1];

            if (not occupancy_map->isFree(Vector3d(x, y, 0.0)))
                continue;

            a = random::uniform() * 2 * M_PI - M_PI;
            break;
        }

        if (j == 1000){
            // We were unable to find free space.
            // This prevents a infinite loop.
            print("UNABLE TO FIND FREE SPACE\n");
            return false;
        }

        Pose2D p(x, y , a);
        Pose3D p3(Vector3d(x, y, 0.0), a);

        // calculate scan likelihood
        Affine3d moving_tf = Translation3d(surface->sensor_origin_) * surface->sensor_orientation_;
        Affine3d fixed_tf = Translation3d(Vector3d(x,y,0.0)) * AngleAxisd(a, Vector3d::UnitZ());

        Affine3d tf = fixed_tf * moving_tf;

        double likelihood = 0;
        double rmse = 0.0;
        for (size_t i = 0; i < num_points; ++i){
            Vector3d hit = tf * surface->points[i];
            double dist = distance_map->distance(distance_map->w2m(hit));
            likelihood += - (dist*dist) / options_.meas_sigma;
            rmse += dist*dist;
        } // end for

        rmse = std::sqrt( rmse / num_points);

        if (likelihood > best_likelihood){
            best_likelihood = likelihood;
            best_pose = p;
            best_rmse = rmse;
        }
    } // end for

    if (best_rmse > 0.1)
        return false;

    setPose(best_pose);
    return true;
}


lama::HybridPFSlam2D::StrategyPtr lama::HybridPFSlam2D::makeStrategy(const std::string& name, const VectorXd& parameters)
{
    if (name == "gn"){
        return StrategyPtr(new GaussNewton);
    }else {
        return StrategyPtr(new LevenbergMarquard);
    }
}

lama::HybridPFSlam2D::RobustCostPtr lama::HybridPFSlam2D::makeRobust(const std::string& name, const double& param)
{
    if (name == "cauchy")
        return RobustCostPtr(new CauchyWeight(0.25));
    else if (name == "tstudent")
        return RobustCostPtr(new TDistributionWeight(3));
    else if (name == "tukey")
        return RobustCostPtr(new TukeyWeight);
    else
        return RobustCostPtr(new UnitWeight);
}

void lama::HybridPFSlam2D::drawFromMotion(const Pose2D& delta, const Pose2D& old_pose, Pose2D& pose)
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

double lama::HybridPFSlam2D::calculateLikelihood(const Particle& particle)
{
    PointCloudXYZ::Ptr surface = current_surface_;

    Affine3d moving_tf = Translation3d(surface->sensor_origin_) * surface->sensor_orientation_;

    Vector3d trans;
    trans << particle.pose.x(), particle.pose.y(), 0.0;

    Affine3d fixed_tf = Translation3d(trans) * AngleAxisd(particle.pose.rotation(), Vector3d::UnitZ());

    PointCloudXYZ::Ptr cloud(new PointCloudXYZ);
    const size_t num_points = surface->points.size();
    //== transform point cloud
    Affine3d tf = fixed_tf * moving_tf;
    cloud->points.reserve(num_points);
    for (size_t i = 0; i < num_points; ++i)
        cloud->points.push_back(tf * surface->points[i]);
    //==

    double likelihood = 0;
    for (size_t i = 0; i < num_points; ++i){
        Vector3d hit = cloud->points[i];
        double dist = particle.dm->distance(hit, 0);
        likelihood += - (dist*dist) / options_.meas_sigma;
    } // end for

    return likelihood;
}


bool lama::HybridPFSlam2D::handleFirstData(const PointCloudXYZ::Ptr& surface, const DynamicArray<Landmark>& landmarks, const GNSS& gnss)
{
    bool valid_surface   = surface->points.size() >= 50;
    bool valid_landmarks = not landmarks.empty();
    bool valid_gnss      = gnss.status != -1;

    static bool _first_update = true;
    bool first_update = _first_update;
    _first_update = false;

    // Lets update the maps of a single particle and then replicate the result to the remaining particles.
    // This is faster than applying the first data to each individual particle.
    // This can only be done if it is the very first time data arrives, otherwise the particles poses no
    // longer coincide and the particles map will be updated elsewhere.

    // Laser update
    if (valid_surface && !has_first_scan_){
        if (first_update){
            current_surface_ = surface;
            updateParticleMaps(&particles_[current_particle_set_][0]);

            const uint32_t num_particles = particles_[current_particle_set_].size();
            for (uint32_t i = 1; i < num_particles; ++i){
                int pset = current_particle_set_;
                particles_[pset][i].dm  = DynamicDistanceMapPtr(new DynamicDistanceMap(*particles_[pset][0].dm));
                particles_[pset][i].occ = FrequencyOccupancyMapPtr(new FrequencyOccupancyMap(*particles_[pset][0].occ));
            }// end for
        }// end if

        has_first_scan_ = true;
    }// end if

    // Landmark update
    if (valid_landmarks && !has_first_landmarks_){

        if (first_update){
            updateParticleLandmarks(&(particles_[current_particle_set_][0]), landmarks);

            const uint32_t num_particles = particles_[current_particle_set_].size();
            for (uint32_t i = 1; i < num_particles; ++i){
                int pset = current_particle_set_;
                particles_[pset][i].lm  = SimpleLandmark2DMapPtr( new SimpleLandmark2DMap(*(particles_[pset][0].lm)) );
            }// end for
        }// end if

        has_first_landmarks_ = true;
    }// end if

    if (valid_gnss && !has_first_gnss_){

        // generate a reference
        double refx, refy;
        gnss.toUTM(refx, refy, gnss_zone_);

        gnss_offset_ = getPose();
        gnss_ref_pose_ = Pose2D(refx, refy , 0.0);
        gnss_pose_     = gnss_ref_pose_;

        has_first_gnss_ = true;
    }

    return first_update;
}

void lama::HybridPFSlam2D::scanMatch(Particle* particle)
{
    /* const PointCloudXYZ::Ptr surface = readings_.back(); */
    const PointCloudXYZ::Ptr surface = current_surface_;
    if (surface->points.size() < 50) return;

    MatchSurface2D match_surface(particle->dm.get(), surface, particle->pose.state);

    SolverOptions so;
    so.max_iterations = options_.max_iter;
    /* so.strategy       = makeStrategy(options_.strategy, Vector2d::Zero()); */
    so.strategy.reset(new GaussNewton);
    so.robust_cost.reset(new CauchyWeight(0.15));

    Solve(so, match_surface, 0);
    particle->pose.state = match_surface.getState();

    double l = calculateLikelihood(*particle);
    particle->weight_sum += l;
    particle->weight     += l;
}

void lama::HybridPFSlam2D::updateParticleMaps(Particle* particle)
{
    const PointCloudXYZ::Ptr surface = current_surface_;
    if (surface->points.size() < 50) return;

    // 1. Transform the point cloud to the model coordinates.
    Affine3d moving_tf = Translation3d(surface->sensor_origin_) * surface->sensor_orientation_;
    Affine3d fixed_tf  = Translation3d(particle->pose.x(), particle->pose.y(), 0.0) * AngleAxisd(particle->pose.rotation(), Vector3d::UnitZ());

    // the sensor origin (in map coordinates) is the origin
    // for the ray casting.
    Vector3d wso = (fixed_tf * moving_tf).translation();

    const size_t num_points = surface->points.size();
    Affine3d tf = fixed_tf * moving_tf;

    // 2. generate the free and occupied positions.
    VectorVector3ui free;

    // generate the ray casts
    for (size_t i = 0; i < num_points; ++i)
    {
        Vector3d start = wso;
        Vector3d hit = tf * surface->points[i];
        Vector3d AB;
        double ray_length = 1.0; // this will be overwritten but gcc fails to notice.
        bool mark_hit = true;

        // Attempt to truncate the ray if it is larger than the truncated range
        if (truncated_range_ > 0.0)
        {
            AB = hit - start;
            ray_length = AB.norm();
            if (truncated_range_ < ray_length)
            {
                // Truncate the hit point and choose not to mark an obstacle for it
                hit = start + AB / ray_length * truncated_range_;
                mark_hit = false;
            }
        }

        // Only attempt to truncate a ray if the hit should be marked. If the hit
        // should not be marked then the range has already been truncated
        if (mark_hit and (truncated_ray_ > 0.0))
        {
            // Avoid computing the AB vector again if it has already been calculated
            if (truncated_range_ == 0.0)
            {
                AB = hit - start;
                ray_length = AB.norm();
            }
            if (truncated_ray_ < ray_length)
                start = hit - AB / ray_length * truncated_ray_;
        }

        Vector3ui mhit = particle->occ->w2m(hit);
        if (mark_hit)
        {
            bool changed = particle->occ->setOccupied(mhit);
            if ( changed ) particle->dm->addObstacle(mhit);
        }

        particle->occ->computeRay(particle->occ->w2m(start), mhit, free);
    }

    const size_t num_free = free.size();
    for (size_t i = 0; i < num_free; ++i){
        bool changed = particle->occ->setFree(free[i]);
        if ( changed ) particle->dm->removeObstacle(free[i]);
    }

    // 3. Update the distance map
    particle->dm->update();
}

void lama::HybridPFSlam2D::updateParticleLandmarks(Particle* particle, const DynamicArray<Landmark>& landmarks)
{
    // Get the particle pose in 3D
    auto xyr = particle->pose.xyr();
    Pose3D pose(Vector3d(xyr(0), xyr(1), 0.0), xyr(2));

    for (const auto& landmark : landmarks){

        uint32_t id = landmark.id;
        auto* lm = particle->lm->get(id);

        if ( do_mapping_ and (lm == nullptr) ){
            // This is the first time the landmarks is observed.
            lm = particle->lm->alloc(id);

            // Landmark in map coordinates
            lm->state = pose + Pose3D(landmark.measurement);

            // Calculate covariance
            Matrix6d H;
            H.setIdentity();
            H.topLeftCorner<3,3>() = pose.state.rotationMatrix().transpose();

            Matrix6d Hi = H.inverse();
            lm->covar = Hi * landmark.covar * Hi.transpose();
        } else if ( lm != nullptr ) {

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
            if (do_mapping_ and is_compatible){
                lm->covar = sig - K * H * sig;

                Vector6d s = K * diff;
                lm->state.state.translation() += s.head<3>();
                lm->state.state.so3() = lm->state.state.so3() * SO3d::exp(s.tail<3>());
            }

            // Calculate weight (or likelihood)
            // The likelihood is the log of a non normalized multivariant gaussian pdf.
            // The normalizer is not needed. The final result is the same and we save some caluculation.
            double w = -0.5 * diff.transpose() * Q.inverse() * diff;

            particle->lm_weight += w;
            particle->weight_sum += w;
        }

    }// end for
}

void lama::HybridPFSlam2D::updateParticleGNSS(Particle* particle, const Vector2d& prior, const Matrix2d& covar)
{
    Vector2d diff = particle->pose.xy() - prior;
    double w = -0.5 * (diff.transpose() * covar.inverse() * diff)(0);

    if ( std::isnan(w) )
        return;

    particle->lm_weight += w;
    particle->weight_sum += w;
}

void lama::HybridPFSlam2D::normalize()
{
    const uint32_t num_particles = particles_[current_particle_set_].size();

    double gain   = 1.0 / (options_.meas_sigma_gain * num_particles);
    double lmgain = 1.0 / (options_.landmark_gain * num_particles);

    double max_l   = particles_[current_particle_set_][0].weight;
    double max_llm = particles_[current_particle_set_][0].lm_weight;

    for (uint32_t i = 1; i < num_particles; ++i){
        if (max_l < particles_[current_particle_set_][i].weight)
            max_l = particles_[current_particle_set_][i].weight;

        if (max_llm < particles_[current_particle_set_][i].lm_weight)
            max_llm = particles_[current_particle_set_][i].lm_weight;
    }

    double sum = 0;
    for (uint32_t i = 0; i < num_particles; ++i){

        double l = gain*(particles_[current_particle_set_][i].weight - max_l) +
                   lmgain*(particles_[current_particle_set_][i].lm_weight - max_llm);
        particles_[current_particle_set_][i].normalized_weight  = std::exp(l);

        sum += particles_[current_particle_set_][i].normalized_weight;
    }

    neff_ = 0;
    for (uint32_t i = 0; i < num_particles; ++i){
        particles_[current_particle_set_][i].normalized_weight /= sum;
        neff_ += particles_[current_particle_set_][i].normalized_weight * particles_[current_particle_set_][i].normalized_weight;
    }

    neff_ = 1.0 / neff_;
}

void lama::HybridPFSlam2D::resample(bool reset_weight)
{
    const uint32_t num_particles = particles_[current_particle_set_].size();
    DynamicArray<double> c(num_particles + 1);

    c[0] = 0.0;
    for (size_t i = 0; i < num_particles; ++i)
        c[i+1] = c[i] + particles_[current_particle_set_][i].normalized_weight;

    uint8_t ps = 1 - current_particle_set_;
    particles_[ps].reserve(options_.max_particles);

    kld_.reset();
    kdtree_.reset(kld_.samples_max);
    for (size_t i = 0; i < kld_.samples_max; ++i){

        double r = random::uniform();
        uint32_t idx;
        for (idx = 0; idx < num_particles; ++idx)
            if ((c[idx] <= r) && (r < c[idx+1]))
                break;

        particles_[ps].emplace_back( Particle{} );
        particles_[ps].back() = particles_[current_particle_set_][idx];

        if (reset_weight){
            particles_[ps].back().weight = 0.0;
            particles_[ps].back().lm_weight = 0.0;
        }

        particles_[ps].back().dm  = DynamicDistanceMapPtr(new DynamicDistanceMap(*(particles_[current_particle_set_][idx].dm)));
        particles_[ps].back().occ = FrequencyOccupancyMapPtr(new FrequencyOccupancyMap(*(particles_[current_particle_set_][idx].occ)));
        particles_[ps].back().lm = SimpleLandmark2DMapPtr(new SimpleLandmark2DMap(*(particles_[current_particle_set_][idx].lm)));

        auto& pose = particles_[ps].back().pose;

        kdtree_.insert(pose);
        auto kld_samples = kld_.resample_limit(kdtree_.num_leafs);

        if (particles_[ps].size() >= kld_samples) break;
    }

    particles_[current_particle_set_].clear();
    current_particle_set_ = ps;
}

void lama::HybridPFSlam2D::clusterStats()
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

        // compute covar of linear components
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

