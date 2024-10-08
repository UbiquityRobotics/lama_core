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

#pragma once

#include <memory>
#include <vector>

#include "time.h"
#include "pose2d.h"
#include "nlls/solver.h"

#include "sdm/dynamic_distance_map.h"
#include "sdm/frequency_occupancy_map.h"

#include "lama/kdtree.h"
#include "lama/kld_sampling.h"

#include "lama/gnss.h"
#include "lama/simple_landmark2d_map.h"

#include <Eigen/StdVector>

namespace lama {

struct ThreadPool;

class HybridPFSlam2D {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef Solver::Options SolverOptions;

    typedef Strategy::Ptr   StrategyPtr;
    typedef RobustCost::Ptr RobustCostPtr;

    typedef std::shared_ptr<DynamicDistanceMap>    DynamicDistanceMapPtr;
    typedef std::shared_ptr<FrequencyOccupancyMap> FrequencyOccupancyMapPtr;

    typedef std::shared_ptr<SimpleLandmark2DMap> SimpleLandmark2DMapPtr;


public:

    struct Particle {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        // The weight of the particle.
        // from the occupancy grids
        double weight = 0.0;
        // from the landmarks
        double lm_weight = 0.0;

        double normalized_weight;

        double weight_sum;

        // The pose of this particle in the map
        Pose2D pose;

        // history
        DynamicArray<Pose2D> poses;

        DynamicDistanceMapPtr    dm;
        FrequencyOccupancyMapPtr occ;

        // The landmark map.
        SimpleLandmark2DMapPtr lm;

    };

    struct Cluster {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        // Number of samples in the cluster
        int32_t count = 0;
        // The weight of the cluster
        double weight = 0.0;

        // Pose of the cluster
        Pose2D pose;

        // Covariance of the pose
        Matrix3d covar = Matrix3d::Zero();

        // Workspace used to calculate the covariance
        Vector4d m = Vector4d::Zero();
        Matrix2d c = Matrix2d::Zero();
    };

    struct Summary {
        /// When it happend.
        DynamicArray<double> timestamp;
        /// Total execution time.
        DynamicArray<double> time;
        /// Solving (i.e. optimization) execution time.
        DynamicArray<double> time_solving;
        /// Normalizing execution time.
        DynamicArray<double> time_normalizing;
        /// Resampling execution time.
        DynamicArray<double> time_resampling;
        /// Mapping (occ+distance) execution time.
        DynamicArray<double> time_mapping;

        /// Total memory used by the maps.
        DynamicArray<double> memory;

        std::string report() const;
    };
    Summary* summary = nullptr;

    // Summary help functions.
    inline void probeStamp(double stamp)
    { summary->timestamp.push_back(stamp); }

    inline void probeTime(Duration elapsed)
    { summary->time.push_back(elapsed.toSec()); }

    inline void probeSTime(Duration elapsed)
    { summary->time_solving.push_back(elapsed.toSec()); }

    inline void probeNTime(Duration elapsed)
    { summary->time_normalizing.push_back(elapsed.toSec()); }

    inline void probeRTime(Duration elapsed)
    { summary->time_resampling.push_back(elapsed.toSec()); }

    inline void probeMTime(Duration elapsed)
    { summary->time_mapping.push_back(elapsed.toSec()); }

    inline void probeMem()
    { summary->memory.push_back(getMemoryUsage()); }

    inline uint32_t getNumParticles() const
    { return particles_[current_particle_set_].size(); }

    // All SLAM options are in one place for easy access and passing.
    struct Options {
        Options(){}

        /// The number of particles to use
        uint32_t particles;
        /// Maximum number of particles to use
        uint32_t max_particles;
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
        /// Use this to smooth the measurements likelihood of the laser.
        double meas_sigma_gain = 3;
        /// Use this to smooth the measurements likelihood of the landmarks and GNSS
        double landmark_gain = 0.05;
        /// Minimum variance of the gps
        double gnss_min_var = 0.025;
        /// If set to true, the gnss "exact" position
        /// will be injected in the particle set.
        bool gnss_injection = false;
        /// Probability of injecting the gnss "exact" positon.
        double gnss_injection_prob = 0.01;
        /// The ammount of displacement that the system must
        /// gather before any update takes place.
        double trans_thresh = 0.5;
        /// The ammout of rotation that the system must
        /// gather before any update takes place.
        double rot_thresh = 0.5;
        /// Maximum distance (in meters) of the euclidean distance map.
        double l2_max = 0.5;
        /// If != from zero, truncate the ray lenght (includes the endpoint).
        double truncated_ray = 0.0;
        /// If != from zero and ray length > truncated_range, truncate the ray from the
        /// starting point and do not add an obstacle for the hit
        double truncated_range = 0.0;
        /// Resolutions of the maps.
        double resolution = 0.05;
        /// The side size of a patch
        uint32_t patch_size = 32;
        /// Maximum number of iterations that the optimizer
        /// can achieve.
        uint32_t max_iter = 100;
        /// Strategy to use in the optimization.
        std::string strategy = "gn";
        /// Number of working threads.
        /// -1 for none, 0 for auto, >0 user define number of workers.
        int32_t threads = -1;
        /// Pseudo random generator seed.
        /// Use 0 to generate a new seed.
        uint32_t seed = 0;
        /// Should online data compression be used or not.
        bool use_compression = false;
        /// Size of LRU.
        uint32_t cache_size = 100;
        /// Compression algorithm to use when compression is activated
        std::string calgorithm = "lz4";
        /// Do compatibility test* using the Mahalanobis distance.
        bool do_compatibility_test = true;
        /// number of particles used for global localization.
        int32_t gloc_particles = 3000;
        /// Save data to create an execution summary.
        bool create_summary = false;
        /// Keep particle pose history
        bool keep_pose_history = false;
    };

    HybridPFSlam2D(const Options& options = Options());

    virtual ~HybridPFSlam2D();

    inline const Options& getOptions() const
    {
        return options_;
    }

    uint64_t getMemoryUsage() const;
    uint64_t getMemoryUsage(uint64_t& occmem, uint64_t& dmmem) const;

    // Update the SLAM system.
    bool update(const PointCloudXYZ::Ptr& surface, const DynamicArray<Landmark>& landmarks,
            const GNSS& gnss, const Pose2D& odometry, double timestamp);

    size_t getBestParticleIdx() const;

    Pose2D getPose() const;
    Matrix3d getCovar() const;

    inline const std::deque<double>& getTimestamps() const
    { return timestamps_; }

    inline const std::vector<Particle>& getParticles() const
    { return particles_[current_particle_set_]; }

    inline const FrequencyOccupancyMap* getOccupancyMap() const
    {
        if (!has_first_scan_) return nullptr;

        size_t pidx = getBestParticleIdx();
        return particles_[current_particle_set_][pidx].occ.get();
    }

    inline const DynamicDistanceMap* getDistanceMap() const
    {
        if (!has_first_scan_) return nullptr;

        size_t pidx = getBestParticleIdx();
        return particles_[current_particle_set_][pidx].dm.get();
    }

    inline const SimpleLandmark2DMap* getLandmarkMap() const
    {
        if (!has_first_landmarks_) return nullptr;

        size_t pidx = getBestParticleIdx();
        return particles_[current_particle_set_][pidx].lm.get();
    }

    void saveOccImage(const std::string& name) const;

    inline double getNeff() const
    { return neff_; }

    void setPrior(const Pose2D& prior);

    void setPose(const Pose2D& prior);

    // Tell the slam process to do localization but not the mapping part.
    void pauseMapping();

    // Tell the slam process to do both localization and mapping.
    void resumeMapping();

    bool setMaps(FrequencyOccupancyMap* map, SimpleLandmark2DMap* lm_map);

    inline void triggerGlobalLocalization()
    { do_global_localization_ = true; }

    // Convert local map coordinates to global latitude/longitude coordinates.
    // Returns false if the navsat reference position hasn't been determined yet.
    bool UTMtoLL(double x, double y, double& latitude, double& longitude);

    inline const Pose2D& getGNSSRef() const
    { return gnss_ref_pose_; }

    inline const Pose2D& getGNSSOffset() const
    { return gnss_offset_; }

    inline const std::string& getGNSSZone() const
    { return gnss_zone_; }

    inline void setGNSSInfo(const Pose2D& ref, const Pose2D& offset, const std::string& zone)
    {
        gnss_ref_pose_ = ref;
        gnss_offset_   = offset;
        gnss_zone_     = zone;
        has_first_gnss_     = true;
        gnss_needs_heading_ = false;
    }

private:

    StrategyPtr makeStrategy(const std::string& name, const VectorXd& parameters);
    RobustCostPtr makeRobust(const std::string& name, const double& param);

    void drawFromMotion(const Pose2D& delta, const Pose2D& old_pose, Pose2D& pose);

    double likelihood(const PointCloudXYZ::Ptr& surface, Pose2D& pose);

    double calculateLikelihood(const Particle& particle);

    bool handleFirstData(const PointCloudXYZ::Ptr& surface, const DynamicArray<Landmark>& landmarks, const GNSS& gnss);

    void scanMatch(Particle* particle);
    void updateParticleMaps(Particle* particle);

    void updateParticleLandmarks(Particle* particle, const DynamicArray<Landmark>& landmarks);
    void updateParticleGNSS(Particle* particle, const Vector2d& prior, const Matrix2d& covar);

    void normalize();
    void resample(bool reset_weight = true);

    void clusterStats();

    // Do global localization.
    //
    // Usually, global localization with a particle filter is solved by uniformly distribute
    // the particles over the map's free space, and, as the particle filter evolves, the particles
    // will converge to the correct pose. This simply cannot be done for a SLAM algorithm.
    // Instead of letting the filter converge, we use the pose of particle with the highest weight
    // to set a new initial pose.
    //
    // WARNING: Calling this function does not guarantee to find the correct global pose. You may need
    // to call this function more than once to find the correct pose. It is up to you to evaluate if the
    // global pose is correct or not.
    bool globalLocalization(const PointCloudXYZ::Ptr& surface, const DynamicArray<Landmark>& landmarks);

    void checkForPossibleGlobalLocalizationTrigger(const DynamicArray<Landmark>& landmarks);

private:
    Options options_;
    SolverOptions solver_options_;

    std::vector<Particle> particles_[4];
    uint8_t current_particle_set_;
    uint8_t mapping_particle_set_;

    DynamicArray<Cluster> clusters_;

    KDTree      kdtree_;
    KLDSampling kld_;

    Pose2D odom_;
    Pose2D pose_;

    Pose2D gnss_ref_pose_;
    Pose2D gnss_offset_;
    Pose2D gnss_pose_;

    Vector2f gnss_ref_;
    std::string gnss_zone_;

    double acc_trans_;
    double acc_rot_;

    bool has_first_scan_      = false;
    bool has_first_landmarks_ = false;
    bool has_first_gnss_      = false;
    bool gnss_needs_heading_  = true;

    bool valid_surface_;

    // Controls the execution of the mapping process
    bool do_mapping_;

    double truncated_ray_;
    double truncated_range_;
    double neff_;

    bool do_global_localization_;
    double gloc_particles_;

    bool forced_update_;

    std::deque<double> timestamps_;
    PointCloudXYZ::Ptr current_surface_;

    ThreadPool* thread_pool_;
};

} /* lama */

