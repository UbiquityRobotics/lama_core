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

#include "types.h"
#include "pose2d.h"

static double normalize_angle(double angle)
{
    while (angle > M_PI)
        angle = angle - 2.0 * M_PI;

    while (angle < -M_PI)
        angle = angle + 2 * M_PI;

    return angle;
}

namespace lama {

struct Landmark2D {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    // Each landmark should have a unique ID.
    uint32_t id;
    // The landmark measurement.
    Vector2d mu;
    // The measument covariance.
    Matrix2d sigma;

    // By default, we assume the the measurement is based on polar coordinates.
    // If false, then measument in cartesian coordinates are assumed.
    // This property changes the measurement prediction and jacobian.
    bool is_polar = true;

    // Given a pose, calculate the cartesian coordinates of the landmark.
    inline Vector2d toCartesian(const Pose2D& pose) const
    {
        if (not is_polar)
            return pose * mu;

        Vector2d h;
        h << pose.x() + mu(0) * std::cos(pose.rotation() + mu(1)),
             pose.y() + mu(0) * std::sin(pose.rotation() + mu(1));

        return h;
    }

    // Calculated the difference between two landmark measurements.
    inline Vector2d diff(const Vector2d& other) const
    {
        Vector2d h = mu - other;
        if (is_polar)
            h(1) = normalize_angle(h(1));

        return h;
    }

    inline void predict(const Pose2D& pose, const Vector2d& coords, Vector2d& h, Matrix2d& J) const
    {
        if (is_polar){
            Vector2d delta = coords - pose.xy();
            double range   = delta.norm();
            double bearing = normalize_angle(std::atan2(delta.y(), delta.x()) - pose.rotation());

            h << range, bearing;
            J << delta.x() / range        , delta.y() / range,
                -delta.y() / (range*range), delta.x() / (range*range);
        } else {
            h = coords;
            J = Matrix2d::Identity();
        }// end if
    }
};

}

