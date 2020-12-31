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

static double normalize(double z)
{
    return std::atan2(std::sin(z),std::cos(z));
}

static double angle_diff(double a, double b)
{
    double d1, d2;
    a = normalize(a);
    b = normalize(b);
    d1 = a-b;
    d2 = 2*M_PI - std::fabs(d1);
    if(d1 > 0)
        d2 *= -1.0;
    if(std::fabs(d1) < std::fabs(d2))
        return(d1);
    else
        return(d2);
}

namespace lama {

struct Landmark2D {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    // Each landmark should have a unique ID.
    uint32_t id;

    // The landmark measurement.
    // It has 3 dof: x, y, rotation
    Vector3d mu;
    // The measument covariance.
    Matrix3d sigma;

    // By default, we assume the the measurement is based on polar coordinates.
    // If false, then measument in cartesian coordinates are assumed.
    // This property changes the measurement prediction and jacobian.
    bool is_polar = true;

    // Given a pose, calculate the cartesian coordinates of the landmark.
    inline Vector3d toCartesian(const Pose2D& pose) const
    {
        Vector3d h;

        if (not is_polar){
            h.head<2>() = pose * mu.head<2>();
        } else {
            h << pose.x() + mu(0) * std::cos(pose.rotation() + mu(1)),
                 pose.y() + mu(0) * std::sin(pose.rotation() + mu(1));
        }

        h(2) = normalize(mu(2) + pose.rotation());

        return h;
    }

    // Calculated the difference between two landmark measurements.
    inline Vector3d diff(const Vector3d& other) const
    {
        Vector3d h;
        h.head<2>() = mu.head<2>() - other.head<2>();
        h(2) = angle_diff(mu(2), other(2));

        return h;
    }

    inline void predict(const Pose2D& pose, const Vector3d& mu, Vector3d& h, Matrix3d& J) const
    {
        if (is_polar){
            Vector2d delta = mu.head<2>() - pose.xy();
            double range   = delta.norm();
            double bearing = normalize(std::atan2(delta.y(), delta.x()) - pose.rotation());

            h << range, bearing, normalize(mu(2) - pose.rotation());

            J << delta.x() / range        , delta.y() / range,         0
                -delta.y() / (range*range), delta.x() / (range*range), 0,
                                         0,                         0, 1;

        } else {
            double cs = std::cos(pose.rotation());
            double sn = std::sin(pose.rotation());

            Vector2d delta = mu.head<2>() - pose.xy();

            h << delta.x() * cs + delta.y() * sn,
                 delta.y() * cs - delta.x() * sn,
                 angle_diff(mu(2), pose.rotation());

            J << cs,  sn,  0,
                -sn,  cs,  0,
                  0,   0,  1;
        }// end if
    }
};

}

