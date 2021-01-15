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

#include <utility>

#include "types.h"
#include "cow_ptr.h"
#include "pose3d.h"

namespace lama {

struct SimpleLandmark2DMap {

    struct Landmark {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        Pose3D   state;
        Matrix6d covar;
    };

    // Landmarks are identified by their unique id.
    // For efficiency, landmarks are wrapped around a copy-on-write object.
    Dictionary<uint32_t, COWPtr<Landmark>> landmarks;

    // Default constructor.
    SimpleLandmark2DMap() = default;
    // Default destructor.
    virtual ~SimpleLandmark2DMap() = default;

    // Copy constructor
    SimpleLandmark2DMap(const SimpleLandmark2DMap& other)
    {
        for (auto& kv : other.landmarks)
            landmarks[kv.first] = COWPtr<Landmark>(kv.second);
    }

    // Get an existing landmark.
    // Returns nullptr if the landmark for the given *id* does not exist.
    inline Landmark* get(uint32_t id)
    {
        auto iter = landmarks.find(id);
        if ( iter == landmarks.end() )
            return nullptr;

        return iter->second.operator->();
    }

    // Allocate a new landmark.
    // Returns nullptr if the landmark already exists.
    inline Landmark* alloc(uint32_t id)
    {
        auto result = landmarks.emplace(id, COWPtr<Landmark>(new Landmark));
        if ( result.second == false )
            return nullptr;

        return result.first->second.get();
    }

    // Update the state of a landmark.
    // Returns true if the update is successful.
    bool update(uint32_t id, const Pose3D& pose, const Vector3d& measurement, const Matrix6d& covar);

    // Get the current state of a landmark.
    // Returns false if the landmark does not exist.
    bool get(uint32_t id, Pose3D& state, Matrix6d* covar = nullptr) const;

    // Landmark visitor function
    typedef std::function<void(uint32_t, const Landmark&)> Visitor;

    // For each landmark, call its visitor.
    inline void visit_all_landmarks(const Visitor& visitor) const
    {
        for (auto& kv : landmarks){
            visitor(kv.first, *(kv.second.read_only()));
        }// end for_all
    }

};

} // namespace lama

