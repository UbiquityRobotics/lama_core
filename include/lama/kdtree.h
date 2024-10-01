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

#pragma once

#include <memory>

#include "types.h"
#include "pose2d.h"

namespace lama {

// A k-d tree designed for 2D poses.
//
// This implementation is based on Players' AMCL source code.
// https://github.com/ros-planning/navigation/blob/noetic-devel/amcl/include/amcl/pf/pf_kdtree.h
struct KDTree {

    struct Node {
        // Node key, used for equality comparison.
        Array<int, 3> key;
        // Is this node a leaf?
        bool leaf;
        // Pivot dimension and value
        int32_t pivot_dim;
        double  pivot_value;
        // Cluster label
        int32_t cluster;
        // Child nodes
        Array<int32_t, 2> children;

        // Two node are equal if they have the same key.
        inline bool operator==(const Array<int32_t, 3>& other) const
        {
            return (key[0] == other[0]) &&
                   (key[1] == other[1]) &&
                   (key[2] == other[2]);
        }
    };

    // Cell size (or resolution)
    Array<double, 3> size;

    // Pointer to the tree's root node
    int32_t root;

    // All Nodes
    DynamicArray<Node> nodes;
    // Current number of node
    int32_t num_nodes;
    // Maximum number of expected nodes
    int32_t max_nodes;

    // Number of leafs in the tree
    int32_t num_leafs;
    // Number of clusters in the tree
    int32_t num_clusters;

    // Constructor
    KDTree();

    // Reset the k-d tree
    void reset(int32_t max_size);

    // Insert a pose into the kdtree
    void insert(const Pose2D& pose);

    // Insert a node into de kdtree
    int32_t insertNode(int32_t parent, int32_t node, const Array<int32_t, 3>& key);

    // Recursive search of a node
    int32_t findNode(int32_t node, const Array<int32_t, 3>& key);

    // Cluster the leafs in the k-d tree
    void cluster();

    // Recursively label the node
    void clusterNode(int32_t node);

    // Get the cluster label for the pose
    int32_t getCluster(const Pose2D& pose);

};

}// namespace lama
