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

#include <iostream>
#include <cmath>

#include "lama/kdtree.h"

lama::KDTree::KDTree()
{
    size = {0.5, 0.5, 10 * M_PI / 180.0};
}

void lama::KDTree::reset(int32_t max_size)
{
    num_nodes = 0;
    num_leafs = 0;
    num_clusters = 0;
    max_nodes = 3*max_size;

    nodes.resize(max_nodes);
    root = -1;
}

void lama::KDTree::insert(const Pose2D& pose)
{
    Array<int32_t, 3> key;
    key[0] = std::floor(pose.x() / size[0]);
    key[1] = std::floor(pose.y() / size[1]);
    key[2] = std::floor(pose.rotation() / size[2]);

    root = insertNode(-1, root, key);
}

int32_t lama::KDTree::insertNode(int32_t parent, int32_t node, const Array<int32_t, 3>& key)
{
    // Case #1: The node does not exist
    if (node == -1){

        // make sure we are not adding a node beyond our maximum capacity
        if (num_nodes == max_nodes)
            return -1;

        node = num_nodes;
        num_nodes++;

        nodes[node].leaf  = true;
        nodes[node].key   = key;

        num_leafs++;

        return node;
    }

    // Case #2: The node is a leaf
    if (nodes[node].leaf == true){

        // Check if the keys are the same
        if (nodes[node] == key){
            return node;
        }// end if

        // The keys are not the same, so we split the node
        // at the dimension with the largest variance
        int32_t max_split = 0;
        nodes[node].pivot_dim = -1;
        for (int32_t i = 0; i < 3; ++i){
            int32_t split = std::abs(key[i] - nodes[node].key[i]);
            if (split > max_split){
                max_split = split;
                nodes[node].pivot_dim = i;
            }// end if
        }// end for

        nodes[node].pivot_value = (key[nodes[node].pivot_dim] + nodes[node].key[nodes[node].pivot_dim]) / 2.0;

        if (key[nodes[node].pivot_dim] < nodes[node].pivot_value){
            nodes[node].children[0] = insertNode(node, -1, key);
            nodes[node].children[1] = insertNode(node, -1, nodes[node].key);
        } else {
            nodes[node].children[0] = insertNode(node, -1, nodes[node].key);
            nodes[node].children[1] = insertNode(node, -1, key);
        }

        nodes[node].leaf = false;
        num_leafs--;

        return node;
    }

    // Case #3: The node is not a leaf and has childreen
    if (key[nodes[node].pivot_dim] < nodes[node].pivot_value)
        insertNode(node, nodes[node].children[0], key);
    else
        insertNode(node, nodes[node].children[1], key);

    return node;
}

int32_t lama::KDTree::findNode(int32_t node, const Array<int32_t, 3>& key)
{
    if (node == -1)
        return -1;

    if (nodes[node].leaf == true){
       if (nodes[node] == key)
            return node;
        else
            return -1; // not found
    }

    if (key[nodes[node].pivot_dim] < nodes[node].pivot_value)
        return findNode(nodes[node].children[0], key);
    else
        return findNode(nodes[node].children[1], key);
}

void lama::KDTree::cluster()
{
    // All leafs go into a queue
    DynamicArray<int32_t> queue(num_nodes);
    int32_t num_queue = 0;

    for (int32_t i = 0; i < num_nodes; ++i){
        if (nodes[i].leaf == true){
            nodes[i].cluster   = -1;
            queue[num_queue++] = i;
        }// end if
    }// end for

    //
    num_clusters = 0;
    while (num_queue > 0){
        int32_t node = queue[--num_queue];

        if (nodes[node].cluster >= 0)
            continue; // Already labeled

        // Assign a label
        nodes[node].cluster = num_clusters++;

        clusterNode(node);
    }// end while
}

void lama::KDTree::clusterNode(int32_t node)
{
    if (node == -1)
        return;

    Array<int32_t, 3> key;
    int32_t nnode;

    for (int32_t i = 0; i < 3 * 3 * 3; ++i){
        key[0] = nodes[node].key[0] + (i / 9) - 1;
        key[1] = nodes[node].key[1] + ((i % 9) / 3) - 1;
        key[2] = nodes[node].key[2] + ((i % 9) % 3) - 1;

        nnode = findNode(root, key);
        if (nnode == -1)
            continue;

        if (nodes[nnode].cluster >= 0)
            continue; // Already labeled

        nodes[nnode].cluster = nodes[node].cluster;
        clusterNode(nnode);
    }// end for
}

int32_t lama::KDTree::getCluster(const Pose2D& pose)
{
    Array<int32_t, 3> key;
    key[0] = std::floor(pose.x() / size[0]);
    key[1] = std::floor(pose.y() / size[1]);
    key[2] = std::floor(pose.rotation() / size[2]);

    int32_t node = findNode(root, key);
    if (node == -1)
        return -1;

    return nodes[node].cluster;
}

