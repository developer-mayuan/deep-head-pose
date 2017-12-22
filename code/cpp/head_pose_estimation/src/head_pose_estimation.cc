//
// Created by yuanma on 12/21/17.
//

#include "head_pose_estimation/head_pose_estimation.hpp"

namespace tf = tensorflow;

// class constructor by providing graph file path.
void HeadPoseEstimator::HeadPoseEstimator(std::string graph_file) {
    tf::GraphDef graph_def;
    tf::Status load_graph_status = tf::ReadBinaryProto(tf::Env::Default(),
                                                       graph_file, graph_def);
    if (!load_graph_status.ok()) {
        return tf::errors::NotFound("Failed to load compute graph at '",
                                    graph_file, "'");
    }

    // Allocate a new TensorFlow Session.
    this->session->reset(tf::NewSession(tf::SessionOptions()));

    // create model graph
    tf::Status session_create_status = (*(this->session))->Create(graph_def);
    if (!session_create_status.ok()) {
        return session_create_status;
    }
    return tf::Status::OK();

}


