//
// Created by yuanma on 12/21/17.
//

#ifndef DSM_WORKSPACE_HEAD_POSE_ESTIMATION_HPP
#define DSM_WORKSPACE_HEAD_POSE_ESTIMATION_HPP

#ifndef TENSORFLOW
#define TENSORFLOW
#endif

#include "utils.hpp"

namespace tf = tensorflow;

class HeadPoseEstimator {
public:
    HeadPoseEstimator(std::string graph_file);
    ~HeadPoseEstimator() {}

    // return roll, pitch, and yaw angle
    double*  head_pose_est_result = estimate_head_pose();

private:
    string model_graph_file;
    double yaw_angle;
    double pitch_angle;
    double roll_angle;
    std::unique_ptr<tf::Session>* session;
};

#endif //DSM_WORKSPACE_HEAD_POSE_ESTIMATION_HPP
