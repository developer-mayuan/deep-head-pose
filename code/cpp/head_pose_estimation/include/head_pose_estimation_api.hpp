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
    HeadPoseEstimator(std::string);
    ~HeadPoseEstimator() = default;

    // return roll, pitch, and yaw angle
    double*  estimate_head_pose();

private:
    string model_graph_file;
    double yaw_angle;
    double pitch_angle;
    double roll_angle;
    std::unique_ptr<tf::Session>* session;

    void convert_img_to_tensor(cv::Mat frame, std::string bounding_box);
};

#endif //DSM_WORKSPACE_HEAD_POSE_ESTIMATION_HPP
