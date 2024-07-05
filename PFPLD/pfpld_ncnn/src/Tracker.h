//
// Created by lyp on 29/9/2019.
//

#ifndef MULTIFACE_TRACKER_H
#define MULTIFACE_TRACKER_H

#endif //MULTIFACE_TRACKER_H
#include <opencv2/opencv.hpp>
using namespace cv;


class Cal_distance{
public:
    Cal_distance();
    float cost_dIou(const Rect& rectA, const Rect& rectB);

};

//establish a new kalman tracker!!!
//*******************************************************************//
#define StateType Rect_<float>
class KalmanTracker
{
public:

    KalmanTracker(StateType initRect)
    {
        init_kf(initRect);
        m_time_since_update = 0;
        m_hits = 0;
        m_hit_streak = 0;
        m_age = 0;
        m_id = kf_count;//kf_count;
        kf_count++;
    }

    ~KalmanTracker()
    {
        m_history.clear();
    }

    StateType predict();
    void update(StateType stateMat);

    StateType get_state();
    StateType get_rect_xysr(float cx, float cy, float s, float r);
    std::vector<cv::Mat> face_img;
    static int kf_count;
    //int kf_count;
    int m_time_since_update;
    int m_hits;
    int m_hit_streak;
    int m_age;
    int m_id;
    int refine_id;
    std::vector<cv::Mat> face_roi;    //save the tracker' face roi
    std::vector<std::vector<float>>  angle;   //save the tracker' angle





private:
    void init_kf(StateType stateMat);

    cv::KalmanFilter kf;
    cv::Mat measurement;

    std::vector<StateType> m_history;
};