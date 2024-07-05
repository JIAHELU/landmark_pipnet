//
// Created by bill on 20-6-28.
//
#include "anchor_generator.h"
#include "opencv2/opencv.hpp"
#include "config.h"
#include "tools.h"
#include "pfpld.id.h"
#include "Tracker.h"
#include "Hungarian.h"
using namespace std;
using namespace cv;
typedef struct TrackingBox
{
    int frame;
    int id;
    cv::Rect_<float> box;
}TrackingBox;

vector<KalmanTracker> sort_trackers;
vector<Rect_<float>> predictedBoxes;
vector<vector<double>> iouMatrix;
vector<int> assignment;
set<int> unmatchedDetections;
set<int> unmatchedTrajectories;
set<int> allItems;
set<int> matchedItems;
vector<cv::Point> matchedPairs;
vector<TrackingBox> frameTrackingResult;
unsigned int trkNum = 0;
unsigned int detNum = 0;




// Computes IOU between two bounding boxes
double GetIOU(Rect_<float> bb_test, Rect_<float> bb_gt)
{
    float in = (bb_test & bb_gt).area();
    float un = bb_test.area() + bb_gt.area() - in;

    if (un < DBL_EPSILON)
        return 0;

    return (double)(in / un);
}

typedef struct _FACE_DET_SINGLE_RESULT_
{
    cv::Rect		faceRect;				//face rectangle
    int  faceID=-1;                        //tracking id number
}FACE_DET_SINGLE_RESULT;

int Sort_multiFaceTrack(cv::Mat &imageData,std::vector<cv::Rect>& face_rect, std::vector<FACE_DET_SINGLE_RESULT>& FaceResults){
    //sort tracking part
    //the first frame operation
    if (sort_trackers.size()==0){
        for (int i=0;i<face_rect.size();i++){
            KalmanTracker temp_tracker=KalmanTracker(face_rect[i]);
            sort_trackers.push_back(temp_tracker);
        }
    }

    //get prediction location
    predictedBoxes.clear();
    std::cout<<"tracker 2"<<std::endl;
    for (auto it=sort_trackers.begin();it!=sort_trackers.end();){
        cv::Rect_<float> temp_box=(*it).predict();
        if (temp_box.x>=0 && temp_box.y>=0){
            predictedBoxes.push_back(temp_box);
            it++;
        }
        else{
            sort_trackers.erase(it);
        }
    }
    //associate detection result to the tracking object

    trkNum=predictedBoxes.size();
    detNum=face_rect.size();
    iouMatrix.clear();

    int min_hits=5;
    int max_age=5;

    //find matches,unmatched detections and unmatched predictions
    iouMatrix.clear();
    iouMatrix.resize(trkNum, vector<double>(detNum, 0));
    std::cout<<"tracker 21"<<std::endl;
    for (unsigned int i = 0; i < trkNum; i++) // compute iou matrix as a distance matrix
    {
        for (unsigned int j = 0; j < detNum; j++)
        {
            // use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
            iouMatrix[i][j] = 1 - GetIOU(predictedBoxes[i], face_rect[j]);
        }
    }
    std::cout<<"tracker 22"<<std::endl;
    HungarianAlgorithm HungAlgo;
    assignment.clear();
    if (trkNum>0)
    {HungAlgo.Solve(iouMatrix, assignment);}
    unmatchedDetections.clear();
    unmatchedTrajectories.clear();
    allItems.clear();
    matchedItems.clear();
    std::cout<<"tracking start 3"<<std::endl;
    //acquire the unmatched detections
    if (detNum >trkNum){
        for (int i=0;i<detNum;i++){
            allItems.insert(i);
        }
        for (int j=0;j<trkNum;++j){
            matchedItems.insert(assignment[j]);
        }
        set_difference(allItems.begin(),allItems.end(),matchedItems.begin(),matchedItems.end(),insert_iterator<set<int>>(unmatchedDetections,unmatchedDetections.begin()));
        //set_difference(engine->allItems.begin(),engine->allItems.end(),engine->matchedItems.begin(),engine->matchedItems.end(),engine->unmatchedDetections.begin());
    }
        //unmatched trajectory /predictions
    else if (detNum<trkNum){
        for (int i=0;i<trkNum;++i){
            if(assignment[i]==-1){
                unmatchedTrajectories.insert(i);//unassigned label will be labeled -1
            }
        }
    } else
        ;
    //filter out matched with low IOU
    std::cout<<"tracking start 4"<<std::endl;
    matchedPairs.clear();
    for (int i=0;i<trkNum;++i){
        //std::cout<<"iou saixuan "<<1 - engine->iouMatrix[i][engine->assignment[i]]<<std::endl;
        if (assignment[i]==-1)
            continue;

        if (1 - iouMatrix[i][assignment[i]] < 0.1){
            unmatchedTrajectories.insert(i);
            unmatchedDetections.insert(assignment[i]);
        }
        else
            matchedPairs.push_back(cv::Point(i,assignment[i]));
    }
    std::cout<<"tracking start 5"<<std::endl;
    //updating the trackers
    int detIdx,trkIdx;
    std::cout<<"all face is "<<face_rect.size()<<std::endl;
    for (int i=0;i<matchedPairs.size();i++){
        trkIdx=matchedPairs[i].x;
        detIdx=matchedPairs[i].y;
        FaceResults[detIdx].faceID=sort_trackers[trkIdx].m_id+1;

        std::cout<<"the face id is "<<detIdx<<" tracking id is "<<FaceResults[detIdx].faceID<<std::endl;
        sort_trackers[trkIdx].update(face_rect[detIdx]);
        cv::Mat face_temp;
        //if the detection box is matched, save the image to tracker vector
        //*************************************koutu**************************************//


    }
    //creating and initialize new trackers for unmatched detections
    std::cout<<"enter the unmatch tracker"<<std::endl;

    for (auto umd :unmatchedDetections){ //adding attrubute
        KalmanTracker traker=KalmanTracker(face_rect[umd]);
        std::cout<<"all id is"<<traker.m_id<<std::endl;
        sort_trackers.push_back(traker);
        std::cout<<"add new tracker "<<sort_trackers.size()<<std::endl;
    }
    //get tracker ' output

    frameTrackingResult.clear();
    for (auto it=sort_trackers.begin();it!=sort_trackers.end();){
        if (((*it).m_time_since_update<1) && ((*it).m_hit_streak>=min_hits )){
            TrackingBox res;
            res.box = (*it).get_state();
            res.id = (*it).m_id + 1;
            frameTrackingResult.push_back(res);
            it++;
        }
        else
            it++;

        //remove dead tracklet
        if(it!= sort_trackers.end() && (*it).m_time_since_update >max_age){
            it=sort_trackers.erase(it);
        }

    }
    // koutu alogrithm
}

int testJpg(){
    extern float pixel_mean[3];
    extern float pixel_std[3];
    std::string param_path =  "../models/retina.param";
    std::string bin_path = "../models/retina.bin";
    //std::string pfpld_path = "../models/pfpld.ncnnmodel";
    std::string pfpld_param ="/home/bill/E/dl_base2/nniefacelib/PFPLD/models/my_onnx/ncnn/pfldx0.25-160.param";
    std::string pfpld_bin ="/home/bill/E/dl_base2/nniefacelib/PFPLD/models/my_onnx/ncnn/pfldx0.25-160.bin";
    ncnn::Net _net, pfpld_net;
    _net.load_param(param_path.data());
    _net.load_model(bin_path.data());

    /*FILE *fp = fopen(pfpld_path.c_str(), "rb");
    if (fp != nullptr) {
        pfpld_net.load_param_bin(fp);
        pfpld_net.load_model(fp);
        fclose(fp);
    }*/

    pfpld_net.load_param(pfpld_param.c_str());
    pfpld_net.load_model(pfpld_bin.c_str());

    //cv::Mat img = cv::imread("../images/test1.jpg");
    cv::Mat img = cv::imread("/home/bill/E/dl_base2/nniefacelib/PFPLD/pfpld_ncnn/sample.jpg");
    if(!img.data)
        printf("load error");


    ncnn::Mat input = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR2RGB, img.cols, img.rows, img.cols, img.rows);
//    cv::resize(img, img, cv::Size(300, 300));

    input.substract_mean_normalize(pixel_mean, pixel_std);
    ncnn::Extractor _extractor = _net.create_extractor();
    _extractor.set_num_threads(4);
    _extractor.input("data", input);


    std::vector<AnchorGenerator> ac(_feat_stride_fpn.size());
    for (int i = 0; i < _feat_stride_fpn.size(); ++i) {
        int stride = _feat_stride_fpn[i];
        ac[i].Init(stride, anchor_cfg[stride], false);
    }

    std::vector<Anchor> proposals;
    proposals.clear();

    for (int i = 0; i < _feat_stride_fpn.size(); ++i) {
        ncnn::Mat cls;
        ncnn::Mat reg;
        ncnn::Mat pts;

        // get blob output
        char clsname[100]; sprintf(clsname, "face_rpn_cls_prob_reshape_stride%d", _feat_stride_fpn[i]);
        char regname[100]; sprintf(regname, "face_rpn_bbox_pred_stride%d", _feat_stride_fpn[i]);
        char ptsname[100]; sprintf(ptsname, "face_rpn_landmark_pred_stride%d", _feat_stride_fpn[i]);
        _extractor.extract(clsname, cls);
        _extractor.extract(regname, reg);
        _extractor.extract(ptsname, pts);

        printf("cls %d %d %d\n", cls.c, cls.h, cls.w);
        printf("reg %d %d %d\n", reg.c, reg.h, reg.w);
        printf("pts %d %d %d\n", pts.c, pts.h, pts.w);

        ac[i].FilterAnchor(cls, reg, pts, proposals);

        printf("stride %d, res size %d\n", _feat_stride_fpn[i], proposals.size());

        for (int r = 0; r < proposals.size(); ++r) {
            proposals[r].print();
        }
    }

    // nms
    std::vector<Anchor> result;
    nms_cpu(proposals, nms_threshold, result);

    printf("final result %d\n", result.size());

    for(int i = 0; i < result.size(); i ++)
    {
        //cv::rectangle (img, cv::Point((int)result[i].finalbox.x, (int)result[i].finalbox.y), cv::Point((int)result[i].finalbox.width, (int)result[i].finalbox.height), cv::Scalar(255, 255, 0), 2, 8, 0);
//        for (int j = 0; j < result[i].pts.size(); ++j) {
//        	cv::circle(img, cv::Point((int)result[i].pts[j].x, (int)result[i].pts[j].y), 1, cv::Scalar(225, 0, 225), 2, 8);
//        }
        int x1 = (int)result[i].finalbox.x;
        int y1 = (int)result[i].finalbox.y;
        int x2 = (int)result[i].finalbox.width;
        int y2 = (int)result[i].finalbox.height;
        int height = img.rows;
        int width = img.cols;
        int channel = img.channels();
        int w = x2 - x1 + 1;
        int h = y2 - y1 + 1;
        //track info



        int size_w = (int)(MAX(w, h)*0.9);
        int size_h = (int)(MAX(w, h)*0.9);
        int cx = x1 + w / 2;
        int cy = y1 + h / 2;
        x1 = cx - size_w / 2;
        x2 = x1 + size_w;
        y1 = cy - (int)(size_h * 0.4);
        y2 = y1 + size_h;

        int left = 0;
        int top = 0;
        int bottom = 0;
        int right = 0;
        if(x1 < 0)
            left = -x1;
        if (y1 < 0)
            top = -y1;
        if (x1 >= width)
            right = x2 - width;
        if (y1 >= height)
            bottom = y2 - height;

        x1 = MAX(0, x1);
        y1 = MAX(0, y1);

        x2 = MIN(width, x2);
        y2 = MIN(height, y2);

        cv::Mat face_img = img(cv::Rect(x1, y1, x2 - x1, y2 - y1));
        cv::copyMakeBorder(face_img, face_img, top, bottom, left, right, cv::BORDER_CONSTANT, 0);


        cv::resize(face_img, face_img, cv::Size(112, 112));
        //face_img = cv::imread("/home/bill/E/study/ncnn/ncnn_genderAge/test_img/Aaron_Peirsol_0002.jpg");
        //cv::imshow("tmp",face_img);
        //cv::waitKey(0);
        ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(face_img.data,ncnn::Mat::PIXEL_BGR2RGB, 112, 112);
        const float mean_vals[3] = {0.0f, 0.0f, 0.0f};
        const float norm_vals[3] = {1 / (float)255.0, 1 / (float)255.0, 1 / (float)255.0};//1 / (float)255.0
        //float norm_vals[3] = {1 , 1 , 1 };
        ncnn_img.substract_mean_normalize(mean_vals, norm_vals);
        ncnn::Mat pose, landms;
        std::vector<float> angles;
        std::vector<float> landmarks;
        for(int i=0;i<1;++i) {
            double t_start = cv::getTickCount();
            ncnn::Extractor pfpld_ex = pfpld_net.create_extractor();
            pfpld_ex.set_num_threads(4);//set

            pfpld_ex.input("input", ncnn_img);
            pfpld_ex.extract("pose", pose);
            pfpld_ex.extract("landms", landms);
//		    pfpld_ex.input(pfpld_param_id::BLOB_input, ncnn_img);
//
//
//			pfpld_ex.extract(pfpld_param_id::BLOB_pose, pose);
//			pfpld_ex.extract(pfpld_param_id::BLOB_landms, landms);


            double t_end = cv::getTickCount();
            float costTime = (t_end - t_start) / cv::getTickFrequency();
            std::cout << "cost Time:" << costTime << std::endl;
        }
        for (int j=0; j<pose.w; j++){
            //std::cout<<"pose[j]"<<pose[j]<<std::endl;
            float tmp_angle = pose[j] * 180.0 / CV_PI;
            angles.push_back(tmp_angle);
        }

        for (int j=0; j<landms.w / 2; j++)
        {
            float tmp_x = landms[2 * j] * size_w + x1 - left;
            float tmp_y = landms[2 * j + 1] * size_h + y1 -bottom;
            landmarks.push_back(tmp_x);
            landmarks.push_back(tmp_y);
            cv::circle(img, cv::Point((int)tmp_x, (int)tmp_y), 1, cv::Scalar(0,255,0), -1);
        }
        std::cout<<angles[0]<<"  "<<angles[1]<<"  "<<angles[2]<<std::endl;
        plot_pose_cube(img, angles[0], angles[1], angles[2], (int)result[i].pts[2].x, (int)result[i].pts[2].y, w / 2);
    }
    result[0].print();

    cv::imshow("img", img);
    //cv::imwrite("../images/result.jpg", img);
    cv::waitKey(0);
    return 0;
}

int testVideo(){
    extern float pixel_mean[3];
    extern float pixel_std[3];
    std::string param_path =  "../models/retina.param";
    std::string bin_path = "../models/retina.bin";
    std::string pfpld_param ="/home/bill/E/dl_base2/nniefacelib/PFPLD/models/my_onnx/ncnn/pfldx0.25-138.param";
    std::string pfpld_bin ="/home/bill/E/dl_base2/nniefacelib/PFPLD/models/my_onnx/ncnn/pfldx0.25-138.bin";
    ncnn::Net _net, pfpld_net;
    _net.load_param(param_path.data());
    _net.load_model(bin_path.data());

    cv::VideoCapture cap;
    //cap.open(0);//打开摄像头
    cap.open("/media/bill/Softmare-Document/Work/show_video/MVI_3958.MOV");


    int width = int(cap.get(CV_CAP_PROP_FRAME_WIDTH));
    int height = int(cap.get(CV_CAP_PROP_FRAME_HEIGHT));
    int fps = int(cap.get(CV_CAP_PROP_FPS));

    std::string outputfile ="video.avi";
    cv::VideoWriter writer;
    writer.open(outputfile,CV_FOURCC('M','J','P','G'),fps,cv::Size(width,height), true);
    if(!cap.isOpened())
        return 0;



    pfpld_net.load_param(pfpld_param.c_str());
    pfpld_net.load_model(pfpld_bin.c_str());

    //cv::Mat img = cv::imread("../images/test1.jpg");
    cv::Mat img;
    while(1) {
        cap >> img;//等价于cap.read(frame);
        if (img.empty())
            break;


        ncnn::Mat input = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR2RGB, img.cols, img.rows,
                                                        img.cols, img.rows);
//    cv::resize(img, img, cv::Size(300, 300));

        input.substract_mean_normalize(pixel_mean, pixel_std);
        ncnn::Extractor _extractor = _net.create_extractor();
        _extractor.input("data", input);


        std::vector<AnchorGenerator> ac(_feat_stride_fpn.size());
        for (int i = 0; i < _feat_stride_fpn.size(); ++i) {
            int stride = _feat_stride_fpn[i];
            ac[i].Init(stride, anchor_cfg[stride], false);
        }

        std::vector<Anchor> proposals;
        proposals.clear();

        for (int i = 0; i < _feat_stride_fpn.size(); ++i) {
            ncnn::Mat cls;
            ncnn::Mat reg;
            ncnn::Mat pts;

            // get blob output
            char clsname[100];
            sprintf(clsname, "face_rpn_cls_prob_reshape_stride%d", _feat_stride_fpn[i]);
            char regname[100];
            sprintf(regname, "face_rpn_bbox_pred_stride%d", _feat_stride_fpn[i]);
            char ptsname[100];
            sprintf(ptsname, "face_rpn_landmark_pred_stride%d", _feat_stride_fpn[i]);
            _extractor.extract(clsname, cls);
            _extractor.extract(regname, reg);
            _extractor.extract(ptsname, pts);

            printf("cls %d %d %d\n", cls.c, cls.h, cls.w);
            printf("reg %d %d %d\n", reg.c, reg.h, reg.w);
            printf("pts %d %d %d\n", pts.c, pts.h, pts.w);

            ac[i].FilterAnchor(cls, reg, pts, proposals);

            printf("stride %d, res size %d\n", _feat_stride_fpn[i], proposals.size());

            for (int r = 0; r < proposals.size(); ++r) {
                proposals[r].print();
            }
        }

        // nms
        std::vector<Anchor> result;
        nms_cpu(proposals, nms_threshold, result);

        printf("final result %d\n", result.size());
        std::vector<cv::Rect>face_rects;
        std::vector<_FACE_DET_SINGLE_RESULT_>face_tracks;
        for (int i = 0; i < result.size(); i++) {
            cv::rectangle(img, cv::Point((int) result[i].finalbox.x, (int) result[i].finalbox.y),
                          cv::Point((int) result[i].finalbox.width, (int) result[i].finalbox.height),
                          cv::Scalar(255, 255, 0), 2, 8, 0);

            int x1 = (int) result[i].finalbox.x;
            int y1 = (int) result[i].finalbox.y;
            int x2 = (int) result[i].finalbox.width;
            int y2 = (int) result[i].finalbox.height;
            int w = x2 - x1 + 1;
            int h = y2 - y1 + 1;
            face_rects.push_back(cv::Rect(x1,y1,w,h));
            _FACE_DET_SINGLE_RESULT_ ft;
            ft.faceRect=face_rects[i];
            face_tracks.push_back(ft);
        }
        Sort_multiFaceTrack(img ,face_rects,face_tracks);
        for(int i = 0; i < face_tracks.size(); i ++) {
            cv::rectangle(img, face_tracks[i].faceRect, cv::Scalar(0, 255, 255), 2, 8, 0);
            std::string method_sh=std::to_string(face_tracks[i].faceID);
            cv::putText(img, method_sh, cv::Point(face_tracks[i].faceRect.x, face_tracks[i].faceRect.y- 20),
                        CV_FONT_HERSHEY_SIMPLEX, 1.3, cv::Scalar(0, 255, 255));
            //process
            int height = img.rows;
            int width = img.cols;
            int x1 =face_tracks[i].faceRect.x;
            int y1 =face_tracks[i].faceRect.y;
            int w =face_tracks[i].faceRect.width;
            int h =face_tracks[i].faceRect.height;
            int size_w = (int) (MAX(w, h) * 0.9);
            int size_h = (int) (MAX(w, h) * 0.9);
            int cx = x1 + w / 2;
            int cy = y1 + h / 2;
            x1 = cx - size_w / 2;
            int x2,y2;
            x2 = x1 + size_w;
            y1 = cy - (int) (size_h * 0.4);
            y2 = y1 + size_h;

            int left = 0;
            int top = 0;
            int bottom = 0;
            int right = 0;
            if (x1 < 0)
                left = -x1;
            if (y1 < 0)
                top = -y1;
            if (x1 >= width)
                right = x2 - width;
            if (y1 >= height)
                bottom = y2 - height;

            x1 = MAX(0, x1);
            y1 = MAX(0, y1);

            x2 = MIN(width, x2);
            y2 = MIN(height, y2);

            cv::Mat face_img = img(cv::Rect(x1, y1, x2 - x1, y2 - y1));
            cv::copyMakeBorder(face_img, face_img, top, bottom, left, right, cv::BORDER_CONSTANT, 0);
            cv::resize(face_img, face_img, cv::Size(112, 112));

            ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(face_img.data, ncnn::Mat::PIXEL_BGR2RGB, 112, 112);
            const float mean_vals[3] = {0.0f, 0.0f, 0.0f};
            const float norm_vals[3] = {1 / (float)255.0, 1 / (float)255.0, 1 / (float)255.0};//1 / (float)255.0
            //float norm_vals[3] = {1 , 1 , 1 };
            ncnn_img.substract_mean_normalize(mean_vals, norm_vals);
            ncnn::Mat pose, landms;
            std::vector<float> angles;
            std::vector<float> landmarks;
            for (int i = 0; i < 1; ++i) {
                double t_start = cv::getTickCount();
                ncnn::Extractor pfpld_ex = pfpld_net.create_extractor();
                pfpld_ex.set_num_threads(4);//set

                pfpld_ex.input("input", ncnn_img);
                pfpld_ex.extract("pose", pose);
                pfpld_ex.extract("landms", landms);

                double t_end = cv::getTickCount();
                float costTime = (t_end - t_start) / cv::getTickFrequency();
                std::cout << "cost Time:" << costTime << std::endl;
            }
            for (int j = 0; j < pose.w; j++) {
                float tmp_angle = pose[j] * 180.0 / CV_PI;
                angles.push_back(tmp_angle);
            }

            for (int j = 0; j < landms.w / 2; j++) {
                float tmp_x = landms[2 * j] * size_w + x1 - left;
                float tmp_y = landms[2 * j + 1] * size_h + y1 - bottom;
                landmarks.push_back(tmp_x);
                landmarks.push_back(tmp_y);
                cv::circle(img, cv::Point((int) tmp_x, (int) tmp_y), 2, cv::Scalar(0, 0, 255), -1);
            }
            std::cout << angles[0] << "  " << angles[1] << "  " << angles[2] << std::endl;
            plot_pose_cube(img, angles[0], angles[1], angles[2], (int) result[i].pts[2].x, (int) result[i].pts[2].y,
                           w / 2);
        }

        result[0].print();

        cv::imshow("img", img);
        //cv::imwrite("../images/result.jpg", img);
        writer<<img;
        cv::waitKey(1);
    }
    cap.release();
    writer.release();
    return 0;
}
int main(int argv, char** argc) {
    int s = atoi(argc[1]);
    switch (s) {
        case 0:testJpg();
            break;
            //case 1:testJpgList();
            //break;
        case 2:
            testVideo();
            break;

        default:break;
    }
    std::cout<<"hello2";
}



