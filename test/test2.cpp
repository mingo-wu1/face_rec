#include <iostream>
#include <librealsense2/hpp/rs_frame.hpp>
#include <librealsense2/hpp/rs_pipeline.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/video.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <librealsense2/rs.hpp>

using namespace cv;
using namespace std;

// int main(){
//     VideoCapture capture;
//     capture.release();
//     capture = VideoCapture("/dev/video2");

//     rs2::pipeline p;
//     p.start();
//     rs2::frameset frames = p.wait_for_frames();
//     rs2::depth_frame depth

//     if(capture.isOpened()){
//         std::cout<<"fuck"<<std::endl;
//     }
//     return 0;
// }

#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <iostream>
#include <opencv2/opencv.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <vector>
#include <sstream>
#include <dlib/compress_stream.h>
#include <dlib/base64.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>
#include <dlib/image_io.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <string.h>

using namespace dlib;

void line_one_face_detections(cv::Mat img, std::vector<dlib::full_object_detection> fs)
{
    int i, j;
    for(j=0; j<fs.size(); j++)
    {
        cv::Point p1, p2;
        for(i = 0; i<67; i++)
        {
            // 下巴到脸颊 0 ~ 16
            //左边眉毛 17 ~ 21
            //右边眉毛 21 ~ 26
            //鼻梁     27 ~ 30
            //鼻孔        31 ~ 35
            //左眼        36 ~ 41
            //右眼        42 ~ 47
            //嘴唇外圈  48 ~ 59
            //嘴唇内圈  59 ~ 67
            switch(i)
            {
                case 16:
                case 21:
                case 26:
                case 30:
                case 35:
                case 41:
                case 47:
                case 59:
                    i++;
                    break;
                default:
                    break;
            }

            p1.x = fs[j].part(i).x();
            p1.y = fs[j].part(i).y();
            p2.x = fs[j].part(i+1).x();
            p2.y = fs[j].part(i+1).y();
            cv::line(img, p1, p2, cv::Scalar(0,0,255), 2, 4, 0);
        }
    }
};

int main(int argc, char * argv[]) try
{
// test
    struct timespec ts_start, ts_end;
//test


    rs2::log_to_console(RS2_LOG_SEVERITY_ERROR);

    rs2::config cfg;
    ///设置从设备管道获取的深度图和彩色图的配置对象
    ///配置彩色图像流：分辨率640*480，图像格式：BGR， 帧率：30帧/秒
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    ///配置深度图像流：分辨率640*480，图像格式：Z16， 帧率：30帧/秒
//   cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);

    ///生成Realsense管道，用来封装实际的相机设备
    rs2::pipeline pipe;
    ///根据给定的配置启动相机管道
    pipe.start(cfg);

    rs2::frameset data;
//   while(1)
//   {
    ///等待一帧数据，默认等待5s
    // data = pipe.wait_for_frames();

    // rs2::frame depth  = data.get_depth_frame(); ///获取深度图像数据
    // rs2::frame color = data.get_color_frame();  ///获取彩色图像数据
    // if (!color || !depth) break;            ///如果获取不到数据则退出
    // if(!color) break;

    ///将彩色图像和深度图像转换为Opencv格式
    // cv::Mat image(cv::Size(640, 480), CV_8UC3, (void*)color.get_data(), cv::Mat::AUTO_STEP);
    // cv::Mat depthmat(cv::Size(640, 480), CV_16U, (void*)depth.get_data(), cv::Mat::AUTO_STEP);

    // dlib begin
    cv::Mat image = imread("./man.jpeg");
    //提取灰度图
    cv::cvtColor(image, image, COLOR_BGR2GRAY);

    //加载dlib的人脸识别器
    // dlib::array<array2d<unsigned char> > images;
    // std::vector<std::vector<full_object_detection> > objects;
    frontal_face_detector detector = get_frontal_face_detector();
    // shape_predictor_trainer trainer;
    // trainer.set_tree_depth(2);
    // trainer.set_nu(0.05);
    //trainer.be_verbose();
    // shape_predictor sp = trainer.train(images, objects);     //加载人脸形状探测器
    // It should have been able to perfectly fit the data
    // While we are here, make sure the default face detector works
    // std::vector<dlib::rectangle> dets = detector(images[0]);

    //加载人脸形状探测器
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    dlib::shape_predictor sp;
    dlib::deserialize("./data") >> sp;
    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    printf("CLOCK_MONOTONIC1 reports %ld.%09ld seconds\n", ts_end.tv_sec - ts_start.tv_sec, ts_end.tv_nsec - ts_start.tv_nsec);

    //Mat转化为dlib的matrix
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    dlib::array2d<dlib::bgr_pixel> dlibImg;
    IplImage iplImage = cvIplImage(image);
    dlib::assign_image(dlibImg, dlib::cv_image<unsigned char>(&iplImage));
    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    printf("CLOCK_MONOTONIC2 reports %ld.%09ld seconds\n", ts_end.tv_sec - ts_start.tv_sec, ts_end.tv_nsec - ts_start.tv_nsec);


    //获取一系列人脸所在区域
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    std::vector<dlib::rectangle> dets = detector(dlibImg);
    std::cout << "Number of faces detected: " << dets.size() << std::endl;
    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    printf("CLOCK_MONOTONIC3 reports %ld.%09ld seconds\n", ts_end.tv_sec - ts_start.tv_sec, ts_end.tv_nsec - ts_start.tv_nsec);

    // if (dets.size() == 0)
    //     return 0;

    //获取人脸特征点分布
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    std::vector<dlib::full_object_detection> shapes;
    for(int i = 0; i < dets.size(); i++)
    {
        dlib::full_object_detection shape = sp(dlibImg, dets[i]); //获取指定一个区域的人脸形状
        shapes.push_back(shape);
    }
    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    printf("CLOCK_MONOTONIC4 reports %ld.%09ld seconds\n", ts_end.tv_sec - ts_start.tv_sec, ts_end.tv_nsec - ts_start.tv_nsec);

    //指出每个检测到的人脸的位置
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    for(int i=0; i<dets.size(); i++)
    {
        //画出人脸所在区域
        cv::Rect r;
        r.x = dets[i].left();
        r.y = dets[i].top();
        r.width = dets[i].width();
        r.height = dets[i].height();
        cv::rectangle(image, r, cv::Scalar(0, 0, 255), 1, 1, 0);
    }
    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    printf("CLOCK_MONOTONIC5 reports %ld.%09ld seconds\n", ts_end.tv_sec - ts_start.tv_sec, ts_end.tv_nsec - ts_start.tv_nsec);

    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    line_one_face_detections(image, shapes);
    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    printf("CLOCK_MONOTONIC6 reports %ld.%09ld seconds\n", ts_end.tv_sec - ts_start.tv_sec, ts_end.tv_nsec - ts_start.tv_nsec);

    // dlib end

    ///显示
    cv::imshow("image",image);
    // cv::imshow("depth",depthmat);
    cv::waitKey(1000000);
//   }
    return EXIT_SUCCESS;
}
catch (const rs2::error & e)
{	///捕获相机设备的异常
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception& e)
{
    std::cerr<<"Error1 : " << e.what() << std::endl;
    return EXIT_FAILURE;
}