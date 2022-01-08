#include <iostream>
#include <librealsense2/hpp/rs_frame.hpp>
#include <librealsense2/hpp/rs_pipeline.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/video.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <librealsense2/rs.hpp>
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
#include <dlib/image_io.h>
#include <dlib/opencv.h>
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/clustering.h>
#include <dlib/string.h>

using namespace cv;
using namespace std;

const std::string imagePath = "../images/man.jpeg";
const std::string dlibModel = "../dataset/data";
const std::string cvModel = "../dataset/haarcascade_frontalface_alt2.xml";
const std::string dnnModel = "../dataset/resnet_data";

void getFiles(std::string path, std::map<std::string, std::string> &files){
    DIR *dir;
    struct dirent *ptr;
    char base[1000];
 
    if(path[path.length()-1] != '/')
        path = path + "/";
 
    if((dir = opendir(path.c_str())) == NULL)
    {
        cout<<"open the dir: "<< path <<"error!" <<endl;
        return;
    }
 
    while((ptr=readdir(dir)) !=NULL )
    {
        ///current dir OR parrent dir 
        if(strcmp(ptr->d_name,".")==0 || strcmp(ptr->d_name,"..")==0) 
            continue; 
        else if(ptr->d_type == 8) //file
        {
            string fn(ptr->d_name);
            string name;
            name = fn.substr(0, fn.find_last_of("."));
 
            string p = path + string(ptr->d_name);
            files.insert(pair<string, string>(p, name));
        }
        else if(ptr->d_type == 10)    ///link file
        {}
        else if(ptr->d_type == 4)    ///dir
        {}
    }
 
    closedir(dir);
    return ;
}
 
template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = dlib::add_prev1<block<N,BN,1,dlib::tag1<SUBNET>>>;
 
template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = dlib::add_prev2<dlib::avg_pool<2,2,2,2,dlib::skip1<dlib::tag2<block<N,BN,2,dlib::tag1<SUBNET>>>>>>;
 
template <int N, template <typename> class BN, int stride, typename SUBNET> 
using block  = BN<dlib::con<N,3,3,1,1,dlib::relu<BN<dlib::con<N,3,3,stride,stride,SUBNET>>>>>;
 
template <int N, typename SUBNET> using ares      = dlib::relu<residual<block,N,dlib::affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = dlib::relu<residual_down<block,N,dlib::affine,SUBNET>>;
 
template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;
 
using anet_type = dlib::loss_metric<dlib::fc_no_bias<128,dlib::avg_pool_everything<
                            alevel0<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            dlib::max_pool<3,3,2,2,dlib::relu<dlib::affine<dlib::con<32,7,7,2,2,
                            dlib::input_rgb_image_sized<150>
                            >>>>>>>>>>>>;

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


//   rs2::log_to_console(RS2_LOG_SEVERITY_ERROR);

//   rs2::config cfg;
  ///设置从设备管道获取的深度图和彩色图的配置对象
  ///配置彩色图像流：分辨率640*480，图像格式：BGR， 帧率：30帧/秒
//   cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
  ///配置深度图像流：分辨率640*480，图像格式：Z16， 帧率：30帧/秒
//   cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);

  ///生成Realsense管道，用来封装实际的相机设备
//   rs2::pipeline pipe;
  ///根据给定的配置启动相机管道
//   pipe.start(cfg);
  
//   rs2::frameset data;
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

//-----------
//------------先做一次检测
    dlib::shape_predictor sp;
    dlib::deserialize(dlibModel) >> sp;
    cv::CascadeClassifier faceDetector(cvModel);
    if(faceDetector.empty())
    {
        std::cout << "face detector is empty!" <<std::endl;
        return 0;
    }
    //加载负责人脸识别的DNN
    anet_type net;
    dlib::deserialize(dnnModel) >> net;
 
    //人脸描述符库, face_descriptor ---> name
    map<dlib::matrix<float,0,1>, string> fdlib;
    
    std::map<string, string> files;
    getFiles(argv[1], files);
 
    if(files.empty())
    {
        std::cout<< "No pic files found in "<< argv[1] <<std::endl;
        return 0;
    }
    for(map<string, string>::iterator it = files.begin(); it != files.end(); it++  )
    {
        std::cout << "filename:" << it->second << " filepath:" <<it->first<<std::endl;
 
        cv::Mat frame = cv::imread(it->first);
        cv::Mat src;
        cv::cvtColor(frame, src, CV_BGR2GRAY);
        dlib::array2d<dlib::bgr_pixel> dimg;
        dlib::assign_image(dimg, dlib::cv_image<uchar>(src)); 
 
        //haar级联分类器探测人脸区域，获取一系列人脸所在区域
        std::vector<cv::Rect> objects;
        std::vector<dlib::rectangle> dets;
        faceDetector.detectMultiScale(src, objects);
        for (int i = 0; i < objects.size(); i++)
        {
            //cv::rectangle(frame, objects[i], CV_RGB(200,0,0));
            dlib::rectangle r(objects[i].x, objects[i].y, objects[i].x + objects[i].width, objects[i].y + objects[i].height);
            dets.push_back(r);  //正常情况下应该只检测到一副面容
        }
 
        if (dets.size() == 0)
            continue;
 
        std::vector<dlib::matrix<dlib::rgb_pixel>> faces;
        std::vector<dlib::full_object_detection> shapes;
        for(int i = 0; i < dets.size(); i++)
        {
            dlib::full_object_detection shape = sp(dimg, dets[i]); //获取指定一个区域的人脸形状
            shapes.push_back(shape); 
 
            dlib::matrix<dlib::rgb_pixel> face_chip;
            dlib::extract_image_chip(dimg, dlib::get_face_chip_details(shape,150,0.25), face_chip);
 
            faces.push_back(move(face_chip));
        }
 
        if (faces.size() == 0)
        {
            cout << "No faces found in " << it->second<<endl;
            continue;
        }
 
        std::vector<dlib::matrix<float,0,1>> face_descriptors = net(faces);
 
        for(std::vector<dlib::matrix<float,0,1>>::iterator iter = face_descriptors.begin(); iter != face_descriptors.end(); iter++ )
        {
            fdlib.insert(pair<dlib::matrix<float,0,1>, string>(*iter, it->second));
        }
    }
//------------
//-----------

    // dlib begin 用于非人脸比对
    // cv::Mat image = imread(imagePath);
    // if(image.empty()){
    //     std::cout<<"image empty, change image path."<<std::endl;
    // }
    // //提取灰度图
    // cv::cvtColor(image, image, CV_BGR2GRAY);

    //加载dlib的人脸识别器
    // dlib::array<array2d<unsigned char> > images;
    // std::vector<std::vector<full_object_detection> > objects;
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
    // dlib::shape_predictor sp;
    // dlib::deserialize(dlibModel) >> sp;
    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    printf("CLOCK_MONOTONIC1 reports %ld.%09ld seconds\n", ts_end.tv_sec - ts_start.tv_sec, ts_end.tv_nsec - ts_start.tv_nsec);

    //Mat转化为dlib的matrix 用于非人脸比对
    // clock_gettime(CLOCK_MONOTONIC, &ts_start);
    // dlib::array2d<dlib::bgr_pixel> dlibImg;
    // dlib::assign_image(dlibImg, dlib::cv_image<unsigned char>(image));
    // clock_gettime(CLOCK_MONOTONIC, &ts_end);
    // printf("CLOCK_MONOTONIC2 reports %ld.%09ld seconds\n", ts_end.tv_sec - ts_start.tv_sec, ts_end.tv_nsec - ts_start.tv_nsec);
//----------
    //获取一系列人脸所在区域
    // clock_gettime(CLOCK_MONOTONIC, &ts_start);
    // dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    // std::vector<dlib::rectangle> dets = detector(dlibImg);
    // std::cout << "Number of faces detected: " << dets.size() << std::endl;
    // clock_gettime(CLOCK_MONOTONIC, &ts_end);
    // printf("CLOCK_MONOTONIC3 reports %ld.%09ld seconds\n", ts_end.tv_sec - ts_start.tv_sec, ts_end.tv_nsec - ts_start.tv_nsec);
    //if (dets.size() == 0)
    //     return 0;
//-----------
//-----------
    //haar级联分类器探测人脸区域，获取一系列人脸所在区域
    // clock_gettime(CLOCK_MONOTONIC, &ts_start);
    // cv::CascadeClassifier faceDetector(cvModel);
    // if(faceDetector.empty())
    // {
    //     std::cout << "face detector is empty!" <<std::endl;
    //     // return 0;
    // }
    // std::vector<cv::Rect> objects;
    // std::vector<dlib::rectangle> dets;
    // faceDetector.detectMultiScale(image, objects);
    // for (int i = 0; i < objects.size(); i++)
    // {
    //     cv::rectangle(image, objects[i], CV_RGB(200,0,0));
    //     dlib::rectangle r(objects[i].x, objects[i].y, objects[i].x + objects[i].width, objects[i].y + objects[i].height);
    //     dets.push_back(r);
    // } 
    // clock_gettime(CLOCK_MONOTONIC, &ts_end);
    // printf("CLOCK_MONOTONIC3 reports %ld.%09ld seconds\n", ts_end.tv_sec - ts_start.tv_sec, ts_end.tv_nsec - ts_start.tv_nsec);
//    if (dets.size() == 0)
//        return 0;
//------------
//------------用于人脸比对检测的函数段
    //haar级联分类器探测人脸区域，获取一系列人脸所在区域
    cv::Mat image = cv::imread(argv[2]);
    cv::Mat src;
    cv::cvtColor(image, src, CV_BGR2GRAY);
    dlib::array2d<dlib::bgr_pixel> dlibImg;
    dlib::assign_image(dlibImg, dlib::cv_image<uchar>(src));
    std::vector<cv::Rect> objects;
    std::vector<dlib::rectangle> dets;
    faceDetector.detectMultiScale(image, objects);
    for (int i = 0; i < objects.size(); i++)
    {
        cv::rectangle(image, objects[i], CV_RGB(200,0,0));
        dlib::rectangle r(objects[i].x, objects[i].y, objects[i].x + objects[i].width, objects[i].y + objects[i].height);
        dets.push_back(r);  //正常情况下应该只检测到一副面容
    }
 
    if (dets.size() == 0)
    {
        cout << "there is no faces found in " << argv[2] <<endl;
        return -1;
    }
 
    std::vector<dlib::matrix<dlib::rgb_pixel>> faces;
    std::vector<dlib::full_object_detection> shapes;
    for(int i = 0; i < dets.size(); i++)
    {
        dlib::full_object_detection shape = sp(dlibImg, dets[i]); //获取指定一个区域的人脸形状
        shapes.push_back(shape); 
 
        dlib::matrix<dlib::rgb_pixel> face_chip;
        dlib::extract_image_chip(dlibImg, dlib::get_face_chip_details(shape,150,0.25), face_chip);
 
        faces.push_back(move(face_chip));
    }
    if (faces.size() == 0)
    {
        cout << "No faces found in " << argv[2] <<endl;
        return -1;
    }
//------------
//------------仅仅需要这段用于视频
//加载视频
    // VideoCapture capture(argv[2]);
    // int frames = capture.get(CAP_PROP_FRAME_COUNT);//获取视频针数目(一帧就是一张图片)
    // double fps = capture.get(CAP_PROP_FPS);//获取每针视频的频率
    // // 获取帧的视频宽度，视频高度
    // Size size = Size(capture.get(CAP_PROP_FRAME_WIDTH), capture.get(CAP_PROP_FRAME_HEIGHT));
    // cout << frames << endl;
    // cout << fps << endl;
    // cout << size << endl;
    // //创建写入对象,需要指定，帧率和视频宽高
    // VideoWriter writer;
    // //指定保存文件位置，编码器，帧率，宽高
    // writer.open("test.avi", VideoWriter::fourcc('M','J','P','G'), fps, size);
    // //VideoWriter writer("VideoTest.avi",CV_FOURCC('M', 'J', 'P', 'G'), 20.0,Size(480, 848));  
    // while(true)
    // {
    //     //加载待检测的图片
    //     cv::Mat frame;
    //     capture >> frame;
    //     if (frame.empty())
    //         break;

    //     cv::Mat src;
    //     cv::cvtColor(frame, src, CV_BGR2GRAY);
    //     dlib::array2d<dlib::bgr_pixel> dimg;
    //     dlib::assign_image(dimg, dlib::cv_image<uchar>(src));

    //     //haar级联分类器探测人脸区域，获取一系列人脸所在区域
    //     std::vector<cv::Rect> objects;
    //     std::vector<dlib::rectangle> dets;
    //     faceDetector.detectMultiScale(src, objects);
    //     for (int i = 0; i < objects.size(); i++)
    //     {
    //         cv::rectangle(frame, objects[i], CV_RGB(200,0,0));
    //         dlib::rectangle r(objects[i].x, objects[i].y, objects[i].x + objects[i].width, objects[i].y + objects[i].height);
    //         dets.push_back(r);  //正常情况下应该只检测到一副面容
    //     }

    //     if (dets.size() == 0)
    //     {
    //         continue;
    //     }

    //     std::vector<dlib::matrix<dlib::rgb_pixel>> faces;
    //     std::vector<dlib::full_object_detection> shapes;
    //     for(int i = 0; i < dets.size(); i++)
    //     {
    //         dlib::full_object_detection shape = sp(dimg, dets[i]); //获取指定一个区域的人脸形状
    //         shapes.push_back(shape); 

    //         dlib::matrix<dlib::rgb_pixel> face_chip;
    //         dlib::extract_image_chip(dimg, dlib::get_face_chip_details(shape,150,0.25), face_chip);

    //         faces.push_back(move(face_chip));
    //     }
    //     if (faces.size() == 0)
    //     {
    //         continue;
    //     }
    //     line_one_face_detections(frame, shapes);

    //     std::vector<dlib::matrix<float,0,1>> face_descriptors = net(faces);

    //     //遍历库，查找相似图像
    //     float min_distance = 0.7;
    //     std::string similar_name = "unknown";
    //     for(map<dlib::matrix<float,0,1>, string>::iterator it=fdlib.begin(); it != fdlib.end(); it++ )
    //     {
    //         float distance = length(it->first - face_descriptors[0]);
    //         if( distance < 0.5 )  //应该计算一个最近值
    //         {
    //             if( distance <= min_distance)
    //             {
    //                 min_distance = distance;
    //                 similar_name = it->second;
    //             }
    //         }
    //     }

    //     if(min_distance < 0.5)
    //     {
    //         float similarity = (0.5 - min_distance) * 100 / 0.5;
    //         stringstream strStream; 
    //         strStream << similar_name << ", " << similarity << '%' << endl;
    //         string s = strStream.str();
    //         cv::Point org(objects[0].x, objects[0].y);
    //         cv::putText(frame, s, org, cv::FONT_HERSHEY_SIMPLEX, 1.0, CV_RGB(0, 200, 0));
    //     }
//------------

    //获取人脸特征点分布
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    // std::vector<dlib::full_object_detection> shapes;
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

//-------------------用于人脸比对函数段
    std::vector<dlib::matrix<float,0,1>> face_descriptors = net(faces);
 
    //遍历库，查找相似图像
    for(map<dlib::matrix<float,0,1>, string>::iterator it=fdlib.begin(); it != fdlib.end(); it++ )
    {
        float distance = length(it->first - face_descriptors[0]);
        if( distance < 0.6 )
        {
            cout << "the pic is " << it->second << "!, distance:" << distance << endl;
 
            cv::Point org(objects[0].x, objects[0].y);
            cv::putText(image, it->second, org, cv::FONT_HERSHEY_SIMPLEX, 1.0, CV_RGB(0, 200, 0));
            break;
        }
    }
//-------------------

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






#include "opencv2/opencv.hpp"
#include <iostream>
#include <opencv2/videoio.hpp>
#include <string>
using namespace cv;
using namespace std;

// example 1
// 实时视频采集
// int main(){
//     // TODO: 添加控制代码
//     VideoCapture capture;
//     capture.release();// 释放避免连续打开两次
//     capture = VideoCapture(0);// 初始化摄像头
//     bool isOpen = true;

//     if(!capture.isOpened()){
//         std::cout<<"cannot open camera."<<std::endl;
//     }

//     while(isOpen){
//         Mat frame;
//         capture >> frame;
//         cv::resize(frame, m_dst, cv::Size(x, y), 0,0,1);
//         cv::imshow("view", m_dst);
//         cv::waitKey(25);
//     }
//     return 0;
// }

// example 2
// 图像抓拍
// int main(){
//     string path = "path.jpg";
//     imwrite(path, m_dst);
//     Mat image = imread(path);
//     Mat imagedst;
// }

// example 3
// 人脸检测功能
// int main(){
//     m_bdetect = 1;
//     if(m_brectangle){
//         m_brectangle = 0;
//     }
//     else{
//         m_brectangle = 1;
//     }
//     Mat frame;
//     capture >> frame
//     cv::resize()

//     cv_image<bgr_pixel> cimg(frame) // 检测单帧，图片对象用frame来初始化

//     if(m_bdetect){
//         std::vector<dlib::rectangle> faces = detector(cimg);//所有人脸，dlib 里面的函数 检测人脸s， 你理解就是矩形框，有可能是一个也有可能是多个
//         std::vector<full_object_detection> shapes; // 所有人脸对象，找出每个人脸的位置的对象

//         for(unsigned long i = 0; i < faces.size(); i++){
//             if(m_brectangle){//人脸检测并绘制矩形框
//                 cv::rectangle(frame，Rect(faces[i]))//论讯一次对着人脸绘制一个框
//             }
//             shapes.push_back(pose_model(cimg, faces[i]));//压入人脸对象shapes
//         }
//         if(!shapes.empty())
//         {
            
//         }
//     }
//     cv::imshow("view", frame);
//     cv::waitKey(30);
// }

// example 4
// 人脸检测2
// int main(){
//     //打开摄像头
//     capture = VideoCapture(0);
//     shape_predictor pose_model;
//     deserialize("特征文件.dat") >> pose_model; // 人脸检测模型文件

//     m_bdetect = 1;
//     if(m_brectangle){
//         m_brectangle = 0;
//     }
//     else{
//         m_brectangle = 1;
//     }
//     Mat frame;
//     capture >> frame;
//     cv::resize();

//     cv_image<bgr_pixel> cimg(frame); // 检测单帧，图片对象用frame来初始化

//     if(m_bdetect){
//         std::vector<dlib::rectangle> faces = detector(cimg);//所有人脸，dlib 里面的函数 检测人脸s， 你理解就是矩形框，有可能是一个也有可能是多个
//         std::vector<full_object_detection> shapes; // 所有人脸对象，找出每个人脸的位置的对象

//         for(unsigned long i = 0; i < faces.size(); i++){
//             if(m_brectangle){//人脸检测并绘制矩形框
//                 cv::rectangle(frame，Rect(faces[i]))//论讯一次对着人脸绘制一个框
//             }
//             shapes.push_back(pose_model(cimg, faces[i]));//压入人脸对象shapes， pose_model 人脸特征位置
//         }
//         if(!shapes.empty())
//         {
            
//         }
//     }
//     cv::imshow("view", frame);
//     cv::waitKey(30);   
// }

// example 5
// 特征点的标定
// int main(){
//     // 特征标定： 68
//     if(!shapes.empty())
//     {
//         for(int i = 0; i< 68; ++i)// 标出68个点
//         {
//             if(!m_brectangle){// 绘制特征点
//                 cv::circle(frame, cvPoint(shape[0].part(i)))
//                 putText(frame, to_string(i));
//             }
//         }
//     }
    // 对齐
    //     dlib::array<array2d<bgr_pixel>> face_chips;
    // dlib::extract_image_chips(img.get_face_chip_details(shapes), face_chips);
    // for(size_t i = 0; i< face_chips.size(); i++){
    //     cv::Mat img = dlib::toMat(face_chips[i]);
    //     string picpath;
    //     stringstream stream;
    //     stream << i;
    //     picpath = stream.str();
    //     picpath += ".jpg";
    //     picpath = "d://" + picpath; //把倾斜的人脸对齐以后的目标保存下来然后显示
    //     imwrite(picpath,img);
    //     show(); // 显示
    // }

// }

// example 6
// 人脸特征点矫正，对齐, face_chips 为对齐之后的目标
// int main(){
//     dlib::array<array2d<bgr_pixel>> face_chips;
//     dlib::extract_image_chips(img.get_face_chip_details(shapes), face_chips);
//     for(size_t i = 0; i< face_chips.size(); i++){
//         cv::Mat img = dlib::toMat(face_chips[i]);
//         string picpath;
//         stringstream stream;
//         stream << i;
//         picpath = stream.str();
//         picpath += ".jpg";
//         picpath = "d://" + picpath; //把倾斜的人脸对齐以后的目标保存下来然后显示
//         imwrite(picpath,img);
//         showOnRect(); // 显示
//     }
// }

// example 7
// 目标跟踪
// 1.获取第一帧，确定跟踪位置
// 2.不断跟踪后续的目标
// int main(){
//     correlation_tracker m_tracker; // 目标跟踪
//     if(m_btrack){
//         if(bFirst){
//             m_tracker.start_track(cimg, centered_rect(point(200,110), 86, 86));//开启追踪
//             m_win.set_image(cimg);
//             m_win.add_overlay(m_tracker.get_position());
//             bFirst = 0;
//         }
//         m_tracker.update(cimg);
//         m_win.set_image(cimg);
//         m_win.clear_overlay();
//         m_win.add_overlay(m_tracker.get_position());
//     }
// }

// example 8
// 人脸特征点进行比对， 离线对比
// int main(){
    
//     // 初始化检测器+图片
//     frontal_face_detector detector = get_frontal_face_detector();

//     // 导入模型：特征点标定
//     shape_predictor sp;
//     deserialize("dat") >> sp;
    
//     //导入模型：人脸识别
//     anet_type net;
//     deserialize("") >> net;

//     string path1 = "";//针对图片进行人脸识别
//     string path2 = "";//针对图片进行人脸识别
//     matrix<rgb_pixel> img1;
//     load_image(img1, path1);

//     matrix<rgb_pixel> img2;
//     load_image(img2, path2);

//     show()// 显示图片
//     show()// 显示图片

//     // 特征点对齐
//     std::vector<matrix<rgb_pixel>> faces1;
//     std::vector<matrix<rgb_pixel>> faces2;

//     // 人脸检测img1
//     //注意：检测多张人脸时，会触发中断
//     for(auto face:detector(img1)){}
//     if(faces1.size()==0){

//     }
//     for(auto face:detector(img2)){

//     }
//     if(faces2.size() == 0){

//     }

//     // 人脸特征向量化
//     std::vector<matrix<float, 0, 1>> face_descriptors1 = net(faces1);
//     std::vector<matrix<float, 0, 1>> face_descriptors2 = net(faces2);

//     // 矩利计算
//     float f = length(face_descriptors1[0] - face_descriptors2[0]);// 0.6-0.7 70%的相似度， 0.25是0.75的相似度
//     std::cout<<"oushijuli"<<f<<std::endl;
//     show();
// }

// example 9 
// 活体检测 眨眼识别
int main(){
    
}