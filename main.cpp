#include "dlib/dnn/input.h"
#include "dlib/dnn/layers.h"
#include "dlib/dnn/loss.h"
#include "dlib/image_processing/shape_predictor.h"
#include "dlib/opencv/cv_image.h"
#include <dirent.h>
#include <dlib/dnn.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <iostream>
#include <librealsense2/hpp/rs_pipeline.hpp>
#include <map>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <string.h>

/**
 * @brief 数据接口层
 */
class Subject {
public:
  virtual void Request() = 0;
  virtual void GetFile(std::string path,
                       std::map<std::string, std::string> &files) = 0;
};

class Proxy : public Subject {
public:
  Proxy(Subject *subject) : _subject(subject) {}

  void Request() override { _subject->Request(); }

  void GetFile(std::string path,
               std::map<std::string, std::string> &files) override {
    _subject->GetFile(path, files);
  }

private:
  Subject *_subject;
};

class RealSubject : public Subject {
public:
  void Request() override {}

  void GetFile(std::string path,
               std::map<std::string, std::string> &files) override {
    DIR *dir;
    struct dirent *ptr;
    char base[1000];

    if (path[path.length() - 1] != '/')
      path = path + "/";

    if ((dir = opendir(path.c_str())) == NULL) {
      std::cout << "open the dir: " << path << "error!" << std::endl;
      return;
    }

    while ((ptr = readdir(dir)) != NULL) {
      /// current dir OR parrent dir
      if (strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0)
        continue;
      else if (ptr->d_type == 8) // file
      {
        std::string fn(ptr->d_name);
        std::string name;
        name = fn.substr(0, fn.find_last_of("."));

        std::string p = path + std::string(ptr->d_name);
        files.insert(std::pair<std::string, std::string>(p, name));
      } else if (ptr->d_type == 10) /// link file
      {
      } else if (ptr->d_type == 4) /// dir
      {
      }
    }

    closedir(dir);
    return;
  }
};

template <template <int, template <typename> class, int, typename> class block,
          int N, template <typename> class BN, typename SUBNET>
using residual = dlib::add_prev1<block<N, BN, 1, dlib::tag1<SUBNET>>>;

template <template <int, template <typename> class, int, typename> class block,
          int N, template <typename> class BN, typename SUBNET>
using residual_down = dlib::add_prev2<dlib::avg_pool<
    2, 2, 2, 2, dlib::skip1<dlib::tag2<block<N, BN, 2, dlib::tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block =
    BN<dlib::con<N, 3, 3, 1, 1,
                 dlib::relu<BN<dlib::con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET>
using ares = dlib::relu<residual<block, N, dlib::affine, SUBNET>>;
template <int N, typename SUBNET>
using ares_down = dlib::relu<residual_down<block, N, dlib::affine, SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256, SUBNET>;
template <typename SUBNET>
using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template <typename SUBNET>
using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template <typename SUBNET>
using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

using anet_type = dlib::loss_metric<dlib::fc_no_bias<
    128,
    dlib::avg_pool_everything<
        alevel0<alevel1<alevel2<alevel3<alevel4<dlib::max_pool<
            3, 3, 2, 2,
            dlib::relu<dlib::affine<dlib::con<
                32, 7, 7, 2, 2, dlib::input_rgb_image_sized<150>>>>>>>>>>>>>;

/**
 * @brief 算法逻辑层
 */
class AbstractClass {
public:
  ~AbstractClass() {}

  void TemplateMethod() {
    PrimitiveOperation1();
    PrimitiveOperation2();
  }
  virtual void PrimitiveOperation1() = 0;
  virtual void PrimitiveOperation2() = 0;
};

class ConcreteClass : public AbstractClass {
public:
  void PrimitiveOperation1() override {}
  void PrimitiveOperation2() override {}

private:
  void InitVideo() {
    rs2::config cfg;
    // 设置从设备管道获取的深度图和彩色图的配置对象
    // 配置彩色图像流：分辨率640*480，图像格式：BGR， 帧率：30帧/秒
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    rs2::pipeline pipe;    // 生成Realsense管道，用来封装实际的相机设备
    pipe.start(cfg);// 根据给定的配置启动相机管道
    rs2::frameset data;
    data = pipe.wait_for_frames();
    rs2::frame color = data.get_color_frame();  // 获取彩色图像数据
    if(!color)
        std::cout<<"no color image"<<std::endl;
  }

  void LoadFacesData() {
    // 使用数据库代理加载人脸图片文件
    _subject = new RealSubject();
    _proxy = new Proxy(_subject);
    std::map<std::string, std::string> faceImgFiles;
    _proxy->GetFile(this->faceImgFiles, faceImgFiles);
    if (faceImgFiles.empty()) {
      std::cout << "No face files found in " << this->faceImgFiles << std::endl;
    }

    // 人脸描述符库, face_descriptor ---> name
    for (auto it = faceImgFiles.begin(); it != faceImgFiles.end(); ++it) {
      std::cout << "face image name:" << it->second << " file path:" << it->first
                << std::endl;

      cv::Mat faceImg = cv::imread(it->first); // 获取一张彩色人脸图片
      cv::cvtColor(faceImg, faceImg, cv::COLOR_BGR2GRAY); // 彩色人脸图片转灰度图
      dlib::array2d<dlib::bgr_pixel> dlibFaceImg;
      dlib::assign_image(dlibFaceImg, dlib::cv_image<unsigned char>(faceImg)); // 人脸灰度图转dlib图

      // haar级联分类器探测人脸区域，获取一系列人脸所在区域
      std::vector<cv::Rect> faceImgRects; // 构造对象，多个人脸区域
      cv::CascadeClassifier faceImgDetector(cvModel); // 使用cv级联分类器（cv级联探测器）
      if (faceImgDetector.empty()) {
        std::cout << "face detector is empty!" << std::endl;
      }
      faceImgDetector.detectMultiScale(faceImg, faceImgRects); // 传入一张faceImg并获取所有人脸区域
      std::vector<dlib::rectangle> dlibRects; // 使用dlib的人脸区域
      for (int i = 0; i < faceImgRects.size(); ++i) { // 遍历所有人脸图片区域并存入dlibRects
        // cv::rectangle(frame, objects[i], CV_RGB(200,0,0));
        dlib::rectangle faceImgRect(faceImgRects[i].x, faceImgRects[i].y,
                          faceImgRects[i].x + faceImgRects[i].width,
                          faceImgRects[i].y + faceImgRects[i].height);
        dlibRects.push_back(faceImgRect); //正常情况下应该只检测到一副面容
      }

      if (dlibRects.size() == 0)
        continue;

      std::vector<dlib::matrix<dlib::rgb_pixel>> facesFeaturePoints; // 多个人脸特征点
      std::vector<dlib::full_object_detection> dlibFaceImgRects; // 多个人脸图片区域
      dlib::shape_predictor sp; // 加载dlib人脸形状探测器
      dlib::deserialize(dlibModel) >> sp;
      for (int i = 0; i < dlibRects.size(); i++) {
        dlib::full_object_detection dlibFaceImgRect =
            sp(dlibFaceImg, dlibRects[i]); // 获取指定一个区域的人脸形状
        dlibFaceImgRects.push_back(dlibFaceImgRect); // 将人脸图片区域放入dlibFaceImgRects

        dlib::matrix<dlib::rgb_pixel> faceFeaturePoints; // 人脸特征点
        dlib::extract_image_chip(
            dlibFaceImg, dlib::get_face_chip_details(dlibFaceImgRect, 150, 0.25), faceFeaturePoints);
        facesFeaturePoints.push_back(std::move(faceFeaturePoints));
      }

      if (facesFeaturePoints.size() == 0) { // 传递到下面使用
        std::cout << "No faces found in " << it->second << std::endl;
        continue;
      }

      anet_type net; // 加载负责人脸识别的DNN算法模型
      dlib::deserialize(dnnModel) >> net;
      std::vector<dlib::matrix<float, 0, 1>> facesDescriptor = net(facesFeaturePoints); // 多个人脸描述

      std::map<dlib::matrix<float, 0, 1>, std::string> fdlib; //???
      for (auto iter = facesDescriptor.begin(); iter != facesDescriptor.end();
           iter++) {
        fdlib.insert(std::pair<dlib::matrix<float, 0, 1>, std::string>(
            *iter, it->second));
      }
    }
  }

  void FaceRecognizer() {
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();// dlib的人脸探测器，用于非人脸比对
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
      std::cout << "image empty, change image path." << std::endl;
    }
    cv::cvtColor(image, image, CV_BGR2GRAY); // 提取灰度图

    //加载dlib的人脸识别器
    dlib::array<dlib::array2d<unsigned char>> images;
    std::vector<std::vector<dlib::full_object_detection>> objects;
    dlib::shape_predictor_trainer trainer;
    trainer.set_tree_depth(2);
    trainer.set_nu(0.05);
    trainer.be_verbose();
    dlib::shape_predictor sp =
        trainer.train(images, objects); //加载人脸形状探测器
    // It should have been able to perfectly fit the data
    // While we are here, make sure the default face detector works
    std::vector<dlib::rectangle> dlibRects = detector(images[0]);
  }

  void FaceDectector() {
    //加载人脸形状探测器
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    dlib::shape_predictor sp;
    dlib::deserialize(dlibModel) >> sp;
    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    printf("CLOCK_MONOTONIC1 reports %ld.%09ld seconds\n",
           ts_end.tv_sec - ts_start.tv_sec, ts_end.tv_nsec - ts_start.tv_nsec);
  }

  void Mat2DlibMat() {
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
      std::cout << "image empty, change image path." << std::endl;
    }
    //提取灰度图
    cv::cvtColor(image, image, CV_BGR2GRAY);
    // Mat转化为dlib的matrix 用于非人脸比对
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    dlib::array2d<dlib::bgr_pixel> dlibImg;
    dlib::assign_image(dlibImg, dlib::cv_image<unsigned char>(image));
    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    printf("CLOCK_MONOTONIC2 reports %ld.%09ld seconds\n",
           ts_end.tv_sec - ts_start.tv_sec, ts_end.tv_nsec - ts_start.tv_nsec);
  }

  void GetFaceRectangle() {
    // 获取一系列人脸所在区域
    // Dlib图片格式与OpenCV还是有一定区别的，dlib是以dlib::array2d的形式存在，而oepncv是以cv::Mat的形式存在，关于opencv图像之间的转换
    dlib::array2d<dlib::bgr_pixel> dlibImg; // dlib类型人脸图片，将cv mat 转成dlib mat，//bgr彩色图
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector(); // 检测器，用于人脸检测画框，返回默认的人脸检测器
    std::vector<dlib::rectangle> faceImgRects = detector(dlibImg); // 对图像画人脸框，输入图片，返回矩形框的4个坐标
    std::cout << "Number of faces detected: " << faceImgRects.size() << std::endl;
    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    printf("CLOCK_MONOTONIC3 reports %ld.%09ld seconds\n",
           ts_end.tv_sec - ts_start.tv_sec, ts_end.tv_nsec - ts_start.tv_nsec);
    if (faceImgRects.size() == 0) {
    }
    // return 0;
  }

  void GetFaceRectangle2() {
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    cv::CascadeClassifier faceDetector(cvModel); // haar级联分类器探测人脸区域，获取一系列人脸所在区域
    if (faceDetector.empty()) {
      std::cout << "face detector is empty!" << std::endl;
      // return 0;
    }
    std::vector<cv::Rect> objects; // x, y, w, h 的 vector
    std::vector<dlib::rectangle> dets; // left: int, top: int, right: int, bottom: int
    cv::Mat image = cv::imread(imagePath); // cv 单个人脸图片数据
    faceDetector.detectMultiScale(image, objects); // 人脸 -> 人脸坐标
    for (int i = 0; i < objects.size(); i++) { // 人脸坐标遍历
      cv::rectangle(image, objects[i], CV_RGB(200, 0, 0)); // 绘制矩形轮廓或者填充矩形，其两个相对嘅角为pt1同pt2
      dlib::rectangle dlibRects(objects[i].x, objects[i].y,
                        objects[i].x + objects[i].width,
                        objects[i].y + objects[i].height);
      dets.push_back(dlibRects);
    }
    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    printf("CLOCK_MONOTONIC3 reports %ld.%09ld seconds\n",
           ts_end.tv_sec - ts_start.tv_sec, ts_end.tv_nsec - ts_start.tv_nsec);
    //    if (dets.size() == 0)
    //        return 0;
  }

  void GetFaceRectangle3() {
    // haar级联分类器探测人脸区域，获取一系列人脸所在区域
    // cv::Mat image = cv::imread(argv[2]);
    cv::Mat cvImage = cv::imread(imagePath);
    // cv::Mat src;
    cv::cvtColor(cvImage, cvImage, CV_BGR2GRAY);
    dlib::array2d<dlib::bgr_pixel> dlibImg;
    dlib::assign_image(dlibImg, dlib::cv_image<uchar>(cvImage));
    std::vector<cv::Rect> cvRects;
    std::vector<dlib::rectangle> dlibRects;
    cv::CascadeClassifier faceDetector(cvModel);
    faceDetector.detectMultiScale(cvImage, cvRects);
    for (int i = 0; i < cvRects.size(); i++) {
      cv::rectangle(cvImage, cvRects[i], CV_RGB(200, 0, 0));
      dlib::rectangle r(cvRects[i].x, cvRects[i].y,
                        cvRects[i].x + cvRects[i].width,
                        cvRects[i].y + cvRects[i].height);
      dlibRects.push_back(r); //正常情况下应该只检测到一副面容
    }

    if (dlibRects.size() == 0) {
      // std::cout << "there is no faces found in " << argv[2] <<std::endl;
      // return -1;
    }

    std::vector<dlib::matrix<dlib::rgb_pixel>> faces; //三通道的matrix 这是一个表示RGB彩色图形像素的简单结构。char r g b
    std::vector<dlib::full_object_detection> shapes; //获取人脸68个特征点部位分布
    dlib::shape_predictor sp;
    for (int i = 0; i < dlibRects.size(); i++) {
      dlib::full_object_detection shape =
          sp(dlibImg, dlibRects[i]); //获取指定一个区域的人脸形状
      std::cout << "number of parts: "<< shape.num_parts() << std::endl;
      std::cout << "pixel position of first part:  " << shape.part(0) << std::endl;
      std::cout << "pixel position of second part: " << shape.part(1) << std::endl;
      shapes.push_back(shape);

      dlib::matrix<dlib::rgb_pixel> face_chip;//face_chips 这个对象，就是存一张图像中所有对齐后的人脸;
      // dlib::get_face_chip_details(shape, 150, 0.25)
      // 输入参数：shape为提取的人脸关键点，68个点,
      // 160为希望获取的对齐后的人脸大小,0.1 是希望对人脸关键点区域进行padding的比例
      // 人脸数越多，这个函数用时越多..
      // 截取人脸部分，并将大小调为150*150/截取人脸部分，并将大小调为150*150
      dlib::extract_image_chip(
          dlibImg, dlib::get_face_chip_details(shape, 150, 0.25), face_chip);//dlib 人脸对齐 基本原理

      faces.push_back(std::move(face_chip));
    }
    if (faces.size() == 0) {
      // std::cout << "No faces found in " << argv[2] <<std::endl;
      // return -1;
    }
  }

  void LoadVideo() {
    //加载视频
    cv::VideoCapture capture(imagePath);
    int frames = capture.get(
        cv::CAP_PROP_FRAME_COUNT); //获取视频针数目(一帧就是一张图片)
    double fps = capture.get(cv::CAP_PROP_FPS); //获取每针视频的频率
    // 获取帧的视频宽度，视频高度
    cv::Size size = cv::Size(capture.get(cv::CAP_PROP_FRAME_WIDTH),
                             capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    std::cout << frames << std::endl;
    std::cout << fps << std::endl;
    std::cout << size << std::endl;
    //创建写入对象,需要指定，帧率和视频宽高
    cv::VideoWriter writer;
    //指定保存文件位置，编码器，帧率，宽高
    writer.open("test.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps,
                size);
    // VideoWriter writer("VideoTest.avi",CV_FOURCC('M', 'J', 'P',
    // 'G'), 20.0,Size(480, 848));
    dlib::shape_predictor sp;
    while (true) {
      //加载待检测的图片
      cv::Mat frame;
      capture >> frame;
      if (frame.empty())
        break;

      cv::Mat src;
      cv::cvtColor(frame, src, cv::COLOR_BGR2GRAY);
      dlib::array2d<dlib::bgr_pixel> dimg;
      dlib::assign_image(dimg, dlib::cv_image<uchar>(src));

      // haar级联分类器探测人脸区域，获取一系列人脸所在区域
      std::vector<cv::Rect> objects;
      std::vector<dlib::rectangle> dets;
      cv::CascadeClassifier faceDetector(cvModel);
      faceDetector.detectMultiScale(src, objects);
      for (int i = 0; i < objects.size(); i++) {
        cv::rectangle(frame, objects[i], CV_RGB(200, 0, 0));
        dlib::rectangle r(objects[i].x, objects[i].y,
                          objects[i].x + objects[i].width,
                          objects[i].y + objects[i].height);
        dets.push_back(r); //正常情况下应该只检测到一副面容
      }

      if (dets.size() == 0) {
        continue;
      }

      std::vector<dlib::matrix<dlib::rgb_pixel>> faces;
      std::vector<dlib::full_object_detection> shapes; //获取人脸68个特征点部位分布
      for (int i = 0; i < dets.size(); i++) {
        dlib::full_object_detection shape =
            sp(dimg, dets[i]); //获取指定一个区域的人脸形状
        shapes.push_back(shape);

        dlib::matrix<dlib::rgb_pixel> face_chip;
        dlib::extract_image_chip(
            dimg, dlib::get_face_chip_details(shape, 150, 0.25), face_chip);

        faces.push_back(std::move(face_chip));
      }
      if (faces.size() == 0) {
        continue;
      }
      LineOneFaceDetections(frame, shapes);

      anet_type net; // restnet
      dlib::deserialize(dnnModel) >> net;
      std::vector<dlib::matrix<float, 0, 1>> face_descriptors = net(faces);

      //遍历库，查找相似图像
      float min_distance = 0.7;
      std::string similar_name = "unknown";
      std::map<dlib::matrix<float, 0, 1>, std::string> fdlib;
      for (std::map<dlib::matrix<float, 0, 1>, std::string>::iterator it =
               fdlib.begin();
           it != fdlib.end(); it++) {
        float distance = length(it->first - face_descriptors[0]);
        if (distance < 0.5) //应该计算一个最近值
        {
          if (distance <= min_distance) {
            min_distance = distance;
            similar_name = it->second;
          }
        }
      }

      if (min_distance < 0.5) {
        float similarity = (0.5 - min_distance) * 100 / 0.5;
        std::stringstream strStream;
        strStream << similar_name << ", " << similarity << '%' << std::endl;
        std::string s = strStream.str();
        cv::Point org(objects[0].x, objects[0].y);
        cv::putText(frame, s, org, cv::FONT_HERSHEY_SIMPLEX, 1.0,
                    CV_RGB(0, 200, 0));
      }
    }
  }

  void GetFeaturePoints() {
    //获取人脸特征点分布
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    std::vector<dlib::full_object_detection> shapes;
    std::vector<dlib::rectangle> dlibRects;
    dlib::array2d<dlib::bgr_pixel> dlibImg;
    dlib::shape_predictor sp;
    dlib::deserialize(dlibModel) >> sp;
    cv::Mat image = cv::imread(imagePath);
    for (int i = 0; i < dlibRects.size(); i++) {
      dlib::full_object_detection shape =
          sp(dlibImg, dlibRects[i]); //获取指定一个区域的人脸形状
      shapes.push_back(shape);
    }
    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    printf("CLOCK_MONOTONIC4 reports %ld.%09ld seconds\n",
           ts_end.tv_sec - ts_start.tv_sec, ts_end.tv_nsec - ts_start.tv_nsec);

    //指出每个检测到的人脸的位置
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    for (int i = 0; i < dlibRects.size(); i++) {
      //画出人脸所在区域
      cv::Rect r;
      r.x = dlibRects[i].left();
      r.y = dlibRects[i].top();
      r.width = dlibRects[i].width();
      r.height = dlibRects[i].height();
      cv::rectangle(image, r, cv::Scalar(0, 0, 255), 1, 1, 0);
    }
    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    printf("CLOCK_MONOTONIC5 reports %ld.%09ld seconds\n",
           ts_end.tv_sec - ts_start.tv_sec, ts_end.tv_nsec - ts_start.tv_nsec);

    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    LineOneFaceDetections(image, shapes);
    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    printf("CLOCK_MONOTONIC6 reports %ld.%09ld seconds\n",
           ts_end.tv_sec - ts_start.tv_sec, ts_end.tv_nsec - ts_start.tv_nsec);

    // dlib end
  }

  void FindMatchFace() {
    cv::Mat image = cv::imread(imagePath);
    std::vector<dlib::matrix<dlib::rgb_pixel>> faces;
    std::vector<cv::Rect> objects;
    anet_type net;
    dlib::deserialize(dnnModel) >> net;
    std::vector<dlib::matrix<float, 0, 1>> face_descriptors = net(faces);

    //遍历库，查找相似图像
    std::map<dlib::matrix<float, 0, 1>, std::string> fdlib;
    for (auto it =fdlib.begin(); it != fdlib.end(); ++it) {
      float distance = length(it->first - face_descriptors[0]);
      if (distance < 0.6) {
        std::cout << "the pic is " << it->second << "!, distance:" << distance
                  << std::endl;

        cv::Point org(objects[0].x, objects[0].y);
        cv::putText(image, it->second, org, cv::FONT_HERSHEY_SIMPLEX, 1.0,
                    CV_RGB(0, 200, 0));
        break;
      }
    }
  }

  void EuclideanDistance() {
    //遍历库，查找相似图像
    float min_distance = 0.7;
    std::string similar_name = "unknown";
    std::map<dlib::matrix<float, 0, 1>, std::string> fdlib; // 定义一个向量组，用于存放每一个人脸的编码；
    cv::Mat frame;
    anet_type net; 	// 终于我们加载Resnet模型进行人脸识别
    dlib::deserialize(dnnModel) >> net;
    std::vector<dlib::matrix<dlib::rgb_pixel>> faces; // 定义dlib型图片，彩色
    std::vector<cv::Rect> cvRects;
    std::vector<dlib::matrix<float, 0, 1>> face_descriptors = net(faces); // 将150*150人脸图像载入Resnet残差网络，返回128D人脸特征存于face_descriptors
    for (auto it = fdlib.begin(); it != fdlib.end(); ++it) {
      float distance = length(it->first - face_descriptors[0]); 
      if (distance < 0.5) //应该计算一个最近值
      {
        if (distance <= min_distance) {
          min_distance = distance;
          similar_name = it->second;
        }
      }
    }

    if (min_distance < 0.5) {
      float similarity = (0.5 - min_distance) * 100 / 0.5;
      std::stringstream strStream;
      strStream << similar_name << ", " << similarity << '%' << std::endl;
      std::string s = strStream.str();
      cv::Point org(cvRects[0].x, cvRects[0].y);
      cv::putText(frame, s, org, cv::FONT_HERSHEY_SIMPLEX, 1.0,
                  CV_RGB(0, 200, 0));
    }
  }

  void LineOneFaceDetections(cv::Mat img,
                             std::vector<dlib::full_object_detection> fs) {
    int i, j;
    for (j = 0; j < fs.size(); j++) {
      cv::Point p1, p2;
      for (i = 0; i < 67; i++) {
        //下巴到脸颊 0 ~ 16
        //左边眉毛 17 ~ 21
        //右边眉毛 21 ~ 26
        //鼻梁     27 ~ 30
        //鼻孔        31 ~ 35
        //左眼        36 ~ 41
        //右眼        42 ~ 47
        //嘴唇外圈  48 ~ 59
        //嘴唇内圈  59 ~ 67
        switch (i) {
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
        p2.x = fs[j].part(i + 1).x();
        p2.y = fs[j].part(i + 1).y();
        cv::line(img, p1, p2, cv::Scalar(0, 0, 255), 2, 4, 0);
      }
    }
  }

  void Show(cv::Mat &image) {
    cv::imshow("image", image);
    // cv::imshow("depth",depthmat);
    cv::waitKey(1000000);
  }

private:
  const std::string dlibModel = "../dataset/data";
  const std::string cvModel = "../dataset/haarcascade_frontalface_alt2.xml";
  const std::string dnnModel = "../dataset/resnet_data";
  const std::string faceImgFiles = "../images/faces";
  const std::string imagePath = "../images/man.jpeg";
  Proxy *_proxy;
  Subject *_subject;

  struct timespec ts_start, ts_end;
};

int main() { return 0; }
