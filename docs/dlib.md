- 项目概述篇（案例演示）

  - 完整项目运行演示

    - 人脸特征点（68,72,84,194）

  - 项目系统架构设计

    - ![image-20211228011810963](/home/victor/.config/Typora/typora-user-images/image-20211228011810963.png)
    - ![image-20211228012205665](/home/victor/.config/Typora/typora-user-images/image-20211228012205665.png)
    - ![image-20211228013711001](/home/victor/.config/Typora/typora-user-images/image-20211228013711001.png)
    - ![image-20211228013648421](/home/victor/.config/Typora/typora-user-images/image-20211228013648421.png)

  - 项目关键技术说明

    - ![image-20211228014018039](/home/victor/.config/Typora/typora-user-images/image-20211228014018039.png)
    - ![image-20211228014107365](/home/victor/.config/Typora/typora-user-images/image-20211228014107365.png)
    - ![image-20211228014136921](/home/victor/.config/Typora/typora-user-images/image-20211228014136921.png)
    - ![image-20211228014239779](/home/victor/.config/Typora/typora-user-images/image-20211228014239779.png)
    - ![image-20211228014311130](/home/victor/.config/Typora/typora-user-images/image-20211228014311130.png)

  - 项目业务需求分析

    - ![image-20211228015003976](/home/victor/.config/Typora/typora-user-images/image-20211228015003976.png)
      - ![image-20211228015900748](/home/victor/.config/Typora/typora-user-images/image-20211228015900748.png)
      - 人脸对齐：人脸特征点的标注（标注出特征点）->位置的矫正->特征向量的比对（特征点信息映射到128维向量，再判定欧式距离，如何在阈值之内就是同一个人）->活体检测（常见就是眨眼张嘴点头）![image-20211228015823926](/home/victor/.config/Typora/typora-user-images/image-20211228015823926.png)
      - ![image-20211228020116184](/home/victor/.config/Typora/typora-user-images/image-20211228020116184.png)
      - ![image-20211228020201129](/home/victor/.config/Typora/typora-user-images/image-20211228020201129.png)

  - 项目业务流程分析

    - 样本标注->模型训练->模型应用

      ![image-20211228021752399](/home/victor/.config/Typora/typora-user-images/image-20211228021752399.png)

    - 5.2对应如下5.3

      ![image-20211228021917457](/home/victor/.config/Typora/typora-user-images/image-20211228021917457.png)

    - 5.3对应的核心过程（看图对应其中内容）![image-20211228021949720](/home/victor/.config/Typora/typora-user-images/image-20211228021949720.png)

- 环境部署篇（技术框架）

  - 项目开发环境概述
    - ![image-20211228024834199](/home/victor/.config/Typora/typora-user-images/image-20211228024834199.png)
    - a
    - 
  - Dlib框架源码编译
    - 包含分类、集群、回归![image-20211228025240719](/home/victor/.config/Typora/typora-user-images/image-20211228025240719.png)
  - 项目工程文件创建
  - 项目开发环境配置之Cuda
  - 项目开发环境配置之OpenCV
  - 项目开发环境配置之Dlib
  - 项目性能优化配置

- 程序设计篇（关键技术）

  - 实时视频采集程序设计
  - 实时图像抓拍程序设计
  - 实时人脸检测程序设计
  - 实时人脸特征点标定程序设计
  - 实时人脸特征点对齐程序设计
  - 实时目标跟踪程序设计
  - 实时人脸对比程序设计
    - ![image-20211228055746634](/home/victor/.config/Typora/typora-user-images/image-20211228055746634.png)
  - 实时活体检测之眨眼识别
    - ![image-20211228055914337](/home/victor/.config/Typora/typora-user-images/image-20211228055914337.png)
    - ![image-20211228060102834](/home/victor/.config/Typora/typora-user-images/image-20211228060102834.png)
    - ![image-20211228060404954](/home/victor/.config/Typora/typora-user-images/image-20211228060404954.png)
    - 设置阈值，就是发生眨眼行为了![image-20211228060335335](/home/victor/.config/Typora/typora-user-images/image-20211228060335335.png)
    - ![image-20211228060705729](/home/victor/.config/Typora/typora-user-images/image-20211228060705729.png)
    - 
  - 实时活体检测之张嘴识别
    - ![image-20211228060801302](/home/victor/.config/Typora/typora-user-images/image-20211228060801302.png)
    - ![image-20211228060950899](/home/victor/.config/Typora/typora-user-images/image-20211228060950899.png)
    - ![image-20211228061005137](/home/victor/.config/Typora/typora-user-images/image-20211228061005137.png)
    - ![image-20211228061037573](/home/victor/.config/Typora/typora-user-images/image-20211228061037573.png)
  - 人脸聚类程序设计

- 模型训练篇（模型训练）

  - 人脸区域检测样本标注（imagelab源码编译和人脸检测标注方法）
    - ![image-20211228061416344](/home/victor/.config/Typora/typora-user-images/image-20211228061416344.png)
    - ![image-20211228061448558](/home/victor/.config/Typora/typora-user-images/image-20211228061448558.png)
    - ![image-20211228061811463](/home/victor/.config/Typora/typora-user-images/image-20211228061811463.png)
    - ![image-20211228061848916](/home/victor/.config/Typora/typora-user-images/image-20211228061848916.png)
    - 
  - 人脸区域检测模型训练
    - ![image-20211228062343905](/home/victor/.config/Typora/typora-user-images/image-20211228062343905.png)
    - ![image-20211228062519593](/home/victor/.config/Typora/typora-user-images/image-20211228062519593.png)
    - ![image-20211228062531949](/home/victor/.config/Typora/typora-user-images/image-20211228062531949.png)
    - a
    - 
  - 人脸区域检测模型测试
    - ![image-20211228062931699](/home/victor/.config/Typora/typora-user-images/image-20211228062931699.png)
    - ![image-20211228063034074](/home/victor/.config/Typora/typora-user-images/image-20211228063034074.png)
    - ![image-20211228063226241](/home/victor/.config/Typora/typora-user-images/image-20211228063226241.png)
    - a
  - 人脸特征点标定样本标注
    - ![image-20211228063421761](/home/victor/.config/Typora/typora-user-images/image-20211228063421761.png)
    - ![image-20211228063435752](/home/victor/.config/Typora/typora-user-images/image-20211228063435752.png)
    - ![image-20211228063449829](/home/victor/.config/Typora/typora-user-images/image-20211228063449829.png)
    - ![image-20211228063504413](/home/victor/.config/Typora/typora-user-images/image-20211228063504413.png)
    - ![image-20211228063527817](/home/victor/.config/Typora/typora-user-images/image-20211228063527817.png)
    - ![image-20211228063634374](/home/victor/.config/Typora/typora-user-images/image-20211228063634374.png)
    - ![image-20211228063657944](/home/victor/.config/Typora/typora-user-images/image-20211228063657944.png)
    - ![image-20211228063944854](/home/victor/.config/Typora/typora-user-images/image-20211228063944854.png)
    - a
    - 
    - 
  - 人脸特征点标定模型训练
    - ![image-20211228064305349](/home/victor/.config/Typora/typora-user-images/image-20211228064305349.png)
    - ![image-20211228064319706](/home/victor/.config/Typora/typora-user-images/image-20211228064319706.png)
    - ![image-20211228064342126](/home/victor/.config/Typora/typora-user-images/image-20211228064342126.png)
    - ![image-20211228065007501](/home/victor/.config/Typora/typora-user-images/image-20211228065007501.png)
    - ![image-20211228065403526](/home/victor/.config/Typora/typora-user-images/image-20211228065403526.png)
    - ![image-20211228065110878](/home/victor/.config/Typora/typora-user-images/image-20211228065110878.png)
    - ![image-20211228065255135](/home/victor/.config/Typora/typora-user-images/image-20211228065255135.png)
  - 人脸特征点标定模型测试
    - ![image-20211228065513290](/home/victor/.config/Typora/typora-user-images/image-20211228065513290.png)
    - ![image-20211228065546566](/home/victor/.config/Typora/typora-user-images/image-20211228065546566.png)
    - ![image-20211228065557663](/home/victor/.config/Typora/typora-user-images/image-20211228065557663.png)
    - ![image-20211228065623442](/home/victor/.config/Typora/typora-user-images/image-20211228065623442.png)
    - ![image-20211228065730753](/home/victor/.config/Typora/typora-user-images/image-20211228065730753.png)
    - 偏了，需要优化训练样本![image-20211228065740611](/home/victor/.config/Typora/typora-user-images/image-20211228065740611.png)
    - ![image-20211228065918659](/home/victor/.config/Typora/typora-user-images/image-20211228065918659.png)
    - ![image-20211228070003867](/home/victor/.config/Typora/typora-user-images/image-20211228070003867.png)
    - 往往训练样本不够，跟源码无关![image-20211228070049645](/home/victor/.config/Typora/typora-user-images/image-20211228070049645.png)
    - 

- 人脸注册