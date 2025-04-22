#include "cvfunction.h"
using namespace cv;

CVFunction::CVFunction() {}

CVFunction::~CVFunction() {}

Mat CVFunction::templateSearch(const Mat &src, const Mat &ref, Mat &dst, Method METHOD)
{
    Mat imgResult;      // 存储匹配结果

    // 创建匹配结果矩阵
    int resCols = src.cols - ref.cols + 1;
    int resRows = src.rows - ref.rows + 1;
    imgResult.create(resRows, resCols, CV_32FC1);

    // 执行模板匹配
    matchTemplate(src, ref, imgResult, METHOD);

    // 归一化匹配结果
    normalize(imgResult, imgResult, 0, 1, NORM_MINMAX, -1, Mat());

    // 找到最佳匹配位置
    double minVal, maxVal;
    Point minLoc, maxLoc, matchLoc;
    minMaxLoc(imgResult, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

    // 根据匹配方法选择最佳匹配位置
    if (METHOD == Method::TM_SQDIFF || METHOD == Method::TM_SQDIFF_NORMED)
        matchLoc = minLoc; // 平方差匹配法取最小值
    else
        matchLoc = maxLoc; // 其他方法取最大值

    // 在源图像上绘制矩形框
    rectangle(dst, matchLoc, Point(matchLoc.x + ref.cols, matchLoc.y + ref.rows), Scalar(0, 255, 0), 2); // 绿色矩形框

    // 剪裁出匹配区域
    Rect matchedRegion(matchLoc, Size(ref.cols, ref.rows)); // 定义剪裁区域
    Mat croppedRegion = src(matchedRegion);                // 从源图像中剪裁出区域

    // 返回剪裁出的区域
    return croppedRegion;
}

void CVFunction::track(const Mat &ref)
{
    // 初始化摄像头
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open camera" << std::endl;
        return;
    }

    // ORB 特征检测器和描述符
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    cv::BFMatcher bf(cv::NORM_HAMMING, true);

    // 计算参考图像的关键点和描述符
    std::vector<cv::KeyPoint> kp_ref;
    Mat des_ref;
    orb->detectAndCompute(ref, cv::Mat(), kp_ref, des_ref);

    while (true)
    {
        Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        // 检测当前帧的关键点和描述符
        std::vector<cv::KeyPoint> kp_frame;
        Mat des_frame;
        orb->detectAndCompute(frame, cv::Mat(), kp_frame, des_frame);

        if (!des_frame.empty() && !des_ref.empty()) {
            // 匹配特征点
            std::vector<cv::DMatch> matches;
            bf.match(des_ref, des_frame, matches);
            std::sort(matches.begin(), matches.end(), [](const cv::DMatch& a, const cv::DMatch& b) {
                    return a.distance < b.distance;
            });

            // 提取匹配点的位置
            std::vector<cv::Point2f> src_pts, dst_pts;
            for (size_t i = 0; i < matches.size(); ++i)
            {
                src_pts.push_back(kp_ref[matches[i].queryIdx].pt);
                dst_pts.push_back(kp_frame[matches[i].trainIdx].pt);
            }

            if (src_pts.size() > 4)
            {
                // 计算单应性矩阵并绘制边界框
                Mat H = cv::findHomography(src_pts, dst_pts, cv::RANSAC, 5.0);
                std::vector<cv::Point2f> pts = {cv::Point2f(0, 0), cv::Point2f(0, ref.rows - 1),
                                                    cv::Point2f(ref.cols - 1, ref.rows - 1), cv::Point2f(ref.cols - 1, 0)};
                std::vector<cv::Point2f> dst;
                cv::perspectiveTransform(pts, dst, H);
                std::vector<cv::Point> dst_int;
                for (const auto& pt : dst) dst_int.emplace_back(static_cast<int>(pt.x), static_cast<int>(pt.y));
                cv::polylines(frame, dst_int, true, cv::Scalar(0, 255, 0), 3);
            }
        }

        // 显示结果
        imshow("ORB Tracking", frame);
        if (cv::waitKey(30) >= 0) break;
    }


    // 释放资源
    cap.release();
    destroyAllWindows();
}


Mat CVFunction::faceSearch(const Mat &src, Mat &dst)
{
    Mat imgCut = src.clone(); // 用于显示最终结果

    // 初始化人脸和眼睛检测器
    CascadeClassifier face_detector;
    CascadeClassifier eyes_detector;

    // 检查分类器是否加载成功
    if (!face_detector.load("release/haarcascade_frontalface_alt.xml"))
    {
        std::cerr << "Error: Could not load face detector." << std::endl;
        return imgCut; // 返回原始图像
    }
    if (!eyes_detector.load("release/haarcascade_eye_tree_eyeglasses.xml"))
    {
        std::cerr << "Error: Could not load eyes detector." << std::endl;
        return imgCut; // 返回原始图像
    }

    // 转换为灰度图并进行直方图均衡化
    Mat imgGray;
    cvtColor(src, imgGray, COLOR_BGR2GRAY);
    equalizeHist(imgGray, imgGray);

    // 检测人脸
    std::vector<Rect> faces;
    face_detector.detectMultiScale(imgGray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

    // 如果未检测到人脸，直接返回原始图像
    if (faces.empty())
    {
        std::cerr << "No faces detected." << std::endl;
        return imgCut;
    }

    // 剪裁出第一张人脸区域
    Rect faceRegion = faces[0]; // 取第一个检测到的人脸
    Mat croppedFace = src(faceRegion); // 剪裁出人脸区域

    // 在 dst 图像上绘制标记（使用矩形框）
    for (size_t i = 0; i < faces.size(); i++)
    {
        // 绘制矩形框标记人脸
        rectangle(dst, faces[i], Scalar(0, 255, 0), 2); // 绿色矩形框

        // 在人脸区域内检测眼睛
        Mat faceROI = imgGray(faces[i]);
        std::vector<Rect> eyes;
        eyes_detector.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

        // 遍历检测到的眼睛
        for (size_t j = 0; j < eyes.size(); j++)
        {
            // 计算眼睛的矩形框位置
            Rect eyeRect(faces[i].x + eyes[j].x, faces[i].y + eyes[j].y, eyes[j].width, eyes[j].height);

            // 绘制矩形框标记眼睛
            rectangle(dst, eyeRect, Scalar(255, 0, 0), 2); // 蓝色矩形框
        }
    }

    // 返回剪裁出的人脸区域
    return croppedFace;
}























