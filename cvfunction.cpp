#include "cvfunction.h"
using namespace cv;

CVFunction::CVFunction() {}

CVFunction::~CVFunction() {}

Mat CVFunction::templateSearch(const Mat &src, const Mat &ref, Mat &dst, Method METHOD)
{
    // 将源图像和参考图像转换为灰度图像
    Mat srcGray, refGray;
    cvtColor(src, srcGray, COLOR_RGB2GRAY);
    cvtColor(ref, refGray, COLOR_RGB2GRAY);

    // 创建匹配结果矩阵
    Mat imgResult;
    int resCols = srcGray.cols - refGray.cols + 1;
    int resRows = srcGray.rows - refGray.rows + 1;
    imgResult.create(resRows, resCols, CV_32FC1);

    // 执行模板匹配
    matchTemplate(srcGray, refGray, imgResult, METHOD);

    // 归一化匹配结果（可选）
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

    // 在源图像上绘制矩形框（注意：这里使用原始彩色图像进行绘制，而不是灰度图像）
    rectangle(dst, matchLoc, Point(matchLoc.x + ref.cols, matchLoc.y + ref.rows), Scalar(0, 255, 0), 2); // 绿色矩形框

    // 剪裁出匹配区域（从原始彩色图像中剪裁）
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
    cvtColor(src, imgGray, COLOR_RGB2GRAY);
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


Mat CVFunction::edgeDetection(const Mat& src, Mat& dst, int kernel_size)
{
    // 转换为灰度图
    Mat srcGray;
    cvtColor(src, srcGray, COLOR_RGB2GRAY);

    // 执行边缘检测（Sobel）
    Mat grad_x, grad_y;
    Sobel(srcGray, grad_x, CV_32F, 1, 0, kernel_size);
    Sobel(srcGray, grad_y, CV_32F, 0, 1, kernel_size);

    // 取绝对值并合并梯度
    Mat abs_grad_x, abs_grad_y, edges;
    convertScaleAbs(grad_x, abs_grad_x);
    convertScaleAbs(grad_y, abs_grad_y);
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, edges);

    // 使用Canny算法从Sobel结果中提取边缘（转化为二值图像）
    Mat canny_output;
    Canny(edges, canny_output, 50, 150); // 调整阈值以适应你的需求

    // 寻找轮廓
    std::vector<std::vector<Point>> contours;
    std::vector<Vec4i> hierarchy;
    findContours(canny_output, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    if (contours.empty()) {
        // 如果没有找到任何轮廓，则复制原始图像到dst或者采取其他措施
        src.copyTo(dst);
        return src;
    }

    // 寻找最大的轮廓（假设是我们感兴趣的区域）
    double maxArea = 0;
    int maxIdx = 0;
    for (int i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        if (area > maxArea) {
            maxArea = area;
            maxIdx = i;
        }
    }

    // 设置颜色和厚度
    Scalar color(0, 255, 0); // 绿色
    int thickness = 2;

    // 在dst图像上绘制最大轮廓
    drawContours(dst, contours, maxIdx, color, thickness);

    // 如果需要剪裁出感兴趣区域，可以在绘制轮廓后进行
    Rect boundingBox = boundingRect(contours[maxIdx]);
    Mat croppedImage = src(boundingBox);

    return croppedImage;
}

Mat CVFunction::grabcutForegroundExtraction(const Mat& src, Mat& dst)
{
    // 初始矩形区域：图像中心缩小80%
    int margin = 20;
    Rect rect(margin, margin, src.cols - 2 * margin, src.rows - 2 * margin);

    // 初始化 mask
    Mat mask(src.size(), CV_8UC1, Scalar(GC_BGD));
    mask(rect).setTo(Scalar(GC_PR_FGD));

    // 初始化模型
    Mat bgModel, fgModel;

    // 执行 GrabCut 分割
    grabCut(src, mask, rect, bgModel, fgModel, 5, GC_INIT_WITH_RECT);

    // 转换为前景掩码
    Mat foregroundMask = (mask == GC_FGD) | (mask == GC_PR_FGD);

    // 提取前景图像
    Mat foreground;
    src.copyTo(foreground, foregroundMask);

    // 查找最大轮廓（便于裁剪）
    std::vector<std::vector<Point>> contours;
    findContours(foregroundMask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    if (contours.empty()) {
        src.copyTo(dst);
        return src;
    }

    // 找最大轮廓
    int maxIdx = 0;
    double maxArea = 0;
    for (int i = 0; i < contours.size(); ++i) {
        double area = contourArea(contours[i]);
        if (area > maxArea) {
            maxArea = area;
            maxIdx = i;
        }
    }

    Rect bbox = boundingRect(contours[maxIdx]);
    Mat cropped = src(bbox).clone();

    // 可视化：绘制最大轮廓
    src.copyTo(dst);
    drawContours(dst, contours, maxIdx, Scalar(0, 255, 0), 2);

    return cropped;
}


























