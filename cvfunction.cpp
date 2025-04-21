#include "cvfunction.h"
using namespace cv;

CVFunction::CVFunction() {}

CVFunction::~CVFunction() {}

Mat CVFunction::templateSearch(const Mat &src, const Mat &ref, int METHOD)
{
    Mat imgResult;      // 存储匹配结果
    Mat imgDisplay;     // 用于显示最终结果
    src.copyTo(imgDisplay); // 将源图像复制到 imgDisplay

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
    if (METHOD == TM_SQDIFF || METHOD == TM_SQDIFF_NORMED)
        matchLoc = minLoc; // 平方差匹配法取最小值
    else
        matchLoc = maxLoc; // 其他方法取最大值

    // 在源图像上绘制矩形框
    rectangle(imgDisplay, matchLoc, Point(matchLoc.x + ref.cols, matchLoc.y + ref.rows), Scalar(0, 255, 0), 2); // 绿色矩形框

    // 返回带有标注的图像
    return imgDisplay;
}

Mat CVFunction::faceSearch(const Mat &src)
{
    Mat imgDisplay = src.clone();

    // 初始化人脸和眼睛检测器
    CascadeClassifier face_detector;
    CascadeClassifier eyes_detector;


    // 检查分类器是否加载成功
    if (!face_detector.load("release/haarcascade_frontalface_alt.xml"))
    {
        std::cerr << "Error: Could not load face detector." << std::endl;
        return imgDisplay;
    }
    if (!eyes_detector.load("release/haarcascade_eye_tree_eyeglasses.xml"))
    {
        std::cerr << "Error: Could not load eyes detector." << std::endl;
        return imgDisplay;
    }

    // 转换为灰度图并进行直方图均衡化
    Mat imgGray;
    cvtColor(imgDisplay, imgGray, COLOR_BGR2GRAY);
    equalizeHist(imgGray, imgGray);

    // 检测人脸
    std::vector<Rect> faces;
    face_detector.detectMultiScale(imgGray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

    // 遍历检测到的人脸
    for (size_t i = 0; i < faces.size(); i++)
    {
        // 绘制椭圆标记人脸
        Point center(faces[i].x + faces[i].width * 0.5, faces[i].y + faces[i].height * 0.5);
        ellipse(imgDisplay, center, Size(faces[i].width * 0.5, faces[i].height * 0.5), 0, 0, 360, Scalar(255, 0, 255), 4);

        // 在人脸区域内检测眼睛
        Mat faceROI = imgGray(faces[i]);
        std::vector<Rect> eyes;
        eyes_detector.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

        // 遍历检测到的眼睛
        for (size_t j = 0; j < eyes.size(); j++)
        {
            // 计算眼睛中心点和半径
            Point eye_center(faces[i].x + eyes[j].x + eyes[j].width * 0.5,
                             faces[i].y + eyes[j].y + eyes[j].height * 0.5);
            int radius = cvRound((eyes[j].width + eyes[j].height) * 0.25);

            // 绘制圆形标记眼睛
            circle(imgDisplay, eye_center, radius, Scalar(255, 0, 0), 4);
        }
    }

    // 返回处理后的图像
    return imgDisplay;
}























