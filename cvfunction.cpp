#include "cvfunction.h"
using namespace cv;

CVFunction::CVFunction() {}

CVFunction::~CVFunction() {}

Mat CVFunction::objectSearch(Mat src, Mat ref, int METHOD)
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



















