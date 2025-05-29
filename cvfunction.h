#ifndef CVFUNCTION_H
#define CVFUNCTION_H

#include "opencv2/opencv.hpp"
#include "opencv2/features2d.hpp"
#include <QString>

using namespace cv;

enum Method
{
    TM_SQDIFF,
    TM_SQDIFF_NORMED,
    TM_CCORR,
    TM_CCORR_NORMED,
    TM_CCOEFF,
    TM_CCOEFF_NORMED,
};

class CVFunction
{
public:
    CVFunction();
    ~CVFunction();

    static Mat templateSearch(const Mat &src, const Mat &ref, Mat &dst, Method METHOD);
    static void track(const Mat &ref);
    static Mat faceSearch(const Mat &src, Mat &dst);
    static Mat edgeDetection(const Mat& src, Mat& dst, int kernel_size);
    static Mat grabcutForegroundExtraction(const Mat& src, Mat& dst);
};
#endif // CVFUNCTION_H
