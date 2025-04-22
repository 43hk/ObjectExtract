#ifndef CVFUNCTION_H
#define CVFUNCTION_H

#include "opencv2/opencv.hpp"
#include <QString>

using namespace cv;

class CVFunction
{
public:
    CVFunction();
    ~CVFunction();

    static Mat templateSearch(const Mat &src, const Mat &ref, Mat &dst, int METHOD);
    static Mat faceSearch(const Mat &src, Mat &dst);
};
#endif // CVFUNCTION_H
