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

    static Mat objectSearch(Mat src, Mat ref, int METHOD);
};
#endif // CVFUNCTION_H
