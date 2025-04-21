#ifndef IMAGEPOOL_H
#define IMAGEPOOL_H

#include "opencv2/opencv.hpp"
using namespace cv;

class ImagePool
{
public:
    ImagePool();
    ~ImagePool();
    Mat src, dst, ref;
    Mat newImage();
    int METHOD = TM_SQDIFF;

private:
    std::vector<Mat> imageQue;
};

#endif // IMAGEPOOL_H
