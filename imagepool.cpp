#include "imagepool.h"

ImagePool::ImagePool()
{
    Mat src = newImage();
    Mat dst = newImage();
    Mat ref = newImage();
}

ImagePool::~ImagePool() {}

Mat ImagePool::newImage()
{
    Mat newImg;
    imageQue.push_back(newImg);
    return imageQue.back();
}
