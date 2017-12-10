#ifndef OPENPOSE_UTILITIES_KEYPOINT_HPP
#define OPENPOSE_UTILITIES_KEYPOINT_HPP

#include <openpose/core/common.hpp>

namespace op
{
    OP_API float getDistance(const Array<float>& keypoints, const int person, const int elementA, const int elementB);

    OP_API void averageKeypoints(Array<float>& keypointsA, const Array<float>& keypointsB, const int personA);

    OP_API void scaleKeypoints(Array<float>& keypoints, const float scale);

    OP_API void scaleKeypoints(Array<float>& keypoints, const float scaleX, const float scaleY);

    OP_API void scaleKeypoints(Array<float>& keypoints, const float scaleX, const float scaleY, const float offsetX,
                               const float offsetY);

    OP_API void renderKeypointsCpu(Array<float>& frameArray, const Array<float>& keypoints,
                                   const std::vector<unsigned int>& pairs, const std::vector<float> colors,
                                   const float thicknessCircleRatio, const float thicknessLineRatioWRTCircle,
                                   const float threshold);
    OP_API void CopyTo(cv::Mat& src, cv::Mat& dst, cv::Mat& mask);

    OP_API void fillColor(cv::Mat& src, float fColor, cv::Mat& matMask);
    OP_API void fillColor(cv::Mat& src, float fColor0, cv::Mat& matMask0, float fThreshold);
    OP_API void fillColor(cv::Mat& src, float fColor0, cv::Mat& matMask0, float fColor1, cv::Mat matMask1);    
    
    OP_API void Beauty(cv::Mat& frame, double dcontrast, double nbrightness);

    OP_API Rectangle<float> getKeypointsRectangle(const Array<float>& keypoints, const int person,
                                                  const int numberKeypoints, const float threshold);

    OP_API float getAverageScore(const Array<float>& keypoints, const int person);

    OP_API float getKeypointsArea(const Array<float>& keypoints, const int person, const int numberKeypoints,
                                  const float threshold);

    OP_API int getBiggestPerson(const Array<float>& keypoints, const float threshold);
}

#endif // OPENPOSE_UTILITIES_KEYPOINT_HPP
