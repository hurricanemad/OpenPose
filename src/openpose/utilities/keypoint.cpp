#include <limits> // std::numeric_limits
#include <opencv2/imgproc/imgproc.hpp> // cv::line, cv::circle
#include <opencv2/highgui.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/utilities/keypoint.hpp>

namespace op
{
    const std::string errorMessage = "The Array<float> is not a RGB image. This function is only for array of"
                                     " dimension: [sizeA x sizeB x 3].";

    float getDistance(const Array<float>& keypoints, const int person, const int elementA, const int elementB)
    {
        try
        {
            const auto keypointPtr = keypoints.getConstPtr() + person * keypoints.getSize(1) * keypoints.getSize(2);
            const auto pixelX = keypointPtr[elementA*3] - keypointPtr[elementB*3];
            const auto pixelY = keypointPtr[elementA*3+1] - keypointPtr[elementB*3+1];
            return std::sqrt(pixelX*pixelX+pixelY*pixelY);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return -1.f;
        }
    }

    void averageKeypoints(Array<float>& keypointsA, const Array<float>& keypointsB, const int personA)
    {
        try
        {
            // Security checks
            if (keypointsA.getNumberDimensions() != keypointsB.getNumberDimensions())
                error("keypointsA.getNumberDimensions() != keypointsB.getNumberDimensions().",
                      __LINE__, __FUNCTION__, __FILE__);
            for (auto dimension = 1u ; dimension < keypointsA.getNumberDimensions() ; dimension++)
                if (keypointsA.getSize(dimension) != keypointsB.getSize(dimension))
                    error("keypointsA.getSize() != keypointsB.getSize().", __LINE__, __FUNCTION__, __FILE__);
            // For each body part
            const auto numberParts = keypointsA.getSize(1);
            for (auto part = 0 ; part < numberParts ; part++)
            {
                const auto finalIndexA = keypointsA.getSize(2)*(personA*numberParts + part);
                const auto finalIndexB = keypointsA.getSize(2)*part;
                if (keypointsB[finalIndexB+2] - keypointsA[finalIndexA+2] > 0.05f)
                {
                    keypointsA[finalIndexA] = keypointsB[finalIndexB];
                    keypointsA[finalIndexA+1] = keypointsB[finalIndexB+1];
                    keypointsA[finalIndexA+2] = keypointsB[finalIndexB+2];
                }
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void scaleKeypoints(Array<float>& keypoints, const float scale)
    {
        try
        {
            scaleKeypoints(keypoints, scale, scale);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void scaleKeypoints(Array<float>& keypoints, const float scaleX, const float scaleY)
    {
        try
        {
            if (scaleX != 1. && scaleY != 1.)
            {
                // Error check
                if (!keypoints.empty() && keypoints.getSize(2) != 3)
                    error(errorMessage, __LINE__, __FUNCTION__, __FILE__);
                // Get #people and #parts
                const auto numberPeople = keypoints.getSize(0);
                const auto numberParts = keypoints.getSize(1);
                // For each person
                for (auto person = 0 ; person < numberPeople ; person++)
                {
                    // For each body part
                    for (auto part = 0 ; part < numberParts ; part++)
                    {
                        const auto finalIndex = 3*(person*numberParts + part);
                        keypoints[finalIndex] *= scaleX;
                        keypoints[finalIndex+1] *= scaleY;
                    }
                }
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void scaleKeypoints(Array<float>& keypoints, const float scaleX, const float scaleY, const float offsetX,
                        const float offsetY)
    {
        try
        {
            if (scaleX != 1. && scaleY != 1.)
            {
                // Error check
                if (!keypoints.empty() && keypoints.getSize(2) != 3)
                    error(errorMessage, __LINE__, __FUNCTION__, __FILE__);
                // Get #people and #parts
                const auto numberPeople = keypoints.getSize(0);
                const auto numberParts = keypoints.getSize(1);
                // For each person
                for (auto person = 0 ; person < numberPeople ; person++)
                {
                    // For each body part
                    for (auto part = 0 ; part < numberParts ; part++)
                    {
                        const auto finalIndex = keypoints.getSize(2)*(person*numberParts + part);
                        keypoints[finalIndex] = keypoints[finalIndex] * scaleX + offsetX;
                        keypoints[finalIndex+1] = keypoints[finalIndex+1] * scaleY + offsetY;
                    }
                }
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void renderKeypointsCpu(Array<float>& frameArray, const Array<float>& keypoints,
                            const std::vector<unsigned int>& pairs, const std::vector<float> colors,
                            const float thicknessCircleRatio, const float thicknessLineRatioWRTCircle,
                            const float threshold)
    {
        try
        {
            if (!frameArray.empty())
            {
                // Array<float> --> cv::Mat
                auto frame = frameArray.getCvMat();

                // Security check
                if (frame.dims != 3 || frame.size[0] != 3)
                    error(errorMessage, __LINE__, __FUNCTION__, __FILE__);

                // Get frame channels
                const auto width = frame.size[2];
                const auto height = frame.size[1];
                const auto area = width * height;
                cv::Mat frameB(height, width, CV_32FC1, &frame.data[0]);
                cv::Mat frameG(height, width, CV_32FC1, &frame.data[area * sizeof(float) / sizeof(uchar)]);
                cv::Mat frameR(height, width, CV_32FC1, &frame.data[2 * area * sizeof(float) / sizeof(uchar)]);
                cv::Mat matMask(height, width, CV_32FC1, cv::Scalar(0.0, 0.0, 0.0));
                cv::Mat matULMask(height, width, CV_32FC1, cv::Scalar(0.0, 0.0, 0.0));
                cv::Mat matEyeMask(height, width, CV_32FC1, cv::Scalar(0.0, 0.0, 0.0));
                std::vector<std::vector<cv::Point> >vvpt2fFacePts(keypoints.getSize(0));
                std::vector<std::vector<cv::Point> >vvpt2fUpLipsPts(keypoints.getSize(0));
                std::vector<std::vector<cv::Point> >vvpt2fDownLipsPts(keypoints.getSize(0));
                std::vector<std::vector<cv::Point> >vvpt2fLeftEyePts(keypoints.getSize(0));
                std::vector<std::vector<cv::Point> >vvpt2fRightEyePts(keypoints.getSize(0));

                std::vector<cv::Point>vptFPts;
                std::vector<cv::Point>vptUpLipsPts;
                std::vector<cv::Point>vptDownLipsPts;
                std::vector<cv::Point>vptLeftEyePts;
                std::vector<cv::Point>vptRightEyePts;
                std::vector<cv::Mat>vmatFaceMask(keypoints.getSize(0));
                

                // Parameters
                const auto lineType = 8;
                const auto shift = 0;
                const auto numberColors = colors.size();
                const auto thresholdRectangle = 0.1f;
                const auto numberKeypoints = keypoints.getSize(1);
                bool bLoop = false;
                // Keypoints
                for (auto person = 0 ; person < keypoints.getSize(0) ; person++)
                {
                    vmatFaceMask[person] = cv::Mat::zeros(height, width, CV_8UC1);
                    const auto personRectangle = getKeypointsRectangle(keypoints, person, numberKeypoints,
                                                                       thresholdRectangle);
                    if (personRectangle.area() > 0)
                    {
                        const auto ratioAreas = fastMin(1.f, fastMax(personRectangle.width/(float)width,
                                                                     personRectangle.height/(float)height));
                        // Size-dependent variables
                        const auto thicknessRatio = fastMax(intRound(std::sqrt(area)
                                                                     * thicknessCircleRatio * ratioAreas), 2);
                        // Negative thickness in cv::circle means that a filled circle is to be drawn.
                        const auto thicknessCircle = (ratioAreas > 0.05 ? thicknessRatio : -1);
                        const auto thicknessLine = intRound(thicknessRatio * thicknessLineRatioWRTCircle);
                        const auto radius = thicknessRatio / 2;
                        vptFPts.resize(27);
                        vptUpLipsPts.resize(12);
                        vptDownLipsPts.resize(10);
                        vptLeftEyePts.resize(6);
                        vptRightEyePts.resize(6);

                        // Draw lines
                       /* for (auto pair = 0u ; pair < pairs.size() ; pair+=2)
                        {
                            const auto index1 = (person * numberKeypoints + pairs[pair]) * keypoints.getSize(2);
                            const auto index2 = (person * numberKeypoints + pairs[pair+1]) * keypoints.getSize(2);
                            if (keypoints[index1+2] > threshold && keypoints[index2+2] > threshold)
                            {
                                const auto colorIndex = pairs[pair+1]*3; // Before: colorIndex = pair/2*3;
                                const cv::Scalar color{colors[colorIndex % numberColors],
                                                       colors[(colorIndex+1) % numberColors],
                                                       colors[(colorIndex+2) % numberColors]};
                                const cv::Point keypoint1{intRound(keypoints[index1]), intRound(keypoints[index1+1])};
                                const cv::Point keypoint2{intRound(keypoints[index2]), intRound(keypoints[index2+1])};
                                cv::line(frameR, keypoint1, keypoint2, color[0], thicknessLine, lineType, shift);
                                cv::line(frameG, keypoint1, keypoint2, color[1], thicknessLine, lineType, shift);
                                cv::line(frameB, keypoint1, keypoint2, color[2], thicknessLine, lineType, shift);
                            }
                        }

                        // Draw circles
                        for (auto part = 0 ; part < numberKeypoints ; part++)
                        {
                            const auto faceIndex = (person * numberKeypoints + part) * keypoints.getSize(2);
                            if (keypoints[faceIndex+2] > threshold)
                            {
                                const auto colorIndex = part*3;
                                const cv::Scalar color{colors[colorIndex % numberColors],
                                                       colors[(colorIndex+1) % numberColors],
                                                       colors[(colorIndex+2) % numberColors]};
                                const cv::Point center{intRound(keypoints[faceIndex]),
                                                       intRound(keypoints[faceIndex+1])};
                                cv::circle(frameR, center, radius, color[0], thicknessCircle, lineType, shift);
                                cv::circle(frameG, center, radius, color[1], thicknessCircle, lineType, shift);
                                cv::circle(frameB, center, radius, color[2], thicknessCircle, lineType, shift);
                            }
                        }*/
                        
                        auto n = 0;
                        auto m = 0;
                        auto l = 0;
                        auto o = 0;
                        auto p = 0;
                        auto k19 = keypoints[(person * numberKeypoints + 19) * keypoints.getSize(2) + 1];
                        auto k8 = keypoints[(person * numberKeypoints + 8) * keypoints.getSize(2) + 1];
                        auto h = (k8 - k19) * 0.3f;
                        for (auto part = 0 ; part < 17 ; part++)
                        {
                            const auto faceIndex = (person * numberKeypoints + part) * keypoints.getSize(2);
                            vptFPts[n] = cv::Point(keypoints[faceIndex], keypoints[faceIndex+1]);
                            n++;
                        }
                        
                        for (auto part = 26 ; part >= 17 ; part--)
                        {
                            const auto faceIndex = (person * numberKeypoints + part) * keypoints.getSize(2);
                            auto th = (keypoints[faceIndex+1] - h) > 0 ? (keypoints[faceIndex+1] - h):0;
                            vptFPts[n] = cv::Point(keypoints[faceIndex], th);
                            n++;
                        }
                        vvpt2fFacePts[person] = vptFPts;
                        
                        for (auto part = 48 ; part < 55 ; part++)
                        {
                            const auto faceIndex = (person * numberKeypoints + part) * keypoints.getSize(2);
                            vptUpLipsPts[m] = cv::Point(keypoints[faceIndex], keypoints[faceIndex+1]);
                            m++;
                        }
                        
                        for (auto part = 64 ; part >= 60 ; part--)
                        {
                            const auto faceIndex = (person * numberKeypoints + part) * keypoints.getSize(2);
                            vptUpLipsPts[m] = cv::Point(keypoints[faceIndex], keypoints[faceIndex+1]);
                            m++;
                        }
                        vvpt2fUpLipsPts[person] = vptUpLipsPts;
                        
                        for (auto part = 54 ; part < 60 ; part++)
                        {
                            const auto faceIndex = (person * numberKeypoints + part) * keypoints.getSize(2);
                            vptDownLipsPts[l] = cv::Point(keypoints[faceIndex], keypoints[faceIndex+1]);
                            l++;
                        }
                        
                        vptDownLipsPts[l++] = cv::Point(keypoints[(person * numberKeypoints + 48) * keypoints.getSize(2)], keypoints[(person * numberKeypoints + 48) * keypoints.getSize(2)+1]);
                        
                        for (auto part = 67 ; part >= 65 ; part--)
                        {
                            const auto faceIndex = (person * numberKeypoints + part) * keypoints.getSize(2);
                            
                            vptDownLipsPts[l] = cv::Point(keypoints[faceIndex], keypoints[faceIndex+1]);
                            l++;
                        }
                        vvpt2fDownLipsPts[person] = vptDownLipsPts;
                        
                        for (auto part = 36 ; part < 42 ; part++)
                        {
                            const auto faceIndex = (person * numberKeypoints + part) * keypoints.getSize(2);
                            vptLeftEyePts[o] = cv::Point(keypoints[faceIndex], keypoints[faceIndex+1]);
                            o++;
                        }
                        vvpt2fLeftEyePts[person] = vptLeftEyePts;
                        for (auto part = 42 ; part < 48 ; part++)
                        {
                            const auto faceIndex = (person * numberKeypoints + part) * keypoints.getSize(2);
                            
                            vptRightEyePts[p] = cv::Point(keypoints[faceIndex], keypoints[faceIndex+1]);
                            p++;
                        }
                        vvpt2fRightEyePts[person] = vptRightEyePts;
                        bLoop = true;
                        
                        //auto faceIndex = (person * numberKeypoints + 68) * keypoints.getSize(2);
                        //cv::Point ptLeftEye = cv::Point(keypoints[faceIndex], keypoints[faceIndex+1]);
                        //cv::Point ptRightEye = cv::Point(keypoints[faceIndex+2], keypoints[faceIndex+3]);
                        //circle(matEyeMask, ptLeftEye, 5, cv::Scalar(255.0, 255.0, 255.0), -1);
                        //circle(matEyeMask, ptRightEye, 5, cv::Scalar(255.0, 255.0, 255.0), -1);
                    }
                }
                if(bLoop){
                    cv::fillPoly(matMask, vvpt2fFacePts, cv::Scalar(255.0, 255.0, 255.0));
                    cv::fillPoly(matULMask, vvpt2fUpLipsPts, cv::Scalar(255.0, 255.0, 255.0));
                    cv::fillPoly(matULMask, vvpt2fDownLipsPts, cv::Scalar(255.0, 255.0, 255.0));
                    cv::fillPoly(matEyeMask, vvpt2fLeftEyePts, cv::Scalar(255.0, 255.0, 255.0));
                    cv::fillPoly(matEyeMask, vvpt2fRightEyePts, cv::Scalar(255.0, 255.0, 255.0));
                
                    //cv::imshow("matMask", matEyeMask);
                    //cv::waitKey(-1);
                    cv::GaussianBlur(matMask, matMask, cv::Size(65, 65), 0, 0);
                    cv::GaussianBlur(matULMask, matULMask, cv::Size(15, 15), 0, 0);
                    cv::GaussianBlur(matEyeMask, matEyeMask, cv::Size(5, 5), 0, 0);
                    matULMask = matULMask / 1.2f;
                    matEyeMask = matEyeMask / 2.0;
                    //cv::imshow("matULMask", matULMask);
                    //cv::waitKey(-1);
                    
                
                    cv::Mat matframeR = frameR.clone();
                    cv::Mat matframeB = frameB.clone();
                    cv::Mat matframeG = frameG.clone();
                
                    Beauty(frameR, 1.1, 32.0);
                    Beauty(frameB, 1.1, 32.0);
                    Beauty(frameG, 1.1, 32.0);              
                    
                    cv::Mat aGMat;
                    cv::Mat aBMat;
                    cv::GaussianBlur(frameR, aGMat, cv::Size(5, 5), 0, 0);
                    cv::bilateralFilter(aGMat, aBMat, 5, // 整体磨皮  
                    30 * 2, 30 / 2);
                    cv::addWeighted(aGMat, 1.5, aBMat, -0.5, 0, frameR);
            
                    cv::GaussianBlur(frameG, aGMat, cv::Size(5, 5), 0, 0);
                    cv::bilateralFilter(aGMat, aBMat, 5, // 整体磨皮  
                    30 * 2, 30 / 2);
                    cv::addWeighted(aGMat, 1.5, aBMat, -0.5, 0, frameG);
                
                    cv::GaussianBlur(frameB, aGMat, cv::Size(5, 5), 0, 0);
                    cv::bilateralFilter(aGMat, aBMat, 5, // 整体磨皮  
                    30 * 2, 30 / 2);
                    cv::addWeighted(aGMat, 1.5, aBMat, -0.5, 0, frameB);
                
               
                    CopyTo(matframeR, frameR, matMask);
                    CopyTo(matframeB, frameB, matMask);
                    CopyTo(matframeG, frameG, matMask);
                
                    fillColor(frameR, 255.0f, matULMask);
                    fillColor(frameG, 100.0f, matULMask);
                    fillColor(frameB, 10.0f, matULMask);
                    fillColor(frameR, 10.0f, matEyeMask, 120.0f);
                    fillColor(frameG, 2550.0f, matEyeMask, 120.0);
                    fillColor(frameB, 255.0f, matEyeMask, 120.0f);
                }
        
                
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
    
    OP_API void fillColor(cv::Mat& src, float fColor0, cv::Mat& matMask0, float fColor1, cv::Mat matMask1){
        try{
            if(src.empty())
                error("Input image is empty!", __LINE__, __FUNCTION__, __FILE__);
            
            matMask0 = matMask0 / 255.0f;
            matMask1 = matMask1 / 255.0f;
            
            float* pfsrc = src.ptr<float>(0);
            float* pfmask = matMask0.ptr<float>(0);
            float* pfmask1 = matMask1.ptr<float>(0);
            
            
            
            int nwidth = src.cols;
            int nheight = src.rows;
            int c;
            //for(r = 0; r < nheight; r++){
            for(c = 0; c < nwidth * nheight; c++){
                if(fabs(pfmask[c]) > 1e-6 )
                    pfsrc[c] =  ((1.0f - pfmask[c])) * pfsrc[c] + (pfmask[c]) * fColor0 ;
                if(fabs(pfmask1[c]) > 1e-6)
                    pfsrc[c] =  ((1.0f - pfmask1[c])) * pfsrc[c] + (pfmask1[c]) * fColor1;
            }
            //}
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return ;
        }   
    }
    
    OP_API void fillColor(cv::Mat& src, float fColor0, cv::Mat& matMask0){
        try{
            if(src.empty())
                error("Input image is empty!", __LINE__, __FUNCTION__, __FILE__);
            
            matMask0 = matMask0 / 255.0f;
            
            float* pfsrc = src.ptr<float>(0);
            float* pfmask = matMask0.ptr<float>(0);           
            
            int nwidth = src.cols;
            int nheight = src.rows;
            int c;
            //for(r = 0; r < nheight; r++){
            for(c = 0; c < nwidth * nheight; c++){
                if(fabs(pfmask[c]) > 1e-6 )
                    pfsrc[c] =  ((1.0f - pfmask[c])) * pfsrc[c] + (pfmask[c]) * fColor0 ;
            }
            //}
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return ;
        }   
    }
    
    OP_API void fillColor(cv::Mat& src, float fColor0, cv::Mat& matMask0, float fThreshold){
        try{
            if(src.empty())
                error("Input image is empty!", __LINE__, __FUNCTION__, __FILE__);
            
            matMask0 = matMask0 / 255.0f;
            
            float* pfsrc = src.ptr<float>(0);
            float* pfmask = matMask0.ptr<float>(0);           
            
            int nwidth = src.cols;
            int nheight = src.rows;
            int c;
            //for(r = 0; r < nheight; r++){
            for(c = 0; c < nwidth * nheight; c++){
                if(pfsrc[c] < fThreshold  && fabs(pfmask[c]) > 1e-6 )
                    pfsrc[c] =  ((1.0f - pfmask[c])) * pfsrc[c] + (pfmask[c]) * fColor0 ;
            }
            //}
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return ;
        }   
    }
    
    OP_API void CopyTo(cv::Mat& src, cv::Mat& dst, cv::Mat& mask){
        try{
            if(src.empty())
                error("Input image is empty!", __LINE__, __FUNCTION__, __FILE__);
            float* pfsrc = src.ptr<float>(0);
            float* pfdst = dst.ptr<float>(0);
            float* pfmask = mask.ptr<float>(0);
            
            int nwidth = src.cols;
            int nheight = src.rows;
            int c;
            //for(r = 0; r < nheight; r++){
            for(c = 0; c < nwidth * nheight; c++){
                pfdst[c] = ((255.0f - pfmask[c])/255.0f) * pfsrc[c] + (pfmask[c]/255.0f) * pfdst[c];    
            }
            //}
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return ;
        }   
    }

    OP_API void Beauty(cv::Mat& frame, double dcontrast, double nbrightness)
    {
        try{
            if(frame.empty())
                error("Input image is empty!", __LINE__, __FUNCTION__, __FILE__);
            float* pv3bframe = frame.ptr<float>(0);
            
            int nwidth = frame.cols;
            int nheight = frame.rows;
            int r, c;
            //for(r = 0; r < nheight; r++){
            for(c = 0; c < nwidth * nheight; c++){
                pv3bframe[c] = static_cast<float>(dcontrast*(pv3bframe[c]) + nbrightness);
            }
            //}
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return ;
        }       
    }
    
    Rectangle<float> getKeypointsRectangle(const Array<float>& keypoints, const int person, const int numberKeypoints,
                                           const float threshold)
    {
        try
        {
            // Security checks
            if (numberKeypoints < 1)
                error("Number body parts must be > 0", __LINE__, __FUNCTION__, __FILE__);
            // Define keypointPtr
            const auto keypointPtr = keypoints.getConstPtr() + person * keypoints.getSize(1) * keypoints.getSize(2);
            float minX = std::numeric_limits<float>::max();
            float maxX = 0.f;
            float minY = minX;
            float maxY = maxX;
            for (auto part = 0 ; part < numberKeypoints ; part++)
            {
                const auto score = keypointPtr[3*part + 2];
                if (score > threshold)
                {
                    const auto x = keypointPtr[3*part];
                    const auto y = keypointPtr[3*part + 1];
                    // Set X
                    if (maxX < x)
                        maxX = x;
                    if (minX > x)
                        minX = x;
                    // Set Y
                    if (maxY < y)
                        maxY = y;
                    if (minY > y)
                        minY = y;
                }
            }
            if (maxX >= minX && maxY >= minY)
                return Rectangle<float>{minX, minY, maxX-minX, maxY-minY};
            else
                return Rectangle<float>{};
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return Rectangle<float>{};
        }
    }

    float getAverageScore(const Array<float>& keypoints, const int person)
    {
        try
        {
            // Security checks
            if (person >= keypoints.getSize(0))
                error("Person index out of bounds.", __LINE__, __FUNCTION__, __FILE__);
            // Get average score
            auto score = 0.f;
            const auto numberKeypoints = keypoints.getSize(1);
            const auto area = numberKeypoints * keypoints.getSize(2);
            const auto personOffset = person * area;
            for (auto part = 0 ; part < numberKeypoints ; part++)
                score += keypoints[personOffset + part*keypoints.getSize(2) + 2];
            return score / numberKeypoints;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return 0.f;
        }
    }

    float getKeypointsArea(const Array<float>& keypoints, const int person, const int numberKeypoints,
                           const float threshold)
    {
        try
        {
            return getKeypointsRectangle(keypoints, person, numberKeypoints, threshold).area();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return 0.f;
        }
    }

    int getBiggestPerson(const Array<float>& keypoints, const float threshold)
    {
        try
        {
            if (!keypoints.empty())
            {
                const auto numberPeople = keypoints.getSize(0);
                const auto numberKeypoints = keypoints.getSize(1);
                auto biggestPoseIndex = -1;
                auto biggestArea = -1.f;
                for (auto person = 0 ; person < numberPeople ; person++)
                {
                    const auto newPersonArea = getKeypointsArea(keypoints, person, numberKeypoints, threshold);
                    if (newPersonArea > biggestArea)
                    {
                        biggestArea = newPersonArea;
                        biggestPoseIndex = person;
                    }
                }
                return biggestPoseIndex;
            }
            else
                return -1;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return -1;
        }
    }
}
