#ifndef SVM_MODEL_HPP_INCLUDED
#define SVM_MODEL_HPP_INCLUDED

#include <algorithm>
#include <random>
#include <opencv2/ml.hpp>
using namespace cv;
using namespace cv::ml;






class SVM_Images{
    vector<Mat> trainCells;
    vector<Mat> testCells;
    vector<int> trainLabels;
    vector<int> testLabels;
    std::vector<std::vector<float> > trainHOG;
    std::vector<std::vector<float> > testHOG;
    Ptr<SVM> model;
    Mat testMat;

    HOGDescriptor hog;

    public:
        char classNames[20][20];

        void SVM_reset(){
            trainLabels.clear();
            testLabels.clear();
            trainHOG.clear();
            testHOG.clear();
        }

        void SVM_loadNewClass(float ratioTrain, String path, int idClass, char classN[]){
            vector<cv::String> fn;
            cv::glob(path,fn,true);
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            shuffle (fn.begin(), fn.end(), std::default_random_engine(seed));


            hog= HOGDescriptor(
                Size(20,20), //winSize
                Size(10,10), //blocksize
                Size(5,5), //blockStride,
                Size(5,5), //cellSize,
                9,   //nbins,
                1,   //derivAper,
                -1,  //winSigma,
                cv::HOGDescriptor::HistogramNormType::L2Hys, //histogramNormType,
                0.2, //L2HysThresh,
                0,   //gammal correction,
                64,  //nlevels=64
                1
            );

            int img_index = 0;
            strcpy(classNames[idClass], classN);
            printf("Loading class %d\n", idClass);
            for (size_t k=0; k<fn.size(); ++k)
            {
                Mat im = imread(fn[k], IMREAD_GRAYSCALE);

                if(im.cols == 0)
                    continue;
                Size s(20, 20);
                resize(im, im, s);

                if (im.empty()) continue;

                if(img_index <= fn.size()*ratioTrain){
                    trainCells.push_back(im);
                    trainLabels.push_back(idClass);
                    vector<float> descriptors;
                    hog.compute(im,descriptors);
                    trainHOG.push_back(descriptors);
                }
                else{
                    testCells.push_back(im);
                    testLabels.push_back(idClass);
                    vector<float> descriptors;
                    hog.compute(im,descriptors);
                    testHOG.push_back(descriptors);
                }
                img_index++;
            }
        }

        void SVM_loadModel(char modelName[]){
            char path[200];
            strcpy(path, "/home/pi/WORKSPACE/provaOpencv4/objects_detection/");
            strcat(path, modelName);
            strcat(path, ".yml");
            model = cv::Algorithm::load<ml::SVM>(path);
        }

        void SVM_trainModel(char modelName[]){
            printf("testHOG %d\n", testHOG.size());
            printf("testLabels %d\n", testLabels.size());
            int descriptor_size = trainHOG[0].size();
            cout << "Descriptor Size : " << descriptor_size << endl;
            Mat trainMat(trainHOG.size(),descriptor_size,CV_32FC1);
            testMat = Mat(testHOG.size(),descriptor_size,CV_32FC1);

            for(int i = 0;i<trainHOG.size();i++){
                for(int j = 0;j<descriptor_size;j++){
                   trainMat.at<float>(i,j) = trainHOG[i][j];
                }
            }
            for(int i = 0;i<testHOG.size();i++){
                for(int j = 0;j<descriptor_size;j++){
                    testMat.at<float>(i,j) = testHOG[i][j];
                }
            }
            //showImage(trainMat, 0);
            //showImage(testMat, 0);
            float C = 0.5, gamma = 0.5;
            model = svmInit(C, gamma);
            svmTrain(model, trainMat, trainLabels, modelName);
        }

        float SVM_getAccuracy(){
            Mat testResponse;
            svmPredict(model, testResponse, testMat);

            float count = 0;
            float accuracy = 0;
            SVMevaluate(testResponse, count, accuracy, testLabels);

            cout << "the accuracy is :" << accuracy << endl;
            return accuracy;
        }

        int SVM_predictImage(Mat m){
            Size s(20, 20);
            resize(m, m, s);
            cvtColor(m, m, COLOR_BGR2GRAY);
            vector<float> descriptors;
            hog.compute(m, descriptors);
            std::vector<std::vector<float>> pHOG;
            pHOG.push_back(descriptors);
            int descriptor_size = pHOG[0].size();
            Mat mm(pHOG.size(),descriptor_size,CV_32FC1);
            for(int i = 0;i<pHOG.size();i++){
                for(int j = 0;j<descriptor_size;j++){
                    mm.at<float>(i,j) = pHOG[i][j];
                }
            }
            Mat testR;
            model->predict(mm, testR);

            return (int)testR.at<float>(0,0);
        }



    private:

        Ptr<SVM> svmInit(float C, float gamma)
        {
          Ptr<SVM> svm = SVM::create();
          svm->setGamma(gamma);
          svm->setC(C);
          svm->setKernel(SVM::INTER);
          svm->setType(SVM::C_SVC);

          return svm;
        }

        void svmTrain(Ptr<SVM> svm, Mat &trainMat, vector<int> &trainLabels, char modelName[])
        {
          Ptr<TrainData> td = TrainData::create(trainMat, ROW_SAMPLE, trainLabels);
          svm->train(td);
        }

        void svmPredict(Ptr<SVM> svm, Mat &testResponse, Mat &testMat )
        {
          svm->predict(testMat, testResponse);
        }

        void SVMevaluate(Mat &testResponse, float &count, float &accuracy, vector<int> &testLabels)
        {
          int totalPerClass[20] = {0};
          int truePerClass[20] = {0};

          for(int i = 0; i < testResponse.rows; i++)
          {
            //cout << testResponse.at<float>(i,0) << " " << testLabels[i] << endl;
            totalPerClass[testLabels[i]] += 1;
            if(testResponse.at<float>(i,0) == testLabels[i]){
              count = count + 1;
              truePerClass[testLabels[i]] += 1;
            }
          }
          for(int i=0; i<20; i++){
            printf("Class:%d -> %f %\n", i, (float)((float)truePerClass[i]/(float)totalPerClass[i])*100.0f);
          }
          accuracy = (count/testResponse.rows)*100;
        }
};



#endif // SVM_MODEL_HPP_INCLUDED
