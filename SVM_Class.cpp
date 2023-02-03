
/*
C++ example of a model that predicts a domino number with a voting system
*/

SVM_Images svm1[25];


int predictNumber(Mat src_gray){
    Scalar me = mean(src_gray);
    int predictions[7] = {0};
    Mat src_gray2;
    threshold(src_gray, src_gray2, me[0], 255, 0 );
    Mat mBGR;
    cvtColor( src_gray2, mBGR, COLOR_GRAY2BGR );

    showImage(mBGR, 0);

    for(int i=0; i<25; i++){
        predictions[svm1[i].SVM_predictImage(mBGR)]++;
    }

    int ind = 0;
    int maxVal = 0;
    for(int k=0;k<7;k++){
        if(predictions[k] > maxVal){
            maxVal = predictions[k];
            ind = k;
        }
    }
    return ind;
}



int main(){
    for(int i=0; i<25; i++){
        svm1[i].SVM_reset();
        float ratio = 0.5f;
        if(i > 18)
            ratio = 0.9f;
        else if(i > 12)
            ratio = 0.7f;
        svm1[i].SVM_loadNewClass(ratio, String("/home/pi/WORKSPACE/provaOpencv4/objects_detection/domino2/0/*.jpg"), 0, "0");
        svm1[i].SVM_loadNewClass(ratio, String("/home/pi/WORKSPACE/provaOpencv4/objects_detection/domino2/1/*.jpg"), 1, "1");
        svm1[i].SVM_loadNewClass(ratio, String("/home/pi/WORKSPACE/provaOpencv4/objects_detection/domino2/2/*.jpg"), 2, "2");
        svm1[i].SVM_loadNewClass(ratio, String("/home/pi/WORKSPACE/provaOpencv4/objects_detection/domino2/3/*.jpg"), 3, "3");
        svm1[i].SVM_loadNewClass(ratio, String("/home/pi/WORKSPACE/provaOpencv4/objects_detection/domino2/4/*.jpg"), 4, "4");
        svm1[i].SVM_loadNewClass(ratio, String("/home/pi/WORKSPACE/provaOpencv4/objects_detection/domino2/5/*.jpg"), 5, "5");
        svm1[i].SVM_loadNewClass(ratio, String("/home/pi/WORKSPACE/provaOpencv4/objects_detection/domino2/6/*.jpg"), 6, "6");
        svm1[i].SVM_trainModel("ElMeuModel_SVM");
        svm1[i].SVM_getAccuracy();
    }
	//Call the "predictNumber" function and pass a grayscale picture as a parameter
	
	//Example
	printf("Predicted number: %d\n", predictNumber(loadBW_Picture()));
	return 0;
}