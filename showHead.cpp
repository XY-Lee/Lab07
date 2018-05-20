#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <sstream>
using namespace cv;
using namespace std;
// the minimum object size
int min_face_height = 50;
int min_face_width = 50;

# define HEAD_SHOW_HEIGHT 100
# define HEAD_SHOW_WIDTH 100

# define ANI_HEAD_MAX 29
const string gHeadName[29] = { "01.png", "02.png", "03.png", "04.png", "05.png", "06.png",
  "07.png", "08.png", "09.png", "10.png", "11.png", "12.png",
  "13.png", "14.png", "15.png", "16.png", "17.png", "18.png",
  "19.png", "20.png", "21.png", "22.png", "23.png", "24.png",
  "25.png", "26.png", "27.png", "28.png", "29.png" } ; // animal head file name

class Img {
public :
  string name ;
  string path ;
  Mat img ;

  Img() {} // Img() 

  Img( string n ) {
    name = n ;
    path = "./" ;
    img = imread( path + n, CV_LOAD_IMAGE_UNCHANGED);
  } // Img()

  Img( string n, string inPath ) {
    name = n ;
    path = inPath ;
    img = imread( inPath + n, CV_LOAD_IMAGE_UNCHANGED);
  } // Img()
} ;

Img gAniHead[29] ;

bool cvAdd4cMat_q( Mat &dst, Mat &scr, double alpha) {   
  if (dst.channels() != 3 || scr.channels() != 4) {    
    return false;    
  } // if 
  if (alpha < 0.01)    
    return false;    
  vector<Mat>scr_channels;    
  vector<Mat>dstt_channels;    
  split(scr, scr_channels);    
  split(dst, dstt_channels);    
  CV_Assert(scr_channels.size() == 4 && dstt_channels.size() == 3);    

  if (alpha < 1) {    
    scr_channels[3] *= alpha;    
    alpha = 1;    
  } // if

  for (int i = 0; i < 3; i++) {    
    dstt_channels[i] = dstt_channels[i].mul(255.0 / alpha - scr_channels[3], alpha / 255.0);    
    dstt_channels[i] += scr_channels[i].mul(scr_channels[3], alpha / 255.0);    
  } // for

  merge(dstt_channels, dst);    
  return true;    
} // cvAdd4cMat_q()

void ShowAnimalHead() {
  Scalar color = Scalar(255,255,255,255);
  Scalar fontColor = Scalar(0,0,255);
  Mat backMat( HEAD_SHOW_WIDTH * 6, HEAD_SHOW_HEIGHT * 5, CV_8UC4, color ) ;
  Mat temp ;
  Mat resizeImg ;
  for ( int i = 0 ; i < 6 ; i++ ) {
    for ( int j = 0 ; j < 5 && j + 5 * i < ANI_HEAD_MAX ; j++ ) {
      resize( gAniHead[j + 5 * i].img, resizeImg, Size( HEAD_SHOW_WIDTH, HEAD_SHOW_HEIGHT ) ) ;
      temp = backMat( Rect( j * HEAD_SHOW_WIDTH , i * HEAD_SHOW_HEIGHT, resizeImg.cols, resizeImg.rows ) );  //指定插入的大小和位置
      /*
      Mat mask ;
      resizeImg.convertTo( mask, 0 ) ;
      resizeImg.copyTo( temp ) ;
      */
      if ( !cvAdd4cMat_q( temp, resizeImg, 1.0 ) )
      addWeighted( temp, 0, resizeImg, 1.0, 0, temp );
      stringstream temp ;
      temp << j + 5 * i ;
      string w ;
      temp >> w ;
      w = w + ":" + gAniHead[j + 5 * i].name ;
      putText( backMat, w, Point( j * HEAD_SHOW_WIDTH, i * HEAD_SHOW_HEIGHT + 15 ),  FONT_HERSHEY_DUPLEX, 0.6, fontColor ) ;

    } // for
  } // for

  namedWindow("Animal Head", 1);
  imshow("Animal Head", backMat ) ;
} // ShowAnimalHead()

void ShowDetectFace( Img mainImg, CvSeq *&faces ) {
  // Load image
  IplImage *image_detect = new IplImage( mainImg.img );
  string cascade_name = "/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml" ;
  // Load cascade
  CvHaarClassifierCascade* classifier=(CvHaarClassifierCascade*)cvLoad(cascade_name.c_str(), 0, 0, 0);
  if ( !classifier ) {
    cerr<<"ERROR: Could not load classifier cascade."<<endl;
    return ;
  } // if

  CvMemStorage *facesMemStorage = cvCreateMemStorage( 0 ) ;
  IplImage *tempFrame=cvCreateImage( cvSize( image_detect->width, image_detect->height ), IPL_DEPTH_8U, image_detect->nChannels );
  if ( image_detect->origin == IPL_ORIGIN_TL ) {
    cvCopy(image_detect, tempFrame, 0);
  } // if
  else{
    cvFlip( image_detect, tempFrame, 0 );
  } // else

  cvClearMemStorage(facesMemStorage);
  faces = cvHaarDetectObjects( tempFrame, classifier, facesMemStorage, 1.1, 3, 
    CV_HAAR_DO_CANNY_PRUNING, cvSize(min_face_width, min_face_height ) );

  namedWindow("Face Detection Result", 1);
  CvFont font;
  double hScale = 1.0;
  double vScale = 1.0;
  int    lineWidth = 2;
  cvInitFont( &font, CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC, hScale, vScale, 0, lineWidth );

  if ( faces ) {
    for ( int i = 0 ; i < faces->total ; ++i ) {
      /* get openCV analysis faces' point */
      /* hint : google CvRect、CvPoint */
      /* point1 is rectangle's left top point */
      /* point2 is rectangle's right bottom point */
      CvPoint point1, point2;
      CvRect* rectangle = (CvRect*)cvGetSeqElem(faces, i);
      point1.*** = rectangle->***;
      point2.*** = rectangle->*** + rectangle->***;
      point1.*** = rectangle->***;
      point2.*** = rectangle->*** + rectangle->***;
      cvRectangle( tempFrame, point1, point2, CV_RGB(255,0,0), 3, 8, 0); // drow rectangle whith red
      stringstream temp ;
      temp << (char)('A' + i) ;
      string w ;
      temp >> w ;
      /* write word to photo in point1 */
      cvPutText( tempFrame, w.c_str(), Point( point1.*** + 2, point1.*** + 25 ),  &font, cvScalar( 200, 200, 100 ) ) ;
      temp.clear() ;
    } // for
  } // if

  imshow("Face Detection Result", Mat( tempFrame ) ) ; // tempFrame);
} // ShowDetectFace()

bool ChangeFace( CvSeq *faces, Img &mainImg, int faceNum, int headIndex ) {
  if ( !faces ) {
    cout << "error : no face" << endl ;
    return false ;
  } // if

  if ( faces->total <= faceNum || faceNum < 0 ) {
    cout << "error : faceNum out of range" << endl ;
    return false ;
  } // if

  if ( headIndex >= ANI_HEAD_MAX || headIndex < 0 ) {
    cout << "error : headIndex out of range" << endl ;
    return false ;
  } // if

  /* get openCV analysis faces' point */
  /* hint : google CvRect、CvPoint */
  CvPoint point;
  CvRect* rectangle = (CvRect*)cvGetSeqElem( faces, *** );
  point.*** = rectangle->*** ;
  point.*** = rectangle->***;
  
  Mat reScaleTemp;
  Mat imgROI ;
  resize( gAniHead[ *** ].img, reScaleTemp, Size( rectangle->***, rectangle->*** ) ) ; // make head's size to rectangle's size
  /* make reSized animal head copy to face */
  // get the face area's memory data to imgROI from point in order to set reSized animal head's area
  imgROI = mainImg.img( Rect( point.***, point.***, reScaleTemp.cols, reScaleTemp.rows ) ); 
  if ( !cvAdd4cMat_q( imgROI, reScaleTemp, 1.0 ) )
    addWeighted(imgROI, 0, reScaleTemp, 1.0, 0, imgROI );
  
  imshow("Change Face", *** ) ; // show mainImg

} // ChangeFace()

int main( int argc , char ** argv ){

  /* init head */
  for ( int i = 0 ; i < ANI_HEAD_MAX ; i++ ) {
    gAniHead[i] = Img( gHeadName[i], "./animalHead/" ) ;
  } // for

  ShowAnimalHead() ;

  Img mainImg( "XD2.jpg" ) ; // your Photo
  CvSeq *faces = NULL ;
  ShowDetectFace( mainImg, faces ) ;

  namedWindow("Change Face", 1);

  /* you can add your parameter */


  while (  ) { // you need to write your break condition
    cvWaitKey( 5000 ) ; // must need to wait another process drow photo
    /* design your input method */
    
    
    
    
    // when you want to drow, please call ChangeFace( faces, mainImg, ***1, ***2 ) ;
    // ***1 is face index. When photo show A, its index is 0.
    // ***2 is head index. When photo show 5:06.png, its index is 5.
    /* design your input method */
  } // while

  cvWaitKey( 0 );
  return EXIT_SUCCESS;
}
