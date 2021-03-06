/**
 * Author: Bilkit Githinji
 * Version: 3.0
 * Description:T his program tracks foreground objects within a video file.
 * It performs background subtraction based on Gaussian Mixture Model (GMM)
 * with shadow detection.
 * Command: ./moth_tracker <video_file> <data_file>
 **/


/*** @TODO: produce visitation sequence to compare with human data
***/

//NOTES:
// For background update, use pMOG2->operator()(frame_rectified,fgMaskMOG2) and initialize with
// average of all frames. pMOG2->operator()(frame_rectified,fgMaskMOG2, learningRate) is another
// option, but adjusting learning rate causes relatively stationary objects to blend
// in with the background model. This is problematic for tracking when moth is visiting.

//opencv
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
//c++
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <boost/lexical_cast.hpp> //  string concatination with doubles
#include <math.h>
#include <limits.h> // access quiet NAN
#include <algorithm> // for max heap
#include <vector>


#define _USE_MATH_DEFINES

using namespace cv;
using namespace std;
using namespace boost;


typedef numeric_limits<float> LIMITS;

int const DEPTH = CV_16S;// 16short is used to prevent overflow during gradient cal
int const SCALE = 1,DELTA = 0,RADIUS = 5;
int const MIN_AREA = 100;  // *** increasing this value --> more spotty trajectory *** //
int const MAX_AREA = 500;
int const ROI_HEIGHT = 50;

/* Calibration parameters used to undistort points */
//Matx33f const camera_mat( 233.7251, 0,         323.25797,\
//                          0,        233.01072, 238.90632,\
//                          0,        0,          1 );
//Matx14f const dist_coeff( -.14484,0.01317,-0.00036,0.00091);

Matx33f const camera_mat( 232.90480, 0,        321.92218,\
                          0,        234.09177, 225.46547,\
                          0,        0,          1 );
Matx14f const dist_coeff( -0.20724,0.02757,0.00007,-0.00041 );

Matx33f new_camera_mat(0,0,0,\
                       0,0,0,\
                       0,0,0);


int keyboard, frameID;

bool bgSet; // flag indicates background averaging

Mat model0, frame_rectified, fgMaskMOG2, result,foreground; // binary mask containing foreground
Ptr<BackgroundSubtractor> pMOG2;
// finding contours
vector< vector<Point> > contours,treeCentroids;
vector< Vec4i > heirarchy;
RNG range(12345); // used to calc contours
Point centroid; // obj center
ofstream data_out;

// function prototypes
void getBGModel(char* video_file);
void draw(Point);
void drawForeground();
Point retrieveAvg(vector<int>,int);
Point2f findObjectCentroid();
bool isNan(float);
void reportCentroid();
void rectifySrc(Mat*,Mat*);
void segObjects();
void writeToVideo(VideoWriter* outputVideo,bool output_result);
void framePercentProgress(VideoCapture* cap);
void processVideo(char* video_file);


int main(int argc, char* argv[])
{
  pMOG2 = new BackgroundSubtractorMOG2(10,16,false); //MOG2 approach


  findObjectCentroid();
  //check for the input parameter correctness
  if(argc != 3)
  {
    cerr <<"Incorret input list" << endl;
    cerr <<"exiting..." << endl;
    return EXIT_FAILURE;
  }
  //create data file
  data_out.open(argv[2]);
  //initialize background
  getBGModel(argv[1]);
//  bgSet=true;
  //input data coming from a video
  if(bgSet){ processVideo(argv[1]); }

  //destroy GUI windows
  destroyAllWindows();
  data_out.close();
  return EXIT_SUCCESS;
}

// draws a rectangle or cross centered at a point
void draw(Point pt)
{
  Point a(pt.x+10,pt.y), b(pt.x-10,pt.y), c(pt.x,pt.y+10), d(pt.x,pt.y-10);
  line( result,a,b,Scalar(255,0,0),2 ); line( result,c,d,Scalar(255,0,0),2 );
}

// draws all foreground objects
void drawForeground()
{
  Mat rgb[3],mask1,mask2;
  vector<Mat> components;
  // get rgb components
  split(frame_rectified,rgb);
  bitwise_xor(fgMaskMOG2,rgb[1],mask2);
  bitwise_xor(fgMaskMOG2,rgb[2],mask1);
  components.push_back(rgb[0]);components.push_back(mask2);components.push_back(mask1);
  merge(components,foreground);
  result = frame_rectified;

}

// returns average position between all segments that exceed the threshold
Point retrieveAvg(vector<int> a,int asize)
{ // a contains INDECES of contours
  Point avg; //avg.x=0; avg.y=0;
  Moments mu;

// *** cleaned up ***
  // calculate centroids of each contour
  for(int i=0; i<asize; i++)
  {
    // get central moments
    mu = moments( contours[a[i]], false ); // true --> binary image
    // calc mass center based on spacial moments x=m10/m00, y=m01/m00;
    Point ci = Point2f( mu.m10/mu.m00, mu.m01/mu.m00 );
    // skip averaging if less than two candidates
    if(asize == 1){ return ci; }
//    return ci;
    avg.x += ci.x; avg.y += ci.y;
  }
  avg.x /= asize; avg.y /= asize;
  return avg;
}

//// returns position of largest contour and checks calculated pos against prev
//Point findObjectCentroid()
//{
//  Point centroid;
//  // vector: threshed areas
//  // heap: keep areas
//  // rectange: roi
//  // bool: search_flag=false
//  // int: largestArea,height


//  // height=INIT_HEIGHT

//  // if area larger than thresh:
//  //   copy to vector and count

//  // make_heap using vector

//  // largestArea=pop_heap
//  // calc centroid for contour at index of largestArea

//  // if history is null:
//  //   return centroid
//  // else:
//  //   search_flag=true
//  //   create roi centered at history (call createROI)

//  // while search:
//  //   for ii in size vector:
//  //      if centroid in roi: return centroid
//  //      largestArea=pop_heap
//  //      calc centroid for contour at index of largestArea
//  //
//  //   // didn't find contour in roi
//  //   expandROI by 2*height
//  //   make_heap using vector


//  // update
//  // history=centroid

//  return centroid;
//}

Point2f history=Point2f(LIMITS::quiet_NaN(),LIMITS::quiet_NaN());

Point2f findObjectCentroid()
{
  vector< double > thresh_areas(contours.size());
  int area_to_contour[contours.size()];
  double largest_area[5];
  Point2f cc; Rect roi; Moments mu;
  int area_count=0,height=ROI_HEIGHT;
  bool found=false;

  // calaculate and threshold areas
  for(int ii=0; ii<contours.size(); ii++)
  {
    double area=contourArea(contours[ii]);
    if(MIN_AREA<area && area<MAX_AREA)
    {
      thresh_areas.push_back(area); area_count++;
      area_to_contour[ii]=area; // keep track of which contour the area belongs to
    }
    area_to_contour[ii]=-1; // mark areas outside threshold bounds
  }
  if(area_count==0){ return Point2f(LIMITS::quiet_NaN(),LIMITS::quiet_NaN()); } // no detectable obj
//    mu = moments( contours[], false ); // true --> binary image
//    // calc mass center based on spacial moments x=m10/m00, y=m01/m00;
//    Point ci = Point2f( mu.m10/mu.m00, mu.m01/mu.m00 );



  // create max heap of all areas and collect top ten
  make_heap(thresh_areas.begin(),thresh_areas.end());
  for(int ii=0; ii<5; ii++)
  {
    if(ii<=area_count){
      largest_area[ii]=thresh_areas.front();
      pop_heap(thresh_areas.begin(),thresh_areas.end());
      cout<<"area "<<ii<<"="<<largest_area[ii]<<endl;
    }
    else{
      largest_area[ii]=0.f;
      cout<<"area "<<ii<<"="<<largest_area[ii]<<endl;
    }
  }

//  while(!found)
//  {



//  }

  return cc;
}

// checks if val is nan
bool isNan(float d){  return d != d; }
// createROI(int height,Point center)
// expandROI(int newHeight)

// calculates and displays centroid of objects with size > threshold value
void reportCentroid()
{
  vector< double > areas(contours.size());
  vector< int > icandidates(contours.size()); // INDEX of candidate contours
  //Point centroid;
  int n_cands=0; // for icandidates

  //printf( "FrameID\tLocation\n" );			//INFO//
  for(int i=0; i<contours.size(); i++)
  {
    // calculate area
    areas[i] = contourArea(contours[i]);
    // for multiple centroids passing threshold, take only the largest centroid (i.e. ignore reflections)
    if( areas[i] > MIN_AREA && areas[i] < MAX_AREA){ icandidates[n_cands++] = i; }
  }
  if( n_cands > 0 )
  {

    //draw these n contours 			//INFO//
    centroid = retrieveAvg(icandidates,n_cands);
    // Write position to file
    data_out << lexical_cast<string>(frameID) +","+ \
                lexical_cast<string>(centroid.x) +","+ \
                lexical_cast<string>(centroid.y) << \
                '\n';
  }
  else
  { // Display only last known centriod
    // Write nan to file b/c no detectable object
    //ROS_INFO("Object not found: ignore or reduce threshold");

#ifdef NAN
    data_out << lexical_cast<string>(frameID) +","+ \
                lexical_cast<string>(LIMITS::quiet_NaN()) +","+ \
                lexical_cast<string>(LIMITS::quiet_NaN()) << \
                '\n';
#endif
  }

// non-averaging (get max)
//  centroid=findObjectCentroid();
//  // Write position to file
//  data_out << lexical_cast<string>(frameID) +","+ \
//              lexical_cast<string>(centroid.x) +","+ \
//              lexical_cast<string>(centroid.y) << \
//              '\n';


  draw(centroid);
  //printf( "[%d]\t\t(%d,%d)\n", frameID, centroid.x, centroid.y );  //INFO//
}


// Averages all frames within video file into one image which represents the initial background model
// NOTE: the frames are read in as unsigned char type, then converted to floating point type to compute
// the average intensity.
void getBGModel(char* videoFilename)
{
  Mat src,rectified_src;
  VideoCapture capture(videoFilename);
  // check if file can be read
  if(!capture.isOpened())
  {
    cerr << "Unable to open video file: " << videoFilename << endl;
    exit(EXIT_FAILURE);
  }
  cout << "Frame count: " << capture.get(CV_CAP_PROP_FRAME_COUNT) << endl;   //INFO//

  cout << "Obtaining initial background...\n"; //INFO//
  // read first frame
  if(!capture.read(src))
  {
    cerr <<"Unable to read next frame.\nExiting..." << endl;
    exit(EXIT_FAILURE);
  }
  // retreive new camera matrix to get uncropped rectified image
  new_camera_mat = getOptimalNewCameraMatrix(camera_mat,dist_coeff,src.size(),1);
  // undistort raw input
  rectifySrc(&src,&rectified_src);
  // initialize model0: still range [0,255]; for imshow, use src.convertTo(model0,CV_32FC3, 1.0/255)
  rectified_src.convertTo(model0,CV_32FC3);
  while( capture.get(CV_CAP_PROP_POS_FRAMES) < capture.get(CV_CAP_PROP_FRAME_COUNT)-2)
  {
    // read new frame as second source img
    if(!capture.read(src))
    {
      cerr <<"Unable to read next frame.\nExiting..." << endl;
      exit(EXIT_FAILURE);
    }
    // undistort raw input
    rectifySrc(&src,&rectified_src);
    rectified_src.convertTo(rectified_src,CV_32FC3);
    double a = 0.5; // new input gets less weight
    // apply simple linear blending operation
    addWeighted(model0,a,rectified_src,1.0-a,0.0,model0);

  }
  // convert back to uchar for bs operator
  model0.convertTo(model0,CV_8UC3);
//  imshow("frame",model0); keyboard = waitKey(0);
  // initialize the background model
  pMOG2->operator()(model0,fgMaskMOG2);
  bgSet = true;
  cout << "Initial background is set.\n"; //INFO//
}


void rectifySrc(Mat* src,Mat* rect)
{
  try
  { undistort(*src,*rect,camera_mat,dist_coeff,new_camera_mat); }
  catch(Exception)
  {
    cerr <<"Unable to undistort frame.\nExiting..." << endl;
    exit(EXIT_FAILURE);
  }
}


// finds objects in the foreground of a frame, then displays their location
void segObjects()
{
  Mat mask = fgMaskMOG2.clone(); // avoid altering fgmask --> make deep copy
  findContours( mask, contours, heirarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0) );
  drawForeground();
  reportCentroid();
}


void writeToVideo(VideoWriter* outputVideo,bool output_tracking_result)
{
  if(!outputVideo->isOpened())
  {
    //error in opening the video output
    cerr <<"Unable to open output video for writing.\nExiting..." << endl;
    exit(EXIT_FAILURE);
  }
  if(output_tracking_result)
    outputVideo->write(result);
  else
    outputVideo->write(foreground);
}

void framePercentProgress(VideoCapture* cap)
{
  // get current frame position
  frameID = cap->get(CV_CAP_PROP_POS_FRAMES);
  double p = 100*(frameID/cap->get(CV_CAP_PROP_FRAME_COUNT));

  // write frame number out of frame count on the current frame
  stringstream ss;
  rectangle(frame_rectified, cv::Point(10, 2), cv::Point(100,20),
            cv::Scalar(255,255,255), -1);
  ss << (int)(p-fmod(p,1));    // percent progress floored in the ones place
  string percentProgress = ss.str()+"%";
  putText(frame_rectified, percentProgress.c_str(), cv::Point(15, 15),
          FONT_HERSHEY_SIMPLEX, 0.5 , cv::Scalar(0,0,0));
  //cout << percentProgress << endl;

}

// captures frames from a video file, then detects obejects in the foreground and displays them
void processVideo(char* videoFilename)
{
  VideoCapture capture(videoFilename);
  VideoWriter highlighted_fg_video;
  VideoWriter tracking_result_video;
  Mat frame;
  // open video file
  if(!capture.isOpened())
  {
    //error in opening the video input
    cerr <<"Unable to open input video.\nExiting..." << endl;
    exit(EXIT_FAILURE);
  }
  // prepare output video file
  int codecType   = static_cast<int>( capture.get(CV_CAP_PROP_FOURCC) ); // make output video have same codec type as input
  Size frameSize  = Size( (int)capture.get(CV_CAP_PROP_FRAME_WIDTH),(int)capture.get(CV_CAP_PROP_FRAME_HEIGHT) );
  string hvideo_name = "highlighted_fg.avi", tvideo_name = "tracking_result.avi";
  highlighted_fg_video.open(hvideo_name ,codecType,capture.get(CV_CAP_PROP_FPS),frameSize,true);
  tracking_result_video.open(tvideo_name,codecType,capture.get(CV_CAP_PROP_FPS),frameSize,true);

  // capture and process frames
  cout << "Processing video... (enter ESC or 'q' to quit)\n";                //INFO//
  cout << "Recording video: " << hvideo_name << " and " << tvideo_name << endl;
  cout << "Object threshold: " << lexical_cast<string>(MIN_AREA) << "pel < obj_area < " << lexical_cast<string>(MAX_AREA) << "pel\n";

  frameID = capture.get(CV_CAP_PROP_POS_FRAMES); // used to indicate progress in video process
  while( frameID < capture.get(CV_CAP_PROP_FRAME_COUNT)-2 && ((char)keyboard != 'q' && (char)keyboard != 27) )
  {
    if(!capture.read(frame))
    {
      cerr <<"Unable to read next frame.\nExiting..." << endl;
      exit(EXIT_FAILURE);
    }

    // retreive new camera matrix to get uncropped rectified frame
    new_camera_mat = getOptimalNewCameraMatrix(camera_mat,dist_coeff,frame.size(),1);
    // undistort raw input
    rectifySrc(&frame,&frame_rectified);

  //write progress
    framePercentProgress(&capture);

    // get foreground mask and update the background model
    pMOG2->operator()(frame_rectified,fgMaskMOG2);
    // segment objects larger than maximum threshold (ignore noise)
    segObjects();
//    // add result and hi-fg frames to video output
    writeToVideo(&highlighted_fg_video,false);
    writeToVideo(&tracking_result_video,true);

    // display result
    try                                              // INFO //
    {                                                //
      imshow("Highlighted Foreground", foreground);  //
      imshow("Result", result);                      //
    }                                                //
    catch(Exception &e)                              //
    {                                                //
      cerr <<"failed to display result" << endl;     //
      exit(EXIT_FAILURE);                            //
    }                                                //
    // quit upon user input                          // INFO //
    keyboard = waitKey(1);//(int)1000.0/capture.get(CV_CAP_PROP_FPS) ); // delay in millisec
  }
  // delete capture object
  capture.release();
  cout << "Fin!\n";  //INFO//

}
