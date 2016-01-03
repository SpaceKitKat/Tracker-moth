/**
 * Author: Bilkit Githinji
 * Version: 3.0
 * Description: This program tracks foreground objects within a video file.
 * It performs background subtraction based on Gaussian Mixture Model (GMM)
 * with shadow detection.
 * Command: ./moth_tracker <video_file> <data_file>
 *
 * Note: octavio's camera is recording at 30fps
 **/

/***
 * Todo
 *   only dedistort points when not visualizing
 *   break file up into headers and at source files
***/

//NOTES:
// For background update, use pMOG2->operator()(frame,fgMaskMOG2) and initialize with
// average of all frames. pMOG2->operator()(frame,fgMaskMOG2, learningRate) is another
// option, but adjusting learning rate causes relatively stationary objects to blend
// in with the background model. This is problematic for tracking when moth is visiting.

//-- testing something --//

//opencv
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
//c++
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstdio> //timing test
#include <ctime>
#include <boost/lexical_cast.hpp> //  string concatination with doubles
#include <boost/program_options.hpp> // command line options: edit CMakeLists.txt to include boost libs
#include <math.h>
#include <limits.h> // access quiet NAN
#include <vector>


#define _USE_MATH_DEFINES

using namespace cv;
using namespace std;
using namespace boost;

typedef numeric_limits<float> FLT;
typedef numeric_limits<double> DBL;
namespace op=program_options;

int const DEPTH = CV_16S;// 16short is used to prevent overflow during gradient cal
int const SCALE = 1,DELTA = 0,RADIUS = 5;
int const MIN_AREA = 50;  // *** increasing this value --> more spotty trajectory *** //
int const MAX_AREA = 1000; // decreasing this will interfere with distorted video tracking
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

// Background subtractor object
Ptr<BackgroundSubtractor> pMOG2;

Mat model0 // empty background
, frame // raw frame
, frame_rectified // dedistorted frame
, fgMaskMOG2 // binary foreground mask
, result // frame with marker on object centroid
, foreground; // frame with highlighted foreground mask

// objects for finding contours
vector< vector<Point> > contours,treeCentroids;
vector< Vec4i > heirarchy;
RNG range(12345); // used to calc contours
Point centroid; // obj center

// output video stream
ofstream data_out;

// flags for command line options
int display;

// frame identifier used to report data
// and progress
int frameID;

// main processing
bool getBGModel(char* video_file);
void processVideo(char* video_file);
void segObjects();
void reportCentroid();
Point retrieveAvg(vector<int>,int);

// helper funcitons
bool isNan(float);
void rectifySrc(Mat*,Mat*);
void writeToVideo(VideoWriter* outputVideo,bool output_result);

// visualization
void draw(Point);
void drawForeground();
void displayPercentProgress(VideoCapture* cap,int);

int main(int argc, char* argv[])
{
  clock_t start;
  pMOG2 = new BackgroundSubtractorMOG2(10,16,false); //MOG2 approach

  //check for the input parameter correctness
  if(argc > 7)
  {
    cerr <<"ERROR: see usage\n";
    cerr <<"USAGE: ./moth_tracker <path_to_video> <output_txt_file> [options][-d 1_or_0]\n";
    cerr <<"exiting..." << endl;
    return EXIT_FAILURE;
  }

  // process commandline options
  // sets visualization and dedistortion flags
  op::options_description desc("Program options specified in command line");
  desc.add_options()
    ("dedistort,d",op::value<int>(& display)->default_value(0),"Record dedistorted video output option")
  ;
  op::variables_map var_map;
  op::store(op::parse_command_line(argc,argv,desc),var_map);
  op::notify(var_map);

  // initialize background and process input video
  data_out.open(argv[2]);
  start = clock();

  if( getBGModel(argv[1]) ){ processVideo(argv[1]); }
  cout << "Total processing time: " << fixed << (double)(clock() - start)/CLOCKS_PER_SEC << "s\n";

  // cleanup
  destroyAllWindows();
  data_out.close();
  cout << "Fin!\n";
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
  if(display ){ split(frame_rectified,rgb);result=frame_rectified; }
  else { split(frame,rgb);result=frame; }
  bitwise_xor(fgMaskMOG2,rgb[1],mask2);
  bitwise_xor(fgMaskMOG2,rgb[2],mask1);
  components.push_back(rgb[0]);components.push_back(mask2);components.push_back(mask1);
  merge(components,foreground);


}

// returns average position between all segments that exceed the threshold
Point retrieveAvg(vector<int> a,int asize)
{ // a contains INDECES of contours
  Point avg; //avg.x=0; avg.y=0;
  Moments mmts;

  // calculate centroids of each contour
  for(int i=0; i<asize; i++)
  {
    // get moments
    mmts = moments( contours[a[i]], false ); // true --> binary image
    // calc mass center based on spacial moments x=m10/m00, y=m01/m00;
    Point ci = Point2f( mmts.m10/mmts.m00, mmts.m01/mmts.m00 );
    // skip averaging if less than two candidates
    if(asize == 1){ return ci; }
    avg.x += ci.x; avg.y += ci.y;
  }
  avg.x /= asize; avg.y /= asize;
  return avg;
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

  //printf( "FrameID\tLocation\n" );      //INFO//
  for(int i=0; i<contours.size(); i++)
  {
    // calculate area
    areas[i] = contourArea(contours[i]);
    // collect multiple contours passing threshold
    if( areas[i] > MIN_AREA && areas[i] < MAX_AREA){ icandidates[n_cands++] = i; }
  }
  if( n_cands > 0 )
  {

    centroid = retrieveAvg(icandidates,n_cands);
    // remove distortion
    //--(!) TODO

    // Write position to file
    data_out << lexical_cast<string>(frameID) +","+ \
                lexical_cast<string>(centroid.x) +","+ \
                lexical_cast<string>(centroid.y) << \
                '\n';
  }
  else
  { // Display only last known centriod
    // Write nan to file b/c no detectable object
#ifdef NAN
    data_out << lexical_cast<string>(frameID) +","+ \
                lexical_cast<string>(FLT::quiet_NaN()) +","+ \
                lexical_cast<string>(FLT::quiet_NaN()) << \
                '\n';
#endif
  }

//  printf( "[%d]\t\t(%d,%d)\n", frameID, centroid.x, centroid.y );  //INFO//
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
  // move to processVideo
  reportCentroid();
  draw(centroid);
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

// This function displays percent progress onto either the frame or the terminal. Percent progress
// is determined by the number of frames processed over total frame count.
void displayPercentProgress(VideoCapture* cap, int lc)
{
  // get current frame position
  bool can_print=(lc%( (int)round(cap->get(CV_CAP_PROP_FRAME_COUNT)/100.0) )==0);
  frameID = cap->get(CV_CAP_PROP_POS_FRAMES);
  double p = 100*(frameID/cap->get(CV_CAP_PROP_FRAME_COUNT));

  // write frame number out of frame count on the current frame
  stringstream ss;
  rectangle(result, cv::Point(10, 2), cv::Point(100,20),
            cv::Scalar(255,255,255), -1);
//  ss << (int)(p-fmod(p,1));    // percent progress floored in the ones place
  ss << (int)round(p);    // percent progress rounded
  string percentProgress = ss.str()+"%";
  putText(result, percentProgress.c_str(), cv::Point(15, 15),
          FONT_HERSHEY_SIMPLEX, 0.5 , cv::Scalar(0,0,0));
  if(!display && can_print){ cout << percentProgress << endl; }

}

// Averages all frames within video file into one image which represents the initial background model.
// The initial background model represents an empty frame which is passed to background subtractor.
// NOTE: the frames are read in as unsigned char type, then converted to floating point type to compute
// the average intensity.
bool getBGModel(char* videoFilename)
{
  Mat src,rectified_src;double beta = 1.0; // new input gets less weight
  int nFrames;
  clock_t start;
  double duration;
  VideoCapture capture(videoFilename);

  // check if file can be read
  if(!capture.isOpened())
  {
    cerr << "Unable to open video file: " << videoFilename << endl;
    exit(EXIT_FAILURE);
  }
  cout << "Frame count: " << capture.get(CV_CAP_PROP_FRAME_COUNT) << endl;   //INFO//

  cout << "Obtaining initial background...\n"; //INFO//

  // measure time to average all video frames
  start = clock();

  //--(?) why process first frame separately?
  // read first frame
  if(!capture.read(src))
  {
    cerr <<"Unable to read next frame.\nExiting..." << endl;
    exit(EXIT_FAILURE);
  }
  // preprocess frame
  if(display )
  {
    // retreive new camera matrix to get uncropped rectified image
    new_camera_mat = getOptimalNewCameraMatrix(camera_mat,dist_coeff,src.size(),1);
    // undistort raw input
    rectifySrc(&src,&rectified_src);
    // initialize model0: still range [0,255]; for imshow, use src.convertTo(model0,CV_32FC3, 1.0/255)
    rectified_src.convertTo(model0,CV_32FC3);
  }

  // initialize model0: still range [0,255]; for imshow, use src.convertTo(model0,CV_32FC3, 1.0/255)
  src.convertTo(model0,CV_32FC3);

  // average all frames
  while( capture.get(CV_CAP_PROP_POS_FRAMES) < capture.get(CV_CAP_PROP_FRAME_COUNT)-2)
  {
    // read new frame as second source img
    if(!capture.read(src))
    {
      cerr <<"Unable to read next frame.\nExiting..." << endl;
      exit(EXIT_FAILURE);
    }
    // preprocess frame
    if(display )
    {
      // undistort raw input
      rectifySrc(&src,&rectified_src);
      rectified_src.convertTo(src,CV_32FC3);
    }
    nFrames = capture.get(CV_CAP_PROP_POS_FRAMES); // track number of frames read (range from [1:frameCount])
    src.convertTo(src,CV_32FC3);

    // weighting: alpha = 1-(1/N), beta=(1/N) N=frame number, alpha+beta = 1
    beta = 1.0/nFrames;
    // apply simple linear blending operation
    addWeighted(model0,1.0-beta,src,beta,0.0,model0);
  }

  duration = (double)(clock()-start)/CLOCKS_PER_SEC;

  // convert back to uchar for bs operator
  model0.convertTo(model0,CV_8UC3);
  // initialize the background model
  pMOG2->operator()(model0,fgMaskMOG2);
  cout << "Initial background is set.\n"; //INFO//
  cout << "Total averaging time: " << fixed << duration << "s\n";

  return true;

}

// captures frames from a video file, then detects obejects in the foreground and displays them; uses distorted frames
void processVideo(char* videoFilename)
{
  int keyboard;
  VideoCapture capture(videoFilename);
  VideoWriter highlighted_fg_video;
  VideoWriter tracking_result_video;
  int loopcount=0;

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
  cout << "Processing distorted video...\n";              //INFO//
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

    // preprocess frame
    if(display )
    {
      // retreive new camera matrix to get uncropped rectified frame
      new_camera_mat = getOptimalNewCameraMatrix(camera_mat,dist_coeff,frame.size(),1);
      // undistort raw input
      rectifySrc(&frame,&frame_rectified);
      frame=frame_rectified.clone(); // make sure to deep copy
    }


    // get foreground mask and update the background model
    pMOG2->operator()(frame,fgMaskMOG2);

    // segment objects larger than maximum threshold (ignore noise)
    segObjects();
    // report and draw point


    //write progress
    displayPercentProgress(&capture,++loopcount);

    // add result and hi-fg frames to video output
    writeToVideo(&highlighted_fg_video,false);
    writeToVideo(&tracking_result_video,true);

    if(display)
    {
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
      // quit upon user input                          //
  //    keyboard = waitKey( 10 );                        // INFO //
      keyboard = waitKey(1);// (int)1000.0/capture.get(CV_CAP_PROP_FPS) ); // delay in millisec
    }
  }
  // delete capture object
  capture.release();
}
