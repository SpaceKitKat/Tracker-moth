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
// For background update, use pMOG2->operator()(frame,fgMaskMOG2) and initialize with
// average of all frames. pMOG2->operator()(frame,fgMaskMOG2, learningRate) is another
// option, but adjusting learning rate causes relatively stationary objects to blend
// in with the background model. This is problematic for tracking when moth is visiting.

//opencv
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//c++
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <boost/lexical_cast.hpp> //  string concatination with doubles
#include <math.h>
#include <limits.h> // access quiet NAN


#define _USE_MATH_DEFINES

using namespace cv;
using namespace std;
using namespace boost;

int const DEPTH = CV_16S;// 16short is used to prevent overflow during gradient cal
int SCALE = 1,DELTA = 0;
int RADIUS = 5;
int MIN_AREA = 100;  // *** increasing this value --> more spotty trajectory *** //
int keyboard, frameID;

bool bgSet; // flag indicates background averaging

Mat model0, frame, fgMaskMOG2, result,foreground; // binary mask containing foreground
Ptr<BackgroundSubtractor> pMOG2;
// finding contours
vector< vector<Point> > contours,treeCentroids;
vector< Vec4i > heirarchy;
RNG range(12345); // used to calc contours
Point centroid; // obj center
ofstream data_out;

// function prototypes
void getBGModel(char* video_file);
void processVideo(char* video_file);
void draw(Point);
void drawForeground();
Point retrieveAvg(vector<int>,int);
void getCentroid();
void segObjects();
void writeToVideo(VideoWriter* outputVideo,bool output_result);

int main(int argc, char* argv[])
{
  pMOG2 = new BackgroundSubtractorMOG2(10,16,false); //MOG2 approach
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
  split(frame,rgb);
  bitwise_xor(fgMaskMOG2,rgb[1],mask2);
  bitwise_xor(fgMaskMOG2,rgb[2],mask1);
  components.push_back(rgb[0]);components.push_back(mask2);components.push_back(mask1);
  merge(components,foreground);
  result = frame;

}

// returns average position between all segments that exceed the threshold
Point retrieveAvg(vector<int> a,int asize)
{ // a contains INDECES of contours
  vector< Moments > mu(asize); // central moments
  vector< Point2f > centroids(asize); // centers of mass
  Point avg; //avg.x=0; avg.y=0;

  // calculate centroids of each contour
  for(int i=0; i<asize; i++)
  {
    // get central moments
    mu[i] = moments( contours[a[i]], false ); // true --> binary image
    // calc mass center based on spacial moments x=m10/m00, y=m01/m00;
    centroids[i] = Point2f( mu[i].m10/mu[i].m00, mu[i].m01/mu[i].m00 );
  }
  // skip averaging if less than two candidates
  if(asize == 1){ return centroids[0]; }

  // otherwise, calc the average centroid
  for(int i=0; i < asize; i++)
  {
      Point ci = centroids[i];
      avg.x += ci.x; avg.y += ci.y;
  }
  avg.x /= centroids.size(); avg.y /= centroids.size();
  return avg;
}

// calculates and displays centroid of objects with size > threshold value
void getCentroid()
{
  vector< double > areas(contours.size());
  vector< int > icandidates(contours.size()); // INDEX of candidate contours
  //Point centroid;
  int j=0; // for icandidates

  //printf( "FrameID\tLocation\n" );			//INFO//
  for(int i=0; i<contours.size(); i++)
  {
    // calculate area
    areas[i] = contourArea(contours[i]);
    // for multiple centroids passing threshold, take only the largest centroid (i.e. ignore reflections)
    if( areas[i] > MIN_AREA ){ icandidates[j++] = i; }
  }
  if( j > 0 )
  {
    centroid = retrieveAvg(icandidates,j);
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
                lexical_cast<string>(numeric_limits<double>::quiet_NaN()) +","+ \
                lexical_cast<string>(numeric_limits<double>::quiet_NaN()) << \
                '\n';
#endif
  }
  draw(centroid);
  //printf( "[%d]\t\t(%d,%d)\n", frameID, centroid.x, centroid.y );  //INFO//
}

// Averages all frames within video file into one image which represents the initial background model
// NOTE: the frames are read in as unsigned char type, then converted to floating point type to compute
// the average intensity.
void getBGModel(char* videoFilename)
{
  Mat src1;
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
  if(!capture.read(src1))
  {
    cerr <<"Unable to read next frame.\nExiting..." << endl;
    exit(EXIT_FAILURE);
  }
  // initialize model0: still range [0,255]; for imshow, use src1.convertTo(model0,CV_32FC3, 1.0/255)
  src1.convertTo(model0,CV_32FC3);
  while( capture.get(CV_CAP_PROP_POS_FRAMES) < capture.get(CV_CAP_PROP_FRAME_COUNT)-2)
  {
    // read new frame as second source img
    if(!capture.read(src1))
    {
      cerr <<"Unable to read next frame.\nExiting..." << endl;
      exit(EXIT_FAILURE);
    }
    src1.convertTo(src1,CV_32FC3);
    double a = 0.5; // new input gets less weight
    // apply simple linear blending operation
    addWeighted(model0,a,src1,1.0-a,0.0,model0);

  }
  // convert back to uchar for bs operator
  model0.convertTo(model0,CV_8UC3);
  // initialize the background model
  pMOG2->operator()(model0,fgMaskMOG2);
  bgSet = true;
  cout << "Initial background is set.\n"; //INFO//
}

// finds objects in the foreground of a frame, then displays their location
void segObjects()
{
  Mat mask = fgMaskMOG2.clone();
  findContours( mask, contours, heirarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0) );
  drawForeground();
  getCentroid();
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

// captures frames from a video file, then detects obejects in the foreground and displays them
void processVideo(char* videoFilename)
{
  VideoCapture capture(videoFilename);
  VideoWriter highlighted_fg_video;
  VideoWriter tracking_result_video;

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
  highlighted_fg_video.open("highlighted_fg.avi",codecType,capture.get(CV_CAP_PROP_FPS),frameSize,true);
  tracking_result_video.open("tracking_result.avi",codecType,capture.get(CV_CAP_PROP_FPS),frameSize,true);

  // capture and process frames
  cout << "Processing video... (enter ESC or 'q' to quit)\n";                //INFO//
  frameID = capture.get(CV_CAP_PROP_POS_FRAMES); // used to indicate progress in video process
  while( frameID < capture.get(CV_CAP_PROP_FRAME_COUNT)-2 && ((char)keyboard != 'q' && (char)keyboard != 27) )
  {
    if(!capture.read(frame))
    {
      cerr <<"Unable to read next frame.\nExiting..." << endl;
      exit(EXIT_FAILURE);
    }
    frameID = capture.get(CV_CAP_PROP_POS_FRAMES);
    double p = 100*(frameID/capture.get(CV_CAP_PROP_FRAME_COUNT));
    // get foreground mask and update the background model
    pMOG2->operator()(frame,fgMaskMOG2);
    // write frame number on the current frame
    stringstream ss;
    rectangle(frame, cv::Point(10, 2), cv::Point(100,20),
              cv::Scalar(255,255,255), -1);
    ss << (int)(p-fmod(p,1));    // percent progress floored in the ones place
    string percentProgress = ss.str()+"%";
    putText(frame, percentProgress.c_str(), cv::Point(15, 15),
            FONT_HERSHEY_SIMPLEX, 0.5 , cv::Scalar(0,0,0));
    //cout << percentProgress << endl;
    // segment objects larger than maximum threshold (ignore noise)
    segObjects();
    // add result and hi-fg frames to video output
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
    // quit upon user input                          //
//    keyboard = waitKey( 10 );                        // INFO //
    keyboard = waitKey( (int)1000.0/capture.get(CV_CAP_PROP_FPS) ); // delay in millisec
  }
  // delete capture object
  capture.release();
  cout << "Fin!\n";  //INFO//

}
