#include </usr/local/include/opencv2/opencv.hpp>

using namespace cv;
using namespace std;

Mat frame;
int main(int argc, char* argv[])
{
  Point a(10,10), b(10,20);
  VideoCapture capture(argv[1]);
  if(!capture.isOpened()){
    exit(EXIT_FAILURE);
  }
  while( !capture.read(frame))
  {
    line( frame,a,b,Scalar(255,0,0),2 );
  }

  return EXIT_SUCCESS;
}
