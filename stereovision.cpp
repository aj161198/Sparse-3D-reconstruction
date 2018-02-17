#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
# include "opencv2/features2d/features2d.hpp"
# include "opencv2/nonfree/features2d.hpp"
#include <stdio.h>
#include <iostream>
using namespace std;
using namespace cv;


float distancePointLine( const Point point, const Vec<float,3>& line)
{
  //Line is given as a*x + b*y + c = 0
  return fabsf(line(0)*point.x + line(1)*point.y + line(2))
      / std::sqrt(line(0)*line(0)+line(1)*line(1));
}

int main( )
{
	Mat img_1 = imread( "7.jpeg", CV_LOAD_IMAGE_GRAYSCALE );
	Mat img_2 = imread( "8.jpeg", CV_LOAD_IMAGE_GRAYSCALE );
	
    	//-- Step 1: Detect the keypoints using SURF Detector
    	int minHessian = 400;

    	SurfFeatureDetector detector( minHessian );

    	std::vector<KeyPoint> keypoints_1, keypoints_2;

    	detector.detect( img_1, keypoints_1 );
    	detector.detect( img_2, keypoints_2 );

    	//-- Step 2: Calculate descriptors (feature vectors)
    	SurfDescriptorExtractor extractor;

    	Mat descriptors_1, descriptors_2;

      	extractor.compute( img_1, keypoints_1, descriptors_1 );
      	extractor.compute( img_2, keypoints_2, descriptors_2 );

   	//-- Step 3: Matching descriptor vectors using FLANN matcher
    	BFMatcher matcher;
    	std::vector< DMatch > matches;
    	matcher.match( descriptors_1, descriptors_2, matches );

    	double max_dist = 0; double min_dist = 100;
    
    	//-- Quick calculation of max and min distances between keypoints
    	for( int i = 0; i < descriptors_1.rows; i++ )
    	{ 
		double dist = matches[i].distance;
      		if( dist < min_dist ) min_dist = dist;
      		if( dist > max_dist ) max_dist = dist;
    	}
    	//-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
    	//-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
    	//-- small)
    
    	std::vector< DMatch > good_matches;
    	vector<Point2f> points1,points2;
    
    	for( int i = 0; i < descriptors_1.rows; i++ )
    	{ 
		if( matches[i].distance <= max(2*min_dist, 0.02) )
      		{ 
      			good_matches.push_back( matches[i]);
      			points2.push_back( keypoints_2[matches[i].trainIdx].pt );
      			points1.push_back( keypoints_1[matches[i].queryIdx].pt ); 
      		}
    	}
	cout << good_matches.size() << endl;
  //***************** Finding Fundamental Matrix **************************

  	vector<uchar> states;
  
  	Mat F = findFundamentalMat(points1, points2, FM_LMEDS, 3, 0.99, states);
  	string window = "epipolarlines";
	cout << F << endl;

  	vector<Vec<float,3> > epilines1, epilines2;
  	computeCorrespondEpilines(points1, 1, F, epilines1); //Index starts with 1
  	computeCorrespondEpilines(points2, 2, F, epilines2);

  	cvtColor(img_1,img_1,COLOR_GRAY2BGR);
  	cvtColor(img_2,img_2,COLOR_GRAY2BGR);

  
  	vector <Point2f> im1_pts, im2_pts;
	int count = 0;
  	cv::RNG rng(0);
  	for(int i=0; i<points1.size(); i++)
  	{
		if (states[i] == 1)
		{
			count++;
			cv::Scalar color(rng(256),rng(256),rng(256));
     			line(img_1, Point(0,-epilines2[i][2]/epilines2[i][1]), Point(img_1.cols,-(epilines2[i][2]+epilines2[i][0]*img_1.cols)/epilines2[i][1]),color);
     			circle(img_1, points1[i], 3, color, -1, CV_AA);

     			line(img_2, Point(0,-epilines1[i][2]/epilines1[i][1]), Point(img_2.cols,-(epilines1[i][2]+epilines1[i][0]*img_2.cols)/epilines1[i][1]),color);
     			circle(img_2, points2[i], 3, color, -1, CV_AA);	
			im1_pts.push_back(points1[i]);
			im2_pts.push_back(points2[i]);
			
			
		}
  	} 
	
	vector <Point3f> h1, h2;
	convertPointsHomogeneous(im1_pts, h1);
	convertPointsHomogeneous(im2_pts, h2);
	
  	imshow("title", img_1);
  	imshow("title2", img_2);
	
  //***************** Finding Essential Matrix **************************
	
	Mat camera = (Mat_<double>(3, 3) <<  6.1027873034347340e+02, 0., 2.9759109094204882e+02, 0.,
       6.0873096284124119e+02, 2.5788588920518663e+02, 0., 0., 1.);
	double fx = camera.at<double>(0,0);
	double fy = camera.at<double>(1,1);
	double cx = camera.at<double>(0,2);
	double cy = camera.at<double>(1,2);
	
	
	Mat him1(im1_pts.size(), 3, CV_64FC1), him2(im1_pts.size(), 3, CV_64FC1);
	for (int i = 0; i < im1_pts.size(); i++)
	{
		him1.at<double>(i, 0) = h1[i].x;
		him1.at<double>(i, 1) = h1[i].y;
		him1.at<double>(i, 2) = h1[i].z;

		him2.at<double>(i, 0) = h2[i].x;
		him2.at<double>(i, 1) = h2[i].y;
		him2.at<double>(i, 2) = h2[i].z;
	}
	/*for (int i = 0; i < im1_pts.size(); i++)
	cout << him1(Range(i, i + 1), Range::all()) * F * (him2.t())( Range::all(), Range(i, i + 1)) << endl;*/

	Mat distCoeffs = (Mat_<double>(5, 1) << -1.1330772187420178e-01, 2.5613927657371685e+00,
       6.2256281395848944e-03, -4.8001634307816400e-03,
       -1.4475237512839797e+01 );	
	
	Mat temp, essential;
	gemm(camera.t(), F, 1, 0, 0, temp, 0);	
	gemm(temp, camera, 1, 0, 0, essential, 0);		// E = K^T . F . K
	
	Mat D = (Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 0);
	SVD svd_(essential);
	
	Mat E = svd_.u * D * -svd_.vt;
	
	SVD svd(E);
	cout << svd_.w << endl;

	Mat W = (Mat_<double>(3, 3) << 0, -1, 0, 1, 0, 0, 0, 0, 1);
	Mat Z = (Mat_<double>(3, 3) << 0, 1, 0, -1, 0, 0, 0, 0, 0);
	Mat u = svd.u;
	Mat vt = svd.vt;

	cout << determinant(u * vt) << endl;
	Mat r1, r2, T1, T2, s1, s2;
	
	s1 = -u * Z * u.t();
	T1 = u(Range::all(), Range(2,3));
	
	s2 = s1;
	T2 = -T1;
	
	
	r1 = u * W.t() * vt;
	r2 = u * W * vt;
	
	Mat P1, P2, P3, P4; 
	hconcat(r1, T1, P1);
	hconcat(r1, -T1, P2);
	hconcat(r2, T2, P3);
	hconcat(r2, -T2, P4);
	
	cout << r1 << endl;
	
	Mat e1 = camera.inv() * him1.t(), e2 = camera.inv() * him2.t();
	/*for (int i = 0; i < im1_pts.size(); i++)
		cout << e1(Range::all(), Range(i, i + 1)).t()  * E *  e2( Range::all(), Range(i, i + 1)) << endl;*/	
	Mat P = (Mat_<double>(3, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);
	
//********************************************************************************************************************
	
	Mat points4d;
	
	vector<Point2f> upts1, upts2;
	undistortPoints(im1_pts, upts1, camera, distCoeffs);
	undistortPoints(im2_pts, upts2, camera, distCoeffs);
	
	
	triangulatePoints(P, P2, upts1, upts2, points4d);
	
	
	Mat points3d(im1_pts.size(), 3, CV_32F);
	for(int i = 0; i < im1_pts.size(); i++)
	{
		points3d.at<float>(i, 0) = points4d.at<float>(0, i) / points4d.at<float>(3, i); 
		points3d.at<float>(i, 1) = points4d.at<float>(1, i) / points4d.at<float>(3, i); 
		points3d.at<float>(i, 2) = points4d.at<float>(2, i) / points4d.at<float>(3, i); 
	}
	
	Mat rvec, imagePoints;
	Rodrigues(r1, rvec);
	
	projectPoints(points3d, rvec, -T1, camera, distCoeffs, imagePoints);
	for (int i = 0; i < im1_pts.size(); i++)
		cout << 1000 * points3d.at<float>(i, 0) << ", " << 1000 * points3d.at<float>(i, 1) << ", " << 1000 * points3d.at<float>(i, 2) << ")" << " -> " << im2_pts[i] << " -> " << "(" << imagePoints.at<float>(i, 0) << ", " << imagePoints.at<float>(i, 1) << ")" << endl;
	
	
  //***********************************************************************	*/
  if (waitKey(0) == 27);

  return 0;
}
