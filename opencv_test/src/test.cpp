#include "ros/ros.h"
#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char** argv)
{
	ros::init(argc, argv, "test_node");
	ros::NodeHandle nh;

	std::cout << CV_VERSION << std::endl;

	return 0;
}
