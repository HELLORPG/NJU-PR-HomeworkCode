// PREx3_6_6.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//
//

//
//#include <iostream>
//
//int main()
//{
//    std::cout << "hello world!\n";
//}
//
//// 运行程序: ctrl + f5 或调试 >“开始执行(不调试)”菜单
//// 调试程序: f5 或调试 >“开始调试”菜单
//
//// 入门使用技巧: 
////   1. 使用解决方案资源管理器窗口添加/管理文件
////   2. 使用团队资源管理器窗口连接到源代码管理
////   3. 使用输出窗口查看生成输出和其他消息
////   4. 使用错误列表窗口查看错误
////   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
////   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件


#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;

static Mat norm_0_255(InputArray _src) {
	Mat src = _src.getMat();
	// Create and return normalized image:
	Mat dst;
	switch (src.channels()) {
	case 1:
		cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
		break;
	case 3:
		cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
		break;
	default:
		src.copyTo(dst);
		break;
	}
	return dst;
}

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
	std::ifstream file(filename.c_str(), ifstream::in);
	if (!file) {
		string error_message = "No valid input file was given, please check the given filename.";
		CV_Error(CV_StsBadArg, error_message);
	}
	string line, path, classlabel;
	while (getline(file, line)) {
		stringstream liness(line);
		getline(liness, path, separator);
		getline(liness, classlabel);
		if (!path.empty() && !classlabel.empty()) {
			images.push_back(imread(path, 0));
			labels.push_back(atoi(classlabel.c_str()));
		}
	}
}

void read_ORLFaces(const string ORLpath, vector<Mat>& images, vector<int>& labels, vector<Mat>& test_image, vector<int>& test_labels) // 为了读取ORL数据集而专门设计的读取函数，用来替代原本的read_csv
{
	for (int i = 1; i <= 40; ++i) {
		std::string directory = ORLpath + "s" + std::to_string(i) + "/";

		for (int j = 1; j <= 10; ++j) 
		{
			std::string image_name = directory + std::to_string(j) + ".pgm";
			cout << image_name << endl;
			cv::Mat mat = cv::imread(image_name, 0);
			if (!mat.data) // 如果无法读取，则报错
			{
				fprintf(stderr, "read image fail: %s\n", image_name.c_str());
			}
			if (j == i % 10 + 1 || j == 11 - i % 10 - 1)
			{
				test_image.push_back(mat);
				test_labels.push_back(i);
			}
			else
			{
				images.push_back(mat);
				labels.push_back(i);
			}
		}
	}
	cout << "read all ORLFaces success!" << endl;
	return;
}

int Eigenface(string save_path)
{
	// Check for valid command line arguments, print usage
	// if no arguments were given.
	/*if (argc < 2) {
		cout << "usage: " << argv[0] << " <csv.ext> <output_folder> " << endl;
		exit(1);
	}
	string output_folder = ".";
	if (argc == 3) {
		output_folder = string(argv[2]);
	}*/
	string output_folder(save_path); // 仿照原代码创造保存的路径
	// These vectors hold the images and corresponding labels.
	vector<Mat> images, test_images;
	vector<int> labels, test_labels;
	
	read_ORLFaces("D:/OneDrive/CSMajor/2019to2020_2nd/PR/Homeworks/Ex3/database/ORL_Faces/", images, labels,test_images, test_labels);
	// Quit if there are not enough images for this demo.
	if (images.size() <= 1) {
		string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
		CV_Error(CV_StsError, error_message);
	}
	// Get the height from the first image. We'll need this
	// later in code to reshape the images to their original
	// size:
	int height = images[0].rows;
	// The following lines simply get the last images from
	// your dataset and remove it from the vector. This is
	// done, so that the training data (which we learn the
	// cv::FaceRecognizer on) and the test data we test
	// the model with, do not overlap.
	/*Mat testSample = images[images.size() - 1];
	int testLabel = labels[labels.size() - 1];
	images.pop_back();
	labels.pop_back();*/
	// The following lines create an Eigenfaces model for
	// face recognition and train it with the images and
	// labels read from the given CSV file.
	// This here is a full PCA, if you just want to keep
	// 10 principal components (read Eigenfaces), then call
	// the factory method like this:
	//
	//      cv::createEigenFaceRecognizer(10);
	//
	// If you want to create a FaceRecognizer with a
	// confidence threshold (e.g. 123.0), call it with:
	//
	//      cv::createEigenFaceRecognizer(10, 123.0);
	//

	int components_num = 300;
	Ptr<FaceRecognizer> model = createEigenFaceRecognizer();
	model->train(images, labels);
	for (int i(0); i < test_images.size(); ++i)
	{
		int predictedLabel = model->predict(test_images[i]);
		string result_message = format("Eigenfaces Predicted class = %d / Actual class = %d.", predictedLabel, test_labels[i]);	// 162
		cout << result_message << endl;
	}

	// Here is how to get the eigenvalues of this Eigenfaces model:
	Mat eigenvalues = model->getMat("eigenvalues");
	// And we can do the same to display the Eigenvectors (read Eigenfaces):
	Mat W = model->getMat("eigenvectors");
	// Get the sample mean from the training data
	Mat mean = model->getMat("mean");
	// Display or save:
	/*if (argc == 2) {
		imshow("mean", norm_0_255(mean.reshape(1, images[0].rows)));
	}
	else {*/
	cv::imwrite(format("%s/mean.png", output_folder.c_str()), norm_0_255(mean.reshape(1, images[0].rows)));
	//}
	// Display or save the Eigenfaces:
	for (int i = 0; i < min(10, W.cols); i++) 
	{
		string msg = format("Eigenvalue #%d = %.5f", i, eigenvalues.at<double>(i));
		cout << msg << endl;
		// get eigenvector #i
		Mat ev = W.col(i).clone();
		// Reshape to original size & normalize to [0...255] for imshow.
		Mat grayscale = norm_0_255(ev.reshape(1, height));
		// Show the image & apply a Jet colormap for better sensing.
		Mat cgrayscale;
		applyColorMap(grayscale, cgrayscale, COLORMAP_JET);
		cv::imwrite(format("%s/eigenface_%d.png", output_folder.c_str(), i), norm_0_255(cgrayscale));
		//}
	}

	// Display or save the image reconstruction at some predefined steps:
	for (int num_components = min(W.cols, 10); num_components < min(W.cols, 400); num_components += 10) 
	{
		// slice the eigenvectors from the model
		Mat evs = Mat(W, Range::all(), Range(0, num_components));
		Mat projection = subspaceProject(evs, mean, images[0].reshape(1, 1));
		Mat reconstruction = subspaceReconstruct(evs, mean, projection);
		// Normalize the result:
		reconstruction = norm_0_255(reconstruction.reshape(1, images[0].rows));
		// Display or save:
		/*if (argc == 2) {
			imshow(format("eigenface_reconstruction_%d", num_components), reconstruction);
		}
		else {*/
		cv::imwrite(format("%s/eigenface_reconstruction_%d.png", output_folder.c_str(), num_components), reconstruction);
		/*}*/
	}
	// Display if we are not writing to an output folder:
	/*if (argc == 2) {
		waitKey(0);
	}*/
	return 0;
}

int ORLFacestoImage()
{
	const std::string path{ "D:/OneDrive/CSMajor/2019to2020_2nd/PR/Homeworks/Ex3/database/ORL_Faces/" };
	cv::Mat dst;
	int height, width;

	for (int i = 1; i <= 40; ++i) {
		std::string directory = path + "s" + std::to_string(i) + "/";

		for (int j = 1; j <= 10; ++j) {
			std::string image_name = directory + std::to_string(j) + ".pgm";
			cv::Mat mat = cv::imread(image_name, 0);
			if (!mat.data) {
				fprintf(stderr, "read image fail: %s\n", image_name.c_str());
			}

			//std::string save_image_name = directory + std::to_string(j) + ".png";
			//cv::imwrite(save_image_name, mat);

			if (i == 1 && j == 1) {
				height = mat.rows;
				width = mat.cols;
				dst = cv::Mat(height * 20, width * 20, CV_8UC1);
			}

			int y_start = (i - 1) / 2 * height;
			int y_end = y_start + height;
			int x_start = (i - 1) % 2 * 10 * width + (j - 1) * width;
			int x_end = x_start + width;
			cv::Mat copy = dst(cv::Range(y_start, y_end), cv::Range(x_start, x_end));
			mat.copyTo(copy);
		}
	}

	int new_width = 750;
	float factor = dst.cols * 1.f / new_width;
	int new_height = dst.rows / factor;
	cv::resize(dst, dst, cv::Size(new_width, new_height));
	cv::imwrite("D:/OneDrive/CSMajor/2019to2020_2nd/PR/Homeworks/Ex3/database/ORL_Faces/test.png", dst);

	return 0;
}

int Fisherface(string save_path)
{
	string output_folder(save_path);
	vector<Mat> images, test_images;
	vector<int> labels, test_labels;
	read_ORLFaces("D:/OneDrive/CSMajor/2019to2020_2nd/PR/Homeworks/Ex3/database/ORL_Faces/", images, labels, test_images, test_labels);
	// Quit if there are not enough images for this demo.
	if (images.size() <= 1) {
		string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
		CV_Error(CV_StsError, error_message);
	}
	int height = images[0].rows;
	/*Mat testSample = images[images.size() - 1];
	int testLabel = labels[labels.size() - 1];
	images.pop_back();
	labels.pop_back();*/
	// The following lines create an Fisherfaces model for
	// face recognition and train it with the images and
	// labels read from the given CSV file.
	// If you just want to keep 10 Fisherfaces, then call
	// the factory method like this:
	//
	//      cv::createFisherFaceRecognizer(10);
	//
	// However it is not useful to discard Fisherfaces! Please
	// always try to use _all_ available Fisherfaces for
	// classification.
	//
	// If you want to create a FaceRecognizer with a
	// confidence threshold (e.g. 123.0) and use _all_
	// Fisherfaces, then call it with:
	//
	//      cv::createFisherFaceRecognizer(0, 123.0);
	//
	Ptr<FaceRecognizer> model = createFisherFaceRecognizer();
	model->train(images, labels);
	for (int i(0); i < test_images.size(); ++i)
	{
		int predictedLabel = model->predict(test_images[i]);
		string result_message = format("Fisherfaces Predicted class = %d / Actual class = %d.", predictedLabel, test_labels[i]);	// 162
		cout << result_message << endl;
	}
	// Here is how to get the eigenvalues of this Eigenfaces model:
	Mat eigenvalues = model->getMat("eigenvalues");
	// And we can do the same to display the Eigenvectors (read Eigenfaces):
	Mat W = model->getMat("eigenvectors");
	// Get the sample mean from the training data
	Mat mean = model->getMat("mean");
	
	imwrite(format("%s/mean.png", output_folder.c_str()), norm_0_255(mean.reshape(1, images[0].rows)));
	// Display or save the first, at most 16 Fisherfaces:
	for (int i = 0; i < min(16, W.cols); i++) {
		string msg = format("Eigenvalue #%d = %.5f", i, eigenvalues.at<double>(i));
		cout << msg << endl;
		// get eigenvector #i
		Mat ev = W.col(i).clone();
		// Reshape to original size & normalize to [0...255] for imshow.
		Mat grayscale = norm_0_255(ev.reshape(1, height));
		// Show the image & apply a Bone colormap for better sensing.
		Mat cgrayscale;
		applyColorMap(grayscale, cgrayscale, COLORMAP_BONE);
		// Display or save:
		imwrite(format("%s/fisherface_%d.png", output_folder.c_str(), i), norm_0_255(cgrayscale));
	}
	// Display or save the image reconstruction at some predefined steps:
	for (int num_component = 0; num_component < min(16, W.cols); num_component++) {
		// Slice the Fisherface from the model:
		Mat ev = W.col(num_component);
		Mat projection = subspaceProject(ev, mean, images[0].reshape(1, 1));
		Mat reconstruction = subspaceReconstruct(ev, mean, projection);
		// Normalize the result:
		reconstruction = norm_0_255(reconstruction.reshape(1, images[0].rows));
		imwrite(format("%s/fisherface_reconstruction_%d.png", output_folder.c_str(), num_component), reconstruction);
	}
	return 0;
}


int main()
{
	//ORLFacestoImage();
	Eigenface("D:/OneDrive/CSMajor/2019to2020_2nd/PR/Homeworks/Ex3/code/6_6/PREx3_6_6/output/PCA/");
	Fisherface("D:/OneDrive/CSMajor/2019to2020_2nd/PR/Homeworks/Ex3/code/6_6/PREx3_6_6/output/Fisher/");
	return 0;
}