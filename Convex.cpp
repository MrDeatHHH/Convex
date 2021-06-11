#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

const int modK = 21;
const int modNt = 2;
const double infinity = 10000000000;

const int amount_of_samples = 4;
const double lambda = 100;
const double starting_ksi = 10;
const int total_iter = 100;

void save_and_show(int* res, const int width, const int height, string name, bool save = false)
{
	Mat* result = new Mat[3];
	for (int c = 0; c < 3; ++c)
	{
		result[c] = Mat::zeros(Size(width, height), CV_8UC1);
		for (int x = 0; x < width; ++x)
			for (int y = 0; y < height; ++y)
			{
				result[c].at<uchar>(y, x) = uchar(res[x * height + y] * 8);
			}
	}

	Mat rez;
	vector<Mat> channels;

	channels.push_back(result[0]);
	channels.push_back(result[1]);
	channels.push_back(result[2]);

	merge(channels, rez);

	namedWindow(name, WINDOW_AUTOSIZE);
	cv::imshow(name, rez);
	if (save)
		imwrite(name + ".png", rez);

	delete[] result;
}

inline double get_q(double* q, int* lcolors, int* rcolors, const int t, const int k, const int height, double* L)
{
	return (q[abs(lcolors[t * 3] - rcolors[(t + k * height) * 3])] +
		q[abs(lcolors[t * 3 + 1] - rcolors[(t + k * height) * 3 + 1])] +
		q[abs(lcolors[t * 3 + 2] - rcolors[(t + k * height) * 3 + 2])] -
		L[t * modK + k]);
}

inline double get_g(double* g, const int c1, const int c2)
{
	return g[abs(c1 - c2)];
}

int find(int* arr, const int start, const int length, const int t)
{
	for (int i = 0; i < length; ++i)
		if (arr[start + i] == t)
			return i;

	cout << "Something is wrong... I can feel it" << endl;
	return -1;
}

int* dynamics(const int width, const int height,
	int* lcolors, int* rcolors,
	double* g, double* q, double* L)
{
	const int modT = width * height;
	int* result = new int[modT]();

	for (int y = 0; y < height; ++y)
	{
		int* ks = new int[width * modK];
		double* f = new double[width * modK]();

		// For f_0(k_1) to f_n-1(k_n)
		for (int x = 0; x < width - 1; ++x)
		{

			for (int k = 0; k < modK; ++k)
			{
				int ind = -1;
				double best = infinity;
				for (int k_ = 0; k_ < min(modK, width - x); ++k_)
				{
					double cur;
					if (x == 0)
						cur = get_q(q, lcolors, rcolors, x * height + y, k_, height, L) + get_g(g, k, k_);
					else
						cur = get_q(q, lcolors, rcolors, x * height + y, k_, height, L) + get_g(g, k, k_) + f[(x - 1) * modK + k_];

					if (cur < best)
					{
						ind = k_;
						best = cur;
					}
				}

				ks[x * modK + k] = ind;
				f[x * modK + k] = best;
			}
		}

		// For max_k_n(f_n-1(k_n) + q_n)
		int ind = -1;
		double best = infinity;
		for (int k_ = 0; k_ < 1; ++k_)
		{
			double cur = get_q(q, lcolors, rcolors, (width - 1) * height + y, k_, height, L) + f[(width - 1) * modK + k_];

			if (cur < best)
			{
				ind = k_;
				best = cur;
			}
		}

		result[(width - 1) * height + y] = ind;
		for (int x = width - 2; x >= 0; --x)
		{
			result[x * height + y] = ks[x * modK + result[(x + 1) * height + y]];
		}

		delete[] ks;
		delete[] f;
	}

	return result;
}

inline double L_func(const int k1, const int k2)
{
	return abs(k1 - k2);
}

void update_q_g(int** lcolors, int** rcolors, int** gcolors, int* widthes, int* heightes, double* q, double* g)
{
	for (int it = 0; it < total_iter; ++it)
	{
		double ksi = starting_ksi / double(it + 1);

		cout << it << " iteration of learning" << endl;
		for (int i = 0; i < amount_of_samples; ++i)
		{
			cout << i << " sample" << endl;
			const int modT = widthes[i] * heightes[i];

			double* L = new double[modT * modK];
			for (int t = 0; t < modT; ++t)
				for (int k = 0; k < modK; ++k)
					L[t * modK + k] = L_func(gcolors[i][t], k);

			int* res = dynamics(widthes[i], heightes[i], lcolors[i], rcolors[i], g, q, L);

			double* grad_q = new double[256]();
			double* grad_g = new double[modK]();

			// For safety
			for (int a = 0; a < 256; ++a)
				grad_q[a] = 0;

			for (int b = 0; b < modK; ++b)
				grad_g[b] = 0;

			for (int a = 0; a < 256; ++a)
				grad_q[a] -= 2 * lambda * q[a];

			for (int b = 0; b < modK; ++b)
				grad_g[b] -= 2 * lambda * g[b];
			
			// Update q
			for (int t = 0; t < modT; ++t)
			{
				if (res[t] == -1)
					cout << "ALARM!!!" << endl;

				if ((t + gcolors[i][t] * heightes[i]) < modT)
				{
					grad_q[abs(lcolors[i][t * 3] - rcolors[i][(t + res[t] * heightes[i]) * 3])] -= 1;
					grad_q[abs(lcolors[i][t * 3 + 1] - rcolors[i][(t + res[t] * heightes[i]) * 3 + 1])] -= 1;
					grad_q[abs(lcolors[i][t * 3 + 2] - rcolors[i][(t + res[t] * heightes[i]) * 3 + 2])] -= 1;

					grad_q[abs(lcolors[i][t * 3] - rcolors[i][(t + gcolors[i][t] * heightes[i]) * 3])] += 1;
					grad_q[abs(lcolors[i][t * 3 + 1] - rcolors[i][(t + gcolors[i][t] * heightes[i]) * 3 + 1])] += 1;
					grad_q[abs(lcolors[i][t * 3 + 2] - rcolors[i][(t + gcolors[i][t] * heightes[i]) * 3 + 2])] += 1;
				}
			}

			// Update g
			for (int t = 0; t < modT - heightes[i]; ++t)
			{
				grad_g[abs(res[t] - res[t + heightes[i]])] -= 1;
				grad_g[abs(gcolors[i][t] - gcolors[i][t + heightes[i]])] += 1;
			}

			// Normalizing grads
			double sum_grad = 0.;
			
			for (int a = 0; a < 256; ++a)
				sum_grad += pow(grad_q[a], 2);

			for (int b = 0; b < modK; ++b)
				sum_grad += pow(grad_g[b], 2);

			for (int a = 0; a < 256; ++a)
				q[a] += grad_q[a]  * ksi / sqrt(sum_grad);

			for (int b = 0; b < modK; ++b)
				g[b] += grad_g[b] * ksi / sqrt(sum_grad);

			delete[] grad_q;
			delete[] grad_g;
			delete[] L;
			delete[] res;
		}
	}
}

int main()
{
	int** lcolors = new int* [amount_of_samples];
	int** rcolors = new int* [amount_of_samples];
	int** gcolors = new int* [amount_of_samples];

	int* heightes = new int[amount_of_samples];
	int* widthes = new int[amount_of_samples];

	double* q = new double[256]();
	for (int i = 0; i < 256; ++i)
		q[i] = i;
	double* g = new double[modK]();
	for (int i = 0; i < modK; ++i)
		g[i] = i;

	for (int i = 0; i < amount_of_samples; ++i)
	{
		Mat limage_, limage[4];
		limage_ = imread("./dataset/" + to_string(i + 1) + "/im6.ppm", IMREAD_UNCHANGED);
		split(limage_, limage);

		Mat rimage_, rimage[4];
		rimage_ = imread("./dataset/" + to_string(i + 1) + "/im2.ppm", IMREAD_UNCHANGED);
		split(rimage_, rimage);

		Mat gimage;
		gimage = imread("./dataset/" + to_string(i + 1) + "/disp2.pgm", IMREAD_UNCHANGED);

		heightes[i] = limage[0].size().height;
		widthes[i] = limage[0].size().width;

		lcolors[i] = new int[widthes[i] * heightes[i] * 3];
		rcolors[i] = new int[widthes[i] * heightes[i] * 3];
		gcolors[i] = new int[widthes[i] * heightes[i]];

		for (int x = 0; x < widthes[i]; ++x)
		{
			for (int y = 0; y < heightes[i]; ++y)
			{
				for (int c = 0; c < 3; ++c)
				{
					lcolors[i][x * heightes[i] * 3 + y * 3 + c] = int(limage[c].at<uchar>(y, x));
					rcolors[i][x * heightes[i] * 3 + y * 3 + c] = int(rimage[c].at<uchar>(y, x));
				}
				gcolors[i][x * heightes[i] + y] = int((gimage.at<uchar>(y, x) + 1) / 8);
			}
		}
	}

	update_q_g(lcolors, rcolors, gcolors, widthes, heightes, q, g);

	for (int i = 0; i < amount_of_samples; ++i)
	{
		const int modT = widthes[i] * heightes[i];

		double* L = new double[modT * modK];
		for (int t = 0; t < modT; ++t)
			for (int k = 0; k < modK; ++k)
				L[t * modK + k] = 0.;

		int* res = dynamics(widthes[i], heightes[i], lcolors[i], rcolors[i], g, q, L);

		save_and_show(res, widthes[i], heightes[i], "sample number " + to_string(i), false);

		delete[] L;
		delete[] res;
	}

	Mat limage_, limage[4];
	limage_ = imread("./test/im6.ppm", IMREAD_UNCHANGED);
	split(limage_, limage);

	Mat rimage_, rimage[4];
	rimage_ = imread("./test/im2.ppm", IMREAD_UNCHANGED);
	split(rimage_, rimage);

	Mat gimage;
	gimage = imread("./test/disp2.pgm", IMREAD_UNCHANGED);

	const int height = limage[0].size().height;
	const int width = limage[0].size().width;

	int* ltest = new int[width * height * 3];
	int* rtest = new int[width * height * 3];
	int* gtest = new int[width * height];

	for (int x = 0; x < width; ++x)
	{
		for (int y = 0; y < height; ++y)
		{
			for (int c = 0; c < 3; ++c)
			{
				ltest[x * height * 3 + y * 3 + c] = int(limage[c].at<uchar>(y, x));
				rtest[x * height * 3 + y * 3 + c] = int(rimage[c].at<uchar>(y, x));
			}
			gtest[x * height + y] = int(gimage.at<uchar>(y, x));
		}
	}

	// Create neighbour structure
	const int modT = width * height;
	double* L = new double[modT * modK]();
	for (int t = 0; t < modT; ++t)
		for (int k = 0; k < modK; ++k)
			L[t * modK + k] = 0.;

	int* res = dynamics(width, height, ltest, rtest, g, q, L);

	save_and_show(res, width, height, "result", true);
	
	std::cout << "q:\n";
	for (int i = 0; i < 256; ++i) {
		std::cout << q[i] << "  ";
	}
	std::cout << "\n\n";
	std::cout << "g:\n";
	for (int i = 0; i < modK; ++i) {
		std::cout << g[i] << "  ";
	}
	std::cout << "\n\n";
	
	waitKey(0);
	return 0;
}