#pragma once

#include <tuple>
#include "matrix.h"
#include "EasyBMP/EasyBMP.h"
using namespace std;

typedef tuple<int, int, int> RGB;
typedef int Monochrome;
typedef Matrix<Monochrome> MonochromeImage;

template<typename ValueType>
std::vector<Matrix<ValueType>> split_image(const Matrix<ValueType> &im);
Image gaussian(const Image &im, double sigma, int radius);
Image gaussian_separable(const Image &im, double sigma, int radius);
Image sobel_x(const Image &im);
Image sobel_y(const Image &im);
Image canny(const Image &im, int threshold1, int threshold2);
Image align(const Image &im, const string &postprocessing = "", double fraction = 0);
Image gray_world(const Image &im);
Image unsharp(const Image &im);
Image autocontrast(const Image &im, double fraction);
Image resize(const Image &im, double scale);

Matrix<Monochrome> ImageToMonochrome(const Image &im);

template<typename ValueType>
Image MonochromeToImage(const Matrix<ValueType> &im);
void save_image(const Image &im, const char *path);

template <typename KernelType>
static Image custom(const Image &im, const KernelType &kernel);

template <typename ValueType>
static Matrix<ValueType> custom2(const Matrix<ValueType> &im, const Matrix<ValueType> &kernel);

template<typename KernelType>
class ConvolutionFunctor
{
public:
	ConvolutionFunctor(const KernelType &k_);

	RGB operator()(const Image &f);

private:
	const KernelType &kernel;
	const int diameter;
	const int row_shift;
	const int col_shift;
public:
	const int radius;
};

template<typename ValueType>
class ConvolutionFunctor2
{
public:
	ConvolutionFunctor2(const Matrix<ValueType> &k_);

	ValueType operator()(const Matrix<ValueType> &f);

private:
	const Matrix<ValueType> &kernel;
	const int diameter;
	const int row_shift;
	const int col_shift;
public:
	const int radius;
};

class NormalizeFunctor
{
	int norm(int pix)
	{
		return max(threshold1, min(threshold2, pix));
	}
public:
	NormalizeFunctor(int t1 = 0, int t2 = 255)
		: threshold1(t1),
		  threshold2(t2)
	{}

	template<typename ValueType>
	ValueType operator()(const Matrix<ValueType> &im)
	{
		return norm(round(im(0, 0)));
	}

	RGB operator()(const Image &pix)
	{
		return RGB(
			norm(get<0>(pix(0, 0))),
			norm(get<1>(pix(0, 0))),
			norm(get<2>(pix(0, 0)))
		);
	}
	const int threshold1;
	const int threshold2;
	static const int radius = 0;	
};

template<typename ValueType>
Matrix<ValueType> normalize(const Matrix<ValueType> &im, int t1 = 0, int t2 = 255);

#include "editor_impl.h"