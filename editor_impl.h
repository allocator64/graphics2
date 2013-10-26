#pragma once

#include <cmath>
#include <iostream>

using namespace std;

template<typename LeftType, typename RightType>
static
tuple<LeftType,LeftType,LeftType>
operator+(const tuple<LeftType,LeftType,LeftType> &l, const tuple<RightType,RightType,RightType> &r)
{
	return tuple<LeftType,LeftType,LeftType>(
		get<0>(l) + get<0>(r),
		get<1>(l) + get<1>(r),
		get<2>(l) + get<2>(r)
	);
}

template<typename LeftType, typename RightType>
static
tuple<LeftType,LeftType,LeftType>
operator*(const LeftType &l, const tuple<RightType,RightType,RightType> &r)
{
	return tuple<LeftType,LeftType,LeftType>(
		l * get<0>(r),
		l * get<1>(r),
		l * get<2>(r)
	);
}

template<typename KernelType>
ConvolutionFunctor<KernelType>\
	::ConvolutionFunctor(const KernelType &k_)
	:kernel(k_),
	 diameter(max(k_.n_rows, k_.n_cols)),
	 row_shift((diameter - k_.n_rows) / 2),
	 col_shift((diameter - k_.n_cols) / 2),
	 radius(diameter / 2)
{}

template<typename KernelType>
RGB ConvolutionFunctor<KernelType>\
	::operator()(const Image &f)
{
	typedef typename KernelType::value_type value_type;
	tuple<value_type, value_type, value_type> sum;
	for (int i = 0; i < kernel.n_rows; ++i)
		for (int j = 0; j < kernel.n_cols; ++j) {
			auto tmp = kernel(i, j) * f(i + row_shift, j + col_shift);
			sum = sum + tmp;
		}
	return RGB(
		round(get<0>(sum)),
		round(get<1>(sum)),
		round(get<2>(sum))
	);
}

template <typename KernelType>
static Image custom(const Image &im, const KernelType &kernel)
{
	auto conv = ConvolutionFunctor<KernelType>(kernel);
	return im.unary_map(conv);
}

template<typename ValueType>
Matrix<ValueType> normalize(const Matrix<ValueType> &im, int t1 = 0, int t2 = 255)
{
    auto norm = NormalizeFunctor(t1, t2);
    return im.unary_map(norm);
}

template<typename ValueType>
std::vector<Matrix<ValueType>> split_image(const Matrix<ValueType> &im)
{
	auto rows = im.n_rows / 3;
	auto cols = im.n_cols;

	return std::vector<Matrix<ValueType>>({
		im.submatrix(0, 0, rows, cols),
		im.submatrix(rows, 0, rows, cols),
		im.submatrix(2 * rows, 0, rows, cols),
	});
}
