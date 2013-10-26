#include "editor.h"

#include <queue>
#include <cassert>

template<typename ValueType>
static inline ValueType pow_2(ValueType val)
{
	return val * val;
}

static int cmp(double l, double r)
{
	double tmp = l - r;
	if (fabs(tmp) < 1e-5)
		return 0;
	return tmp > 0 ? 1 : -1;
}

Matrix<Monochrome> ImageToMonochrome(const Image &im)
{
	Matrix<Monochrome> result(im.n_rows, im.n_cols);
	for (int i = 0; i < im.n_rows; ++i)
		for (int j = 0; j < im.n_cols; ++j)
			result(i, j) = get<0>(im(i, j));
	return result;
}

Image gaussian(const Image &im, double sigma, int radius)
{
	int n = radius * 2 + 1;
	Matrix<double> kernel(n, n);
	double k = 1 / (2 * M_PI * sigma * sigma);
	double d = -1 / (2 * sigma * sigma);
	double sum = 0;
	for (int i = 0; i < n; ++i)
		for (int j = 0; j < n; ++j) {
			kernel(i, j) = k * exp(d * (pow_2(i - radius) + pow_2(j - radius)));
			sum += kernel(i, j);
		}
	for (int i = 0; i < n; ++i)
		for (int j = 0; j < n; ++j)
			kernel(i, j) /= sum;
	return custom(im, kernel);
}

Image gaussian_separable(const Image &im, double sigma, int radius)
{
	double k = 1 / (2 * M_PI * sigma * sigma);
	double d = -1 / (2 * sigma * sigma);
	int n = radius * 2 + 1;
	Matrix<double> kernel_col(n, 1);
	Matrix<double> kernel_row(1, n);
	double s = 0;
	for (int i = 0; i < n; ++i) {
		kernel_col(i, 0) = kernel_row(0, i) = k * exp(d * (pow_2(i - radius)));
		s += kernel_col(i, 0);
	}
	for (int i = 0; i < n; ++i) {
		kernel_col(i, 0) /= s;
		kernel_row(0, i) /= s;
	}


	// vertical - kernel_col
	Image tmp(im.n_rows, im.n_cols);
	for (int i = radius; i < im.n_rows - radius; ++i)
		for (int j = 0; j < im.n_cols; ++j) {
			tuple<double, double, double> sum;
			for (int l = 0; l < n; ++l) {
				auto tmp3 = kernel_col(l, 0) * im(i - radius + l, j);
				sum = sum + tmp3;
			}

			tmp(i, j) = RGB(
				round(get<0>(sum)),
				round(get<1>(sum)),
				round(get<2>(sum))
			);
		}

	// horisontal - kernel_row
	Image tmp2(im.n_rows, im.n_cols);
	for (int i = 0; i < im.n_rows; ++i)
		for (int j = radius; j < im.n_cols - radius; ++j) {
			tuple<double, double, double> sum;
			for (int l = 0; l < n; ++l) {
				auto tmp3 = kernel_row(0, l) * tmp(i, j - radius + l);
				sum = sum + tmp3;
			}

			tmp2(i, j) = RGB(
				round(get<0>(sum)),
				round(get<1>(sum)),
				round(get<2>(sum))
			);
		}

	return tmp2;
}

Image sobel_x(const Image &im)
{
	Matrix<Monochrome> kernel = {
		{-1, 0, 1},
		{-2, 0, 2},
		{-1, 0, 1},
	};
	return custom(im, kernel);
}

Image sobel_y(const Image &im)
{
	Matrix<Monochrome> kernel = {
		{1, 2, 1},
		{0, 0, 0},
		{-1, -2, -1},
	};
	return custom(im, kernel);
}

class AbsGradientFunctor
{
public:
	template<typename InputType>
	Monochrome operator()(const InputType &left, const InputType &right) const
	{
		return sqrt(pow_2(left(0, 0)) + pow_2(right(0, 0)));
	}
	static const int radius = 0;
};

class DirectGradientFunctor
{
public:
	template<typename InputType>
	double operator()(const InputType &left, const InputType &right) const
	{
		return atan2(right(0, 0), left(0, 0));
	}
	static const int radius = 0;
};

class SuppressionFunctor
{
public:
	template<typename InputType, typename InputType2>
	Monochrome operator()(const InputType &abs, const InputType2 &direct) const
	{
		int i = lower_bound(mas.begin(), mas.end(), direct(1, 1)) - mas.begin();
		int first = (num[i] + 2) % 8;
		int second = (first + 4) % 8;
		if (
			abs(1 + di[first], 1 + dj[first]) > abs(1, 1) ||
			abs(1 + di[second], 1 + dj[second]) > abs(1, 1)
		)
			return 0;
		return abs(1, 1);
	}
	static const int radius = 1;
private:
	static const vector<double> mas;
	static const vector<int> num;
	static const vector<int> di;
	static const vector<int> dj;
};
static const double pp = M_PI / 8;
const vector<double> SuppressionFunctor::mas = {-7*pp, -5*pp, -3*pp, -pp, pp, 3*pp, 5*pp, 7*pp, 8*pp};
const vector<int> SuppressionFunctor::num    = {   4,    5,    6,  7, 0,   1,   2,   3,   4};
const vector<int> SuppressionFunctor::dj = {-1, -1, 0, 1, 1, 1, 0, -1};
const vector<int> SuppressionFunctor::di = {0, 1, 1, 1, 0, -1, -1, -1};

class TreshholdFunctor
{
public:
	TreshholdFunctor(int min_, int max_)
		: t_min(min_), t_max(max_)
	{}

	Monochrome operator()(const MonochromeImage &im) const
	{
		if (im(0, 0) <= t_min)
			return 0;
		if (im(0, 0) >= t_max)
			return 255;
		return 128;
	}
	static const int radius = 0;
	const int t_min;
	const int t_max;	
};

template <typename InputType>
void hysteresis_bfs(InputType &abs_grad)
{
	const vector<int> di = {-1, -1, 0, 1, 1, 1, 0, -1};
	const vector<int> dj = {0, 1, 1, 1, 0, -1, -1, -1};
	auto exist = [&](int i, int j){ return 0 <= i && i < abs_grad.n_rows && 0 <= j && j < abs_grad.n_cols;};
	vector<vector<char>> visited(abs_grad.n_rows, vector<char>(abs_grad.n_cols, 0));
	for (int i = 0; i < abs_grad.n_rows; ++i)
		for (int j = 0; j < abs_grad.n_cols; ++j)
			if (!visited[i][j] && abs_grad(i, j) == 255) {
				queue<pair<int, int>> Q;
				Q.push(make_pair(i, j));
				visited[i][j] = 1;
				abs_grad(i, j) = 255;
				while (!Q.empty()) {
					int ti = Q.front().first;
					int tj = Q.front().second;
					Q.pop();
					for (int k = 0; k < 8; ++k) {
						int ni = ti + di[k];
						int nj = tj + dj[k];
						if (
							exist(ni, nj) &&
							!visited[ni][nj] &&
							abs_grad(ni, nj) >= 128
						) {
							visited[ni][nj] = 1;
							abs_grad(ni, nj) = 255;
							Q.push(make_pair(ni, nj));
						}
					}
				}
			}
	for (int i = 0; i < abs_grad.n_rows; ++i)
		for (int j = 0; j < abs_grad.n_cols; ++j)
			if (abs_grad(i, j) == 128)
				abs_grad(i, j) = 0;
}

class DSU
{
public:
	DSU(int n)
		: parent(n), rank(n), marks(n)
	{}

	void make_set(int v)
	{
		parent[v] = v;
		rank[v] = 0;
		marks[v] = 0;
	}
	
	void set_mark(int v)
	{
		marks[find_set(v)] = 1;
	}

	bool check_mark(int v)
	{
		return marks[find_set(v)];
	}

	int find_set(int v)
	{
		if (parent[v] == v)
			return v;
		return parent[v] = find_set(parent[v]);
	}
	 
	int union_sets(int a, int b)
	{
		a = find_set(a);
		b = find_set(b);
		if (a != b) {
			if (rank[a] < rank[b])
				swap(a, b);
			parent[b] = a;
			marks[a] = (marks[a] | marks[b]);
			if (rank[a] == rank[b])
				++rank[a];
		}
		return a;
	}

private:
	vector<int> parent;
	vector<int> rank;
	vector<char> marks;
};

template <typename InputType>
void hysteresis_dsu(InputType &abs_grad)
{
	int len = abs_grad.n_rows * abs_grad.n_cols;
	auto dsu = DSU(len);
	auto exist = [&](int i, int j){ return 0 <= i && i < abs_grad.n_rows && 0 <= j && j < abs_grad.n_cols;};
	auto check = [&](int i, int j){ return exist(i, j) && (abs_grad(i, j) >= 128);};
	auto strong = [&](int i, int j){ return exist(i, j) && (abs_grad(i, j) == 255);};
	vector<vector<int>> groups(abs_grad.n_rows, vector<int>(abs_grad.n_cols));
	int count = 0;
	for (int i = 0; i < abs_grad.n_rows; ++i)
		for (int j = 0; j < abs_grad.n_cols; ++j) {
			bool A = check(i, j);
			bool C = check(i - 1, j);
			bool B = check(i, j - 1);
			if (!A)
				continue;
			int current = 0;
			if (!B && !C) {
				current = count++;
				dsu.make_set(current);
			}

			if (B && !C)
				current = dsu.find_set(groups[i][j - 1]);
			if (!B && C)
				current = dsu.find_set(groups[i - 1][j]);
			if (B && C)
				current = dsu.union_sets(groups[i - 1][j], groups[i][j - 1]);

			if (strong(i, j))
				dsu.set_mark(current);

			groups[i][j] = current;
		}
	for (int i = 0; i < abs_grad.n_rows; ++i)
		for (int j = 0; j < abs_grad.n_cols; ++j)
			abs_grad(i, j) = 255 * (check(i, j) && dsu.check_mark(groups[i][j]));
}

Image canny(const Image &im, int threshold1, int threshold2)
{
	auto blured = gaussian_separable(im, 1.4, 2);

	auto derivative_x = ImageToMonochrome(sobel_x(blured));
	auto derivative_y = ImageToMonochrome(sobel_y(blured));

	auto abs_grad_functor = AbsGradientFunctor();
	auto abs_grad = binary_map(abs_grad_functor, derivative_x, derivative_y);

	const int di[] = {1, 0, 1, 1};
	const int dj[] = {0, 1, 1, -1};

	MonochromeImage new_grad(abs_grad.n_rows, abs_grad.n_cols);
	for (int i = 1; i < abs_grad.n_rows - 1; ++i)
		for (int j = 1; j < abs_grad.n_cols - 1; ++j) {
			int x = derivative_x(i, j);
			int y = derivative_y(i, j);
			int k = 3;
			if (100000 * abs(x) < 41421 * abs(y))
				k = 0;
			else if (100000 * abs(y) < 41421 * abs(x))
				k = 1;
			else if ((x < 0 && y > 0) || (x > 0 && y < 0))
				k = 2;
			new_grad(i, j) = abs_grad(i, j) *(
				abs_grad(i + di[k], j + dj[k]) <= abs_grad(i, j) &&
				abs_grad(i - di[k], j - dj[k]) <  abs_grad(i, j));
		}
	abs_grad = new_grad;

	auto treshold_functor = TreshholdFunctor(threshold1, threshold2);
	abs_grad = abs_grad.unary_map(treshold_functor);

	hysteresis_bfs(abs_grad);

	auto result = MonochromeToImage(normalize(abs_grad));
	return result;
}

template<typename MetricType>
static pair<int, int> match(MonochromeImage &r, MonochromeImage &g, MetricType metric, int i = 0, int j = 0, int rd = 15)
{
	vector<pair<double, pair<int, int>>> v;
	for (int di = -rd; di <= rd; ++di)
		for (int dj = -rd; dj <= rd; ++dj) {
			MonochromeImage r_ = r.submatrix(max(0, di + i), max(0, dj + j), min(r.n_rows, g.n_rows + di + i) - max(0, di + i), min(r.n_cols, g.n_cols + dj + j) - max(0, dj + j));
			MonochromeImage g_ = g.submatrix(max(0, -di + i), max(0, -dj + j), min(g.n_rows, r.n_rows - di + i) - max(0, -di + i), min(g.n_cols, r.n_cols - dj + j) - max(0, -dj + j));
			v.push_back(make_pair(metric(r_, g_), make_pair(di + i, dj + j)));
		}
	sort(v.begin(), v.end());
	return v[0].second;
}

template<typename MetricType>
pair<int, int> scale_match(MonochromeImage &l, MonochromeImage &r, MetricType metric)
{
	if (l.n_rows < 4000 && l.n_cols < 4000) {
		return match(l, r, metric);
	} else {
		auto L = ImageToMonochrome(resize(MonochromeToImage(l), 0.5));
		auto R = ImageToMonochrome(resize(MonochromeToImage(r), 0.5));
		auto shift = scale_match(L, R, metric);
		shift.first *= 2;
		shift.second *= 2;
		return match(l, r, metric, shift.first, shift.second, 2);
	}
}

template<typename MetricType>
void normal_match(MonochromeImage &r, MonochromeImage &g, MetricType metric)
{
	auto shift = scale_match(r, g, metric);
	int row_shift = shift.first;
	int col_shift = shift.second;
	MonochromeImage r_ = r.submatrix(max(0, row_shift), max(0, col_shift), min(r.n_rows, g.n_rows + row_shift) - max(0, row_shift), min(r.n_cols, g.n_cols + col_shift) - max(0, col_shift));
	MonochromeImage g_ = g.submatrix(max(0, -row_shift), max(0, -col_shift), min(g.n_rows, r.n_rows - row_shift) - max(0, -row_shift), min(g.n_cols, r.n_cols - col_shift) - max(0, -col_shift));
	r = r_;
	g = g_;
}

Image align(const Image &image, const string &postprocessing, double fraction)
{
	auto tmp = split_image(image);
	for (auto &im : tmp) {
		auto borders = ImageToMonochrome(canny(im, 36, 100));
		vector<int> sum_row(borders.n_rows);
		vector<int> sum_col(borders.n_cols);
		for (int i = 0; i < borders.n_rows; ++i)
			for (int j = 0; j < borders.n_cols; ++j) {
				sum_row[i] += (borders(i, j) != 0);
				sum_col[j] += (borders(i, j) != 0);
			}

		vector<pair<int, int>> v;
		for (int i = 0; i < borders.n_rows * 0.05; ++i)
			v.push_back(make_pair(sum_row[i], i));
		sort(v.rbegin(), v.rend());
		int row1 = max(v[0].second, v[1].second) + 1;

		v.clear();
		for (int i = 0; i < borders.n_cols * 0.05; ++i)
			v.push_back(make_pair(sum_col[i], i));
		sort(v.rbegin(), v.rend());
		int col1 = max(v[0].second, v[1].second) + 1;

		v.clear();
		for (int i = borders.n_rows * 0.95; i < borders.n_rows; ++i)
			v.push_back(make_pair(sum_row[i], i));
		sort(v.rbegin(), v.rend());
		int row2 = min(v[0].second, v[1].second);

		v.clear();
		for (int i = borders.n_cols * 0.95; i < borders.n_cols; ++i)
			v.push_back(make_pair(sum_col[i], i));
		sort(v.rbegin(), v.rend());
		int col2 = min(v[0].second, v[1].second);

		im = im.submatrix(row1, col1, row2 - row1, col2 - col1);
	}
	vector<MonochromeImage> rgb(3);
	for (int i = 0; i < 3; ++i)
		rgb[i] = ImageToMonochrome(tmp[i]);
	MonochromeImage &r = rgb[0];
	MonochromeImage &g = rgb[1];
	MonochromeImage &b = rgb[2];

	auto mse = [](MonochromeImage &im1, MonochromeImage &im2) {
		double sum = 0;
		int row_lim = min(im1.n_rows, im2.n_rows);
		int col_lim = min(im1.n_cols, im2.n_cols);
		for (int i = 0; i < row_lim; ++i)
			for (int j = 0; j < col_lim; ++j)
				sum += pow_2(im1(i, j) - im2(i, j));
		sum /= (row_lim * col_lim);
		return sum;
	};

	auto cc __attribute__((unused)) = [](MonochromeImage &im1, MonochromeImage &im2) {
		double sum = 0;
		for (int i = 0; i < im1.n_rows; ++i)
			for (int j = 0; j < im2.n_cols; ++j)
				sum += im1(i, j) * im2(i, j);
		sum *= -1;
		return sum;
	};

	normal_match(r, g, mse);
	normal_match(r, b, mse);
	normal_match(b, g, mse);

	auto result = MonochromeToImage(r);

	auto safe_get = [](MonochromeImage &im, int i, int j) { return (i < im.n_rows && j < im.n_cols) ? im(i, j) : 0;};

	for (int i = 0; i < result.n_rows; ++i)
		for (int j = 0; j < result.n_cols; ++j)
			result(i, j) = RGB(
				safe_get(b, i, j), 
				safe_get(g, i, j), 
				safe_get(r, i, j)
			);
	
	if (postprocessing == "--gray-world")
		result = gray_world(result);
	else if (postprocessing == "--unsharp")
		result = unsharp(result);
	else if (postprocessing == "--autocontrast")
		result = autocontrast(result, fraction);
	return result;
}

Image gray_world(const Image &im)
{
	Image result(im.n_rows, im.n_cols);
	double Sr = 0;
	double Sg = 0;
	double Sb = 0;
	for (int i = 0; i < im.n_rows; ++i)
		for (int j = 0; j < im.n_cols; ++j) {
			Sr += get<0>(im(i, j));
			Sg += get<1>(im(i, j));
			Sb += get<2>(im(i, j));
		}
	double s = (Sr + Sg + Sb) / 3;
	for (int i = 0; i < im.n_rows; ++i)
		for (int j = 0; j < im.n_cols; ++j)
			result(i, j) = RGB(
				round(cmp(Sr, 0) ? (s * get<0>(im(i, j)) / Sr) : (s / im.n_rows / im.n_cols)),
				round(cmp(Sg, 0) ? (s * get<1>(im(i, j)) / Sg) : (s / im.n_rows / im.n_cols)),
				round(cmp(Sb, 0) ? (s * get<2>(im(i, j)) / Sb) : (s / im.n_rows / im.n_cols))
			);
	return normalize(result);
}

Image unsharp(const Image &im)
{
	Matrix<double> kernel = {
		{-1.0/6, -2.0/3, -1.0/6},
		{-2.0/3, 13.0/3, -2.0/3},
		{-1.0/6, -2.0/3, -1.0/6}
	};
	return normalize(custom(im, kernel));
}

class ContrastFunctor
{
public:
	ContrastFunctor(int area, double f)
		:n(area),
		 low(area * f),
		 hi(area - low),
		 hyst()
	{}
	
	RGB operator()(const Image &im) const
	{
		hyst.push_back(Y(
			get<0>(im(0,0)),
			get<1>(im(0,0)),
			get<2>(im(0,0))
		));
		return im(0, 0);
	}
	
	static double Y(Monochrome R, Monochrome G, Monochrome B)
	{
		return R * 0.2125 + G * 0.7154 + B * 0.0721;
	}

	bool all_done() const
	{
		return !(1 <= hi && low < n && low <= hi);
	}

	pair<double, double> get_limits() const
	{
		if (low > hi || low >= n || hi < 0)
			return make_pair(0, 0);
		sort(hyst.begin(), hyst.end());
		return make_pair(hyst[low], hyst[hi - 1]);
	}
	
	int n;
	int low, hi;
	mutable vector<double> hyst;
	static const int radius = 0;
};

class StretchHyst
{
public:
	StretchHyst(double t1, double t2)
		:threshold1(t1),
		 threshold2(t2)
	{}

	RGB operator()(const Image &im) const
	{
		return RGB(
			(get<0>(im(0, 0)) - threshold1) * 255 / (threshold2 - threshold1),
			(get<1>(im(0, 0)) - threshold1) * 255 / (threshold2 - threshold1),
			(get<2>(im(0, 0)) - threshold1) * 255 / (threshold2 - threshold1)
		);
	}
	double threshold1;
	double threshold2;
	static const int radius = 0;
};

Image autocontrast(const Image &im, double fraction)
{
	auto contrast = ContrastFunctor(im.n_rows * im.n_cols, fraction);
	auto result = im.unary_map(contrast);
	auto limits = contrast.get_limits();
	cerr << limits.first << " " << limits.second << endl;
	if (cmp(limits.first, limits.second) == 0)
		return im;
	auto strech_hyst = StretchHyst(limits.first, limits.second);
	return normalize(result.unary_map(strech_hyst)); 
}

Image resize(const Image &im, double scale)
{
	Image out(im.n_rows * scale, im.n_cols * scale);
    for (int i = 0; i < out.n_rows; i++) {
            double tmp1 = (.0 + i) / (out.n_rows - 1) * (im.n_rows - 1);
            int h = floor(tmp1);
            h = max(0, min(im.n_rows - 2, h));
            double u = tmp1 - h;
        for (int j = 0; j < out.n_cols; j++) {
            double tmp2 = (.0 + j) / (out.n_cols - 1) * (im.n_cols - 1);
            int w = floor(tmp2);
            w = max(0, min(im.n_cols - 2, w));
            double t = tmp2 - w;
 
            double d1 = (1 - t) * (1 - u);
            double d2 = t * (1 - u);
            double d3 = t * u;
            double d4 = (1 - t) * u;
 
            RGB p1 = im(h, w);
            RGB p2 = im(h, w + 1);
            RGB p3 = im(h + 1, w + 1);
            RGB p4 = im(h + 1, w);
 			
 			out(i, j) = RGB(
	            get<0>(p1) * d1 +
	            get<0>(p2) * d2 +
	            get<0>(p3) * d3 +
	            get<0>(p4) * d4,

	            get<1>(p1) * d1 +
	            get<1>(p2) * d2 +
	            get<1>(p3) * d3 +
	            get<1>(p4) * d4,

	            get<2>(p1) * d1 +
	            get<2>(p2) * d2 +
	            get<2>(p3) * d3 +
	            get<2>(p4) * d4
			);
        }
    }
	return out;
}

void save_image(const Image &im, const char *path)
{
    BMP out;
    out.SetSize(im.n_cols, im.n_rows);

    int r, g, b;
    RGBApixel p;
    p.Alpha = 255;
    for (int i = 0; i < im.n_rows; ++i) {
        for (int j = 0; j < im.n_cols; ++j) {
            tie(r, g, b) = im(i, j);
            p.Red = r; p.Green = g; p.Blue = b;
            out.SetPixel(j, i, p);
        }
    }

    if (!out.WriteToFile(path))
        throw string("Error writing file ") + string(path);
}
