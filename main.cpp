#include <string>
#include <vector>
#include <fstream>
#include <cassert>
#include <iostream>
#include <cmath>

#include "classifier.h"
#include "EasyBMP/EasyBMP.h"
#include "liblinear-1.93/linear.h"
#include "argvparser/argvparser.h"
#include "matrix.h"
#include "editor.h"

using std::string;
using std::vector;
using std::ifstream;
using std::ofstream;
using std::pair;
using std::make_pair;
using std::make_tuple;
using std::cout;
using std::cerr;
using std::endl;
using std::get;

using CommandLineProcessing::ArgvParser;

typedef vector<pair<Image*, int> > TDataSet;
typedef vector<pair<string, int> > TFileList;
typedef vector<pair<vector<float>, int> > TFeatures;

// Load list of files and its labels from 'data_file' and
// stores it in 'file_list'
void LoadFileList(const string& data_file, TFileList* file_list) {
    ifstream stream(data_file.c_str());

    string filename;
    int label;
    
    int char_idx = data_file.size() - 1;
    for (; char_idx >= 0; --char_idx)
        if (data_file[char_idx] == '/' || data_file[char_idx] == '\\')
            break;
    string data_path = data_file.substr(0,char_idx+1);
    
    while(!stream.eof() && !stream.fail()) {
        stream >> filename >> label;
        if (filename.size())
            file_list->push_back(make_pair(data_path + filename, label));
    }

    stream.close();
}

// Load images by list of files 'file_list' and store them in 'data_set'
void LoadImages(const TFileList& file_list, TDataSet* data_set) {
    for (size_t img_idx = 0; img_idx < file_list.size(); ++img_idx) {
            // Create image

        BMP in;
        auto path = file_list[img_idx].first;

        if (!in.ReadFromFile(path.c_str())) {
            cerr << "Cannot load " << path;
            exit(1);
        }

        Image *res = new Image(in.TellHeight(), in.TellWidth());

        for (int i = 0; i < res->n_rows; ++i) {
            for (int j = 0; j < res->n_cols; ++j) {
                RGBApixel *p = in(j, i);
                (*res)(i, j) = make_tuple(p->Red, p->Green, p->Blue);
            }
        }

        // Add image and it's label to dataset
        data_set->push_back(make_pair(res, file_list[img_idx].second));
    }
}

// Save result of prediction to file
void SavePredictions(const TFileList& file_list,
                     const TLabels& labels, 
                     const string& prediction_file) {
        // Check that list of files and list of labels has equal size 
    assert(file_list.size() == labels.size());
        // Open 'prediction_file' for writing
    ofstream stream(prediction_file.c_str());

        // Write file names and labels to stream
    for (size_t image_idx = 0; image_idx < file_list.size(); ++image_idx)
        stream << file_list[image_idx].first << " " << labels[image_idx] << endl;
    stream.close();
}

// Exatract features from dataset.
// You should implement this function by yourself =)
void ExtractFeatures(const TDataSet& data_set, TFeatures* features) {
    for (size_t image_idx = 0; image_idx < data_set.size(); ++image_idx) {
        
        Image *im = data_set[image_idx].first;

        Matrix<int> res(im->n_rows, im->n_cols);
        
        for (int i = 0; i < im->n_rows; ++i)
            for (int j = 0; j < im->n_cols; ++j)
                res(i, j) = (
                    get<0>((*im)(i, j)) * 0.299 +
                    get<1>((*im)(i, j)) * 0.587 +
                    get<2>((*im)(i, j)) * 0.114
                );

        // auto p = MonochromeToImage(res);
        Matrix<int> x_sobel = {
            {-1, 0, 1},
            {-2, 0, 2},
            {-1, 0, 1},
        };
        auto dx = custom2(res, x_sobel);

        Matrix<int> y_sobel = {
            {-1, -2, -1},
            {0, 0, 0},
            {1, 2, 1},
        };
        auto dy = custom2(res, y_sobel);

        Matrix<double> grad(res.n_rows, res.n_cols);
        for (int i = 0; i < grad.n_rows; ++i)
            for (int j = 0; j < grad.n_cols; ++j) {
                grad(i, j) = sqrt(dx(i, j) * dx(i, j) + dy(i, j) * dy(i, j));
            }

        Matrix<float> angle(res.n_rows, res.n_cols);
        for (int i = 0; i < angle.n_rows; ++i)
            for (int j = 0; j < angle.n_cols; ++j) {
                angle(i, j) = atan2(dy(i, j), dx(i, j));
            }
        
        vector<float> one_image_features;
        int shift = 4;
        int sectors = 10;
        for (int i = 0; i < grad.n_rows; i += shift)
            for (int j = 0; j < grad.n_cols; j += shift) {
                vector<float> local(sectors);
                for (int ii = i; ii < min(i + shift, grad.n_rows); ++ii)
                    for (int jj = j; jj < min(j + shift, grad.n_cols); ++jj) {
                        auto alpha = angle(ii, jj);
                        local[int((alpha + M_PI / 2) / (2 * M_PI) * sectors)] += grad(ii, jj);
                    }
                float sum = 1e-9;
                for (auto &k : local) {
                    sum += k * k;
                }
                sum = sqrt(sum);
                for (auto &k : local)
                    k /= sum;
                copy(local.begin(), local.end(), back_inserter(one_image_features));
            }

        // save_image(*im, "im.bmp");
        // save_image(MonochromeToImage(res), "res.bmp");
        // save_image(MonochromeToImage(normalize(dy)), "dy.bmp");
        // save_image(MonochromeToImage(normalize(dx)), "dx.bmp");
        // save_image(MonochromeToImage(normalize(grad)), "grad.bmp");

        // PLACE YOUR CODE HERE
        // Remove this sample code and place your feature extraction code here
        // one_image_features.push_back(1.0);
        features->push_back(make_pair(one_image_features, data_set[image_idx].second));
        // for (auto &i : one_image_features)
        //     cerr << i << " ";

        // End of sample code

    }
}

// Clear dataset structure
void ClearDataset(TDataSet* data_set) {
        // Delete all images from dataset
    for (size_t image_idx = 0; image_idx < data_set->size(); ++image_idx)
        delete (*data_set)[image_idx].first;
        // Clear dataset
    data_set->clear();
}

// Train SVM classifier using data from 'data_file' and save trained model
// to 'model_file'
void TrainClassifier(const string& data_file, const string& model_file) {
        // List of image file names and its labels
    TFileList file_list;
        // Structure of images and its labels
    TDataSet data_set;
        // Structure of features of images and its labels
    TFeatures features;
        // Model which would be trained
    TModel model;
        // Parameters of classifier
    TClassifierParams params;
    
        // Load list of image file names and its labels
    LoadFileList(data_file, &file_list);
        // Load images
    LoadImages(file_list, &data_set);
        // Extract features from images
    ExtractFeatures(data_set, &features);

        // PLACE YOUR CODE HERE
        // You can change parameters of classifier here
    params.C = 0.01;
    TClassifier classifier(params);
        // Train classifier
    classifier.Train(features, &model);
        // Save model to file
    model.Save(model_file);
        // Clear dataset structure
    ClearDataset(&data_set);
}

// Predict data from 'data_file' using model from 'model_file' and
// save predictions to 'prediction_file'
void PredictData(const string& data_file,
                 const string& model_file,
                 const string& prediction_file) {
        // List of image file names and its labels
    TFileList file_list;
        // Structure of images and its labels
    TDataSet data_set;
        // Structure of features of images and its labels
    TFeatures features;
        // List of image labels
    TLabels labels;

        // Load list of image file names and its labels
    LoadFileList(data_file, &file_list);
        // Load images
    LoadImages(file_list, &data_set);
        // Extract features from images
    ExtractFeatures(data_set, &features);

        // Classifier 
    TClassifier classifier = TClassifier(TClassifierParams());
        // Trained model
    TModel model;
        // Load model from file
    model.Load(model_file);
        // Predict images by its features using 'model' and store predictions
        // to 'labels'
    classifier.Predict(features, model, &labels);

        // Save predictions
    SavePredictions(file_list, labels, prediction_file);
        // Clear dataset structure
    ClearDataset(&data_set);
}

YUV toYUV(const RGB &rgb) {
    int R, G, B;
    tie(R, G, B) = rgb;
    return YUV(
        0.299 * R + 0.587 * G + 0.114 * B,
        -0.14713 * R - 0.28886 * G + 0.436 * B,
        0.615 * R - 0.51499 * G - 0.10001 * B
    );
}

RGB toRGB(const YUV &yuv) {
    int Y, U, V;
    tie(Y, U, V) = yuv;
    return RGB(
        1 * Y + 0 * U + 1.13983 * V,
        1 * Y - 0.39465 * U - 0.58060 * V,
        1 * Y + 2.03211 * U + 0 * V
    );
}

tuple<vector<vector<int>>, int> areas(Image &abs_grad)
{
    int len = abs_grad.n_rows * abs_grad.n_cols;
    DSU dsu(len + 1);
    auto exist = [&](int i, int j){ return 0 <= i && i < abs_grad.n_rows && 0 <= j && j < abs_grad.n_cols;};
    auto check = [&](int i, int j){ return exist(i, j) && (get<0>(abs_grad(i, j)));};
    vector<vector<int>> groups(abs_grad.n_rows, vector<int>(abs_grad.n_cols));
    int count = 1;
    dsu.make_set(0);
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

            groups[i][j] = current;
        }
    for (auto &i : groups)
        for (auto &j : i)
            j = dsu.find_set(j);
    return make_tuple(groups, count);
}


void do_detect(Image &im, const char *output_name) {
    // Image contrast = autocontrast(im, 0.1);
    Image conv(im.n_rows, im.n_cols);
    for (int i = 0; i < im.n_rows; ++i)
        for (int j = 0; j < im.n_cols; ++j)
            conv(i, j) = toYUV(im(i, j));

    auto check_color = [](YUV &in)->bool{
        int Y, U, V;
        tie(Y, U, V) = in;
        return Y > 2 && Y < 200 && (
            U - V < 0 && U + V > 10 ||
            Y < 128 && U - V > 30 && 2 * U + V > 30
        );
    };

    for (int i = 0; i < conv.n_rows; ++i)
        for (int j = 0; j < conv.n_cols; ++j)
            if (!(check_color(conv(i, j))))
                conv(i, j) = YUV(0, 0, 0);

    vector<vector<int>> groups;
    int cnt;
    tie(groups, cnt) = areas(conv);

    vector<pair<int, int>> min_corners(cnt, pair<int, int>(1e9, 1e9));
    vector<pair<int, int>> max_corners(cnt, pair<int, int>(-1, -1));

    for (int i = 0; i < conv.n_rows; ++i)
        for (int j = 0; j < conv.n_cols; ++j)
            if (groups[i][j]) {
                auto &minc = min_corners[groups[i][j]];
                minc.first = min(minc.first, i);
                minc.second = min(minc.second, j);

                auto &maxc = max_corners[groups[i][j]];
                maxc.first = max(maxc.first, i);
                maxc.second = max(maxc.second, j);
            }
    
    auto check = [](int left, int right)->bool{
        int l = right - left;
        return l > 10 && l < 1000;
    };
    auto check2 = [](int x, int y)->bool{
        int s = x * y;
        return s > 100 && s < 10000;
    };

    cnt = 0;

    for (int t = 1; t < int(min_corners.size()); ++t) {
        auto &minc = min_corners[t];
        auto &maxc = max_corners[t];

        if (minc.first > maxc.first || minc.second > maxc.second || !(check(minc.first, maxc.first) &&
              check(minc.second, maxc.second) &&
              check2(maxc.first - minc.first, maxc.second - minc.second)))
            continue;
        min_corners[cnt] = min_corners[t];
        max_corners[cnt] = max_corners[t];
        cnt++;
    }
    min_corners.resize(cnt);
    max_corners.resize(cnt);
    DSU dsu(cnt);
    for (int i = 0; i < cnt; ++i)
        dsu.make_set(i);

    auto cross = [](pair<int, int> A, pair<int, int> B, pair<int, int> C, pair<int, int> D)->bool{
        if (A > C) {
            swap(A, C);
            swap(B, D);
        }
        return A.first <= C.first && C.first <= B.first && (
               A.second <= C.second && C.second <= B.second ||
               A.second <= D.second && D.second <= B.second
               ) || C.first <= B.first && B.first <= D.first &&
                    C.second <= B.second && B.second <= D.second;
    };

    {
        auto &minc = min_corners;
        auto &maxc = max_corners;
        for (bool found = 1; found;) {
            found = 0;
            for (int i = 0; i < cnt; ++i)
                for (int j = i + 1; j < cnt; ++j) {
                    int t1 = dsu.find_set(i);
                    int t2 = dsu.find_set(j);
                    if (t1 == t2)
                        continue;
                    if (cross(minc[t1], maxc[t1], minc[t2], maxc[t2])) {
                        int k = dsu.union_sets(t1, t2);
                        minc[k].first = min(minc[t1].first, minc[t2].first);
                        minc[k].second = min(minc[t1].second, minc[t2].second);
                        maxc[k].first = max(maxc[t1].first, maxc[t2].first);
                        maxc[k].second = max(maxc[t1].second, maxc[t2].second);
                        found = 1;
                    }
                }
        }
    }

    Image out(conv.n_rows, conv.n_cols);
    for (int i = 0; i < conv.n_rows; ++i)
        for (int j = 0; j < conv.n_cols; ++j)
            out(i, j) = toRGB(conv(i, j));

    for (int t = 0; t < cnt; ++t)
        if (dsu.find_set(t) == t){
            auto &minc = min_corners[t];
            auto &maxc = max_corners[t];
            // cerr << minc.first << " " << minc.second << " " << maxc.first << " " << maxc.second << endl;
            for (int i = minc.first; i <= maxc.first; ++i)
                for (int j = minc.second; j <= maxc.second; ++j)
                    im(i, minc.second) = \
                    im(i, maxc.second) = \
                    im(minc.first, j) = \
                    im(maxc.first, j) = RGB(0, 255, 0);
        }
    normalize(im);
    save_image(im, output_name);
    cerr << output_name << endl;
}

void DetectImages(const string& data_file) {
    TFileList file_list;
    TDataSet data_set;

    LoadFileList(data_file, &file_list);

    LoadImages(file_list, &data_set);

    for (int i = 0; i < file_list.size(); ++i) {
        string name = file_list[i].first;
        name = name.substr(0, name.size() - 4) + ".res.bmp";

        do_detect(*data_set[i].first, name.c_str());
    }

    ClearDataset(&data_set);
}

int main(int argc, char** argv) {
        // Command line options parser
    ArgvParser cmd;
        // Description of program
    cmd.setIntroductoryDescription("Machine graphics course, task 2. CMC MSU, 2013.");
        // Add help option
    cmd.setHelpOption("h", "help", "Print this help message");
        // Add other options
    cmd.defineOption("data_set", "File with dataset",
        ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("model", "Path to file to save or load model",
        ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("predicted_labels", "Path to file to save prediction results",
        ArgvParser::OptionRequiresValue);
    cmd.defineOption("train", "Train classifier");
    cmd.defineOption("predict", "Predict dataset");
    cmd.defineOption("detect", "Detect dataset");
        
        // Add options aliases
    cmd.defineOptionAlternative("data_set", "d");
    cmd.defineOptionAlternative("model", "m");
    cmd.defineOptionAlternative("predicted_labels", "l");
    cmd.defineOptionAlternative("train", "t");
    cmd.defineOptionAlternative("predict", "p");

        // Parse options
    int result = cmd.parse(argc, argv);

        // Check for errors or help option
    if (result) {
        cout << cmd.parseErrorDescription(result) << endl;
        return result;
    }

        // Get values 
    string data_file = cmd.optionValue("data_set");
    string model_file = cmd.optionValue("model");
    bool train = cmd.foundOption("train");
    bool predict = cmd.foundOption("predict");
    bool detect = cmd.foundOption("detect");
        // If we need to train classifier
    if (train)
        TrainClassifier(data_file, model_file);
        // If we need to predict data
    if (predict) {
            // You must declare file to save images
        if (!cmd.foundOption("predicted_labels")) {
            cerr << "Error! Option --predicted_labels not found!" << endl;
            return 1;
        }
            // File to save predictions
        string prediction_file = cmd.optionValue("predicted_labels");
            // Predict data
        PredictData(data_file, model_file, prediction_file);
    }
    
    if (detect) {
        DetectImages(data_file);
    }
}