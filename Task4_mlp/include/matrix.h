#include <sstream>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>

using namespace std;

void load_data_from_line(const string &line, vector<half> &data) {
  stringstream ss(line);
  float d;
  while (ss >> d) {
    half dd = __float2half(d);
    data.push_back(dd);
  }
}

template <class T>
void load_data_from_line(const string &line, vector<T> &data) {
  stringstream ss(line);
  T d;
  while (ss >> d) {
    data.push_back(d);
  }
}

template <class T>
struct Matrix {
  vector<T> data;
  vector<int> shape; // [row_cnt, col_cnt]
  Matrix(string path) {
    // open(path);
    ifstream ifs(path);
    string line;
    getline(ifs, line);
    fflush(stdout);
    load_data_from_line(line, shape);
    getline(ifs, line);
    load_data_from_line(line, data);
    ifs.close();
  }

  void log_info() {
    if (shape.size() == 2) {
      printf("shape: [%d, %d]\n", shape[0], shape[1]);
    } else {
      printf("shape: [%d]\n", shape[0]);
    }
  }
  
  bool operator==(const Matrix &rhs) const {
    return shape == rhs.shape && data == rhs.data;
  }
  bool operator!=(const Matrix &rhs) const { return !(*this == rhs); }
};