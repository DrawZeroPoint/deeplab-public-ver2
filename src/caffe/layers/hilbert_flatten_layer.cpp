#include <vector>

#include "caffe/layers/hilbert_flatten_layer.h"

namespace caffe {

template <typename Dtype>
void HilbertFlattenLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
                                 "allow in-place computation.";
  const int start_axis = bottom[0]->CanonicalAxisIndex(
        this->layer_param_.hilbert_flatten_param().axis());
  const int end_axis = bottom[0]->CanonicalAxisIndex(
        this->layer_param_.hilbert_flatten_param().end_axis());
  vector<int> top_shape;
  for (int i = 0; i < start_axis; ++i) {
    top_shape.push_back(bottom[0]->shape(i));
  }
  const int flattened_dim = bottom[0]->count(start_axis, end_axis + 1);
  top_shape.push_back(flattened_dim);
  for (int i = end_axis + 1; i < bottom[0]->num_axes(); ++i) {
    top_shape.push_back(bottom[0]->shape(i));
  }
  
  top[0]->Reshape(top_shape);
  CHECK_EQ(top[0]->count(), bottom[0]->count());
}

template <typename Dtype>
void HilbertFlattenLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
  int n = bottom[0]->shape(0);
  int c = bottom[0]->shape(1);
  int h = bottom[0]->shape(2);
  int w = bottom[0]->shape(3);
  if (divide_two(h) && divide_two(w)) {
    Dtype* top_data = top[0]->mutable_cpu_data();
    for (int k = 0; k < n; ++k) {
      for (int cn = 0; cn < c; ++cn) {
        for (int row = 0; row < h; ++row) {
          for (int col = 0; col < w; ++col) {
            int d = xy2d(w, col, row);
            const Dtype* data = bottom[0]->cpu_data_at(k, cn, row, col);
            top_data[top[0]->offset(k, d)] = *data;
          }
        }
      }
    }
  }
  else {
    int f = two_floor(w);
    Dtype* top_data = top[0]->mutable_cpu_data();
    for (int k = 0; k < n; ++k) {
      for (int cn = 0; cn < c; ++cn) {
        for (int row = 0; row < f; ++row) {
          for (int col = 0; col < f; ++col) {
            int d = xy2d(f, col, row);
            const Dtype* data = bottom[0]->cpu_data_at(k, cn, row, col);
            top_data[d] = *data;
          }
        }
      }
    }
  }
}

template <typename Dtype>
void HilbertFlattenLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                              const vector<bool>& propagate_down, 
                                              const vector<Blob<Dtype>*>& bottom) {
  int n = bottom[0]->shape(0);
  int c = bottom[0]->shape(1);
  int h = bottom[0]->shape(2);
  int w = bottom[0]->shape(3);
  if (divide_two(h) && divide_two(w)) {
    Dtype* top_diff = top[0]->mutable_cpu_diff();
    for (int k = 0; k < n; ++k) {
      for (int cn = 0; cn < c; ++cn) {
        for (int row = 0; row < h; ++row) {
          for (int col = 0; col < w; ++col) {
            int d = xy2d(w, col, row);
            const Dtype* diff = bottom[0]->cpu_diff_at(k, cn, row, col);
            top_diff[top[0]->offset(k, d)] = *diff;
          }
        }
      }
    }
  }
  else {
    int f = two_floor(w);
    Dtype* top_diff = top[0]->mutable_cpu_diff();
    for (int k = 0; k < n; ++k) {
      for (int cn = 0; cn < c; ++cn) {
        for (int row = 0; row < f; ++row) {
          for (int col = 0; col < f; ++col) {
            int d = xy2d(f, col, row);
            const Dtype* diff = bottom[0]->cpu_diff_at(k, cn, row, col);
            top_diff[top[0]->offset(k, d)] = *diff;
          }
        }
      }
    }
  }
}

template <typename Dtype>
bool HilbertFlattenLayer<Dtype>::divide_two(int n) {
  if (n % 2 != 0) { return false; }
  while (n / 2 > 0) {
    if (n / 2 == 1) {
      if (n % 2 != 0) {return false;}
      else {return true;}
    }
    n /= 2;
  }
  return false;
}

template <typename Dtype>
void HilbertFlattenLayer<Dtype>::rot(int n, int *x, int *y, int rx, int ry) {
  if (ry == 0) {
    if (rx == 1) {
      *x = n-1 - *x;
      *y = n-1 - *y;
    }
    
    //Swap x and y
    int t  = *x;
    *x = *y;
    *y = t;
  }
}

//convert (x,y) to d
template <typename Dtype>
int HilbertFlattenLayer<Dtype>::xy2d (int n, int x, int y) {
  int rx, ry, s, d=0;
  for (s=n/2; s>0; s/=2) {
    rx = (x & s) > 0;
    ry = (y & s) > 0;
    d += s * s * ((3 * rx) ^ ry);
    rot(s, &x, &y, rx, ry);
  }
  return d;
}

template <typename Dtype>
int HilbertFlattenLayer<Dtype>::two_floor(int n) {
  int t = 1;
  while (n/2 > 1) {
    n /= 2;
    t++;
  }
  return pow(2, t);
}

INSTANTIATE_CLASS(HilbertFlattenLayer);
REGISTER_LAYER_CLASS(HilbertFlatten);

}  // namespace caffe
