// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <vector>
#include <cmath>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/math_functions.hpp"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>


using std::max;

namespace caffe {


template <typename Dtype>
void DataAugmentationLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 1) << "Data Layer takes one input blobs.";
  CHECK_EQ(top->size(), 1) << "Data Layer takes one output blobs.";

  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();

  //num pixels to crop left/right and top/bottom
  int crop_size = this->layer_param_.data_param().crop_size();
  cropped_height = height - 2 * crop_size;
  CHECK_GE(cropped_height, 1) << "crop size greater than original";
  cropped_width = width - 2 * crop_size;
  CHECK_GE(cropped_width, 1) << "crop size greater than original";

  (*top)[0]->Reshape(num, channels, cropped_height, cropped_width);
}

template <typename Dtype>
Dtype DataAugmentationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {

  const int crop_size = this->layer_param_.data_param().crop_size();
  const bool mirror_enabled = this->layer_param_.data_param().mirror();

  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();

  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();

  for (int item_id = 0; item_id < num; ++item_id) {
    int h_off, w_off, w_idx;
    bool mirror;
    if (Caffe::phase() == Caffe::TRAIN) {
      //take random crop:
      h_off = caffe_rng_rand() % (2*crop_size+1);
      w_off = caffe_rng_rand() % (2*crop_size+1);
      if (mirror_enabled && caffe_rng_rand() % 2) {
        mirror = true;
      } else {
        mirror = false;
      }

    } else {
      //take center crop:
      h_off = crop_size;
      w_off = crop_size;
      mirror = false;
    }
    for (int c = 0; c < channels; ++c) {
      for (int h = 0; h < this->cropped_height; ++h) {
        for (int w = 0; w < this->cropped_width; ++w)  {
          int bottom_idx;
          if (mirror) {
            bottom_idx = item_id * (channels * width * height) + c * (height*width)
              + (h_off+h) * width + width - 1 - w_off - w;
          } else {
            bottom_idx = item_id * (channels * width * height) + c * (height*width)
              + (h_off+h)*width + (w_off + w);
          }
          int top_idx = item_id * (channels * this->cropped_height * this->cropped_width)
            + c * (this->cropped_height * this->cropped_width) + h * this->cropped_width + w;
          top_data[top_idx] = bottom_data[bottom_idx];
        }
      }
    }
  }

  return Dtype(0);
}

    /*cv::Rect cropRect(0, 0, width-8, height-8);
  cv::Mat adj_cv_img_cropped;
  adj_cv_img_cropped = adj_cv_img(cropRect);

  cv::imwrite("test_cropped.png", adj_cv_img_cropped);

  cv::resize(adj_cv_img_cropped, adj_cv_img_cropped, cv::Size(height, width));
  cv::imwrite("test_cropped_interpol.png", adj_cv_img_cropped);

  //http://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders

  cv::Point2f center(height/2., width/2.);
  cv::Mat rot_mat = cv::getRotationMatrix2D(center, 30, 1);
  cv::warpAffine(adj_cv_img, adj_cv_img, rot_mat, cv::Size(height, width));

  cv::imwrite("test_rotated.png", adj_cv_img);*/

  //gaussian noisy
  //boost::mt19937 *rng = new boost::mt19937();
  //rng->seed(time(NULL));
  //boost::normal_distribution<> distribution(0, 2);
  //boost::variate_generator< boost::mt19937, boost::normal_distribution<> > dist(*rng, distribution);






/*template <typename Dtype>
void DataAugmentationLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 1) << "Data Layer takes one input blobs.";
  CHECK_EQ(top->size(), 1) << "Data Layer takes one output blobs.";

  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();

  //num pixels to crop left/right and top/bottom
  int crop_size = this->layer_param_.data_param().crop_size();
  cropped_height = height - 2 * crop_size;
  CHECK_GE(cropped_height, 1) << "crop size greater than original";
  cropped_width = width - 2 * crop_size;
  CHECK_GE(cropped_width, 1) << "crop size greater than original";

  //we scale up to get back to the original size:
  (*top)[0]->Reshape(num, channels, height, width);
}

void rotated_rect_max_area(int w, int h, double angle, int& w_out, int& h_out) {
  //balenzly copied from:
  //http://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
  if(w <= 0 || h <= 0) {
    w_out = 0;
    h_out = 0;
    return;
  }
  bool width_is_longer = (w >= h) ? true : false;
  int side_long;
  int side_short;
  if (width_is_longer) {
    side_long = w;
    side_short = h;
  } else {
    side_long = h;
    side_short = w;
  }

  // since the solutions for angle, -angle and 180-angle are all the same,
  // it suffices to look at the first quadrant and the absolute values of sin,cos:
  double sin_a = fabs(sin(angle));
  double cos_a = fabs(cos(angle));

  if (side_short <= 2.*sin_a*cos_a*side_long) {
    double x = 0.5*side_short;
    if (width_is_longer) {
      w_out = (int) (x/sin_a);
      h_out = (int) (x/cos_a);
    } else {
      w_out = (int) (x/cos_a);
      h_out = (int) (x/sin_a);
    }
  } else {
    double cos_2a = cos_a*cos_a - sin_a*sin_a;
    w_out = (int) ((w*cos_a - h*sin_a)/cos_2a);
    h_out = (int) ((h*cos_a - w*sin_a)/cos_2a);
  }
}*/


/*template <typename Dtype>
Dtype DataAugmentationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {

  const int crop_size = this->layer_param_.data_param().crop_size();
  const bool mirror_enabled = this->layer_param_.data_param().mirror();

  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();

  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();

  CHECK_EQ(bottom[0]->count(), (*top)[0]->count());

  if (Caffe::phase() == Caffe::TEST) {  
    //just copy:
    int count =  bottom[0]->count();
    for (int i = 0; i < count; ++i)
    {
      top_data[i] = bottom_data[i];
    }
  } else {
    for (int item_id = 0; item_id < num; ++item_id) {

      const Dtype* image_data = &bottom_data[item_id * (channels * width * height)];
      Dtype* top_image_data = &top_data[item_id * (channels * width * height)];

      if (caffe_rng_rand() % 2) {
        //do a simple copy:
        int image_size = channels * width * height;
        for (int i = 0; i < image_size; ++i)
        {
          top_image_data[i] = image_data[i];
        }
        continue;
      }

      //otherwise we augment:
      bool mirror;
      if (mirror_enabled && caffe_rng_rand() % 2) {
        mirror = true;
      } else {
        mirror = false;
      }

      //convert to Opencv
      cv::Mat cv_img(height, width, CV_64FC3);
      cv_img.setTo(0);

      for (int c = 0; c < channels; ++c){
        for (int h = 0; h < height ; ++h){
          for (int w = 0; w < width; ++w){
            Dtype data;
            if(mirror) {
              data = image_data[c * (height * width) + h * width + width - 1 - w];
            } else {
              data = image_data[c * (height * width) + h * width + w];
            }
            cv_img.at<cv::Vec3d>(h, w)[c] = data;
          }
        }
      }

      //take random crop:
      int h_off = caffe_rng_rand() % (2*crop_size+1);
      int w_off = caffe_rng_rand() % (2*crop_size+1);
      
      cv::Rect cropRect(h_off, w_off, this->cropped_height, this->cropped_width);
      cv::Mat cv_img_cropped = cv_img(cropRect);

      //rotate:
      cv::Point2f center(height/2., width/2.);
      int angle = caffe_rng_rand() % (10+1) - 5;
      cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, 1);
      cv::warpAffine(cv_img_cropped, cv_img_cropped, rot_mat, cv::Size(height, width));
      //int out_w
      //rotated_rect_max_area(this->cropped_height, this->cropped_width, 30, );

      //scale back up to the original size:
      cv::resize(cv_img_cropped, cv_img, cv::Size(height, width));

      //copy back:
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height ; ++h) {
          for (int w = 0; w < width; ++w) {
            Dtype data = cv_img.at<cv::Vec3d>(h, w)[c];
            top_image_data[c * (height * width) + h * width + w] = data;
          }
        }
      }
    }
  }
  return Dtype(0);
}*/

INSTANTIATE_CLASS(DataAugmentationLayer);


}  // namespace caffe
