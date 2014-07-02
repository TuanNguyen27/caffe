// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <vector>
#include <cmath>

#include "google/protobuf/descriptor.h"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/math_functions.hpp"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>

#include <iostream>
#include <fstream>
#include <omp.h>

using std::max;

namespace caffe {
  
template <typename T> int sgn(T val) {
  return (T(0) < val) - (val < T(0));
}
  
template <typename Dtype>
Dtype caffe_rng_generate(const RandomGeneratorParameter param) {
  const std::string rand_type =  param.rand_type();
  //std::cout << rand_type << " " << rand_type.compare("uniform") << " " << rand_type.compare("gaussian") << " " << rand_type.compare("bernoulli");
  Dtype rand;
  if (rand_type.compare("uniform") == 0) {
    float tmp;
    if (param.spread() > 0.)
      caffe_rng_uniform(1, param.mean() - param.spread(), param.mean() + param.spread(), &tmp);
    else
      tmp = param.mean();
    if (param.exp())
      tmp = exp(tmp);
    rand = static_cast<Dtype>(tmp);
  }
  else if (rand_type.compare("gaussian") == 0) {
    float tmp;
    if (param.spread() > 0.)
      caffe_rng_gaussian(1, param.mean(), param.spread(), &tmp);
    else
      tmp = param.mean();
    if (param.exp())
      tmp = exp(tmp);
    rand = static_cast<Dtype>(tmp);
  }
  else if (rand_type.compare("bernoulli") == 0) {
    int tmp;
    if (param.prob() > 0.)
      caffe_rng_bernoulli(1, param.prob(), &tmp);
    else
      tmp = 0;
    rand = static_cast<Dtype>(tmp);
  }
  else if (rand_type.compare("uniform_bernoulli") == 0) {
    float tmp1;
    int tmp2;
    
    if (param.spread() > 0.) 
      caffe_rng_uniform(1, param.mean() - param.spread(), param.mean() + param.spread(), &tmp1);
    else
      tmp1 = param.mean();
    
    if (param.prob() > 0.)
      caffe_rng_bernoulli(1, param.prob(), &tmp2);
    else
      tmp2 = 0;
    
    tmp1 = tmp1 * static_cast<float>(tmp2);
    
    if (param.exp())
      tmp1 = exp(tmp1);
    
    rand = static_cast<Dtype>(tmp1);
  }
  else if (rand_type.compare("gaussian_bernoulli") == 0) {
    float tmp1;
    int tmp2;
    
    if (param.spread() > 0.) 
      caffe_rng_gaussian(1, param.mean() - param.spread(), param.mean() + param.spread(), &tmp1);
    else
      tmp1 = param.mean();
    
    if (param.prob() > 0.)
      caffe_rng_bernoulli(1, param.prob(), &tmp2);
    else
      tmp2 = 0;
    
    tmp1 = tmp1 * static_cast<float>(tmp2);
    
    if (param.exp())
      tmp1 = exp(tmp1);
    
    rand = static_cast<Dtype>(tmp1);
  }
  else {
    LOG(ERROR) << "Unknown random type " << rand_type;
    rand = NAN;
  }
  return rand;
}  


template <typename Dtype>
void DataAugmentationLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 1) << "Data Layer takes one input blobs.";
  CHECK_EQ(top->size(), 1) << "Data Layer takes one output blobs.";

  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  
//   AugmentationParameter augmentation_param = this->layer_param_.augmentation_param();
//   
//   if (augmentation_param.has_rotate()) {
//     std::cout << "Rotation: " << std::endl;
//     for(int i=0;i<20;i++)
//       std::cout << "  " << caffe_rng_generate<float>(augmentation_param.rotate()) << std::endl;
//   }
//   if (augmentation_param.has_translate()) {
//     std::cout << "Translation: " << std::endl;
//     for(int i=0;i<20;i++)
//       std::cout << "  " << caffe_rng_generate<float>(augmentation_param.translate()) << std::endl;
//   }
//   if (augmentation_param.has_mirror()) {
//     std::cout << "Mirror: " << std::endl;
//     for(int i=0;i<20;i++)
//       std::cout << "  " << caffe_rng_generate<bool>(augmentation_param.mirror()) << std::endl;
//   }  
//   std::cin.get();

  //num pixels to crop left/right and top/bottom
  int crop_size = this->layer_param_.augmentation_param().crop_size();
  CHECK_GE(height, crop_size) << "crop size greater than original";
  CHECK_GE(width, crop_size) << "crop size greater than original";
  
  cropped_height = crop_size;
  cropped_width = crop_size;

  (*top)[0]->Reshape(num, channels, crop_size, crop_size);
}

template <typename Dtype>
Dtype DataAugmentationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {

  AugmentationParameter aug = this->layer_param_.augmentation_param();
  if (!aug.has_crop_size())
    LOG(ERROR) << "Please enter crop_size if you want to perform augmentation";
  const int crop_size = aug.crop_size();
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();

  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data(); 
  
  
  std::string write_augmented;
  if (aug.has_write_augmented())
    write_augmented = aug.write_augmented();
  else
    write_augmented = std::string("");
  
  
  
#pragma omp parallel for
  for (int item_id = 0; item_id < num; ++item_id) {
    
    int x, y, c, top_idx, bottom_idx, h_off, w_off;
    float x1, y1, x2, y2;
    
    bool do_spatial_transform, do_chromatic_transform;
    
    //   We only do transformations during training.
    if (Caffe::phase() != Caffe::TRAIN) {
      do_spatial_transform   = false;
      do_chromatic_transform = false;
    }
    
    Dtype angle = 0.;
    Dtype zoom_coeff = 1.;
    Dtype dx = 0.;
    Dtype dy = 0.;
    bool mirror = false;  
    
    Dtype lmult_pow_coeff = 1.;
    Dtype lmult_mult_coeff = 1.;
    Dtype lmult_add_coeff = 0.;  
    Dtype pow_coeffs [3] = {1., 1., 1.};
    Dtype mult_coeffs [3] = {1., 1., 1.};
    Dtype add_coeffs [3] = {0., 0., 0.};  
    Dtype pow_factor = 1.;
    Dtype mult_factor = 1.;
    Dtype add_factor = 1.;
    
    //LOG(INFO) <<  " === thread " << omp_get_thread_num() << "/" << omp_get_num_threads() << " === ";
    
    do_spatial_transform   = (aug.has_mirror()    || aug.has_translate()  || aug.has_rotate()    || aug.has_zoom());
    do_chromatic_transform = (aug.has_lmult_pow() || aug.has_lmult_mult() || aug.has_lmult_add() || 
                                 aug.has_sat_pow()   || aug.has_sat_mult()   || aug.has_sat_add()   ||
                                 aug.has_col_pow()   || aug.has_col_mult()   || aug.has_col_add()   ||
                                 aug.has_ladd_pow()  || aug.has_ladd_mult()  || aug.has_ladd_add());
    
    // sample the parameters of the transoformations  
    if (do_spatial_transform) {
      int counter = 0;
      int max_num_tries = 20;    
      int good_params = 0;
      
      // try to sample parameters for which transformed image doesn't go outside the borders of the original one
      // in order to do check this, just apply the transformations to 4 corners
      while (good_params < 4 && counter < max_num_tries) {
        good_params = 0;
        if (aug.has_rotate())
          angle = caffe_rng_generate<float>(aug.rotate());
        if (aug.has_zoom())
          zoom_coeff = caffe_rng_generate<float>(aug.zoom());
        if (aug.has_translate()) {
          dx = caffe_rng_generate<float>(aug.translate());
          dy = caffe_rng_generate<float>(aug.translate());
        }
        if (aug.has_mirror())
          mirror = caffe_rng_generate<bool>(aug.mirror());
  
        //LOG(INFO) << "angle: " << angle << ", zoom: " << zoom_coeff << ", dx: " << dx << ", dy: " << dy << ", mirror: " << mirror;
        
        for (x = 0; x < crop_size; x += crop_size-1) {
          for (y = 0; y < crop_size; y += crop_size-1) {
            // move the origin and mirror
            if (mirror) {
              x1 =  static_cast<Dtype>(x) - .5 * static_cast<Dtype>(crop_size);
              y1 = -static_cast<Dtype>(y) + .5 * static_cast<Dtype>(crop_size);            
            } 
            else {
              x1 = static_cast<Dtype>(x) - .5 * static_cast<Dtype>(crop_size);
              y1 = static_cast<Dtype>(y) - .5 * static_cast<Dtype>(crop_size);
            }
            // rotate
            x2 =  cos(angle) * x1 - sin(angle) * y1;
            y2 =  sin(angle) * x1 + cos(angle) * y1;
            // translate
            x2 = x2 + dx * static_cast<Dtype>(crop_size);
            y2 = y2 + dy * static_cast<Dtype>(crop_size);
            // zoom
            x2 = x2 / zoom_coeff;
            y2 = y2 / zoom_coeff;
            // move the origin back
            x2 = x2 + .5 * static_cast<Dtype>(width);
            y2 = y2 + .5 * static_cast<Dtype>(height);
            
            if (!(floor(x2) < 0 || floor(x2) > static_cast<Dtype>(width - 2) || floor(y2) < 0 || floor(y2) > static_cast<Dtype>(height - 2)))
                good_params++;
            //mexPrintf(" (%f,%f) ", x2, y2);
          }
        }
        //mexPrintf("\n");
        counter++;
      }
      if (counter >= max_num_tries) {
        angle=0.;
        zoom_coeff=1.;
        dx=0.;
        dy=0.;
        mirror = false;
      }
      
      if (write_augmented.size()) { 
        if (do_spatial_transform)
          LOG(INFO) << " Augmenting. angle: " << angle << ", zoom: " << zoom_coeff << ", dx: " << dx << ", dy: " << dy << ", mirror: " << mirror;
        else
          LOG(INFO) << "Couldn't find appropriate spatial augmentation parameters";
      }
    } 
    
    if (do_chromatic_transform) {
      if (aug.has_lmult_pow())
        lmult_pow_coeff = caffe_rng_generate<float>(aug.lmult_pow());
      if (aug.has_lmult_mult())
        lmult_mult_coeff = caffe_rng_generate<float>(aug.lmult_mult());
      if (aug.has_lmult_add())
        lmult_add_coeff = caffe_rng_generate<float>(aug.lmult_add());
      
      if (aug.has_ladd_pow())
        pow_coeffs[0] = caffe_rng_generate<float>(aug.ladd_pow());
      if (aug.has_ladd_mult())
        mult_coeffs[0] = caffe_rng_generate<float>(aug.ladd_mult());
      if (aug.has_ladd_add())
        add_coeffs[0] = caffe_rng_generate<float>(aug.ladd_add());
      
      for (c=1; c<3; c++) {
        if (aug.has_sat_pow())
          pow_coeffs[c] = caffe_rng_generate<float>(aug.sat_pow());
        if (aug.has_sat_mult())
          mult_coeffs[c] = caffe_rng_generate<float>(aug.sat_mult());
        if (aug.has_sat_add())
          add_coeffs[c] = caffe_rng_generate<float>(aug.sat_add());
      }
      
      if (aug.has_col_pow())
        pow_factor = caffe_rng_generate<float>(aug.col_pow());
      if (aug.has_col_mult())
        mult_factor = caffe_rng_generate<float>(aug.col_mult());
      if (aug.has_col_add())
        add_factor = caffe_rng_generate<float>(aug.col_add());
      
      pow_coeffs[1] = pow_coeffs[1] * pow_factor;
      pow_coeffs[2] = pow_coeffs[2] / pow_factor;
      mult_coeffs[1] = mult_coeffs[1] * mult_factor;
      mult_coeffs[2] = mult_coeffs[2] / mult_factor;
      add_coeffs[1] = add_coeffs[1] + add_factor;
      add_coeffs[2] = add_coeffs[2] - add_factor;
      
      if (write_augmented.size()) {
        LOG(INFO) << "Augmenting. lmult_pow: " << lmult_pow_coeff << ", lmult_mult: " << lmult_mult_coeff << ", lmult_add: " << lmult_add_coeff;      
      }
    }
    
    bool do_rotate = aug.has_rotate();
    bool do_translate = aug.has_translate();
    bool do_mirror = aug.has_mirror();
    bool do_zoom = aug.has_zoom();
    
    if (do_rotate)
      do_rotate = (fabs(angle) >1e-2);
    if (do_translate)
      do_translate = ( fabs(dx) > 1e-2 || fabs(dy) > 1e-2) ;
    if (do_mirror)
      do_mirror = mirror;
    if (do_zoom)
      do_zoom = (fabs(zoom_coeff - 1.) >1e-2);
    
    do_spatial_transform = (do_rotate || do_translate || do_mirror || do_zoom);
    
    bool do_pow [3] = {false, false, false};
    bool do_mult [3] = {false, false, false};
    bool do_add [3] = {false, false, false};
    bool do_lmult_pow = aug.has_lmult_pow();
    bool do_lmult_add = aug.has_lmult_add();
    bool do_lmult_mult = aug.has_lmult_mult();;
    
    do_chromatic_transform = false;
    
    for (c=0; c<3; c++) {
      if (((c==1 || c==2) && (aug.has_sat_pow() || aug.has_col_pow())) || (c==0 && aug.has_ladd_pow()))
        do_pow[c] = true;
      if (((c==1 || c==2) && (aug.has_sat_add() || aug.has_col_add())) || (c==0 && aug.has_ladd_add()))
        do_add[c] = true;
      if (((c==1 || c==2) && (aug.has_sat_mult() || aug.has_col_mult())) || (c==0 && aug.has_ladd_mult()))
        do_mult[c] = true;
      if (do_pow[c])
        do_pow[c] = (fabs(pow_coeffs[c] - 1.) > 1e-2);
      if (do_add[c])
        do_add[c] = (fabs(add_coeffs[c]) > 1e-2);
      if (do_mult[c])
        do_mult[c] = (fabs(mult_coeffs[c] - 1.) > 1e-2);
      do_chromatic_transform = (do_chromatic_transform || do_pow[c] || do_add[c] || do_mult[c]);
    }
    if (do_lmult_pow)
      do_lmult_pow = (fabs(lmult_pow_coeff - 1.) > 1e-2);
    if (do_lmult_add)
      do_lmult_add = (fabs(lmult_add_coeff) > 1e-2);
    if (do_lmult_mult)
      do_lmult_mult = (fabs(lmult_mult_coeff - 1.) > 1e-2);
    do_chromatic_transform = (do_chromatic_transform || do_lmult_pow || do_lmult_add || do_lmult_mult);
    
//     LOG(INFO) << "item_id " << item_id << " do_translate " << do_translate << " do_rotate " << do_rotate << " do_zoom " << do_zoom;
//     LOG(INFO) << "angle: " << angle << ", zoom: " << zoom_coeff << ", dx: " << dx << ", dy: " << dy << ", mirror: " << mirror;
    // actually apply the transformation
    if (do_spatial_transform) { 
//       LOG(INFO) << " >>> do spatial transform " << item_id;
      int i00,i01,i10,i11;
      for (x = 0; x < crop_size; x++) {
        for (y = 0; y < crop_size; y++) {
          // move the origin and mirror
          if (mirror) {
            x1 =  static_cast<Dtype>(x) - .5 * static_cast<Dtype>(crop_size);
            y1 = -static_cast<Dtype>(y) + .5 * static_cast<Dtype>(crop_size);            
          } 
          else {
            x1 = static_cast<Dtype>(x) - .5 * static_cast<Dtype>(crop_size);
            y1 = static_cast<Dtype>(y) - .5 * static_cast<Dtype>(crop_size);
          }
          // rotate
          if (do_rotate) {
            x2 =  cos(angle) * x1 - sin(angle) * y1;
            y2 =  sin(angle) * x1 + cos(angle) * y1;
          }
          else {
            x2 = x1;
            y2 = y1;
          }
          // translate
          if (do_translate) {
            x2 = x2 + dx * static_cast<Dtype>(crop_size);
            y2 = y2 + dy * static_cast<Dtype>(crop_size);
          }
          // zoom
          if (do_zoom) {
            x2 = x2 / zoom_coeff;
            y2 = y2 / zoom_coeff;
          }
          // move the origin back
          x2 = x2 + .5 * static_cast<Dtype>(width);
          y2 = y2 + .5 * static_cast<Dtype>(height);
          

          for (c = 0; c < channels; c++) {
            top_idx = ((item_id*channels + c)*crop_size + x)*crop_size + y;
            if (floor(x2) < 0. || floor(x2) > static_cast<Dtype>(width - 2) || floor(y2) < 0. || floor(y2) > static_cast<Dtype>(height - 2))
              top_data[top_idx] = 0.;
            else {
              if (do_rotate || do_zoom) {
                i00 = static_cast<int>(((item_id*channels + c) * width +  floor(x2)) *height + floor(y2));
                i01 = i00 + 1;
                i10 = i00 + height;
                i11 = i00 + height + 1;
                
                top_data[top_idx] = bottom_data[i00] * ((floor(x2)+1)  - x2) * ((floor(y2)+1)  - y2) +
                                    bottom_data[i01] * ((floor(x2)+1)  - x2) * (y2 - floor(y2))      +
                                    bottom_data[i10] * (x2 - floor(x2))      * ((floor(y2)+1)  - y2) +
                                    bottom_data[i11] * (x2 - floor(x2))      * (y2 - floor(y2));                
              } 
              else {
                i00 = static_cast<int>(((item_id*channels + c) * width +  floor(x2)) *height + floor(y2));              
                top_data[top_idx] = bottom_data[i00];
              }
            }         
            // TODO: return the mean when end debugging
            //top_data[i] = (top_data[i] - 127.5) * scale;
          }
          //mexPrintf(" (%f,%f) ", x2, y2);        
        }
      }
    }
    else {
      h_off = (height - crop_size)/2;
      w_off = (width - crop_size)/2;
      for (x = 0; x < crop_size; x++) {
        for (y = 0; y < crop_size; y++) {
          for (c = 0; c < channels; c++) {
            top_idx = ((item_id*channels + c)*crop_size + x)*crop_size + y;
            bottom_idx = ((item_id*channels + c)*width + x + w_off)*height + y + h_off;
            top_data[top_idx] = bottom_data[bottom_idx];
          }
        }
      }
    }
    
    if (do_chromatic_transform) {
//       LOG(INFO) << " >>> do chromatic transform " << item_id;
      Dtype l;
      Dtype rgb [3];
      Dtype eig [3];
      Dtype max_abs_eig[3] = {0., 0., 0.};
      Dtype max_abs_rgb[3] = {0., 0., 0.};
      const Dtype eigvec [9] = {0.5579, 0.5859, 0.5878, 0.8021, -0.1989, -0.5631, -0.2130, 0.7856, -0.5809};
      // compute max abs values of eigs (projections onto color space eigenvectors)
      for (x=0; x<crop_size; x++) {
        for (y=0; y<crop_size; y++) {
          for (c=0; c<channels; c++)
            rgb[c] = top_data[((item_id*channels + c)*crop_size + x)*crop_size + y];
          for (c=0; c<channels; c++) {
            eig[c] = eigvec[3*c] * rgb[0] + eigvec[3*c+1] * rgb[1] + eigvec[3*c+2] * rgb[2];
            if (fabs(eig[c]) > max_abs_eig[c])
              max_abs_eig[c] = fabs(eig[c]);
            if (fabs(rgb[c]) > max_abs_rgb[c])
              max_abs_rgb[c] = fabs(rgb[c]);
          }
        }
      }
      // actually apply the transform
      for (x=0; x<crop_size; x++) {
        for (y=0; y<crop_size; y++) {
          for (c=0; c<channels; c++)
            rgb[c] = top_data[((item_id*channels + c)*crop_size + x)*crop_size + y];
          for (c=0; c<channels; c++) {
            eig[c] = eigvec[3*c] * rgb[0] + eigvec[3*c+1] * rgb[1] + eigvec[3*c+2] * rgb[2];
            if ( max_abs_eig[c] > 1e-5 ) {
              eig[c] = eig[c] / max_abs_eig[c]; 
              if (do_pow[c])            
                eig[c] = static_cast<float>(sgn(eig[c])) * pow(fabs(eig[c]), pow_coeffs[c]);
              if (do_add[c])
                eig[c] = eig[c] + add_coeffs[c];
              if (do_mult[c])
                eig[c] = eig[c] * mult_coeffs[c];
              eig[c] = eig[c] * max_abs_eig[c]; 
            }
          }
          if (do_lmult_pow)
            l = pow(fabs(eig[0]), lmult_pow_coeff);
          else
            l = fabs(eig[0]);
          if (do_lmult_add) {
            l = l + lmult_add_coeff;
            if (l < 0.)
              l = 0.;
          }
          if (do_lmult_mult)
            l = l * lmult_mult_coeff;
          if ((do_lmult_pow || do_lmult_add || do_lmult_mult) && fabs(eig[0]) > 1e-5) {
            for (c=channels-1; c>=0; c--) {
              eig[c] = eig[c] / fabs(eig[0]) * l;
            }
          }
          for (c=0; c<channels; c++) {
            rgb[c] = eigvec[c] * eig[0] + eigvec[3+c] * eig[1] + eigvec[6+c] * eig[2];
            if (rgb[c] > aug.max_multiplier()*max_abs_rgb[c])
              rgb[c] = aug.max_multiplier()*max_abs_rgb[c];
            if (rgb[c] < -aug.max_multiplier()*max_abs_rgb[c])
              rgb[c] = -aug.max_multiplier()*max_abs_rgb[c];
            top_data[((item_id*channels + c)*crop_size + x)*crop_size + y] = rgb[c];
          }
          
        }
      } 
    }
    
  }
  
  if (write_augmented.size()) {  
    std::ofstream out_file (write_augmented.data(), std::ios::out | std::ios::binary);
    if (out_file.is_open()) { 
      uint32_t imsize[4];
      imsize[0] = num; 
      imsize[1] = channels; 
      imsize[2] = crop_size; 
      imsize[3] = crop_size;
      LOG(INFO) << "Writing blob size " << imsize[0] << "x" << imsize[1] << "x" << imsize[2] << "x" << imsize[3];
      out_file.write(reinterpret_cast<char*>(&imsize[0]), 4*4);
      out_file.write(reinterpret_cast<const char*>(top_data), imsize[0]*imsize[1]*imsize[2]*imsize[3]*sizeof(float));
      out_file.close();
      LOG(INFO) << " finished augmenting a batch === PAUSED === ";
      std::cout << " finished augmenting a batch === PAUSED === ";
      std::cin.get();
    }
    else
      LOG(INFO) << "WARNING: Could not open the file" << write_augmented;
  }
 
  
//   for (int item_id = 0; item_id < num; ++item_id) {
//     for (int c = 0; c < channels; ++c) {
//       for (int h = 0; h < this->cropped_height; ++h) {
//         for (int w = 0; w < this->cropped_width; ++w)  {
//           int bottom_idx;
//           if (mirror) {
//             bottom_idx = item_id * (channels * width * height) + c * (height*width)
//               + (h_off+h) * width + width - 1 - w_off - w;
//           } else {
//             bottom_idx = item_id * (channels * width * height) + c * (height*width)
//               + (h_off+h)*width + (w_off + w);
//           }
//           int top_idx = item_id * (channels * this->cropped_height * this->cropped_width)
//             + c * (this->cropped_height * this->cropped_width) + h * this->cropped_width + w;
//           top_data[top_idx] = bottom_data[bottom_idx];
//         }
//       }
//     }
//   }

  return Dtype(0);
}


INSTANTIATE_CLASS(DataAugmentationLayer);


}  // namespace caffe
