// Gain Robust Tracker - Joint Radiometric Calibration and Feature Tracking System
// Copyright (c) 2022, Algorithm Development Team. All rights reserved.
//
// This software was developed of Jacob.lsx

#include "gain_robust_tracker.h"

#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>

GainRobustTracker::GainRobustTracker(int patch_size, int pyramid_levels, bool only_exposure_ratio) {
  // Initialize patch size and pyramid levels
  patch_size_ = patch_size;
  pyramid_levels_ = pyramid_levels;
  only_exposure_ratio_ = only_exposure_ratio;
  radiometric_converged_ = false;
}

double GainRobustTracker::trackImagePyramids(const cv::Mat &frame_1,
                                             const cv::Mat &frame_2,
                                             const std::vector<cv::Point2f> &pts_1,
                                             std::vector<cv::Point2f> &pts_2,
                                             std::vector<uchar> &point_status) {
  // If the radiometric parameters have converged, the image is first radiometrically corrected
  // and then tracking only with exposure.
  cv::Mat frame_new, frame_old;
  if (radiometric_converged_) {
    frame_old = mapImage(frame_1);
    frame_new = mapImage(frame_2);
  } else {
    frame_old = frame_1;
    frame_new = frame_2;
  }

  // ALL point valid in the beginning of tracking
  std::vector<uchar> point_validity(pts_1.size(), 1);

  // Calculate image pyramid of frame 1 and frame 2
  std::vector<cv::Mat> new_pyramid;
  cv::buildPyramid(frame_new, new_pyramid, pyramid_levels_);

  std::vector<cv::Mat> old_pyramid;
  cv::buildPyramid(frame_old, old_pyramid, pyramid_levels_);

  // Temporary vector to update tracking estimates over time
  std::vector<cv::Point2f> tracking_estimates = pts_1;

  double all_exp_estimates = 0.0;
  int nr_estimates = 0;

  // Iterate all pyramid levels and perform gain robust KLT on each level (coarse to fine)
  for (int level = (int)new_pyramid.size()-1; level >= 0; level--) {
    // Scale the input points and tracking estimates to the current pyramid level
    std::vector<cv::Point2f> scaled_tracked_points;
    std::vector<cv::Point2f> scaled_tracking_estimates;
    for (int i = 0; i < pts_1.size(); i++) {
      cv::Point2f scaled_point;
      scaled_point.x = (float)(pts_1.at(i).x / pow(2, level));
      scaled_point.y = (float)(pts_1.at(i).y / pow(2, level));
      scaled_tracked_points.emplace_back(scaled_point);

      cv::Point2f scaled_estimate;
      scaled_estimate.x = (float)(tracking_estimates.at(i).x / pow(2, level));
      scaled_estimate.y = (float)(tracking_estimates.at(i).y / pow(2, level));
      scaled_tracking_estimates.emplace_back(scaled_estimate);
    }

    // Perform tracking on current level
    double exp_estimate = trackImageRobustPyr(old_pyramid.at(level),
                                              new_pyramid.at(level),
                                              scaled_tracked_points,
                                              scaled_tracking_estimates,
                                              point_validity);

    // Optional: Do something with the estimated exposure ratio
    // std::cout << "Estimated exposure ratio of current level: " << exp_estimate << std::endl;

    // Average estimates of each level later
    all_exp_estimates += exp_estimate;
    nr_estimates++;

    // Update the current tracking result by scaling down to pyramid level 0
    for (int i = 0; i < scaled_tracking_estimates.size(); i++) {
      if (point_validity.at(i) == 0)
        continue;

      cv::Point2f scaled_point;
      scaled_point.x = (float)(scaled_tracking_estimates.at(i).x * pow(2, level));
      scaled_point.y = (float)(scaled_tracking_estimates.at(i).y * pow(2, level));

      tracking_estimates.at(i) = scaled_point;
    }
  }

  // Write result to output vectors passed by reference
  std::swap(pts_2, tracking_estimates);
  std::swap(point_status, point_validity);

  // Average exposure ratio estimate
  double overall_exp_estimate = all_exp_estimates / nr_estimates;
  return overall_exp_estimate;
}

/**
 * For a reference on the meaning of the optimization variables and the overall concept of this function
 * refer to the photometric calibration paper
 * introducing gain robust KLT tracking by Kim et al.
 */
double GainRobustTracker::trackImageRobustPyr(const cv::Mat &old_image,
                                              const cv::Mat &new_image,
                                              const std::vector<cv::Point2f> &input_points,
                                              std::vector<cv::Point2f> &output_points,
                                              std::vector<uchar> &point_validity) {
  // Number of points to track
  int nr_points = static_cast<int>(input_points.size());

  // Update point locations which are update throughout the iterations
  if (output_points.size() == 0) {
    output_points = input_points;
  } else if (output_points.size() != input_points.size()) {
    std::cout << "ERROR - OUTPUT POINT SIZE != INPUT POINT SIZE!" << std::endl;
    return -1;
  }

  // Input image dimensions
  const int image_rows = new_image.rows;
  const int image_cols = new_image.cols;

  // Final exposure time estimate
  double K_total = 0.0;

  if (only_exposure_ratio_ || radiometric_converged_) {
    for (int round = 0; round < 1; round++) {
      // Get the currently valid points
      int nr_valid_points = getNrValidPoints(point_validity);

      // Allocate space for W & V matrices
      cv::Mat W(2*nr_valid_points, 1, CV_64F, 0.0);
      cv::Mat V(2*nr_valid_points, 1, CV_64F, 0.0);

      // Allocate space for U_inv and the original Us
      cv::Mat U_inv(2*nr_valid_points, 2*nr_valid_points, CV_64F, 0.0);
      std::vector<cv::Mat> Us;

      double lambda = 0;
      double m = 0;

      int absolute_point_index = -1;

      for (int p = 0; p < input_points.size(); p++) {
        if (point_validity.at(p) == 0)
          continue;

        absolute_point_index++;

        // Build U matrix
        cv::Mat U(2, 2, CV_64F, 0.0);

        // Bilinear image interpolation
        cv::Mat patch_intensities_1;
        cv::Mat patch_intensities_2;
        int absolute_patch_size = ((patch_size_ + 1) * 2 + 1);  // Todo: why patch_size_ + 1 ?
        cv::getRectSubPix(new_image, cv::Size(absolute_patch_size, absolute_patch_size), output_points.at(p), patch_intensities_2, CV_32F);
        cv::getRectSubPix(old_image, cv::Size(absolute_patch_size, absolute_patch_size), input_points.at(p), patch_intensities_1, CV_32F);

        // Go through image patch around this point
        for (int r = 0; r < 2 * patch_size_ + 1; r++) {
          for (int c = 0; c < 2 * patch_size_ + 1; c++) {
            // Fetch patch intensity values
            double i_frame_1 = patch_intensities_1.at<float>(1 + r, 1 + c);
            double i_frame_2 = patch_intensities_2.at<float>(1 + r, 1 + c);

            if (i_frame_1 < 1)
              i_frame_1 = 1;
            if (i_frame_2 < 1)
              i_frame_2 = 1;

            // Estimate patch gradient values
            double grad_1_x = (patch_intensities_1.at<float>(1+r, 1+c+1) - patch_intensities_1.at<float>(1+r, 1+c-1))/2;
            double grad_1_y = (patch_intensities_1.at<float>(1+r+1, 1+c) - patch_intensities_1.at<float>(1+r-1, 1+c))/2;

            double grad_2_x = (patch_intensities_2.at<float>(1+r, 1+c+1) - patch_intensities_2.at<float>(1+r, 1+c-1))/2;
            double grad_2_y = (patch_intensities_2.at<float>(1+r+1, 1+c) - patch_intensities_2.at<float>(1+r-1, 1+c))/2;

            double a = (1.0/i_frame_2)*grad_2_x + (1.0/i_frame_1)*grad_1_x;
            double b = (1.0/i_frame_2)*grad_2_y + (1.0/i_frame_1)*grad_1_y;
            double beta = log(i_frame_2/255.0) - log(i_frame_1/255.0);

            U.at<double>(0, 0) += 0.5*a*a;
            U.at<double>(1, 0) += 0.5*a*b;
            U.at<double>(0, 1) += 0.5*a*b;
            U.at<double>(1, 1) += 0.5*b*b;

            W.at<double>(2*absolute_point_index,   0) -= a;
            W.at<double>(2*absolute_point_index+1, 0) -= b;

            V.at<double>(2*absolute_point_index,   0) -= beta*a;
            V.at<double>(2*absolute_point_index+1, 0) -= beta*b;

            lambda += 2;
            m += 2*beta;
          }
        }

        // Back up U for re-substitution
        Us.push_back(U);

        // Invert matrix U for this point and write it to diagonal of overall U_inv matrix
        cv::Mat U_inv_p = U.inv();
        // std::cout << cv::determinant(U_inv_p) << std::endl;
        // std::cout << U_inv_p << std::endl;
        // std::cout << U << std::endl;

        U_inv.at<double>(2*absolute_point_index,   2*absolute_point_index  ) = U_inv_p.at<double>(0, 0);
        U_inv.at<double>(2*absolute_point_index+1, 2*absolute_point_index  ) = U_inv_p.at<double>(1, 0);
        U_inv.at<double>(2*absolute_point_index,   2*absolute_point_index+1) = U_inv_p.at<double>(0, 1);
        U_inv.at<double>(2*absolute_point_index+1, 2*absolute_point_index+1) = U_inv_p.at<double>(1, 1);
      }

      // Todo: check if opencv utilizes the sparsity of U, change use Eigen?
      // solve for the exposure
      cv::Mat K_mat;
      cv::solve(-W.t()*U_inv*W+lambda, -W.t()*U_inv*V+m, K_mat);
      double K = K_mat.at<double>(0, 0);

      // std::cout << -W.t()*U_inv*W+lambda << std::endl;
      // std::cout << -W.t()*U_inv*V+m << std::endl;
      // std::cout << K_mat << std::endl;

      // Solve for the displacements
      absolute_point_index = -1;
      for (int p = 0; p < nr_points; p++) {
        if (point_validity.at(p) == 0)
          continue;

        absolute_point_index++;

        cv::Mat U_p = Us.at(absolute_point_index);
        cv::Mat V_p = V(cv::Rect(0, 2 * absolute_point_index, 1, 2));
        cv::Mat W_p = W(cv::Rect(0, 2 * absolute_point_index, 1, 2));

        cv::Mat displacement;
        cv::solve(U_p, V_p - K * W_p, displacement);

        // std::cout << displacement << std::endl;

        output_points.at(p).x += (float) displacement.at<double>(0, 0);
        output_points.at(p).y += (float) displacement.at<double>(1, 0);

        // Filter out this point if too close at the boundaries
        const int filter_margin = 2;
        double x = output_points.at(p).x;
        double y = output_points.at(p).y;
        if (x < filter_margin || y < filter_margin || x > image_cols-filter_margin || y > image_rows-filter_margin) {
          point_validity.at(p) = 0;
        }
      }

      K_total += K;
    }
  } else {
    for (int round = 0; round < 1; round++) {
      // Get the currently valid points
      int nr_valid_points = getNrValidPoints(point_validity);
      if (nr_valid_points == 0) continue;

      // Allocate space for W, V, lambda, m matrices
      cv::Mat W(8*nr_valid_points, 4, CV_64F, 0.0);
      cv::Mat V(8*nr_valid_points, 1, CV_64F, 0.0);
      cv::Mat lambda(4, 4, CV_64F, 0.0);
      cv::Mat m(4, 1, CV_64F, 0.0);

      // Allocate space for U_inv and the original Us
      cv::Mat U_inv(8*nr_valid_points, 8*nr_valid_points, CV_64F, 0.0);
      std::vector<cv::Mat> Us;

      int absolute_point_index = -1;

      for (int p = 0; p < input_points.size(); p++) {
        if (point_validity.at(p) == 0)
          continue;

        absolute_point_index++;

        // Build U matrix
        cv::Mat U(8, 8, CV_64F, 0.0);

        // Bilinear image interpolation
        cv::Mat patch_intensities_1;
        cv::Mat patch_intensities_2;
        int absolute_patch_size = ((patch_size_ + 1) * 2 + 1);  // Todo: why patch_size_ + 1 ?
        cv::getRectSubPix(new_image, cv::Size(absolute_patch_size, absolute_patch_size), output_points.at(p), patch_intensities_2, CV_32F);
        cv::getRectSubPix(old_image, cv::Size(absolute_patch_size, absolute_patch_size), input_points.at(p), patch_intensities_1, CV_32F);

        // Go through image patch around this point
        for (int r = 0; r < 2*patch_size_; r++) {
          for (int c = 0; c < 2*patch_size_; c++) {
            // Fetch patch intensity values
            double i_frame_1 = patch_intensities_1.at<float>(1 + r, 1 + c) / 255.0;
            double i_frame_2 = patch_intensities_2.at<float>(1 + r, 1 + c) / 255.0;

            // calculate value of Grossberg base function or their derivatives that matched point intensity
            double g_0_1 = evaluateGrossbergBaseFunction(0, false, i_frame_1);
            double hinv_1_1 = evaluateGrossbergBaseFunction(1, false, i_frame_1);
            double hinv_2_1 = evaluateGrossbergBaseFunction(2, false, i_frame_1);
            double hinv_3_1 = evaluateGrossbergBaseFunction(3, false, i_frame_1);

            double g_0_der_1 = evaluateGrossbergBaseFunction(0, true, i_frame_1);
            double hinv_1_der_1 = evaluateGrossbergBaseFunction(1, true, i_frame_1);
            double hinv_2_der_1 = evaluateGrossbergBaseFunction(2, true, i_frame_1);
            double hinv_3_der_1 = evaluateGrossbergBaseFunction(3, true, i_frame_1);

            double g_0_2 = evaluateGrossbergBaseFunction(0, false, i_frame_2);
            double hinv_1_2 = evaluateGrossbergBaseFunction(1, false, i_frame_2);
            double hinv_2_2 = evaluateGrossbergBaseFunction(2, false, i_frame_2);
            double hinv_3_2 = evaluateGrossbergBaseFunction(3, false, i_frame_2);

            double g_0_der_2 = evaluateGrossbergBaseFunction(0, true, i_frame_2);
            double hinv_1_der_2 = evaluateGrossbergBaseFunction(1, true, i_frame_2);
            double hinv_2_der_2 = evaluateGrossbergBaseFunction(2, true, i_frame_2);
            double hinv_3_der_2 = evaluateGrossbergBaseFunction(3, true, i_frame_2);

            // Estimate patch gradient values
            double grad_1_x = (patch_intensities_1.at<float>(1+r, 1+c+1) - patch_intensities_1.at<float>(1+r, 1+c-1))/2/255;
            double grad_1_y = (patch_intensities_1.at<float>(1+r+1, 1+c) - patch_intensities_1.at<float>(1+r-1, 1+c))/2/255;

            double grad_2_x = (patch_intensities_2.at<float>(1+r, 1+c+1) - patch_intensities_2.at<float>(1+r, 1+c-1))/2/255;
            double grad_2_y = (patch_intensities_2.at<float>(1+r+1, 1+c) - patch_intensities_2.at<float>(1+r-1, 1+c))/2/255;

            double a = (g_0_der_2*grad_2_x + g_0_der_1*grad_1_x)/2;
            double b = (g_0_der_2*grad_2_y + g_0_der_1*grad_1_y)/2;

            double r1 = hinv_1_2 - hinv_1_1;
            double r2 = hinv_2_2 - hinv_2_1;
            double r3 = hinv_3_2 - hinv_3_1;

            double p1 = (hinv_1_der_2*grad_2_x + hinv_1_der_1*grad_1_x)/2;
            double p2 = (hinv_2_der_2*grad_2_x + hinv_2_der_1*grad_1_x)/2;
            double p3 = (hinv_3_der_2*grad_2_x + hinv_3_der_1*grad_1_x)/2;

            double q1 = (hinv_1_der_2*grad_2_y + hinv_1_der_1*grad_1_y)/2;
            double q2 = (hinv_2_der_2*grad_2_y + hinv_2_der_1*grad_1_y)/2;
            double q3 = (hinv_3_der_2*grad_2_y + hinv_3_der_1*grad_1_y)/2;

            double d = g_0_2 - g_0_1;

            cv::Mat mu = (cv::Mat_<double>(8, 1) << a, p1, p2, p3, b, q1, q2, q3);
            cv::Mat nu = (cv::Mat_<double>(4, 1) << r1, r2, r3, -1.0);

            U += mu * mu.t();
            W(cv::Rect(0, 8*absolute_point_index, 4, 8)) += mu * nu.t();
            lambda += nu * nu.t();
            V(cv::Rect(0, 8*absolute_point_index, 1, 8)) -= d * mu;
            m -= d * nu;
          }
        }

        // Back up U for re-substitution
        Us.push_back(U);

        // Invert matrix U for this point and write it to diagonal of overall U_inv matrix
        cv::Mat U_inv_p = U.inv();
        // std::cout << cv::determinant(U_inv_p) << std::endl;
        // std::cout << U_inv_p << std::endl;
        // std::cout << U << std::endl;

//        U_inv(cv::Rect(8*absolute_point_index, 8*absolute_point_index, 8, 8)) = U_inv_p; // Todo: why?
        U_inv_p.copyTo(U_inv(cv::Rect(8*absolute_point_index, 8*absolute_point_index, 8, 8)));
      }

      // Deal with gamma ambiguity problem
      double g_0_128 = evaluateGrossbergBaseFunction(0, false, 128.0/255.0);
      double hinv_1_128 = evaluateGrossbergBaseFunction(1, false, 128.0/255.0);
      double hinv_2_128 = evaluateGrossbergBaseFunction(2, false, 128.0/255.0);
      double hinv_3_128 = evaluateGrossbergBaseFunction(3, false, 128.0/255.0);

      // Todo: How to chose value for omega
      double omega = 1.0;
      double tua = 0.5;

      // Todo: check if opencv utilizes the sparsity of U, change use Eigen?
      cv::Mat D(5, 4, CV_64F, 0.0);
//      D(cv::Rect(0, 0, 4, 4)) = - W.t() * U_inv * W + lambda;
      cv::Mat(- W.t() * U_inv * W + lambda).copyTo(D(cv::Rect(0, 0, 4, 4)));
//      D(cv::Rect(0, 4, 4, 1)) = (cv::Mat_<double>(1, 4) << hinv_1_128, hinv_2_128, hinv_3_128, 0) * omega;
      cv::Mat((cv::Mat_<double>(1, 4) << hinv_1_128, hinv_2_128, hinv_3_128, 0) * omega).copyTo( D(cv::Rect(0, 4, 4, 1)));

      cv::Mat b (5, 1, CV_64F, 0.0);
//      b.rowRange(0, 4) = - W.t() * U_inv * V + m;
      cv::Mat(- W.t() * U_inv * V + m).copyTo(b.rowRange(0, 4));
      b.at<double>(4, 0) = (tua - g_0_128) * omega;

      // solve for the exposure & radiometric parameters and their covariance matrix
      cv::Mat theta;
      cv::solve(D, b, theta, cv::DECOMP_QR);
      double K = theta.at<double>(3);
      double c1 = theta.at<double>(0);
      double c2 = theta.at<double>(1);
      double c3 = theta.at<double>(2);

      cv::Matx33d R = cv::Mat(((D.t() * D).inv() * cv::Mat((D * theta - b).t() * (D * theta - b)).at<double>(0, 0))(cv::Rect(0, 0, 3, 3)));

      // std::cout << A << std::endl;
      // std::cout << b << std::endl;
      // std::cout << K << std::endl;
       std::cout << c1 << ", " << c2 << ", " << c3 << std::endl;
      // std::cout << R << std::endl;

      // Todo: create thread to update radiometric parameters
      // Update radiometric parameters
      updateRadiometricParameters(c1, c2, c3, R);

      // Solve for the displacements
      absolute_point_index = -1;
      for (int p = 0; p < nr_points; p++) {
        if (point_validity.at(p) == 0)
          continue;

        absolute_point_index++;

        cv::Mat U_p = Us.at(absolute_point_index);
        cv::Mat V_p = V(cv::Rect(0, 8*absolute_point_index, 1, 8));
        cv::Mat W_p = W(cv::Rect(0, 8*absolute_point_index, 4, 8));
//        std::cout << U_p << std::endl;
//        std::cout << V_p << std::endl;
//        std::cout << W_p << std::endl;

        cv::Mat Y(8, 2, CV_64F, 0.0);
        cv::Mat c_vector = (cv::Mat_<double>(4, 1) << 1.0, c1, c2, c3);
        Y.colRange(0, 1) = U_p.colRange(0, 4) * c_vector;
        Y.colRange(1, 2) = U_p.colRange(4, 8) * c_vector;
//        std::cout << Y << std::endl;

        cv::Mat displacement;
        cv::solve(Y, V_p - W_p * theta, displacement, cv::DECOMP_QR);

//        std::cout << displacement << std::endl;

        output_points.at(p).x += (float) displacement.at<double>(0, 0);
        output_points.at(p).y += (float) displacement.at<double>(1, 0);
        if (std::isnan(output_points.at(p).x) || std::isnan(output_points.at(p).y))
          point_validity.at(p) = 0;

        // Filter out this point if too close at the boundaries
        const int filter_margin = 2;
        double x = output_points.at(p).x;
        double y = output_points.at(p).y;
        if (x < filter_margin || y < filter_margin || x > image_cols-filter_margin || y > image_rows-filter_margin) {
          point_validity.at(p) = 0;
        }
      }

      K_total += K;
    }
  }

  return exp(K_total);
}

int GainRobustTracker::getNrValidPoints(const std::vector<uchar>& validity_vector) {
  // Simply sum up the validity vector
  int result = 0;
  for (const auto& item : validity_vector) {
    result += item;
  }
  return result;
}

int GainRobustTracker::updateRadiometricParameters(double c1, double c2, double c3, cv::Matx33d& R) {
  // is first estimation
  static bool is_initialized = false;
  if (!is_initialized) {
    radiometric_parameters_[0] = c1;
    radiometric_parameters_[1] = c2;
    radiometric_parameters_[2] = c3;
    P_ = R;

    is_initialized = true;
  } else {
    // Using EKF filter update estimation radiometric parameters
    cv::Vec3d z(c1, c2, c3);

    cv::Vec3d status_ = radiometric_parameters_;
    cv::Matx33d covariance_ = P_;

    cv::Matx33d k = covariance_ * (covariance_ + R).inv();
    radiometric_parameters_ = status_ + k * (z - status_);
    P_ = (cv::Matx33d::eye() - k) * covariance_;

    cv::SVD svd(P_);
    if (svd.w.row(svd.w.cols-1).at<double>(0) < 1e-5) {
      radiometric_converged_ = true;
      getInverseResponse();
    }
      std::cout << radiometric_parameters_ << std::endl;
     // std::cout << P_ << std::endl;
     // std::cout << svd.w << std::endl;
  }
}

double GainRobustTracker::evaluateGrossbergBaseFunction(int base_function_index, bool is_derivative, double x) {
  if (x < 0) x = 0.0;
  else if (x > 1) x = 1.0;

  int x_int = (int)round(x * 1023.0);
  int x_der_int = (int)round(x * 1021);

  double result;
  switch (base_function_index) {
    case 0: {
      if (!is_derivative)
        result = g_0_[x_int];
      else
        result = g_0_der_[x_der_int];
      break;
    }
    case 1: {
      if (!is_derivative)
        result = hinv_1_[x_int];
      else
        result = hinv_1_der_[x_der_int];
      break;
    }
    case 2: {
      if (!is_derivative)
        result = hinv_2_[x_int];
      else
        result = hinv_2_der_[x_der_int];
      break;
    }
    case 3: {
      if (!is_derivative)
        result = hinv_3_[x_int];
      else
        result = hinv_3_der_[x_der_int];
      break;
    }
    default: {
      assert(false);
    }
  }

  return result;
}

void GainRobustTracker::getInverseResponse() {
  inverse_response_function_.resize(256);

  // set boundaries of the inverse response
  inverse_response_function_.at(0) = 0;
  inverse_response_function_.at(255) = 1.0;

  double min = applyResponse(0);
  double max = applyResponse(255 / 255.0);

  // For each inverse response value i find s
  for (int i = 1; i < 255; i++) {
    inverse_response_function_.at(i) = (applyResponse(i / 255.0) - min) / (max - min);
  }
}

double GainRobustTracker::applyResponse(double x) {
  double v0 = evaluateGrossbergBaseFunction(0, false, x);
  double v1 = evaluateGrossbergBaseFunction(1, false, x);
  double v2 = evaluateGrossbergBaseFunction(2, false, x);
  double v3 = evaluateGrossbergBaseFunction(3, false, x);

  double c1 = radiometric_parameters_[0];
  double c2 = radiometric_parameters_[1];
  double c3 = radiometric_parameters_[2];

  return v0 + c1*v1 + c2*v2 + c3*v3;
}

cv::Mat GainRobustTracker::mapImage(const cv::Mat& image) {
  cv::Mat result(image.size(), CV_32F);
  for (int row = 0; row < image.rows; row++) {
    for (int col = 0; col < image.cols; col++) {
      result.at<float>(row, col) = static_cast<float>(inverse_response_function_.at(image.at<uchar>(row, col)) * 255.0);
    }
  }
  return result;
}

const double GainRobustTracker::g_0_[1024] = {
    0.00000000, 0.00147726, 0.00273790, 0.00385811, 0.00487750, 0.00581978, 0.00670052, 0.00753202,
    0.00832390, 0.00908511, 0.00981674, 0.01051719, 0.01119348, 0.01184927, 0.01248706, 0.01310836,
    0.01371501, 0.01430845, 0.01488919, 0.01545915, 0.01601947, 0.01656868, 0.01710786, 0.01763954,
    0.01816409, 0.01868202, 0.01919384, 0.01970013, 0.02020145, 0.02069803, 0.02119012, 0.02167658,
    0.02215623, 0.02263206, 0.02310436, 0.02357320, 0.02403910, 0.02450196, 0.02496192, 0.02541903,
    0.02587329, 0.02632477, 0.02677024, 0.02721375, 0.02765540, 0.02809528, 0.02853420, 0.02897388,
    0.02941151, 0.02984418, 0.03027385, 0.03070138, 0.03112383, 0.03154393, 0.03196235, 0.03237907,
    0.03279429, 0.03320812, 0.03362059, 0.03403191, 0.03444217, 0.03485135, 0.03525754, 0.03566134,
    0.03606424, 0.03646630, 0.03686738, 0.03726786, 0.03766775, 0.03806709, 0.03846602, 0.03886461,
    0.03926180, 0.03965717, 0.04005235, 0.04044736, 0.04084235, 0.04123742, 0.04163233, 0.04202693,
    0.04242082, 0.04281423, 0.04320657, 0.04359637, 0.04398566, 0.04437448, 0.04476288, 0.04515076,
    0.04553832, 0.04592557, 0.04631260, 0.04669941, 0.04708620, 0.04747111, 0.04785581, 0.04824055,
    0.04862516, 0.04900963, 0.04939392, 0.04977822, 0.05016234, 0.05054640, 0.05093011, 0.05131254,
    0.05169442, 0.05207628, 0.05245792, 0.05283941, 0.05322091, 0.05360254, 0.05398451, 0.05436656,
    0.05474869, 0.05513042, 0.05551149, 0.05589278, 0.05627396, 0.05665528, 0.05703689, 0.05741860,
    0.05780036, 0.05818207, 0.05856357, 0.05894459, 0.05932454, 0.05970394, 0.06008290, 0.06046159,
    0.06084004, 0.06121843, 0.06159673, 0.06197504, 0.06235353, 0.06273225, 0.06311054, 0.06348897,
    0.06386763, 0.06424637, 0.06462534, 0.06500475, 0.06538453, 0.06576474, 0.06614533, 0.06652623,
    0.06690713, 0.06728824, 0.06766955, 0.06805103, 0.06843249, 0.06881406, 0.06919566, 0.06957729,
    0.06995908, 0.07034095, 0.07072298, 0.07110500, 0.07148700, 0.07186898, 0.07225112, 0.07263336,
    0.07301572, 0.07339807, 0.07378068, 0.07416369, 0.07454717, 0.07493111, 0.07531555, 0.07570061,
    0.07608625, 0.07647253, 0.07685955, 0.07724725, 0.07763534, 0.07802366, 0.07841244, 0.07880214,
    0.07919242, 0.07958317, 0.07997455, 0.08036661, 0.08075916, 0.08115193, 0.08154477, 0.08193767,
    0.08233068, 0.08272424, 0.08311811, 0.08351236, 0.08390691, 0.08430197, 0.08469743, 0.08509340,
    0.08548978, 0.08588655, 0.08628385, 0.08668222, 0.08708152, 0.08748141, 0.08788185, 0.08828293,
    0.08868463, 0.08908685, 0.08948966, 0.08989305, 0.09029707, 0.09070218, 0.09110851, 0.09151570,
    0.09192383, 0.09233269, 0.09274223, 0.09315233, 0.09356289, 0.09397371, 0.09438483, 0.09479646,
    0.09520934, 0.09562284, 0.09603692, 0.09645145, 0.09686642, 0.09728162, 0.09769687, 0.09811201,
    0.09852718, 0.09894242, 0.09935901, 0.09977592, 0.10019310, 0.10061070, 0.10102860, 0.10144680,
    0.10186540, 0.10228440, 0.10270370, 0.10312340, 0.10354420, 0.10396580, 0.10438780, 0.10481030,
    0.10523330, 0.10565680, 0.10608090, 0.10650570, 0.10693120, 0.10735760, 0.10778520, 0.10821450,
    0.10864480, 0.10907590, 0.10950790, 0.10994030, 0.11037310, 0.11080630, 0.11123970, 0.11167340,
    0.11210780, 0.11254350, 0.11297960, 0.11341620, 0.11385310, 0.11429040, 0.11472790, 0.11516570,
    0.11560380, 0.11604230, 0.11648140, 0.11692210, 0.11736330, 0.11780510, 0.11824740, 0.11869040,
    0.11913420, 0.11957890, 0.12002450, 0.12047100, 0.12091850, 0.12136800, 0.12181860, 0.12227010,
    0.12272240, 0.12317570, 0.12363000, 0.12408500, 0.12454090, 0.12499780, 0.12545550, 0.12591490,
    0.12637570, 0.12683690, 0.12729830, 0.12775990, 0.12822170, 0.12868370, 0.12914590, 0.12960840,
    0.13007120, 0.13053470, 0.13099960, 0.13146490, 0.13193080, 0.13239720, 0.13286410, 0.13333150,
    0.13379960, 0.13426820, 0.13473750, 0.13520750, 0.13567960, 0.13615240, 0.13662580, 0.13710000,
    0.13757490, 0.13805050, 0.13852690, 0.13900420, 0.13948240, 0.13996120, 0.14044230, 0.14092390,
    0.14140600, 0.14188860, 0.14237150, 0.14285500, 0.14333900, 0.14382330, 0.14430820, 0.14479350,
    0.14528030, 0.14576790, 0.14625560, 0.14674350, 0.14723180, 0.14772030, 0.14820940, 0.14869900,
    0.14918910, 0.14967980, 0.15017170, 0.15066550, 0.15115980, 0.15165490, 0.15215080, 0.15264770,
    0.15314540, 0.15364400, 0.15414360, 0.15464440, 0.15514650, 0.15565100, 0.15615660, 0.15666330,
    0.15717060, 0.15767820, 0.15818610, 0.15869440, 0.15920290, 0.15971150, 0.16022050, 0.16073140,
    0.16124290, 0.16175470, 0.16226720, 0.16278010, 0.16329350, 0.16380760, 0.16432250, 0.16483820,
    0.16535500, 0.16587390, 0.16639410, 0.16691530, 0.16743750, 0.16796050, 0.16848440, 0.16900950,
    0.16953550, 0.17006240, 0.17059030, 0.17112010, 0.17165170, 0.17218420, 0.17271720, 0.17325120,
    0.17378630, 0.17432200, 0.17485860, 0.17539610, 0.17593410, 0.17647330, 0.17701410, 0.17755530,
    0.17809690, 0.17863930, 0.17918240, 0.17972590, 0.18027010, 0.18081490, 0.18136030, 0.18190620,
    0.18245430, 0.18300290, 0.18355220, 0.18410210, 0.18465260, 0.18520370, 0.18575570, 0.18630820,
    0.18686130, 0.18741510, 0.18797080, 0.18852780, 0.18908540, 0.18964360, 0.19020280, 0.19076300,
    0.19132390, 0.19188560, 0.19244810, 0.19301130, 0.19357600, 0.19414170, 0.19470820, 0.19527530,
    0.19584280, 0.19641100, 0.19697980, 0.19754900, 0.19811880, 0.19868910, 0.19926060, 0.19983350,
    0.20040690, 0.20098090, 0.20155560, 0.20213130, 0.20270750, 0.20328450, 0.20386240, 0.20444130,
    0.20502140, 0.20560380, 0.20618750, 0.20677230, 0.20735810, 0.20794460, 0.20853210, 0.20912040,
    0.20970960, 0.21030000, 0.21089140, 0.21148510, 0.21208010, 0.21267610, 0.21327310, 0.21387070,
    0.21446860, 0.21506690, 0.21566600, 0.21626580, 0.21686630, 0.21746840, 0.21807180, 0.21867600,
    0.21928110, 0.21988720, 0.22049400, 0.22110180, 0.22171070, 0.22232050, 0.22293140, 0.22354360,
    0.22415750, 0.22477230, 0.22538810, 0.22600480, 0.22662240, 0.22724100, 0.22786050, 0.22848100,
    0.22910250, 0.22972520, 0.23034960, 0.23097500, 0.23160140, 0.23222880, 0.23285710, 0.23348660,
    0.23411730, 0.23474960, 0.23538320, 0.23601780, 0.23665450, 0.23729240, 0.23793130, 0.23857140,
    0.23921250, 0.23985480, 0.24049800, 0.24114240, 0.24178770, 0.24243400, 0.24308180, 0.24373080,
    0.24438060, 0.24503100, 0.24568240, 0.24633450, 0.24698750, 0.24764150, 0.24829660, 0.24895240,
    0.24960980, 0.25026870, 0.25092850, 0.25158940, 0.25225170, 0.25291520, 0.25358000, 0.25424630,
    0.25491440, 0.25558410, 0.25625550, 0.25692870, 0.25760290, 0.25827830, 0.25895450, 0.25963230,
    0.26031140, 0.26099210, 0.26167430, 0.26235800, 0.26304350, 0.26373090, 0.26441970, 0.26510980,
    0.26580100, 0.26649340, 0.26718680, 0.26788140, 0.26857700, 0.26927380, 0.26997170, 0.27067150,
    0.27137260, 0.27207500, 0.27277880, 0.27348410, 0.27419060, 0.27489870, 0.27560840, 0.27631930,
    0.27703150, 0.27774530, 0.27846020, 0.27917610, 0.27989310, 0.28061080, 0.28132910, 0.28204840,
    0.28276870, 0.28348940, 0.28421070, 0.28493280, 0.28565590, 0.28637940, 0.28710330, 0.28782800,
    0.28855380, 0.28928060, 0.29000850, 0.29073770, 0.29146780, 0.29219900, 0.29293220, 0.29366660,
    0.29440220, 0.29513920, 0.29587730, 0.29661680, 0.29735750, 0.29809940, 0.29884250, 0.29958720,
    0.30033410, 0.30108280, 0.30183270, 0.30258430, 0.30333770, 0.30409330, 0.30485090, 0.30561020,
    0.30637070, 0.30713260, 0.30789650, 0.30866190, 0.30942880, 0.31019710, 0.31096670, 0.31173770,
    0.31251020, 0.31328430, 0.31405970, 0.31483620, 0.31561430, 0.31639420, 0.31717570, 0.31795860,
    0.31874310, 0.31952910, 0.32031680, 0.32110610, 0.32189670, 0.32268900, 0.32348320, 0.32428060,
    0.32508000, 0.32588120, 0.32668440, 0.32748930, 0.32829610, 0.32910470, 0.32991510, 0.33072760,
    0.33154260, 0.33236150, 0.33318230, 0.33400520, 0.33482960, 0.33565590, 0.33648360, 0.33731290,
    0.33814400, 0.33897680, 0.33981110, 0.34064860, 0.34148810, 0.34232880, 0.34317140, 0.34401590,
    0.34486180, 0.34570910, 0.34655850, 0.34740940, 0.34826190, 0.34911700, 0.34997500, 0.35083460,
    0.35169600, 0.35255900, 0.35342390, 0.35429060, 0.35515890, 0.35602880, 0.35690020, 0.35777390,
    0.35865110, 0.35952980, 0.36041010, 0.36129190, 0.36217540, 0.36306020, 0.36394640, 0.36483370,
    0.36572260, 0.36661300, 0.36750790, 0.36840470, 0.36930310, 0.37020280, 0.37110380, 0.37200550,
    0.37290850, 0.37381290, 0.37471890, 0.37562590, 0.37653720, 0.37745030, 0.37836550, 0.37928230,
    0.38020040, 0.38111990, 0.38204130, 0.38296440, 0.38388950, 0.38481650, 0.38574730, 0.38668140,
    0.38761770, 0.38855560, 0.38949560, 0.39043720, 0.39138070, 0.39232610, 0.39327340, 0.39422310,
    0.39517650, 0.39613460, 0.39709470, 0.39805680, 0.39902080, 0.39998670, 0.40095430, 0.40192410,
    0.40289630, 0.40387030, 0.40484710, 0.40583010, 0.40681540, 0.40780220, 0.40879090, 0.40978130,
    0.41077380, 0.41176860, 0.41276550, 0.41376440, 0.41476510, 0.41577250, 0.41678210, 0.41779430,
    0.41880990, 0.41982860, 0.42085020, 0.42187510, 0.42290330, 0.42393420, 0.42496810, 0.42600870,
    0.42705450, 0.42810330, 0.42915440, 0.43020830, 0.43126480, 0.43232430, 0.43338590, 0.43444980,
    0.43551620, 0.43658690, 0.43766330, 0.43874190, 0.43982260, 0.44090540, 0.44199040, 0.44307720,
    0.44416610, 0.44525770, 0.44635160, 0.44744850, 0.44855270, 0.44965860, 0.45076680, 0.45187760,
    0.45299090, 0.45410650, 0.45522410, 0.45634390, 0.45746590, 0.45858990, 0.45972260, 0.46085680,
    0.46199250, 0.46312930, 0.46426760, 0.46540810, 0.46655110, 0.46769670, 0.46884440, 0.46999390,
    0.47115070, 0.47231170, 0.47347520, 0.47464060, 0.47580810, 0.47697820, 0.47815110, 0.47932590,
    0.48050270, 0.48168280, 0.48286970, 0.48406330, 0.48525920, 0.48645750, 0.48765920, 0.48886390,
    0.49007150, 0.49128170, 0.49249510, 0.49371200, 0.49493370, 0.49616400, 0.49739770, 0.49863410,
    0.49987300, 0.50111380, 0.50235700, 0.50360280, 0.50485040, 0.50609950, 0.50735150, 0.50861400,
    0.50987940, 0.51114690, 0.51241610, 0.51368770, 0.51496200, 0.51623950, 0.51751940, 0.51880130,
    0.52008620, 0.52138090, 0.52268010, 0.52398160, 0.52528640, 0.52659440, 0.52790530, 0.52921930,
    0.53053590, 0.53185610, 0.53317890, 0.53451050, 0.53584940, 0.53719190, 0.53853800, 0.53988770,
    0.54124160, 0.54259850, 0.54395880, 0.54532200, 0.54668860, 0.54806120, 0.54944250, 0.55082700,
    0.55221520, 0.55360640, 0.55500100, 0.55639890, 0.55780120, 0.55920640, 0.56061410, 0.56202630,
    0.56344950, 0.56487590, 0.56630600, 0.56773990, 0.56917660, 0.57061650, 0.57206030, 0.57350810,
    0.57495960, 0.57641540, 0.57788440, 0.57935810, 0.58083580, 0.58231750, 0.58380360, 0.58529380,
    0.58678840, 0.58828800, 0.58979260, 0.59130130, 0.59282100, 0.59434710, 0.59587640, 0.59740980,
    0.59894700, 0.60048870, 0.60203340, 0.60358170, 0.60513450, 0.60669070, 0.60825640, 0.60983260,
    0.61141360, 0.61299890, 0.61458880, 0.61618290, 0.61778160, 0.61938580, 0.62099500, 0.62260820,
    0.62422840, 0.62586220, 0.62750110, 0.62914440, 0.63079200, 0.63244570, 0.63410360, 0.63576650,
    0.63743540, 0.63910980, 0.64078880, 0.64248650, 0.64419000, 0.64589980, 0.64761400, 0.64933390,
    0.65105940, 0.65279050, 0.65452700, 0.65626910, 0.65801760, 0.65978290, 0.66155640, 0.66333600,
    0.66512160, 0.66691260, 0.66870860, 0.67051120, 0.67232080, 0.67413800, 0.67596130, 0.67779970,
    0.67965340, 0.68151400, 0.68337990, 0.68525280, 0.68713150, 0.68901630, 0.69090870, 0.69280680,
    0.69471080, 0.69662900, 0.69856710, 0.70051320, 0.70246720, 0.70442980, 0.70639800, 0.70837280,
    0.71035460, 0.71234520, 0.71434440, 0.71635240, 0.71838960, 0.72043590, 0.72249050, 0.72455250,
    0.72662460, 0.72870660, 0.73079610, 0.73289470, 0.73500280, 0.73712110, 0.73927170, 0.74143430,
    0.74360610, 0.74578850, 0.74798400, 0.75019060, 0.75240640, 0.75463060, 0.75686390, 0.75910790,
    0.76138020, 0.76367690, 0.76598340, 0.76830370, 0.77063730, 0.77298140, 0.77533580, 0.77770120,
    0.78007810, 0.78246780, 0.78488260, 0.78733370, 0.78979650, 0.79227340, 0.79476270, 0.79726710,
    0.79978640, 0.80231790, 0.80486680, 0.80742880, 0.81001200, 0.81264690, 0.81529650, 0.81796220,
    0.82064830, 0.82335160, 0.82607340, 0.82881310, 0.83157270, 0.83434460, 0.83713420, 0.83999340,
    0.84287730, 0.84578880, 0.84872720, 0.85169420, 0.85468870, 0.85770070, 0.86073610, 0.86381080,
    0.86695710, 0.87015640, 0.87340270, 0.87667810, 0.87997270, 0.88329600, 0.88665530, 0.89005950,
    0.89351050, 0.89699390, 0.90051200, 0.90453230, 0.90822410, 0.91193950, 0.91571120, 0.91954820,
    0.92343280, 0.92737860, 0.93141100, 0.93554070, 0.93979770, 0.94501960, 0.95093510, 0.95590720,
    0.96078810, 0.96567640, 0.97074810, 0.97739200, 0.98294170, 0.98851590, 0.99415250, 1.00000000
};

const double GainRobustTracker::hinv_1_[1024] = {
    0.00000000, -0.00037067, -0.00068149, -0.00095414, -0.00120025, -0.00142671, -0.00163781, -0.00183627,
    -0.00202431, -0.00220365, -0.00237603, -0.00254291, -0.00270451, -0.00286124, -0.00301371, -0.00316222,
    -0.00330721, -0.00344903, -0.00358787, -0.00372394, -0.00385759, -0.00398924, -0.00411904, -0.00424688,
    -0.00437280, -0.00449698, -0.00461946, -0.00474035, -0.00485984, -0.00497804, -0.00509497, -0.00521079,
    -0.00532572, -0.00543940, -0.00555192, -0.00566335, -0.00577378, -0.00588321, -0.00599162, -0.00609905,
    -0.00620550, -0.00631101, -0.00641596, -0.00652005, -0.00662325, -0.00672558, -0.00682687, -0.00692683,
    -0.00702612, -0.00712529, -0.00722396, -0.00732193, -0.00741962, -0.00751677, -0.00761337, -0.00770939,
    -0.00780486, -0.00789979, -0.00799422, -0.00808816, -0.00818163, -0.00827463, -0.00836735, -0.00845971,
    -0.00855165, -0.00864318, -0.00873431, -0.00882512, -0.00891563, -0.00900585, -0.00909580, -0.00918548,
    -0.00927506, -0.00936462, -0.00945405, -0.00954329, -0.00963247, -0.00972153, -0.00981044, -0.00989901,
    -0.00998707, -0.01007457, -0.01016154, -0.01024815, -0.01033421, -0.01041978, -0.01050487, -0.01058952,
    -0.01067369, -0.01075741, -0.01084071, -0.01092361, -0.01100617, -0.01108853, -0.01117067, -0.01125255,
    -0.01133419, -0.01141559, -0.01149677, -0.01157775, -0.01165856, -0.01173920, -0.01181969, -0.01190009,
    -0.01198039, -0.01206064, -0.01214071, -0.01222067, -0.01230052, -0.01238033, -0.01246016, -0.01253995,
    -0.01261963, -0.01269928, -0.01277886, -0.01285834, -0.01293768, -0.01301696, -0.01309619, -0.01317537,
    -0.01325442, -0.01333328, -0.01341188, -0.01349014, -0.01356790, -0.01364519, -0.01372204, -0.01379850,
    -0.01387464, -0.01395054, -0.01402616, -0.01410158, -0.01417690, -0.01425209, -0.01432713, -0.01440206,
    -0.01447688, -0.01455162, -0.01462635, -0.01470105, -0.01477576, -0.01485047, -0.01492514, -0.01499979,
    -0.01507431, -0.01514875, -0.01522309, -0.01529729, -0.01537128, -0.01544510, -0.01551872, -0.01559211,
    -0.01566534, -0.01573837, -0.01581116, -0.01588368, -0.01595599, -0.01602805, -0.01609989, -0.01617154,
    -0.01624302, -0.01631431, -0.01638549, -0.01645658, -0.01652758, -0.01659843, -0.01666931, -0.01674024,
    -0.01681128, -0.01688244, -0.01695371, -0.01702501, -0.01709624, -0.01716739, -0.01723849, -0.01730947,
    -0.01738051, -0.01745163, -0.01752283, -0.01759406, -0.01766524, -0.01773627, -0.01780703, -0.01787745,
    -0.01794756, -0.01801728, -0.01808679, -0.01815616, -0.01822537, -0.01829451, -0.01836356, -0.01843253,
    -0.01850141, -0.01857021, -0.01863898, -0.01870764, -0.01877626, -0.01884488, -0.01891349, -0.01898213,
    -0.01905077, -0.01911940, -0.01918802, -0.01925665, -0.01932529, -0.01939390, -0.01946250, -0.01953126,
    -0.01960022, -0.01966929, -0.01973839, -0.01980742, -0.01987635, -0.01994515, -0.02001378, -0.02008226,
    -0.02015051, -0.02021878, -0.02028705, -0.02035529, -0.02042343, -0.02049135, -0.02055896, -0.02062620,
    -0.02069308, -0.02075966, -0.02082588, -0.02089190, -0.02095780, -0.02102356, -0.02108920, -0.02115471,
    -0.02122013, -0.02128543, -0.02135060, -0.02141563, -0.02148041, -0.02154502, -0.02160958, -0.02167410,
    -0.02173858, -0.02180305, -0.02186752, -0.02193206, -0.02199664, -0.02206132, -0.02212600, -0.02219079,
    -0.02225584, -0.02232107, -0.02238638, -0.02245158, -0.02251668, -0.02258156, -0.02264624, -0.02271071,
    -0.02277493, -0.02283878, -0.02290247, -0.02296599, -0.02302931, -0.02309241, -0.02315521, -0.02321774,
    -0.02328006, -0.02334218, -0.02340415, -0.02346577, -0.02352728, -0.02358872, -0.02365008, -0.02371140,
    -0.02377271, -0.02383406, -0.02389548, -0.02395701, -0.02401868, -0.02408025, -0.02414183, -0.02420345,
    -0.02426509, -0.02432674, -0.02438843, -0.02445014, -0.02451192, -0.02457374, -0.02463560, -0.02469736,
    -0.02475893, -0.02482029, -0.02488130, -0.02494192, -0.02500218, -0.02506217, -0.02512190, -0.02518141,
    -0.02524069, -0.02529971, -0.02535839, -0.02541694, -0.02547538, -0.02553373, -0.02559201, -0.02565024,
    -0.02570845, -0.02576659, -0.02582471, -0.02588276, -0.02594057, -0.02599835, -0.02605615, -0.02611396,
    -0.02617183, -0.02622978, -0.02628788, -0.02634617, -0.02640463, -0.02646321, -0.02652163, -0.02657998,
    -0.02663823, -0.02669640, -0.02675448, -0.02681251, -0.02687051, -0.02692845, -0.02698636, -0.02704422,
    -0.02710181, -0.02715913, -0.02721620, -0.02727303, -0.02732961, -0.02738607, -0.02744239, -0.02749860,
    -0.02755473, -0.02761084, -0.02766681, -0.02772261, -0.02777842, -0.02783429, -0.02789024, -0.02794629,
    -0.02800244, -0.02805873, -0.02811521, -0.02817192, -0.02822882, -0.02828570, -0.02834280, -0.02840004,
    -0.02845724, -0.02851419, -0.02857078, -0.02862705, -0.02868299, -0.02873860, -0.02879393, -0.02884878,
    -0.02890338, -0.02895779, -0.02901203, -0.02906603, -0.02911989, -0.02917366, -0.02922736, -0.02928097,
    -0.02933450, -0.02938782, -0.02944102, -0.02949424, -0.02954742, -0.02960054, -0.02965366, -0.02970681,
    -0.02975995, -0.02981312, -0.02986630, -0.02991938, -0.02997238, -0.03002534, -0.03007822, -0.03013107,
    -0.03018389, -0.03023667, -0.03028943, -0.03034215, -0.03039476, -0.03044718, -0.03049926, -0.03055110,
    -0.03060276, -0.03065431, -0.03070576, -0.03075712, -0.03080840, -0.03085961, -0.03091070, -0.03096169,
    -0.03101243, -0.03106309, -0.03111367, -0.03116420, -0.03121469, -0.03126514, -0.03131556, -0.03136598,
    -0.03141643, -0.03146690, -0.03151726, -0.03156767, -0.03161813, -0.03166863, -0.03171924, -0.03176988,
    -0.03182052, -0.03187118, -0.03192187, -0.03197253, -0.03202301, -0.03207326, -0.03212334, -0.03217328,
    -0.03222305, -0.03227273, -0.03232229, -0.03237167, -0.03242090, -0.03246988, -0.03251861, -0.03256704,
    -0.03261532, -0.03266345, -0.03271149, -0.03275949, -0.03280743, -0.03285532, -0.03290325, -0.03295123,
    -0.03299933, -0.03304744, -0.03309576, -0.03314423, -0.03319275, -0.03324120, -0.03328962, -0.03333797,
    -0.03338634, -0.03343478, -0.03348335, -0.03353190, -0.03358053, -0.03362921, -0.03367782, -0.03372615,
    -0.03377403, -0.03382148, -0.03386861, -0.03391553, -0.03396222, -0.03400862, -0.03405479, -0.03410073,
    -0.03414653, -0.03419220, -0.03423773, -0.03428312, -0.03432841, -0.03437359, -0.03441868, -0.03446361,
    -0.03450831, -0.03455297, -0.03459747, -0.03464185, -0.03468607, -0.03473020, -0.03477423, -0.03481817,
    -0.03486201, -0.03490574, -0.03494928, -0.03499275, -0.03503618, -0.03507956, -0.03512293, -0.03516630,
    -0.03520975, -0.03525324, -0.03529676, -0.03534028, -0.03538366, -0.03542705, -0.03547042, -0.03551377,
    -0.03555710, -0.03560045, -0.03564378, -0.03568709, -0.03573027, -0.03577331, -0.03581614, -0.03585881,
    -0.03590129, -0.03594360, -0.03598576, -0.03602773, -0.03606952, -0.03611116, -0.03615268, -0.03619413,
    -0.03623546, -0.03627668, -0.03631784, -0.03635900, -0.03640014, -0.03644132, -0.03648262, -0.03652412,
    -0.03656581, -0.03660775, -0.03664982, -0.03669177, -0.03673361, -0.03677528, -0.03681678, -0.03685804,
    -0.03689910, -0.03693993, -0.03698057, -0.03702106, -0.03706138, -0.03710139, -0.03714113, -0.03718066,
    -0.03721993, -0.03725895, -0.03729775, -0.03733637, -0.03737485, -0.03741322, -0.03745144, -0.03748950,
    -0.03752750, -0.03756548, -0.03760350, -0.03764158, -0.03767977, -0.03771805, -0.03775641, -0.03779489,
    -0.03783337, -0.03787178, -0.03791007, -0.03794830, -0.03798639, -0.03802438, -0.03806230, -0.03810014,
    -0.03813793, -0.03817563, -0.03821318, -0.03825045, -0.03828737, -0.03832406, -0.03836050, -0.03839677,
    -0.03843287, -0.03846884, -0.03850470, -0.03854052, -0.03857625, -0.03861182, -0.03864716, -0.03868239,
    -0.03871746, -0.03875244, -0.03878733, -0.03882217, -0.03885693, -0.03889170, -0.03892640, -0.03896107,
    -0.03899562, -0.03903022, -0.03906484, -0.03909959, -0.03913451, -0.03916969, -0.03920516, -0.03924068,
    -0.03927615, -0.03931141, -0.03934628, -0.03938083, -0.03941521, -0.03944940, -0.03948354, -0.03951762,
    -0.03955159, -0.03958545, -0.03961908, -0.03965240, -0.03968527, -0.03971775, -0.03975003, -0.03978216,
    -0.03981413, -0.03984592, -0.03987765, -0.03990928, -0.03994083, -0.03997228, -0.04000372, -0.04003494,
    -0.04006621, -0.04009753, -0.04012889, -0.04016029, -0.04019175, -0.04022323, -0.04025482, -0.04028651,
    -0.04031816, -0.04034928, -0.04038001, -0.04041021, -0.04043987, -0.04046904, -0.04049777, -0.04052604,
    -0.04055386, -0.04058123, -0.04060819, -0.04063445, -0.04066016, -0.04068559, -0.04071071, -0.04073550,
    -0.04075998, -0.04078422, -0.04080817, -0.04083200, -0.04085568, -0.04087897, -0.04090187, -0.04092459,
    -0.04094719, -0.04096969, -0.04099210, -0.04101449, -0.04103686, -0.04105920, -0.04108158, -0.04110390,
    -0.04112606, -0.04114836, -0.04117073, -0.04119310, -0.04121553, -0.04123802, -0.04126054, -0.04128316,
    -0.04130584, -0.04132856, -0.04135085, -0.04137302, -0.04139503, -0.04141678, -0.04143807, -0.04145884,
    -0.04147922, -0.04149924, -0.04151896, -0.04153841, -0.04155709, -0.04157537, -0.04159343, -0.04161128,
    -0.04162887, -0.04164627, -0.04166347, -0.04168053, -0.04169741, -0.04171409, -0.04173020, -0.04174598,
    -0.04176170, -0.04177730, -0.04179274, -0.04180804, -0.04182322, -0.04183832, -0.04185342, -0.04186857,
    -0.04188354, -0.04189808, -0.04191258, -0.04192692, -0.04194102, -0.04195492, -0.04196866, -0.04198221,
    -0.04199574, -0.04200906, -0.04202212, -0.04203427, -0.04204614, -0.04205764, -0.04206872, -0.04207938,
    -0.04208975, -0.04209985, -0.04210967, -0.04211916, -0.04212840, -0.04213661, -0.04214455, -0.04215232,
    -0.04216000, -0.04216734, -0.04217447, -0.04218140, -0.04218806, -0.04219447, -0.04220065, -0.04220590,
    -0.04221059, -0.04221494, -0.04221905, -0.04222291, -0.04222653, -0.04222977, -0.04223266, -0.04223517,
    -0.04223729, -0.04223869, -0.04223901, -0.04223891, -0.04223840, -0.04223757, -0.04223638, -0.04223492,
    -0.04223307, -0.04223087, -0.04222838, -0.04222556, -0.04222162, -0.04221757, -0.04221348, -0.04220947,
    -0.04220531, -0.04220112, -0.04219708, -0.04219321, -0.04218940, -0.04218563, -0.04218053, -0.04217517,
    -0.04216952, -0.04216340, -0.04215699, -0.04215031, -0.04214338, -0.04213612, -0.04212862, -0.04212073,
    -0.04211140, -0.04210105, -0.04209019, -0.04207889, -0.04206721, -0.04205518, -0.04204283, -0.04203028,
    -0.04201757, -0.04200449, -0.04199045, -0.04197544, -0.04196038, -0.04194520, -0.04192981, -0.04191421,
    -0.04189845, -0.04188271, -0.04186697, -0.04185102, -0.04183452, -0.04181683, -0.04179888, -0.04178052,
    -0.04176171, -0.04174234, -0.04172245, -0.04170208, -0.04168123, -0.04165968, -0.04163740, -0.04161329,
    -0.04158860, -0.04156345, -0.04153785, -0.04151176, -0.04148527, -0.04145833, -0.04143115, -0.04140336,
    -0.04137520, -0.04134533, -0.04131476, -0.04128380, -0.04125250, -0.04122076, -0.04118866, -0.04115623,
    -0.04112346, -0.04109010, -0.04105640, -0.04102135, -0.04098537, -0.04094922, -0.04091286, -0.04087644,
    -0.04083970, -0.04080263, -0.04076525, -0.04072758, -0.04068947, -0.04065014, -0.04060925, -0.04056795,
    -0.04052634, -0.04048442, -0.04044213, -0.04039922, -0.04035567, -0.04031139, -0.04026642, -0.04022038,
    -0.04017213, -0.04012325, -0.04007387, -0.04002381, -0.03997331, -0.03992227, -0.03987075, -0.03981876,
    -0.03976641, -0.03971344, -0.03965825, -0.03960252, -0.03954651, -0.03949009, -0.03943321, -0.03937609,
    -0.03931847, -0.03926059, -0.03920225, -0.03914344, -0.03908245, -0.03902008, -0.03895700, -0.03889304,
    -0.03882843, -0.03876326, -0.03869705, -0.03863003, -0.03856213, -0.03849353, -0.03842306, -0.03835093,
    -0.03827807, -0.03820454, -0.03813013, -0.03805510, -0.03797951, -0.03790342, -0.03782664, -0.03774921,
    -0.03767049, -0.03758917, -0.03750738, -0.03742498, -0.03734200, -0.03725849, -0.03717452, -0.03708996,
    -0.03700494, -0.03691950, -0.03683353, -0.03674412, -0.03665405, -0.03656320, -0.03647169, -0.03637960,
    -0.03628669, -0.03619290, -0.03609827, -0.03600281, -0.03590650, -0.03580699, -0.03570604, -0.03560407,
    -0.03550142, -0.03539778, -0.03529324, -0.03518797, -0.03508205, -0.03497529, -0.03486767, -0.03475733,
    -0.03464435, -0.03453039, -0.03441555, -0.03429960, -0.03418293, -0.03406567, -0.03394739, -0.03382821,
    -0.03370800, -0.03358524, -0.03345891, -0.03333127, -0.03320242, -0.03307204, -0.03294059, -0.03280808,
    -0.03267451, -0.03253969, -0.03240345, -0.03226576, -0.03212292, -0.03197874, -0.03183292, -0.03168610,
    -0.03153797, -0.03138799, -0.03123657, -0.03108369, -0.03092927, -0.03077294, -0.03061155, -0.03044834,
    -0.03028347, -0.03011670, -0.02994799, -0.02977748, -0.02960529, -0.02943169, -0.02925652, -0.02907966,
    -0.02889846, -0.02871340, -0.02852629, -0.02833704, -0.02814575, -0.02795228, -0.02775726, -0.02755995,
    -0.02736048, -0.02715863, -0.02695249, -0.02674042, -0.02652666, -0.02630985, -0.02609062, -0.02586830,
    -0.02564352, -0.02541669, -0.02518592, -0.02495234, -0.02471529, -0.02447100, -0.02422366, -0.02397329,
    -0.02371875, -0.02346003, -0.02319774, -0.02293137, -0.02266066, -0.02238920, -0.02211452, -0.02182980,
    -0.02154045, -0.02124386, -0.02094103, -0.02063227, -0.02031858, -0.02000225, -0.01968033, -0.01934677,
    -0.01898635, -0.01862573, -0.01826109, -0.01788969, -0.01751771, -0.01714099, -0.01675907, -0.01636672,
    -0.01596240, -0.01554893, -0.01512689, -0.01446370, -0.01400828, -0.01355645, -0.01309401, -0.01261733,
    -0.01213440, -0.01164163, -0.01113086, -0.01059870, -0.01003953, -0.00912132, -0.00787181, -0.00711156,
    -0.00641178, -0.00571864, -0.00496208, -0.00350506, -0.00259309, -0.00170919, -0.00086507, 0.00000000
};

const double GainRobustTracker::hinv_2_[1024] = {
    0.00000000, -0.00138071, -0.00253762, -0.00354823, -0.00445211, -0.00527324, -0.00602843, -0.00672853,
    -0.00738366, -0.00799980, -0.00858306, -0.00913581, -0.00966243, -0.01016601, -0.01064938, -0.01111449,
    -0.01156316, -0.01199686, -0.01241708, -0.01282460, -0.01321998, -0.01360340, -0.01397568, -0.01433806,
    -0.01469119, -0.01503557, -0.01537149, -0.01569976, -0.01602076, -0.01633465, -0.01664175, -0.01694207,
    -0.01723519, -0.01752263, -0.01780453, -0.01808108, -0.01835268, -0.01861944, -0.01888151, -0.01913918,
    -0.01939260, -0.01964168, -0.01988536, -0.02012495, -0.02036070, -0.02059275, -0.02082114, -0.02104594,
    -0.02126740, -0.02148601, -0.02170167, -0.02191439, -0.02212292, -0.02232825, -0.02253078, -0.02273052,
    -0.02292768, -0.02312206, -0.02331385, -0.02350313, -0.02368987, -0.02387418, -0.02405547, -0.02423400,
    -0.02441034, -0.02458450, -0.02475648, -0.02492637, -0.02509422, -0.02526006, -0.02542396, -0.02558596,
    -0.02574561, -0.02590270, -0.02605803, -0.02621151, -0.02636305, -0.02651292, -0.02666100, -0.02680745,
    -0.02695221, -0.02709542, -0.02723671, -0.02737560, -0.02751293, -0.02764863, -0.02778284, -0.02791539,
    -0.02804663, -0.02817625, -0.02830440, -0.02843113, -0.02855648, -0.02867969, -0.02880139, -0.02892175,
    -0.02904081, -0.02915861, -0.02927501, -0.02939023, -0.02950419, -0.02961702, -0.02972864, -0.02983874,
    -0.02994745, -0.03005493, -0.03016126, -0.03026651, -0.03037072, -0.03047370, -0.03057557, -0.03067618,
    -0.03077586, -0.03087431, -0.03097135, -0.03106753, -0.03116268, -0.03125682, -0.03134997, -0.03144200,
    -0.03153315, -0.03162334, -0.03171247, -0.03180041, -0.03188698, -0.03197249, -0.03205701, -0.03214052,
    -0.03222297, -0.03230442, -0.03238497, -0.03246459, -0.03254320, -0.03262084, -0.03269717, -0.03277273,
    -0.03284746, -0.03292126, -0.03299417, -0.03306629, -0.03313748, -0.03320776, -0.03327724, -0.03334596,
    -0.03341351, -0.03348020, -0.03354618, -0.03361134, -0.03367560, -0.03373913, -0.03380186, -0.03386380,
    -0.03392484, -0.03398514, -0.03404450, -0.03410286, -0.03416045, -0.03421727, -0.03427326, -0.03432849,
    -0.03438296, -0.03443667, -0.03448964, -0.03454186, -0.03459316, -0.03464345, -0.03469297, -0.03474187,
    -0.03478997, -0.03483721, -0.03488369, -0.03492943, -0.03497440, -0.03501855, -0.03506201, -0.03510461,
    -0.03514654, -0.03518767, -0.03522810, -0.03526789, -0.03530703, -0.03534542, -0.03538307, -0.03542007,
    -0.03545643, -0.03549198, -0.03552687, -0.03556111, -0.03559473, -0.03562756, -0.03565969, -0.03569121,
    -0.03572212, -0.03575238, -0.03578202, -0.03581092, -0.03583915, -0.03586663, -0.03589350, -0.03591977,
    -0.03594546, -0.03597052, -0.03599497, -0.03601882, -0.03604210, -0.03606463, -0.03608645, -0.03610773,
    -0.03612835, -0.03614836, -0.03616768, -0.03618631, -0.03620431, -0.03622168, -0.03623839, -0.03625450,
    -0.03627005, -0.03628503, -0.03629932, -0.03631296, -0.03632598, -0.03633847, -0.03635038, -0.03636174,
    -0.03637250, -0.03638264, -0.03639216, -0.03640108, -0.03640939, -0.03641717, -0.03642438, -0.03643106,
    -0.03643709, -0.03644258, -0.03644748, -0.03645181, -0.03645561, -0.03645881, -0.03646151, -0.03646367,
    -0.03646530, -0.03646631, -0.03646684, -0.03646674, -0.03646602, -0.03646472, -0.03646297, -0.03646074,
    -0.03645806, -0.03645487, -0.03645102, -0.03644667, -0.03644175, -0.03643632, -0.03643045, -0.03642403,
    -0.03641717, -0.03640988, -0.03640208, -0.03639374, -0.03638486, -0.03637550, -0.03636551, -0.03635498,
    -0.03634391, -0.03633231, -0.03632014, -0.03630763, -0.03629470, -0.03628122, -0.03626727, -0.03625272,
    -0.03623763, -0.03622199, -0.03620586, -0.03618916, -0.03617198, -0.03615439, -0.03613635, -0.03611781,
    -0.03609870, -0.03607906, -0.03605882, -0.03603807, -0.03601676, -0.03599492, -0.03597259, -0.03594984,
    -0.03592654, -0.03590272, -0.03587838, -0.03585353, -0.03582817, -0.03580239, -0.03577611, -0.03574945,
    -0.03572249, -0.03569521, -0.03566761, -0.03563950, -0.03561093, -0.03558190, -0.03555245, -0.03552254,
    -0.03549221, -0.03546147, -0.03543020, -0.03539856, -0.03536657, -0.03533405, -0.03530107, -0.03526770,
    -0.03523396, -0.03519978, -0.03516531, -0.03513042, -0.03509512, -0.03505951, -0.03502372, -0.03498764,
    -0.03495120, -0.03491428, -0.03487697, -0.03483931, -0.03480125, -0.03476283, -0.03472405, -0.03468486,
    -0.03464545, -0.03460576, -0.03456561, -0.03452506, -0.03448406, -0.03444270, -0.03440093, -0.03435868,
    -0.03431606, -0.03427303, -0.03422971, -0.03418606, -0.03414212, -0.03409779, -0.03405301, -0.03400776,
    -0.03396208, -0.03391605, -0.03386957, -0.03382267, -0.03377538, -0.03372800, -0.03368010, -0.03363166,
    -0.03358280, -0.03353354, -0.03348386, -0.03343371, -0.03338318, -0.03333228, -0.03328091, -0.03322929,
    -0.03317728, -0.03312482, -0.03307187, -0.03301850, -0.03296473, -0.03291048, -0.03285585, -0.03280075,
    -0.03274511, -0.03268922, -0.03263287, -0.03257599, -0.03251866, -0.03246094, -0.03240277, -0.03234405,
    -0.03228488, -0.03222529, -0.03216523, -0.03210483, -0.03204412, -0.03198296, -0.03192145, -0.03185941,
    -0.03179673, -0.03173373, -0.03167030, -0.03160634, -0.03154205, -0.03147731, -0.03141236, -0.03134714,
    -0.03128146, -0.03121519, -0.03114855, -0.03108153, -0.03101401, -0.03094607, -0.03087780, -0.03080908,
    -0.03074024, -0.03067110, -0.03060149, -0.03053140, -0.03046100, -0.03039022, -0.03031896, -0.03024738,
    -0.03017548, -0.03010320, -0.03003072, -0.02995782, -0.02988463, -0.02981105, -0.02973687, -0.02966223,
    -0.02958726, -0.02951187, -0.02943599, -0.02935970, -0.02928309, -0.02920624, -0.02912890, -0.02905129,
    -0.02897328, -0.02889482, -0.02881596, -0.02873677, -0.02865708, -0.02857700, -0.02849654, -0.02841578,
    -0.02833468, -0.02825317, -0.02817117, -0.02808878, -0.02800621, -0.02792334, -0.02783999, -0.02775609,
    -0.02767181, -0.02758743, -0.02750263, -0.02741739, -0.02733178, -0.02724585, -0.02715948, -0.02707270,
    -0.02698558, -0.02689802, -0.02680999, -0.02672187, -0.02663339, -0.02654454, -0.02645518, -0.02636532,
    -0.02627522, -0.02618472, -0.02609368, -0.02600227, -0.02591044, -0.02581830, -0.02572573, -0.02563273,
    -0.02553920, -0.02544516, -0.02535077, -0.02525613, -0.02516088, -0.02506522, -0.02496913, -0.02487276,
    -0.02477599, -0.02467869, -0.02458091, -0.02448272, -0.02438410, -0.02428506, -0.02418560, -0.02408555,
    -0.02398503, -0.02388409, -0.02378300, -0.02368153, -0.02357962, -0.02347728, -0.02337452, -0.02327135,
    -0.02316767, -0.02306357, -0.02295886, -0.02285371, -0.02274818, -0.02264224, -0.02253584, -0.02242896,
    -0.02232174, -0.02221403, -0.02210588, -0.02199720, -0.02188807, -0.02177841, -0.02166840, -0.02155796,
    -0.02144709, -0.02133579, -0.02122389, -0.02111151, -0.02099879, -0.02088554, -0.02077186, -0.02065786,
    -0.02054348, -0.02042852, -0.02031323, -0.02019760, -0.02008134, -0.01996468, -0.01984768, -0.01973028,
    -0.01961242, -0.01949408, -0.01937530, -0.01925612, -0.01913650, -0.01901646, -0.01889604, -0.01877485,
    -0.01865308, -0.01853067, -0.01840764, -0.01828390, -0.01815937, -0.01803439, -0.01790873, -0.01778231,
    -0.01765525, -0.01752768, -0.01739967, -0.01727136, -0.01714254, -0.01701325, -0.01688353, -0.01675350,
    -0.01662305, -0.01649209, -0.01636058, -0.01622845, -0.01609593, -0.01596322, -0.01582997, -0.01569629,
    -0.01556213, -0.01542766, -0.01529277, -0.01515758, -0.01502197, -0.01488613, -0.01475005, -0.01461369,
    -0.01447692, -0.01433997, -0.01420284, -0.01406561, -0.01392816, -0.01379033, -0.01365225, -0.01351381,
    -0.01337487, -0.01323562, -0.01309598, -0.01295600, -0.01281558, -0.01267476, -0.01253351, -0.01239184,
    -0.01224976, -0.01210743, -0.01196473, -0.01182152, -0.01167779, -0.01153375, -0.01138952, -0.01124491,
    -0.01110006, -0.01095480, -0.01080910, -0.01066301, -0.01051664, -0.01036987, -0.01022263, -0.01007491,
    -0.00992686, -0.00977830, -0.00962921, -0.00947989, -0.00933022, -0.00918025, -0.00902974, -0.00887877,
    -0.00872734, -0.00857552, -0.00842331, -0.00827084, -0.00811794, -0.00796466, -0.00781099, -0.00765679,
    -0.00750212, -0.00734719, -0.00719161, -0.00703553, -0.00687904, -0.00672214, -0.00656489, -0.00640749,
    -0.00624941, -0.00609087, -0.00593183, -0.00577241, -0.00561244, -0.00545204, -0.00529102, -0.00512970,
    -0.00496779, -0.00480546, -0.00464242, -0.00447878, -0.00431468, -0.00414998, -0.00398467, -0.00381868,
    -0.00365195, -0.00348471, -0.00331710, -0.00314917, -0.00298081, -0.00281181, -0.00264203, -0.00247187,
    -0.00230129, -0.00213041, -0.00195880, -0.00178664, -0.00161392, -0.00144086, -0.00126742, -0.00109343,
    -0.00091885, -0.00074376, -0.00056816, -0.00039189, -0.00021526, -0.00003841, 0.00013889, 0.00031672,
    0.00049454, 0.00067285, 0.00085141, 0.00103022, 0.00120953, 0.00138911, 0.00156928, 0.00174976,
    0.00193062, 0.00211194, 0.00229342, 0.00247530, 0.00265754, 0.00284009, 0.00302297, 0.00320629,
    0.00339023, 0.00357467, 0.00375942, 0.00394465, 0.00412990, 0.00431534, 0.00450120, 0.00468732,
    0.00487399, 0.00506123, 0.00524898, 0.00543724, 0.00562571, 0.00581444, 0.00600317, 0.00619220,
    0.00638177, 0.00657164, 0.00676194, 0.00695250, 0.00714353, 0.00733513, 0.00752690, 0.00771894,
    0.00791124, 0.00810350, 0.00829616, 0.00848927, 0.00868297, 0.00887714, 0.00907173, 0.00926659,
    0.00946167, 0.00965708, 0.00985272, 0.01004835, 0.01024472, 0.01044162, 0.01063897, 0.01083673,
    0.01103515, 0.01123390, 0.01143302, 0.01163246, 0.01183230, 0.01203209, 0.01223234, 0.01243312,
    0.01263429, 0.01283589, 0.01303791, 0.01324075, 0.01344416, 0.01364830, 0.01385287, 0.01405732,
    0.01426189, 0.01446701, 0.01467249, 0.01487877, 0.01508553, 0.01529243, 0.01549970, 0.01570735,
    0.01591524, 0.01612325, 0.01633149, 0.01653964, 0.01674808, 0.01695718, 0.01716665, 0.01737638,
    0.01758647, 0.01779674, 0.01800720, 0.01821791, 0.01842861, 0.01863974, 0.01885109, 0.01906238,
    0.01927413, 0.01948622, 0.01969859, 0.01991134, 0.02012457, 0.02033795, 0.02055052, 0.02076334,
    0.02097614, 0.02118909, 0.02140257, 0.02161672, 0.02183073, 0.02204468, 0.02225936, 0.02247434,
    0.02268879, 0.02290243, 0.02311654, 0.02333120, 0.02354619, 0.02376138, 0.02397638, 0.02419221,
    0.02440812, 0.02462415, 0.02483977, 0.02505449, 0.02526956, 0.02548532, 0.02570141, 0.02591753,
    0.02613366, 0.02635010, 0.02656685, 0.02678406, 0.02700066, 0.02721644, 0.02743297, 0.02765008,
    0.02786691, 0.02808376, 0.02830037, 0.02851711, 0.02873420, 0.02895173, 0.02916914, 0.02938571,
    0.02960230, 0.02981918, 0.03003642, 0.03025388, 0.03047129, 0.03068832, 0.03090576, 0.03112341,
    0.03134130, 0.03155832, 0.03177516, 0.03199213, 0.03220943, 0.03242669, 0.03264375, 0.03286111,
    0.03307851, 0.03329647, 0.03351449, 0.03373148, 0.03394802, 0.03416480, 0.03438177, 0.03459905,
    0.03481659, 0.03503397, 0.03525162, 0.03546922, 0.03568671, 0.03590381, 0.03611971, 0.03633615,
    0.03655233, 0.03676820, 0.03698390, 0.03719954, 0.03741477, 0.03762999, 0.03784502, 0.03805927,
    0.03827239, 0.03848548, 0.03869846, 0.03891112, 0.03912377, 0.03933628, 0.03954853, 0.03976029,
    0.03997216, 0.04018378, 0.04039365, 0.04060349, 0.04081284, 0.04102173, 0.04123006, 0.04143837,
    0.04164652, 0.04185434, 0.04206080, 0.04226710, 0.04247193, 0.04267628, 0.04288001, 0.04308339,
    0.04328656, 0.04348910, 0.04369158, 0.04389370, 0.04409512, 0.04429601, 0.04449555, 0.04469309,
    0.04489005, 0.04508657, 0.04528221, 0.04547716, 0.04567085, 0.04586390, 0.04605641, 0.04624848,
    0.04643978, 0.04662839, 0.04681564, 0.04700235, 0.04718857, 0.04737394, 0.04755836, 0.04774268,
    0.04792619, 0.04810848, 0.04829004, 0.04846807, 0.04864474, 0.04882018, 0.04899443, 0.04916763,
    0.04933912, 0.04950947, 0.04967827, 0.04984503, 0.05001084, 0.05017339, 0.05033537, 0.05049553,
    0.05065465, 0.05081302, 0.05097031, 0.05112593, 0.05128044, 0.05143252, 0.05158333, 0.05173167,
    0.05187521, 0.05201761, 0.05215924, 0.05229866, 0.05243674, 0.05257302, 0.05270632, 0.05283815,
    0.05296871, 0.05309521, 0.05321711, 0.05333508, 0.05345026, 0.05356186, 0.05367134, 0.05377853,
    0.05388269, 0.05398251, 0.05407914, 0.05417283, 0.05425936, 0.05434245, 0.05442081, 0.05449691,
    0.05456919, 0.05463745, 0.05470278, 0.05476401, 0.05482035, 0.05487145, 0.05491466, 0.05495483,
    0.05499070, 0.05502202, 0.05504626, 0.05506480, 0.05507908, 0.05508837, 0.05509176, 0.05509037,
    0.05508130, 0.05506469, 0.05504226, 0.05501149, 0.05497314, 0.05492907, 0.05487915, 0.05482218,
    0.05475930, 0.05468910, 0.05461169, 0.05452295, 0.05442757, 0.05432349, 0.05421169, 0.05409030,
    0.05396074, 0.05382497, 0.05367777, 0.05352149, 0.05335712, 0.05318157, 0.05299479, 0.05279605,
    0.05258339, 0.05235704, 0.05211555, 0.05186179, 0.05159240, 0.05132070, 0.05103851, 0.05074100,
    0.05043067, 0.05009421, 0.04973575, 0.04935512, 0.04895530, 0.04854740, 0.04811690, 0.04764371,
    0.04706587, 0.04650470, 0.04593629, 0.04534170, 0.04474376, 0.04412589, 0.04348096, 0.04279121,
    0.04205265, 0.04128471, 0.04048974, 0.03877263, 0.03788250, 0.03701317, 0.03610274, 0.03513388,
    0.03413506, 0.03308996, 0.03196537, 0.03074525, 0.02939912, 0.02658509, 0.02261612, 0.02050075,
    0.01859532, 0.01673826, 0.01467582, 0.01015304, 0.00753902, 0.00501200, 0.00259783, 0.00000000
};

const double GainRobustTracker::hinv_3_[1024] = {
    0.00000000, -0.00196684, -0.00359118, -0.00499315, -0.00623904, -0.00736154, -0.00838951, -0.00933980,
    -0.01022619, -0.01106100, -0.01185191, -0.01259061, -0.01329160, -0.01396254, -0.01460719, -0.01522683,
    -0.01582421, -0.01639999, -0.01695729, -0.01749608, -0.01801787, -0.01851602, -0.01899179, -0.01945418,
    -0.01990387, -0.02034123, -0.02076620, -0.02117981, -0.02158230, -0.02197431, -0.02235616, -0.02272505,
    -0.02307672, -0.02342042, -0.02375600, -0.02408425, -0.02440564, -0.02472019, -0.02502813, -0.02532968,
    -0.02562541, -0.02591441, -0.02618794, -0.02645629, -0.02671985, -0.02697872, -0.02723388, -0.02748576,
    -0.02773307, -0.02797437, -0.02821109, -0.02844384, -0.02866481, -0.02887993, -0.02909166, -0.02929993,
    -0.02950482, -0.02970608, -0.02990367, -0.03009803, -0.03028875, -0.03047628, -0.03065546, -0.03082844,
    -0.03099849, -0.03116545, -0.03132970, -0.03149063, -0.03164834, -0.03180300, -0.03195478, -0.03210327,
    -0.03224521, -0.03237865, -0.03250892, -0.03263511, -0.03275758, -0.03287599, -0.03299138, -0.03310449,
    -0.03321632, -0.03332669, -0.03343441, -0.03353524, -0.03363447, -0.03373225, -0.03382829, -0.03392293,
    -0.03401612, -0.03410786, -0.03419812, -0.03428680, -0.03437388, -0.03445333, -0.03453058, -0.03460547,
    -0.03467862, -0.03474991, -0.03481909, -0.03488623, -0.03495141, -0.03501494, -0.03507679, -0.03513295,
    -0.03518527, -0.03523577, -0.03528423, -0.03533110, -0.03537587, -0.03541824, -0.03545852, -0.03549612,
    -0.03553209, -0.03556406, -0.03559189, -0.03561836, -0.03564298, -0.03566592, -0.03568723, -0.03570689,
    -0.03572536, -0.03574251, -0.03575848, -0.03577326, -0.03578487, -0.03579626, -0.03580776, -0.03581877,
    -0.03582928, -0.03583928, -0.03584837, -0.03585611, -0.03586263, -0.03586778, -0.03586777, -0.03586676,
    -0.03586409, -0.03585986, -0.03585433, -0.03584722, -0.03583791, -0.03582700, -0.03581450, -0.03580064,
    -0.03578282, -0.03576323, -0.03574255, -0.03572062, -0.03569772, -0.03567434, -0.03565025, -0.03562539,
    -0.03559966, -0.03557335, -0.03554447, -0.03551383, -0.03548306, -0.03545177, -0.03541966, -0.03538698,
    -0.03535371, -0.03531956, -0.03528466, -0.03524860, -0.03521048, -0.03516928, -0.03512678, -0.03508278,
    -0.03503683, -0.03498877, -0.03493888, -0.03488733, -0.03483457, -0.03478079, -0.03472583, -0.03466768,
    -0.03460757, -0.03454567, -0.03448217, -0.03441719, -0.03435115, -0.03428442, -0.03421753, -0.03415069,
    -0.03408377, -0.03401517, -0.03394597, -0.03387597, -0.03380535, -0.03373329, -0.03365997, -0.03358578,
    -0.03351067, -0.03343428, -0.03335670, -0.03327714, -0.03319585, -0.03311291, -0.03302862, -0.03294302,
    -0.03285606, -0.03276788, -0.03267851, -0.03258794, -0.03249626, -0.03240283, -0.03230723, -0.03220990,
    -0.03211058, -0.03200970, -0.03190722, -0.03180380, -0.03169973, -0.03159527, -0.03149043, -0.03138474,
    -0.03127840, -0.03117102, -0.03106265, -0.03095331, -0.03084328, -0.03073346, -0.03062434, -0.03051620,
    -0.03040860, -0.03030118, -0.03019367, -0.03008615, -0.02997844, -0.02987045, -0.02976202, -0.02965337,
    -0.02954412, -0.02943475, -0.02932485, -0.02921462, -0.02910414, -0.02899348, -0.02888244, -0.02877074,
    -0.02865821, -0.02854463, -0.02843042, -0.02831500, -0.02819846, -0.02808050, -0.02796133, -0.02784078,
    -0.02771842, -0.02759459, -0.02746935, -0.02734355, -0.02721726, -0.02709102, -0.02696490, -0.02683883,
    -0.02671293, -0.02658809, -0.02646330, -0.02633841, -0.02621351, -0.02608870, -0.02596409, -0.02583992,
    -0.02571572, -0.02559167, -0.02546747, -0.02534397, -0.02522034, -0.02509622, -0.02497177, -0.02484657,
    -0.02472044, -0.02459337, -0.02446517, -0.02433571, -0.02420496, -0.02407371, -0.02394179, -0.02380879,
    -0.02367483, -0.02353998, -0.02340412, -0.02326734, -0.02312955, -0.02299089, -0.02285134, -0.02271140,
    -0.02257144, -0.02243179, -0.02229282, -0.02215476, -0.02201766, -0.02188115, -0.02174519, -0.02160985,
    -0.02147536, -0.02134149, -0.02120844, -0.02107539, -0.02094222, -0.02080879, -0.02067517, -0.02054106,
    -0.02040646, -0.02027174, -0.02013647, -0.02000085, -0.01986546, -0.01972941, -0.01959259, -0.01945515,
    -0.01931705, -0.01917788, -0.01903774, -0.01889622, -0.01875334, -0.01860944, -0.01846543, -0.01832141,
    -0.01817748, -0.01803336, -0.01788895, -0.01774428, -0.01759905, -0.01745355, -0.01730772, -0.01716162,
    -0.01701596, -0.01687088, -0.01672629, -0.01658218, -0.01643854, -0.01629489, -0.01615134, -0.01600750,
    -0.01586341, -0.01571896, -0.01557404, -0.01542910, -0.01528392, -0.01513796, -0.01499090, -0.01484271,
    -0.01469360, -0.01454325, -0.01439152, -0.01423813, -0.01408320, -0.01392718, -0.01376950, -0.01361051,
    -0.01345110, -0.01329235, -0.01313460, -0.01297773, -0.01282185, -0.01266683, -0.01251242, -0.01235853,
    -0.01220519, -0.01205217, -0.01189931, -0.01174701, -0.01159466, -0.01144194, -0.01128911, -0.01113615,
    -0.01098294, -0.01082924, -0.01067501, -0.01051984, -0.01036434, -0.01020873, -0.01005245, -0.00989547,
    -0.00973804, -0.00957985, -0.00942082, -0.00926136, -0.00910115, -0.00894041, -0.00877963, -0.00861827,
    -0.00845607, -0.00829373, -0.00813102, -0.00796784, -0.00780487, -0.00764193, -0.00747929, -0.00731738,
    -0.00715556, -0.00699330, -0.00683116, -0.00666924, -0.00650726, -0.00634522, -0.00618365, -0.00602222,
    -0.00586029, -0.00569869, -0.00553675, -0.00537449, -0.00521214, -0.00504967, -0.00488685, -0.00472399,
    -0.00456102, -0.00439773, -0.00423372, -0.00406892, -0.00390360, -0.00373775, -0.00357073, -0.00340351,
    -0.00323587, -0.00306766, -0.00289893, -0.00272981, -0.00256036, -0.00239112, -0.00222249, -0.00205428,
    -0.00188663, -0.00171894, -0.00155151, -0.00138440, -0.00121758, -0.00105158, -0.00088581, -0.00072012,
    -0.00055473, -0.00038958, -0.00022434, -0.00005922, 0.00010598, 0.00027121, 0.00043702, 0.00060366,
    0.00077139, 0.00094004, 0.00111021, 0.00128155, 0.00145337, 0.00162521, 0.00179767, 0.00197035,
    0.00214334, 0.00231726, 0.00249220, 0.00266814, 0.00284477, 0.00302195, 0.00319932, 0.00337603,
    0.00355114, 0.00372489, 0.00389757, 0.00406971, 0.00424139, 0.00441272, 0.00458378, 0.00475394,
    0.00492403, 0.00509390, 0.00526333, 0.00543235, 0.00560143, 0.00577047, 0.00593940, 0.00610805,
    0.00627609, 0.00644428, 0.00661221, 0.00677971, 0.00694669, 0.00711333, 0.00727978, 0.00744602,
    0.00761219, 0.00777805, 0.00794352, 0.00810863, 0.00827379, 0.00843854, 0.00860330, 0.00876835,
    0.00893422, 0.00910018, 0.00926675, 0.00943355, 0.00959983, 0.00976601, 0.00993224, 0.01009857,
    0.01026489, 0.01043112, 0.01059739, 0.01076378, 0.01092990, 0.01109568, 0.01126048, 0.01142460,
    0.01158801, 0.01175065, 0.01191302, 0.01207466, 0.01223591, 0.01239666, 0.01255705, 0.01271685,
    0.01287597, 0.01303478, 0.01319331, 0.01335189, 0.01351070, 0.01366987, 0.01382948, 0.01398999,
    0.01415172, 0.01431458, 0.01447820, 0.01464091, 0.01480350, 0.01496528, 0.01512640, 0.01528697,
    0.01544713, 0.01560691, 0.01576672, 0.01592610, 0.01608575, 0.01624329, 0.01639994, 0.01655645,
    0.01671218, 0.01686687, 0.01702112, 0.01717496, 0.01732864, 0.01748210, 0.01763514, 0.01778652,
    0.01793727, 0.01808838, 0.01823991, 0.01839187, 0.01854457, 0.01869797, 0.01885149, 0.01900568,
    0.01915961, 0.01931247, 0.01946403, 0.01961523, 0.01976594, 0.01991587, 0.02006527, 0.02021437,
    0.02036332, 0.02051137, 0.02065815, 0.02080310, 0.02094557, 0.02108679, 0.02122632, 0.02136482,
    0.02150281, 0.02164089, 0.02177871, 0.02191563, 0.02205207, 0.02218757, 0.02232051, 0.02245280,
    0.02258438, 0.02271551, 0.02284626, 0.02297666, 0.02310622, 0.02323546, 0.02336454, 0.02349375,
    0.02362020, 0.02374685, 0.02387338, 0.02400049, 0.02412878, 0.02425854, 0.02438938, 0.02452046,
    0.02465088, 0.02478022, 0.02490717, 0.02503224, 0.02515708, 0.02528110, 0.02540465, 0.02552781,
    0.02565030, 0.02577199, 0.02589283, 0.02601235, 0.02612937, 0.02624383, 0.02635685, 0.02646946,
    0.02658120, 0.02669239, 0.02680348, 0.02691442, 0.02702523, 0.02713526, 0.02724457, 0.02735194,
    0.02746013, 0.02756777, 0.02767555, 0.02778380, 0.02789195, 0.02799997, 0.02810846, 0.02821642,
    0.02832466, 0.02843016, 0.02853420, 0.02863659, 0.02873672, 0.02883550, 0.02893277, 0.02902819,
    0.02912165, 0.02921310, 0.02930243, 0.02938811, 0.02947135, 0.02955326, 0.02963369, 0.02971291,
    0.02979015, 0.02986640, 0.02994159, 0.03001591, 0.03008902, 0.03016003, 0.03022845, 0.03029631,
    0.03036281, 0.03042880, 0.03049392, 0.03055882, 0.03062330, 0.03068711, 0.03075038, 0.03081252,
    0.03087263, 0.03093255, 0.03099232, 0.03105140, 0.03111032, 0.03116886, 0.03122663, 0.03128395,
    0.03134037, 0.03139612, 0.03144927, 0.03150173, 0.03155311, 0.03160231, 0.03164882, 0.03169281,
    0.03173454, 0.03177476, 0.03181250, 0.03184878, 0.03188169, 0.03191346, 0.03194373, 0.03197212,
    0.03199901, 0.03202554, 0.03205045, 0.03207361, 0.03209555, 0.03211650, 0.03213518, 0.03215201,
    0.03216799, 0.03218322, 0.03219685, 0.03220924, 0.03222039, 0.03223028, 0.03224050, 0.03225093,
    0.03226026, 0.03226707, 0.03227286, 0.03227752, 0.03228153, 0.03228387, 0.03228520, 0.03228611,
    0.03228593, 0.03228524, 0.03228232, 0.03227557, 0.03226718, 0.03225747, 0.03224612, 0.03223264,
    0.03221771, 0.03220026, 0.03218160, 0.03216118, 0.03213981, 0.03211469, 0.03208782, 0.03206007,
    0.03203126, 0.03200167, 0.03197165, 0.03194031, 0.03190743, 0.03187400, 0.03183946, 0.03180115,
    0.03175928, 0.03171587, 0.03167146, 0.03162530, 0.03157741, 0.03152789, 0.03147663, 0.03142246,
    0.03136501, 0.03130346, 0.03123699, 0.03116953, 0.03109936, 0.03102779, 0.03095295, 0.03087560,
    0.03079631, 0.03071589, 0.03063382, 0.03054973, 0.03045998, 0.03036949, 0.03027739, 0.03018344,
    0.03008859, 0.02999360, 0.02989827, 0.02980283, 0.02970704, 0.02960970, 0.02950349, 0.02939527,
    0.02928523, 0.02917266, 0.02905757, 0.02894071, 0.02882180, 0.02869959, 0.02857575, 0.02845008,
    0.02831566, 0.02817780, 0.02803695, 0.02789301, 0.02774673, 0.02759854, 0.02744850, 0.02729682,
    0.02714465, 0.02699143, 0.02683162, 0.02666496, 0.02649675, 0.02632743, 0.02615595, 0.02598346,
    0.02581048, 0.02563700, 0.02546115, 0.02528375, 0.02510156, 0.02491103, 0.02471859, 0.02452439,
    0.02432807, 0.02412877, 0.02392610, 0.02372015, 0.02351167, 0.02329986, 0.02308513, 0.02285558,
    0.02262340, 0.02238864, 0.02215123, 0.02191134, 0.02166893, 0.02142404, 0.02117648, 0.02092594,
    0.02067282, 0.02040455, 0.02013124, 0.01985568, 0.01957826, 0.01929783, 0.01901505, 0.01872934,
    0.01844247, 0.01815118, 0.01785771, 0.01755223, 0.01723858, 0.01692252, 0.01660536, 0.01628761,
    0.01596799, 0.01564605, 0.01532118, 0.01499559, 0.01466648, 0.01432733, 0.01397388, 0.01361818,
    0.01326007, 0.01290029, 0.01253904, 0.01217461, 0.01180746, 0.01143719, 0.01106476, 0.01068498,
    0.01028033, 0.00987262, 0.00946176, 0.00904934, 0.00863453, 0.00821637, 0.00779541, 0.00737077,
    0.00694393, 0.00651484, 0.00606136, 0.00560211, 0.00514105, 0.00467773, 0.00421315, 0.00374692,
    0.00328135, 0.00281269, 0.00234143, 0.00186754, 0.00137286, 0.00086571, 0.00035597, -0.00015658,
    -0.00067244, -0.00119456, -0.00171943, -0.00224873, -0.00278138, -0.00331512, -0.00386649, -0.00443873,
    -0.00501560, -0.00559457, -0.00617583, -0.00675983, -0.00734646, -0.00793634, -0.00852859, -0.00912207,
    -0.00972508, -0.01035609, -0.01098837, -0.01162109, -0.01225259, -0.01288795, -0.01352328, -0.01415860,
    -0.01479598, -0.01543840, -0.01608183, -0.01675517, -0.01743171, -0.01810839, -0.01878658, -0.01946581,
    -0.02014412, -0.02082697, -0.02151165, -0.02219821, -0.02288776, -0.02360226, -0.02432553, -0.02504945,
    -0.02577604, -0.02650347, -0.02723117, -0.02796133, -0.02869360, -0.02942647, -0.03016289, -0.03091634,
    -0.03168529, -0.03245680, -0.03322924, -0.03400264, -0.03477588, -0.03554869, -0.03632230, -0.03709553,
    -0.03786996, -0.03864972, -0.03944912, -0.04024791, -0.04104873, -0.04185212, -0.04265380, -0.04345909,
    -0.04426347, -0.04506652, -0.04586677, -0.04666974, -0.04749296, -0.04831592, -0.04913525, -0.04995770,
    -0.05078286, -0.05160197, -0.05242122, -0.05323735, -0.05404900, -0.05485658, -0.05567221, -0.05648935,
    -0.05730473, -0.05811605, -0.05892537, -0.05972585, -0.06052435, -0.06131953, -0.06210712, -0.06288800,
    -0.06366737, -0.06444464, -0.06521699, -0.06598779, -0.06675729, -0.06751857, -0.06827863, -0.06902994,
    -0.06977257, -0.07051074, -0.07123471, -0.07194008, -0.07264710, -0.07334053, -0.07402887, -0.07471274,
    -0.07539262, -0.07607043, -0.07672512, -0.07736703, -0.07799883, -0.07859128, -0.07916153, -0.07970872,
    -0.08023127, -0.08071652, -0.08117552, -0.08161300, -0.08202531, -0.08244722, -0.08285887, -0.08320079,
    -0.08351897, -0.08378219, -0.08401684, -0.08422844, -0.08442230, -0.08459404, -0.08471103, -0.08468607,
    -0.08432258, -0.08403595, -0.08378662, -0.08346766, -0.08314768, -0.08278308, -0.08236903, -0.08184399,
    -0.08119819, -0.08046697, -0.07966105, -0.07533033, -0.07419390, -0.07321201, -0.07215726, -0.07100417,
    -0.06984035, -0.06859966, -0.06718541, -0.06555212, -0.06358785, -0.05725615, -0.04660865, -0.04258397,
    -0.03927632, -0.03603078, -0.03204807, -0.01945943, -0.01376409, -0.00858815, -0.00424043, 0.00000000
};

const double GainRobustTracker::g_0_der_[1022] = {
    1.40043483, 1.21780426, 1.09440693, 1.00339523, 0.93247371, 0.87580974, 0.83036040, 0.79440809,
    0.76358766, 0.73250892, 0.70420149, 0.68135892, 0.66166617, 0.64402453, 0.62809643, 0.61384604,
    0.60059307, 0.58858305, 0.57813822, 0.56752459, 0.55671148, 0.54774489, 0.54026165, 0.53322852,
    0.52671712, 0.52076326, 0.51539251, 0.51042585, 0.50570471, 0.50052833, 0.49416526, 0.48872802,
    0.48496850, 0.48139311, 0.47811951, 0.47506074, 0.47202243, 0.46908130, 0.46616576, 0.46328601,
    0.45878993, 0.45471327, 0.45275934, 0.45090260, 0.44950620, 0.44940390, 0.44874406, 0.44515845,
    0.44108691, 0.43845780, 0.43476477, 0.43096432, 0.42890298, 0.42717411, 0.42553731, 0.42405907,
    0.42265245, 0.42136858, 0.42023817, 0.41914356, 0.41706175, 0.41430988, 0.41262705, 0.41173704,
    0.41080611, 0.40999794, 0.40938926, 0.40880614, 0.40831510, 0.40793148, 0.40704147, 0.40539444,
    0.40436633, 0.40418219, 0.40408500, 0.40411569, 0.40407477, 0.40383437, 0.40331263, 0.40270395,
    0.40191113, 0.40006461, 0.39850454, 0.39800326, 0.39754803, 0.39706722, 0.39663756, 0.39631532,
    0.39604422, 0.39581916, 0.39569640, 0.39472455, 0.39365551, 0.39356856, 0.39352253, 0.39338442,
    0.39322074, 0.39313378, 0.39304683, 0.39292407, 0.39271436, 0.39188061, 0.39094456, 0.39065301,
    0.39053025, 0.39034100, 0.39026939, 0.39034099, 0.39058140, 0.39079623, 0.39087807, 0.39071439,
    0.39017220, 0.38994714, 0.39000341, 0.39001875, 0.39023870, 0.39043818, 0.39051491, 0.39051491,
    0.39038192, 0.39002898, 0.38923615, 0.38840752, 0.38790114, 0.38753798, 0.38727711, 0.38712366,
    0.38704694, 0.38700602, 0.38710320, 0.38731292, 0.38721062, 0.38706228, 0.38725153, 0.38741010,
    0.38756867, 0.38791137, 0.38832568, 0.38873489, 0.38914920, 0.38950214, 0.38966070, 0.38976811,
    0.38997783, 0.39016708, 0.39024381, 0.39028984, 0.39036146, 0.39039215, 0.39048933, 0.39061209,
    0.39073485, 0.39081158, 0.39079623, 0.39077577, 0.39084738, 0.39098037, 0.39109290, 0.39114917,
    0.39127704, 0.39161463, 0.39205963, 0.39253533, 0.39302637, 0.39359925, 0.39421305, 0.39483708,
    0.39554295, 0.39626928, 0.39681658, 0.39713371, 0.39748665, 0.39819252, 0.39895977, 0.39949684,
    0.40005950, 0.40072956, 0.40132801, 0.40169118, 0.40183952, 0.40190601, 0.40199297, 0.40233056,
    0.40277044, 0.40312338, 0.40347120, 0.40388552, 0.40435098, 0.40481644, 0.40528702, 0.40569623,
    0.40616680, 0.40698520, 0.40800821, 0.40878568, 0.40936880, 0.40997748, 0.41062197, 0.41120508,
    0.41177284, 0.41237130, 0.41299021, 0.41386999, 0.41505156, 0.41611548, 0.41703618, 0.41789038,
    0.41861160, 0.41924586, 0.41976759, 0.42013587, 0.42042231, 0.42083662, 0.42173687, 0.42269337,
    0.42330717, 0.42383401, 0.42428925, 0.42463196, 0.42477518, 0.42474449, 0.42470357, 0.42475472,
    0.42548104, 0.42633525, 0.42663703, 0.42698997, 0.42735825, 0.42766515, 0.42802320, 0.42843240,
    0.42879045, 0.42914850, 0.42991575, 0.43088760, 0.43150140, 0.43196175, 0.43247325, 0.43298475,
    0.43354740, 0.43421235, 0.43492845, 0.43574685, 0.43682100, 0.43830435, 0.43968540, 0.44060610,
    0.44147565, 0.44214060, 0.44254980, 0.44295900, 0.44326590, 0.44352165, 0.44403315, 0.44505615,
    0.44592570, 0.44638605, 0.44679525, 0.44715330, 0.44746020, 0.44771595, 0.44802285, 0.44838090,
    0.44889240, 0.45001770, 0.45109185, 0.45165450, 0.45221715, 0.45283095, 0.45359820, 0.45446775,
    0.45538845, 0.45630915, 0.45728100, 0.45881550, 0.46040115, 0.46142415, 0.46229370, 0.46321440,
    0.46423740, 0.46510695, 0.46592535, 0.46689720, 0.46781790, 0.46909665, 0.47068230, 0.47160300,
    0.47190990, 0.47211450, 0.47231910, 0.47252370, 0.47272830, 0.47298405, 0.47329095, 0.47380245,
    0.47487660, 0.47579730, 0.47630880, 0.47687145, 0.47738295, 0.47789445, 0.47850825, 0.47912205,
    0.47973585, 0.48045195, 0.48188415, 0.48331635, 0.48398130, 0.48469740, 0.48546465, 0.48618075,
    0.48694800, 0.48781755, 0.48873825, 0.48950550, 0.49098885, 0.49242105, 0.49293255, 0.49344405,
    0.49385325, 0.49431360, 0.49487625, 0.49528545, 0.49574580, 0.49625730, 0.49722915, 0.49840560,
    0.49886595, 0.49901940, 0.49932630, 0.49963320, 0.50004240, 0.50060505, 0.50111655, 0.50167920,
    0.50259990, 0.50418555, 0.50541315, 0.50607810, 0.50689650, 0.50781720, 0.50873790, 0.50960745,
    0.51057930, 0.51170460, 0.51298335, 0.51487590, 0.51666615, 0.51779145, 0.51866100, 0.51912135,
    0.51942825, 0.51978630, 0.52009320, 0.52024665, 0.52050240, 0.52167885, 0.52295760, 0.52341795,
    0.52392945, 0.52449210, 0.52495245, 0.52556625, 0.52633350, 0.52715190, 0.52812375, 0.52976055,
    0.53149965, 0.53267610, 0.53369910, 0.53461980, 0.53548935, 0.53656350, 0.53763765, 0.53855835,
    0.53953020, 0.54101355, 0.54290610, 0.54428715, 0.54500325, 0.54577050, 0.54684465, 0.54771420,
    0.54848145, 0.54940215, 0.55011825, 0.55098780, 0.55242000, 0.55344300, 0.55385220, 0.55446600,
    0.55523325, 0.55579590, 0.55635855, 0.55702350, 0.55763730, 0.55819995, 0.55958100, 0.56096205,
    0.56157585, 0.56224080, 0.56285460, 0.56346840, 0.56423565, 0.56495175, 0.56551440, 0.56617935,
    0.56750925, 0.56914605, 0.57011790, 0.57073170, 0.57155010, 0.57257310, 0.57344265, 0.57420990,
    0.57502830, 0.57579555, 0.57692085, 0.57819960, 0.57912030, 0.57983640, 0.58034790, 0.58091055,
    0.58157550, 0.58208700, 0.58259850, 0.58316115, 0.58403070, 0.58536060, 0.58633245, 0.58689510,
    0.58756005, 0.58842960, 0.58919685, 0.58986180, 0.59073135, 0.59170320, 0.59282850, 0.59461875,
    0.59646015, 0.59768775, 0.59876190, 0.59963145, 0.60050100, 0.60142170, 0.60229125, 0.60336540,
    0.60449070, 0.60617865, 0.60802005, 0.60919650, 0.61021950, 0.61103790, 0.61149825, 0.61185630,
    0.61247010, 0.61323735, 0.61395345, 0.61512990, 0.61661325, 0.61768740, 0.61855695, 0.61952880,
    0.62039835, 0.62126790, 0.62234205, 0.62336505, 0.62438805, 0.62561565, 0.62715015, 0.62848005,
    0.62945190, 0.63042375, 0.63134445, 0.63231630, 0.63328815, 0.63426000, 0.63528300, 0.63640830,
    0.63789165, 0.63927270, 0.64029570, 0.64131870, 0.64229055, 0.64336470, 0.64459230, 0.64602450,
    0.64750785, 0.64868430, 0.65026995, 0.65195790, 0.65308320, 0.65420850, 0.65533380, 0.65645910,
    0.65753325, 0.65860740, 0.65968155, 0.66065340, 0.66193215, 0.66331320, 0.66433620, 0.66505230,
    0.66587070, 0.66674025, 0.66755865, 0.66853050, 0.66960465, 0.67052535, 0.67170180, 0.67328745,
    0.67451505, 0.67553805, 0.67681680, 0.67814670, 0.67942545, 0.68085765, 0.68254560, 0.68428470,
    0.68597265, 0.68776290, 0.68919510, 0.69032040, 0.69134340, 0.69257100, 0.69405435, 0.69553770,
    0.69712335, 0.69865785, 0.70034580, 0.70223835, 0.70392630, 0.70530735, 0.70653495, 0.70771140,
    0.70883670, 0.70996200, 0.71108730, 0.71221260, 0.71338905, 0.71492355, 0.71656035, 0.71789025,
    0.71927130, 0.72075465, 0.72213570, 0.72356790, 0.72520470, 0.72663690, 0.72791565, 0.72939900,
    0.73078005, 0.73185420, 0.73292835, 0.73384905, 0.73451400, 0.73533240, 0.73635540, 0.73707150,
    0.73758300, 0.73829910, 0.73921980, 0.73993590, 0.74034510, 0.74095890, 0.74193075, 0.74300490,
    0.74407905, 0.74530665, 0.74643195, 0.74745495, 0.74904060, 0.75067740, 0.75190500, 0.75323490,
    0.75451365, 0.75579240, 0.75712230, 0.75834990, 0.75957750, 0.76100970, 0.76295340, 0.76499940,
    0.76653390, 0.76801725, 0.76980750, 0.77185350, 0.77400180, 0.77589435, 0.77737770, 0.77870760,
    0.78044670, 0.78223695, 0.78377145, 0.78525480, 0.78663585, 0.78801690, 0.78950025, 0.79108590,
    0.79256925, 0.79379685, 0.79517790, 0.79691700, 0.79865610, 0.80019060, 0.80172510, 0.80331075,
    0.80494755, 0.80663550, 0.80811885, 0.80965335, 0.81149475, 0.81410340, 0.81676320, 0.81870690,
    0.82065060, 0.82254315, 0.82438455, 0.82627710, 0.82811850, 0.83011335, 0.83246625, 0.83573985,
    0.83870655, 0.84075255, 0.84259395, 0.84433305, 0.84602100, 0.84755550, 0.84929460, 0.85108485,
    0.85272165, 0.85512570, 0.85778550, 0.85942230, 0.86100795, 0.86295165, 0.86463960, 0.86607180,
    0.86786205, 0.86970345, 0.87128910, 0.87343740, 0.87625065, 0.87855240, 0.88029150, 0.88203060,
    0.88382085, 0.88571340, 0.88745250, 0.88908930, 0.89067495, 0.89261865, 0.89558535, 0.89814285,
    0.89972850, 0.90131415, 0.90295095, 0.90448545, 0.90586650, 0.90714525, 0.90852630, 0.91011195,
    0.91318095, 0.91645455, 0.91824480, 0.91972815, 0.92105805, 0.92208105, 0.92310405, 0.92448510,
    0.92601960, 0.92734950, 0.93006045, 0.93318060, 0.93517545, 0.93706800, 0.93855135, 0.93993240,
    0.94162035, 0.94346175, 0.94535430, 0.94734915, 0.95026470, 0.95389635, 0.95670960, 0.95865330,
    0.96054585, 0.96243840, 0.96422865, 0.96617235, 0.96811605, 0.97031550, 0.97343565, 0.97773225,
    0.98115930, 0.98320530, 0.98520015, 0.98714385, 0.98898525, 0.99098010, 0.99333300, 0.99548130,
    0.99783420, 1.00243770, 1.00678545, 1.00872915, 1.01046825, 1.01230965, 1.01425335, 1.01650395,
    1.01875455, 1.02085170, 1.02279540, 1.02714315, 1.03169550, 1.03415070, 1.03721970, 1.04054445,
    1.04361345, 1.04678475, 1.05016065, 1.05322965, 1.05614520, 1.06110675, 1.06719360, 1.07138790,
    1.07409885, 1.07670750, 1.07946960, 1.08233400, 1.08494265, 1.08719325, 1.08964845, 1.09312665,
    1.09824165, 1.10228250, 1.10448195, 1.10663025, 1.10882970, 1.11087570, 1.11287055, 1.11532575,
    1.11788325, 1.12059420, 1.12586265, 1.13046615, 1.13251215, 1.13501850, 1.13762715, 1.14008235,
    1.14228180, 1.14443010, 1.14668070, 1.14882900, 1.15430205, 1.15951935, 1.16105385, 1.16238375,
    1.16371365, 1.16560620, 1.16801025, 1.17061890, 1.17302295, 1.17501780, 1.17967245, 1.18555470,
    1.18898175, 1.19123235, 1.19327835, 1.19568240, 1.19844450, 1.20084855, 1.20284340, 1.20555435,
    1.21072050, 1.21762575, 1.22222925, 1.22463330, 1.22760000, 1.23087360, 1.23389145, 1.23670470,
    1.23967140, 1.24309845, 1.24734390, 1.25419800, 1.26033600, 1.26345615, 1.26611595, 1.26836655,
    1.27056600, 1.27312350, 1.27537410, 1.27706205, 1.27931265, 1.28616675, 1.29302085, 1.29557835,
    1.29752205, 1.29961920, 1.30222785, 1.30524570, 1.30811010, 1.31036070, 1.31291820, 1.31946540,
    1.32677985, 1.33025805, 1.33312245, 1.33644720, 1.33956735, 1.34263635, 1.34555190, 1.34872320,
    1.35189450, 1.35772560, 1.36596075, 1.37153610, 1.37521890, 1.37890170, 1.38289140, 1.38657420,
    1.38984780, 1.39307025, 1.39629270, 1.40110080, 1.40861985, 1.41470670, 1.41823605, 1.42166310,
    1.42493670, 1.42836375, 1.43230230, 1.43603625, 1.43879835, 1.44237885, 1.45030710, 1.45757040,
    1.46109975, 1.46493600, 1.46831190, 1.47138090, 1.47501255, 1.47905340, 1.48299195, 1.48708395,
    1.49603520, 1.50519105, 1.50964110, 1.51373310, 1.51802970, 1.52237745, 1.52672520, 1.53153330,
    1.53664830, 1.54130295, 1.54902660, 1.55792670, 1.56283710, 1.56657105, 1.57061190, 1.57485735,
    1.57869360, 1.58206950, 1.58621265, 1.59025350, 1.59685185, 1.60708185, 1.61490780, 1.61956245,
    1.62411480, 1.62861600, 1.63311720, 1.63828335, 1.64365410, 1.64825760, 1.65388410, 1.66442100,
    1.67398605, 1.67884530, 1.68329535, 1.68861495, 1.69388340, 1.69858920, 1.70421570, 1.71009795,
    1.71526410, 1.72718205, 1.73971380, 1.74590295, 1.75137600, 1.75654215, 1.76232210, 1.76805090,
    1.77367740, 1.77930390, 1.78544190, 1.79730870, 1.81009620, 1.81741065, 1.82359980, 1.82943090,
    1.83475050, 1.84068390, 1.84764030, 1.85510820, 1.86211575, 1.87295955, 1.88850915, 1.89986445,
    1.90610475, 1.91239620, 1.91894340, 1.92503025, 1.93203780, 1.93884075, 1.94477415, 1.95505530,
    1.97249745, 1.98676830, 1.99490115, 2.00334090, 2.01060420, 2.01684450, 2.02380090, 2.03188260,
    2.04078270, 2.04968280, 2.06911980, 2.08871025, 2.09761035, 2.10564090, 2.11459215, 2.12482215,
    2.13372225, 2.14221315, 2.15172705, 2.16180360, 2.18354235, 2.20620180, 2.21704560, 2.22717330,
    2.23929585, 2.25167415, 2.26205760, 2.27106000, 2.28001125, 2.29013895, 2.31008745, 2.33704350,
    2.35453680, 2.36660820, 2.38046985, 2.39264355, 2.40328275, 2.41417770, 2.42568645, 2.43811590,
    2.45750175, 2.48890785, 2.51345985, 2.52665655, 2.54021130, 2.55427755, 2.56962255, 2.58348420,
    2.59862460, 2.61422535, 2.63176980, 2.66905815, 2.70302175, 2.71877595, 2.73744570, 2.75667810,
    2.77493865, 2.79355725, 2.81289195, 2.82936225, 2.84470725, 2.88936120, 2.93759565, 2.96434710,
    2.99222385, 3.02061210, 3.04930725, 3.07232475, 3.09324510, 3.12531615, 3.18204150, 3.24577440,
    3.29692440, 3.33584955, 3.36055500, 3.38505585, 3.41814990, 3.45953025, 3.50643480, 3.54694560,
    3.58126725, 3.85589160, 3.94473915, 3.78878280, 3.82965165, 3.89185005, 3.94959840, 4.00524960,
    4.08084930, 4.17491415, 4.28979705, 4.84845735, 5.69678010, 5.56900740, 5.03980950, 4.99694580,
    5.09454000, 5.99252940, 6.23702640, 5.68987485, 5.73432420, 5.87411715
};

const double GainRobustTracker::hinv_1_der_[1022] = {
    -0.34858183, -0.29844424, -0.26534554, -0.24172232, -0.22381041, -0.20948738, -0.19769475, -0.18791589,
    -0.17990580, -0.17353251, -0.16801701, -0.16282477, -0.15815836, -0.15395025, -0.15012372, -0.14670280,
    -0.14355861, -0.14061698, -0.13796127, -0.13570504, -0.13373065, -0.13178133, -0.12980131, -0.12792462,
    -0.12616454, -0.12448222, -0.12295284, -0.12157843, -0.12027155, -0.11905623, -0.11802709, -0.11693350,
    -0.11570283, -0.11454787, -0.11347883, -0.11245788, -0.11142823, -0.11040523, -0.10939604, -0.10841601,
    -0.10765234, -0.10692549, -0.10603139, -0.10512553, -0.10414754, -0.10293989, -0.10191637, -0.10151331,
    -0.10119567, -0.10058136, -0.10007907, -0.09965913, -0.09910364, -0.09852820, -0.09794611, -0.09738909,
    -0.09685969, -0.09634972, -0.09585919, -0.09537838, -0.09499834, -0.09467047, -0.09426638, -0.09384235,
    -0.09343059, -0.09306487, -0.09274876, -0.09243981, -0.09215440, -0.09188228, -0.09169200, -0.09163369,
    -0.09155492, -0.09138817, -0.09125927, -0.09117078, -0.09103575, -0.09078051, -0.09034420, -0.08979741,
    -0.08924089, -0.08878617, -0.08832070, -0.08778875, -0.08729259, -0.08682201, -0.08635143, -0.08587574,
    -0.08543073, -0.08501130, -0.08463279, -0.08435658, -0.08414175, -0.08389623, -0.08364048, -0.08339496,
    -0.08315967, -0.08294484, -0.08275559, -0.08258168, -0.08241800, -0.08229524, -0.08219805, -0.08212132,
    -0.08200368, -0.08185535, -0.08174281, -0.08166609, -0.08165586, -0.08164563, -0.08156891, -0.08149730,
    -0.08144614, -0.08135919, -0.08123643, -0.08113413, -0.08107787, -0.08102672, -0.08093464, -0.08077096,
    -0.08054079, -0.08023389, -0.07980423, -0.07930808, -0.07884261, -0.07841807, -0.07805490, -0.07776846,
    -0.07750248, -0.07725696, -0.07710351, -0.07698586, -0.07684265, -0.07670965, -0.07659712, -0.07649994,
    -0.07645391, -0.07643345, -0.07642322, -0.07642833, -0.07640787, -0.07637718, -0.07630046, -0.07619304,
    -0.07610097, -0.07597821, -0.07579918, -0.07560481, -0.07541556, -0.07519561, -0.07499613, -0.07481199,
    -0.07458693, -0.07432606, -0.07408054, -0.07384525, -0.07360485, -0.07339514, -0.07321100, -0.07302685,
    -0.07287340, -0.07277110, -0.07267904, -0.07255628, -0.07249489, -0.07253581, -0.07261766, -0.07273530,
    -0.07285294, -0.07292456, -0.07290409, -0.07282737, -0.07276088, -0.07267392, -0.07264323, -0.07271484,
    -0.07279668, -0.07285295, -0.07284271, -0.07274041, -0.07252559, -0.07221357, -0.07188109, -0.07152305,
    -0.07121614, -0.07103712, -0.07088367, -0.07076602, -0.07068418, -0.07059723, -0.07051027, -0.07042332,
    -0.07036706, -0.07029545, -0.07021872, -0.07019826, -0.07019315, -0.07020337, -0.07021872, -0.07021361,
    -0.07020337, -0.07020337, -0.07021360, -0.07020337, -0.07018292, -0.07025964, -0.07044378, -0.07060234,
    -0.07067395, -0.07065350, -0.07056654, -0.07044889, -0.07029544, -0.07013177, -0.06993740, -0.06982998,
    -0.06984021, -0.06982487, -0.06975837, -0.06959469, -0.06932360, -0.06897578, -0.06860238, -0.06826479,
    -0.06792720, -0.06764076, -0.06747708, -0.06734409, -0.06721110, -0.06708323, -0.06697070, -0.06686328,
    -0.06673541, -0.06659730, -0.06639781, -0.06618299, -0.06607046, -0.06602442, -0.06598350, -0.06595792,
    -0.06595281, -0.06598861, -0.06604488, -0.06611649, -0.06616764, -0.06622390, -0.06641316, -0.06663822,
    -0.06677121, -0.06675587, -0.06664845, -0.06648477, -0.06626994, -0.06606022, -0.06582493, -0.06550781,
    -0.06523671, -0.06506791, -0.06487866, -0.06466383, -0.06439785, -0.06410629, -0.06386078, -0.06365106,
    -0.06347203, -0.06321628, -0.06298100, -0.06288893, -0.06281220, -0.06275082, -0.06272524, -0.06274059,
    -0.06279686, -0.06288893, -0.06301680, -0.06303726, -0.06299122, -0.06301680, -0.06304749, -0.06306283,
    -0.06308841, -0.06311910, -0.06316514, -0.06322140, -0.06326232, -0.06323163, -0.06308329, -0.06287869,
    -0.06259226, -0.06221375, -0.06183012, -0.06150788, -0.06123678, -0.06099126, -0.06076109, -0.06051045,
    -0.06020355, -0.05996314, -0.05984038, -0.05973809, -0.05965624, -0.05959487, -0.05955906, -0.05951302,
    -0.05946699, -0.05942096, -0.05926239, -0.05912428, -0.05911917, -0.05913451, -0.05917032, -0.05924193,
    -0.05935957, -0.05953348, -0.05971762, -0.05986596, -0.05984550, -0.05972785, -0.05964090, -0.05954883,
    -0.05946188, -0.05939027, -0.05934934, -0.05930331, -0.05925728, -0.05921636, -0.05905267, -0.05877646,
    -0.05851048, -0.05825985, -0.05800922, -0.05781996, -0.05768697, -0.05755910, -0.05746191, -0.05741076,
    -0.05732892, -0.05717036, -0.05708852, -0.05712432, -0.05719593, -0.05728800, -0.05739030, -0.05751306,
    -0.05768186, -0.05789668, -0.05811152, -0.05819847, -0.05830077, -0.05848491, -0.05853606, -0.05838772,
    -0.05807571, -0.05772789, -0.05739541, -0.05705782, -0.05674581, -0.05635707, -0.05598368, -0.05575861,
    -0.05557448, -0.05536476, -0.05517039, -0.05505275, -0.05497091, -0.05488906, -0.05480211, -0.05465377,
    -0.05448498, -0.05443383, -0.05442360, -0.05437245, -0.05434176, -0.05435710, -0.05436734, -0.05437756,
    -0.05439802, -0.05435199, -0.05425992, -0.05419854, -0.05413716, -0.05408089, -0.05405021, -0.05401440,
    -0.05398371, -0.05395302, -0.05387630, -0.05372285, -0.05345175, -0.05315508, -0.05294025, -0.05279191,
    -0.05268450, -0.05258731, -0.05250036, -0.05242364, -0.05232645, -0.05221392, -0.05203490, -0.05186610,
    -0.05178426, -0.05171776, -0.05167173, -0.05163081, -0.05159500, -0.05157966, -0.05159501, -0.05162058,
    -0.05157454, -0.05154386, -0.05159501, -0.05164104, -0.05171777, -0.05178937, -0.05180472, -0.05181495,
    -0.05184053, -0.05184053, -0.05173311, -0.05152340, -0.05131880, -0.05116023, -0.05100167, -0.05086868,
    -0.05076126, -0.05060781, -0.05043902, -0.05023442, -0.04997867, -0.04969734, -0.04946716, -0.04931371,
    -0.04919095, -0.04912446, -0.04907331, -0.04901704, -0.04901193, -0.04905796, -0.04914492, -0.04921141,
    -0.04932395, -0.04950808, -0.04961038, -0.04960016, -0.04954900, -0.04949786, -0.04947228, -0.04951831,
    -0.04962062, -0.04967688, -0.04970757, -0.04977406, -0.04976383, -0.04958481, -0.04921141, -0.04876129,
    -0.04837767, -0.04810657, -0.04788152, -0.04761554, -0.04734956, -0.04711426, -0.04692501, -0.04678690,
    -0.04664880, -0.04650558, -0.04638282, -0.04627541, -0.04617311, -0.04604523, -0.04584575, -0.04570764,
    -0.04560534, -0.04546212, -0.04531890, -0.04519103, -0.04509384, -0.04499665, -0.04489947, -0.04479205,
    -0.04463861, -0.04450562, -0.04444935, -0.04440331, -0.04437262, -0.04436751, -0.04440843, -0.04446981,
    -0.04450562, -0.04452096, -0.04444935, -0.04438285, -0.04437774, -0.04435728, -0.04433682, -0.04433682,
    -0.04433682, -0.04431636, -0.04423964, -0.04410153, -0.04392251, -0.04373325, -0.04355422, -0.04337009,
    -0.04320640, -0.04303250, -0.04284324, -0.04267445, -0.04253634, -0.04243915, -0.04234197, -0.04222433,
    -0.04213737, -0.04210668, -0.04209645, -0.04210668, -0.04218852, -0.04235220, -0.04255168, -0.04277675,
    -0.04297112, -0.04297623, -0.04285859, -0.04271537, -0.04254146, -0.04233174, -0.04210668, -0.04188674,
    -0.04167191, -0.04149799, -0.04133432, -0.04108879, -0.04079213, -0.04054660, -0.04030620, -0.04004534,
    -0.03980493, -0.03960033, -0.03943665, -0.03930877, -0.03917579, -0.03901722, -0.03890469, -0.03886377,
    -0.03887400, -0.03892515, -0.03901211, -0.03911441, -0.03920136, -0.03930366, -0.03936504, -0.03932924,
    -0.03923205, -0.03913998, -0.03903768, -0.03891492, -0.03882796, -0.03875124, -0.03868475, -0.03861314,
    -0.03849037, -0.03827043, -0.03794819, -0.03765152, -0.03740599, -0.03719116, -0.03701726, -0.03686381,
    -0.03674105, -0.03666432, -0.03659782, -0.03646995, -0.03627047, -0.03609655, -0.03595845, -0.03583058,
    -0.03573850, -0.03566690, -0.03560040, -0.03556459, -0.03553391, -0.03548275, -0.03540603, -0.03537022,
    -0.03540603, -0.03548275, -0.03563620, -0.03585615, -0.03613748, -0.03631138, -0.03631138, -0.03617839,
    -0.03587149, -0.03550833, -0.03525770, -0.03507356, -0.03495079, -0.03489453, -0.03480758, -0.03469504,
    -0.03452114, -0.03424492, -0.03385619, -0.03342653, -0.03312474, -0.03294571, -0.03278715, -0.03261324,
    -0.03249048, -0.03240864, -0.03231657, -0.03222450, -0.03216824, -0.03205059, -0.03196363, -0.03201479,
    -0.03206082, -0.03210174, -0.03215289, -0.03219381, -0.03226030, -0.03236772, -0.03239841, -0.03210686,
    -0.03163628, -0.03116570, -0.03061839, -0.03009154, -0.02961585, -0.02915550, -0.02869003, -0.02822969,
    -0.02778980, -0.02722203, -0.02658265, -0.02615811, -0.02585632, -0.02552897, -0.02520161, -0.02492028,
    -0.02464918, -0.02443947, -0.02430137, -0.02402516, -0.02362619, -0.02333463, -0.02318118, -0.02306865,
    -0.02297146, -0.02291520, -0.02289474, -0.02286916, -0.02287428, -0.02286405, -0.02275152, -0.02274129,
    -0.02284871, -0.02288451, -0.02291520, -0.02297658, -0.02302261, -0.02308911, -0.02317095, -0.02322210,
    -0.02302261, -0.02274129, -0.02259807, -0.02238324, -0.02201496, -0.02151369, -0.02104822, -0.02066460,
    -0.02032701, -0.02003545, -0.01950349, -0.01890504, -0.01858791, -0.01836797, -0.01812756, -0.01789738,
    -0.01769790, -0.01752399, -0.01736031, -0.01716594, -0.01677209, -0.01631173, -0.01611225, -0.01602018,
    -0.01587696, -0.01572351, -0.01559052, -0.01548822, -0.01544730, -0.01547288, -0.01540638, -0.01509436,
    -0.01485396, -0.01475166, -0.01454706, -0.01432200, -0.01413786, -0.01395883, -0.01385142, -0.01373378,
    -0.01349337, -0.01289491, -0.01228623, -0.01195376, -0.01154967, -0.01112001, -0.01075685, -0.01047041,
    -0.01018908, -0.00987707, -0.00958040, -0.00892567, -0.00826072, -0.00803567, -0.00790268, -0.00768273,
    -0.00740140, -0.00719169, -0.00695129, -0.00668530, -0.00643978, -0.00584644, -0.00508431, -0.00462396,
    -0.00432729, -0.00407666, -0.00382602, -0.00350889, -0.00313550, -0.00276210, -0.00236824, -0.00180048,
    -0.00087978, -0.00011253, 0.00031201, 0.00068541, 0.00103323, 0.00135547, 0.00169306, 0.00207158,
    0.00239893, 0.00271606, 0.00345774, 0.00408688, 0.00416361, 0.00414315, 0.00417895, 0.00427102,
    0.00420965, 0.00404596, 0.00392832, 0.00387717, 0.00453701, 0.00535029, 0.00563161, 0.00602036,
    0.00640910, 0.00669553, 0.00696151, 0.00725819, 0.00754974, 0.00787198, 0.00880803, 0.01006632,
    0.01084892, 0.01133484, 0.01175427, 0.01212767, 0.01247037, 0.01273635, 0.01292049, 0.01319159,
    0.01387188, 0.01485907, 0.01538081, 0.01546776, 0.01563656, 0.01585139, 0.01604064, 0.01611225,
    0.01610202, 0.01620943, 0.01659818, 0.01748819, 0.01822986, 0.01857256, 0.01901245, 0.01952907,
    0.02008149, 0.02059299, 0.02108403, 0.02168760, 0.02241905, 0.02372849, 0.02496120, 0.02549316,
    0.02595863, 0.02643944, 0.02689467, 0.02732945, 0.02768238, 0.02811716, 0.02861842, 0.02968235,
    0.03091506, 0.03147259, 0.03184599, 0.03224496, 0.03265416, 0.03300710, 0.03334980, 0.03382550,
    0.03430119, 0.03516562, 0.03633184, 0.03689449, 0.03708886, 0.03722697, 0.03742134, 0.03775382,
    0.03808117, 0.03838808, 0.03876147, 0.03961056, 0.04103253, 0.04204019, 0.04240846, 0.04272559,
    0.04307342, 0.04357980, 0.04422429, 0.04492504, 0.04565138, 0.04655162, 0.04822933, 0.04968200,
    0.05025999, 0.05086356, 0.05143644, 0.05193771, 0.05245944, 0.05294537, 0.05336991, 0.05387118,
    0.05532384, 0.05673558, 0.05715501, 0.05750795, 0.05795295, 0.05831100, 0.05868951, 0.05907825,
    0.05944653, 0.05992222, 0.06127770, 0.06309864, 0.06416767, 0.06498096, 0.06576356, 0.06638247,
    0.06720087, 0.06814714, 0.06901158, 0.06981975, 0.07113430, 0.07293990, 0.07416239, 0.07487848,
    0.07567131, 0.07643856, 0.07704213, 0.07758432, 0.07819300, 0.07887842, 0.07987073, 0.08186046,
    0.08343077, 0.08398319, 0.08459187, 0.08515963, 0.08566602, 0.08620310, 0.08674017, 0.08719029,
    0.08767621, 0.08970687, 0.09180402, 0.09254058, 0.09327714, 0.09391140, 0.09462750, 0.09549705,
    0.09637683, 0.09723103, 0.09809036, 0.10016193, 0.10253529, 0.10379358, 0.10466313, 0.10551734,
    0.10648407, 0.10731781, 0.10802368, 0.10878582, 0.10965537, 0.11148654, 0.11422818, 0.11607981,
    0.11703120, 0.11804909, 0.11898513, 0.11965520, 0.12047871, 0.12146079, 0.12244798, 0.12427915,
    0.12740954, 0.12990565, 0.13119463, 0.13259615, 0.13392605, 0.13501554, 0.13609992, 0.13728149,
    0.13864719, 0.14011520, 0.14349110, 0.14681073, 0.14833500, 0.14968536, 0.15086693, 0.15248327,
    0.15416610, 0.15564945, 0.15718395, 0.15894863, 0.16251378, 0.16603290, 0.16781292, 0.16963386,
    0.17159802, 0.17351103, 0.17529105, 0.17687158, 0.17839585, 0.18006334, 0.18314769, 0.18734199,
    0.19036496, 0.19250814, 0.19464621, 0.19680474, 0.19871264, 0.20067679, 0.20295297, 0.20527518,
    0.20868689, 0.21391441, 0.21781204, 0.22023656, 0.22303446, 0.22585282, 0.22869165, 0.23099852,
    0.23406240, 0.23751502, 0.24072724, 0.24620541, 0.25146874, 0.25457866, 0.25826147, 0.26253249,
    0.26649661, 0.27040959, 0.27471642, 0.27731995, 0.27935061, 0.28613310, 0.29363680, 0.29970831,
    0.30660333, 0.31282828, 0.31838318, 0.32225523, 0.32646487, 0.33527802, 0.35497077, 0.36881196,
    0.37097049, 0.37648446, 0.38023887, 0.38296005, 0.38804436, 0.39603910, 0.40749670, 0.41829959,
    0.42736337, 0.55509515, 0.57216902, 0.46405838, 0.46764911, 0.48035988, 0.49084051, 0.49907055,
    0.51331071, 0.53345869, 0.55821529, 0.75567834, 1.10878725, 1.02799582, 0.74680483, 0.71247756,
    0.74152513, 1.13224310, 1.21173583, 0.91859211, 0.88388387, 0.87424813
};

const double GainRobustTracker::hinv_2_der_[1022] = {
    -1.29799212, -1.10868955, -0.97926266, -0.88234057, -0.80628870, -0.74438186, -0.69319861, -0.65025307,
    -0.61349157, -0.58106963, -0.55209724, -0.52694832, -0.50482697, -0.48514752, -0.46739847, -0.45133226,
    -0.43678008, -0.42338901, -0.41068335, -0.39835620, -0.38654055, -0.37577859, -0.36598336, -0.35677637,
    -0.34797345, -0.33973319, -0.33210160, -0.32474623, -0.31763639, -0.31069533, -0.30354456, -0.29695644,
    -0.29121741, -0.28564717, -0.28037873, -0.27537114, -0.27049655, -0.26584701, -0.26142253, -0.25702875,
    -0.25204674, -0.24719260, -0.24313641, -0.23927970, -0.23551506, -0.23180668, -0.22826199, -0.22509581,
    -0.22212911, -0.21911637, -0.21546938, -0.21168939, -0.20862039, -0.20576110, -0.20301435, -0.20027271,
    -0.19752596, -0.19491731, -0.19233423, -0.18979207, -0.18700440, -0.18404793, -0.18151601, -0.17928075,
    -0.17705061, -0.17486651, -0.17275401, -0.17068244, -0.16866201, -0.16669785, -0.16452397, -0.16201251,
    -0.15980283, -0.15795631, -0.15601773, -0.15417121, -0.15240143, -0.15065210, -0.14895392, -0.14729665,
    -0.14552175, -0.14331207, -0.14128653, -0.13965485, -0.13805896, -0.13644774, -0.13492858, -0.13342989,
    -0.13184936, -0.13037112, -0.12893892, -0.12713844, -0.12527147, -0.12381369, -0.12246333, -0.12115389,
    -0.11979330, -0.11847363, -0.11722557, -0.11600309, -0.11480618, -0.11340978, -0.11192131, -0.11058119,
    -0.10936381, -0.10822317, -0.10713879, -0.10597768, -0.10478077, -0.10356852, -0.10244833, -0.10134349,
    -0.09999314, -0.09883203, -0.09786529, -0.09682183, -0.09579883, -0.09471957, -0.09369657, -0.09275541,
    -0.09172218, -0.09057131, -0.08926187, -0.08801892, -0.08697034, -0.08594734, -0.08488854, -0.08383485,
    -0.08286300, -0.08192696, -0.08093465, -0.07992187, -0.07875565, -0.07769174, -0.07687334, -0.07597310,
    -0.07504217, -0.07418284, -0.07330306, -0.07236191, -0.07148724, -0.07068930, -0.06970211, -0.06866376,
    -0.06786071, -0.06707811, -0.06619833, -0.06536458, -0.06458199, -0.06376871, -0.06290427, -0.06206541,
    -0.06120609, -0.06021378, -0.05930843, -0.05852072, -0.05770231, -0.05688903, -0.05611155, -0.05533407,
    -0.05456682, -0.05380468, -0.05295048, -0.05196328, -0.05105281, -0.05034183, -0.04961550, -0.04876641,
    -0.04793778, -0.04717053, -0.04639817, -0.04558488, -0.04481251, -0.04401969, -0.04323709, -0.04248519,
    -0.04171794, -0.04103253, -0.04037270, -0.03965659, -0.03889446, -0.03818347, -0.03752364, -0.03678197,
    -0.03603006, -0.03536000, -0.03471039, -0.03398918, -0.03322704, -0.03255697, -0.03193295, -0.03128845,
    -0.03063885, -0.02994321, -0.02922200, -0.02849566, -0.02780003, -0.02718111, -0.02657754, -0.02595863,
    -0.02532437, -0.02470545, -0.02410699, -0.02343182, -0.02268502, -0.02204565, -0.02143185, -0.02078224,
    -0.02011730, -0.01941143, -0.01873625, -0.01809175, -0.01743192, -0.01678743, -0.01619409, -0.01561610,
    -0.01497161, -0.01428619, -0.01363659, -0.01304836, -0.01248060, -0.01190260, -0.01131438, -0.01069035,
    -0.01005609, -0.00943206, -0.00881315, -0.00823003, -0.00766738, -0.00710474, -0.00650116, -0.00589248,
    -0.00531448, -0.00472114, -0.00415850, -0.00358050, -0.00301785, -0.00248589, -0.00193858, -0.00135036,
    -0.00078771, -0.00021994, 0.00041943, 0.00103323, 0.00156008, 0.00203577, 0.00251146, 0.00300250,
    0.00360096, 0.00419430, 0.00474160, 0.00529402, 0.00577995, 0.00628633, 0.00679272, 0.00723773,
    0.00771853, 0.00825561, 0.00880803, 0.00932976, 0.00989753, 0.01049598, 0.01104840, 0.01159571,
    0.01215835, 0.01262382, 0.01301256, 0.01350871, 0.01403045, 0.01457775, 0.01516086, 0.01571839,
    0.01625036, 0.01679255, 0.01732962, 0.01778485, 0.01822475, 0.01871067, 0.01925798, 0.01982062,
    0.02039862, 0.02096639, 0.02151369, 0.02207122, 0.02259296, 0.02305842, 0.02355457, 0.02410188,
    0.02463384, 0.02516068, 0.02568242, 0.02615811, 0.02662869, 0.02707881, 0.02742663, 0.02774376,
    0.02807112, 0.02849567, 0.02899182, 0.02946240, 0.02991252, 0.03036264, 0.03081276, 0.03123730,
    0.03171812, 0.03217846, 0.03254675, 0.03299687, 0.03350325, 0.03393802, 0.03432676, 0.03474108,
    0.03511448, 0.03547764, 0.03590218, 0.03627047, 0.03652110, 0.03676151, 0.03709398, 0.03752364,
    0.03796865, 0.03834716, 0.03873078, 0.03911952, 0.03948780, 0.03988166, 0.04020390, 0.04045965,
    0.04083816, 0.04127805, 0.04171282, 0.04212714, 0.04252099, 0.04297623, 0.04341101, 0.04380997,
    0.04416802, 0.04448516, 0.04480229, 0.04515010, 0.04557976, 0.04605035, 0.04651070, 0.04690966,
    0.04731886, 0.04776387, 0.04817819, 0.04842370, 0.04873572, 0.04927791, 0.04976895, 0.05018838,
    0.05060781, 0.05106305, 0.05149782, 0.05188144, 0.05231111, 0.05267938, 0.05300674, 0.05343641,
    0.05391722, 0.05438268, 0.05480211, 0.05525223, 0.05569212, 0.05612690, 0.05664351, 0.05704759,
    0.05741076, 0.05791714, 0.05841842, 0.05884807, 0.05927774, 0.05978924, 0.06030073, 0.06074574,
    0.06120097, 0.06161529, 0.06194777, 0.06233651, 0.06274571, 0.06319583, 0.06379428, 0.06428532,
    0.06466895, 0.06515999, 0.06559987, 0.06599884, 0.06633644, 0.06658195, 0.06695535, 0.06749243,
    0.06798346, 0.06836709, 0.06881721, 0.06928779, 0.06967142, 0.07007039, 0.07036194, 0.07057677,
    0.07097062, 0.07145655, 0.07186064, 0.07221357, 0.07265346, 0.07306266, 0.07339002, 0.07374807,
    0.07404474, 0.07436187, 0.07472504, 0.07507286, 0.07557924, 0.07612143, 0.07652551, 0.07690914,
    0.07737461, 0.07783496, 0.07820835, 0.07849479, 0.07886819, 0.07925693, 0.07959963, 0.08003440,
    0.08046918, 0.08084258, 0.08126712, 0.08172235, 0.08211621, 0.08246403, 0.08279139, 0.08317502,
    0.08363537, 0.08408549, 0.08437704, 0.08462256, 0.08502153, 0.08554838, 0.08602407, 0.08626959,
    0.08653557, 0.08697546, 0.08738978, 0.08774271, 0.08813145, 0.08856622, 0.08894985, 0.08934882,
    0.08981428, 0.09010072, 0.09033090, 0.09070430, 0.09115442, 0.09167103, 0.09204954, 0.09237690,
    0.09285771, 0.09332318, 0.09372726, 0.09410066, 0.09447917, 0.09491905, 0.09541009, 0.09594205,
    0.09638194, 0.09668884, 0.09712874, 0.09765047, 0.09808013, 0.09844329, 0.09879111, 0.09926681,
    0.09978342, 0.10023866, 0.10066832, 0.10110309, 0.10153275, 0.10204936, 0.10259156, 0.10304679,
    0.10333835, 0.10360944, 0.10402887, 0.10447387, 0.10490865, 0.10533320, 0.10580377, 0.10627947,
    0.10680632, 0.10734339, 0.10776282, 0.10816690, 0.10861191, 0.10909272, 0.10951215, 0.10993670,
    0.11041239, 0.11090854, 0.11140982, 0.11191108, 0.11236120, 0.11276018, 0.11320007, 0.11363996,
    0.11416680, 0.11471922, 0.11513865, 0.11558365, 0.11607470, 0.11645832, 0.11681637, 0.11730741,
    0.11777287, 0.11811558, 0.11861174, 0.11913858, 0.11951709, 0.11989560, 0.12033549, 0.12081630,
    0.12128688, 0.12171654, 0.12214620, 0.12258609, 0.12299529, 0.12358351, 0.12427404, 0.12489807,
    0.12554256, 0.12622286, 0.12699010, 0.12762436, 0.12820236, 0.12893892, 0.12965502, 0.13024324,
    0.13072917, 0.13110768, 0.13152199, 0.13202326, 0.13248362, 0.13286212, 0.13323552, 0.13371122,
    0.13425341, 0.13485186, 0.13536847, 0.13566514, 0.13603854, 0.13653469, 0.13700016, 0.13740425,
    0.13777764, 0.13814592, 0.13851420, 0.13884668, 0.13908708, 0.13935306, 0.13970599, 0.14000778,
    0.14019192, 0.14033514, 0.14049882, 0.14080572, 0.14112797, 0.14143998, 0.14187987, 0.14229418,
    0.14265224, 0.14302563, 0.14342460, 0.14385426, 0.14427881, 0.14471358, 0.14513812, 0.14547571,
    0.14579285, 0.14624297, 0.14676981, 0.14719435, 0.14745011, 0.14774166, 0.14805879, 0.14839126,
    0.14882604, 0.14925059, 0.14959329, 0.14994111, 0.15038612, 0.15087204, 0.15128482, 0.15171653,
    0.15225156, 0.15263723, 0.15292878, 0.15326484, 0.15370012, 0.15420549, 0.15467658, 0.15511544,
    0.15551288, 0.15584433, 0.15619369, 0.15660851, 0.15700544, 0.15747755, 0.15798803, 0.15836091,
    0.15882740, 0.15941153, 0.15987751, 0.16029745, 0.16068773, 0.16094706, 0.16136597, 0.16194908,
    0.16244319, 0.16289536, 0.16336850, 0.16386926, 0.16440786, 0.16487538, 0.16533112, 0.16584671,
    0.16642778, 0.16709733, 0.16763952, 0.16818120, 0.16879602, 0.16945842, 0.17018730, 0.17082617,
    0.17127424, 0.17162973, 0.17201643, 0.17255913, 0.17328239, 0.17387880, 0.17428851, 0.17466037,
    0.17518364, 0.17583631, 0.17640612, 0.17686391, 0.17723424, 0.17771198, 0.17829616, 0.17885948,
    0.17937814, 0.17997782, 0.18050636, 0.18080679, 0.18114834, 0.18164671, 0.18191748, 0.18216116,
    0.18253880, 0.18279757, 0.18317669, 0.18356968, 0.18401366, 0.18447350, 0.18482285, 0.18525712,
    0.18557373, 0.18585506, 0.18624840, 0.18659367, 0.18691847, 0.18730874, 0.18784991, 0.18842637,
    0.18884222, 0.18924528, 0.18950257, 0.18960742, 0.18991790, 0.19026726, 0.19068464, 0.19125803,
    0.19180534, 0.19232656, 0.19269944, 0.19293678, 0.19307028, 0.19322475, 0.19365032, 0.19408509,
    0.19445951, 0.19480989, 0.19518277, 0.19571371, 0.19609273, 0.19631830, 0.19658991, 0.19670551,
    0.19688709, 0.19732084, 0.19785178, 0.19839704, 0.19885330, 0.19920112, 0.19945533, 0.19973359,
    0.20001798, 0.20013665, 0.20051005, 0.20115761, 0.20165887, 0.20209876, 0.20264607, 0.20315246,
    0.20351050, 0.20386344, 0.20423172, 0.20441074, 0.20462046, 0.20512685, 0.20559742, 0.20601686,
    0.20645163, 0.20708589, 0.20779688, 0.20846182, 0.20905516, 0.20921373, 0.20921373, 0.20955644,
    0.21002190, 0.21061524, 0.21126996, 0.21158709, 0.21184796, 0.21223158, 0.21254871, 0.21273285,
    0.21291187, 0.21298349, 0.21308578, 0.21357171, 0.21409856, 0.21442080, 0.21473793, 0.21501414,
    0.21520340, 0.21542846, 0.21555122, 0.21576604, 0.21609852, 0.21618036, 0.21638496, 0.21679416,
    0.21711129, 0.21744888, 0.21788877, 0.21821102, 0.21787342, 0.21758699, 0.21770463, 0.21777112,
    0.21811894, 0.21873274, 0.21900384, 0.21890154, 0.21924424, 0.21977109, 0.21965345, 0.21896804,
    0.21879413, 0.21931585, 0.21976598, 0.22003707, 0.22004218, 0.22036955, 0.22083501, 0.22093731,
    0.22078898, 0.22011891, 0.21983758, 0.22036955, 0.22089128, 0.22107542, 0.22109587, 0.22125956,
    0.22157668, 0.22197054, 0.22189381, 0.22116237, 0.22112657, 0.22180686, 0.22196031, 0.22182732,
    0.22171479, 0.22165852, 0.22190405, 0.22230813, 0.22247181, 0.22198077, 0.22156134, 0.22171991,
    0.22205238, 0.22234905, 0.22243601, 0.22221606, 0.22223140, 0.22254853, 0.22277871, 0.22245647,
    0.22191939, 0.22189381, 0.22212911, 0.22227744, 0.22215468, 0.22220583, 0.22237974, 0.22268664,
    0.22300377, 0.22250761, 0.22175060, 0.22164318, 0.22186312, 0.22211887, 0.22241043, 0.22246158,
    0.22251784, 0.22263038, 0.22254854, 0.22229278, 0.22147950, 0.22114191, 0.22128513, 0.22099358,
    0.22074805, 0.22063041, 0.22039000, 0.22017517, 0.22007288, 0.21957672, 0.21859976, 0.21800642,
    0.21793480, 0.21771486, 0.21754607, 0.21746934, 0.21726474, 0.21688111, 0.21668675, 0.21661514,
    0.21559214, 0.21468166, 0.21441568, 0.21392976, 0.21340803, 0.21311136, 0.21301929, 0.21276865,
    0.21190422, 0.21112674, 0.21029299, 0.20929557, 0.20873292, 0.20823677, 0.20795032, 0.20752067,
    0.20716773, 0.20695290, 0.20641071, 0.20578156, 0.20481995, 0.20310642, 0.20178675, 0.20126502,
    0.20058984, 0.19978679, 0.19878936, 0.19781751, 0.19721394, 0.19671267, 0.19609376, 0.19432396,
    0.19225239, 0.19128054, 0.19075370, 0.19006829, 0.18914759, 0.18861051, 0.18814504, 0.18710670,
    0.18610927, 0.18393028, 0.18142905, 0.18010426, 0.17886643, 0.17772067, 0.17630894, 0.17485116,
    0.17347522, 0.17163894, 0.17010956, 0.16795614, 0.16599709, 0.16477461, 0.16331172, 0.16239614,
    0.16146009, 0.16005346, 0.15863149, 0.15682079, 0.15492824, 0.15301522, 0.14929662, 0.14625831,
    0.14528135, 0.14375707, 0.14194125, 0.14033514, 0.13789017, 0.13561400, 0.13421249, 0.13148619,
    0.12705660, 0.12269351, 0.11925623, 0.11599797, 0.11308242, 0.11082670, 0.10810552, 0.10433577,
    0.10048418, 0.09734868, 0.09218253, 0.08676063, 0.08258168, 0.07900629, 0.07589637, 0.07188621,
    0.06833129, 0.06473544, 0.06013705, 0.05495556, 0.04823957, 0.04264887, 0.03889446, 0.03436768,
    0.02841894, 0.02188197, 0.01678743, 0.01205606, 0.00648582, 0.00102300, -0.00535029, -0.01313532,
    -0.01996896, -0.02721180, -0.03535488, -0.04215783, -0.04807588, -0.05467424, -0.06130328, -0.06807042,
    -0.07550251, -0.08498572, -0.09417738, -0.10202379, -0.11042262, -0.11927669, -0.12836092, -0.13571630,
    -0.14473916, -0.15523002, -0.16401247, -0.17386908, -0.18533180, -0.19719348, -0.21043110, -0.22455361,
    -0.23930016, -0.25332038, -0.26759122, -0.27676754, -0.28331473, -0.29651655, -0.31091016, -0.33083309,
    -0.35545158, -0.37804453, -0.39920018, -0.41314878, -0.42884160, -0.46223743, -0.53760184, -0.58260361,
    -0.57778017, -0.59487450, -0.60997909, -0.62188681, -0.64592220, -0.68268882, -0.73058057, -0.77057475,
    -0.79942846, -1.28492892, -1.33360326, -0.89996379, -0.91034724, -0.96125683, -1.00646832, -1.04546508,
    -1.10979644, -1.19931917, -1.31263687, -2.12792184, -3.46950450, -3.11213991, -2.05663920, -1.92451364,
    -2.00482425, -3.36834003, -3.65047473, -2.62964349, -2.52741715, -2.56363647
};

const double GainRobustTracker::hinv_3_der_[1022] = {
    -1.83688857, -1.54795705, -1.35438244, -1.21143302, -1.09996387, -1.01187999, -0.93946131, -0.88039329,
    -0.83155578, -0.78239552, -0.73640144, -0.70174220, -0.67292429, -0.64668433, -0.62250573, -0.60007134,
    -0.57957042, -0.56065004, -0.54248667, -0.52169931, -0.49816008, -0.47986884, -0.46652892, -0.45372607,
    -0.44108179, -0.42893367, -0.41743515, -0.40638675, -0.39582939, -0.38400351, -0.36856644, -0.35568176,
    -0.34745172, -0.33954905, -0.33229086, -0.32528331, -0.31840363, -0.31175414, -0.30550872, -0.29908939,
    -0.28773409, -0.27717162, -0.27207197, -0.26722295, -0.26292634, -0.25935096, -0.25533569, -0.24992401,
    -0.24450723, -0.24013391, -0.23207778, -0.22306004, -0.21833377, -0.21483000, -0.21133134, -0.20774573,
    -0.20401177, -0.20048243, -0.19696842, -0.19347488, -0.18757216, -0.18012984, -0.17545985, -0.17238062,
    -0.16941392, -0.16632957, -0.16298436, -0.15977725, -0.15674406, -0.15358811, -0.14855495, -0.14085687,
    -0.13488766, -0.13117929, -0.12718959, -0.12321012, -0.11958870, -0.11687775, -0.11505681, -0.11365530,
    -0.11155303, -0.10667333, -0.10233069, -0.10077061, -0.09913893, -0.09753282, -0.09607504, -0.09459169,
    -0.09309300, -0.09152781, -0.08990124, -0.08518009, -0.08015205, -0.07781961, -0.07572246, -0.07388106,
    -0.07185040, -0.06972768, -0.06768168, -0.06583517, -0.06413187, -0.06036212, -0.05548752, -0.05259243,
    -0.05061804, -0.04876129, -0.04687386, -0.04457211, -0.04227548, -0.03983562, -0.03763106, -0.03475131,
    -0.03058770, -0.02777445, -0.02613253, -0.02432694, -0.02263388, -0.02095616, -0.01950349, -0.01821963,
    -0.01694088, -0.01572863, -0.01349849, -0.01176450, -0.01170823, -0.01151386, -0.01100748, -0.01049087,
    -0.00976453, -0.00860854, -0.00729399, -0.00596921, -0.00262911, 0.00052173, 0.00188232, 0.00352935,
    0.00499224, 0.00646536, 0.00839883, 0.01034253, 0.01197422, 0.01348314, 0.01620432, 0.01913522,
    0.02059811, 0.02179501, 0.02293054, 0.02367222, 0.02428091, 0.02503792, 0.02587679, 0.02661846,
    0.02822968, 0.03044448, 0.03141122, 0.03174369, 0.03242910, 0.03314009, 0.03373342, 0.03448533,
    0.03531908, 0.03629604, 0.03794307, 0.04057218, 0.04281255, 0.04424475, 0.04600943, 0.04808611,
    0.05010142, 0.05188656, 0.05335456, 0.05449521, 0.05562051, 0.05785576, 0.06048999, 0.06240811,
    0.06414210, 0.06571752, 0.06701673, 0.06791185, 0.06834663, 0.06840290, 0.06841824, 0.06931848,
    0.07048470, 0.07120080, 0.07192713, 0.07298082, 0.07436187, 0.07545136, 0.07636695, 0.07749225,
    0.07875565, 0.08037711, 0.08227478, 0.08400364, 0.08553814, 0.08689873, 0.08826444, 0.08958411,
    0.09081682, 0.09203931, 0.09322088, 0.09468377, 0.09668884, 0.09868370, 0.10058647, 0.10240230,
    0.10401864, 0.10531785, 0.10613113, 0.10666309, 0.10705695, 0.10768610, 0.10845335, 0.10931778,
    0.11035612, 0.11135866, 0.11220775, 0.11245328, 0.11198781, 0.11112849, 0.11035101, 0.10998273,
    0.10993670, 0.10998785, 0.11009015, 0.11033055, 0.11069883, 0.11103642, 0.11145585, 0.11182413,
    0.11215661, 0.11259649, 0.11289316, 0.11311311, 0.11339955, 0.11393151, 0.11469364, 0.11565526,
    0.11651458, 0.11745575, 0.11864754, 0.11994675, 0.12129200, 0.12261678, 0.12424846, 0.12592619,
    0.12739931, 0.12840696, 0.12894403, 0.12916910, 0.12908214, 0.12899518, 0.12888266, 0.12825351,
    0.12768575, 0.12771132, 0.12776758, 0.12772667, 0.12757833, 0.12725097, 0.12704125, 0.12697987,
    0.12697988, 0.12669855, 0.12640699, 0.12672413, 0.12714356, 0.12769598, 0.12855530, 0.12951180,
    0.13057060, 0.13179309, 0.13309742, 0.13401300, 0.13461146, 0.13550658, 0.13655004, 0.13749632,
    0.13846817, 0.13945536, 0.14044256, 0.14140418, 0.14230441, 0.14295914, 0.14316885, 0.14302051,
    0.14251413, 0.14170085, 0.14074434, 0.13995152, 0.13936841, 0.13876995, 0.13801805, 0.13726614,
    0.13652958, 0.13611015, 0.13617153, 0.13636590, 0.13659608, 0.13694390, 0.13744516, 0.13775718,
    0.13809989, 0.13856023, 0.13862161, 0.13884156, 0.13957301, 0.14028399, 0.14093871, 0.14182360,
    0.14286706, 0.14406909, 0.14547060, 0.14668797, 0.14726596, 0.14732735, 0.14728643, 0.14733758,
    0.14758309, 0.14786442, 0.14828385, 0.14870839, 0.14901530, 0.14932220, 0.14923524, 0.14871351,
    0.14816620, 0.14767005, 0.14718412, 0.14694883, 0.14690280, 0.14699998, 0.14727619, 0.14758821,
    0.14801276, 0.14826339, 0.14839638, 0.14891811, 0.14987973, 0.15102038, 0.15206895, 0.15317379,
    0.15451392, 0.15606888, 0.15770568, 0.15905092, 0.16045755, 0.16197671, 0.16286160, 0.16273884,
    0.16188975, 0.16092813, 0.15997163, 0.15902535, 0.15827344, 0.15769545, 0.15714815, 0.15670314,
    0.15645762, 0.15608934, 0.15582848, 0.15604331, 0.15628882, 0.15641159, 0.15660595, 0.15698446,
    0.15750619, 0.15825810, 0.15890771, 0.15913277, 0.15953174, 0.16023505, 0.16082020, 0.16143758,
    0.16225701, 0.16290968, 0.16351325, 0.16416541, 0.16445646, 0.16477410, 0.16549941, 0.16600272,
    0.16626512, 0.16669222, 0.16682419, 0.16669990, 0.16653570, 0.16600681, 0.16558687, 0.16577101,
    0.16593316, 0.16575669, 0.16567280, 0.16573316, 0.16552907, 0.16521655, 0.16539506, 0.16548355,
    0.16549327, 0.16582728, 0.16603546, 0.16614492, 0.16638481, 0.16658737, 0.16666205, 0.16688352,
    0.16741600, 0.16818580, 0.16885485, 0.16939345, 0.17026505, 0.17096171, 0.17128191, 0.17179034,
    0.17234379, 0.17280721, 0.17317702, 0.17324352, 0.17282051, 0.17229008, 0.17179648, 0.17152999,
    0.17141030, 0.17111619, 0.17080417, 0.17023590, 0.16970276, 0.16954445, 0.16934624, 0.16906661,
    0.16899612, 0.16898275, 0.16895899, 0.16901538, 0.16932415, 0.17004598, 0.17103220, 0.17205939,
    0.17330822, 0.17468451, 0.17552429, 0.17578056, 0.17610996, 0.17654013, 0.17681123, 0.17744037,
    0.17844036, 0.17947665, 0.18034058, 0.18097484, 0.18135181, 0.18111243, 0.17995695, 0.17843882,
    0.17719690, 0.17637594, 0.17586444, 0.17544859, 0.17513504, 0.17453608, 0.17403634, 0.17388749,
    0.17354990, 0.17312024, 0.17294071, 0.17294787, 0.17287114, 0.17266961, 0.17221438, 0.17198523,
    0.17192487, 0.17157040, 0.17108703, 0.17064663, 0.17037860, 0.17017042, 0.17002413, 0.16983386,
    0.16947734, 0.16909320, 0.16893464, 0.16874897, 0.16854283, 0.16869781, 0.16926609, 0.16973002,
    0.17008858, 0.17051773, 0.17036889, 0.17005585, 0.17002771, 0.17010342, 0.17015201, 0.17009933,
    0.17007375, 0.17015559, 0.17007886, 0.16976685, 0.16909167, 0.16824258, 0.16753159, 0.16677457,
    0.16624262, 0.16573111, 0.16515823, 0.16470300, 0.16426311, 0.16377718, 0.16312758, 0.16262119,
    0.16231941, 0.16220177, 0.16234499, 0.16264677, 0.16305597, 0.16374138, 0.16482576, 0.16602778,
    0.16699452, 0.16691779, 0.16639095, 0.16591525, 0.16516335, 0.16454444, 0.16405339, 0.16364931,
    0.16347029, 0.16326568, 0.16318384, 0.16224269, 0.16070818, 0.16018134, 0.15971076, 0.15877983,
    0.15802281, 0.15758804, 0.15729648, 0.15710211, 0.15677475, 0.15571083, 0.15453950, 0.15440139,
    0.15480036, 0.15523513, 0.15583359, 0.15657015, 0.15698958, 0.15739366, 0.15760338, 0.15692309,
    0.15571083, 0.15486174, 0.15442696, 0.15377736, 0.15310730, 0.15268275, 0.15245258, 0.15191550,
    0.15080554, 0.14921990, 0.14701533, 0.14510743, 0.14360362, 0.14221234, 0.14142464, 0.14120980,
    0.14112285, 0.14052951, 0.13982364, 0.13909731, 0.13730706, 0.13566514, 0.13496951, 0.13437617,
    0.13395162, 0.13357822, 0.13296954, 0.13237620, 0.13213068, 0.13211534, 0.13077009, 0.12946065,
    0.12950157, 0.12973686, 0.13063710, 0.13199257, 0.13329690, 0.13397208, 0.13375725, 0.13286724,
    0.13109233, 0.12890823, 0.12782896, 0.12729189, 0.12663206, 0.12619217, 0.12564998, 0.12489807,
    0.12405409, 0.12294414, 0.12099021, 0.11840202, 0.11635602, 0.11540975, 0.11475502, 0.11402869,
    0.11369622, 0.11356835, 0.11342512, 0.11295966, 0.11219241, 0.11083182, 0.11025894, 0.11039704,
    0.11018733, 0.11049935, 0.11068860, 0.11057095, 0.11074486, 0.11071417, 0.11058630, 0.10932801,
    0.10717971, 0.10558895, 0.10358898, 0.10174247, 0.10027957, 0.09856093, 0.09661212, 0.09458146,
    0.09246897, 0.08951761, 0.08640258, 0.08447423, 0.08303691, 0.08166097, 0.08002929, 0.07851014,
    0.07746156, 0.07647436, 0.07541045, 0.07371738, 0.07131844, 0.06970722, 0.06872514, 0.06776863,
    0.06706277, 0.06650523, 0.06617787, 0.06562033, 0.06500142, 0.06414721, 0.06253088, 0.06139535,
    0.06122143, 0.06079178, 0.06035700, 0.06008079, 0.05949256, 0.05886853, 0.05817801, 0.05737496,
    0.05570235, 0.05401951, 0.05311416, 0.05144667, 0.04895566, 0.04629075, 0.04384578, 0.04191742,
    0.03987654, 0.03786123, 0.03539068, 0.03308382, 0.03173346, 0.03000459, 0.02827572, 0.02732433,
    0.02631156, 0.02458781, 0.02306865, 0.02193823, 0.02027074, 0.01816337, 0.01678232, 0.01596391,
    0.01476189, 0.01330923, 0.01204071, 0.01076196, 0.01028626, 0.01056247, 0.01010724, 0.00825561,
    0.00644490, 0.00534517, 0.00443471, 0.00324803, 0.00187720, 0.00114576, 0.00037339, -0.00044501,
    -0.00184651, -0.00494620, -0.00774411, -0.00925815, -0.01077219, -0.01270055, -0.01453172, -0.01656237,
    -0.01847026, -0.01998942, -0.02137558, -0.02377963, -0.02659288, -0.02793813, -0.02893044, -0.02987160,
    -0.03049052, -0.03138564, -0.03284853, -0.03391756, -0.03476666, -0.03726277, -0.04101207, -0.04362072,
    -0.04491993, -0.04632655, -0.04810657, -0.04982521, -0.05154897, -0.05392745, -0.05709363, -0.06086850,
    -0.06548223, -0.06850519, -0.07039774, -0.07250001, -0.07488871, -0.07784518, -0.08012136, -0.08169167,
    -0.08311364, -0.08499084, -0.08891916, -0.09219276, -0.09339478, -0.09516458, -0.09657120, -0.09710316,
    -0.09734868, -0.09757885, -0.09781415, -0.09878599, -0.10411583, -0.10968094, -0.11163999, -0.11386502,
    -0.11644809, -0.11864242, -0.12059635, -0.12333288, -0.12585458, -0.12762437, -0.13303604, -0.13927122,
    -0.14256016, -0.14567009, -0.14844753, -0.15062140, -0.15254465, -0.15432978, -0.15541928, -0.15620698,
    -0.16011484, -0.16698940, -0.17128601, -0.17264660, -0.17431920, -0.17594065, -0.17670791, -0.17721429,
    -0.17868230, -0.18068737, -0.18393028, -0.19064628, -0.19588915, -0.19776636, -0.19975098, -0.20235963,
    -0.20560766, -0.20900913, -0.21198095, -0.21497834, -0.21817521, -0.22724922, -0.23617490, -0.23883981,
    -0.24151495, -0.24413895, -0.24669645, -0.24925395, -0.25188817, -0.25477815, -0.25762209, -0.26669099,
    -0.27701817, -0.28074700, -0.28284927, -0.28534028, -0.28808192, -0.29078263, -0.29287467, -0.29572884,
    -0.29910474, -0.30636293, -0.31668499, -0.32209666, -0.32389203, -0.32475646, -0.32601475, -0.32815794,
    -0.33084332, -0.33271029, -0.33487905, -0.34181499, -0.35426490, -0.36273022, -0.36511381, -0.36720074,
    -0.36880685, -0.37118532, -0.37420317, -0.37719033, -0.37989105, -0.38475541, -0.40123594, -0.41552009,
    -0.41869702, -0.42110823, -0.42312763, -0.42606671, -0.42921039, -0.43252389, -0.43553509, -0.43780615,
    -0.45143149, -0.46686498, -0.47074112, -0.47281781, -0.47462085, -0.47611034, -0.47661570, -0.47786018,
    -0.48077010, -0.48344269, -0.49542253, -0.51243461, -0.52014046, -0.52290277, -0.52602732, -0.53092779,
    -0.53553753, -0.53920847, -0.54318538, -0.54545644, -0.55503735, -0.57472754, -0.58776670, -0.59120909,
    -0.59345918, -0.59603100, -0.59877929, -0.60178589, -0.60465643, -0.60650192, -0.61200463, -0.63120072,
    -0.64617386, -0.64704750, -0.64664853, -0.64799889, -0.64995793, -0.64993747, -0.65098605, -0.65461770,
    -0.65771227, -0.67352786, -0.69046362, -0.69217203, -0.69301601, -0.69432033, -0.69438171, -0.69623334,
    -0.69949159, -0.70138926, -0.70388026, -0.71817158, -0.73541936, -0.74023769, -0.74193586, -0.74373123,
    -0.74429899, -0.74569539, -0.74803295, -0.74941911, -0.75154184, -0.76206851, -0.77870760, -0.78794529,
    -0.78973043, -0.79069716, -0.79110636, -0.79080458, -0.79099383, -0.79120866, -0.79162809, -0.79496818,
    -0.80774034, -0.81747418, -0.81820052, -0.82055342, -0.82099331, -0.82196515, -0.82334621, -0.82220045,
    -0.82008795, -0.82004703, -0.83179618, -0.84202107, -0.84003133, -0.83977047, -0.84275252, -0.84104410,
    -0.83802114, -0.83649687, -0.83260947, -0.82823614, -0.83027191, -0.83516185, -0.83503398, -0.83205705,
    -0.82895736, -0.82341270, -0.81787827, -0.81516732, -0.80958686, -0.80227241, -0.79806787, -0.79622136,
    -0.79263063, -0.78932123, -0.78786345, -0.78299397, -0.77816541, -0.77306575, -0.76415031, -0.75742920,
    -0.74788461, -0.73110741, -0.72243748, -0.71633017, -0.70677535, -0.70188541, -0.69755812, -0.69445843,
    -0.68157375, -0.66321090, -0.65150267, -0.62620387, -0.59472105, -0.57157056, -0.54717201, -0.51548970,
    -0.48298388, -0.45854952, -0.43466759, -0.42670353, -0.42636594, -0.38545105, -0.33764115, -0.29738610,
    -0.25466050, -0.22825687, -0.20739279, -0.18700440, -0.14768540, -0.04707335, 0.19869218, 0.33253638,
    0.27414354, 0.29068033, 0.32681781, 0.35016267, 0.39827948, 0.48034453, 0.59888466, 0.70434573,
    0.78624711, 2.62739136, 2.79644723, 1.08352068, 1.04174136, 1.12931016, 1.18509947, 1.22990686,
    1.35800181, 1.55881671, 1.84015194, 4.24338866, 8.68486080, 7.50482007, 3.75048679, 3.35195669,
    3.69724988, 8.47624552, 9.35225577, 5.56066125, 4.87135158, 4.39283719
};


