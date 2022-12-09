// Gain Robust Tracker - Joint Radiometric Calibration and Feature Tracking System
// Copyright (c) 2022, Algorithm Development Team. All rights reserved.
//
// This software was developed of Jacob.lsx

#ifndef GAIN_ROBUST_TRACKER_H_
#define GAIN_ROBUST_TRACKER_H_

#include <vector>
#include <opencv2/core/core.hpp>

/**
 * This class implements gain robust KLT tracking
 * optimizing jointly for displacements of features, radiometric parameters and an exposure ratio between input frames
 */
class GainRobustTracker {
 public:
  /**
   * Constructor
   *
   * @param patch_size Size of tracking patches
   * @param pyramid_levels Number of pyramid levels used for KLT tracking
   * @param only_exposure_ratio Only optimizing exposure ratio between input frames,
   *                            if images was be photometrically rectify first.
   */
  GainRobustTracker(int patch_size, int pyramid_levels, bool only_exposure_ratio = false);

  /**
   * Track a new image using radiometric parameters calibration, exposure estimation & image pyramids
   *
   * @param frame_1 Frame to track points from
   * @param frame_2 Frame to track points to
   * @param pts_1 Given point locations in frame_1
   * @param pts_2 Output point locations in frame_2 (tracked from frame_1 to frame_2)
   * @param point_status Vector indicating point validity (set to 0 by tracker if e.g. tracked patches leave input images)
   * @returns Exposure ratio estimate between frame_1 and frame_2 based on KLT optimization
   */
  double trackImagePyramids(const cv::Mat& frame_1,
                            const cv::Mat& frame_2,
                            const std::vector<cv::Point2f>& pts_1,
                            std::vector<cv::Point2f>& pts_2,
                            std::vector<uchar>& point_status);

 private:
  /**
   * Track points on a specific pyramid layer
   *
   * @param old_image First input image
   * @param new_image Second input image, track new features to this image
   * @param input_points Original points in first input image
   * @param output_points Tracked point locations in second input image
   * @returns Exposure ratio estimate between first and second input image
   */
  double trackImageRobustPyr(const cv::Mat& old_image,
                             const cv::Mat& new_image,
                             const std::vector<cv::Point2f>& input_points,
                             std::vector<cv::Point2f>& output_points,
                             std::vector<uchar>& point_validity);

  /**
   * Get number of valid points inside the specified validity vector
   *
   * @param validity_vector Vector of validity flags corresponding to tracking points
   * @returns Number of valid flags inside the input vector
   */
  int getNrValidPoints(const std::vector<uchar>& validity_vector);

  /**
   * Update radiometric parameters
   * @param c1, c2, c3 the coefficients for the Grossberg base functions (hinv_1, hinv_2, hinv_3)
   * @param R the covariance matrix for estimation radiometric parameters
   * @return none.
   */
  int updateRadiometricParameters(double c1, double c2, double c3, cv::Matx33d& R);

  /**
   * Evaluate one of the Grossberg base functions or their derivatives (g_0, hinv_1, hinv_2, hinv_3, ...) at location x
   * Base functions are approximated by ploynomials (of Max degree 10)
   *
   * @param base_function_index index of which base function be calculation
   * @param is_derivative Whether to solve derivative of Grossberg base functions
   * @param x intensity value [0 ~ 1]
   * @return irradiance value or its derivative value
   */
  double evaluateGrossbergBaseFunction(int base_function_index, bool is_derivative, double x);

  /**
   * Get mapping table of inverse response function
   */
  void getInverseResponse();

  /**
   * get a value of inverse response function
   */
  double applyResponse(double x);

  /**
   * change in intensity image to irradiance image by inverse response function
   */
  cv::Mat mapImage(const cv::Mat& image);

  /**
   * Patch size used for tracking of image patches
   */
  int patch_size_;

  /**
   * Number of pyramid levels used for tracking
   */
  int pyramid_levels_;

  /**
   * Only optimizing exposure ratio between input frames
   */
  bool only_exposure_ratio_;

  /**
   * Parameters of radiometric have converged
   */
  bool radiometric_converged_;

  /**
   * parameters of radiometric
   */
  cv::Vec3d radiometric_parameters_;

  /**
   * the estimate error covariance matrix for parameters of radiometric
   */
  cv::Matx33d P_;

  /**
   * Inverse response function
   */
  std::vector<double> inverse_response_function_;

  /**
   * Here follow the PCA components of the first 3 basis functions of the Grossberg model as discrete 1024 component vectors
   * The derivatives have 1023 components, obtained by symmetric differences
   */
  static const double g_0_[1024];
  static const double hinv_1_[1024];
  static const double hinv_2_[1024];
  static const double hinv_3_[1024];

  static const double g_0_der_[1022];
  static const double hinv_1_der_[1022];
  static const double hinv_2_der_[1022];
  static const double hinv_3_der_[1022];
};

#endif // GAIN_ROBUST_TRACKER_H_

