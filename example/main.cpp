// Gain Robust Tracker - Joint Radiometric Calibration and Feature Tracking System
// Copyright (c) 2022, Algorithm Development Team. All rights reserved.
//
// This software was developed of Jacob.lsx

#include <dirent.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "gain_robust_tracker.h"

// Number of active features aimed at when tracking
#define MAX_NR_ACTIVE_FEATURES (300)

// Number of pyramid levels used for KLT tracking
#define NR_PYRAMID_LEVELS (3)

// Patch size used for KLT offset optimization
#define KLT_PATCH_SIZE (2)

// Forward backward tracking error threshold (if error is larger than this the track is set invalid)
#define FWD_BWD_TRACKING_THRESH (2.0)


inline int getdir(std::string dir, std::vector<std::string>& files);
std::vector<cv::Point2f> extractFeatures(const cv::Mat& frame, const std::vector<cv::Point2f>& old_features);

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cout << "usage: gain_robust_tracker [PATH_TO_IMAGE]" << std::endl;
    return 1;
  }

  std::vector<std::string> files;
  int img_number = getdir(argv[1], files);

  cv::Mat last_frame;
  std::vector<cv::Point2f> old_features, new_features;
  GainRobustTracker gain_robust_tracker(KLT_PATCH_SIZE, NR_PYRAMID_LEVELS, false);

  for (int i = 0; i < img_number; i++) {
    new_features.clear();
    cv::Mat input_image = cv::imread(files.at(i), cv::IMREAD_GRAYSCALE);

    // First frame - extract features and push them back
    if (i == 0) {
      auto new_feature_locations = extractFeatures(input_image, new_features);
      new_features.insert(new_features.end(), new_feature_locations.begin(), new_feature_locations.end());
      std::swap(old_features, new_features);
      last_frame = input_image;
      continue;
    }

    // Track the feature locations forward using gain robust KLT
    std::vector<cv::Point2f> tracked_points_new_frame;
    std::vector<uchar> tracked_point_status_int1;
    gain_robust_tracker.trackImagePyramids(last_frame, input_image,
                                           old_features, tracked_points_new_frame,
                                           tracked_point_status_int1);

    // Bidirectional tracking filter: Track points backwards and make sure its consistent
    std::vector<cv::Point2f> tracked_points_backtracking;
    std::vector<uchar> tracked_point_status_int2;
    gain_robust_tracker.trackImagePyramids(input_image, last_frame,
                                           tracked_points_new_frame, tracked_points_backtracking,
                                           tracked_point_status_int2);

    // Tracked points from backtracking and old frame should be the same -> check and filter by distance
    std::vector<uchar> tracked_point_status(old_features.size(), 0);
    for (int p = 0; p < old_features.size(); p++) {
      // Point already set invalid by forward tracking or backtracking -> ignore
      if (tracked_point_status_int1.at(p) == 0 || tracked_point_status_int2.at(p) == 0)
        continue;

      // Valid in front + backtracked images -> calculate displacement error
      cv::Point2d d_p = old_features.at(p) - tracked_points_backtracking.at(p);
      double distance = sqrt(d_p.x * d_p.x + d_p.y * d_p.y);

      if (distance > FWD_BWD_TRACKING_THRESH)
        continue;

      tracked_point_status.at(p) = 1;
      new_features.push_back(tracked_points_new_frame.at(p));
    }

    // Extract new features
    std::vector<cv::Point2f> new_feature_locations = extractFeatures(input_image, new_features);

    // Show tracking result
    cv::Mat draw_image = input_image.clone();
    cv::cvtColor(draw_image, draw_image, CV_GRAY2RGB);

    for (int p = 0; p < old_features.size(); p++) {
      if (tracked_point_status.at(p) == 0)
        continue;

      cv::circle(draw_image, tracked_points_new_frame.at(p), 3, cv::Scalar(0, 255, 0));
      cv::line(draw_image, old_features.at(p), tracked_points_new_frame.at(p), cv::Scalar(0, 255, 0));
    }

    for (const auto& p : new_feature_locations) {
      cv::circle(draw_image, p, 3, cv::Scalar(255, 0, 0));
    }

    std::cout << "tracking points number: " << new_features.size() << ", lost points number: "
              << old_features.size() - new_features.size() << std::endl;
    std::cout << "extract new points number: " << new_feature_locations.size() << std::endl;

    cv::imshow("Tracking", draw_image);
    cv::waitKey(10);

    // Update frame
    last_frame = input_image;
    std::swap(old_features, new_features);
    old_features.insert(old_features.end(), new_feature_locations.begin(), new_feature_locations.end());
  }

  cv::destroyAllWindows();

  return 0;
}

int getdir(std::string dir, std::vector<std::string>& files) {
  DIR *dp;
  struct dirent *dirp;
  if ((dp = opendir(dir.c_str())) == NULL) {
    return -1;
  }

  while ((dirp = readdir(dp)) != NULL) {
    std::string name = std::string(dirp->d_name);
    if (name != "." && name != "..")
      files.emplace_back(name);
  }
  closedir(dp);

  std::sort(files.begin(), files.end());

  if (dir.at(dir.length()-1) != '/')
    dir += "/";

  for (auto& item : files) {
    if (item.at(0) != '/')
      item = dir + item;
  }

  return files.size();
}

std::vector<cv::Point2f> extractFeatures(const cv::Mat& frame, const std::vector<cv::Point2f>& old_features) {
  std::vector<cv::Point2f> new_features;

  // No new feature have to be extracted
  if (old_features.size() >= MAX_NR_ACTIVE_FEATURES) {
    return new_features;
  }

  int nr_features_to_extract = static_cast<int>(MAX_NR_ACTIVE_FEATURES - old_features.size());

  // Build spatial distribution map to check where to extract features
  const int cells_r = 10;
  const int cells_c = 10;

  int cell_height = frame.rows / cells_r;
  int cell_width = frame.cols / cells_c;

  int point_distribution_map[cells_r][cells_c] = {0};

  // Build the point distribution map to check where features need to be extracted mostly
  for (int p = 0; p < old_features.size(); p++) {
    float x_value = old_features.at(p).x;
    float y_value = old_features.at(p).y;

    int c_bin = x_value / cell_width;
    if (c_bin >= cells_c)
      c_bin = cells_c - 1;

    int r_bin = y_value / cell_height;
    if (r_bin >= cells_r)
      r_bin = cells_r - 1;

    point_distribution_map[r_bin][c_bin]++;
  }

  // Identify empty cells
  std::vector<int> empty_row_indices;
  std::vector<int> empty_col_indices;

  for (int r = 0; r < cells_r; r++) {
    for (int c = 0;c < cells_c;c++) {
      if (point_distribution_map[r][c] == 0) {
        empty_row_indices.push_back(r);
        empty_col_indices.push_back(c);
      }
    }
  }

  // Todo: empty_col_indices might be 0!!!
  // Todo: Another bad case is: only one cell is empty and all other cells have only 1 feature inside,
  // Todo: then all the features to extract will be extracted from the single empty cell.
  int points_per_cell = ceil(nr_features_to_extract / (empty_col_indices.size()*1.0));

  // Extract "points per cell" features from each empty cell
  for(int i = 0;i < empty_col_indices.size();i++) {
    // Select random cell from where to extract features
    int random_index = rand() % empty_row_indices.size();

    // Select row and col
    int selected_row = empty_row_indices.at(random_index);
    int selected_col = empty_col_indices.at(random_index);

    // Define the region of interest where to detect a feature
    cv::Rect ROI(selected_col * cell_width, selected_row * cell_height, cell_width, cell_height);

    // Extract features from this frame
    cv::Mat frame_roi = frame(ROI);

    // Extract features
    std::vector<cv::Point2f> good_corners;
    cv::goodFeaturesToTrack(frame_roi,
                            good_corners,
                            points_per_cell,
                            0.01,
                            7,
                            cv::Mat(),
                            7,
                            false,
                            0.04);

    // Add the strongest "points per cell" features from this extraction
    for (int k = 0; k < good_corners.size(); k++) {
      if (k == points_per_cell)
        break;

      // Add the offset to the point location
      cv::Point2f point_location = good_corners.at(k);
      point_location.x += selected_col*cell_width;
      point_location.y += selected_row*cell_height;

      new_features.emplace_back(point_location);
    }
  }

  return new_features;
}
