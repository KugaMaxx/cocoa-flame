#pragma once

#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include <dv-toolkit/toolkit.hpp>
#include <dv-processing/processing.hpp>

namespace py = pybind11;

namespace dv::detection {

template<class EventStoreClass = dv::EventStore>
class FlameScout {
public:
  explicit FlameScout(const cv::Size &resolution, float_t min_area,
                      size_t candidate_num, float_t threshold)
      : height_(resolution.height), 
        width_(resolution.width),
        min_area_(min_area), 
        candidate_num_(candidate_num),
        threshold_(threshold) {}

  void accept(const EventStoreClass &store){
    if (store.isEmpty()) {
      return;
    }

		if (store.getLowestTime() < highest_time_) {
			throw std::out_of_range{"Tried adding event store to store out of order. Ignoring packet."};
		}

    buffer_.add(store);
  }

  std::vector<py::array_t<int32_t>> detect() {
    // discharge return
    if (buffer_.isEmpty()) {
      return {{}};
    }

    // convert to binary image
    cv::Mat image = cv::Mat::zeros(width_, height_, CV_8UC1);
    for (const auto& event : buffer_) {
      image.at<uint8_t>(event.x(), event.y()) = 255;
    }

    // find possible regions
    RegionSet region_set = findContoursRect(image);

    // selective search
    std::vector<py::array_t<int32_t>> results = selectiveBoundingBox(region_set);

    return results;
  }

private:
  struct RegionSet {
  private:
    size_t _ind = 0;
    size_t _size = 0;

  public:
    std::vector<cv::Rect> rect;
    std::vector<cv::Point2f> center;
    std::vector<float_t> radius;
    std::vector<uint32_t> rank;
    std::vector<uint32_t> label;

    size_t size() { return _size; }

    RegionSet(){};
    RegionSet(size_t length) : _size(length), _ind(0) {
      rect.resize(_size);
      center.resize(_size);
      radius.resize(_size);
      rank.resize(_size);
      label.resize(_size);
    }

    inline void push_back(const cv::Rect &rect_, const cv::Point2f center_,
                          const float radius_) {
      rect[_ind] = rect_;
      center[_ind] = center_;
      radius[_ind] = radius_;
      label[_ind] = _ind;
      _ind++;
    }

    inline int find(int i) { return (label[i] == i) ? i : find(label[i]); }

    inline void group(int i, int j) {
      int x = find(i), y = find(j);
      if (x != y) {
        if (rank[x] <= rank[y]) {
          label[x] = y;
        } else {
          label[y] = x;
        }
        if (rank[x] == rank[y]) {
          rank[y]++;
        }
      }
      return;
    }
  };

  RegionSet findContoursRect(cv::Mat image) {
    // find contours
    std::vector<cv::Vec4i> hierarchy;
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(image, contours, hierarchy, cv::RETR_TREE,
                     cv::CHAIN_APPROX_SIMPLE);

    // construct set of rectangle
    RegionSet region_set(contours.size());
    for (const auto &contour : contours) {
      // approximate polyline
      std::vector<cv::Point> contour_poly;
      cv::approxPolyDP(contour, contour_poly, 3, true);

      // get boundary rectangle
      cv::Rect bound_rect = cv::boundingRect(contour_poly);

      // search minimal enclosing circle
      float radius;
      cv::Point2f center;
      cv::minEnclosingCircle(contour_poly, center, radius);

      // emplace back
      region_set.push_back(bound_rect, center, radius);
    }

    return region_set;
  }

  std::vector<py::array_t<int32_t>> selectiveBoundingBox(RegionSet &region_set) {
    // group rectangles by calculate similarity
    for (size_t i = 0; i < region_set.size(); i++) {
      for (size_t j = i + 1; j < region_set.size(); j++) {
        if (calcSimilarity(region_set, i, j) >= threshold_) {
          region_set.group(i, j);
        }
      }
    }

    // merge rectangles within same group
    std::map<int32_t, cv::Rect> rects;
    for (size_t i = 0; i < region_set.size(); i++) {
      int k = region_set.find(i);
      if (!rects.count(k)) {
        rects[k] = region_set.rect[i];
        continue;
      }
      rects[k] |= region_set.rect[i];
    }

    // get candidate regions
    std::vector<std::pair<int32_t, cv::Rect>> rankedRect;
    for (size_t i = 0; i < rects.size(); i++) {
      if (rects[i].area() < min_area_)
        continue;
      rankedRect.push_back(std::make_pair(i, rects[i]));
    }
    std::sort(rankedRect.begin(), rankedRect.end(),
              [](auto &left, auto &right) {
                return left.second.area() > right.second.area();
              });

    // convert to lists of coordinates
    std::vector<py::array_t<int32_t>> result;
    for (size_t i = 0; i < rankedRect.size() && i < candidate_num_; i++) {
        auto rect = rankedRect[i].second;
        std::vector<int32_t> vect = {rect.x, rect.y, rect.width, rect.height};
        result.push_back(py::cast(vect));
    }

    return result;
  }

  inline float_t calcSimilarity(RegionSet &region_set, int i, int j) {
    // calculate intersect
    float_t intsR = (region_set.rect[i] & region_set.rect[j]).area();
    float_t unitR =
        std::min(region_set.rect[i].area(), region_set.rect[j].area());
    float_t score_1 = intsR / unitR;

    // calculate radius
    float_t dist = cv::norm(region_set.center[i] - region_set.center[j]);
    float_t sumR = region_set.radius[i] + region_set.radius[j];
    float_t score_2 = dist < sumR ? 1. : sumR / dist;

    return 0.2 * score_1 + 0.8 * score_2;
  }

  size_t height_;
  size_t width_;
  size_t candidate_num_;
  float_t min_area_;
  float_t threshold_;
  cv::Mat binary_img_;
  EventStoreClass buffer_;
  int64_t highest_time_ = -1;
};

} // namespace dv::detection