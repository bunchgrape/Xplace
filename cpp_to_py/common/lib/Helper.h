#pragma once

#include <iostream>
#include <iomanip>
#include <fstream>
#include <thread>
#include <sstream>
#include <mutex>
#include <shared_mutex>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <dirent.h>
#include <vector>
#include <cstring>
#include <string_view>
#include <memory>
#include <map>
#include <future>
#include <atomic>
#include <list>
#include <forward_list>
#include <unordered_map>
#include <set>
#include <stack>
#include <queue>
#include <deque>
#include <tuple>
#include <unordered_set>
#include <numeric>
#include <iterator>
#include <functional>
#include <cstddef>
#include <type_traits>
#include <algorithm>
#include <cassert>
#include <random>
#include <regex>
#include <ratio>
#include <optional>
#include <unistd.h>
#include <sys/wait.h>
#include <variant>

#if defined(__has_include) && __has_include(<filesystem>)
#include <filesystem>
#else
#include <experimental/filesystem>
namespace std {
namespace filesystem = experimental::filesystem;
};
#endif

#include "units.hpp"
#include "common/common.h"

namespace gt {

using namespace std::chrono_literals;
using namespace std::literals::string_literals;

enum Split {
  MIN = 0,
  MAX = 1
};

enum Tran {
  RISE = 0,
  FALL = 1
};

constexpr int MAX_SPLIT = 2;
constexpr int MAX_TRAN = 2;

constexpr std::initializer_list<Split> SPLIT = {MIN, MAX};
constexpr std::initializer_list<Tran> TRAN = {RISE, FALL};

// #define FOR_EACH_EL(el) for(auto el : SPLIT)
// #define FOR_EACH_RF(rf) for(auto rf : TRAN)
// #define FOR_EACH_RF_RF(irf, orf) for(auto [irf, orf] : TRANX2)
// #define FOR_EACH_EL_RF(el, rf) for(auto [el, rf] : SPLIT_TRAN)
// #define FOR_EACH_EL_RF_RF(el, rf1, rf2) for(auto [el, rf1, rf2] : SPLIT_TRANx2)

// #define FOR_EACH_EL_IF(el, c) for(auto el : SPLIT) if(c)
// #define FOR_EACH_RF_IF(rf, c) for(auto rf : TRAN) if(c)
// #define FOR_EACH_RF_RF_IF(irf, orf, c) for(auto [irf, orf] : TRANX2) if(c)
// #define FOR_EACH_EL_RF_IF(el, rf, c) for(auto [el, rf] : SPLIT_TRAN) if(c)
// #define FOR_EACH_EL_RF_RF_IF(el, rf1, rf2, c) for(auto [el, rf1, rf2] : SPLIT_TRANx2) if(c)

// #define FOR_EACH(i, C) for(auto& i : C)
// #define FOR_EACH_IF(i, C, s) for(auto& i : C) if(s)

};  // namespace gt
