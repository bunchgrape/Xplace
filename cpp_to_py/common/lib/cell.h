#pragma once
#include "cellpin.h"

namespace gt {

struct Cell {
    std::string name;
    std::string cell_footprint;

    std::optional<float> leakage_power;
    std::optional<float> area;

    std::optional<string> primary_ground;
    std::optional<string> primary_power;

    int num_bits = 0;

    std::unordered_map<std::string, Cellpin> cellpins;

    float average_delay();

    void scale_time(float s);
    void scale_capacitance(float s);

    Cellpin* cellpin(const std::string&);
};

std::ostream& operator<<(std::ostream&, const Cell&);

using CellView = std::array<const Cell*, MAX_SPLIT>;

};  // namespace gt
