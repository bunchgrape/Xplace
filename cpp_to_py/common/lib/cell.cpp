#include "cell.h"

namespace gt {

void Cell::scale_time(float s) {
    for (auto& kvp : cellpins) {
        kvp.second.scale_time(s);
    }
}

void Cell::scale_capacitance(float s) {
    for (auto& kvp : cellpins) {
        kvp.second.scale_capacitance(s);
    }
}

Cellpin* Cell::cellpin(const std::string& name) {
    if (auto itr = cellpins.find(name); itr == cellpins.end()) {
        return nullptr;
    } else
        return &(itr->second);
}

std::ostream& operator<<(std::ostream& os, const Cell& c) {
    // Write the cell name.
    os << "cell (\"" << c.name << "\") {\n";

    if (!c.cell_footprint.empty()) {
        os << "  cell_footprint : " << c.cell_footprint << ";\n";
    }

    if (c.leakage_power) {
        os << "  cell_leakage_power : " << *c.leakage_power << ";\n";
    }

    if (c.area) {
        os << "  area : " << *c.area << ";\n";
    }

    // Write the cellpins.
    for (const auto& kvp : c.cellpins) {
        os << kvp.second;
    }

    // Write the ending group symbol.
    os << "}\n";

    return os;
}

float Cell::average_delay() {
    float total_delay = 0;
    int table_cnt = 0;
    for (const auto& kvp : cellpins) {
        auto cellpin = kvp.second;
        if (cellpin.name == "QN" || cellpin.name == "QN1") {
            for (auto timing : cellpin.timings) {
                auto delay = timing.delay(gt::Tran::RISE, gt::Tran::FALL, 30, 30);
                if (delay) {
                    total_delay += *delay;
                    table_cnt++;
                }
            }
        }
    }
    if (table_cnt == 0) {
        return 0;
    }
    return total_delay / table_cnt;
}

};  // namespace gt
