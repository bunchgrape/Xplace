

#pragma once
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "Helper.h"
#include "EnumNameMap.h"
#include "tokenizer.h"
#include "unit.h"

using std::optional;
using std::string;
using std::unordered_map;
using std::variant;
using std::vector;

namespace db {
class Database;
};

namespace gt {

class CellLib;
class LibertyCell;
class LibertyPort;
class TimingArc;
struct LutTemplate;
class Lut;

// using CellView = std::array<const Cell *, MAX_SPLIT>;
// using CellpinView = std::array<Cellpin *, MAX_SPLIT>;
enum class DelayModel { generic_cmos, table_lookup, cmos2, piecewise_cmos, dcm, polynomial, unknown};
enum class CellPortDirection { input, output, inout, internal, unknown};
DelayModel findDelayModel(const std::string model_name);
CellPortDirection findPortDirection(const std::string dir_name);


class CellLib {
public:
    ~CellLib() { logger.info("Destruct celllib"); }
    CellLib() = default;
    CellLib(db::Database *rawdb_) : rawdb(rawdb_) {}
    db::Database *rawdb = nullptr;

    using token_iterator = std::vector<std::string_view>::iterator;
    DelayModel delay_model;

    string name;

    // unordered_map<string,
    //               variant<optional<second_t>,
    //                       optional<watt_t>,
    //                       optional<ohm_t>,
    //                       optional<farad_t>,
    //                       optional<ampere_t>,
    //                       optional<volt_t>>>
    //     default_units = {{"time_unit", optional<second_t>{}},
    //                      {"power_unit", optional<watt_t>{}},
    //                      {"resistance_unit", optional<ohm_t>{}},
    //                      {"capacitance_unit", optional<farad_t>{}},
    //                      {"current_unit", optional<ampere_t>{}},
    //                      {"voltage_unit", optional<volt_t>{}}};
    optional<second_t> time_unit_;
    optional<watt_t> power_unit_;
    optional<ohm_t> resistance_unit_;
    optional<farad_t> capacitance_unit_;
    optional<ampere_t> current_unit_;
    optional<volt_t> voltage_unit_;

    unordered_map<string, optional<float>> default_values = {
        {"default_cell_leakage_power", optional<float>{}},
        {"default_inout_pin_cap", optional<float>{}},
        {"default_input_pin_cap", optional<float>{}},
        {"default_output_pin_cap", optional<float>{}},
        {"default_fanout_load", optional<float>{}},
        {"default_max_fanout", optional<float>{}},
        {"default_max_transition", optional<float>{}},
        {"voltage", optional<float>{}},
    };

    unordered_map<string, float> scale_factors = {
        {"time", 1.0},
        {"resistance", 1.0},
        {"power", 1.0},
        {"capacitance", 1.0},
        {"current", 1.0},
        {"voltage", 1.0},
    };

    unordered_map<string, LutTemplate *> lut_templates_;
    unordered_map<string, LibertyCell *> lib_cells_;

    void read(const string &file);
    const LutTemplate *lut_template(const string &) const;
    LutTemplate *lut_template(const string &);
    // const Cell *cell(const string &) const;
    // Cell *cell(const string &);

    LibertyCell *extractLibertyCell(token_iterator &, const token_iterator);
    LibertyPort *extractLibertyPort(token_iterator &, const token_iterator, LibertyCell *);
    TimingArc *extractTimingArc(token_iterator &, const token_iterator, LibertyPort *);
    std::optional<float> extract_operating_conditions(token_iterator &itr, const token_iterator end);
    LutTemplate *extract_lut_template(token_iterator &, const token_iterator);
    Lut *extract_lut(token_iterator &, const token_iterator);

private:
    void apply_default_values();
    void uncomment(std::vector<char> &);
    void tokenize(const std::vector<char> &, std::vector<std::string_view> &);
};

class LibertyCell {
public:
    LibertyCell() = default;
    string name;
    vector<LibertyPort *> ports_;

    vector<float> leakage_powers_;
    optional<float> leakage_power_;
    optional<float> area_;

    bool is_seq_ = false;
    int num_bits_ = 0;

    // std::unordered_map<std::string, Cellpin> cellpins;
    // float average_delay();
};

class LibertyPort {
public:
    LibertyPort() = default;
    // LibertyPort(LibertyCell *, string name, bool is_bundle, LibertyPortSeq *members);

public:
    string name;
    LibertyCell *cell_;
    CellPortDirection direction_;

    bool is_bundle_ = false;
    vector<LibertyPort *> member_ports_;

    vector<TimingArc *> timing_arcs_;
    optional<float> port_capacitance_[3];

    optional<bool> is_clock;
    // optional<float> capacitance;
    // optional<float> fall_capacitance;
    // optional<float> rise_capacitance;

    optional<float> fanout_load;
    optional<float> max_fanout;
    optional<float> min_fanout;
    optional<float> max_capacitance;
    optional<float> min_capacitance;
    optional<float> max_transition;
    optional<float> min_transition;

    // std::unordered_map<std::string, std::unordered_map<TimingType, int>> timings_map;
};

};  // namespace gt
