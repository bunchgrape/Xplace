

#pragma once
#include "common/common.h"
#include "common/db/Database.h"
#include "gputimer/base.h"
#include "common/lib/celllib.h"
#include "common/lib/sdc/sdc.h"
#include "../traits.h"

namespace gp {
class GPDatabase;
class GPPin;
class GPNet;
}

namespace db {
class Database;
};

namespace gt {
class CellLib;
class LibertyCell;
class LibertyPort;
class TimingArc;
class Lut;
};

namespace gt {
class TimingTorchRawDB;
class clock;
class Pin;
class Net;
class Arc;
class cellpin;

class Pin {
public:
    const gp::GPPin* gppin;
    string name;
    int io = -1;  // 0:i/1:o
    vector<index_type> fanin;
    vector<index_type> fanout;
    set<index_type> fanin_idx;
    set<index_type> fanout_idx;
    int id() const;
    Pin(string& name_, const gp::GPPin* gppin_) : name(name_), gppin(gppin_) { ; }
};
class Clock {
private:
    std::string _name;
    Pin* _source{nullptr};
    float _period{.0f};
    std::array<float, MAX_TRAN> _waveform;
public:
    Clock(const std::string& name, float period) : _name{name}, _source{nullptr}, _period{period}, _waveform{0.0f, period / 2.0f} {};
    Clock(const std::string& name, Pin* source, float period) : _name{name}, _source{source}, _period{period}, _waveform{0.0f, period / 2.0f} {};
    inline const std::string& name() const;
    inline float period() const { return _period; }
    inline float waveform(Tran rf) const { return _waveform[rf]; }
    inline string source_name() { return _source->name; }
    Pin* source() { return _source; }
};
class Net {
public:
    const gp::GPNet* gpnet;
    string name;
    Net(const gp::GPNet* gpnet_);
};
class Arc {
public:
    Pin* from;
    Pin* to;
    int el;
    TimingType type;
    string related_pin;
    string timing_pin;
    string lib_pin_name;
    std::variant<Net*, array<int, 2>> Attribute;
    Arc(Pin* from_, Pin* to_, Net* net_) : from(from_), to(to_), Attribute(net_) { ; }
    Arc(Pin* from_, Pin* to_, array<int, 2> tv_idx_) : from(from_), to(to_), Attribute(tv_idx_) { ; }
};
class cellpin {
public:
    const CellpinView cpv;
    std::array<vector<int>, MAX_SPLIT> timings;
    std::array<vector<float>, MAX_SPLIT> capacitance;
    std::array<vector<float>, MAX_SPLIT> max_transition;
    cellpin(const CellpinView cpv_) : cpv(cpv_) {
        capacitance[MIN].resize(3, nanf(""));
        capacitance[MAX].resize(3, nanf(""));
    }
};
class test {
public:
    Arc* arc;
    test(Arc* arc_) : arc(arc_) { ; }
};


// ======================================================================================
// GPU Timing Database
// 
class GTDatabase {
public:
    db::Database& rawdb;
    gp::GPDatabase& gpdb;
    TimingTorchRawDB& timing_raw_db;
    GTDatabase(shared_ptr<db::Database> rawdb_, shared_ptr<gp::GPDatabase> gpdb_, shared_ptr<TimingTorchRawDB> timing_raw_db_);
    ~GTDatabase() { logger.info("destruct gtdb"); }
    void CreateNetlist();
    void readSpef(const std::string& file);
    void readSdc(sdc::SDC& sdc);
    void _read_sdc(sdc::SetInputDelay&);
    void _read_sdc(sdc::SetInputTransition&);
    void _read_sdc(sdc::SetOutputDelay&);
    void _read_sdc(sdc::SetLoad&);
    void _read_sdc(sdc::CreateClock&);
    void _read_sdc(sdc::SetUnits&);

public:
    // Units
    optional<float> sdc_res_unit;
    optional<float> sdc_cap_unit;
    optional<float> sdc_time_unit;

    optional<float> spef_res_unit;
    optional<float> spef_cap_unit;
    optional<float> spef_time_unit;

public:
    // Incremental update
    void swap_gate_type(int node_id, int bit_group, int bit_index);

public:
    // Timing Liberty
    /// @param celllib                  Cell libraries
    /// @param cell_views               vector of celllib views
    /// @param cell2view                map from cell name to cell view index
    TimingData<std::shared_ptr<gt::Celllib>, MAX_SPLIT> celllib;
    vector<CellView> cell_views;
    unordered_map<string, int> cell2view;
    unordered_map<int, vector<int>> cell_gpid2arcs;
    unordered_map<int, vector<int>> pin_id2arcs;

    /// @param timings                  vector of timing data
    /// @param pin2timing               map from pin name to cellpin timing view
    /// @param pin_capacitance          vector of pin capacitance
    vector<Timing*> timings;
    unordered_map<string, cellpin*> pin2timing;
    vector<float> pin_capacitance;


    // Pin variables
    /// @param pins                     vector of pin objects
    /// @param pin_names                vector of pin names
    /// @param pin2idx                  map from pin name to pin index
    /// @param arcs                     vector of arc objects
    vector<Pin*> pins;
    vector<string> pin_names;
    unordered_map<string, index_type> pin2idx;
    vector<Arc*> arcs;

    vector<string> net_names;
    vector<Net*> nets;
    vector<test> tests;

    unordered_map<std::string, Clock> clocks;
    bool is_redundant_timing(const Timing& timing, Split el, const string& cpname);

public:
    int num_pins, num_arcs, num_timings, num_fanout_pins, num_tests, num_POs;
    vector<index_type> pin_ins, pin_outs;
    vector<index_type> pin_frontiers;
    void stack_pins(int idx);
    unordered_map<string, index_type> pi2idx;
    unordered_map<string, index_type> po2idx;

    vector<int> pin_num_fanin;
    vector<index_type> pin_fanout_list_end, pin_fanout_list;
    vector<index_type> pin_f_arc_list_end, pin_f_arc_list;
    vector<index_type> pin_b_arc_list_end, pin_b_arc_list;
    vector<index_type> arc_from_pin, arc_to_pin;
    vector<int> arc_types, arc_timings, arc_tests;
    vector<int> test_to_arc;
    vector<int> net_is_clock;

    // timing data
    float res_unit;
    float cap_unit;
    float time_unit;
};

class TimingTorchRawDB {
public:
    TimingTorchRawDB(torch::Tensor node_lpos_init_,
                     torch::Tensor node_size_,
                     torch::Tensor pin_rel_lpos_,
                     torch::Tensor pin_id2node_id_,
                     torch::Tensor pin_id2net_id_,
                     torch::Tensor node2pin_list_,
                     torch::Tensor node2pin_list_end_,
                     torch::Tensor hyperedge_list_,
                     torch::Tensor hyperedge_list_end_,
                     torch::Tensor net_mask_,
                     int num_movable_nodes_,
                     float scale_factor_,
                     int microns_,
                     float wire_resistance_per_micron_,
                     float wire_capacitance_per_micron_);

    void commit_from(torch::Tensor x_, torch::Tensor y_);
    torch::Tensor get_curr_cposx();
    torch::Tensor get_curr_cposy();
    torch::Tensor get_curr_lposx();
    torch::Tensor get_curr_lposy();

public:
    /* node info */
    // for backup
    torch::Tensor node_lpos_init;
    torch::Tensor node_size;
    torch::Tensor pin_rel_lpos;

    torch::Tensor init_x;  // original pos (keep it const except committing)
    torch::Tensor init_y;  // original pos (keep it const except committing)
    torch::Tensor x;       // mutable/cached pos (current)
    torch::Tensor y;       // mutable/cached pos (current)
    torch::Tensor node_size_x;
    torch::Tensor node_size_y;

    /* pin info */
    torch::Tensor pin_offset_x;
    torch::Tensor pin_offset_y;

    // gputimer api tensors
    torch::Tensor at_prefix_pin;
    torch::Tensor at_prefix_arc;
    torch::Tensor at_prefix_attr;

    torch::Tensor flat_node2pin_start_map;
    torch::Tensor flat_node2pin_map;
    torch::Tensor pin2node_map;

    /* net info */
    torch::Tensor flat_net2pin_start_map;
    torch::Tensor flat_net2pin_map;
    torch::Tensor pin2net_map;
    torch::Tensor net_mask;

    /* chip info */
    int num_pins;
    int num_nets;
    int num_nodes;
    int num_movable_nodes;

    int num_threads;

public:
    float scale_factor;
    float wire_resistance_per_micron = 2.535;
    float wire_capacitance_per_micron = 0.16e-15;
    int microns = 2000;

// Timer variables
/// @param pinSlew                  Slew value on a pin
/// @param pinLoad                  Load value on a pin
/// @param pinRat                   Required arrival time on a pin
/// @param pinAt                    Arrival time on a pin
/// @param pinImpulse               Impulse value on a pin
/// @param pinRootDelay             Root delay value on a sink pin
public:
    // vector<float> pinSlew, pinLoad, pinRat, pinAt;
    // vector<float> pinImpulse, pinRootDelay;
    torch::Tensor pinSlew;
    torch::Tensor pinLoad;
    torch::Tensor pinRat;
    torch::Tensor pinAt;
    torch::Tensor pinImpulse;
    torch::Tensor pinRootDelay;

// Timer RC Tree variables
/// @param endpoints_index          Index of the endpoints
/// @param arcDelay                 Delay value of an arc
/// @param pinImpulse_ref           Reference impulse value of a accurate Timer
/// @param pinLoad_ref              Reference load value of a accurate Timer
/// @param pinLoad_ratio            Load ratio value of a accurate Timer
/// @param pinRootDelay_ref         Reference root delay value of a accurate Timer
/// @param pinRootDelay_ratio       Root delay ratio value of a accurate Timer
/// @param pinRootDelay_compensation Root delay compensation value compared to a accurate Timer
public:
    // vector<float> arcDelay;
    torch::Tensor endpoints_index;
    torch::Tensor arcDelay;
    torch::Tensor pinImpulse_ref;
    torch::Tensor pinLoad_ref;
    torch::Tensor pinLoad_ratio;
    torch::Tensor pinRootDelay_ref;
    torch::Tensor pinRootDelay_ratio;
    torch::Tensor pinRootDelay_compensation;

// Timer graph topology variables
/// @param pin_f_arc_list           List of forward arcs of a pin
/// @param pin_f_arc_list_end       Star & End index of the forward arcs lists
/// @param pin_b_arc_list           List of backward arcs of a pin
/// @param pin_b_arc_list_end       Star & End index of the backward arcs lists
/// @param arc_from_pin             From pin index of an arc
/// @param arc_to_pin               To pin index of an arc
/// @param pin_num_fanin            Number of fanin pins of a pin
/// @param pin_fanout_list          List of fanout pins of a pin
/// @param pin_fanout_list_end      Star & End index of the fanout pins lists
public:
    torch::Tensor pin_f_arc_list;
    torch::Tensor pin_f_arc_list_end;
    torch::Tensor pin_b_arc_list;
    torch::Tensor pin_b_arc_list_end;
    torch::Tensor arc_from_pin;
    torch::Tensor arc_to_pin;
    torch::Tensor pin_num_fanin;
    torch::Tensor pin_fanout_list;
    torch::Tensor pin_fanout_list_end;

// Timer timing liberty variables
/// @param arc_types                Types of an arc: 0/1
/// @param arc_timings              Timing liberty index of an arc
/// @param arc_tests                Timing test index of an arc: -1 for non-test arcs
public:
    torch::Tensor arc_types;
    torch::Tensor arc_timings;
    torch::Tensor arc_tests;
};

}  // namespace gt