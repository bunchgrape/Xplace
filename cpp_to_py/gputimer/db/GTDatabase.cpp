

#include "GTDatabase.h"
#include "io_parser/gp/GPDatabase.h"

namespace gt {

int Pin::id() const { return gppin->getId(); }

Net::Net(const gp::GPNet* gpnet_) : gpnet(gpnet_) { name = gpnet->getName(); }

void GTDatabase::swap_gate_type(int node_id, int bit_group, int bit_index) {
    int libcell_id = rawdb.cells[gpdb.getNodes()[node_id].getOriDBId()]->ctype()->libcell();
    int change_libcell_id = rawdb.bit_to_celltypes[bit_group][bit_index];
    // logger.info("changing celllib %s to %s",  rawdb.celltypes[libcell_id]->name.c_str(), rawdb.celltypes[change_libcell_id]->name.c_str());
    string change_celltype_name = rawdb.celltypes[change_libcell_id]->name;
     
    for (auto arc_id : cell_gpid2arcs[node_id]) {
        int el = arcs[arc_id]->el;
        int timing_id = arc_timings[2 * arc_id + el];
        if (timing_id == -1) continue;

        string timing_pin_name = arcs[arc_id]->timing_pin;
        string related_pin_name = arcs[arc_id]->related_pin;
        string lib_pin_name = arcs[arc_id]->lib_pin_name;
        string lib_pin_name_change = change_celltype_name + ":" + timing_pin_name + "1";
        if (pin2idx.find(lib_pin_name) == pin2idx.end()) {
            logger.warning("timing pin %s not found", lib_pin_name.c_str());
            continue;
        }
        // logger.info("changing %s to %s", lib_pin_name.c_str(), lib_pin_name_change.c_str());
        int pin_idx = pin2idx[lib_pin_name];
        TimingType type = arcs[arc_id]->type;

        // change cellpin lib
        auto change_pin_view = pin2timing[lib_pin_name_change];
        auto cp = change_pin_view->cpv[el];
        
        // search the timing according to related pin and timing type
        if (cp->timings_map.find(related_pin_name) == cp->timings_map.end()) {
            logger.warning("timing %s not found", related_pin_name.c_str());
            continue;
        }
        auto& pin_timings = cp->timings_map.find(related_pin_name)->second;

        if (pin_timings.find(type) == pin_timings.end()) {
            logger.warning("timing %s not found", to_string(type).c_str());
            continue;
        }
        int timing_id_change = cp->timings[pin_timings.find(type)->second].gtdb_id;

        arc_timings[2 * arc_id + el] = timing_id_change;
        timing_raw_db.arc_timings[2 * arc_id + el] = timing_id_change;
        // logger.info("el %d change timing %d to %d",el,  timing_id, timing_id_change);
    }
}


bool GTDatabase::is_redundant_timing(const Timing& timing, Split el, const string& cpname) {
    if (timing.related_pin.empty()) return true;
    if (timing.related_pin == cpname) return true;
    if (timing.type == TimingType::NON_SEQ_SETUP_RISING || timing.type == TimingType::NON_SEQ_SETUP_FALLING ||
        timing.type == TimingType::NON_SEQ_HOLD_RISING || timing.type == TimingType::NON_SEQ_HOLD_FALLING)
        return true;
    // if (timing.type == TimingType::CLEAR || timing.type == TimingType::COMBINATIONAL || timing.type == TimingType::COMBINATIONAL_RISE ||
    //     timing.type == TimingType::COMBINATIONAL_FALL || timing.type == TimingType::FALLING_EDGE || timing.type == TimingType::PRESET ||
    //     timing.type == TimingType::RISING_EDGE || timing.type == TimingType::THREE_STATE_DISABLE ||
    //     timing.type == TimingType::THREE_STATE_DISABLE_RISE || timing.type == TimingType::THREE_STATE_DISABLE_FALL ||
    //     timing.type == TimingType::THREE_STATE_ENABLE || timing.type == TimingType::THREE_STATE_ENABLE_RISE ||
    //     timing.type == TimingType::THREE_STATE_ENABLE_FALL)
    //     return true;
    switch (el) {
        case MIN:
            if (timing.is_max_constraint()) {
                return true;
            }
            break;
        case MAX:
            if (timing.is_min_constraint()) {
                return true;
            }
            break;
    }
    return false;
}

GTDatabase::GTDatabase(shared_ptr<db::Database> rawdb_, shared_ptr<gp::GPDatabase> gpdb_, shared_ptr<TimingTorchRawDB> timing_raw_db_)
    : rawdb(*rawdb_), gpdb(*gpdb_), timing_raw_db(*timing_raw_db_) {
        // celllib = *(&rawdb.celllib);
        celllib[MIN] = rawdb.celllib[MIN];
        celllib[MAX] = rawdb.celllib[MAX];
}

void GTDatabase::CreateNetlist() {
    res_unit = celllib[MIN]->resistance_unit->value();
    cap_unit = celllib[MIN]->capacitance_unit->value();
    time_unit = celllib[MIN]->time_unit->value();
    pin_names = gpdb.getPinNames();
    net_names = gpdb.getNetNames();
    logger.info(" %.4E, %.4E, %.4E", res_unit, cap_unit, time_unit);

    double elapsed = utils::tstamp.elapsed();
    // ========================================== Create cell library ==========================================
    //
    //     const string& cname = dbcell->ctype()->name;
    for (const auto& dbcelltype : rawdb.celltypes) {
        const string& cname = dbcelltype->name;
        if (cell2view.find(cname) != cell2view.end()) continue;
        auto cell_min = celllib[MIN]->cell(cname);
        auto cell_max = celllib[MAX]->cell(cname);
        CellView cell{cell_min, cell_max};
        if (!cell[MIN] || !cell[MAX]) {
            // logger.warning("cell %s not found in celllib", cname.c_str());
            continue;
        }
        cell_views.push_back(cell);
        cell2view[cname] = cell_views.size() - 1;
        for (auto& [cpname, ecpin] : cell_min->cellpins) {
            string lib_pin_name = cname + ':' + cpname;
            CellpinView cpv{&ecpin, cell_max->cellpin(cpname)};
            if (!cpv[MIN] || !cpv[MAX]) {
                logger.warning("cellpin %s mismatched in celllib", cpname.c_str());
            }
            auto cellpin_in_lib = pin2timing.try_emplace(lib_pin_name, new cellpin(cpv)).first->second;
            FOR_EACH_EL(el) {
                gt::Cellpin* cp = cpv[el];
                if (cp->rise_capacitance) cellpin_in_lib->capacitance[el][0] = cp->rise_capacitance.value();
                if (cp->fall_capacitance) cellpin_in_lib->capacitance[el][1] = cp->fall_capacitance.value();
                cellpin_in_lib->capacitance[el][2] = cp->capacitance ? cp->capacitance.value() : 0.0f;
                for (gt::Timing& tm : cp->timings) {
                    timings.push_back(&tm);
                    cellpin_in_lib->timings[el].push_back(timings.size() - 1);
                    tm.gtdb_id = timings.size() - 1;
                }
            }
        }
    }
    num_timings = timings.size();
    num_pins = gpdb.getPins().size();
    vector<index_type> endpoints_index;
    elapsed = utils::tstamp.elapsed();
    logger.info("Create cell library: %.2fs", elapsed);

    // ========================================== Traverse pins ==========================================
    //
    pin_capacitance.resize(2 * 3 * num_pins, 0.0f);
    for (auto& gppin : gpdb.getPins()) {
        string pin_name = gppin.getName();
        int pin_id = gppin.getId();
        auto pin = pins.emplace_back(new Pin(pin_name, &gppin));
        pin2idx[pin_name] = pin_id;
        auto [ori_node_id, ori_node_pin_id, ori_net_id] = gppin.getOriDBInfo();
        if (ori_node_pin_id == -1) {
            auto dbiopin = rawdb.iopins[ori_node_id];
            if (dbiopin->type->direction() == 'i') {
                pin_outs.push_back(pin_id);
                po2idx[pin_name] = pin_id;
                pin->io = 1;
                endpoints_index.push_back(pin_id);
            } else if (dbiopin->type->direction() == 'o') {
                pin_ins.push_back(pin_id);
                pi2idx[pin_name] = pin_id;
                pin->io = 0;
            }
        } else {
            auto& dbcell = rawdb.cells[ori_node_id];
            string cname = dbcell->ctype()->name;
            string macro_pin_name = dbcell->ctype()->pins[ori_node_pin_id]->name();
            string lib_pin_name = cname + ':' + macro_pin_name;
            auto pin_lib_iter = pin2timing.find(lib_pin_name);
            if (pin_lib_iter == pin2timing.end()) {
                logger.warning("pin %s found in celllib", lib_pin_name.c_str());
            }
            auto cpv = pin_lib_iter->second->cpv;
            FOR_EACH_EL(el) {
                auto& cp = cpv[el];
                pin_capacitance[pin_id * 6 + el * 2 + 0] = cp->rise_capacitance ? cp->rise_capacitance.value() : nanf("");
                pin_capacitance[pin_id * 6 + el * 2 + 1] = cp->fall_capacitance ? cp->fall_capacitance.value() : nanf("");
                pin_capacitance[pin_id * 6 + 4 + el] = cp->capacitance ? cp->capacitance.value() : 0.0f;
            }
        }
    }
    num_POs = pin_outs.size();
    elapsed = utils::tstamp.elapsed();
    logger.info("Create pins: %.2fs", elapsed);

    // ========================================== Connect from-to pins ==========================================
    //
    auto connect_from_to_pin = [&](string from_pin_name, string to_pin_name) -> pair<Pin*, Pin*> {
        if (pin2idx.find(from_pin_name) == pin2idx.end() || pin2idx.find(to_pin_name) == pin2idx.end()) {
            return {nullptr, nullptr};
        }
        index_type from_pin_idx = pin2idx[from_pin_name];
        index_type to_pin_idx = pin2idx[to_pin_name];
        auto from_pin = pins[from_pin_idx];
        auto to_pin = pins[to_pin_idx];
        from_pin->fanout_idx.insert(to_pin_idx);
        to_pin->fanin_idx.insert(from_pin_idx);
        arc_from_pin.push_back(from_pin_idx);
        arc_to_pin.push_back(to_pin_idx);
        from_pin->fanout.push_back(arcs.size());
        to_pin->fanin.push_back(arcs.size());
        // printf("connect %s %s\n", from_pin_name.c_str(), to_pin_name.c_str());
        return {from_pin, to_pin};
    };

    // ========================================== Traverse nets ==========================================
    //
    // net_is_clock.resize(rawdb.nets.size(), 0);
    for (auto& gpnet : gpdb.getNets()) {
        auto net = nets.emplace_back(new Net(&gpnet));
        // if (gpnet.pins().size() == 0) continue;
        string root_name = pins[gpnet.pins()[0]]->name;
        // if (net->name == "iccad_clk") net_is_clock[nets.size() - 1] = 1;
        for (index_type i = 1; i < static_cast<index_type>(gpnet.pins().size()); i++) {
            string sink_name = pins[gpnet.pins()[i]]->name;
            auto [from_pin, to_pin] = connect_from_to_pin(root_name, sink_name);
            auto arc = arcs.emplace_back(new Arc(from_pin, to_pin, net));
            arc_types.push_back(0);
            arc_timings.push_back(-1);
            arc_timings.push_back(-1);
            arc_tests.push_back(-1);
        }
    }
    elapsed = utils::tstamp.elapsed();
    logger.info("Create nets: %.2fs", elapsed);

    // ========================================== Traverse gates ==========================================
    //
    for (auto& dbcell : rawdb.cells) {
        string cname = dbcell->ctype()->name;
        if (cname == "DFFASRHQNx1_ASAP7_75t_R") {
            bool debug = true;
        }
        string gname = dbcell->name();
        int gpdb_id = dbcell->gpdb_id;
        auto cell_lib_iter = cell2view.find(cname);
        if (cell_lib_iter == cell2view.end()) {
            // logger.warning("cell %s not found in celllib", cname.c_str());
            continue;
        }
        auto cell_lib = cell_views[cell2view.find(cname)->second];
        FOR_EACH_EL(el) {
            for (const auto& [cpname, cp] : cell_lib[el]->cellpins) {
                string to_pin_name = gname + ':' + cpname;
                string lib_pin_name = cname + ':' + cpname;
                index_type pin_idx = pin2idx[lib_pin_name];
                // if (cp.is_clock) continue; // FIXME: we currently do not support clock signal timing
                for (auto& tm_idx : pin2timing[lib_pin_name]->timings[el]) {
                    auto& tm = *timings[tm_idx];
                    if (is_redundant_timing(tm, el, cpname)) continue;
                    array<int, 2> tv = {-1, -1};
                    tv[el] = tm_idx;
                    string from_pin_name = gname + ':' + tm.related_pin;
                    auto [from_pin, to_pin] = connect_from_to_pin(from_pin_name, to_pin_name);
                    if (!from_pin || !to_pin) {
                        // logger.warning("pin %s or %s not found", from_pin_name.c_str(), to_pin_name.c_str());
                        continue;
                    }
                    if (to_pin_name == "_133424_:QN") {
                        bool debug = true;
                    } 
                    auto arc = arcs.emplace_back(new Arc(from_pin, to_pin, tv));
                    arc_types.push_back(1);
                    arc_timings.push_back(tv[MIN]);
                    arc_timings.push_back(tv[MAX]);
                    cell_gpid2arcs[gpdb_id].push_back(arcs.size() - 1);
                    pin_id2arcs[pin_idx].push_back(arcs.size() - 1);
                    arc->related_pin = tm.related_pin;
                    arc->timing_pin = cpname;
                    arc->type = *tm.type;
                    arc->lib_pin_name = lib_pin_name;
                    arc->el = el;

                    if (tm.is_constraint()) {
                        tests.emplace_back(arc);
                        test_to_arc.push_back(arcs.size() - 1);
                        arc_tests.push_back(tests.size() - 1);
                        endpoints_index.push_back(pin2idx[to_pin_name]);
                    } else {
                        arc_tests.push_back(-1);
                    }
                }
            }
        }
    }
    num_tests = tests.size();
    elapsed = utils::tstamp.elapsed();
    logger.info("Create gates: %.2fs", elapsed);

    // ========================================== Create hyperedge list of pin connection ==========================================
    //
    num_arcs = arcs.size();
    num_fanout_pins = 0;
    for (index_type i = 0; i < static_cast<index_type>(num_pins); i++) {
        num_fanout_pins += pins[i]->fanout_idx.size();
    }

    pin_fanout_list_end.resize(num_pins + 1);
    pin_fanout_list_end[0] = 0;
    pin_fanout_list.resize(num_fanout_pins);
    pin_num_fanin.resize(num_pins);
    index_type ptr = 0;
    index_type last_idx = 0;
    pin_fanout_list[0] = 1;
    // write to file
    // auto f = fopen("netlist.txt", "w");
    for (index_type i = 0; i < static_cast<index_type>(num_pins); i++) {
        // string& from_pin_name = pins[i]->name;
        for (auto fanout_pin_idx : pins[i]->fanout_idx) {
            pin_fanout_list[ptr++] = fanout_pin_idx;
            string& to_pin_name = pins[fanout_pin_idx]->name;
            // // printf("connect %s %s\n", from_pin_name.c_str(), to_pin_name.c_str());
            // fprintf(f, "%s %s\n", from_pin_name.c_str(), to_pin_name.c_str());
        }
        last_idx += pins[i]->fanout_idx.size();
        pin_fanout_list_end[i + 1] = last_idx;
        pin_num_fanin[i] = pins[i]->fanin_idx.size();
    }
    // fclose(f);
    for (int i = 0; i < num_pins; i++) {
        if (pin_num_fanin[i] == 0) {
            pin_frontiers.push_back(i);
        }
    }

    // ========================================== Create hyperedge list of pin-arc connection ==========================================
    //
    pin_f_arc_list_end.push_back(0);
    pin_b_arc_list_end.push_back(0);
    for (index_type i = 0; i < static_cast<index_type>(num_pins); i++) {
        for (auto fanout_arc : pins[i]->fanout) {
            pin_f_arc_list.push_back(fanout_arc);
        }
        pin_f_arc_list_end.push_back(pin_f_arc_list.size());
        for (auto fanin_arc : pins[i]->fanin) {
            pin_b_arc_list.push_back(fanin_arc);
        }
        pin_b_arc_list_end.push_back(pin_b_arc_list.size());
    }

    auto device = timing_raw_db.node_size.device();
    auto options = torch::TensorOptions().dtype(torch::kInt32);
    // Timer graph topology variables
    timing_raw_db.pin_f_arc_list = torch::from_blob(pin_f_arc_list.data(), {static_cast<index_type>(pin_f_arc_list.size())}, options).contiguous().to(device);
    timing_raw_db.pin_f_arc_list_end = torch::from_blob(pin_f_arc_list_end.data(), {static_cast<index_type>(pin_f_arc_list_end.size())}, options).contiguous().to(device);;
    timing_raw_db.pin_b_arc_list = torch::from_blob(pin_b_arc_list.data(), {static_cast<index_type>(pin_b_arc_list.size())}, options).contiguous().to(device);
    timing_raw_db.pin_b_arc_list_end = torch::from_blob(pin_b_arc_list_end.data(), {static_cast<index_type>(pin_b_arc_list_end.size())}, options).contiguous().to(device);;
    timing_raw_db.arc_from_pin = torch::from_blob(arc_from_pin.data(), {static_cast<index_type>(arc_from_pin.size())}, options).contiguous().to(device);
    timing_raw_db.arc_to_pin = torch::from_blob(arc_to_pin.data(), {static_cast<index_type>(arc_to_pin.size())}, options).contiguous().to(device);
    timing_raw_db.pin_num_fanin = torch::from_blob(pin_num_fanin.data(), {static_cast<index_type>(pin_num_fanin.size())}, options).contiguous().to(device);
    timing_raw_db.pin_fanout_list = torch::from_blob(pin_fanout_list.data(), {static_cast<index_type>(pin_fanout_list.size())}, options).contiguous().to(device);;
    timing_raw_db.pin_fanout_list_end = torch::from_blob(pin_fanout_list_end.data(), {static_cast<index_type>(pin_fanout_list_end.size())}, options).contiguous().to(device);;

    // Timer timing liberty variables
    timing_raw_db.arc_types = torch::from_blob(arc_types.data(), {static_cast<int>(arc_types.size())}, options).contiguous().to(device);
    timing_raw_db.arc_timings = torch::from_blob(arc_timings.data(), {static_cast<int>(arc_timings.size())}, options).contiguous().to(device);
    timing_raw_db.arc_tests = torch::from_blob(arc_tests.data(), {static_cast<int>(arc_tests.size())}, options).contiguous().to(device);
    
    // gputimer arrays
    timing_raw_db.endpoints_index = torch::from_blob(endpoints_index.data(), {static_cast<index_type>(endpoints_index.size())}, options).contiguous().to(device);
    timing_raw_db.pinSlew = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::Device(device))).contiguous();
    timing_raw_db.pinLoad = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::Device(device))).contiguous();
    timing_raw_db.pinRat = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::Device(device))).contiguous();
    timing_raw_db.pinAt = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::Device(device))).contiguous();
    timing_raw_db.pinImpulse = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::Device(device))).contiguous();
    timing_raw_db.pinRootDelay = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::Device(device))).contiguous();
    torch::fill_(timing_raw_db.pinSlew, nanf(""));
    torch::fill_(timing_raw_db.pinRat, nanf(""));
    torch::fill_(timing_raw_db.pinAt, nanf(""));
    torch::fill_(timing_raw_db.pinImpulse, nanf(""));
    torch::fill_(timing_raw_db.pinRootDelay, nanf(""));

    timing_raw_db.arcDelay = torch::zeros({num_arcs, 2 * NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::Device(device))).contiguous();
    timing_raw_db.pinImpulse_ref = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::Device(device))).contiguous();
    timing_raw_db.pinLoad_ref = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::Device(device))).contiguous();
    timing_raw_db.pinLoad_ratio = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::Device(device))).contiguous();
    timing_raw_db.pinRootDelay_ref = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::Device(device))).contiguous();
    timing_raw_db.pinRootDelay_ratio = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::Device(device))).contiguous();
    timing_raw_db.pinRootDelay_compensation = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::Device(device))).contiguous();

    elapsed = utils::tstamp.elapsed();
    logger.info("Create hyperedge list: %.2fs", elapsed);
    logger.info("Design info: %d pins, %d arcs, %d tests %d timings", num_pins, num_arcs, num_tests, num_timings);
}

void GTDatabase::readSdc(sdc::SDC& sdc) {
    for (auto& command : sdc.commands) {
        std::visit(Functors{[this](auto&& cmd) { _read_sdc(cmd); }}, command);
    }

    string clock_name = clocks.begin()->second.source_name();
    float period = clocks.begin()->second.period();
    logger.info("clock: %s, period: %.2f\n", clock_name.c_str(), period);

    net_is_clock.resize(rawdb.nets.size(), 0);
    for (int i = 0; i < nets.size(); i++) {
        auto& net = nets[i];
        if (net->name == clock_name) {
            net_is_clock[i] = 1;
            // int clock_port_id = net->gpnet->pins()[0];
        }
    }

    // set nan slew of PIs to half period
    for (auto& pi : pin_ins) {
        if (torch::isnan(timing_raw_db.pinSlew[pi][0]).item<bool>()) timing_raw_db.pinSlew[pi][0] = 0.0f;
        if (torch::isnan(timing_raw_db.pinSlew[pi][1]).item<bool>()) timing_raw_db.pinSlew[pi][1] = 0.0f;
        if (torch::isnan(timing_raw_db.pinSlew[pi][2]).item<bool>()) timing_raw_db.pinSlew[pi][2] = 0.0f;
        if (torch::isnan(timing_raw_db.pinSlew[pi][3]).item<bool>()) timing_raw_db.pinSlew[pi][3] = 0.0f;
        // if (torch::isnan(pinAt[pi][0]).item<bool>()) pinAt[pi][0] = 0.0f;
        // if (torch::isnan(pinAt[pi][1]).item<bool>()) pinAt[pi][1] = period / 2.0;
        // if (torch::isnan(pinAt[pi][2]).item<bool>()) pinAt[pi][2] = 0.0f;
        // if (torch::isnan(pinAt[pi][3]).item<bool>()) pinAt[pi][3] = period / 2.0;
    }

    if (clocks.begin()->second.source() != nullptr) {
        int clock_pin_id = clocks.begin()->second.source()->id();
        if (torch::isnan(timing_raw_db.pinAt[clock_pin_id][0]).item<bool>()) timing_raw_db.pinAt[clock_pin_id][0] = 0.0f;
        if (torch::isnan(timing_raw_db.pinAt[clock_pin_id][1]).item<bool>()) timing_raw_db.pinAt[clock_pin_id][1] = period / 2.0;
        if (torch::isnan(timing_raw_db.pinAt[clock_pin_id][2]).item<bool>()) timing_raw_db.pinAt[clock_pin_id][2] = 0.0f;
        if (torch::isnan(timing_raw_db.pinAt[clock_pin_id][3]).item<bool>()) timing_raw_db.pinAt[clock_pin_id][3] = period / 2.0;
    }

    // for (auto& pi : pin_ins) {
    //     std::cout << "PI " << pins[pi]->name << " slew: min " << pinSlew[pi][0].item<float>() << ":" << pinSlew[pi][2].item<float>() << " max " <<
    //     pinSlew[pi][1].item<float>() << ":" << pinSlew[pi][3].item<float>() << '\n'; std::cout << "PI " << pins[pi]->name << " at: min " <<
    //     pinAt[pi][0].item<float>() << ":" << pinAt[pi][2].item<float>() << " max " << pinAt[pi][1].item<float>() << ":" <<
    //     pinAt[pi][3].item<float>() << '\n';
    // }
}

// Sets input delay on pins or input ports relative to a clock signal.
void GTDatabase::_read_sdc(sdc::SetUnits& obj) {
    if (obj.time.has_value()) {
        auto s = *obj.time;
        if (s == "ps") sdc_time_unit = 1e-12;
        if (s == "ns") sdc_time_unit = 1e-9;
        if (s == "us") sdc_time_unit = 1e-6;
        if (s == "ms") sdc_time_unit = 1e-3;
        if (s == "s") sdc_time_unit = 1.0; ;
    }
    if (obj.capacitance.has_value()) {
        auto s = *obj.capacitance;
        if (s == "fF") sdc_cap_unit = 1e-15;
        if (s == "pF") sdc_cap_unit = 1e-12;
        if (s == "nF") sdc_cap_unit = 1e-9;
        if (s == "uF") sdc_cap_unit = 1e-6;
        if (s == "F") sdc_cap_unit = 1.0;
    }
    if (obj.resistance.has_value()) {
        auto s = *obj.resistance;
        if (s == "Ohm") sdc_res_unit = 1.0;
        if (s == "kOhm") sdc_res_unit = 1e3;
        if (s == "MOhm") sdc_res_unit = 1e6;
    }
    if (sdc_time_unit.has_value()) printf("sdc time unit: %.2E\n", *sdc_time_unit);
    if (sdc_cap_unit.has_value()) printf("sdc capacitance unit: %.2E\n", *sdc_cap_unit);
    if (sdc_res_unit.has_value()) printf("sdc resistance unit: %.2E\n", *sdc_res_unit);
}


// Sets input delay on pins or input ports relative to a clock signal.
void GTDatabase::_read_sdc(sdc::SetInputDelay& obj) {
    assert(obj.delay_value && obj.port_pin_list);

    auto mask = sdc::TimingMask(obj.min, obj.max, obj.rise, obj.fall);

    std::visit(Functors{[&](sdc::AllInputs&) {
                            for (auto& pi : pin_ins) {
                                FOR_EACH_EL_RF_IF(el, rf, (mask | el) && (mask | rf)) {
                                    // pinAt[pi * NUM_ATTR + (el << 1) + rf] = *obj.delay_value;
                                    float delay = *obj.delay_value;
                                    if (sdc_time_unit.has_value()) delay = delay * *sdc_time_unit / time_unit;
                                    timing_raw_db.pinAt[pi][(el << 1) + rf] = delay;
                                }
                            }
                        },
                        [&](sdc::GetPorts& get_ports) {
                            for (auto& port : get_ports.ports) {
                                if (auto itr = pi2idx.find(port); itr != pi2idx.end()) {
                                    FOR_EACH_EL_RF_IF(el, rf, (mask | el) && (mask | rf)) {
                                        float delay = *obj.delay_value;
                                        if (sdc_time_unit.has_value()) delay = delay * *sdc_time_unit / time_unit;
                                        timing_raw_db.pinAt[itr->second][(el << 1) + rf] = delay;
                                    }
                                } else {
                                    printf(obj.command, ": port ", std::quoted(port), " not found");
                                }
                            }
                        },
                        [](auto&&) { assert(false); }},
               *obj.port_pin_list);
}

// Sets input transition on pins or input ports relative to a clock signal.
void GTDatabase::_read_sdc(sdc::SetInputTransition& obj) {
    assert(obj.transition && obj.port_list);

    auto mask = sdc::TimingMask(obj.min, obj.max, obj.rise, obj.fall);

    std::visit(Functors{[&](sdc::AllInputs&) {
                            for (auto& pi : pin_ins) {
                                FOR_EACH_EL_RF_IF(el, rf, (mask | el) && (mask | rf)) {
                                    float transition = *obj.transition;
                                    if (sdc_time_unit.has_value()) transition = transition * *sdc_time_unit / time_unit;
                                    timing_raw_db.pinSlew[pi][(el << 1) + rf] = transition;
                                }
                            }
                        },
                        [&](sdc::GetPorts& get_ports) {
                            for (auto& port : get_ports.ports) {
                                if (auto itr = pi2idx.find(port); itr != pi2idx.end()) {
                                    FOR_EACH_EL_RF_IF(el, rf, (mask | el) && (mask | rf)) {
                                        float transition = *obj.transition;
                                        if (sdc_time_unit.has_value()) transition = transition * *sdc_time_unit / time_unit;
                                        timing_raw_db.pinSlew[itr->second][(el << 1) + rf] = transition;
                                    }
                                } else {
                                    printf(obj.command, ": port ", std::quoted(port), " not found");
                                }
                            }
                        },
                        [](auto&&) { assert(false); }},
               *obj.port_list);
}

// Sets output delay on pins or input ports relative to a clock signal.
void GTDatabase::_read_sdc(sdc::SetOutputDelay& obj) {
    assert(obj.delay_value && obj.port_pin_list);

    if (clocks.find(obj.clock) == clocks.end()) {
        printf(obj.command, ": clock ", std::quoted(obj.clock), " not found");
        return;
    }

    auto& clock = clocks.at(obj.clock);

    auto mask = sdc::TimingMask(obj.min, obj.max, obj.rise, obj.fall);

    std::visit(Functors{[&](sdc::AllOutputs&) {
                            for (auto& po : pin_outs) {
                                FOR_EACH_EL_RF_IF(el, rf, (mask | el) && (mask | rf)) {
                                    float delay = *obj.delay_value;
                                    if (sdc_time_unit.has_value()) delay = delay * *sdc_time_unit / time_unit;
                                    timing_raw_db.pinRat[po][(el << 1) + rf] = el == MIN ? -delay : clock.period() - delay;
                                }
                            }
                        },
                        [&](sdc::GetPorts& get_ports) {
                            for (auto& port : get_ports.ports) {
                                if (auto itr = po2idx.find(port); itr != po2idx.end()) {
                                    FOR_EACH_EL_RF_IF(el, rf, (mask | el) && (mask | rf)) {
                                        float delay = *obj.delay_value;
                                        if (sdc_time_unit.has_value()) delay = delay * *sdc_time_unit / time_unit;
                                        timing_raw_db.pinRat[itr->second][(el << 1) + rf] = el == MIN ? -delay : clock.period() - delay;
                                    }
                                } else {
                                    printf(obj.command, ": port ", std::quoted(port), " not found");
                                }
                            }
                        },
                        [](auto&&) { assert(false); }},
               *obj.port_pin_list);
}

// Sets the load attribute to a specified value on specified ports and nets.
void GTDatabase::_read_sdc(sdc::SetLoad& obj) {
    assert(obj.value && obj.objects);

    auto mask = sdc::TimingMask(obj.min, obj.max, std::nullopt, std::nullopt);

    std::visit(Functors{[&](sdc::AllOutputs&) {
                            for (auto& po : pin_outs) {
                                FOR_EACH_EL_RF_IF(el, rf, (mask | el) && (mask | rf)) {
                                    float load = *obj.value;
                                    if (sdc_res_unit.has_value()) load = load * *sdc_res_unit / res_unit;
                                    timing_raw_db.pinLoad[po][(el << 1) + rf] = load;
                                }
                            }
                        },
                        [&](sdc::GetPorts& get_ports) {
                            for (auto& port : get_ports.ports) {
                                if (auto itr = po2idx.find(port); itr != po2idx.end()) {
                                    FOR_EACH_EL_RF_IF(el, rf, (mask | el) && (mask | rf)) {
                                        float load = *obj.value;
                                        if (sdc_res_unit.has_value()) load = load * *sdc_res_unit / res_unit;
                                        timing_raw_db.pinLoad[itr->second][(el << 1) + rf] = load;
                                    }
                                } else {
                                    printf(obj.command, ": port ", std::quoted(port), " not found");
                                }
                            }
                        },
                        [](auto&&) { assert(false); }},
               *obj.objects);
}

void GTDatabase::_read_sdc(sdc::CreateClock& obj) {
    assert(obj.period && !obj.name.empty());

    // create clock from given sources
    if (obj.port_pin_list) {
        std::visit(Functors{[&](sdc::GetPorts& get_ports) {
                                auto& ports = get_ports.ports;
                                assert(ports.size() == 1);
                                if (auto itr = pi2idx.find(ports.front()); itr != pi2idx.end()) {
                                    clocks.try_emplace(obj.name, obj.name, pins[itr->second], *obj.period);
                                } else {
                                    printf(obj.command, ": port ", std::quoted(ports.front()), " not found");
                                }
                            },
                            [](auto&&) { assert(false); }},
                   *obj.port_pin_list);
    }
    // create virtual clock
    else {
        clocks.try_emplace(obj.name, obj.name, *obj.period);
    }
}

TimingTorchRawDB::TimingTorchRawDB(torch::Tensor node_lpos_init_,
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
                                   float wire_capacitance_per_micron_) {
    node_lpos_init = node_lpos_init_;
    node_size = node_size_;
    pin_rel_lpos = pin_rel_lpos_;

    node_size_x = node_size.index({"...", 0}).clone().contiguous();
    node_size_y = node_size.index({"...", 1}).clone().contiguous();
    init_x = node_lpos_init.index({"...", 0}).clone().contiguous();
    init_y = node_lpos_init.index({"...", 1}).clone().contiguous();
    pin_offset_x = pin_rel_lpos.index({"...", 0}).clone().contiguous();
    pin_offset_y = pin_rel_lpos.index({"...", 1}).clone().contiguous();
    x = init_x.clone().contiguous();
    y = init_y.clone().contiguous();

    num_nodes = node_size.size(0);
    num_pins = pin_id2node_id_.size(0);
    num_nets = hyperedge_list_end_.size(0);
    num_movable_nodes = num_movable_nodes_;
    net_mask = net_mask_;

    pinAt = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::Device(node_size.device()))).contiguous();
    pinRat =
        torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::Device(node_size.device()))).contiguous();
    at_prefix_pin = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kInt32).device(torch::Device(node_size.device()))).contiguous();
    at_prefix_arc = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kInt32).device(torch::Device(node_size.device()))).contiguous();
    at_prefix_attr = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kInt32).device(torch::Device(node_size.device()))).contiguous();

    flat_node2pin_start_map =
        torch::cat({torch::zeros({1}, torch::dtype(torch::kInt32).device(torch::Device(node_size.device()))), node2pin_list_end_}, 0)
            .to(torch::kInt32)
            .contiguous();
    flat_node2pin_map = node2pin_list_.to(torch::kInt32);
    pin2node_map = pin_id2node_id_.to(torch::kInt32);

    flat_net2pin_start_map =
        torch::cat({torch::zeros({1}, torch::dtype(torch::kInt32).device(torch::Device(node_size.device()))), hyperedge_list_end_}, 0)
            .to(torch::kInt32)
            .contiguous();
    flat_net2pin_map = hyperedge_list_.to(torch::kInt32);
    pin2net_map = pin_id2net_id_.to(torch::kInt32);

    num_threads = std::max(6, 1);
    scale_factor = scale_factor_;
    microns = microns_;
    wire_resistance_per_micron = wire_resistance_per_micron_;
    wire_capacitance_per_micron = wire_capacitance_per_micron_;
}

void TimingTorchRawDB::commit_from(torch::Tensor x_, torch::Tensor y_) {
    // commit external pos to original pos
    init_x.index({torch::indexing::Slice(0, num_movable_nodes)}).data().copy_(x_.index({torch::indexing::Slice(0, num_movable_nodes)}));
    init_y.index({torch::indexing::Slice(0, num_movable_nodes)}).data().copy_(y_.index({torch::indexing::Slice(0, num_movable_nodes)}));
    x.index({torch::indexing::Slice(0, num_movable_nodes)}).data().copy_(x_.index({torch::indexing::Slice(0, num_movable_nodes)}));
    y.index({torch::indexing::Slice(0, num_movable_nodes)}).data().copy_(y_.index({torch::indexing::Slice(0, num_movable_nodes)}));
}

torch::Tensor TimingTorchRawDB::get_curr_cposx() { return x + node_size_x / 2; }
torch::Tensor TimingTorchRawDB::get_curr_cposy() { return y + node_size_y / 2; }
torch::Tensor TimingTorchRawDB::get_curr_lposx() { return x; }
torch::Tensor TimingTorchRawDB::get_curr_lposy() { return y; }

}  // namespace gt