#include <zlib.h>
#include <iostream>

#include "Liberty.h"
#include "Timing.h"
#include "lut.h"

using std::ifstream;

namespace gt {

// const LutTemplate* CellLib::lut_template(const std::string& name) const {
//     if (auto itr = lut_templates_.find(name); itr == lut_templates_.end()) {
//         return nullptr;
//     } else {
//         return itr->second;
//     }
// }

// LutTemplate* CellLib::lut_template(const std::string& name) {
//     if (auto itr = lut_templates_.find(name); itr == lut_templates_.end()) {
//         return nullptr;
//     } else {
//         return itr->second;
//     }
// }

// const Cell* CellLib::cell(const std::string& name) const {
//     if (auto itr = cells.find(name); itr == cells.end()) {
//         return nullptr;
//     } else {
//         return itr->second;
//     }
// }

// Cell* CellLib::cell(const std::string& name) {
//     if (auto itr = cells.find(name); itr == cells.end()) {
//         return nullptr;
//     } else {
//         return itr->second;
//     }
// }

std::optional<float> CellLib::extract_operating_conditions(token_iterator& itr, const token_iterator end) {
    std::optional<float> voltage;
    std::string operating_condition_name;
    if (itr = on_next_parentheses(itr, end, [&](auto& name) mutable { operating_condition_name = name; }); itr == end) {
        logger.info("can't find lut template name");
    }
    // Extract the lut template group
    if (itr = std::find(itr, end, "{"); itr == end) {
        logger.info("can't find lut template group brace '{'");
    }

    int stack = 1;

    while (stack && ++itr != end) {
        // variable 1
        if (*itr == "voltage") {  // Read the variable.

            if (++itr == end) {
                logger.info("volate error in operating_conditions template ", operating_condition_name);
            }

            voltage = std::strtof(std::string(*itr).c_str(), nullptr);
        } else if (*itr == "}") {
            stack--;
        } else if (*itr == "{") {
            stack++;
        } else {
        }
    }

    if (stack != 0 || *itr != "}") {
        logger.info("can't find operating_conditions template group brace '}'");
    }

    return voltage;
}

LutTemplate* CellLib::extract_lut_template(token_iterator& itr, const token_iterator end) {
    LutTemplate* lt = new LutTemplate();

    if (itr = on_next_parentheses(itr, end, [&](auto& name) mutable { lt->name = name; }); itr == end) {
        logger.info("can't find lut template name");
    }

    if (itr = std::find(itr, end, "{"); itr == end) {
        logger.info("can't find lut template group brace '{'");
    }

    int stack = 1;
    while (stack && ++itr != end) {
        if (*itr == "variable_1") {
            if (++itr == end) {
                logger.info("variable_1 error in lut template %s", lt->name.c_str());
            }

            if (auto vitr = lut_vars.find(*itr); vitr != lut_vars.end()) {
                lt->variable1 = vitr->second;
            } else {
                logger.warning(
                    "unexpected lut template variable %.*s", static_cast<int>((*itr).length()), (*itr).data());
            }
        } else if (*itr == "variable_2") {
            if (++itr == end) {
                logger.info("variable_2 error in lut template %s", lt->name.c_str());
            }
            if (auto vitr = lut_vars.find(*itr); vitr != lut_vars.end()) {
                lt->variable2 = vitr->second;
            } else {
                logger.warning(
                    "unexpected lut template variable %.*s", static_cast<int>((*itr).length()), (*itr).data());
            }
        } else if (*itr == "index_1") {
            itr = on_next_parentheses(
                itr, end, [&](auto& str) { lt->indices1.push_back(std::strtof(str.data(), nullptr)); });
        } else if (*itr == "index_2") {
            itr = on_next_parentheses(
                itr, end, [&](auto& str) { lt->indices2.push_back(std::strtof(str.data(), nullptr)); });
        } else if (*itr == "}") {
            stack--;
        } else if (*itr == "{") {
            stack++;
        } else {
        }
    }

    lut_templates_[lt->name] = lt;

    return lt;
}

Lut* CellLib::extract_lut(token_iterator& itr, const token_iterator end) {
    Lut* lut_ = new Lut();

    if (itr = on_next_parentheses(itr, end, [&](auto& name) mutable { lut_->name = name; }); itr == end) {
        logger.info("can't find lut template name");
    }

    // Set up the template
    lut_->lut_template = lut_template(lut_->name);

    // Extract the lut group
    if (itr = std::find(itr, end, "{"); itr == end) {
        logger.info("group brace '{' error in lut ", lut_->name);
    }

    int stack = 1;

    size_t size1 = 1;
    size_t size2 = 1;

    while (stack && ++itr != end) {
        if (*itr == "index_1") {
            itr = on_next_parentheses(
                itr, end, [&](auto& v) mutable { lut_->indices1.push_back(std::strtof(v.data(), nullptr)); });

            if (lut_->indices1.size() == 0) {
                logger.info("syntax error in ", lut_->name, " index_1");
            }

            size1 = lut_->indices1.size();
        } else if (*itr == "index_2") {
            itr = on_next_parentheses(
                itr, end, [&](auto& v) mutable { lut_->indices2.push_back(std::strtof(v.data(), nullptr)); });

            if (lut_->indices2.size() == 0) {
                logger.info("syntax error in ", lut_->name, " index_2");
            }

            size2 = lut_->indices2.size();
        } else if (*itr == "values") {
            if (lut_->indices1.empty()) {
                if (size1 != 1) {
                    logger.info("empty indices1 in non-scalar lut ", lut_->name);
                }
                lut_->indices1.resize(size1);
            }

            if (lut_->indices2.empty()) {
                if (size2 != 1) {
                    logger.info("empty indices2 in non-scalar lut ", lut_->name);
                }
                lut_->indices2.resize(size2);
            }

            lut_->table.resize(size1 * size2);

            int id{0};
            itr = on_next_parentheses(
                itr, end, [&](auto& v) mutable { lut_->table[id++] = std::strtof(v.data(), nullptr); });
        } else if (*itr == "}") {
            stack--;
        } else if (*itr == "{") {
            stack++;
        } else {
        }
    }

    lut_->set_ = true;

    return lut_;
}

TimingArc* CellLib::extractTimingArc(token_iterator& itr, const token_iterator end, LibertyPort* cell_port_) {
    TimingArc* timing_arc_ = new TimingArc();
    timing_arc_->liberty_port_ = cell_port_;
    cell_port_->timing_arcs_.push_back(timing_arc_);

    std::find(itr, end, "{");
    int stack = 1;
    while (stack && ++itr != end) {
        if (*itr == "cell_fall") {
            timing_arc_->cell_delay_[1] = extract_lut(itr, end);
        } else if (*itr == "cell_rise") {
            timing_arc_->cell_delay_[0] = extract_lut(itr, end);
        } else if (*itr == "fall_transition") {
            timing_arc_->transition_[1] = extract_lut(itr, end);
        } else if (*itr == "rise_transition") {
            timing_arc_->transition_[0] = extract_lut(itr, end);
        } else if (*itr == "fall_constraint") {
            timing_arc_->constraint_[1] = extract_lut(itr, end);
        } else if (*itr == "rise_constraint") {
            timing_arc_->constraint_[0] = extract_lut(itr, end);
        } else if (*itr == "timing_sense") {
            ++itr;
            timing_arc_->timing_sense_ = findTimingSense(string(*itr));
        } else if (*itr == "timing_type") {
            ++itr;
            timing_arc_->timing_type_ = findTimingType(string(*itr));
        } else if (*itr == "related_pin") {
            ++itr;
            timing_arc_->related_pin_name_ = *itr;
        } else if (*itr == "}") {
            stack--;
        } else if (*itr == "{") {
            stack++;
        } else {
        }
    }

    return timing_arc_;
}

LibertyPort* CellLib::extractLibertyPort(token_iterator& itr, const token_iterator end, LibertyCell* cell) {
    LibertyPort* cell_port_ = new LibertyPort();
    cell_port_->cell_ = cell;

    on_next_parentheses(itr, end, [&](auto& name) mutable { cell_port_->name = name; });
    std::find(itr, end, "{");

    int stack = 1;
    while (stack && ++itr != end) {
        if (*itr == "direction") {
            ++itr;
            cell_port_->direction_ = findPortDirection(string(*itr));
        } else if (*itr == "capacitance") {
            logger.infoif(++itr == end, "can't get the capacitance in cellpin");
            cell_port_->port_capacitance_[2] = std::strtof(itr->data(), nullptr);
        } else if (*itr == "fall_capacitance") {
            logger.infoif(++itr == end, "can't get fall_capacitance in cellpin");
            cell_port_->port_capacitance_[1] = std::strtof(itr->data(), nullptr);
        } else if (*itr == "rise_capacitance") {
            logger.infoif(++itr == end, "can't get rise_capacitance in cellpin");
            cell_port_->port_capacitance_[0] = std::strtof(itr->data(), nullptr);
        } else if (*itr == "max_capacitance") {
            logger.infoif(++itr == end, "can't get the max_capacitance in cellpin");
            cell_port_->max_capacitance = std::strtof(itr->data(), nullptr);
        } else if (*itr == "min_capacitance") {
            logger.infoif(++itr == end, "can't get the min_capacitance in cellpin");
            cell_port_->min_capacitance = std::strtof(itr->data(), nullptr);
        } else if (*itr == "max_transition") {
            logger.infoif(++itr == end, "can't get the max_transition in cellpin");
            cell_port_->max_transition = std::strtof(itr->data(), nullptr);
        } else if (*itr == "min_transition") {
            logger.infoif(++itr == end, "can't get the min_transition in cellpin");
            cell_port_->min_transition = std::strtof(itr->data(), nullptr);
        } else if (*itr == "fanout_load") {
            logger.infoif(++itr == end, "can't get fanout_load in cellpin");
            cell_port_->fanout_load = std::strtof(itr->data(), nullptr);
        } else if (*itr == "max_fanout") {
            logger.infoif(++itr == end, "can't get max_fanout in cellpin");
            cell_port_->max_fanout = std::strtof(itr->data(), nullptr);
        } else if (*itr == "min_fanout") {
            logger.infoif(++itr == end, "can't get min_fanout in cellpin");
            cell_port_->min_fanout = std::strtof(itr->data(), nullptr);
        } else if (*itr == "clock") {
            logger.infoif(++itr == end, "can't get the clock status in cellpin");
            cell_port_->is_clock = (*itr == "true") ? true : false;
        } else if (*itr == "timing") {
            TimingArc* timing_arc_ = extractTimingArc(itr, end, cell_port_);
            // if (cellpin.timings.back().type) {
            //     std::string related_pin = cellpin.timings.back().related_pin;
            //     if (related_pin.empty()) continue;
            //     cellpin.timings_map[related_pin][*cellpin.timings.back().type] = cellpin.timings.size() - 1;
            // }
        } else if (*itr == "}") {
            stack--;
        } else if (*itr == "{") {
            stack++;
        } else {
        }
    }
    return cell_port_;
}

LibertyCell* CellLib::extractLibertyCell(token_iterator& itr, const token_iterator end) {
    LibertyCell* liberty_cell_ = new LibertyCell();

    on_next_parentheses(itr, end, [&](auto& name) mutable { liberty_cell_->name = name; });
    itr = std::find(itr, end, "{");

    int stack = 1;
    while (stack && ++itr != end) {
        if (*itr == "cell_leakage_power") {
            ++itr;
            liberty_cell_->leakage_power_ = scale_factors["power"] * std::strtof(itr->data(), nullptr);
        }
        if (*itr == "leakage_power") {
            itr = std::find(itr, end, "{");
            int stack_1 = 1;
            while (stack_1 && ++itr != end) {
                if (*itr == "value") {
                    liberty_cell_->leakage_powers_.push_back(std::strtof(itr->data(), nullptr));
                } else if (*itr == "}")
                    stack_1--;
                else if (*itr == "{")
                    stack_1++;
            }
        } else if (*itr == "area") {
            liberty_cell_->area_ = std::strtof(itr->data(), nullptr);
        } else if (*itr == "pin") {
            LibertyPort* cell_port_ = extractLibertyPort(itr, end, liberty_cell_);
        } else if (*itr == "bundle") {
            LibertyPort* cell_port_bundle_ = new LibertyPort();
            cell_port_bundle_->cell_ = liberty_cell_;
            cell_port_bundle_->is_bundle_ = true;
            on_next_parentheses(itr, end, [&](auto& name) mutable { cell_port_bundle_->name = name; });
            itr = std::find(itr, end, "{");
            int stack_1 = 1;
            while (stack_1 && ++itr != end) {
                if (*itr == "direction") {
                    ++itr;
                    cell_port_bundle_->direction_ = findPortDirection(string(*itr));
                } else if (*itr == "pin") {
                    LibertyPort* cell_port_ = extractLibertyPort(itr, end, liberty_cell_);
                    cell_port_bundle_->member_ports_.push_back(cell_port_);
                } else if (*itr == "}")
                    stack_1--;
                else if (*itr == "{")
                    stack_1++;
            }
        } else if (*itr == "}") {
            stack--;
        } else if (*itr == "{") {
            stack++;
        } else {
        }
    }
    return liberty_cell_;
}

void CellLib::read(const std::string& file) {
    // process .gz file with zlib
    std::vector<char> buffer;
    if (file.substr(file.find_last_of(".") + 1) == "gz") {
        logger.info("reading gzip celllib %s ...", file.c_str());
        gzFile fs = gzopen(file.c_str(), "rb");
        if (!fs) {
            logger.error("cannot open verilog file: %s", file.c_str());
        }
        char buf[1024];
        int len = 0;
        while ((len = gzread(fs, buf, 1024)) > 0) {
            buffer.insert(buffer.end(), buf, buf + len);
        }
        gzclose(fs);
        buffer.push_back(0);
    } else {
        ifstream fs(file.c_str(), std::ios::ate);
        if (!fs.good()) {
            logger.error("cannot open liberty file: %s", file.c_str());
        }
        logger.info("reading celllib %s ...", file.c_str());

        size_t fsize = fs.tellg();
        fs.seekg(0, std::ios::beg);
        buffer.resize(fsize + 1);
        fs.read(buffer.data(), fsize);
        buffer[fsize] = 0;
    }

    // get tokens
    std::vector<std::string_view> tokens;
    tokens.reserve(buffer.size() / sizeof(std::string));

    uncomment(buffer);
    tokenize(buffer, tokens);

    // Set up the iterator
    auto itr = tokens.begin();
    auto end = tokens.end();

    // Read the library name.
    if (itr = std::find(itr, end, "library"); itr == end) {
        logger.error("can't find keyword %s", "library");
    }

    if (itr = on_next_parentheses(itr, end, [&](auto& str) mutable { name = str; }); itr == end) {
        logger.info("can't find library name");
    }

    if (itr = std::find(itr, tokens.end(), "{"); itr == tokens.end()) {
        logger.info("can't find library group symbol '{'");
    }

    int stack = 1;

    while (stack && ++itr != end) {
        if (*itr == "lu_table_template") {
            auto lut = extract_lut_template(itr, end);
        } else if (*itr == "power_lut_template") {
            auto lut = extract_lut_template(itr, end);
        } else if (*itr == "delay_model") {
            logger.infoif(++itr == end, "syntax error in delay_model");
            delay_model = findDelayModel(string(*itr));
        } else if (*itr == "default_cell_leakage_power" || *itr == "default_inout_pin_cap" ||
                   *itr == "default_input_pin_cap" || *itr == "default_output_pin_cap" ||
                   *itr == "default_fanout_load" || *itr == "default_max_fanout" || *itr == "default_max_transition") {
            logger.infoif(++itr == end, "syntax error");
            default_values[std::string(*itr)] = std::strtof(itr->data(), nullptr);
        } else if (*itr == "operating_conditions") {
            logger.infoif(++itr == end, "syntax error");
            default_values["voltage"] = extract_operating_conditions(itr, end);
        } else if (*itr == "time_unit") {
            logger.infoif(++itr == end, "syntax error");
            time_unit_ = make_time_unit(*itr);
        } else if (*itr == "voltage_unit") {
            logger.infoif(++itr == end, "syntax error");
            voltage_unit_ = make_voltage_unit(*itr);
        } else if (*itr == "current_unit") {
            logger.infoif(++itr == end, "syntax error");
            current_unit_ = make_current_unit(*itr);
        } else if (*itr == "pulling_resistance_unit") {
            logger.infoif(++itr == end, "syntax error");
            resistance_unit_ = make_resistance_unit(*itr);
        } else if (*itr == "capacitive_load_unit") {
            string unit;
            on_next_parentheses(itr, end, [&](auto& str) mutable { unit += str; });
            capacitance_unit_ = make_capacitance_unit(unit);
        } else if (*itr == "leakage_power_unit") {
            logger.infoif(++itr == end, "syntax error");
            auto current_power_unit_ = make_power_unit(*itr);
            if (!power_unit_)
                power_unit_ = current_power_unit_;
            scale_factors["power"] = *current_power_unit_ / *power_unit_;
        } else if (*itr == "cell") {
            LibertyCell* libterty_cell_ = extractLibertyCell(itr, end);
            lib_cells_[libterty_cell_->name] = libterty_cell_;
            // string cell_name = cell.name;
            // cells[cell_name] = std::move(cell);
            // if (rawdb) {
            //     auto celltype = rawdb->getCellType(cell_name);
            //     celltype->libCell = &cells[cell_name];
            // }
        } else if (*itr == "}") {
            stack--;
        } else if (*itr == "{") {
            stack++;
        } else {
        }
    }

    // apply_default_values();
}

// void CellLib::apply_default_values() {
//     for (auto& ckvp : cells) {
//         auto& cell = ckvp.second;

//         // apply the default leakage power
//         if (!cell.leakage_power) cell.leakage_power = default_cell_leakage_power;

//         for (auto& pkvp : cell.cellpins) {
//             auto& cpin = pkvp.second;

//             // direction-specific default values
//             if (!cpin.direction) {
//                 logger.warning("cellpin %s:%s has no direction defined", cell.name.c_str(), cpin.name.c_str());
//                 continue;
//             }

//             switch (*cpin.direction) {
//                 case CellpinDirection::INPUT:
//                     if (!cpin.capacitance) {
//                         cpin.capacitance = operating_conditions["default_input_pin_cap"];
//                     }

//                     if (!cpin.fanout_load) {
//                         cpin.fanout_load = operating_conditions["default_fanout_load"];
//                     }
//                     break;

//                 case CellpinDirection::OUTPUT:
//                     if (!cpin.capacitance) {
//                         cpin.capacitance = operating_conditions["default_output_pin_cap"];
//                     }

//                     if (!cpin.max_fanout) {
//                         cpin.max_fanout = operating_conditions["default_max_fanout"];
//                     }

//                     if (!cpin.max_transition) {
//                         cpin.max_transition = operating_conditions["default_max_transition"];
//                     }
//                     break;

//                 case CellpinDirection::INOUT:
//                     if (!cpin.capacitance) {
//                         cpin.capacitance = operating_conditions["default_inout_pin_cap"];
//                     }
//                     break;

//                 case CellpinDirection::INTERNAL:
//                     break;
//             }
//         }
//     }
// }

};  // namespace gt
