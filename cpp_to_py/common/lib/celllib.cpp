#include "celllib.h"

#include "common/db/Database.h"

namespace gt {

void Celllib::_uncomment(std::vector<char>& buffer) {
    auto fsize = buffer.size() > 0 ? buffer.size() - 1 : 0;

    // Mart out the comment
    for (size_t i = 0; i < fsize; ++i) {
        // Block comment
        if (buffer[i] == '/' && buffer[i + 1] == '*') {
            buffer[i] = buffer[i + 1] = ' ';
            for (i = i + 2; i < fsize; buffer[i++] = ' ') {
                if (buffer[i] == '*' && buffer[i + 1] == '/') {
                    buffer[i] = buffer[i + 1] = ' ';
                    i = i + 1;
                    break;
                }
            }
        }

        // Line comment
        if (buffer[i] == '/' && buffer[i + 1] == '/') {
            buffer[i] = buffer[i + 1] = ' ';
            for (i = i + 2; i < fsize; ++i) {
                if (buffer[i] == '\n' || buffer[i] == '\r') {
                    break;
                } else
                    buffer[i] = ' ';
            }
        }

        // Pond comment
        if (buffer[i] == '#') {
            buffer[i] = ' ';
            for (i = i + 1; i < fsize; ++i) {
                if (buffer[i] == '\n' || buffer[i] == '\r') {
                    break;
                } else
                    buffer[i] = ' ';
            }
        }
    }
}

void Celllib::_tokenize(const std::vector<char>& buf, std::vector<std::string_view>& tokens) {
    static std::string_view dels = "(),:;/#[]{}*\"\\";

    // get the position
    const char* beg = buf.data();
    const char* end = buf.data() + buf.size();

    // Parse the token.
    const char* token{nullptr};
    size_t len{0};

    tokens.clear();

    for (const char* itr = beg; itr != end && *itr != 0; ++itr) {
        // extract the entire quoted string as a token
        bool is_del = (dels.find(*itr) != std::string_view::npos);

        if (std::isspace(*itr) || is_del) {
            if (len > 0) {  // Add the current token.
                tokens.push_back({token, len});
                token = nullptr;
                len = 0;
            }
            // group delimiter is liberty token
            if (*itr == '(' || *itr == ')' || *itr == '{' || *itr == '}') {
                tokens.push_back({itr, 1});
            }
            // extract the entire quoted string (this is buggy now...)
            // else if(*itr == '"') {
            //  for(++itr; itr != end && *itr != '"'; ++itr, ++len) ;
            //  if(len > 0) {
            //    tokens.push_back({itr-len, len});
            //    len = 0;
            //  }
            //}
        } else {
            if (len == 0) {
                token = itr;
            }
            ++len;
        }
    }

    if (len > 0) {
        tokens.push_back({token, len});
    }
}

std::string to_string(DelayModel m) {
    switch (m) {
        case DelayModel::GENERIC_CMOS:
            return "generic_cmos";
            break;

        case DelayModel::TABLE_LOOKUP:
            return "table_lookup";
            break;

        case DelayModel::CMOS2:
            return "cmos2";
            break;

        case DelayModel::PIECEWISE_CMOS:
            return "piecewise_cmos";
            break;

        case DelayModel::DCM:
            return "dcm";
            break;

        case DelayModel::POLYNOMIAL:
            return "polynomial";
            break;

        default:
            return "undefined";
            break;
    }
}

const LutTemplate* Celllib::lut_template(const std::string& name) const {
    if (auto itr = lut_templates.find(name); itr == lut_templates.end()) {
        return nullptr;
    } else {
        return &(itr->second);
    }
}

LutTemplate* Celllib::lut_template(const std::string& name) {
    if (auto itr = lut_templates.find(name); itr == lut_templates.end()) {
        return nullptr;
    } else {
        return &(itr->second);
    }
}

const Cell* Celllib::cell(const std::string& name) const {
    if (auto itr = cells.find(name); itr == cells.end()) {
        return nullptr;
    } else {
        return &(itr->second);
    }
}

Cell* Celllib::cell(const std::string& name) {
    if (auto itr = cells.find(name); itr == cells.end()) {
        return nullptr;
    } else {
        return &(itr->second);
    }
}

std::optional<float> Celllib::_extract_operating_conditions(token_iterator& itr, const token_iterator end) {
    std::optional<float> voltage;
    std::string operating_condition_name;

    if (itr = on_next_parentheses(itr, end, [&](auto& name) mutable { operating_condition_name = name; }); itr == end) {
        logger.info("can't find lut template name");
    }

    // Extract the lut template group
    if (itr = std::find(itr, end, "{"); itr == end) {
        logger.info("can't find lut template group brace '{'");
    }

    // std::cout << lt.name << std::endl;

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

LutTemplate Celllib::_extract_lut_template(token_iterator& itr, const token_iterator end) {
    LutTemplate lt;

    if (itr = on_next_parentheses(itr, end, [&](auto& name) mutable { lt.name = name; }); itr == end) {
        logger.info("can't find lut template name");
    }

    // Extract the lut template group
    if (itr = std::find(itr, end, "{"); itr == end) {
        logger.info("can't find lut template group brace '{'");
    }

    // std::cout << lt.name << std::endl;

    int stack = 1;

    while (stack && ++itr != end) {
        // variable 1
        if (*itr == "variable_1") {  // Read the variable.

            if (++itr == end) {
                logger.info("variable_1 error in lut template %s", lt.name.c_str());
            }

            if (auto vitr = lut_vars.find(*itr); vitr != lut_vars.end()) {
                lt.variable1 = vitr->second;
            } else {
                logger.warning(
                    "unexpected lut template variable %.*s", static_cast<int>((*itr).length()), (*itr).data());
            }
        }
        // variable 2
        else if (*itr == "variable_2") {
            if (++itr == end) {
                logger.info("variable_2 error in lut template %s", lt.name.c_str());
            }

            if (auto vitr = lut_vars.find(*itr); vitr != lut_vars.end()) {
                lt.variable2 = vitr->second;
            } else {
                logger.warning(
                    "unexpected lut template variable %.*s", static_cast<int>((*itr).length()), (*itr).data());
            }
        }
        // index_1
        else if (*itr == "index_1") {
            itr = on_next_parentheses(
                itr, end, [&](auto& str) { lt.indices1.push_back(std::strtof(str.data(), nullptr)); });
        }
        // index_2
        else if (*itr == "index_2") {
            itr = on_next_parentheses(
                itr, end, [&](auto& str) { lt.indices2.push_back(std::strtof(str.data(), nullptr)); });
        } else if (*itr == "}") {
            stack--;
        } else if (*itr == "{") {
            stack++;
        } else {
        }
    }

    if (stack != 0 || *itr != "}") {
        logger.info("can't find lut template group brace '}'");
    }

    return lt;
}

Lut Celllib::_extract_lut(token_iterator& itr, const token_iterator end) {
    Lut lut;

    if (itr = on_next_parentheses(itr, end, [&](auto& name) mutable { lut.name = name; }); itr == end) {
        logger.info("can't find lut template name");
    }

    // Set up the template
    lut.lut_template = lut_template(lut.name);

    // Extract the lut group
    if (itr = std::find(itr, end, "{"); itr == end) {
        logger.info("group brace '{' error in lut ", lut.name);
    }

    int stack = 1;

    size_t size1 = 1;
    size_t size2 = 1;

    while (stack && ++itr != end) {
        if (*itr == "index_1") {
            itr = on_next_parentheses(
                itr, end, [&](auto& v) mutable { lut.indices1.push_back(std::strtof(v.data(), nullptr)); });

            if (lut.indices1.size() == 0) {
                logger.info("syntax error in ", lut.name, " index_1");
            }

            size1 = lut.indices1.size();
        } else if (*itr == "index_2") {
            itr = on_next_parentheses(
                itr, end, [&](auto& v) mutable { lut.indices2.push_back(std::strtof(v.data(), nullptr)); });

            if (lut.indices2.size() == 0) {
                logger.info("syntax error in ", lut.name, " index_2");
            }

            size2 = lut.indices2.size();
        } else if (*itr == "values") {
            if (lut.indices1.empty()) {
                if (size1 != 1) {
                    logger.info("empty indices1 in non-scalar lut ", lut.name);
                }
                lut.indices1.resize(size1);
            }

            if (lut.indices2.empty()) {
                if (size2 != 1) {
                    logger.info("empty indices2 in non-scalar lut ", lut.name);
                }
                lut.indices2.resize(size2);
            }

            lut.table.resize(size1 * size2);

            int id{0};
            itr = on_next_parentheses(
                itr, end, [&](auto& v) mutable { lut.table[id++] = std::strtof(v.data(), nullptr); });
        } else if (*itr == "}") {
            stack--;
        } else if (*itr == "{") {
            stack++;
        } else {
        }
    }

    if (stack != 0 || *itr != "}") {
        logger.info("group brace '}' error in lut ", lut.name);
    }

    return lut;
}

InternalPower Celllib::_extract_internal_power(token_iterator& itr, const token_iterator end) {
    InternalPower power;

    // Extract the lut template group
    if (itr = std::find(itr, end, "{"); itr == end) {
        logger.info("can't find group brace '{' in timing");
    }

    int stack = 1;

    while (stack && ++itr != end) {
        if (*itr == "rise_power") {
            power.rise_power = _extract_lut(itr, end);
        } else if (*itr == "fall_power") {  // Rise delay.
            power.fall_power = _extract_lut(itr, end);
        } else if (*itr == "related_pin") {
            if (++itr == end) {
                logger.info("syntax error in related_pin");
            }

            power.related_pin = *itr;
        } else if (*itr == "}") {
            stack--;
        } else if (*itr == "{") {
            stack++;
        } else {
        }
    }

    if (stack != 0 || *itr != "}") {
        logger.info("can't find group brace '}' in internal_power");
    }

    return power;
}

Timing Celllib::_extract_timing(token_iterator& itr, const token_iterator end, const std::string& cell_name) {
    Timing timing;
    timing.cell_name = cell_name;

    // Extract the lut template group
    if (itr = std::find(itr, end, "{"); itr == end) {
        logger.info("can't find group brace '{' in timing");
    }

    int stack = 1;

    while (stack && ++itr != end) {
        if (*itr == "cell_fall") {
            timing.cell_fall = _extract_lut(itr, end);
        } else if (*itr == "cell_rise") {  // Rise delay.
            timing.cell_rise = _extract_lut(itr, end);
        } else if (*itr == "fall_transition") {  // Fall slew.
            timing.fall_transition = _extract_lut(itr, end);
        } else if (*itr == "rise_transition") {  // Rise slew.
            timing.rise_transition = _extract_lut(itr, end);
        } else if (*itr == "rise_constraint") {  // FF rise constraint.
            timing.rise_constraint = _extract_lut(itr, end);
        } else if (*itr == "fall_constraint") {  // FF fall constraint.
            timing.fall_constraint = _extract_lut(itr, end);
        } else if (*itr == "timing_sense") {  // Read the timing sense.

            logger.infoif(++itr == end, "syntex error in timing_sense");

            if (*itr == "negative_unate") {
                timing.sense = TimingSense::NEGATIVE_UNATE;  // Negative unate.
            } else if (*itr == "positive_unate") {           // Positive unate.
                timing.sense = TimingSense::POSITIVE_UNATE;
            } else if (*itr == "non_unate") {  // Non unate.
                timing.sense = TimingSense::NON_UNATE;
            } else {
                logger.info("unexpected timing sense ", *itr);
            }
        } else if (*itr == "timing_type") {
            if (++itr == end) {
                logger.info("syntax error in timing_type");
            }

            if (auto titr = timing_types.find(*itr); titr != timing_types.end()) {
                timing.type = titr->second;
            } else {
                logger.warning("unexpected timing type ", *itr);
            }
        } else if (*itr == "related_pin") {
            if (++itr == end) {
                logger.info("syntax error in related_pin");
            }

            timing.related_pin = *itr;
        } else if (*itr == "}") {
            stack--;
        } else if (*itr == "{") {
            stack++;
        } else {
        }
    }

    if (stack != 0 || *itr != "}") {
        logger.info("can't find group brace '}' in timing");
    }

    return timing;
}

Cellpin Celllib::_extract_cellpin(token_iterator& itr, const token_iterator end, const std::string& cell_name) {
    Cellpin cellpin;

    if (itr = on_next_parentheses(itr, end, [&](auto& name) mutable { cellpin.name = name; }); itr == end) {
        logger.info("can't find cellpin name");
    }

    // Extract the lut template group
    if (itr = std::find(itr, end, "{"); itr == end) {
        logger.info("can't find group brace '{' in cellpin ", cellpin.name);
    }

    cellpin.cell_name = cell_name;

    // std::cout << "  -->" << cellpin.name << std::endl;

    int stack = 1;

    while (stack && ++itr != end) {
        if (*itr == "direction") {
            if (++itr == end) {
                logger.info("can't get the direction in cellpin ", cellpin.name);
            }

            if (auto ditr = cellpin_directions.find(*itr); ditr != cellpin_directions.end()) {
                cellpin.direction = ditr->second;
            } else {
                logger.warning("unexpected cellpin direction ", *itr);
            }
        } else if (*itr == "capacitance") {
            logger.infoif(++itr == end, "can't get the capacitance in cellpin ", cellpin.name);
            cellpin.capacitance = std::strtof(itr->data(), nullptr);
        } else if (*itr == "max_capacitance") {
            logger.infoif(++itr == end, "can't get the max_capacitance in cellpin ", cellpin.name);
            cellpin.max_capacitance = std::strtof(itr->data(), nullptr);
        } else if (*itr == "min_capacitance") {
            logger.infoif(++itr == end, "can't get the min_capacitance in cellpin ", cellpin.name);
            cellpin.min_capacitance = std::strtof(itr->data(), nullptr);
        } else if (*itr == "max_transition") {
            logger.infoif(++itr == end, "can't get the max_transition in cellpin ", cellpin.name);
            cellpin.max_transition = std::strtof(itr->data(), nullptr);
        } else if (*itr == "min_transition") {
            logger.infoif(++itr == end, "can't get the min_transition in cellpin ", cellpin.name);
            cellpin.min_transition = std::strtof(itr->data(), nullptr);
        } else if (*itr == "fall_capacitance") {
            logger.infoif(++itr == end, "can't get fall_capacitance in cellpin ", cellpin.name);
            cellpin.fall_capacitance = std::strtof(itr->data(), nullptr);
        } else if (*itr == "rise_capacitance") {
            logger.infoif(++itr == end, "can't get rise_capacitance in cellpin ", cellpin.name);
            cellpin.rise_capacitance = std::strtof(itr->data(), nullptr);
        } else if (*itr == "fanout_load") {
            logger.infoif(++itr == end, "can't get fanout_load in cellpin ", cellpin.name);
            cellpin.fanout_load = std::strtof(itr->data(), nullptr);
        } else if (*itr == "max_fanout") {
            logger.infoif(++itr == end, "can't get max_fanout in cellpin ", cellpin.name);
            cellpin.max_fanout = std::strtof(itr->data(), nullptr);
        } else if (*itr == "min_fanout") {
            logger.infoif(++itr == end, "can't get min_fanout in cellpin ", cellpin.name);
            cellpin.min_fanout = std::strtof(itr->data(), nullptr);
        } else if (*itr == "clock") {
            logger.infoif(++itr == end, "can't get the clock status in cellpin ", cellpin.name);
            cellpin.is_clock = (*itr == "true") ? true : false;
        } else if (*itr == "original_pin") {
            logger.infoif(++itr == end, "can't get the original pin in cellpin ", cellpin.name);
            cellpin.original_pin = *itr;
        } else if (*itr == "internal_power") {
            // auto ipower = _extract_internal_power(itr, end);
            // bool found = false;
            // for(auto &t:cellpin.timings) {
            //   if (t.related_pin != ipower.related_pin)
            //     continue;

            //   t.internal_power = ipower;
            //   found = true;
            //   break;
            // }
            // if (!found) {
            //   Timing t;
            //   t.related_pin    = ipower.related_pin;
            //   t.internal_power = ipower;
            //   cellpin.timings.emplace_back(t);
            // }
            // FIXME: currently not support internal power
        } else if (*itr == "timing") {
            cellpin.timings.push_back(_extract_timing(itr, end, cell_name));
            if (cellpin.timings.back().type) {
                // if (cellpin.timings_map.find(*cellpin.timings.back().type) != cellpin.timings_map.end()) {
                //   logger.warning("duplicated timing table cellname: %s, pinname: %s, type: %s", cell_name.c_str(),
                //   cellpin.name.c_str(), to_string(*cellpin.timings.back().type).c_str());
                // }
                std::string related_pin = cellpin.timings.back().related_pin;
                if (related_pin.empty()) continue;
                cellpin.timings_map[related_pin][*cellpin.timings.back().type] = cellpin.timings.size() - 1;
                // cellpin.timings_map[*cellpin.timings.back().type] = cellpin.timings.back();
            }
        } else if (*itr == "}") {
            stack--;
        } else if (*itr == "{") {
            stack++;
        } else {
        }
    }

    if (stack != 0 || *itr != "}") {
        logger.info("can't find group brace '}' in cellpin ", cellpin.name);
    }

    return cellpin;
}

Cell Celllib::_extract_cell(token_iterator& itr, const token_iterator end) {
    Cell cell;

    if (itr = on_next_parentheses(itr, end, [&](auto& name) mutable { cell.name = name; }); itr == end) {
        logger.info("can't find cell name");
    }

    // Extract the lut template group
    if (itr = std::find(itr, end, "{"); itr == end) {
        logger.info("can't find group brace '{' in cell ", cell.name);
    }

    int stack = 1;
    unordered_map<string, CellpinDirection> members_direction;
    // vector<string> members;
    queue<string> members;
    while (stack && ++itr != end) {
        if (*itr == "cell_leakage_power") {  // Read the leakage power.
            logger.infoif(++itr == end, "can't get leakage power in cell ", cell.name);
            cell.leakage_power = power_scale * std::strtof(itr->data(), nullptr);
        }
        if (*itr == "leakage_power") {  // Read the leakage power.
            if (itr = std::find(itr, end, "{"); itr == end) logger.info("can't find group brace '{' in timing");
            float value;
            int stack = 1;
            while (stack && ++itr != end) {
                if (*itr == "value") {
                    if (++itr == end) {
                        logger.info("can't get the leakage power value in cell ", cell.name);
                    }
                    value = std::strtof(itr->data(), nullptr);
                } else if (*itr == "related_pg_pin") {
                    if (++itr == end) {
                        logger.info("can't get the related_pg_pin in cell ", cell.name);
                    }
                    string related_pg_pin = string(*itr);
                    if (cell.primary_power && cell.primary_power == related_pg_pin) {
                        cell.leakage_power = power_scale * value;
                    }
                } else if (*itr == "}") {
                    stack--;
                } else if (*itr == "{") {
                    stack++;
                } else {
                }
            }
        } else if (*itr == "cell_footprint") {  // Read the footprint.
            logger.infoif(++itr == end, "can't get footprint in cell ", cell.name);
            cell.cell_footprint = *itr;
        } else if (*itr == "area") {  // Read the area.
            logger.infoif(++itr == end, "can't get area in cell ", cell.name);
            cell.area = std::strtof(itr->data(), nullptr);
        } else if (*itr == "pg_pin") {  // Read power
            std::string pg_name;
            itr = on_next_parentheses(itr, end, [&](auto& name) mutable { pg_name = name; });
            if (itr = std::find(itr, end, "{"); itr == end) logger.info("can't find group brace '{' in timing");
            int stack = 1;
            while (stack && ++itr != end) {
                if (*itr == "voltage_name") {
                    // TODO:
                } else if (*itr == "pg_type") {
                    if (++itr == end) {
                        logger.info("can't get the pg_type in cell pg_pin ");
                    }
                    if (*itr == "primary_power") {
                        cell.primary_power = pg_name;
                    } else if (*itr == "primary_ground") {
                        cell.primary_ground = pg_name;
                    }
                } else if (*itr == "}") {
                    stack--;
                } else if (*itr == "{") {
                    stack++;
                } else {
                }
            }
        } else if (*itr == "pin") {  // Read the cell pin group.
            auto pin = _extract_cellpin(itr, end, cell.name);
            string pin_name = pin.name;
            cell.cellpins[pin_name] = std::move(pin);
            if (members_direction.find(pin_name) != members_direction.end()) {
                cell.cellpins[pin_name].direction = members_direction[pin_name];
            }
            if (cell.cellpins[pin_name].is_clock) {
                if (*cell.cellpins[pin_name].is_clock) cell.num_bits = cell.num_bits == 0 ? 1 : cell.num_bits;
            }
        } else if (*itr == "members") {
            itr = on_next_parentheses(itr, end, [&](auto& name) mutable { members.push(std::string{name}); });
            cell.num_bits = members.size();
        } else if (*itr == "direction") {
            if (++itr == end) {
                logger.info("can't get the direction in cellpin");
            }
            if (auto ditr = cellpin_directions.find(*itr); ditr != cellpin_directions.end()) {
                while (!members.empty()) {
                    members_direction[members.front()] = ditr->second;
                    members.pop();
                }
            } else {
                logger.warning("unexpected cellpin direction ");
            }
        } else if (*itr == "}") {
            stack--;
        } else if (*itr == "{") {
            stack++;
        } else {
            // logger.warning("unexpected token %.*s", static_cast<int>((*itr).length()), (*itr).data());
        }
    }

    if (stack != 0 || *itr != "}") {
        logger.info("can't find group brace '}' in cell ", cell.name);
    }

    return cell;
}

#include <zlib.h>

void Celllib::read(const std::string& file) {
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

    _uncomment(buffer);
    _tokenize(buffer, tokens);

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

    // Extract the library group
    if (itr = std::find(itr, tokens.end(), "{"); itr == tokens.end()) {
        logger.info("can't find library group symbol '{'");
    }

    int stack = 1;

    while (stack && ++itr != end) {
        if (*itr == "lu_table_template") {
            auto lut = _extract_lut_template(itr, end);
            lut_templates[lut.name] = lut;
        } else if (*itr == "power_lut_template") {
            auto lut = _extract_lut_template(itr, end);
            lut_templates[lut.name] = lut;
        } else if (*itr == "delay_model") {
            logger.infoif(++itr == end, "syntax error in delay_model");
            if (auto ditr = delay_models.find(*itr); ditr != delay_models.end()) {
                delay_model = ditr->second;
            } else {
                logger.warning("unexpected delay model ", *itr);
            }
        } else if (*itr == "default_cell_leakage_power") {
            logger.infoif(++itr == end, "syntax error in default_cell_leakage_power");
            default_cell_leakage_power = std::strtof(itr->data(), nullptr);
        } else if (*itr == "default_inout_pin_cap") {
            logger.infoif(++itr == end, "syntax error in default_inout_pin_cap");
            default_inout_pin_cap = std::strtof(itr->data(), nullptr);
        } else if (*itr == "default_input_pin_cap") {
            logger.infoif(++itr == end, "syntax error in default_input_pin_cap");
            default_input_pin_cap = std::strtof(itr->data(), nullptr);
        } else if (*itr == "default_output_pin_cap") {
            logger.infoif(++itr == end, "syntax error in default_output_pin_cap");
            default_output_pin_cap = std::strtof(itr->data(), nullptr);
        } else if (*itr == "default_fanout_load") {
            logger.infoif(++itr == end, "syntax error in default_fanout_load");
            default_fanout_load = std::strtof(itr->data(), nullptr);
        } else if (*itr == "default_max_fanout") {
            logger.infoif(++itr == end, "syntax error in default_max_fanout");
            default_max_fanout = std::strtof(itr->data(), nullptr);
        } else if (*itr == "default_max_transition") {
            logger.infoif(++itr == end, "syntax error in default_max_transition");
            default_max_transition = std::strtof(itr->data(), nullptr);
        } else if (*itr == "operating_conditions") {
            logger.infoif(++itr == end, "syntax error in operating_conditions");
            voltage = _extract_operating_conditions(itr, end);
            // TODO: Unit field.
        } else if (*itr == "time_unit") {
            logger.infoif(++itr == end, "time_unit syntax error");
            time_unit = make_time_unit(*itr);
        } else if (*itr == "voltage_unit") {
            logger.infoif(++itr == end, "voltage_unit syntax error");
            voltage_unit = make_voltage_unit(*itr);
        } else if (*itr == "current_unit") {
            logger.infoif(++itr == end, "current_unit syntax error");
            current_unit = make_current_unit(*itr);
        } else if (*itr == "pulling_resistance_unit") {
            logger.infoif(++itr == end, "pulling_resistance_unit syntax error");
            resistance_unit = make_resistance_unit(*itr);
        } else if (*itr == "leakage_power_unit") {
            logger.infoif(++itr == end, "leakage_power_unit syntax error");
            // power_unit = make_power_unit(*itr);
            current_power_unit = make_power_unit(*itr);
            if (!power_unit) power_unit = current_power_unit;
            power_scale = *current_power_unit / *power_unit;
        } else if (*itr == "capacitive_load_unit") {
            std::string unit;
            if (itr = on_next_parentheses(itr, end, [&](auto& str) mutable { unit += str; }); itr == end) {
                logger.info("capacitive_load_unit syntax error");
            }
            capacitance_unit = make_capacitance_unit(unit);
        } else if (*itr == "cell") {
            auto cell = _extract_cell(itr, end);
            string cell_name = cell.name;
            cells[cell_name] = std::move(cell);
            if (rawdb) {
                auto celltype = rawdb->getCellType(cell_name);
                celltype->libCell = &cells[cell_name];
            }
        } else if (*itr == "}") {
            stack--;
        } else if (*itr == "{") {
            stack++;
        } else {
        }
    }

    if (stack != 0 || *itr != "}") {
        logger.info("can't find library group brace '}'");
    }

    _apply_default_values();
}

void Celllib::_apply_default_values() {
    for (auto& ckvp : cells) {
        auto& cell = ckvp.second;

        // apply the default leakage power
        if (!cell.leakage_power) cell.leakage_power = default_cell_leakage_power;

        for (auto& pkvp : cell.cellpins) {
            auto& cpin = pkvp.second;

            // direction-specific default values
            if (!cpin.direction) {
                logger.warning("cellpin %s:%s has no direction defined", cell.name.c_str(), cpin.name.c_str());
                continue;
            }

            switch (*cpin.direction) {
                case CellpinDirection::INPUT:
                    if (!cpin.capacitance) {
                        cpin.capacitance = default_input_pin_cap;
                    }

                    if (!cpin.fanout_load) {
                        cpin.fanout_load = default_fanout_load;
                    }
                    break;

                case CellpinDirection::OUTPUT:
                    if (!cpin.capacitance) {
                        cpin.capacitance = default_output_pin_cap;
                    }

                    if (!cpin.max_fanout) {
                        cpin.max_fanout = default_max_fanout;
                    }

                    if (!cpin.max_transition) {
                        cpin.max_transition = default_max_transition;
                    }
                    break;

                case CellpinDirection::INOUT:
                    if (!cpin.capacitance) {
                        cpin.capacitance = default_inout_pin_cap;
                    }
                    break;

                case CellpinDirection::INTERNAL:
                    break;
            }
        }
    }
}

// Convert the numerics to the new unit
void Celllib::scale_time(float s) {
    if (default_max_transition) {
        default_max_transition = *default_max_transition * s;
    }

    for (auto& c : cells) {
        c.second.scale_time(s);
    }
}

void Celllib::scale_capacitance(float s) {
    if (default_inout_pin_cap) {
        default_inout_pin_cap = *default_inout_pin_cap * s;
    }

    if (default_input_pin_cap) {
        default_input_pin_cap = *default_input_pin_cap * s;
    }

    if (default_output_pin_cap) {
        default_output_pin_cap = *default_output_pin_cap * s;
    }

    for (auto& c : cells) {
        c.second.scale_capacitance(s);
    }
}

void Celllib::scale_voltage(float s) {
    // TODO
}

void Celllib::scale_current(float s) {
    // TODO
}

void Celllib::scale_resistance(float s) {
    // TODO
}

void Celllib::scale_power(float s) {
    // TODO
}

std::ostream& operator<<(std::ostream& os, const Celllib& c) {
    // Write the comment.
    os << "/* Generated by OpenTimer "
       << " */\n";

    // Write library name.
    os << "library (\"" << c.name << "\") {\n\n";

    // Delay modeA
    if (c.delay_model) {
        os << "delay_model : " << to_string(*(c.delay_model)) << ";\n";
    }

    // Library units
    if (auto u = c.time_unit; u) {
        os << "time_unit : \"" << u->value() << "s\"\n";
    }

    if (auto u = c.voltage_unit; u) {
        os << "voltage_unit : \"" << u->value() << "V\"\n";
    }

    if (auto u = c.current_unit; u) {
        os << "current_unit : \"" << u->value() << "A\"\n";
    }

    if (auto u = c.resistance_unit; u) {
        os << "pulling_resistance_unit : \"" << u->value() << "ohm\"\n";
    }

    if (auto u = c.power_unit; u) {
        os << "leakage_power_unit : \"" << u->value() << "W\"\n";
    }

    if (auto u = c.capacitance_unit; u) {
        os << "capacitive_load_unit (" << u->value() << ",F)\"\n";
    }

    // default values
    if (c.default_cell_leakage_power) {
        os << *c.default_cell_leakage_power << '\n';
    }

    if (c.default_inout_pin_cap) {
        os << *c.default_inout_pin_cap << '\n';
    }

    if (c.default_input_pin_cap) {
        os << *c.default_input_pin_cap << '\n';
    }

    if (c.default_output_pin_cap) {
        os << *c.default_fanout_load << '\n';
    }

    if (c.default_max_fanout) {
        os << *c.default_max_fanout << '\n';
    }

    if (c.default_max_transition) {
        os << *c.default_max_transition << '\n';
    }

    // Write the lut templates
    for (const auto& kvp : c.lut_templates) {
        os << kvp.second << '\n';
    }

    // Write all cells.
    for (const auto& kvp : c.cells) {
        os << kvp.second << '\n';
    }

    // Write library ending group symbol.
    os << "}\n";

    return os;
}

};  // namespace gt
