#include "common/db/Database.h"
#include "verilog/verilog_driver.hpp"

namespace db {

// helper type for the visitor #4
template <class... Ts>
struct overloaded : Ts... {
    using Ts::operator()...;
};
// explicit deduction guide (not needed as of C++20)
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

class VerilogParser : public verilog::ParserVerilogInterface {
public:
    Database &db;
    bool verilog_only;
    VerilogParser() = default;
    VerilogParser(Database *db, bool verilog_only) : db(*db), verilog_only(verilog_only) {}
    virtual ~VerilogParser() {}
    vector<verilog::Port> ports;

    void add_module(std::string &&name) { 
        std::cout << "Module name = " << name << '\n'; 
        db.module_name = name;
    }

    void add_port(verilog::Port &&port) {
        ports.push_back(port);
        // std::cout << "Port: " << port << '\n';
        auto addIOPin = [&](const std::string &iopinname, const char direction) {
            IOPin *iopin;
            if (!verilog_only) {
                iopin = db.getIOPin(iopinname);
                if (!iopin) {
                    logger.warning("IO pin is not defined: %s", iopinname.c_str());
                    return;
                }
            } else {
                iopin = db.addIOPin(iopinname, iopinname, direction);
            }

            Pin *pin = iopin->pin;
            iopin->is_connected = true;
            Net *net = db.addNet(iopinname);
            net->is_port = true;
            // std::cout << "Net: " << iopinname << '\n';

            pin->net = net;
            pin->is_connected = true;
            net->addPin(iopin->pin);
        };
        for (auto &name : port.names) {
            if (port.dir == verilog::PortDirection::INPUT || port.dir == verilog::PortDirection::OUTPUT) {
                char direction = 'x';
                direction = (port.dir == verilog::PortDirection::INPUT) ? 'o' : 'i';
                if ((port.beg != -1) && (port.end != -1)) {
                    for (int i = port.beg; i >= port.end; i--) {
                        // INPUT to the chip, output from external
                        // OUTPUT to the chip, input to external
                        std::string iopinname(name + "[" + std::to_string(i) + "]");
                        addIOPin(iopinname, direction);
                    }
                } else {
                    std::string iopinname(name);
                    addIOPin(iopinname, direction);
                }
            }
        }
    }

    void add_net(verilog::Net &&net) {
        // std::cout << "Net: " << net << '\n';
        auto addNet = [&](const std::string &netName) { db.addNet(netName); };
        for (auto &name : net.names) {
            if ((net.beg != -1) && (net.end != -1)) {
                for (int i = net.beg; i >= net.end; i--) {
                    std::string netName(name + "[" + std::to_string(i) + "]");
                    netName = validate_token(netName);
                    addNet(netName);
                }
            } else {
                std::string netName(name);
                netName = validate_token(netName);
                addNet(netName);
            }
        }
    }

    void add_assignment(verilog::Assignment &&ast) {
        // std::cout << "Assignment: " << ast << '\n';
    }

    void add_instance(verilog::Instance &&inst) {
        // std::cout << "Instance: " << inst << '\n';
        // remove '\' in the head
        string cellName(inst.inst_name);
        cellName = validate_token(cellName);
        Cell *cell;
        if (!verilog_only) {
            cell = db.getCell(cellName);
            if (!cell) {
                logger.error("Cell is not defined: %s", cellName.c_str());
                return;
            }
        } else {
            string macroName(inst.module_name);

            CellType *celltype = db.getCellType(macroName);
            if (!celltype) {
                celltype = db.addCellType(macroName, db.celltypes.size());
                for (auto [cellpin_name, cellpin] : db.celllib[0]->cell(macroName)->cellpins) {
                    char direction = 'x';
                    char type = 's';
                    switch (*cellpin.direction) {
                        case gt::CellpinDirection::INPUT:
                            direction = 'i';
                            break;
                        case gt::CellpinDirection::OUTPUT:
                            direction = 'o';
                            break;
                        case gt::CellpinDirection::INOUT:
                            if (cellpin_name != "VDD" && cellpin_name != "vdd" && cellpin_name != "VSS" && cellpin_name != "vss") {
                                logger.warning("unknown pin %s.%s direction: %s", macroName.c_str(), cellpin_name.c_str(), "INOUT");
                            }
                            break;
                        default:
                            logger.error("unknown pin %s.%s direction: %s", macroName.c_str(), cellpin_name.c_str(), "UNKNOWN");
                            break;
                    }
                    PinType* pintype = celltype->addPin(cellpin_name, direction, type);
                }
            }

            cell = db.addCell(cellName, celltype);
        }
        for (size_t i = 0; i < inst.pin_names.size(); i++) {
            if (inst.net_names[i].size() > 1) logger.error("Bus net name is not supported\n");
            // define std::string and NetBit visit methods
            std::string pin_name =
                std::visit(overloaded{
                               [](std::string &v) { return v; },
                               [](verilog::NetBit &v) { return v.name + '[' + std::to_string(v.bit) + ']'; },
                               [](verilog::NetRange &v) { return v.name + '[' + std::to_string(v.beg) + ':' + std::to_string(v.end) + ']'; },
                               [](verilog::Constant &v) { return v.value; },
                           },
                           inst.pin_names[i]);
            // printf("gate:pin %s:%s\n", cellName.c_str(), pin_name.c_str());

            std::string net_name =
                std::visit(overloaded{
                               [](std::string &v) { return v; },
                               [](verilog::NetBit &v) { return v.name + '[' + std::to_string(v.bit) + ']'; },
                               [](verilog::NetRange &v) { return v.name + '[' + std::to_string(v.beg) + ':' + std::to_string(v.end) + ']'; },
                               [](verilog::Constant &v) { return v.value; },
                           },
                           inst.net_names[i][0]);
            // std::cout << "Net: " << net_name << '\n';

            std::string pinName(pin_name);
            std::string netName(net_name);
            netName = validate_token(netName);
            Net *net = db.getNet(netName);
            if (!net) logger.error("Net is not defined: %s", netName.c_str());
            Pin *pin;
            if (!verilog_only) {
                pin = cell->pin(pinName);
                if (!pin) logger.error("Pin is not defined: %s", pinName.c_str());
            } else {
                pin = cell->pin(pinName);
                if (!pin) logger.error("Pin is not defined: %s", pinName.c_str());
            }
            pin->net = net;
            net->addPin(pin);
            pin->is_connected = true;
            // if (!strcmp(netName.c_str(), "perm_state[0]")) {
            //     printf("Net -- pin %s -- %s:%s\n", netName.c_str(), cellName.c_str(), pinName.c_str());
            // }
            // if (!strcmp(cellName.c_str(), "round_reg_5_")) {
            //     bool debug = true;
            // }
        }
        cell->is_connected = true;
    }
};

template <typename T>
void removeDuplicates(std::vector<T> &vec) {
    std::unordered_set<T> uniqueElements;
    vec.erase(std::remove_if(vec.begin(), vec.end(), [&uniqueElements](const T &element) { return !uniqueElements.insert(element).second; }),
              vec.end());
}

bool Database::readVerilog_yy(const std::string &file, bool verilog_only) {
    // VerilogParser parser(this, verilog_only);
    verilog_parser = new VerilogParser(this, verilog_only);
    verilog_parser->read(file);

    // remove empty nets
    for (int i = 0; i < (int)nets.size(); i++) {
        if (nets[i]->pins.size() == 0) {
            nets.erase(nets.begin() + i);
            logger.warning("Empty net %s", nets[i]->name.c_str());
            i--;
        }
    }

    // remove duplicates in net pins
    for (auto &net : nets) {
        auto &pins = net->pins;

        // std::sort(pins.begin(), pins.end());
        // pins.erase(std::unique(pins.begin(), pins.end()), pins.end());
        // auto ip = std::unique(pins.begin(), pins.end());
        // // Resizing the vector so as to remove the undefined terms
        // pins.resize(std::distance(pins.begin(), ip));
        // set<Pin*> s( pins.begin(), pins.end() );
        // pins.assign( s.begin(), s.end() );

        removeDuplicates(net->pins);
    }

    // debug
    // for (auto &dbnet : nets) {
    //     if (!strcmp(dbnet->name.c_str(), "CTS_19")) {
    //         for (auto &pin : dbnet->pins) {
    //             string instName;
    //             if (pin->iopin != nullptr)
    //                 instName = "";
    //             else
    //                 instName = pin->cell->name();
    //             auto pin_name = instName + ":" + pin->type->name();
    //             printf("Net: %s, Pin: %s\n", dbnet->name.c_str(), pin_name.c_str());
    //         }
    //     }
    // }

    // exit(0);

    return true;
}

string tokenize_name(const string& name) {
    // if '[' or ']' in the name, add '\' in the front
    string new_name;
    for (auto &c : name) {
        if (c == '[' || c == ']' || c == '$' || c == '.') {
            new_name += '\\';
            break;
        }
    }
    new_name += name;
    return new_name;
}

bool Database::write_verilog(const std::string& file) {
    ofstream ofs(file.c_str());
    if (!ofs.good()) {
        logger.error("cannot open verilog file: %s", file.c_str());
        return false;
    }

    ofs << "module " << module_name << " (" << endl;

    for (int i = 0; i < verilog_parser->ports.size(); i++) {
        auto &port = verilog_parser->ports[i];
        for (auto &name : port.names) {
            ofs << "  " << name;
        }
        if (i != verilog_parser->ports.size() - 1) ofs << ",\n";
        else ofs << ");\n";
    }

    for (int i = 0; i < verilog_parser->ports.size(); i++) {
        auto &port = verilog_parser->ports[i];
        for (auto &name : port.names) {
            if (port.dir == verilog::PortDirection::INPUT || port.dir == verilog::PortDirection::OUTPUT) {
                if ((port.beg != -1) && (port.end != -1)) {
                    ofs << "  " << ((port.dir == verilog::PortDirection::INPUT) ? "input " : "output ") << "[" << port.beg << ":" << port.end << "] " << name ;
                } else {
                    ofs << "  " << ((port.dir == verilog::PortDirection::INPUT) ? "input " : "output ") << name;
                }
            }
        }
        ofs << ";\n";
    }

    for (Net* net : nets) {
        if (net->is_port) continue;
        ofs << " wire " << tokenize_name(net->name) << " ; " << endl;
    } 
    ofs << endl;

    for (Cell* cell : cells) {
        if (cell->removed) continue;
        string raw_name = cell->name();
        string cell_name = tokenize_name(raw_name);
        ofs << " " << cell->ctype()->name << " " << cell_name << " (";
        vector<Pin*> cell_pins = cell->pins();
        bool first_line = true;
        for (int i = 0; i < cell_pins.size(); i++) {
            Pin* pin = cell_pins[i];
            if (!pin->net) continue;
            if (!first_line) ofs << " ,\n";
            ofs << "    ." << pin->type->name() << "(" << tokenize_name(pin->net->name) << " )";
            first_line = false;
        }
        ofs << " );\n";
    } 


    ofs << "endmodule" << endl;
    ofs.close();
    return true;
}

}