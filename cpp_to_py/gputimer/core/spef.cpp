#include "GPUTimer.h"
#include "gputimer/db/GTDatabase.h"

using std::ofstream;
using std::string;
using std::cerr;
using std::endl;
using std::stringstream;

namespace gt {

void GPUTimer::read_spef(const std::string& file) {
    logger.info("reading spef: %s", file.c_str());
    if (not std::filesystem::exists(file)) {
        std::cerr << "can't find " << file << '\n';
        std::exit(EXIT_FAILURE);
    }

    // Invoke the read function and check the return value
    if (not spef.read(file)) {
        std::cerr << *spef.error;
        std::exit(EXIT_FAILURE);
    }


    if (spef.time_unit == "1 PS") gtdb.spef_time_unit = 1e-12;
    if (spef.time_unit == "1 NS") gtdb.spef_time_unit = 1e-9;
    if (spef.time_unit == "1 US") gtdb.spef_time_unit = 1e-6;
    if (spef.time_unit == "1 MS") gtdb.spef_time_unit = 1e-3;
    if (spef.time_unit == "1 S") gtdb.spef_time_unit = 1.0; ;

    if (spef.capacitance_unit == "1 FF") gtdb.spef_cap_unit = 1e-15;
    if (spef.capacitance_unit == "1 PF") gtdb.spef_cap_unit = 1e-12;
    if (spef.capacitance_unit == "1 NF") gtdb.spef_cap_unit = 1e-9;
    if (spef.capacitance_unit == "1 UF") gtdb.spef_cap_unit = 1e-6;
    if (spef.capacitance_unit == "1 F") gtdb.spef_cap_unit = 1.0;

    if (spef.resistance_unit == "1 OHM") gtdb.spef_res_unit = 1.0;
    if (spef.resistance_unit == "1 KOHM") gtdb.spef_res_unit = 1e3;
    if (spef.resistance_unit == "1 MOHM") gtdb.spef_res_unit = 1e6;

    logger.info("spef time_unit: %.5E s", *gtdb.spef_time_unit);
    logger.info("spef capacitance_unit: %.5E F", *gtdb.spef_cap_unit);
    logger.info("spef resistance_unit: %.5E Ohm", *gtdb.spef_res_unit);

    spef.expand_name();
}

void GPUTimer::write_spef(const std::string& file) {
    ofstream dot_spef(file.c_str());
    if (!dot_spef.good()) {
        cerr << "write_incremental_spef:: cannot open `" << file << "' for writing." << endl;
        exit(1);
    }
    stringstream feed;
    feed.precision(5);

    time_t rawtime;
    time(&rawtime);
    string t(ctime(&rawtime));
    feed << "*SPEF \"IEEE 1481-1998\"" << endl;
    feed << "*DATE \"" << t.substr(0, t.length() - 1) << "\"" << endl;
    feed << "*VENDOR \"ICCAD 2015 Contest\"" << endl;
    feed << "*PROGRAM \"ICCAD 2015 Contest Spef Generator\"" << endl;
    feed << "*VERSION \"0.0\"" << endl;
    feed << "*DIVIDER /" << endl;
    feed << "*DELIMITER :" << endl;
    feed << "*BUS_DELIMITER [ ]" << endl;



    feed << "*T_UNIT 1 PS" << endl;
    feed << "*C_UNIT 1 FF" << endl;
    feed << "*R_UNIT 1 KOHM" << endl;
    feed << "*L_UNIT 1 UH" << endl << endl;
}

}  // namespace gt