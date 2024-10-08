#include <pybind11/pybind11.h>
#include <module/DataMiningModule.hpp>
#include <main/main_test.hpp>
#include <string>

#include <util/logging/BasicLogger.hpp>
#include <util/logging/GlobalLogger.hpp>

// ***  INITIALIZATION  *** //
// ************************ //
// Initialize C++ logging system
std::shared_ptr<BasicLogger> LOGGER = make_shared<BasicLogger>("VL3DPP.log");


// ***  DANGLING FUNCTIONS  *** //
// **************************** //
std::string get_hello_world() {
    return "HELLO WORLD";
}


// ***   PYVL3DPP MODULE   *** //
// *************************** //
PYBIND11_MODULE(pyvl3dpp, m){
    // Hello world
    m.def(
        "get_hello_world",
        &get_hello_world,
        "Return the HELLO WORLD string"
    );
    // TODO Rethink : Move main_test to vl3dpp_test ?
    m.def(
        "main_test",
        &vl3dpp::test::main_test,
        "Return the number of failed tests"
    );




    // ***  DATA MINING MODULES  *** //
    // ***************************** //
    // Mine smooth features <XDecimalType, FDecimalType>
    m.def(
        "mine_smooth_feats_dd",                             // Function name
        &vl3dpp::pymods::mine_smooth_feats<double, double>, // Function wrapper
        "Mine smooth features"                              // Description
    );
    m.def(
        "mine_smooth_feats_df",
        &vl3dpp::pymods::mine_smooth_feats<double, float>,
        "Mine smooth features"
    );
    m.def(
        "mine_smooth_feats_fd",
        &vl3dpp::pymods::mine_smooth_feats<float, double>,
        "Mine smooth features"
    );
    m.def(
        "mine_smooth_feats_ff",
        &vl3dpp::pymods::mine_smooth_feats<float, float>,
        "Mine smooth features"
    );
    // Mine height features <XDecimalType, FDecimalType>
    m.def(
        "mine_height_feats_dd",                               // Name
        &vl3dpp::pymods::mine_height_feats<double, double>,   // Wrapper
        "Mine height features"                                // Descrip.
    );
    m.def(
        "mine_height_feats_df",
        &vl3dpp::pymods::mine_height_feats<double, float>,
        "Mine height features"
    );
    m.def(
        "mine_height_feats_fd",
        &vl3dpp::pymods::mine_height_feats<float, double>,
        "Mine height features"
    );
    m.def(
        "mine_height_feats_ff",
        &vl3dpp::pymods::mine_height_feats<float, float>,
        "Mine height features"
    );
    // Recount-based features <XDecimalType, FDecimalType>
    m.def(
        "mine_recount_dd",                              // Function name
        &vl3dpp::pymods::mine_recount<double, double>,  // Function wrapper
        "Mine recount-based features"                   // Description
    );
    m.def(
        "mine_recount_df",
        &vl3dpp::pymods::mine_recount<double, float>,
        "Mine recount-based features"
    );
    m.def(
        "mine_recount_fd",
        &vl3dpp::pymods::mine_recount<float, double>,
        "Mine recount-based features"
    );
    m.def(
        "mine_recount_ff",
        &vl3dpp::pymods::mine_recount<float, float>,
        "Mine recount-based features"
    );
}