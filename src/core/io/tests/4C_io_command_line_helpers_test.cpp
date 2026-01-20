// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_io_command_line_helpers.hpp"

#include <vector>

namespace
{
  using namespace FourC;

  TEST(AdaptLegacyCliArguments, ConvertsAndCombines)
  {
    std::vector<std::string> in = {"-ngroup=2", "-glayout=3,2", "-nptype=separateInputFiles",
        "inp1", "out1", "restart=1", "restartfrom=xxx", "inp2", "out2", "restart=2",
        "restartfrom=yyy"};


    LegacyCliOptions legacy_options = {.single_dash_legacy_names = {"ngroup", "glayout", "nptype"},
        .nodash_legacy_names = {"restart", "restartfrom"}};
    std::vector<std::string> out = adapt_legacy_cli_arguments(in, legacy_options);

    // Expected: converted --ngroup, combined --restart, combined --restartfrom
    std::vector<std::string> expect = {"--ngroup=2", "--glayout=3,2", "--nptype=separateInputFiles",
        "inp1", "out1", "inp2", "out2", "--restart=1,2", "--restartfrom=xxx,yyy"};
    EXPECT_EQ(out, expect);
  }

}  // namespace