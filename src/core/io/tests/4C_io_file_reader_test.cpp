// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_io_file_reader.hpp"

#include "4C_unittest_utils_assertions_test.hpp"

#include <fstream>
#include <vector>

namespace
{

  using namespace FourC;

  TEST(CsvReaderTest, DataProcessingCsvStream)
  {
    const std::vector<double> x = {0.2, 0.4, 0.45};
    const std::vector<double> y = {4.3, 4.1, 4.15};
    const std::vector<double> z = {-1.0, 0.1, 1.3};

    std::stringstream test_csv_file_stream;
    test_csv_file_stream << "#x,y,z" << std::endl;
    for (std::size_t i = 0; i < x.size(); ++i)
      test_csv_file_stream << std::to_string(x[i]) << "," << std::to_string(y[i]) << ","
                           << std::to_string(z[i]) << std::endl;

    auto csv_values = Core::IO::read_csv_as_columns(3, test_csv_file_stream);

    EXPECT_EQ(csv_values[0], x);
    EXPECT_EQ(csv_values[1], y);
    EXPECT_EQ(csv_values[2], z);
  }

  TEST(CsvReaderTest, DataProcessingCsvFile)
  {
    const std::vector<double> x = {0.3, 0.4, 0.45};
    const std::vector<double> y = {4.3, 4.1, 4.15};
    const std::vector<double> z = {-1.0, 0.1, 1.3};

    const std::string csv_template_file_name = "test.csv";
    std::ofstream test_csv_file(csv_template_file_name);

    // include header line
    test_csv_file << "#x,y,z" << std::endl;
    for (std::size_t i = 0; i < x.size(); ++i)
      test_csv_file << std::to_string(x[i]) << "," << std::to_string(y[i]) << ","
                    << std::to_string(z[i]) << std::endl;
    // close template file
    test_csv_file.close();

    auto csv_values = Core::IO::read_csv_as_columns(3, csv_template_file_name);

    EXPECT_EQ(csv_values[0], x);
    EXPECT_EQ(csv_values[1], y);
    EXPECT_EQ(csv_values[2], z);
  }


  TEST(CsvReaderTest, DifferentColumnLengthThrows)
  {
    std::stringstream test_csv_file;
    test_csv_file << "#x,y" << std::endl;
    test_csv_file << "0.30,4.40" << std::endl;
    test_csv_file << "0.30," << std::endl;

    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        Core::IO::read_csv_as_columns(3, test_csv_file), Core::Exception, "same length");
  }

  TEST(CsvReaderTest, TrailingCommaThrows)
  {
    std::stringstream test_csv_file;
    test_csv_file << "#x,y" << std::endl;
    test_csv_file << "0.30,4.40," << std::endl;

    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        Core::IO::read_csv_as_columns(2, test_csv_file), Core::Exception, "trailing comma");
  }

  TEST(CsvReaderTest, WrongColumnNumberThrows)
  {
    std::stringstream test_csv_file;
    test_csv_file << "#x,y" << std::endl;
    test_csv_file << "0.30,4.40" << std::endl;

    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        Core::IO::read_csv_as_columns(3, test_csv_file), Core::Exception, "");
  }

  TEST(CsvReaderTest, WrongHeaderStyleThrows)
  {
    std::stringstream test_csv_file;
    test_csv_file << "x,y" << std::endl;
    test_csv_file << "0.30,4.40" << std::endl;

    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        Core::IO::read_csv_as_columns(2, test_csv_file), Core::Exception, "header");
  }

  TEST(CsvReaderTest, WrongInputDataTypeThrows)
  {
    std::stringstream test_csv_file;
    test_csv_file << "x,y" << std::endl;
    test_csv_file << "0.30,a" << std::endl;

    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        Core::IO::read_csv_as_columns(2, test_csv_file), Core::Exception, "numbers");
  }

  TEST(CsvReaderTest, WrongSeparatorThrows)
  {
    std::stringstream test_csv_file;
    test_csv_file << "x;y" << std::endl;
    test_csv_file << "0.30;4.40" << std::endl;

    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        Core::IO::read_csv_as_columns(2, test_csv_file), Core::Exception, "separated by commas");
  }
}  // namespace
