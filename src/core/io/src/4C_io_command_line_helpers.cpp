// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"

#include "4C_io_command_line_helpers.hpp"


FOUR_C_NAMESPACE_OPEN


std::vector<std::string> adapt_legacy_cli_arguments(
    const std::vector<std::string>& args, LegacyCliOptions& legacy_options)
{
  if (args.empty()) return {};

  std::vector<std::string> new_args;
  new_args.reserve(args.size());
  std::vector<std::vector<std::string>> pending_vals(legacy_options.nodash_legacy_names.size());

  auto warn = [](const std::string& name, const std::string& to)
  {
    std::cerr << "DEPRECATION WARNING: Legacy argument '" << name << "' has been converted to '"
              << to
              << "'. Please update your command line arguments. This legacy form will be removed "
                 "in a future release.\n";
  };

  auto combine_and_warn = [&warn](std::vector<std::string>& out_args, const std::string& name,
                              std::vector<std::string>& vals)
  {
    if (!vals.empty())
    {
      std::string combined = std::string("--") + name + "=";
      for (size_t i = 0; i < vals.size(); ++i)
      {
        if (i) combined += ",";
        combined += vals[i];
      }
      out_args.push_back(combined);
      warn(name, combined);
      vals.clear();
    }
  };

  for (const std::string& arg : args)
  {
    bool handled = false;

    // Check nodash legacy names first (e.g., "restart=1") and collect values.
    for (size_t j = 0; j < legacy_options.nodash_legacy_names.size(); ++j)
    {
      const std::string& name = legacy_options.nodash_legacy_names[j];
      std::string prefix = name + "=";
      if (arg.rfind(prefix, 0) == 0)
      {
        pending_vals[j].push_back(arg.substr(prefix.size()));
        handled = true;
        break;
      }
    }
    if (handled) continue;

    // Convert known single-dash legacy options to double-dash (e.g., "-ngroup=2" -> "--ngroup=2").
    for (const auto& name : legacy_options.single_dash_legacy_names)
    {
      std::string prefix = std::string("-") + name + "=";
      if (arg.rfind(prefix, 0) == 0)
      {
        std::string new_arg = std::string("-") + arg;
        warn(arg, new_arg);
        new_args.push_back(new_arg);
        handled = true;
        break;
      }
    }
    if (handled) continue;

    // Keep everything else unchanged (positional args, two dashed args, -p, -h, etc.)
    new_args.push_back(arg);
  }

  for (size_t j = 0; j < legacy_options.nodash_legacy_names.size(); ++j)
  {
    combine_and_warn(new_args, legacy_options.nodash_legacy_names[j], pending_vals[j]);
  }

  return new_args;
}

FOUR_C_NAMESPACE_CLOSE