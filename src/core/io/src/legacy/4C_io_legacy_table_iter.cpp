// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_io_legacy_table_iter.hpp"

#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

#include <stack>

/*----------------------------------------------------------------------*/
/*!
  \brief map iterator constructor
 */
/*----------------------------------------------------------------------*/
void init_map_iterator(MapIterator* iterator, MAP* map)
{
  iterator->map = map;
  iterator->stack = std::stack<MapNode*>{};
}

/*----------------------------------------------------------------------*/
/*!
  \brief map iterator

  \param iterator (i/o) the map iterator to be advanced
  \return true if a new node was found
 */
/*----------------------------------------------------------------------*/
int next_map_node(MapIterator* iterator)
{
  if (!iterator->map) return 0;

  if (iterator->stack.empty())
  {
    if (iterator->map->root.rhs) iterator->stack.push(iterator->map->root.rhs);
    if (iterator->map->root.lhs) iterator->stack.push(iterator->map->root.lhs);
    return !iterator->stack.empty();
  }
  else
  {
    auto tmp = iterator->stack.top();
    iterator->stack.pop();
    if (tmp->rhs) iterator->stack.push(tmp->rhs);
    if (tmp->lhs) iterator->stack.push(tmp->lhs);
    return !iterator->stack.empty();
  }
}
/*----------------------------------------------------------------------*/
/*!
  \brief map iterator current node

 */
/*----------------------------------------------------------------------*/
MapNode* iterator_get_node(MapIterator* iterator) { return iterator->stack.top(); }

FOUR_C_NAMESPACE_CLOSE
