// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_TSI_DEFINES_HPP
#define FOUR_C_TSI_DEFINES_HPP

#include "4C_config.hpp"

FOUR_C_NAMESPACE_OPEN

/************************************************************************/
/* Debugging options for TSI algorithm                                  */
/************************************************************************/
// #define COUPLEINITTEMPERATURE  /* flag for Thermo: constant temperature T_0 in couple term  */
// #define TSIPARALLEL            /* flag for parallel TSI */
// #define MonTSWithoutTHERMO       /* flag to comment out all coupling terms in STR field */
// #define MonTSWithoutSTR       /* flag to comment out all coupling terms in Thermo field */
// #define TFSI                   /* flag to reduce the output to screen */
// #define TSI_DEBUG              /* output to screen information */


/************************************************************************/
/* Debugging options dependent on problem type                          */
/************************************************************************/

// GENERAL DEBUGGING OPTIONS
// #define TSIASOUTPUT            /* flag for detailed active set output */
// #define TSIMONOLITHASOUTPUT    /* flag for detailed output in monolithic TSI */
// #define TSIPARTITIONEDASOUTPUT /* flag for detailed output in partitioned TSI */


/************************************************************************/
/* Output options for thermal problems                                  */
/************************************************************************/

// #define THRASOUTPUT            /* flag for output in the thermal field */
// #define CALCSTABILOFREACTTERM  /* flag for output of stabilisation parameter in Thermo equation
// */


/************************************************************************/
/* Debugging options for plastic materials                              */
/************************************************************************/
// #define DEBUGMATERIAL          /* flag for material debug output */

/************************************************************************/
/* Debugging options for pseudo-SLM in TSI                              */
/************************************************************************/
// #define TSISLMFDCHECK         /* flag for enabling FDChecks*/
// #define TSISLMFDCHECKDEBUG    /* prints element matrices in FDcheck to a file
// "FDCheck_capa.txt"*/

FOUR_C_NAMESPACE_CLOSE

#endif
