## ---------------------------------------------------------------------
##
## Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
## authors.
##
## This file is part of the DFT-FE code.
##
## The DFT-FE code is free software; you can use it, redistribute
## it, and/or modify it under the terms of the GNU Lesser General
## Public License as published by the Free Software Foundation; either
## version 2.1 of the License, or (at your option) any later version.
## The full text of the license can be found in the file LICENSE at
## the top level of the DFT-FE distribution.
##
## ---------------------------------------------------------------------
#
# Removes run-dependent text from a test's output before it is compared with
# the reference; applied with "perl -pi normalize.pl output". The reference
# outputs were normalized with deal.II's tests/normalize.pl, whose rules that
# can apply to DFT-FE output are repeated here.

# Windows line endings
s/\r$//;

# job, date and time stamps
s/JobId.*//;
s/# Input file name:.*//;
s/# Time =.*//;
s/# Date =.*//;
s/^\s+Time =.*//;
s/^\s+Date =.*//;
s/Time tag:.*//g;

# the line number in an exception message
s/line <\d+> of file <.*\//file </;

# -0.0... not followed by another digit is written without the sign
s/-0\.(0+)(?!\d)/0.\1/g;

# Library notices printed on stderr depending on the environment, such as
# "ELPA: Default for option 'omp_threads' is set to '1' due to environment
# variable ELPA_DEFAULT_omp_threads", are not part of the result
$_ = "" if m/^ELPA: /;

# X server authorization warnings from dependency libraries
if (m/Authorization required, but no authorization protocol specified/)
{
    <>;
    $_ = "";
}
