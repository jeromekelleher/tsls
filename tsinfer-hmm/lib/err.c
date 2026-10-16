/*
** MIT License
**
** Copyright (C) 2020 University of Oxford
**
** Permission is hereby granted, free of charge, to any person obtaining a copy
** of this software and associated documentation files (the "Software"), to deal
** in the Software without restriction, including without limitation the rights
** to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
** copies of the Software, and to permit persons to whom the Software is
** furnished to do so, subject to the following conditions:
**
** The above copyright notice and this permission notice shall be included in all
** copies or substantial portions of the Software.
**
** THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
** IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
** FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
** AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
** LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
** OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
** SOFTWARE.
*/

#include "err.h"
#include <tskit.h>

const char *
tsi_strerror(int err)
{
    const char *ret = "Unknown error";

    switch (err) {
        case 0:
            ret = "Normal exit condition. This is not an error!";
            break;

        case TSI_ERR_GENERIC:
            ret = "Generic tsinfer error - please file a bug report.";
            break;
        case TSI_ERR_NO_MEMORY:
            ret = "Out of memory";
            break;
        case TSI_ERR_NONCONTIGUOUS_EDGES:
            ret = "Edges must be contiguous";
            break;
        case TSI_ERR_UNSORTED_EDGES:
            ret = "Edges must be sorted";
            break;
        case TSI_ERR_PC_ANCESTOR_TIME:
            ret = "Failure generating time for path compression ancestor";
            break;
        case TSI_ERR_BAD_PATH_CHILD:
            ret = "Bad path information: child node";
            break;
        case TSI_ERR_BAD_PATH_PARENT:
            ret = "Bad path information: parent node";
            break;
        case TSI_ERR_BAD_PATH_TIME:
            ret = "Bad path information: time";
            break;
        case TSI_ERR_BAD_PATH_INTERVAL:
            ret = "Bad path information: left >= right";
            break;
        case TSI_ERR_BAD_PATH_LEFT_LESS_ZERO:
            ret = "Bad path information: left < 0";
            break;
        case TSI_ERR_BAD_PATH_RIGHT_GREATER_NUM_SITES:
            ret = "Bad path information: right > num_sites";
            break;
        case TSI_ERR_MATCH_IMPOSSIBLE:
            ret = "Unexpected failure to find matching haplotype; please open "
                  "an issue on GitHub";
            break;
        case TSI_ERR_MATCH_IMPOSSIBLE_EXTREME_MUTATION_PROBA:
            ret = "Cannot find match: the specified mismatch probability is "
                  "0 or 1 and no matches are possible with these parameters";
            break;
        case TSI_ERR_MATCH_IMPOSSIBLE_ZERO_RECOMB_PRECISION:
            ret = "Cannot find match: the specified recombination probability is"
                  "zero and no matches could be found. Increasing the 'precision' "
                  "may help, but recombination values of 0 are not recommended.";
            break;
        case TSI_ERR_BAD_HAPLOTYPE_ALLELE:
            ret = "Input haplotype contains bad allele information.";
            break;
        case TSI_ERR_BAD_NUM_ALLELES:
            ret = "The number of alleles must be between 2 and 127";
            break;
        case TSI_ERR_BAD_MUTATION_NODE:
            ret = "Bad mutation information: node";
            break;
        case TSI_ERR_BAD_MUTATION_SITE:
            ret = "Bad mutation information: site";
            break;
        case TSI_ERR_BAD_MUTATION_DERIVED_STATE:
            ret = "Bad mutation information: derived state";
            break;
        case TSI_ERR_BAD_MUTATION_DUPLICATE_NODE:
            ret = "Bad mutation information: mutation already exists for this node.";
            break;
        case TSI_ERR_BAD_NUM_SAMPLES:
            ret = "Must have at least 2 samples.";
            break;
        case TSI_ERR_TOO_MANY_SITES:
            ret = "Cannot add more sites than the specified maximum.";
            break;
        case TSI_ERR_BAD_FOCAL_SITE:
            ret = "Bad focal site.";
            break;
        case TSI_ERR_ONE_BIT_NON_BINARY:
            ret = "One-bit genotype encoding only supports binary 0/1 data";
            break;
        case TSI_ERR_IO:
            ret = tsk_strerror(TSK_ERR_IO);
            break;
        case TSI_ERR_BAD_ANCESTRAL_STATE:
            ret = "Ancestral state must be >= 0 and < num_alleles";
            break;
        case TSI_ERR_BAD_EDGE_NODE:
            ret = "Bad edge: parent or child node out of bounds";
            break;
        case TSI_ERR_NODE_0_NOT_ROOT:
            ret = "Node 0 must be the root: no edges have parent 0";
            break;
    }
    return ret;
}

/* shim allocator moved to lib/tsi_blkalloc.c */
