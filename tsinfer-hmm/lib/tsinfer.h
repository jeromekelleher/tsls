/*
** MIT License
**
** Copyright (C) 2018-2026 University of Oxford
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

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

/* No atomics on Windows for now */
#if defined(_WIN32)
#define TSI_NO_ATOMICS 1
#endif

#if !defined(TSI_NO_ATOMICS)
#include <stdatomic.h>
#define TSI_ATOMIC(T) _Atomic T
#else
#define TSI_ATOMIC(T) T
#endif

#include "tskit.h"
#include "err.h"

/* TODO remove this when we update tskit version. */
#define TSK_MISSING_DATA (-1)

/* NULL_LIKELIHOOD represents a compressed path and NONZERO_ROOT_LIKELIHOOD
 * marks a node that is not in the current tree. */
#define NULL_LIKELIHOOD         (-1)
#define NONZERO_ROOT_LIKELIHOOD (-2)

#define NULL_NODE   (-1)
#define CACHE_UNSET (-1)

#define TSI_EXTENDED_CHECKS     1
#define TSI_DISABLE_WEIGHT_BY_N 2

typedef int8_t allele_t;
/* TODO should probably change to uint32 when we have removed the old code.*/
typedef tsk_id_t coordinate_t;

/* Allow the CPython accessor AncestorMatcher.total_memory to read the
 * current allocator usage accurately while other threads may be allocating.
 * Allocation semantics remain identical to tsk_blkalloc; only total_size is
 * atomic so it can be observed safely from the getter without data races.
 * See https://github.com/tskit-dev/tsinfer/commit/57f9209d for context. */
typedef struct {
    size_t chunk_size; /* number of bytes per chunk */
    size_t top;        /* the offset of the next available byte in the current chunk */
    size_t current_chunk;          /* the index of the chunk currently being used */
    TSI_ATOMIC(size_t) total_size; /* the total number of bytes allocated + overhead. */
    size_t total_allocated;        /* the total number of bytes allocated. */
    size_t num_chunks;             /* the number of memory chunks. */
    char **mem_chunks;             /* the memory chunks */
} tsi_blkalloc_t;

extern void tsi_blkalloc_print_state(tsi_blkalloc_t *self, FILE *out);
extern int tsi_blkalloc_reset(tsi_blkalloc_t *self);
extern int tsi_blkalloc_init(tsi_blkalloc_t *self, size_t chunk_size);
extern void *tsi_blkalloc_get(tsi_blkalloc_t *self, size_t size);
extern void tsi_blkalloc_free(tsi_blkalloc_t *self);

typedef struct {
    coordinate_t left;
    coordinate_t right;
    tsk_id_t parent;
    tsk_id_t child;
} edge_t;

typedef struct _mutation_list_node_t {
    tsk_id_t node;
    allele_t derived_state;
    struct _mutation_list_node_t *next;
} mutation_list_node_t;

typedef struct {
    int32_t size;
    tsk_id_t *node;
    int8_t *recombination_required;
} node_state_list_t;

typedef struct {
    tsk_flags_t flags;
    size_t num_sites;
    size_t num_nodes;
    size_t num_mutations;
    size_t num_edges;
    struct {
        coordinate_t *position;
        mutation_list_node_t **mutations;
        tsk_size_t *num_alleles;
        allele_t *ancestral_state;
    } sites;
    edge_t *left_index_edges;
    edge_t *right_index_edges;
    tsi_blkalloc_t allocator;
} matcher_indexes_t;

typedef struct {
    int flags;
    const matcher_indexes_t *matcher_indexes;
    size_t num_nodes;
    size_t num_sites;
    /* Input LS model rates */
    double likelihood_threshold;
    double *recombination_rate;
    double *mismatch_rate;
    /* The quintuply linked tree */
    tsk_id_t *parent;
    tsk_id_t *left_child;
    tsk_id_t *right_child;
    tsk_id_t *left_sib;
    tsk_id_t *right_sib;
    double *likelihood;
    double *likelihood_cache;
    allele_t *allelic_state;
    int num_likelihood_nodes;
    /* At each site, record a node with the maximum likelihood. */
    tsk_id_t *max_likelihood_node;
    /* Used during traceback to map nodes where recombination is required. */
    int8_t *recombination_required;
    tsk_id_t *likelihood_nodes_tmp;
    tsk_id_t *likelihood_nodes;
    node_state_list_t *traceback;
    tsi_blkalloc_t traceback_allocator;
    size_t total_traceback_size;
    size_t traceback_block_size;
    size_t traceback_realloc_size;
} ancestor_matcher_t;

int matcher_indexes_alloc(matcher_indexes_t *self, const tsk_table_collection_t *tables,
    const tsk_size_t *num_alleles, const allele_t *ancestral_state,
    const allele_t *derived_state, tsk_flags_t options);
int matcher_indexes_print_state(const matcher_indexes_t *self, FILE *out);
int matcher_indexes_free(matcher_indexes_t *self);

int ancestor_matcher_alloc(ancestor_matcher_t *self,
    const matcher_indexes_t *matcher_indexes, double *recombination_rate,
    double *mismatch_rate, double likelihood_threshold, int flags);
int ancestor_matcher_free(ancestor_matcher_t *self);
int ancestor_matcher_find_path(ancestor_matcher_t *self, tsk_id_t start, tsk_id_t end,
    const allele_t *haplotype, allele_t *matched_haplotype, size_t *path_length,
    tsk_id_t *path_left, tsk_id_t *path_right, tsk_id_t *path_parent);
int ancestor_matcher_print_state(ancestor_matcher_t *self, FILE *out);
double ancestor_matcher_get_mean_traceback_size(ancestor_matcher_t *self);
size_t ancestor_matcher_get_total_memory(ancestor_matcher_t *self);

#define tsi_safe_free(pointer)                                                          \
    do {                                                                                \
        if (pointer != NULL) {                                                          \
            free(pointer);                                                              \
        }                                                                               \
    } while (0)
