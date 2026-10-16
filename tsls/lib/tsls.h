#ifndef __TSLS_H__
#define __TSLS_H__

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

#define TSLS_MISSING_DATA (-1)

/* Special values in the likelihood array. A node with NULL_LIKELIHOOD
 * inherits the value of its nearest ancestor with a likelihood, and a node
 * that is NOT_IN_TREE is not in the current tree (it has no parent and no
 * children), and so is not a state of the HMM. */
#define NULL_LIKELIHOOD (-1)
#define NOT_IN_TREE     (-2)

#define NULL_NODE   (-1)
#define CACHE_UNSET (-1)

/* Matcher flags */
#define TSLS_EXTENDED_CHECKS     (1 << 0)
#define TSLS_DISABLE_WEIGHT_BY_N (1 << 1)
#define TSLS_COUNT_DISTINCT      (1 << 2)

/* Experimental pruning modes. After normalisation at each site, pruned
 * nodes are given the value likelihood_threshold. */
#define TSLS_PRUNE_NONE     0
/* Prune nodes with likelihood < parameter * max likelihood */
#define TSLS_PRUNE_RELATIVE 1
/* Prune all but the parameter nodes with the largest likelihoods */
#define TSLS_PRUNE_TOP_N    2

typedef int8_t allele_t;
typedef tsk_id_t coordinate_t;

/* Allow the CPython accessor AncestorMatcher.total_memory to read the
 * current allocator usage accurately while other threads may be allocating.
 * Only total_size is atomic so it can be observed safely from the getter
 * without data races. */
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

/* The read-only information about a tree sequence needed for matching,
 * which can be shared between matchers in different threads. */
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

/* The compressed output of the forward and backward algorithms. At each
 * site we store a list of (node, value) pairs: the value for a node in the
 * tree is the value of its nearest ancestor (including itself) in this list.
 */
typedef struct {
    size_t num_sites;
    double *normalisation_factor;
    tsk_size_t *num_values;
    tsk_id_t **nodes;
    double **values;
    /* For Viterbi, the recombination_required flag of each node */
    int8_t **flags;
    tsi_blkalloc_t allocator;
} compressed_matrix_t;

typedef struct {
    int flags;
    const matcher_indexes_t *matcher_indexes;
    size_t num_nodes;
    size_t num_sites;
    /* Input LS model rates */
    double likelihood_threshold;
    int pruning_mode;
    double pruning_parameter;
    double *recombination_rate;
    double *mismatch_rate;
    /* The quintuply linked tree */
    tsk_id_t *parent;
    tsk_id_t *left_child;
    tsk_id_t *right_child;
    tsk_id_t *left_sib;
    tsk_id_t *right_sib;
    /* The number of nodes in the subtree below each node, including itself.
     * Only maintained for the forward and backward algorithms. */
    int64_t *num_below;
    bool track_num_below;
    /* The number of nodes in the tree that inherit each likelihood value. */
    int64_t *num_inheriting;
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
    /* The total number of distinct likelihood values over sites, if
     * TSLS_COUNT_DISTINCT is set. */
    size_t total_distinct_values;
    /* Scratch buffer for values */
    double *values_tmp;
    size_t traceback_block_size;
    size_t traceback_realloc_size;
} ancestor_matcher_t;

int matcher_indexes_alloc(matcher_indexes_t *self, const tsk_table_collection_t *tables,
    const tsk_size_t *num_alleles, const allele_t *ancestral_state,
    const allele_t *derived_state, tsk_flags_t options);
int matcher_indexes_print_state(const matcher_indexes_t *self, FILE *out);
int matcher_indexes_free(matcher_indexes_t *self);

int compressed_matrix_alloc(
    compressed_matrix_t *self, size_t num_sites, size_t block_size);
int compressed_matrix_free(compressed_matrix_t *self);
size_t compressed_matrix_get_total_values(const compressed_matrix_t *self);

int ancestor_matcher_alloc(ancestor_matcher_t *self,
    const matcher_indexes_t *matcher_indexes, const double *recombination_rate,
    const double *mismatch_rate, double likelihood_threshold, int flags);
int ancestor_matcher_set_pruning(ancestor_matcher_t *self, int mode, double parameter);
int ancestor_matcher_free(ancestor_matcher_t *self);
int ancestor_matcher_find_path(ancestor_matcher_t *self, tsk_id_t start, tsk_id_t end,
    const allele_t *haplotype, allele_t *matched_haplotype, size_t *path_length,
    tsk_id_t *path_left, tsk_id_t *path_right, tsk_id_t *path_parent);
int ancestor_matcher_find_path_matrix(ancestor_matcher_t *self, tsk_id_t start,
    tsk_id_t end, const allele_t *haplotype, allele_t *matched_haplotype,
    size_t *path_length, tsk_id_t *path_left, tsk_id_t *path_right,
    tsk_id_t *path_parent, compressed_matrix_t *output);
int ancestor_matcher_forward(ancestor_matcher_t *self, const allele_t *haplotype,
    compressed_matrix_t *output);
int ancestor_matcher_backward(ancestor_matcher_t *self, const allele_t *haplotype,
    const double *forward_normalisation_factor, compressed_matrix_t *output);
int ancestor_matcher_print_state(ancestor_matcher_t *self, FILE *out);
double ancestor_matcher_get_mean_traceback_size(ancestor_matcher_t *self);
double ancestor_matcher_get_mean_num_distinct_values(ancestor_matcher_t *self);
size_t ancestor_matcher_get_total_memory(ancestor_matcher_t *self);

#define tsi_safe_free(pointer)                                                          \
    do {                                                                                \
        if (pointer != NULL) {                                                          \
            free(pointer);                                                              \
        }                                                                               \
    } while (0)

#endif /* __TSLS_H__ */
