/* Lean C tests. The algorithms are tested in detail against a dense
 * implementation in the Python tests. */

#include "tsls.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include <CUnit/Basic.h>

static FILE *_devnull;

/* A forest over [0, 10): roots 3 (above 0 and 1) and 4 (above 2) on [0, 5),
 * joined by node 6 on [5, 10). Node 5 is isolated. */
static void
build_forest(tsk_table_collection_t *tables)
{
    int ret;
    int j;
    double time[] = { 0, 0, 0, 1, 2, 4, 3 };

    ret = tsk_table_collection_init(tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tables->sequence_length = 10;
    for (j = 0; j < 7; j++) {
        tsk_node_table_add_row(&tables->nodes, time[j] == 0 ? TSK_NODE_IS_SAMPLE : 0,
            time[j], TSK_NULL, TSK_NULL, NULL, 0);
    }
    tsk_edge_table_add_row(&tables->edges, 0, 10, 3, 0, NULL, 0);
    tsk_edge_table_add_row(&tables->edges, 0, 10, 3, 1, NULL, 0);
    tsk_edge_table_add_row(&tables->edges, 0, 10, 4, 2, NULL, 0);
    tsk_edge_table_add_row(&tables->edges, 5, 10, 6, 3, NULL, 0);
    tsk_edge_table_add_row(&tables->edges, 5, 10, 6, 4, NULL, 0);
    tsk_site_table_add_row(&tables->sites, 1, "A", 1, NULL, 0);
    tsk_site_table_add_row(&tables->sites, 3, "C", 1, NULL, 0);
    tsk_site_table_add_row(&tables->sites, 6, "G", 1, NULL, 0);
    tsk_site_table_add_row(&tables->sites, 8, "A", 1, NULL, 0);
    tsk_mutation_table_add_row(
        &tables->mutations, 0, 3, TSK_NULL, TSK_UNKNOWN_TIME, "C", 1, NULL, 0);
    tsk_mutation_table_add_row(
        &tables->mutations, 0, 2, TSK_NULL, TSK_UNKNOWN_TIME, "G", 1, NULL, 0);
    tsk_mutation_table_add_row(
        &tables->mutations, 1, 4, TSK_NULL, TSK_UNKNOWN_TIME, "A", 1, NULL, 0);
    tsk_mutation_table_add_row(
        &tables->mutations, 2, 0, TSK_NULL, TSK_UNKNOWN_TIME, "A", 1, NULL, 0);
    tsk_mutation_table_add_row(
        &tables->mutations, 3, 6, TSK_NULL, TSK_UNKNOWN_TIME, "T", 1, NULL, 0);
    tsk_mutation_table_add_row(
        &tables->mutations, 3, 1, TSK_NULL, TSK_UNKNOWN_TIME, "A", 1, NULL, 0);
    ret = tsk_table_collection_sort(tables, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_build_index(tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
}

/* Alleles are A=0, C=1, G=2, T=3 */
static tsk_size_t num_alleles[] = { 4, 4, 4, 4 };
static allele_t ancestral_state[] = { 0, 1, 2, 0 };
static allele_t derived_state[] = { 1, 2, 0, 0, 3, 0 };

static void
test_matcher_indexes_errors(void)
{
    int ret;
    matcher_indexes_t mi;
    tsk_table_collection_t tables;
    allele_t bad_ancestral_state[] = { 0, 4, 2, 0 };
    allele_t bad_derived_state[] = { 1, 2, -1, 0, 3, 0 };

    build_forest(&tables);

    ret = matcher_indexes_alloc(&mi, &tables, num_alleles, bad_ancestral_state, NULL, 0);
    CU_ASSERT_EQUAL(ret, TSLS_ERR_BAD_ANCESTRAL_STATE);
    matcher_indexes_free(&mi);

    ret = matcher_indexes_alloc(&mi, &tables, num_alleles, NULL, bad_derived_state, 0);
    CU_ASSERT_EQUAL(ret, TSLS_ERR_BAD_MUTATION_DERIVED_STATE);
    matcher_indexes_free(&mi);

    /* Default of 2 alleles */
    ret = matcher_indexes_alloc(&mi, &tables, NULL, ancestral_state, NULL, 0);
    CU_ASSERT_EQUAL(ret, TSLS_ERR_BAD_ANCESTRAL_STATE);
    matcher_indexes_free(&mi);

    tables.mutations.node[0] = 100;
    ret = matcher_indexes_alloc(&mi, &tables, num_alleles, NULL, NULL, 0);
    CU_ASSERT_EQUAL(ret, TSLS_ERR_BAD_MUTATION_NODE);
    matcher_indexes_free(&mi);
    tables.mutations.node[0] = 0;

    tables.mutations.site[0] = 100;
    ret = matcher_indexes_alloc(&mi, &tables, num_alleles, NULL, NULL, 0);
    CU_ASSERT_EQUAL(ret, TSLS_ERR_BAD_MUTATION_SITE);
    matcher_indexes_free(&mi);
    tables.mutations.site[0] = 0;

    tables.edges.child[0] = -1;
    ret = matcher_indexes_alloc(&mi, &tables, num_alleles, NULL, NULL, 0);
    CU_ASSERT_EQUAL(ret, TSLS_ERR_BAD_EDGE_NODE);
    matcher_indexes_free(&mi);
    tables.edges.child[0] = 0;

    tables.sites.position[0] = 1.5;
    ret = matcher_indexes_alloc(&mi, &tables, num_alleles, NULL, NULL, 0);
    CU_ASSERT_EQUAL(ret, TSLS_ERR_BAD_COORDINATE);
    matcher_indexes_free(&mi);

    tsk_table_collection_free(&tables);
}

static void
test_forest_matching(void)
{
    int ret;
    tsk_table_collection_t tables;
    matcher_indexes_t mi;
    ancestor_matcher_t am;
    compressed_matrix_t fm, bm;
    double rho[] = { 0.01, 0.01, 0.01, 0.01 };
    double mu[] = { 0.01, 0.01, 0.01, 0.01 };
    /* The haplotype of node 1: C (under 3), C (ancestral), G, A (back mutation) */
    allele_t h[] = { 1, 1, 2, 0 };
    allele_t match[4];
    tsk_id_t left[4], right[4], parent[4];
    size_t path_length, j, k, l;
    double total;

    build_forest(&tables);
    ret = matcher_indexes_alloc(&mi, &tables, num_alleles, ancestral_state, derived_state, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    matcher_indexes_print_state(&mi, _devnull);
    ret = ancestor_matcher_alloc(&am, &mi, rho, mu, DBL_MIN, TSLS_EXTENDED_CHECKS);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = ancestor_matcher_find_path(
        &am, 0, 4, h, match, &path_length, left, right, parent);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL(path_length, 1);
    CU_ASSERT_EQUAL(parent[0], 1);
    CU_ASSERT_EQUAL(left[0], 0);
    CU_ASSERT_EQUAL(right[0], 10);
    for (j = 0; j < 4; j++) {
        CU_ASSERT_EQUAL(match[j], h[j]);
    }
    CU_ASSERT(ancestor_matcher_get_mean_traceback_size(&am) > 0);
    CU_ASSERT(ancestor_matcher_get_total_memory(&am) > 0);
    ancestor_matcher_print_state(&am, _devnull);

    ret = ancestor_matcher_find_path(
        &am, 2, 1, h, match, &path_length, left, right, parent);
    CU_ASSERT_EQUAL(ret, TSLS_ERR_BAD_SITE_RANGE);

    ret = compressed_matrix_alloc(&fm, 4, 1024);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = compressed_matrix_alloc(&bm, 4, 1024);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = ancestor_matcher_forward(&am, h, &fm);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = ancestor_matcher_backward(&am, h, fm.normalisation_factor, &bm);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT(compressed_matrix_get_total_values(&fm) > 0);

    /* The Python tests check the values against a dense implementation;
     * here we just check that the outputs are populated. */
    for (j = 0; j < 4; j++) {
        CU_ASSERT(fm.normalisation_factor[j] > 0);
        total = 0;
        for (k = 0; k < fm.num_values[j]; k++) {
            for (l = 0; l < bm.num_values[j]; l++) {
                if (fm.nodes[j][k] == bm.nodes[j][l]) {
                    total += fm.values[j][k] * bm.values[j][l];
                }
            }
        }
        CU_ASSERT(total > 0);
    }
    compressed_matrix_free(&fm);
    compressed_matrix_free(&bm);

    ancestor_matcher_free(&am);
    matcher_indexes_free(&mi);
    tsk_table_collection_free(&tables);
}

static void
test_bad_inputs(void)
{
    int ret;
    tsk_table_collection_t tables;
    matcher_indexes_t mi;
    ancestor_matcher_t am;
    compressed_matrix_t cm;
    double rho[] = { 0.01, 0.01, 0.01, 0.01 };
    double mu[] = { 0, 0, 0, 0 };
    double norm[] = { 1, 1, 0, 1 };
    allele_t h[] = { 1, 1, 2, 0 };
    allele_t bad_h[] = { 1, 4, 2, 0 };
    allele_t impossible_h[] = { 3, 3, 3, 3 };
    allele_t match[4];
    tsk_id_t left[4], right[4], parent[4];
    size_t path_length;

    build_forest(&tables);
    ret = matcher_indexes_alloc(&mi, &tables, num_alleles, ancestral_state, derived_state, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = ancestor_matcher_alloc(&am, &mi, rho, mu, DBL_MIN, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = ancestor_matcher_find_path(
        &am, 0, 4, bad_h, match, &path_length, left, right, parent);
    CU_ASSERT_EQUAL(ret, TSLS_ERR_BAD_HAPLOTYPE_ALLELE);
    ret = ancestor_matcher_find_path(
        &am, 0, 4, impossible_h, match, &path_length, left, right, parent);
    CU_ASSERT_EQUAL(ret, TSLS_ERR_MATCH_IMPOSSIBLE_EXTREME_MUTATION_PROBA);

    ret = compressed_matrix_alloc(&cm, 4, 1024);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = ancestor_matcher_forward(&am, impossible_h, &cm);
    CU_ASSERT_EQUAL(ret, TSLS_ERR_MATCH_IMPOSSIBLE);
    ret = ancestor_matcher_backward(&am, h, norm, &cm);
    CU_ASSERT_EQUAL(ret, TSLS_ERR_BAD_NORMALISATION_FACTOR);
    compressed_matrix_free(&cm);

    ret = compressed_matrix_alloc(&cm, 3, 1024);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = ancestor_matcher_forward(&am, h, &cm);
    CU_ASSERT_EQUAL(ret, TSLS_ERR_BAD_SITE_RANGE);
    compressed_matrix_free(&cm);

    ancestor_matcher_free(&am);
    matcher_indexes_free(&mi);
    tsk_table_collection_free(&tables);
}

static void
test_strerror(void)
{
    int j;
    const char *msg;

    for (j = 0; j < 100; j++) {
        msg = tsls_strerror(-j);
        CU_ASSERT_FATAL(msg != NULL);
        CU_ASSERT(strlen(msg) > 0);
    }
}

static int
tsls_suite_init(void)
{
    _devnull = fopen("/dev/null", "w");
    if (_devnull == NULL) {
        return CUE_SINIT_FAILED;
    }
    return CUE_SUCCESS;
}

static int
tsls_suite_cleanup(void)
{
    if (_devnull != NULL) {
        fclose(_devnull);
    }
    return CUE_SUCCESS;
}

static void
handle_cunit_error(void)
{
    fprintf(stderr, "CUnit error occured: %d: %s\n", CU_get_error(),
        CU_get_error_msg());
    exit(EXIT_FAILURE);
}

int
main(int argc, char **argv)
{
    int ret;
    CU_pTest test;
    CU_pSuite suite;
    CU_TestInfo tests[] = {
        { "test_matcher_indexes_errors", test_matcher_indexes_errors },
        { "test_forest_matching", test_forest_matching },
        { "test_bad_inputs", test_bad_inputs },
        { "test_strerror", test_strerror },
        { NULL, NULL },
    };

    if (CUE_SUCCESS != CU_initialize_registry()) {
        handle_cunit_error();
    }
    suite = CU_add_suite("tsls", tsls_suite_init, tsls_suite_cleanup);
    if (NULL == suite) {
        handle_cunit_error();
    }
    for (ret = 0; tests[ret].pName != NULL; ret++) {
        if (NULL == CU_add_test(suite, tests[ret].pName, tests[ret].pTestFunc)) {
            handle_cunit_error();
        }
    }
    CU_basic_set_mode(CU_BRM_VERBOSE);
    if (argc == 1) {
        CU_basic_run_tests();
    } else {
        test = CU_get_test_by_name(argv[1], suite);
        if (test == NULL) {
            printf("Test '%s' not found\n", argv[1]);
        } else {
            CU_basic_run_test(suite, test);
        }
    }
    ret = CU_get_number_of_tests_failed() != 0;
    CU_cleanup_registry();
    return ret;
}
