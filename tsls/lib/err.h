#ifndef __TSLS_ERR_H__
#define __TSLS_ERR_H__

// clang-format off
#define TSLS_ERR_GENERIC                                    -1
#define TSLS_ERR_NO_MEMORY                                  -2
#define TSLS_ERR_MATCH_IMPOSSIBLE                           -3
#define TSLS_ERR_MATCH_IMPOSSIBLE_EXTREME_MUTATION_PROBA    -4
#define TSLS_ERR_MATCH_IMPOSSIBLE_ZERO_RECOMB_PRECISION     -5
#define TSLS_ERR_BAD_HAPLOTYPE_ALLELE                       -6
#define TSLS_ERR_BAD_ANCESTRAL_STATE                        -7
#define TSLS_ERR_BAD_MUTATION_DERIVED_STATE                 -8
#define TSLS_ERR_BAD_MUTATION_NODE                          -9
#define TSLS_ERR_BAD_MUTATION_SITE                          -10
#define TSLS_ERR_BAD_EDGE_NODE                              -11
#define TSLS_ERR_BAD_COORDINATE                             -12
#define TSLS_ERR_BAD_SITE_RANGE                             -13
#define TSLS_ERR_BAD_NORMALISATION_FACTOR                   -14
// clang-format on

#ifdef __GNUC__
#define WARN_UNUSED    __attribute__((warn_unused_result))
#define unlikely(expr) __builtin_expect(!!(expr), 0)
#define likely(expr)   __builtin_expect(!!(expr), 1)
#else
/* On windows we don't do any perf related stuff */
#define WARN_UNUSED
#define restrict
#define unlikely(expr) (expr)
#define likely(expr)   (expr)
#endif

const char *tsls_strerror(int err);

#endif /*__TSLS_ERR_H__*/
