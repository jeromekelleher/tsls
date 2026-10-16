#include "err.h"

const char *
tsls_strerror(int err)
{
    const char *ret = "Unknown error";

    switch (err) {
        case 0:
            ret = "Normal exit condition. This is not an error!";
            break;
        case TSLS_ERR_GENERIC:
            ret = "Generic tsls error - please file a bug report.";
            break;
        case TSLS_ERR_NO_MEMORY:
            ret = "Out of memory";
            break;
        case TSLS_ERR_MATCH_IMPOSSIBLE:
            ret = "Unexpected failure to find matching haplotype; please open "
                  "an issue on GitHub";
            break;
        case TSLS_ERR_MATCH_IMPOSSIBLE_EXTREME_MUTATION_PROBA:
            ret = "Cannot find match: the specified mismatch probability is "
                  "0 or 1 and no matches are possible with these parameters";
            break;
        case TSLS_ERR_MATCH_IMPOSSIBLE_ZERO_RECOMB_PRECISION:
            ret = "Cannot find match: the specified recombination probability is "
                  "zero and no matches could be found";
            break;
        case TSLS_ERR_BAD_HAPLOTYPE_ALLELE:
            ret = "Haplotype allele must be missing (-1) or >= 0 and < num_alleles";
            break;
        case TSLS_ERR_BAD_ANCESTRAL_STATE:
            ret = "Ancestral state must be >= 0 and < num_alleles";
            break;
        case TSLS_ERR_BAD_MUTATION_DERIVED_STATE:
            ret = "Mutation derived state must be >= 0 and < num_alleles";
            break;
        case TSLS_ERR_BAD_MUTATION_NODE:
            ret = "Bad mutation: node out of bounds";
            break;
        case TSLS_ERR_BAD_MUTATION_SITE:
            ret = "Bad mutation: site out of bounds";
            break;
        case TSLS_ERR_BAD_EDGE_NODE:
            ret = "Bad edge: parent or child node out of bounds";
            break;
        case TSLS_ERR_BAD_COORDINATE:
            ret = "Edge and site coordinates must be integers";
            break;
        case TSLS_ERR_BAD_SITE_RANGE:
            ret = "Bad site range: must have 0 <= start < end <= num_sites";
            break;
        case TSLS_ERR_BAD_NORMALISATION_FACTOR:
            ret = "Forward normalisation factors must be > 0";
            break;
    }
    return ret;
}
