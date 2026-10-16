
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <structmember.h>
#include <float.h>
#include <stdbool.h>

#include "lib/tsls.h"

#if PY_MAJOR_VERSION >= 3
#define IS_PY3K
#endif

#define MODULE_DOC "Low-level tsls HMM interface."

static PyObject *TslsLibraryError;
static PyObject *TslsMatchImpossible;

#include "tskit_lwt_interface.h"

typedef struct {
    PyObject_HEAD
    matcher_indexes_t *matcher_indexes;
} MatcherIndexes;

typedef struct {
    PyObject_HEAD
    ancestor_matcher_t *ancestor_matcher;
    MatcherIndexes *matcher_indexes;
} AncestorMatcher;

static void
handle_library_error(int err)
{
    if (err == TSLS_ERR_NO_MEMORY) {
        PyErr_NoMemory();
    } else if (err == TSLS_ERR_MATCH_IMPOSSIBLE
               || err == TSLS_ERR_MATCH_IMPOSSIBLE_EXTREME_MUTATION_PROBA
               || err == TSLS_ERR_MATCH_IMPOSSIBLE_ZERO_RECOMB_PRECISION) {
        PyErr_Format(TslsMatchImpossible, "%s", tsls_strerror(err));
    } else {
        PyErr_Format(TslsLibraryError, "%s", tsls_strerror(err));
    }
}

static FILE *
make_file(PyObject *fileobj, const char *mode)
{
    FILE *ret = NULL;
    FILE *file = NULL;
    int fileobj_fd, new_fd;

    fileobj_fd = PyObject_AsFileDescriptor(fileobj);
    if (fileobj_fd == -1) {
        goto out;
    }
    new_fd = dup(fileobj_fd);
    if (new_fd == -1) {
        PyErr_SetFromErrno(PyExc_OSError);
        goto out;
    }
    file = fdopen(new_fd, mode);
    if (file == NULL) {
        (void) close(new_fd);
        PyErr_SetFromErrno(PyExc_OSError);
        goto out;
    }
    ret = file;
out:
    return ret;
}

/*===================================================================
 * MatcherIndexes
 *===================================================================
 */

static void
MatcherIndexes_dealloc(MatcherIndexes *self)
{
    if (self->matcher_indexes != NULL) {
        matcher_indexes_free(self->matcher_indexes);
        PyMem_Free(self->matcher_indexes);
        self->matcher_indexes = NULL;
    }
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static int
MatcherIndexes_init(MatcherIndexes *self, PyObject *args, PyObject *kwds)
{
    int ret = -1;
    int err;
    LightweightTableCollection *tables;
    PyObject *num_alleles_obj = NULL;
    PyObject *ancestral_state_obj = NULL;
    PyObject *derived_state_obj = NULL;
    PyArrayObject *num_alleles_array = NULL;
    PyArrayObject *ancestral_state_array = NULL;
    PyArrayObject *derived_state_array = NULL;
    tsk_size_t *num_alleles_data = NULL;
    allele_t *ancestral_state_data = NULL;
    allele_t *derived_state_data = NULL;
    static char *kwlist[]
        = { "tables", "num_alleles", "ancestral_state", "derived_state", NULL };

    self->matcher_indexes = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|OOO", kwlist,
            &LightweightTableCollectionType, &tables, &num_alleles_obj,
            &ancestral_state_obj, &derived_state_obj)) {
        goto out;
    }
    if (LightweightTableCollection_check_state(tables) != 0) {
        goto out;
    }
    if (num_alleles_obj != NULL && num_alleles_obj != Py_None) {
        num_alleles_array = (PyArrayObject *) PyArray_FROMANY(
            num_alleles_obj, NPY_UINT64, 1, 1, NPY_ARRAY_IN_ARRAY);
        if (num_alleles_array == NULL) {
            goto out;
        }
        if (PyArray_DIM(num_alleles_array, 0)
            != (npy_intp) tables->tables->sites.num_rows) {
            PyErr_SetString(PyExc_ValueError, "num_alleles must have length num_sites");
            goto out;
        }
        num_alleles_data = PyArray_DATA(num_alleles_array);
    }
    if (ancestral_state_obj != NULL && ancestral_state_obj != Py_None) {
        ancestral_state_array = (PyArrayObject *) PyArray_FROMANY(
            ancestral_state_obj, NPY_INT8, 1, 1, NPY_ARRAY_IN_ARRAY);
        if (ancestral_state_array == NULL) {
            goto out;
        }
        if (PyArray_DIM(ancestral_state_array, 0)
            != (npy_intp) tables->tables->sites.num_rows) {
            PyErr_SetString(
                PyExc_ValueError, "ancestral_state must have length num_sites");
            goto out;
        }
        ancestral_state_data = PyArray_DATA(ancestral_state_array);
    }
    if (derived_state_obj != NULL && derived_state_obj != Py_None) {
        derived_state_array = (PyArrayObject *) PyArray_FROMANY(
            derived_state_obj, NPY_INT8, 1, 1, NPY_ARRAY_IN_ARRAY);
        if (derived_state_array == NULL) {
            goto out;
        }
        if (PyArray_DIM(derived_state_array, 0)
            != (npy_intp) tables->tables->mutations.num_rows) {
            PyErr_SetString(
                PyExc_ValueError, "derived_state must have length num_mutations");
            goto out;
        }
        derived_state_data = PyArray_DATA(derived_state_array);
    }

    self->matcher_indexes = PyMem_Calloc(1, sizeof(*self->matcher_indexes));
    if (self->matcher_indexes == NULL) {
        PyErr_NoMemory();
        goto out;
    }
    err = matcher_indexes_alloc(self->matcher_indexes, tables->tables,
        num_alleles_data, ancestral_state_data, derived_state_data, 0);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = 0;
out:
    Py_XDECREF(num_alleles_array);
    Py_XDECREF(ancestral_state_array);
    Py_XDECREF(derived_state_array);
    return ret;
}

static int
MatcherIndexes_check_state(MatcherIndexes *self)
{
    int ret = 0;
    if (self->matcher_indexes == NULL) {
        PyErr_SetString(PyExc_SystemError, "MatcherIndexes not initialised");
        ret = -1;
    }
    return ret;
}

static PyObject *
MatcherIndexes_print_state(MatcherIndexes *self, PyObject *args)
{
    PyObject *ret = NULL;
    PyObject *fileobj;
    FILE *file = NULL;

    if (MatcherIndexes_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTuple(args, "O", &fileobj)) {
        goto out;
    }
    file = make_file(fileobj, "w");
    if (file == NULL) {
        goto out;
    }
    matcher_indexes_print_state(self->matcher_indexes, file);
    ret = Py_BuildValue("");
out:
    if (file != NULL) {
        (void) fclose(file);
    }
    return ret;
}

static PyMethodDef MatcherIndexes_methods[] = {
    { "print_state", (PyCFunction) MatcherIndexes_print_state, METH_VARARGS,
        "Low-level debug method" },
    { NULL } /* Sentinel */
};

static PyTypeObject MatcherIndexesType = {
    // clang-format off
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_tsls.MatcherIndexes",
    .tp_basicsize = sizeof(MatcherIndexes),
    .tp_dealloc = (destructor) MatcherIndexes_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "MatcherIndexes objects",
    .tp_methods = MatcherIndexes_methods,
    .tp_init = (initproc) MatcherIndexes_init,
    .tp_new = PyType_GenericNew,
    // clang-format on
};

/*===================================================================
 * AncestorMatcher
 *===================================================================
 */

static int
AncestorMatcher_check_state(AncestorMatcher *self)
{
    int ret = 0;
    if (self->ancestor_matcher == NULL) {
        PyErr_SetString(PyExc_SystemError, "AncestorMatcher not initialised");
        ret = -1;
    }
    return ret;
}

static void
AncestorMatcher_dealloc(AncestorMatcher *self)
{
    if (self->ancestor_matcher != NULL) {
        ancestor_matcher_free(self->ancestor_matcher);
        PyMem_Free(self->ancestor_matcher);
        self->ancestor_matcher = NULL;
    }
    Py_XDECREF(self->matcher_indexes);
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static int
AncestorMatcher_init(AncestorMatcher *self, PyObject *args, PyObject *kwds)
{
    int ret = -1;
    int err;
    int extended_checks = 0;
    int weight_by_n = 1;
    static char *kwlist[] = { "matcher_indexes", "recombination", "mismatch",
        "extended_checks", "likelihood_threshold", "weight_by_n", NULL };
    MatcherIndexes *matcher_indexes = NULL;
    PyObject *recombination = NULL;
    PyObject *mismatch = NULL;
    PyArrayObject *recombination_array = NULL;
    PyArrayObject *mismatch_array = NULL;
    npy_intp *shape;
    double likelihood_threshold = DBL_MIN;
    int flags = 0;

    self->ancestor_matcher = NULL;
    self->matcher_indexes = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!OO|idi", kwlist, &MatcherIndexesType,
            &matcher_indexes, &recombination, &mismatch, &extended_checks,
            &likelihood_threshold, &weight_by_n)) {
        goto out;
    }
    self->matcher_indexes = matcher_indexes;
    Py_INCREF(self->matcher_indexes);
    if (MatcherIndexes_check_state(self->matcher_indexes) != 0) {
        goto out;
    }

    recombination_array = (PyArrayObject *) PyArray_FromAny(recombination,
        PyArray_DescrFromType(NPY_FLOAT64), 1, 1, NPY_ARRAY_IN_ARRAY, NULL);
    if (recombination_array == NULL) {
        goto out;
    }
    shape = PyArray_DIMS(recombination_array);
    if (shape[0] != (npy_intp) matcher_indexes->matcher_indexes->num_sites) {
        PyErr_SetString(
            PyExc_ValueError, "Size of recombination array must be num_sites");
        goto out;
    }
    mismatch_array = (PyArrayObject *) PyArray_FromAny(
        mismatch, PyArray_DescrFromType(NPY_FLOAT64), 1, 1, NPY_ARRAY_IN_ARRAY, NULL);
    if (mismatch_array == NULL) {
        goto out;
    }
    shape = PyArray_DIMS(mismatch_array);
    if (shape[0] != (npy_intp) matcher_indexes->matcher_indexes->num_sites) {
        PyErr_SetString(PyExc_ValueError, "Size of mismatch array must be num_sites");
        goto out;
    }

    self->ancestor_matcher = PyMem_Malloc(sizeof(ancestor_matcher_t));
    if (self->ancestor_matcher == NULL) {
        PyErr_NoMemory();
        goto out;
    }
    if (extended_checks) {
        flags |= TSLS_EXTENDED_CHECKS;
    }
    if (!weight_by_n) {
        flags |= TSLS_DISABLE_WEIGHT_BY_N;
    }
    err = ancestor_matcher_alloc(self->ancestor_matcher,
        self->matcher_indexes->matcher_indexes, PyArray_DATA(recombination_array),
        PyArray_DATA(mismatch_array), likelihood_threshold, flags);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = 0;
out:
    Py_XDECREF(recombination_array);
    Py_XDECREF(mismatch_array);
    return ret;
}

static PyObject *
AncestorMatcher_find_path(AncestorMatcher *self, PyObject *args, PyObject *kwds)
{
    int err;
    PyObject *ret = NULL;
    static char *kwlist[] = { "haplotype", "start", "end", NULL };
    PyObject *haplotype = NULL;
    PyArrayObject *haplotype_array = NULL;
    npy_intp *shape;
    size_t num_edges;
    int start, end;
    PyArrayObject *left = NULL;
    PyArrayObject *right = NULL;
    PyArrayObject *parent = NULL;
    PyArrayObject *match = NULL;
    npy_intp dims[1];

    if (AncestorMatcher_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "Oii", kwlist, &haplotype, &start, &end)) {
        goto out;
    }
    haplotype_array
        = (PyArrayObject *) PyArray_FROM_OTF(haplotype, NPY_INT8, NPY_ARRAY_IN_ARRAY);
    if (haplotype_array == NULL) {
        goto out;
    }
    if (PyArray_NDIM(haplotype_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto out;
    }
    shape = PyArray_DIMS(haplotype_array);
    if (shape[0] != (npy_intp) self->ancestor_matcher->num_sites) {
        PyErr_SetString(PyExc_ValueError, "Incorrect size for input haplotype.");
        goto out;
    }

    dims[0] = self->ancestor_matcher->num_sites;
    left = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_UINT32);
    right = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_UINT32);
    parent = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_INT32);
    match = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_INT8);
    if (left == NULL || right == NULL || parent == NULL || match == NULL) {
        goto out;
    }

    Py_BEGIN_ALLOW_THREADS
    err = ancestor_matcher_find_path(self->ancestor_matcher, (tsk_id_t) start,
        (tsk_id_t) end, (allele_t *) PyArray_DATA(haplotype_array),
        (allele_t *) PyArray_DATA(match), &num_edges, PyArray_DATA(left),
        PyArray_DATA(right), PyArray_DATA(parent));
    Py_END_ALLOW_THREADS
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue(
        "(kNNNN)", (unsigned long) num_edges, left, right, parent, match);
    if (ret == NULL) {
        goto out;
    }
    left = NULL;
    right = NULL;
    parent = NULL;
    match = NULL;
out:
    Py_XDECREF(haplotype_array);
    Py_XDECREF(match);
    Py_XDECREF(left);
    Py_XDECREF(right);
    Py_XDECREF(parent);
    return ret;
}

/* Run the forward (if forward_norm is NULL) or backward algorithm, and
 * return the compressed matrix as a tuple of flat arrays
 * (normalisation_factor, offsets, nodes, values), where the values at site
 * j are nodes[offsets[j]: offsets[j + 1]] and similarly for values. */
static PyObject *
AncestorMatcher_run_compressed_matrix(
    AncestorMatcher *self, PyObject *haplotype, PyObject *forward_norm)
{
    int err;
    PyObject *ret = NULL;
    PyArrayObject *haplotype_array = NULL;
    PyArrayObject *forward_norm_array = NULL;
    PyArrayObject *norm = NULL;
    PyArrayObject *offsets = NULL;
    PyArrayObject *nodes = NULL;
    PyArrayObject *values = NULL;
    const size_t num_sites = self->ancestor_matcher->num_sites;
    compressed_matrix_t cm;
    bool cm_allocated = false;
    npy_intp dims[1];
    uint64_t *offsets_data;
    tsk_id_t *nodes_data;
    double *values_data;
    size_t j, k, offset;

    haplotype_array
        = (PyArrayObject *) PyArray_FROMANY(haplotype, NPY_INT8, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (haplotype_array == NULL) {
        goto out;
    }
    if (PyArray_DIM(haplotype_array, 0) != (npy_intp) num_sites) {
        PyErr_SetString(PyExc_ValueError, "Incorrect size for input haplotype.");
        goto out;
    }
    if (forward_norm != NULL) {
        forward_norm_array = (PyArrayObject *) PyArray_FROMANY(
            forward_norm, NPY_FLOAT64, 1, 1, NPY_ARRAY_IN_ARRAY);
        if (forward_norm_array == NULL) {
            goto out;
        }
        if (PyArray_DIM(forward_norm_array, 0) != (npy_intp) num_sites) {
            PyErr_SetString(
                PyExc_ValueError, "normalisation_factor must have length num_sites");
            goto out;
        }
    }
    err = compressed_matrix_alloc(&cm, num_sites,
        2 * (self->ancestor_matcher->num_nodes + 1) * sizeof(double));
    cm_allocated = true;
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }

    Py_BEGIN_ALLOW_THREADS
    if (forward_norm_array == NULL) {
        err = ancestor_matcher_forward(self->ancestor_matcher,
            (allele_t *) PyArray_DATA(haplotype_array), &cm);
    } else {
        err = ancestor_matcher_backward(self->ancestor_matcher,
            (allele_t *) PyArray_DATA(haplotype_array),
            (double *) PyArray_DATA(forward_norm_array), &cm);
    }
    Py_END_ALLOW_THREADS
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }

    dims[0] = (npy_intp) num_sites;
    norm = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_FLOAT64);
    dims[0] = (npy_intp) num_sites + 1;
    offsets = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_UINT64);
    dims[0] = (npy_intp) compressed_matrix_get_total_values(&cm);
    nodes = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_INT32);
    values = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_FLOAT64);
    if (norm == NULL || offsets == NULL || nodes == NULL || values == NULL) {
        goto out;
    }
    memcpy(PyArray_DATA(norm), cm.normalisation_factor, num_sites * sizeof(double));
    offsets_data = PyArray_DATA(offsets);
    nodes_data = PyArray_DATA(nodes);
    values_data = PyArray_DATA(values);
    offset = 0;
    for (j = 0; j < num_sites; j++) {
        offsets_data[j] = offset;
        for (k = 0; k < cm.num_values[j]; k++) {
            nodes_data[offset] = cm.nodes[j][k];
            values_data[offset] = cm.values[j][k];
            offset++;
        }
    }
    offsets_data[num_sites] = offset;

    ret = Py_BuildValue("(NNNN)", norm, offsets, nodes, values);
    if (ret == NULL) {
        goto out;
    }
    norm = NULL;
    offsets = NULL;
    nodes = NULL;
    values = NULL;
out:
    if (cm_allocated) {
        compressed_matrix_free(&cm);
    }
    Py_XDECREF(haplotype_array);
    Py_XDECREF(forward_norm_array);
    Py_XDECREF(norm);
    Py_XDECREF(offsets);
    Py_XDECREF(nodes);
    Py_XDECREF(values);
    return ret;
}

static PyObject *
AncestorMatcher_forward_matrix(AncestorMatcher *self, PyObject *args, PyObject *kwds)
{
    PyObject *ret = NULL;
    static char *kwlist[] = { "haplotype", NULL };
    PyObject *haplotype = NULL;

    if (AncestorMatcher_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &haplotype)) {
        goto out;
    }
    ret = AncestorMatcher_run_compressed_matrix(self, haplotype, NULL);
out:
    return ret;
}

static PyObject *
AncestorMatcher_backward_matrix(AncestorMatcher *self, PyObject *args, PyObject *kwds)
{
    PyObject *ret = NULL;
    static char *kwlist[] = { "haplotype", "normalisation_factor", NULL };
    PyObject *haplotype = NULL;
    PyObject *forward_norm = NULL;

    if (AncestorMatcher_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "OO", kwlist, &haplotype, &forward_norm)) {
        goto out;
    }
    ret = AncestorMatcher_run_compressed_matrix(self, haplotype, forward_norm);
out:
    return ret;
}

static PyObject *
AncestorMatcher_get_traceback(AncestorMatcher *self, PyObject *args)
{
    PyObject *ret = NULL;
    unsigned long site;
    node_state_list_t *list;
    PyObject *dict = NULL;
    PyObject *key = NULL;
    PyObject *value = NULL;
    int j;

    if (AncestorMatcher_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTuple(args, "k", &site)) {
        goto out;
    }
    if (site >= self->ancestor_matcher->num_sites) {
        PyErr_SetString(PyExc_ValueError, "site out of range");
        goto out;
    }
    dict = PyDict_New();
    if (dict == NULL) {
        goto out;
    }
    list = &self->ancestor_matcher->traceback[site];
    for (j = 0; j < list->size; j++) {
        key = Py_BuildValue("k", (unsigned long) list->node[j]);
        value = Py_BuildValue("i", (int) list->recombination_required[j]);
        if (key == NULL || value == NULL) {
            goto out;
        }
        if (PyDict_SetItem(dict, key, value) != 0) {
            goto out;
        }
        Py_DECREF(key);
        key = NULL;
        Py_DECREF(value);
        value = NULL;
    }
    ret = dict;
    dict = NULL;
out:
    Py_XDECREF(key);
    Py_XDECREF(value);
    Py_XDECREF(dict);
    return ret;
}

static PyObject *
AncestorMatcher_get_mean_traceback_size(AncestorMatcher *self, void *closure)
{
    PyObject *ret = NULL;

    if (AncestorMatcher_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue(
        "d", ancestor_matcher_get_mean_traceback_size(self->ancestor_matcher));
out:
    return ret;
}

static PyObject *
AncestorMatcher_get_total_memory(AncestorMatcher *self, void *closure)
{
    PyObject *ret = NULL;
    unsigned long val;

    if (AncestorMatcher_check_state(self) != 0) {
        goto out;
    }

#if defined(TSI_NO_ATOMICS)
    /* Without atomics, return an obviously wrong value */
    val = (unsigned long) PY_SSIZE_T_MAX;
#else
    val = ancestor_matcher_get_total_memory(self->ancestor_matcher);
#endif
    ret = Py_BuildValue("k", val);

out:
    return ret;
}

static PyGetSetDef AncestorMatcher_getsetters[] = {
    { "mean_traceback_size", (getter) AncestorMatcher_get_mean_traceback_size, NULL,
        "The mean size of the traceback per site." },
    { "total_memory", (getter) AncestorMatcher_get_total_memory, NULL,
        "The total amount of memory used by this matcher." },
    { NULL } /* Sentinel */
};

static PyMethodDef AncestorMatcher_methods[] = {
    { "find_path", (PyCFunction) AncestorMatcher_find_path, METH_VARARGS | METH_KEYWORDS,
        "Returns a best match path for the specified haplotype through the ancestors." },
    { "forward_matrix", (PyCFunction) AncestorMatcher_forward_matrix,
        METH_VARARGS | METH_KEYWORDS,
        "Runs the forward algorithm and returns the compressed matrix arrays." },
    { "backward_matrix", (PyCFunction) AncestorMatcher_backward_matrix,
        METH_VARARGS | METH_KEYWORDS,
        "Runs the backward algorithm and returns the compressed matrix arrays." },
    { "get_traceback", (PyCFunction) AncestorMatcher_get_traceback, METH_VARARGS,
        "Returns the traceback likelihood dictionary at the specified site." },
    { NULL } /* Sentinel */
};

static PyTypeObject AncestorMatcherType = {
    // clang-format off
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_tsls.AncestorMatcher",
    .tp_basicsize = sizeof(AncestorMatcher),
    .tp_dealloc = (destructor) AncestorMatcher_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "AncestorMatcher objects",
    .tp_methods = AncestorMatcher_methods,
    .tp_getset = AncestorMatcher_getsetters,
    .tp_init = (initproc) AncestorMatcher_init,
    .tp_new = PyType_GenericNew,
    // clang-format on
};

/*===================================================================
 * Module level code.
 *===================================================================
 */

static PyMethodDef tsls_methods[] = {
    { NULL } /* Sentinel */
};

/* Initialisation code supports Python 2.x and 3.x. The framework uses the
 * recommended structure from http://docs.python.org/howto/cporting.html.
 * I've ignored the point about storing state in globals, as the examples
 * from the Python documentation still use this idiom.
 */

#if PY_MAJOR_VERSION >= 3

static struct PyModuleDef tslsmodule
    = { PyModuleDef_HEAD_INIT, "_tsls", /* name of module */
          MODULE_DOC,                      /* module documentation, may be NULL */
          -1, tsls_methods, NULL, NULL, NULL, NULL };

#define INITERROR return NULL

PyObject *
PyInit__tsls(void)

#else
#define INITERROR return

void
init_tsls(void)
#endif
{
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&tslsmodule);
#else
    PyObject *module = Py_InitModule3("_tsls", tsls_methods, MODULE_DOC);
#endif
    if (module == NULL) {
        INITERROR;
    }
    /* Initialise numpy */
    import_array();

    register_lwt_class(module);

    /* MatcherIndexes type */
    MatcherIndexesType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&MatcherIndexesType) < 0) {
        INITERROR;
    }
    Py_INCREF(&MatcherIndexesType);
    PyModule_AddObject(module, "MatcherIndexes", (PyObject *) &MatcherIndexesType);

    /* AncestorMatcher type */
    AncestorMatcherType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&AncestorMatcherType) < 0) {
        INITERROR;
    }
    Py_INCREF(&AncestorMatcherType);
    PyModule_AddObject(module, "AncestorMatcher", (PyObject *) &AncestorMatcherType);

    TslsLibraryError = PyErr_NewException("_tsls.LibraryError", NULL, NULL);
    Py_INCREF(TslsLibraryError);
    PyModule_AddObject(module, "LibraryError", TslsLibraryError);

    TslsMatchImpossible = PyErr_NewException("_tsls.MatchImpossible", NULL, NULL);
    Py_INCREF(TslsMatchImpossible);
    PyModule_AddObject(module, "MatchImpossible", TslsMatchImpossible);

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}
