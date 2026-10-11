/*
 * MIT License
 *
 * Copyright (c) 2019-2025 Tskit Developers
 * Copyright (c) 2015-2018 University of Oxford
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#include <numpy/arrayobject.h>

#include <structmember.h>
#include <float.h>

#include "kastore.h"
/* Include our copy of the HMM header before tskit.h, so that the include
 * guard stops tskit's own copy from being used. */
#include "haplotype_matching.h"
#include "tskit.h"

static PyObject *TskitHmmLibraryError;

#include "tskit_lwt_interface.h"

static void
handle_library_error(int err)
{
    if (err == TSK_ERR_NO_MEMORY) {
        PyErr_NoMemory();
    } else {
        PyErr_SetString(TskitHmmLibraryError, tsk_strerror(err));
    }
}

/*===================================================================
 * TreeSequence
 *===================================================================
 */

/* A minimal tree sequence type, built from a LightweightTableCollection,
 * standing in for the _tskit.TreeSequence that the HMM classes were
 * written against. */
typedef struct {
    PyObject_HEAD
    tsk_treeseq_t *tree_sequence;
} TreeSequence;

static int
TreeSequence_check_state(TreeSequence *self)
{
    int ret = 0;
    if (self->tree_sequence == NULL) {
        PyErr_SetString(PyExc_ValueError, "tree_sequence not initialised");
        ret = -1;
    }
    return ret;
}

static void
TreeSequence_dealloc(TreeSequence *self)
{
    if (self->tree_sequence != NULL) {
        tsk_treeseq_free(self->tree_sequence);
        PyMem_Free(self->tree_sequence);
        self->tree_sequence = NULL;
    }
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static int
TreeSequence_init(TreeSequence *self, PyObject *args, PyObject *kwds)
{
    int ret = -1;
    int err;
    LightweightTableCollection *tables = NULL;
    static char *kwlist[] = { "tables", NULL };

    self->tree_sequence = NULL;
    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "|O!", kwlist, &LightweightTableCollectionType, &tables)) {
        goto out;
    }
    if (tables == NULL) {
        /* Leave uninitialised */
        ret = 0;
        goto out;
    }
    if (LightweightTableCollection_check_state(tables) != 0) {
        goto out;
    }
    self->tree_sequence = PyMem_Calloc(1, sizeof(*self->tree_sequence));
    if (self->tree_sequence == NULL) {
        PyErr_NoMemory();
        goto out;
    }
    err = tsk_treeseq_init(
        self->tree_sequence, tables->tables, TSK_TS_INIT_BUILD_INDEXES);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = 0;
out:
    return ret;
}

static PyObject *
TreeSequence_get_num_nodes(TreeSequence *self, PyObject *Py_UNUSED(ignored))
{
    if (TreeSequence_check_state(self) != 0) {
        return NULL;
    }
    return Py_BuildValue(
        "n", (Py_ssize_t) tsk_treeseq_get_num_nodes(self->tree_sequence));
}

static PyObject *
TreeSequence_get_num_samples(TreeSequence *self, PyObject *Py_UNUSED(ignored))
{
    if (TreeSequence_check_state(self) != 0) {
        return NULL;
    }
    return Py_BuildValue(
        "n", (Py_ssize_t) tsk_treeseq_get_num_samples(self->tree_sequence));
}

static PyObject *
TreeSequence_get_num_sites(TreeSequence *self, PyObject *Py_UNUSED(ignored))
{
    if (TreeSequence_check_state(self) != 0) {
        return NULL;
    }
    return Py_BuildValue(
        "n", (Py_ssize_t) tsk_treeseq_get_num_sites(self->tree_sequence));
}

static PyMethodDef TreeSequence_methods[] = {
    { .ml_name = "get_num_nodes",
        .ml_meth = (PyCFunction) TreeSequence_get_num_nodes,
        .ml_flags = METH_NOARGS,
        .ml_doc = "Returns the number of nodes in the tree sequence." },
    { .ml_name = "get_num_samples",
        .ml_meth = (PyCFunction) TreeSequence_get_num_samples,
        .ml_flags = METH_NOARGS,
        .ml_doc = "Returns the number of samples in the tree sequence." },
    { .ml_name = "get_num_sites",
        .ml_meth = (PyCFunction) TreeSequence_get_num_sites,
        .ml_flags = METH_NOARGS,
        .ml_doc = "Returns the number of sites in the tree sequence." },
    { NULL } /* Sentinel */
};

static PyTypeObject TreeSequenceType = {
    // clang-format off
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_tskit_hmm.TreeSequence",
    .tp_basicsize = sizeof(TreeSequence),
    .tp_dealloc = (destructor) TreeSequence_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "TreeSequence objects",
    .tp_methods = TreeSequence_methods,
    .tp_init = (initproc) TreeSequence_init,
    .tp_new = PyType_GenericNew,
    // clang-format on
};

typedef struct {
    PyObject_HEAD
    TreeSequence *tree_sequence;
    tsk_ls_hmm_t *ls_hmm;
} LsHmm;

typedef struct {
    PyObject_HEAD
    TreeSequence *tree_sequence;
    tsk_compressed_matrix_t *compressed_matrix;
} CompressedMatrix;

typedef struct {
    PyObject_HEAD
    TreeSequence *tree_sequence;
    tsk_viterbi_matrix_t *viterbi_matrix;
} ViterbiMatrix;

/* TODO: this should really be a dict we're returning */
static PyObject *
convert_compressed_matrix_site(tsk_compressed_matrix_t *matrix, unsigned int site)
{
    PyObject *ret = NULL;
    PyObject *list = NULL;
    PyObject *item = NULL;
    tsk_size_t j, num_values;

    if (site >= matrix->num_sites) {
        PyErr_SetString(PyExc_ValueError, "Site index out of bounds");
        goto out;
    }

    num_values = matrix->num_transitions[site];
    list = PyList_New(num_values);
    if (list == NULL) {
        goto out;
    }
    for (j = 0; j < num_values; j++) {
        item = Py_BuildValue("id", matrix->nodes[site][j], matrix->values[site][j]);
        if (item == NULL) {
            goto out;
        }
        PyList_SET_ITEM(list, j, item);
        item = NULL;
    }
    ret = list;
    list = NULL;
out:
    Py_XDECREF(item);
    Py_XDECREF(list);
    return ret;
}

static PyObject *
decode_compressed_matrix(tsk_compressed_matrix_t *matrix)
{
    int err;
    PyObject *ret = NULL;
    PyArrayObject *decoded = NULL;
    npy_intp dims[2];

    dims[0] = tsk_treeseq_get_num_sites(matrix->tree_sequence);
    dims[1] = tsk_treeseq_get_num_samples(matrix->tree_sequence);
    decoded = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_FLOAT64);
    if (decoded == NULL) {
        goto out;
    }
    err = tsk_compressed_matrix_decode(matrix, PyArray_DATA(decoded));
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = (PyObject *) decoded;
    decoded = NULL;
out:
    Py_XDECREF(decoded);
    return ret;
}

/*===================================================================
 * CompressedMatrix
 *===================================================================
 */

static int
CompressedMatrix_check_state(CompressedMatrix *self)
{
    int ret = -1;
    if (self->compressed_matrix == NULL) {
        PyErr_SetString(PyExc_SystemError, "CompressedMatrix not initialised");
        goto out;
    }
    ret = 0;
out:
    return ret;
}

static void
CompressedMatrix_dealloc(CompressedMatrix *self)
{
    if (self->compressed_matrix != NULL) {
        tsk_compressed_matrix_free(self->compressed_matrix);
        PyMem_Free(self->compressed_matrix);
        self->compressed_matrix = NULL;
    }
    Py_XDECREF(self->tree_sequence);
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static int
CompressedMatrix_init(CompressedMatrix *self, PyObject *args, PyObject *kwds)
{
    int ret = -1;
    int err;
    static char *kwlist[] = { "tree_sequence", "block_size", NULL };
    TreeSequence *tree_sequence = NULL;
    Py_ssize_t block_size = 0;

    self->compressed_matrix = NULL;
    self->tree_sequence = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|n", kwlist, &TreeSequenceType,
            &tree_sequence, &block_size)) {
        goto out;
    }
    self->tree_sequence = tree_sequence;
    Py_INCREF(self->tree_sequence);
    if (TreeSequence_check_state(self->tree_sequence) != 0) {
        goto out;
    }
    self->compressed_matrix = PyMem_Malloc(sizeof(tsk_compressed_matrix_t));
    if (self->compressed_matrix == NULL) {
        PyErr_NoMemory();
        goto out;
    }
    memset(self->compressed_matrix, 0, sizeof(tsk_compressed_matrix_t));

    err = tsk_compressed_matrix_init(
        self->compressed_matrix, self->tree_sequence->tree_sequence, block_size, 0);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = 0;
out:
    return ret;
}

static PyObject *
CompressedMatrix_get_num_sites(CompressedMatrix *self, void *closure)
{
    PyObject *ret = NULL;

    if (CompressedMatrix_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("n", (Py_ssize_t) self->compressed_matrix->num_sites);
out:
    return ret;
}

static PyObject *
CompressedMatrix_get_normalisation_factor(CompressedMatrix *self, void *closure)
{
    PyObject *ret = NULL;
    PyArrayObject *array;
    tsk_size_t num_sites;
    npy_intp dims;

    if (CompressedMatrix_check_state(self) != 0) {
        goto out;
    }
    num_sites = self->compressed_matrix->num_sites;
    dims = (npy_intp) num_sites;
    array = (PyArrayObject *) PyArray_EMPTY(1, &dims, NPY_FLOAT64, 0);
    if (array == NULL) {
        goto out;
    }
    memcpy(PyArray_DATA(array), self->compressed_matrix->normalisation_factor,
        num_sites * sizeof(*self->compressed_matrix->normalisation_factor));
    ret = (PyObject *) array;
out:
    return ret;
}

static PyObject *
CompressedMatrix_get_num_transitions(CompressedMatrix *self, void *closure)
{
    PyObject *ret = NULL;
    PyArrayObject *array;
    tsk_size_t num_sites;
    npy_intp dims;

    if (CompressedMatrix_check_state(self) != 0) {
        goto out;
    }
    num_sites = self->compressed_matrix->num_sites;
    dims = (npy_intp) num_sites;
    array = (PyArrayObject *) PyArray_EMPTY(1, &dims, NPY_UINT64, 0);
    if (array == NULL) {
        goto out;
    }
    memcpy(PyArray_DATA(array), self->compressed_matrix->num_transitions,
        num_sites * sizeof(*self->compressed_matrix->num_transitions));
    ret = (PyObject *) array;
out:
    return ret;
}

static PyObject *
CompressedMatrix_get_site(CompressedMatrix *self, PyObject *args)
{
    PyObject *ret = NULL;
    unsigned int site;

    if (CompressedMatrix_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTuple(args, "I", &site)) {
        goto out;
    }
    ret = convert_compressed_matrix_site(self->compressed_matrix, site);
out:
    return ret;
}

static PyObject *
CompressedMatrix_decode(CompressedMatrix *self)
{
    PyObject *ret = NULL;
    if (CompressedMatrix_check_state(self) != 0) {
        goto out;
    }
    ret = decode_compressed_matrix(self->compressed_matrix);
out:
    return ret;
}

static PyGetSetDef CompressedMatrix_getsetters[] = {
    { .name = "num_sites",
        .get = (getter) CompressedMatrix_get_num_sites,
        .doc = "The number of sites." },
    { .name = "normalisation_factor",
        .get = (getter) CompressedMatrix_get_normalisation_factor,
        .doc = "The per-site normalisation factor." },
    { .name = "num_transitions",
        .get = (getter) CompressedMatrix_get_num_transitions,
        .doc = "The per-site number of transitions in the compressed matrix." },
    { NULL } /* Sentinel */
};

static PyMethodDef CompressedMatrix_methods[] = {
    { .ml_name = "get_site",
        .ml_meth = (PyCFunction) CompressedMatrix_get_site,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Returns the list of (node, value) tuples for the specified site." },
    { .ml_name = "decode",
        .ml_meth = (PyCFunction) CompressedMatrix_decode,
        .ml_flags = METH_NOARGS,
        .ml_doc = "Returns the full decoded forward matrix." },
    { NULL } /* Sentinel */
};

static PyTypeObject CompressedMatrixType = {
    // clang-format off
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_tskit_hmm.CompressedMatrix",
    .tp_basicsize = sizeof(CompressedMatrix),
    .tp_dealloc = (destructor) CompressedMatrix_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "CompressedMatrix objects",
    .tp_methods = CompressedMatrix_methods,
    .tp_getset = CompressedMatrix_getsetters,
    .tp_init = (initproc) CompressedMatrix_init,
    .tp_new = PyType_GenericNew,
    // clang-format on
};

/*===================================================================
 * ViterbiMatrix
 *===================================================================
 */

static int
ViterbiMatrix_check_state(ViterbiMatrix *self)
{
    int ret = -1;
    if (self->viterbi_matrix == NULL) {
        PyErr_SetString(PyExc_SystemError, "ViterbiMatrix not initialised");
        goto out;
    }
    ret = 0;
out:
    return ret;
}

static void
ViterbiMatrix_dealloc(ViterbiMatrix *self)
{
    if (self->viterbi_matrix != NULL) {
        tsk_viterbi_matrix_free(self->viterbi_matrix);
        PyMem_Free(self->viterbi_matrix);
        self->viterbi_matrix = NULL;
    }
    Py_XDECREF(self->tree_sequence);
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static int
ViterbiMatrix_init(ViterbiMatrix *self, PyObject *args, PyObject *kwds)
{
    int ret = -1;
    int err;
    static char *kwlist[] = { "tree_sequence", "num_records", NULL };
    TreeSequence *tree_sequence = NULL;
    Py_ssize_t num_records = 0;

    self->viterbi_matrix = NULL;
    self->tree_sequence = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|n", kwlist, &TreeSequenceType,
            &tree_sequence, &num_records)) {
        goto out;
    }
    self->tree_sequence = tree_sequence;
    Py_INCREF(self->tree_sequence);
    if (TreeSequence_check_state(self->tree_sequence) != 0) {
        goto out;
    }
    self->viterbi_matrix = PyMem_Malloc(sizeof(tsk_viterbi_matrix_t));
    if (self->viterbi_matrix == NULL) {
        PyErr_NoMemory();
        goto out;
    }
    memset(self->viterbi_matrix, 0, sizeof(tsk_viterbi_matrix_t));

    err = tsk_viterbi_matrix_init(
        self->viterbi_matrix, self->tree_sequence->tree_sequence, num_records, 0);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = 0;
out:
    return ret;
}

static PyObject *
ViterbiMatrix_traceback(ViterbiMatrix *self)
{
    PyObject *ret = NULL;
    PyArrayObject *path = NULL;
    npy_intp dims;
    int err;

    if (ViterbiMatrix_check_state(self) != 0) {
        goto out;
    }
    dims = self->viterbi_matrix->matrix.num_sites;
    path = (PyArrayObject *) PyArray_SimpleNew(1, &dims, NPY_INT32);
    if (path == NULL) {
        goto out;
    }

    err = tsk_viterbi_matrix_traceback(self->viterbi_matrix, PyArray_DATA(path), 0);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = (PyObject *) path;
    path = NULL;
out:
    Py_XDECREF(path);
    return ret;
}

static PyObject *
ViterbiMatrix_get_num_sites(ViterbiMatrix *self, void *closure)
{
    PyObject *ret = NULL;

    if (ViterbiMatrix_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("n", (Py_ssize_t) self->viterbi_matrix->matrix.num_sites);
out:
    return ret;
}

/* NOTE: We're doing something pretty ugly here in that we're duplicating the
 * methods from the CompressedMatrix class to provide access to the
 * viterbi_matrix struct's embedded compressed_matrix. It would be more
 * elegant if the ViterbiMatrix class had a CompressedMatrix member,
 * but the memory management is tricky, so it doesn't seem worth the
 * hassle.
 */

static PyObject *
ViterbiMatrix_get_normalisation_factor(ViterbiMatrix *self, void *closure)
{
    PyObject *ret = NULL;
    PyArrayObject *array;
    tsk_size_t num_sites;
    npy_intp dims;

    if (ViterbiMatrix_check_state(self) != 0) {
        goto out;
    }
    num_sites = self->viterbi_matrix->matrix.num_sites;
    dims = (npy_intp) num_sites;
    array = (PyArrayObject *) PyArray_EMPTY(1, &dims, NPY_FLOAT64, 0);
    if (array == NULL) {
        goto out;
    }
    memcpy(PyArray_DATA(array), self->viterbi_matrix->matrix.normalisation_factor,
        num_sites * sizeof(*self->viterbi_matrix->matrix.normalisation_factor));
    ret = (PyObject *) array;
out:
    return ret;
}

static PyObject *
ViterbiMatrix_get_num_transitions(ViterbiMatrix *self, void *closure)
{
    PyObject *ret = NULL;
    PyArrayObject *array;
    tsk_size_t num_sites;
    npy_intp dims;

    if (ViterbiMatrix_check_state(self) != 0) {
        goto out;
    }
    num_sites = self->viterbi_matrix->matrix.num_sites;
    dims = (npy_intp) num_sites;

    array = (PyArrayObject *) PyArray_EMPTY(1, &dims, NPY_UINT64, 0);
    if (array == NULL) {
        goto out;
    }
    memcpy(PyArray_DATA(array), self->viterbi_matrix->matrix.num_transitions,
        num_sites * sizeof(*self->viterbi_matrix->matrix.num_transitions));
    ret = (PyObject *) array;
out:
    return ret;
}

static PyObject *
ViterbiMatrix_get_site(ViterbiMatrix *self, PyObject *args)
{
    PyObject *ret = NULL;
    unsigned int site;

    if (ViterbiMatrix_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTuple(args, "I", &site)) {
        goto out;
    }
    ret = convert_compressed_matrix_site(&self->viterbi_matrix->matrix, site);
out:
    return ret;
}

static PyObject *
ViterbiMatrix_decode(ViterbiMatrix *self)
{
    PyObject *ret = NULL;
    if (ViterbiMatrix_check_state(self) != 0) {
        goto out;
    }
    ret = decode_compressed_matrix(&self->viterbi_matrix->matrix);
out:
    return ret;
}

static PyGetSetDef ViterbiMatrix_getsetters[] = {
    { .name = "num_sites",
        .get = (getter) ViterbiMatrix_get_num_sites,
        .doc = "The number of sites." },
    { .name = "normalisation_factor",
        .get = (getter) ViterbiMatrix_get_normalisation_factor,
        .doc = "The per-site normalisation factor." },
    { .name = "num_transitions",
        .get = (getter) ViterbiMatrix_get_num_transitions,
        .doc = "The per-site number of transitions in the compressed matrix." },
    { NULL } /* Sentinel */
};

static PyMethodDef ViterbiMatrix_methods[] = {
    { .ml_name = "traceback",
        .ml_meth = (PyCFunction) ViterbiMatrix_traceback,
        .ml_flags = METH_NOARGS,
        .ml_doc = "Returns a path for a given haplotype." },
    { .ml_name = "get_site",
        .ml_meth = (PyCFunction) ViterbiMatrix_get_site,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Returns the list of (node, value) tuples for the specified site." },
    { .ml_name = "decode",
        .ml_meth = (PyCFunction) ViterbiMatrix_decode,
        .ml_flags = METH_NOARGS,
        .ml_doc = "Returns the full decoded forward matrix." },
    { NULL } /* Sentinel */
};

static PyTypeObject ViterbiMatrixType = {
    // clang-format off
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_tskit_hmm.ViterbiMatrix",
    .tp_basicsize = sizeof(ViterbiMatrix),
    .tp_dealloc = (destructor) ViterbiMatrix_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "ViterbiMatrix objects",
    .tp_methods = ViterbiMatrix_methods,
    .tp_getset = ViterbiMatrix_getsetters,
    .tp_init = (initproc) ViterbiMatrix_init,
    .tp_new = PyType_GenericNew,
    // clang-format on
};

/*===================================================================
 * LsHmm
 *===================================================================
 */

static int
LsHmm_check_state(LsHmm *self)
{
    int ret = -1;
    if (self->ls_hmm == NULL) {
        PyErr_SetString(PyExc_SystemError, "LsHmm not initialised");
        goto out;
    }
    ret = 0;
out:
    return ret;
}

static void
LsHmm_dealloc(LsHmm *self)
{
    if (self->ls_hmm != NULL) {
        tsk_ls_hmm_free(self->ls_hmm);
        PyMem_Free(self->ls_hmm);
        self->ls_hmm = NULL;
    }
    Py_XDECREF(self->tree_sequence);
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static int
LsHmm_init(LsHmm *self, PyObject *args, PyObject *kwds)
{
    int ret = -1;
    int err;
    static char *kwlist[] = { "tree_sequence", "recombination_rate", "mutation_rate",
        "precision", "acgt_alleles", NULL };
    PyObject *recombination_rate = NULL;
    PyArrayObject *recombination_rate_array = NULL;
    PyObject *mutation_rate = NULL;
    PyArrayObject *mutation_rate_array = NULL;
    TreeSequence *tree_sequence = NULL;
    unsigned int precision = 23;
    int acgt_alleles = 0;
    tsk_flags_t options = 0;
    npy_intp *shape, num_sites;

    self->ls_hmm = NULL;
    self->tree_sequence = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!OO|Ii", kwlist, &TreeSequenceType,
            &tree_sequence, &recombination_rate, &mutation_rate, &precision,
            &acgt_alleles)) {
        goto out;
    }
    self->tree_sequence = tree_sequence;
    Py_INCREF(self->tree_sequence);
    if (TreeSequence_check_state(self->tree_sequence) != 0) {
        goto out;
    }
    self->ls_hmm = PyMem_Malloc(sizeof(tsk_ls_hmm_t));
    if (self->ls_hmm == NULL) {
        PyErr_NoMemory();
        goto out;
    }
    memset(self->ls_hmm, 0, sizeof(tsk_ls_hmm_t));

    num_sites = (npy_intp) tsk_treeseq_get_num_sites(self->tree_sequence->tree_sequence);
    recombination_rate_array = (PyArrayObject *) PyArray_FROMANY(
        recombination_rate, NPY_FLOAT64, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (recombination_rate_array == NULL) {
        goto out;
    }
    shape = PyArray_DIMS(recombination_rate_array);
    if (shape[0] != num_sites) {
        PyErr_SetString(PyExc_ValueError,
            "recombination_rate array must have dimension (num_sites,)");
        goto out;
    }
    mutation_rate_array = (PyArrayObject *) PyArray_FROMANY(
        mutation_rate, NPY_FLOAT64, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (mutation_rate_array == NULL) {
        goto out;
    }
    shape = PyArray_DIMS(mutation_rate_array);
    if (shape[0] != num_sites) {
        PyErr_SetString(
            PyExc_ValueError, "mutation_rate array must have dimension (num_sites,)");
        goto out;
    }
    if (acgt_alleles) {
        options |= TSK_ALLELES_ACGT;
    }

    err = tsk_ls_hmm_init(self->ls_hmm, self->tree_sequence->tree_sequence,
        PyArray_DATA(recombination_rate_array), PyArray_DATA(mutation_rate_array),
        options);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    tsk_ls_hmm_set_precision(self->ls_hmm, precision);
    ret = 0;
out:
    Py_XDECREF(recombination_rate_array);
    Py_XDECREF(mutation_rate_array);
    return ret;
}

static PyObject *
LsHmm_forward_matrix(LsHmm *self, PyObject *args)
{
    int err;
    PyObject *ret = NULL;
    PyObject *haplotype = NULL;
    CompressedMatrix *compressed_matrix = NULL;
    PyArrayObject *haplotype_array = NULL;
    npy_intp *shape, num_sites;

    if (LsHmm_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTuple(
            args, "OO!", &haplotype, &CompressedMatrixType, &compressed_matrix)) {
        goto out;
    }
    num_sites = (npy_intp) tsk_treeseq_get_num_sites(self->tree_sequence->tree_sequence);
    haplotype_array = (PyArrayObject *) PyArray_FROMANY(
        haplotype, NPY_INT32, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (haplotype_array == NULL) {
        goto out;
    }
    shape = PyArray_DIMS(haplotype_array);
    if (shape[0] != num_sites) {
        PyErr_SetString(
            PyExc_ValueError, "haplotype array must have dimension (num_sites,)");
        goto out;
    }
    err = tsk_ls_hmm_forward(self->ls_hmm, PyArray_DATA(haplotype_array),
        compressed_matrix->compressed_matrix, TSK_NO_INIT);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("");
out:
    Py_XDECREF(haplotype_array);
    return ret;
}

static PyObject *
LsHmm_backward_matrix(LsHmm *self, PyObject *args)
{
    int err;
    PyObject *ret = NULL;
    PyObject *haplotype = NULL;
    PyObject *forward_norm = NULL;
    CompressedMatrix *compressed_matrix = NULL;
    PyArrayObject *haplotype_array = NULL;
    PyArrayObject *forward_norm_array = NULL;
    npy_intp *shape, num_sites;

    if (LsHmm_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTuple(args, "OOO!", &haplotype, &forward_norm, &CompressedMatrixType,
            &compressed_matrix)) {
        goto out;
    }
    num_sites = (npy_intp) tsk_treeseq_get_num_sites(self->tree_sequence->tree_sequence);

    haplotype_array = (PyArrayObject *) PyArray_FROMANY(
        haplotype, NPY_INT32, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (haplotype_array == NULL) {
        goto out;
    }
    shape = PyArray_DIMS(haplotype_array);
    if (shape[0] != num_sites) {
        PyErr_SetString(
            PyExc_ValueError, "haplotype array must have dimension (num_sites,)");
        goto out;
    }

    forward_norm_array = (PyArrayObject *) PyArray_FROMANY(
        forward_norm, NPY_FLOAT64, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (forward_norm_array == NULL) {
        goto out;
    }
    shape = PyArray_DIMS(forward_norm_array);
    if (shape[0] != num_sites) {
        PyErr_SetString(
            PyExc_ValueError, "forward_norm array must have dimension (num_sites,)");
        goto out;
    }
    err = tsk_ls_hmm_backward(self->ls_hmm, PyArray_DATA(haplotype_array),
        PyArray_DATA(forward_norm_array), compressed_matrix->compressed_matrix,
        TSK_NO_INIT);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("");
out:
    Py_XDECREF(haplotype_array);
    Py_XDECREF(forward_norm_array);
    return ret;
}

static PyObject *
LsHmm_viterbi_matrix(LsHmm *self, PyObject *args)
{
    int err;
    PyObject *ret = NULL;
    PyObject *haplotype = NULL;
    ViterbiMatrix *viterbi_matrix = NULL;
    PyArrayObject *haplotype_array = NULL;
    npy_intp *shape, num_sites;

    if (LsHmm_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTuple(
            args, "OO!", &haplotype, &ViterbiMatrixType, &viterbi_matrix)) {
        goto out;
    }
    num_sites = (npy_intp) tsk_treeseq_get_num_sites(self->tree_sequence->tree_sequence);
    haplotype_array = (PyArrayObject *) PyArray_FROMANY(
        haplotype, NPY_INT32, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (haplotype_array == NULL) {
        goto out;
    }
    shape = PyArray_DIMS(haplotype_array);
    if (shape[0] != num_sites) {
        PyErr_SetString(
            PyExc_ValueError, "haplotype array must have dimension (num_sites,)");
        goto out;
    }
    err = tsk_ls_hmm_viterbi(self->ls_hmm, PyArray_DATA(haplotype_array),
        viterbi_matrix->viterbi_matrix, TSK_NO_INIT);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("");
out:
    Py_XDECREF(haplotype_array);
    return ret;
}

static PyMethodDef LsHmm_methods[] = {
    { .ml_name = "forward_matrix",
        .ml_meth = (PyCFunction) LsHmm_forward_matrix,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Returns the tree encoded forward matrix for a given haplotype" },
    { .ml_name = "backward_matrix",
        .ml_meth = (PyCFunction) LsHmm_backward_matrix,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Returns the tree encoded backward matrix for a given haplotype" },
    { .ml_name = "viterbi_matrix",
        .ml_meth = (PyCFunction) LsHmm_viterbi_matrix,
        .ml_flags = METH_VARARGS,
        .ml_doc = "Returns the tree encoded Viterbi matrix for a given haplotype" },
    { NULL } /* Sentinel */
};

static PyTypeObject LsHmmType = {
    // clang-format off
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_tskit_hmm.LsHmm",
    .tp_basicsize = sizeof(LsHmm),
    .tp_dealloc = (destructor) LsHmm_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "LsHmm objects",
    .tp_methods = LsHmm_methods,
    .tp_init = (initproc) LsHmm_init,
    .tp_new = PyType_GenericNew,
    // clang-format on
};

/*===================================================================
 * Module level code.
 *===================================================================
 */

static struct PyModuleDef tskithmmmodule = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "_tskit_hmm",
    .m_doc = "Low-level tskit HMM interface.",
    .m_size = -1,
};

PyMODINIT_FUNC
PyInit__tskit_hmm(void)
{
    PyObject *module = PyModule_Create(&tskithmmmodule);
    if (module == NULL) {
        return NULL;
    }
    import_array();

    if (register_lwt_class(module) != 0) {
        return NULL;
    }

    /* TreeSequence type */
    if (PyType_Ready(&TreeSequenceType) < 0) {
        return NULL;
    }
    Py_INCREF(&TreeSequenceType);
    PyModule_AddObject(module, "TreeSequence", (PyObject *) &TreeSequenceType);

    /* CompressedMatrix type */
    if (PyType_Ready(&CompressedMatrixType) < 0) {
        return NULL;
    }
    Py_INCREF(&CompressedMatrixType);
    PyModule_AddObject(module, "CompressedMatrix", (PyObject *) &CompressedMatrixType);

    /* ViterbiMatrix type */
    if (PyType_Ready(&ViterbiMatrixType) < 0) {
        return NULL;
    }
    Py_INCREF(&ViterbiMatrixType);
    PyModule_AddObject(module, "ViterbiMatrix", (PyObject *) &ViterbiMatrixType);

    /* LsHmm type */
    if (PyType_Ready(&LsHmmType) < 0) {
        return NULL;
    }
    Py_INCREF(&LsHmmType);
    PyModule_AddObject(module, "LsHmm", (PyObject *) &LsHmmType);

    TskitHmmLibraryError
        = PyErr_NewException("_tskit_hmm.LibraryError", NULL, NULL);
    Py_INCREF(TskitHmmLibraryError);
    PyModule_AddObject(module, "LibraryError", TskitHmmLibraryError);

    return module;
}
