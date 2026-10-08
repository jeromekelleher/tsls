project('tskit-hmm', 'c', default_options: ['c_std=c99'])

# lib/subprojects/tskit is a symlink to the tskit C library in the
# git-submodules/tskit submodule at the root of the repository.
tskit_proj = subproject('tskit')
tskit_dep = tskit_proj.get_variable('tskit_dep')
kastore_dep = tskit_proj.get_variable('kastore_dep')

cc = meson.get_compiler('c')
m_dep = cc.find_library('m', required : false)
cunit_dep = dependency('cunit')

extra_c_args = [
    '-Wall', '-Wextra', '-Werror', '-Wpedantic', '-W',
    '-Wmissing-prototypes',  '-Wstrict-prototypes',
    '-Wconversion', '-Wshadow', '-Wpointer-arith', '-Wcast-align',
    '-Wcast-qual', '-Wwrite-strings', '-Wnested-externs',
    '-fshort-enums', '-fno-common']

# NOTE: the tskit static library also contains tskit's own copy of
# haplotype_matching.c. Our objects are linked first, so the archive
# member is never pulled in.
hmm_inc = include_directories('.')
hmm_lib = static_library('tskit_hmm',
    sources: ['haplotype_matching.c'], include_directories: hmm_inc,
    dependencies: [m_dep, tskit_dep], c_args: extra_c_args)

# We don't specify extra C args here as CUnit won't pass the checks.
test_lib = static_library('testlib',
    sources: ['tests/testlib.c'], dependencies: [cunit_dep, kastore_dep, tskit_dep])

test_haplotype_matching = executable('test_haplotype_matching',
    sources: ['tests/test_haplotype_matching.c'],
    include_directories: hmm_inc,
    link_with: [hmm_lib, test_lib],
    c_args: extra_c_args,
    dependencies: [cunit_dep, kastore_dep, tskit_dep])
test('haplotype_matching', test_haplotype_matching)
