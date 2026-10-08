# tsls
Li and Stephens HMM on a tree sequence

tsls will replace the two separate Li and Stephens HMM implementations in
[tsinfer](https://github.com/tskit-dev/tsinfer) and
[tskit](https://github.com/tskit-dev/tskit) with a single package.

## Current status

As a first step, the two existing implementations have been copied into
this repository. Each builds and runs its test suite on its own:

- [`tsinfer-hmm/`](tsinfer-hmm): the tsinfer ancestor matching engine,
  copied from tsinfer at commit `53d7363b`. Python package `tsinfer_hmm`,
  extension module `_tsinfer_hmm`.
- [`tskit-hmm/`](tskit-hmm): the tskit `LsHmm` implementation, copied from
  tskit at commit `0e38fc3d`. Python package `tskit_hmm`, extension module
  `_tskit_hmm`.

Both build against the tskit C library in the `git-submodules/tskit`
submodule, so clone with `--recurse-submodules` or run
`git submodule update --init`.

The git history records exactly what was copied from where. Each
directory was first added verbatim, then changed in separate commits to
make it build on its own.

## Licence

tsls is distributed under the MIT License. tsinfer is licensed under
GPL-3.0-or-later; the tsinfer code copied here was relicensed to MIT by
its author. See [tsinfer-hmm/LICENSING.md](tsinfer-hmm/LICENSING.md) for
details. The tskit code is MIT licensed.

## Development

In either `tsinfer-hmm/` or `tskit-hmm/`:

```
# Python tests
uv sync --group test
uv run pytest -n 4

# C tests (needs CUnit)
uvx meson setup build lib
ninja -C build
uvx meson test -C build
```
