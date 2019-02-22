# pyEEG

pyEEG is a library fo processing EEG data build mostly on top of MNE-py and scikit-learn. It allows anlaysis of raw data and generation of temporal response functions with continuous signals as stimuli or real-valued events (e.g. word-level features).

If on the netwrok of Imperial College, you can access the documentation here: [pyeeg-docs](http://bg-hw2512.bg.ic.ac.uk).

------

## TODOs

Out of my head, some things to finish now:

### Priority

- [x] Computing or loading envelopes (12/02/2019)
- [x] An object for all speech related features, aligne with segement (not necessarily word-level features!), e.g. what about pitch or envelope? (12/02/2019)
- [x] Getting the indices to align stim and repsonse shoul be doable independently of stim type (12/02/2019) **-> STILL TO BE TESTED?**
- [x] Format EEG for DNN input (12/02/2019) **-> check _chunk_data_ ?**
- [ ] Loading/computing syntactic features (12/02/2019)

> Release 1.0 on completion of all above

### Future enhancements

- [ ] Functional connectivity methods:
  - [ ] Estimate connectivity
  - [ ] Graph theory metrics (path length, clustering coeff.)
- [ ] Pipeline `pyRiemann` and `pyeeg` [this one](https://github.com/freole/pyeeg) into some workflows..

------

## Installation

### Dependencies

pyEEG requires:

- Python (>= 3.5)
- NumPy (>= 1.11.0)
- SciPy (>= 1.0.0)
- mne (>= 0.16)
- pandas (>= 0.23.0)
- scikit-learn (>= 0.20.0)
- matplotlib (>= 2.0)
- h5py (>= 2.8.0)

Install requirements:

```bash
pip install requirements.txt
```

To generate the doc, Python package `sphinx` (>= 1.1.0), `sphinx_rtd_theme` and `nbsphinx` are required (`sphinx` is installable from `conda` and the others from `pip`).

### User Installation

From terminal (or `conda` shell in Windows), `cd` in root directory of the library (directory containing `setup.py` file) and type:

To get the package installed only through symbolic links, namely so that you can modify the source code and use modified versions at will when importing the package in your python scripts do:

```bash
python setup.py develop
```

Otherwise, for a standard installation (but this will require to be installed if you need to install another version of the library):

```bash
python setup.py install
```

## Basic Examples

See files in [`examples/`](docs/source/examples/).

### Computing Envelope TRF and spatial map from CCA

See [examples/CCA_envelope.ipynb](docs/source/examples/CCA_envelope.ipynb)


### Computing Word-feature TRF

See [examples/TRF_wordonsets.ipynb](docs/source/examples/TRF_wordonsets.ipynb)

### Working with Word vectors

See [examples/import_WordVectors.ipynb](docs/source/examples/importWordVectors.ipynb)

## Documentation

The simplest way is to access it from Imperial College Network (or via VPN) [here](http://pyeeg-docs).
But you can also generate an _offline_ version, or a PDF file of all the docs by following the following instructions.

### Generate the documentation

To generate the documentation you will need `sphinx` to be installed in your Python environment. If it is not installed, install it with:

```bash
conda install sphinx
conda install -c conda-forge nbsphinx
pip install sphinx_rtd_theme
```

You can access the doc as HTML or PDF format.
To generate the documentation HTML pages, type in a terminal (does not work Windows!):

```bash
make doc
```

And for PDF version:

```bash
make docpdf
```

Then you can open the `docs/build/html/index.html` page in your favourite browser or open `docs/build/latex/pyEEG.pdf` in a PDF viewer.

**The PDF documentation can only be generated if `latex` and `latxmk` are present on the machine**

To clean files created during build process:

```bash
make clean
```