# Changelog

## [0.1.0][2016-01-11]
### Changed
- add algorithm tests for corpus construction and model fitting
- remove dependency on Cython for intallation, the required .c and .cpp files are now included
- use py.test for testing
- removed dependency on C++11 features by using a different sparse matrix structure for corpus construction
- faster coocurrence matrix construction

### Removed
- max_map_size argument removed from Corpus.fit
