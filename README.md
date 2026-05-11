# PREDICT
PREDICT is an ML-based capability to predict chemical toxicity directly from chemical sensor spectra. 
Research was done in collaboration with MIT Lincoln Laboratories.

Terminology Guide: 
- In file names conditions are labeled numberically as follows:
    - 1: ChemNet
    - 2: Toxicicty
    - 3: Morgan fingerprints
    - 4: Filtered Morgan Fingerprints
    - e1: Instrument group and ionization mode
    - e2: Collision energy level

Files with scripts to train models that use any of these conditions are labelled with the numbers of each condition used.
Data used saved as csv and parquet files in data folders. Processing using must be done to get the same format as data used to train the models themselves.

DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
This material is based upon work supported by the Department of the Air Force under Air Force Contract No. FA8702-15-D-0001 or FA8702-25-D-B002. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Department of the Air Force.
