# rcs_wearable_analysis

This code uses CSV files from the [rcs_lfp_analysis repo](https://github.com/molaruna/rcs_lfp_analysis) 
to decode dyskinetic state using spectral features from chronic human brain recordings collected using the Summit RC+S system.

## Getting started

Directory structure:
```bash
├── code
├── data
│   └── RCS02
│       └── L
│           ├── neural
│           │   ├── coh
│           │   └── psd
│           ├── notes
│           └── wearable
│               ├── apple
│               └── pkg
├── plots
└── tables
```

This code uses Python 3.8.3


## Data
Spectral data are derived from the Summit RC+S neurostimulator (Medtronic). 
See the [rcs_lfp_analysis repo](https://github.com/molaruna/rcs_lfp_analysis) for more information. 
Specifically, these are the CSV files that are required in the data directory

