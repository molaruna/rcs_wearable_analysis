# rcs_wearable_analysis

This code uses CSV files from the [rcs_lfp_analysis repo](https://github.com/molaruna/rcs_lfp_analysis) 
to decode dyskinetic state using spectral features from chronic human brain recordings collected using the Summit RC+S system.

## Getting started
Directory structure:
```bash
├── code
├── data
│   ├── neural
│   │   ├── coh
│   │   └── psd
│   ├── notes
│   └── wearable
│       ├── apple
│       └── pkg
├── plots
└── tables
```

This code uses Python 3.8.3

## Data
Neural spectral data are derived from the Summit RC+S neurostimulator (Medtronic). See the [rcs_lfp_analysis repo](https://github.com/molaruna/rcs_lfp_analysis) for more information. Wearable data are derived from the PKG system and Apple watch. PKG data are available on [UCSF Box](https://ucsf.app.box.com/folder/0), and apple watch data are available on [Rune Labs](https://app.runelabs.io/patients). You can request access from me. Then, these data are organized within the repo directory structure. Specifically, these are the CSV files that are required in the data directory:<br/>
* data/neural/psd/subj_id_psd_total.csv
* data/neural/psd/subj_id_psd_flat.csv
* data/neural/psd/subj_id_psd_peak_rm.csv
* data/neural/coh/subj_id_msc.csv
* data/wearable/apple/subj_id_dys_scores.csv
* data/sidedness/wearable/apple/subj_id_tremor_scores.csv
* data/wearable/apple/subj_id_accel_data.csv
* data/wearable/pkg/subj_id_dk_bk_scores.csv
* data/wearable/pkg/subj_id_dose_times.csv
* data/notes/subj_id_selfrate_dys_scores.csv
* data/notes/subj_id_sleep_times.csv
  
## Analysis
* correlation between wearable scores & spectral features
* baseline linear regression decoding
* machine learning decoding
  
## License
This software is open source and under an MIT license.
  
