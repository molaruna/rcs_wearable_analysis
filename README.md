# rcs_wearable_analysis

This code uses neural data that have been preprocessed using [rcs_lfp_analysis repo](https://github.com/molaruna/rcs_lfp_analysis) 
into overlaid power spectra (details in [readme]https://github.com/molaruna/rcs_lfp_analysis/blob/main/README.md)) to continuously decode motor signs using spectral features from chronic human brain recordings collected using the Summit RC+S system, in addition to sensor wearable measures of motor signs. 

## Getting started
This code uses Python 3.8.3


## Data
Neural electrophysiological data are derived from the Summit RC+S neurostimulator (Medtronic). See the [rcs_lfp_analysis repo](https://github.com/molaruna/rcs_lfp_analysis) for more information. Wearable sensor data are derived from the PKG system and Apple watch. PKG data are available on [UCSF Box](https://ucsf.app.box.com/folder/0), and apple watch data are available on [Rune Labs](https://app.runelabs.io/patients). You can request access from me. Then, these data are organized within the repo directory structure. Specifically, these are the CSV file keywords that are used by the function:<br/>
* "psd" processes as neural electrophysiological spectral data
* "apple" processes as apple watch sensor data
* "pkg" processes as pkg sensor data
* "accel" processes  as accelerometry data
* "symptom" processes as patient ratings of symptoms

Note: Data directories are created with respect to (1) patient and (2) sense channel configurations. 
 
## Analysis
First, the neural electrophysiological data is represented within the frequency domain using super imposed power spectra of 1-10 minute durations. Then, principle components of each neural spectral feature over time are visualized to derive independently fluctuating components of the spectral data. Next, correlations are derived between each neural spectral feature and wearable motor score over time. The top correlations of each motor score are visualized within the time-series domain and cross-correlated for any temporal lags. Then,  principal components within the frequency domain are used to predict motor scores with linear regression as a baseline assessment. Lastly, a bidirectional long short-term memory (blstm) is trained to non-linearly predict motor scores.
  
## License
This software is open source and under an MIT license.
