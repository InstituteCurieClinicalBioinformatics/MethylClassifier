# MethylClassifier

## Requirements

List of python packages:
* pandas==2.2.3
* scikit-learn==1.6.0
* umap-learn==0.5.7

## Build the classifiers
```python
python3 methylation_classifier.py build --CpGs 10KCpGs.txt --betaValueMatrix 954RefSamples_400kCpG_BetaVal.csv --samplesLabel samplesheet.tsv --saveDetailsReport report.tsv
```

## Predict label of new samples
```python
python3 methylation_classifier.py  predict --CpGs 10KCpGs.txt  --betaValueMatrix SamplesToPredict_400kCpG_BetaVal.csv
```
