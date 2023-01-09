---
language: en
license: mit
tags:
- text classification
- fact checking
datasets:
- mwong/fever-evidence-related
- mwong/climate-evidence-related
widget:
- text: "Earth’s changing climate is a critical issue and poses the risk of significant environmental, social and economic disruptions around the globe.</s></s>Because of fears of climate change and adverse effects of drilling explosions and oil spills in the Gulf of Mexico, legislation has been considered, and governmental regulations and orders have been issued, which, combined with the local economic and employment conditions caused by both, could materially adversely impact the oil and gas industries and the economic health of areas in which a significant number of our stores are located."
  example_title: "Evidence related to claim"
metrics: f1
---

# ClimateRoberta

ClimateRoberta is a classifier model that predicts if climate related evidence is related to query claim. The model achieved F1 score of 80.13% with test dataset "mwong/climate-evidence-related". Using pretrained roberta-base model, the classifier head is trained on Fever dataset and adapted to climate domain using ClimateFever dataset.