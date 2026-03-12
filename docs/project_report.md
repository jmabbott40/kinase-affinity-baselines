# When Do Complex Models Beat Simple Baselines for Kinase Inhibitor Affinity Prediction?

## 1. Introduction

*TODO: Motivate the scientific question. Discuss the gap between published model performance and real-world utility. Explain why rigorous evaluation matters for drug discovery.*

## 2. Dataset Construction

*TODO: Describe the data curation pipeline, inclusion criteria, standardization, and quality analysis. Reference the data card for details.*

## 3. Baseline Model Comparison

*TODO: Present results for RF, XGBoost, ElasticNet, and MLP across all split strategies. Discuss which models work best and why.*

## 4. Impact of Splitting Strategy

*TODO: Show how performance degrades from random → scaffold → target split. Discuss what this means for claims of model generalization.*

## 5. Uncertainty Quantification

*TODO: Present calibration analysis. Show error-uncertainty correlation and selective prediction results. Discuss when model predictions can be trusted.*

## 6. Error Analysis

*TODO: Analyze failure modes — activity cliffs, rare scaffolds, noisy measurements. Discuss what makes predictions hard.*

## 7. Case Study: [Target Family TBD]

*TODO: Deep dive on a specific kinase subfamily. Compound-level analysis of good and bad predictions.*

## 8. Conclusions and Future Directions

*TODO: Summarize findings. Outline plans for advanced models (GNN, protein embeddings) and how they will be evaluated using the same framework.*

## References

- Mendez et al. (2019) ChEMBL: towards direct deposition of bioassay data. *Nucleic Acids Research*, 47(D1), D930-D940.
