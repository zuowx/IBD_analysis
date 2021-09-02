# IBD_analysis

In `Analysis_funcs.py`
  - `shannon` is used to calculate alpha diversity (shannon index).
  - `multi_boxplot` is used to draw boxplot between cases and controls with the p value of ks test.
  - `draw_heatmap` is used to display the significant of taxa (used to generate Figure 3 and 5).
  - `run_rm_cv` is used to run random forest classifier with multiple times.
  - `predict_score_cv` is used to run random forest classifier with k-fold cross validation (used to generate Table 4).

In `Analysis_funcs.R`
  - `plot.shannon` is used to plot the boxplot of alpha diversity labeled with the p value of Wilcoxon Rank Sum test. (used to generate Figure 2)
  - `plot.pcoa` is used to plot the PCoA plot of beta diversity. (used to generate Table 3)
  - `lm.analysis` is used to conduct linear regression on alpha diversity. (used to generate Figure 2)
  - `mrm.analysis` is used to conduct multiple regression on matrices on the Bray-Curtis distances between samples. (used to generate Table 4)
