# Hate Speech Adapter Fusion Repository

This repo contains resources and source code for adapter fusion on hate speech data.

## Source code to train adapters in each language:
 - *[stella_eng_hate_train_adapter_en.py](stella_eng_hate_train_adapter_en.py)* - English adapter training and evaluation, takes many days due to 360k large dataset
 - *[stella_pl_hate_train_adapter.py](stella_pl_hate_train_adapter.py)* - Polish adapter training

## Adapter fusion implementations:
 - *[stella_hate_adapter_fusion.py](stella_hate_adapter_fusion.py)* - implementations of Layer-weighted task-vector scaling, Layer-selective interpolation, Head-only transfer and Knowledge distillation
 - *[stella_hate_adapter_fusion_gating.py](stella_hate_adapter_fusion_gating.py)* - Gated logit fusion
 - *[stella_hate_weight_merge_ties.py](stella_hate_weight_merge_ties.py)* - TIES + DARE method
 - *[stella_hate_weight_merge_ties_fisher.py](stella_hate_weight_merge_ties_fisher.py)* - Fisher-weighted TIES + DARE
