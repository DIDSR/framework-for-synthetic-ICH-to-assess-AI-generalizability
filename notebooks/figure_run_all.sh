#!/usr/bin/env bash
jupyter nbconvert --to notebook --execute fig3-5_view_six_examples.ipynb
jupyter nbconvert --to notebook --execute fig4_compare_synth_real_distributions.ipynb
jupyter nbconvert --to notebook --execute fig6_kV_mA_variation.ipynb
jupyter nbconvert --to notebook --execute fig7_create_multimodel_ROC.ipynb
jupyter nbconvert --to notebook --execute fig8_compare_false_negatives.ipynb
rm -rf *.nbconvert.*