#!/usr/bin/env bash
echo "Setting environment variables..."
export BASE_DIR=/projects01/didsr-aiml/jayse.weaver/insilicoich/
export STUDY_NAME=manuscript_100_280mA_wME
export EXPERIMENT_NAME=kVp_mA_variation
echo "Running all notebooks..."
jupyter nbconvert --to notebook --execute fig3-5_view_six_examples.ipynb
jupyter nbconvert --to notebook --execute fig4_compare_synth_real_distributions.ipynb
jupyter nbconvert --to notebook --execute fig6_kV_mA_variation.ipynb # uses EXPERIMENT_NAME for kVp/mA variation
jupyter nbconvert --to notebook --execute fig7_create_multimodel_ROC.ipynb
jupyter nbconvert --to notebook --execute fig8_compare_false_negatives.ipynb
rm -rf *.nbconvert.*
echo "Done."