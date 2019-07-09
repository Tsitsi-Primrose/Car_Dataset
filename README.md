# Keras CNNs on Stanford car dataset trained on Google Cloud ML Engine

For training locally: MODEL_DIR is any location where you would like to store your model.
gcloud ai-platform local train     --module-name trainer.task     --package-path trainer/     --job-dir $MODEL_DIR 

gcloud ai-platform jobs submit training JOB_NAME         --package-path $TRAINER_PACKAGE_PATH         --module-name $MAIN_TRAINER_MODULE         --region $REGION  --python-version 2.7    --runtime-version 1.13    --job-dir $JOB_DIR

where TRAINER_PACKAGE_PATH=trainer; MAIN_TRAINER_MODULE=trainer.task; REGION=chosen region; JOB_DIR is whatever location you would like to store the model


