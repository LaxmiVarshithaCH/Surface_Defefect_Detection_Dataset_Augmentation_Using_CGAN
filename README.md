pip install -r requirements.txt

python src/preprocess_surface_images.py

python src/train_cgan_surface.py

python src/inference_surface_cgan.py

python src/defect_classifier_train.py

python src/defect_classifier_eval.py

python src/monitor_surface_cgan.py

uvicorn src.api_surface_cgan:app --reload

streamlit run src/app_surface_cgan.py