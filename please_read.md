
Run with these environment
pyenv activate pyannote-env
python -m streamlit run new_app/app.py

pyenv activate pyannote-env
python -m streamlit run halal_product/app.py

Here is the things that we need to download, including accept all model gate 

hbredin/wespeaker-voxceleb-resnet34-LM


For halal
ultralytics
opencv-python-headless
pillow
numpy

pip install ultralytics opencv-python-headless pillow numpy


git checkout main_new
nohup streamlit run halal_product/app.py --server.address 0.0.0.0 --server.port 8502 > streamlit.log 2>&1 &
