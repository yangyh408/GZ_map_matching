conda create -n map_match
conda activate map_match
conda config --env --add channels conda-forge
conda config --env --set channel_priority strict
conda install python=3.7 geopandas
pip install -r requirements.txt
python -u 'main.py'
