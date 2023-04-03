python3 -m venv env 
source env/bin/activate
pip3 install ipykernel
python3 -m ipykernel install --user --name=env
pip install -r requirements.txt
open the notebook in that environment, and you should be good to go

