sudo apt update
yes Y | sudo apt install python3-pip
python3 -m pip --version
python3 -m pip install ipykernel
python3 -m ipykernel install --user
pip3 install --upgrade pip

yes Y | sudo apt install awscli
yes Y | sudo apt install linuxbrew-wrapper

python3 -m pip3 install -r requirements.txt

# sudo apt install postgresql-client-common
# sudo apt-get install postgresql-client

# OMG ... the security group has to be correct too... Yeah have to add it ... unfortunately

# have to log back in and out
jupyter lab --FileCheckpoints.checkpoint_dir=/home/ubuntu/environment/.jupyter_checkpoints
