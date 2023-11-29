# this script can be attached as a startup script in jarvislab.ai

## gotcha! you must go to /home, if you stay in /root you will run out of disk space
cd /home

## get training code
git clone https://github.com/rmminusrslash/ml_prototyping.git
cd ml_prototyping
git checkout custom_embeddings

## set up dependencies
curl -sSL https://install.python-poetry.org | python3 -
export PATH="/home/.local/bin:$PATH"
echo export PATH="/home/.local/bin:$PATH" >> ~/.bashrc
poetry completions bash >> ~/.bash_completion
# install without env so that the already installed torch package is accessible
poetry config virtualenvs.create false
poetry install --without dev

## Download data

echo "Writing Kaggle API key to ~/.kaggle/kaggle.json"
mkdir -p ~/.kaggle
cat <<EOF > ~/.kaggle/kaggle.json
{"username":"lina261486","key":"yourkeys"}
EOF

cat ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
echo "Kaggle API Key successfully linked !!!"

echo "Downloading data"
mkdir data/
cd data/
kaggle datasets download -d atharvaingle/uspppm-data
unzip uspppm-data.zip
rm uspppm-data.zip


# run training
#cd ..
#python -m nlp.kaggle_patents.train debug=False data.input_dir=$PWD/data 
