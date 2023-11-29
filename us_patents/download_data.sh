echo "Writing your Kaggle API '${KAGGLE_API_KEY}' and Kaggle user name '${KAGGLE_USER}' to ~/.kaggle/kaggle.json"

if [ -z "${KAGGLE_API_KEY}" ]; then
    echo "Error: Environment variable 'KAGGLE_API_KEY' is not set."
    exit 1
fi

if [ -z "${KAGGLE_USER}" ]; then
    echo "Error: Environment variable 'KAGGLE_USER' is not set."
    exit 1
fi

mkdir -p ~/.kaggle
cat <<EOF > ~/.kaggle/kaggle.json
{"username":"$KAGGLE_USER","key":"$KAGGLE_API_KEY"}
EOF
cat ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
echo "Kaggle API Key successfully linked !!!"

mkdir data/
cd data
kaggle datasets download -d atharvaingle/uspppm-data
unzip uspppm-data.zip
rm uspppm-data.zip
echo "Data downloaded and unzipped successfully !!!"
