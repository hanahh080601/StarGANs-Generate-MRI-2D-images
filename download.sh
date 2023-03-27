FILE=$1
if [ $FILE == 'pretrained-both-256x256' ]; then

    # StarGAN trained on CelebA (Black_Hair, Blond_Hair, Brown_Hair, Male, Young), 256x256 resolution
    URL=https://www.dropbox.com/s/4gxqe169axc1ezw/pretrained-both-256x256.zip?dl=0
    ZIP_FILE=./stargan_both_256/models/pretrained-both-256x256.zip
    mkdir -p ./stargan_both_256/models/
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./stargan_both_256/models/
    rm $ZIP_FILE

else
    echo "Available arguments are pretrained-both-256x256"
    exit 1
fi