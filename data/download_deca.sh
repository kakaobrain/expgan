# The script is copied and edited from https://github.com/radekd91/emoca/blob/release/EMOCA/gdl_apps/EMOCA/demos/download_assets.sh

echo "In order to run EMOCA, you need to download FLAME. Before you continue, you must register and agree to license terms at:"
echo -e '\e]8;;https://flame.is.tue.mpg.de\ahttps://flame.is.tue.mpg.de\e]8;;\a'

while true; do
    read -p "I have registered and agreed to the license terms at https://flame.is.tue.mpg.de? (y/n)" yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done

echo "If you wish to use EMOCA, please register at:" 
echo -e '\e]8;;https://emoca.is.tue.mpg.de\ahttps://emoca.is.tue.mpg.de\e]8;;\a'
while true; do
    read -p "I have registered and agreed to the license terms at https://emoca.is.tue.mpg.de? (y/n)" yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done

echo "Downloading DECA related assets"
wget https://download.is.tue.mpg.de/emoca/assets/DECA.zip -O DECA.zip
echo "Extracting DECA related assets..."
unzip DECA.zip
rm DECA.zip

# echo "Downloading FLAME related assets"
# wget https://download.is.tue.mpg.de/emoca/assets/FLAME.zip -O FLAME.zip
# echo "Extracting FLAME..."
# unzip FLAME.zip
# echo "Assets for EMOCA downloaded and extracted."