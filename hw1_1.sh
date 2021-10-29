filename='model2500.pth'
if [! -f model2500.pth]; then
        wget https://www.dropbox.com/s/2t4p9ss5yzcvns9/model2500.pth?dl=0 -O $filename
fi
python3 ./p1/inference.py --img_dir $1 --save_dir $2 --model_path $filename