EPOCHS=30
SAVEPATH='/root/ktg/Capstone2/checkpoint/SupCon/extrapolate_intra'


python /root/ktg/Capstone2/src/augment.py\
        --epochs=${EPOCHS}\
        --save-path=${SAVEPATH}