export LC_ALL=en_US.utf-8
export LANG=en_US.utf-8
#ln -s /apdcephfs_cq2/share_1290939/gavinyuan/ /gavin
wandb offline
cd /apdcephfs_cq2/share_1290939/gavinyuan/code/FaceSwapping/trainer/faceshifter || exit
export PYTHONPATH=/apdcephfs_cq2/share_1290939/gavinyuan/code/FaceSwapping/:/gavin/code/FaceSwapping/modules/third_party:$PYTHONPATH
python3 /apdcephfs_cq2/share_1290939/gavinyuan/code/FaceSwapping/trainer/faceshifter/faceshifter_trainer.py -n hello -bs 1