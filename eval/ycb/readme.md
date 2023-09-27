# YCB dataset evaluation

Currently we use YCB evaluation directly from densefusion to ensure same criteria.(Still modified a little bit) (Densefusion's evaluation is also taken from posecnn LOL)

Run `eval.sh`. It will download YCB toolbox automatically. Change the parameters to load correct dataset, model, and save.

Note that we are not using refinenet currently, so we removed all these parts.

update:

`eval_ycb_gt.py` evaluates the model on YCB dataset with ground truth label. `eval_ycb_posecnn.py` evaluates the model on YCB dataset with the labels that provided by YCB_tool_box.