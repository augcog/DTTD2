import os
import glob
import re
from moviepy.editor import *

base_dir = os.path.realpath("./dttd_iphone/eval_results_densefusion_gtmask/visualize")
# print(base_dir)

gif_name = 'pic'
fps = 30

# file_list = glob.glob(base_dir, '*.png')  # Get all the pngs in the current directory
# file_list_sorted = sorted(file_list,reverse=False)  # Sort the images

files = [os.path.join(base_dir, f) for f in sorted(os.listdir(base_dir))]
# print(files)
nf = []
for i in range(len(files)):
    res = re.match(r"(\d+)\.png", os.path.basename(files[i]))
    if res != None:
        nf.append((int(str(res.group(1))), files[i]))
    print(int(str(res.group(1))))

# nf = sorted(nf, key=lambda x: x[0])


clips = [ImageClip(m[1]).set_duration(1/fps) for m in nf]
concat_clip = concatenate_videoclips(clips)
concat_clip.write_videofile("eval.mp4", fps=fps)