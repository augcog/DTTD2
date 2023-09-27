import os
import re

def get_checkpoint(model_path, isAcc=False):
    if os.path.isfile(model_path):
        return model_path
    elif os.path.isdir(model_path):
        checkpoints = sorted(os.listdir(model_path))
        record = []
        for cp in checkpoints:
            find = re.match(r'epoch_(\d+)_(.+)_([\d\.]+).pth', cp)
            if find == None:
                if cp == "best_save.pth":
                    return os.path.join(model_path, cp) # return best save directly
                print("[Warning] checkpoint name should be in epoch_(\d+)_dist_([\d\.]+).pth form")
            else:
                val = find.group(3)
                record.append((val, cp))
        if len(record) > 0:
            record = sorted(record, key=lambda x: x[0], reverse=isAcc)
            return os.path.join(model_path, record[0][1])
        else:
            return None
    else:
        return None
    