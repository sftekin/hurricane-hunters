import os
import numpy as np

from static_helpers import haversine_dist

folder_path = os.path.join("data", "hurricanes")
file_path_list = os.listdir(folder_path)
test_ratio = 0.2
test_count = int(test_ratio * len(file_path_list))

for step in range(1, 17):
    error = 0
    count = 0
    for file_path in file_path_list[:test_count]:
        hurricane_file_path = os.path.join(folder_path, file_path)
        hurr = np.load(hurricane_file_path)[:, :2]

        if len(hurr) < 2 + step:
            continue

        target_list = []
        pred_list = []
        for i in range(1+step, len(hurr)):
            target = hurr[i]
            cur_hurr = hurr[:i]
            for s in range(step):
                pred = 2 * cur_hurr[-1] - cur_hurr[-2]
                cur_hurr = np.concatenate([cur_hurr, pred[None, :]], axis=0)
            pred_list.append(pred)
            target_list.append(target)

        pred = np.stack(pred_list, axis=0)
        target = np.stack(target_list, axis=0)

        cur_error = haversine_dist(pred, target)
        error += cur_error.sum()
        count += pred.shape[0]

    print(f"Error (km): for {step*3}-hour: {error/count}")

