"""Download activitynet videos"""

import json
import multiprocessing
import os
import sys

from concurrent.futures import ProcessPoolExecutor, as_completed

from utils.video import download_youtube


def print_progress(iteration, total, prefix='', suffix='', decimals=3, bar_length=100):
    """
    Call in a loop to create terminal progress bar

    :param iteration: current iteration
    :param total: total iterations
    :param prefix: prefix string
    :param suffix: suffix string
    :param decimals: positive number of decimals in percent complete
    :param bar_length: character length of bar
    :return:
    """

    format_str = "{0:." + str(decimals) + "f}"
    percents = format_str.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\x1b[2K\r')
    sys.stdout.flush()


def download_activitynet():
    dataset_path = os.path.join('datasets', 'ActivityNet')
    with open(os.path.join(dataset_path, 'activity_net.v1-3.min.json'), 'r') as f:
        d = json.load(f)
    
    ids = list(d['database'].keys())

    each_vid_size_approx_gb = .029620394  # i have guestimated this
    expected_size = each_vid_size_approx_gb * len(ids)
    print("\n\nYou have set keep_vids=True ..."
          "\nBe warned that this will try to download {} videos with an approximate size of {} GBs."
          " This could take weeks or even months depending on your download speeds."
          "\n\nContinue? (y/n)".format(len(ids), int(expected_size)))

    response = input()
    if response.lower() not in ['y', 'yes']:
        print("User Cancelled")
        return

    # Download the files
    videos_dir = os.path.join(dataset_path, 'Videos')
    os.makedirs(videos_dir, exist_ok=True)
    errors = set()
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:

        futures = [executor.submit(download_youtube, videos_dir, v_id) for v_id in ids]

        for i, f in enumerate(as_completed(futures)):
            print_progress(i, len(ids), prefix="Downloading Videos:", suffix='Complete', decimals=5)
            result = f.result()
            if result[0] < 1:
                errors.add(result[1])

        print("Successfully processed {} / {} videos".format(len(ids) - len(errors), len(ids)))

    if len(errors) > 0:
        print("Saving Error file: {}".format(os.path.join(dataset_path, "frame_get_errors.txt")))
        with open(os.path.join(dataset_path, "frame_get_errors.txt"), "a") as f:
            for v_id in errors:
                f.write(v_id + "\n")

    return len(errors)


if __name__ == "__main__":

    download_activitynet()
