from collections import defaultdict
import sys
import os
import logging
import requests
import math
import zipfile
from contextlib import contextmanager
from tempfile import TemporaryDirectory
from tqdm import tqdm
from retrying import retry
import gzip
import shutil


@retry(wait_random_min=1000, wait_random_max=5000, stop_max_attempt_number=5)
def maybe_download(url, filename=None, work_directory=".", expected_bytes=None):
    """Download a file if it is not already downloaded.

    Args:
        filename (str): File name.
        work_directory (str): Working directory.
        url (str): URL of the file to download.
        expected_bytes (int): Expected file size in bytes.

    Returns:
        str: File path of the file downloaded.
    """
    if filename is None:
        filename = url.split("/")[-1]
    os.makedirs(work_directory, exist_ok=True)
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            print(f"Downloading {url}")
            total_size = int(r.headers.get("content-length", 0))
            block_size = 1024
            num_iterables = math.ceil(total_size / block_size)
            with open(filepath, "wb") as file:
                for data in tqdm(
                    r.iter_content(block_size),
                    total=num_iterables,
                    unit="KB",
                    unit_scale=True,
                ):
                    file.write(data)
        else:
            print(f"Problem downloading {url}")
            r.raise_for_status()
    else:
        print(f"File {filepath} already downloaded")
    if expected_bytes is not None:
        statinfo = os.stat(filepath)
        if statinfo.st_size != expected_bytes:
            os.remove(filepath)
            raise IOError(f"Failed to verify {filepath}")

    return filepath


@contextmanager
def download_path(path=None):
    """Return a path to download data. If `path=None`, then it yields a temporal path that is eventually deleted,
    otherwise the real path of the input.

    Args:
        path (str): Path to download data.

    Returns:
        str: Real path where the data is stored.

    Examples:
        >>> with download_path() as path:
        >>> ... maybe_download(url="http://example.com/file.zip", work_directory=path)

    """
    if path is None:
        tmp_dir = TemporaryDirectory()
        try:
            yield tmp_dir.name
        finally:
            tmp_dir.cleanup()
    else:
        path = os.path.realpath(path)
        yield path


def unzip_file(zip_src, dst_dir, clean_zip_file=False):
    """Unzip a file

    Args:
        zip_src (str): Zip file.
        dst_dir (str): Destination folder.
        clean_zip_file (bool): Whether or not to clean the zip file.
    """
    fz = zipfile.ZipFile(zip_src, "r")
    for file in fz.namelist():
        fz.extract(file, dst_dir)
    if clean_zip_file:
        os.remove(zip_src)


def download_and_extract(name, dest_path):
    """Downloads and extracts Amazon reviews and meta datafiles if they don’t already exist

    Args:
        name (str): Category of reviews.
        dest_path (str): File path for the downloaded file.

    Returns:
        str: File path for the extracted file.
    """
    dirs, _ = os.path.split(dest_path)
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    file_path = os.path.join(dirs, name)
    if not os.path.exists(file_path):
        _download_reviews(name, dest_path)
        _extract_reviews(file_path, dest_path)

    return file_path


def _download_reviews2(name, dest_path):
    """Downloads Amazon reviews datafile.

    Args:
        name (str): Category of reviews
        dest_path (str): File path for the downloaded file
    """

    print(f"Downloading {name} data")
    if "meta" in name:
        url = "http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles2/" + name + ".gz"
    else:
        url = "http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/" + name + ".gz"

    dirs, file = os.path.split(dest_path)
    maybe_download(url, file + ".gz", work_directory=dirs)


def _download_reviews(name, dest_path):
    """Downloads Amazon reviews datafile.

    Args:
        name (str): Category of reviews
        dest_path (str): File path for the downloaded file
    """

    print(f"Downloading {name} data")
    url = (
        "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/"
        + name
        + ".gz"
    )

    # url = ("http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/"
    #     + name
    #     + ".gz"
    # )
    # http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/All_Beauty_5.json.gz

    dirs, file = os.path.split(dest_path)
    maybe_download(url, file + ".gz", work_directory=dirs)


def _extract_reviews(file_path, zip_path):
    """Extract Amazon reviews and meta datafiles from the raw zip files.

    To extract all files,
    use ZipFile's extractall(path) instead.

    Args:
        file_path (str): Destination path for datafile
        zip_path (str): zipfile path
    """
    print(f"Extracting from {zip_path} data")
    with gzip.open(zip_path + ".gz", "rb") as zf, open(file_path, "wb") as f:
        shutil.copyfileobj(zf, f)


def _reviews_preprocessing(reviews_readfile):
    print("start reviews preprocessing...")
    reviews_writefile = reviews_readfile + "_output"
    reviews_r = open(reviews_readfile, "r")
    reviews_w = open(reviews_writefile, "w")
    for line in reviews_r:
        line = line.replace("true", "True")
        line = line.replace("false", "False")
        # print(line)
        line_new = eval(line.strip())
        reviews_w.write(
            str(line_new["reviewerID"])
            + "\t"
            + str(line_new["asin"])
            + "\t"
            + str(line_new["unixReviewTime"])
            + "\n"
        )
    reviews_r.close()
    reviews_w.close()
    print(f"Processed data in {reviews_writefile}")
    return reviews_writefile


def _meta_preprocessing(meta_readfile):
    # logger.info("start meta preprocessing...")
    meta_writefile = meta_readfile + "_output"
    meta_r = open(meta_readfile, "r")
    meta_w = open(meta_writefile, "w")
    for line in meta_r:
        line_new = eval(line)
        meta_w.write(line_new["asin"] + "\t" + line_new["categories"][0][-1] + "\n")
    meta_r.close()
    meta_w.close()
    return meta_writefile


def parse(path):
    g = gzip.open(path, "r")
    for line in g:
        yield eval(line)


def timeSlice(time_set):
    time_min = min(time_set)
    time_map = dict()
    for time in time_set:
        time_map[time] = int(round(float(time - time_min)))
    return time_map


def cleanAndsort(User, time_map):
    User_filted = dict()
    user_set = set()
    item_set = set()
    for user, items in User.items():
        user_set.add(user)
        User_filted[user] = items
        for item in items:
            item_set.add(item[0])
    user_map = dict()
    item_map = dict()
    for u, user in enumerate(user_set):
        user_map[user] = u + 1
    for i, item in enumerate(item_set):
        item_map[item] = i + 1

    for user, items in User_filted.items():
        User_filted[user] = sorted(items, key=lambda x: x[1])

    User_res = dict()
    for user, items in User_filted.items():
        User_res[user_map[user]] = list(
            map(lambda x: [item_map[x[0]], time_map[x[1]]], items)
        )

    time_max = set()
    for user, items in User_res.items():
        time_list = list(map(lambda x: x[1], items))
        time_diff = set()
        for i in range(len(time_list) - 1):
            if time_list[i + 1] - time_list[i] != 0:
                time_diff.add(time_list[i + 1] - time_list[i])
        if len(time_diff) == 0:
            time_scale = 1
        else:
            time_scale = min(time_diff)
        time_min = min(time_list)
        User_res[user] = list(
            map(lambda x: [x[0], int(round((x[1] - time_min) / time_scale) + 1)], items)
        )
        time_max.add(max(set(map(lambda x: x[1], User_res[user]))))

    return User_res, len(user_set), len(item_set), max(time_max)


def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}

    print("Preparing data...")
    f = open(fname, "r")
    time_set = set()

    user_count = defaultdict(int)
    item_count = defaultdict(int)
    for line in f:
        try:
            u, i, rating, timestamp = line.rstrip().split("\t")
        except:
            u, i, timestamp = line.rstrip().split("\t")
        u = int(u)
        i = int(i)
        user_count[u] += 1
        item_count[i] += 1
    f.close()
    f = open(fname, "r")
    for line in f:
        try:
            u, i, rating, timestamp = line.rstrip().split("\t")
        except:
            u, i, timestamp = line.rstrip().split("\t")
        u = int(u)
        i = int(i)
        timestamp = float(timestamp)
        if user_count[u] < 5 or item_count[i] < 5:
            continue
        time_set.add(timestamp)
        User[u].append([i, timestamp])
    f.close()
    time_map = timeSlice(time_set)

    User, usernum, itemnum, timenum = cleanAndsort(User, time_map)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    print("Preparing done...")
    return [user_train, user_valid, user_test, usernum, itemnum, timenum]


def data_process_with_time(fname, pname, K=10, sep=" ", item_set=None, add_time=False):
    User = defaultdict(list)
    Users = set()
    Items = set()
    user_dict, item_dict = {}, {}

    item_counter = defaultdict(lambda: 0)
    user_counter = defaultdict(lambda: 0)
    with open(fname, "r") as fr:
        for line in fr:
            u, i, t = line.rstrip().split(sep)
            User[u].append((i, t))
            Items.add(i)
            Users.add(u)
            item_counter[i] += 1
            user_counter[u] += 1
            # if i in item_counter:
            #     item_counter[i] += 1
            # else:
            #     item_counter[i] = 1

    # print(item_counter['1304351475'], item_counter['1304482685'])

    # remove items with less than K interactions
    print(f"Read {len(User)} users and {len(Items)} items")
    remove_items = set()
    count_remove, count_missing = 0, 0
    for item in Items:
        if item_counter[item] < K:
            count_remove += 1
            remove_items.add(item)
        elif item_set and item not in item_set:
            count_missing += 1
            remove_items.add(item)

    if count_remove > 0:
        print(f"{count_remove} items have less than {K} interactions")

    if count_missing > 0:
        print(f"{count_missing} items are not in the meta data")

    Items = Items - remove_items

    # remove users with less than K interactions
    remove_users = set()
    count_remove = 0
    # Users = set(User.keys())
    for user in Users:
        if user_counter[user] < K:
            remove_users.add(user)
            count_remove += 1
    if count_remove > 0:
        print(f"{count_remove} users have less than {K} interactions")
        Users = Users - remove_users

    print(f"Total {len(Users)} users and {len(Items)} items")
    item_count = 1
    for item in Items:
        item_dict[item] = item_count
        item_count += 1

    count_del = 0
    user_count = 1
    with open(pname, "w") as fw:
        for user in Users:
            items = User[user]
            items = [tup for tup in items if tup[0] in Items]
            if len(items) < K:
                # del User[user]
                count_del += 1
            else:
                user_dict[user] = user_count
                # sort by time
                items = sorted(items, key=lambda x: x[1])

                # replace by the item-code
                timestamps = [x[1] for x in items]
                items = [item_dict[x[0]] for x in items]
                for i, t in zip(items, timestamps):
                    out_txt = [str(user_count), str(i)]
                    if add_time:
                        out_txt.append(str(t))
                    fw.write(sep.join(out_txt) + "\n")
                user_count += 1

    print(f"Total {user_count-1} users, {count_del} removed")
    print(f"Processed model input data in {pname}")
    return user_dict, item_dict


def nwords(txt, n):
    return " ".join(txt.split()[:n])


if __name__ == "__main__":
    """
    example usage: python download_and_process_amazon.py Beauty 5 0

    """
    keep_words = 100  # maximum number of words in the item description
    K = 10  # filter, minimum number of interactions (item and user)
    core = 5
    url_type = 1
    data_path = "data/."

    if len(sys.argv) > 1:
        category = sys.argv[1]
    else:
        category = "Beauty"  # 'Toys_and_Games' # 'Movies_and_TV', 'Beauty', 'Electronics' (too big)

    if len(sys.argv) > 2:
        K = int(sys.argv[2])

    if len(sys.argv) > 3:
        core = int(sys.argv[3])

    if len(sys.argv) > 4:
        url_type = int(sys.argv[4])

    if url_type == 1:
        if core > 0:
            reviews_name = "reviews_" + category + "_" + str(core) + ".json"
        else:
            reviews_name = "reviews_" + category + ".json"
    else:
        reviews_name = category + ".json"
    meta_name = "meta_" + category + ".json"

    reviews_file = os.path.join(data_path, reviews_name)
    meta_file = os.path.join(data_path, meta_name)
    text_file = os.path.join(data_path, category + "_item_description.txt")

    print(f"Generating data for ***{category}***")
    download_and_extract(reviews_name, reviews_file)
    download_and_extract(meta_name, meta_file)
    reviews_output = _reviews_preprocessing(reviews_file)

    all_items = set()
    ignore_item = 0
    with open(meta_file, "r") as fr:
        for line in fr.readlines():
            jdict = eval(line)
            if "description" in jdict.keys() or "title" in jdict.keys():
                all_items.add(jdict["asin"])
            else:
                ignore_item += 1
    print(f"Read {len(all_items)} from {meta_file} (ignored {ignore_item} items)")

    model_input = os.path.join(data_path, category + ".txt")
    udict, idict = data_process_with_time(
        reviews_output, model_input, K, "\t", all_items
    )
    if len(udict) == 0 or len(idict) == 0:
        print(f"{len(udict)} users and {len(idict)} items")
        exit()

    num_items = len(idict)
    # number to name
    inv_udict = {v: k for k, v in udict.items()}
    inv_idict = {v: k for k, v in idict.items()}

    # get item descriptions from the meta file
    ddict = {}
    with open(os.path.join(data_path, meta_name), "r") as fr:
        for line in fr.readlines():
            jdict = eval(line)
            if jdict["asin"] in idict:
                ddict[jdict["asin"]] = {}
                if "title" in jdict.keys():
                    ddict[jdict["asin"]]["title"] = jdict["title"]
                if "description" in jdict.keys():
                    if type(jdict["description"]) is str:
                        ddict[jdict["asin"]]["description"] = jdict["description"]
                    elif (
                        type(jdict["description"]) is list
                        and len(jdict["description"]) > 0
                    ):
                        ddict[jdict["asin"]]["description"] = jdict["description"][0]

    # print(len(ddict), num_items)
    count = 0
    for item in ddict:
        if len(ddict[item]) == 0:
            count += 1
    print(f"{count} items do not have descriptions and title")

    max_len = 0
    with open(text_file, "w") as fw:
        for item_number in range(1, num_items + 1):
            item_name = inv_idict[item_number]
            temp_dict = ddict[item_name]
            title, desc = "", ""
            if "title" in temp_dict:
                title = temp_dict["title"].replace("\n", "")
            if "description" in temp_dict:
                desc = temp_dict["description"].replace("\n", "")
            otxt = title + " " + desc
            if len(otxt.split()) > max_len:
                max_len = len(otxt.split())
            otxt = nwords(otxt, keep_words)
            if len(otxt) == 0:
                otxt = "not available"
            fw.write(otxt + "\n")
    print(f"Maximum number of words {max_len}")
    print(f"Processed model input data with text in {text_file}")
