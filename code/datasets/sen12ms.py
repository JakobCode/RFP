"""
This script is taken and adapted from https://github.com/JakobCode/explaining_cloud_effects/ .
There the authors extended the original data loader from https://github.com/schmitt-muc/SEN12MS/blob/master/ 
to a data loader handling cloudy and clear versions of the SEN12MSCR data set.
"""

import os
import rasterio
import numpy as np
import pickle as pkl
from tqdm import tqdm

import torch
import torch.utils.data as data

# indices of sentinel-2 bands related to land
S2_BANDS_LD = [2, 3, 4, 5, 6, 7, 8, 9, 12, 13]
S2_BANDS_RGB = [2,3,4]  # B(2),G(3),R(4)

CLASS_NAMES = [
    "Forest",
    "Shrubland",
    "Savanna",
    "Grassland",
    "Wetland",
    "Croplands",
    "Urban",
    "Snow & Ice",
    "Bareland",
    "Water",
]

# define mean/std of the training set (for data normalization)
bands_mean = {'s1_mean': [-11.76858, -18.294598],
                's2_mean': [1226.4215, 1137.3799, 1139.6792, 1350.9973, 1932.9058,
                            2211.1584, 2154.9846, 2409.1128, 2001.8622, 1356.0801]}
                
bands_std = {'s1_std': [4.525339, 4.3586307],
                's2_std': [741.6254, 740.883, 960.1045, 946.76056, 985.52747,
                        1082.4341, 1057.7628, 1136.1942, 1132.7898, 991.48016]} 

# util function for reading s2 data
def load_s2(path, img_transform, s2_band):
    bands_selected = s2_band

    with rasterio.open(path) as data_file:
        s2 = data_file.read(bands_selected)
    s2 = s2.astype(np.float32)  # convert to float using astype
    if not img_transform:
        s2 = np.clip(s2, 0, 10000)  # Limit values within 0 ti 10000
        s2 /= 10000  # normalize between 0 and 1
    s2 = s2.astype(np.float32)  # convert the normalized value to float
    return s2


# util function for reading s1 data
def load_s1(path, img_transform):
    with rasterio.open(path) as data_file:
        s1 = data_file.read()
    s1 = s1.astype(np.float32)
    s1 = np.nan_to_num(s1)
    s1 = np.clip(s1, -25, 0)
    if not img_transform:
        s1 /= 25
        s1 += 1
    s1 = s1.astype(np.float32)
    return s1


# util function for reading data from single sample
def load_sample(
    sample,
    labels,
    label_type,
    threshold,
    img_transform,
    use_s1,
    use_s2,
    use_rgb,
    use_cloudy,
    cloud_frac,
    igbp_s,
):
    # load s2 data
    cloudy = False
    if use_s2:
        if not use_cloudy:
            img = load_s2(sample["s2"], img_transform, s2_band=S2_BANDS_LD)
        else:
            if sample["s2_cloudy"] is not None and cloud_frac > np.random.rand(1):
                cloudy = True
                img = load_s2(sample["s2_cloudy"], img_transform, s2_band=S2_BANDS_LD)
            else:
                img = load_s2(sample["s2"], img_transform, s2_band=S2_BANDS_LD)

    # load only RGB
    if use_rgb and not use_s2:
        if not use_cloudy:
            img = load_s2(sample["s2"], img_transform, s2_band=S2_BANDS_RGB)
        else:
            if sample["s2_cloudy"] is not None and cloud_frac > np.random.rand(1):
                img = load_s2(sample["s2_cloudy"], img_transform, s2_band=S2_BANDS_RGB)
            else:
                img = load_s2(sample["s2"], img_transform, s2_band=S2_BANDS_RGB)

    # load s1 data
    if use_s1:
        if use_s2 or use_rgb:
            img = np.concatenate((img, load_s1(sample["s1"], img_transform)), axis=0)
        else:
            img = load_s1(sample["s1"], img_transform)

    # load label
    lc = labels[sample["id"]]

    # covert label to IGBP simplified scheme
    if igbp_s:
        cls1 = sum(lc[0:5])
        cls2 = sum(lc[5:7])
        cls3 = sum(lc[7:9])
        cls6 = lc[11] + lc[13]
        lc = np.asarray(
            [cls1, cls2, cls3, lc[9], lc[10], cls6, lc[12], lc[14], lc[15], lc[16]]
        )

    if label_type == "multi_label":
        lc_hot = (lc >= threshold).astype(np.float32)
    else:
        loc = np.argmax(lc, axis=-1)
        lc_hot = np.zeros_like(lc).astype(np.float32)
        lc_hot[loc] = 1

    rt_sample = {"image": img, "label": lc_hot, "id": sample["id"], "cloudy": cloudy}

    if img_transform is not None:
        rt_sample = img_transform(rt_sample)

    return rt_sample

#  calculate number of input channels
def get_ninputs(use_s1, use_s2, use_rgb):
    n_inputs = 0
    if use_s2:
        n_inputs += len(S2_BANDS_LD)
    if use_s1:
        n_inputs += 2
    if use_rgb and not use_s2:
        n_inputs += 3

    return n_inputs


# class SEN12MS..............................
class SEN12MS(data.Dataset):
    """PyTorch dataset class for the SEN12MS dataset

    expects dataset dir as:
        - SEN12MS_holdOutScenes.txt
        - ROIsxxxx_y
            - lc_n
            - s1_n
            - s2_n

    SEN12SEN12MS_holdOutScenes.txt contains the subdirs for the official
    train/val/test split and can be obtained from:
    https://github.com/MSchmitt1984/SEN12MS/
    """

    def __init__(
        self,
        path_clear=None,
        path_cloudy=None,
        ls_dir=None,
        img_transform=None,
        label_type="multi_label",
        threshold=0.1,
        subset="val",
        use_s2=True,
        use_s1=False,
        use_rgb=True,
        use_cloudy=False,
        cloud_frac=1,
        igbp_s=True,
        label_filter=None,
        pre_load=False,
    ):
        """Initialize the dataset"""

        print("path clear: ", path_clear)
        print("path cloudy: ", path_cloudy)
        
        # inizialize
        super().__init__()
        self.img_transform = img_transform
        self.threshold = threshold
        self.label_type = label_type
        self.label_filter = label_filter

        self.pre_load = pre_load
        self.loaded_samples = {}

        if not (use_s2 or use_s1 or use_rgb):
            raise ValueError(
                "No input specified, set at least one of "
                + "use_[s2, s1, RGB] to True!"
            )

        self.cloud_frac = cloud_frac
        self.use_s2 = use_s2
        self.use_s1 = use_s1
        self.use_rgb = use_rgb
        self.use_cloudy = use_cloudy
        self.igbp_s = igbp_s

        assert subset in ["train", "val", "test", "test_cloudy"]
        assert label_type in ["multi_label", "single_label"]  # new !!

        # provide number of input channels
        self.n_inputs = get_ninputs(use_s1, use_s2, use_rgb)

        # provide number of IGBP classes
        if igbp_s:
            self.n_classes = 10
        else:
            self.n_classes = 17

        # make sure parent dir exists
        assert os.path.exists(path_clear), path_clear
        if use_cloudy:
            assert os.path.exists(path_cloudy), path_cloudy

        assert os.path.exists(ls_dir), ls_dir
        # -------------------------------- import split lists--------------------------------
        if label_type in ("multi_label", "single_label"):
            # find and index samples
            self.samples = []
            if subset == "train" or subset == "train_cloudy":
                pbar = tqdm(total=162555)  # 162556-1 samples in train set
                # 1 broken file "ROIs1868_summer_s2_146_p202" had been removed
                # from the list already
            elif subset == "val":
                pbar = tqdm(total=784)  # 18550 samples in val set
            elif subset == "test":
                pbar = tqdm(total=18106)  # 18106 samples in test set
            if subset == "test_cloudy":
                pbar = tqdm(total=12666)  # 12666 samples in cloudy test set
            pbar.set_description("[Load]")

            fix = ""

            if ls_dir.endswith(".pkl"):
                sample_list = pkl.load(open(ls_dir, "rb"))
                ls_dir = os.path.join(*ls_dir.split("/")[:-1])
            elif isinstance(ls_dir, list):
                sample_list = np.concatenate(
                    [pkl.load(open(ls_dir_s, "rb")) for ls_dir_s in ls_dir]
                )
                ls_dir = os.path.join(*ls_dir[0].split("/")[:-1])
            elif subset == "train":
                file = os.path.join(ls_dir, "train_list.pkl")
                sample_list = pkl.load(open(file, "rb"))

            elif subset == "train_cloudy": 
                file = os.path.join(ls_dir, "train_list.pkl")
                sample_list = pkl.load(open(file, "rb"))

            elif subset == "val":
                file = os.path.join(ls_dir, "val_list.pkl")
                sample_list = pkl.load(open(file, "rb"))

            elif subset == "test":
                file = os.path.join(ls_dir, "test_list.pkl")
                sample_list = pkl.load(open(file, "rb"))
            elif subset == "test_cloudy":
                file = os.path.join(ls_dir, "test_list_cloudy.pkl")
                sample_list = pkl.load(open(file, "rb"))

            # remove broken file
            broken_files = [
                "ROIs1868_summer_s2_146_p202.tif",
                "ROIs1158_spring_s2_1_p127.tif",
                "ROIs1158_spring_s2_1_p132.tif",
                "ROIs1158_spring_s2_1_p219.tif",
                "ROIs2017_winter_s2_59_p318.tif"
            ]
            for broken_file in broken_files:
                if broken_file in sample_list:
                    sample_list.remove(broken_file)

            pbar.set_description("[Load]")


            for s2_id in sample_list:
                
                mini_name = s2_id.split("_")
                s2_loc = os.path.join(
                    path_clear,
                    "_".join([mini_name[0], mini_name[1]]),
                    "_".join([mini_name[2], mini_name[3]]),
                    s2_id,
                )

                s2_loc_cloudy = os.path.join(
                    path_cloudy,
                    "_".join([mini_name[0], mini_name[1]]) + "_s2",
                    "_".join([mini_name[2], mini_name[3]]),
                    s2_id,
                )

                s1_loc = s2_loc.replace("s2_", "s1_").replace("_s2", "_s1")
                s1_cloudy_loc = s2_loc_cloudy.replace("s2_", "s1_").replace("_s2", "_s1")
                s2_cloudy_loc = s1_cloudy_loc.replace("s1_", "s2_cloudy_").replace(
                    "_s1", "_s2_cloudy"
                )

                pbar.update()

                if not os.path.exists(s2_loc):
                    print("Skip file: ", s2_loc)
                    continue

                if self.use_cloudy:

                    if os.path.exists(s2_cloudy_loc):
                        self.samples.append(
                            {
                                "s1": s1_loc,
                                "s2": s2_loc,
                                "s2_cloudy": s2_cloudy_loc,
                                "id": s2_id,
                            })
                    else:
                        self.samples.append(
                            {
                                "s1": s1_loc,
                                "s2": s2_loc,
                                "s2_cloudy": None,
                                "id": s2_id,
                            })

                    
                else:
                    self.samples.append({"s1": s1_loc, "s2": s2_loc, "id": s2_id})

            pbar.close()

        if use_cloudy:
            print(f"Loaded {len(self.samples)} files with {len([1 for s in self.samples if s['s2_cloudy'] is not None])} Cloudy versions!")

        # ----------------------------------------------------------------------

        # sort list of samples
        self.samples = sorted(self.samples, key=lambda i: i["id"])

        print("loaded", len(self.samples), "samples from the sen12ms subset", subset)

        # import lables as a dictionary
        label_file = os.path.join(ls_dir, "IGBP_probability_labels.pkl")

        a_file = open(label_file, "rb")
        self.labels = pkl.load(a_file)

        if self.label_filter:
            print("filter labels from ", len(self.labels))
            self.labels = dict(
                filter(
                    lambda i: np.argmax(i[1]) in self.label_filter, self.labels.items()
                )
            )
            print("to ", len(self.labels))
            print("filter samples from ", len(self.samples))
            self.samples = list(
                filter(lambda sample: sample["id"] in self.labels, self.samples)
            )
            print("to ", len(self.samples))

        a_file.close()
        
        self.labels_filtered = self.get_labels()

    def get_labels(self):
        res = {}

        for sample in self.samples:
            sample_id = sample["id"]
            lc = self.labels[sample_id]

            cls1 = np.sum(lc[[0,1,2,3,4,5]], axis=-1)
            cls2 = np.sum(lc[5:7], axis=-1)
            cls3 = np.sum(lc[7:9], axis=-1)
            cls6 = lc[11] + lc[13]
            lc = np.stack(
                [cls1, cls2, cls3, lc[9], lc[10], cls6, lc[12], lc[14], lc[15], lc[16]], axis=-1)

            res[sample_id] = np.argmax(lc, axis=-1)

        return res

    def __getitem__(self, index, supres_random=False):
        """Get a single example from the dataset"""

        # get and load sample from index file

        sample = self.samples[index]

        labels = self.labels
        s = load_sample(
            sample=sample,
            labels=labels,
            label_type=self.label_type,
            threshold=self.threshold,
            img_transform=self.img_transform,
            use_s1=self.use_s1,
            use_s2=self.use_s2,
            use_rgb=self.use_rgb,
            use_cloudy=self.use_cloudy,
            cloud_frac=self.cloud_frac,
            igbp_s=self.igbp_s,
        )

        return s
    
    def load_by_img_name(self, name):
        """Get a single example from the dataset"""

        # get and load sample from index file
        sample = [
            index
            for index in range(len(self.samples))
            if self.samples[index]["id"] == name
        ]
        assert len(sample) == 1
        return self[sample[0]]

    def __len__(self):
        """Get number of samples in the dataset"""
        return len(self.samples)

    def get_class_distribution(self):
        classes = [self.labels[s["id"]].argmax(-1) for s in self.samples]
        dist = torch.unique(classes, return_counts=True)
        return dist

    def get_print_data(self, sample_id):
        sample = self.samples[sample_id]

        img_trafo_low = lambda x: torch.clip(x, 0, 2500) / 2500
        img_trafo_high = lambda x: torch.clip(x, 0, 4000) / 4000
        img2 = torch.as_tensor(load_s1(sample["s1"], img_transform=None))
        img1_clear = torch.as_tensor(load_s2(sample["s2"], s2_band=S2_BANDS_RGB, img_transform=True))
        img1_clear = img_trafo_low(img1_clear)

        if "s2_cloudy" in sample:
            img1_cloudy = torch.as_tensor(load_s2(sample["s2_cloudy"], s2_band=S2_BANDS_RGB, img_transform=True))
        
            if img1_cloudy.mean() > 2000:
                img1_cloudy = img_trafo_high(img1_cloudy)
            else:
                img1_cloudy = img_trafo_low(img1_cloudy)
        else:
            img1_cloudy = torch.ones_like(img1_clear)


        img1_clear = torch.stack([img1_clear[2], img1_clear[1], img1_clear[0]],-1)
        img1_cloudy = torch.stack([img1_cloudy[2], img1_cloudy[1], img1_cloudy[0]],-1)
        img2 = torch.cat([img2, img2.sum(0, keepdims=True) * 0.5], 0).permute(dims=[1,2,0])
            
        label = self.labels_filtered[sample['id']]
        label_str = CLASS_NAMES[label]
        return {"img1_clear": img1_clear, "img1_cloudy": img1_cloudy, "img2": img2, "id": sample["id"], "label": label, "label_str": label_str}

    def plot_imgs(self, axs, num_examples, fontsize=12, cloudy_only=False):
        plot_idx = torch.randperm(len(self))

        if self.use_cloudy:
            plot_idx = [idx for idx in plot_idx if "s2_cloudy" in self.samples[idx] and self.samples[idx]["s2_cloudy"] is not None and self.labels_filtered[self.samples[idx]["id"]] != 2]

        for row_id in range(num_examples[0]):
            for col_id in range(num_examples[1]):

                sample = self.get_print_data(plot_idx[row_id*num_examples[1]+col_id])

                img1_clear = sample["img1_clear"]
                img1_cloudy = sample["img1_cloudy"]
                img2 = sample["img2"]
                
                img1 = torch.cat([img1_clear, torch.ones([img1_clear.shape[0],10,3]), img1_cloudy], -2)

                img = torch.cat([img1, torch.ones([10, img1.shape[1],3]), torch.cat([img2, torch.ones([img1.shape[0],10,3]), torch.ones_like(img2)], -2)], 0)
                axs[row_id,col_id].matshow(img, cmap="gray")
                axs[row_id,col_id].axis("off")
                axs[row_id,col_id].set_title(f"Class {CLASS_NAMES[self.labels_filtered[sample['id']]]}", fontsize=fontsize)

                if col_id == 0:
                    axs[row_id,col_id].set_ylabel("SAR - RGB (Clear) - RGB (Cloudy)", rotation=90)


#%% data normalization
class Normalize(object):
    """Pytorch Normalization classes"""
    def __init__(self, bands_mean_p, bands_std_p):

        self.bands_s1_mean = torch.as_tensor(bands_mean_p["s1_mean"]).view(-1, 1, 1)
        self.bands_s1_std = torch.as_tensor(bands_std_p["s1_std"]).view(-1, 1, 1)

        self.bands_s2_mean = torch.as_tensor(bands_mean_p["s2_mean"]).view(-1, 1, 1)
        self.bands_s2_std = torch.as_tensor(bands_std_p["s2_std"]).view(-1, 1, 1)

        self.bands_rgb_mean = torch.as_tensor(bands_mean_p["s2_mean"][0:3]).view(-1, 1, 1)
        self.bands_rgb_std = torch.as_tensor(bands_std_p["s2_std"][0:3]).view(-1, 1, 1)

        if self.bands_s1_mean is not None:
            self.bands_all_mean = torch.cat([self.bands_s2_mean, self.bands_s1_mean], 0)
            self.bands_all_std = torch.cat([self.bands_s2_std, self.bands_s1_std], 0)
        else:
            self.bands_all_mean = None
            self.bands_all_std = None

    def __call__(self, rt_sample):

        img, label, sample_id, cloudy = rt_sample["image"], rt_sample["label"], rt_sample["id"], rt_sample["cloudy"]
        if img.shape[0] == 12:
            img = (img - self.bands_all_mean) / self.bands_all_std
        elif img.shape[0] == 10:
            img = (img - self.bands_s2_mean) / self.bands_s2_std

        else:
            raise Exception("Not implemented")

        return {"image": img, "label": label, "id": sample_id, "cloudy": cloudy}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, rt_sample):

        img, label, sample_id, cloudy = rt_sample["image"], rt_sample["label"], rt_sample["id"], rt_sample["cloudy"]

        rt_sample = {"image": torch.as_tensor(img), "label": label, "id": sample_id, "cloudy": cloudy}
        return rt_sample