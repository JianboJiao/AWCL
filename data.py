from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import yaml
import torch
import numpy as np
import os
import cv2
import random
from pulsepytools_local.pulsepytools.pulsedata.data import Scan, LiveScan# tool for loading our PULSE data, please replace with your own data loading tools


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(227),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

class AbstractData(Dataset):

    def __init__(self, scan_id_list, val=False, config_path=None, device=None,
                 plot=False, data_config_path=None, config_file="spdata_SPD.yml",
                 default_cfg_folder='config'):
        super().__init__()
        if config_path is None:
            config_dir = Path(__file__).parent / default_cfg_folder
        else:
            config_dir = Path(config_path)
        self.config_file = config_dir.expanduser().resolve() / config_file
        with open(self.config_file, 'r') as stream_:
            self.config = yaml.load(stream_)
        self.data_config_path = data_config_path
        self.scan_id_list = scan_id_list
        self.val = val
        if device is None:
            device = torch.device('cuda')
        self.device = device
        self.plot = plot
        self.pre_sf = self.config['pre_scaling_factor']
        self.out_res = np.array(self.config['out_res'], dtype=np.uint64)
        self.to_augment = self.config['to_augment'] and not self.val
        if self.to_augment:
            self.augm_cfg = self.config['augment_cfg']
        else:
            self.augm_cfg = None
        self.pre_res = None

    def apply_gamma_brighness(self, frame):
        if not (self.augm_cfg['gamma'] or self.augm_cfg['brightness']):
            return frame
        frame = Image.fromarray(frame)
        if self.augm_cfg['gamma'] is not None:
            gamma = random.uniform(*self.augm_cfg['gamma'])
            frame = transforms.functional.adjust_gamma(frame, gamma, gain=1)
        if self.augm_cfg['brightness'] is not None:
            brightness = random.uniform(*self.augm_cfg['brightness'])
            frame = transforms.functional.adjust_brightness(frame, brightness)
        return np.array(frame)

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError



class PULSELoader(AbstractData):
    """PULSE Dataset.
    """
    sononet_labels = ['3VT',
                     '4CH',
                     'ACP',
                     'BG',
                     'SOB',
                     'HCP',
                     'FLP',
                     'KIDNEYS_CORONAL',
                     'NOSE_LIPS',
                     'LVOT',
                     'PROFILE',
                     'RVOT',
                     'SPINE_COR',
                     'SPINE_SAG']

    def __init__(self, scan_id_list, config_file="spdata.yml", transform=None,
                 default_cfg_folder='config', data_dir=None,
                 classes=None, sononet=False, **kwargs):
        super().__init__(scan_id_list, config_file=config_file,
                         default_cfg_folder=default_cfg_folder, **kwargs)

        self.scan_list = None
        self.data = None
        self.transform=transform

        if self.val and 'frame_modulo_val' in self.config:
            self.frame_modulo = self.config['frame_modulo_val']
        else:
            self.frame_modulo = self.config['frame_modulo']

        ################################################################
        if data_dir is None:
            local_data_dir = Path(os.environ['LOCAL_DATA_DIR'])
            data_dir = local_data_dir.expanduser()
        self.data_dir = data_dir / self.config['data_folder']
        self.sononet = sononet
        if classes is None:
            if sononet:
                classes = self.sononet_labels
            elif 'classes' in self.config:
                classes = self.config['classes']
            else:
                classes = None
        self.classes = classes
        self.class_mapping = None
        self.label_mapping = None
        self.num_classes = None
        self.c = None
        self.normalize = self.config['normalize']
        if not self.val:
            self.num_samples = self.config['num_samples']
        else:
            self.num_samples = None
        self.class_count = None
        self.frames = None
        self.samples = None
        self.weights = None
        self.sononet_sampling = True if 'sononet_sampling' not in self.config\
            else self.config['sononet_sampling']

        postproc = [
            transforms.Resize(tuple(self.out_res[::-1])),
            transforms.ToTensor(),
        ]
        if not (sononet or self.normalize):
            postproc.append(transforms.Normalize([44.02], [48.29]))
        self.postproc = transforms.Compose(postproc)

        self.load_data()
        self.set_weights()

    def set_weights(self):
        class_weights = {class_: 1. / count if count != 0 else 1
                         for class_, count in self.class_count.items()}
        if self.sononet_sampling:
            if 'BG' in class_weights:
                class_weights['BG'] *= len(self.class_count) - 1
        self.weights = [class_weights[class_] for _, class_ in self.samples]

    def load_data(self):
        #load original data w/o labels
        self.scan_list = []
        self.data = []
        for scan_idx, scan_id in enumerate(self.scan_id_list):
            scan = Scan(scan_id, config_path=self.data_config_path)
            selection = LiveScan(scan)
            self.scan_list.append(scan)
            if self.pre_res is None:
                self.pre_res = scan.get_scaled_res(self.pre_sf)
            for frame_nr in selection.frames:
                if not ((frame_nr - 1) % self.frame_modulo == 0):
                    continue
                self.data.append((scan_idx, frame_nr, None))

        #load data w/ labels
        class_dirs = self.data_dir.glob('*')
        class_dirs = [dir_ for dir_ in class_dirs if dir_.is_dir()]

        if self.classes is None:
            self.classes = sorted([dir_.name for dir_ in class_dirs])

        self.class_mapping = {key: idx for key, idx in zip(
            self.classes, range(len(self.classes)))}
        self.label_mapping = {
            val: key for key, val in self.class_mapping.items()}
        self.num_classes = len(self.label_mapping)
        self.c = self.num_classes
        print(self.class_mapping)

        self.frames = {class_: [] for class_ in self.classes}
        self.class_count = {class_: 0 for class_ in self.classes}
        self.samples = []

        def load_class(class_):
            for scan_id in self.scan_id_list:
                dir_ = self.data_dir / class_
                frames = dir_.glob(scan_id + '*.png')
                for frame in frames:
                    if not frame.exists():
                        continue
                    self.frames[class_].append(frame.name)
                    self.class_count[class_] += 1
                    self.samples.append(
                        (frame.name, class_))
                    if class_ != 'BG' and self.num_samples is not None and\
                        self.class_count[class_] >= self.num_samples:
                        return

        for this_class in self.classes:
            load_class(this_class)
        print(self.class_count)

    def __getitem__(self, index):
        #load original data w/o labels
        if random.random() >= len(self.samples)/len(self.data):
            target = -1
            scan_idx, frame_nr, point = self.data[index]
            frame = self.scan_list[scan_idx].get_scaled_frame(frame_nr, self.pre_sf)
            frame = np.stack((frame,frame,frame),axis=2)
            img = Image.fromarray(frame)
            if self.transform is not None:
                pos_1 = self.transform(img)
                pos_2 = self.transform(img)

        #load data w/ labels
        else:
            index = (index%len(self.samples))
            file, class_ = self.samples[index]
            target = self.class_mapping[class_]
            file = str(self.data_dir / class_ / file)
            frame = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            if frame is None:
                print(file)
                print('Is None')
            if self.to_augment:
                frame = self.apply_gamma_brighness(frame)
            frame = frame.astype(np.uint8)
            frame=np.stack((frame,frame,frame),axis=2)

            img = Image.fromarray(frame)#.convert('L')
            if self.transform is not None:
                pos_1 = self.transform(img)
                pos_2 = self.transform(img)
        
            # Impact of anatomy analysis (Sec. 6.4 in the paper)
            # if random.random() >0.8: #proportion of labels (0.1, 0.3, 0.5, 0.8)
                # target = -1
        
        return pos_1, pos_2, target

    def __len__(self):
        return len(self.data)
