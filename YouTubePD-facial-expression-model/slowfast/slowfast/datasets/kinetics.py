#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import numpy as np
import os
import random
import pandas
import torch
import torch.utils.data
from torchvision import transforms
from torchvision.utils import save_image
import pickle

import slowfast.utils.logging as logging
from slowfast.utils.env import pathmgr

# from . import decoder as decoder
from . import pd_decoder as decoder
from . import transform as transform
from . import utils as utils
from . import video_container as container
from .build import DATASET_REGISTRY
from .random_erasing import RandomErasing
from .transform import create_random_augment

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Kinetics(torch.utils.data.Dataset):
    """
    Kinetics video loader. Construct the Kinetics video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(self, cfg, mode, num_retries=100):
        """
        Construct the Kinetics video loader with a given csv file. The format of
        the csv file is:
        ```
        path_to_video_1 label_1
        path_to_video_2 label_2
        ...
        path_to_video_N label_N
        ```
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries.
        """
        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for Kinetics".format(mode)
        
        self.mode = mode
        self.cfg = cfg
        self._video_meta = {}
        self._num_retries = num_retries
        self._num_epoch = 0.0
        self._num_yielded = 0
        self.skip_rows = self.cfg.DATA.SKIP_ROWS
        self.p_convert_dt = self.cfg.DATA.TIME_DIFF_PROB
        self.use_chunk_loading = (
            True
            if self.mode in ["train"] and self.cfg.DATA.LOADER_CHUNK_SIZE > 0
            else False
        )
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.

        logger.info("Constructing Kinetics {}...".format(mode))
        self._construct_loader()
        self.kpts = {}


        if self.mode in ["test"]:
            print("self.mode: ", self.mode)
            print("self.cfg: ", self.cfg)
            print("self._video_meta: ", self._video_meta)
            print("self._num_retries: ", self._num_retries)
            print("self._num_epoch: ", self._num_epoch)
            print("self._num_yielded: ", self._num_yielded)
            print("self.skip_rows: ", self.skip_rows)
            print("self.use_chunk_loading: ", self.use_chunk_loading)
            print("self._path_to_videos: ", self._path_to_videos)
            print("self._labels: ", self._labels)
            print("self._spatial_temporal_idx: ", self._spatial_temporal_idx)
            print("self.cur_iter: ", self.cur_iter)
            print("self.chunk_epoch: ", self.chunk_epoch)
            print("self.epoch: ", self.epoch)
            print("self.skip_rows: ", self.skip_rows)


    def _construct_loader(self):
        """
        Construct the video loader.
        """
        path_to_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR, "{}.csv".format(self.mode)
        )
        assert pathmgr.exists(path_to_file), "{} dir not found".format(
            path_to_file
        )

        self._path_to_videos = []
        self._labels = []

        self._region_labels = []

        self._spatial_temporal_idx = []
        self.cur_iter = 0
        self.chunk_epoch = 0
        self.epoch = 0.0
        self.skip_rows = self.cfg.DATA.SKIP_ROWS
        self.reg_annotation_path = self.cfg.DATA.REGION_ANNOTATION_PATH
        self.video_path = self.cfg.DATA.PATH_TO_DATA_DIR

        with pathmgr.open(path_to_file, "r") as f:
            if self.use_chunk_loading:
                rows = self._get_chunk(f, self.cfg.DATA.LOADER_CHUNK_SIZE)
            else:
                rows = f.read().splitlines()

            print("rows: ", rows)
            for clip_idx, path_label in enumerate(rows):
                print("clip_idx: ", clip_idx, " and path_label: ", path_label)
                fetch_info = path_label.split(
                    self.cfg.DATA.PATH_LABEL_SEPARATOR
                )
                if len(fetch_info) == 2:
                    path, label = fetch_info
                elif len(fetch_info) == 3:
                    path, fn, label = fetch_info
                elif len(fetch_info) == 1:
                    path, label = fetch_info[0], 0
                else:
                    raise RuntimeError(
                        "Failed to parse video fetch {} info {} retries.".format(
                            path_to_file, fetch_info
                        )
                    )

                print("fetch_info: ", fetch_info)
                self._path_to_videos.append(
                    os.path.join(self.cfg.DATA.PATH_PREFIX, path)
                )
                self._labels.append(int(label))
                self._spatial_temporal_idx.append(0)
                self._video_meta[clip_idx] = {}

            with open(self.reg_annotation_path, 'rb') as f:
                self._region_labels = pickle.load(f)
        
        assert (
            len(self._path_to_videos) > 0
        ), "Failed to load Kinetics split {} from {}".format(
            self._split_idx, path_to_file
        )
        logger.info(
            "Constructing kinetics dataloader (size: {} skip_rows {}) from {} ".format(
                len(self._path_to_videos), self.skip_rows, path_to_file
            )
        )

    def _set_epoch_num(self, epoch):
        self.epoch = epoch

    def _get_chunk(self, path_to_file, chunksize):
        try:
            for chunk in pandas.read_csv(
                path_to_file,
                chunksize=self.cfg.DATA.LOADER_CHUNK_SIZE,
                skiprows=self.skip_rows,
            ):
                break
        except Exception:
            self.skip_rows = 0
            return self._get_chunk(path_to_file, chunksize)
        else:
            return pandas.array(chunk.values.flatten(), dtype="string")

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """

        if self.mode in ["train", "val"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1] 
            # crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
            crop_size = 256


            assert min_scale == max_scale
            assert max_scale == crop_size

        elif self.mode in ["test"]:
            temporal_sample_index = (
                self._spatial_temporal_idx[index]
                // self.cfg.TEST.NUM_SPATIAL_CROPS
            )

            
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = (
                (
                    self._spatial_temporal_idx[index]
                    % self.cfg.TEST.NUM_SPATIAL_CROPS
                )
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else 1
            )

            min_scale, max_scale, crop_size = (
                [self.cfg.DATA.TEST_CROP_SIZE] * 3
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else [self.cfg.DATA.TRAIN_JITTER_SCALES[0]] * 2
                + [self.cfg.DATA.TEST_CROP_SIZE]
            )
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale}) == 1

        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )


        num_decode = 1

        min_scale, max_scale, crop_size = [min_scale], [max_scale], [crop_size]


        # DEBUG: HARDCODE INDEX FOR PREPROCESSED VIDEO
        # index = 30 for test
        # index = 58
        
        # Try to decode and sample a clip from a video. If the video can not be
        # decoded, repeatly find a random video replacement that can be decoded.
        for i_try in range(self._num_retries):
            video_container = None

            try:
                video_container = container.get_video_container(
                    self._path_to_videos[index],
                    self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                    self.cfg.DATA.DECODING_BACKEND,
                )
            except Exception as e:
                logger.info(
                    "Failed to load video from {} with error {}".format(
                        self._path_to_videos[index], e
                    )
                )
            # Select a random video if the current video was not able to access.
            if video_container is None:
                logger.warning(
                    "Failed to meta load video idx {} from {}; trial {}".format(
                        index, self._path_to_videos[index], i_try
                    )
                )
                if self.mode not in ["test"] and i_try > self._num_retries // 8:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            frames_decoded, time_idx_decoded, bbox_index_decoded = (
                [None] * num_decode,
                [None] * num_decode,
                [None] * num_decode
            )

            # for i in range(num_decode):
            num_frames = [self.cfg.DATA.NUM_FRAMES]

            sampling_rate = self.cfg.DATA.SAMPLING_RATE
            sampling_rate = [sampling_rate]

            assert (
                len(min_scale)
                == len(max_scale)
                == len(crop_size)
                == num_decode
            )

            target_fps = self.cfg.DATA.TARGET_FPS
            # Decode video. Meta info is used to perform selective decoding.
            frames, time_idx, tdiff, bbox_index = decoder.decode(
                video_container,
                sampling_rate, # 8
                num_frames,   # 8
                temporal_sample_index,
                self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
                video_meta=self._video_meta[index]
                if len(self._video_meta) < 5e6
                else {},  # do not cache on huge datasets
                target_fps=target_fps,
                backend=self.cfg.DATA.DECODING_BACKEND,
                use_offset=self.cfg.DATA.USE_OFFSET_SAMPLING,
                max_spatial_scale=min_scale[0]
                if all(x == min_scale[0] for x in min_scale)
                else 0,  # if self.mode in ["test"] else 0,
                time_diff_prob=self.p_convert_dt
                if self.mode in ["train"]
                else 0.0,
                temporally_rnd_clips=True,
                min_delta=self.cfg.CONTRASTIVE.DELTA_CLIPS_MIN,
                max_delta=self.cfg.CONTRASTIVE.DELTA_CLIPS_MAX,
            )

            frames_decoded = frames
            time_idx_decoded = time_idx
            bbox_index_decoded = bbox_index

            region_labels = [0 for i in range(14)]
            video_idx_start = len(self.video_path) + 6

            if index not in self.kpts:
                index_str = self._path_to_videos[index]
                video_idx = index_str[video_idx_start:-10]
                with open(self.video_path + f"/video{video_idx}_kpts", 'rb') as handle:
                    kpts = pickle.load(handle)
                    self.kpts[index] = kpts

            index_str = self._path_to_videos[index]
            video_idx = index_str[video_idx_start:-10]
            if f"video{video_idx}" in self._region_labels:
                regions = self._region_labels[f"video{video_idx}"]
                for reg in regions:
                    if reg == '':
                        continue

                    curr_reg = reg.split(",")[1]

                    if curr_reg == 'all_regions':
                        region_labels = [1 for i in range(14)]
                    elif curr_reg == '':
                        continue
                    else:
                        region_labels[int(curr_reg)] = 1
                
            kpts = self.kpts[index]
            bboxes = []

            # If decoding failed (wrong format, video is too short, and etc),
            # select another video.
            if frames_decoded is None or None in frames_decoded:
                logger.warning(
                    "Failed to decode video idx {} from {}; trial {}".format(
                        index, self._path_to_videos[index], i_try
                    )
                )
                if (
                    self.mode not in ["test"]
                    and (i_try % (self._num_retries // 8)) == 0
                ):
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            for frame_num in bbox_index_decoded[0].cpu().detach().numpy():
                frame_name = "frame" + str(frame_num)
                vid_name = self._path_to_videos[index].split("/")[-1]
                vid_name = vid_name[:-10] + ".mp4"
                bounding_boxes = kpts[vid_name][frame_name]

                
                small_bboxes = []
                for box in bounding_boxes:
                    actual_pts = []
                    x_pts = []
                    y_pts = []
                    for coord in box:
                        x_pts.append(int(coord[0]))
                        y_pts.append(int(coord[1]))
                    
                    x_min, x_max = min(x_pts), max(x_pts)
                    y_min, y_max = min(y_pts), max(y_pts)

                    actual_pts = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
                    small_bboxes.append(actual_pts)
                bboxes.append(small_bboxes)

            num_aug = 1
            num_out = num_aug * num_decode


            f_out, time_idx_out = [None] * num_out, [None] * num_out
            idx = -1
            label = self._labels[index]



            if self.cfg.MODEL.NUM_CLASSES == 2 and label > 0:
                label = 1

            for i in range(num_decode):
                for _ in range(num_aug):
                    idx += 1
                    f_out[idx] = frames_decoded[i].clone()
                    time_idx_out[idx] = time_idx_decoded[i, :]

                    f_out[idx] = f_out[idx].float()
                    f_out[idx] = f_out[idx] / 255.0

                    # T H W C -> C T H W.
                    f_out[idx] = f_out[idx].permute(3, 0, 1, 2)

                    relative_scales = None
                    relative_aspect = None

                    f_out[idx] = utils.spatial_sampling(
                        f_out[idx],
                        spatial_idx=spatial_sample_index,
                        min_scale=min_scale[i],
                        max_scale=max_scale[i],
                        crop_size=crop_size[i],
                        random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                        inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
                        aspect_ratio=relative_aspect,
                        scale=relative_scales,
                        motion_shift=self.cfg.DATA.TRAIN_JITTER_MOTION_SHIFT
                        if self.mode in ["train"]
                        else False,
                    )


                    f_out[idx] = utils.pack_pathway_output(self.cfg, f_out[idx])
            frames = f_out[0] if num_out == 1 else f_out
            time_idx = np.array(time_idx_out)
            if num_aug > 1:
                label = [label] * num_aug
                index = [index] * num_aug

            return frames, label, index, time_idx, {}, torch.Tensor(bboxes), torch.Tensor(region_labels)
        else:
            raise RuntimeError(
                "Failed to fetch video idx {} from {}; after {} trials".format(
                    index, self._path_to_videos[index], i_try
                )
            )

    def _frame_to_list_img(self, frames):
        img_list = [
            transforms.ToPILImage()(frames[i]) for i in range(frames.size(0))
        ]
        return img_list

    def _list_img_to_frames(self, img_list):
        img_list = [transforms.ToTensor()(img) for img in img_list]
        return torch.stack(img_list)

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return self.num_videos

    @property
    def num_videos(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)
