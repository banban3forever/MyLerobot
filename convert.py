"""
Script to convert Aloha hdf5 data to the LeRobot dataset v2.0 format.

Example usage: uv run examples/aloha_real/convert_aloha_data_to_lerobot.py --raw-dir /path/to/raw/data --repo-id <org>/<dataset-name>
"""

import dataclasses
from pathlib import Path
import shutil
from typing import Literal
import os
import h5py
from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import torch
import tqdm

# import tyro

# UNI_STATE_INDICES = [50,51,52,53,54,55] + [60,] + [0,1,2,3,4,5,] + [10,]


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    motors = [
        "left_arm1",
        "left_arm2",
        "left_arm3",
        "left_arm4",
        "left_arm5",
        "left_arm6",
        "left_arm7",
        "left_arm8",
        "left_gripper",
        "right_arm1",
        "right_arm2",
        "right_arm3",
        "right_arm4",
        "right_arm5",
        "right_arm6",
        "right_arm7",
        "right_arm8",
        "right_gripper",
    ]
    cameras = [
        "camera_front",
        "camera_left",
        "camera_right",
        "camera_top",
        "camera_wrist_left",
        "camera_wrist_right",
    ]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
    }

    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    for cam in cameras:
        if cam == "camera_top" or cam == "camera_front":
            features[f"observation.images.{cam}"] = {
                "dtype": mode,
                "shape": (3, 720, 1280),
                "names": [
                    "channels",
                    "height",
                    "width",
                ],
            }
            features[f"observation.depths.{cam}"] = {
                "dtype": "uint16",
                "shape": (720, 1280),
                "names": [
                    "height",
                    "width",
                ],
            }
        else:
            features[f"observation.images.{cam}"] = {
                "dtype": mode,
                "shape": (3, 480, 640),
                "names": [
                    "channels",
                    "height",
                    "width",
                ],
            }
            features[f"observation.depths.{cam}"] = {
                "dtype": "uint16",
                "shape": (480, 640),
                "names": [
                    "height",
                    "width",
                ],
            }

    if Path(HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=30,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )



def load_raw_images_per_camera(ep: str) -> dict[str, np.ndarray]:
    import cv2

    def processIMG(raw_img):
        # deocode and convert to rgb
        img_array = np.frombuffer(raw_img, dtype=np.uint8)
        img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = np.transpose(img_rgb, (2, 0, 1))  # from HWC to CHW
        return img_rgb

    imgs_per_cam = {
        "camera_front":[],
        "camera_left":[],
        "camera_right":[],
        "camera_top":[],
        "camera_wrist_left":[],
        "camera_wrist_right":[],
    }

    with h5py.File(ep) as f:
        for cam in imgs_per_cam.keys():
            raw_img = f[f'/observations/rgb_images/{cam}']
            for i in range(len(raw_img)):
                imgs_per_cam[cam].append(processIMG(raw_img[i]))
    

    return imgs_per_cam

def load_raw_depths_per_camera(ep: str) -> dict[str, np.ndarray]:
    import cv2

    def processIMG(raw_depth):
        # deocode and convert to rgb
        depth_array = np.array(raw_depth, dtype=np.uint8)
        depth = cv2.imdecode(depth_array, cv2.IMREAD_UNCHANGED)
        return depth

    depths_per_cam = {
        "camera_front":[],
        "camera_left":[],
        "camera_right":[],
        "camera_top":[],
        "camera_wrist_left":[],
        "camera_wrist_right":[],
    }

    with h5py.File(ep) as f:
        for cam in depths_per_cam.keys():
            raw_depth = f[f'/observations/depth_images/{cam}']
            for i in range(len(raw_depth)):
                depths_per_cam[cam].append(processIMG(raw_depth[i]))
    

    return depths_per_cam


def load_raw_episode_data(
    ep: Path,
) -> tuple[
    dict[str, np.ndarray],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    with h5py.File(ep) as f:
        state_arm = np.array(f["/puppet/arm_joint_position"]).astype(np.float32)
        state_end_effector = np.array(f["/puppet/end_effector"]).astype(np.float32) ### TODO Haven't save eef yet.
        state_hand = np.array(f["/puppet/hand_joint_position"]).astype(np.float32)

        action_arm = np.array(f["/master/arm_joint_position"]).astype(np.float32)
        action_hand = np.array(f["/master/hand_joint_position"]).astype(np.float32)

    arm_joint_num = 8
    hand_joint_num = 1

    state = np.concatenate(
        [
            state_arm[:, :arm_joint_num],
            state_hand[:, :hand_joint_num],
            state_arm[:, arm_joint_num:],
            state_hand[:, hand_joint_num:],
        ],
        axis=-1,
    )

    action = np.concatenate(
        [
            action_arm[:, :arm_joint_num],
            action_hand[:, :hand_joint_num],
            action_arm[:, arm_joint_num:],
            action_hand[:, hand_joint_num:],
        ],
        axis=-1,
    )

    imgs_per_cam = load_raw_images_per_camera(ep)
    depths_per_cam = load_raw_depths_per_camera(ep)

    return imgs_per_cam, depths_per_cam, state, action


def populate_dataset(
    dataset: LeRobotDataset,
    ep_files: list[Path],
    task: str,
    episodes: list[int] | None = None,
) -> LeRobotDataset:
    if episodes is None:
        episodes = range(len(ep_files))

    for ep_idx in tqdm.tqdm(episodes):
        ep_path = ep_files[ep_idx]

        imgs_per_cam, depths_per_cam, state, action = load_raw_episode_data(ep_path)
        num_frames = state.shape[0]

        for i in range(num_frames):
            frame = {
                "observation.state": state[i],
                "action": action[i],
            }

            for camera, img_array in imgs_per_cam.items():
                frame[f"observation.images.{camera}"] = img_array[i]

            for camera, depth_array in depths_per_cam.items():
                frame[f"observation.depths.{camera}"] = depth_array[i]

            dataset.add_frame(frame, task=task)

        dataset.save_episode()

    return dataset


def port_aloha(
    raw_dir: Path,
    repo_id: str,
    raw_repo_id: str | None = None,
    *,
    episodes: list[int] | None = None,
    push_to_hub: bool = False,
    is_mobile: bool = False,
    mode: Literal["video", "image"] = "video",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    if (HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    ep_files = [raw_dir / Path(path) for path in os.listdir(raw_dir)]

    dataset = create_empty_dataset(
        repo_id,
        robot_type="mobile_aloha" if is_mobile else "aloha",
        mode=mode,
        has_effort=False,
        has_velocity=False,
        dataset_config=dataset_config,
    )
    for task_dir in os.listdir(raw_dir):
        task_dir_path = os.path.join(raw_dir, task_dir)
        ep_paths = []
        for root, dirs, files in os.walk(task_dir_path):
            for file in files:
                if file == "trajectory.hdf5":
                    ep_paths.append(os.path.join(root, file))

        if len(ep_paths) == 0:
            continue
        else:
            path = ep_paths[0]
            with h5py.File(path, "r") as f:
                task_description = f['language_instruction'].asstr()[()]
                print(task_description)


        dataset = populate_dataset(
            dataset,
            [Path(path) for path in ep_paths],
            task=task_description,
            episodes=episodes,
        )

        ###
        # feature 'consolidate' has been removed in the new version of lerobot
        ###
        #dataset.consolidate()

    if push_to_hub:
        dataset.push_to_hub()


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ###
    # path is the top dir which contains all the task dirs, we process every child dir to get all tasks.
    ###
    
    ###                                            |---trajectory
    ###                             |--task1_dir---|...
    ### data structure:  root_dir --|--task2_dir
    ###                             |....   

    parser.add_argument("--path", type=str) #/share/project/hejingyang/data/RoboMINDV2
    ###
    # repo_id is the name of the dataset to be created in the hub.
    ###
    parser.add_argument("--repo_id", type=str) #RoboMINDV2
    args = parser.parse_args()
    root_dir = args.path
    port_aloha(
        raw_dir=Path(root_dir),
        repo_id=args.repo_id,
    )