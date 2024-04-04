from multiprocessing import Process, Manager
from typing import Type, List, Callable
import glob
import os
import pickle
from subprocess import call

from pyrep.const import RenderMode

from rlbench import ObservationConfig
from rlbench.backend.observation import Observation
from rlbench.demo import Demo
from rlbench.backend.task import Task
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend.utils import task_file_to_task_class
from rlbench.environment import Environment
from rlbench.task_environment import (
    TaskEnvironment,
    _MAX_RESET_ATTEMPTS,
    _MAX_DEMO_ATTEMPTS
)
import rlbench.backend.task as task

from PIL import Image
from rlbench.backend import utils
from rlbench.backend.const import (
    LEFT_SHOULDER_RGB_FOLDER,
    LEFT_SHOULDER_DEPTH_FOLDER,
    LEFT_SHOULDER_MASK_FOLDER,
    RIGHT_SHOULDER_RGB_FOLDER,
    RIGHT_SHOULDER_DEPTH_FOLDER,
    RIGHT_SHOULDER_MASK_FOLDER,
    OVERHEAD_RGB_FOLDER,
    OVERHEAD_DEPTH_FOLDER,
    OVERHEAD_MASK_FOLDER,
    WRIST_RGB_FOLDER,
    WRIST_DEPTH_FOLDER,
    WRIST_MASK_FOLDER,
    FRONT_RGB_FOLDER,
    FRONT_DEPTH_FOLDER,
    FRONT_MASK_FOLDER,
    DEPTH_SCALE,
    IMAGE_FORMAT,
    LOW_DIM_PICKLE,
    VARIATION_NUMBER,
    VARIATIONS_ALL_FOLDER,
    EPISODES_FOLDER,
    EPISODE_FOLDER,
    VARIATION_DESCRIPTIONS
)
import numpy as np

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('save_path',
                    '/tmp/rlbench_data/',
                    'Where to save the demos.')
flags.DEFINE_string('demo_path',
                    '/tmp/rlbench_data/',
                    'Where to existing demos.')
flags.DEFINE_list('tasks', [],
                  'The tasks to collect. If empty, all tasks are collected.')
flags.DEFINE_list('image_size', [128, 128],
                  'The size of the images tp save.')
flags.DEFINE_enum('renderer',  'opengl3', ['opengl', 'opengl3'],
                  'The renderer to use. opengl does not include shadows, '
                  'but is faster.')
flags.DEFINE_integer('processes', 1,
                     'The number of parallel processes during collection.')
flags.DEFINE_integer('variations', -1,
                     'Number of variations to collect per task. -1 for all.')
flags.DEFINE_bool('all_variations', True,
                  'Include all variations when sampling epsiodes')


class CustomizedTaskEnvironment(TaskEnvironment):
    """Modify TaskEnvironment class, so that we can provide random seed
    when generating live demos.
    """
    def get_demos(self, amount: int, live_demos: bool = False,
                  image_paths: bool = False,
                  callable_each_step: Callable[[Observation], None] = None,
                  max_attempts: int = _MAX_DEMO_ATTEMPTS,
                  random_selection: bool = True,
                  from_episode_number: int = 0,
                  random_seed_state = None,
                  ) -> List[Demo]:
        """Negative means all demos"""

        if not live_demos and (self._dataset_root is None
                               or len(self._dataset_root) == 0):
            raise RuntimeError(
                "Can't ask for a stored demo when no dataset root provided.")

        if not live_demos:
            if self._dataset_root is None or len(self._dataset_root) == 0:
                raise RuntimeError(
                    "Can't ask for stored demo when no dataset root provided.")
            demos = utils.get_stored_demos(
                amount, image_paths, self._dataset_root, self._variation_number,
                self._task.get_name(), self._obs_config,
                random_selection, from_episode_number)
        else:
            ctr_loop = self._robot.arm.joints[0].is_control_loop_enabled()
            self._robot.arm.set_control_loop_enabled(True)
            demos = self._get_live_demos(
                amount, callable_each_step, max_attempts, random_seed_state)
            self._robot.arm.set_control_loop_enabled(ctr_loop)
        return demos

    def _get_live_demos(self, amount: int,
                        callable_each_step: Callable[
                            [Observation], None] = None,
                        max_attempts: int = _MAX_DEMO_ATTEMPTS,
                        random_seed_state = None) -> List[Demo]:
        demos = []
        for i in range(amount):
            attempts = max_attempts
            while attempts > 0:
                if random_seed_state is None:
                    random_seed = np.random.get_state()
                else:
                    random_seed = random_seed_state
                    np.random.set_state(random_seed)
                self.reset()
                try:
                    demo = self._scene.get_demo(
                        callable_each_step=callable_each_step)
                    demo.random_seed = random_seed
                    demos.append(demo)
                    break
                except Exception as e:
                    attempts -= 1
                    logging.info('Bad demo. ' + str(e) + ' Attempts left: ' + str(attempts))
            if attempts <= 0:
                raise RuntimeError(
                    'Could not collect demos. Maybe a problem with the task?')
        return demos



class CustomizedEnvironment(Environment):

    def get_task(self, task_class: Type[Task]) -> CustomizedTaskEnvironment:

        # If user hasn't called launch, implicitly call it.
        if self._pyrep is None:
            self.launch()

        self._scene.unload()
        task = task_class(self._pyrep, self._robot)
        self._prev_task = task
        return CustomizedTaskEnvironment(
            self._pyrep, self._robot, self._scene, task,
            self._action_mode, self._dataset_root, self._obs_config,
            self._static_positions, self._attach_grasped_objects)



def check_and_make(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def save_demo(demo, example_path, variation):

    # Save image data first, and then None the image data, and pickle
    left_shoulder_rgb_path = os.path.join(
        example_path, LEFT_SHOULDER_RGB_FOLDER)
    left_shoulder_depth_path = os.path.join(
        example_path, LEFT_SHOULDER_DEPTH_FOLDER)
    left_shoulder_mask_path = os.path.join(
        example_path, LEFT_SHOULDER_MASK_FOLDER)
    right_shoulder_rgb_path = os.path.join(
        example_path, RIGHT_SHOULDER_RGB_FOLDER)
    right_shoulder_depth_path = os.path.join(
        example_path, RIGHT_SHOULDER_DEPTH_FOLDER)
    right_shoulder_mask_path = os.path.join(
        example_path, RIGHT_SHOULDER_MASK_FOLDER)
    overhead_rgb_path = os.path.join(
        example_path, OVERHEAD_RGB_FOLDER)
    overhead_depth_path = os.path.join(
        example_path, OVERHEAD_DEPTH_FOLDER)
    overhead_mask_path = os.path.join(
        example_path, OVERHEAD_MASK_FOLDER)
    wrist_rgb_path = os.path.join(example_path, WRIST_RGB_FOLDER)
    wrist_depth_path = os.path.join(example_path, WRIST_DEPTH_FOLDER)
    wrist_mask_path = os.path.join(example_path, WRIST_MASK_FOLDER)
    front_rgb_path = os.path.join(example_path, FRONT_RGB_FOLDER)
    front_depth_path = os.path.join(example_path, FRONT_DEPTH_FOLDER)
    front_mask_path = os.path.join(example_path, FRONT_MASK_FOLDER)

    check_and_make(left_shoulder_rgb_path)
    check_and_make(left_shoulder_depth_path)
    check_and_make(left_shoulder_mask_path)
    check_and_make(right_shoulder_rgb_path)
    check_and_make(right_shoulder_depth_path)
    check_and_make(right_shoulder_mask_path)
    check_and_make(overhead_rgb_path)
    check_and_make(overhead_depth_path)
    check_and_make(overhead_mask_path)
    check_and_make(wrist_rgb_path)
    check_and_make(wrist_depth_path)
    check_and_make(wrist_mask_path)
    check_and_make(front_rgb_path)
    check_and_make(front_depth_path)
    check_and_make(front_mask_path)

    for i, obs in enumerate(demo):
        left_shoulder_rgb = Image.fromarray(obs.left_shoulder_rgb)
        left_shoulder_depth = utils.float_array_to_rgb_image(
            obs.left_shoulder_depth, scale_factor=DEPTH_SCALE)
        left_shoulder_mask = Image.fromarray(
            (obs.left_shoulder_mask * 255).astype(np.uint8))
        right_shoulder_rgb = Image.fromarray(obs.right_shoulder_rgb)
        right_shoulder_depth = utils.float_array_to_rgb_image(
            obs.right_shoulder_depth, scale_factor=DEPTH_SCALE)
        right_shoulder_mask = Image.fromarray(
            (obs.right_shoulder_mask * 255).astype(np.uint8))
        overhead_rgb = Image.fromarray(obs.overhead_rgb)
        overhead_depth = utils.float_array_to_rgb_image(
            obs.overhead_depth, scale_factor=DEPTH_SCALE)
        overhead_mask = Image.fromarray(
            (obs.overhead_mask * 255).astype(np.uint8))
        wrist_rgb = Image.fromarray(obs.wrist_rgb)
        wrist_depth = utils.float_array_to_rgb_image(
            obs.wrist_depth, scale_factor=DEPTH_SCALE)
        wrist_mask = Image.fromarray((obs.wrist_mask * 255).astype(np.uint8))
        front_rgb = Image.fromarray(obs.front_rgb)
        front_depth = utils.float_array_to_rgb_image(
            obs.front_depth, scale_factor=DEPTH_SCALE)
        front_mask = Image.fromarray((obs.front_mask * 255).astype(np.uint8))

        left_shoulder_rgb.save(
            os.path.join(left_shoulder_rgb_path, IMAGE_FORMAT % i))
        left_shoulder_depth.save(
            os.path.join(left_shoulder_depth_path, IMAGE_FORMAT % i))
        left_shoulder_mask.save(
            os.path.join(left_shoulder_mask_path, IMAGE_FORMAT % i))
        right_shoulder_rgb.save(
            os.path.join(right_shoulder_rgb_path, IMAGE_FORMAT % i))
        right_shoulder_depth.save(
            os.path.join(right_shoulder_depth_path, IMAGE_FORMAT % i))
        right_shoulder_mask.save(
            os.path.join(right_shoulder_mask_path, IMAGE_FORMAT % i))
        overhead_rgb.save(
            os.path.join(overhead_rgb_path, IMAGE_FORMAT % i))
        overhead_depth.save(
            os.path.join(overhead_depth_path, IMAGE_FORMAT % i))
        overhead_mask.save(
            os.path.join(overhead_mask_path, IMAGE_FORMAT % i))
        wrist_rgb.save(os.path.join(wrist_rgb_path, IMAGE_FORMAT % i))
        wrist_depth.save(os.path.join(wrist_depth_path, IMAGE_FORMAT % i))
        wrist_mask.save(os.path.join(wrist_mask_path, IMAGE_FORMAT % i))
        front_rgb.save(os.path.join(front_rgb_path, IMAGE_FORMAT % i))
        front_depth.save(os.path.join(front_depth_path, IMAGE_FORMAT % i))
        front_mask.save(os.path.join(front_mask_path, IMAGE_FORMAT % i))

        # We save the images separately, so set these to None for pickling.
        obs.left_shoulder_rgb = None
        obs.left_shoulder_depth = None
        obs.left_shoulder_point_cloud = None
        obs.left_shoulder_mask = None
        obs.right_shoulder_rgb = None
        obs.right_shoulder_depth = None
        obs.right_shoulder_point_cloud = None
        obs.right_shoulder_mask = None
        obs.overhead_rgb = None
        obs.overhead_depth = None
        obs.overhead_point_cloud = None
        obs.overhead_mask = None
        obs.wrist_rgb = None
        obs.wrist_depth = None
        obs.wrist_point_cloud = None
        obs.wrist_mask = None
        obs.front_rgb = None
        obs.front_depth = None
        obs.front_point_cloud = None
        obs.front_mask = None

    # Save the low-dimension data
    with open(os.path.join(example_path, LOW_DIM_PICKLE), 'wb') as f:
        pickle.dump(demo, f)

    with open(os.path.join(example_path, VARIATION_NUMBER), 'wb') as f:
        pickle.dump(variation, f)


def verify_demo_and_rgbs(demo, example_path):
    left_shoulder_rgb_path = os.path.join(
        example_path, LEFT_SHOULDER_RGB_FOLDER)
    left_shoulder_depth_path = os.path.join(
        example_path, LEFT_SHOULDER_DEPTH_FOLDER)
    left_shoulder_mask_path = os.path.join(
        example_path, LEFT_SHOULDER_MASK_FOLDER)
    right_shoulder_rgb_path = os.path.join(
        example_path, RIGHT_SHOULDER_RGB_FOLDER)
    right_shoulder_depth_path = os.path.join(
        example_path, RIGHT_SHOULDER_DEPTH_FOLDER)
    right_shoulder_mask_path = os.path.join(
        example_path, RIGHT_SHOULDER_MASK_FOLDER)
    overhead_rgb_path = os.path.join(
        example_path, OVERHEAD_RGB_FOLDER)
    overhead_depth_path = os.path.join(
        example_path, OVERHEAD_DEPTH_FOLDER)
    overhead_mask_path = os.path.join(
        example_path, OVERHEAD_MASK_FOLDER)
    wrist_rgb_path = os.path.join(example_path, WRIST_RGB_FOLDER)
    wrist_depth_path = os.path.join(example_path, WRIST_DEPTH_FOLDER)
    wrist_mask_path = os.path.join(example_path, WRIST_MASK_FOLDER)
    front_rgb_path = os.path.join(example_path, FRONT_RGB_FOLDER)
    front_depth_path = os.path.join(example_path, FRONT_DEPTH_FOLDER)
    front_mask_path = os.path.join(example_path, FRONT_MASK_FOLDER)

    num_ls_rgb = len(os.listdir(left_shoulder_rgb_path))
    num_ls_depth = len(os.listdir(left_shoulder_depth_path))
    num_ls_mask = len(os.listdir(left_shoulder_mask_path))
    num_rs_rgb = len(os.listdir(right_shoulder_rgb_path))
    num_rs_depth = len(os.listdir(right_shoulder_depth_path))
    num_rs_mask = len(os.listdir(right_shoulder_mask_path))
    num_oh_rgb = len(os.listdir(overhead_rgb_path))
    num_oh_depth = len(os.listdir(overhead_depth_path))
    num_oh_mask = len(os.listdir(overhead_mask_path))
    num_wrist_rgb = len(os.listdir(wrist_rgb_path))
    num_wrist_depth = len(os.listdir(wrist_depth_path))
    num_wrist_mask = len(os.listdir(wrist_mask_path))
    num_front_rgb = len(os.listdir(front_rgb_path))
    num_front_depth = len(os.listdir(front_depth_path))
    num_front_mask = len(os.listdir(front_mask_path))

    print(len(demo), num_ls_rgb, num_rs_rgb, num_oh_rgb, num_front_rgb)
    assert len(demo) == num_ls_rgb
    assert len(demo) == num_ls_depth
    assert len(demo) == num_ls_mask
    assert len(demo) == num_rs_rgb
    assert len(demo) == num_rs_depth
    assert len(demo) == num_rs_mask
    assert len(demo) == num_oh_rgb
    assert len(demo) == num_oh_depth
    assert len(demo) == num_oh_mask
    assert len(demo) == num_front_rgb
    assert len(demo) == num_front_depth
    assert len(demo) == num_front_mask
    assert len(demo) == num_wrist_rgb
    assert len(demo) == num_wrist_depth
    assert len(demo) == num_wrist_mask


def run_all_variations(i, lock, task_index, variation_count, results, file_lock, tasks):
    """Each thread will choose one task and variation, and then gather
    all the episodes_per_task for that variation."""

    # Initialise each thread with random seed
    np.random.seed(None)
    num_tasks = len(tasks)

    img_size = list(map(int, FLAGS.image_size))

    obs_config = ObservationConfig()
    obs_config.set_all(True)
    obs_config.right_shoulder_camera.image_size = img_size
    obs_config.left_shoulder_camera.image_size = img_size
    obs_config.overhead_camera.image_size = img_size
    obs_config.wrist_camera.image_size = img_size
    obs_config.front_camera.image_size = img_size

    # Store depth as 0 - 1
    obs_config.right_shoulder_camera.depth_in_meters = False
    obs_config.left_shoulder_camera.depth_in_meters = False
    obs_config.overhead_camera.depth_in_meters = False
    obs_config.wrist_camera.depth_in_meters = False
    obs_config.front_camera.depth_in_meters = False

    # We want to save the masks as rgb encodings.
    obs_config.left_shoulder_camera.masks_as_one_channel = False
    obs_config.right_shoulder_camera.masks_as_one_channel = False
    obs_config.overhead_camera.masks_as_one_channel = False
    obs_config.wrist_camera.masks_as_one_channel = False
    obs_config.front_camera.masks_as_one_channel = False

    if FLAGS.renderer == 'opengl':
        obs_config.right_shoulder_camera.render_mode = RenderMode.OPENGL
        obs_config.left_shoulder_camera.render_mode = RenderMode.OPENGL
        obs_config.overhead_camera.render_mode = RenderMode.OPENGL
        obs_config.wrist_camera.render_mode = RenderMode.OPENGL
        obs_config.front_camera.render_mode = RenderMode.OPENGL

    rlbench_env = CustomizedEnvironment(
        action_mode=MoveArmThenGripper(JointVelocity(), Discrete()),
        obs_config=obs_config,
        headless=True)
    rlbench_env.launch()

    task_env = None

    tasks_with_problems = results[i] = ''

    while True:
        # with lock:
        if task_index.value >= num_tasks:
            print('Process', i, 'finished')
            break

        t = tasks[task_index.value]

        task_env = rlbench_env.get_task(t)
        possible_variations = task_env.variation_count()

        variation_path = os.path.join(
            FLAGS.save_path, task_env.get_name(),
            VARIATIONS_ALL_FOLDER)
        check_and_make(variation_path)

        episodes_path = os.path.join(variation_path, EPISODES_FOLDER)
        check_and_make(episodes_path)

        existing_episodes_path = os.path.join(
            FLAGS.demo_path, task_env.get_name(), VARIATIONS_ALL_FOLDER, EPISODES_FOLDER
        )
        episodes_per_task = len(glob.glob(os.path.join(existing_episodes_path, "episode*")))

        abort_variation = False
        for ex_idx in range(episodes_per_task):
            attempts = 100
            existing_episode_path = os.path.join(
                existing_episodes_path, EPISODE_FOLDER % ex_idx
            )
            episode_path = os.path.join(episodes_path, EPISODE_FOLDER % ex_idx)

            while attempts > 0:
                try:
                    #variation = np.random.randint(possible_variations)
                    variation = pickle.load(
                        open(
                            os.path.join(existing_episode_path, VARIATION_NUMBER),
                            'rb'
                        )
                    )
                    existing_demo = pickle.load(
                        open(
                            os.path.join(existing_episode_path, LOW_DIM_PICKLE),
                            'rb'
                        )
                    )
                    random_seed_state = existing_demo.random_seed
                    task_env = rlbench_env.get_task(t)
                    task_env.set_variation(variation)
                    descriptions, obs = task_env.reset()

                    print('Process', i, '// Task:', task_env.get_name(),
                          '// Variation:', variation, '// Demo:', ex_idx)

                    # TODO: for now we do the explicit looping.
                    demo, = task_env.get_demos(
                        amount=1,
                        live_demos=True,
                        random_seed_state=random_seed_state)

                    with file_lock:
                        save_demo(demo, episode_path, variation)

                        with open(os.path.join(
                                episode_path, VARIATION_DESCRIPTIONS), 'wb') as f:
                            pickle.dump(descriptions, f)

                    # verify demo
                    verify_demo_and_rgbs(demo, episode_path)
                except Exception as e:
                    attempts -= 1

                    # clean up previously saved RGBs
                    call(['rm', '-r', episode_path])

                    if attempts > 0:
                        continue
                    problem = (
                        'Process %d failed collecting task %s (variation: %d, '
                        'example: %d). Skipping this task/variation.\n%s\n' % (
                            i, task_env.get_name(), variation, ex_idx,
                            str(e))
                    )
                    print(problem)
                    tasks_with_problems += problem
                    abort_variation = True
                    break
                break
            if abort_variation:
                break

        # with lock:
        task_index.value += 1

    results[i] = tasks_with_problems
    rlbench_env.shutdown()


def main(argv):

    task_files = [t.replace('.py', '') for t in os.listdir(task.TASKS_PATH)
                  if t != '__init__.py' and t.endswith('.py')]

    if len(FLAGS.tasks) > 0:
        for t in FLAGS.tasks:
            if t not in task_files:
                raise ValueError('Task %s not recognised!.' % t)
        task_files = FLAGS.tasks

    tasks = [task_file_to_task_class(t) for t in task_files]

    manager = Manager()

    result_dict = manager.dict()
    file_lock = manager.Lock()

    task_index = manager.Value('i', 0)
    variation_count = manager.Value('i', 0)
    lock = manager.Lock()

    check_and_make(FLAGS.save_path)

    # multiprocessing for all_variations not support (for now)
    run_all_variations(0, lock, task_index, variation_count, result_dict, file_lock, tasks)

    print('Data collection done!')
    for i in range(FLAGS.processes):
        print(result_dict[i])


if __name__ == '__main__':
  app.run(main)

