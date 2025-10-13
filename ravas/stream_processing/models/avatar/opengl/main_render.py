import json
import logging
import os.path
import time
from io import BytesIO
from multiprocessing import Queue
from typing import Optional, List, Dict, Literal

import glm
import moderngl
from PIL import Image
from moderngl_window import WindowConfig
from moderngl_window.meta import SceneDescription
from moderngl_window.scene import MeshProgram, Scene
from moderngl_window.scene.camera import KeyboardCamera, Camera
from moderngl_window.text.bitmapped import TextWriter2D

from .gltf2 import Loader
from .mesh import Mesh
from .node import Node
from .texture_light_skeleton_morph_program import TextureLightSkeletonMorphProgram


class GLTFRenderer(WindowConfig):
    """
    Main window for avatar renderer:
    uses different implementations for the rendering-function (see `on_render()` and `_render_xxx()`).
    These are set / enabled via the `enable_xxx()` methods.

    The main rendering function for using the avatar renderer is `_render_queue_io()`
    (see also `enable_render_queue_io()`):
    this function waits on the `input_queue` until blendshape data becomes available (i.e. blocks the rendering loop),
    then renders the new blendshape data to an image (by default JPEG) and sends it via the `output_queue`
    (then waits again on the `input_queue`).

    For more information etc., see moderngl-window examples:
    https://github.com/moderngl/moderngl-window/tree/master/examples
    """

    title = "Avatar Renderer"
    window_size = 1024, 600  # 1280, 720  # TODO make configurable
    aspect_ratio = None

    hidden_window_framerate_limit = -1
    visible = False
    is_headless = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.wnd.mouse_exclusivity = True

        self.play_data_path: Optional[str] = None
        self.model_path: Optional[str] = None

        self.input_queue: Optional[Queue] = None
        self.output_queue: Optional[Queue] = None

        # number of render loops (e.g. for storing images)
        self.step = 0
        # close window after (recorded) data was looped once?
        self.stop_on_data_end = False
        # self.image_format: Literal['png', 'JPEG'] = 'JPEG'

        # output directory for storing images
        self.out_dir: Optional[str] = None

        self.scene: Optional[Scene] = None
        self.camera: Optional[Camera] = None
        self.texture_morph_prog: Optional[MeshProgram] = None

        # TODO make configurable
        self.camera_config = {
            "fov": 10.0,  # ~ zoom for camera (smaller value ~ larger zoom)
            "near": 0.1,
            "far": 20.0,
        }

        # TODO make configurable
        self.camera_position = glm.vec3(0.0, 1.6, 3.2)

        self.render_stats = False
        self.writer: Optional[TextWriter2D] = None

        # position/transformation for the scene
        self.global_matrix: glm.mat4 = glm.mat4()

        # shortcuts for animating model:
        self.model_mesh: Optional[Mesh] = None

        # shortcuts for animating head movements:
        self.node_head: Optional[Node] = None
        self.node_neck: Optional[Node] = None
        self.node_spine2: Optional[Node] = None

        # for "animating" a recorded morph and head rotation sequence:
        self.morph_data: Optional[List[Dict[str, any]]] = None
        self.morph_data_index = 0
        self.morph_data_size = 0
        self.morph_data_start_time = 0
        self.has_morph_data: bool = False

        # register event methods
        self.wnd.close_func = self.on_close

        # register public methods to window:
        self.wnd.load_model = self.load_model
        self.wnd.load_recorded_blend_shapes = self.load_recorded_blend_shapes
        self.wnd.enable_render_and_store_images = self.enable_render_and_store_images
        self.wnd.enable_render_queue_io = self.enable_render_queue_io
        self.wnd._DEBUG_enable_render_to_queue = self._DEBUG_enable_render_to_queue
        self.wnd.clear_queues = self.clear_queues
        self.wnd.enable_stats = self.enable_stats

    def enable_render_queue_io(
        self,
        in_queue: Queue,
        out_queue: Queue,
        image_format: Literal["png", "JPEG"] = "JPEG",
    ):
        """
        enable rendering the images (from blendshapes data via `in_queue`) to `out_queue`

        :param in_queue: the queue for (sending) input data (sent as `dict`, __not__ as JSON!)
        :param out_queue: the queue for (receiving) output data (i.e. rendered images); these will be the images
                          according to `image_format` parameter in `bytes`
        :param image_format: the format for the rendered images
                             __NOTE__ that `JPEG` is considerable faster than `PNG`!
                                      But on the downside, `JPEG` does not support transparency.
        """

        self.output_queue = out_queue
        self.input_queue = in_queue

        self.stop_on_data_end = False

        self._set_image_format(image_format, for_storing=False)
        self.wnd.render_func = self._render_queue_io

    def _DEBUG_enable_render_to_queue(
        self,
        out_queue: Queue,
        stop_on_data_end: bool = True,
        image_format: Literal["png", "JPEG"] = "JPEG",
    ):
        """enable rendering the images (from recorded blendshapes data) to `out_queue`

        NOTE: requires recorded blendshapes (see [[load_blendshapes_data()]]) to be loaded

        :param out_queue: the queue for (receiving) output data (i.e. rendered images); these will be the images
                          according to `image_format` parameter in `bytes`
        :param stop_on_data_end: if `True` the renderer will close after rendering all blendshapes from the recorded
                                 blendshape data once.
        :param image_format: the format for the rendered images
                             __NOTE__ that `JPEG` is considerable faster than `PNG`!
                                      But on the downside, `JPEG` does not support transparency.
        """
        self.output_queue = out_queue

        self.stop_on_data_end = stop_on_data_end

        self._set_image_format(image_format, for_storing=False)
        self.wnd.render_func = self._render_to_queue

    def enable_render_and_store_images(
        self,
        out_dir: Optional[str] = None,
        stop_on_data_end: bool = True,
        image_format: Literal["png", "JPEG"] = "JPEG",
    ):
        """enable rendering the images (from recorded blendshapes data) & store them in directory `out_dir`

        NOTE: requires recorded blendshapes (see [[load_blendshapes_data()]]) to be loaded

        :param out_dir: the directory for storing the rendered images; these will be the images
                          according to `image_format` parameter in `bytes`
        :param stop_on_data_end: if `True` the renderer will close after rendering all blendshapes from the recorded
                                 blendshape data once.
        :param image_format: the format for the rendered images
                             __NOTE__ that `JPEG` is considerable faster than `PNG`!
                                      But on the downside, `JPEG` does not support transparency.
        """
        if not out_dir:
            out_dir = self.out_dir
        else:
            self.out_dir = out_dir

        self.stop_on_data_end = stop_on_data_end

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        self._set_image_format(image_format, for_storing=True)

        self.wnd.render_func = self._render_to_images

    def enable_stats(self, enable: bool):
        """NOTE: only supported in default render function (i.e. if none of the `enable_render_xxx()` were invoked)"""
        if enable:
            self.render_stats = True
            if not self.writer:
                self.writer = TextWriter2D()
        else:
            self.render_stats = False

    def _set_image_format(
        self, image_format: Literal["png", "JPEG"] = "JPEG", for_storing: bool = True
    ):
        if image_format == "png":
            get_func = self._get_png_image
            set_func = (
                self._save_png_image_to_file
                if for_storing
                else self._put_png_image_in_queue
            )
        elif image_format == "JPEG":
            get_func = self._get_jpg_image
            set_func = (
                self._save_jpg_image_to_file
                if for_storing
                else self._put_jpg_image_in_queue
            )
        else:
            logging.getLogger().warning(
                'INVALID image format "%s", using "JPEG" instead', image_format
            )
            get_func = self._get_jpg_image
            set_func = (
                self._save_jpg_image_to_file
                if for_storing
                else self._put_jpg_image_in_queue
            )
        self.get_image_from_buffer = get_func
        self.do_output_image = set_func

    def load_recorded_blend_shapes(self, blend_shapes_path: Optional[str] = None):

        if not blend_shapes_path:
            blend_shapes_path = self.play_data_path
        else:
            self.play_data_path = blend_shapes_path

        self.morph_data = self.load_blendshapes_data(blend_shapes_path)
        self.morph_data_index = 0
        self.morph_data_start_time = 0
        self.morph_data_size = len(self.morph_data)
        self.has_morph_data = True

    def load_model(self, model_path: Optional[str] = None):

        if not model_path:
            model_path = self.model_path
        else:
            self.model_path = model_path

        loader = Loader(SceneDescription(path=model_path))
        self.scene = loader.load()

        self.model_mesh = self.scene.meshes[0]

        self.wnd.mouse_exclusivity = False
        if self.is_headless:
            self.camera = Camera(
                aspect_ratio=self.wnd.aspect_ratio,
                **self.camera_config,
            )
        else:
            self.camera = KeyboardCamera(
                self.wnd.keys,
                aspect_ratio=self.wnd.aspect_ratio,
                **self.camera_config,
            )
            self.camera.velocity = 10.0
            self.camera.mouse_sensitivity = 0.25
            # Use this for gltf scenes for better camera controls
            if self.scene.diagonal_size > 0:
                self.camera.velocity = self.scene.diagonal_size / 5.0

            self.wnd.key_event_func = self._on_camera_key_event

        if self.camera_position:
            self.camera.position = self.camera_position
        else:
            self.camera.position = self.scene.get_center() + glm.vec3(
                0.0, 0.75, self.scene.diagonal_size / 1.5
            )

        self.texture_morph_prog = self.create_program()
        self.global_matrix = glm.mat4()

        # update global transformations for all nodes in (first) scene:
        self.scene.root_nodes[0].calc_model_mat(self.global_matrix)

        self.node_head = self.scene.find_node("Head")
        self.node_neck = self.scene.find_node("Neck")
        self.node_spine2 = self.scene.find_node("Spine2")

    def create_program(self) -> MeshProgram:
        texture_morph_prog = TextureLightSkeletonMorphProgram()
        for mesh in self.scene.meshes:
            instance = texture_morph_prog.apply(mesh)
            if instance is not None:
                if isinstance(instance, MeshProgram):
                    mesh.mesh_program = instance
                    break
                else:
                    raise ValueError(
                        "apply() must return a MeshProgram instance, not {}".format(
                            type(instance)
                        )
                    )
        return texture_morph_prog

    def load_blendshapes_data(self, data_path: str) -> List[Dict]:
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def update_morph_from_data(
        self, mesh: Mesh, time: float, force: bool = False
    ) -> bool:

        if self.morph_data_index >= self.morph_data_size:
            self.morph_data_index = 0

        md = self.morph_data[self.morph_data_index]

        if self.morph_data_index == 0:
            self.morph_data_start_time = time
        elif not force and time - self.morph_data_start_time < 0.05:
            return False
        self.morph_data_start_time = time

        # md_time = md.get('ts', 0) / 1000
        # if self.morph_data_index == 0:
        #     self.morph_data_start_time = md_time
        # else:
        #     curr_time = md_time - self.morph_data_start_time
        #     if curr_time < time:
        #         return

        self.morph_data_index += 1

        return self.apply_transformations(md, mesh)

    def apply_transformations(self, md: Dict[str, any], mesh: Mesh):
        did_modify = False

        blend_shapes: Dict[str, any] = md.get("blendshapes")
        if blend_shapes is not None:
            # apply morphs for face expressions:
            for item in blend_shapes:
                idx = mesh.morph_target_dictionary.get(item.get("categoryName"))
                if idx is None:
                    continue
                score = item.get("score", 0.0)
                mesh.morph_target_influences[idx] = score
            did_modify = True

        rot: List[float] = md.get("transformation_matrix")
        if rot is not None:
            # apply rotation of head
            rot_mat = glm.mat4(*rot)
            rot_quat = glm.quat(rot_mat)
            euler = glm.eulerAngles(rot_quat)

            # head_rot_quat = glm.quat(glm.vec3(euler.x, euler.y, euler.z))
            head_rot_inv = glm.inverse(
                glm.mat4_cast(glm.quat(self.node_head.matrix))
            )  # <- "un-rotate" head, then apply new rotation
            self.node_head.matrix = (
                self.node_head.matrix * head_rot_inv * glm.mat4_cast(rot_quat)
            )  # rot_mat

            neck_rot_quat = glm.quat(
                glm.vec3(euler.x / 5 + 0.3, euler.y / 5, euler.z / 5)
            )
            neck_rot_inv = glm.inverse(
                glm.mat4_cast(glm.quat(self.node_neck.matrix))
            )  # <- "un-rotate" neck, then apply new rotation
            self.node_neck.matrix = (
                self.node_neck.matrix * neck_rot_inv * glm.mat4_cast(neck_rot_quat)
            )

            spine_rot_quat = glm.quat(
                glm.vec3(euler.x / 10, euler.y / 10, euler.z / 10)
            )
            spine_rot_inv = glm.inverse(
                glm.mat4_cast(glm.quat(self.node_spine2.matrix))
            )  # <- "un-rotate" spine, then apply new rotation
            self.node_spine2.matrix = (
                self.node_spine2.matrix * spine_rot_inv * glm.mat4_cast(spine_rot_quat)
            )

            did_modify = True

        if did_modify:
            # update global transformation matrices for nodes/bones
            self.scene.root_nodes[0].calc_model_mat(self.global_matrix)
            return True

        return True

    def on_render(self, time: float, frame_time: float):
        """Render the scene (default render function)"""

        if self.has_morph_data:
            if self.update_morph_from_data(self.model_mesh, time):
                self.step += 1

        self.ctx.enable_only(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

        self.scene.draw(
            projection_matrix=self.camera.projection.matrix,
            camera_matrix=self.camera.matrix,
            time=time,
        )

        # # Draw bounding boxes
        # self.scene.draw_bbox(
        #     projection_matrix=self.camera.projection.matrix,
        #     camera_matrix=self.camera.matrix,
        #     children=True,
        #     color=(0.75, 0.75, 0.75),
        # )

        # self.scene.draw_wireframe(
        #     projection_matrix=self.camera.projection.matrix,
        #     camera_matrix=self.camera.matrix,
        #     color=(1, 1, 1, 1),
        # )

        if self.render_stats:
            self.writer.text = f"Camera Position: {str(self.camera.position)}"
            self.writer.draw((10, self.window_size[1] - 20), size=20)

            self.writer.text = "Keyboard: UP = Q, DOWN = E, LEFT = A, RIGHT = D, FORWARD = W, BACKWARD = S"
            self.writer.draw((10, self.window_size[1] - 40), size=20)

            self.writer.text = "FPS: {:.2f}".format(self.timer.fps_average)
            self.writer.draw((10, self.window_size[1] - 80), size=20)

            if self.has_morph_data:
                self.writer.text = "Morph no. {:04} / {:04}".format(
                    self.step % self.morph_data_size, self.morph_data_size
                )
                self.writer.draw((10, self.window_size[1] - 110), size=20)

        if (
            self.has_morph_data
            and self.stop_on_data_end
            and self.step > self.morph_data_size
        ):
            self.wnd.close()

    def on_close(self):
        self.clear_queues()
        if self.scene:
            self.scene.destroy()

            # FIXME scene.destroy() currently doesn't release the additional textures that
            #       were created for the morph targets & bone animation
            #       -> do this "manually" for now
            for m in self.scene.meshes:
                m.release()  # NOTE the Mesh.release() method is added in our modification, and does not exist on the original implementation in moderngl_window

    def _on_camera_key_event(self, key, action, modifiers):
        # NOTE only enable this, if the camera is a KeyboardCamera!
        self.camera.key_input(key, action, modifiers)
        # TODO add key interaction for changing camera's fov (and show fov in stats in default render method)

    def clear_queues(self):
        # ensure that queues are empty:
        if self.output_queue:
            while not self.output_queue.empty():
                self.output_queue.get()
            # self.output_queue.close()
        if self.input_queue:
            while not self.input_queue.empty():
                self.input_queue.get()
            # self.input_queue.close()

    def _render_to_images(self, time: float, frame_time: float):
        """Render the scene and store to images in self.out_dir (will overwrite existing images!)"""

        if self.wnd.is_closing:
            return

        self.update_morph_from_data(self.model_mesh, time, force=True)

        self.ctx.enable_only(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

        self.scene.draw(
            projection_matrix=self.camera.projection.matrix,
            camera_matrix=self.camera.matrix,
            time=time,
        )

        # wait for rendering to complete:
        self.ctx.finish()

        image = self.get_image_from_buffer()
        self.do_output_image(image)
        self.step += 1

        if self.stop_on_data_end and self.step > self.morph_data_size:
            self.wnd.close()

    def _render_to_queue(self, time: float, frame_time: float):
        """Render the scene and store put image to self.out_queue"""

        if self.wnd.is_closing:
            return

        self.update_morph_from_data(self.model_mesh, time, force=True)

        self.ctx.enable_only(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

        self.scene.draw(
            projection_matrix=self.camera.projection.matrix,
            camera_matrix=self.camera.matrix,
            time=time,
        )

        # wait for rendering to complete:
        self.ctx.finish()

        image = self.get_image_from_buffer()
        self.do_output_image(image)
        self.step += 1

        if self.stop_on_data_end and self.step > self.morph_data_size:
            self.wnd.close()

    def _render_queue_io(self, time: float, frame_time: float):
        """Render the scene and store put image to self.out_queue"""

        if self.wnd.is_closing:
            return

        shapes = self.input_queue.get()
        if shapes is None:
            logging.getLogger().info("received empty input: stop rendering now.")
            self.wnd.close()
            return

        data = shapes  # json.loads(shapes)
        if not self.apply_transformations(data, self.model_mesh):
            self.output_queue.put(None)  # FIXME should we ignore this!
            return

        self.ctx.enable_only(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

        self.scene.draw(
            projection_matrix=self.camera.projection.matrix,
            camera_matrix=self.camera.matrix,
            time=time,
        )

        # wait for rendering to complete:
        self.ctx.finish()

        image = self.get_image_from_buffer()
        self.do_output_image(image)
        self.step += 1

    def get_image_from_buffer(self) -> Image:
        raise NotImplementedError("set to self._get_jpg_image or self._get_png_image")

    def _get_jpg_image(self) -> Image:
        image = Image.frombytes(
            "RGB", self.wnd.fbo.size, self.wnd.fbo.read(components=3)
        )
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        return image

    def _get_png_image(self) -> Image:
        """WARNING: note that using PNG over JPEG almost doubles the rendering time!"""
        image = Image.frombytes(
            "RGBA", self.wnd.fbo.size, self.wnd.fbo.read(components=4)
        )
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        return image

    def do_output_image(self, image: Image) -> None:
        raise NotImplementedError(
            "set to self._save_[jpg | png]_image_to_file or self._put_[jpg | png]_image_in_queue"
        )

    def _save_jpg_image_to_file(self, image: Image) -> None:
        image.save(
            os.path.join(self.out_dir, f"scene_{self.step}.jpg"),
            format="JPEG",
            quality=95,
            subsampling=0,
        )

    def _put_jpg_image_in_queue(self, image: Image) -> None:
        temp = BytesIO()
        image.save(temp, format="JPEG", quality=95, subsampling=0)
        self.output_queue.put(temp.getvalue())

    def _save_png_image_to_file(self, image: Image) -> None:
        image.save(os.path.join(self.out_dir, f"scene_{self.step}.png"), format="png")

    def _put_png_image_in_queue(self, image: Image) -> None:
        temp = BytesIO()
        image.save(temp, format="png")
        self.output_queue.put(temp.getvalue())


if __name__ == "__main__":

    import moderngl_window

    headless = True
    the_blendshapes_data_path = "detectionLog.json"
    the_model_path = os.join(__file__, "../../../../../../rpm/public/avatar_1_f.glb")
    the_out_dir_path = os.path.realpath("output")

    if headless:
        GLTFRenderer.hidden_window_framerate_limit = -1
        GLTFRenderer.visible = False
        GLTFRenderer.is_headless = True
        args = ("--window", "headless")
    else:
        GLTFRenderer.hidden_window_framerate_limit = 30
        GLTFRenderer.visible = True
        GLTFRenderer.is_headless = False
        args = None

    config = moderngl_window.create_window_config_instance(GLTFRenderer, args=args)
    win: GLTFRenderer = config.wnd

    win.load_model(the_model_path)
    win.load_recorded_blend_shapes(the_blendshapes_data_path)

    win.enable_render_and_store_images(the_out_dir_path)

    start_time = time.time()
    moderngl_window.run_window_config_instance(config)
    print("duration: ", time.time() - start_time)
