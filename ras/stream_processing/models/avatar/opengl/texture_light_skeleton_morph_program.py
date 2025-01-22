from __future__ import annotations

import os
from typing import Any, Optional

import glm
import moderngl

import moderngl_window
from moderngl_window.conf import settings
from moderngl_window.meta import ProgramDescription
from moderngl_window.resources.programs import programs

from .mesh import Mesh

settings.PROGRAM_DIRS.append(os.path.join(os.path.dirname(__file__), "programs"))


class TextureLightProgram(MeshProgram):
    """
    Simple texture program
    """

    def __init__(self, program: Optional[moderngl.Program] = None, **kwargs: Any) -> None:
        super().__init__(program=None)
        self.program = programs.load(ProgramDescription(path="scene_default/texture_light.glsl"))

    def draw(
        self,
        mesh: Mesh,
        projection_matrix: glm.mat4,
        model_matrix: glm.mat4,
        camera_matrix: glm.mat4,
        time: float = 0.0,
    ) -> None:
        assert self.program is not None, "There is no program to draw"
        assert mesh.vao is not None, "There is no vao to render"
        assert mesh.material is not None, "There is no material to render"
        assert (
            mesh.material.mat_texture is not None
        ), "The material does not have a texture to render"
        assert (
            mesh.material.mat_texture.texture is not None
        ), "The material texture is not linked to a texture, so it can not be rendered"

        # if mesh.material.double_sided:
        #     self.ctx.disable(moderngl.CULL_FACE)
        # else:
        #     self.ctx.enable(moderngl.CULL_FACE)

        mesh.material.mat_texture.texture.use()
        self.program["texture0"].value = 0
        self.program["m_proj"].write(projection_matrix)
        self.program["m_model"].write(model_matrix)
        self.program["m_cam"].write(camera_matrix)
        mesh.vao.render(self.program)

    def apply(self, mesh: Mesh) -> MeshProgram | None:
        if not mesh.material:
            return None

        if not mesh.attributes.get("NORMAL"):
            return None

        if not mesh.attributes.get("TEXCOORD_0"):
            return None

        if mesh.material.mat_texture is not None:
            return self

        return None
