# ##############################################################
# [russa] MODIFIED source: implementation based on class TextureLightProgram in
# https://github.com/mmig/moderngl-window/blob/16746555a299e3df9ec00dfa597be33b59143050/moderngl_window/scene/programs.py
#
# MODIFICATIONS:
#  * support for morph targets (morph-based animation)
#  * support for skeleton / bone-based animation
# ##############################################################


from pathlib import Path
from typing import Optional, Any

import glm
import moderngl
import numpy
from moderngl_window.meta import ProgramDescription
from moderngl_window.resources import programs
from moderngl_window.scene import MeshProgram

from .mesh import Mesh


class TextureLightSkeletonMorphProgram(MeshProgram):
    """
    Simple texture program for "skinned" mesh with Skeleton & Morph Targets

    based on moderngl_window's TextureLightProgram with adapted handling for skeleton & morph targets from three.js
    """

    def __init__(
        self, program: Optional[moderngl.Program] = None, **kwargs: Any
    ) -> None:
        super().__init__(program=None)
        self.program: Optional[moderngl.Program] = program

    def _load_program(self, mesh: Mesh):
        prog_path = Path(__file__, "../texture_light_skeleton_morph.glsl").resolve()
        # TODO find a good way a #define and #undefine USE_SKINNING & USE_MORPHTARGETS
        #      depending on whether a skeleton & morph targets are defined in the mesh
        #      ... for now these are hardcoded in the shader, i.e. shader only works,
        #          if mesh has skeleton & morph targets
        morph_targets_count = (
            len(mesh.morph_target_dictionary)
            if mesh.morph_target_dictionary is not None
            else 1
        )  # FIXME set to 0 if None, but for this USE_MORPHTARGETS needs to get #undefined!
        self.program = programs.load(
            ProgramDescription(
                path=str(prog_path),
                defines={
                    # 'USE_SKINNING': True,
                    # 'USE_MORPHTARGETS': True,
                    "MORPHTARGETS_COUNT": morph_targets_count
                },
            )
        )

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
        assert mesh.morph_texture is not None, "The are no morph targets"
        assert mesh.skeleton is not None, "The is no skeleton"

        # if mesh.material.double_sided:
        #     self.ctx.disable(moderngl.CULL_FACE)
        # else:
        #     self.ctx.enable(moderngl.CULL_FACE)

        self.update_morph_texture(mesh)
        self.update_skeleton(mesh, self.program.ctx)

        self.program["texture0"].value = 0
        mesh.material.mat_texture.texture.use(location=0)
        self.program["m_proj"].write(projection_matrix)
        self.program["m_model"].write(model_matrix)
        self.program["m_cam"].write(camera_matrix)
        mesh.vao.render(self.program)

    def update_skeleton(self, mesh: Mesh, ctx: moderngl.Context):
        if not mesh.skeleton:
            return

        skeleton = mesh.skeleton

        # TODO only update if necessary
        skeleton.update()

        if not skeleton.boneTexture:
            skeleton.create_bone_texture(ctx)
        else:
            skeleton.boneTexture.write(
                skeleton.boneMatrices
            )  # TODO only update if necessary

        self.program["boneTexture"].value = 2
        skeleton.boneTexture.use(location=2)

    def update_morph_texture(self, mesh: Mesh):

        if len(mesh.morph_target_influences) < 1:
            mesh.morph_target_influences = numpy.zeros(
                len(mesh.morph_target_dictionary)
            )

        object_influences = mesh.morph_target_influences

        # TODO support non-relative morph target manipulation?
        # morph_influences_sum = 0
        # if not mesh.morph_targets_relative:
        # #     for infl in object_influences:
        # #         morph_influences_sum += infl
        #     morph_influences_sum = numpy.sum(object_influences)
        morph_base_influence = (
            1  # mesh.morph_targets_relative ? 1: 1 - morph_influences_sum;
        )

        self.program["morphTargetBaseInfluence"].value = morph_base_influence
        self.program["morphTargetInfluences"].value = object_influences.tolist()
        # self.program["morphTargetsTexture"].value = entry.texture, textures);
        self.program["morphTargetsTextureSize"].write(
            mesh.morph_texture_size
        )  # entry.size

        self.program["morphTargetsTexture"].value = 1
        mesh.morph_texture.use(location=1)

    def apply(self, mesh: Mesh) -> Optional["MeshProgram"]:
        if not mesh.material:
            return None

        if not mesh.attributes.get("NORMAL"):
            return None

        if not mesh.attributes.get("TEXCOORD_0"):
            return None

        if not mesh.morph_texture:
            return None

        if (
            not mesh.morph_target_dictionary
        ):  # FIXME alternatively, mesh would need a field with count of morph targets
            return None

        if not mesh.skeleton:
            return None

        if mesh.material.mat_texture is not None:
            if not self.program:
                self._load_program(mesh)
                return self
            else:
                # program depends on morph target count:
                #  if one is already loaded, do create a new one for the (different) mesh:
                # TODO create cache & cache-key depending on mesh features (-> skeleton & morph-targets/-count)
                new_prog = TextureLightSkeletonMorphProgram()
                new_prog._load_program(mesh)
                return new_prog

        return None
