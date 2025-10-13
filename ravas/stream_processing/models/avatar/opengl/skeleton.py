from typing import TYPE_CHECKING, List, Optional, Tuple

import glm
from moderngl import Texture, Context

import numpy
import math


if TYPE_CHECKING:
    from .node import Node


class Skeleton:

    def __init__(self, bones: List["Node"] = None, boneInverses=None):
        sorted_bones, indices = (
            self.to_topological_sorted(bones) if bones is not None else ([], [])
        )
        self.boneInversesIndices: List[int] = indices
        self.bones: List["Node"] = sorted_bones
        # self.boneInverses = self.apply_reordering(self.prepare_bone_inverses(boneInverses), indices) if boneInverses is not None else []

        # self.bones = bones if bones is not None else []
        self.boneInverses = (
            self.prepare_bone_inverses(boneInverses) if boneInverses is not None else []
        )

        self.boneMatrices = numpy.zeros(len(self.bones) * 16, dtype=numpy.float32)
        self.boneTexture: Optional[Texture] = None
        self.boneTextureSize: int = -1

    def create_bone_texture(self, ctx: Context):

        size = math.sqrt(len(self.bones) * 4)  # 4 pixels needed for 1 matrix
        size = math.ceil(size / 4) * 4
        size = max(size, 4)

        bone_matrices = numpy.zeros(size * size * 4, dtype=numpy.float32)
        bone_matrices_view = self.boneMatrices.view(numpy.float32).reshape(
            (len(self.boneMatrices),)
        )  # FIXME remove?
        bone_matrices[0 : len(bone_matrices_view)] = bone_matrices_view

        self.boneTexture = ctx.texture((size, size), 4, dtype="f4", data=bone_matrices)
        self.boneMatrices = bone_matrices
        self.boneTextureSize = size

    def update(self):
        for i, bone in enumerate(self.bones):
            idx = self.boneInversesIndices[i]
            matrix = bone.matrix_global if bone else glm.mat4()
            # offset_matrix = Matrix4()
            # offset_matrix.multiplyMatrices(matrix, self.boneInverses[i])
            offset_matrix = (
                matrix * self.boneInverses[idx]
            )  # glm.mat4(*self.boneInverses[i])  # FIXME pre-cal mat4 !!!
            # offset_matrix.toArray(self.boneMatrices, i * 16)
            offset = idx * 16
            # self.boneMatrices[offset:offset+16] = offset_matrix.to_tuple()
            self.boneMatrices[offset : offset + 16] = numpy.array(
                offset_matrix.to_list()
            ).reshape((16,))

        # if self.boneTexture:
        #     self.boneTexture.needsUpdate = True  # TODO?

    @staticmethod
    def to_topological_sorted(bones: List["Node"]) -> Tuple[List["Node"], List[int]]:
        """
        HELPER create topological sorted list of bones
               (for efficient traversing & updating node transforms for rendering)

        see
        https://en.wikipedia.org/wiki/Topological_sorting#Depth-first_search
        """
        lst = []
        indices = []

        # find root node (i.e. node NOT referenced as child)
        nodes = set(bones)
        for i, b in enumerate(bones):
            if b.children:
                for c in b.children:
                    nodes.remove(c)
        root = nodes.pop()

        def proc(n: "Node"):
            # 1) visit node children:
            for c in n.children:
                proc(c)
            # 2) prepend to result list:
            lst.insert(0, n)
            # 3) keep track of indices w.r.t. to original list
            idx = next(ii for ii, bb in enumerate(bones) if bb == n)
            indices.insert(0, idx)

        proc(root)
        return lst, indices

    def release(self):
        if self.boneTexture:
            self.boneTexture.release()

    @staticmethod
    def apply_reordering(
        bone_inverses: List[glm.mat4], indices: List[int]
    ) -> List[glm.mat4]:
        lst: List[glm.mat4] = []
        for i in range(len(bone_inverses)):
            lst.append(bone_inverses[indices[i]])
        return lst

    @staticmethod
    def prepare_bone_inverses(bone_inverses: numpy.ndarray):
        """
        :param bone_inverses: the inverse-bind matrices for the bones as `numpy` array with shape
                              `(<num bones>, 4, 4)`
        """
        lst = []
        for mat in bone_inverses:
            lst.append(glm.mat4(*mat))
        return lst
