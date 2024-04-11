import "./App.css";
//import three types

import { Euler } from "three";

function Avatar({
  nodes,
  scene,
  blendshapes,
  rotation,
  headMesh,

  onRenderFinished,
}: {
  nodes: any;
  scene: any;
  blendshapes: any[];
  rotation: Euler;
  headMesh: any[];
  onRenderFinished: () => void;
}) {
  // apply blendshapes to the head mesh
  if (blendshapes.length > 0) {
    blendshapes.forEach((element) => {
      headMesh.forEach((mesh) => {
        let index = mesh.morphTargetDictionary[element.categoryName];
        if (index >= 0) {
          mesh.morphTargetInfluences[index] = element.score;
        }
      });
    });
  }

  // apply rotation to the head and neck
  nodes.Head.rotation.set(rotation.x, rotation.y, rotation.z);
  nodes.Neck.rotation.set(rotation.x / 5 + 0.3, rotation.y / 5, rotation.z / 5);
  nodes.Spine2.rotation.set(rotation.x / 10, rotation.y / 10, rotation.z / 10);

  // call onRenderFinished when the avatar is rendered
  nodes.Wolf3D_Avatar.onAfterRender = () => {
    onRenderFinished();
  };
  return <primitive object={scene} position={[0, -1.75, 3]} />;
}
export default Avatar;
