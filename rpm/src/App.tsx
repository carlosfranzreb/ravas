import { useGLTF } from "@react-three/drei";
import { Canvas, useGraph } from "@react-three/fiber";
import { useEffect, useRef, useState } from "react";
import { useDropzone } from "react-dropzone";
import useWebSocket, { ReadyState } from "react-use-websocket";
import { Color, Euler, Matrix4 } from "three";
import "./App.css";
import Avatar from "./Avatar";

function App() {
  const [url, setUrl] = useState<string>("./default_avatar.glb");
  const { scene } = useGLTF(url);
  const { nodes } = useGraph(scene);

  // get head mesh
  const headMesh = [];
  if (nodes.Wolf3D_Head) headMesh.push(nodes.Wolf3D_Head);
  if (nodes.Wolf3D_Teeth) headMesh.push(nodes.Wolf3D_Teeth);
  if (nodes.Wolf3D_Beard) headMesh.push(nodes.Wolf3D_Beard);
  if (nodes.Wolf3D_Avatar) headMesh.push(nodes.Wolf3D_Avatar);
  if (nodes.Wolf3D_Head_Custom) headMesh.push(nodes.Wolf3D_Head_Custom);
  const [rotation, setRotation] = useState<Euler>(new Euler());
  const [blendshapes, setBlendshapes] = useState<any[]>([]);
  const updateRef = useRef(false);

  // create websocket connection
  const ws_url = process.env.REACT_APP_WS_URL || "ws://localhost:8888";
  const { sendMessage, lastMessage, readyState } = useWebSocket(ws_url, {
    shouldReconnect: (closeEvent) => true,
    reconnectAttempts: 6000, // try to reconnect every second for 100 minutes
    reconnectInterval: 1000,
  });
  const connectionStatus = {
    [ReadyState.CONNECTING]: "Connecting",
    [ReadyState.OPEN]: "Connected",
    [ReadyState.CLOSING]: "Closing",
    [ReadyState.CLOSED]: "Closed",
    [ReadyState.UNINSTANTIATED]: "Uninstantiated",
  }[readyState];

  function get_canvas_url() {
    // get canvas as image
    const canvas = document.getElementsByTagName("canvas")[0];
    const img_data = canvas.toDataURL("image/jpeg");
    return img_data.split(",")[1];
  }
  useEffect(() => {
    if (lastMessage !== null) {
      const strmessage = lastMessage.data;
      const message = JSON.parse(strmessage);

      const blendshapes_temp = message.blendshapes;
      const matrix = new Matrix4().fromArray(message.transformation_matrix);
      const rotation_temp = new Euler().setFromRotationMatrix(matrix);
      setBlendshapes(blendshapes_temp);
      setRotation(rotation_temp);
      // set updateRef to true to send image to server after the avatar is rendered
      updateRef.current = true;
    }
  }, [lastMessage]);

  // handle paste url or file to load avatar
  const { getRootProps } = useDropzone({
    onDrop: (files) => {
      const file = files[0];
      const reader = new FileReader();
      reader.onload = () => {
        setUrl(reader.result as string);
      };
      reader.readAsDataURL(file);
    },
  });

  function onRenderFinished() {
    if (updateRef.current) {
      // send image generated from canvas to server
      const b64_img = get_canvas_url();
      sendMessage(b64_img);
      updateRef.current = false;
    }
  }

  return (
    <div className="App">
      <div {...getRootProps({ className: "dropzone" })}>
        <p>Drag & drop RPM avatar GLB file here</p>
      </div>
      <Canvas
        style={{ height: 600 }}
        camera={{ fov: 25 }}
        shadows
        gl={{ preserveDrawingBuffer: true }}
      >
        <ambientLight intensity={0.5} />
        <pointLight
          position={[10, 10, 10]}
          color={new Color(1, 1, 0)}
          intensity={0.5}
          castShadow
        />
        <pointLight
          position={[-10, 0, 10]}
          color={new Color(1, 0, 0)}
          intensity={0.5}
          castShadow
        />
        <pointLight position={[0, 0, 10]} intensity={0.5} castShadow />

        <Avatar
          scene={scene}
          nodes={nodes}
          blendshapes={blendshapes}
          rotation={rotation}
          headMesh={headMesh}
          onRenderFinished={onRenderFinished}
        />
      </Canvas>
      <span>State: {connectionStatus}</span>
      <span></span>
    </div>
  );
}

export default App;
