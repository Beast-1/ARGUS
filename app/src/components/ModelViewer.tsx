import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";
import { apiUrl } from "../api/client";
import "./ModelViewer.css";

interface ModelViewerProps {
  /** API-relative GLB url, e.g. /api/projects/foo/files/foo.glb */
  src: string | null;
  wireframe: boolean;
}

/** three.js directly rather than <model-viewer>: model-viewer has no wireframe mode,
 *  and doing it here gives one component that serves both the shaded and wireframe
 *  tabs off the same loaded scene. */
export function ModelViewer({ src, wireframe }: ModelViewerProps) {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const modelRef = useRef<THREE.Object3D | null>(null);
  const [status, setStatus] = useState<"idle" | "loading" | "ready" | "error">("idle");

  // Scene is built once; only the model swaps when `src` changes.
  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const scene = new THREE.Scene();
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(mount.clientWidth, mount.clientHeight);
    mount.appendChild(renderer.domElement);

    const camera = new THREE.PerspectiveCamera(
      40,
      mount.clientWidth / mount.clientHeight,
      0.01,
      100,
    );
    camera.position.set(1.6, 1.2, 1.9);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;

    // Neutral studio IBL so PBR materials read correctly without shipping an HDRI.
    const pmrem = new THREE.PMREMGenerator(renderer);
    scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;

    const key = new THREE.DirectionalLight(0xffffff, 2.0);
    key.position.set(3, 5, 2);
    scene.add(key);
    scene.add(new THREE.AmbientLight(0xffffff, 0.35));

    const grid = new THREE.GridHelper(4, 16, 0x2b2b2f, 0x1c1c1f);
    (grid.material as THREE.Material).transparent = true;
    (grid.material as THREE.Material).opacity = 0.5;
    scene.add(grid);

    let raf = 0;
    const tick = () => {
      controls.update();
      renderer.render(scene, camera);
      raf = requestAnimationFrame(tick);
    };
    tick();

    const onResize = () => {
      if (!mount.clientWidth || !mount.clientHeight) return;
      camera.aspect = mount.clientWidth / mount.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(mount.clientWidth, mount.clientHeight);
    };
    const observer = new ResizeObserver(onResize);
    observer.observe(mount);

    // Expose for the loader effect below.
    (mount as any).__three = { scene, camera, controls };

    return () => {
      cancelAnimationFrame(raf);
      observer.disconnect();
      controls.dispose();
      pmrem.dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode === mount) {
        mount.removeChild(renderer.domElement);
      }
    };
  }, []);

  // Load / swap the model.
  useEffect(() => {
    const mount = mountRef.current;
    const ctx = mount && (mount as any).__three;
    if (!mount || !ctx) return;
    const { scene, camera, controls } = ctx as {
      scene: THREE.Scene;
      camera: THREE.PerspectiveCamera;
      controls: OrbitControls;
    };

    if (modelRef.current) {
      scene.remove(modelRef.current);
      modelRef.current = null;
    }
    if (!src) {
      setStatus("idle");
      return;
    }

    setStatus("loading");
    let cancelled = false;
    new GLTFLoader().load(
      apiUrl(src),
      (gltf) => {
        if (cancelled) return;
        const model = gltf.scene;

        // Frame the model: recentre on origin and pull the camera back to fit.
        const box = new THREE.Box3().setFromObject(model);
        const size = box.getSize(new THREE.Vector3());
        const centre = box.getCenter(new THREE.Vector3());
        model.position.sub(new THREE.Vector3(centre.x, box.min.y, centre.z));

        const extent = Math.max(size.x, size.y, size.z) || 1;
        const dist = extent * 2.4;
        camera.position.set(dist * 0.62, extent * 0.9 + dist * 0.28, dist * 0.72);
        controls.target.set(0, size.y * 0.45, 0);
        controls.update();

        scene.add(model);
        modelRef.current = model;
        setStatus("ready");
      },
      undefined,
      () => !cancelled && setStatus("error"),
    );

    return () => {
      cancelled = true;
    };
  }, [src]);

  // Wireframe toggle walks the loaded materials in place — no reload.
  useEffect(() => {
    const model = modelRef.current;
    if (!model) return;
    model.traverse((obj) => {
      const mesh = obj as THREE.Mesh;
      if (!mesh.isMesh) return;
      const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
      mats.forEach((m) => {
        const mat = m as THREE.MeshStandardMaterial;
        if ("wireframe" in mat) mat.wireframe = wireframe;
      });
    });
  }, [wireframe, status]);

  return (
    <div className="model-viewer">
      <div className="model-viewer-canvas" ref={mountRef} />
      {status !== "ready" && (
        <div className="model-viewer-status">
          {status === "loading"
            ? "Loading model…"
            : status === "error"
              ? "Couldn't load the model"
              : "No model yet"}
        </div>
      )}
      {status === "ready" && (
        <div className="model-viewer-hint">drag — orbit · scroll — zoom · right-drag — pan</div>
      )}
    </div>
  );
}
