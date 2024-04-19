import React, { useEffect } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CameraHelper } from './CameraHelper';

async function createBackground() {
  const response = await fetch("assets/background_meta.json")
  const data = await response.json();

  const textureLoader = new THREE.TextureLoader();
  const geometry = new THREE.SphereGeometry( 500, 60, 40 );
  geometry.scale( - 1, 1, 1 );
  
  const url = `./assets/backgrounds/${Math.floor(Math.random() * data.count - 1)}.jpg`
  const texture = textureLoader.load(url);
  texture.colorSpace = THREE.SRGBColorSpace;
  const material = new THREE.MeshBasicMaterial( { map: texture } );
  return new THREE.Mesh(geometry, material)
}

async function screenshot(renderer) {
  return new Promise((resolve, reject) => {
    // Use requestAnimationFrame to wait for the next frame
    requestAnimationFrame(() => {
      const canvas = renderer.domElement;
      const context = renderer.getContext();
      const left = 0;
      const top = 0;
      const width = canvas.width;
      const height = canvas.height;
      const pixelData = new Uint8Array(width * height * 4);

      // Read pixel data from the renderer's context
      context.readPixels(left, top, width, height, context.RGBA, context.UNSIGNED_BYTE, pixelData);

      // Resolve the Promise with pixelData
      resolve(pixelData);
    });
  });
}


async function getTriPhoto(camera, renderer, controls, object) {
    const photos = []
    for (let i = 0; i < 3; i++) {
      let rotation = [0, 0, 0]; // Initialize rotation

      // Set rotation based on the axis
      switch (i) {
        case 0: // x-axis
          rotation[0] = Math.PI / 2; // Rotate around x-axis by 90 degrees
          break;
        case 1: // y-axis
          rotation[1] = - Math.PI / 2; // Rotate around y-axis by 90 degrees
          break;
        case 2: // z-axis
          rotation[2] =  Math.PI / 2; // Rotate around z-axis by 90 degrees
          break;
      }

      controls.target.fromArray(rotation);
      controls.update();
      
      // Extract camera properties
      const position = camera.position.toArray();
      rotation = camera.rotation.toArray();
      const fov = camera.fov;
      const aspect = camera.aspect;
      const near = camera.near;
      const far = camera.far;
      const image = await screenshot(renderer)
  
      const cameraData = {
        position: position,
        rotation: rotation,
        fov: fov,
        aspect: aspect,
        near: near,
        far: far,
        image: image,
      };

      photos.push(cameraData)
    }

    console.log(photos)
}

function App() {
  useEffect(() => {
    const scene = new THREE.Scene();

    const light = new THREE.AmbientLight( 0xffffff, 3 );
    scene.add( light );

    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.z = 1;
    
    const renderer = new THREE.WebGLRenderer();
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.update();

    createBackground().then(res=>scene.add(res))

    window.getData = ()=>getTriPhoto(camera, renderer, controls)

    // Render the scene
    function animate() {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }
    animate();

    // Clean up
    return () => {
      // Remove the renderer when the component unmounts
      document.body.removeChild(renderer.domElement);
    };
  }, []);

  return null;
}

export default App;
