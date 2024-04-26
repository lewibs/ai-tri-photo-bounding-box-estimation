import React, { useEffect } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import {v4 as uuid} from "uuid";
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader';

const SAVE = true;

async function screenshot(renderer) {
  return new Promise((resolve, reject) => {
    // Create an "invisible" link element
    const canvas = renderer.domElement;
    const link = document.createElement('a');
    link.style.display = 'none';

    requestAnimationFrame(() => {
      // Convert the canvas content to a data URL
      const dataURL = canvas.toDataURL('image/jpeg');
    
      // Create a link element
      const link = document.createElement('a');
      link.href = dataURL;
      const url = `${uuid()}.jpg`;
      link.download = url;
      
      document.body.appendChild(link);
      SAVE && link.click();
      
      document.body.removeChild(link);

      resolve(url);
    });
  });

  // return new Promise((resolve, reject) => {
  //   // Use requestAnimationFrame to wait for the next frame
  //   requestAnimationFrame(() => {
  //     const canvas = renderer.domElement;
  //     const context = renderer.getContext();
  //     const left = 0;
  //     const top = 0;
  //     const width = canvas.width;
  //     const height = canvas.height;
  //     const pixelData = new Uint8Array(width * height * 4);

  //     // Read pixel data from the renderer's context
  //     context.readPixels(left, top, width, height, context.RGBA, context.UNSIGNED_BYTE, pixelData);

  //     // Resolve the Promise with pixelData
  //     resolve(pixelData);
  //   });
  // });
}


async function getSceneData(camera, renderer, controls, object) {
    const boundingBox = new THREE.Box3();
    boundingBox.setFromObject(object);
    const maxX = boundingBox.max.x;
    const maxY = boundingBox.max.y;
    const maxZ = boundingBox.max.z;

    const hypotinuse = Math.sqrt(Math.pow(maxX, 2)+Math.pow(maxY, 2)+Math.pow(maxZ, 2));

    const photos = []

    const fov = camera.fov;
    const aspect = camera.aspect;
    const near = camera.near;
    const far = camera.far;
    const bounding_box = {
      max: {
        x:boundingBox.max.x,
        y:boundingBox.max.y,
        z:boundingBox.max.z
      },
      min: {
        x:boundingBox.min.x,
        y:boundingBox.min.y,
        z:boundingBox.min.z
      }
    }


    for (let i = 0; i < 3; i++) {
      let rotation = [0, 0, 0]; // Initialize rotation
      let camera_pos = [0, 0, 0];
      
      // Set rotation based on the axis
      switch (i) {
        case 0: // x-axis
          rotation[0] = Math.PI / 2; // Rotate around x-axis by 90 degrees
          camera_pos[0] = hypotinuse * 2
          break;
        case 1: // y-axis
          rotation[1] = - Math.PI / 2; // Rotate around y-axis by 90 degrees
          camera_pos[1] = hypotinuse * 2
          break;
        case 2: // z-axis
          rotation[2] =  Math.PI / 2; // Rotate around z-axis by 90 degrees
          camera_pos[2] = hypotinuse * 2
          break;
      }

      camera.position.set(camera_pos[0], camera_pos[1], camera_pos[2])
      controls.target.fromArray(rotation);
      controls.update();
      
      // Extract camera properties
      const position = camera.position.toArray();
      rotation = camera.rotation.toArray();
      const up = camera.up.toArray();
      const image = await screenshot(renderer)
  
      const photoData = {
        position: position,
        rotation: rotation,
        //up: up, //TODO add this only if we start using real world cameras since three cameras will always be up.
        image: image,
      };

      photos.push(photoData)
    }

    const id = uuid()

    const dat = {
      id,
      fov,
      aspect,
      near,
      far,
      bounding_box,
      photos,
    }

    // Convert data object to JSON string
    const jsonData = JSON.stringify(dat);

    // Create a Blob from the JSON string
    const blob = new Blob([jsonData], { type: 'application/json' });

    // Create a URL for the Blob
    const url = URL.createObjectURL(blob);

    // Create a link element
    const link = document.createElement('a');
    link.href = url;
    link.download = `${id}.json`; // Set the desired file name

    // Simulate a click on the link to trigger the download
    document.body.appendChild(link); // Append the link to the document body
    SAVE && link.click(); // Simulate a click

    // Clean up
    document.body.removeChild(link);
    URL.revokeObjectURL(url); // Release the object URL

    return dat;
}

async function getGltf(data, index) {
  // Load GLB file
  const loader = new GLTFLoader();
  return await new Promise((resolve, reject)=>{
    loader.load(`assets/furniture/${data.furniture_names[index]}`, (gltf)=>{resolve(gltf.scene)})
  })
}

async function getScene() {
  // Load GLB file
  const loader = new GLTFLoader();
  return await new Promise((resolve, reject)=>{
    loader.load(`assets/scenes/0.glb`, (gltf)=>{resolve(gltf.scene)})
  })
}

async function getBackground(data, index) {
  const textureLoader = new THREE.TextureLoader();
  const geometry = new THREE.SphereGeometry( 500, 60, 40 );
  geometry.scale( - 1, 1, 1 );
  const url = `assets/backgrounds/${data.photo_names[index]}`
  const texture = textureLoader.load(url);
  texture.colorSpace = THREE.SRGBColorSpace;
  const material = new THREE.MeshBasicMaterial( { map: texture } );
  return new THREE.Mesh(geometry, material)
}

async function generateDataSet(camera, renderer, controls, scene) {
  function dispose(object) {
    object.material?.dispose();
    object.geometry?.dispose();
    object.children.forEach(dispose)
  }

  const response = await fetch("assets/meta.json")
  const data = await response.json();
  const dataset = []
  const raycaster = new THREE.Raycaster(new THREE.Vector3(0,0,0), new THREE.Vector3(0,-1,0));
  const intersects = raycaster.intersectObject(scene)

  //data.furniture_count
  // for(let i = 0; i < 10; i++) {
    const gltf = await getGltf(data, 50);
    let bbox = new THREE.Box3().setFromObject(gltf);
    let size = new THREE.Vector3();
    bbox.getSize(size);
    gltf.position.y -= intersects.distance - (size.y / 2);
    gltf.scale.set(40,40,40)
    scene.add(gltf)

    console.log(gltf)

    // const dat = await getSceneData(camera, renderer, controls, gltf);
    // dataset.push(dat);
    console.log("new item loaded")

    // scene.remove(gltf)
    // dispose(gltf)
  // }

  console.log("done");
}

function App() {
  useEffect(() => {
    const scene = new THREE.Scene();

    const light = new THREE.AmbientLight( 0xffffff, 1 );
    scene.add( light );

    const directionalLight = new THREE.DirectionalLight(0xffffff, 5);
    directionalLight.position.set(50, 50, 0); // Set the direction of the light
    scene.add(directionalLight);

    scene.background = new THREE.Color('skyblue');

    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.x = 115;
    camera.position.z = 20;
    
    const renderer = new THREE.WebGLRenderer();
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.update();

    getScene().then((room)=>{
      room.position.y -= 50
      room.position.x += 130

      scene.add(room)
      generateDataSet(camera, renderer, controls, scene)
    })

    // createBackground().then(res=>scene.add(res))
    // window.getData = ()=>getTriPhoto(camera, renderer, controls)

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
