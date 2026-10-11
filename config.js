// Конфигурация на визуализатора
// За тестова/продукционна среда промени AI_API_URL
window.DC_CONFIG = {
  // AI сървър за измерване
  // За локален тест: 'http://localhost:8000'
  // Продукционен Render сървър:
  AI_API_URL: 'https://kimi-oppolia.onrender.com',

  // Демо GLB модели за тест
  DEMO_MODELS: {
    parrot: 'https://threejs.org/examples/models/gltf/Parrot.glb',
    flamingo: 'https://threejs.org/examples/models/gltf/Flamingo.glb',
    tokyo: 'https://threejs.org/examples/models/gltf/LittlestTokyo.glb',
  },
};
