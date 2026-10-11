const puppeteer = require('puppeteer-core');
const path = require('path');

(async () => {
  const browser = await puppeteer.launch({
    headless: true,
    executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  });

  const page = await browser.newPage();
  page.on('console', msg => console.log('PAGE LOG:', msg.text()));
  page.on('pageerror', err => console.error('PAGE ERROR:', err.message));

  await page.goto('http://localhost:8765/visualizer.html', { waitUntil: 'networkidle2' });
  console.log('Страницата е заредена');

  // Изчакваме Three.js canvas да се инициализира
  await page.waitForSelector('#v-canvas');
  await new Promise(r => setTimeout(r, 2000));

  // Проверяваме дали WebGL canvas е инициализиран
  const canvasInfo = await page.evaluate(() => {
    const canvas = document.getElementById('v-canvas');
    return {
      canvasExists: !!canvas,
      canvasWidth: canvas ? canvas.width : 0,
      canvasHeight: canvas ? canvas.height : 0,
    };
  });

  console.log('Canvas тест:', canvasInfo);
  await new Promise(r => setTimeout(r, 1000));

  // Симулираме качване на снимка
  const testImageBase64 = 'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAn/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdAB//2Q==';
  await page.evaluate((base64) => {
    return new Promise((resolve) => {
      const input = document.getElementById('roomPhoto');
      fetch(base64)
        .then(r => r.blob())
        .then(blob => {
          const file = new File([blob], 'test-room.jpg', { type: 'image/jpeg' });
          const dt = new DataTransfer();
          dt.items.add(file);
          input.files = dt.files;
          input.dispatchEvent(new Event('change', { bubbles: true }));
          resolve();
        });
    });
  }, testImageBase64);

  console.log('Снимката е качена');
  await new Promise(r => setTimeout(r, 1000));

  // Натискаме бутона за анализ през JS (по-надеждно от click() в headless)
  await page.evaluate(() => document.getElementById('analyzeBtn').click());
  console.log('Анализът стартира');

  // Изчакваме резултатите
  await page.waitForFunction(() => {
    const el = document.getElementById('resWidth');
    return el && el.textContent !== '—';
  }, { timeout: 15000 });

  const results = await page.evaluate(() => ({
    width: document.getElementById('resWidth').textContent,
    length: document.getElementById('resLength').textContent,
    height: document.getElementById('resHeight').textContent,
  }));

  console.log('Резултати от AI:', results);

  await browser.close();
})();
