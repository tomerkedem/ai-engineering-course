const pptxgen = require('pptxgenjs');

let pres = new pptxgen();
pres.layout = 'LAYOUT_16x9';
pres.author = 'World Cup';
pres.title = 'World Cup Final';

let slide = pres.addSlide();

// ===== BACKGROUND: DARK STADIUM GRADIENT =====
// Create a deep dark background (stadium at night)
slide.background = { color: '0F1419' };

// ===== ADD RADIAL GLOW EFFECT =====
// Create concentric circles to simulate stadium lights/glow
const glowCircles = [
  { r: 3.2, color: '1a472a', transparency: 85 },  // outer green glow
  { r: 2.8, color: '2d5f3f', transparency: 75 },  // green glow
  { r: 2.4, color: '3d7c52', transparency: 60 },  // brighter green
  { r: 2.0, color: '4da662', transparency: 40 },  // stadium field green
];

const centerX = 5;  // center of 10" width
const centerY = 2.8125;  // center of 5.625" height

glowCircles.forEach((glow) => {
  const offsetX = centerX - (glow.r / 2);
  const offsetY = centerY - (glow.r / 2);
  
  slide.addShape(pres.shapes.OVAL, {
    x: offsetX, y: offsetY, w: glow.r, h: glow.r,
    fill: { color: glow.color, transparency: glow.transparency },
    line: { type: 'none' }
  });
});

// ===== ADD STADIUM LIGHTS (GOLDEN GLOW) =====
const stadiumLights = [
  { x: 1.5, y: 0.5 },
  { x: 8.5, y: 0.5 },
  { x: 1, y: 4.8 },
  { x: 9, y: 4.8 }
];

stadiumLights.forEach((light) => {
  // Large glow halo
  slide.addShape(pres.shapes.OVAL, {
    x: light.x - 0.6, y: light.y - 0.6, w: 1.2, h: 1.2,
    fill: { color: 'FFD700', transparency: 85 },
    line: { type: 'none' }
  });
  
  // Brighter core
  slide.addShape(pres.shapes.OVAL, {
    x: light.x - 0.15, y: light.y - 0.15, w: 0.3, h: 0.3,
    fill: { color: 'FFFF00', transparency: 20 },
    line: { type: 'none' }
  });
});

// ===== ADD TROPHY GOLDEN GLOW (INTENSE YELLOW) =====
// Large golden halo behind trophy
slide.addShape(pres.shapes.OVAL, {
  x: 4.2, y: 1.8, w: 1.6, h: 1.8,
  fill: { color: 'FFA500', transparency: 60 },
  line: { type: 'none' }
});

// Brighter golden light
slide.addShape(pres.shapes.OVAL, {
  x: 4.4, y: 2.0, w: 1.2, h: 1.4,
  fill: { color: 'FFD700', transparency: 40 },
  line: { type: 'none' }
});

// ===== TROPHY SHAPE (STYLIZED) =====
// Trophy base/cup (simplified geometric)
const trophyX = 5;
const trophyY = 2.5;

// Cup bowl
slide.addShape(pres.shapes.OVAL, {
  x: trophyX - 0.25, y: trophyY - 0.3, w: 0.5, h: 0.35,
  fill: { color: 'FFD700' },
  line: { color: 'FFA500', width: 2 }
});

// Stem
slide.addShape(pres.shapes.RECTANGLE, {
  x: trophyX - 0.05, y: trophyY + 0.08, w: 0.1, h: 0.25,
  fill: { color: 'C0A000' },
  line: { color: 'FFA500', width: 1 }
});

// Base (wide)
slide.addShape(pres.shapes.RECTANGLE, {
  x: trophyX - 0.3, y: trophyY + 0.35, w: 0.6, h: 0.12,
  fill: { color: 'DAA520' },
  line: { color: 'FFA500', width: 2 }
});

// Trophy shine/highlight
slide.addShape(pres.shapes.OVAL, {
  x: trophyX - 0.15, y: trophyY - 0.25, w: 0.15, h: 0.15,
  fill: { color: 'FFFACD' },
  line: { type: 'none' }
});

// ===== ACCENT RAYS (LIGHT BEAMS) =====
const rays = [
  { angle: 0, x: 5.8, y: 2.8 },    // right
  { angle: 45, x: 5.5, y: 2.3 },   // upper right
  { angle: 90, x: 5, y: 1.8 },     // up
  { angle: 135, x: 4.5, y: 2.3 },  // upper left
  { angle: 180, x: 4.2, y: 2.8 },  // left
  { angle: 225, x: 4.5, y: 3.3 },  // lower left
  { angle: 270, x: 5, y: 3.8 },    // down
  { angle: 315, x: 5.5, y: 3.3 }   // lower right
];

rays.forEach((ray) => {
  slide.addShape(pres.shapes.RECTANGLE, {
    x: ray.x, y: ray.y, w: 1.5, h: 0.08,
    fill: { color: 'FFFF99', transparency: 70 },
    line: { type: 'none' },
    rotate: ray.angle
  });
});

// ===== SPOTLIGHTS FROM SIDES =====
// Left spotlight beam
slide.addShape(pres.shapes.RECTANGLE, {
  x: 0.5, y: 2.0, w: 3, h: 1.5,
  fill: { color: 'FFB347', transparency: 80 },
  line: { type: 'none' },
  rotate: 30
});

// Right spotlight beam
slide.addShape(pres.shapes.RECTANGLE, {
  x: 6.5, y: 2.0, w: 3, h: 1.5,
  fill: { color: 'FFB347', transparency: 80 },
  line: { type: 'none' },
  rotate: -30
});

// ===== TEXT: CELEBRATION MESSAGE =====
// Trophy won text
slide.addText('CHAMPIONS', {
  x: 0.5, y: 0.3, w: 9, h: 0.6,
  fontSize: 48, bold: true, color: 'FFFF99',
  align: 'center', fontFace: 'Arial Black',
  shadow: { type: 'outer', blur: 8, offset: 3, color: '000000', opacity: 0.4 }
});

// Subtitle
slide.addText("The World's Greatest Trophy", {
  x: 0.5, y: 4.9, w: 9, h: 0.5,
  fontSize: 28, italic: true, color: 'FFD700',
  align: 'center', fontFace: 'Calibri',
  shadow: { type: 'outer', blur: 6, offset: 2, color: '000000', opacity: 0.3 }
});

pres.writeFile({ fileName: 'world_cup_final.pptx' });
console.log('✅ Presentation created: world_cup_final.pptx');
