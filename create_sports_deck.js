const pptxgen = require("pptxgenjs");

let pres = new pptxgen();
pres.layout = 'LAYOUT_16x9';
pres.author = 'Sports Broadcast Team';
pres.title = 'Host Countries Showcase';

// Color palette - energetic sports broadcast style
const colors = {
  primary: "FF2D2D",      // Vibrant red
  secondary: "FFD700",    // Gold accent
  dark: "1A1A1A",         // Deep black
  white: "FFFFFF",
  accent1: "00A3FF",      // Electric blue
  accent2: "00FF88",      // Neon green
  accent3: "FF6B35"       // Energetic orange
};

// ====== SLIDE 1: TITLE SLIDE ======
let slide1 = pres.addSlide();
slide1.background = { color: colors.dark };

// Main title with dramatic styling
slide1.addText("HOST NATIONS", {
  x: 0.5, y: 1.5, w: 9, h: 1,
  fontSize: 54, bold: true, color: colors.primary,
  align: "center", fontFace: "Arial Black"
});

// Subtitle
slide1.addText("Competitive Excellence | Global Championship", {
  x: 0.5, y: 2.7, w: 9, h: 0.6,
  fontSize: 18, color: colors.secondary,
  align: "center", fontFace: "Arial", italic: true
});

// Decorative line under title
slide1.addShape(pres.shapes.RECTANGLE, {
  x: 3.5, y: 2.3, w: 3, h: 0.08,
  fill: { color: colors.secondary }, line: { type: "none" }
});

// ====== SLIDE 2: 3-PART CONNECTED HOST COUNTRIES ======
let slide2 = pres.addSlide();
slide2.background = { color: colors.white };

// Main title
slide2.addText("Three Legendary Host Countries", {
  x: 0.5, y: 0.3, w: 9, h: 0.5,
  fontSize: 32, bold: true, color: colors.primary,
  align: "left", fontFace: "Arial Black"
});

// Decorative accent line
slide2.addShape(pres.shapes.RECTANGLE, {
  x: 0.5, y: 0.9, w: 9, h: 0.06,
  fill: { color: colors.secondary }, line: { type: "none" }
});

// ===== PART 1: JAPAN =====
const part1X = 0.5;
const partWidth = 2.8;
const partHeight = 3.8;

// Card background
slide2.addShape(pres.shapes.RECTANGLE, {
  x: part1X, y: 1.2, w: partWidth, h: partHeight,
  fill: { color: "#FAFAFA" }, 
  shadow: { type: "outer", color: "000000", blur: 8, offset: 3, angle: 135, opacity: 0.15 }
});

// Top accent bar
slide2.addShape(pres.shapes.RECTANGLE, {
  x: part1X, y: 1.2, w: partWidth, h: 0.12,
  fill: { color: "#E63946" }, line: { type: "none" }
});

// Flag emoji / visual
slide2.addText("🇯🇵", {
  x: part1X, y: 1.4, w: partWidth, h: 0.6,
  fontSize: 48, align: "center", valign: "middle"
});

// Country name
slide2.addText("JAPAN", {
  x: part1X, y: 2.1, w: partWidth, h: 0.4,
  fontSize: 20, bold: true, color: colors.primary,
  align: "center", fontFace: "Arial Black"
});

// Divider line
slide2.addShape(pres.shapes.RECTANGLE, {
  x: part1X + 0.2, y: 2.55, w: partWidth - 0.4, h: 0.04,
  fill: { color: colors.secondary }, line: { type: "none" }
});

// Content
slide2.addText([
  { text: "Tokyo 2020\nOlympics", options: { breakLine: true, fontSize: 13, bold: true, color: colors.dark } },
  { text: "\nPrecision & Innovation\nWorld-Class Venues\nCultural Excellence", options: { fontSize: 11, color: "#555555" } }
], {
  x: part1X + 0.15, y: 2.7, w: partWidth - 0.3, h: 1.3,
  align: "center", valign: "middle", lineSpacing: 16
});

// Badge
slide2.addShape(pres.shapes.OVAL, {
  x: part1X + (partWidth - 0.4) / 2 - 0.2, y: 4.7, w: 0.4, h: 0.4,
  fill: { color: colors.accent1 }
});

slide2.addText("🏆", {
  x: part1X + (partWidth - 0.4) / 2 - 0.2, y: 4.7, w: 0.4, h: 0.4,
  fontSize: 20, align: "center", valign: "middle"
});

// ===== CONNECTING LINE 1 =====
slide2.addShape(pres.shapes.RECTANGLE, {
  x: part1X + partWidth, y: 3.0, w: 0.5, h: 0.08,
  fill: { color: colors.secondary }, line: { type: "none" }
});

// Connector dot
slide2.addShape(pres.shapes.OVAL, {
  x: part1X + partWidth + 0.2, y: 2.96, w: 0.16, h: 0.16,
  fill: { color: colors.secondary }
});

// ===== PART 2: USA =====
const part2X = 3.65;

// Card background
slide2.addShape(pres.shapes.RECTANGLE, {
  x: part2X, y: 1.2, w: partWidth, h: partHeight,
  fill: { color: "#F0F8FF" },
  shadow: { type: "outer", color: "000000", blur: 8, offset: 3, angle: 135, opacity: 0.15 }
});

// Top accent bar
slide2.addShape(pres.shapes.RECTANGLE, {
  x: part2X, y: 1.2, w: partWidth, h: 0.12,
  fill: { color: colors.accent1 }, line: { type: "none" }
});

// Flag emoji
slide2.addText("🇺🇸", {
  x: part2X, y: 1.4, w: partWidth, h: 0.6,
  fontSize: 48, align: "center", valign: "middle"
});

// Country name
slide2.addText("USA", {
  x: part2X, y: 2.1, w: partWidth, h: 0.4,
  fontSize: 20, bold: true, color: colors.accent1,
  align: "center", fontFace: "Arial Black"
});

// Divider line
slide2.addShape(pres.shapes.RECTANGLE, {
  x: part2X + 0.2, y: 2.55, w: partWidth - 0.4, h: 0.04,
  fill: { color: colors.accent1 }, line: { type: "none" }
});

// Content
slide2.addText([
  { text: "Los Angeles 2028\nOlympics", options: { breakLine: true, fontSize: 13, bold: true, color: colors.dark } },
  { text: "\nLargest Sports Market\nState-of-Art Technology\nInconic Venues", options: { fontSize: 11, color: "#555555" } }
], {
  x: part2X + 0.15, y: 2.7, w: partWidth - 0.3, h: 1.3,
  align: "center", valign: "middle", lineSpacing: 16
});

// Badge
slide2.addShape(pres.shapes.OVAL, {
  x: part2X + (partWidth - 0.4) / 2 - 0.2, y: 4.7, w: 0.4, h: 0.4,
  fill: { color: colors.secondary }
});

slide2.addText("⭐", {
  x: part2X + (partWidth - 0.4) / 2 - 0.2, y: 4.7, w: 0.4, h: 0.4,
  fontSize: 20, align: "center", valign: "middle"
});

// ===== CONNECTING LINE 2 =====
slide2.addShape(pres.shapes.RECTANGLE, {
  x: part2X + partWidth, y: 3.0, w: 0.5, h: 0.08,
  fill: { color: colors.secondary }, line: { type: "none" }
});

// Connector dot
slide2.addShape(pres.shapes.OVAL, {
  x: part2X + partWidth + 0.2, y: 2.96, w: 0.16, h: 0.16,
  fill: { color: colors.secondary }
});

// ===== PART 3: FRANCE =====
const part3X = 6.8;

// Card background
slide2.addShape(pres.shapes.RECTANGLE, {
  x: part3X, y: 1.2, w: partWidth, h: partHeight,
  fill: { color: "#FFF8DC" },
  shadow: { type: "outer", color: "000000", blur: 8, offset: 3, angle: 135, opacity: 0.15 }
});

// Top accent bar
slide2.addShape(pres.shapes.RECTANGLE, {
  x: part3X, y: 1.2, w: partWidth, h: 0.12,
  fill: { color: colors.accent3 }, line: { type: "none" }
});

// Flag emoji
slide2.addText("🇫🇷", {
  x: part3X, y: 1.4, w: partWidth, h: 0.6,
  fontSize: 48, align: "center", valign: "middle"
});

// Country name
slide2.addText("FRANCE", {
  x: part3X, y: 2.1, w: partWidth, h: 0.4,
  fontSize: 20, bold: true, color: colors.accent3,
  align: "center", fontFace: "Arial Black"
});

// Divider line
slide2.addShape(pres.shapes.RECTANGLE, {
  x: part3X + 0.2, y: 2.55, w: partWidth - 0.4, h: 0.04,
  fill: { color: colors.accent3 }, line: { type: "none" }
});

// Content
slide2.addText([
  { text: "Paris 2024\nOlympics", options: { breakLine: true, fontSize: 13, bold: true, color: colors.dark } },
  { text: "\nHistoric Venues\nFrench Elegance\nSustainability Focus", options: { fontSize: 11, color: "#555555" } }
], {
  x: part3X + 0.15, y: 2.7, w: partWidth - 0.3, h: 1.3,
  align: "center", valign: "middle", lineSpacing: 16
});

// Badge
slide2.addShape(pres.shapes.OVAL, {
  x: part3X + (partWidth - 0.4) / 2 - 0.2, y: 4.7, w: 0.4, h: 0.4,
  fill: { color: colors.accent2 }
});

slide2.addText("🎯", {
  x: part3X + (partWidth - 0.4) / 2 - 0.2, y: 4.7, w: 0.4, h: 0.4,
  fontSize: 20, align: "center", valign: "middle"
});

// ====== SLIDE 3: FINAL STATS SLIDE ======
let slide3 = pres.addSlide();
slide3.background = { color: colors.dark };

slide3.addText("Global Championship Impact", {
  x: 0.5, y: 0.5, w: 9, h: 0.6,
  fontSize: 40, bold: true, color: colors.secondary,
  align: "center", fontFace: "Arial Black"
});

// Three stat boxes
const statBoxWidth = 2.6;
const statBoxX = [1, 3.7, 6.4];
const statColors = [colors.primary, colors.accent1, colors.accent3];
const statLabels = ["Athlete Nations", "Global Viewers", "Events Showcased"];
const statValues = ["206", "4.7B", "329"];

for (let i = 0; i < 3; i++) {
  // Box background
  slide3.addShape(pres.shapes.RECTANGLE, {
    x: statBoxX[i], y: 1.8, w: statBoxWidth, h: 2.5,
    fill: { color: "2A2A2A" },
    line: { color: statColors[i], width: 3 }
  });

  // Stat number
  slide3.addText(statValues[i], {
    x: statBoxX[i], y: 2.2, w: statBoxWidth, h: 0.8,
    fontSize: 48, bold: true, color: statColors[i],
    align: "center", fontFace: "Arial Black"
  });

  // Stat label
  slide3.addText(statLabels[i], {
    x: statBoxX[i], y: 3.2, w: statBoxWidth, h: 0.6,
    fontSize: 14, color: colors.white,
    align: "center"
  });

  // Accent dot
  slide3.addShape(pres.shapes.OVAL, {
    x: statBoxX[i] + statBoxWidth / 2 - 0.15, y: 3.95, w: 0.3, h: 0.3,
    fill: { color: statColors[i] }
  });
}

// Bottom tagline
slide3.addText("Excellence in Motion", {
  x: 0.5, y: 4.8, w: 9, h: 0.5,
  fontSize: 24, bold: true, color: colors.secondary,
  align: "center", fontFace: "Arial", italic: true
});

// Write the presentation
pres.writeFile({ fileName: "Host_Countries_Sports_Broadcast.pptx" });
console.log("✅ Presentation created: Host_Countries_Sports_Broadcast.pptx");
