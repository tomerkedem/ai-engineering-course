const pptxgen = require("pptxgenjs");

let pres = new pptxgen();
pres.layout = 'LAYOUT_16x9';
pres.author = 'Sports Broadcasting Team';
pres.title = 'FIFA World Cup 2026 - The Ultimate Tournament';

// Color Palette - High Energy Sports Broadcast
const colors = {
  darkBlue: "0C1E3C",      // Deep navy (dominant)
  brightGold: "FFD700",     // Gold accent
  electricOrange: "FF6B35", // Energy orange
  vibrantRed: "EF4444",     // Vibrant red
  brightWhite: "FFFFFF",    // Clean white
  lightBlue: "E0F2FE",      // Light sky blue
  darkGray: "1F2937"        // Dark gray text
};

// ============================================================================
// SLIDE 1: TITLE SLIDE - THE GREATEST TOURNAMENT ON EARTH
// ============================================================================
let slide1 = pres.addSlide();
slide1.background = { color: colors.darkBlue };

// Accent bar on left
slide1.addShape(pres.shapes.RECTANGLE, {
  x: 0, y: 0, w: 0.15, h: 5.625,
  fill: { color: colors.electricOrange },
  line: { type: "none" }
});

// Gold stripe accent
slide1.addShape(pres.shapes.RECTANGLE, {
  x: 0.15, y: 0, w: 0.05, h: 5.625,
  fill: { color: colors.brightGold },
  line: { type: "none" }
});

// Main title with styling
slide1.addText("FIFA WORLD CUP 2026", {
  x: 1, y: 1.2, w: 8.5, h: 1.2,
  fontSize: 54, bold: true, color: colors.brightGold,
  fontFace: "Arial Black", align: "left", valign: "middle"
});

// Subtitle
slide1.addText("THE GREATEST TOURNAMENT ON EARTH", {
  x: 1, y: 2.5, w: 8.5, h: 0.6,
  fontSize: 28, color: colors.brightWhite,
  fontFace: "Arial", align: "left", valign: "middle", italic: true
});

// Decorative text blocks
slide1.addShape(pres.shapes.RECTANGLE, {
  x: 1, y: 3.3, w: 0.08, h: 1.8,
  fill: { color: colors.vibrantRed },
  line: { type: "none" }
});

slide1.addText("USA  |  MEXICO  |  CANADA", {
  x: 1.3, y: 3.3, w: 7, h: 0.5,
  fontSize: 18, color: colors.brightGold, bold: true,
  fontFace: "Arial", align: "left"
});

slide1.addText("32 TEAMS • 80 MATCHES • 1 CHAMPION", {
  x: 1.3, y: 3.9, w: 7, h: 0.5,
  fontSize: 16, color: colors.brightWhite,
  fontFace: "Arial", align: "left"
});

slide1.addText("Coming Summer 2026", {
  x: 1.3, y: 4.6, w: 7, h: 0.4,
  fontSize: 14, color: colors.lightBlue,
  fontFace: "Arial", align: "left"
});

// ============================================================================
// SLIDE 2: TOURNAMENT OVERVIEW - THE FACTS
// ============================================================================
let slide2 = pres.addSlide();
slide2.background = { color: colors.brightWhite };

// Header bar
slide2.addShape(pres.shapes.RECTANGLE, {
  x: 0, y: 0, w: 10, h: 0.8,
  fill: { color: colors.darkBlue },
  line: { type: "none" }
});

// Gold underline
slide2.addShape(pres.shapes.RECTANGLE, {
  x: 0, y: 0.8, w: 10, h: 0.08,
  fill: { color: colors.brightGold },
  line: { type: "none" }
});

// Title
slide2.addText("TOURNAMENT OVERVIEW", {
  x: 0.5, y: 0.15, w: 9, h: 0.5,
  fontSize: 36, bold: true, color: colors.brightGold,
  fontFace: "Arial Black", valign: "middle"
});

// Content - 2 columns of stats
// Left column
slide2.addShape(pres.shapes.RECTANGLE, {
  x: 0.5, y: 1.2, w: 4, h: 1,
  fill: { color: colors.electricOrange },
  line: { type: "none" }
});

slide2.addText([
  { text: "32", options: { fontSize: 44, bold: true, color: colors.brightWhite, breakLine: true } },
  { text: "TEAMS", options: { fontSize: 14, color: colors.brightWhite } }
], {
  x: 0.5, y: 1.2, w: 4, h: 1,
  align: "center", valign: "middle", margin: 0
});

// Right column stat 1
slide2.addShape(pres.shapes.RECTANGLE, {
  x: 5.5, y: 1.2, w: 4, h: 1,
  fill: { color: colors.vibrantRed },
  line: { type: "none" }
});

slide2.addText([
  { text: "80", options: { fontSize: 44, bold: true, color: colors.brightWhite, breakLine: true } },
  { text: "MATCHES", options: { fontSize: 14, color: colors.brightWhite } }
], {
  x: 5.5, y: 1.2, w: 4, h: 1,
  align: "center", valign: "middle", margin: 0
});

// Left column stat 2
slide2.addShape(pres.shapes.RECTANGLE, {
  x: 0.5, y: 2.4, w: 4, h: 1,
  fill: { color: colors.darkBlue },
  line: { type: "none" }
});

slide2.addText([
  { text: "12", options: { fontSize: 44, bold: true, color: colors.brightGold, breakLine: true } },
  { text: "STADIUMS", options: { fontSize: 14, color: colors.brightGold } }
], {
  x: 0.5, y: 2.4, w: 4, h: 1,
  align: "center", valign: "middle", margin: 0
});

// Right column stat 2
slide2.addShape(pres.shapes.RECTANGLE, {
  x: 5.5, y: 2.4, w: 4, h: 1,
  fill: { color: colors.electricOrange },
  line: { type: "none" }
});

slide2.addText([
  { text: "5B+", options: { fontSize: 44, bold: true, color: colors.brightWhite, breakLine: true } },
  { text: "VIEWERS", options: { fontSize: 14, color: colors.brightWhite } }
], {
  x: 5.5, y: 2.4, w: 4, h: 1,
  align: "center", valign: "middle", margin: 0
});

// Key highlights
slide2.addText("TOURNAMENT HIGHLIGHTS", {
  x: 0.5, y: 3.7, w: 9, h: 0.4,
  fontSize: 18, bold: true, color: colors.darkBlue,
  fontFace: "Arial"
});

// Bullet points with icons
slide2.addText([
  { text: "First tournament hosted by 3 nations (USA, Mexico, Canada)", options: { bullet: true, breakLine: true } },
  { text: "Expanded format with 80 matches across 2 months", options: { bullet: true, breakLine: true } },
  { text: "Record viewership expected globally", options: { bullet: true } }
], {
  x: 0.7, y: 4.2, w: 8.6, h: 1.2,
  fontSize: 14, color: colors.darkGray,
  fontFace: "Arial"
});

// ============================================================================
// SLIDE 3: THE HOSTING NATIONS - UNITED FRONT
// ============================================================================
let slide3 = pres.addSlide();
slide3.background = { color: colors.darkBlue };

// Top accent
slide3.addShape(pres.shapes.RECTANGLE, {
  x: 0, y: 0, w: 10, h: 0.08,
  fill: { color: colors.brightGold },
  line: { type: "none" }
});

// Title
slide3.addText("THE HOST NATIONS", {
  x: 0.5, y: 0.3, w: 9, h: 0.5,
  fontSize: 36, bold: true, color: colors.brightGold,
  fontFace: "Arial Black", align: "left"
});

// Three country sections
// USA
slide3.addShape(pres.shapes.RECTANGLE, {
  x: 0.5, y: 1.1, w: 2.8, h: 4,
  fill: { color: colors.brightWhite },
  line: { type: "none" }
});

slide3.addShape(pres.shapes.RECTANGLE, {
  x: 0.5, y: 1.1, w: 2.8, h: 0.7,
  fill: { color: colors.vibrantRed },
  line: { type: "none" }
});

slide3.addText("🇺🇸 USA", {
  x: 0.5, y: 1.15, w: 2.8, h: 0.6,
  fontSize: 24, bold: true, color: colors.brightWhite,
  fontFace: "Arial Black", align: "center", valign: "middle", margin: 0
});

slide3.addText([
  { text: "Dallas", options: { bullet: true, breakLine: true } },
  { text: "Los Angeles", options: { bullet: true, breakLine: true } },
  { text: "New York", options: { bullet: true, breakLine: true } },
  { text: "Miami", options: { bullet: true, breakLine: true } },
  { text: "Atlanta", options: { bullet: true } }
], {
  x: 0.7, y: 2, w: 2.4, h: 2.8,
  fontSize: 12, color: colors.darkGray,
  fontFace: "Arial"
});

// MEXICO
slide3.addShape(pres.shapes.RECTANGLE, {
  x: 3.6, y: 1.1, w: 2.8, h: 4,
  fill: { color: colors.brightWhite },
  line: { type: "none" }
});

slide3.addShape(pres.shapes.RECTANGLE, {
  x: 3.6, y: 1.1, w: 2.8, h: 0.7,
  fill: { color: colors.electricOrange },
  line: { type: "none" }
});

slide3.addText("🇲🇽 MEXICO", {
  x: 3.6, y: 1.15, w: 2.8, h: 0.6,
  fontSize: 24, bold: true, color: colors.brightWhite,
  fontFace: "Arial Black", align: "center", valign: "middle", margin: 0
});

slide3.addText([
  { text: "Mexico City", options: { bullet: true, breakLine: true } },
  { text: "Guadalajara", options: { bullet: true, breakLine: true } },
  { text: "Monterrey", options: { bullet: true, breakLine: true } },
  { text: "Querétaro", options: { bullet: true } }
], {
  x: 3.8, y: 2, w: 2.4, h: 2.8,
  fontSize: 12, color: colors.darkGray,
  fontFace: "Arial"
});

// CANADA
slide3.addShape(pres.shapes.RECTANGLE, {
  x: 6.7, y: 1.1, w: 2.8, h: 4,
  fill: { color: colors.brightWhite },
  line: { type: "none" }
});

slide3.addShape(pres.shapes.RECTANGLE, {
  x: 6.7, y: 1.1, w: 2.8, h: 0.7,
  fill: { color: colors.darkBlue },
  line: { type: "none" }
});

slide3.addText("🇨🇦 CANADA", {
  x: 6.7, y: 1.15, w: 2.8, h: 0.6,
  fontSize: 24, bold: true, color: colors.brightWhite,
  fontFace: "Arial Black", align: "center", valign: "middle", margin: 0
});

slide3.addText([
  { text: "Toronto", options: { bullet: true, breakLine: true } },
  { text: "Vancouver", options: { bullet: true, breakLine: true } },
  { text: "Montreal", options: { bullet: true, breakLine: true } },
  { text: "Edmonton", options: { bullet: true } }
], {
  x: 6.9, y: 2, w: 2.4, h: 2.8,
  fontSize: 12, color: colors.darkGray,
  fontFace: "Arial"
});

// ============================================================================
// SLIDE 4: WHAT TO EXPECT - THE EXCITEMENT
// ============================================================================
let slide4 = pres.addSlide();
slide4.background = { color: colors.brightWhite };

// Header bar with gradient effect using shapes
slide4.addShape(pres.shapes.RECTANGLE, {
  x: 0, y: 0, w: 10, h: 0.08,
  fill: { color: colors.brightGold },
  line: { type: "none" }
});

// Title with left accent
slide4.addShape(pres.shapes.RECTANGLE, {
  x: 0.5, y: 0.3, w: 0.08, h: 0.5,
  fill: { color: colors.electricOrange },
  line: { type: "none" }
});

slide4.addText("WHAT TO EXPECT", {
  x: 0.7, y: 0.3, w: 8.8, h: 0.5,
  fontSize: 36, bold: true, color: colors.darkBlue,
  fontFace: "Arial Black", align: "left", valign: "middle", margin: 0
});

// Four feature cards in 2x2 grid
const features = [
  {
    title: "⚡ HIGH-OCTANE ACTION",
    desc: "32 teams competing for glory in 80 intense matches",
    x: 0.5, y: 1.2
  },
  {
    title: "🌎 GLOBAL SPECTACLE",
    desc: "Billions of fans worldwide watching history unfold",
    x: 5.25, y: 1.2
  },
  {
    title: "🏆 LEGENDARY MOMENTS",
    desc: "Unforgettable goals, dramatic comebacks, and upsets",
    x: 0.5, y: 3.0
  },
  {
    title: "🎊 CELEBRATION CULTURE",
    desc: "Nations united through passion for the beautiful game",
    x: 5.25, y: 3.0
  }
];

features.forEach((feature, idx) => {
  // Card background
  slide4.addShape(pres.shapes.RECTANGLE, {
    x: feature.x, y: feature.y, w: 4.25, h: 1.6,
    fill: { color: colors.lightBlue },
    line: { color: colors.electricOrange, width: 2 }
  });

  // Title
  slide4.addText(feature.title, {
    x: feature.x + 0.2, y: feature.y + 0.15, w: 3.85, h: 0.5,
    fontSize: 14, bold: true, color: colors.darkBlue,
    fontFace: "Arial", align: "left"
  });

  // Description
  slide4.addText(feature.desc, {
    x: feature.x + 0.2, y: feature.y + 0.65, w: 3.85, h: 0.85,
    fontSize: 12, color: colors.darkGray,
    fontFace: "Arial", align: "left", valign: "top"
  });
});

// Bottom CTA
slide4.addText("THE WORLD'S MOST WATCHED SPORTING EVENT", {
  x: 0.5, y: 4.9, w: 9, h: 0.5,
  fontSize: 16, bold: true, color: colors.vibrantRed,
  fontFace: "Arial Black", align: "center", italic: true
});

// ============================================================================
// SLIDE 5: FINALE - COUNTDOWN TO GLORY
// ============================================================================
let slide5 = pres.addSlide();
slide5.background = { color: colors.darkBlue };

// Multi-color accent bars at top
slide5.addShape(pres.shapes.RECTANGLE, {
  x: 0, y: 0, w: 2.5, h: 0.15,
  fill: { color: colors.vibrantRed },
  line: { type: "none" }
});

slide5.addShape(pres.shapes.RECTANGLE, {
  x: 2.5, y: 0, w: 2.5, h: 0.15,
  fill: { color: colors.brightGold },
  line: { type: "none" }
});

slide5.addShape(pres.shapes.RECTANGLE, {
  x: 5, y: 0, w: 2.5, h: 0.15,
  fill: { color: colors.electricOrange },
  line: { type: "none" }
});

slide5.addShape(pres.shapes.RECTANGLE, {
  x: 7.5, y: 0, w: 2.5, h: 0.15,
  fill: { color: colors.brightGold },
  line: { type: "none" }
});

// Main message
slide5.addText("COUNTDOWN TO GLORY", {
  x: 0.5, y: 1.2, w: 9, h: 0.8,
  fontSize: 48, bold: true, color: colors.brightGold,
  fontFace: "Arial Black", align: "center", valign: "middle"
});

// Date/period
slide5.addShape(pres.shapes.RECTANGLE, {
  x: 2, y: 2.2, w: 6, h: 0.8,
  fill: { color: colors.vibrantRed },
  line: { type: "none" }
});

slide5.addText("SUMMER 2026 - USA, MEXICO, CANADA", {
  x: 2, y: 2.2, w: 6, h: 0.8,
  fontSize: 28, bold: true, color: colors.brightWhite,
  fontFace: "Arial Black", align: "center", valign: "middle", margin: 0
});

// Feature highlights
slide5.addText([
  { text: "Be there for the greatest tournament on Earth", options: { breakLine: true, fontSize: 16, bold: true, color: colors.brightWhite } },
  { text: "", options: { breakLine: true } },
  { text: "Experience the passion of 32 nations", options: { breakLine: true, fontSize: 14, color: colors.brightGold } },
  { text: "Witness historic moments that will live forever", options: { breakLine: true, fontSize: 14, color: colors.brightGold } },
  { text: "Be part of the global celebration", options: { fontSize: 14, color: colors.brightGold } }
], {
  x: 1, y: 3.2, w: 8, h: 1.8,
  fontFace: "Arial", align: "center", valign: "middle"
});

// Footer tagline
slide5.addShape(pres.shapes.RECTANGLE, {
  x: 0, y: 5.2, w: 10, h: 0.425,
  fill: { color: colors.electricOrange },
  line: { type: "none" }
});

slide5.addText("⚽ WHERE NATIONS COLLIDE • LEGENDS ARE BORN • HISTORY IS WRITTEN ⚽", {
  x: 0.5, y: 5.25, w: 9, h: 0.325,
  fontSize: 14, bold: true, color: colors.brightWhite,
  fontFace: "Arial Black", align: "center", valign: "middle", margin: 0
});

// Write the presentation
pres.writeFile({ fileName: "FIFA_World_Cup_2026.pptx" });
console.log("✅ Presentation created: FIFA_World_Cup_2026.pptx");
