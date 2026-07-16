const pptxgen = require("pptxgenjs");

let pres = new pptxgen();
pres.layout = 'LAYOUT_16x9';
pres.author = 'AI Engineering Course';
pres.title = 'Sports Broadcast Opener';

let slide = pres.addSlide();

// ============================================
// BACKGROUND: Dark gradient effect with shapes
// ============================================
slide.background = { color: "0F1419" };  // Deep charcoal

// Add dynamic gradient-like effect with overlapping shapes
slide.addShape(pres.shapes.RECTANGLE, {
  x: 0, y: 0, w: 10, h: 2.8,
  fill: { color: "1A1F2E" }, line: { type: "none" }
});

// Accent bar along left edge
slide.addShape(pres.shapes.RECTANGLE, {
  x: 0, y: 0, w: 0.12, h: 5.625,
  fill: { color: "FF4444" }, line: { type: "none" }  // Bold red accent
});

// Top accent bar
slide.addShape(pres.shapes.RECTANGLE, {
  x: 0, y: 0, w: 10, h: 0.15,
  fill: { color: "FF4444" }, line: { type: "none" }
});

// Diagonal accent shape (lower right area)
slide.addShape(pres.shapes.RECTANGLE, {
  x: 7.5, y: 3.8, w: 2.5, h: 1.825,
  fill: { color: "1F2937" }, line: { type: "none" },
  rotate: 15
});

// ============================================
// MAIN HEADLINE - BOLD & COMMANDING
// ============================================
slide.addText("LIVE FROM THE ARENA", {
  x: 0.5, y: 0.4, w: 8, h: 1.2,
  fontSize: 72, bold: true, fontFace: "Arial Black",
  color: "FFFFFF",
  align: "left", valign: "top",
  margin: 0,
  charSpacing: 4  // Wide letter spacing for broadcast impact
});

// ============================================
// SECONDARY LINE - Dynamic accent
// ============================================
slide.addText("CHAMPIONSHIP PLAYOFFS 2026", {
  x: 0.5, y: 1.6, w: 8, h: 0.5,
  fontSize: 28, bold: false, fontFace: "Arial",
  color: "FF4444",  // Red accent color
  align: "left", valign: "top",
  margin: 0
});

// ============================================
// DECORATIVE LINE SEPARATOR
// ============================================
slide.addShape(pres.shapes.RECTANGLE, {
  x: 0.5, y: 2.15, w: 3.5, h: 0.08,
  fill: { color: "FF4444" }, line: { type: "none" }
});

// ============================================
// LARGE STAT CALLOUT (Left side)
// ============================================
slide.addShape(pres.shapes.RECTANGLE, {
  x: 0.5, y: 3.2, w: 2.8, h: 2.15,
  fill: { color: "1A2332" },
  line: { color: "FF4444", width: 2 },
  shadow: { type: "outer", color: "000000", blur: 8, offset: 3, angle: 135, opacity: 0.3 }
});

slide.addText("42.3M", {
  x: 0.5, y: 3.4, w: 2.8, h: 0.8,
  fontSize: 48, bold: true, fontFace: "Arial Black",
  color: "FF4444",
  align: "center", valign: "top",
  margin: 0
});

slide.addText("VIEWERS WORLDWIDE", {
  x: 0.5, y: 4.25, w: 2.8, h: 0.6,
  fontSize: 12, bold: true, fontFace: "Arial",
  color: "AAAAAA",
  align: "center", valign: "top",
  margin: 0
});

// ============================================
// LIVE INDICATOR (animated-effect)
// ============================================
slide.addShape(pres.shapes.OVAL, {
  x: 1, y: 2.45, w: 0.25, h: 0.25,
  fill: { color: "FF4444" },
  line: { type: "none" },
  shadow: { type: "outer", color: "FF4444", blur: 6, offset: 0, angle: 0, opacity: 0.6 }
});

slide.addText("LIVE", {
  x: 1.35, y: 2.42, w: 0.8, h: 0.3,
  fontSize: 14, bold: true, fontFace: "Arial Black",
  color: "FF4444",
  align: "left", valign: "middle",
  margin: 0
});

// ============================================
// COUNTDOWN/TIMER STYLE BOX (Right side)
// ============================================
slide.addShape(pres.shapes.RECTANGLE, {
  x: 6.8, y: 3.2, w: 2.8, h: 2.15,
  fill: { color: "1A2332" },
  line: { color: "4488FF", width: 2 },
  shadow: { type: "outer", color: "000000", blur: 8, offset: 3, angle: 135, opacity: 0.3 }
});

slide.addText("GAME TIME", {
  x: 6.8, y: 3.4, w: 2.8, h: 0.5,
  fontSize: 14, bold: true, fontFace: "Arial",
  color: "4488FF",
  align: "center", valign: "top",
  margin: 0
});

slide.addText("8:00 PM EST", {
  x: 6.8, y: 3.95, w: 2.8, h: 0.7,
  fontSize: 42, bold: true, fontFace: "Arial Black",
  color: "FFFFFF",
  align: "center", valign: "middle",
  margin: 0
});

slide.addText("01:47:23", {
  x: 6.8, y: 4.75, w: 2.8, h: 0.5,
  fontSize: 16, fontFace: "Courier New",
  color: "4488FF",
  align: "center", valign: "top",
  margin: 0
});

// ============================================
// BOTTOM INFORMATION BAR
// ============================================
slide.addShape(pres.shapes.RECTANGLE, {
  x: 0, y: 5.15, w: 10, h: 0.475,
  fill: { color: "1A1F2E" },
  line: { type: "none" }
});

slide.addText("Champions League Finals  •  Stadium Capacity: 98,500  •  Venue: International Sports Complex", {
  x: 0.5, y: 5.2, w: 9, h: 0.4,
  fontSize: 11, fontFace: "Arial",
  color: "999999",
  align: "left", valign: "middle",
  margin: 0
});

// ============================================
// Save presentation
// ============================================
pres.writeFile({ fileName: "sports_broadcast_opener.pptx" });
console.log("✅ Sports Broadcast Opening Slide created: sports_broadcast_opener.pptx");
