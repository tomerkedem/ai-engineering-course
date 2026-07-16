const pptxgen = require('pptxgenjs');

let pres = new pptxgen();
pres.layout = 'LAYOUT_16x9';
pres.author = 'AI Engineering Course';
pres.title = 'Mondial';

// Create slide with dark background
let slide = pres.addSlide();
slide.background = { color: '065A82' };

// Main title
slide.addText('MONDIAL', {
  x: 0.5, y: 0.5, w: 9, h: 0.8,
  fontSize: 48, bold: true, color: 'FFFFFF',
  align: 'center', fontFace: 'Calibri'
});

// Subtitle
slide.addText('The World\'s Premier International Football Tournament', {
  x: 0.5, y: 1.3, w: 9, h: 0.5,
  fontSize: 24, color: '21295C', italic: true,
  align: 'center', fontFace: 'Calibri'
});

// Left column - What is Mondial
slide.addShape(pres.shapes.RECTANGLE, {
  x: 0.5, y: 2.0, w: 4.5, h: 3.4,
  fill: { color: 'FFFFFF' },
  line: { color: '00A896', width: 3 }
});

slide.addText('What is Mondial?', {
  x: 0.7, y: 2.15, w: 4.1, h: 0.4,
  fontSize: 16, bold: true, color: '028090',
  fontFace: 'Calibri'
});

slide.addText([
  { text: 'The FIFA World Cup - held every 4 years', options: { bullet: true, breakLine: true } },
  { text: 'Features 32 national teams (expanded to 48 in 2026)', options: { bullet: true, breakLine: true } },
  { text: 'Began in 1930 in Uruguay', options: { bullet: true, breakLine: true } },
  { text: 'Watched by 4+ billion people globally', options: { bullet: true, breakLine: true } },
  { text: 'Symbol of international competition & unity', options: { bullet: true } }
], {
  x: 0.7, y: 2.65, w: 4.1, h: 2.6,
  fontSize: 12, color: '36454F',
  fontFace: 'Calibri'
});

// Right column - Key Stats
slide.addShape(pres.shapes.RECTANGLE, {
  x: 5.2, y: 2.0, w: 4.3, h: 3.4,
  fill: { color: 'FFFFFF' },
  line: { color: '00A896', width: 3 }
});

slide.addText('Key Facts', {
  x: 5.4, y: 2.15, w: 3.9, h: 0.4,
  fontSize: 16, bold: true, color: '028090',
  fontFace: 'Calibri'
});

// Stats boxes
slide.addShape(pres.shapes.RECTANGLE, {
  x: 5.4, y: 2.75, w: 1.8, h: 0.8,
  fill: { color: '1C7293' }
});
slide.addText('2026 Host\nUSA/Mexico/Canada', {
  x: 5.4, y: 2.75, w: 1.8, h: 0.8,
  fontSize: 10, color: 'FFFFFF', bold: true,
  align: 'center', valign: 'middle'
});

slide.addShape(pres.shapes.RECTANGLE, {
  x: 7.4, y: 2.75, w: 1.8, h: 0.8,
  fill: { color: '1C7293' }
});
slide.addText('104 Matches\nTotal Games', {
  x: 7.4, y: 2.75, w: 1.8, h: 0.8,
  fontSize: 10, color: 'FFFFFF', bold: true,
  align: 'center', valign: 'middle'
});

slide.addShape(pres.shapes.RECTANGLE, {
  x: 5.4, y: 3.75, w: 1.8, h: 0.8,
  fill: { color: '1C7293' }
});
slide.addText('France\nCurrent Champion', {
  x: 5.4, y: 3.75, w: 1.8, h: 0.8,
  fontSize: 10, color: 'FFFFFF', bold: true,
  align: 'center', valign: 'middle'
});

slide.addShape(pres.shapes.RECTANGLE, {
  x: 7.4, y: 3.75, w: 1.8, h: 0.8,
  fill: { color: '1C7293' }
});
slide.addText('Brazil\nMost Titles (5)', {
  x: 7.4, y: 3.75, w: 1.8, h: 0.8,
  fontSize: 10, color: 'FFFFFF', bold: true,
  align: 'center', valign: 'middle'
});

// Footer
slide.addText('The greatest spectacle in international sports', {
  x: 0.5, y: 5.2, w: 9, h: 0.35,
  fontSize: 12, italic: true, color: 'FFFFFF',
  align: 'center', fontFace: 'Calibri'
});

pres.writeFile({ fileName: 'Mondial.pptx' });
console.log('Presentation created: Mondial.pptx');
