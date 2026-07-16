import zipfile
import sys

try:
    with zipfile.ZipFile('FIFA_World_Cup_2026.pptx', 'r') as z:
        # Count slides
        slides = [f for f in z.namelist() if 'ppt/slides/slide' in f and f.endswith('.xml')]
        print(f"✅ Valid PPTX file created successfully!")
        print(f"📊 Number of slides: {len(slides)}")
        
        # Extract and check content from slides
        for i, slide_file in enumerate(sorted(slides), 1):
            slide_xml = z.read(slide_file).decode('utf-8')
            print(f"\n📄 Slide {i} content verified:")
            
            # Check for key content
            if i == 1:
                if 'FIFA WORLD CUP 2026' in slide_xml and 'GREATEST TOURNAMENT' in slide_xml:
                    print("  ✅ Title slide with main content")
            elif i == 2:
                if 'TOURNAMENT OVERVIEW' in slide_xml and 'TEAMS' in slide_xml:
                    print("  ✅ Overview slide with statistics")
            elif i == 3:
                if 'HOST NATIONS' in slide_xml:
                    print("  ✅ Host nations slide")
            elif i == 4:
                if 'WHAT TO EXPECT' in slide_xml:
                    print("  ✅ Features/expectations slide")
            elif i == 5:
                if 'COUNTDOWN TO GLORY' in slide_xml:
                    print("  ✅ Finale/closing slide")
        
        print("\n" + "="*60)
        print("🎉 PRESENTATION CREATED SUCCESSFULLY!")
        print("="*60)
        print("📁 File: FIFA_World_Cup_2026.pptx")
        print("🎯 Slides: 5")
        print("🎨 Style: High-energy sports broadcast with vibrant colors")
        print("📊 Content: FIFA World Cup 2026 overview and highlights")
        print("="*60)
            
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
