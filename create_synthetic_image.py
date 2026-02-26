
from PIL import Image, ImageDraw

def create_synthetic_image(filename='demo_dog.jpg'):
    # Create a 224x224 RGB image
    img = Image.new('RGB', (224, 224), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Draw a red rectangle
    draw.rectangle([50, 50, 150, 150], fill=(255, 0, 0))
    
    # Draw a blue circle
    draw.ellipse([100, 100, 200, 200], fill=(0, 0, 255))
    
    # Draw a green triangle
    draw.polygon([(20, 200), (100, 50), (180, 200)], outline=(0, 255, 0))
    
    img.save(filename)
    print(f"Created synthetic image: {filename}")

if __name__ == "__main__":
    create_synthetic_image()
