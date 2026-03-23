from PIL import Image, ImageDraw, ImageFont

# Create a simple white image
img = Image.new('RGB', (400, 200), color = (255, 255, 255))
d = ImageDraw.Draw(img)

# Add some text to it
text = "INVOICE #9999\nTotal: $150.00\nThank you for your business!"

# We just use default font if none available
d.text((50, 50), text, fill=(0, 0, 0))

# Save the image
img.save('dataset/test_image.png')
print("Test image created at dataset/test_image.png")
