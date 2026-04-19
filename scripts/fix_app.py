from pathlib import Path

p = Path('/workspace/moe_medical_vision/app.py')
text = p.read_text()

# Fix the float64 to float32 issue permanently
text = text.replace("tensor_224 = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)", "tensor_224 = torch.from_numpy(img_array).float().permute(2, 0, 1).unsqueeze(0)")

# We make sure process_3d_volume also returns float32
text = text.replace("tensor_3d = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0)", "tensor_3d = torch.from_numpy(vol).float().unsqueeze(0).unsqueeze(0)")

p.write_text(text)
print('APP PARCHEADA CON FLOAT32')
