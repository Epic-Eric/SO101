from model.src.data import collect_images_with_teleoperation

result = collect_images_with_teleoperation(
    duration_sec=30,
    save_dir="data/captured_images",
    show_window=True,
)
print(result.metadata)