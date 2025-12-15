from model.src.data import collect_images_with_teleoperation

duration_time = None

while True:
    try:
        duration_time = int(input("Enter duration in seconds for data collection (or press Enter for unlimited): "))
        break
    except ValueError:
        print("Invalid input. Please enter a valid integer or press Enter for unlimited duration.")
        duration_time = None
    except KeyboardInterrupt:
        print("\nData collection cancelled by user.")
        exit(0)

result = collect_images_with_teleoperation(
    duration_sec=duration_time,
    save_dir="data/captured_images",
    show_window=True,
)
print(result.metadata)

