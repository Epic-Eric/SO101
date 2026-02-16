from model.src.data import collect_images_with_teleoperation


def _prompt_duration() -> int | None:
    while True:
        try:
            raw = input("Enter duration in seconds for data collection (or press Enter for unlimited): ").strip()
            if raw == "":
                return None
            return int(raw)
        except ValueError:
            print("Invalid input. Please enter a valid integer or press Enter for unlimited duration.")
        except KeyboardInterrupt:
            print("\nData collection cancelled by user.")
            raise SystemExit(0)


def main() -> None:
    duration_time = _prompt_duration()
    result = collect_images_with_teleoperation(
        duration_sec=duration_time,
        save_dir="data/captured_images",
        show_window=True,
    )
    print(result.metadata)


if __name__ == "__main__":
    main()
