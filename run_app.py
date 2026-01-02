import argparse
import os
import subprocess
import sys


def _workspace_root() -> str:
	return os.path.abspath(os.path.dirname(__file__))


def _build_streamlit_cmd(app_path: str) -> list[str]:
	return [
		sys.executable,
		"-m",
		"streamlit",
		"run",
		app_path,
		"--server.headless",
		"false",
	]


def main() -> None:
	parser = argparse.ArgumentParser(description="Run Streamlit apps in this workspace")
	parser.add_argument(
		"app",
		nargs="?",
		default=None,
		help="Which app to run: reward_camera | world_collect (omit to select interactively)",
	)
	args = parser.parse_args()

	apps: dict[str, dict[str, str]] = {
		"reward_camera": {
			"label": "Red Bead Detector App",
			"path": "model/src/reward_camera/streamlit_app.py",
		},
		"world_collect": {
			"label": "World Model Data Collector",
			"path": "model/src/data/app/streamlit_collect_world_model.py",
		},
	}

	app_choice = args.app
	if app_choice is None:
		# Interactive selector (best effort). Falls back to simple stdin prompt.
		if sys.stdin.isatty():
			try:
				# InquirerPy API varies slightly by version; keep this best-effort.
				from InquirerPy import inquirer as _inq  

				options = [
					{"name": f"{apps[k]['label']}  ({k})", "value": k}
					for k in apps.keys()
				]
				select_fn = getattr(_inq, "select", None)
				if callable(select_fn):
					prompt: object = select_fn(message="Select an app to run:", choices=options)
					exec_fn = getattr(prompt, "execute", None)
					if callable(exec_fn):
						app_choice = exec_fn()
			except Exception:
				pass

		if app_choice is None:
			print("Select an app to run:")
			for i, k in enumerate(apps.keys(), start=1):
				print(f"  {i}) {apps[k]['label']} ({k})")
			try:
				idx = int(input("Enter choice (1-2): ").strip())
				app_choice = list(apps.keys())[idx - 1]
			except Exception:
				app_choice = "reward_camera"

	if not isinstance(app_choice, str) or app_choice not in apps:
		raise SystemExit(f"Unknown app '{app_choice}'. Valid: {', '.join(apps.keys())}")
	app_choice = str(app_choice)

	workspace_root = _workspace_root()

	label = apps[app_choice]["label"]
	app_path = os.path.join(workspace_root, apps[app_choice]["path"])

	print(f"Starting {label}...")
	print(f"Workspace Root: {workspace_root}")
	print(f"App Path: {app_path}")

	cmd = _build_streamlit_cmd(app_path)

	env = os.environ.copy()
	# Ensure workspace root is on PYTHONPATH for absolute imports like `model.src...`
	if "PYTHONPATH" in env and env["PYTHONPATH"].strip():
		env["PYTHONPATH"] = workspace_root + os.pathsep + env["PYTHONPATH"]
	else:
		env["PYTHONPATH"] = workspace_root

	print(f"Running command: {' '.join(cmd)}")
	try:
		subprocess.run(cmd, env=env, check=True)
	except KeyboardInterrupt:
		print("\nStopped by user.")
	except subprocess.CalledProcessError as e:
		print(f"Error running streamlit app: {e}")


if __name__ == "__main__":
	main()
