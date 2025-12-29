# SO101 PyBullet Simulator

This simulator loads the SO101 robot from `URDF_so101.urdf` into a PyBullet GUI and provides:
- Joint control via keys 1..6 (reverse with Shift+1..6)
- Camera rotate: hold Shift and drag with left mouse (two-finger drag on trackpad)
- Camera pan: hold Ctrl and drag with left mouse
- Target point: adjust via sliders (Target X/Y/Z) and use the "Randomize Target" button (or press `R`) to randomize

Note: please obtain the correct URDF from https://cad.onshape.com/documents/7715cc284bb430fe6dab4ffd/w/4fd0791b683777b02f8d975a/e/826c553ede3b7592eb9ca800.

## Run

```bash
conda activate lerobot
python test_use/SETS/simulator/main.py
```

## Keys

- 1..6: move joints named "1".."6"
- Shift+1..6: move in reverse
- R: randomize target sphere within bounds

## Sliders

In the GUI, use Target X/Y/Z to place the red target sphere. Press the "Randomize Target" button to randomize its position.

## Troubleshooting
- If the GUI closes unexpectedly, re-run the command.
- On macOS, ensure the window has focus for mouse/keyboard events.
- Install PyBullet (conda-forge recommended): `conda install -c conda-forge pybullet`
