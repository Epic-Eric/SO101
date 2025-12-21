"""Model project package root.

Keep this import-light: training and inference code may run in environments
that do not have robot teleop dependencies or teleop interface files.
"""

__all__ = []

try:
	from .src.interfaces.teleop import TeleopSample, TeleopSessionResult  # re-export

	__all__ += [
		"TeleopSample",
		"TeleopSessionResult",
	]
except Exception:
	# Teleop interfaces are optional for pure training usage.
	pass
