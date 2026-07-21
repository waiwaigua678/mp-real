# Known Limitations

- H6 does not claim new real-robot motion support.
- Piper and RM2 move-to-state and trajectory replay remain hardware blocked
  until H4/H5 hardware gates are recorded.
- The RM2 CLI defaults match the operator-provided command that was validated
  on hardware; that does not validate replay, move-to-state or Web replay.
- Piper hardware validation is not claimed by H6 documentation.
- Core deployment does not install `av` or `pyarrow`. Recording, data
  inspection, data validation, data audit, offline viewing and open-loop
  evaluation require an optional data extra.
- Legacy datasets may have unknown observation/action alignment and should not
  be used for training without audit evidence.
- The in-process Web resource manager does not coordinate multiple controller
  processes.
- ROS camera support depends on system ROS packages and is not represented as a
  PyPI dependency.

