When you press F5:
VSCode reads launch.json
Sees preLaunchTask: "build"
Runs the build task from tasks.json
tasks.json runs make build
Make checks file timestamps and:
If source files changed: recompiles only those files
If no changes: does nothing
Then the debugger launches with the settings from launch.json