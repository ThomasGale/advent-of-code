# [Advent Of Code](https://adventofcode.com)
Turns out I'm really bad at solving these problems quickly...
Anyway, here are my various attempts.

## Build notes
- Docker + VSCode + Dev Containers Extension
- For non linux, PDCurses X11 window forwarding requires either Xming (win) or XQuartz (macOS) installed on host
- Open folder in container (allowing Docker to build the devcontainer)

## Running
- `F5` (should build and run debug - tweak `launch.json` args to change things like `year` / `day`)

## Side notes
Due to developing on both win / linux, git can grumble about line endings, suppres with  
`git config --global core.autocrlf true`  