# Vision-Utils Repository

## Info:
This repository is used by RoMeLa UCLA for deep learning computer vision models and its utils.

## User-Manual
- Please do not edit main. For development on this repo, use branches and pull requests. 
- For use in other projects, create a submodule inside another repository.

### Setting up as submodule
Enter this command into your own repo
```bash
git submodule add https://github.com/alvister88/Vision-Utils.git <path/to/submodule>
```
```bash
git submodule update --init --recursive
```

### Installation
This library is pip installable
```bash
pip install visionml-utils
```
#### Dev Install
Run the command inside the root directory
```bash
pip install -e .
```

