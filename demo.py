import os
from subprocess import Popen
from time import time

all_command = []

for file, size, p, t in [
    ("./img/dragon-girl.png", 256, 6, 1),
    ("./img/house.png", 128, 8, 2),
    ("./img/horse-girl.png", 256, 8, 2),
]:
    output_file = os.path.join("./demo", os.path.splitext(os.path.basename(file))[0])
    for thickness in [0, t]:
        for mode in ["nearest", "bicubic", "center", "k-centroid", "contrast"]:
            all_command.append(
                "python -m pixeloe.cli "
                f"{file} --thickness {thickness} "
                f"--target_size {size} --patch_size {p} "
                f"-M {mode} -O {output_file}-t{thickness}-{mode}.png"
            )


## Use Popen to run all commands in parallel
t0 = time()
all_process = []
for command in all_command:
    print(command)
    all_process.append(Popen(command, shell=True))


## Wait for all processes to finish
for process in all_process:
    process.wait()
t1 = time()
print("=" * 50)
print(f"Total time   : {t1 - t0:.3f}sec")
print(f"Total process: {len(all_process)}")
print(f"Average cost : {(t1 - t0) / len(all_process):.3f}sec")
