import get_data
import learner
import numpy as np

import os

actors = list(set([a.split("\n")[0] for a in open("./subset_actors.txt").readlines()]))
extensions = [".jpg", ".JPG", ".png", ".PNG", ".jpeg", ".JPEG"]

#get_data.get_data()
image_count = get_data.image_count("./uncropped")

training_sets = {key: [] for key in actors}
validation_sets = {key: [] for key in actors}
test_sets = {key: [] for key in actors}

for actor in actors:
    for i in range(image_count[actor] - 20):
        for extension in extensions:
            if(os.path.isfile("./uncropped/" + actor.split()[1].lower() + str(i) + extension)):
                training_sets[actor].append((actor, actor.split()[1].lower() + str(i) + extension))
    for i in range(image_count[actor] - 20, image_count[actor] - 10):
        for extension in extensions:
            if(os.path.isfile("./uncropped/" + actor.split()[1].lower() + str(i) + extension)):
                validation_sets[actor].append((actor, actor.split()[1].lower() + str(i) + extension))
    for i in range(image_count[actor] - 10, image_count[actor]):
        for extension in extensions:
            if(os.path.isfile("./uncropped/" + actor.split()[1].lower() + str(i) + extension)):
                test_sets[actor].append((actor, actor.split()[1].lower() + str(i) + extension))

thetas = np.zeros((1,1024))
thetas[0][123] = 123
test = learner.exp_loss(thetas, test_sets["Alec Baldwin"] + test_sets["Steve Carell"], "Steve Carell", "Alec Baldwin")