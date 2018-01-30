import get_data
import learner
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import *

# Getting and processing data
actors = list(set([a.split("\n")[0] for a in open("./subset_actors.txt").readlines()]))
actors_orig = ["Alec Baldwin", "Steve Carell", "Bill Hader", "Lorraine Bracco", "Angie Harmon", "Peri Gilpin"]
actors_new = ["Michael Vartan", "Gerard Butler", "Daniel Radcliffe", "Kristin Chenoweth", "America Ferrera",
              "Fran Drescher"]
extensions = [".jpg", ".JPG", ".png", ".PNG", ".jpeg", ".JPEG"]

# get_data.get_data() # Uncomment only if both ./cropped and ./uncropped are empty and we need to get new images
image_count = get_data.image_count("./cropped")
training_sets, validation_sets, test_sets = get_data.generate_sets(actors)
np.set_printoptions(precision=2)



# # Part 3: Steve Carell vs Alec Baldwin
# # Steve Carell: 1
# # Alec Baldwin: -1
# print("\n\n >>>PART 3<<<")
#
# x, y, thetas = learner.generate_xyt(training_sets["Steve Carell"] + training_sets["Alec Baldwin"], [1 for i in range(len(training_sets["Steve Carell"]))] + [-1 for i in range(len(training_sets["Alec Baldwin"]))])
# thetas_p3 = learner.grad_descent(learner.quad_loss, learner.quad_loss_grad, x, y, thetas, 0.001, 10000)
#
# test_sets_p3 = {key: test_sets[key] for key in ["Alec Baldwin", "Steve Carell"]}
# validation_sets_p3 = {key: validation_sets[key] for key in ["Alec Baldwin", "Steve Carell"]}
# answers_p3 = {"Alec Baldwin": np.array((-1)), "Steve Carell": np.array((1))}
#
# print("\nTesting Set")
# learner.test(test_sets_p3, answers_p3, thetas_p3)
# print("\nValidation Set")
# learner.test(validation_sets_p3, answers_p3, thetas_p3)
#
#
#
# # Part 4: Showing Thetas
# print("\n\n >>>PART 4<<<")
# # 4a
# x, y, thetas = learner.generate_xyt(training_sets["Steve Carell"][:2] + training_sets["Alec Baldwin"][:2], [1, 1, -1, -1])
# thetas_p4a_2 = learner.grad_descent(learner.quad_loss, learner.quad_loss_grad, x, y, thetas, 0.001, 10000)
# plt.imshow(thetas_p4a_2[1:].reshape((32, 32)))
# plt.title("Part 3 Thetas, 2 Images")
# plt.show()
#
# x, y, thetas = learner.generate_xyt(training_sets["Steve Carell"] + training_sets["Alec Baldwin"], [1 for i in range(len(training_sets["Steve Carell"]))] + [-1 for i in range(len(training_sets["Alec Baldwin"]))])
# thetas_p4a_all = learner.grad_descent(learner.quad_loss, learner.quad_loss_grad, x, y, thetas, 0.001, 10000)
# plt.imshow(thetas_p4a_all[1:].reshape((32, 32)))
# plt.title("Part 3 Thetas, All Images")
# plt.show()
#
# # 4b
# x, y, thetas = learner.generate_xyt(training_sets["Steve Carell"] + training_sets["Alec Baldwin"], [1 for i in range(len(training_sets["Steve Carell"]))] + [-1 for i in range(len(training_sets["Alec Baldwin"]))])
# thetas_p4b_10 = learner.grad_descent(learner.quad_loss, learner.quad_loss_grad, x, y, thetas, 0.001, 10)
# plt.imshow(thetas_p4b_10[1:].reshape((32, 32)))
# plt.title("Part 3 Thetas, 10 Iterations")
# plt.show()
#
# x, y, thetas = learner.generate_xyt(training_sets["Steve Carell"] + training_sets["Alec Baldwin"], [1 for i in range(len(training_sets["Steve Carell"]))] + [-1 for i in range(len(training_sets["Alec Baldwin"]))])
# thetas_p4b_100000 = learner.grad_descent(learner.quad_loss, learner.quad_loss_grad, x, y, thetas, 0.001, 10000) #TODO: change back to 100000
# plt.imshow(thetas_p4b_100000[1:].reshape((32, 32)))
# plt.title("Part 3 Thetas, 100000 Iterations")
# plt.show()
#
#
#
# # Part 5: Overfitting
# # Male: 1
# # Female: -1
# print("\n\n >>>PART 5<<<")
#
# testanswers_p5 = {"Steve Carell": np.array((1)), "Alec Baldwin": np.array((1)), "Bill Hader": np.array((1)),
#                    "Lorraine Bracco": np.array((-1)), "Peri Gilpin": np.array((-1)), "Angie Harmon": np.array((-1))}
#
# test_results_training = np.zeros((22,1))
# test_results_validation = np.zeros((22,1))
# for i in range (22):
#     print("\nUsing %i Training Images" % ((i+1)*5))
#     training_set_orig_6 = []
#     labels_malefemale = []
#
#     for actor in actors_orig:
#         training_set_orig_6 += training_sets[actor][:(i+1)*5]
#         if actor in ["Steve Carell", "Alec Baldwin", "Bill Hader"]:
#             labels_malefemale += [1 for i in range(len(training_sets[actor][:(i+1)*5]))]
#         elif actor in ["Lorraine Bracco", "Peri Gilpin", "Angie Harmon"]:
#             labels_malefemale += [-1 for i in range(len(training_sets[actor][:(i+1)*5]))]
#
#     x, y, thetas = learner.generate_xyt(training_set_orig_6, labels_malefemale)
#     thetas_p5 = learner.grad_descent(learner.quad_loss, learner.quad_loss_grad, x, y, thetas, 0.001, 100000)
#
#     training_sets_p5 = {key: training_sets[key][:(i+1)*5] for key in actors_orig}
#     validation_sets_p5 = {key:validation_sets[key] for key in actors_orig}
#     test_results_training[i] = learner.test(training_sets_p5, testanswers_p5, thetas_p5, False)
#     test_results_validation[i] = learner.test(validation_sets_p5, testanswers_p5, thetas_p5, False)
#
# plt.plot(range(5,115,5), test_results_validation*100, label="Validation")
# plt.plot(range(5,115,5), test_results_training*100, label="Training")
# plt.ylabel("% Correct")
# plt.xlabel("Number of Training Images Used")
# plt.axis([0,115,0,110])
# plt.legend()
# plt.title("Part 5 Test Results for Varying Number of Training Images")
# plt.show()
#
# testactors_p5b = {key: test_sets[key] for key in actors_new}
# testanswers_p5b = {"Michael Vartan": np.array((1)), "Gerard Butler": np.array((1)), "Daniel Radcliffe": np.array((1)),
#                    "Kristin Chenoweth": np.array((-1)), "America Ferrera": np.array((-1)),
#                    "Fran Drescher": np.array((-1))}
# learner.test(testactors_p5b, testanswers_p5b, thetas_p5)



# Part 7: Multiple Actor Classification
# Alec Baldwin:    [1,0,0,0,0,0]
# Steve Carell:    [0,1,0,0,0,0]
# Bill Hader:      [0,0,1,0,0,0]
# Lorraine Bracco: [0,0,0,1,0,0]
# Angie Harmon:    [0,0,0,0,1,0]
# Peri Gilpin:     [0,0,0,0,0,1]
print("\n\n >>>PART 7<<<")

training_set_orig_6 = []
labels_by_actor = []

for actor in actors_orig:
    training_set_orig_6 += training_sets[actor]
    if actor == "Alec Baldwin":
        labels_by_actor += [[1, 0, 0, 0, 0, 0] for i in range(len(training_sets[actor]))]
    elif actor == "Steve Carell":
        labels_by_actor += [[0, 1, 0, 0, 0, 0] for i in range(len(training_sets[actor]))]
    elif actor == "Bill Hader":
        labels_by_actor += [[0, 0, 1, 0, 0, 0] for i in range(len(training_sets[actor]))]
    elif actor == "Lorraine Bracco":
        labels_by_actor += [[0, 0, 0, 1, 0, 0] for i in range(len(training_sets[actor]))]
    elif actor == "Angie Harmon":
        labels_by_actor += [[0, 0, 0, 0, 1, 0] for i in range(len(training_sets[actor]))]
    elif actor == "Peri Gilpin":
        labels_by_actor += [[0, 0, 0, 0, 0, 1] for i in range(len(training_sets[actor]))]

x, y, thetas = learner.generate_xyt(training_set_orig_6, labels_by_actor)
thetas_p7 = learner.grad_descent(learner.quad_loss, learner.quad_loss_grad, x, y, thetas, 0.003, 5000)

test_set_p7 = {key: test_sets[key] for key in actors_orig}
training_set_p7 = {key: training_sets[key] for key in actors_orig}
test_answers_p7 = {"Alec Baldwin": [1, 0, 0, 0, 0, 0], "Steve Carell": [0, 1, 0, 0, 0, 0],
                  "Bill Hader": [0, 0, 1, 0, 0, 0], "Lorraine Bracco": [0, 0, 0, 1, 0, 0],
                  "Angie Harmon": [0, 0, 0, 0, 1, 0], "Peri Gilpin": [0, 0, 0, 0, 0, 1]}
learner.test(test_set_p7, test_answers_p7, thetas_p7)
learner.test(training_set_p7, test_answers_p7, thetas_p7, False)
learner.grad_est(x, y, thetas_p7, x.shape[1], learner.quad_loss, learner.quad_loss_grad)



# Part 8: Plotting Thetas for Part 7
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(thetas_p7[1:, i].reshape((32, 32)))
    plt.title(actors_orig[i])
plt.suptitle("Part 7 Thetas")
plt.show()
